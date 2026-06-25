"""
DreamOn quickstart-faithful infilling for the AISE Java-Identifier finetunes.

WHY THIS EXISTS (verified against BOTH tokenizers on 2026-06-11):
  ``__MASKED_VAR__`` is NOT a token. Grepping tokenizer.json of both
  checkpoints finds no such entry; the only mask token is ``<|mask|>``
  (7B id 151666, 0.5B id 248077). A DreamOn diffusion model denoises ONLY
  positions whose id equals its single ``mask_token_id``. So if you feed the
  literal string ``__MASKED_VAR__`` it becomes 5 ordinary BPE tokens that are
  never masked -> they pass straight through to the output unchanged. That is
  exactly the ``__MASKED_VAR__`` predictions seen in placeholder mode.

  The faithful realization of "the variable was written as __MASKED_VAR__ and
  then masked for diffusion" is therefore: put ``<|mask|>`` at each site, sized
  by the token length of __MASKED_VAR__ (=5). This script follows the official
  DreamOn README (process_infilling_prompt + diffusion_generate) EXACTLY and
  exposes BOTH interpretations so the experiment settles it:

    --mask-mode real     (default) each [MASK] site -> <|mask|> * number_of_mask
                         (the real diffusion mask; correct).
    --mask-mode literal  each [MASK] site -> the token ids of "__MASKED_VAR__"
                         (your hypothesis; these are NOT mask ids, so DreamOn
                         leaves them untouched -> they reappear in the output,
                         proving __MASKED_VAR__ cannot act as the mask token).

It reuses ONLY model loading + window tiling + anchor extraction from
benchmark_dreamon_java_identifiers; the prompt build + diffusion_generate are
inline here and mirror the README, so there is no hidden wrapper to blame.

Usage on the HPC cluster (run separately from the main benchmark):
    # smoke, real mask (should give clean identifiers):
    python experiments/benchmark_dreamon_java_quickstart.py \
        --model dreamon-7b-Java --mask-mode real --max-samples 20 --debug
    # smoke, literal __MASKED_VAR__ (should reproduce the pass-through):
    python experiments/benchmark_dreamon_java_quickstart.py \
        --model dreamon-7b-Java --mask-mode literal --max-samples 20 --debug
    # full run, real mask:
    python experiments/benchmark_dreamon_java_quickstart.py --model dreamon-7b-Java --mask-mode real
"""

import os
import sys
import csv
import gc
import argparse
import time
from collections import Counter
from datetime import datetime

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import benchmark_dreamon_java_identifiers as J   # model loading / tiling / extraction

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RESULTS_DIR = "results/dreamon_java"

# README quickstart params (DreamOn 7B, variable-length).
DREAMON_GEN = dict(temperature=0.2, top_p=0.9, alg="entropy", alg_temp=0,
                   number_transfer_tokens=1)
# Fixed-length params for the 0.5B "dream" backend.
DREAM_GEN = dict(temperature=0.3, top_p=0.95, alg="entropy", alg_temp=0.0)


def diagnose_tokenizer(tokenizer, mask_pattern):
    """Print, loudly, how the mask token and __MASKED_VAR__ are represented."""
    mid = tokenizer.mask_token_id
    pieces = tokenizer.encode(mask_pattern, add_special_tokens=False)
    try:
        single = tokenizer.convert_tokens_to_ids(mask_pattern)
    except Exception:
        single = tokenizer.unk_token_id
    is_single = (single is not None and single != tokenizer.unk_token_id
                 and len(pieces) == 1)
    print("  --- tokenizer diagnosis -----------------------------------------")
    print(f"  mask_token        : {tokenizer.mask_token!r}  id={mid}")
    print(f"  {mask_pattern!r} as single token? {is_single}  (convert_tokens_to_ids -> {single})")
    print(f"  {mask_pattern!r} BPE pieces      : {tokenizer.tokenize(mask_pattern)}  ({len(pieces)} tokens)")
    if not is_single:
        print(f"  => __MASKED_VAR__ is NOT a mask token; diffusion masks <|mask|> only.")
        print(f"  => 'real' mode masks with <|mask|>*{len(pieces)}; 'literal' mode will pass through.")
    print("  -----------------------------------------------------------------")
    return len(pieces)


def build_prompt(window_text, tokenizer, number_of_mask, mask_mode, mask_pattern):
    """README process_infilling_prompt shape, generalised to N sites in a window.

    real    : each [MASK] -> [mask_token_id] * number_of_mask
    literal : each [MASK] -> encode("__MASKED_VAR__")  (NOT mask ids)
    """
    parts = window_text.split("[MASK]")
    bos = tokenizer.bos_token_id
    eos = tokenizer.eos_token_id
    mask_id = tokenizer.mask_token_id
    lit_ids = tokenizer.encode(mask_pattern, add_special_tokens=False)

    ids = [bos] if bos is not None else []
    for i, seg in enumerate(parts):
        ids.extend(tokenizer.encode(seg, add_special_tokens=False))
        if i < len(parts) - 1:
            if mask_mode == "literal":
                ids.extend(lit_ids)
            else:
                ids.extend([mask_id] * number_of_mask)
    if eos is not None:
        ids.append(eos)
    return ids


def run_window(model, tokenizer, window_text, number_of_mask, gen_mode,
               mask_mode, mask_pattern, max_new_tokens, steps, debug=False):
    ids = build_prompt(window_text, tokenizer, number_of_mask, mask_mode, mask_pattern)
    input_t = torch.LongTensor([ids]).to(model.device)
    with torch.no_grad():
        if gen_mode == "dreamon":
            eff = max(number_of_mask, max_new_tokens)
            out = model.diffusion_generate(
                input_t, max_new_tokens=eff, return_dict_in_generate=True,
                output_history=False, **DREAMON_GEN)
        else:  # 0.5B fixed-length
            am = torch.ones_like(input_t)
            out = model.diffusion_generate(
                input_t, attention_mask=am, max_new_tokens=1, steps=steps,
                return_dict_in_generate=True, output_history=False, **DREAM_GEN)
    seq = out.sequences[0] if hasattr(out, "sequences") else out[0]
    full = tokenizer.decode(seq, skip_special_tokens=True)
    preds = J.extract_idents_by_anchor(full, window_text)
    if debug:
        print(f"  [debug:{gen_mode}/{mask_mode}] sites={window_text.count('[MASK]')} "
              f"masks/site={number_of_mask if mask_mode=='real' else len(tokenizer.encode(mask_pattern, add_special_tokens=False))}")
        print(f"  [debug] preds={preds}")
        print(f"  [debug] decoded[:240]={full[:240]!r}")
    return preds


def predict_one(model, tokenizer, masked_code, number_of_mask, gen_mode,
                mask_mode, mask_pattern, ctx, max_sites, max_new_tokens, steps, debug=False):
    n_total = masked_code.count("[MASK]")
    if n_total == 0:
        return [], 0
    windows = J.tile_windows(masked_code, target_chars=ctx, max_sites=max_sites)
    site_preds = [""] * n_total
    for w_idx, (win, gidx) in enumerate(windows):
        preds = run_window(model, tokenizer, win, number_of_mask, gen_mode,
                           mask_mode, mask_pattern, max_new_tokens, steps,
                           debug=(debug and w_idx == 0))
        for li, gi in enumerate(gidx):
            site_preds[gi] = preds[li] if li < len(preds) else ""
    return site_preds, len(windows)


def main():
    p = argparse.ArgumentParser(description="DreamOn quickstart-faithful infilling (Java identifiers).")
    p.add_argument("--model", default="dreamon-7b-Java", choices=list(J.MODEL_REGISTRY))
    p.add_argument("--mask-mode", choices=["real", "literal"], default="real",
                   help="real = <|mask|>*N (correct); literal = feed __MASKED_VAR__ ids (pass-through test).")
    p.add_argument("--mask-pattern", default=J.MASK_PLACEHOLDER)
    p.add_argument("--number-of-mask", type=int, default=None,
                   help="masks per site for real mode (default = len(tokenize(__MASKED_VAR__))).")
    p.add_argument("--data-path", default=J.DATA_PATH)
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--context-chars", type=int, default=J.CONTEXT_CHARS)
    p.add_argument("--max-sites", type=int, default=J.MAX_SITES_IN_WINDOW)
    p.add_argument("--max-new-tokens", type=int, default=J.DREAMON_MAX_NEW_TOKENS)
    p.add_argument("--steps", type=int, default=J.DREAM_STEPS)
    p.add_argument("--debug", action="store_true")
    p.add_argument("--hf-repo", default=None)
    p.add_argument("--hf-token", default=os.environ.get("HF_TOKEN"))
    args = p.parse_args()

    meta = J.MODEL_REGISTRY[args.model]
    gen_mode = meta["gen_mode"]
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print(f"Loading data from {args.data_path} ...")
    data = J.load_data(args.data_path)
    if args.start:
        data = data[args.start:]
    if args.max_samples is not None:
        data = data[:args.max_samples]
    print(f"Loaded {len(data)} samples (start={args.start}) on device={DEVICE}.")

    print(f"\n{'='*64}\n  {meta['label']}  ({meta['id']})")
    print(f"  gen_mode={gen_mode}  mask_mode={args.mask_mode}\n{'='*64}")
    t0 = time.time()
    tokenizer, model = J.load_model(meta, hf_token=args.hf_token)
    print(f"  loaded in {time.time()-t0:.1f}s")

    n_mask = args.number_of_mask or diagnose_tokenizer(tokenizer, args.mask_pattern)
    print(f"  number_of_mask/site = {n_mask}")

    if args.debug:
        print("\n[sanity] 'return a + [MASK];'")
        sp, _ = predict_one(model, tokenizer,
                            "public int add(int a, int b) {\n    return a + [MASK];\n}\n",
                            n_mask, gen_mode, args.mask_mode, args.mask_pattern,
                            10000, args.max_sites, args.max_new_tokens, args.steps, debug=True)
        print(f"[sanity] preds={sp}\n")

    rows = []
    n = len(data)
    n_first = n_any = n_consistent = n_all = errors = 0
    site_c = site_t = 0
    for idx, r in enumerate(J.tqdm(data, desc=f"  {meta['label']}")):
        gt = r["target"]
        n_masks = r["masked_code"].count("[MASK]")
        try:
            preds, nw = predict_one(model, tokenizer, r["masked_code"], n_mask,
                                    gen_mode, args.mask_mode, args.mask_pattern,
                                    args.context_chars, args.max_sites,
                                    args.max_new_tokens, args.steps,
                                    debug=(args.debug and idx < 2))
            for pr in preds:
                site_t += 1
                site_c += int(pr == gt)
            first = preds[0] if preds else ""
            maj = Counter(preds).most_common(1)[0][0] if preds else ""
            consistent = (len(preds) > 0 and len(set(preds)) == 1)
            allok = (len(preds) > 0 and all(x == gt for x in preds))
            n_first += int(first == gt)
            n_any += int(any(x == gt for x in preds))
            n_consistent += int(consistent)
            n_all += int(allok)
            rows.append({"id": r["id"], "ground_truth": gt, "n_total_masks": n_masks,
                         "n_windows": nw, "predictions": "|".join(preds),
                         "first_pred": first, "first_correct": (first == gt),
                         "majority_pred": maj, "any_correct": any(x == gt for x in preds),
                         "all_sites_consistent": consistent, "all_sites_correct": allok,
                         "error": ""})
        except Exception as e:           # noqa: BLE001
            errors += 1
            rows.append({"id": r["id"], "ground_truth": gt, "n_total_masks": n_masks,
                         "n_windows": 0, "predictions": "", "first_pred": "",
                         "first_correct": False, "majority_pred": "",
                         "any_correct": False, "all_sites_consistent": False,
                         "all_sites_correct": False, "error": str(e)[:200]})
            if errors <= 5:
                print(f"  error on {r['id']}: {e}")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = f"{meta['label']}_quickstart-{args.mask_mode}"
    if args.start:
        tag += f"_start{args.start}"
    out = os.path.join(RESULTS_DIR, f"{tag}_per_sample_{ts}.csv")
    fields = ["id", "ground_truth", "n_total_masks", "n_windows", "predictions",
              "first_pred", "first_correct", "majority_pred", "any_correct",
              "all_sites_consistent", "all_sites_correct", "error"]
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})

    site_acc = site_c / site_t if site_t else 0.0
    print(f"\n=== {meta['label']} quickstart mask_mode={args.mask_mode} ===")
    print(f"Site-level EM     : {site_c}/{site_t} = {site_acc:.2%}")
    print(f"First-mask EM     : {n_first}/{n} = {n_first/n:.2%}")
    print(f"Any-site EM       : {n_any}/{n} = {n_any/n:.2%}")
    print(f"All-sites CONSIST : {n_consistent}/{n} = {n_consistent/n:.2%}")
    print(f"All-sites CORRECT : {n_all}/{n} = {n_all/n:.2%}")
    print(f"Errors            : {errors}")
    print(f"Saved             : {out}")

    if args.hf_repo:
        J.upload_to_hf(out, args.hf_repo, args.hf_token)

    del model, tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


if __name__ == "__main__":
    main()
