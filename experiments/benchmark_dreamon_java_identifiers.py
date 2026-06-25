"""
Exact-Match benchmark for the two AISE-TUDelft DreamOn Java-Identifier models
on the refineID (variable-naming refactoring) task.

    * AISE-TUDelft/dreamon-0.5b-Java-Identifiers   (581M, Qwen-base + Dream code)
    * AISE-TUDelft/dreamon-7b-Java-Identifiers     (7.6B, DreamOn-v0-7B fine-tune)

This reuses our existing dLLM EM setup (the tiled, all-site coverage protocol
from ``experiments/benchmark_dreamon.py``) and ADDS explicit all-sites
consistency metrics, which the previous EM scripts did NOT compute (see the
"Consistency" note below).

Why two generation backends
---------------------------
The two checkpoints publish DIFFERENT generation recipes (their own
``generation_config.json``), so we honour each:

  * 7B   -> DreamOn variable-length infilling
            (alg="entropy", number_transfer_tokens=1, <|expand|> canvas,
             NO `steps`).  Same protocol as benchmark_dreamon.py.
            Its repo does NOT ship modeling_dream.py / generation_utils.py and
            its auto_map is LOCAL, so we materialise the code from the base
            repo Dream-org/DreamOn-v0-7B (architecture + tokens are identical;
            _name_or_path == checkpoints/DreamOn-v0-7B).

  * 0.5B -> fixed-length masked diffusion
            (alg="entropy"/"origin", `steps`, no expand/contract).  Same
            protocol as the DiffuCoder/DreamCoder branch of
            benchmark_diffusion_models.py.  Its auto_map points to
            Dream-org/Dream-Coder-v0-Base-7B, whose generation_utils.py is the
            fixed-length one; we materialise that code locally so the run is
            offline-safe on the HPC cluster.

Both backends produce ONE predicted identifier per [MASK] site, so the EM /
consistency metrics below are computed identically for both models.

Metrics (per sample; all [MASK] in a sample refer to the SAME variable)
-----------------------------------------------------------------------
  site_acc            per-[MASK]-site EM   (each occurrence vs ground truth)
  first_mask_acc      EM on the FIRST site (matches benchmark_diffusion_models
                      and the DreamOn `tiled_first_mask` baseline)
  majority_acc        EM of the majority-vote name across sites
  any_acc             >=1 site equals ground truth
  consistent_rate     fraction of samples where ALL sites predict the SAME
                      name (regardless of correctness)  -- NEW
  all_correct_acc     STRICT EM: ALL sites equal ground truth               -- NEW
                      (== consistent AND the agreed name is correct)

Mask-token pattern of the fine-tuned checkpoints (--mask-style)
---------------------------------------------------------------
The Java-Identifier fine-tunes were trained with the site placeholder
``__MASKED_VAR__``. It is LITERAL TEXT, not a tokenizer special token (both
repos still register <|mask|> as the only mask token, and diffusion can only
denoise <|mask|> positions), so the placeholder must enter the prompt at the
data level. The exact training layout is not recorded in the model repos
(empty model cards), so BOTH plausible layouts are implemented:

  --mask-style canvas       (default) every [MASK] site becomes a <|mask|>
                            canvas of length len(tokenize("__MASKED_VAR__"))
                            -- assumes training masked the placeholder's
                            token positions in place, so the canvas length
                            prior matches the fine-tuning distribution.
  --mask-style placeholder  the FIRST site of each window is the <|mask|>
                            canvas; every OTHER site shows the literal
                            __MASKED_VAR__ text as co-reference context.
                            The window's single prediction is assigned to
                            all of its sites.
  --mask-style legacy       the original benchmark behaviour (k=4 dreamon /
                            k=2 dream canvas at every site), to reproduce /
                            compare with the Jun-07 runs.

Input data may mark sites with either ``[MASK]`` (our test.csv) or
``__MASKED_VAR__`` (the fine-tuning convention); both are accepted.

Pick the style empirically: run both with --max-samples 20 --debug, keep the
one whose predictions look sane, then do the full run.

Usage on the HPC cluster:
    # smoke test, both styles (per model):
    python experiments/benchmark_dreamon_java_identifiers.py \
        --model dreamon-7b-Java --mask-style canvas --max-samples 20 --debug
    python experiments/benchmark_dreamon_java_identifiers.py \
        --model dreamon-7b-Java --mask-style placeholder --max-samples 20 --debug
    # full 1000-sample set, both models, chosen style:
    python experiments/benchmark_dreamon_java_identifiers.py --mask-style canvas
    # old behaviour:
    python experiments/benchmark_dreamon_java_identifiers.py --mask-style legacy
"""

import os
import sys
import csv
import re
import gc
import json
import shutil
import argparse
import time
from collections import Counter
from datetime import datetime

import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

try:
    from huggingface_hub import HfApi, snapshot_download, hf_hub_download
    HAS_HF_HUB = True
except ImportError:
    HAS_HF_HUB = False


# --- Mock torchvision (some Dream/DreamOn checkpoints try to import it) ------
class _MockModule:
    def __getattr__(self, name): return _MockModule()
    def __call__(self, *args, **kwargs): return _MockModule()

sys.modules.setdefault('torchvision', _MockModule())
sys.modules.setdefault('torchvision.ops', _MockModule())
sys.modules.setdefault('torchvision.transforms', _MockModule())
if not hasattr(torch.ops, 'torchvision'):
    class _DummyOps:
        def nms(*args, **kwargs): return torch.tensor([])
    torch.ops.torchvision = _DummyOps()


# ---- Configuration ---------------------------------------------------------

DATA_PATH = "data/test.csv"
RESULTS_DIR = "results/dreamon_java"
LOCAL_MODELS_DIR = "hf_models"          # where we materialise weights + code
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Window cropping around [MASK] sites (chars). ~4 chars/token -> 3000 chars is
# ~750 tokens of context, well inside the 32k position budget after masks.
CONTEXT_CHARS = 3000
MAX_SITES_IN_WINDOW = 8

# Per-mode initial mask-token count per [MASK] site (legacy style only).
NUM_MASK_PER_SITE = {"dreamon": 4, "dream": 2}

# Literal site placeholder used in the Java-Identifier fine-tuning data.
# NOT a tokenizer special token -- it BPE-tokenizes into ~5 normal pieces.
# canvas style: per-site canvas length = len(tokenize(MASK_PLACEHOLDER)).
# placeholder style: non-first sites show this text verbatim as context.
MASK_PLACEHOLDER = "__MASKED_VAR__"
# DreamOn: max canvas size after <|expand|> growth (must be >= num_mask).
DREAMON_MAX_NEW_TOKENS = 64
# Fixed-length diffusion steps for the 0.5B "dream" backend.
DREAM_STEPS = 64

# Generation kwargs that replicate our previous dLLM EM setup.
DREAMON_GEN_KWARGS = dict(
    temperature=0.2, top_p=0.9, alg="entropy", alg_temp=0,
    number_transfer_tokens=1,
)
DREAM_GEN_KWARGS = dict(
    temperature=0.3, top_p=0.95, alg="entropy", alg_temp=0.0, steps=DREAM_STEPS,
)


# ---- Model registry --------------------------------------------------------
# code_repo: where to fetch modeling_dream.py / generation_utils.py /
#            tokenization_dream.py from, since the AISE repos ship only
#            configuration_dream.py + weights + tokenizer.
MODEL_REGISTRY = {
    "dreamon-0.5b-Java": {
        "id":        "AISE-TUDelft/dreamon-0.5b-Java-Identifiers",
        "gen_mode":  "dream",
        "code_repo": "Dream-org/Dream-Coder-v0-Base-7B",
        "label":     "DreamOn-0.5B-Java",
    },
    "dreamon-7b-Java": {
        "id":        "AISE-TUDelft/dreamon-7b-Java-Identifiers",
        "gen_mode":  "dreamon",
        "code_repo": "Dream-org/DreamOn-v0-7B",
        "label":     "DreamOn-7B-Java",
    },
}

CODE_FILES = ["modeling_dream.py", "generation_utils.py",
              "configuration_dream.py", "tokenization_dream.py"]


# ---- Window tiling (shared) ------------------------------------------------

_MASK_RE = re.compile(r'\[MASK\]')
_IDENT_RE = re.compile(r'[a-zA-Z_$][a-zA-Z0-9_$]*')


def tile_windows(code, target_chars=CONTEXT_CHARS, max_sites=MAX_SITES_IN_WINDOW):
    """Tile the FULL sample into non-overlapping windows so EVERY [MASK] falls
    into exactly one window. Returns list of (window_text, global_site_indices).
    """
    spans = [m.span() for m in _MASK_RE.finditer(code)]
    if not spans:
        return []

    half = target_chars // 2
    windows = []
    i = 0
    while i < len(spans):
        group = [i]
        while (len(group) < max_sites
               and group[-1] + 1 < len(spans)
               and spans[group[-1] + 1][1] - spans[group[0]][0] <= target_chars):
            group.append(group[-1] + 1)

        first_site_start = spans[group[0]][0]
        last_site_end = spans[group[-1]][1]
        win_start = max(0, first_site_start - half)
        win_end = min(len(code), last_site_end + half)

        # Clip so adjacent external [MASK]s never bleed into this window.
        if group[0] > 0:
            win_start = max(win_start, spans[group[0] - 1][1])
        if group[-1] + 1 < len(spans):
            win_end = min(win_end, spans[group[-1] + 1][0])

        win_text = code[win_start:win_end]
        n_in_win = win_text.count("[MASK]")
        assert n_in_win == len(group), (
            f"tile_windows mismatch: window has {n_in_win} [MASK]s but "
            f"group has {len(group)}"
        )
        windows.append((win_text, group))
        i = group[-1] + 1
    return windows


def extract_idents_by_anchor(full_text, window_text):
    """Return one identifier per [MASK] site in window_text by anchoring on the
    surrounding (unmasked) text segments. Robust for both backends because the
    fixed text around each site is never rewritten."""
    parts = window_text.split("[MASK]")
    if len(parts) <= 1:
        return []
    preds = []
    cursor = 0
    for i in range(len(parts) - 1):
        pre = parts[i].strip()
        post = parts[i + 1].strip()
        pre_anchor = pre[-30:] if len(pre) > 30 else pre
        post_anchor = post[:30] if len(post) > 30 else post

        if pre_anchor:
            idx = full_text.find(pre_anchor, cursor)
            if idx == -1:
                preds.append("")
                continue
            cursor = idx + len(pre_anchor)
        if post_anchor:
            end_idx = full_text.find(post_anchor, cursor)
            if end_idx == -1:
                end_idx = min(cursor + 60, len(full_text))
        else:
            end_idx = min(cursor + 60, len(full_text))

        gap = full_text[cursor:end_idx] if end_idx > cursor else full_text[cursor:cursor + 60]
        m = _IDENT_RE.search(gap)
        preds.append(m.group(0) if m else gap.strip()[:20])
        cursor = end_idx
    return preds


def window_with_placeholders(window_text, placeholder):
    """placeholder style: keep ONE [MASK] (the first site) as the canvas
    marker and render every other site as the literal placeholder text, so
    the model sees where the variable recurs without extra canvases."""
    parts = window_text.split("[MASK]")
    return parts[0] + "[MASK]" + placeholder.join(parts[1:])


# ---- DreamOn (variable-length) backend -------------------------------------

def build_dreamon_prompt(window_text, tokenizer, num_mask_per_site):
    """BOS + prefix + (<|mask|>*k per site) + suffix + EOS."""
    parts = window_text.split("[MASK]")
    bos = tokenizer.bos_token_id
    eos = tokenizer.eos_token_id
    mask_id = tokenizer.mask_token_id
    ids = [bos] if bos is not None else []
    for i, segment in enumerate(parts):
        ids.extend(tokenizer.encode(segment, add_special_tokens=False))
        if i < len(parts) - 1:
            ids.extend([mask_id] * num_mask_per_site)
    if eos is not None:
        ids.append(eos)
    return ids


def run_window_dreamon(model, tokenizer, window_text, num_mask_per_site,
                       max_new_tokens, gen_kwargs, debug=False):
    input_ids = build_dreamon_prompt(window_text, tokenizer, num_mask_per_site)
    eff_max_new = max(num_mask_per_site, max_new_tokens)
    input_t = torch.LongTensor([input_ids]).to(model.device)
    with torch.no_grad():
        output = model.diffusion_generate(
            input_t, max_new_tokens=eff_max_new,
            return_dict_in_generate=True, output_history=False, **gen_kwargs,
        )
    seq = output.sequences[0] if hasattr(output, "sequences") else output[0]
    full = tokenizer.decode(seq, skip_special_tokens=True)
    preds = extract_idents_by_anchor(full, window_text)
    if debug:
        print(f"  [debug:dreamon] sites={window_text.count('[MASK]')} preds={preds}")
        print(f"  [debug:dreamon] decoded[:240]={full[:240]!r}")
    return preds


# ---- Dream (fixed-length) backend ------------------------------------------

def run_window_dream(model, tokenizer, window_text, num_mask_per_site,
                     steps, gen_kwargs, debug=False):
    mask_tok = tokenizer.mask_token
    input_code = window_text.replace("[MASK]", mask_tok * num_mask_per_site)
    inputs = tokenizer(input_code, return_tensors="pt")
    input_ids = inputs.input_ids.to(model.device)
    attention_mask = inputs.attention_mask.to(model.device)
    with torch.no_grad():
        output = model.diffusion_generate(
            input_ids, attention_mask=attention_mask,
            max_new_tokens=1, steps=steps, **gen_kwargs,
        )
    seq = output.sequences[0] if hasattr(output, "sequences") else output[0]
    full = tokenizer.decode(seq, skip_special_tokens=True)
    preds = extract_idents_by_anchor(full, window_text)
    if debug:
        print(f"  [debug:dream] sites={window_text.count('[MASK]')} preds={preds}")
        print(f"  [debug:dream] decoded[:240]={full[:240]!r}")
    return preds


# ---- Per-sample prediction (tile + dispatch) -------------------------------

def predict_one(model, tokenizer, masked_code, gen_mode, num_mask_per_site,
                context_chars, max_sites, max_new_tokens, steps,
                mask_style="legacy", placeholder=MASK_PLACEHOLDER, debug=False):
    n_total = masked_code.count("[MASK]")
    if n_total == 0:
        return [], 0
    windows = tile_windows(masked_code, target_chars=context_chars, max_sites=max_sites)
    site_preds = [""] * n_total
    for w_idx, (win_text, global_indices) in enumerate(windows):
        win_for_model = win_text
        if mask_style == "placeholder":
            # one canvas (first site) + literal placeholder at the others
            win_for_model = window_with_placeholders(win_text, placeholder)
        if gen_mode == "dreamon":
            preds = run_window_dreamon(
                model, tokenizer, win_for_model, num_mask_per_site,
                max_new_tokens, DREAMON_GEN_KWARGS, debug=(debug and w_idx == 0))
        else:
            preds = run_window_dream(
                model, tokenizer, win_for_model, num_mask_per_site,
                steps, {k: v for k, v in DREAM_GEN_KWARGS.items() if k != "steps"},
                debug=(debug and w_idx == 0))
        if mask_style == "placeholder":
            # the window has a single canvas -> its prediction names ALL
            # co-referent sites of this window
            one = preds[0] if preds else ""
            preds = [one] * len(global_indices)
        for local_i, gi in enumerate(global_indices):
            site_preds[gi] = preds[local_i] if local_i < len(preds) else ""
    return site_preds, len(windows)


# ---- Data loading ----------------------------------------------------------

def load_data(data_path, max_samples=None):
    csv.field_size_limit(sys.maxsize)
    rows = []
    with open(data_path, "r", encoding="utf-8") as f:
        for i, row in enumerate(csv.reader(f)):
            if max_samples is not None and i >= max_samples:
                break
            if len(row) < 3:
                continue
            code = row[1]
            # Accept either site marker: our test.csv uses [MASK]; the
            # fine-tuning data convention marks sites with __MASKED_VAR__.
            if MASK_PLACEHOLDER in code:
                code = code.replace(MASK_PLACEHOLDER, "[MASK]")
            rows.append({"id": row[0], "masked_code": code, "target": row[2].strip()})
    return rows


# ---- Model materialisation + loading ---------------------------------------

def materialize_model(model_id, code_repo, hf_token=None):
    """Download the model into a writable local dir, fill in any missing custom
    code from ``code_repo``, and rewrite auto_map to LOCAL references so loading
    is deterministic and offline-safe."""
    if not HAS_HF_HUB:
        # No hub helpers -> hope the repo is self-contained / cached.
        return model_id
    local_dir = os.path.join(LOCAL_MODELS_DIR, model_id.replace("/", "__"))
    print(f"  Materialising {model_id} -> {local_dir}")
    # Skip the training-only blobs the AISE repos ship (optimizer_state.pt /
    # training_state.pt are ~2-3x the model size and would blow Colab disk).
    snapshot_download(repo_id=model_id, local_dir=local_dir, token=hf_token,
                      ignore_patterns=["optimizer_state.pt", "training_state.pt"])

    for fn in CODE_FILES:
        dst = os.path.join(local_dir, fn)
        if os.path.exists(dst) or not code_repo:
            continue
        try:
            src = hf_hub_download(repo_id=code_repo, filename=fn, token=hf_token)
            shutil.copy(src, dst)
            print(f"    + injected {fn} from {code_repo}")
        except Exception as e:
            print(f"    (skip {fn}: {e})")

    cfg_path = os.path.join(local_dir, "config.json")
    with open(cfg_path) as f:
        cfg = json.load(f)
    if "auto_map" in cfg:
        cfg["auto_map"] = {k: v.split("--")[-1] for k, v in cfg["auto_map"].items()}
        with open(cfg_path, "w") as f:
            json.dump(cfg, f, indent=2)
    return local_dir


def load_model(meta, hf_token=None):
    path = materialize_model(meta["id"], meta.get("code_repo"), hf_token=hf_token)
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        path,
        torch_dtype=torch.bfloat16 if DEVICE == "cuda" else torch.float32,
        trust_remote_code=True,
    ).to(DEVICE).eval()
    print(f"  bos={tokenizer.bos_token_id} eos={tokenizer.eos_token_id} "
          f"mask={tokenizer.mask_token_id} ({tokenizer.mask_token!r})")
    return tokenizer, model


# ---- HF upload helper ------------------------------------------------------

def upload_to_hf(file_path, repo_id, token, path_in_repo=None):
    if not HAS_HF_HUB or not repo_id:
        return
    try:
        api = HfApi(token=token)
        api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True)
        if path_in_repo is None:
            path_in_repo = f"dreamon_java/{os.path.basename(file_path)}"
        api.upload_file(path_or_fileobj=file_path, path_in_repo=path_in_repo,
                        repo_id=repo_id, token=token, repo_type="dataset")
        print(f"    uploaded {os.path.basename(file_path)}")
    except Exception as e:
        print(f"    upload failed: {e}")


# ---- Benchmark one model ---------------------------------------------------

def run_one_model(key, meta, data, args, timestamp):
    print(f"\n{'='*64}\n  Model: {meta['label']}  ({meta['id']})")
    print(f"  gen_mode={meta['gen_mode']}  mask_style={args.mask_style}  "
          f"code_repo={meta.get('code_repo')}\n{'='*64}")
    t0 = time.time()
    tokenizer, model = load_model(meta, hf_token=args.hf_token)
    print(f"  loaded in {time.time()-t0:.1f}s")

    gen_mode = meta["gen_mode"]
    if args.num_mask_per_site:
        num_mask = args.num_mask_per_site
    elif args.mask_style in ("canvas", "placeholder"):
        # canvas length prior = token length of the fine-tuning placeholder
        num_mask = len(tokenizer.encode(args.mask_pattern, add_special_tokens=False))
        print(f"  canvas/site = len(tokenize({args.mask_pattern!r})) = {num_mask} tokens")
    else:
        num_mask = NUM_MASK_PER_SITE[gen_mode]

    if args.debug:
        print("\n[sanity] tiny infill: 'return a + [MASK];'")
        sp, nw = predict_one(model, tokenizer,
                             "public int add(int a, int b) {\n    return a + [MASK];\n}\n",
                             gen_mode, num_mask, 10000, args.max_sites,
                             args.max_new_tokens, args.steps,
                             mask_style=args.mask_style,
                             placeholder=args.mask_pattern, debug=True)
        print(f"[sanity] preds={sp} (any of a/b/1 is fine)\n")

    site_rows, sample_rows = [], []
    site_correct = site_total = 0
    n_first = n_majority = n_any = n_consistent = n_all_correct = 0
    errors = 0

    for idx, row in enumerate(tqdm(data, desc=f"  {meta['label']}")):
        item_id, masked_code, gt = row["id"], row["masked_code"], row["target"]
        n_masks = masked_code.count("[MASK]")
        try:
            site_preds, n_windows = predict_one(
                model, tokenizer, masked_code, gen_mode, num_mask,
                args.context_chars, args.max_sites, args.max_new_tokens,
                args.steps, mask_style=args.mask_style,
                placeholder=args.mask_pattern, debug=(args.debug and idx < 2))

            for s_idx, pred in enumerate(site_preds):
                is_c = (pred == gt)
                site_total += 1
                site_correct += int(is_c)
                site_rows.append({"id": item_id, "site_idx": s_idx,
                                  "ground_truth": gt, "prediction": pred,
                                  "correct": is_c})

            nonempty = [p for p in site_preds if p != ""]
            first_pred = site_preds[0] if site_preds else ""
            if site_preds:
                maj_pred, maj_count = Counter(site_preds).most_common(1)[0]
            else:
                maj_pred, maj_count = "", 0

            first_correct = (first_pred == gt)
            majority_correct = (maj_pred == gt)
            any_correct = any(p == gt for p in site_preds)
            # all-sites consistency: every site produced the SAME identifier.
            consistent = (len(site_preds) > 0 and len(set(site_preds)) == 1)
            # strict EM: ALL sites equal ground truth.
            all_correct = (len(site_preds) > 0 and all(p == gt for p in site_preds))

            n_first += int(first_correct)
            n_majority += int(majority_correct)
            n_any += int(any_correct)
            n_consistent += int(consistent)
            n_all_correct += int(all_correct)

            sample_rows.append({
                "id": item_id, "ground_truth": gt, "n_total_masks": n_masks,
                "n_windows": n_windows, "predictions": "|".join(site_preds),
                "first_pred": first_pred, "first_correct": first_correct,
                "majority_pred": maj_pred, "majority_count": maj_count,
                "majority_correct": majority_correct, "any_correct": any_correct,
                "all_sites_consistent": consistent,
                "all_sites_correct": all_correct,
                "n_distinct_preds": len(set(nonempty)),
            })
        except Exception as e:
            errors += 1
            sample_rows.append({
                "id": item_id, "ground_truth": gt, "n_total_masks": n_masks,
                "n_windows": 0, "predictions": "", "first_pred": "",
                "first_correct": False, "majority_pred": "", "majority_count": 0,
                "majority_correct": False, "any_correct": False,
                "all_sites_consistent": False, "all_sites_correct": False,
                "n_distinct_preds": 0, "error": str(e)})
            if errors <= 5:
                print(f"  Error on {item_id}: {e}")

    # ---- Save ----------------------------------------------------------
    safe = f"{meta['label']}_style-{args.mask_style}"
    if args.start:
        safe += f"_start{args.start}"
    site_file = os.path.join(RESULTS_DIR, f"{safe}_per_site_{timestamp}.csv")
    sample_file = os.path.join(RESULTS_DIR, f"{safe}_per_sample_{timestamp}.csv")

    with open(site_file, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["id", "site_idx", "ground_truth",
                                          "prediction", "correct"])
        w.writeheader()
        w.writerows(site_rows)

    sample_fields = ["id", "ground_truth", "n_total_masks", "n_windows",
                     "predictions", "first_pred", "first_correct",
                     "majority_pred", "majority_count", "majority_correct",
                     "any_correct", "all_sites_consistent", "all_sites_correct",
                     "n_distinct_preds", "error"]
    with open(sample_file, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=sample_fields)
        w.writeheader()
        for r in sample_rows:
            w.writerow({k: r.get(k, "") for k in sample_fields})

    n = len(data)
    site_acc = site_correct / site_total if site_total else 0.0
    print(f"\n=== {meta['label']} (tiled, all-site coverage, gen_mode={gen_mode}, "
          f"mask_style={args.mask_style}) ===")
    print(f"Site-level EM            : {site_correct}/{site_total} = {site_acc:.2%}")
    print(f"First-mask EM            : {n_first}/{n} = {n_first/n:.2%}")
    print(f"Majority-vote EM         : {n_majority}/{n} = {n_majority/n:.2%}")
    print(f"Any-site EM              : {n_any}/{n} = {n_any/n:.2%}")
    print(f"All-sites CONSISTENT     : {n_consistent}/{n} = {n_consistent/n:.2%}  (same name at every site)")
    print(f"All-sites CORRECT (strict): {n_all_correct}/{n} = {n_all_correct/n:.2%}  (every site == ground truth)")
    print(f"Errors                   : {errors}")
    print(f"Per-site : {site_file}")
    print(f"Per-sample: {sample_file}")

    summary = {
        "model": meta["label"], "hf_id": meta["id"], "gen_mode": gen_mode,
        "mask_style": args.mask_style, "mask_pattern": args.mask_pattern,
        "samples": n, "errors": errors,
        "site_acc": f"{site_acc:.4f}", "site_correct": site_correct, "site_total": site_total,
        "first_mask_acc": f"{n_first/n:.4f}",
        "majority_acc": f"{n_majority/n:.4f}",
        "any_acc": f"{n_any/n:.4f}",
        "consistent_rate": f"{n_consistent/n:.4f}",
        "all_correct_acc": f"{n_all_correct/n:.4f}",
        "num_mask_per_site": num_mask, "context_chars": args.context_chars,
        "max_sites": args.max_sites,
        "max_new_tokens": args.max_new_tokens, "steps": args.steps,
        "start": args.start,
        "gen_kwargs": str(DREAMON_GEN_KWARGS if gen_mode == "dreamon" else DREAM_GEN_KWARGS),
        "site_file": site_file, "sample_file": sample_file,
    }

    if args.hf_repo:
        upload_to_hf(site_file, args.hf_repo, args.hf_token)
        upload_to_hf(sample_file, args.hf_repo, args.hf_token)

    del model, tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    return summary


# ---- Main ------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="EM benchmark for AISE DreamOn Java-Identifier models.")
    p.add_argument("--model", action="append", default=None,
                   help=f"One or more of {list(MODEL_REGISTRY)}. Default: both.")
    p.add_argument("--data-path", default=DATA_PATH)
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--mask-style", choices=["canvas", "placeholder", "legacy"],
                   default="canvas",
                   help="How [MASK] sites are presented to the fine-tuned model: "
                        "canvas = <|mask|> run sized by tokenize(__MASKED_VAR__) at "
                        "EVERY site (default); placeholder = canvas at the FIRST "
                        "site per window, literal __MASKED_VAR__ text at the rest; "
                        "legacy = original k=4/k=2 canvases (Jun-07 behaviour).")
    p.add_argument("--mask-pattern", default=MASK_PLACEHOLDER,
                   help="Literal site placeholder used in fine-tuning (default "
                        f"{MASK_PLACEHOLDER!r}). Sets the canvas length prior and "
                        "the context placeholder text.")
    p.add_argument("--num-mask-per-site", type=int, default=None,
                   help="Hard override of mask tokens per site (bypasses the "
                        "mask-style sizing; legacy defaults are 4 dreamon / 2 dream).")
    p.add_argument("--context-chars", type=int, default=CONTEXT_CHARS)
    p.add_argument("--max-sites", type=int, default=MAX_SITES_IN_WINDOW)
    p.add_argument("--max-new-tokens", type=int, default=DREAMON_MAX_NEW_TOKENS,
                   help="DreamOn canvas size after <|expand|> (dreamon mode).")
    p.add_argument("--steps", type=int, default=DREAM_STEPS,
                   help="Fixed-length diffusion steps (dream mode). The 0.5B "
                        "checkpoint's own generation_config says 512.")
    p.add_argument("--alg", choices=["entropy", "origin", "maskgit_plus", "topk_margin"],
                   default=None,
                   help="Override the denoising algorithm for BOTH gen modes "
                        "(0.5B's own generation_config says 'origin').")
    p.add_argument("--temperature", type=float, default=None,
                   help="Override sampling temperature (0.5B's own config says 0).")
    p.add_argument("--top-p", type=float, default=None,
                   help="Override nucleus top_p.")
    p.add_argument("--start", type=int, default=0,
                   help="Skip the first N samples (chunked runs on Colab: "
                        "--start 0/250/500/750 with --max-samples 250; "
                        "merge the chunk CSVs afterwards).")
    p.add_argument("--debug", action="store_true")
    p.add_argument("--hf-repo", default=None)
    p.add_argument("--hf-token", default=os.environ.get("HF_TOKEN"))
    p.add_argument("--list-models", action="store_true")
    args = p.parse_args()

    if args.list_models:
        for k, m in MODEL_REGISTRY.items():
            print(f"  {k:<20} {m['id']}  [{m['gen_mode']}]")
        return

    keys = args.model or list(MODEL_REGISTRY)
    for k in keys:
        if k not in MODEL_REGISTRY:
            sys.exit(f"Unknown model '{k}'. Available: {list(MODEL_REGISTRY)}")

    # Apply sampling overrides to both gen-mode recipes.
    for kw in (DREAMON_GEN_KWARGS, DREAM_GEN_KWARGS):
        if args.alg:
            kw["alg"] = args.alg
        if args.temperature is not None:
            kw["temperature"] = args.temperature
        if args.top_p is not None:
            kw["top_p"] = args.top_p
    if args.alg or args.temperature is not None or args.top_p is not None:
        print(f"Sampling overrides -> dreamon={DREAMON_GEN_KWARGS} dream={DREAM_GEN_KWARGS}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    print(f"Loading data from {args.data_path} ...")
    data = load_data(args.data_path)
    if args.start:
        data = data[args.start:]
    if args.max_samples is not None:
        data = data[:args.max_samples]
    print(f"Loaded {len(data)} samples (start={args.start}) on device={DEVICE}.")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summaries = [run_one_model(k, MODEL_REGISTRY[k], data, args, timestamp) for k in keys]

    summary_file = os.path.join(RESULTS_DIR, f"summary_{timestamp}.csv")
    with open(summary_file, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summaries[0].keys()))
        w.writeheader()
        w.writerows(summaries)
    if args.hf_repo:
        upload_to_hf(summary_file, args.hf_repo, args.hf_token)

    print(f"\n{'='*64}\n  SUMMARY\n{'='*64}")
    print(f"  {'Model':<20}{'site':>7}{'first':>7}{'maj':>7}{'any':>7}{'consist':>9}{'allOK':>7}")
    for s in summaries:
        print(f"  {s['model']:<20}{s['site_acc']:>7}{s['first_mask_acc']:>7}"
              f"{s['majority_acc']:>7}{s['any_acc']:>7}{s['consistent_rate']:>9}"
              f"{s['all_correct_acc']:>7}")
    print(f"\n  Saved: {summary_file}")


if __name__ == "__main__":
    main()
