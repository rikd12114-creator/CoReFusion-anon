"""
Unified refineID benchmark + consistency-gated metric runner for ALL paper models.

Re-runs inference (PER-SITE predictions, needed for the all-sites consistency
gate) for every model in the paper, then scores each with the new identifier
metrics (analysis/identifier_similarity_metrics.py). Everything is written to a
single fixed folder:

    results/unified_refineID/
        predictions/<Model>.csv     id, ground_truth, n_total_masks,
                                    predictions (|-joined per site),
                                    first_pred, first_correct, error
        metrics/<Model>_persample.csv
        leaderboard.csv             one row per model: consistency_rate +
                                    M1/M2/M3 (consistent subset & gated)
        manifest.csv                per-model run status

It reuses the EXACT per-sample inference functions from the existing benchmark
scripts so the numbers stay consistent with the paper:
    fim       -> benchmark_ar_models_fim.run_fim_on_sample
    t5        -> benchmark_t5_models.run_t5_on_sample
    codet5p   -> benchmark_codet5p_16b.run_codet5p_large_on_sample
    diffusion -> benchmark_diffusion_models.extract_all_predictions (+ inline fwd)
    dreamon   -> benchmark_dreamon.predict_one
    dgemma    -> benchmark_diffusiongemma.run_dgemma_on_sample (prompted per-site)

Consistency = HARD GATE: a sample is usable only if every [MASK] site emits the
SAME non-empty identifier (else the renamed code won't compile). Metrics score
the single agreed name; inconsistent samples are failures. See
analysis/identifier_similarity_metrics.py.

Usage (GPU server / the HPC cluster):
    # everything (sequential; long -- prefer per-arch jobs below):
    python experiments/run_all_refineID_unified.py
    # one architecture family at a time (good for separate SLURM jobs):
    python experiments/run_all_refineID_unified.py --arch "Decoder-only"
    python experiments/run_all_refineID_unified.py --arch "dLLM (fixed-canvas)"
    # specific models / quick smoke test / resume / metrics-only:
    python experiments/run_all_refineID_unified.py --only DreamOn-7B --only DiffuCoder-7B
    python experiments/run_all_refineID_unified.py --max-samples 20
    python experiments/run_all_refineID_unified.py --resume
    python experiments/run_all_refineID_unified.py --skip-inference   # re-score existing predictions only
    python experiments/run_all_refineID_unified.py --list
"""

import os
import sys
import csv
import gc
import argparse
import time
import traceback

# torch is imported lazily (inside inference functions) so that --list and
# --skip-inference (pure-Python metrics) work on machines without torch/GPU.

# import paths: this file lives in experiments/; analysis/ is a sibling.
_HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(_HERE)
for p in (_HERE, os.path.join(REPO, "analysis")):
    if p not in sys.path:
        sys.path.insert(0, p)

DATA_PATH = os.path.join(REPO, "data", "test.csv")
OUT_DIR = os.path.join(REPO, "results", "unified_refineID")
PRED_DIR = os.path.join(OUT_DIR, "predictions")
MET_DIR = os.path.join(OUT_DIR, "metrics")

_DEVICE = None


def dev():
    global _DEVICE
    if _DEVICE is None:
        import torch
        _DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    return _DEVICE


# ---------------------------------------------------------------------------
# Model registry -- all paper models. engine selects the inference path.
# ---------------------------------------------------------------------------

REG = {
    # --- dLLM fixed-canvas (single-pass, replace [MASK] with <|mask|>*2) ----
    "DreamCoder-7B": dict(engine="diffusion", arch="dLLM (fixed-canvas)", params="7B",
                          id="Dream-org/Dream-Coder-v0-Instruct-7B", mask_token="<|mask|>"),
    "DiffuCoder-7B": dict(engine="diffusion", arch="dLLM (fixed-canvas)", params="7B",
                          id="apple/DiffuCoder-7B-Base", mask_token="<|mask|>"),
    # --- dLLM variable-canvas (tiled all-site DreamOn) ---------------------
    "DreamOn-7B": dict(engine="dreamon", arch="dLLM (variable canvas)", params="7B",
                       id="Dream-org/DreamOn-v0-7B"),
    # --- dLLM block-AR (canvas appended after prompt; prompted per-site fill,
    #     NOT in-place infill -- footnote the protocol difference) ----------
    "DiffusionGemma-26B-A4B": dict(engine="dgemma", arch="dLLM (block-AR)", params="26B-A4B",
                                   id="google/diffusiongemma-26B-A4B-it", max_ctx=16384),
    # --- Decoder-only (FIM) ------------------------------------------------
    "CodeLlama-13B": dict(engine="fim", arch="Decoder-only", params="13B",
                          id="codellama/CodeLlama-13b-hf", model_type="codellama", max_ctx=16384),
    "CodeLlama-7B": dict(engine="fim", arch="Decoder-only", params="7B",
                         id="codellama/CodeLlama-7b-hf", model_type="codellama", max_ctx=16384),
    "StarCoder2-15B": dict(engine="fim", arch="Decoder-only", params="15B",
                           id="bigcode/starcoder2-15b", model_type="starcoder", max_ctx=16384),
    "StarCoder2-7B": dict(engine="fim", arch="Decoder-only", params="7B",
                          id="bigcode/starcoder2-7b", model_type="starcoder", max_ctx=16384),
    "StarCoder2-3B": dict(engine="fim", arch="Decoder-only", params="3B",
                          id="bigcode/starcoder2-3b", model_type="starcoder", max_ctx=16384),
    "DeepSeek-Coder-6.7B": dict(engine="fim", arch="Decoder-only", params="6.7B",
                                id="deepseek-ai/deepseek-coder-6.7b-base", model_type="deepseek", max_ctx=16384),
    "DeepSeek-Coder-1.3B": dict(engine="fim", arch="Decoder-only", params="1.3B",
                                id="deepseek-ai/deepseek-coder-1.3b-base", model_type="deepseek", max_ctx=16384),
    "Qwen2.5-Coder-14B": dict(engine="fim", arch="Decoder-only", params="14B",
                              id="Qwen/Qwen2.5-Coder-14B", model_type="qwen25coder", max_ctx=32768),
    "Qwen2.5-Coder-7B": dict(engine="fim", arch="Decoder-only", params="7B",
                             id="Qwen/Qwen2.5-Coder-7B", model_type="qwen25coder", max_ctx=32768),
    "Qwen2.5-Coder-3B": dict(engine="fim", arch="Decoder-only", params="3B",
                             id="Qwen/Qwen2.5-Coder-3B", model_type="qwen25coder", max_ctx=32768),
    "Qwen2.5-Coder-1.5B": dict(engine="fim", arch="Decoder-only", params="1.5B",
                               id="Qwen/Qwen2.5-Coder-1.5B", model_type="qwen25coder", max_ctx=32768),
    "CodeGemma-7B": dict(engine="fim", arch="Decoder-only", params="7B",
                         id="google/codegemma-7b", model_type="codegemma", max_ctx=8192),
    "CodeGemma-2B": dict(engine="fim", arch="Decoder-only", params="2B",
                         id="google/codegemma-2b", model_type="codegemma", max_ctx=8192),
    # --- Encoder-decoder: CodeT5+ large (prefix-completion) ----------------
    "CodeT5p-16B": dict(engine="codet5p", arch="Encoder-decoder", params="16B",
                        id="Salesforce/codet5p-16b", max_ctx=2048),
    "CodeT5p-6B": dict(engine="codet5p", arch="Encoder-decoder", params="6B",
                       id="Salesforce/codet5p-6b", max_ctx=2048),
    "CodeT5p-2B": dict(engine="codet5p", arch="Encoder-decoder", params="2B",
                       id="Salesforce/codet5p-2b", max_ctx=2048),
    # --- Encoder-decoder: CodeT5 (sentinel) --------------------------------
    "CodeT5-large": dict(engine="t5", arch="Encoder-decoder", params="770M",
                         id="Salesforce/codet5-large", max_ctx=512),
    "CodeT5-base": dict(engine="t5", arch="Encoder-decoder", params="220M",
                        id="Salesforce/codet5-base", max_ctx=512),
    "CodeT5-small": dict(engine="t5", arch="Encoder-decoder", params="60M",
                         id="Salesforce/codet5-small", max_ctx=512),
}

_ENGINE_CACHE = {}


def _engine(name):
    if name not in _ENGINE_CACHE:
        if name == "fim":
            import benchmark_ar_models_fim as m
        elif name == "t5":
            import benchmark_t5_models as m
        elif name == "codet5p":
            import benchmark_codet5p_16b as m
        elif name == "diffusion":
            import benchmark_diffusion_models as m
        elif name == "dreamon":
            import benchmark_dreamon as m
        elif name == "dgemma":
            import benchmark_diffusiongemma as m
        else:
            raise ValueError(name)
        _ENGINE_CACHE[name] = m
    return _ENGINE_CACHE[name]


# ---------------------------------------------------------------------------
# Loading + per-sample inference dispatch
# ---------------------------------------------------------------------------

def load_model(meta, hf_token=None):
    import torch
    eng, hf_id = meta["engine"], meta["id"]
    device = dev()
    bf16 = torch.bfloat16 if device == "cuda" else torch.float32
    if eng == "fim":
        from transformers import AutoTokenizer, AutoModelForCausalLM
        tok = AutoTokenizer.from_pretrained(hf_id, trust_remote_code=True, token=hf_token)
        model = AutoModelForCausalLM.from_pretrained(
            hf_id, torch_dtype=bf16, device_map="auto" if device == "cuda" else None,
            trust_remote_code=True, token=hf_token)
        if device == "cpu":
            model = model.to("cpu")
    elif eng == "t5":
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        tok = AutoTokenizer.from_pretrained(hf_id, trust_remote_code=True, token=hf_token)
        model = AutoModelForSeq2SeqLM.from_pretrained(
            hf_id, torch_dtype=bf16, device_map="auto" if device == "cuda" else None,
            trust_remote_code=True, token=hf_token)
        if device == "cpu":
            model = model.to("cpu")
    elif eng == "codet5p":
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        e = _engine("codet5p")
        tok = AutoTokenizer.from_pretrained(hf_id, trust_remote_code=True, token=hf_token)
        config = e.load_codet5p_config(hf_id)
        model = AutoModelForSeq2SeqLM.from_pretrained(
            hf_id, config=config,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            low_cpu_mem_usage=True, trust_remote_code=True, token=hf_token).to(device)
        # transformers>=4.50: generate() would re-derive the generation config
        # from CodeT5pConfig and die on its encoder/decoder assertion.
        e.ensure_generate_compatible(model)
    elif eng == "dgemma":
        # AutoProcessor stands in for tok; needs transformers w/ diffusion_gemma
        # (pylibs_dgemma on the HPC cluster, NOT the 4.57.1 Dream/DreamOn tree).
        tok, model = _engine("dgemma").load_diffusiongemma(hf_id, hf_token=hf_token)
    else:  # diffusion / dreamon
        from transformers import AutoTokenizer, AutoModel
        tok = AutoTokenizer.from_pretrained(hf_id, trust_remote_code=True, token=hf_token)
        model = AutoModel.from_pretrained(
            hf_id, torch_dtype=bf16, trust_remote_code=True, token=hf_token).to(device)
    model.eval()
    return tok, model


def _diffusion_sample(masked_code, model, tok, mask_token, num_mask=2, steps=64):
    import torch
    e = _engine("diffusion")
    input_code = masked_code.replace("[MASK]", mask_token * num_mask)
    inputs = tok(input_code, return_tensors="pt")
    ids = inputs.input_ids.to(model.device)
    am = inputs.attention_mask.to(model.device)
    with torch.no_grad():
        out = model.diffusion_generate(
            ids, attention_mask=am, max_new_tokens=1, steps=steps,
            temperature=0.3, top_p=0.95, alg="entropy", alg_temp=0.)
    seq = out.sequences[0] if hasattr(out, "sequences") else out[0]
    full = tok.decode(seq, skip_special_tokens=True)
    return e.extract_all_predictions(full, masked_code)


def infer_sample(meta, masked_code, model, tok):
    eng = meta["engine"]
    if eng == "fim":
        return _engine("fim").run_fim_on_sample(
            masked_code, model, tok, meta["model_type"], meta["max_ctx"])[0]
    if eng == "t5":
        return _engine("t5").run_t5_on_sample(masked_code, model, tok, meta["max_ctx"])[0]
    if eng == "codet5p":
        return _engine("codet5p").run_codet5p_large_on_sample(
            masked_code, model, tok, meta["max_ctx"])[0]
    if eng == "diffusion":
        return _diffusion_sample(masked_code, model, tok, meta["mask_token"])
    if eng == "dreamon":
        return _engine("dreamon").predict_one(model, tok, masked_code)[0]
    if eng == "dgemma":
        return _engine("dgemma").run_dgemma_on_sample(
            masked_code, model, tok, meta["max_ctx"])[0]
    raise ValueError(eng)


# ---------------------------------------------------------------------------
# Data + inference loop
# ---------------------------------------------------------------------------

def load_data(max_samples=None):
    csv.field_size_limit(2**31 - 1)
    rows = []
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        for i, row in enumerate(csv.reader(f)):
            if max_samples is not None and i >= max_samples:
                break
            if len(row) < 3:
                continue
            rows.append({"id": row[0], "masked_code": row[1], "target": row[2].strip()})
    return rows


def run_inference(name, meta, data, hf_token=None, debug=False):
    """Run a model over the dataset and write its per-site prediction CSV."""
    import torch
    from tqdm import tqdm
    print(f"\n{'='*64}\n  {name}  [{meta['arch']} / {meta['params']} / {meta['engine']}]\n  {meta['id']}\n{'='*64}")
    t0 = time.time()
    tok, model = load_model(meta, hf_token=hf_token)
    print(f"  loaded in {time.time()-t0:.1f}s on {dev()}")

    out_path = os.path.join(PRED_DIR, f"{name}.csv")
    fields = ["id", "ground_truth", "n_total_masks", "predictions",
              "first_pred", "first_correct", "error"]
    n_err = 0
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for idx, row in enumerate(tqdm(data, desc=f"  {name}")):
            gt = row["target"]
            n_masks = row["masked_code"].count("[MASK]")
            try:
                preds = infer_sample(meta, row["masked_code"], model, tok)
                first = preds[0] if preds else ""
                w.writerow({"id": row["id"], "ground_truth": gt, "n_total_masks": n_masks,
                            "predictions": "|".join(preds), "first_pred": first,
                            "first_correct": (first == gt), "error": ""})
                if debug and idx < 2:
                    print(f"    [{row['id']}] gt={gt!r} preds={preds}")
            except Exception as ex:
                n_err += 1
                w.writerow({"id": row["id"], "ground_truth": gt, "n_total_masks": n_masks,
                            "predictions": "", "first_pred": "", "first_correct": False,
                            "error": str(ex)[:200]})
                if n_err <= 3:
                    print(f"    error on {row['id']}: {ex}")
            f.flush()

    del model, tok
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    print(f"  wrote {out_path}  (errors={n_err})")
    return out_path, n_err


# ---------------------------------------------------------------------------
# Metrics + leaderboard
# ---------------------------------------------------------------------------

def score_predictions(name, meta, pred_path, dictionary):
    import identifier_similarity_metrics as M
    res = M.evaluate_csv(pred_path, dictionary)
    if res is None:
        return None
    summary, rows = res
    os.makedirs(MET_DIR, exist_ok=True)
    persample = os.path.join(MET_DIR, f"{name}_persample.csv")
    with open(persample, "w", newline="", encoding="utf-8") as f:
        keys = list(rows[0].keys())
        wr = csv.DictWriter(f, fieldnames=keys)
        wr.writeheader()
        wr.writerows(rows)
    summary["model"] = name
    summary["arch"] = meta["arch"]
    summary["params"] = meta["params"]
    summary["engine"] = meta["engine"]
    return summary


LEADERBOARD_COLS = [
    "model", "arch", "params", "engine", "n", "n_consistent", "consistency_rate",
    "em_consistent", "lev_sim_consistent", "nw_sim_consistent",
    "subtok_jaccard_consistent", "subtok_fuzzy_consistent", "qual_char_consistent",
    "em_gated", "lev_sim_gated", "subtok_fuzzy_gated", "qual_char_gated",
    "gt_qual_char",
]


def write_leaderboard(summaries):
    summaries = sorted(summaries, key=lambda s: s["consistency_rate"], reverse=True)
    path = os.path.join(OUT_DIR, "leaderboard.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=LEADERBOARD_COLS, extrasaction="ignore")
        w.writeheader()
        for s in summaries:
            w.writerow(s)
    print(f"\n{'='*100}")
    print("  UNIFIED LEADERBOARD  (sorted by all-sites consistency rate)")
    print(f"{'='*100}")
    print(f"{'model':<22}{'arch':<22}{'consist%':>9}{'EM_cons':>8}"
          f"{'lev':>6}{'fuzzy':>6}{'qualC':>7}{'EM_gat':>8}")
    print("  " + "-" * 96)
    for s in summaries:
        print(f"{s['model']:<22}{s['arch'][:21]:<22}{s['consistency_rate']*100:>8.1f}%"
              f"{s['em_consistent']:>8.3f}{s['lev_sim_consistent']:>6.3f}"
              f"{s['subtok_fuzzy_consistent']:>6.3f}{s['qual_char_consistent']:>7.3f}"
              f"{s['em_gated']:>8.3f}")
    print(f"\n  Full table -> {path}")
    return path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", action="append", default=None, help="model name(s), repeatable")
    ap.add_argument("--arch", action="append", default=None, help="filter by arch label")
    ap.add_argument("--engine", action="append", default=None, help="filter by engine")
    ap.add_argument("--max-samples", type=int, default=None)
    ap.add_argument("--resume", action="store_true",
                    help="skip inference for models whose prediction CSV already exists")
    ap.add_argument("--skip-inference", action="store_true",
                    help="only (re)compute metrics from existing prediction CSVs")
    ap.add_argument("--dict", default="/usr/share/dict/words")
    ap.add_argument("--hf-token", default=os.environ.get("HF_TOKEN"))
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()

    if args.list:
        for n, m in REG.items():
            print(f"  {n:<22} {m['arch']:<22} {m['params']:<6} {m['engine']:<10} {m['id']}")
        return

    selected = list(REG)
    if args.only:
        selected = [n for n in selected if n in set(args.only)]
    if args.arch:
        selected = [n for n in selected if REG[n]["arch"] in set(args.arch)]
    if args.engine:
        selected = [n for n in selected if REG[n]["engine"] in set(args.engine)]
    if not selected:
        sys.exit("No models selected. Use --list to see names.")

    os.makedirs(PRED_DIR, exist_ok=True)
    os.makedirs(MET_DIR, exist_ok=True)

    import identifier_similarity_metrics as M
    dictionary = M.load_dictionary(args.dict)
    dev_str = "metrics-only" if args.skip_inference else dev()
    print(f"Models: {len(selected)} | device={dev_str} | dict={len(dictionary)} words")
    print(f"Output -> {OUT_DIR}")

    data = None if args.skip_inference else load_data(args.max_samples)
    if data is not None:
        print(f"Loaded {len(data)} samples from {DATA_PATH}")

    manifest, summaries = [], []
    for name in selected:
        meta = REG[name]
        pred_path = os.path.join(PRED_DIR, f"{name}.csv")
        status, n_err = "ok", 0
        try:
            if args.skip_inference:
                if not os.path.exists(pred_path):
                    print(f"  [skip] {name}: no prediction CSV at {pred_path}")
                    manifest.append(dict(model=name, engine=meta["engine"],
                                         status="missing_predictions", n_err=""))
                    continue
            elif args.resume and os.path.exists(pred_path):
                print(f"  [resume] {name}: reuse existing {pred_path}")
            else:
                pred_path, n_err = run_inference(name, meta, data,
                                                 hf_token=args.hf_token, debug=args.debug)
            summary = score_predictions(name, meta, pred_path, dictionary)
            if summary:
                summaries.append(summary)
                print(f"  {name}: consist={summary['consistency_rate']:.1%} "
                      f"EM_cons={summary['em_consistent']:.3f} lev={summary['lev_sim_consistent']:.3f} "
                      f"fuzzy={summary['subtok_fuzzy_consistent']:.3f} qualC={summary['qual_char_consistent']:.3f}")
            else:
                status = "no_rows"
        except Exception as ex:
            status = f"FAILED: {ex}"
            print(f"  !! {name} failed: {ex}")
            if args.debug:
                traceback.print_exc()
        manifest.append(dict(model=name, engine=meta["engine"], status=status, n_err=n_err))

    # manifest
    with open(os.path.join(OUT_DIR, "manifest.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["model", "engine", "status", "n_err"])
        w.writeheader()
        w.writerows(manifest)

    if summaries:
        write_leaderboard(summaries)
    print(f"\nManifest -> {os.path.join(OUT_DIR, 'manifest.csv')}")
    failed = [m for m in manifest if m["status"] != "ok"]
    if failed:
        print(f"  {len(failed)} model(s) not ok: " + ", ".join(f"{m['model']}({m['status'][:20]})" for m in failed))


if __name__ == "__main__":
    main()
