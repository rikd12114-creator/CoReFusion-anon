"""
Experiment: Diffusion Steps Sensitivity on RefineID
=====================================================

MOTIVATION
----------
Diffusion language models (DiffuCoder, DreamCoder) perform iterative
denoising over T steps.  Increasing T costs proportionally more GPU time,
but does extra denoising actually help for identifier renaming?

This experiment holds the *model* fixed (same checkpoint, same number of
parameters) and sweeps only the `steps` argument to `diffusion_generate`.

NULL HYPOTHESIS
---------------
Exact Match Rate is NOT significantly different across the evaluated step
counts.  If confirmed, practitioners can safely use far fewer steps and
trade compute for throughput with negligible accuracy loss.

EXPERIMENT DESIGN
-----------------
For every (model, steps_count) combination:
  1. Run full or sub-sampled refineID benchmark (data/test.csv)
  2. Record per-sample: prediction, correct flag, wall-clock time
  3. Compute aggregate: exact_match_rate, mean_time_per_sample, throughput

STEP GRID
---------
  STEPS_GRID = [1, 2, 4, 8, 16, 32, 64, 128]

  • 1 / 2  steps : quasi-single-forward, maximum throughput
  • 4 / 8  steps : minimal denoising
  • 16 / 32 steps : standard "fast" setting
  • 64 / 128 steps : high-quality baselines

EFFICIENCY METRICS (per step-count)
------------------------------------
  exact_match_rate    : fraction of samples where prediction == ground_truth
  mean_time_per_sample: wall-clock seconds averaged over all samples
  throughput          : samples / second
  relative_speedup    : throughput(steps) / throughput(max_steps)
                        — how many times faster vs. the most expensive config
  accuracy_drop       : exact_match_rate(max_steps) - exact_match_rate(steps)
                        — absolute accuracy cost of the speedup

OUTPUT
------
  results/diffusion_steps_benchmark/{model}_{date}.csv
  results/diffusion_steps_benchmark/summary_{date}.csv

Columns in per-model CSV:
  steps, sample_id, ground_truth, prediction, correct,
  time_per_sample

Columns in summary CSV:
  model, steps, exact_match_rate, correct, total,
  mean_time_per_sample, throughput, relative_speedup, accuracy_drop

Usage:
  # Run both models, full test set, default step grid:
  python experiments/exp_diffusion_steps_benchmark.py

  # Single model, quick test with 100 samples:
  python experiments/exp_diffusion_steps_benchmark.py \\
      --model DiffuCoder-7B-Base --max-samples 100

  # Custom step grid:
  python experiments/exp_diffusion_steps_benchmark.py \\
      --steps 1 4 16 64

  # Upload results to Hugging Face:
  python experiments/exp_diffusion_steps_benchmark.py \\
      --hf-repo anonymous/IdentifierRefactoringRes

  # List available models:
  python experiments/exp_diffusion_steps_benchmark.py --list-models
"""

import os
import sys
import csv
import gc
import re
import argparse
import time
import numpy as np
from datetime import datetime

import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

try:
    from huggingface_hub import HfApi
    HAS_HF_HUB = True
except ImportError:
    HAS_HF_HUB = False

# ── torchvision mock (required for DiffuCoder/DreamCoder) ─────────────────────
class _MockModule:
    def __getattr__(self, name): return _MockModule()
    def __call__(self, *args, **kwargs): return _MockModule()

sys.modules['torchvision'] = _MockModule()
sys.modules['torchvision.ops'] = _MockModule()
sys.modules['torchvision.transforms'] = _MockModule()
if not hasattr(torch.ops, 'torchvision'):
    class _DummyOps:
        def nms(*args, **kwargs): return torch.tensor([])
    torch.ops.torchvision = _DummyOps()

# ══════════════════════════════════════════════════════════════════════════════
# Configuration
# ══════════════════════════════════════════════════════════════════════════════

ROOT_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Try multiple paths for data — works from project root, experiments/, or Colab
_DATA_CANDIDATES = [
    os.path.join(ROOT_DIR, "data", "test.csv"),
    "data/test.csv",
    "CoRefusion/data/test.csv",
]
DATA_PATH   = next((p for p in _DATA_CANDIDATES if os.path.exists(p)),
                   _DATA_CANDIDATES[0])
RESULTS_DIR = os.path.join(ROOT_DIR, "results", "diffusion_steps_benchmark")
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"

# Step grid: span from near single-forward to the standard 128-step setting.
# The LAST element is treated as the "reference" (highest quality) baseline
# for computing speedup and accuracy_drop.
DEFAULT_STEPS_GRID = [1, 2, 4, 8, 16, 32, 64, 128]

# Number of mask tokens substituted per [MASK] site (same as existing scripts)
NUM_MASK_TOKENS = 2

# ── diffusion_generate() keyword arguments ────────────────────────────────────
# These are held constant across all step counts; only `steps` varies.
GEN_KWARGS = dict(
    max_new_tokens=1,   # identifier fills ≈ 1–3 tokens; 1 forces a compact pred
    temperature=0.3,
    top_p=0.95,
    alg="entropy",
    alg_temp=0.,
)

# ══════════════════════════════════════════════════════════════════════════════
# Model registry
# ══════════════════════════════════════════════════════════════════════════════

MODEL_REGISTRY = {
    # Same two models as RQ1 benchmark (benchmark_diffusion_models.py)
    "DiffuCoder-7B": {
        "id":         "apple/DiffuCoder-7B-Base",
        "mask_token": "<|mask|>",
    },
    "DreamCoder-7B": {
        "id":         "Dream-org/Dream-Coder-v0-Instruct-7B",
        "mask_token": "<|mask|>",
    },
}

# ══════════════════════════════════════════════════════════════════════════════
# Utilities
# ══════════════════════════════════════════════════════════════════════════════

def load_model(model_id: str):
    """Load tokenizer + model onto DEVICE."""
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        model_id,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if DEVICE == "cuda" else torch.float32,
    ).to(DEVICE).eval()
    return tokenizer, model


def load_data(data_path: str, max_samples: int = None) -> list:
    """Load data/test.csv → list of dicts {id, masked_code, target}."""
    csv.field_size_limit(sys.maxsize)
    rows = []
    with open(data_path, "r", encoding="utf-8") as f:
        for i, row in enumerate(csv.reader(f)):
            if max_samples is not None and i >= max_samples:
                break
            if len(row) < 3:
                continue
            rows.append({
                "id":          row[0],
                "masked_code": row[1],
                "target":      row[2].strip(),
            })
    return rows


def extract_prediction(full_code: str, masked_code: str) -> str:
    """
    Given the fully denoised `full_code` and the original `masked_code`
    (with [MASK] placeholders), extract the token that filled the first [MASK].

    Strategy: anchor on the text before and after the first [MASK] site,
    identify the gap in full_code, and return the first valid Java identifier.
    """
    parts = masked_code.split("[MASK]", 1)
    if len(parts) < 2:
        return ""

    pre_anchor  = parts[0].strip()[-30:] if len(parts[0].strip()) > 30 else parts[0].strip()
    post_anchor = parts[1].strip()[:30]  if len(parts[1].strip()) > 30 else parts[1].strip()

    idx_start = full_code.find(pre_anchor)
    if idx_start == -1:
        idx_start = 0
    else:
        idx_start += len(pre_anchor)

    idx_end = full_code.find(post_anchor, idx_start) if post_anchor else -1
    gap = full_code[idx_start:idx_end].strip() if idx_end != -1 else full_code[idx_start:idx_start + 60].strip()

    m = re.search(r"[a-zA-Z_$][a-zA-Z0-9_$]*", gap)
    return m.group(0) if m else gap[:20]


def run_single_step(model, tokenizer, mask_token: str, masked_code: str,
                    steps: int) -> tuple:
    """
    Run diffusion_generate for a single sample with the given number of steps.

    Returns (prediction_str, elapsed_seconds).
    Raises on any error so the caller can mark the sample as an error.
    """
    multi_mask  = mask_token * NUM_MASK_TOKENS
    input_code  = masked_code.replace("[MASK]", multi_mask)

    inputs      = tokenizer(input_code, return_tensors="pt")
    input_ids   = inputs.input_ids.to(DEVICE)
    attn_mask   = inputs.attention_mask.to(DEVICE)

    t0 = time.perf_counter()
    with torch.no_grad():
        output = model.diffusion_generate(
            input_ids,
            attention_mask=attn_mask,
            steps=steps,
            **GEN_KWARGS,
        )
    elapsed = time.perf_counter() - t0

    generated_ids = (output.sequences[0]
                     if hasattr(output, "sequences") else output[0])
    full_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    prediction = extract_prediction(full_text, masked_code)
    return prediction, elapsed


def upload_to_hf(path: str, repo_id: str, token: str) -> None:
    """Upload a file to a HuggingFace dataset repository."""
    if not HAS_HF_HUB or not repo_id:
        return
    try:
        api = HfApi(token=token)
        api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True)
        api.upload_file(
            path_or_fileobj=path,
            path_in_repo=f"diffusion_steps_benchmark/{os.path.basename(path)}",
            repo_id=repo_id,
            repo_type="dataset",
        )
        print(f"    Uploaded → {repo_id}")
    except Exception as e:
        print(f"    Upload failed: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# Core experiment
# ══════════════════════════════════════════════════════════════════════════════

def run_experiment(
    target_models=None,
    steps_grid=None,
    max_samples=None,
    hf_repo=None,
    hf_token=None,
    seed=42,
):
    """
    Main entry point.

    For each model in `target_models` (default: all in MODEL_REGISTRY):
      For each step count in `steps_grid`:
        Evaluate exact match rate on refineID.
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)
    torch.manual_seed(seed)

    if steps_grid is None or len(steps_grid) == 0:
        steps_grid = DEFAULT_STEPS_GRID
    steps_grid = sorted(set(int(s) for s in steps_grid))
    ref_steps  = steps_grid[-1]   # the most expensive setting = reference baseline

    models_to_run = (
        {n: MODEL_REGISTRY[n] for n in target_models if n in MODEL_REGISTRY}
        if target_models else MODEL_REGISTRY
    )
    if not models_to_run:
        print("ERROR: No valid models found.  Run with --list-models to see options.")
        return

    print(f"Dataset     : {DATA_PATH}")
    data = load_data(DATA_PATH, max_samples)
    print(f"Samples     : {len(data)}")
    print(f"Step grid   : {steps_grid}  (reference = {ref_steps})")
    print(f"Device      : {DEVICE}")
    print(f"NUM_MASK_TOKS: {NUM_MASK_TOKENS}\n")

    timestamp       = datetime.now().strftime("%Y%m%d_%H%M%S")
    all_summaries   = []   # one row per (model, steps)

    for model_name, meta in models_to_run.items():
        print(f"\n{'='*68}")
        print(f"  Model: {model_name}  ({meta['id']})")
        print(f"{'='*68}")

        # ── Load model ────────────────────────────────────────────────────────
        t_load = time.time()
        try:
            tokenizer, model = load_model(meta["id"])
            print(f"  Loaded in {time.time() - t_load:.1f}s")
        except Exception as exc:
            print(f"  FAILED to load: {exc}")
            continue

        mask_token = meta["mask_token"]

        # Per-model detail CSV (one row per sample × step_count)
        detail_file = os.path.join(RESULTS_DIR, f"{model_name}_{timestamp}.csv")
        detail_fields = [
            "model", "steps",
            "sample_id", "ground_truth", "prediction",
            "correct", "time_per_sample",
        ]
        fout   = open(detail_file, "w", newline="", encoding="utf-8")
        writer = csv.DictWriter(fout, fieldnames=detail_fields, extrasaction="ignore")
        writer.writeheader()

        # ── Step sweep ────────────────────────────────────────────────────────
        step_summaries = {}   # steps → {correct, total, times}

        for steps in steps_grid:
            print(f"\n  ── steps = {steps:>4} ───────────────────────────────────────")

            correct = 0
            errors  = 0
            times   = []

            for row in tqdm(data, desc=f"    steps={steps}", leave=False):
                item_id     = row["id"]
                masked_code = row["masked_code"]
                gt          = row["target"]

                try:
                    prediction, elapsed = run_single_step(
                        model, tokenizer, mask_token, masked_code, steps
                    )
                except Exception as exc:
                    errors += 1
                    if errors <= 3:
                        print(f"\n    [WARN] sample {item_id}: {exc}")
                    prediction = ""
                    elapsed    = float("nan")

                is_correct = (prediction == gt)
                if is_correct:
                    correct += 1
                times.append(elapsed)

                writer.writerow({
                    "model":           model_name,
                    "steps":           steps,
                    "sample_id":       item_id,
                    "ground_truth":    gt,
                    "prediction":      prediction,
                    "correct":         int(is_correct),
                    "time_per_sample": f"{elapsed:.4f}",
                })
                fout.flush()

            valid_times  = [t for t in times if not np.isnan(t)]
            mean_time    = float(np.mean(valid_times))  if valid_times else float("nan")
            throughput   = 1.0 / mean_time              if mean_time > 0  else float("nan")
            em_rate      = correct / len(data)           if data else 0.0

            step_summaries[steps] = {
                "correct":           correct,
                "total":             len(data),
                "errors":            errors,
                "mean_time":         mean_time,
                "throughput":        throughput,
                "exact_match_rate":  em_rate,
                "times":             valid_times,
            }

            print(f"    Exact Match : {correct}/{len(data)}  = {em_rate:.2%}")
            print(f"    Mean time/s : {mean_time:.3f} s")
            print(f"    Throughput  : {throughput:.2f} samples/s")
            if errors:
                print(f"    Errors      : {errors}")

        fout.close()
        print(f"\n  Detail CSV → {detail_file}")

        # ── Compute relative speedup and accuracy drop ─────────────────────────
        ref = step_summaries.get(ref_steps)
        ref_throughput = ref["throughput"] if ref else float("nan")
        ref_em         = ref["exact_match_rate"] if ref else float("nan")

        # ── Print formatted efficiency table ─────────────────────────────────
        print(f"\n  {'='*66}")
        print(f"  DIFFUSION STEP SENSITIVITY — {model_name}")
        print(f"  {'='*66}")
        col_w = [6, 12, 12, 14, 14, 14]
        hdr = (f"  {'Steps':>{col_w[0]}}  "
               f"{'EM Rate':>{col_w[1]}}  "
               f"{'Correct':>{col_w[2]}}  "
               f"{'Time/s':>{col_w[3]}}  "
               f"{'Speedup':>{col_w[4]}}  "
               f"{'EM Drop':>{col_w[5]}}")
        print(hdr)
        print(f"  {'-'*66}")

        for steps in steps_grid:
            s = step_summaries[steps]
            rel_speedup = (s["throughput"] / ref_throughput
                           if ref_throughput > 0 else float("nan"))
            em_drop     = ref_em - s["exact_match_rate"]

            speedup_str = f"{rel_speedup:.2f}×" if not np.isnan(rel_speedup) else "—"
            drop_str    = f"{em_drop:+.2%}"

            bar_fill = int(s["exact_match_rate"] * 20)
            bar      = "█" * bar_fill + "░" * (20 - bar_fill)

            print(
                f"  {steps:>{col_w[0]}}  "
                f"{s['exact_match_rate']:>{col_w[1]}.2%}  "
                f"{s['correct']:>{col_w[2]}}/{s['total']:<5}  "
                f"{s['mean_time']:>{col_w[3]}.3f}s  "
                f"{speedup_str:>{col_w[4]}}  "
                f"{drop_str:>{col_w[5]}}  "
                f"[{bar}]"
            )

        print(f"  {'='*66}")
        print(f"  Reference baseline: steps={ref_steps} "
              f"(EM={ref_em:.2%}, throughput={ref_throughput:.2f} sps)")
        print(f"  Speedup = throughput(steps) / throughput({ref_steps})")
        print(f"  EM Drop = EM({ref_steps}) − EM(steps)  "
              f"[negative means faster setting is BETTER]")

        # ── Collect per-(model, steps) summary rows ────────────────────────────
        for steps in steps_grid:
            s = step_summaries[steps]
            rel_speedup = (s["throughput"] / ref_throughput
                           if ref_throughput > 0 else float("nan"))
            em_drop = ref_em - s["exact_match_rate"]

            all_summaries.append({
                "model":              model_name,
                "model_hf_id":        meta["id"],
                "steps":              steps,
                "exact_match_rate":   round(s["exact_match_rate"], 6),
                "correct":            s["correct"],
                "total":              s["total"],
                "errors":             s["errors"],
                "mean_time_per_sample": round(s["mean_time"], 4),
                "throughput_sps":     round(s["throughput"], 4),
                "relative_speedup":   round(rel_speedup, 4),
                "accuracy_drop":      round(em_drop, 6),
                "is_reference":       int(steps == ref_steps),
            })

        if hf_repo:
            upload_to_hf(detail_file, hf_repo, hf_token)

        # ── Clean up GPU memory between models ────────────────────────────────
        del model, tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    # ── Global summary CSV ─────────────────────────────────────────────────────
    if not all_summaries:
        print("\nNo results collected.")
        return

    summary_file = os.path.join(RESULTS_DIR, f"summary_{timestamp}.csv")
    pd.DataFrame(all_summaries).to_csv(summary_file, index=False)
    print(f"\n  Summary CSV → {summary_file}")

    if hf_repo:
        upload_to_hf(summary_file, hf_repo, hf_token)

    # ── Global print ──────────────────────────────────────────────────────────
    print(f"\n{'='*68}")
    print("  GLOBAL SUMMARY: Diffusion Step Sensitivity on RefineID")
    print(f"{'='*68}")

    df_sum = pd.DataFrame(all_summaries)

    for model_name in df_sum["model"].unique():
        sub = df_sum[df_sum["model"] == model_name].sort_values("steps")
        print(f"\n  ► {model_name}")
        print(f"    {'Steps':>6}  {'EM Rate':>8}  {'Speedup':>8}  {'EM Drop':>8}")
        print(f"    {'-'*40}")
        for _, r in sub.iterrows():
            flag = "  ← reference" if r["is_reference"] else ""
            print(f"    {int(r['steps']):>6}  {r['exact_match_rate']:>8.2%}  "
                  f"{r['relative_speedup']:>8.2f}×  {r['accuracy_drop']:>+8.2%}{flag}")

    # ── Cross-model EM consistency check ─────────────────────────────────────
    print(f"\n{'='*68}")
    print("  EFFICIENCY CONCLUSION")
    print(f"{'='*68}")
    for model_name in df_sum["model"].unique():
        sub = df_sum[df_sum["model"] == model_name].sort_values("steps")
        em_vals   = sub["exact_match_rate"].values
        em_range  = em_vals.max() - em_vals.min()
        best_fast = sub.loc[sub["relative_speedup"].idxmax()]
        print(f"\n  {model_name}:")
        print(f"    EM range across steps : {em_range:.2%}  "
              f"(min={em_vals.min():.2%}, max={em_vals.max():.2%})")
        print(f"    Fastest config (steps={int(best_fast['steps'])}) : "
              f"speedup={best_fast['relative_speedup']:.2f}×, "
              f"EM drop={best_fast['accuracy_drop']:+.2%}")
        note = ("✓ ROBUST — step count does not significantly affect accuracy"
                if em_range < 0.03
                else "△ SENSITIVE — accuracy degrades noticeably at low steps")
        print(f"    Assessment : {note}")

    print(f"\n{'='*68}\n")


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Experiment — Diffusion Step Sensitivity on RefineID (Exact Match)"
    )
    parser.add_argument(
        "--model", action="append", default=None, metavar="MODEL",
        help="Model key(s) from MODEL_REGISTRY. Repeatable. Default: all.",
    )
    parser.add_argument(
        "--steps", nargs="+", type=int, default=None, metavar="N",
        help=f"Diffusion step counts to sweep. Default: {DEFAULT_STEPS_GRID}.",
    )
    parser.add_argument(
        "--max-samples", type=int, default=None, metavar="N",
        help="Truncate dataset to N samples (for quick tests).",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility (default: 42).",
    )
    parser.add_argument(
        "--hf-repo", type=str, default=None, metavar="REPO",
        help="HuggingFace dataset repo ID to upload results (e.g. user/repo).",
    )
    parser.add_argument(
        "--hf-token", type=str, default=os.environ.get("HF_TOKEN"), metavar="TOKEN",
        help="HF write token (or set HF_TOKEN env var).",
    )
    parser.add_argument(
        "--list-models", action="store_true",
        help="Print available models and exit.",
    )
    args = parser.parse_args()

    if args.list_models:
        print("Available models:")
        for k, v in MODEL_REGISTRY.items():
            print(f"  {k:<40} {v['id']}")
        sys.exit(0)

    run_experiment(
        target_models=args.model,
        steps_grid=args.steps,
        max_samples=args.max_samples,
        hf_repo=args.hf_repo,
        hf_token=args.hf_token,
        seed=args.seed,
    )
