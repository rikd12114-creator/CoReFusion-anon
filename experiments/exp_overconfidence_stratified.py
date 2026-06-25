"""
Experiment 1b: Over-confidence Trap — Stratified Design
========================================================

MOTIVATION (from 10-sample pilot analysis)
-------------------------------------------
The naive experiment (compare all gt names vs injected smells) is confounded
because gt names span two radically different regimes:

  REGIME A — Short/generic gt names (style, status, bytes):
    gt_rank ≈ 18–68 → the model is ALREADY over-confident about these.
    The gt name IS the smell. Injecting smell probes adds no information.

  REGIME B — Long/specific gt names (decodedCapacity, methodHandlesLookup):
    gt_rank ≈ 860–4322 → model is uncertain about the exact token.
    When smell probes are injected here, many get rank << gt_rank.
    This IS the Over-confidence Trap.

STRATIFIED DESIGN
-----------------
Step 1 — Measure base gt_rank for every sample (no injection, just mask the
         target position and read the gt token's rank).

Step 2 — Classify each sample into one of three regimes:
    OVERCONFIDENT (gt_rank ≤ THRESH_LOW)   → gt name IS already smell-like
                                              → verify by comparing gt entropy
                                                vs a known-good (rare) token
    UNCERTAIN      (THRESH_LOW < gt_rank ≤ THRESH_HIGH) → ambiguous, keep as control
    CONFIDENT_RARE (gt_rank > THRESH_HIGH)  → specific name, here we expect
                                              smell probes to have LOWER rank
                                              = the over-confidence trap fires

Step 3 — For OVERCONFIDENT and CONFIDENT_RARE samples, inject smell probes
         and measure their ranks at the SAME position and context.

Step 4 — Report:
    a) Distribution of samples across regimes
    b) For CONFIDENT_RARE: mean rank_gt vs mean rank_smell → gap is the signal
    c) For OVERCONFIDENT:  gt already behaves like smell → count as "caught smells"

THRESHOLDS (tunable)
--------------------
THRESH_LOW  = 200   samples where gt_rank ≤ 200 → model very confident in gt
THRESH_HIGH = 1000  samples where gt_rank > 1000 → model genuinely uncertain

METRICS
-------
  base_rank    : rank of probed token (gt or smell) in the softmax distribution
  base_prob    : raw probability of that token
  base_entropy : Shannon entropy at the target position (context-dependent)
  gt_rank_pass1: gt_rank measured in first pass (used for regime classification)

OUTPUT
------
  results/overconfidence_stratified/{model}_{timestamp}.csv
  results/overconfidence_stratified/summary_{timestamp}.csv

Usage:
    python experiments/exp_overconfidence_stratified.py
    python experiments/exp_overconfidence_stratified.py --model DiffuCoder-7B-Base --max-samples 500
    python experiments/exp_overconfidence_stratified.py --thresh-low 200 --thresh-high 1000
    python experiments/exp_overconfidence_stratified.py --list-models
"""

import os
import sys
import csv
import gc
import argparse
import time
import random
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

# ── torchvision guard ──────────────────────────────────────────────────────────
if "torchvision" in sys.modules and (
        getattr(sys.modules["torchvision"], "__spec__", None) is None):
    del sys.modules["torchvision"]

# ── Configuration ──────────────────────────────────────────────────────────────

ROOT_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH   = os.path.join(ROOT_DIR, "data", "test.csv")
RESULTS_DIR = os.path.join(ROOT_DIR, "results", "overconfidence_stratified")
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
MAX_TOKS    = 512
LEFT_CTX    = MAX_TOKS // 2      # 256 tokens of left context (centered window)

# Default regime thresholds — override with --thresh-low / --thresh-high
THRESH_LOW_DEFAULT  = 200
THRESH_HIGH_DEFAULT = 1000

# Diffusion steps — set low for fast logit probing (single position).
# 32 steps is sufficient; the default 256 is ~8x slower with minimal gain.
NUM_STEPS = 32

# Smell probe vocabulary (identical tiers to partial_masking_noise_map.py)
SMELL_PROBES = {
    "severe":   ["x", "a", "n", "i"],
    "moderate": ["tmp", "val", "foo", "res"],
    "mild":     ["myVar", "temp1", "result1", "value1"],
}

# ── Model Registry ─────────────────────────────────────────────────────────────

MODEL_REGISTRY = {
    "DiffuCoder-7B-Instruct": {
        "id":         "apple/DiffuCoder-7B-Instruct",
        "mask_token": "<|mask|>",
    },
    "DiffuCoder-7B-Base": {
        "id":         "apple/DiffuCoder-7B-Base",
        "mask_token": "<|mask|>",
    },
    "DreamCoder-7B": {
        "id":         "Dream-org/Dream-Coder-v0-Instruct-7B",
        "mask_token": "<|mask|>",
    },
}

# ── Utilities ──────────────────────────────────────────────────────────────────

def load_model(model_id: str, mask_token: str):
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        model_id,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if DEVICE == "cuda" else torch.float32,
    ).to(DEVICE).eval()
    mask_id = tokenizer.convert_tokens_to_ids(mask_token)
    # Patch generation_config so diffusion_generate also respects NUM_STEPS
    if hasattr(model, "generation_config") and model.generation_config is not None:
        if hasattr(model.generation_config, "steps"):
            model.generation_config.steps = NUM_STEPS
    return tokenizer, model, mask_id


def single_forward(model, input_ids,
                   num_steps: int = NUM_STEPS) -> torch.Tensor:
    """Forward pass returning logits [1, seq, vocab].

    Tries to pass num_steps to limit internal diffusion iterations
    (DiffuCoder / DreamCoder). Falls back gracefully if not supported.
    """
    with torch.no_grad():
        try:
            out = model(input_ids=input_ids,
                        attention_mask=None,
                        num_steps=num_steps)
        except TypeError:
            out = model(input_ids=input_ids, attention_mask=None)
    if hasattr(out, "logits"):
        return out.logits
    if isinstance(out, tuple):
        return out[0]
    return out


def metrics_at(logits, position: int, token_id: int) -> dict:
    lp     = logits[0, position, :].float()
    probs  = torch.softmax(lp, dim=-1)
    lprobs = torch.log(probs + 1e-12)
    entropy = -(probs * lprobs).sum().item()

    sorted_ids  = torch.argsort(probs, descending=True)
    rank_map    = {int(t): r + 1 for r, t in enumerate(sorted_ids)}
    target_rank = rank_map.get(int(token_id), len(rank_map))
    target_prob = float(probs[token_id])

    return {
        "base_entropy": entropy,
        "base_rank":    target_rank,
        "base_prob":    target_prob,
    }


def build_centered_window(tokenizer, original_code: str,
                           mask_char: int, gt_name: str,
                           mask_id: int) -> tuple:
    """
    Build a MAX_TOKS-length token window centered on the masked identifier.

    Returns (input_ids [1, seq], target_idx).
    """
    prefix_ids = tokenizer.encode(original_code[:mask_char],
                                  add_special_tokens=False)
    suffix_ids = tokenizer.encode(original_code[mask_char + len(gt_name):],
                                  add_special_tokens=False)

    right_ctx  = MAX_TOKS - LEFT_CTX - 1
    prefix_win = prefix_ids[-LEFT_CTX:]
    suffix_win = suffix_ids[:right_ctx]

    bos = ([tokenizer.bos_token_id]
           if getattr(tokenizer, "bos_token_id", None) is not None else [])

    token_seq  = bos + prefix_win + [mask_id] + suffix_win
    target_idx = len(bos) + len(prefix_win)

    input_ids = torch.tensor([token_seq], dtype=torch.long).to(DEVICE)
    return input_ids, target_idx


def classify_regime(gt_rank: int, thresh_low: int, thresh_high: int) -> str:
    """
    Classify a sample into a confidence regime based on the model's certainty
    about the original gt token.

    OVERCONFIDENT  : gt_rank ≤ thresh_low
        → the model ALREADY assigns a high probability to this name
        → the name behaves like a generic/smell token even if it looks clean
        → these samples ARE the over-confidence trap

    UNCERTAIN      : thresh_low < gt_rank ≤ thresh_high
        → middle ground; included in output but not the focus

    CONFIDENT_RARE : gt_rank > thresh_high
        → model is very uncertain about this specific name
        → when smell probes are evaluated here they should get MUCH lower ranks
        → this is where the trap fires most clearly
    """
    if gt_rank <= thresh_low:
        return "OVERCONFIDENT"
    if gt_rank <= thresh_high:
        return "UNCERTAIN"
    return "CONFIDENT_RARE"


def load_data(data_path: str, max_samples: int = None) -> list:
    csv.field_size_limit(sys.maxsize)
    rows = []
    with open(data_path, "r", encoding="utf-8") as f:
        for i, row in enumerate(csv.reader(f)):
            if max_samples and i >= max_samples:
                break
            if len(row) < 3:
                continue
            rows.append({"id": row[0], "masked_code": row[1], "target": row[2].strip()})
    return rows


def upload_to_hf(file_path, repo_id, token):
    if not HAS_HF_HUB or not repo_id:
        return
    try:
        api = HfApi(token=token)
        api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True)
        api.upload_file(
            path_or_fileobj=file_path,
            path_in_repo=f"overconfidence_stratified/{os.path.basename(file_path)}",
            repo_id=repo_id, repo_type="dataset",
        )
        print(f"    Uploaded → {repo_id}")
    except Exception as e:
        print(f"    Upload failed: {e}")

# ── Main experiment ────────────────────────────────────────────────────────────

def run_experiment(target_models=None, max_samples=None,
                   thresh_low=THRESH_LOW_DEFAULT, thresh_high=THRESH_HIGH_DEFAULT,
                   hf_repo=None, hf_token=None):

    os.makedirs(RESULTS_DIR, exist_ok=True)
    rng = random.Random(42)

    models_to_run = (
        {n: MODEL_REGISTRY[n] for n in target_models if n in MODEL_REGISTRY}
        if target_models else MODEL_REGISTRY
    )
    if not models_to_run:
        print("ERROR: No valid models.")
        return

    print(f"Loading dataset: {DATA_PATH}")
    data = load_data(DATA_PATH, max_samples)
    print(f"  {len(data)} samples loaded.\n")
    print(f"  Regime thresholds: OVERCONFIDENT ≤ {thresh_low} < UNCERTAIN ≤ {thresh_high} < CONFIDENT_RARE\n")

    timestamp    = datetime.now().strftime("%Y%m%d_%H%M%S")
    all_summaries = []

    for model_name, meta in models_to_run.items():
        print(f"{'='*66}")
        print(f"  Stratified Over-confidence Trap  |  {model_name}")
        print(f"{'='*66}")

        t0 = time.time()
        try:
            tokenizer, model, mask_id = load_model(meta["id"], meta["mask_token"])
            print(f"  Loaded {meta['id']} in {time.time()-t0:.1f}s  |  device={DEVICE}")
        except Exception as e:
            print(f"  FAILED: {e}")
            continue

        out_file = os.path.join(RESULTS_DIR, f"{model_name}_{timestamp}.csv")
        fieldnames = [
            "id", "gt_name", "gt_len",
            "gt_rank_pass1",        # rank of gt in an unmodified forward pass
            "regime",               # OVERCONFIDENT / UNCERTAIN / CONFIDENT_RARE
            "probe_type",           # gt / smell_severe / smell_moderate / smell_mild
            "probe_name",
            "target_idx",
            "base_entropy", "base_rank", "base_prob",
        ]
        fout   = open(out_file, "w", newline="", encoding="utf-8")
        writer = csv.DictWriter(fout, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()

        total_rows = 0
        regime_counts = {"OVERCONFIDENT": 0, "UNCERTAIN": 0, "CONFIDENT_RARE": 0}

        for row in tqdm(data, desc=f"  {model_name}"):
            masked_code = row["masked_code"]
            gt_name     = row["target"]
            mask_char   = masked_code.find("[MASK]")
            if mask_char == -1:
                continue

            original_code = masked_code.replace("[MASK]", gt_name, 1)

            # ── Build centered 512-token window ─────────────────────────────
            try:
                input_ids, target_idx = build_centered_window(
                    tokenizer, original_code, mask_char, gt_name, mask_id
                )
            except Exception:
                continue

            # ── PASS 1: measure gt baseline (no injection, just mask & predict)
            gt_tok_ids  = tokenizer.encode(gt_name, add_special_tokens=False)
            gt_token_id = gt_tok_ids[0] if gt_tok_ids else (tokenizer.unk_token_id or 0)

            try:
                logits       = single_forward(model, input_ids)
                gt_metrics   = metrics_at(logits, target_idx, gt_token_id)
            except Exception:
                continue

            gt_rank_p1 = gt_metrics["base_rank"]
            regime     = classify_regime(gt_rank_p1, thresh_low, thresh_high)
            regime_counts[regime] += 1

            # ── Build probes list ────────────────────────────────────────────
            # Always include the gt probe (its metrics come from pass 1 above)
            probes = [("gt", gt_name, gt_token_id, gt_metrics)]

            # Inject smell probes for ALL regimes so we have the full picture
            for sev, names in SMELL_PROBES.items():
                smell_name   = rng.choice(names)
                smell_tok    = tokenizer.encode(smell_name, add_special_tokens=False)
                smell_tok_id = smell_tok[0] if smell_tok else (tokenizer.unk_token_id or 0)
                try:
                    # We reuse the same logits tensor — no extra forward pass needed
                    # because we only care about WHICH token's rank we look up in the
                    # SAME softmax distribution (the context didn't change).
                    s_metrics = metrics_at(logits, target_idx, smell_tok_id)
                except Exception:
                    s_metrics = {"base_entropy": None, "base_rank": None, "base_prob": None}
                probes.append((f"smell_{sev}", smell_name, smell_tok_id, s_metrics))

            # ── Write all probe rows for this sample ─────────────────────────
            for probe_type, probe_name, _, probe_metrics in probes:
                writer.writerow({
                    "id":            row["id"],
                    "gt_name":       gt_name,
                    "gt_len":        len(gt_name),
                    "gt_rank_pass1": gt_rank_p1,
                    "regime":        regime,
                    "probe_type":    probe_type,
                    "probe_name":    probe_name,
                    "target_idx":    target_idx,
                    **probe_metrics,
                })
                total_rows += 1

            fout.flush()

        fout.close()
        print(f"\n  {total_rows} rows written → {out_file}")

        # ── Summary ──────────────────────────────────────────────────────────
        df = pd.read_csv(out_file)

        print(f"\n  {'='*60}")
        print(f"  REGIME DISTRIBUTION")
        print(f"  {'='*60}")
        for regime, cnt in regime_counts.items():
            pct = cnt / len(data) * 100
            bar = "█" * int(pct / 2)
            print(f"  {regime:<18} {cnt:>5} ({pct:5.1f}%)  {bar}")

        print(f"\n  {'='*60}")
        print(f"  CORE SIGNAL: Mean Rank by Regime × Probe Type")
        print(f"  (Lower rank = model MORE confident = higher probability)")
        print(f"  {'='*60}")
        pivot = (df.groupby(["regime", "probe_type"])["base_rank"]
                   .mean()
                   .unstack("probe_type")
                   .round(1))
        # Reorder columns
        col_order = [c for c in ["gt","smell_severe","smell_moderate","smell_mild"]
                     if c in pivot.columns]
        print(pivot[col_order].to_string())

        print(f"\n  {'='*60}")
        print(f"  OVER-CONFIDENCE TRAP SIGNAL (CONFIDENT_RARE samples only)")
        print(f"  Hypothesis: smell_rank << gt_rank in this regime")
        print(f"  {'='*60}")
        rare = df[df.regime == "CONFIDENT_RARE"]
        if len(rare) > 0:
            gt_r   = rare[rare.probe_type == "gt"]["base_rank"].mean()
            sev_r  = rare[rare.probe_type == "smell_severe"]["base_rank"].mean()
            mod_r  = rare[rare.probe_type == "smell_moderate"]["base_rank"].mean()
            mild_r = rare[rare.probe_type == "smell_mild"]["base_rank"].mean()
            n_rare = rare["id"].nunique()
            print(f"  n = {n_rare} CONFIDENT_RARE samples")
            print(f"  gt rank   (developer name):  {gt_r:>10.1f}")
            print(f"  smell_severe rank:           {sev_r:>10.1f}   ratio = {gt_r/max(sev_r,1):.1f}x")
            print(f"  smell_moderate rank:         {mod_r:>10.1f}   ratio = {gt_r/max(mod_r,1):.1f}x")
            print(f"  smell_mild rank:             {mild_r:>10.1f}   ratio = {gt_r/max(mild_r,1):.1f}x")
            trap_fires = sev_r < gt_r
            print(f"\n  → Over-confidence trap {'CONFIRMED ✓' if trap_fires else 'NOT confirmed ✗'}")
            print(f"    (smell_severe rank {'<' if trap_fires else '>'} gt rank)")
        else:
            print(f"  No CONFIDENT_RARE samples found. Lower --thresh-high.")

        print(f"\n  {'='*60}")
        print(f"  OVERCONFIDENT SAMPLES (gt already smells like a smell)")
        print(f"  {'='*60}")
        over = df[(df.regime == "OVERCONFIDENT") & (df.probe_type == "gt")]
        if len(over) > 0:
            print(f"  {len(over)} samples where model is already over-confident about gt name:")
            print(f"  {'gt_name':<30} {'gt_rank':>9} {'gt_len':>7}")
            print(f"  {'-'*50}")
            for _, r in over.sort_values("base_rank").head(20).iterrows():
                print(f"  {r.gt_name:<30} {r.base_rank:>9.0f} {r.gt_len:>7.0f}")

        # Save per-model summary
        summary_file = out_file.replace(".csv", "_summary.csv")
        summary = (df.groupby(["regime", "probe_type"])["base_rank"]
                     .agg(["mean", "median", "std"])
                     .round(2)
                     .reset_index())
        summary.to_csv(summary_file, index=False)
        print(f"\n  Summary → {summary_file}")

        # Global row
        rare_gt   = rare[rare.probe_type=="gt"]["base_rank"].mean() if len(rare) > 0 else np.nan
        rare_sev  = rare[rare.probe_type=="smell_severe"]["base_rank"].mean() if len(rare) > 0 else np.nan
        n_rare    = rare["id"].nunique() if len(rare) > 0 else 0
        n_over    = regime_counts["OVERCONFIDENT"]
        all_summaries.append({
            "model":         model_name,
            "n_total":       len(data),
            "n_overconfident": n_over,
            "n_uncertain":   regime_counts["UNCERTAIN"],
            "n_confident_rare": n_rare,
            "rare_gt_rank":  rare_gt,
            "rare_sev_rank": rare_sev,
            "trap_ratio":    rare_gt / max(rare_sev, 1) if not np.isnan(rare_gt) else np.nan,
        })

        if hf_repo:
            upload_to_hf(out_file,     hf_repo, hf_token)
            upload_to_hf(summary_file, hf_repo, hf_token)

        del model, tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    # ── Global summary ─────────────────────────────────────────────────────────
    global_summary = os.path.join(RESULTS_DIR, f"global_summary_{timestamp}.csv")
    if all_summaries:
        pd.DataFrame(all_summaries).to_csv(global_summary, index=False)
        print(f"\n{'='*66}")
        print("  GLOBAL SUMMARY: Stratified Over-confidence Trap")
        print(f"{'='*66}")
        print(f"  {'Model':<32} {'Rare N':>7} {'GT Rank':>9} {'Sev Rank':>9} {'Ratio':>7}")
        print(f"  {'-'*68}")
        for s in all_summaries:
            ratio = f"{s['trap_ratio']:.1f}x" if not np.isnan(s.get('trap_ratio', np.nan)) else "—"
            print(f"  {s['model']:<32} {s['n_confident_rare']:>7} "
                  f"{s['rare_gt_rank']:>9.1f} {s['rare_sev_rank']:>9.1f} {ratio:>7}")
        print(f"\n  Trap ratio = gt_rank / sev_rank in CONFIDENT_RARE samples.")
        print(f"  Larger ratio = stronger evidence of over-confidence trap.")
        if hf_repo:
            upload_to_hf(global_summary, hf_repo, hf_token)

# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Experiment 1b — Stratified Over-confidence Trap"
    )
    parser.add_argument("--model", action="append", default=None,
        help="Model key(s) from MODEL_REGISTRY. Repeatable. Default: all.")
    parser.add_argument("--max-samples", type=int, default=None,
        help="Max samples (default: full test set).")
    parser.add_argument("--thresh-low", type=int, default=THRESH_LOW_DEFAULT,
        help=f"Upper rank bound for OVERCONFIDENT regime (default: {THRESH_LOW_DEFAULT}).")
    parser.add_argument("--thresh-high", type=int, default=THRESH_HIGH_DEFAULT,
        help=f"Lower rank bound for CONFIDENT_RARE regime (default: {THRESH_HIGH_DEFAULT}).")
    parser.add_argument("--hf-repo", type=str, default=None,
        help="HuggingFace dataset repo ID to upload results.")
    parser.add_argument("--hf-token", type=str, default=os.environ.get("HF_TOKEN"),
        help="HF write token (or set HF_TOKEN env var).")
    parser.add_argument("--list-models", action="store_true",
        help="Print available models and exit.")
    args = parser.parse_args()

    if args.list_models:
        print("Available models:")
        for k, v in MODEL_REGISTRY.items():
            print(f"  {k:<40} {v['id']}")
        sys.exit(0)

    run_experiment(
        target_models=args.model,
        max_samples=args.max_samples,
        thresh_low=args.thresh_low,
        thresh_high=args.thresh_high,
        hf_repo=args.hf_repo,
        hf_token=args.hf_token,
    )
