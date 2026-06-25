"""
Experiment 2b: Context Sensitivity Probe — Stratified Design
=============================================================

BACKGROUND (from overconfidence experiment, Exp 1b)
----------------------------------------------------
We now know that gt_rank at α=0 divides samples into three regimes:

  OVERCONFIDENT  (gt_rank ≤ 200)  : model highly certain about gt name
                                     → name is generic-like, context-agnostic
                                     → expect FLAT entropy curve (low ΔH)

  UNCERTAIN      (200 < rank ≤ 1000): middle ground

  CONFIDENT_RARE (gt_rank > 1000) : model uncertain about gt name
                                     → name is context-dependent, specific
                                     → expect STEEP entropy curve (high ΔH)

HYPOTHESIS
----------
Context sensitivity (ΔH = H(α=0.8) − H(α=0.0)) should be:

  • HIGH for CONFIDENT_RARE samples (gt name strongly bound to context)
  • LOW  for OVERCONFIDENT samples  (name is context-agnostic)
  • HIGHEST of all for gt,  LOWEST for smell probes at the same position

EXPERIMENT DESIGN
-----------------
For every sample:
  1. Measure gt_rank at α=0 → classify regime (OVERCONFIDENT / UNCERTAIN / CONFIDENT_RARE)
  2. For BOTH gt token AND smell probes injected at the same position:
       For each α ∈ ALPHA_GRID:
         a. Apply partial masking of the context (excluding target position)
         b. Run forward pass
         c. Record H(x | C_α) — entropy at target, and rank of the probe token

KEY INSIGHT: shared forward pass
---------------------------------
At each α level we only need ONE forward pass.  The entropy H is computed from
the FULL softmax (context-dependent).  We then look up each probe token's rank
and probability from the SAME distribution — no extra GPU calls needed.
This makes the experiment as fast as running it for just gt alone.

METRICS RECORDED
----------------
  entropy        : Shannon entropy of the FULL softmax at the target position
                   (this captures overall uncertainty and does NOT depend on
                    which probe token we are looking at)
  probe_rank     : rank of the specific probe token (gt / smell_severe / …)
  probe_prob     : probability of the specific probe token

KEY SIGNALS
-----------
  ΔH_entropy = H(α=0.8) − H(α=0.0)   → context sensitivity of the POSITION
               (this is the SAME for all probe types at the same sample/alpha)

  Δprob_rank = rank(α=0) − rank(α=0.8) → how much the probe token rises/falls
               in rank when context is stripped (probe-specific)

OUTPUT
------
  results/context_sensitivity_stratified/{model}_{date}.csv
  results/context_sensitivity_stratified/{model}_{date}_summary.csv

Columns in the CSV:
  sample_id, gt_name, regime, gt_rank_base,
  probe_type, probe_name,
  alpha, entropy, probe_rank, probe_prob

Usage:
  python experiments/exp_context_sensitivity_stratified.py
  python experiments/exp_context_sensitivity_stratified.py \\
      --model DiffuCoder-7B-Base --max-samples 200
  python experiments/exp_context_sensitivity_stratified.py \\
      --model DiffuCoder-7B-Base --max-samples 500 \\
      --repeats 3 --seed 42 \\
      --hf-repo anonymous/IdentifierRefactoringRes
  python experiments/exp_context_sensitivity_stratified.py --list-models
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

# ══════════════════════════════════════════════════════════════════════════════
# Configuration
# ══════════════════════════════════════════════════════════════════════════════

ROOT_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH   = os.path.join(ROOT_DIR, "data", "test.csv")
RESULTS_DIR = os.path.join(ROOT_DIR, "results", "context_sensitivity_stratified")
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
MAX_TOKS    = 512
LEFT_CTX    = MAX_TOKS // 2        # 256 tokens — centered window

# Regime thresholds (from Exp-1b pilot: ~52% OVERCONFIDENT, ~30% CONFIDENT_RARE)
THRESH_LOW_DEFAULT  = 200
THRESH_HIGH_DEFAULT = 1000

# Diffusion steps — lower = faster inference, slight quality trade-off.
# 32 steps is enough for single-position logit probing.
NUM_STEPS = 32

# Alpha sweep grid — mirrors partial_masking_noise_map.py
ALPHA_GRID = [0.0, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50,
              0.60, 0.70, 0.80, 0.90, 0.95, 1.0]

# Smell probe sets (same as Exp-1b so signals are directly comparable)
SMELL_PROBES = {
    "severe":   ["x", "a", "n", "i"],
    "moderate": ["tmp", "val", "foo", "res"],
    "mild":     ["myVar", "temp1", "result1", "value1"],
}

# ══════════════════════════════════════════════════════════════════════════════
# Model registry
# ══════════════════════════════════════════════════════════════════════════════

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

# ══════════════════════════════════════════════════════════════════════════════
# Model / tokenisation utilities
# ══════════════════════════════════════════════════════════════════════════════

def load_model(model_id: str, mask_token: str):
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        model_id,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if DEVICE == "cuda" else torch.float32,
    ).to(DEVICE).eval()
    mask_id = tokenizer.convert_tokens_to_ids(mask_token)
    # Reduce internal diffusion steps for faster logit probing.
    # DiffuCoder uses generation_config.steps; patch it here as well as
    # passing num_steps= at call time so both code paths are covered.
    if hasattr(model, "generation_config") and model.generation_config is not None:
        if hasattr(model.generation_config, "steps"):
            model.generation_config.steps = NUM_STEPS
    return tokenizer, model, mask_id


def single_forward(model, input_ids: torch.Tensor,
                   num_steps: int = NUM_STEPS) -> torch.Tensor:
    """Single forward pass — returns logits [1, seq, vocab].

    num_steps controls internal diffusion iterations for models that expose
    this parameter (DiffuCoder, DreamCoder). Lower = faster.
    """
    with torch.no_grad():
        try:
            out = model(input_ids=input_ids,
                        attention_mask=None,
                        num_steps=num_steps)
        except TypeError:
            # Model does not accept num_steps in forward() — fall back.
            out = model(input_ids=input_ids, attention_mask=None)
    if hasattr(out, "logits"):
        return out.logits
    if isinstance(out, tuple):
        return out[0]
    return out


def build_centered_window(tokenizer, original_code: str,
                           mask_char: int, gt_name: str,
                           mask_id: int) -> tuple:
    """
    Build a MAX_TOKS-token input with <|mask|> centred at LEFT_CTX.

    Returns (input_ids [1, seq], target_idx).
    The context window is the same regardless of alpha — alpha masking is
    applied ON TOP of this base window.
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
    target_idx = len(bos) + len(prefix_win)   # exact position of <|mask|>

    input_ids = torch.tensor([token_seq], dtype=torch.long).to(DEVICE)
    return input_ids, target_idx


def apply_partial_masking(base_ids: torch.Tensor,
                           mask_id: int,
                           target_idx: int,
                           alpha: float,
                           rng: random.Random) -> torch.Tensor:
    """Randomly mask α fraction of context tokens (never target_idx)."""
    ids = base_ids.clone()
    n   = ids.shape[1]
    eligible = [i for i in range(1, n - 1) if i != target_idx]
    k        = int(round(alpha * len(eligible)))
    for i in rng.sample(eligible, k) if k else []:
        ids[0, i] = mask_id
    return ids


def entropy_and_probe_stats(logits: torch.Tensor,
                             position: int,
                             probe_token_id: int) -> dict:
    """
    Compute:
      entropy    — Shannon entropy of the full softmax at `position`
                   (this is SHARED across all probe types at α)
      probe_rank — rank of probe_token_id in the softmax
      probe_prob — probability of probe_token_id
    """
    lp     = logits[0, position, :].float()
    probs  = torch.softmax(lp, dim=-1)
    lprobs = torch.log(probs + 1e-12)

    entropy    = float(-(probs * lprobs).sum())
    probe_prob = float(probs[probe_token_id])

    sorted_ids = torch.argsort(probs, descending=True)
    rank_map   = {int(t): r + 1 for r, t in enumerate(sorted_ids)}
    probe_rank = rank_map.get(int(probe_token_id), len(rank_map))

    return {"entropy": entropy, "probe_rank": probe_rank, "probe_prob": probe_prob}


def classify_regime(gt_rank: int, thresh_low: int, thresh_high: int) -> str:
    if gt_rank <= thresh_low:
        return "OVERCONFIDENT"
    if gt_rank <= thresh_high:
        return "UNCERTAIN"
    return "CONFIDENT_RARE"

# ══════════════════════════════════════════════════════════════════════════════
# Data loading
# ══════════════════════════════════════════════════════════════════════════════

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

# ══════════════════════════════════════════════════════════════════════════════
# HF upload
# ══════════════════════════════════════════════════════════════════════════════

def upload_to_hf(path, repo_id, token):
    if not HAS_HF_HUB or not repo_id:
        return
    try:
        api = HfApi(token=token)
        api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True)
        api.upload_file(
            path_or_fileobj=path,
            path_in_repo=f"context_sensitivity_stratified/{os.path.basename(path)}",
            repo_id=repo_id, repo_type="dataset",
        )
        print(f"    Uploaded → {repo_id}")
    except Exception as e:
        print(f"    Upload failed: {e}")

# ══════════════════════════════════════════════════════════════════════════════
# Main experiment
# ══════════════════════════════════════════════════════════════════════════════

def run_experiment(target_models=None, max_samples=None, repeats=1, seed=42,
                   thresh_low=THRESH_LOW_DEFAULT, thresh_high=THRESH_HIGH_DEFAULT,
                   hf_repo=None, hf_token=None):

    os.makedirs(RESULTS_DIR, exist_ok=True)
    rng = random.Random(seed)

    models_to_run = (
        {n: MODEL_REGISTRY[n] for n in target_models if n in MODEL_REGISTRY}
        if target_models else MODEL_REGISTRY
    )
    if not models_to_run:
        print("ERROR: No valid models.")
        return

    print(f"Dataset : {DATA_PATH}")
    data = load_data(DATA_PATH, max_samples)
    print(f"Samples : {len(data)}  |  repeats={repeats}  |  device={DEVICE}")
    print(f"α grid  : {ALPHA_GRID}")
    print(f"Regimes : OVERCONFIDENT(≤{thresh_low}) / UNCERTAIN / CONFIDENT_RARE(>{thresh_high})\n")

    timestamp    = datetime.now().strftime("%Y%m%d_%H%M%S")
    all_summaries = []

    for model_name, meta in models_to_run.items():
        print(f"{'='*68}")
        print(f"  Context Sensitivity Probe (Stratified)  |  {model_name}")
        print(f"{'='*68}")

        t0 = time.time()
        try:
            tokenizer, model, mask_id = load_model(meta["id"], meta["mask_token"])
            print(f"  Loaded in {time.time()-t0:.1f}s  |  mask_id={mask_id}  |  device={DEVICE}")
        except Exception as e:
            print(f"  FAILED to load: {e}")
            continue

        out_file  = os.path.join(RESULTS_DIR, f"{model_name}_{timestamp}.csv")
        fieldnames = [
            "sample_id", "run_id", "gt_name", "gt_len",
            "gt_rank_base",   # rank of gt at α=0 (used for regime classification)
            "regime",         # OVERCONFIDENT / UNCERTAIN / CONFIDENT_RARE
            "probe_type",     # gt / smell_severe / smell_moderate / smell_mild
            "probe_name",     # the actual token being probed
            "alpha",          # masking fraction α
            "entropy",        # H( · | C_α) — position entropy, SHARED across probes
            "probe_rank",     # rank of probe_name in softmax at α
            "probe_prob",     # probability of probe_name at α
        ]
        fout   = open(out_file, "w", newline="", encoding="utf-8")
        writer = csv.DictWriter(fout, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()

        total_rows    = 0
        regime_counts = {"OVERCONFIDENT": 0, "UNCERTAIN": 0, "CONFIDENT_RARE": 0}

        for run_id in range(repeats):
            run_label = f"Run {run_id+1}/{repeats}" if repeats > 1 else model_name
            for row in tqdm(data, desc=f"  {run_label}"):
                masked_code = row["masked_code"]
                gt_name     = row["target"]
                mask_char   = masked_code.find("[MASK]")
                if mask_char == -1:
                    continue

                original_code = masked_code.replace("[MASK]", gt_name, 1)

                # ── Build centered 512-token window ──────────────────────────
                try:
                    base_ids, target_idx = build_centered_window(
                        tokenizer, original_code, mask_char, gt_name, mask_id
                    )
                except Exception:
                    continue

                # ── Probe token ids ───────────────────────────────────────────
                gt_toks      = tokenizer.encode(gt_name, add_special_tokens=False)
                gt_token_id  = gt_toks[0] if gt_toks else (tokenizer.unk_token_id or 0)

                # Build smell probes, guarding against tokenization collision:
                # e.g. 'nPredicates' → first subword is 'n' = smell_severe 'n'
                # If all candidates in a tier collide with gt, skip that tier.
                smell_probe_ids = {}
                for sev, names in SMELL_PROBES.items():
                    candidates = list(names)   # copy so we can shuffle
                    rng.shuffle(candidates)
                    chosen_name, chosen_id = None, None
                    for cand in candidates:
                        toks = tokenizer.encode(cand, add_special_tokens=False)
                        cand_id = toks[0] if toks else (tokenizer.unk_token_id or 0)
                        if cand_id != gt_token_id:
                            chosen_name, chosen_id = cand, cand_id
                            break
                    if chosen_name is None:
                        # All candidates in this tier collide — skip tier
                        continue
                    smell_probe_ids[sev] = (chosen_name, chosen_id)

                # ── PASS 0: α=0 forward pass (no extra context masking) ───────
                # Needed to determine gt_rank for regime classification.
                try:
                    logits_0 = single_forward(model, base_ids)
                    gt_stats_0 = entropy_and_probe_stats(logits_0, target_idx, gt_token_id)
                except Exception:
                    continue

                gt_rank_base = gt_stats_0["probe_rank"]
                regime       = classify_regime(gt_rank_base, thresh_low, thresh_high)
                regime_counts[regime] += 1

                # Build probe list: (probe_type, probe_name, probe_token_id)
                probes = [("gt", gt_name, gt_token_id)]
                for sev, (name, tok_id) in smell_probe_ids.items():
                    probes.append((f"smell_{sev}", name, tok_id))

                # ── Alpha sweep ───────────────────────────────────────────────
                # At each α we do ONE forward pass and look up all probe tokens.
                for alpha in ALPHA_GRID:
                    if alpha == 0.0:
                        # Reuse logits_0 — no masking
                        logits = logits_0
                    else:
                        probe_ids = apply_partial_masking(
                            base_ids, mask_id, target_idx, alpha, rng
                        )
                        try:
                            logits = single_forward(model, probe_ids)
                        except Exception:
                            continue

                    # entropy is position-level → same for all probe tokens
                    lp     = logits[0, target_idx, :].float()
                    probs  = torch.softmax(lp, dim=-1)
                    lprobs = torch.log(probs + 1e-12)
                    entropy = float(-(probs * lprobs).sum())

                    sorted_ids = torch.argsort(probs, descending=True)
                    rank_map   = {int(t): r + 1 for r, t in enumerate(sorted_ids)}

                    for probe_type, probe_name, probe_tok_id in probes:
                        probe_rank = rank_map.get(int(probe_tok_id), len(rank_map))
                        probe_prob = float(probs[probe_tok_id])

                        writer.writerow({
                            "sample_id":    row["id"],
                            "run_id":       run_id,
                            "gt_name":      gt_name,
                            "gt_len":       len(gt_name),
                            "gt_rank_base": gt_rank_base,
                            "regime":       regime,
                            "probe_type":   probe_type,
                            "probe_name":   probe_name,
                            "alpha":        alpha,
                            "entropy":      entropy,
                            "probe_rank":   probe_rank,
                            "probe_prob":   probe_prob,
                        })
                        total_rows += 1
                    fout.flush()

        fout.close()
        print(f"\n  {total_rows} rows written → {out_file}")

        # ── Summary ───────────────────────────────────────────────────────────
        df = pd.read_csv(out_file)

        print(f"\n  {'='*62}")
        print(f"  REGIME DISTRIBUTION")
        print(f"  {'='*62}")
        for regime, cnt in regime_counts.items():
            pct = cnt / len(data) * 100
            bar = "█" * int(pct / 2)
            print(f"  {regime:<18} {cnt:>5} ({pct:5.1f}%)  {bar}")

        # ── ΔH: key metric ────────────────────────────────────────────────────
        # Entropy is the same for all probes at the same (sample_id, alpha),
        # so only use gt rows to compute ΔH (avoids quadruple-counting).
        df_gt = df[df.probe_type == "gt"]
        h0    = df_gt[df_gt.alpha == 0.0].groupby(["sample_id","regime"])["entropy"].mean()
        h08   = df_gt[df_gt.alpha == 0.8].groupby(["sample_id","regime"])["entropy"].mean()
        delta = (h08 - h0).reset_index().rename(columns={"entropy": "delta_H"})

        print(f"\n  {'='*62}")
        print(f"  CONTEXT SENSITIVITY  ΔH = H(α=0.8) − H(α=0.0)")
        print(f"  (larger ΔH = name is MORE tightly bound to context)")
        print(f"  {'='*62}")
        dh_by_regime = delta.groupby("regime")["delta_H"].agg(["mean","sem"]).round(4)
        print(dh_by_regime.to_string())

        print(f"\n  {'='*62}")
        print(f"  PROBE RANK AT α=0 vs α=0.8  (does rank rise when context stripped?)")
        print(f"  (Rising rank at high α = probe token no longer fits → context-bound)")
        print(f"  {'='*62}")
        rank_pivot = (df.groupby(["regime","probe_type","alpha"])["probe_rank"]
                        .mean()
                        .unstack("alpha")
                        .round(0))
        show_alphas = [c for c in [0.0, 0.5, 0.8, 1.0] if c in rank_pivot.columns]
        print(rank_pivot[show_alphas].to_string())

        print(f"\n  {'='*62}")
        print(f"  CORE SIGNAL: ΔH by Regime  (CONFIDENT_RARE should be highest)")
        print(f"  {'='*62}")
        for regime in ["OVERCONFIDENT", "UNCERTAIN", "CONFIDENT_RARE"]:
            row_d = dh_by_regime.loc[regime] if regime in dh_by_regime.index else None
            if row_d is not None:
                bar  = "█" * int(abs(row_d["mean"]) * 30)
                print(f"  {regime:<18}  ΔH = {row_d['mean']:+.4f} ± {row_d['sem']:.4f}  {bar}")
            else:
                print(f"  {regime:<18}  (no samples)")

        # Full entropy curves for each regime (gt only)
        print(f"\n  {'='*62}")
        print(f"  ENTROPY CURVES  H(α) — gt probe only, by regime")
        print(f"  {'='*62}")
        curve = (df_gt.groupby(["regime","alpha"])["entropy"]
                       .mean().unstack("regime").round(4))
        print(curve.to_string())

        # ── Save summary CSV ──────────────────────────────────────────────────
        summary_file = out_file.replace(".csv", "_summary.csv")
        # regime × probe_type × alpha → mean entropy + mean probe_rank
        summ = (df.groupby(["regime","probe_type","alpha"])
                  .agg(
                      entropy_mean   = ("entropy",    "mean"),
                      entropy_sem    = ("entropy",    "sem"),
                      probe_rank_mean= ("probe_rank", "mean"),
                      probe_rank_sem = ("probe_rank", "sem"),
                  )
                  .round(4)
                  .reset_index())
        summ.to_csv(summary_file, index=False)
        print(f"\n  Summary saved → {summary_file}")

        # Global summary row
        rare = delta[delta.regime == "CONFIDENT_RARE"]["delta_H"]
        over = delta[delta.regime == "OVERCONFIDENT"]["delta_H"]
        all_summaries.append({
            "model":           model_name,
            "n_total":         len(data),
            "n_overconfident": regime_counts["OVERCONFIDENT"],
            "n_uncertain":     regime_counts["UNCERTAIN"],
            "n_confident_rare":regime_counts["CONFIDENT_RARE"],
            "dH_overconfident_mean": over.mean()  if len(over)  > 0 else float("nan"),
            "dH_overconfident_sem":  over.sem()   if len(over)  > 1 else float("nan"),
            "dH_confident_rare_mean":rare.mean()  if len(rare)  > 0 else float("nan"),
            "dH_confident_rare_sem": rare.sem()   if len(rare)  > 1 else float("nan"),
            "dH_ratio": rare.mean() / over.mean() if over.mean() > 0 else float("nan"),
        })

        if hf_repo:
            upload_to_hf(out_file,     hf_repo, hf_token)
            upload_to_hf(summary_file, hf_repo, hf_token)

        del model, tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    # ── Global summary ─────────────────────────────────────────────────────────
    global_file = os.path.join(RESULTS_DIR, f"global_summary_{timestamp}.csv")
    if all_summaries:
        pd.DataFrame(all_summaries).to_csv(global_file, index=False)
        print(f"\n{'='*68}")
        print("  GLOBAL SUMMARY: Context Sensitivity Probe (Stratified)")
        print(f"{'='*68}")
        print(f"  {'Model':<32} {'ΔH(OC)':>8} {'ΔH(CR)':>8} {'Ratio':>7}  Interpretation")
        print(f"  {'-'*70}")
        for s in all_summaries:
            ratio_str = f"{s['dH_ratio']:.1f}x" if not np.isnan(s['dH_ratio']) else "—"
            interp    = ("CR >> OC ✓" if not np.isnan(s['dH_ratio']) and s['dH_ratio'] > 1.5
                         else "no clear separation")
            print(f"  {s['model']:<32} {s['dH_overconfident_mean']:>8.4f} "
                  f"{s['dH_confident_rare_mean']:>8.4f} {ratio_str:>7}  {interp}")
        print(f"\n  ΔH(OC) = context sensitivity in OVERCONFIDENT regime (should be LOW)")
        print(f"  ΔH(CR) = context sensitivity in CONFIDENT_RARE regime (should be HIGH)")
        print(f"  Ratio  = ΔH(CR) / ΔH(OC)  → large ratio confirms hypothesis")

        if hf_repo:
            upload_to_hf(global_file, hf_repo, hf_token)

# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Experiment 2b — Context Sensitivity Probe (Stratified)"
    )
    parser.add_argument("--model", action="append", default=None,
        help="Model key(s) from MODEL_REGISTRY. Repeatable. Default: all.")
    parser.add_argument("--max-samples", type=int, default=None,
        help="Max samples (default: full test set).")
    parser.add_argument("--repeats", type=int, default=1,
        help="Repeats per sample for variance estimation (default: 1).")
    parser.add_argument("--seed", type=int, default=42,
        help="RNG seed for partial masking (default: 42).")
    parser.add_argument("--thresh-low", type=int, default=THRESH_LOW_DEFAULT,
        help=f"OVERCONFIDENT upper bound on gt_rank (default: {THRESH_LOW_DEFAULT}).")
    parser.add_argument("--thresh-high", type=int, default=THRESH_HIGH_DEFAULT,
        help=f"CONFIDENT_RARE lower bound on gt_rank (default: {THRESH_HIGH_DEFAULT}).")
    parser.add_argument("--hf-repo", type=str, default=None,
        help="HuggingFace dataset repo to upload results.")
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
        repeats=args.repeats,
        seed=args.seed,
        thresh_low=args.thresh_low,
        thresh_high=args.thresh_high,
        hf_repo=args.hf_repo,
        hf_token=args.hf_token,
    )
