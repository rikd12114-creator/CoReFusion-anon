"""
Supplementary Experiment: Noise Mapping via Partial Masking Interpolation
=========================================================================

WHY THIS EXPERIMENT EXISTS — The calibration gap problem
---------------------------------------------------------
In the main experiment, the diffusion calibration curve operates at entropy
levels of 1.64–1.69 nats (the model sees mostly masked tokens), while all
code smell groups cluster at 0.27–0.56 nats (the model sees full real tokens).

These two entropy ranges are NON-OVERLAPPING. This means we CANNOT directly
map "which diffusion step does a code smell correspond to?" using entropy
comparison alone — the calibration ruler doesn't reach the smell region.

The root cause: the diffusion process in DiffuCoder is highly non-linear.
The model resolves most uncertainty in a single step (step 1 → step 0),
leaving the entire semantic variation space of code quality operating in
a completely different probability regime from the masking regime.

PROPOSED SOLUTION — Partial Masking Interpolation
--------------------------------------------------
Instead of asking "where on the calibration curve does smell entropy fall?",
we ask a better question:

    "What fraction of tokens need to be randomly masked ALONGSIDE the
     smell token before the model treats the position the same way
     it treats a fully masked position?"

This creates a bridge between the two regimes by gradually mixing:
    - α = 0.0: only the smell token is present (smell probe)
    - α = 0.5: 50% of context randomly masked + smell token present
    - α = 1.0: all context masked (equivalent to full diffusion step 0)

The α at which smell entropy MATCHES mask entropy gives us the
"equivalent partial-context noise level" of a code smell.

EXPERIMENT DESIGN
-----------------
For each sample in the dataset:
  1. Take the clean code snippet with a smell identifier (e.g. `tmp`)
  2. For each masking fraction α ∈ {0.0, 0.1, 0.2, ..., 1.0}:
       a. Randomly mask α% of NON-TARGET context tokens
       b. Run one model forward pass
       c. Record entropy at the target (smell) position
  3. Find α* where entropy(smell, α) ≈ entropy(full_mask, α=1.0)
  4. Compare α* across severity levels and with clean code

ADDITIONAL METRIC: KL-Divergence Fingerprint
---------------------------------------------
At each α level, compute the KL divergence between the model's output
distribution for the smell token and for the mask token:

    KL(P_smell(α) || P_mask(α))

If smell tokens reduce to noise as masking fraction increases, we expect
KL → 0 as α → 1.0. The rate of KL decay characterises the "noise distance"
of each smell severity level.
"""

import os, sys, json, re, random
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from datetime import datetime
from tqdm import tqdm

# ── Remove any broken torchvision mock BEFORE importing transformers ──────
# transformers checks torchvision.__spec__ via importlib; a bare ModuleType
# has __spec__=None which raises ValueError. Solution: don't mock it at all —
# just let transformers handle the missing package gracefully on its own.
if "torchvision" in sys.modules and sys.modules["torchvision"].__spec__ is None:
    del sys.modules["torchvision"]

from transformers import AutoTokenizer, AutoModel

# ─────────────────────────────── Config ──────────────────────────────────────
DATA_PATH   = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "..", "data", "test_filtered_1024.csv")
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "..", "results", "noise_mapping_supplementary")
os.makedirs(RESULTS_DIR, exist_ok=True)

MODEL_CFG = {
    "id":         "apple/DiffuCoder-7B-Instruct",
    "mask_token": "<|mask|>",
}

# Masking fraction grid
ALPHA_GRID = [0.0, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50,
              0.60, 0.70, 0.80, 0.90, 0.95, 1.0]

# Smell sets (one token per severity, kept simple for clean tokenisation)
BAD_NAMES = {
    "severe":   ["x", "a", "n", "i"],
    "moderate": ["tmp", "val", "foo", "res"],
    "mild":     ["myVar", "temp1", "result1", "value1"],
}

LIMIT    = 50    # samples — override via --limit
REPEATS  = 3     # per sample — override via --repeats
MAX_TOKS = 512
TEMPERATURE = 0.0   # greedy for reproducibility
DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"


# ─────────────────────────────── Utilities ───────────────────────────────────

def load_model(model_id: str, mask_token: str):
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model     = AutoModel.from_pretrained(
        model_id, trust_remote_code=True,
        torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
    ).to(DEVICE).eval()
    mask_id = tokenizer.convert_tokens_to_ids(mask_token)
    assert mask_id is not None and mask_id != tokenizer.unk_token_id, \
        f"Mask token '{mask_token}' not found in vocabulary"
    return tokenizer, model, mask_id


def single_forward(model, input_ids, attention_mask):
    """One model forward pass → (logits: Tensor[1, seq, vocab])."""
    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=attention_mask)
    if hasattr(out, "logits"):
        return out.logits
    if hasattr(out, "last_hidden_state"):
        # Fallback for models that expose a LM head separately
        return out.last_hidden_state
    raise AttributeError("Cannot extract logits from model output")


def extract_metrics_at_position(logits, position: int, current_token_id: int,
                                 gt_token_id: int):
    """
    Given logits [1, seq, vocab], extract per-position metrics.

    Returns dict with:
      entropy           — Shannon entropy of the softmax distribution (nats)
      current_token_rank — rank of the current (possibly smell) token
      gt_token_rank     — rank of the ground-truth token
      gt_token_prob     — probability of the ground-truth token
      kl_vs_uniform     — KL divergence from the uniform distribution
                          (higher → model is more "sure", lower → more noise-like)
    """
    logit_pos = logits[0, position, :]            # (vocab,)
    probs = torch.softmax(logit_pos.float(), dim=-1)

    # Entropy
    log_probs = torch.log(probs + 1e-12)
    entropy   = -(probs * log_probs).sum().item()

    # Ranks (1-indexed, lower is better)
    sorted_ids = torch.argsort(probs, descending=True)
    rank_map   = {int(tid): rank+1 for rank, tid in enumerate(sorted_ids)}

    current_rank = rank_map.get(int(current_token_id), len(rank_map))
    gt_rank      = rank_map.get(int(gt_token_id),      len(rank_map))
    gt_prob      = float(probs[gt_token_id])

    # KL from uniform
    vocab_size    = len(probs)
    uniform_prob  = 1.0 / vocab_size
    kl_vs_uniform = float(
        (probs * (log_probs - np.log(uniform_prob + 1e-12))).sum()
    )

    return {
        "entropy":            entropy,
        "current_token_rank": current_rank,
        "gt_token_rank":      gt_rank,
        "gt_token_prob":      gt_prob,
        "kl_vs_uniform":      kl_vs_uniform,   # higher → more confident → less noisy
    }


def apply_partial_masking(input_ids: torch.Tensor,
                           mask_id:   int,
                           target_idx: int,
                           alpha:      float,
                           rng:        random.Random) -> torch.Tensor:
    """
    Randomly mask α fraction of context tokens (all except the target position
    and special tokens at the beginning/end).

    The target position itself is NOT masked here — it stays as the smell
    token (or clean token) throughout, so we measure how contextual masking
    changes the model's assessment of that token.
    """
    ids   = input_ids.clone()
    seq   = ids[0].tolist()
    n     = len(seq)

    # Positions eligible for masking (not the target, not position 0 or n-1)
    eligible = [i for i in range(1, n-1) if i != target_idx]
    k        = int(round(alpha * len(eligible)))
    to_mask  = rng.sample(eligible, k) if k else []

    for i in to_mask:
        ids[0, i] = mask_id

    return ids


# ─────────────────────────────── Core Experiment ─────────────────────────────

def probe_sample_with_alpha_sweep(
    tokenizer, model, mask_id,
    original_code: str,
    gt_name: str,
    mask_start_char: int,
    sample_id: int,
    run_id: int,
    rng: random.Random,
):
    """
    For a single sample, sweep alpha from 0 to 1 for each smell severity
    and for the clean (control) condition.

    Returns a list of dicts — one per (severity, bad_name, alpha) combination.
    """
    results = []

    # ── tokenise the original (clean) code first ───────────────────────────
    enc = tokenizer(
        original_code,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_TOKS,
    ).to(DEVICE)
    original_ids   = enc["input_ids"]
    attention_mask = enc["attention_mask"]
    seq_len        = original_ids.shape[1]

    # Find which token position corresponds to the target identifier
    # We do this by comparing prefix lengths
    prefix    = original_code[:mask_start_char]
    prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
    target_idx = min(len(prefix_ids), seq_len - 2)   # clamp to valid range

    # Ground-truth token id
    gt_ids = tokenizer.encode(gt_name, add_special_tokens=False)
    gt_token_id = gt_ids[0] if gt_ids else tokenizer.unk_token_id

    # ── loop over severities ────────────────────────────────────────────────
    conditions = [
        ("control", "none", gt_name, gt_token_id),
    ]
    for sev, names in BAD_NAMES.items():
        bad_name = rng.choice(names)
        bad_ids  = tokenizer.encode(bad_name, add_special_tokens=False)
        bad_tok  = bad_ids[0] if bad_ids else tokenizer.unk_token_id
        conditions.append((f"smell_{sev}", sev, bad_name, bad_tok))

    for group_key, severity, token_str, token_id in conditions:
        # Replace the target position with the smell / clean token
        probe_ids = original_ids.clone()
        probe_ids[0, target_idx] = token_id

        for alpha in ALPHA_GRID:
            # Apply contextual masking
            masked_ids = apply_partial_masking(
                probe_ids, mask_id, target_idx, alpha, rng
            )

            try:
                logits  = single_forward(model, masked_ids, attention_mask)
                metrics = extract_metrics_at_position(
                    logits, target_idx, token_id, gt_token_id
                )
            except Exception as e:
                metrics = {
                    "entropy": None, "current_token_rank": None,
                    "gt_token_rank": None, "gt_token_prob": None,
                    "kl_vs_uniform": None,
                }

            results.append({
                "sample_id":          sample_id,
                "run_id":             run_id,
                "group":              group_key,
                "severity":           severity,
                "token":              token_str,
                "gt_name":            gt_name,
                "alpha":              alpha,
                "context_masked_pct": alpha * 100,
                **metrics,
            })

    return results


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Supplementary Experiment: Noise Mapping via Partial Masking"
    )
    parser.add_argument("--limit",   type=int, default=LIMIT,   help="Number of samples")
    parser.add_argument("--repeats", type=int, default=REPEATS,  help="Repeats per sample")
    parser.add_argument("--seed",    type=int, default=42,       help="Random seed")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    np.random.seed(args.seed)

    ts        = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_csv = os.path.join(RESULTS_DIR, f"partial_masking_probe_{ts}.csv")

    print("=" * 64)
    print(" Supplementary Experiment: Partial Masking Noise Mapping")
    print("=" * 64)
    print(f"  Model:   {MODEL_CFG['id']}")
    print(f"  Samples: {args.limit}  |  Repeats: {args.repeats}")
    print(f"  Alpha grid: {ALPHA_GRID}")
    print(f"  Output:  {output_csv}")
    print()

    # Load data
    print("Loading dataset …")
    df = pd.read_csv(DATA_PATH, header=None, names=["id", "X", "y"],
                     nrows=args.limit)
    print(f"  Loaded {len(df)} samples.")

    # Load model
    print(f"\nLoading model {MODEL_CFG['id']} …")
    tokenizer, model, mask_id = load_model(MODEL_CFG["id"], MODEL_CFG["mask_token"])
    print(f"  Model loaded on {DEVICE}, mask_id={mask_id}")

    # Open CSV writer
    fieldnames = [
        "sample_id", "run_id", "group", "severity", "token", "gt_name",
        "alpha", "context_masked_pct",
        "entropy", "current_token_rank", "gt_token_rank",
        "gt_token_prob", "kl_vs_uniform",
    ]
    import csv
    fout = open(output_csv, "w", newline="")
    writer = csv.DictWriter(fout, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()

    total_rows = 0

    for run_id in tqdm(range(args.repeats), desc="Runs"):
        for _, row in tqdm(df.iterrows(), total=len(df),
                           desc=f"Run {run_id+1}/{args.repeats}", leave=False):
            code_with_mask = str(row["X"])
            gt_name        = str(row["y"])
            sample_id      = row["id"]

            # Find where [MASK] appears in the original code
            mask_pos = code_with_mask.find("[MASK]")
            if mask_pos == -1:
                continue
            original_code = code_with_mask.replace("[MASK]", gt_name, 1)

            try:
                batch_results = probe_sample_with_alpha_sweep(
                    tokenizer, model, mask_id,
                    original_code, gt_name, mask_pos,
                    sample_id, run_id, rng,
                )
                writer.writerows(batch_results)
                total_rows += len(batch_results)
                fout.flush()
            except Exception as e:
                print(f"\n  [Warning] sample {sample_id} run {run_id}: {e}")
                continue

    fout.close()
    print(f"\n  Done. {total_rows} rows written → {output_csv}")

    # ── Quick summary ──────────────────────────────────────────────────────
    print("\nRunning quick summary …")
    data = pd.read_csv(output_csv)
    summary = data.groupby(["group", "alpha"])["entropy"].agg(["mean", "sem"]).reset_index()
    print(summary.to_string(index=False))

    # Save summary
    summary_path = output_csv.replace(".csv", "_summary.csv")
    summary.to_csv(summary_path, index=False)
    print(f"\n  Summary saved → {summary_path}")
    print(f"\nRun analysis:  python analysis/analyze_partial_masking.py --input {output_csv}")


if __name__ == "__main__":
    main()
