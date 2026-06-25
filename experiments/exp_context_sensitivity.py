"""
Experiment 2: Context Sensitivity Probe (Phase 2)
==================================================

HYPOTHESIS
----------
High-quality, semantically specific identifiers (userAuthToken, encodedURLStr) are
TIGHTLY BOUND to their surrounding context. If you remove the context, the model
can no longer guess this particular name — the entropy at the target position rises
sharply as the masking fraction α increases.

Generic Code Smell identifiers (temp, data, val) are CONTEXT-AGNOSTIC. Even after
masking 80% of the surrounding tokens, the model still considers these names
equally plausible, so entropy barely changes.

EXPERIMENT DESIGN (directly mirrors partial_masking_noise_map.py)
-----------------------------------------------------------------
For every sample in the test dataset:
  1. Restore the original identifier into the [MASK] slot → full clean code.
  2. Classify the identifier as SMELL / SPECIFIC / AVERAGE.
  3. Mask EXACTLY the target identifier position with <|mask|>.
  4. For α ∈ ALPHA_GRID, randomly mask α fraction of the OTHER context tokens.
  5. Run one forward pass, record:
       H(x^i | C_α)   — Shannon entropy at the target position
       Rank(x^i | C_α) — rank of the gt token in the softmax distribution
       KL(P || Uniform) — model confidence relative to uniform (↑ = more certain)

DIFFERENTIATOR
--------------
  Clean / SPECIFIC identifiers: ΔH = H(α=0.8) − H(α=0.0) >> 0  (steep curve)
  Code Smell / SMELL identifiers: ΔH ≈ 0                         (flat curve)

The key advantage over full-mask diffusion: this experiment operates in the SAME
probability regime as full natural language forward passes (α=0 → α=1 is a
continuous bridge). No "calibration gap" exists.

OUTPUT
------
  results/context_sensitivity/context_probe_{model}_{timestamp}.csv
  results/context_sensitivity/summary_{model}_{timestamp}.csv

Usage:
    python experiments/exp_context_sensitivity.py
    python experiments/exp_context_sensitivity.py --model DiffuCoder-7B-Instruct --max-samples 200
    python experiments/exp_context_sensitivity.py --limit 200 --repeats 3 --seed 42
    python experiments/exp_context_sensitivity.py --list-models
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

# ── torchvision guard (identical to partial_masking_noise_map.py) ──────────────
if "torchvision" in sys.modules and (
        getattr(sys.modules["torchvision"], "__spec__", None) is None):
    del sys.modules["torchvision"]

# ── Configuration ──────────────────────────────────────────────────────────────

ROOT_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH   = os.path.join(ROOT_DIR, "data", "test.csv")
RESULTS_DIR = os.path.join(ROOT_DIR, "results", "context_sensitivity")
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
MAX_TOKS    = 512

# Half-window: center the 512-token budget on [MASK] so it is always visible.
LEFT_CTX    = MAX_TOKS // 2   # 256 tokens of left context

# Masking fraction grid — matches partial_masking_noise_map.py exactly
ALPHA_GRID = [0.0, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50,
              0.60, 0.70, 0.80, 0.90, 0.95, 1.0]

# Code-smell vocabulary
SMELL_VOCAB = {
    "x", "y", "z", "a", "b", "c", "n", "k", "v",
    "i", "j", "l", "m",
    "tmp", "temp", "res", "val", "ret", "obj",
    "data", "buf", "str", "idx", "num", "cnt",
    "foo", "bar", "baz", "result", "value", "item",
    "myVar", "myval", "flag",
}

# ── Model Registry ─────────────────────────────────────────────────────────────

MODEL_REGISTRY = {
    "DiffuCoder-7B-Instruct": {
        "id":         "apple/DiffuCoder-7B-Instruct",
        "type":       "diffucoder",
        "mask_token": "<|mask|>",
    },
    "DiffuCoder-7B-Base": {
        "id":         "apple/DiffuCoder-7B-Base",
        "type":       "diffucoder",
        "mask_token": "<|mask|>",
    },
    "DreamCoder-7B": {
        "id":         "Dream-org/Dream-Coder-v0-Instruct-7B",
        "type":       "dreamcoder",
        "mask_token": "<|mask|>",
    },
}

# ── Identifier classification ──────────────────────────────────────────────────

def classify_identifier(name: str) -> str:
    if name.lower() in SMELL_VOCAB or name in SMELL_VOCAB:
        return "SMELL"
    if len(name) >= 12:
        return "SPECIFIC"
    return "AVERAGE"

# ── Model utilities ────────────────────────────────────────────────────────────

def load_model(model_id: str, mask_token: str):
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        model_id,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if DEVICE == "cuda" else torch.float32,
    ).to(DEVICE).eval()
    mask_id = tokenizer.convert_tokens_to_ids(mask_token)
    return tokenizer, model, mask_id


def single_forward(model, input_ids) -> torch.Tensor:
    """Forward pass — returns logits [1, seq, vocab]."""
    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=None)
    if hasattr(out, "logits"):
        return out.logits
    if isinstance(out, tuple):
        return out[0]
    return out


def metrics_at_position(logits, position: int,
                         target_token_id: int,
                         gt_token_id: int) -> dict:
    """
    Extract per-position metrics from a logits tensor.

    target_token_id : the id we are probing (same as gt_token_id here, since
                      we always evaluate the original name in its original slot)
    gt_token_id     : kept separate for API compatibility with partial_masking_noise_map
    """
    lp     = logits[0, position, :].float()
    probs  = torch.softmax(lp, dim=-1)
    lprobs = torch.log(probs + 1e-12)

    entropy = -(probs * lprobs).sum().item()

    sorted_ids  = torch.argsort(probs, descending=True)
    rank_map    = {int(t): r + 1 for r, t in enumerate(sorted_ids)}
    target_rank = rank_map.get(int(target_token_id), len(rank_map))
    target_prob = float(probs[target_token_id])

    vocab_size    = len(probs)
    kl_vs_uniform = float(
        (probs * (lprobs - np.log(1.0 / vocab_size + 1e-12))).sum()
    )

    return {
        "entropy":            entropy,
        "current_token_rank": target_rank,
        "gt_token_prob":      target_prob,
        "kl_vs_uniform":      kl_vs_uniform,
    }

# ── Partial-masking utility ────────────────────────────────────────────────────

def apply_partial_masking(input_ids: torch.Tensor,
                           mask_id: int,
                           target_idx: int,
                           alpha: float,
                           rng: random.Random) -> torch.Tensor:
    """
    Randomly mask α fraction of tokens that are NOT the target position.

    Directly mirrors apply_partial_masking() in partial_masking_noise_map.py.
    """
    ids = input_ids.clone()
    seq = ids[0].tolist()
    n   = len(seq)

    eligible = [i for i in range(1, n - 1) if i != target_idx]
    k        = int(round(alpha * len(eligible)))
    to_mask  = rng.sample(eligible, k) if k else []
    for i in to_mask:
        ids[0, i] = mask_id

    return ids

# ── Centered tokenisation window ─────────────────────────────────────────────

def build_centered_window(tokenizer, original_code: str,
                           mask_char: int, gt_name: str,
                           mask_id: int) -> tuple:
    """
    Build a 512-token input centered on the identifier.

    Tokenises prefix and suffix separately, keeps last LEFT_CTX prefix tokens
    and first (MAX_TOKS - LEFT_CTX - 1) suffix tokens, inserts <|mask|> at
    the join point.  target_idx is the exact position of the mask token.
    """
    prefix_text = original_code[:mask_char]
    suffix_text = original_code[mask_char + len(gt_name):]

    prefix_ids = tokenizer.encode(prefix_text, add_special_tokens=False)
    suffix_ids = tokenizer.encode(suffix_text, add_special_tokens=False)

    right_ctx  = MAX_TOKS - LEFT_CTX - 1
    prefix_win = prefix_ids[-LEFT_CTX:]
    suffix_win = suffix_ids[:right_ctx]

    bos = ([tokenizer.bos_token_id]
           if getattr(tokenizer, "bos_token_id", None) is not None else [])

    token_seq  = bos + prefix_win + [mask_id] + suffix_win
    target_idx = len(bos) + len(prefix_win)

    input_ids = torch.tensor([token_seq], dtype=torch.long).to(DEVICE)
    return input_ids, target_idx


# ── Data loading ───────────────────────────────────────────────────────────────

def load_data(data_path: str, max_samples: int = None) -> list:
    csv.field_size_limit(sys.maxsize)
    rows = []
    with open(data_path, "r", encoding="utf-8") as f:
        for i, row in enumerate(csv.reader(f)):
            if max_samples and i >= max_samples:
                break
            if len(row) < 3:
                continue
            rows.append({
                "id":          row[0],
                "masked_code": row[1],
                "target":      row[2].strip(),
            })
    return rows

# ── HF upload ─────────────────────────────────────────────────────────────────

def upload_to_hf(file_path, repo_id, token):
    if not HAS_HF_HUB or not repo_id:
        return
    try:
        api = HfApi(token=token)
        api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True)
        api.upload_file(
            path_or_fileobj=file_path,
            path_in_repo=f"context_sensitivity/{os.path.basename(file_path)}",
            repo_id=repo_id,
            repo_type="dataset",
        )
        print(f"    Uploaded → {repo_id}")
    except Exception as e:
        print(f"    Upload failed: {e}")

# ── Core per-sample probe ──────────────────────────────────────────────────────

def probe_sample(tokenizer, model, mask_id,
                 original_code: str,
                 gt_name: str,
                 gt_token_id: int,
                 mask_char: int,
                 sample_id,
                 run_id: int,
                 rng: random.Random) -> list:
    """
    Sweep α from 0 to 1 for a single sample.

    Uses a CENTERED 512-token window so [MASK] is always at position LEFT_CTX,
    regardless of how long the surrounding file is.
    """
    # Build the centered window once; reuse for all α levels
    base_ids, target_idx = build_centered_window(
        tokenizer, original_code, mask_char, gt_name, mask_id
    )

    results = []
    for alpha in ALPHA_GRID:
        probe_ids = apply_partial_masking(base_ids, mask_id, target_idx, alpha, rng)
        try:
            logits  = single_forward(model, probe_ids)
            metrics = metrics_at_position(logits, target_idx, gt_token_id, gt_token_id)
        except Exception as e:
            metrics = {
                "entropy": None,
                "current_token_rank": None,
                "gt_token_prob": None,
                "kl_vs_uniform": None,
            }
        results.append({
            "sample_id":   sample_id,
            "run_id":      run_id,
            "gt_name":     gt_name,
            "alpha":       alpha,
            **metrics,
        })
    return results

# ── Main experiment ────────────────────────────────────────────────────────────

def run_experiment(target_models=None, max_samples=None, repeats=1, seed=42,
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

    print(f"Loading dataset from {DATA_PATH} …")
    data = load_data(DATA_PATH, max_samples)
    print(f"  {len(data)} samples loaded.\n")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    for model_name, meta in models_to_run.items():
        print(f"{'='*62}")
        print(f"  Experiment: Context Sensitivity Probe")
        print(f"  Model:  {model_name}  ({meta['id']})")
        print(f"  α grid: {ALPHA_GRID}")
        print(f"  Repeats per sample: {repeats}")
        print(f"{'='*62}")

        t0 = time.time()
        try:
            tokenizer, model, mask_id = load_model(meta["id"], meta["mask_token"])
            print(f"  Loaded in {time.time()-t0:.1f}s  |  mask_id={mask_id}  |  device={DEVICE}")
        except Exception as e:
            print(f"  FAILED to load: {e}")
            continue

        out_file = os.path.join(
            RESULTS_DIR, f"context_probe_{model_name}_{timestamp}.csv"
        )
        fieldnames = [
            "sample_id", "run_id", "gt_name", "category", "alpha",
            "entropy", "current_token_rank", "gt_token_prob", "kl_vs_uniform",
        ]
        fout   = open(out_file, "w", newline="", encoding="utf-8")
        writer = csv.DictWriter(fout, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()

        total_rows = 0
        for run_id in tqdm(range(repeats), desc=f"  Runs"):
            for row in tqdm(data, desc=f"    Run {run_id+1}/{repeats}", leave=False):
                masked_code = row["masked_code"]
                gt_name     = row["target"]
                category    = classify_identifier(gt_name)

                mask_char = masked_code.find("[MASK]")
                if mask_char == -1:
                    continue
                original_code = masked_code.replace("[MASK]", gt_name, 1)

                gt_tok_ids  = tokenizer.encode(gt_name, add_special_tokens=False)
                gt_token_id = gt_tok_ids[0] if gt_tok_ids else (tokenizer.unk_token_id or 0)

                try:
                    batch = probe_sample(
                        tokenizer, model, mask_id,
                        original_code, gt_name, gt_token_id,
                        mask_char, row["id"], run_id, rng,
                    )
                    # Annotate with category before writing
                    for r in batch:
                        r["category"] = category
                    writer.writerows(batch)
                    total_rows += len(batch)
                    fout.flush()
                except Exception as e:
                    print(f"\n  [Warning] sample {row['id']} run {run_id}: {e}")
                    continue

        fout.close()
        print(f"\n  {total_rows} rows written → {out_file}")

        # ── Quick summary ─────────────────────────────────────────────────────
        df = pd.read_csv(out_file)
        print(f"\n  --- Context Sensitivity Summary  ({model_name}) ---")
        print(f"  Mean entropy by category × alpha (showing α=0.0, 0.5, 0.8, 1.0):")

        pivot = df.groupby(["category", "alpha"])["entropy"].mean().unstack("alpha")
        cols_to_show = [c for c in [0.0, 0.5, 0.8, 1.0] if c in pivot.columns]
        print(pivot[cols_to_show].round(4).to_string())

        # ΔH = H(α=0.8) - H(α=0.0) — the key discriminator
        h0   = df[df.alpha == 0.0].groupby("category")["entropy"].mean()
        h08  = df[df.alpha == 0.8].groupby("category")["entropy"].mean()
        dh   = (h08 - h0).rename("delta_entropy_0.8_vs_0.0")
        print(f"\n  ΔH (α=0.8 − α=0.0)  — larger value = more context-dependent:")
        print(dh.to_string())

        # Save per-model summary
        summary_file = out_file.replace("context_probe_", "summary_")
        df.groupby(["category", "alpha"])["entropy"].agg(["mean", "sem"]).reset_index().to_csv(
            summary_file, index=False
        )
        print(f"\n  Summary → {summary_file}")

        if hf_repo:
            upload_to_hf(out_file, hf_repo, hf_token)
            upload_to_hf(summary_file, hf_repo, hf_token)

        del model, tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Experiment 2 — Context Sensitivity Probe (Partial Masking)"
    )
    parser.add_argument(
        "--model", action="append", default=None,
        help="Model key(s) from MODEL_REGISTRY. Repeatable. Default: all."
    )
    parser.add_argument(
        "--max-samples", type=int, default=None,
        help="Max samples from the dataset (None = full test set)."
    )
    parser.add_argument(
        "--repeats", type=int, default=1,
        help="Number of repeats per sample (for variance estimation). Default: 1."
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for masking."
    )
    parser.add_argument(
        "--hf-repo", type=str, default=None,
        help="HuggingFace dataset repo to upload results."
    )
    parser.add_argument(
        "--hf-token", type=str, default=os.environ.get("HF_TOKEN"),
        help="HF write token (or set HF_TOKEN env var)."
    )
    parser.add_argument(
        "--list-models", action="store_true",
        help="Print available models and exit."
    )
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
        hf_repo=args.hf_repo,
        hf_token=args.hf_token,
    )
