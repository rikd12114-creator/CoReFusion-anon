"""
Part 1: Token Length Analysis of Ground-Truth Variable Names
=============================================================

Reads the RefineID dataset (data/test.csv) and computes, under BOTH
the DiffuCoder  (apple/DiffuCoder-7B-Instruct) and the DreamCoder
(Dream-org/Dream-Coder-v0-Instruct-7B) tokenizers:

  • Mean character length of the GT variable names
  • Mean token length of the GT variable names
  • Full token-length distribution (histogram + CDF saved as PNG/PDF)
  • Detailed percentile table (P10, P25, P50, P75, P90, P95, P99)
  • Summary CSV saved to results/1t5t_exp/part1_token_length_analysis.csv

Usage (from repo root):
    python experiments/1t5t_exp/part1_token_length_analysis.py \
        --data data/test.csv \
        --max-samples 5000

The --max-samples flag is optional (default: use all data). Useful for a
quick sanity-check run (e.g., --max-samples 500).
"""

import os
import sys
import csv
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # headless
import matplotlib.pyplot as plt
from collections import Counter
from datetime import datetime

# ---------------------------------------------------------------------------
# torchvision mock (required to import transformers on machines that have a
# conflicting torchvision version)
# ---------------------------------------------------------------------------
class _MockModule:
    def __getattr__(self, name): return _MockModule()
    def __call__(self, *a, **k): return _MockModule()

import sys as _sys, torch as _torch
_sys.modules.setdefault("torchvision", _MockModule())
_sys.modules.setdefault("torchvision.ops", _MockModule())
_sys.modules.setdefault("torchvision.transforms", _MockModule())
if not hasattr(_torch.ops, "torchvision"):
    class _DummyOps:
        def nms(*a, **k): return _torch.tensor([])
    _torch.ops.torchvision = _DummyOps()
# ---------------------------------------------------------------------------

from transformers import AutoTokenizer

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────
MODELS = {
    "DiffuCoder-7B": "apple/DiffuCoder-7B-Instruct",
    "DreamCoder-7B": "Dream-org/Dream-Coder-v0-Instruct-7B",
}

DEFAULT_DATA    = "data/test.csv"
RESULTS_DIR     = "results/1t5t_exp"
FIGURES_DIR     = os.path.join(RESULTS_DIR, "figures")

plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update({"font.size": 12, "axes.titlesize": 14, "figure.dpi": 180})

COLORS = {
    "DiffuCoder-7B": "#3a86ff",
    "DreamCoder-7B": "#ff6b6b",
}

# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_ground_truth_names(data_path: str, max_samples: int | None = None) -> list[str]:
    """
    Load ground-truth variable names from test.csv.
    Expected format (no header):  id | masked_code | ground_truth
    """
    csv.field_size_limit(sys.maxsize)
    names = []
    with open(data_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 3:
                continue
            gt = row[2].strip()
            if gt and gt.lower() not in ("ground_truth", "target"):
                names.append(gt)
            if max_samples and len(names) >= max_samples:
                break
    return names


# ─────────────────────────────────────────────────────────────────────────────
# Analysis helpers
# ─────────────────────────────────────────────────────────────────────────────

def compute_token_lengths(names: list[str], tokenizer) -> np.ndarray:
    """Tokenise each name individually and return an array of token counts."""
    lengths = []
    for name in names:
        toks = tokenizer.encode(name, add_special_tokens=False)
        lengths.append(len(toks))
    return np.array(lengths)


def percentile_table(lengths: np.ndarray) -> pd.DataFrame:
    pcts = [10, 25, 50, 75, 90, 95, 99]
    rows = [{
        "percentile": p,
        "token_length": int(np.percentile(lengths, p)),
    } for p in pcts]
    return pd.DataFrame(rows)


def token_count_distribution(lengths: np.ndarray) -> pd.DataFrame:
    counter = Counter(lengths.tolist())
    total   = len(lengths)
    rows = []
    for k in sorted(counter.keys()):
        rows.append({
            "num_tokens": k,
            "count": counter[k],
            "fraction": counter[k] / total,
        })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Plotting helpers
# ─────────────────────────────────────────────────────────────────────────────

def plot_histogram(data: dict[str, np.ndarray], out_prefix: str):
    """Side-by-side histograms of token-length distributions."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=False)

    for ax, (model_name, lengths) in zip(axes, data.items()):
        max_len = int(np.percentile(lengths, 99)) + 1
        bins = range(1, max_len + 2)
        ax.hist(lengths, bins=bins, color=COLORS[model_name], alpha=0.8,
                edgecolor="white", linewidth=0.4)
        ax.axvline(lengths.mean(), color="black", ls="--", lw=1.8,
                   label=f"Mean = {lengths.mean():.2f}")
        ax.axvline(np.median(lengths), color="orange", ls=":", lw=1.8,
                   label=f"Median = {int(np.median(lengths))}")
        ax.set_title(f"{model_name}\nToken-Length Distribution", fontweight="bold")
        ax.set_xlabel("Number of tokens per variable name")
        ax.set_ylabel("Count")
        ax.legend(fontsize=10)
        ax.set_xticks(range(1, max_len + 1))

    plt.suptitle("GT Variable Name Token-Length Distribution\n(RefineID test set)",
                 fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(f"{out_prefix}_histogram.png", bbox_inches="tight")
    plt.savefig(f"{out_prefix}_histogram.pdf", bbox_inches="tight")
    plt.close()
    print(f"  → Saved: {out_prefix}_histogram.png")


def plot_cdf(data: dict[str, np.ndarray], out_prefix: str):
    """Overlaid CDF for all models."""
    fig, ax = plt.subplots(figsize=(8, 5))
    for model_name, lengths in data.items():
        sorted_l = np.sort(lengths)
        cdf = np.arange(1, len(sorted_l) + 1) / len(sorted_l)
        ax.step(sorted_l, cdf, where="post", color=COLORS[model_name],
                lw=2, label=model_name)
        mean_v = lengths.mean()
        ax.axvline(mean_v, color=COLORS[model_name], ls="--", lw=1.2, alpha=0.6)

    ax.set_xlabel("Number of tokens per variable name")
    ax.set_ylabel("Cumulative fraction")
    ax.set_title("CDF of Variable Name Token Lengths", fontweight="bold")
    ax.legend(fontsize=11)
    ax.set_xlim(left=0)
    ax.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.savefig(f"{out_prefix}_cdf.png", bbox_inches="tight")
    plt.savefig(f"{out_prefix}_cdf.pdf", bbox_inches="tight")
    plt.close()
    print(f"  → Saved: {out_prefix}_cdf.png")


def plot_stacked_bar(dist_data: dict[str, pd.DataFrame], out_prefix: str):
    """
    Grouped bar chart: fraction of variable names whose token length = k,
    for k = 1, 2, 3, 4, 5, 6+ — for each tokenizer.
    """
    label_map = {1: "1", 2: "2", 3: "3", 4: "4", 5: "5", 6: "6+"}

    def bucket(k):
        return k if k <= 5 else 6

    models = list(dist_data.keys())
    buckets = [1, 2, 3, 4, 5, 6]
    x = np.arange(len(buckets))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, model_name in enumerate(models):
        df = dist_data[model_name]
        total = df["count"].sum()
        fracs = []
        for b in buckets:
            if b == 6:
                cnt = df[df["num_tokens"] >= 6]["count"].sum()
            else:
                row = df[df["num_tokens"] == b]
                cnt = row["count"].sum() if len(row) else 0
            fracs.append(cnt / total * 100)
        offset = (i - len(models) / 2 + 0.5) * width
        bars = ax.bar(x + offset, fracs, width, color=COLORS[model_name],
                      label=model_name, alpha=0.85)
        for bar, val in zip(bars, fracs):
            if val > 1.5:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                        f"{val:.1f}%", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels([label_map[b] for b in buckets])
    ax.set_xlabel("Number of tokens per variable name")
    ax.set_ylabel("Percentage of variable names (%)")
    ax.set_title("Token-Length Distribution of GT Variable Names\n(bucketed at 6+)",
                 fontweight="bold")
    ax.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(f"{out_prefix}_bar.png", bbox_inches="tight")
    plt.savefig(f"{out_prefix}_bar.pdf", bbox_inches="tight")
    plt.close()
    print(f"  → Saved: {out_prefix}_bar.png")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Part 1 – Analyse GT variable-name token lengths."
    )
    parser.add_argument("--data", default=DEFAULT_DATA,
                        help=f"Path to test.csv  (default: {DEFAULT_DATA})")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="If set, use only the first N samples.")
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("=" * 65)
    print("  Part 1 – GT Variable Name Token-Length Analysis")
    print("=" * 65)

    # 1. Load GT names
    print(f"\n[1] Loading ground-truth names from  {args.data} …")
    names = load_ground_truth_names(args.data, args.max_samples)
    print(f"    Loaded {len(names):,} variable names.")
    print(f"    Char-length:  mean={np.mean([len(n) for n in names]):.2f}  "
          f"median={np.median([len(n) for n in names]):.0f}  "
          f"max={max(len(n) for n in names)}")

    # 2. Tokenise under each model's tokeniser
    all_lengths: dict[str, np.ndarray] = {}
    all_dist:    dict[str, pd.DataFrame] = {}
    summary_rows = []

    for model_friendly, model_id in MODELS.items():
        print(f"\n[2] Loading tokeniser for  {model_id} …")
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

        print(f"    Tokenising {len(names):,} names …")
        lengths = compute_token_lengths(names, tokenizer)
        all_lengths[model_friendly] = lengths

        dist_df = token_count_distribution(lengths)
        all_dist[model_friendly] = dist_df

        pct_df = percentile_table(lengths)

        print(f"\n  ── {model_friendly} ──")
        print(f"     Mean token length : {lengths.mean():.3f}")
        print(f"     Median            : {int(np.median(lengths))}")
        print(f"     Std               : {lengths.std():.3f}")
        print(f"     Min / Max         : {lengths.min()} / {lengths.max()}")
        print(f"\n  Percentile table:")
        print(pct_df.to_string(index=False))

        print(f"\n  Token-count distribution (top rows):")
        print(dist_df.head(10).to_string(index=False))

        # Save per-model distribution CSV
        dist_path = os.path.join(RESULTS_DIR, f"part1_dist_{model_friendly.replace('-','_')}_{timestamp}.csv")
        dist_df.to_csv(dist_path, index=False)
        print(f"\n  → Distribution saved: {dist_path}")

        # Accumulate summary
        summary_rows.append({
            "model":              model_friendly,
            "n_samples":          len(names),
            "mean_char_length":   round(np.mean([len(n) for n in names]), 3),
            "mean_token_length":  round(lengths.mean(), 3),
            "median_token_length": int(np.median(lengths)),
            "std_token_length":   round(lengths.std(), 3),
            "max_token_length":   int(lengths.max()),
            "pct90_token_length": int(np.percentile(lengths, 90)),
            "pct_1tok":           round((lengths == 1).mean() * 100, 2),
            "pct_2tok":           round((lengths == 2).mean() * 100, 2),
            "pct_3tok":           round((lengths == 3).mean() * 100, 2),
            "pct_4tok":           round((lengths == 4).mean() * 100, 2),
            "pct_5tok":           round((lengths == 5).mean() * 100, 2),
            "pct_6plus_tok":      round((lengths >= 6).mean() * 100, 2),
        })

    # 3. Summary CSV
    summary_df = pd.DataFrame(summary_rows)
    summary_path = os.path.join(RESULTS_DIR, f"part1_summary_{timestamp}.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"\n[3] Summary saved → {summary_path}")
    print(summary_df.to_string(index=False))

    # 4. Plots
    print("\n[4] Generating plots …")
    fig_prefix = os.path.join(FIGURES_DIR, f"part1_{timestamp}")
    plot_histogram(all_lengths, fig_prefix)
    plot_cdf(all_lengths, fig_prefix)
    plot_stacked_bar(all_dist, fig_prefix)

    # 5. Print actionable recommendation
    print("\n" + "=" * 65)
    print("  KEY FINDINGS")
    print("=" * 65)
    for row in summary_rows:
        m = row["model"]
        print(f"\n  {m}:")
        print(f"    Average GT variable name  →  {row['mean_token_length']:.2f} tokens")
        print(f"    Distribution:")
        for k in [1, 2, 3, 4, 5]:
            print(f"      {k} token(s): {row[f'pct_{k}tok']:.1f}%")
        print(f"      6+ tokens : {row['pct_6plus_tok']:.1f}%")
        med = row["median_token_length"]
        print(f"\n    Recommendation for Part 2: test MASK counts [1, 2, 3, {med}, 5]")
    print("=" * 65)


if __name__ == "__main__":
    main()
