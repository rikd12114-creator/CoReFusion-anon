"""
Part 2 Analysis Script
======================

Reads all part2_raw_*.csv files from results/1t5t_exp/ and produces:

  1. Summary table  – EM per (model, k), with OOM rate and valid sample count
  2. EM trend plot  – line chart of EM vs k for each model
  3. GT length vs EM – does the model do better on short vs long variable names?
  4. Prediction analysis – top predicted tokens per k, truncation rate
  5. Error analysis    – OOM distribution across token lengths

Usage (from repo root):
    python experiments/1t5t_exp/analyze_part2.py
    python experiments/1t5t_exp/analyze_part2.py --results-dir results/1t5t_exp
"""

import os
import re
import sys
import glob
import argparse
from datetime import datetime

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from collections import Counter

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

RESULTS_DIR = "results/1t5t_exp"
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")

PALETTE = {
    "DiffuCoder-7B": "#3a86ff",
    "DreamCoder-7B": "#ff6b6b",
}
FALLBACK_COLORS = ["#06d6a0", "#ffd166", "#ef476f", "#118ab2"]

plt.rcParams.update({
    "font.family":  "DejaVu Sans",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":    True,
    "grid.alpha":   0.35,
    "grid.linestyle": "--",
})


# ─────────────────────────────────────────────────────────────────────────────
# Loading
# ─────────────────────────────────────────────────────────────────────────────

def load_all_part2(results_dir: str) -> pd.DataFrame:
    """Load all part2_raw_*.csv files and concat into one DataFrame."""
    pattern = os.path.join(results_dir, "part2_raw_*.csv")
    files   = sorted(glob.glob(pattern))
    if not files:
        print(f"ERROR: no part2_raw_*.csv found in {results_dir}")
        sys.exit(1)

    dfs = []
    for f in files:
        df = pd.read_csv(f, dtype={"id": str})
        dfs.append(df)
        print(f"  Loaded: {os.path.basename(f):60s}  ({len(df):4d} rows)")

    combined = pd.concat(dfs, ignore_index=True)

    # Normalise columns
    combined["mask_count"]  = combined["mask_count"].astype(int)
    combined["exact_match"] = combined["exact_match"].fillna(0).astype(int)
    combined["llm_verdict"] = combined["llm_verdict"].fillna(-2).astype(int)

    # Derive helper columns
    combined["is_oom"]   = combined.get("error", pd.Series("", index=combined.index)).astype(str).str.contains("CUDA out of memory", na=False)
    combined["is_error"] = combined["llm_verdict"] == -1   # any error (incl OOM)
    combined["is_valid"] = ~combined["is_error"]           # successful inference

    # GT character length
    combined["gt_len"] = combined["ground_truth"].astype(str).apply(len)

    # Prediction length (0 if error)
    combined["pred_len"] = combined["prediction"].fillna("").astype(str).apply(len)

    return combined


# ─────────────────────────────────────────────────────────────────────────────
# Helper: colour for model
# ─────────────────────────────────────────────────────────────────────────────

def _color(model_name: str, idx: int = 0) -> str:
    return PALETTE.get(model_name, FALLBACK_COLORS[idx % len(FALLBACK_COLORS)])


# ─────────────────────────────────────────────────────────────────────────────
# Analysis 1 – Core EM summary table
# ─────────────────────────────────────────────────────────────────────────────

def compute_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model, k), grp in df.groupby(["model", "mask_count"]):
        valid       = grp[grp["is_valid"]]
        n_total     = len(grp)
        n_oom       = grp["is_oom"].sum()
        n_valid     = len(valid)
        em          = valid["exact_match"].mean() if n_valid > 0 else float("nan")

        # LLM judge (only rows with verdict >= 0)
        judged      = grp[grp["llm_verdict"] >= 0]
        lj          = judged["llm_verdict"].mean() if len(judged) > 0 else float("nan")

        rows.append({
            "model":       model,
            "k":           k,
            "n_total":     n_total,
            "n_oom":       n_oom,
            "oom_rate":    round(n_oom / n_total, 4) if n_total > 0 else float("nan"),
            "n_valid":     n_valid,
            "exact_match": round(em, 4),
            "llm_judge":   round(lj, 4) if not np.isnan(lj) else None,
        })
    return pd.DataFrame(rows).sort_values(["model", "k"]).reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# Plot 1 – EM vs k (line + bar combo)
# ─────────────────────────────────────────────────────────────────────────────

def plot_em_vs_k(summary: pd.DataFrame, ts: str):
    models = summary["model"].unique()
    k_vals = sorted(summary["k"].unique())

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # ── Left: line chart ──────────────────────────────────────────────────────
    ax = axes[0]
    for i, model in enumerate(models):
        sub = summary[summary["model"] == model].set_index("k")
        em_vals = [sub.loc[k, "exact_match"] if k in sub.index else float("nan")
                   for k in k_vals]
        ax.plot(k_vals, em_vals, marker="o", linewidth=2.2, markersize=8,
                label=model, color=_color(model, i))
        for k, v in zip(k_vals, em_vals):
            if not np.isnan(v):
                ax.annotate(f"{v:.3f}", (k, v),
                            textcoords="offset points", xytext=(0, 9),
                            ha="center", fontsize=8.5, color=_color(model, i))

    ax.set_xlabel("Number of <|mask|> tokens (k)", fontsize=11)
    ax.set_ylabel("Exact Match (EM)", fontsize=11)
    ax.set_title("EM vs MASK-Token Count", fontweight="bold")
    ax.set_xticks(k_vals)
    ax.legend(fontsize=10)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=1))

    # ── Right: grouped bar chart ──────────────────────────────────────────────
    ax = axes[1]
    x       = np.arange(len(k_vals))
    n_mod   = len(models)
    width   = 0.7 / n_mod
    offsets = np.linspace(-(n_mod - 1) * width / 2, (n_mod - 1) * width / 2, n_mod)

    for i, model in enumerate(models):
        sub   = summary[summary["model"] == model].set_index("k")
        vals  = [sub.loc[k, "exact_match"] if k in sub.index else 0. for k in k_vals]
        bars  = ax.bar(x + offsets[i], vals, width, label=model,
                       color=_color(model, i), alpha=0.82)
        for bar, v in zip(bars, vals):
            if v > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
                        f"{v:.3f}", ha="center", va="bottom", fontsize=8)

    ax.set_xlabel("Number of <|mask|> tokens (k)", fontsize=11)
    ax.set_ylabel("Exact Match (EM)", fontsize=11)
    ax.set_title("EM per k (Bar)", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([str(k) for k in k_vals])
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=1))
    ax.legend(fontsize=10)

    plt.suptitle("Part 2 – Static MASK-Token Count Ablation\n"
                 "DiffuCoder-7B on RefineID (Java) · 32 diffusion steps",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    out = os.path.join(FIGURES_DIR, f"part2_em_vs_k_{ts}.png")
    plt.savefig(out, bbox_inches="tight", dpi=150)
    plt.savefig(out.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close()
    print(f"  → {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Plot 2 – EM by GT character-length bucket
# ─────────────────────────────────────────────────────────────────────────────

def plot_em_by_gt_length(df: pd.DataFrame, ts: str):
    valid = df[df["is_valid"]].copy()

    # Bucket GT length
    bins   = [0, 4, 8, 12, 17, 50]
    labels = ["≤4", "5-8", "9-12", "13-17", "18+"]
    valid["gt_bucket"] = pd.cut(valid["gt_len"], bins=bins, labels=labels)

    models = sorted(valid["model"].unique())
    k_vals = sorted(valid["mask_count"].unique())

    fig, axes = plt.subplots(1, len(models), figsize=(7 * len(models), 5), squeeze=False)

    for col, model in enumerate(models):
        ax  = axes[0][col]
        sub = valid[valid["model"] == model]

        pivot = (sub.groupby(["gt_bucket", "mask_count"])["exact_match"]
                   .mean()
                   .unstack("mask_count"))

        x      = np.arange(len(labels))
        n_k    = len(k_vals)
        width  = 0.7 / n_k
        offs   = np.linspace(-(n_k - 1) * width / 2, (n_k - 1) * width / 2, n_k)
        cmap   = plt.cm.Blues(np.linspace(0.4, 0.9, n_k))

        for ki, (k, off, c) in enumerate(zip(k_vals, offs, cmap)):
            if k not in pivot.columns:
                continue
            vals = [pivot.loc[label, k] if label in pivot.index else 0.
                    for label in labels]
            ax.bar(x + off, vals, width, label=f"k={k}", color=c, alpha=0.9)

        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_xlabel("GT variable name length (chars)", fontsize=10)
        ax.set_ylabel("Exact Match", fontsize=10)
        ax.set_title(f"{model}\nEM by GT Length Bucket", fontweight="bold")
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=1))
        ax.legend(fontsize=8, ncol=3)

        # Count labels on x-axis
        for xi, label in enumerate(labels):
            n = len(sub[sub["gt_bucket"] == label])
            ax.text(xi, -0.04, f"n={n}", ha="center", va="top",
                    fontsize=7.5, color="grey", transform=ax.get_xaxis_transform())

    plt.suptitle("EM by GT Variable Name Length", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    out = os.path.join(FIGURES_DIR, f"part2_em_by_gt_length_{ts}.png")
    plt.savefig(out, bbox_inches="tight", dpi=150)
    plt.savefig(out.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close()
    print(f"  → {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Plot 3 – OOM rate per k
# ─────────────────────────────────────────────────────────────────────────────

def plot_oom_rate(summary: pd.DataFrame, ts: str):
    if summary["oom_rate"].fillna(0).sum() == 0:
        print("  (No OOM errors found – skipping OOM rate plot)")
        return

    models = summary["model"].unique()
    k_vals = sorted(summary["k"].unique())
    x      = np.arange(len(k_vals))
    n_mod  = len(models)
    width  = 0.6 / n_mod
    offsets = np.linspace(-(n_mod-1)*width/2, (n_mod-1)*width/2, n_mod)

    fig, ax = plt.subplots(figsize=(9, 4))

    for i, model in enumerate(models):
        sub  = summary[summary["model"] == model].set_index("k")
        vals = [sub.loc[k, "oom_rate"] if k in sub.index else 0. for k in k_vals]
        bars = ax.bar(x + offsets[i], vals, width,
                      label=model, color=_color(model, i), alpha=0.82)
        for bar, v in zip(bars, vals):
            if v > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
                        f"{v:.1%}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels([str(k) for k in k_vals])
    ax.set_xlabel("Number of <|mask|> tokens (k)", fontsize=11)
    ax.set_ylabel("OOM Error Rate", fontsize=11)
    ax.set_title("OOM Rate per MASK-Token Count\n"
                 "(OOM samples are excluded from EM calculation)",
                 fontweight="bold")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=1))
    ax.legend(fontsize=10)
    plt.tight_layout()
    out = os.path.join(FIGURES_DIR, f"part2_oom_rate_{ts}.png")
    plt.savefig(out, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"  → {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Plot 4 – Prediction distribution: top 15 predicted tokens per k
# ─────────────────────────────────────────────────────────────────────────────

def plot_top_predictions(df: pd.DataFrame, ts: str):
    valid  = df[df["is_valid"] & (df["prediction"].fillna("") != "")]
    models = sorted(valid["model"].unique())
    k_vals = sorted(valid["mask_count"].unique())

    for model in models:
        sub      = valid[valid["model"] == model]
        n_cols   = min(len(k_vals), 5)
        fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 5))
        if n_cols == 1:
            axes = [axes]

        for ax, k in zip(axes, k_vals):
            preds   = sub[sub["mask_count"] == k]["prediction"].dropna()
            counter = Counter(preds.astype(str))
            top15   = counter.most_common(15)
            if not top15:
                ax.axis("off")
                continue
            names, counts = zip(*top15)
            y = np.arange(len(names))
            ax.barh(y, counts, color=_color(model), alpha=0.8)
            ax.set_yticks(y)
            ax.set_yticklabels(names, fontsize=8)
            ax.invert_yaxis()
            ax.set_xlabel("Frequency", fontsize=9)
            ax.set_title(f"k={k}", fontweight="bold")

        fig.suptitle(f"{model} – Top 15 Predicted Tokens per k",
                     fontsize=12, fontweight="bold", y=1.02)
        plt.tight_layout()
        safe = model.replace("-", "_")
        out  = os.path.join(FIGURES_DIR, f"part2_top_preds_{safe}_{ts}.png")
        plt.savefig(out, bbox_inches="tight", dpi=150)
        plt.close()
        print(f"  → {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Plot 5 – GT length distribution in dataset
# ─────────────────────────────────────────────────────────────────────────────

def plot_gt_length_distribution(df: pd.DataFrame, ts: str):
    # Use k=1 to avoid duplicates (all k share same GT)
    sample = df[(df["mask_count"] == df["mask_count"].min()) & df["is_valid"]].copy()
    sample = sample.drop_duplicates(subset=["id"])

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    # Histogram
    ax = axes[0]
    ax.hist(sample["gt_len"], bins=range(1, 35), color="#3a86ff", alpha=0.8,
            edgecolor="white", linewidth=0.5)
    ax.axvline(sample["gt_len"].mean(), color="#ef476f", linestyle="--",
               linewidth=1.8, label=f"Mean={sample['gt_len'].mean():.1f}")
    ax.axvline(sample["gt_len"].median(), color="#06d6a0", linestyle=":",
               linewidth=1.8, label=f"Median={sample['gt_len'].median():.0f}")
    ax.set_xlabel("GT variable name length (chars)", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title("GT Variable Name Length Distribution", fontweight="bold")
    ax.legend(fontsize=9)

    # CDF
    ax = axes[1]
    sorted_lens = np.sort(sample["gt_len"])
    cdf = np.arange(1, len(sorted_lens) + 1) / len(sorted_lens)
    ax.plot(sorted_lens, cdf, color="#3a86ff", linewidth=2)
    for pct in [0.5, 0.75, 0.90]:
        idx = np.searchsorted(cdf, pct)
        if idx < len(sorted_lens):
            ax.axhline(pct, color="grey", linestyle="--", linewidth=0.8, alpha=0.6)
            ax.axvline(sorted_lens[idx], color="grey", linestyle="--",
                       linewidth=0.8, alpha=0.6)
            ax.text(sorted_lens[idx] + 0.3, pct + 0.01,
                    f"P{int(pct*100)}={sorted_lens[idx]}", fontsize=8, color="grey")
    ax.set_xlabel("GT variable name length (chars)", fontsize=11)
    ax.set_ylabel("Cumulative fraction", fontsize=11)
    ax.set_title("CDF of GT Variable Name Length", fontweight="bold")

    plt.suptitle("RefineID Dataset – GT Variable Name Length Stats",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    out = os.path.join(FIGURES_DIR, f"part2_gt_length_{ts}.png")
    plt.savefig(out, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"  → {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Text report
# ─────────────────────────────────────────────────────────────────────────────

def print_report(df: pd.DataFrame, summary: pd.DataFrame):
    print("\n" + "═" * 70)
    print("  PART 2 ANALYSIS REPORT")
    print("═" * 70)

    print("\n📊  SUMMARY TABLE  (valid samples only, OOM excluded from EM)")
    print(summary.to_string(index=False))

    print("\n🏆  BEST k PER MODEL")
    for model in summary["model"].unique():
        sub = summary[summary["model"] == model]
        best = sub.loc[sub["exact_match"].idxmax()]
        print(f"\n  {model}")
        print(f"    Best EM  →  k={int(best['k'])}   "
              f"EM={best['exact_match']:.4f}  "
              f"(valid n={int(best['n_valid'])}/{int(best['n_total'])})")
        lj_col = sub["llm_judge"].dropna()
        if len(lj_col) > 0:
            best_lj = sub.loc[lj_col.idxmax()]
            print(f"    Best LLM →  k={int(best_lj['k'])}   "
                  f"LJ={best_lj['llm_judge']:.4f}")

    print("\n⚠️   OOM ERRORS")
    oom_rows = summary[summary["n_oom"] > 0]
    if len(oom_rows) == 0:
        print("  No OOM errors found.")
    else:
        for _, r in oom_rows.iterrows():
            print(f"  {r['model']:20s}  k={int(r['k'])}  "
                  f"OOM={int(r['n_oom'])}/{int(r['n_total'])}  "
                  f"({r['oom_rate']:.1%})")
        print("\n  TIP: OOM samples excluded from EM.  Re-run those samples with")
        print("       PYTORCH_ALLOC_CONF=expandable_segments:True")

    print("\n📈  GT LENGTH STATISTICS (from valid k=1 samples, unique ids)")
    sample = df[(df["mask_count"] == df["mask_count"].min()) & df["is_valid"]]
    sample = sample.drop_duplicates(subset=["id"])
    gl     = sample["gt_len"]
    print(f"  n={len(gl)}  mean={gl.mean():.1f}  "
          f"median={gl.median():.0f}  "
          f"P90={np.percentile(gl, 90):.0f}  "
          f"max={gl.max()}")

    print("\n📉  EM TREND  (delta EM from k=1 to k=5)")
    for model in summary["model"].unique():
        sub  = summary[summary["model"] == model].set_index("k")
        k1   = sub["exact_match"].get(1, float("nan"))
        k5   = sub["exact_match"].get(5, float("nan"))
        delta = k5 - k1 if not (np.isnan(k1) or np.isnan(k5)) else float("nan")
        trend = "↑ improves" if delta > 0 else ("↓ degrades" if delta < 0 else "→ flat")
        print(f"  {model:20s}  k=1→k=5  Δ={delta:+.4f}  {trend}")

    print("\n" + "═" * 70)


# ─────────────────────────────────────────────────────────────────────────────
# Save summary CSV
# ─────────────────────────────────────────────────────────────────────────────

def save_summary(summary: pd.DataFrame, results_dir: str, ts: str):
    path = os.path.join(results_dir, f"part2_analysis_summary_{ts}.csv")
    summary.to_csv(path, index=False)
    print(f"\n  Summary CSV → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Analyse Part 2 raw results from results/1t5t_exp/")
    parser.add_argument("--results-dir", default=RESULTS_DIR,
                        help=f"Directory containing part2_raw_*.csv (default: {RESULTS_DIR})")
    parser.add_argument("--no-plots", action="store_true",
                        help="Skip figure generation (text report only)")
    args = parser.parse_args()

    os.makedirs(FIGURES_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("=" * 70)
    print("  Part 2 Analysis")
    print("=" * 70)
    print(f"\nLoading part2_raw_*.csv from {args.results_dir} …")

    df      = load_all_part2(args.results_dir)
    summary = compute_summary(df)

    print_report(df, summary)
    save_summary(summary, args.results_dir, ts)

    if not args.no_plots:
        print("\nGenerating figures …")
        plot_em_vs_k(summary, ts)
        plot_em_by_gt_length(df, ts)
        plot_oom_rate(summary, ts)
        plot_top_predictions(df, ts)
        plot_gt_length_distribution(df, ts)
        print(f"\nAll figures saved to {FIGURES_DIR}/")

    print("\nDone ✓")


if __name__ == "__main__":
    main()
