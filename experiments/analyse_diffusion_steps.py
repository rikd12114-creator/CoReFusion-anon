"""
Analysis & Visualisation — Diffusion Step Sensitivity
======================================================

Reads the summary CSV produced by exp_diffusion_steps_benchmark.py and
generates publication-quality figures + a concise printed report.

Usage
-----
  # Auto-detect the latest summary in results/diffusion_steps_benchmark/:
  python experiments/analyse_diffusion_steps.py

  # Or point to a specific summary file:
  python experiments/analyse_diffusion_steps.py \\
      --summary results/diffusion_steps_benchmark/summary_20260303_120000.csv

  # Save figures to a custom directory:
  python experiments/analyse_diffusion_steps.py --out-dir results/paper_figures
"""

import os
import sys
import glob
import argparse

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # headless-safe backend
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy import stats         # for Wilcoxon signed-rank test (optional)

# ══════════════════════════════════════════════════════════════════════════════
# Config
# ══════════════════════════════════════════════════════════════════════════════

ROOT_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BENCH_DIR   = os.path.join(ROOT_DIR, "results", "diffusion_steps_benchmark")

# Colour palette — one colour per model, consistent across all plots
MODEL_COLOURS = {
    "DiffuCoder-7B-Base":      "#4C72B0",
    "DiffuCoder-7B-Instruct":  "#DD8452",
    "DreamCoder-7B":           "#55A868",
}
DEFAULT_COLOUR = "#8C8C8C"

# ══════════════════════════════════════════════════════════════════════════════
# Utilities
# ══════════════════════════════════════════════════════════════════════════════

def find_latest_summary(bench_dir: str) -> str:
    """Return the most recently modified summary_*.csv in bench_dir."""
    pattern = os.path.join(bench_dir, "summary_*.csv")
    files   = sorted(glob.glob(pattern), key=os.path.getmtime)
    if not files:
        raise FileNotFoundError(
            f"No summary_*.csv found in {bench_dir}.  "
            "Run exp_diffusion_steps_benchmark.py first."
        )
    return files[-1]


def load_summary(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"model", "steps", "exact_match_rate",
                "mean_time_per_sample", "relative_speedup", "accuracy_drop"}
    missing  = required - set(df.columns)
    if missing:
        raise ValueError(f"Summary CSV is missing columns: {missing}")
    df["steps"] = df["steps"].astype(int)
    return df


def model_colour(model_name: str) -> str:
    return MODEL_COLOURS.get(model_name, DEFAULT_COLOUR)


# ══════════════════════════════════════════════════════════════════════════════
# Figure 1 — Exact Match Rate vs. Diffusion Steps
# ══════════════════════════════════════════════════════════════════════════════

def plot_em_vs_steps(df: pd.DataFrame, out_dir: str) -> str:
    fig, ax = plt.subplots(figsize=(8, 4.5))

    for model in df["model"].unique():
        sub   = df[df["model"] == model].sort_values("steps")
        col   = model_colour(model)
        ax.plot(
            sub["steps"], sub["exact_match_rate"] * 100,
            marker="o", linewidth=2.0, markersize=6,
            color=col, label=model,
        )
        # Shade ±1 pp band around each point (visual guide only)
        ax.fill_between(
            sub["steps"],
            sub["exact_match_rate"] * 100 - 1,
            sub["exact_match_rate"] * 100 + 1,
            alpha=0.10, color=col,
        )

    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
    ax.xaxis.set_major_locator(mticker.FixedLocator(df["steps"].unique()))
    ax.set_xlabel("Diffusion Steps (log₂ scale)", fontsize=12)
    ax.set_ylabel("Exact Match Rate (%)", fontsize=12)
    ax.set_title("RefineID: Exact Match vs. Diffusion Step Count\n"
                 "(same model, same parameters — only steps varies)", fontsize=12)
    ax.legend(fontsize=10, loc="lower right")
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.5)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))

    plt.tight_layout()
    out_path = os.path.join(out_dir, "em_vs_steps.pdf")
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    out_png = out_path.replace(".pdf", ".png")
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [Fig 1] {out_path}")
    return out_path


# ══════════════════════════════════════════════════════════════════════════════
# Figure 2 — Relative Speedup vs. Accuracy Drop  (Pareto frontier)
# ══════════════════════════════════════════════════════════════════════════════

def plot_speedup_vs_drop(df: pd.DataFrame, out_dir: str) -> str:
    fig, ax = plt.subplots(figsize=(8, 4.5))

    for model in df["model"].unique():
        sub = df[df["model"] == model].sort_values("steps")
        col = model_colour(model)

        sc = ax.scatter(
            sub["relative_speedup"],
            sub["accuracy_drop"] * 100,   # convert to pp
            c=np.log2(sub["steps"]),       # colour encodes step count
            cmap="viridis", s=80, zorder=3,
            edgecolors=col, linewidths=1.2,
            label=model,
        )
        # Annotate step count beside each point
        for _, r in sub.iterrows():
            ax.annotate(
                f"{int(r['steps'])}",
                (r["relative_speedup"], r["accuracy_drop"] * 100),
                textcoords="offset points", xytext=(4, 4),
                fontsize=7, color=col,
            )

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", zorder=2)
    ax.axhline(1, color="red",   linewidth=0.7, linestyle=":",  zorder=2, alpha=0.6,
               label="1 pp accuracy drop threshold")

    # Ideal region: bottom-right corner
    ax.fill_betweenx([-5, 0.5], ax.get_xlim()[0] if ax.get_xlim()[0] > 1 else 1,
                     ax.get_xlim()[1] if ax.get_xlim() else 200,
                     alpha=0.04, color="green", zorder=1)

    ax.set_xlabel("Relative Throughput Speedup  (×ref)", fontsize=12)
    ax.set_ylabel("Accuracy Drop vs. Reference (percentage points)", fontsize=12)
    ax.set_title("Efficiency Pareto: Speedup vs. Accuracy Cost\n"
                 "(bottom-right = faster AND at least as accurate)", fontsize=12)
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)

    plt.tight_layout()
    out_path = os.path.join(out_dir, "speedup_vs_accuracy_drop.pdf")
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    fig.savefig(out_path.replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [Fig 2] {out_path}")
    return out_path


# ══════════════════════════════════════════════════════════════════════════════
# Figure 3 — Throughput (samples/s) vs. Steps
# ══════════════════════════════════════════════════════════════════════════════

def plot_throughput(df: pd.DataFrame, out_dir: str) -> str:
    if "throughput_sps" not in df.columns:
        print("  [Fig 3] Skipped — 'throughput_sps' column not found.")
        return ""

    fig, ax = plt.subplots(figsize=(8, 4))

    for model in df["model"].unique():
        sub = df[df["model"] == model].sort_values("steps")
        col = model_colour(model)
        ax.plot(sub["steps"], sub["throughput_sps"],
                marker="s", linewidth=2.0, markersize=6,
                color=col, label=model)

    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
    ax.xaxis.set_major_locator(mticker.FixedLocator(df["steps"].unique()))
    ax.set_xlabel("Diffusion Steps (log₂ scale)", fontsize=12)
    ax.set_ylabel("Throughput (samples / second)", fontsize=12)
    ax.set_title("Inference Throughput by Step Count", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.5)

    plt.tight_layout()
    out_path = os.path.join(out_dir, "throughput_vs_steps.pdf")
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    fig.savefig(out_path.replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [Fig 3] {out_path}")
    return out_path


# ══════════════════════════════════════════════════════════════════════════════
# Figure 4 — Combined 2×2 panel for paper
# ══════════════════════════════════════════════════════════════════════════════

def plot_combined_panel(df: pd.DataFrame, out_dir: str) -> str:
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    # — Panel A: EM vs steps —
    ax = axes[0]
    for model in df["model"].unique():
        sub = df[df["model"] == model].sort_values("steps")
        col = model_colour(model)
        ax.plot(sub["steps"], sub["exact_match_rate"] * 100,
                marker="o", linewidth=2.0, markersize=6,
                color=col, label=model.replace("-7B", ""))
    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
    ax.xaxis.set_major_locator(mticker.FixedLocator(sorted(df["steps"].unique())))
    ax.set_xlabel("Steps (log₂)", fontsize=11)
    ax.set_ylabel("Exact Match Rate (%)", fontsize=11)
    ax.set_title("(A) Accuracy vs. Steps", fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)

    # — Panel B: Throughput vs steps —
    ax = axes[1]
    if "throughput_sps" in df.columns:
        for model in df["model"].unique():
            sub = df[df["model"] == model].sort_values("steps")
            col = model_colour(model)
            ax.plot(sub["steps"], sub["throughput_sps"],
                    marker="s", linewidth=2.0, markersize=6,
                    color=col, label=model.replace("-7B", ""))
        ax.set_xscale("log", base=2)
        ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
        ax.xaxis.set_major_locator(mticker.FixedLocator(sorted(df["steps"].unique())))
        ax.set_xlabel("Steps (log₂)", fontsize=11)
        ax.set_ylabel("Samples / second", fontsize=11)
        ax.set_title("(B) Throughput vs. Steps", fontsize=11)
        ax.legend(fontsize=8)
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)

    # — Panel C: Speedup vs EM drop —
    ax = axes[2]
    for model in df["model"].unique():
        sub = df[df["model"] == model].sort_values("steps")
        col = model_colour(model)
        ax.scatter(sub["relative_speedup"],
                   sub["accuracy_drop"] * 100,
                   c=col, s=70, label=model.replace("-7B", ""), zorder=3,
                   edgecolors="white", linewidths=0.5)
        for _, r in sub.iterrows():
            ax.annotate(f"{int(r['steps'])}",
                        (r["relative_speedup"], r["accuracy_drop"] * 100),
                        textcoords="offset points", xytext=(3, 3),
                        fontsize=7, color=col)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Relative Speedup (×)", fontsize=11)
    ax.set_ylabel("EM Drop vs. Ref (pp)", fontsize=11)
    ax.set_title("(C) Speed–Accuracy Trade-off", fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)

    plt.suptitle(
        "Diffusion Step Sensitivity — RefineID Task\n"
        "(Same model weights, only step count varies)",
        fontsize=12, y=1.01,
    )
    plt.tight_layout()
    out_path = os.path.join(out_dir, "diffusion_steps_sensitivity_panel.pdf")
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    fig.savefig(out_path.replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [Fig 4] {out_path}")
    return out_path


# ══════════════════════════════════════════════════════════════════════════════
# Text report
# ══════════════════════════════════════════════════════════════════════════════

def print_report(df: pd.DataFrame) -> None:
    print("\n" + "="*72)
    print("  DIFFUSION STEP SENSITIVITY — STATISTICAL REPORT")
    print("="*72)

    for model in df["model"].unique():
        sub = df[df["model"] == model].sort_values("steps")
        em  = sub["exact_match_rate"].values

        em_range = em.max() - em.min()
        mean_em  = em.mean()
        std_em   = em.std()

        # Try Kruskal-Wallis / Friedman test via detail CSV if available
        # (here we only have the summary, so we report summary stats)
        print(f"\n  ► {model}")
        print(f"    {'Steps':>6}  {'EM Rate':>8}  {'Time/s':>8}  {'Speedup':>8}  {'EM Drop':>8}")
        print(f"    {'-'*48}")
        for _, r in sub.iterrows():
            flag = "  ← ref" if r.get("is_reference", 0) else ""
            t_str = f"{r['mean_time_per_sample']:.3f}" if "mean_time_per_sample" in r else "—"
            print(f"    {int(r['steps']):>6}  {r['exact_match_rate']:>8.2%}  "
                  f"{t_str:>8}  {r['relative_speedup']:>8.2f}×  "
                  f"{r['accuracy_drop']:>+8.2%}{flag}")

        print(f"\n    EM range  : {em_range:.2%}  (max−min across all step counts)")
        print(f"    Mean EM   : {mean_em:.2%} ± {std_em:.2%} (std)")

        # Fastest config that drops EM by at most 1 pp
        THRESHOLD_PP = 0.01
        fast_sub = sub[sub["accuracy_drop"] <= THRESHOLD_PP]
        if not fast_sub.empty:
            best = fast_sub.loc[fast_sub["relative_speedup"].idxmax()]
            print(f"    Best fast : steps={int(best['steps'])}  "
                  f"speedup={best['relative_speedup']:.2f}×  "
                  f"EM drop={best['accuracy_drop']:+.2%}  "
                  f"(within 1 pp of reference EM)")
        else:
            print("    Best fast : no config within 1 pp threshold")

        verdict = ("✓ ROBUST — diffusion step count does NOT significantly impact accuracy"
                   if em_range < 0.03
                   else "△ SENSITIVE — accuracy degrades noticeably at very low step counts")
        print(f"    Verdict   : {verdict}")

    print("\n" + "="*72 + "\n")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Analyse & visualise diffusion step sensitivity results"
    )
    parser.add_argument(
        "--summary", type=str, default=None,
        help="Path to summary CSV. If omitted, auto-detects latest in results/diffusion_steps_benchmark/.",
    )
    parser.add_argument(
        "--out-dir", type=str, default=None,
        help="Directory to write figures. Defaults to same directory as summary CSV.",
    )
    args = parser.parse_args()

    summary_path = args.summary or find_latest_summary(BENCH_DIR)
    print(f"Summary: {summary_path}")

    df       = load_summary(summary_path)
    out_dir  = args.out_dir or os.path.dirname(summary_path)
    os.makedirs(out_dir, exist_ok=True)

    print_report(df)

    print("Generating figures…")
    plot_em_vs_steps(df, out_dir)
    plot_speedup_vs_drop(df, out_dir)
    plot_throughput(df, out_dir)
    plot_combined_panel(df, out_dir)

    print(f"\nAll figures saved to: {out_dir}")


if __name__ == "__main__":
    main()
