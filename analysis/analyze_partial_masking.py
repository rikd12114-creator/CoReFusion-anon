"""
Analysis: Partial Masking Noise Mapping Experiment
===================================================
Reads the output of partial_masking_noise_map.py and produces:
  - fig_S1: Alpha–Entropy curves per group (the key "bridge" plot)
  - fig_S2: KL divergence decay curves
  - fig_S3: α* crossing-point summary (the noise position estimate)
  - stats_supplement.txt
"""

import os, sys, argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy import stats
from scipy.interpolate import interp1d

OUTDIR = "results/noise_mapping_supplementary"
os.makedirs(OUTDIR, exist_ok=True)

# ────────────────── paper style (same as main figures) ─────────────────────
matplotlib.rcParams.update({
    "font.family":      "serif",
    "font.serif":       ["Times New Roman", "DejaVu Serif"],
    "font.size":        8,
    "axes.titlesize":   8,
    "axes.labelsize":   8,
    "xtick.labelsize":  7,
    "ytick.labelsize":  7,
    "legend.fontsize":  7,
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "axes.grid":        True,
    "grid.linewidth":   0.4,
    "grid.alpha":       0.4,
})
SINGLE_W = 3.5
DOUBLE_W = 7.16
DPI = 300

C = {
    "smell_severe":   "#FF7F0E",
    "smell_moderate": "#BCBD22",
    "smell_mild":     "#2CA02C",
    "control":        "#1F77B4",
}
MARKERS = {
    "smell_severe":   "^",
    "smell_moderate": "s",
    "smell_mild":     "D",
    "control":        "o",
}
LABELS = {
    "smell_severe":   "Severe smell (x, a)",
    "smell_moderate": "Moderate smell (tmp, val)",
    "smell_mild":     "Mild smell (myVar)",
    "control":        "Clean code (GT)",
}

def load_data(path):
    df = pd.read_csv(path)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    return df

def agg_by_alpha(df, group_col="group", alpha_col="alpha", metric="entropy"):
    return (df.groupby([group_col, alpha_col])[metric]
              .agg(mean="mean", se=lambda x: x.sem(), n="count")
              .reset_index())


# ═══════════════════════════════════════════════════════════════════════════
# FIG S1 — Alpha–Entropy curves: the bridge between regimes
# ═══════════════════════════════════════════════════════════════════════════
def plot_alpha_entropy(df, outdir):
    agg = agg_by_alpha(df)
    groups = [g for g in df["group"].unique() if g in C]

    fig, ax = plt.subplots(figsize=(SINGLE_W * 1.3, 2.9))

    full_mask_at_alpha1 = None   # entropy of the mask group at α=1.0

    for grp in ["control", "smell_mild", "smell_severe", "smell_moderate"]:
        if grp not in groups:
            continue
        sub = agg[agg["group"] == grp].sort_values("alpha")
        x   = sub["alpha"].values
        y   = sub["mean"].values
        ye  = sub["se"].values
        col = C[grp]

        ax.plot(x, y, color=col, lw=1.3, marker=MARKERS[grp],
                markersize=4, label=LABELS[grp], zorder=5)
        ax.fill_between(x, y - ye, y + ye, color=col, alpha=0.10, zorder=3)

        if grp == "control" and 1.0 in sub["alpha"].values:
            full_mask_at_alpha1 = sub.loc[sub["alpha"] == 1.0, "mean"].values[0]

    # Reference: horizontal line at full mask entropy (α=1.0 of any group ≈ same)
    if full_mask_at_alpha1 is not None:
        ax.axhline(full_mask_at_alpha1, color="#888", ls=":", lw=1.0,
                   label=f"Full-mask entropy ≈ {full_mask_at_alpha1:.3f}")

    # Mark α* crossing points for each smell group
    for grp in ["smell_severe", "smell_moderate", "smell_mild"]:
        if grp not in groups or full_mask_at_alpha1 is None:
            continue
        sub = agg[agg["group"] == grp].sort_values("alpha")
        x, y = sub["alpha"].values, sub["mean"].values
        try:
            f = interp1d(x, y, kind="linear")
            alphas_fine = np.linspace(x.min(), x.max(), 1000)
            ys_fine     = f(alphas_fine)
            idx = np.argmin(np.abs(ys_fine - full_mask_at_alpha1))
            alpha_star = alphas_fine[idx]
            ax.axvline(alpha_star, color=C[grp], ls="--", lw=0.8, alpha=0.6)
            ax.text(alpha_star + 0.01, full_mask_at_alpha1 * 0.97,
                    f"α*={alpha_star:.2f}", color=C[grp], fontsize=6,
                    va="top")
        except Exception:
            pass

    ax.set_xlabel("Context masking fraction α\n(α=0 → no masking, α=1 → fully masked)")
    ax.set_ylabel("Shannon entropy at target position (nats)")
    ax.set_title("(S1) Partial masking bridges the noise regimes\n"
                 "α* = fraction at which smell entropy meets full-mask entropy",
                 fontsize=8, fontweight="bold")
    ax.legend(loc="upper left", frameon=False)
    ax.set_xlim(-0.02, 1.02)

    fig.tight_layout(pad=0.3)
    fig.savefig(f"{outdir}/figS1_alpha_entropy.pdf", bbox_inches="tight")
    fig.savefig(f"{outdir}/figS1_alpha_entropy.png", bbox_inches="tight", dpi=DPI)
    plt.close()
    print("  ✓ figS1_alpha_entropy")


# ═══════════════════════════════════════════════════════════════════════════
# FIG S2 — KL divergence from uniform (confidence proxy)
# ═══════════════════════════════════════════════════════════════════════════
def plot_kl_decay(df, outdir):
    if "kl_vs_uniform" not in df.columns:
        print("  KL column missing – skipping figS2")
        return

    agg = agg_by_alpha(df, metric="kl_vs_uniform")
    groups = [g for g in df["group"].unique() if g in C]

    fig, ax = plt.subplots(figsize=(SINGLE_W * 1.3, 2.9))

    for grp in ["control", "smell_mild", "smell_severe", "smell_moderate"]:
        if grp not in groups:
            continue
        sub = agg[agg["group"] == grp].sort_values("alpha")
        x, y, ye = sub["alpha"].values, sub["mean"].values, sub["se"].values
        ax.plot(x, y, color=C[grp], lw=1.3, marker=MARKERS[grp],
                markersize=4, label=LABELS[grp])
        ax.fill_between(x, y - ye, y + ye, color=C[grp], alpha=0.10)

    ax.set_xlabel("Context masking fraction α")
    ax.set_ylabel("KL divergence from uniform (nats)\n↑ = more confident; ↓ = more noise-like")
    ax.set_title("(S2) Model confidence decays toward uniform as\n"
                 "context masking increases — smell converges with mask",
                 fontsize=8, fontweight="bold")
    ax.legend(loc="upper right", frameon=False)
    ax.set_xlim(-0.02, 1.02)

    fig.tight_layout(pad=0.3)
    fig.savefig(f"{outdir}/figS2_kl_decay.pdf", bbox_inches="tight")
    fig.savefig(f"{outdir}/figS2_kl_decay.png", bbox_inches="tight", dpi=DPI)
    plt.close()
    print("  ✓ figS2_kl_decay")


# ═══════════════════════════════════════════════════════════════════════════
# FIG S3 — α* summary bar chart
# ═══════════════════════════════════════════════════════════════════════════
def plot_alpha_star(df, outdir):
    agg = agg_by_alpha(df)

    # Compute full-mask reference entropy at α=1.0
    ref_rows = agg[agg["alpha"] == 1.0]
    if ref_rows.empty:
        print("  No α=1.0 data – skipping figS3")
        return
    ref_entropy = ref_rows["mean"].mean()

    alpha_stars = {}
    for grp in ["smell_severe", "smell_moderate", "smell_mild", "control"]:
        sub = agg[agg["group"] == grp].sort_values("alpha")
        if sub.empty:
            continue
        x, y = sub["alpha"].values, sub["mean"].values
        try:
            f = interp1d(x, y, kind="linear")
            alphas_fine = np.linspace(x.min(), x.max(), 10000)
            ys_fine     = f(alphas_fine)
            idx = np.argmin(np.abs(ys_fine - ref_entropy))
            alpha_stars[grp] = alphas_fine[idx]
        except Exception:
            alpha_stars[grp] = float("nan")

    if not alpha_stars:
        print("  No α* values – skipping figS3")
        return

    fig, ax = plt.subplots(figsize=(SINGLE_W, 2.4))
    keys_ord = ["smell_severe", "smell_moderate", "smell_mild", "control"]
    labels_ord = ["Severe\nSmell", "Moderate\nSmell", "Mild\nSmell", "Clean\nCode"]
    vals  = [alpha_stars.get(k, np.nan) for k in keys_ord]
    cols  = [C[k] for k in keys_ord]
    x_pos = np.arange(len(keys_ord))

    bars = ax.bar(x_pos, vals, color=cols, width=0.55,
                  edgecolor="white", linewidth=0.6, alpha=0.82)
    for bar, val in zip(bars, vals):
        if not np.isnan(val):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.01,
                    f"α*={val:.2f}", ha="center", va="bottom", fontsize=6.5,
                    fontweight="bold")

    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels_ord)
    ax.set_ylabel("α* — masking fraction at noise convergence")
    ax.set_title("(S3) Equivalent noise masking fraction α*\n"
                 "for each code quality level",
                 fontsize=8, fontweight="bold")
    ax.set_ylim(0, 1.15)

    # Annotation
    ax.axhline(1.0, color="#888", ls=":", lw=0.9, label="Full mask (α=1)")
    ax.legend(fontsize=6.5, frameon=False)

    fig.tight_layout(pad=0.3)
    fig.savefig(f"{outdir}/figS3_alpha_star.pdf", bbox_inches="tight")
    fig.savefig(f"{outdir}/figS3_alpha_star.png", bbox_inches="tight", dpi=DPI)
    plt.close()
    print(f"  ✓ figS3_alpha_star  (values: {alpha_stars})")


# ═══════════════════════════════════════════════════════════════════════════
# Statistical report
# ═══════════════════════════════════════════════════════════════════════════
def write_stats(df, outdir):
    lines = ["=" * 66,
             "Supplementary Statistical Analysis — Partial Masking Experiment",
             "=" * 66, ""]

    # At each alpha, test entropy differences between smell groups and control
    for alpha in sorted(df["alpha"].unique()):
        sub = df[df["alpha"] == alpha]
        lines.append(f"α = {alpha:.2f}")
        ctrl = sub[sub["group"] == "control"]["entropy"].dropna()
        for grp in ["smell_severe", "smell_moderate", "smell_mild"]:
            g_vals = sub[sub["group"] == grp]["entropy"].dropna()
            if len(g_vals) < 3 or len(ctrl) < 3:
                continue
            t, p = stats.ttest_ind(g_vals, ctrl, equal_var=False)
            d = (g_vals.mean() - ctrl.mean()) / (
                np.sqrt((g_vals.std()**2 + ctrl.std()**2) / 2) + 1e-12)
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
            lines.append(
                f"  {grp:20s} vs control: "
                f"Δμ={g_vals.mean()-ctrl.mean():+.4f}, "
                f"d={d:+.3f}, p={p:.4f} {sig}"
            )
        lines.append("")

    report = "\n".join(lines)
    path   = os.path.join(outdir, "stats_supplement.txt")
    with open(path, "w") as f:
        f.write(report)
    print(f"\n  Stats saved → {path}")
    print(report[:1200])   # preview


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True,
                        help="CSV output from partial_masking_noise_map.py")
    args = parser.parse_args()

    print(f"Loading {args.input} …")
    df = load_data(args.input)
    print(f"  {len(df)} rows, groups: {df['group'].unique().tolist()}")

    plot_alpha_entropy(df, OUTDIR)
    plot_kl_decay(df,      OUTDIR)
    plot_alpha_star(df,    OUTDIR)
    write_stats(df,        OUTDIR)

    print(f"\nAll supplementary figures saved to: {OUTDIR}/")


if __name__ == "__main__":
    main()
