"""
Deep Analysis: Code Smell as Noise in Diffusion Language Models
===============================================================
Analyzes the 256-step DiffuCoder experiment results.

Outputs:
  results/smell_analysis/  — all figures (PDF + PNG)
  results/smell_analysis/stats_report.txt — full statistical report
"""

import os, re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from scipy import stats
from scipy.stats import mannwhitneyu, kruskal

# ─────────────────────────── Config ────────────────────────────────────────
BASE   = "data/code_smell_256Steps_CoRefusion/"
OUTDIR = "results/smell_analysis"
os.makedirs(OUTDIR, exist_ok=True)

CALIB_F   = BASE + "noise_calibration_diffucoder_20260225_200758.csv"
PROBE_F   = BASE + "noise_probe_diffucoder_20260225_200758.csv"
MAPPING_F = BASE + "noise_mapping_diffucoder_20260225_200758.csv"
NUM_F     = BASE + "noise_numeric_summary_diffucoder_20260225_200758.csv"

# Color palette
COLORS = {
    "mask":     "#E84855",   # red   — pure noise
    "severe":   "#F4791F",   # orange
    "moderate": "#F4B942",   # yellow
    "mild":     "#4DAA57",   # green
    "control":  "#3D7EBF",   # blue  — clean code
}
SEVERITY_ORDER = ["mask", "severe", "moderate", "mild", "control"]
SEVERITY_LABELS = {
    "mask":     "Full Mask\n(Step 0 = 100% noise)",
    "severe":   "Severe Smell\n(e.g. x, a)",
    "moderate": "Moderate Smell\n(e.g. tmp, val)",
    "mild":     "Mild Smell\n(e.g. myVar)",
    "control":  "Clean Code\n(ground truth)",
}

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
})

# ─────────────────────────── Load Data ─────────────────────────────────────
print("Loading data...")
calib   = pd.read_csv(CALIB_F)
probe   = pd.read_csv(PROBE_F)
mapping = pd.read_csv(MAPPING_F)
num     = pd.read_csv(NUM_F)

# Normalise column names (lower + strip)
calib.columns   = [c.strip().lower().replace(" ", "_") for c in calib.columns]
probe.columns   = [c.strip().lower().replace(" ", "_") for c in probe.columns]
mapping.columns = [c.strip().lower().replace(" ", "_") for c in mapping.columns]
num.columns     = [c.strip().lower().replace(" ", "_") for c in num.columns]

print("Calibration shape:", calib.shape)
print("Calibration columns:", list(calib.columns))
print("Probe shape:", probe.shape)
print("Probe columns:", list(probe.columns))
print("Mapping columns:", list(mapping.columns))
print("Numeric summary:\n", num)

# ─────────── Helper: identify step / entropy / confidence columns ───────────
def find_col(df, *candidates):
    """Return the first column name that matches any candidate (case-insensitive)."""
    lower_cols = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower_cols:
            return lower_cols[cand.lower()]
    raise KeyError(f"None of {candidates} found in {list(df.columns)}")

# Calibration columns
calib_step_col   = find_col(calib, "step", "diffusion_step", "noise_step")
calib_entropy_col= find_col(calib, "entropy", "mean_entropy", "avg_entropy")
try:
    calib_conf_col = find_col(calib, "confidence", "argmax_confidence", "mean_confidence")
except KeyError:
    calib_conf_col = None
try:
    calib_gtp_col  = find_col(calib, "gt_token_prob", "gt_prob", "ground_truth_prob")
except KeyError:
    calib_gtp_col = None

# Probe columns
pgroup_col  = find_col(probe, "group", "condition", "type")
psev_col    = find_col(probe, "severity", "smell_severity", "level")
pentropy_col= find_col(probe, "entropy")
ptok_rank_col = find_col(probe, "current_token_rank", "token_rank", "rank")
try:
    pgt_prob_col = find_col(probe, "gt_token_prob", "gt_prob")
except KeyError:
    pgt_prob_col = None
try:
    pconf_col = find_col(probe, "argmax_confidence", "confidence", "argmax_conf")
except KeyError:
    pconf_col = None

print("\n  Using calibration cols → step:", calib_step_col,
      "| entropy:", calib_entropy_col,
      "| confidence:", calib_conf_col,
      "| gt_prob:", calib_gtp_col)
print("  Using probe cols → group:", pgroup_col,
      "| severity:", psev_col,
      "| entropy:", pentropy_col,
      "| rank:", ptok_rank_col)

# ─────────────── Build per-severity probe subsets ───────────────────────────
# Masks / Control
mask_df    = probe[probe[pgroup_col].str.lower() == "mask"].copy()
control_df = probe[probe[pgroup_col].str.lower() == "control"].copy()
severe_df  = probe[(probe[pgroup_col].str.lower() == "smell") &
                   (probe[psev_col].str.lower() == "severe")].copy()
moderate_df= probe[(probe[pgroup_col].str.lower() == "smell") &
                   (probe[psev_col].str.lower() == "moderate")].copy()
mild_df    = probe[(probe[pgroup_col].str.lower() == "smell") &
                   (probe[psev_col].str.lower() == "mild")].copy()

group_dfs = {
    "mask":     mask_df,
    "severe":   severe_df,
    "moderate": moderate_df,
    "mild":     mild_df,
    "control":  control_df,
}
print("\nGroup sizes:")
for k, df in group_dfs.items():
    print(f"  {k}: {len(df)}")

# ─────────── Aggregate calibration curve ────────────────────────────────────
calib_agg = calib.groupby(calib_step_col).agg({
    calib_entropy_col: "mean",
    **({calib_conf_col: "mean"} if calib_conf_col else {}),
    **({calib_gtp_col:  "mean"} if calib_gtp_col else {}),
}).reset_index().sort_values(calib_step_col)

# Step 0 = max noise (all masked), last step = denoised
# Entropy should be highest at step 0
max_step = calib_agg[calib_step_col].max()
print(f"\n  Calibration: max_step={max_step}, entropy range "
      f"[{calib_agg[calib_entropy_col].min():.3f}, {calib_agg[calib_entropy_col].max():.3f}]")

# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 1 — Calibration Curve + Smell Horizontal Lines
# ═══════════════════════════════════════════════════════════════════════════
print("\n[Figure 1] Calibration curve + smell overlays...")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Diffusion Noise Spectrum: Where Do Code Smells Land?",
             fontsize=14, fontweight="bold", y=1.02)

for ax_idx, (metric_col, metric_label, probe_metric) in enumerate([
    (calib_entropy_col, "Shannon Entropy (nats)", pentropy_col),
    (calib_conf_col,    "Argmax Confidence",       pconf_col),
]):
    ax = axes[ax_idx]
    if metric_col is None or probe_metric is None:
        ax.set_visible(False)
        continue

    # Calibration curve
    ax.plot(calib_agg[calib_step_col], calib_agg[metric_col],
            color="#555555", lw=2, label="Diffusion Calibration Curve", zorder=5)

    # Fill the "high noise" zone
    ax.axvspan(0, 30, color="#FFDDDD", alpha=0.35, label="High-noise zone")
    ax.axvspan(200, max_step, color="#DDFFDD", alpha=0.25, label="Low-noise zone")

    # Horizontal dashed lines for each smell severity
    for grp_key in ["mask", "severe", "moderate", "mild", "control"]:
        df = group_dfs[grp_key]
        if df.empty or probe_metric not in df.columns:
            continue
        val  = df[probe_metric].mean()
        sem  = df[probe_metric].sem()
        col  = COLORS[grp_key]
        label = SEVERITY_LABELS[grp_key].split("\n")[0]
        ax.axhline(val, color=col, linestyle="--", lw=1.5, alpha=0.85, label=f"{label}: {val:.3f}")
        ax.fill_between(calib_agg[calib_step_col],
                        val - sem, val + sem, color=col, alpha=0.08)

    ax.set_xlabel(f"Diffusion Step (0 = full noise → {int(max_step)} = denoised)", fontsize=10)
    ax.set_ylabel(metric_label, fontsize=10)
    ax.set_title(f"{metric_label} Along Noise Spectrum", fontsize=11)
    ax.legend(fontsize=7.5, loc="upper left" if ax_idx == 0 else "lower left")
    ax.set_xlim(0, max_step)

fig.tight_layout()
fig.savefig(f"{OUTDIR}/fig1_calibration_overlay.png", bbox_inches="tight")
fig.savefig(f"{OUTDIR}/fig1_calibration_overlay.pdf", bbox_inches="tight")
plt.close(fig)
print("  Saved fig1_calibration_overlay.")

# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 2 — Token-Rank (Model Satisfaction) Bar Chart
# ═══════════════════════════════════════════════════════════════════════════
print("[Figure 2] Token rank satisfaction chart...")
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Model 'Satisfaction' with Current Token\n"
             "(Lower rank = model prefers this token more)",
             fontsize=13, fontweight="bold")

def bar_panel(ax, metric_col, ylabel, title, annotate_delta=True):
    if metric_col is None:
        return
    keys = ["mask", "severe", "moderate", "mild", "control"]
    means, sems, cols, labels = [], [], [], []
    for k in keys:
        df = group_dfs[k]
        if df.empty or metric_col not in df.columns:
            continue
        means.append(df[metric_col].mean())
        sems.append(df[metric_col].sem())
        cols.append(COLORS[k])
        labels.append(SEVERITY_LABELS[k].split("\n")[0])

    x = np.arange(len(means))
    bars = ax.bar(x, means, color=cols, width=0.6, edgecolor="white",
                  linewidth=1.2, yerr=sems, capsize=4, error_kw={"elinewidth":1.2})
    for bar, val in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(means)*0.01,
                f"{val:,.0f}", ha="center", va="bottom", fontsize=9.5, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=11)

bar_panel(axes[0], ptok_rank_col,
          "Avg Rank of Current Token\n(lower = model 'satisfied')",
          "Model Satisfaction with Current Token")
bar_panel(axes[1], pgt_prob_col,
          "P(Ground Truth Token)",
          "Model Preference for Ground Truth Name")

fig.tight_layout()
fig.savefig(f"{OUTDIR}/fig2_token_rank.png", bbox_inches="tight")
fig.savefig(f"{OUTDIR}/fig2_token_rank.pdf", bbox_inches="tight")
plt.close(fig)
print("  Saved fig2_token_rank.")

# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 3 — Entropy Distribution (Box + Violin)
# ═══════════════════════════════════════════════════════════════════════════
print("[Figure 3] Entropy distribution violin plot...")
fig, ax = plt.subplots(figsize=(10, 5.5))
ax.set_title("Entropy Distribution by Code Quality Level\n"
             "(Higher entropy → more noise-like uncertainty)",
             fontsize=13, fontweight="bold")

keys = ["mask", "severe", "moderate", "mild", "control"]
data_list, x_pos, tick_labels, vc = [], [], [], []
for i, k in enumerate(keys):
    df = group_dfs[k]
    if df.empty or pentropy_col not in df.columns:
        continue
    vals = df[pentropy_col].dropna().values
    data_list.append(vals)
    x_pos.append(i)
    tick_labels.append(SEVERITY_LABELS[k])
    vc.append(COLORS[k])

parts = ax.violinplot(data_list, positions=x_pos, showmedians=True,
                      showextrema=False, widths=0.7)
for i, (pc, col) in enumerate(zip(parts["bodies"], vc)):
    pc.set_facecolor(col); pc.set_alpha(0.6)
parts["cmedians"].set_color("black"); parts["cmedians"].set_linewidth(2)

# Overlay individual means
for i, (vals, col) in enumerate(zip(data_list, vc)):
    ax.scatter(i, np.mean(vals), color=col, s=80, zorder=5,
               edgecolors="white", linewidths=1.5)
    ax.text(i, np.mean(vals) + 0.02, f"μ={np.mean(vals):.3f}",
            ha="center", va="bottom", fontsize=8.5, color=col, fontweight="bold")

ax.set_xticks(x_pos)
ax.set_xticklabels(tick_labels, fontsize=9)
ax.set_ylabel("Shannon Entropy (nats)", fontsize=11)
ax.set_xlabel("Code Quality Group", fontsize=11)

# Annotation arrow for the noise interpretation
ax.annotate("← More noise-like\n(uncertain distribution)",
            xy=(0, max([np.mean(d) for d in data_list])),
            fontsize=8.5, color="gray", ha="left")

fig.tight_layout()
fig.savefig(f"{OUTDIR}/fig3_entropy_violin.png", bbox_inches="tight")
fig.savefig(f"{OUTDIR}/fig3_entropy_violin.pdf", bbox_inches="tight")
plt.close(fig)
print("  Saved fig3_entropy_violin.")

# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 4 — Severity Gradient (Key Claim: Monotonic Trend?)
# ═══════════════════════════════════════════════════════════════════════════
print("[Figure 4] Severity gradient (monotonicity check)...")
fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
fig.suptitle("Severity Gradient: Is Code Smell a Continuous Noise Scale?\n"
             "(Monotonic trend → smells form a noise spectrum)",
             fontsize=12, fontweight="bold")

smell_keys   = ["control", "mild", "moderate", "severe"]
smell_labels = ["Clean\nCode", "Mild\nSmell", "Moderate\nSmell", "Severe\nSmell"]
smell_colors = [COLORS["control"], COLORS["mild"], COLORS["moderate"], COLORS["severe"]]

for ax, (metric_col, ylabel, title) in zip(axes, [
    (pentropy_col,   "Shannon Entropy (nats)",         "Entropy (↑ = more noise)"),
    (ptok_rank_col,  "Current Token Rank\n(lower = more satisfied)", "Model Satisfaction (Rank)"),
    (pgt_prob_col,   "P(Ground Truth)",                "P(GT) — Model wants clean name"),
]):
    if metric_col is None:
        ax.set_visible(False)
        continue

    vals = []
    for k in smell_keys:
        df = group_dfs[k]
        if df.empty or metric_col not in df.columns:
            vals.append(np.nan)
        else:
            vals.append(df[metric_col].mean())

    # Line + scatter
    ax.plot(range(len(smell_keys)), vals, color="#555", lw=1.5, zorder=1)
    for xi, (val, col) in enumerate(zip(vals, smell_colors)):
        ax.scatter(xi, val, s=120, color=col, zorder=5,
                   edgecolors="white", linewidths=1.5)
        ax.text(xi, val * 1.03, f"{val:.3f}", ha="center", va="bottom",
                fontsize=9, fontweight="bold")

    # Shade under to emphasise direction
    ax.fill_between(range(len(smell_keys)), vals, min(vals)*0.95,
                    alpha=0.10, color="#888")

    ax.set_xticks(range(len(smell_keys)))
    ax.set_xticklabels(smell_labels, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_title(title, fontsize=10)

fig.tight_layout()
fig.savefig(f"{OUTDIR}/fig4_severity_gradient.png", bbox_inches="tight")
fig.savefig(f"{OUTDIR}/fig4_severity_gradient.pdf", bbox_inches="tight")
plt.close(fig)
print("  Saved fig4_severity_gradient.")

# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 5 — Calibration Curve Zoom: first 60 steps + smell markers
# ═══════════════════════════════════════════════════════════════════════════
print("[Figure 5] Zoomed calibration curve (first 60 steps)...")
fig, ax = plt.subplots(figsize=(10, 5))
ax.set_title("Code Smell Position on Diffusion Noise Schedule (Zoomed)\n"
             "DiffuCoder-7B — 256 steps, 1000 samples",
             fontsize=12, fontweight="bold")

valid = calib_agg[calib_agg[calib_step_col] <= 60]
ax.plot(valid[calib_step_col], valid[calib_entropy_col],
        color="#333", lw=2.5, zorder=5, label="Calibration curve")

# Shade regions
ax.axvspan(0, 15, color="#FFE0E0", alpha=0.4)
ax.axvspan(15, 40, color="#FFF3E0", alpha=0.4)
ax.axvspan(40, 60, color="#E8F5E9", alpha=0.3)

probe_order = ["severe", "moderate", "mild", "control"]
for k in probe_order:
    df = group_dfs[k]
    if df.empty or pentropy_col not in df.columns:
        continue
    val  = df[pentropy_col].mean()
    sem  = df[pentropy_col].sem()
    col  = COLORS[k]
    lab  = SEVERITY_LABELS[k].split("\n")[0]

    # Find equivalent step
    diff = (calib_agg[calib_entropy_col] - val).abs()
    eq_step_row = calib_agg.loc[diff.idxmin()]
    eq_step = eq_step_row[calib_step_col]

    ax.axhline(val, color=col, ls="--", lw=1.4, alpha=0.8)
    ax.fill_between([0, 60], val - sem, val + sem, color=col, alpha=0.07)
    ax.scatter([eq_step], [val], color=col, s=100, zorder=7,
               edgecolors="white", lw=1.5)
    ax.text(eq_step + 0.8, val, f"{lab}\n≈ step {eq_step:.0f}",
            color=col, fontsize=8.5, va="center", fontweight="bold")

ax.set_xlabel(f"Diffusion Step (0 = full noise → {int(max_step)} = denoised)", fontsize=11)
ax.set_ylabel("Shannon Entropy (nats)", fontsize=11)
ax.set_xlim(0, 60)
ax.legend(fontsize=9)

fig.tight_layout()
fig.savefig(f"{OUTDIR}/fig5_calibration_zoomed.png", bbox_inches="tight")
fig.savefig(f"{OUTDIR}/fig5_calibration_zoomed.pdf", bbox_inches="tight")
plt.close(fig)
print("  Saved fig5_calibration_zoomed.")

# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 6 — Calibration Full Curve Overview
# ═══════════════════════════════════════════════════════════════════════════
print("[Figure 6] Full calibration dynamics...")
ncols = sum([calib_entropy_col is not None,
             calib_conf_col is not None,
             calib_gtp_col is not None])
fig, axes = plt.subplots(1, ncols, figsize=(6*ncols, 4.5))
if ncols == 1:
    axes = [axes]
fig.suptitle("Diffusion Denoising Dynamics (Full 256 Steps)\nDiffuCoder-7B",
             fontsize=12, fontweight="bold")

active_metrics = [(c, l) for c, l in [
    (calib_entropy_col, "Shannon Entropy (nats)"),
    (calib_conf_col,    "Argmax Confidence"),
    (calib_gtp_col,     "P(Ground Truth Token)"),
] if c is not None]

for ax, (metric_col, ylabel) in zip(axes, active_metrics):
    ax.plot(calib_agg[calib_step_col], calib_agg[metric_col],
            color="#3D7EBF", lw=2)
    ax.axvspan(0, max_step*0.1, color="#FFE0E0", alpha=0.3, label="High-noise region")
    ax.axvspan(max_step*0.7, max_step, color="#E8F5E9", alpha=0.3, label="Low-noise region")
    ax.set_xlabel(f"Diffusion Step (0→{int(max_step)})", fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(ylabel, fontsize=10)
    ax.set_xlim(0, max_step)
    ax.legend(fontsize=8)

fig.tight_layout()
fig.savefig(f"{OUTDIR}/fig6_calibration_full.png", bbox_inches="tight")
fig.savefig(f"{OUTDIR}/fig6_calibration_full.pdf", bbox_inches="tight")
plt.close(fig)
print("  Saved fig6_calibration_full.")

# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 7 — Summary Heatmap (Numeric Summary)
# ═══════════════════════════════════════════════════════════════════════════
print("[Figure 7] Summary heatmap...")
wanted_cols = ["current_token_rank", "gt_token_rank", "entropy",
               "equiv_step_entropy", "equiv_step_gt_prob"]
available   = [c for c in wanted_cols if c in num.columns]
group_col_n = find_col(num, "group")

if available and group_col_n in num.columns:
    heatmap_data = num.set_index(group_col_n)[available].copy()
    # Normalize each column to [0, 1] for a unified display
    normed = heatmap_data.copy()
    for c in available:
        col_min, col_max = heatmap_data[c].min(), heatmap_data[c].max()
        if col_max > col_min:
            normed[c] = (heatmap_data[c] - col_min) / (col_max - col_min)

    fig, ax = plt.subplots(figsize=(10, 4))
    im = ax.imshow(normed.values, cmap="RdYlGn_r", aspect="auto",
                   vmin=0, vmax=1)
    ax.set_xticks(range(len(available)))
    ax.set_xticklabels([c.replace("_", "\n") for c in available], fontsize=9)
    ax.set_yticks(range(len(heatmap_data)))
    ax.set_yticklabels(heatmap_data.index, fontsize=10)
    ax.set_title("Code Quality Groups — Normalized Metric Heatmap\n"
                 "(Red = noise-like, Green = clean-like)", fontsize=11, fontweight="bold")
    plt.colorbar(im, ax=ax, label="Normalized value (0=clean, 1=noise-like)")

    # Add raw numbers
    for i in range(len(heatmap_data)):
        for j in range(len(available)):
            raw_val = heatmap_data.iloc[i, j]
            fmt = f"{raw_val:,.0f}" if raw_val > 100 else f"{raw_val:.3f}"
            ax.text(j, i, fmt, ha="center", va="center", fontsize=7.5, color="black")

    fig.tight_layout()
    fig.savefig(f"{OUTDIR}/fig7_summary_heatmap.png", bbox_inches="tight")
    fig.savefig(f"{OUTDIR}/fig7_summary_heatmap.pdf", bbox_inches="tight")
    plt.close(fig)
    print("  Saved fig7_summary_heatmap.")

# ═══════════════════════════════════════════════════════════════════════════
# STATISTICAL TESTS
# ═══════════════════════════════════════════════════════════════════════════
print("\n[Statistics] Running tests...")

def stats_between(name_a, name_b, df_a, df_b, col):
    if col is None or col not in df_a.columns or col not in df_b.columns:
        return None
    a = df_a[col].dropna().values
    b = df_b[col].dropna().values
    if len(a) < 3 or len(b) < 3:
        return None
    t, p_t   = stats.ttest_ind(a, b, equal_var=False)
    u, p_u   = mannwhitneyu(a, b, alternative="two-sided")
    d        = (np.mean(a) - np.mean(b)) / (np.sqrt((np.std(a)**2 + np.std(b)**2) / 2) + 1e-12)
    eff_size = ("negligible" if abs(d) < 0.2 else
                "small"      if abs(d) < 0.5 else
                "medium"     if abs(d) < 0.8 else "large")
    sig = ("***" if p_t < 0.001 else "**" if p_t < 0.01 else "*" if p_t < 0.05 else "ns")
    return {
        "A": name_a, "B": name_b, "col": col,
        "mean_A": np.mean(a), "std_A": np.std(a), "n_A": len(a),
        "mean_B": np.mean(b), "std_B": np.std(b), "n_B": len(b),
        "t": t, "p_t": p_t, "U": u, "p_u": p_u,
        "cohen_d": d, "effect": eff_size, "sig": sig,
    }

pairs = [
    ("mask",     "severe"),
    ("mask",     "moderate"),
    ("mask",     "mild"),
    ("mask",     "control"),
    ("severe",   "control"),
    ("moderate", "control"),
    ("mild",     "control"),
    ("severe",   "moderate"),
    ("severe",   "mild"),
    ("moderate", "mild"),
]

metrics = [(pentropy_col, "Entropy"),
           (ptok_rank_col, "CurrentTokenRank"),
           (pgt_prob_col, "GT_Prob")]

report_lines = []
report_lines.append("=" * 72)
report_lines.append("STATISTICAL ANALYSIS — Code Smell as Noise in DiffuCoder-7B")
report_lines.append(f"n per group ≈ 1000 | DiffuCoder-7B-Instruct | 256 diffusion steps")
report_lines.append("=" * 72)

all_results = []
for metric_col, metric_name in metrics:
    if metric_col is None:
        continue
    report_lines.append(f"\n{'─'*60}")
    report_lines.append(f"  Metric: {metric_name}")
    report_lines.append(f"{'─'*60}")
    for (na, nb) in pairs:
        r = stats_between(na, nb, group_dfs[na], group_dfs[nb], metric_col)
        if r is None:
            continue
        all_results.append({**r, "metric": metric_name})
        report_lines.append(
            f"\n  {na} vs {nb}:\n"
            f"    {na}: μ={r['mean_A']:.4f}, σ={r['std_A']:.4f}, n={r['n_A']}\n"
            f"    {nb}: μ={r['mean_B']:.4f}, σ={r['std_B']:.4f}, n={r['n_B']}\n"
            f"    Welch's t: t={r['t']:.4f}, p={r['p_t']:.6f} {r['sig']}\n"
            f"    Mann-Whitney U: U={r['U']:.0f}, p={r['p_u']:.6f}\n"
            f"    Cohen's d: {r['cohen_d']:.4f} ({r['effect']})"
        )

# Kruskal-Wallis across all smell groups
kw_keys = ["severe", "moderate", "mild", "control"]
for metric_col, metric_name in metrics:
    if metric_col is None:
        continue
    data_for_kw = [group_dfs[k][metric_col].dropna().values
                   for k in kw_keys if not group_dfs[k].empty and metric_col in group_dfs[k].columns]
    if len(data_for_kw) >= 2:
        h, p = kruskal(*data_for_kw)
        sig = ("***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns")
        report_lines.append(
            f"\n  Kruskal-Wallis across [severe/moderate/mild/control] — {metric_name}:\n"
            f"    H={h:.4f}, p={p:.6f} {sig}"
        )

report_lines.append("\n" + "=" * 72)
report_lines.append("Significance: *** p<0.001 | ** p<0.01 | * p<0.05 | ns = not significant")
report_lines.append("=" * 72)

report_text = "\n".join(report_lines)
print(report_text)

with open(f"{OUTDIR}/stats_report.txt", "w") as f:
    f.write(report_text)
print(f"\n  Saved stats_report.txt")

# ═══════════════════════════════════════════════════════════════════════════
# KEY FINDINGS SUMMARY
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 72)
print("KEY FINDINGS SUMMARY")
print("=" * 72)

num_g = num.set_index(group_col_n) if group_col_n in num.columns else num

print("\n1. Entropy Gradient (Null hypothesis: all groups equal)")
for k in ["mask", "smell_severe", "smell_moderate", "smell_mild", "control"]:
    if k in num_g.index and "entropy" in num_g.columns:
        print(f"   {k:20s}: entropy = {num_g.loc[k, 'entropy']:.4f}")

print("\n2. Token Rank (model satisfaction with current token)")
for k in ["mask", "smell_severe", "smell_moderate", "smell_mild", "control"]:
    if k in num_g.index and "current_token_rank" in num_g.columns:
        print(f"   {k:20s}: rank = {num_g.loc[k, 'current_token_rank']:.1f}")

print("\n3. Equivalent Noise Step")
for k in ["smell_severe", "smell_moderate", "smell_mild", "control"]:
    if k in num_g.index and "equiv_step_entropy" in num_g.columns:
        print(f"   {k:20s}: ≈ step {num_g.loc[k, 'equiv_step_entropy']:.2f} / 256")

print(f"\nAll figures saved to: {OUTDIR}/")
print("Files: fig1 – fig7 (PNG + PDF), stats_report.txt")
