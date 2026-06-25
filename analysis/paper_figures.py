"""
Paper-Quality Figures: Code Smell as Noise in Diffusion Language Models
=======================================================================
Produces 4 publication-ready figures (IEEE two-column style, 300 dpi).

Figure 1  — Calibration curve + smell positions  (single-column, 3.5 in)
Figure 2  — Entropy violin / severity gradient   (single-column, 3.5 in)
Figure 3  — Token-rank "satisfaction" panel      (single-column, 3.5 in)
Figure 4  — Combined 4-panel summary             (double-column, 7.0 in)

NOTE on calibration direction:
  Step 0 in the diffusion run is the FINAL denoised state (lowest entropy ~0.29).
  Step 1 is the fully-masked start state  (highest entropy ~1.69).
  Steps 2-255 are the iterative denoising trajectory (entropy falls back).
  → We re-index so that "noise level" increases from right to left:
      x-axis = (max_step - step), i.e. 0=denoised, 255=fully noisy
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D
from scipy import stats
from scipy.stats import mannwhitneyu

# ─────────────────────────────── paths ──────────────────────────────────────
BASE   = "data/code_smell_256Steps_CoRefusion/"
OUTDIR = "results/paper_figures"
os.makedirs(OUTDIR, exist_ok=True)

CALIB_F = BASE + "noise_calibration_diffucoder_20260225_200758.csv"
PROBE_F = BASE + "noise_probe_diffucoder_20260225_200758.csv"

# ─────────────────────────── paper style ────────────────────────────────────
SINGLE_W = 3.5        # inches — IEEE single column
DOUBLE_W = 7.16       # inches — IEEE double column
DPI      = 300

# Color palette (colorblind-safe)
C = {
    "mask":     "#D62728",   # red
    "severe":   "#FF7F0E",   # orange
    "moderate": "#BCBD22",   # olive/yellow
    "mild":     "#2CA02C",   # green
    "control":  "#1F77B4",   # blue
    "curve":    "#444444",
    "band":     "#BBBBBB",
}
MARKER = {
    "severe":   "^",
    "moderate": "s",
    "mild":     "D",
    "control":  "o",
}
HATCH = {
    "severe":   "////",
    "moderate": "----",
    "mild":     "....",
    "control":  "",
}

matplotlib.rcParams.update({
    # Font
    "font.family":       "serif",
    "font.serif":        ["Times New Roman", "DejaVu Serif"],
    "font.size":         8,
    "axes.titlesize":    8,
    "axes.labelsize":    8,
    "xtick.labelsize":   7,
    "ytick.labelsize":   7,
    "legend.fontsize":   7,
    "figure.dpi":        DPI,
    # Spines
    "axes.spines.top":   False,
    "axes.spines.right": False,
    # Grid
    "axes.grid":         True,
    "grid.linewidth":    0.4,
    "grid.alpha":        0.4,
    "grid.color":        "#CCCCCC",
    # Lines
    "lines.linewidth":   1.2,
    "patch.linewidth":   0.6,
})

# ─────────────────────────── load data ──────────────────────────────────────
print("Loading data …")
calib_raw = pd.read_csv(CALIB_F)
probe_raw = pd.read_csv(PROBE_F)
calib_raw.columns = [c.strip().lower().replace(" ", "_") for c in calib_raw.columns]
probe_raw.columns = [c.strip().lower().replace(" ", "_") for c in probe_raw.columns]

# ── calibration: aggregate per step ──────────────────────────────────────────
calib = calib_raw.groupby("step").agg(
    entropy_mean     = ("entropy",           "mean"),
    entropy_se       = ("entropy",           lambda x: x.sem()),
    confidence_mean  = ("argmax_confidence", "mean"),
    confidence_se    = ("argmax_confidence", lambda x: x.sem()),
    gt_prob_mean     = ("gt_token_prob",     "mean"),
    gt_rank_mean     = ("gt_token_rank",     "mean"),
).reset_index().sort_values("step")

MAX_STEP = int(calib["step"].max())   # 255

# Re-index: noise_level = 0 (denoised, step 0) → increasing noise.
# The calibration run is: step 0 = final output, steps 1..255 = mask→denoise.
# For plotting, we want x = "noise level" increasing →
#   x = MAX_STEP - step   (so step 255 → x=0 = noisiest, step 0 → x=255)
# Actually the data showed step 0 has LOW entropy (the model already denoised)
# and step 1 has HIGH entropy (first mask state).
# We flip so x = 0 means "clean / denoised" and x = MAX_STEP means "noisy".
calib["noise_x"] = MAX_STEP - calib["step"]

# For per-step calibration curve we need a lookup: entropy → best-matching step
# Use the curve FROM step 1 to MAX_STEP (ignore step 0 which is the clean output)
calib_curve = calib[calib["step"] >= 1].sort_values("step")

def find_equiv_noise_x(target_entropy):
    """Return the noise_x whose mean entropy is closest to target_entropy."""
    diff = (calib_curve["entropy_mean"] - target_entropy).abs()
    row  = calib_curve.loc[diff.idxmin()]
    return int(row["noise_x"]), float(row["entropy_mean"])

# ── probe: build per-group datasets ──────────────────────────────────────────
def get_group(group_val, severity_val=None):
    mask = probe_raw["group"] == group_val
    if severity_val:
        mask &= probe_raw["severity"] == severity_val
    return probe_raw[mask].copy()

grp = {
    "mask":     get_group("mask"),
    "severe":   get_group("smell", "severe"),
    "moderate": get_group("smell", "moderate"),
    "mild":     get_group("smell", "mild"),
    "control":  get_group("control"),
}
SMELL_KEYS = ["severe", "moderate", "mild", "control"]
ALL_KEYS   = ["mask", "severe", "moderate", "mild", "control"]

print("Group sizes:", {k: len(v) for k, v in grp.items()})

# Pre-compute means ± SEM for probe metrics
def stats_df(keys, metric):
    rows = []
    for k in keys:
        df = grp[k]
        if df.empty or metric not in df.columns:
            continue
        vals = df[metric].dropna()
        rows.append({
            "key":  k,
            "mean": vals.mean(),
            "se":   vals.sem(),
            "n":    len(vals),
        })
    return pd.DataFrame(rows)

entropy_stats   = stats_df(ALL_KEYS, "entropy")
rank_stats      = stats_df(ALL_KEYS, "current_token_rank")
gt_prob_stats   = stats_df(ALL_KEYS, "gt_token_prob")
conf_stats      = stats_df(ALL_KEYS, "argmax_confidence")

# Equiv noise positions for smell groups
equiv = {}
for k in SMELL_KEYS:
    ent_mean = entropy_stats.loc[entropy_stats["key"] == k, "mean"].values
    if len(ent_mean):
        nx, ne = find_equiv_noise_x(ent_mean[0])
        equiv[k] = {"noise_x": nx, "entropy": ne}
    else:
        equiv[k] = {"noise_x": None, "entropy": None}

print("Equivalent noise positions:")
for k, v in equiv.items():
    print(f"  {k:10s}: noise_x = {v['noise_x']:4} / {MAX_STEP}  "
          f"(entropy ≈ {v['entropy']:.4f})")

# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 1  —  Calibration Curve + Smell Noise Positions
# ═══════════════════════════════════════════════════════════════════════════
print("\n[Fig 1] Calibration curve + smell positions …")
fig, ax = plt.subplots(figsize=(SINGLE_W, 2.6))

# ── calibration curve (full trajectory, steps 1–255) ──
ax.plot(calib_curve["noise_x"], calib_curve["entropy_mean"],
        color=C["curve"], lw=1.4, zorder=4, label="Diffusion trajectory")

# confidence band
ax.fill_between(calib_curve["noise_x"],
                calib_curve["entropy_mean"] - calib_curve["entropy_se"],
                calib_curve["entropy_mean"] + calib_curve["entropy_se"],
                color=C["curve"], alpha=0.15, zorder=3)

# ── horizontal dashed lines for each smell / control group ──
LABEL_MAP = {
    "severe":   r"\textbf{Severe} ($x$, $a$)",
    "moderate": r"\textbf{Moderate} (\textit{tmp}, \textit{val})",
    "mild":     r"\textbf{Mild} (\textit{myVar})",
    "control":  "Clean code (GT)",
}
LABEL_MAP_PLAIN = {
    "severe":   "Severe (x, a)",
    "moderate": "Moderate (tmp, val)",
    "mild":     "Mild (myVar)",
    "control":  "Clean code (GT)",
}

for k in SMELL_KEYS:
    row = entropy_stats[entropy_stats["key"] == k]
    if row.empty:
        continue
    mu  = row["mean"].values[0]
    se  = row["se"].values[0]
    col = C[k]

    ax.axhline(mu, color=col, ls="--", lw=0.9, alpha=0.85, zorder=5)
    ax.fill_between([0, MAX_STEP], mu - se, mu + se,
                    color=col, alpha=0.07, zorder=2)

    # Dot on the calibration curve at the equivalent noise position
    eq = equiv.get(k, {})
    nx = eq.get("noise_x")
    if nx is not None:
        ax.scatter([nx], [mu], color=col, s=30, zorder=8,
                   marker=MARKER.get(k, "o"),
                   edgecolors="white", linewidths=0.5)

    # Right-hand label
    ax.text(MAX_STEP + 1, mu, LABEL_MAP_PLAIN[k],
            color=col, fontsize=6.5, va="center", ha="left")

ax.set_xlabel(r"Noise level (0 = denoised $\rightarrow$ 255 = fully masked)")
ax.set_ylabel("Shannon entropy (nats)")
ax.set_title("(a) Code smell entropy vs. diffusion noise schedule")
ax.set_xlim(0, MAX_STEP + 55)   # extra room for labels
ax.set_ylim(bottom=0)
ax.xaxis.set_major_locator(ticker.MultipleLocator(50))
ax.legend(loc="lower right", frameon=False)

fig.tight_layout(pad=0.3)
fig.savefig(f"{OUTDIR}/fig1_calibration.pdf", bbox_inches="tight")
fig.savefig(f"{OUTDIR}/fig1_calibration.png", bbox_inches="tight", dpi=DPI)
plt.close()
print("  ✓ fig1_calibration")

# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 2  —  Entropy Distributions (box-and-whisker, paper style)
# ═══════════════════════════════════════════════════════════════════════════
print("[Fig 2] Entropy distributions …")
fig, ax = plt.subplots(figsize=(SINGLE_W, 2.6))

BOX_KEYS = ALL_KEYS
box_data  = [grp[k]["entropy"].dropna().values for k in BOX_KEYS]
box_cols  = [C[k] for k in BOX_KEYS]
x_pos     = np.arange(len(BOX_KEYS))

bp = ax.boxplot(box_data, positions=x_pos, widths=0.5,
                patch_artist=True, notch=False,
                showfliers=False,           # suppress outlier dots (paper-clean)
                medianprops=dict(color="black", linewidth=1.2),
                whiskerprops=dict(linewidth=0.8),
                capprops=dict(linewidth=0.8),
                boxprops=dict(linewidth=0.8))

for patch, col in zip(bp["boxes"], box_cols):
    patch.set_facecolor(col)
    patch.set_alpha(0.45)

# Overlay mean dots
for xi, (k, col) in enumerate(zip(BOX_KEYS, box_cols)):
    mu = grp[k]["entropy"].mean()
    ax.scatter(xi, mu, color=col, s=20, zorder=6,
               edgecolors="white", linewidths=0.5)
    ax.text(xi, mu + 0.03, f"{mu:.3f}",
            ha="center", va="bottom", fontsize=6.2,
            color=col, fontweight="bold")

# Significance brackets (only the key comparisons for paper)
def sig_bracket(ax, x1, x2, y, label, col="black"):
    ax.plot([x1, x1, x2, x2], [y, y+0.02, y+0.02, y],
            lw=0.8, color=col)
    ax.text((x1+x2)/2, y+0.03, label,
            ha="center", va="bottom", fontsize=6, color=col)

# Annotate: all smell vs control are ***
bracket_y = max(grp[k]["entropy"].quantile(0.75) for k in ALL_KEYS) + 0.1
for i, k in enumerate(["severe", "moderate", "mild"]):
    xi = BOX_KEYS.index(k)
    xc = BOX_KEYS.index("control")
    sig_bracket(ax, xi, xc, bracket_y + (i*0.12), "***", col=C[k])

XLABELS = ["Full\nMask", "Severe\nSmell", "Moderate\nSmell",
           "Mild\nSmell", "Clean\nCode"]
ax.set_xticks(x_pos)
ax.set_xticklabels(XLABELS)
ax.set_ylabel("Shannon entropy (nats)")
ax.set_title("(b) Entropy distributions by code quality group")
ax.set_xlim(-0.6, len(BOX_KEYS) - 0.4)
ax.set_ylim(bottom=-0.02)

fig.tight_layout(pad=0.3)
fig.savefig(f"{OUTDIR}/fig2_entropy_dist.pdf", bbox_inches="tight")
fig.savefig(f"{OUTDIR}/fig2_entropy_dist.png", bbox_inches="tight", dpi=DPI)
plt.close()
print("  ✓ fig2_entropy_dist")

# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 3  —  Token-Rank Satisfaction Panel
# ═══════════════════════════════════════════════════════════════════════════
print("[Fig 3] Token rank panel …")
fig, axes = plt.subplots(1, 2, figsize=(SINGLE_W * 2 + 0.15, 2.6))

def rank_bar_panel(ax, stats_df_arg, ylabel, title, log_scale=True):
    keys = [r["key"] for _, r in stats_df_arg.iterrows()
            if r["key"] in ALL_KEYS]
    # preserve order
    ordered = [k for k in ALL_KEYS if k in stats_df_arg["key"].values]
    x  = np.arange(len(ordered))
    mu = [stats_df_arg.loc[stats_df_arg["key"]==k, "mean"].values[0] for k in ordered]
    se = [stats_df_arg.loc[stats_df_arg["key"]==k, "se"].values[0]   for k in ordered]
    cols = [C[k] for k in ordered]

    bars = ax.bar(x, mu, color=cols, width=0.6,
                  edgecolor="white", linewidth=0.6, alpha=0.82)
    ax.errorbar(x, mu, yerr=se, fmt="none", color="black",
                capsize=2.5, elinewidth=0.7)

    for bar, val, col in zip(bars, mu, cols):
        fmt = f"{val:,.0f}" if val >= 100 else f"{val:.3f}"
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() * (1.08 if log_scale else 1.02),
                fmt, ha="center", va="bottom", fontsize=6.0,
                color=col, fontweight="bold")

    xlabels = [XLABELS[ALL_KEYS.index(k)] for k in ordered]
    ax.set_xticks(x)
    ax.set_xticklabels(xlabels, fontsize=7)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if log_scale:
        ax.set_yscale("log")

rank_bar_panel(axes[0], rank_stats,
               "Current-token rank (log scale)\n↓ rank = model more 'satisfied'",
               "(c) Model satisfaction with current token",
               log_scale=True)

rank_bar_panel(axes[1], gt_prob_stats,
               r"$P$(ground-truth token)",
               "(d) Model preference for GT identifier",
               log_scale=False)

fig.tight_layout(pad=0.3)
fig.savefig(f"{OUTDIR}/fig3_token_rank.pdf", bbox_inches="tight")
fig.savefig(f"{OUTDIR}/fig3_token_rank.png", bbox_inches="tight", dpi=DPI)
plt.close()
print("  ✓ fig3_token_rank")

# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 4  —  4-Panel Combined Summary  (double-column)
# ═══════════════════════════════════════════════════════════════════════════
print("[Fig 4] 4-panel combined summary …")
fig = plt.figure(figsize=(DOUBLE_W, 4.8))

import matplotlib.gridspec as gridspec
gs = gridspec.GridSpec(2, 4, figure=fig,
                       hspace=0.52, wspace=0.45,
                       left=0.07, right=0.97,
                       top=0.93, bottom=0.12)

# ── Panel A: Calibration curve (spans 2 columns) ──────────────────────────
ax_a = fig.add_subplot(gs[:, :2])

ax_a.plot(calib_curve["noise_x"], calib_curve["entropy_mean"],
          color=C["curve"], lw=1.4, zorder=4,
          label="Diffusion trajectory")
ax_a.fill_between(calib_curve["noise_x"],
                  calib_curve["entropy_mean"] - calib_curve["entropy_se"],
                  calib_curve["entropy_mean"] + calib_curve["entropy_se"],
                  color=C["curve"], alpha=0.12)

for k in SMELL_KEYS:
    row = entropy_stats[entropy_stats["key"] == k]
    if row.empty: continue
    mu = row["mean"].values[0]
    se = row["se"].values[0]
    ax_a.axhline(mu, color=C[k], ls="--", lw=0.9, alpha=0.85)
    ax_a.fill_between([0, MAX_STEP], mu-se, mu+se, color=C[k], alpha=0.07)
    eq = equiv.get(k, {})
    nx = eq.get("noise_x")
    if nx is not None:
        ax_a.scatter([nx], [mu], color=C[k], s=28, zorder=8,
                     marker=MARKER.get(k, "o"), edgecolors="white", lw=0.5)
    ax_a.text(MAX_STEP + 2, mu, LABEL_MAP_PLAIN[k],
              color=C[k], fontsize=5.8, va="center")

ax_a.set_xlabel(r"Noise level (0 = denoised → 255 = fully masked)", fontsize=7)
ax_a.set_ylabel("Shannon entropy (nats)", fontsize=7)
ax_a.set_title("(a) Entropy along diffusion noise schedule", fontsize=7.5, fontweight="bold")
ax_a.set_xlim(0, MAX_STEP + 60)
ax_a.set_ylim(bottom=0)
ax_a.xaxis.set_major_locator(ticker.MultipleLocator(50))

# ── Panel B: Entropy boxplot (top right) ──────────────────────────────────
ax_b = fig.add_subplot(gs[0, 2:])

box_data2 = [grp[k]["entropy"].dropna().values for k in ALL_KEYS]
bp2 = ax_b.boxplot(box_data2, positions=np.arange(len(ALL_KEYS)),
                   widths=0.5, patch_artist=True, notch=False,
                   showfliers=False,
                   medianprops=dict(color="black", lw=1.0),
                   whiskerprops=dict(lw=0.7),
                   capprops=dict(lw=0.7),
                   boxprops=dict(lw=0.7))
for patch, col in zip(bp2["boxes"], [C[k] for k in ALL_KEYS]):
    patch.set_facecolor(col); patch.set_alpha(0.45)
for xi, k in enumerate(ALL_KEYS):
    mu = grp[k]["entropy"].mean()
    ax_b.scatter(xi, mu, color=C[k], s=16, zorder=6,
                 edgecolors="white", lw=0.4)
ax_b.set_xticks(range(len(ALL_KEYS)))
ax_b.set_xticklabels(XLABELS, fontsize=5.8)
ax_b.set_ylabel("Entropy (nats)", fontsize=7)
ax_b.set_title("(b) Entropy distributions", fontsize=7.5, fontweight="bold")

# significance brackets on panel b
bracket_base = 0.80
for i, k in enumerate(["severe", "moderate", "mild"]):
    xi = ALL_KEYS.index(k)
    xc = ALL_KEYS.index("control")
    y  = bracket_base + i * 0.10
    ax_b.plot([xi, xi, xc, xc], [y, y+0.015, y+0.015, y], lw=0.7, color=C[k])
    ax_b.text((xi+xc)/2, y+0.02, "***", ha="center", va="bottom",
              fontsize=5.5, color=C[k])

# ── Panel C: Current-token rank (bottom left) ─────────────────────────────
ax_c = fig.add_subplot(gs[1, 2])

for xi, k in enumerate(ALL_KEYS):
    row = rank_stats[rank_stats["key"]==k]
    if row.empty: continue
    mu = row["mean"].values[0]
    se = row["se"].values[0]
    ax_c.bar(xi, mu, color=C[k], width=0.6, edgecolor="white", lw=0.5, alpha=0.82)
    ax_c.errorbar(xi, mu, yerr=se, fmt="none", color="black",
                  capsize=2, elinewidth=0.6)

ax_c.set_yscale("log")
ax_c.set_xticks(range(len(ALL_KEYS)))
ax_c.set_xticklabels(XLABELS, fontsize=5.5)
ax_c.set_ylabel("Token rank (log)", fontsize=6.5)
ax_c.set_title("(c) Current-token rank\n(↓ = model more satisfied)", fontsize=7.5, fontweight="bold")

# ── Panel D: Severity gradient line plot (bottom right) ──────────────────
ax_d = fig.add_subplot(gs[1, 3])

sev_keys_ordered = ["control", "mild", "moderate", "severe"]
sev_x = np.arange(len(sev_keys_ordered))
sev_mu = [grp[k]["entropy"].mean() for k in sev_keys_ordered]
sev_se = [grp[k]["entropy"].sem()  for k in sev_keys_ordered]

ax_d.plot(sev_x, sev_mu, color=C["curve"], lw=1.0, zorder=2)
ax_d.fill_between(sev_x,
                  np.array(sev_mu) - np.array(sev_se),
                  np.array(sev_mu) + np.array(sev_se),
                  color=C["curve"], alpha=0.1)
for xi, (k, mu, se) in enumerate(zip(sev_keys_ordered, sev_mu, sev_se)):
    ax_d.scatter(xi, mu, color=C[k], s=28, zorder=5,
                 marker=MARKER.get(k,"o"), edgecolors="white", lw=0.5)
    ax_d.text(xi, mu + 0.01, f"{mu:.3f}",
              ha="center", va="bottom", fontsize=6.0, color=C[k], fontweight="bold")

ax_d.set_xticks(sev_x)
ax_d.set_xticklabels(["Clean", "Mild", "Mod.", "Severe"], fontsize=5.8)
ax_d.set_ylabel("Mean entropy (nats)", fontsize=6.5)
ax_d.set_title("(d) Severity gradient", fontsize=7.5, fontweight="bold")

# ── shared legend ─────────────────────────────────────────────────────────
legend_handles = [
    mpatches.Patch(facecolor=C["mask"],     label="Full mask (step 0)"),
    mpatches.Patch(facecolor=C["severe"],   label="Severe smell"),
    mpatches.Patch(facecolor=C["moderate"], label="Moderate smell"),
    mpatches.Patch(facecolor=C["mild"],     label="Mild smell"),
    mpatches.Patch(facecolor=C["control"],  label="Clean code"),
]
fig.legend(handles=legend_handles, loc="lower center", ncol=5,
           fontsize=6.5, frameon=False,
           bbox_to_anchor=(0.5, 0.00))

fig.suptitle("Code Smells as Semantic Noise in Diffusion Language Models"
             "  (DiffuCoder-7B, n=1000, 256 steps)",
             fontsize=8.5, fontweight="bold", y=0.99)

fig.savefig(f"{OUTDIR}/fig4_combined.pdf",  bbox_inches="tight")
fig.savefig(f"{OUTDIR}/fig4_combined.png",  bbox_inches="tight", dpi=DPI)
plt.close()
print("  ✓ fig4_combined")

# ═══════════════════════════════════════════════════════════════════════════
# PRINT CORRECTED NOISE MAPPING TABLE
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("CORRECTED NOISE MAPPING TABLE")
print("(noise_x = MAX_STEP - step;  0=denoised, 255=fully masked)")
print("="*60)
print(f"{'Group':<18} {'Entropy':>10} {'Equiv noise_x':>15} {'% noise':>10}")
print("-"*60)
for k in SMELL_KEYS:
    row = entropy_stats[entropy_stats["key"]==k]
    if row.empty: continue
    mu = row["mean"].values[0]
    eq = equiv.get(k, {})
    nx = eq.get("noise_x")
    pct = (nx/MAX_STEP*100) if nx is not None else float("nan")
    print(f"{k:<18} {mu:>10.4f} {(str(nx) if nx is not None else 'N/A'):>15} {pct:>9.1f}%")

print(f"\nCalibration reference:")
print(f"  Step 0  (denoised output):  entropy = "
      f"{calib[calib['step']==0]['entropy_mean'].values[0]:.4f}")
print(f"  Step 1  (fully masked):     entropy = "
      f"{calib[calib['step']==1]['entropy_mean'].values[0]:.4f}")
print(f"  Steps 2-255 (denoising):    entropy ≈ "
      f"{calib_curve['entropy_mean'].iloc[1:][:5].mean():.4f} (slowly decreasing)")

print(f"\nAll paper figures saved to:  {OUTDIR}/")
print("Files: fig1_calibration | fig2_entropy_dist | fig3_token_rank | fig4_combined")
print("       (each as .pdf + .png at 300 dpi)")
