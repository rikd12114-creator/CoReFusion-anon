"""
Analyze DreamOn-7B refineID benchmark (tiled all-site coverage run) and
produce the figures + summary stats used in the paper.

Inputs (from full-benchmark run on 1000 test samples):
    results/dreamon/DreamOn-7B_per_site_20260509_134411.csv
    results/dreamon/DreamOn-7B_per_sample_20260509_134411.csv
    results/dreamon/DreamOn-7B_refineID_20260509_113722.csv   (old single-window run)

Outputs (under analysis/dreamon_results/):
    plots/fig1_headline_metrics.png
    plots/fig2_em_vs_mask_count.png
    plots/fig3_em_vs_identifier_complexity.png
    plots/fig4_em_vs_site_idx.png
    plots/fig5_top_wrong_predictions.png
    plots/fig6_consistency_vs_accuracy.png
    summary_stats.csv
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

# ---- Paths -----------------------------------------------------------------

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS_DIR = os.path.join(REPO, "results", "dreamon")
OUT_DIR = os.path.join(REPO, "analysis", "dreamon_results")
PLOT_DIR = os.path.join(OUT_DIR, "plots")
os.makedirs(PLOT_DIR, exist_ok=True)

SITE_CSV = os.path.join(RESULTS_DIR, "DreamOn-7B_per_site_20260509_134411.csv")
SAMPLE_CSV = os.path.join(RESULTS_DIR, "DreamOn-7B_per_sample_20260509_134411.csv")
OLD_CSV = os.path.join(RESULTS_DIR, "DreamOn-7B_refineID_20260509_113722.csv")

# Consistent style
plt.rcParams.update({
    "figure.dpi": 130,
    "savefig.dpi": 200,
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": "--",
})
ACCENT = "#1f6feb"
ACCENT_ANY = "#9a6dd7"
WARN = "#c0392b"
GRAY = "#777777"


# ---- Load ------------------------------------------------------------------

site = pd.read_csv(SITE_CSV)
samp = pd.read_csv(SAMPLE_CSV)
old = pd.read_csv(OLD_CSV)

site_em = site["correct"].mean()
samp_maj = samp["majority_correct"].mean()
samp_any = samp["any_correct"].mean()
old_em = old["correct"].mean()


# ---- FIG 1: Headline metrics bar chart -------------------------------------

fig, ax = plt.subplots(figsize=(6.5, 4))
metrics = [
    ("Old (first-mask\nonly, 1 window)", old_em, GRAY),
    ("Sample-level EM\n(majority vote)", samp_maj, ACCENT),
    ("Site-level EM\n(all sites)", site_em, ACCENT),
    ("Sample-level EM\n(any-site correct)", samp_any, ACCENT_ANY),
]
labels, vals, colors = zip(*metrics)
bars = ax.bar(labels, vals, color=colors, edgecolor="white", linewidth=1.2)
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width()/2, v + 0.005, f"{v:.1%}",
            ha="center", va="bottom", fontsize=10, fontweight="bold")
ax.set_ylim(0, max(vals) * 1.25)
ax.set_ylabel("Exact-match accuracy")
ax.set_title("DreamOn-7B on refineID: headline metrics", fontweight="bold")
fig.tight_layout()
fig.savefig(os.path.join(PLOT_DIR, "fig1_headline_metrics.png"))
plt.close(fig)


# ---- FIG 2: EM vs number of [MASK] sites in sample -------------------------

bins = [0, 1, 2, 3, 5, 10, 20, 50, 1000]
labels_b = ["1", "2", "3", "4-5", "6-10", "11-20", "21-50", "50+"]
samp["mask_bucket"] = pd.cut(samp["n_total_masks"], bins=bins, labels=labels_b)

agg = samp.groupby("mask_bucket", observed=True).agg(
    n=("id", "count"),
    maj=("majority_correct", "mean"),
    any_c=("any_correct", "mean"),
).reset_index()

fig, ax = plt.subplots(figsize=(7.5, 4.5))
x = np.arange(len(agg))
w = 0.38
ax.bar(x - w/2, agg["maj"], w, label="Majority vote", color=ACCENT, edgecolor="white")
ax.bar(x + w/2, agg["any_c"], w, label="Any-site correct", color=ACCENT_ANY, edgecolor="white")
ax.set_xticks(x)
ax.set_xticklabels(agg["mask_bucket"])
ax.set_xlabel("Number of [MASK] sites per sample")
ax.set_ylabel("Sample-level EM")
ax.set_title("Accuracy vs. number of mask sites per sample", fontweight="bold")
ax.legend(frameon=False)
# Annotate sample counts above x labels
for i, n in enumerate(agg["n"]):
    ax.text(i, -0.04, f"n={n}", ha="center", va="top", fontsize=8, color=GRAY,
            transform=ax.get_xaxis_transform())
ax.set_ylim(0, max(agg["any_c"].max(), agg["maj"].max()) * 1.20)
fig.tight_layout()
fig.savefig(os.path.join(PLOT_DIR, "fig2_em_vs_mask_count.png"))
plt.close(fig)


# ---- FIG 3: EM vs identifier complexity ------------------------------------

site["gt_chars"] = site["ground_truth"].astype(str).str.len()
site["gt_segs"] = site["ground_truth"].astype(str).str.findall(
    r'[A-Z]?[a-z]+|[A-Z]+').str.len()

site["len_bucket"] = pd.cut(site["gt_chars"], bins=[0, 3, 6, 10, 15, 100],
                             labels=["1-3", "4-6", "7-10", "11-15", "16+"])
len_agg = site.groupby("len_bucket", observed=True).agg(
    em=("correct", "mean"), n=("correct", "count")).reset_index()

site["seg_bucket"] = pd.cut(site["gt_segs"], bins=[-1, 1, 2, 3, 100],
                             labels=["1 word", "2 words", "3 words", "4+ words"])
seg_agg = site.groupby("seg_bucket", observed=True).agg(
    em=("correct", "mean"), n=("correct", "count")).reset_index()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
b1 = ax1.bar(len_agg["len_bucket"].astype(str), len_agg["em"],
             color=ACCENT, edgecolor="white", linewidth=1.2)
for b, v, n in zip(b1, len_agg["em"], len_agg["n"]):
    ax1.text(b.get_x()+b.get_width()/2, v+0.005, f"{v:.1%}\nn={n}",
             ha="center", va="bottom", fontsize=8)
ax1.set_xlabel("Ground-truth identifier length (chars)")
ax1.set_ylabel("Site-level EM")
ax1.set_title("By character length")
ax1.set_ylim(0, len_agg["em"].max() * 1.35)

b2 = ax2.bar(seg_agg["seg_bucket"].astype(str), seg_agg["em"],
             color=ACCENT_ANY, edgecolor="white", linewidth=1.2)
for b, v, n in zip(b2, seg_agg["em"], seg_agg["n"]):
    ax2.text(b.get_x()+b.get_width()/2, v+0.005, f"{v:.1%}\nn={n}",
             ha="center", va="bottom", fontsize=8)
ax2.set_xlabel("CamelCase segment count")
ax2.set_title("By CamelCase complexity")
ax2.set_ylim(0, seg_agg["em"].max() * 1.20)

fig.suptitle("DreamOn EM rises with identifier complexity",
             fontweight="bold", y=1.02)
fig.tight_layout()
fig.savefig(os.path.join(PLOT_DIR, "fig3_em_vs_identifier_complexity.png"),
            bbox_inches="tight")
plt.close(fig)


# ---- FIG 4: EM vs site_idx ------------------------------------------------

site["site_bucket"] = pd.cut(site["site_idx"], bins=[-1, 0, 1, 3, 7, 15, 1000],
                              labels=["#0", "#1", "#2-3", "#4-7", "#8-15", "#16+"])
site_idx_agg = site.groupby("site_bucket", observed=True).agg(
    em=("correct", "mean"), n=("correct", "count")).reset_index()

fig, ax = plt.subplots(figsize=(6.5, 4))
b = ax.bar(site_idx_agg["site_bucket"].astype(str), site_idx_agg["em"],
           color=ACCENT, edgecolor="white", linewidth=1.2)
for bb, v, n in zip(b, site_idx_agg["em"], site_idx_agg["n"]):
    ax.text(bb.get_x()+bb.get_width()/2, v+0.003, f"{v:.1%}\nn={n}",
            ha="center", va="bottom", fontsize=8)
ax.axhline(site_em, color=WARN, linestyle="--", linewidth=1.2,
           label=f"Overall site EM = {site_em:.1%}")
ax.set_xlabel("Site position within sample (0 = first [MASK])")
ax.set_ylabel("Site-level EM")
ax.set_title("Later sites drift: EM drops for #16+", fontweight="bold")
ax.legend(frameon=False, loc="upper right")
ax.set_ylim(0, site_idx_agg["em"].max() * 1.35)
fig.tight_layout()
fig.savefig(os.path.join(PLOT_DIR, "fig4_em_vs_site_idx.png"))
plt.close(fig)


# ---- FIG 5: Top wrong predictions ------------------------------------------

wrong = site[~site["correct"]]
top_wrong = wrong["prediction"].value_counts().head(12)

fig, ax = plt.subplots(figsize=(7.5, 4.5))
y = np.arange(len(top_wrong))
ax.barh(y, top_wrong.values, color=WARN, edgecolor="white", linewidth=1.2)
ax.set_yticks(y)
ax.set_yticklabels(top_wrong.index)
ax.invert_yaxis()
ax.set_xlabel("Count among wrong predictions")
ax.set_title("Most common wrong predictions are generic Java tokens",
             fontweight="bold")
for i, v in enumerate(top_wrong.values):
    ax.text(v + 1, i, str(v), va="center", fontsize=9)
fig.tight_layout()
fig.savefig(os.path.join(PLOT_DIR, "fig5_top_wrong_predictions.png"))
plt.close(fig)


# ---- FIG 6: Self-consistency vs accuracy -----------------------------------

def consistency_score(predictions_str):
    if not predictions_str or pd.isna(predictions_str):
        return 0.0
    preds = predictions_str.split("|")
    if len(preds) <= 1:
        return 1.0
    counts = Counter(preds)
    return counts.most_common(1)[0][1] / len(preds)

samp["consistency"] = samp["predictions"].fillna("").apply(consistency_score)
samp["cons_bucket"] = pd.cut(samp["consistency"],
                              bins=[0, 0.25, 0.5, 0.75, 0.99, 1.01],
                              labels=["0-25%", "25-50%", "50-75%", "75-99%", "100%"])
cons_agg = samp.groupby("cons_bucket", observed=True).agg(
    n=("id", "count"),
    maj=("majority_correct", "mean"),
).reset_index()

fig, ax = plt.subplots(figsize=(7, 4.2))
x = np.arange(len(cons_agg))
b = ax.bar(x, cons_agg["maj"], color=ACCENT, edgecolor="white", linewidth=1.2)
for bb, v, n in zip(b, cons_agg["maj"], cons_agg["n"]):
    ax.text(bb.get_x()+bb.get_width()/2, v+0.004, f"{v:.1%}\nn={n}",
            ha="center", va="bottom", fontsize=8)
ax.set_xticks(x)
ax.set_xticklabels(cons_agg["cons_bucket"].astype(str))
ax.set_xlabel("Self-consistency across sites (mode-count / n_sites)")
ax.set_ylabel("Sample-level EM (majority)")
ax.set_title("Self-consistency is a poor predictor of correctness",
             fontweight="bold")
ax.axhline(samp_maj, color=WARN, linestyle="--", linewidth=1.2,
           label=f"Overall majority EM = {samp_maj:.1%}")
ax.legend(frameon=False, loc="upper left")
ax.set_ylim(0, max(cons_agg["maj"].max(), samp_maj) * 1.45)
fig.tight_layout()
fig.savefig(os.path.join(PLOT_DIR, "fig6_consistency_vs_accuracy.png"))
plt.close(fig)


# ---- summary_stats.csv -----------------------------------------------------

stats = pd.DataFrame([
    {"metric": "site_level_em",          "value": f"{site_em:.4f}",   "n": len(site)},
    {"metric": "sample_majority_em",     "value": f"{samp_maj:.4f}",  "n": len(samp)},
    {"metric": "sample_any_em",          "value": f"{samp_any:.4f}",  "n": len(samp)},
    {"metric": "old_first_mask_em",      "value": f"{old_em:.4f}",    "n": len(old)},
    {"metric": "avg_masks_per_sample",   "value": f"{samp['n_total_masks'].mean():.2f}",  "n": len(samp)},
    {"metric": "median_masks_per_sample","value": f"{samp['n_total_masks'].median():.0f}", "n": len(samp)},
    {"metric": "avg_windows_per_sample", "value": f"{samp['n_windows'].mean():.2f}",      "n": len(samp)},
    {"metric": "total_forward_passes",   "value": f"{samp['n_windows'].sum()}",           "n": len(samp)},
    {"metric": "recoverable_gap_any_vs_majority", "value": f"{samp_any - samp_maj:.4f}",  "n": len(samp)},
    {"metric": "mean_self_consistency",  "value": f"{samp['consistency'].mean():.4f}",    "n": len(samp)},
])
stats.to_csv(os.path.join(OUT_DIR, "summary_stats.csv"), index=False)

print("Wrote:")
for f in sorted(os.listdir(PLOT_DIR)):
    print(" ", os.path.join("plots", f))
print(" ", "summary_stats.csv")
