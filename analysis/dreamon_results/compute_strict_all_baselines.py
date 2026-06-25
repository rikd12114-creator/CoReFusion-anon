"""
Compute the strict 'all-sites-correct' EM for EVERY refineID baseline.

A sample is correct iff every [MASK] site in the sample is recovered
identically to the ground-truth identifier.

For CSVs with the `all_predictions` column (pipe-separated per-site
predictions stored at benchmark time), we use it directly. For CSVs
that only stored `full_code` (the model's generated code with masks
filled in), we re-extract per-site predictions by aligning the
ground-truth masked_code against full_code using the same anchor-based
extractor used by ``experiments/benchmark_diffusion_models.py``.

Schema-C dumps (``X_original / y_ground_truth / output_full`` with
hundreds-of-thousands of rows) are intermediate per-token logs, not
sample-level results -- we skip them.

Outputs:
    analysis/dreamon_results/strict_all_correct_comparison.csv
    analysis/dreamon_results/plots/fig7_strict_all_correct_comparison.png
"""

import os
import sys
import re
import csv
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ---- Paths -----------------------------------------------------------------

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS = os.path.join(REPO, "results")
OUT_DIR = os.path.join(REPO, "analysis", "dreamon_results")
PLOTS = os.path.join(OUT_DIR, "plots")
DATA_CSV = os.path.join(REPO, "data", "test.csv")
os.makedirs(PLOTS, exist_ok=True)


# ---- Per-site extractor (verbatim from benchmark_diffusion_models.py) ------

def extract_all_predictions(full_code, masked_code):
    """Extract one identifier per [MASK] site by aligning masked_code on full_code."""
    parts = str(masked_code).split("[MASK]")
    if len(parts) <= 1:
        return []
    predictions = []
    cursor = 0
    for i in range(len(parts) - 1):
        pre = parts[i]
        post = parts[i + 1]
        pre_anchor = pre.strip()[-30:] if len(pre.strip()) > 30 else pre.strip()
        post_anchor = post.strip()[:30] if len(post.strip()) > 30 else post.strip()
        if pre_anchor:
            idx_start = full_code.find(pre_anchor, cursor)
            idx_start = idx_start + len(pre_anchor) if idx_start != -1 else cursor
        else:
            idx_start = cursor
        if post_anchor:
            idx_end = full_code.find(post_anchor, idx_start)
        else:
            idx_end = -1
        if idx_end != -1:
            gap = full_code[idx_start:idx_end].strip()
            cursor = idx_end
        else:
            gap = full_code[idx_start:idx_start + 60].strip()
            cursor = idx_start + 60
        m = re.search(r'[a-zA-Z_$][a-zA-Z0-9_$]*', gap)
        predictions.append(m.group(0) if m else gap[:20])
    return predictions


# ---- Test-set lookup: id -> masked_code -----------------------------------

print(f"Loading {DATA_CSV}...")
csv.field_size_limit(sys.maxsize)
id_to_masked = {}
with open(DATA_CSV, "r", encoding="utf-8") as f:
    reader = csv.reader(f)
    for row in reader:
        try:
            id_to_masked[int(row[0])] = row[1]
        except (ValueError, IndexError):
            continue
print(f"Loaded masked_code for {len(id_to_masked)} samples.\n")


# ---- Scoring helpers -------------------------------------------------------

def strict_all(preds, gt):
    """True iff preds is non-empty and every entry equals gt."""
    if not preds:
        return False
    gt = str(gt).strip()
    return all(str(p).strip() == gt for p in preds)


def score_with_all_predictions(df):
    """For CSVs that have an `all_predictions` column."""
    n_total, n_correct, n_first_correct = 0, 0, 0
    for _, row in df.iterrows():
        preds_str = row.get("all_predictions", "")
        if not isinstance(preds_str, str) or not preds_str.strip():
            n_total += 1
            continue
        preds = [p.strip() for p in preds_str.split("|")]
        n_total += 1
        if strict_all(preds, row["ground_truth"]):
            n_correct += 1
        if preds and preds[0] == str(row["ground_truth"]).strip():
            n_first_correct += 1
    return n_correct, n_first_correct, n_total


def score_with_full_code(df, id_to_masked):
    """For CSVs that only have `full_code`; re-extract per-site predictions."""
    n_total, n_correct, n_first_correct, n_skipped = 0, 0, 0, 0
    for _, row in df.iterrows():
        n_total += 1
        try:
            sid = int(row["id"])
        except (ValueError, TypeError):
            n_skipped += 1
            continue
        masked = id_to_masked.get(sid)
        full = row.get("full_code", "")
        if not masked or not isinstance(full, str) or not full:
            n_skipped += 1
            continue
        preds = extract_all_predictions(full, masked)
        if not preds:
            continue
        if strict_all(preds, row["ground_truth"]):
            n_correct += 1
        if preds[0] == str(row["ground_truth"]).strip():
            n_first_correct += 1
    return n_correct, n_first_correct, n_total, n_skipped


# ---- File registry --------------------------------------------------------

REGISTRY = [
    # (display_name, path, scoring_mode)
    # scoring_mode in {"all_preds", "full_code", "stored_correct_only"}
    # ----- Existing baselines (refine_id_benchmark/) -----
    ("StarCoder2-7B",
     f"{RESULTS}/refine_id_benchmark/StarCoder2-7B_refineID_results_20260206_123938.csv",
     "all_preds"),
    ("DeepSeek-Coder-6.7B-Base",
     f"{RESULTS}/refine_id_benchmark/DeepSeek-Coder-6.7B-Base_refineID_results_20260206_124149.csv",
     "all_preds"),
    ("DeepSeek-Coder-6.7B-Instruct",
     f"{RESULTS}/refine_id_benchmark/DeepSeek-Coder-6.7B-Instruct_refineID_results(1).csv",
     "full_code"),
    ("DiffuCoder-7B (Base, noPrompt)",
     f"{RESULTS}/refine_id_benchmark/DiffuCoder_Base_noPrompt_20260206_091137.csv",
     "full_code"),
    ("DiffuCoder-7B",
     f"{RESULTS}/refine_id_benchmark/DiffuCoder_refineID_results_20260206_091826.csv",
     "full_code"),
    ("DreamCoder-7B (32 steps, noPrompt)",
     f"{RESULTS}/refine_id_benchmark/DreamCoder32_noPrompt_20260206_094435.csv",
     "full_code"),
    ("DreamCoder-7B (64 steps, noPrompt)",
     f"{RESULTS}/refine_id_benchmark/DreamCoder64_noPrompt_20260206_093622.csv",
     "full_code"),
    ("Llama-3.1-8B-Instruct",
     f"{RESULTS}/refine_id_benchmark/Llama-3.1-8B-Instruct_refineID_results(1).csv",
     "full_code"),
    ("Qwen2.5-Coder-7B-Instruct",
     f"{RESULTS}/refine_id_benchmark/Qwen2.5-Coder-7B-Instruct_refineID_results(1).csv",
     "full_code"),
    # ----- CodeT5p (newer) -----
    ("CodeT5p-16B",
     f"{RESULTS}/codeT5results/CodeT5p-16B_refineID_fim_20260509_134747.csv",
     "all_preds"),
]


# ---- Score every baseline -------------------------------------------------

print("=" * 95)
print(f"{'Model':<38} {'n':>5} {'1st-EM':>10} {'STRICT all-EM':>14}  source")
print("=" * 95)

rows_out = []
for name, path, mode in REGISTRY:
    if not os.path.exists(path):
        print(f"{name:<38} {'-':>5} {'-':>10} {'-':>14}  MISSING")
        continue
    df = pd.read_csv(path)
    n = len(df)

    if mode == "all_preds":
        nc, nfc, nt = score_with_all_predictions(df)
        strict = nc / nt if nt else 0.0
        first = nfc / nt if nt else 0.0
        src = "all_predictions"
    elif mode == "full_code":
        nc, nfc, nt, nsk = score_with_full_code(df, id_to_masked)
        strict = nc / nt if nt else 0.0
        first = nfc / nt if nt else 0.0
        src = f"full_code re-extracted ({nsk} skipped)"
    else:
        strict, first, src = None, None, "n/a"

    print(f"{name:<38} {n:>5} {first:>10.4f} {strict:>14.4f}  {src}")
    rows_out.append({
        "model": name, "n": n,
        "first_mask_em": f"{first:.4f}",
        "strict_all_em": f"{strict:.4f}",
        "source": src,
    })


# ---- Add DreamOn rows -----------------------------------------------------

dream_new = pd.read_csv(
    f"{RESULTS}/dreamon/DreamOn-7B_per_sample_20260509_134411.csv")
nc, nfc = 0, 0
for _, row in dream_new.iterrows():
    preds = [p.strip() for p in str(row.get("predictions", "")).split("|") if p.strip()]
    if strict_all(preds, row["ground_truth"]):
        nc += 1
    if preds and preds[0] == str(row["ground_truth"]).strip():
        nfc += 1
strict_new = nc / len(dream_new)
first_new = nfc / len(dream_new)
rows_out.append({
    "model": "DreamOn-7B (NEW: tiled, all-site coverage)",
    "n": len(dream_new),
    "first_mask_em": f"{first_new:.4f}",
    "strict_all_em": f"{strict_new:.4f}",
    "source": "predictions (per_sample CSV)",
})
print(f"{'DreamOn-7B (NEW: tiled)':<38} {len(dream_new):>5} "
      f"{first_new:>10.4f} {strict_new:>14.4f}  predictions column")

dream_old = pd.read_csv(
    f"{RESULTS}/dreamon/DreamOn-7B_refineID_20260509_113722.csv")
first_old = dream_old["correct"].mean()
rows_out.append({
    "model": "DreamOn-7B (OLD: single-window, first-mask)",
    "n": len(dream_old),
    "first_mask_em": f"{first_old:.4f}",
    "strict_all_em": "N/A",
    "source": "per-site preds not stored",
})
print(f"{'DreamOn-7B (OLD)':<38} {len(dream_old):>5} {first_old:>10.4f} "
      f"{'N/A':>14}  per-site preds not stored")


# ---- Save CSV -------------------------------------------------------------

out_csv = f"{OUT_DIR}/strict_all_correct_comparison.csv"
with open(out_csv, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["model", "n", "first_mask_em",
                                       "strict_all_em", "source"])
    w.writeheader()
    w.writerows(rows_out)
print(f"\nWrote {out_csv}")


# ---- Figure: strict vs first-mask, sorted by strict EM -------------------

plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 200, "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linestyle": "--",
})

plot_rows = [r for r in rows_out if r["strict_all_em"] not in ("N/A", None)]
plot_rows.sort(key=lambda r: float(r["strict_all_em"]), reverse=True)

labels = [r["model"] for r in plot_rows]
first_vals = [float(r["first_mask_em"]) for r in plot_rows]
strict_vals = [float(r["strict_all_em"]) for r in plot_rows]

fig, ax = plt.subplots(figsize=(11, 5.5))
y = np.arange(len(labels))
h = 0.38
b1 = ax.barh(y + h/2, first_vals, h, label="First-mask EM (lenient)",
             color="#bbbbbb", edgecolor="white", linewidth=1.2)
b2 = ax.barh(y - h/2, strict_vals, h, label="Strict all-sites-correct EM",
             color="#1f6feb", edgecolor="white", linewidth=1.2)

# Annotate values
for bb, v in zip(b1, first_vals):
    ax.text(v + 0.003, bb.get_y() + bb.get_height()/2, f"{v:.1%}",
            va="center", fontsize=8.5)
for bb, v in zip(b2, strict_vals):
    ax.text(v + 0.003, bb.get_y() + bb.get_height()/2, f"{v:.1%}",
            va="center", fontsize=8.5, fontweight="bold")

# Highlight DreamOn rows
for i, lbl in enumerate(labels):
    if "DreamOn" in lbl:
        ax.get_yticklabels()  # ensure ticks exist
        ax.axhspan(i - 0.5, i + 0.5, color="#fff3cd", alpha=0.6, zorder=0)

ax.set_yticks(y)
ax.set_yticklabels(labels, fontsize=9.5)
ax.invert_yaxis()
ax.set_xlabel("Exact-match accuracy")
ax.set_title("refineID: Strict all-sites-correct vs. First-mask EM (all baselines)",
             fontweight="bold")
ax.legend(frameon=False, loc="lower right")
ax.set_xlim(0, max(first_vals + strict_vals) * 1.15)

fig.tight_layout()
fig.savefig(f"{PLOTS}/fig7_strict_all_correct_comparison.png",
            bbox_inches="tight")
plt.close(fig)
print(f"Wrote {PLOTS}/fig7_strict_all_correct_comparison.png")
