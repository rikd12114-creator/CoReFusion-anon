"""Plot Exact Match (EM) stratified by number of smell positions |S| per sample.

Reads per-model benchmark CSVs (one row per smell position with `correct` and
`mask_count`) and aggregates EM per sample, then bins by |S| in
{1, 2, 3-5, 6-10, 11+}.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIFF_DIR = os.path.join(ROOT, "data/benchmark_ReFineID_Diffusion/diffusion_benchmark")
FIM_DIR  = os.path.join(ROOT, "data/benchmark_ReFineID_FIM/ar_fim_benchmark")
OUTDIR   = os.path.join(ROOT, "results/paper_figures")
os.makedirs(OUTDIR, exist_ok=True)

# (display_name, file_path, family)
MODELS = [
    ("DreamCoder-7B",    f"{DIFF_DIR}/DreamCoder-7B_refineID_diffusion_20260226_131226.csv", "diffusion"),
    ("DiffuCoder-7B",    f"{DIFF_DIR}/DiffuCoder-7B_refineID_diffusion_20260226_125550.csv", "diffusion"),
    ("CodeLlama-7B",     f"{FIM_DIR}/CodeLlama-7B_refineID_fim_20260226_100325.csv",         "ar"),
    ("CodeLlama-13B",    f"{FIM_DIR}/CodeLlama-13B_refineID_fim_20260226_102052.csv",        "ar"),
    ("StarCoder2-15B",   f"{FIM_DIR}/StarCoder2-15B_refineID_fim_20260225_193323.csv",       "ar"),
    ("Qwen2.5-Coder-7B", f"{FIM_DIR}/Qwen2.5-Coder-7B_refineID_fim_20260226_014318.csv",     "ar"),
]

BIN_EDGES  = [(1, 1), (2, 2), (3, 5), (6, 10), (11, np.inf)]
BIN_LABELS = ["|S| = 1", "|S| = 2", "|S| = 3-5", "|S| = 6-10", "|S| = 11+"]

STYLE = {
    "DreamCoder-7B":    dict(color="#D62728", marker="o", mfc="#D62728"),
    "DiffuCoder-7B":    dict(color="#D62728", marker="o", mfc="white"),
    "CodeLlama-7B":     dict(color="#1F77B4", marker="o", mfc="white"),
    "CodeLlama-13B":    dict(color="#1F77B4", marker="s", mfc="#1F77B4"),
    "StarCoder2-15B":   dict(color="#1F77B4", marker="^", mfc="white"),
    "Qwen2.5-Coder-7B": dict(color="#1F77B4", marker="D", mfc="white"),
}

def load_per_sample_em(path: str) -> pd.DataFrame:
    """Return DataFrame with one row per sample id: columns [id, em, mask_count].

    The benchmark CSV has one row per (id, masked-position). A sample is
    considered exact-matched if every position is correct.
    """
    df = pd.read_csv(path)
    df["correct"] = df["correct"].astype(str).str.lower().isin(["true", "1"])
    df["mask_count"] = pd.to_numeric(df["mask_count"], errors="coerce")
    df = df.dropna(subset=["mask_count"])
    g = df.groupby("id").agg(
        em=("correct", "all"),
        mask_count=("mask_count", "first"),
    ).reset_index()
    return g

def bin_index(n: float) -> int:
    for i, (lo, hi) in enumerate(BIN_EDGES):
        if lo <= n <= hi:
            return i
    return -1

def em_per_bin(df: pd.DataFrame):
    out = np.full(len(BIN_EDGES), np.nan)
    counts = np.zeros(len(BIN_EDGES), dtype=int)
    df = df.copy()
    df["bin"] = df["mask_count"].apply(bin_index)
    for i in range(len(BIN_EDGES)):
        sub = df[df["bin"] == i]
        counts[i] = len(sub)
        if len(sub) > 0:
            out[i] = sub["em"].mean() * 100.0
    return out, counts

# ── compute ─────────────────────────────────────────────────────────────────
results = {}
for name, path, fam in MODELS:
    if not os.path.exists(path):
        print(f"[skip] missing: {path}")
        continue
    per_sample = load_per_sample_em(path)
    em, n = em_per_bin(per_sample)
    results[name] = (em, n)
    print(f"{name:<22}  n_per_bin={n.tolist()}  EM%={np.round(em,2).tolist()}")

# ── plot ────────────────────────────────────────────────────────────────────
plt.rcParams.update({"font.family": "serif", "font.size": 9})
fig, ax = plt.subplots(figsize=(7.0, 3.2))
ax.set_facecolor("#FFF9E6")

x = np.arange(len(BIN_LABELS))
for name, _, _ in MODELS:
    if name not in results:
        continue
    em, n = results[name]
    s = STYLE[name]
    ax.plot(x, em,
            color=s["color"], marker=s["marker"],
            mfc=s["mfc"], mec=s["color"],
            mew=1.2, ms=7, lw=1.4, label=name)
    # Annotate any bin where the marker would be hidden by another series:
    # we still ensure the point itself is plotted (even when n is small).
    for xi, yi, ni in zip(x, em, n):
        if not np.isnan(yi) and ni < 30:
            ax.annotate(f"n={ni}", (xi, yi), textcoords="offset points",
                        xytext=(0, 8), fontsize=6, color=s["color"], ha="center")

ax.set_xticks(x)
ax.set_xticklabels(BIN_LABELS)
ax.set_xlabel("Number of smell positions per sample")
ax.set_ylabel("Exact Match (%)")
ax.set_ylim(0, max(60, np.nanmax([results[m][0].max() for m in results if m in results]) + 8))
ax.grid(True, ls="--", lw=0.4, alpha=0.5)
ax.text(0.02, 0.96, "multi-s", transform=ax.transAxes,
        fontsize=8, style="italic", color="#B8860B", va="top")

ax.legend(loc="upper right", ncol=2, frameon=True, fontsize=7.5,
          framealpha=0.95)

fig.tight_layout()
out_png = os.path.join(OUTDIR, "fig_em_stratified_by_smell_positions.png")
out_pdf = os.path.join(OUTDIR, "fig_em_stratified_by_smell_positions.pdf")
fig.savefig(out_png, dpi=300, bbox_inches="tight")
fig.savefig(out_pdf, bbox_inches="tight")
plt.close()
print(f"\nSaved:\n  {out_png}\n  {out_pdf}")
