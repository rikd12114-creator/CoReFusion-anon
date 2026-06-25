#!/usr/bin/env python3
"""
RQ3 v2 — figures (matplotlib only; reads precomputed JSON, no model/sklearn).

  Fig 1 (Exp1, 2 panels): contextual-AUC(layer) curve with bootstrap band and the
        baseline/control reference lines (left) + the 1D probe-projection histogram
        good vs bad at the chosen layer (right) -> REPLACES the UMAP scatter.
  Fig 2 (Exp2): detection AUC(step) on still-masked positions + commitment CDF on
        one shared step axis; peak-ROC and median-commit marked, summary in caption.
  Fig 3 (output-distribution inversion): r_gt vs r_smell across the three regimes,
        showing the RareConfident sign flip. Numbers default to the paper's Table VII
        (override with --regime-csv id,regime,r_gt,r_smell).

Outputs to figures/new/rq3/ as .png + .pdf.
Python 3.11 safe.
"""

import os
import json
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

matplotlib.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 6.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.linewidth": 0.4, "grid.alpha": 0.4,
})

# Table VII (paper) defaults — median r_gt / r_smell by regime.
PAPER_REGIMES = [("HighConfident", 77, 492), ("Uncertain", 524, 676),
                 ("RareConfident", 10989, 733)]


def rank_auc(scores, labels):
    """AUC via Mann-Whitney U, no sklearn."""
    scores = np.asarray(scores); labels = np.asarray(labels)
    pos = scores[labels == 1]; neg = scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(np.concatenate([pos, neg]))
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(order) + 1)
    r_pos = ranks[:len(pos)].sum()
    return float((r_pos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def save(fig, outdir, name):
    fig.savefig(os.path.join(outdir, name + ".png"), dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(outdir, name + ".pdf"), bbox_inches="tight")
    plt.close(fig)
    print("  -> %s.{png,pdf}" % name)


def fig1(out_dir, fig_dir, cap_n):
    d = json.load(open(os.path.join(out_dir, "exp1_curves.json")))
    c = d["curves"]
    L = d["n_layers"]
    xs = np.arange(L)
    chosen = d["chosen_layer"]

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(7.0, 2.7))

    # LEFT: contextual AUC + references
    contextual = np.array(c["contextual_pkg"])
    axL.plot(xs, c["intact_pkg"], color="#999999", lw=0.9, ls="--", label="AUC intact")
    axL.plot(xs, c["scrambled_pkg"], color="#cccccc", lw=0.9, ls=":", label="AUC scrambled ctx")
    lo = np.array(c["intact_pkg_lo"]); hi = np.array(c["intact_pkg_hi"])
    finite = ~np.isnan(lo)
    if finite.any():
        axL.fill_between(xs[finite], lo[finite], hi[finite], color="#3498db", alpha=0.15, lw=0)
    axL.plot(xs, contextual, color="#e74c3c", lw=1.6, label="AUC contextual (intact-scrambled)")
    axL.axhline(0.5, color="k", lw=0.5, alpha=0.5)
    axL.plot(xs, c["control"], color="#7f8c8d", lw=0.7, alpha=0.7, label="random-label control")
    axL.axhline(d["length_baseline_auc"], color="#27ae60", lw=0.7, ls="-.",
                alpha=0.8, label="sub-word-count baseline")
    third = max(1, L // 3)
    axL.axvspan(L - third, L - 1, color="#f1c40f", alpha=0.10, lw=0)
    axL.axvline(chosen, color="#e74c3c", lw=0.6, alpha=0.6)
    axL.annotate("chosen L=%d\ncontextual AUC=%.2f" % (chosen, d["contextual_at_chosen"]),
                 xy=(chosen, contextual[chosen]), xytext=(max(0, chosen - 11), 0.78),
                 fontsize=6.5, arrowprops=dict(arrowstyle="->", lw=0.5))
    axL.set_xlabel("Transformer layer (depth)")
    axL.set_ylabel("ROC-AUC  (good vs length-matched misplaced name)")
    axL.set_ylim(0.45, 1.0)
    axL.set_title("(a) Name-fit decodability vs depth")
    axL.legend(loc="lower left", frameon=False, ncol=1)

    # RIGHT: 1D probe-projection histogram (REPLACES UMAP)
    sc = np.array(d["hist"]["scores"]); lab = np.array(d["hist"]["labels"])
    auc = rank_auc(sc, lab)
    bins = np.linspace(sc.min(), sc.max(), 30)
    axR.hist(sc[lab == 1], bins=bins, color="#3498db", alpha=0.6, label="good (developer name)")
    axR.hist(sc[lab == 0], bins=bins, color="#e74c3c", alpha=0.6, label="bad (misplaced name)")
    axR.axvline(0.0, color="k", lw=0.6, ls="--", alpha=0.7)
    axR.set_xlabel("probe-weight projection  w.h  (signed distance to boundary)")
    axR.set_ylabel("count")
    axR.set_title("(b) Layer %d separation (AUC=%.2f)" % (chosen, auc))
    axR.legend(loc="upper right", frameon=False)

    fig.suptitle("RQ3 Exp1 — DiffuCoder-7B-Base, %s" % cap_n, fontsize=7.5, y=1.02)
    save(fig, fig_dir, "fig_rq3_exp1_depth_probe")


def fig2(out_dir, fig_dir, cap_n):
    d = json.load(open(os.path.join(out_dir, "exp2_curves.json")))
    T = d["T"]
    xs = np.arange(T)
    roc = np.array(d["roc"], dtype=float)
    cdf = np.array(d["commit_cdf"], dtype=float)

    fig, ax = plt.subplots(figsize=(4.4, 3.0))
    ax.plot(xs, cdf, color="#2c3e50", lw=1.8, label="commitment CDF (first unmask)")
    ax.plot(xs, roc, color="#e74c3c", lw=1.8, label="detection ROC-AUC (still-masked)")
    ax.plot(xs, d["pr"], color="#e67e22", lw=1.0, ls="--", label="detection PR-AUC")
    ax.axhline(d["base_rate"], color="#7f8c8d", lw=0.8, ls=":",
               label="no-skill (EM=%.2f)" % d["base_rate"])
    # Subtle reference lines at the detection peak and the commitment median.
    # The quantitative summary lives in the caption, not in an on-figure box,
    # and the all-censored 3x3 DCL inset (detection never clears ROC 0.70) is
    # dropped because it carried no readable content.
    pk = int(np.nanargmax(roc))
    cmed = next((t for t in range(T) if cdf[t] >= 0.5), T)
    ax.axvline(pk, color="#e74c3c", lw=0.7, ls=":", alpha=0.45)
    ax.axvline(cmed, color="#2c3e50", lw=0.7, ls=":", alpha=0.45)
    ax.set_xlabel("Denoising step")
    ax.set_ylabel("AUC / committed fraction")
    ax.set_xlim(0, T - 1)
    ax.set_ylim(0, 1.02)
    # no title -- the paper caption carries it
    ax.legend(loc="upper left", frameon=True, framealpha=0.9, fontsize=7,
              borderpad=0.4, handlelength=1.8, labelspacing=0.3)
    save(fig, fig_dir, "fig_rq3_exp2_dcl")


def fig3(fig_dir, regime_csv, cap_n):
    regimes = PAPER_REGIMES
    if regime_csv and os.path.exists(regime_csv):
        import csv
        agg = {}
        for r in csv.DictReader(open(regime_csv)):
            agg.setdefault(r["regime"], []).append((float(r["r_gt"]), float(r["r_smell"])))
        order = ["HighConfident", "Uncertain", "RareConfident"]
        regimes = [(k, float(np.median([a for a, _ in agg[k]])),
                    float(np.median([b for _, b in agg[k]]))) for k in order if k in agg]

    fig, ax = plt.subplots(figsize=(3.6, 2.8))
    for i, (name, rgt, rsm) in enumerate(regimes):
        ax.plot([0, 1], [rgt, rsm], "-o", lw=1.4, ms=4,
                color=["#3498db", "#9b59b6", "#e74c3c"][i % 3], label=name)
        ax.annotate("%d" % rgt, (0, rgt), textcoords="offset points", xytext=(-3, 2),
                    ha="right", fontsize=6)
        ax.annotate("%d" % rsm, (1, rsm), textcoords="offset points", xytext=(3, 2),
                    ha="left", fontsize=6)
    ax.set_yscale("log")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["good\n(developer)", "smell\n(injected)"])
    ax.set_ylabel("median rank in output distribution (log)")
    ax.set_title("RQ3 — regime-dependent ranking; rare names invert")
    ax.legend(frameon=False, loc="upper center")
    save(fig, fig_dir, "fig_rq3_regime_inversion")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default="results/rq3_probe")
    ap.add_argument("--fig-dir", default="figures/new/rq3")
    ap.add_argument("--regime-csv", default=None,
                    help="optional id,regime,r_gt,r_smell to recompute Fig 3 on real data.")
    args = ap.parse_args()
    os.makedirs(args.fig_dir, exist_ok=True)

    meta_p = os.path.join(args.out_dir, "extract_meta.json")
    cap_n = "n=230 snippets / 1055 sites / 64 packages"
    if os.path.exists(meta_p):
        m = json.load(open(meta_p))
        cap_n = "n=%d snippets / %d sites / %d packages" % (m["snippets"], m["sites"], m["packages"])

    if os.path.exists(os.path.join(args.out_dir, "exp1_curves.json")):
        fig1(args.out_dir, args.fig_dir, cap_n)
    if os.path.exists(os.path.join(args.out_dir, "exp2_curves.json")):
        fig2(args.out_dir, args.fig_dir, cap_n)
    fig3(args.fig_dir, args.regime_csv, cap_n)


if __name__ == "__main__":
    main()
