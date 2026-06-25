#!/usr/bin/env python3
"""
RQ3 cross-model comparison (Mali's request): overlay the contextual name-fit
decodability curve AUC_intact - AUC_scrambled vs layer for two (or more) models,
e.g. DiffuCoder-7B-Base vs DreamCoder-7B. Reads each model's exp1_curves.json
from its own --dir. Robust if a model's dir is missing (skips with a warning), so
it can be run now with DiffuCoder alone and re-run once DreamCoder is extracted.

Run DreamCoder Exp1 first (no code change, just the checkpoint):
  sbatch server/jobs/rq3_extract.slurm \
     --model Dream-org/Dream-Coder-v0-Instruct-7B \
     --out-dir results/rq3_probe_dreamcoder --skip-exp2
  python experiments/rq3_probe/train_probes.py --out-dir results/rq3_probe_dreamcoder --skip-exp2
then:
  python experiments/rq3_probe/plot_compare_models.py \
     --dirs results/rq3_probe results/rq3_probe_dreamcoder \
     --labels DiffuCoder-7B-Base DreamCoder-7B

Outputs figures/new/rq3/fig_rq3_compare_models.{png,pdf}.
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
    "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.linewidth": 0.4, "grid.alpha": 0.4,
})

COLORS = ["#e74c3c", "#2980b9", "#27ae60", "#8e44ad"]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dirs", nargs="+",
                    default=["results/rq3_probe", "results/rq3_probe_dreamcoder"])
    ap.add_argument("--labels", nargs="+",
                    default=["DiffuCoder-7B-Base", "DreamCoder-7B"])
    ap.add_argument("--fig-dir", default="figures/new/rq3")
    args = ap.parse_args()
    os.makedirs(args.fig_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    plotted = 0
    summary = []
    for i, (d, lab) in enumerate(zip(args.dirs, args.labels)):
        p = os.path.join(d, "exp1_curves.json")
        if not os.path.exists(p):
            print("[skip] %s missing (run Exp1 for %s first)" % (p, lab))
            continue
        c = json.load(open(p))
        ctx = np.array(c["curves"]["contextual_pkg"], dtype=float)
        L = c["n_layers"]
        xs = np.arange(L) / (L - 1)  # normalised depth so different layer counts align
        col = COLORS[i % len(COLORS)]
        ax.plot(xs, ctx, color=col, lw=1.6, label="%s (peak %.2f)" % (lab, np.nanmax(ctx)))
        ax.axhline(0.0, color="k", lw=0.5, alpha=0.4)
        summary.append((lab, float(np.nanmax(ctx)), int(np.nanargmax(ctx)), L,
                        c.get("late_third_contextual"), c.get("early_third_contextual")))
        plotted += 1

    ax.axvspan(2 / 3, 1.0, color="#f1c40f", alpha=0.10, lw=0)
    ax.set_xlabel("relative network depth (layer / final layer)")
    ax.set_ylabel("contextual AUC (intact - scrambled)")
    ax.set_title("RQ3: name-fit decodability across dLLMs")
    ax.legend(loc="upper left", frameon=False)
    if plotted == 0:
        print("[error] no curves to plot — extract+train at least one model first.")
        return
    fig.savefig(os.path.join(args.fig_dir, "fig_rq3_compare_models.png"),
                dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(args.fig_dir, "fig_rq3_compare_models.pdf"), bbox_inches="tight")
    plt.close(fig)
    print("\nmodel                  peak_ctxAUC  peak_layer/L   late3rd  early3rd")
    for lab, pk, pl, L, late, early in summary:
        print("  %-20s  %.3f        %d/%d         %s     %s"
              % (lab, pk, pl, L - 1,
                 ("%.3f" % late) if late is not None else "na",
                 ("%.3f" % early) if early is not None else "na"))
    print("-> figures/new/rq3/fig_rq3_compare_models.{png,pdf}")


if __name__ == "__main__":
    main()
