#!/usr/bin/env python3
"""
RQ3 Exp2 — robustness with a CIS-based soft success label (CPU, no GPU re-run).

The strict-EM "will-succeed" label is noisy because the fixed k=2 canvas duplicates
short identifiers (request -> requestRequest), marking near-correct fills as wrong.
This script re-runs the per-step detection probe using the project's CoReFusion
Identifier Score (CIS = mean of lev_sim, nw_sim, subtok_jaccard, subtok_fuzzy;
analysis/metric_eval_final.py) as a graded success criterion, computed from the
pred/target already stored in exp2_meta.csv. It reports detection under strict EM
and under CIS>=thresholds, overlaid with the SAME commitment CDF, so we can see
whether a less-noisy success label recovers an earlier/stronger detection signal.

Conclusion to expect (verified): softer labels lift the detection peak only
modestly and it stays LATE (>= step ~22), so "the success signal does not precede
commitment" holds regardless of label.

Outputs results/rq3_probe/exp2_cis_curves.json (read by plot_rq3.py-style plots).
Python 3.11 safe.
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # train_probes.py
sys.path.insert(0, os.path.join(os.getcwd(), "analysis"))         # CIS metrics

from train_probes import grouped_oof_proba                       # noqa: E402
from sklearn.metrics import roc_auc_score, average_precision_score
from identifier_similarity_metrics import (                      # noqa: E402
    lev_sim, nw_sim, subtok_jaccard, subtok_fuzzy)


def cis(pred, gt):
    p, g = str(pred), str(gt)
    return (lev_sim(p, g) + nw_sim(p, g) + subtok_jaccard(p, g) + subtok_fuzzy(p, g)) / 4.0


def per_step_curve(X, still, y, snip, T, n_splits=5):
    roc, pr, ne = [], [], []
    for t in range(T):
        m = still[:, t]
        yt = y[m]
        if m.sum() < 30 or len(np.unique(yt)) < 2:
            roc.append(np.nan); pr.append(np.nan); ne.append(int(m.sum())); continue
        oof = grouped_oof_proba(X[m, t, :], yt, snip[m], n_splits)
        roc.append(float(roc_auc_score(yt, oof)))
        pr.append(float(average_precision_score(yt, oof)))
        ne.append(int(m.sum()))
    return np.array(roc), np.array(pr), ne


def summarize(name, roc, cdf, base):
    pk = int(np.nanargmax(roc))
    cmed = next((t for t in range(len(cdf)) if cdf[t] >= 0.5), len(cdf))
    print("  %-18s base=%.3f  peakROC=%.3f@step%d  meanROC=%.3f  committed@peak=%.0f%%  precedes_commit=%s"
          % (name, base, float(np.nanmax(roc)), pk, float(np.nanmean(roc)),
             100 * cdf[pk], "YES" if pk < cmed else "no"))
    return dict(peak_roc=float(np.nanmax(roc)), peak_step=pk, mean_roc=float(np.nanmean(roc)),
                committed_at_peak=float(cdf[pk]), precedes_commit=bool(pk < cmed))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default="results/rq3_probe")
    ap.add_argument("--thresholds", default="0.5,0.7",
                    help="comma-separated CIS success thresholds")
    ap.add_argument("--n-splits", type=int, default=5)
    args = ap.parse_args()

    X = np.load(os.path.join(args.out_dir, "exp2_states.npz"))["X"].astype(np.float32)
    m = pd.read_csv(os.path.join(args.out_dir, "exp2_meta.csv"))
    T = X.shape[1]
    snip = m["snippet_id"].values
    still = np.array([json.loads(s) for s in m["still_masked"].values])
    commit = np.where(m["first_conf"].values >= 0, m["first_conf"].values,
                      np.where(m["flip_step"].values >= 0, m["flip_step"].values, T)).astype(float)
    cdf = np.array([float(np.mean(commit <= t)) for t in range(T)])

    cis_vals = np.array([cis(p, g) for p, g in zip(m["pred"], m["target"])])
    print("CIS distribution: mean=%.3f  >=0.5: %.1f%%  >=0.7: %.1f%%  ==1.0(EM-ish): %.1f%%"
          % (cis_vals.mean(), 100 * (cis_vals >= 0.5).mean(),
             100 * (cis_vals >= 0.7).mean(), 100 * (cis_vals >= 0.999).mean()))
    print("\nper-step detection probe (still-masked positions, snippet-grouped CV):")

    out = {"T": T, "commit_cdf": cdf.tolist(), "labels": {}}
    # strict EM reference
    y_em = m["em_correct"].values.astype(int)
    roc_em, pr_em, _ = per_step_curve(X, still, y_em, snip, T, args.n_splits)
    out["labels"]["em"] = dict(roc=roc_em.tolist(), pr=pr_em.tolist(),
                               base=float(y_em.mean()),
                               **summarize("strict EM", roc_em, cdf, float(y_em.mean())))
    # CIS thresholds
    best = None
    for thr in [float(x) for x in args.thresholds.split(",")]:
        y = (cis_vals >= thr).astype(int)
        if len(np.unique(y)) < 2:
            print("  CIS>=%.2f skipped (single class)" % thr); continue
        roc, pr, _ = per_step_curve(X, still, y, snip, T, args.n_splits)
        s = summarize("CIS>=%.2f" % thr, roc, cdf, float(y.mean()))
        out["labels"]["cis_%.2f" % thr] = dict(roc=roc.tolist(), pr=pr.tolist(),
                                               base=float(y.mean()), **s)
        if best is None or s["peak_roc"] > best[1]:
            best = ("cis_%.2f" % thr, s["peak_roc"])
    out["primary_cis_label"] = best[0] if best else None
    with open(os.path.join(args.out_dir, "exp2_cis_curves.json"), "w") as f:
        json.dump(out, f)

    print("\nREADING: if every CIS peak is still at a LATE step (>= ~22) and committed@peak")
    print("is high, the 'success signal does not precede commitment' conclusion is robust to")
    print("label noise — the canvas-doubling EM artifact is NOT what drives the null.")
    print("-> wrote %s/exp2_cis_curves.json" % args.out_dir)


if __name__ == "__main__":
    main()
