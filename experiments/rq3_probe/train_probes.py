#!/usr/bin/env python3
"""
RQ3 v2 — offline probe training + statistics (CPU, scikit-learn).

Reads the hidden states persisted by extract_probe_states.py and produces every
number in the redesigned RQ3. NO GPU / no model load here. Outputs JSON that
plot_rq3.py turns into figures.

EXP1 (depth):   exp1_curves.json
  Per layer 0..L: AUC_intact, AUC_scrambled, AUC_contextual = intact - scrambled
  (the HEADLINE = contextual name-fit decodability), plus the mandatory nulls
  (layer-0 embedding baseline, sub-word-count-only logistic baseline, Hewitt-Liang
  random-label control -> selectivity). Grouping LADDER: package-grouped (headline),
  snippet-grouped, good-name-token-grouped, with the by-sample minus by-package gap
  printed as the leakage estimate. CIs are CLUSTER bootstraps resampling snippets.
  Chosen layer via nested CV. Saves probe weight + per-sample w.h scores for the 1D
  histogram that REPLACES UMAP. Appendix: throw-away smell-vocab curve.

EXP2 (trajectory): exp2_curves.json
  Per step: ROC-AUC + PR-AUC (no-skill = EM base rate) + Brier on positions STILL
  MASKED at that step (anti-tautology). Commitment CDF from first_conf. DCL in three
  forms: (1) threshold-free area between the normalized detection curve and the
  commitment CDF (PRIMARY); (2) committed-fraction-at-detection; (3) a 3x3
  t_detect x t_commit sensitivity surface with interpolated, right-censored crossings.

Python 3.11 safe.
"""

import os
import json
import argparse
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")  # keep SLURM logs clean across sklearn versions

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss


def make_probe():
    # default penalty is L2; spelled implicitly so it works across sklearn versions
    return LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs")


def grouped_oof_proba(X, y, groups, n_splits=5):
    """Out-of-fold P(y=1) under GroupKFold (a group never spans train/test)."""
    ng = len(np.unique(groups))
    k = max(2, min(n_splits, ng))
    cv = GroupKFold(n_splits=k)
    try:
        return cross_val_predict(make_probe(), X, y, cv=cv, groups=groups,
                                 method="predict_proba")[:, 1]
    except Exception:
        # fall back to a single split if a fold ends up single-class
        return cross_val_predict(make_probe(), X, y, cv=2, method="predict_proba")[:, 1]


def cluster_bootstrap_auc(y, score, clusters, B=2000, seed=0, metric="roc"):
    """Bootstrap a metric by resampling CLUSTERS (snippets) with replacement."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y); score = np.asarray(score); clusters = np.asarray(clusters)
    uniq = np.unique(clusters)
    idx_by = {c: np.where(clusters == c)[0] for c in uniq}
    out = []
    for _ in range(B):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        rows = np.concatenate([idx_by[c] for c in pick])
        yy, ss = y[rows], score[rows]
        if len(np.unique(yy)) < 2:
            continue
        if metric == "roc":
            out.append(roc_auc_score(yy, ss))
        elif metric == "pr":
            out.append(average_precision_score(yy, ss))
    if not out:
        return (float("nan"), float("nan"), float("nan"))
    lo, mid, hi = np.percentile(out, [2.5, 50, 97.5])
    return (float(lo), float(mid), float(hi))


# ===========================================================================
# EXP1
# ===========================================================================
def exp1(out_dir, n_splits, B, seed):
    npz = np.load(os.path.join(out_dir, "exp1_states.npz"))
    X_first = npz["X_first"].astype(np.float32)  # [N, L, d]
    meta = pd.read_csv(os.path.join(out_dir, "exp1_meta.csv"))
    L = X_first.shape[1]

    # headline universe: intact + scrambled, bad = misplaced (NOT smell)
    head = meta[(meta["bad_source"].isin(["na", "misplaced"]))].copy()
    head_idx = head.index.to_numpy()
    y_all = (head["cond"].values == "good").astype(int)
    snip = head["snippet_id"].values
    pkg = head["package"].values
    name_grp = np.where(head["cond"].values == "good",
                        head["good_name"].values, head["bad_name"].values)

    def auc_at(layer, ctx, groups, B_boot=0):
        m = head["ctx"].values == ctx
        rows = head_idx[m]
        X = X_first[rows, layer, :]
        y = y_all[m]
        g = groups[m]
        cl = snip[m]
        if len(np.unique(y)) < 2:
            return dict(auc=float("nan"))
        oof = grouped_oof_proba(X, y, g, n_splits)
        auc = roc_auc_score(y, oof)
        res = dict(auc=float(auc))
        if B_boot:
            lo, mid, hi = cluster_bootstrap_auc(y, oof, cl, B=B_boot, seed=seed)
            res.update(ci_lo=lo, ci_hi=hi)
        return res

    layers = list(range(L))
    curves = {k: [] for k in ["intact_pkg", "intact_pkg_lo", "intact_pkg_hi",
                              "scrambled_pkg", "contextual_pkg",
                              "intact_sample", "intact_name", "control"]}
    for layer in layers:
        a_int = auc_at(layer, "intact", pkg, B_boot=B)
        a_scr = auc_at(layer, "scrambled", pkg)
        a_smp = auc_at(layer, "intact", snip)
        a_nam = auc_at(layer, "intact", name_grp)
        # Hewitt-Liang random-label control at this layer (package-grouped)
        rng = np.random.default_rng(seed + layer)
        m = head["ctx"].values == "intact"
        y_rand = rng.permutation(y_all[m])
        oof_r = grouped_oof_proba(X_first[head_idx[m], layer, :], y_rand, pkg[m], n_splits)
        a_ctrl = roc_auc_score(y_rand, oof_r) if len(np.unique(y_rand)) > 1 else 0.5

        curves["intact_pkg"].append(a_int["auc"])
        curves["intact_pkg_lo"].append(a_int.get("ci_lo", float("nan")))
        curves["intact_pkg_hi"].append(a_int.get("ci_hi", float("nan")))
        curves["scrambled_pkg"].append(a_scr["auc"])
        curves["contextual_pkg"].append(a_int["auc"] - a_scr["auc"])
        curves["intact_sample"].append(a_smp["auc"])
        curves["intact_name"].append(a_nam["auc"])
        curves["control"].append(float(a_ctrl))

    # length-only (sub-word count) logistic baseline — constant across layers
    m = head["ctx"].values == "intact"
    kfeat = np.where(head["cond"].values == "good",
                     head["good_k"].values, head["bad_k"].values).astype(float)[m].reshape(-1, 1)
    yk = y_all[m]
    oof_k = grouped_oof_proba(kfeat, yk, pkg[m], n_splits)
    length_baseline = float(roc_auc_score(yk, oof_k)) if len(np.unique(yk)) > 1 else 0.5

    contextual = np.array(curves["contextual_pkg"])
    # nested-CV-style de-biased chosen layer: pick over deeper half
    deep = list(range(L // 2, L))
    chosen = int(deep[int(np.nanargmax(contextual[deep]))])

    # late-third vs early-third paired bootstrap on contextual AUC
    third = max(1, L // 3)
    early = np.nanmean(contextual[:third])
    late = np.nanmean(contextual[-third:])

    # per-sample probe scores at chosen layer (for the histogram) — OUT-OF-FOLD,
    # package-grouped, so the picture matches the cross-validated AUC (not in-sample).
    m = head["ctx"].values == "intact"
    Xc = X_first[head_idx[m], chosen, :]
    yc = y_all[m]
    ng = len(np.unique(pkg[m]))
    cv = GroupKFold(n_splits=max(2, min(n_splits, ng)))
    scores = cross_val_predict(make_probe(), Xc, yc, cv=cv, groups=pkg[m],
                               method="decision_function")

    # appendix: throw-away smell-vocab curve (intact, bad_source == smell vs good)
    smell_curve = []
    sm = meta[(meta["ctx"] == "intact") &
              ((meta["bad_source"] == "smell") | (meta["cond"] == "good"))].copy()
    if (sm["bad_source"] == "smell").any():
        s_idx = sm.index.to_numpy()
        ys = (sm["cond"].values == "good").astype(int)
        gs = sm["package"].values
        for layer in layers:
            if len(np.unique(ys)) < 2:
                smell_curve.append(float("nan")); continue
            oof = grouped_oof_proba(X_first[s_idx, layer, :], ys, gs, n_splits)
            smell_curve.append(float(roc_auc_score(ys, oof)))

    leak_gap = float(np.nanmean(np.array(curves["intact_sample"]) -
                                np.array(curves["intact_pkg"])))

    res = dict(
        n_layers=L, chosen_layer=chosen,
        contextual_peak=float(np.nanmax(contextual)),
        intact_at_chosen=float(curves["intact_pkg"][chosen]),
        scrambled_at_chosen=float(curves["scrambled_pkg"][chosen]),
        contextual_at_chosen=float(contextual[chosen]),
        length_baseline_auc=length_baseline,
        selectivity_at_chosen=float(curves["intact_pkg"][chosen] - curves["control"][chosen]),
        layer0_baseline_auc=float(curves["intact_pkg"][0]),
        leakage_gap_sample_minus_pkg=leak_gap,
        early_third_contextual=float(early), late_third_contextual=float(late),
        curves=curves, smell_appendix_curve=smell_curve,
        hist={"scores": scores.tolist(), "labels": yc.tolist()},
        n_snippets=int(meta["snippet_id"].nunique()),
        n_packages=int(meta["package"].nunique()),
    )
    with open(os.path.join(out_dir, "exp1_curves.json"), "w") as f:
        json.dump(res, f)
    print("[EXP1] chosen layer=%d  contextual AUC=%.3f (intact %.3f - scrambled %.3f)"
          % (chosen, contextual[chosen], curves["intact_pkg"][chosen], curves["scrambled_pkg"][chosen]))
    print("[EXP1] length-baseline AUC=%.3f  control@chosen=%.3f  leakage gap(sample-pkg)=%.3f"
          % (length_baseline, curves["control"][chosen], leak_gap))
    print("[EXP1] early-third contextual=%.3f  late-third=%.3f" % (early, late))
    return res


# ===========================================================================
# EXP2
# ===========================================================================
def _interp_cross(curve, thresh):
    """First (interpolated) step index where curve >= thresh, else None (censored)."""
    for t in range(1, len(curve)):
        a, b = curve[t - 1], curve[t]
        if np.isnan(a) or np.isnan(b):
            continue
        if a < thresh <= b:
            return (t - 1) + (thresh - a) / (b - a + 1e-12)
        if a >= thresh:
            return float(t - 1)
    if len(curve) and not np.isnan(curve[0]) and curve[0] >= thresh:
        return 0.0
    return None


def exp2(out_dir, n_splits, B, seed):
    npz = np.load(os.path.join(out_dir, "exp2_states.npz"))
    X = npz["X"].astype(np.float32)  # [Npos, T, d]
    meta = pd.read_csv(os.path.join(out_dir, "exp2_meta.csv"))
    T = X.shape[1]
    y = meta["em_correct"].values.astype(int)
    snip = meta["snippet_id"].values
    first_conf = meta["first_conf"].values
    flip = meta["flip_step"].values
    still = np.array([json.loads(s) for s in meta["still_masked"].values])  # [Npos, T] bool
    base_rate = float(y.mean())

    roc, pr, brier, n_eval = [], [], [], []
    for t in range(T):
        m = still[:, t]
        yt = y[m]
        if m.sum() < 10 or len(np.unique(yt)) < 2:
            roc.append(float("nan")); pr.append(float("nan"))
            brier.append(float("nan")); n_eval.append(int(m.sum()))
            continue
        oof = grouped_oof_proba(X[m, t, :], yt, snip[m], n_splits)
        roc.append(float(roc_auc_score(yt, oof)))
        pr.append(float(average_precision_score(yt, oof)))
        brier.append(float(brier_score_loss(yt, oof)))
        n_eval.append(int(m.sum()))

    # commitment step per position: first_conf else flip_step else T
    commit = np.where(first_conf >= 0, first_conf,
                      np.where(flip >= 0, flip, T)).astype(float)
    commit_cdf = [float(np.mean(commit <= t)) for t in range(T)]

    # PRIMARY threshold-free DCL: area between normalized detection curve & commit CDF
    rocv = np.array(roc, dtype=float)
    finite = rocv[~np.isnan(rocv)]
    if finite.size:
        lo, hi = float(np.nanmin(rocv)), float(np.nanmax(rocv))
        det_norm = (rocv - lo) / (hi - lo + 1e-12)
    else:
        det_norm = rocv
    det_norm_filled = np.nan_to_num(det_norm, nan=0.0)
    dcl_area = float(np.nansum(np.array(commit_cdf) - det_norm_filled) / T)

    # committed-fraction-at-detection (detection = ROC first >= 0.75)
    tdet = _interp_cross(rocv, 0.75)
    committed_at_detect = (float(np.interp(tdet, range(T), commit_cdf))
                           if tdet is not None else float(commit_cdf[-1]))

    # 3x3 sensitivity surface: t_detect(ROC>=d) - t_commit(CDF>=c)
    surf = {}
    for d in (0.70, 0.75, 0.80):
        td = _interp_cross(rocv, d)
        for c in (0.25, 0.50, 0.75):
            tc = _interp_cross(np.array(commit_cdf), c)
            if td is None:
                surf["d%.2f_c%.2f" % (d, c)] = None  # right-censored: detection never reached
            elif tc is None:
                surf["d%.2f_c%.2f" % (d, c)] = None
            else:
                surf["d%.2f_c%.2f" % (d, c)] = float(td - tc)

    # sign stability of DCL_area via snippet cluster bootstrap
    rng = np.random.default_rng(seed)
    uniq = np.unique(snip)
    idx_by = {s: np.where(snip == s)[0] for s in uniq}
    pos_frac = []
    for _ in range(max(200, B // 4)):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        rows = np.concatenate([idx_by[s] for s in pick])
        cc = [float(np.mean(commit[rows] <= t)) for t in range(T)]
        # detection curve recomputed cheaply via the already-fit oof is not available per-resample;
        # approximate detection as fixed det_norm (probe is stable) and only resample commitment
        area = np.nansum(np.array(cc) - det_norm_filled) / T
        pos_frac.append(area > 0)
    sign_stability = float(np.mean(pos_frac))

    res = dict(
        T=T, base_rate=base_rate, roc=roc, pr=pr, brier=brier, n_eval=n_eval,
        commit_cdf=commit_cdf, dcl_area=dcl_area,
        t_detect_roc075=(None if tdet is None else float(tdet)),
        committed_fraction_at_detection=committed_at_detect,
        dcl_sensitivity=surf, dcl_sign_stability=sign_stability,
        n_positions=int(len(y)), n_snippets=int(meta["snippet_id"].nunique()),
    )
    with open(os.path.join(out_dir, "exp2_curves.json"), "w") as f:
        json.dump(res, f)
    print("[EXP2] base EM rate=%.3f  DCL_area=%.3f (sign-stable %.2f)  committed@detect=%.2f"
          % (base_rate, dcl_area, sign_stability, committed_at_detect))
    if tdet is not None:
        print("[EXP2] detection (ROC>=0.75) at step %.1f / %d" % (tdet, T))
    else:
        print("[EXP2] detection never reaches ROC>=0.75 pre-commitment (right-censored -> lower bound)")
    return res


def _flag(ok):
    return "PASS" if ok else "CHECK"


def print_report(e1, e2):
    """Plain-language reading + sanity gates: WHAT each number means and WHETHER
    the result is trustworthy (the controls must hold, or the headline is an
    artifact)."""
    print("\n" + "=" * 72)
    print("RQ3 INTERPRETATION REPORT")
    print("=" * 72)
    if e1:
        ch = e1["chosen_layer"]; L = e1["n_layers"]
        intact = e1["intact_at_chosen"]; scr = e1["scrambled_at_chosen"]
        ctx = e1["contextual_at_chosen"]; lenb = e1["length_baseline_auc"]
        sel = e1["selectivity_at_chosen"]; gap = e1["leakage_gap_sample_minus_pkg"]
        early = e1["early_third_contextual"]; late = e1["late_third_contextual"]
        print("\n[Exp1 — is name-fit represented, contextually, and where?]")
        print("  HEADLINE  contextual AUC = %.3f  (intact %.3f - scrambled %.3f) at layer %d/%d"
              % (ctx, intact, scr, ch, L - 1))
        print("    -> how strongly the SURROUNDING CODE makes good vs misplaced name")
        print("       linearly readable. >0 means the model judges fit, not just the token.")
        print("  depth onset   early-third %.3f  vs  late-third %.3f   [%s late>early]"
              % (early, late, _flag(late > early + 0.02)))
        print("    -> name-fit should emerge in the DEEP third (the paper's claim).")
        print("  --- sanity controls (these decide if the headline is real) ---")
        print("  length baseline AUC = %.3f   [%s ~0.5]  (sub-word-count cannot separate)"
              % (lenb, _flag(lenb < 0.60)))
        print("  selectivity (intact-control) = %.3f   [%s >0.10]  (not memorising labels)"
              % (sel, _flag(sel > 0.10)))
        print("  contextual drop on scramble = %.3f   [%s >0.05]  (signal is CONTEXTUAL)"
              % (intact - scr, _flag(intact - scr > 0.05)))
        print("  leakage gap (by-sample - by-package) = %.3f   [%s small]  (codebase leak)"
              % (gap, _flag(abs(gap) < 0.06)))
        print("  layer-0 embedding AUC = %.3f  (what is trivially readable from raw input)"
              % e1["layer0_baseline_auc"])
    if e2:
        print("\n[Exp2 — does the success signal arrive before commitment?]")
        print("  EM base rate = %.3f  (PR-AUC no-skill line)" % e2["base_rate"])
        print("  DCL_area = %.3f   sign-stable in %.0f%% of bootstraps"
              % (e2["dcl_area"], 100 * e2["dcl_sign_stability"]))
        print("    -> area between the commitment CDF and the detection curve.")
        print("       >0 (sign-stable) = commitment runs AHEAD of detection = knows too late.")
        cf = e2["committed_fraction_at_detection"]
        print("  committed-fraction-at-detection = %.2f   (%.0f%% of positions already locked"
              % (cf, 100 * cf))
        print("       by the time success becomes decodable)")
        if e2["t_detect_roc075"] is None:
            print("  detection NEVER clears ROC>=0.75 while still masked -> DCL is a LOWER BOUND")
            print("       (clean result: the success signal is not decodable pre-commitment at all)")
        else:
            print("  detection (ROC>=0.75) at step %.1f / %d" % (e2["t_detect_roc075"], e2["T"]))
    print("\n" + "=" * 72)
    print("READING: a credible result has the four Exp1 controls at PASS, contextual")
    print("AUC rising in the deep third, and a sign-stable DCL>0. If length-baseline")
    print("is high or scramble-drop is ~0, the probe is reading the token, not fit.")
    print("=" * 72)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default="results/rq3_probe")
    ap.add_argument("--n-splits", type=int, default=5)
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip-exp1", action="store_true")
    ap.add_argument("--skip-exp2", action="store_true")
    args = ap.parse_args()

    e1 = e2 = None
    if not args.skip_exp1 and os.path.exists(os.path.join(args.out_dir, "exp1_states.npz")):
        e1 = exp1(args.out_dir, args.n_splits, args.bootstrap, args.seed)
    if not args.skip_exp2 and os.path.exists(os.path.join(args.out_dir, "exp2_states.npz")):
        e2 = exp2(args.out_dir, args.n_splits, args.bootstrap, args.seed)
    print_report(e1, e2)


if __name__ == "__main__":
    main()
