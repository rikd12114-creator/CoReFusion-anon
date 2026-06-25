"""
Metric-reasonableness study + metric-fusion model for the refineID
identifier-naming task.

Backbone for the question: "we used to report only EM and LJ; we now have
several new identifier metrics (M1 IdBench lev/nw, M2 subtoken jaccard/fuzzy,
M3 quality char/word). (1) Is each new metric a reasonable evaluation axis vs
EM and LJ? (2) Can we fuse them into ONE score that represents identifier
goodness better than any single metric?"

Ground truth dataset: results/identifier_metrics/fusion_consensus.csv
(one row per model x sample; metrics recomputed on the SAME agreed_pred/gt the
judge saw; LJ = majority / mean vote across up to 5 judge models).

Outputs:
  figures/new/metric_eval/*.png
  results/identifier_metrics/metric_eval_summary.json
  results/identifier_metrics/fused_scores.csv  (per-sample fused score)
  results/identifier_metrics/leaderboard_fused.csv
"""
import os, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve
from sklearn.calibration import calibration_curve

BLUE, ORANGE = "#0076C2", "#FF8000"
GREY = "#888888"
FIGDIR = "figures/new/metric_eval"
os.makedirs(FIGDIR, exist_ok=True)
os.makedirs("results/identifier_metrics", exist_ok=True)

SIM = ["lev_sim", "nw_sim", "subtok_jaccard", "subtok_fuzzy"]   # M1+M2 (pred vs gt)
QUAL = ["qual_char", "qual_word"]                               # M3 (intrinsic)
METRICS = SIM + QUAL
PRETTY = {"lev_sim": "lev (M1)", "nw_sim": "NW (M1)", "subtok_jaccard": "jacc (M2)",
          "subtok_fuzzy": "fuzzy (M2)", "qual_char": "qualC (M3)", "qual_word": "qualW (M3)",
          "em": "EM"}

# --------------------------------------------------------------------------
# Load: consistent samples with a real LJ label
# --------------------------------------------------------------------------
df = pd.read_csv("results/identifier_metrics/fusion_consensus.csv")
df = df[df.consistent == 1].copy()
df = df[df.lj_n_judges > 0].copy()
df = df[df.lj_mean >= 0].copy()          # drop all-judge-error rows (-1)
df["lj"] = (df.lj_majority > 0).astype(int)
print(f"consistent labelled samples: {len(df)} | models: {df.model.nunique()} | "
      f"EM={df.em.mean():.3f} LJ={df.lj.mean():.3f}")
summary = {"n": int(len(df)), "n_models": int(df.model.nunique()),
           "em_rate": float(df.em.mean()), "lj_rate": float(df.lj.mean())}

# ==========================================================================
# PART A  --  is each new metric reasonable vs EM and LJ?
# ==========================================================================
rows = []
e0 = df[df.em == 0]
for m in METRICS:
    sp_em = stats.spearmanr(df[m], df.em).correlation
    sp_lj = stats.spearmanr(df[m], df.lj_mean).correlation
    auc_lj = roc_auc_score(df.lj, df[m])
    auc_lj_e0 = roc_auc_score(e0.lj, e0[m]) if e0.lj.nunique() > 1 else float("nan")
    # separation: mean among LJ=1 vs LJ=0
    mu1, mu0 = df[df.lj == 1][m].mean(), df[df.lj == 0][m].mean()
    rows.append({"metric": m, "spearman_EM": sp_em, "spearman_LJ": sp_lj,
                 "AUC_LJ": auc_lj, "AUC_LJ_EM0": auc_lj_e0,
                 "mean_LJ1": mu1, "mean_LJ0": mu0, "sep": mu1 - mu0})
rA = pd.DataFrame(rows)
# EM as reference predictor of LJ
rA_em = {"metric": "em", "spearman_EM": 1.0, "spearman_LJ": stats.spearmanr(df.em, df.lj_mean).correlation,
         "AUC_LJ": roc_auc_score(df.lj, df.em), "AUC_LJ_EM0": float("nan"),
         "mean_LJ1": df[df.lj == 1].em.mean(), "mean_LJ0": df[df.lj == 0].em.mean(),
         "sep": df[df.lj == 1].em.mean() - df[df.lj == 0].em.mean()}
print("\n== PART A: metric reasonableness (vs EM, vs LJ) ==")
print(rA.round(3).to_string(index=False))
print("  [ref] EM->LJ AUC = %.3f" % rA_em["AUC_LJ"])
summary["partA"] = {r["metric"]: {k: (None if pd.isna(v) else float(v)) for k, v in r.items() if k != "metric"}
                    for r in rows}
summary["partA"]["em_ref"] = {k: (None if pd.isna(v) else float(v)) for k, v in rA_em.items() if k != "metric"}

# ---- Fig A1: correlation heatmap (metrics + EM + LJ) ----
cols = METRICS + ["em", "lj_mean"]
corr = df[cols].corr(method="spearman")
fig, ax = plt.subplots(figsize=(7, 6))
im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xticks(range(len(cols))); ax.set_yticks(range(len(cols)))
lab = [PRETTY.get(c, c.replace("lj_mean", "LJ")) for c in cols]
ax.set_xticklabels(lab, rotation=45, ha="right"); ax.set_yticklabels(lab)
for i in range(len(cols)):
    for j in range(len(cols)):
        ax.text(j, i, f"{corr.iloc[i,j]:.2f}", ha="center", va="center",
                color="white" if abs(corr.iloc[i, j]) > 0.5 else "black", fontsize=8)
ax.set_title("Spearman correlation: new metrics vs EM vs LJ\n(consistent samples, n=%d)" % len(df))
fig.colorbar(im, fraction=0.046, pad=0.04)
fig.tight_layout(); fig.savefig(f"{FIGDIR}/A1_corr_heatmap.png", dpi=140); plt.close(fig)

# ---- Fig A2: AUC bars (each metric predicting LJ acceptance) ----
order = rA.sort_values("AUC_LJ", ascending=True)
fig, ax = plt.subplots(figsize=(8, 5))
y = np.arange(len(order))
ax.barh(y - 0.2, order.AUC_LJ, 0.4, color=BLUE, label="all consistent")
ax.barh(y + 0.2, order.AUC_LJ_EM0, 0.4, color=ORANGE, label="within EM=0 (synonym rescue)")
ax.axvline(0.5, color=GREY, ls="--", lw=1)
ax.axvline(rA_em["AUC_LJ"], color="k", ls=":", lw=1.5, label="EM alone (AUC=%.2f)" % rA_em["AUC_LJ"])
ax.set_yticks(y); ax.set_yticklabels([PRETTY[m] for m in order.metric])
ax.set_xlabel("AUC: single metric -> LJ acceptance"); ax.set_xlim(0.4, 1.0)
ax.set_title("Does each new metric predict semantic (LJ) acceptance?")
ax.legend(loc="lower right", fontsize=8)
fig.tight_layout(); fig.savefig(f"{FIGDIR}/A2_auc_bars.png", dpi=140); plt.close(fig)

# ---- Fig A3: per-metric distribution split by LJ ----
fig, axes = plt.subplots(2, 3, figsize=(13, 7))
for ax, m in zip(axes.ravel(), METRICS):
    d0 = df[df.lj == 0][m]; d1 = df[df.lj == 1][m]
    ax.hist(d0, bins=25, density=True, alpha=0.55, color=BLUE, label="LJ reject")
    ax.hist(d1, bins=25, density=True, alpha=0.55, color=ORANGE, label="LJ accept")
    ax.set_title("%s   (AUC=%.2f)" % (PRETTY[m], rA[rA.metric == m].AUC_LJ.iloc[0]))
    ax.set_yticks([])
axes[0, 0].legend(fontsize=8)
fig.suptitle("New-metric distributions split by LJ acceptance (consistent samples)")
fig.tight_layout(); fig.savefig(f"{FIGDIR}/A3_separation.png", dpi=140); plt.close(fig)

# ---- Fig A4: the synonym-rescue panel (EM=0 only) ----
fig, ax = plt.subplots(figsize=(8, 5))
sim_e0 = e0[SIM].mean(axis=1)
ax.hist(sim_e0[e0.lj == 0], bins=25, density=True, alpha=0.55, color=BLUE, label="EM=0 & LJ reject (n=%d)" % (e0.lj == 0).sum())
ax.hist(sim_e0[e0.lj == 1], bins=25, density=True, alpha=0.55, color=ORANGE, label="EM=0 & LJ accept (n=%d)" % (e0.lj == 1).sum())
ax.set_xlabel("mean similarity metric (lev,nw,jacc,fuzzy)  among EM=0 samples")
ax.set_ylabel("density"); ax.legend()
ax.set_title("EM=0 names the judge ACCEPTS score high on similarity metrics\n(EM can't see them; the new metrics can)")
fig.tight_layout(); fig.savefig(f"{FIGDIR}/A4_synonym_rescue.png", dpi=140); plt.close(fig)

# ==========================================================================
# PART B  --  fuse metrics into ONE identifier-goodness score
# ==========================================================================
X = df[METRICS].values
y = df.lj.values
groups = df.model.values
gkf = GroupKFold(n_splits=min(8, df.model.nunique()))

def cv_auc(estimator, X, y, groups):
    p = cross_val_predict(estimator, X, y, groups=groups, cv=gkf, method="predict_proba")[:, 1]
    return roc_auc_score(y, p), average_precision_score(y, p), p

logreg = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0))
gbdt = GradientBoostingClassifier(n_estimators=200, max_depth=3, learning_rate=0.05, subsample=0.8, random_state=0)

results = {}
auc_lr, ap_lr, p_lr = cv_auc(logreg, X, y, groups)
auc_gb, ap_gb, p_gb = cv_auc(gbdt, X, y, groups)
results["LogReg(6 metrics)"] = (auc_lr, ap_lr)
results["GBDT(6 metrics)"] = (auc_gb, ap_gb)

# baselines
for name, feats in [("EM only", ["em"]), ("best single (fuzzy)", ["subtok_fuzzy"]),
                    ("mean similarity", None)]:
    if feats is None:
        s = df[SIM].mean(axis=1).values
    else:
        s = df[feats].values.ravel()
    results[name] = (roc_auc_score(y, s), average_precision_score(y, s))

# SIM-only vs +QUAL ablation (does M3 add anything to fusion?)
auc_sim, ap_sim, _ = cv_auc(make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
                            df[SIM].values, y, groups)
results["LogReg(SIM only,4)"] = (auc_sim, ap_sim)

print("\n== PART B: fusion CV (GroupKFold by model) ==")
for k, (a, ap) in sorted(results.items(), key=lambda kv: -kv[1][0]):
    print(f"  {k:<24} AUC={a:.3f}  PR-AUC={ap:.3f}")
summary["partB_cv"] = {k: {"auc": float(a), "pr_auc": float(ap)} for k, (a, ap) in results.items()}

# ---- final interpretable model on all data: closed-form weighted score ----
final = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(X, y)
scaler = final.named_steps["standardscaler"]; clf = final.named_steps["logisticregression"]
coef = clf.coef_[0]
print("\n  Closed-form 'CoReFusion Identifier Score' = sigmoid(b + sum w_i * z_i):")
print("    intercept(on z) = %.3f" % clf.intercept_[0])
coef_tbl = []
for m, w, mu, sd in zip(METRICS, coef, scaler.mean_, scaler.scale_):
    print(f"    {PRETTY[m]:<12} w={w:+.3f}  (mean={mu:.3f} sd={sd:.3f})")
    coef_tbl.append({"metric": m, "weight_z": float(w), "mean": float(mu), "sd": float(sd)})
summary["partB_logreg_coef"] = {"intercept": float(clf.intercept_[0]), "coef": coef_tbl}

# GBDT importances (fit on all)
gbdt_full = gbdt.fit(X, y)
imp = dict(zip(METRICS, gbdt_full.feature_importances_))
summary["partB_gbdt_importance"] = {m: float(v) for m, v in imp.items()}
print("\n  GBDT feature importance:", {m: round(v, 3) for m, v in imp.items()})

# fused score for every sample (out-of-fold for honesty) + full-fit for leaderboard
df["fused_cv"] = p_gb            # OOF GBDT prob (best CV model)
df["fused_logreg"] = final.predict_proba(X)[:, 1]
df[["model", "family", "id", "ground_truth", "agreed_pred", "em", "lj", "lj_mean",
    "fused_cv", "fused_logreg"] + METRICS].to_csv(
    "results/identifier_metrics/fused_scores.csv", index=False)

# ---- Fig B1: CV AUC comparison ----
fig, ax = plt.subplots(figsize=(8, 5))
items = sorted(results.items(), key=lambda kv: kv[1][0])
names = [k for k, _ in items]; aucs = [v[0] for _, v in items]
colors = [ORANGE if ("LogReg(6" in n or "GBDT" in n) else (GREY if "only" in n or "single" in n or "EM" in n else BLUE) for n in names]
ax.barh(range(len(names)), aucs, color=colors)
for i, a in enumerate(aucs):
    ax.text(a + 0.003, i, f"{a:.3f}", va="center", fontsize=8)
ax.set_yticks(range(len(names))); ax.set_yticklabels(names)
ax.set_xlim(0.5, 1.0); ax.set_xlabel("AUC -> LJ acceptance (GroupKFold by model)")
ax.set_title("Fused score vs single metrics vs EM")
fig.tight_layout(); fig.savefig(f"{FIGDIR}/B1_fusion_auc.png", dpi=140); plt.close(fig)

# ---- Fig B2: ROC curves ----
fig, ax = plt.subplots(figsize=(6.5, 6))
for s, n, c in [(p_gb, "Fused GBDT", ORANGE), (p_lr, "Fused LogReg", BLUE),
                (df.subtok_fuzzy, "fuzzy (best single)", GREY), (df.em, "EM", "k")]:
    fpr, tpr, _ = roc_curve(y, s)
    ax.plot(fpr, tpr, color=c, lw=2, label="%s (AUC=%.3f)" % (n, roc_auc_score(y, s)))
ax.plot([0, 1], [0, 1], "k--", lw=1)
ax.set_xlabel("FPR"); ax.set_ylabel("TPR"); ax.legend(loc="lower right", fontsize=8)
ax.set_title("ROC: predicting LJ semantic acceptance")
fig.tight_layout(); fig.savefig(f"{FIGDIR}/B2_roc.png", dpi=140); plt.close(fig)

# ---- Fig B3: coefficients + GBDT importance ----
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
o = np.argsort(coef)
ax1.barh(range(len(METRICS)), coef[o], color=[ORANGE if coef[o][i] > 0 else BLUE for i in range(len(METRICS))])
ax1.set_yticks(range(len(METRICS))); ax1.set_yticklabels([PRETTY[METRICS[i]] for i in o])
ax1.axvline(0, color="k", lw=0.8); ax1.set_title("LogReg weights (standardised)")
io = sorted(imp.items(), key=lambda kv: kv[1])
ax2.barh(range(len(io)), [v for _, v in io], color=BLUE)
ax2.set_yticks(range(len(io))); ax2.set_yticklabels([PRETTY[k] for k, _ in io])
ax2.set_title("GBDT feature importance")
fig.tight_layout(); fig.savefig(f"{FIGDIR}/B3_importance.png", dpi=140); plt.close(fig)

# ---- Fig B4: calibration of fused score ----
fig, ax = plt.subplots(figsize=(6, 6))
frac, mean_pred = calibration_curve(y, p_gb, n_bins=10, strategy="quantile")
ax.plot(mean_pred, frac, "o-", color=ORANGE, label="Fused GBDT (OOF)")
ax.plot([0, 1], [0, 1], "k--", label="perfect")
ax.set_xlabel("predicted P(acceptable)"); ax.set_ylabel("observed LJ-accept rate")
ax.set_title("Calibration of the fused identifier score"); ax.legend()
fig.tight_layout(); fig.savefig(f"{FIGDIR}/B4_calibration.png", dpi=140); plt.close(fig)

# ---- Fig B5: model-level leaderboard, fused vs EM vs LJ ----
lb = df.groupby(["model", "family"]).agg(
    EM=("em", "mean"), LJ=("lj", "mean"),
    fused=("fused_cv", "mean"), n=("em", "size")).reset_index()
lb.to_csv("results/identifier_metrics/leaderboard_fused.csv", index=False)
rho_fused = stats.spearmanr(lb.fused, lb.LJ).correlation
rho_em = stats.spearmanr(lb.EM, lb.LJ).correlation
summary["leaderboard_rank_corr_with_LJ"] = {"fused": float(rho_fused), "em": float(rho_em)}
fig, ax = plt.subplots(figsize=(7.5, 6))
for fam, c in [("dLLM", ORANGE), ("FIM", BLUE), ("Seq2Seq", GREY)]:
    s = lb[lb.family == fam]
    ax.scatter(s.EM, s.LJ, s=70, color=c, label=fam, edgecolor="k", zorder=3)
    ax.scatter(s.fused, s.LJ, s=70, color=c, marker="^", edgecolor="k", alpha=0.6, zorder=2)
ax.plot([0, lb.LJ.max()], [0, lb.LJ.max()], "k--", lw=1, alpha=0.5)
ax.set_xlabel("EM (circles)  /  fused score (triangles), model mean")
ax.set_ylabel("LJ acceptance (model mean)")
ax.set_title("Per-model: fused score tracks LJ tighter than EM\n(rank-corr w/ LJ: fused=%.2f, EM=%.2f)" % (rho_fused, rho_em))
ax.legend()
fig.tight_layout(); fig.savefig(f"{FIGDIR}/B5_leaderboard.png", dpi=140); plt.close(fig)

print("\n  leaderboard rank-corr with LJ:  fused=%.3f  EM=%.3f" % (rho_fused, rho_em))

with open("results/identifier_metrics/metric_eval_summary.json", "w") as f:
    json.dump(summary, f, indent=2)
print("\nwrote summary -> results/identifier_metrics/metric_eval_summary.json")
print("figures -> %s/" % FIGDIR)
