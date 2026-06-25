"""
FINAL metric-reasonableness + fusion analysis (MASK-aware, adversarially verified).

Corrects the backbone for the degeneracy the verification workflow surfaced:
~41% of "consistent" predictions are the literal placeholder token 'MASK' (the
model echoing the mask instead of filling it) -- all LJ-rejected, all from AR
models, qual_char=1.0. The consistency gate alone does not catch them, and they
inflate the raw metric AUCs by creating an easy MASK-vs-realname split.

We therefore report every number THREE ways:
  (1) full regime          -- all consistent labelled samples (what a leaderboard sees)
  (2) valid fills (ex-MASK) -- honest metric discrimination on real names
  (3) hardest: EM=0 & valid -- can the metric tell a good synonym from a bad guess

Recommended integrated score (CIS) = unweighted mean of the 4 similarity metrics.
"""
import os, json, re
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy import stats
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.metrics import roc_auc_score, roc_curve, brier_score_loss
from sklearn.isotonic import IsotonicRegression

BLUE, ORANGE, GREY, GREEN = "#0076C2", "#FF8000", "#9aa0a6", "#2ca02c"
FIG = "figures/new/metric_eval"; os.makedirs(FIG, exist_ok=True)
SIM = ["lev_sim", "nw_sim", "subtok_jaccard", "subtok_fuzzy"]
QUAL = ["qual_char", "qual_word"]
METRICS = SIM + QUAL
PRETTY = {"lev_sim": "lev (M1)", "nw_sim": "NW (M1)", "subtok_jaccard": "jacc (M2)",
          "subtok_fuzzy": "fuzzy (M2)", "qual_char": "qualC (M3)", "qual_word": "qualW (M3)",
          "em": "EM", "CIS": "CIS (mean of 4 sim)"}
CEILING = 0.924  # LJ judge noise ceiling (held-out judge vs independent 3-judge consensus)

# placeholder / special-token echoes = non-fills the consistency gate misses
DEGEN = {"mask", "masked", "maskedvar", "__masked_var__", "eot", "file_separator",
         "endoftext", "fim_middle", "fim_prefix", "fim_suffix", "pad", "unk", "bos", "eos"}

def coref_identifier_score(lev_sim, nw_sim, subtok_jaccard, subtok_fuzzy):
    """CoReFusion Identifier Score (CIS): unweighted mean of the four
    ground-truth similarity metrics (M1 char-level + M2 subtoken-level), in [0,1].
    Untrained, deterministic, needs no LJ label."""
    return (lev_sim + nw_sim + subtok_jaccard + subtok_fuzzy) / 4.0

# --------------------------------------------------------------------------
df = pd.read_csv("results/identifier_metrics/fusion_consensus.csv")
df = df[(df.consistent == 1) & (df.lj_n_judges > 0) & (df.lj_mean >= 0)].copy()
df["lj"] = (df.lj_majority > 0).astype(int)
df["degenerate"] = df.agreed_pred.astype(str).str.strip().str.lower().isin(DEGEN)
df["CIS"] = coref_identifier_score(df.lev_sim, df.nw_sim, df.subtok_jaccard, df.subtok_fuzzy)

valid = df[~df.degenerate]
print(f"regime n={len(df)} | degenerate(non-fill)={int(df.degenerate.sum())} "
      f"({df.degenerate.mean():.1%}) | valid fills={len(valid)}")
print(f"  EM={df.em.mean():.3f} LJ={df.lj.mean():.3f} | among valid: EM={valid.em.mean():.3f} LJ={valid.lj.mean():.3f}")

def auc(sub, m):
    return roc_auc_score(sub.lj, sub[m]) if sub.lj.nunique() > 1 else float("nan")

# ============ PART A: reasonableness, 3 views ============
e0v = valid[valid.em == 0]
print("\n== PART A: AUC(metric -> LJ) ==")
print(f"{'metric':<16}{'full':>8}{'ex-MASK':>9}{'EM0&valid':>11}{'sprm_LJ':>9}")
rowsA = []
for m in METRICS + ["CIS"]:
    a_full, a_valid, a_e0 = auc(df, m), auc(valid, m), auc(e0v, m)
    sp = stats.spearmanr(df[m], df.lj_mean).correlation
    rowsA.append(dict(metric=m, auc_full=a_full, auc_valid=a_valid, auc_em0valid=a_e0, spearman_lj=sp))
    print(f"{PRETTY[m]:<16}{a_full:>8.3f}{a_valid:>9.3f}{a_e0:>11.3f}{sp:>9.3f}")
em_full = auc(df.assign(**{}), "em"); print(f"{'EM (ref)':<16}{auc(df,'em'):>8.3f}{auc(valid,'em'):>9.3f}{'--':>11}")
rA = pd.DataFrame(rowsA)
print(f"\n  EM=0 & valid rescued names (LJ=1): {int(e0v.lj.sum())} of {len(e0v)} ({e0v.lj.mean():.1%})")
print(f"  qual_char on MASK rows: mean={df[df.degenerate].qual_char.mean():.2f} (LJ={df[df.degenerate].lj.mean():.2f}) "
      f"<- qual rates the worst failure as max-readable")

# ============ PART B: CIS vs ML vs EM ============
g = GroupKFold(n_splits=8)
gb = GradientBoostingClassifier(n_estimators=200, max_depth=3, learning_rate=0.05, subsample=0.8, random_state=0)
y, grp = df.lj.values, df.model.values
p_gb = cross_val_predict(gb, df[METRICS].values, y, groups=grp, cv=g, method="predict_proba")[:, 1]
df["fused_gbdt"] = p_gb
cv = {"CIS (mean sim)": auc(df, "CIS"), "GBDT(6) OOF": roc_auc_score(y, p_gb),
      "best single (NW)": auc(df, "nw_sim"), "EM": auc(df, "em")}
print("\n== PART B: scoring power (full regime, AUC->LJ) ==")
for k, v in sorted(cv.items(), key=lambda x: -x[1]):
    print(f"  {k:<20}{v:.3f}")
print(f"  [LJ judge noise ceiling = {CEILING}]")

# cross-family transfer: train AR(FIM+Seq2Seq) -> test dLLM
dll = df.family == "dLLM"
gb.fit(df[~dll][METRICS].values, df[~dll].lj.values)
p_tr = gb.predict_proba(df[dll][METRICS].values)[:, 1]
tr_gbdt = roc_auc_score(df[dll].lj, p_tr); tr_cis = roc_auc_score(df[dll].lj, df[dll].CIS)
print(f"\n  cross-family transfer (train AR -> test dLLM):  GBDT={tr_gbdt:.3f}  CIS={tr_cis:.3f}  (edge {tr_gbdt-tr_cis:+.3f})")

# leaderboard rank-corr with LJ
lb = df.groupby(["model", "family"]).agg(EM=("em", "mean"), LJ=("lj", "mean"),
                                         CIS=("CIS", "mean"), n=("em", "size")).reset_index()
rho_cis = stats.spearmanr(lb.CIS, lb.LJ).correlation
rho_em = stats.spearmanr(lb.EM, lb.LJ).correlation
print(f"  leaderboard Spearman with LJ:  CIS={rho_cis:.3f}  EM={rho_em:.3f}")

# CIS calibration via OOF isotonic (GroupKFold by model)
oof = np.zeros(len(df))
for tr, te in g.split(df.CIS.values, y, grp):
    m = IsotonicRegression(out_of_bounds="clip").fit(df.CIS.values[tr], y[tr]); oof[te] = m.predict(df.CIS.values[te])
brier_raw = brier_score_loss(y, df.CIS); brier_cal = brier_score_loss(y, oof)
print(f"  CIS calibration Brier: raw={brier_raw:.3f} -> OOF-isotonic={brier_cal:.3f}")
df["CIS_prob"] = oof

# ----- save outputs -----
df[["model", "family", "id", "ground_truth", "agreed_pred", "degenerate", "em", "lj", "lj_mean",
    "CIS", "CIS_prob", "fused_gbdt"] + METRICS].to_csv("results/identifier_metrics/cis_scores.csv", index=False)
lb.to_csv("results/identifier_metrics/leaderboard_cis.csv", index=False)
summ = {"n": len(df), "degenerate_frac": float(df.degenerate.mean()), "ceiling_auc": CEILING,
        "partA": rA.to_dict("records"), "partB_cv": cv,
        "transfer": {"gbdt": float(tr_gbdt), "cis": float(tr_cis)},
        "leaderboard_spearman": {"cis": float(rho_cis), "em": float(rho_em)},
        "cis_brier": {"raw": float(brier_raw), "oof_isotonic": float(brier_cal)}}
json.dump(summ, open("results/identifier_metrics/metric_eval_final.json", "w"), indent=2)

# ================= FIGURES =================
# F1: reasonableness AUC bars, 3 views
fig, ax = plt.subplots(figsize=(9, 5.5))
order = rA[rA.metric.isin(METRICS)].sort_values("auc_valid")
yy = np.arange(len(order)); h = 0.27
ax.barh(yy + h, order.auc_full, h, color=GREY, label="full regime (incl. MASK)")
ax.barh(yy, order.auc_valid, h, color=BLUE, label="valid fills (ex-MASK)")
ax.barh(yy - h, order.auc_em0valid, h, color=ORANGE, label="hardest: EM=0 & valid (synonym rescue)")
ax.axvline(0.5, color="k", ls="--", lw=1)
ax.axvline(auc(valid, "em"), color="k", ls=":", lw=1.5, label=f"EM alone ({auc(valid,'em'):.2f})")
ax.axvline(CEILING, color=GREEN, ls="-", lw=1.5, label=f"LJ judge ceiling ({CEILING})")
ax.set_yticks(yy); ax.set_yticklabels([PRETTY[m] for m in order.metric])
ax.set_xlim(0.35, 1.0); ax.set_xlabel("AUC: single metric -> LJ acceptance")
ax.set_title("Is each new metric a reasonable LJ proxy?  (MASK-corrected, 3 views)")
ax.legend(fontsize=8, loc="lower right")
fig.tight_layout(); fig.savefig(f"{FIG}/F1_reasonableness_masked.png", dpi=140); plt.close(fig)

# F2: MASK contamination
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
mc = df[df.degenerate].groupby("family").size().reindex(["dLLM", "FIM", "Seq2Seq"]).fillna(0)
tot = df.groupby("family").size().reindex(["dLLM", "FIM", "Seq2Seq"]).fillna(0)
x = np.arange(3)
ax1.bar(x, tot, color=GREY, label="all consistent")
ax1.bar(x, mc, color=ORANGE, label="MASK / special-token echo")
for i, (a, b) in enumerate(zip(tot, mc)):
    ax1.text(i, a + 30, f"{int(b)}/{int(a)}\n({b/a:.0%})" if a else "0", ha="center", fontsize=9)
ax1.set_xticks(x); ax1.set_xticklabels(["dLLM", "FIM", "Seq2Seq"]); ax1.set_ylabel("# consistent samples")
ax1.set_title("Copy-the-mask non-fills are an AR-model failure\n(dLLMs: 0)"); ax1.legend()
# qual trap: with vs without MASK
qa = {"qual_char": (auc(df, "qual_char"), auc(valid, "qual_char")),
      "qual_word": (auc(df, "qual_word"), auc(valid, "qual_word"))}
xx = np.arange(2)
ax2.bar(xx - 0.2, [qa["qual_char"][0], qa["qual_word"][0]], 0.4, color=GREY, label="full (incl MASK)")
ax2.bar(xx + 0.2, [qa["qual_char"][1], qa["qual_word"][1]], 0.4, color=BLUE, label="valid fills")
ax2.axhline(0.5, color="k", ls="--", lw=1)
ax2.set_xticks(xx); ax2.set_xticklabels(["qual_char", "qual_word"]); ax2.set_ylim(0.3, 0.6)
ax2.set_title("M3 quality: rates MASK=1.0 -> looks anti-correlated;\nex-MASK it is ~chance (orthogonal axis)"); ax2.legend(fontsize=8)
fig.tight_layout(); fig.savefig(f"{FIG}/F2_mask_contamination.png", dpi=140); plt.close(fig)

# F3: CIS vs alternatives + ceiling, and leaderboard
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
items = sorted(cv.items(), key=lambda kv: kv[1])
cols = [ORANGE if "CIS" in n else (GREEN if "GBDT" in n else (GREY if "EM" == n else BLUE)) for n, _ in items]
ax1.barh(range(len(items)), [v for _, v in items], color=cols)
for i, (n, v) in enumerate(items):
    ax1.text(v + 0.004, i, f"{v:.3f}", va="center", fontsize=9)
ax1.axvline(CEILING, color=GREEN, ls="--", lw=1.5)
ax1.text(CEILING, -0.6, f"LJ ceiling {CEILING}", color=GREEN, fontsize=8, ha="center")
ax1.set_yticks(range(len(items))); ax1.set_yticklabels([n for n, _ in items])
ax1.set_xlim(0.5, 1.0); ax1.set_xlabel("AUC -> LJ acceptance (full regime)")
ax1.set_title("The integrated CIS = a trained model, at the judge ceiling")
for fam, c in [("dLLM", ORANGE), ("FIM", BLUE), ("Seq2Seq", GREY)]:
    s = lb[lb.family == fam]
    ax2.scatter(s.EM, s.LJ, s=80, color=c, marker="o", edgecolor="k", label=f"{fam} EM", zorder=3)
    ax2.scatter(s.CIS, s.LJ, s=80, color=c, marker="^", edgecolor="k", alpha=0.55, zorder=2)
ax2.plot([0, lb.LJ.max()], [0, lb.LJ.max()], "k--", lw=1, alpha=0.5)
ax2.set_xlabel("EM (circles)  /  CIS (triangles), model mean"); ax2.set_ylabel("LJ acceptance (model mean)")
ax2.set_title(f"Per-model: CIS tracks LJ\n(Spearman: CIS={rho_cis:.2f}, EM={rho_em:.2f})"); ax2.legend(fontsize=8)
fig.tight_layout(); fig.savefig(f"{FIG}/F3_cis_summary.png", dpi=140); plt.close(fig)

# F4: CIS calibration
fig, ax = plt.subplots(figsize=(6, 6))
from sklearn.calibration import calibration_curve
fr, mp = calibration_curve(y, oof, n_bins=10, strategy="quantile")
ax.plot(mp, fr, "o-", color=ORANGE, label=f"CIS+isotonic (Brier {brier_cal:.3f})")
ax.plot([0, 1], [0, 1], "k--", label="perfect")
ax.set_xlabel("predicted P(LJ-accept)"); ax.set_ylabel("observed accept rate")
ax.set_title("CIS -> calibrated acceptance probability"); ax.legend()
fig.tight_layout(); fig.savefig(f"{FIG}/F4_cis_calibration.png", dpi=140); plt.close(fig)

print("\nfigures -> figures/new/metric_eval/F1..F4 ; outputs -> results/identifier_metrics/cis_scores.csv, leaderboard_cis.csv, metric_eval_final.json")
