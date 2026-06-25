"""
Build paper-ready figures + summary table + paragraph for the ML smell
localization head.

Output directory: results/ml_layer_paper/
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "results/ml_layer_paper"
OUT_DIR.mkdir(parents=True, exist_ok=True)

UNMASK_CSV = ROOT / "results/abc_exp/abc/unmasking_order_20260312_225458.csv"
RANKING_CSV = ROOT / "results/abc_exp/abc/token_ranking_20260312_230225.csv"

FEATURES = ["avg_flip_step", "first_confident_step",
            "mean_entropy_change", "max_entropy_change"]
FEATURES_LABEL = [r"$\overline{flip}$", r"$conf$", r"$\overline{\Delta H}$", r"$\max\Delta H$"]

MODEL_COLORS = {"gbdt": "#2c7fb8", "logreg": "#7fcdbb", "mlp": "#fdae6b",
                "heuristic": "#bcbddc", "random": "#bbbbbb"}
MODEL_LABEL = {"gbdt": "GBDT", "logreg": "LogReg", "mlp": "MLP",
               "heuristic": "Heuristic", "random": "Random"}

plt.rcParams.update({
    "font.size": 11, "axes.labelsize": 12, "axes.titlesize": 13,
    "legend.fontsize": 10, "figure.dpi": 200,
})


def load_data():
    u = pd.read_csv(UNMASK_CSV); r = pd.read_csv(RANKING_CSV)
    key = ["sample_id", "identifier_name", "is_smell_token"]
    mask = (u[key].reset_index(drop=True) == r[key].reset_index(drop=True)).all(axis=1)
    df = pd.concat([
        u.loc[mask, key + ["avg_flip_step", "first_confident_step"]].reset_index(drop=True),
        r.loc[mask, ["mean_entropy_change", "max_entropy_change"]].reset_index(drop=True),
    ], axis=1)
    return df


def build_models():
    return {
        "logreg": Pipeline([("s", StandardScaler()),
                            ("c", LogisticRegression(class_weight="balanced",
                                                     max_iter=2000, random_state=42))]),
        "mlp":    Pipeline([("s", StandardScaler()),
                            ("c", MLPClassifier(hidden_layer_sizes=(32, 16),
                                                max_iter=400, random_state=42,
                                                early_stopping=True))]),
        "gbdt":   Pipeline([("c", GradientBoostingClassifier(n_estimators=200,
                                                             max_depth=3,
                                                             random_state=42))]),
    }


def run_loso(df):
    X = df[FEATURES].values
    y = df["is_smell_token"].astype(int).values
    groups = df["sample_id"].values
    probs = {n: np.zeros(len(df)) for n in build_models()}
    cv = LeaveOneGroupOut()
    for i, (tr, te) in enumerate(cv.split(X, y, groups), 1):
        for name, pipe in build_models().items():
            pipe.fit(X[tr], y[tr])
            probs[name][te] = pipe.predict_proba(X[te])[:, 1]
        print(f"  fold {i}/{cv.get_n_splits(X, y, groups)} done")
    # baselines
    probs["heuristic"] = (df["mean_entropy_change"].clip(0, 1.5).values / 1.5)
    rng = np.random.default_rng(42)
    probs["random"] = rng.random(len(df))
    return probs, y


def per_sample_mrr(df, prob, ks=(1, 3, 5, 10)):
    rows = []
    df = df.copy(); df["_p"] = prob
    for sid in sorted(df["sample_id"].unique()):
        sub = df[df["sample_id"] == sid].sort_values("_p", ascending=False).reset_index(drop=True)
        if not sub["is_smell_token"].any():
            continue
        rank = int(sub.index[sub["is_smell_token"]].min()) + 1
        row = {"sample_id": sid, "n_candidates": len(sub), "first_smell_rank": rank,
               "rr": 1.0 / rank}
        for k in ks: row[f"in_top_{k}"] = int(rank <= k)
        rows.append(row)
    return pd.DataFrame(rows)


# ── figures ──────────────────────────────────────────────────────────────────

def fig_localization_bar(loc_df: pd.DataFrame):
    """Headline figure: MRR + R@K bars per model."""
    order = ["random", "heuristic", "logreg", "mlp", "gbdt"]
    metrics = ["MRR", "R@1", "R@3", "R@5", "R@10"]
    loc_df = loc_df.set_index("model").reindex(order)

    fig, ax = plt.subplots(figsize=(9, 4.2))
    x = np.arange(len(metrics))
    w = 0.16
    for i, m in enumerate(order):
        vals = loc_df.loc[m, metrics].values.astype(float)
        ax.bar(x + (i - 2) * w, vals, w, label=MODEL_LABEL[m],
               color=MODEL_COLORS[m], edgecolor="black", linewidth=0.4)
    ax.set_xticks(x); ax.set_xticklabels(metrics)
    ax.set_ylabel("Localization score")
    ax.set_ylim(0, max(0.25, loc_df[metrics].values.max() * 1.2))
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    ax.legend(ncol=5, loc="upper right", frameon=False)
    ax.set_title("Per-(sample, run) localization of injected smell identifier")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "fig_localization_bar.png", dpi=300)
    plt.savefig(OUT_DIR / "fig_localization_bar.pdf", bbox_inches="tight")
    plt.close()


def fig_classification_bar(metrics_json: dict):
    """Token-level F1 / ROC-AUC / PR-AUC."""
    order = ["heuristic", "logreg", "mlp", "gbdt"]
    metrics = ["f1", "roc_auc", "pr_auc"]
    label = {"f1": "F1", "roc_auc": "ROC-AUC", "pr_auc": "PR-AUC"}

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    x = np.arange(len(metrics))
    w = 0.18
    for i, m in enumerate(order):
        vals = [metrics_json[m][k] for k in metrics]
        ax.bar(x + (i - 1.5) * w, vals, w, label=MODEL_LABEL[m],
               color=MODEL_COLORS[m], edgecolor="black", linewidth=0.4)
    ax.set_xticks(x); ax.set_xticklabels([label[k] for k in metrics])
    ax.set_ylabel("Score"); ax.set_ylim(0, 1.0)
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    ax.legend(loc="upper right", frameon=False)
    ax.set_title("Per-token binary classification (smell vs clean)")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "fig_classification_bar.png", dpi=300)
    plt.savefig(OUT_DIR / "fig_classification_bar.pdf", bbox_inches="tight")
    plt.close()


def fig_per_sample_mrr(per_sample: dict[str, pd.DataFrame]):
    """Per-sample first-smell-rank distribution + MRR — exposes variance."""
    order = ["heuristic", "logreg", "mlp", "gbdt"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2),
                             gridspec_kw={"width_ratios": [1.2, 1.0]})

    # left: per-sample rank dots
    for i, m in enumerate(order):
        df = per_sample[m]
        y = np.full(len(df), i)
        axes[0].scatter(df["first_smell_rank"], y,
                        c=MODEL_COLORS[m], s=40, edgecolor="black", linewidth=0.4,
                        alpha=0.85, label=MODEL_LABEL[m])
    axes[0].set_xscale("log")
    axes[0].set_xlabel("First-smell rank within sample (log)")
    axes[0].set_yticks(range(len(order))); axes[0].set_yticklabels([MODEL_LABEL[m] for m in order])
    axes[0].axvline(5, color="red", linestyle="--", linewidth=0.8, alpha=0.6)
    axes[0].text(5.5, len(order) - 0.5, "top-5 cutoff", color="red", fontsize=9)
    axes[0].grid(axis="x", linestyle=":", alpha=0.4)
    axes[0].set_title("Per-sample rank of the injected smell")

    # right: MRR box plot
    data = [per_sample[m]["rr"].values for m in order]
    bp = axes[1].boxplot(data, labels=[MODEL_LABEL[m] for m in order],
                          patch_artist=True, widths=0.55, showfliers=False)
    for patch, m in zip(bp["boxes"], order):
        patch.set_facecolor(MODEL_COLORS[m])
        patch.set_edgecolor("black")
    for m in bp["medians"]: m.set_color("black")
    # overlay mean dots
    means = [np.mean(d) for d in data]
    axes[1].scatter(range(1, len(order) + 1), means, marker="D",
                    color="black", s=30, zorder=10, label="mean (= MRR)")
    axes[1].set_ylabel("Reciprocal rank")
    axes[1].grid(axis="y", linestyle=":", alpha=0.4)
    axes[1].legend(loc="upper left", frameon=False)
    axes[1].set_title("Reciprocal-rank distribution")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "fig_per_sample_mrr.png", dpi=300)
    plt.savefig(OUT_DIR / "fig_per_sample_mrr.pdf", bbox_inches="tight")
    plt.close()


def fig_feature_importance(df: pd.DataFrame, y: np.ndarray):
    """Standardized LogReg coefs + GBDT impurity importances side-by-side."""
    lr = Pipeline([("s", StandardScaler()),
                   ("c", LogisticRegression(class_weight="balanced",
                                            max_iter=2000, random_state=42))]).fit(df[FEATURES].values, y)
    gb = GradientBoostingClassifier(n_estimators=200, max_depth=3,
                                    random_state=42).fit(df[FEATURES].values, y)

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    axes[0].barh(FEATURES_LABEL, lr.named_steps["c"].coef_[0],
                 color=MODEL_COLORS["logreg"], edgecolor="black")
    axes[0].axvline(0, color="grey", linewidth=0.7)
    axes[0].set_title("LogReg coefficient (standardized)")
    axes[1].barh(FEATURES_LABEL, gb.feature_importances_,
                 color=MODEL_COLORS["gbdt"], edgecolor="black")
    axes[1].set_title("GBDT split importance")
    for ax in axes:
        ax.grid(axis="x", linestyle=":", alpha=0.4)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "fig_feature_importance.png", dpi=300)
    plt.savefig(OUT_DIR / "fig_feature_importance.pdf", bbox_inches="tight")
    plt.close()


# ── main ─────────────────────────────────────────────────────────────────────

def aggregate_localization(per_sample: dict[str, pd.DataFrame], ks=(1, 3, 5, 10)) -> pd.DataFrame:
    rows = []
    for m, df in per_sample.items():
        if m == "random":
            # use the analytic expectation instead of one realization
            sizes = df["n_candidates"].values
            row = {"model": "random", "n_queries": len(sizes),
                   "MRR": float(np.mean([1 / n for n in sizes]))}
            for k in ks: row[f"R@{k}"] = float(np.mean([min(k, n) / n for n in sizes]))
        else:
            row = {"model": m, "n_queries": len(df), "MRR": float(df["rr"].mean())}
            for k in ks: row[f"R@{k}"] = float(df[f"in_top_{k}"].mean())
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    print("loading features ...")
    df = load_data()
    print(f"  n={len(df):,} samples={df.sample_id.nunique()} smell={int(df.is_smell_token.sum())}")

    print("running LOSO CV ...")
    probs, y = run_loso(df)

    print("computing per-sample localization ...")
    per_sample = {m: per_sample_mrr(df, p) for m, p in probs.items()}
    per_sample["random"].to_csv(OUT_DIR / "per_sample_random.csv", index=False)
    per_sample["gbdt"].to_csv(OUT_DIR / "per_sample_gbdt.csv", index=False)

    loc_df = aggregate_localization(per_sample)
    loc_df.to_csv(OUT_DIR / "table_localization.csv", index=False)

    # classification metrics
    from sklearn.metrics import f1_score, roc_auc_score, average_precision_score
    cls_rows = []
    for m, p in probs.items():
        pred = (p >= 0.5).astype(int)
        cls_rows.append({
            "model": m, "f1": f1_score(y, pred, zero_division=0),
            "roc_auc": roc_auc_score(y, p) if m != "random" else 0.5,
            "pr_auc": average_precision_score(y, p),
        })
    cls_df = pd.DataFrame(cls_rows)
    cls_df.to_csv(OUT_DIR / "table_classification.csv", index=False)
    cls_dict = {r["model"]: r for r in cls_rows}

    print("rendering figures ...")
    fig_localization_bar(loc_df)
    fig_classification_bar(cls_dict)
    fig_per_sample_mrr(per_sample)
    fig_feature_importance(df, y)

    # combined summary table
    summary = (loc_df.merge(cls_df, on="model", how="outer")
               [["model", "f1", "roc_auc", "pr_auc",
                 "MRR", "R@1", "R@3", "R@5", "R@10"]])
    summary.to_csv(OUT_DIR / "table_summary.csv", index=False)

    print(f"\nDone. Artifacts → {OUT_DIR}")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()
