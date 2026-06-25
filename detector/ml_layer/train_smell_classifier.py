"""
ML Smell Classifier on dLLM probe features.

Trains lightweight classifiers (LogReg, MLP, GradientBoosting) on top of the
DiffuCoder probe signals collected by `generate_features.py` (or the legacy
ABC experiment CSVs):

  * avg_flip_step          (Experiment B: unmasking order)
  * first_confident_step   (Experiment B)
  * mean_entropy_change    (Experiment C: token ranking)
  * max_entropy_change     (Experiment C)

These four scalars per identifier are exactly what the UMAP plots visualize
in 2D: deeper layers / later denoising steps separate smelly from clean
tokens. Here we ask whether a tiny supervised head on top of the same
features can drive that separation into a calibrated smell probability,
which is what the paper's Section RQ2/RQ3 describes as a "lightweight ML
layer for localization".

Two CSV schemas are supported:
  --features <path.csv>   new schema from generate_features.py
                          (sample_id, run_id, version, injected_name,
                           gt_target, identifier_name, is_smell_token, …)
  (no flag)               legacy ABC CSVs joined from results/abc_exp/abc/

Outputs:
  detector/ml_layer/runs/<timestamp>/
    metrics.json          aggregated test metrics for every model
    coef_logreg.json      logistic-regression weights on standardized inputs
    feature_importance.png
    roc_pr_curves.png
    confusion_matrices.png
    localization_examples.csv  top-K smelliest tokens per (sample,run)
    model_<name>.joblib   pickled sklearn pipeline
"""

import argparse
import json
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GroupShuffleSplit, LeaveOneGroupOut
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib

ROOT = Path(__file__).resolve().parents[2]
UNMASK_CSV = ROOT / "results/abc_exp/abc/unmasking_order_20260312_225458.csv"
RANKING_CSV = ROOT / "results/abc_exp/abc/token_ranking_20260312_230225.csv"

FEATURES = [
    "avg_flip_step",
    "first_confident_step",
    "mean_entropy_change",
    "max_entropy_change",
]


def load_legacy_abc() -> pd.DataFrame:
    """Concat the two ABC CSVs side-by-side; rows are 99.99% aligned."""
    unmask = pd.read_csv(UNMASK_CSV)
    ranking = pd.read_csv(RANKING_CSV)
    assert len(unmask) == len(ranking), "row-count mismatch between ABC csvs"
    key = ["sample_id", "identifier_name", "is_smell_token"]
    aligned_mask = (unmask[key].reset_index(drop=True)
                    == ranking[key].reset_index(drop=True)).all(axis=1)
    n_drop = int((~aligned_mask).sum())
    if n_drop:
        print(f"  dropping {n_drop} mis-aligned rows ({n_drop/len(unmask):.4%})")
    df = pd.concat(
        [
            unmask.loc[aligned_mask, key + ["avg_flip_step", "first_confident_step"]].reset_index(drop=True),
            ranking.loc[aligned_mask, ["mean_entropy_change", "max_entropy_change"]].reset_index(drop=True),
        ],
        axis=1,
    )
    # legacy schema didn't track run_id/version — treat as a single smell run
    df["run_id"] = 0
    df["version"] = "smell"
    df["injected_name"] = "SMELL_DUMMY_TOKEN"
    return df


def load_features(path: Path) -> pd.DataFrame:
    """Read a feature CSV produced by generate_features.py."""
    df = pd.read_csv(path)
    required = {"sample_id", "identifier_name", "is_smell_token"} | set(FEATURES)
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"feature CSV missing columns: {sorted(missing)}")
    for c in ("run_id", "version", "injected_name"):
        if c not in df.columns:
            df[c] = 0 if c == "run_id" else ("smell" if c == "version" else "")
    df["is_smell_token"] = df["is_smell_token"].astype(bool)
    return df


def heuristic_baseline(X: pd.DataFrame) -> np.ndarray:
    """The hand-tuned rule used elsewhere in the repo (entropy-change threshold).

    Calibrated from the per-class statistics in `results/abc_exp/ABC.md`:
    smell mean ~0.59 vs non-smell ~0.21, so 0.4 sits midway.
    """
    return (X["mean_entropy_change"] > 0.4).astype(int).values


def split_by_sample(df: pd.DataFrame, test_frac: float = 0.2, seed: int = 42):
    """Group split on `sample_id` so the same code never appears in train+test."""
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_frac, random_state=seed)
    train_idx, test_idx = next(splitter.split(df, groups=df["sample_id"]))
    return df.iloc[train_idx].reset_index(drop=True), df.iloc[test_idx].reset_index(drop=True)


def build_models() -> dict[str, Pipeline]:
    return {
        "logreg": Pipeline([
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(
                class_weight="balanced",
                max_iter=2000,
                solver="lbfgs",
                random_state=42,
            )),
        ]),
        "mlp": Pipeline([
            ("scale", StandardScaler()),
            ("clf", MLPClassifier(
                hidden_layer_sizes=(32, 16),
                max_iter=400,
                random_state=42,
                early_stopping=True,
            )),
        ]),
        "gbdt": Pipeline([
            ("clf", GradientBoostingClassifier(
                n_estimators=200,
                max_depth=3,
                random_state=42,
            )),
        ]),
    }


def evaluate(name: str, y_true: np.ndarray, y_prob: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "model": name,
        "n_test": int(len(y_true)),
        "n_pos": int(y_true.sum()),
        "f1": float(f1_score(y_true, y_pred)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "confusion": confusion_matrix(y_true, y_pred).tolist(),
        "report": classification_report(y_true, y_pred, output_dict=True, zero_division=0),
    }


def plot_curves(out_path: Path, results: dict[str, dict], y_test: np.ndarray, probs: dict[str, np.ndarray]):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for name, prob in probs.items():
        fpr, tpr, _ = roc_curve(y_test, prob)
        axes[0].plot(fpr, tpr, label=f"{name} (AUC={results[name]['roc_auc']:.3f})")
        prec, rec, _ = precision_recall_curve(y_test, prob)
        axes[1].plot(rec, prec, label=f"{name} (AP={results[name]['pr_auc']:.3f})")
    axes[0].plot([0, 1], [0, 1], linestyle="--", color="grey")
    axes[0].set_title("ROC")
    axes[0].set_xlabel("FPR"); axes[0].set_ylabel("TPR"); axes[0].legend()
    axes[1].set_title("Precision-Recall")
    axes[1].set_xlabel("Recall"); axes[1].set_ylabel("Precision"); axes[1].legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_confusions(out_path: Path, results: dict[str, dict]):
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    if n == 1:
        axes = [axes]
    for ax, (name, res) in zip(axes, results.items()):
        cm = np.array(res["confusion"])
        im = ax.imshow(cm, cmap="Blues")
        ax.set_title(f"{name}\nF1={res['f1']:.3f}")
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(["clean", "smell"]); ax.set_yticklabels(["clean", "smell"])
        ax.set_xlabel("pred"); ax.set_ylabel("true")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black")
        fig.colorbar(im, ax=ax, fraction=0.046)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_feature_importance(out_path: Path, models: dict[str, Pipeline]):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    # logreg coefficients on standardized features
    lr = models["logreg"].named_steps["clf"]
    axes[0].barh(FEATURES, lr.coef_[0], color="#3498db")
    axes[0].set_title("LogReg coef (standardized)")
    axes[0].axvline(0, color="grey", linewidth=0.8)
    # gbdt feature importances
    gb = models["gbdt"].named_steps["clf"]
    axes[1].barh(FEATURES, gb.feature_importances_, color="#e74c3c")
    axes[1].set_title("GBDT importance")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def localization_metrics(df_test: pd.DataFrame, probs: dict[str, np.ndarray],
                         ks: tuple[int, ...] = (1, 3, 5, 10)) -> pd.DataFrame:
    """Per-(sample, run) ranking eval — the actual end-to-end task.

    Each smell-injection run is one query: rank that sample's identifiers by
    smell-prob, ask whether the injected name lands in the top-K. Rows from
    clean/GT versions are scored but not used as positive queries.
    """
    df = df_test.copy().reset_index(drop=True)
    df["_orig_idx"] = df.index
    smell = df[df["version"] == "smell"] if "version" in df.columns else df

    key_cols = ["sample_id"] + (["run_id"] if "run_id" in smell.columns else [])
    has_smell = smell.groupby(key_cols)["is_smell_token"].any()
    query_keys = has_smell[has_smell].index.tolist()

    rows = []
    sizes = []
    for name, prob in probs.items():
        df["_p"] = prob
        rrs, hits = [], {k: 0 for k in ks}
        for q in query_keys:
            q = q if isinstance(q, tuple) else (q,)
            mask = pd.Series(True, index=smell.index)
            for col, val in zip(key_cols, q):
                mask &= (smell[col] == val)
            idxs = smell.loc[mask, "_orig_idx"].tolist()
            sub = df.loc[idxs].sort_values("_p", ascending=False).reset_index(drop=True)
            if name == list(probs.keys())[0]:
                sizes.append(len(sub))
            hit_ranks = sub.index[sub["is_smell_token"]].tolist()
            if hit_ranks:
                rank = int(hit_ranks[0]) + 1
                rrs.append(1.0 / rank)
                for k in ks:
                    if rank <= k:
                        hits[k] += 1
            else:
                rrs.append(0.0)
        n = len(query_keys)
        row = {
            "model": name,
            "n_queries": n,
            "MRR": float(np.mean(rrs)) if rrs else 0.0,
            "avg_candidates_per_query": float(np.mean(sizes)) if sizes else 0.0,
        }
        for k in ks:
            row[f"R@{k}"] = hits[k] / n if n else 0.0
        rows.append(row)

    if query_keys and sizes:
        row = {
            "model": "random",
            "n_queries": len(sizes),
            "MRR": float(np.mean([1.0 / n if n > 0 else 0 for n in sizes])),
            "avg_candidates_per_query": float(np.mean(sizes)),
        }
        for k in ks:
            row[f"R@{k}"] = float(np.mean([min(k, n) / n if n > 0 else 0 for n in sizes]))
        rows.append(row)

    return pd.DataFrame(rows)


def localization_examples(df_test: pd.DataFrame, prob: np.ndarray, k_per_sample: int = 5, n_samples: int = 5) -> pd.DataFrame:
    """Top-K identifiers per (sample,run) by smell probability — qualitative dump."""
    out = df_test.copy()
    out["smell_prob"] = prob
    smell_only = out[out["version"] == "smell"] if "version" in out.columns else out
    keys = ["sample_id"] + (["run_id"] if "run_id" in out.columns else [])
    has_smell = (smell_only.groupby(keys)["is_smell_token"].any()
                 .sort_values(ascending=False))
    queries = list(has_smell[has_smell].index)[:n_samples]
    rows = []
    for q in queries:
        if not isinstance(q, tuple):
            q = (q,)
        mask = pd.Series(True, index=smell_only.index)
        for col, val in zip(keys, q):
            mask &= (smell_only[col] == val)
        sub = smell_only[mask].sort_values("smell_prob", ascending=False).head(k_per_sample).copy()
        for col, val in zip(keys, q):
            sub[col] = val
        sub["rank_in_query"] = range(1, len(sub) + 1)
        out_cols = keys + ["rank_in_query", "identifier_name", "is_smell_token",
                           "smell_prob"] + FEATURES
        if "injected_name" in sub.columns:
            out_cols.insert(len(keys), "injected_name")
        rows.append(sub[out_cols])
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def run_loso(df: pd.DataFrame) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Leave-one-sample-out CV: for each sample, train on the other 19 and
    score this sample's tokens. Returns (probs_per_model_concatenated, y_concat)
    in the original DF row order so localization metrics can be computed.

    There are only ~20 unique sample_ids in the ABC experiment, so a flat
    train/test split is unstable. LOSO uses every sample for evaluation.
    """
    X = df[FEATURES].values
    y = df["is_smell_token"].astype(int).values
    groups = df["sample_id"].values

    model_specs = build_models()
    probs_out: dict[str, np.ndarray] = {name: np.zeros(len(df)) for name in model_specs}

    splitter = LeaveOneGroupOut()
    for i, (tr, te) in enumerate(splitter.split(X, y, groups), 1):
        sid = df.iloc[te[0]]["sample_id"]
        for name, pipe in build_models().items():  # fresh pipeline per fold
            pipe.fit(X[tr], y[tr])
            probs_out[name][te] = pipe.predict_proba(X[te])[:, 1]
        if i % 5 == 0:
            print(f"      fold {i}/20 done (held-out sample_id={sid})")
    return probs_out, y


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--features", default=None,
                   help="path to a feature CSV from generate_features.py "
                        "(falls back to legacy ABC CSVs if omitted)")
    args = p.parse_args()

    print("[1/5] loading features ...")
    if args.features:
        df = load_features(Path(args.features))
        print(f"      source: {args.features}")
    else:
        df = load_legacy_abc()
        print(f"      source: legacy ABC CSVs")
    n_samples = df["sample_id"].nunique()
    print(f"      n={len(df):,}  unique_samples={n_samples}  "
          f"smell_pos={int(df.is_smell_token.sum()):,}  "
          f"({df.is_smell_token.mean():.4%} positive)")

    print("[2/5] leave-one-sample-out CV ...")
    probs, y_all = run_loso(df)

    print("[3/5] heuristic baseline (in-sample threshold) ...")
    probs["heuristic"] = (df["mean_entropy_change"].clip(0, 1.5).values / 1.5)

    print("[4/5] aggregating metrics ...")
    fitted = build_models()  # also train one full-data model per kind for export & coefs
    for name, pipe in fitted.items():
        pipe.fit(df[FEATURES].values, y_all)

    results = {}
    for name, prob in probs.items():
        pred = (prob >= 0.5).astype(int)
        results[name] = evaluate(name, y_all, prob, pred)

    train = test = df  # for compatibility w/ downstream code; we evaluate on all rows

    # ── output ─────────────────────────────────────────────────────────────
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = ROOT / f"detector/ml_layer/runs/{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    lr = fitted["logreg"].named_steps["clf"]
    with open(out_dir / "coef_logreg.json", "w") as f:
        json.dump({
            "features": FEATURES,
            "coef_standardized": lr.coef_[0].tolist(),
            "intercept": float(lr.intercept_[0]),
        }, f, indent=2)

    plot_curves(out_dir / "roc_pr_curves.png", results, y_all, probs)
    plot_confusions(out_dir / "confusion_matrices.png", results)
    plot_feature_importance(out_dir / "feature_importance.png", fitted)

    # localization eval — the actual task
    loc_metrics = localization_metrics(test, probs, ks=(1, 3, 5, 10))
    loc_metrics.to_csv(out_dir / "localization_metrics.csv", index=False)

    # qualitative dump using best model (by ROC-AUC)
    best_name = max(["logreg", "mlp", "gbdt"], key=lambda n: results[n]["roc_auc"])
    print(f"   best model by ROC-AUC: {best_name}")
    loc_df = localization_examples(test, probs[best_name], k_per_sample=5, n_samples=8)
    loc_df.to_csv(out_dir / f"localization_examples_{best_name}.csv", index=False)

    for name, pipe in fitted.items():
        joblib.dump(pipe, out_dir / f"model_{name}.joblib")

    print("\n[5/5] summary")
    print(f"  output dir: {out_dir}")
    summary = pd.DataFrame([
        {"model": r["model"], "F1": r["f1"], "ROC-AUC": r["roc_auc"], "PR-AUC": r["pr_auc"]}
        for r in results.values()
    ])
    print("\n  Classification (per-token):")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\n  Localization (per-sample ranking):")
    print(loc_metrics.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()
