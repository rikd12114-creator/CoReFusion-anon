"""
End-to-end smell localizer.

Given:
  * a trained ML head (joblib pickle from train_smell_classifier.py)
  * either a Java source file (GPU mode) OR a feature CSV (CSV mode)

Produces a ranked list of identifiers with their smell probabilities, plus
per-(sample,run) localization metrics if the input CSV contains
`is_smell_token` ground-truth labels.

GPU mode:
  python detector/ml_layer/localize.py \\
    --model-path detector/ml_layer/runs/<ts>/model_gbdt.joblib \\
    --java-file path/to/Foo.java
  → Loads DiffuCoder, runs the 64-step probe, scores every identifier.

CSV mode (no GPU needed — replay precomputed features):
  python detector/ml_layer/localize.py \\
    --model-path detector/ml_layer/runs/<ts>/model_gbdt.joblib \\
    --features detector/ml_layer/features/features_<...>.csv
  → Reads features, scores, prints localization metrics if labels present.
"""

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

FEATURES = [
    "avg_flip_step",
    "first_confident_step",
    "mean_entropy_change",
    "max_entropy_change",
]


def score_features(model, df: pd.DataFrame) -> np.ndarray:
    """Run the trained classifier on a feature DataFrame."""
    X = df[FEATURES].values
    return model.predict_proba(X)[:, 1]


def rank_identifiers(df: pd.DataFrame, probs: np.ndarray, top_k: int = 10) -> pd.DataFrame:
    out = df.copy()
    out["smell_prob"] = probs
    sort_cols = ["smell_prob"] + (["n_occurrences"] if "n_occurrences" in out.columns else [])
    out = out.sort_values(sort_cols, ascending=False).reset_index(drop=True)
    out["rank"] = range(1, len(out) + 1)
    cols = ["rank", "identifier_name", "smell_prob"]
    if "is_smell_token" in out.columns:
        cols.append("is_smell_token")
    if "n_occurrences" in out.columns:
        cols.append("n_occurrences")
    cols += FEATURES
    return out[cols].head(top_k)


def localization_summary(df: pd.DataFrame, probs: np.ndarray, ks=(1, 3, 5, 10)) -> dict:
    """If the CSV has labels, summarize per-query MRR / R@K."""
    if "is_smell_token" not in df.columns:
        return {}
    df = df.copy()
    df["_p"] = probs
    smell = df[df.get("version", "smell") == "smell"]
    key_cols = ["sample_id"] + (["run_id"] if "run_id" in smell.columns else [])
    has_smell = smell.groupby(key_cols)["is_smell_token"].any()
    queries = has_smell[has_smell].index.tolist()
    if not queries:
        return {"n_queries": 0}

    rrs, hits = [], {k: 0 for k in ks}
    sizes = []
    for q in queries:
        q = q if isinstance(q, tuple) else (q,)
        mask = pd.Series(True, index=smell.index)
        for col, val in zip(key_cols, q):
            mask &= (smell[col] == val)
        sub = smell.loc[mask].sort_values("_p", ascending=False).reset_index(drop=True)
        sizes.append(len(sub))
        hit_idx = sub.index[sub["is_smell_token"]].tolist()
        if hit_idx:
            r = hit_idx[0] + 1
            rrs.append(1.0 / r)
            for k in ks:
                if r <= k:
                    hits[k] += 1
        else:
            rrs.append(0.0)
    return {
        "n_queries": len(queries),
        "MRR": float(np.mean(rrs)),
        "avg_candidates_per_query": float(np.mean(sizes)),
        **{f"R@{k}": hits[k] / len(queries) for k in ks},
    }


# ── GPU end-to-end path ──────────────────────────────────────────────────────

def run_on_java_file(java_path: Path, model_pipe, dllm_name: str, total_steps: int = 64,
                     smell_inject: str | None = None):
    """Load DiffuCoder, probe the source, score, return ranked identifiers.

    If `smell_inject` is provided AND the file contains `[MASK]`, the mask is
    replaced by that token before probing — simulating an injection.
    """
    from detector.ml_layer.generate_features import (
        MODEL_REGISTRY, run_probes, aggregate_per_identifier, _MockMod,  # noqa: F401
    )
    import torch
    from transformers import AutoTokenizer, AutoModel

    code = java_path.read_text()
    if smell_inject and "[MASK]" in code:
        code = code.replace("[MASK]", smell_inject)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_id, mask_tok = MODEL_REGISTRY[dllm_name]
    print(f"loading {model_id} on {device} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    dllm = AutoModel.from_pretrained(
        model_id, trust_remote_code=True,
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
    ).to(device).eval()
    mask_token_id = tokenizer.convert_tokens_to_ids(mask_tok)

    result = run_probes(dllm, tokenizer, mask_token_id, code, total_steps, device)
    if result is None:
        print("no identifiers found in source", file=sys.stderr)
        return None
    id_groups, flip_step, first_conf, ent = result
    rows = aggregate_per_identifier(id_groups, flip_step, first_conf, ent,
                                    total_steps, smell_inject)
    feat_df = pd.DataFrame(rows)
    probs = score_features(model_pipe, feat_df)
    return rank_identifiers(feat_df, probs, top_k=20)


# ── CSV mode ─────────────────────────────────────────────────────────────────

def run_on_csv(features_path: Path, model_pipe, sample_id: int | None = None,
               run_id: int | None = None, top_k: int = 20):
    df = pd.read_csv(features_path)
    if sample_id is not None:
        df = df[df["sample_id"] == sample_id]
    if run_id is not None and "run_id" in df.columns:
        df = df[df["run_id"] == run_id]
    if df.empty:
        print("no rows match the requested filter", file=sys.stderr)
        return None
    probs = score_features(model_pipe, df)
    return df, probs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", required=True, help="trained joblib pipeline")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--java-file", help="path to a .java source file (GPU mode)")
    src.add_argument("--features",  help="path to a feature CSV (CSV mode)")

    p.add_argument("--dllm", default="DiffuCoder-7B-Instruct",
                   help="(GPU mode) which dLLM to probe with")
    p.add_argument("--smell-inject", default=None,
                   help="(GPU mode) replace [MASK] with this name before probing")
    p.add_argument("--total-steps", type=int, default=64,
                   help="(GPU mode) diffusion steps")

    p.add_argument("--sample-id", type=int, default=None,
                   help="(CSV mode) only score this sample_id")
    p.add_argument("--run-id", type=int, default=None,
                   help="(CSV mode) only score this run_id")
    p.add_argument("--top-k", type=int, default=20)
    args = p.parse_args()

    print(f"loading classifier: {args.model_path}")
    model_pipe = joblib.load(args.model_path)

    if args.java_file:
        ranked = run_on_java_file(Path(args.java_file), model_pipe, args.dllm,
                                  args.total_steps, args.smell_inject)
        if ranked is None:
            return
        print("\n── Top identifiers by smell probability ──")
        print(ranked.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    else:
        result = run_on_csv(Path(args.features), model_pipe,
                            args.sample_id, args.run_id, args.top_k)
        if result is None:
            return
        df, probs = result
        ranked = rank_identifiers(df, probs, top_k=args.top_k)
        print("\n── Top identifiers by smell probability ──")
        print(ranked.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

        summary = localization_summary(df, probs)
        if summary:
            print("\n── Localization metrics on labeled queries ──")
            for k, v in summary.items():
                print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")


if __name__ == "__main__":
    main()
