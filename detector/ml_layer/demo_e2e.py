"""
End-to-end smell-localization demo on the RefineID dataset.

What this script does (no GPU required for the CSV path):

  1. Pick a sample from `data/test.csv`.
  2. Inject a `smell_name` at every `[MASK]` position.
  3. Look up the precomputed dLLM probe features for this sample (or run
     them live in `--gpu` mode).
  4. Score every identifier with the trained ML head.
  5. Print the top-K ranked identifiers and check whether the injected
     smell name is at rank 1.

Two modes:

  CSV (no GPU):
    python detector/ml_layer/demo_e2e.py \\
        --features /tmp/legacy_features.csv \\
        --model-path detector/ml_layer/runs/<ts>/model_gbdt.joblib \\
        --sample-id 13

  GPU end-to-end:
    python detector/ml_layer/demo_e2e.py \\
        --gpu \\
        --model-path detector/ml_layer/runs/<ts>/model_gbdt.joblib \\
        --sample-id 13 --smell-name tmp --dllm DiffuCoder-7B-Instruct
"""

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

FEATURES = ["avg_flip_step", "first_confident_step",
            "mean_entropy_change", "max_entropy_change"]


def show_code_preview(masked_code: str, smell_name: str, max_lines: int = 12) -> str:
    """Compact preview: the lines that contain a `[MASK]` injection point."""
    code = masked_code.replace("[MASK]", f"⟨{smell_name}⟩")
    lines = code.splitlines()
    hit_lines = [i for i, l in enumerate(lines) if f"⟨{smell_name}⟩" in l]
    if not hit_lines:
        return "\n".join(lines[:max_lines])
    pieces = []
    for ln in hit_lines[:5]:
        s, e = max(0, ln - 1), min(len(lines), ln + 2)
        chunk = "\n".join(f"  L{i+1:3d}: {lines[i]}" for i in range(s, e))
        pieces.append(chunk)
    return "\n  …\n".join(pieces)


def load_refineid(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, header=None, names=["id", "masked_code", "target"])


def csv_path(args, model_pipe):
    feat = pd.read_csv(args.features)

    if args.honest:
        # Re-fit a fresh classifier excluding the queried sample so the score
        # is truly out-of-sample. Uses the same hyperparams as model_pipe.
        from sklearn.base import clone
        train_df = feat[feat["sample_id"] != args.sample_id]
        model_pipe = clone(model_pipe)
        model_pipe.fit(train_df[FEATURES].values,
                       train_df["is_smell_token"].astype(int).values)
        print(f"  [honest] re-fit classifier on {len(train_df):,} rows "
              f"(excluding sample_id={args.sample_id})")

    sub = feat[feat["sample_id"] == args.sample_id]
    if sub.empty:
        sys.exit(f"sample_id={args.sample_id} not found in {args.features}")
    if "run_id" in sub.columns and args.run_id is not None:
        sub = sub[sub["run_id"] == args.run_id]
    sub = sub[sub.get("version", "smell") == "smell"].reset_index(drop=True)

    probs = model_pipe.predict_proba(sub[FEATURES].values)[:, 1]
    sub = sub.assign(smell_prob=probs)
    return sub.sort_values("smell_prob", ascending=False).reset_index(drop=True)


def gpu_path(args, model_pipe):
    refine = load_refineid(Path(args.refineid))
    row = refine[refine["id"] == args.sample_id]
    if row.empty:
        sys.exit(f"sample_id={args.sample_id} not in RefineID")
    masked_code = str(row.iloc[0]["masked_code"])
    target = str(row.iloc[0]["target"]).strip()
    if "[MASK]" not in masked_code:
        sys.exit("this sample has no [MASK] to inject")
    smell_name = args.smell_name or "tmp"
    if smell_name == target:
        sys.exit(f"smell name '{smell_name}' equals the GT target — pick a different one")
    code = masked_code.replace("[MASK]", smell_name)

    print(f"\nInjecting '{smell_name}' (GT target was '{target}')\n")
    print("Code preview at injection sites:")
    print(show_code_preview(masked_code, smell_name))
    print()

    from detector.ml_layer.generate_features import (
        MODEL_REGISTRY, run_probes, aggregate_per_identifier,
    )
    import torch
    from transformers import AutoTokenizer, AutoModel

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_id, mask_tok = MODEL_REGISTRY[args.dllm]
    print(f"loading {model_id} on {device} (this is the slow part) ...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    dllm = AutoModel.from_pretrained(
        model_id, trust_remote_code=True,
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
    ).to(device).eval()
    mask_token_id = tokenizer.convert_tokens_to_ids(mask_tok)

    print(f"running {args.total_steps}-step probe ...")
    result = run_probes(dllm, tokenizer, mask_token_id, code, args.total_steps, device)
    if result is None:
        sys.exit("no identifiers found (parser failure?)")
    id_groups, flip_step, first_conf, ent = result
    rows = aggregate_per_identifier(id_groups, flip_step, first_conf, ent,
                                    args.total_steps, smell_name)
    feat_df = pd.DataFrame(rows)
    probs = model_pipe.predict_proba(feat_df[FEATURES].values)[:, 1]
    feat_df = feat_df.assign(smell_prob=probs)
    return feat_df.sort_values("smell_prob", ascending=False).reset_index(drop=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", required=True,
                   help="trained joblib pipeline from train_smell_classifier.py")
    p.add_argument("--sample-id", type=int, required=True)
    p.add_argument("--top-k", type=int, default=15)

    p.add_argument("--gpu", action="store_true",
                   help="run dLLM inference live (otherwise use --features CSV)")
    p.add_argument("--honest", action="store_true",
                   help="(CSV mode) re-fit the classifier excluding the queried "
                        "sample_id — gives a leakage-free localization score")
    # CSV mode
    p.add_argument("--features", default=None,
                   help="(CSV mode) feature CSV from generate_features.py")
    p.add_argument("--run-id", type=int, default=None,
                   help="(CSV mode) optional run_id filter")
    # GPU mode
    p.add_argument("--refineid", default="data/test.csv")
    p.add_argument("--smell-name", default=None,
                   help="(GPU mode) name to inject; default 'tmp'")
    p.add_argument("--dllm", default="DiffuCoder-7B-Instruct")
    p.add_argument("--total-steps", type=int, default=64)
    args = p.parse_args()

    if not args.gpu and not args.features:
        sys.exit("provide --features <csv> or use --gpu")

    print(f"loading classifier: {args.model_path}")
    model_pipe = joblib.load(args.model_path)

    ranked = (gpu_path(args, model_pipe) if args.gpu
              else csv_path(args, model_pipe))

    out_cols = ["identifier_name", "smell_prob"]
    if "is_smell_token" in ranked.columns:
        out_cols.append("is_smell_token")
    if "n_occurrences" in ranked.columns:
        out_cols.append("n_occurrences")
    if "injected_name" in ranked.columns:
        out_cols.append("injected_name")

    print(f"\n── Top-{args.top_k} identifiers ranked by smell probability ──")
    head = ranked.head(args.top_k).copy()
    head.insert(0, "rank", range(1, len(head) + 1))
    print(head[["rank"] + out_cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    if "is_smell_token" in ranked.columns:
        smell_ranks = ranked.index[ranked["is_smell_token"]].tolist()
        if smell_ranks:
            r1 = smell_ranks[0] + 1
            print(f"\n  → first true-smell hit at rank {r1}  (MRR contribution = {1/r1:.4f})")
            print(f"  → smell positions in top-{args.top_k}: "
                  f"{sum(1 for r in smell_ranks if r < args.top_k)}/{len(smell_ranks)}")
        else:
            print("\n  → no smell label found in this sample (CSV may be unlabeled)")


if __name__ == "__main__":
    main()
