"""Run the multi-identifier / multi-site rename benchmark for one or more models.

Reads the JSONL produced by build_dataset.py, runs each requested model, scores
every instance, and writes:
  * results/<model>_predictions.jsonl  -- per-instance preds + per-identifier scores
  * results/<model>_summary.json       -- aggregate metrics
  * results/summary_all.csv            -- one row per model (appended/rewritten)

Example:
    python experiments/multi_site_rename/run_benchmark.py \
        --data   experiments/multi_site_rename/data/multisite_rename.jsonl \
        --models deepseek-1.3b-instruct qwen2.5-1.5b-instruct \
        --out-dir experiments/multi_site_rename/results --limit 100
"""
import argparse
import json
import os

import metrics as M
from models import MODELS, ModelRunner, list_models

HERE = os.path.dirname(os.path.abspath(__file__))


def load_dataset(path, limit=None):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
            if limit and len(rows) >= limit:
                break
    return rows


def run_model(key, data, out_dir, max_new_tokens):
    try:
        from tqdm import tqdm
    except ImportError:
        def tqdm(x, **k):
            return x

    runner = ModelRunner(key, max_new_tokens=max_new_tokens)
    inst_scores, pred_path = [], os.path.join(out_dir, f"{key}_predictions.jsonl")
    with open(pred_path, "w") as pf:
        for rec in tqdm(data, desc=key):
            try:
                pred, raw = runner.predict(rec)
            except Exception as e:
                pred, raw = {}, f"ERROR: {e}"
            score = M.score_instance(rec["ground_truth"], pred)
            inst_scores.append(score)
            pf.write(json.dumps({
                "id": rec["id"],
                "source_name": rec.get("source_name", ""),
                "repo": rec.get("repo", ""),
                "num_identifiers": rec["num_identifiers"],
                "total_sites": rec.get("total_sites"),
                "ground_truth": rec["ground_truth"],
                "prediction": pred,
                "raw": raw[:2000],
                "score": {k: v for k, v in score.items() if k != "per_identifier"},
                "per_identifier": score["per_identifier"],
            }) + "\n")
    runner.close()

    summary = M.aggregate(inst_scores)
    summary["model"] = key
    summary["model_id"] = MODELS[key]["id"]
    summary["protocol"] = MODELS[key]["protocol"]
    with open(os.path.join(out_dir, f"{key}_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=os.path.join(HERE, "data", "multisite_rename.jsonl"))
    ap.add_argument("--models", nargs="+", default=["deepseek-1.3b-instruct",
                                                    "qwen2.5-1.5b-instruct"],
                    help=f"keys from models.py. Available: {list_models()}")
    ap.add_argument("--out-dir", default=os.path.join(HERE, "results"))
    ap.add_argument("--limit", type=int, default=None, help="cap #instances (debug)")
    ap.add_argument("--max-new-tokens", type=int, default=160)
    args = ap.parse_args()

    if not os.path.exists(args.data):
        raise SystemExit(f"dataset not found: {args.data}\nRun build_dataset.py first.")
    os.makedirs(args.out_dir, exist_ok=True)

    data = load_dataset(args.data, args.limit)
    print(f"Loaded {len(data)} instance(s) from {args.data}")
    if not data:
        raise SystemExit("empty dataset")

    summaries = []
    for key in args.models:
        print(f"\n{'=' * 60}\n{key}\n{'=' * 60}")
        summaries.append(run_model(key, data, args.out_dir, args.max_new_tokens))

    # combined CSV + console table
    cols = ["model", "protocol", "n_instances", "n_identifiers", "em", "em_ci",
            "joint_acc", "edit_sim", "subtoken_f1", "coverage"]
    csv_path = os.path.join(args.out_dir, "summary_all.csv")
    with open(csv_path, "w") as f:
        f.write(",".join(cols) + "\n")
        for s in summaries:
            f.write(",".join(str(s.get(c, "")) for c in cols) + "\n")

    print(f"\n{'=' * 60}\nSUMMARY  (written to {csv_path})\n{'=' * 60}")
    hdr = f"{'model':24} {'EM':>7} {'EM-ci':>7} {'joint':>7} {'editSim':>8} {'subF1':>7}"
    print(hdr)
    print("-" * len(hdr))
    for s in summaries:
        print(f"{s['model']:24} {s['em']:7.3f} {s['em_ci']:7.3f} "
              f"{s['joint_acc']:7.3f} {s['edit_sim']:8.3f} {s['subtoken_f1']:7.3f}")


if __name__ == "__main__":
    main()
