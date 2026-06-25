"""
Convert the AISE DreamOn Java-Identifier per_sample CSVs (produced by
``experiments/benchmark_dreamon_java_identifiers.py``) into the baseline
column format expected by ``experiments/llm_judge_variable_naming.py``:

    id, ground_truth, prediction, correct

For each model we emit TWO judge-ready CSVs (same convention as
prepare_dreamon_for_judge.py):

    <label>_first_mask.csv   prediction = first site  (baseline-equivalent)
    <label>_majority.csv     prediction = majority vote across sites

Outputs go to data/benchmark_ReFineID_DreamOn_Java/ so the judge's --input
flag can point straight at them.

Usage:
    # auto-pick the latest per_sample CSV for every label in results/dreamon_java
    python experiments/prepare_dreamon_java_for_judge.py
    # or point at specific files
    python experiments/prepare_dreamon_java_for_judge.py \
        results/dreamon_java/DreamOn-7B-Java_per_sample_20260607_120000.csv
"""

import os
import csv
import sys
import glob
from collections import Counter

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(REPO, "results", "dreamon_java")
DST_DIR = os.path.join(REPO, "data", "benchmark_ReFineID_DreamOn_Java")

os.makedirs(DST_DIR, exist_ok=True)
csv.field_size_limit(sys.maxsize)


def write_baseline_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["id", "ground_truth", "prediction", "correct"])
        w.writeheader()
        w.writerows(rows)
    print(f"  wrote {os.path.relpath(path, REPO)}  ({len(rows)} rows)")


def label_from_filename(path):
    base = os.path.basename(path)
    return base.split("_per_sample_")[0] if "_per_sample_" in base else base.rsplit(".", 1)[0]


def convert(src):
    label = label_from_filename(src)
    first_rows, maj_rows = [], []
    with open(src, "r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            gt = (r.get("ground_truth") or "").strip()
            preds = [p.strip() for p in (r.get("predictions") or "").split("|") if p.strip()]

            first_pred = preds[0] if preds else ""
            first_rows.append({"id": r["id"], "ground_truth": gt,
                               "prediction": first_pred, "correct": first_pred == gt})

            maj_pred = Counter(preds).most_common(1)[0][0] if preds else ""
            maj_rows.append({"id": r["id"], "ground_truth": gt,
                             "prediction": maj_pred, "correct": maj_pred == gt})

    out_first = os.path.join(DST_DIR, f"{label}_first_mask.csv")
    out_maj = os.path.join(DST_DIR, f"{label}_majority.csv")
    write_baseline_csv(out_first, first_rows)
    write_baseline_csv(out_maj, maj_rows)
    return [out_first, out_maj]


def latest_per_sample_files():
    files = glob.glob(os.path.join(SRC_DIR, "*_per_sample_*.csv"))
    latest = {}
    for f in files:
        latest.setdefault(label_from_filename(f), []).append(f)
    return [sorted(v)[-1] for v in latest.values()]


if __name__ == "__main__":
    srcs = sys.argv[1:] or latest_per_sample_files()
    if not srcs:
        sys.exit(f"No per_sample CSVs found in {SRC_DIR}. Run the EM benchmark first.")
    print(f"Source : {SRC_DIR}\nDest   : {DST_DIR}\n")
    produced = []
    for s in srcs:
        print(f"Converting {os.path.basename(s)}:")
        produced += convert(s)
    print("\nDone. Next: run the judge on each, e.g.")
    for p in produced:
        print(f"  python experiments/llm_judge_variable_naming.py "
              f"--input {os.path.relpath(p, REPO)} --judge-model Qwen2.5-7B-Instruct")
