"""
Convert DreamOn benchmark CSVs into the column format expected by
``experiments/llm_judge_variable_naming.py``: ``id, ground_truth,
prediction, correct``.

DreamOn has two result files:
    OLD (single-window, first-mask only):
        results/dreamon/DreamOn-7B_refineID_20260509_113722.csv
        columns already include id/ground_truth/prediction/correct.
        We just keep those four and write a clean copy.

    NEW (tiled, all-site coverage):
        results/dreamon/DreamOn-7B_per_sample_20260509_134411.csv
        columns: id, ground_truth, n_total_masks, n_windows, predictions,
                 majority_pred, majority_count, majority_correct, any_correct
        ``predictions`` is pipe-separated per-site predictions.

For the NEW file we emit TWO baseline-format CSVs so the judge can be
run on each:
    * ``DreamOn-7B_tiled_first_mask.csv`` -- prediction = first site
      (baseline-equivalent: same rule the other models report under).
    * ``DreamOn-7B_tiled_majority.csv``  -- prediction = majority vote
      across sites (DreamOn's own most-confident answer).

Outputs go to ``data/benchmark_ReFineID_DreamOn/`` so the existing
``--input`` flag of the judge script can point at them directly.

Usage:
    python experiments/prepare_dreamon_for_judge.py
"""

import os
import csv
import sys
from collections import Counter

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(REPO, "results", "dreamon")
DST_DIR = os.path.join(REPO, "data", "benchmark_ReFineID_DreamOn")

OLD_FILE = "DreamOn-7B_refineID_20260509_113722.csv"
NEW_FILE = "DreamOn-7B_per_sample_20260509_134411.csv"

os.makedirs(DST_DIR, exist_ok=True)
csv.field_size_limit(sys.maxsize)


def write_baseline_csv(path, rows):
    """Write rows in (id, ground_truth, prediction, correct) order."""
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["id", "ground_truth",
                                          "prediction", "correct"])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"  wrote {path}  ({len(rows)} rows)")


def convert_old():
    src = os.path.join(SRC_DIR, OLD_FILE)
    dst = os.path.join(DST_DIR, "DreamOn-7B_single_window.csv")
    if not os.path.exists(src):
        print(f"SKIP (missing): {src}")
        return None
    rows_out = []
    with open(src, "r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            gt = (r.get("ground_truth") or "").strip()
            pred = (r.get("prediction") or "").strip()
            rows_out.append({
                "id": r["id"],
                "ground_truth": gt,
                "prediction": pred,
                "correct": pred == gt,
            })
    write_baseline_csv(dst, rows_out)
    return dst


def convert_new():
    src = os.path.join(SRC_DIR, NEW_FILE)
    if not os.path.exists(src):
        print(f"SKIP (missing): {src}")
        return None, None

    first_rows = []
    maj_rows = []
    with open(src, "r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            gt = (r.get("ground_truth") or "").strip()
            preds_raw = (r.get("predictions") or "")
            preds = [p.strip() for p in preds_raw.split("|") if p.strip()]

            # ---- first-site prediction (baseline-equivalent) ----
            first_pred = preds[0] if preds else ""
            first_rows.append({
                "id": r["id"],
                "ground_truth": gt,
                "prediction": first_pred,
                "correct": first_pred == gt,
            })

            # ---- majority-vote prediction ----
            if preds:
                maj_pred, _ = Counter(preds).most_common(1)[0]
            else:
                maj_pred = ""
            maj_rows.append({
                "id": r["id"],
                "ground_truth": gt,
                "prediction": maj_pred,
                "correct": maj_pred == gt,
            })

    dst_first = os.path.join(DST_DIR, "DreamOn-7B_tiled_first_mask.csv")
    dst_maj = os.path.join(DST_DIR, "DreamOn-7B_tiled_majority.csv")
    write_baseline_csv(dst_first, first_rows)
    write_baseline_csv(dst_maj, maj_rows)
    return dst_first, dst_maj


if __name__ == "__main__":
    print(f"Source : {SRC_DIR}")
    print(f"Dest   : {DST_DIR}\n")
    print("Converting OLD (single-window):")
    old_out = convert_old()
    print("\nConverting NEW (tiled):")
    new_first, new_maj = convert_new()
    print("\nDone.")
    print("\nNext step: run the judge on each file, e.g.")
    for p in [old_out, new_first, new_maj]:
        if p:
            rel = os.path.relpath(p, REPO)
            print(f"  python experiments/llm_judge_variable_naming.py "
                  f"--input {rel} --judge-model Qwen2.5-7B-Instruct")
