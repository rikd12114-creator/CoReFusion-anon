"""
Pure-Python (no bash) LLM-as-Judge runner for the AISE DreamOn Java-Identifier
models. Colab-friendly: just `python experiments/run_llm_judge_dreamon_java.py`.

Pipeline (all outputs stay UNDER results/ so a single zip of results/ captures
everything for the Drive upload):
    results/dreamon_java/*_per_sample_*.csv         (EM benchmark output)
        -> results/dreamon_java/judge_inputs/<label>_{first_mask,majority}.csv
        -> results/dreamon_java/llm_judge/<...>__judge_<model>__<ts>.csv

Consistency with the previous LJ: this script does NOT reimplement the judge.
It imports and calls the exact same functions from
``experiments/llm_judge_variable_naming.py`` (same SYSTEM_PROMPT, same chat
template, same Qwen2.5-7B-Instruct default, same --resume, same exact-match
shortcut). The only difference vs run_llm_judge_dreamon.sh is orchestration in
Python and the judge model is loaded ONCE and reused across all variants.

Usage:
    # run the EM benchmark first so per_sample CSVs exist:
    python experiments/benchmark_dreamon_java_identifiers.py
    # then judge:
    python experiments/run_llm_judge_dreamon_java.py
    python experiments/run_llm_judge_dreamon_java.py --judge-model Qwen2.5-14B-Instruct
    python experiments/run_llm_judge_dreamon_java.py --max-samples 50   # quick test
"""

import os
import sys
import csv
import glob
import argparse
from collections import Counter
from datetime import datetime

# Make the sibling judge module importable regardless of cwd.
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
import llm_judge_variable_naming as judge   # reuse the exact judge logic

REPO = os.path.dirname(_HERE)
EM_DIR = os.path.join(REPO, "results", "dreamon_java")
JUDGE_INPUT_DIR = os.path.join(EM_DIR, "judge_inputs")
JUDGE_OUT_DIR = os.path.join(EM_DIR, "llm_judge")
DATA_PATH = os.path.join(REPO, "data", "test.csv")


# ---- Step 1: per_sample CSVs -> baseline-format judge inputs ----------------

def build_judge_inputs(em_dir=EM_DIR):
    """For each model's latest *_per_sample_*.csv emit two baseline CSVs
    (id, ground_truth, prediction, correct): first-mask and majority-vote."""
    os.makedirs(JUDGE_INPUT_DIR, exist_ok=True)
    csv.field_size_limit(sys.maxsize)

    by_label = {}
    for f in glob.glob(os.path.join(em_dir, "*_per_sample_*.csv")):
        label = os.path.basename(f).split("_per_sample_")[0]
        by_label.setdefault(label, []).append(f)

    produced = []
    for label, files in by_label.items():
        src = sorted(files)[-1]                      # latest run for this label
        first_rows, maj_rows = [], []
        with open(src, "r", encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                gt = (r.get("ground_truth") or "").strip()
                preds = [p.strip() for p in (r.get("predictions") or "").split("|") if p.strip()]
                fp = preds[0] if preds else ""
                first_rows.append({"id": r["id"], "ground_truth": gt,
                                   "prediction": fp, "correct": fp == gt})
                mp = Counter(preds).most_common(1)[0][0] if preds else ""
                maj_rows.append({"id": r["id"], "ground_truth": gt,
                                 "prediction": mp, "correct": mp == gt})
        for variant, rows in [("first_mask", first_rows), ("majority", maj_rows)]:
            out = os.path.join(JUDGE_INPUT_DIR, f"{label}_{variant}.csv")
            with open(out, "w", newline="", encoding="utf-8") as fh:
                w = csv.DictWriter(fh, fieldnames=["id", "ground_truth", "prediction", "correct"])
                w.writeheader()
                w.writerows(rows)
            produced.append(out)
            print(f"  wrote {os.path.relpath(out, REPO)}  ({len(rows)} rows)")
    return produced


# ---- Step 2: judge each variant with one shared judge model -----------------

def main():
    p = argparse.ArgumentParser(description="No-bash LLM-as-Judge for DreamOn-Java EM outputs.")
    p.add_argument("--judge-model", default=judge.DEFAULT_JUDGE,
                   help=f"Registry name ({', '.join(judge.JUDGE_REGISTRY)}) or full HF id.")
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--no-resume", action="store_true",
                   help="Disable resume (default: resume like run_llm_judge_dreamon.sh).")
    p.add_argument("--em-dir", default=EM_DIR)
    args = p.parse_args()

    print("=" * 70)
    print("  Step 1: per_sample -> baseline judge inputs")
    print("=" * 70)
    inputs = build_judge_inputs(args.em_dir)
    if not inputs:
        sys.exit(f"No *_per_sample_*.csv in {args.em_dir}. Run the EM benchmark first.")

    # Resolve judge id/name exactly like llm_judge_variable_naming.main().
    if args.judge_model in judge.JUDGE_REGISTRY:
        model_id, judge_name = judge.JUDGE_REGISTRY[args.judge_model], args.judge_model
    else:
        model_id, judge_name = args.judge_model, args.judge_model.split("/")[-1]

    os.makedirs(JUDGE_OUT_DIR, exist_ok=True)
    print("\n" + "=" * 70)
    print(f"  Step 2: LLM-as-Judge with {model_id}")
    print(f"  device={judge.DEVICE}  results -> {os.path.relpath(JUDGE_OUT_DIR, REPO)}")
    print("=" * 70)

    print("\nLoading test.csv for code context ...")
    test_data = judge.load_test_data(DATA_PATH)
    print(f"  {len(test_data)} samples loaded.")

    tokenizer, model = judge.load_judge(model_id)   # loaded ONCE, reused below

    summaries = []
    for path in sorted(inputs):
        print(f"\n{'-'*70}\n  Judging: {os.path.relpath(path, REPO)}\n{'-'*70}")
        summary = judge.evaluate_file(
            benchmark_type="dreamon_java",
            benchmark_path=path,
            test_data=test_data,
            tokenizer=tokenizer,
            model=model,
            results_dir=JUDGE_OUT_DIR,
            judge_name=judge_name,
            max_samples=args.max_samples,
            resume=not args.no_resume,
        )
        summaries.append(summary)
        print(f"  EM acc={summary['exact_match_acc']}  LLM-judge acc={summary['llm_judge_acc']}"
              f"  ({summary['verdict_1_count']}/{summary['total_judged']})")

    # Aggregate summary under results/ so it gets zipped too.
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_path = os.path.join(JUDGE_OUT_DIR, f"summary_judge_{judge_name}_{ts}.csv")
    if summaries:
        with open(summary_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(summaries[0].keys()))
            w.writeheader()
            w.writerows(summaries)

    print(f"\n{'='*70}\n  JUDGE SUMMARY ({model_id})\n{'='*70}")
    print(f"  {'File':<48}{'EM':>8}{'LLM':>8}")
    for s in summaries:
        print(f"  {s['benchmark_file'][:48]:<48}{s['exact_match_acc']:>8}{s['llm_judge_acc']:>8}")
    print(f"\n  Saved: {os.path.relpath(summary_path, REPO)}")
    print(f"  All outputs under results/dreamon_java/  (EM + judge_inputs + llm_judge)")

    del model, tokenizer
    try:
        import torch, gc
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
    except Exception:
        pass


if __name__ == "__main__":
    main()
