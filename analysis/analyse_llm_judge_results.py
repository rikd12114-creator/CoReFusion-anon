"""
Analyse LLM Judge Results
=========================

Reads all judge output CSVs from results/llm_judge/ and produces:
  1. A summary table comparing Exact Match accuracy vs LLM Judge accuracy
     per model.
  2. A CSV with per-sample verdict analysis (useful for error analysis).
  3. Prints examples where the judge overrides the exact-match decision
     (i.e. exact_match=False but llm_verdict=1 → model was actually right).

Usage (from project root):
    python analysis/analyse_llm_judge_results.py
    python analysis/analyse_llm_judge_results.py --results-dir results/llm_judge
    python analysis/analyse_llm_judge_results.py --show-overrides 20
"""

import os
import csv
import sys
import re
import argparse
from pathlib import Path
from collections import defaultdict

import pandas as pd


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DEFAULT_RESULTS_DIR = "results/llm_judge"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def model_name_from_filename(fname: str) -> str:
    """Extract the model name from a judge output filename.

    Filename pattern:
        {benchmark_stem}__judge_{judge_model}__{timestamp}.csv
    We want the benchmark model name (the part before '__judge_').
    """
    stem = Path(fname).stem
    parts = stem.split("__judge_")
    if len(parts) >= 1:
        bench_part = parts[0]
        # e.g. "DiffuCoder-7B_refineID_diffusion_20260226_125550"
        # Strip timestamp suffix (last _YYYYMMDD_HHMMSS)
        bench_part = re.sub(r"_\d{8}_\d{6}$", "", bench_part)
        # Strip task suffix
        bench_part = re.sub(r"_refineID_(diffusion|fim)$", "", bench_part)
        return bench_part
    return stem


def load_judge_csv(path: str) -> pd.DataFrame:
    """Load a judge output CSV, handling large fields."""
    csv.field_size_limit(sys.maxsize)
    return pd.read_csv(path, dtype={"llm_verdict": str})


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def analyse(results_dir: str, show_overrides: int) -> None:
    judge_files = sorted(Path(results_dir).glob("*.csv"))
    # Exclude summary files
    judge_files = [f for f in judge_files if not f.name.startswith("summary_")]

    if not judge_files:
        print(f"No judge result CSVs found in {results_dir}")
        return

    print(f"Found {len(judge_files)} judge result file(s).\n")

    rows = []
    override_examples = []  # (model, gt, pred) where EM=False but judge=1

    for fpath in judge_files:
        model_name = model_name_from_filename(fpath.name)
        try:
            df = load_judge_csv(str(fpath))
        except Exception as e:
            print(f"  WARNING: could not load {fpath.name}: {e}")
            continue

        # Normalise verdict column
        df["verdict"] = pd.to_numeric(df["llm_verdict"], errors="coerce").fillna(0).astype(int)
        df["exact_match"] = df["exact_match"].astype(str).str.lower().isin(["true", "1"])

        total         = len(df)
        em_correct    = df["exact_match"].sum()
        llm_correct   = (df["verdict"] >= 1).sum()
        parse_fail    = (df["verdict"] == -1).sum()

        em_acc  = em_correct  / total if total else 0.0
        llm_acc = llm_correct / total if total else 0.0
        delta   = llm_acc - em_acc

        rows.append({
            "model":          model_name,
            "file":           fpath.name,
            "total":          total,
            "em_correct":     int(em_correct),
            "em_acc":         em_acc,
            "llm_correct":    int(llm_correct),
            "llm_acc":        llm_acc,
            "delta":          delta,
            "parse_failures": int(parse_fail),
        })

        # Collect override examples: EM=False but judge=1
        overrides = df[(~df["exact_match"]) & (df["verdict"] == 1)]
        for _, r in overrides.iterrows():
            override_examples.append({
                "model":        model_name,
                "id":           r.get("id", "?"),
                "ground_truth": r.get("ground_truth", ""),
                "prediction":   r.get("prediction",   ""),
                "llm_raw":      str(r.get("llm_raw_output", ""))[:80],
            })

    if not rows:
        print("No valid results to summarise.")
        return

    summary_df = pd.DataFrame(rows).sort_values("llm_acc", ascending=False)

    # ---------- Print summary table -----------------------------------------
    print("=" * 80)
    print("  VARIABLE RENAMING  —  LLM Judge vs Exact Match")
    print("=" * 80)
    print(
        f"  {'Model':<35} {'Total':>6}  {'EM Acc':>7}  {'LLM Acc':>8}  "
        f"{'Δ (LLM-EM)':>10}  {'Parse ✗':>7}"
    )
    print(f"  {'-'*78}")
    for _, r in summary_df.iterrows():
        delta_str = f"{r['delta']:+.4f}"
        print(
            f"  {r['model']:<35} {r['total']:>6}  "
            f"{r['em_acc']:>7.4f}  {r['llm_acc']:>8.4f}  "
            f"{delta_str:>10}  {r['parse_failures']:>7}"
        )

    overall_em  = summary_df["em_correct"].sum()  / summary_df["total"].sum()
    overall_llm = summary_df["llm_correct"].sum() / summary_df["total"].sum()
    print(f"  {'-'*78}")
    print(
        f"  {'OVERALL':<35} {summary_df['total'].sum():>6}  "
        f"{overall_em:>7.4f}  {overall_llm:>8.4f}  "
        f"{overall_llm - overall_em:>+10.4f}"
    )

    # ---------- Save summary CSV --------------------------------------------
    out_path = os.path.join(results_dir, "analysis_summary.csv")
    summary_df.to_csv(out_path, index=False, float_format="%.4f")
    print(f"\n  Summary saved to: {out_path}")

    # ---------- Show overrides ----------------------------------------------
    if override_examples and show_overrides > 0:
        print(f"\n{'='*80}")
        print(
            f"  TOP {min(show_overrides, len(override_examples))} JUDGE-OVERRIDES "
            f"(EM=False but LLM judged ACCEPTABLE)"
        )
        print(f"  These are cases where the model proposed a different-but-valid name.")
        print(f"{'='*80}")
        for ex in override_examples[:show_overrides]:
            print(
                f"  Model: {ex['model']}\n"
                f"  ID:    {ex['id']}\n"
                f"  GT:    {ex['ground_truth']}\n"
                f"  Pred:  {ex['prediction']}\n"
                f"  Judge: {ex['llm_raw']}\n"
            )

    # ---------- Save override examples --------------------------------------
    override_path = os.path.join(results_dir, "analysis_overrides.csv")
    if override_examples:
        keys = ["model", "id", "ground_truth", "prediction", "llm_raw"]
        with open(override_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(override_examples)
        print(f"\n  Override examples saved to: {override_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Analyse LLM judge results for variable renaming evaluation."
    )
    parser.add_argument(
        "--results-dir", type=str, default=DEFAULT_RESULTS_DIR,
        help=f"Directory with judge output CSVs. Default: {DEFAULT_RESULTS_DIR}"
    )
    parser.add_argument(
        "--show-overrides", type=int, default=10,
        help="Number of override examples to print (EM=False but judge=1). Default: 10"
    )
    args = parser.parse_args()
    analyse(args.results_dir, args.show_overrides)


if __name__ == "__main__":
    main()
