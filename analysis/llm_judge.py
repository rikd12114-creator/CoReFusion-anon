"""
LLM-as-a-Judge for RefineID Benchmark Evaluation
================================================
This script is completely standalone — it has no dependencies on any other
project file. It uses Qwen/Qwen2.5-7B-Instruct as the judge model.

It scans the following two data directories for result CSVs:
  - data/benchmark_ReFineID_FIM/ar_fim_benchmark/
  - data/benchmark_ReFineID_Diffusion/diffusion_benchmark/

For each CSV, it samples failed Exact Match cases and asks the LLM judge
whether the model's prediction is semantically acceptable in context.
"""

import os
import sys
import glob
import argparse
import json
from datetime import datetime

import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


# ─────────────────────────── Judge Model ───────────────────────────────────

class QwenJudge:
    """Thin wrapper around Qwen2.5-7B-Instruct for judge inference."""

    def __init__(self, model_id: str = "Qwen/Qwen2.5-7B-Instruct"):
        self.model_id = model_id
        self.model = None
        self.tokenizer = None

    def load(self):
        print(f"Loading judge model: {self.model_id} ...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype="auto",
            device_map="auto",
        )
        print("Judge model loaded.\n")

    def generate(self, prompt: str, max_new_tokens: int = 200) -> str:
        messages = [
            {
                "role": "system",
                "content": (
                    "You are an expert Java programmer and impartial code reviewer. "
                    "You evaluate identifier substitutions in Java code rigorously."
                ),
            },
            {"role": "user", "content": prompt},
        ]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        generated_ids = self.model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False
        )
        # Trim the input tokens from the output
        output_ids = generated_ids[0][inputs.input_ids.shape[1]:]
        return self.tokenizer.decode(output_ids, skip_special_tokens=True)


# ─────────────────────────── Prompt Builder ────────────────────────────────

JUDGE_PROMPT_TEMPLATE = """\
You are evaluating a code identifier-infilling task (RefineID).

A specific identifier (variable, method, or type name) was **masked** in the original Java code, and a model predicted a replacement.

═══════════════════════════════════
GROUND TRUTH : "{ground_truth}"
MODEL PREDICTION : "{prediction}"
═══════════════════════════════════

FULL CODE CONTEXT (the masked location is where the identifier should appear):
```java
{full_code}
```

Evaluate the prediction by answering:
1. What is the relationship between the prediction and the ground truth in this exact context?
2. Would using the prediction instead of the ground truth preserve the program's correctness and readability?

Respond ONLY in this exact format (3 lines, no extra text):
Relationship: [Identical | Semantically Equivalent | Related but Different | Incorrect | Syntactically Invalid]
Explanation: [max 20 words]
Decision: [Accept | Reject]"""


def build_prompt(ground_truth: str, prediction: str, full_code: str) -> str:
    # Truncate very long code snippets to keep prompt manageable
    if len(full_code) > 3000:
        full_code = full_code[:3000] + "\n... [truncated]"
    return JUDGE_PROMPT_TEMPLATE.format(
        ground_truth=ground_truth,
        prediction=prediction,
        full_code=full_code,
    )


# ─────────────────────────── Output Parser ─────────────────────────────────

def parse_response(response: str) -> dict:
    """Extract Relationship, Explanation, Decision from model output."""
    result = {"relationship": "Unknown", "explanation": "", "decision": "Reject"}
    for line in response.strip().splitlines():
        line = line.strip()
        if line.lower().startswith("relationship:"):
            result["relationship"] = line.split(":", 1)[1].strip()
        elif line.lower().startswith("explanation:"):
            result["explanation"] = line.split(":", 1)[1].strip()
        elif line.lower().startswith("decision:"):
            result["decision"] = line.split(":", 1)[1].strip()
    return result


# ─────────────────────────── File Discovery ────────────────────────────────

def find_benchmark_csvs(data_root: str) -> list:
    """
    Discover all result CSVs in:
      data/benchmark_ReFineID_FIM/ar_fim_benchmark/
      data/benchmark_ReFineID_Diffusion/diffusion_benchmark/
    Excludes 'summary' files.
    """
    patterns = [
        os.path.join(data_root, "benchmark_ReFineID_FIM", "ar_fim_benchmark", "*.csv"),
        os.path.join(data_root, "benchmark_ReFineID_Diffusion", "diffusion_benchmark", "*.csv"),
    ]
    found = []
    for pat in patterns:
        for f in glob.glob(pat):
            if "summary" not in os.path.basename(f).lower():
                found.append(f)
    return sorted(found)


# ─────────────────────────── Per-File Judging ──────────────────────────────

def judge_file(
    file_path: str,
    judge: QwenJudge,
    sample_size: int,
    failed_only: bool,
) -> dict:
    """Run the judge on a single result CSV. Returns a summary dict."""
    df = pd.read_csv(file_path)

    # Validate required columns
    required = {"ground_truth", "prediction"}
    if not required.issubset(df.columns):
        print(f"  [SKIP] Missing columns in {os.path.basename(file_path)}")
        return None

    # Determine which rows to judge
    if failed_only and "correct" in df.columns:
        candidates = df[df["correct"] == False].copy()
    else:
        candidates = df.copy()

    if len(candidates) == 0:
        print(f"  [SKIP] No candidate rows in {os.path.basename(file_path)}")
        return None

    if sample_size > 0 and len(candidates) > sample_size:
        candidates = candidates.sample(n=sample_size, random_state=42)

    # Determine the code column
    code_col = next(
        (c for c in ["full_code", "code", "context"] if c in df.columns),
        None
    )

    model_name = os.path.basename(file_path).split("_refineID")[0].split("_fim")[0]
    rows_for_output = []
    accepted = 0

    for _, row in tqdm(candidates.iterrows(), total=len(candidates), desc=f"  {model_name}"):
        gt = str(row["ground_truth"]).strip()
        pred = str(row["prediction"]).strip()
        code = str(row[code_col]).strip() if code_col else "No context available"

        prompt = build_prompt(gt, pred, code)

        try:
            raw_response = judge.generate(prompt, max_new_tokens=200)
            parsed = parse_response(raw_response)
        except Exception as e:
            parsed = {
                "relationship": "Error",
                "explanation": str(e)[:80],
                "decision": "Reject",
            }
            raw_response = ""

        is_accepted = "accept" in parsed["decision"].lower()
        if is_accepted:
            accepted += 1

        rows_for_output.append({
            "id": row.get("id", "N/A"),
            "ground_truth": gt,
            "prediction": pred,
            "judge_relationship": parsed["relationship"],
            "judge_explanation": parsed["explanation"],
            "judge_decision": parsed["decision"],
            "judge_raw": raw_response,
        })

    total_judged = len(rows_for_output)
    judge_accept_rate = accepted / total_judged if total_judged > 0 else 0

    # Calculate corrected accuracy if 'correct' column exists
    corrected_em = None
    if "correct" in df.columns:
        em_rate = df["correct"].mean()
        # Estimate: failed cases * judge_accept_rate can be added back
        failed_rate = 1 - em_rate
        corrected_em = em_rate + failed_rate * judge_accept_rate

    return {
        "model": model_name,
        "file": os.path.basename(file_path),
        "total_rows": len(df),
        "judged": total_judged,
        "judge_accepted": accepted,
        "judge_accept_rate": judge_accept_rate,
        "em_rate": df["correct"].mean() if "correct" in df.columns else None,
        "corrected_em_estimate": corrected_em,
        "rows": rows_for_output,
    }


# ─────────────────────────── Main ──────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="LLM-as-a-Judge for RefineID benchmarks (standalone)"
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"),
        help="Root of the data directory (default: ../data relative to this script)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save results. Defaults to ../results/llm_judge/<timestamp>/",
    )
    parser.add_argument(
        "--model_id",
        type=str,
        default="Qwen/Qwen2.5-7B-Instruct",
        help="HuggingFace model ID for the judge (default: Qwen/Qwen2.5-7B-Instruct)",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=50,
        help="Number of rows to sample per file (default: 50, 0 = all rows)",
    )
    parser.add_argument(
        "--all_rows",
        action="store_true",
        help="Judge all rows, not just failed Exact Match cases",
    )
    parser.add_argument(
        "--files",
        nargs="*",
        help="Optionally specify explicit CSV file paths to judge instead of auto-discovery",
    )
    args = parser.parse_args()

    # Resolve output dir
    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = os.path.join(
            os.path.dirname(args.data_root), "results", "llm_judge", timestamp
        )
    os.makedirs(args.output_dir, exist_ok=True)

    # Discover CSV files
    if args.files:
        csv_files = args.files
    else:
        csv_files = find_benchmark_csvs(args.data_root)

    if not csv_files:
        print(f"No CSV files found under {args.data_root}. Check your data directory.")
        sys.exit(1)

    print(f"Found {len(csv_files)} CSV file(s) to judge:")
    for f in csv_files:
        print(f"  {os.path.relpath(f)}")
    print()

    # Load judge model
    judge = QwenJudge(model_id=args.model_id)
    judge.load()

    # Run judging
    all_summaries = []
    for file_path in csv_files:
        print(f"\nJudging: {os.path.basename(file_path)}")
        result = judge_file(
            file_path=file_path,
            judge=judge,
            sample_size=args.sample,
            failed_only=not args.all_rows,
        )
        if result is None:
            continue

        # Save per-file detailed CSV
        detail_df = pd.DataFrame(result["rows"])
        model_safe = result["model"].replace("/", "_")
        detail_path = os.path.join(args.output_dir, f"judge_{model_safe}.csv")
        detail_df.to_csv(detail_path, index=False)

        # Collect summary
        all_summaries.append({
            "Model": result["model"],
            "File": result["file"],
            "Total Rows": result["total_rows"],
            "Judged": result["judged"],
            "Judge Accepted": result["judge_accepted"],
            "Judge Accept Rate": f"{result['judge_accept_rate']:.2%}",
            "EM Rate (raw)": f"{result['em_rate']:.2%}" if result["em_rate"] is not None else "N/A",
            "Corrected EM Estimate": (
                f"{result['corrected_em_estimate']:.2%}"
                if result["corrected_em_estimate"] is not None else "N/A"
            ),
        })

    # Save aggregate summary
    if all_summaries:
        summary_df = pd.DataFrame(all_summaries)
        summary_path = os.path.join(args.output_dir, "summary.csv")
        summary_df.to_csv(summary_path, index=False)

        print("\n" + "=" * 70)
        print("=== LLM-as-a-Judge Summary ===")
        print("=" * 70)
        print(summary_df.to_string(index=False))
        print(f"\nAll outputs saved to: {args.output_dir}")
    else:
        print("No results were generated.")


if __name__ == "__main__":
    main()
