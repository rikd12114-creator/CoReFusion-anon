"""
LLM-as-a-Judge: Variable Renaming Evaluation
=============================================

Evaluates variable renaming predictions against ground-truth identifiers
using an open-source LLM judge. The judge sees the full code context
(with [MASK] replaced by the prediction) and the ground-truth name, and
must output a **binary verdict**: 1 (acceptable) or 0 (not acceptable).

A prediction is judged "acceptable" if it conveys the same semantic
meaning as the ground truth even when the exact string does not match
(e.g. `bufSize` vs `bufferSize` may both be fine in context).

Supported judge models (via Hugging Face `transformers`):
  - Qwen/Qwen2.5-7B-Instruct  (default, best quality/speed trade-off)
  - Qwen/Qwen2.5-3B-Instruct
  - Qwen/Qwen2.5-14B-Instruct
  - mistralai/Mistral-7B-Instruct-v0.3
  - google/gemma-2-9b-it
  - meta-llama/Meta-Llama-3.1-8B-Instruct

Usage:
    # Evaluate a single CSV file:
    python experiments/llm_judge_variable_naming.py \\
        --input data/benchmark_ReFineID_Diffusion/diffusion_benchmark/DiffuCoder-7B_refineID_diffusion_20260226_125550.csv

    # Evaluate ALL CSVs in both benchmark directories:
    python experiments/llm_judge_variable_naming.py --all

    # Choose a different judge model:
    python experiments/llm_judge_variable_naming.py --all --judge-model Qwen/Qwen2.5-14B-Instruct

    # Quick smoke-test with 20 samples:
    python experiments/llm_judge_variable_naming.py --all --max-samples 20

    # Resume an interrupted run (skip already-judged rows):
    python experiments/llm_judge_variable_naming.py --all --resume
"""

import os
import sys
import csv
import re
import gc
import json
import time
import argparse
import textwrap
from datetime import datetime
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATA_PATH = "data/test.csv"  # columns: id (int) | masked_code | ground_truth

DIFFUSION_BENCHMARK_DIR = "data/benchmark_ReFineID_Diffusion/diffusion_benchmark"
FIM_BENCHMARK_DIR       = "data/benchmark_ReFineID_FIM/ar_fim_benchmark"

RESULTS_DIR = "results/llm_judge"

# How many characters of code context to show the judge (centered on the mask).
MAX_CONTEXT_CHARS = 2000

# Max tokens the judge may generate per sample.
MAX_NEW_TOKENS = 32

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Judge model registry:  short_name -> HF model ID
JUDGE_REGISTRY = {
    "Qwen2.5-7B-Instruct":   "Qwen/Qwen2.5-7B-Instruct",
    "Qwen2.5-3B-Instruct":   "Qwen/Qwen2.5-3B-Instruct",
    "Qwen2.5-14B-Instruct":  "Qwen/Qwen2.5-14B-Instruct",
    "Mistral-7B-Instruct":   "mistralai/Mistral-7B-Instruct-v0.3",
    "Gemma-2-9B-It":         "google/gemma-2-9b-it",
    "Llama-3.1-8B-Instruct": "meta-llama/Meta-Llama-3.1-8B-Instruct",
    # ---- larger judges: bf16 weights need a >=80 GB card (RTX PRO 6000 96GB).
    # ---- 24B+ do NOT fit the 48 GB A40; request the big card explicitly.
    "Phi-4":                 "microsoft/phi-4",                            # 14B ~28 GB, MIT
    "Mistral-Small-24B":     "mistralai/Mistral-Small-24B-Instruct-2501",  # 24B ~47 GB, Apache-2.0
    "Gemma-2-27B-It":        "google/gemma-2-27b-it",                      # 27B ~54 GB, gated -> HF_TOKEN
    "Qwen2.5-32B-Instruct":  "Qwen/Qwen2.5-32B-Instruct",                  # 32B ~65 GB
    "Qwen3-32B":             "Qwen/Qwen3-32B",                             # 32B ~65 GB, thinking disabled
}
DEFAULT_JUDGE = "Qwen2.5-7B-Instruct"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_test_data(data_path: str) -> dict:
    """Load test.csv into a dict keyed by 0-based integer row id.

    test.csv format (no header):  row_idx | masked_code | ground_truth
    The 'id' column is 1-indexed in the file but benchmark CSVs use 0-indexed.
    We key the returned dict by the integer in the first column as-is so that
    lookups from benchmark CSV 'id' values work directly.
    """
    csv.field_size_limit(sys.maxsize)
    data = {}
    with open(data_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        first_row = next(reader, None)
        if first_row is None:
            return data
        # Detect whether first row is a header
        try:
            row_id = int(first_row[0])
            if len(first_row) >= 3:
                data[row_id] = {
                    "masked_code":  first_row[1],
                    "ground_truth": first_row[2].strip(),
                }
        except ValueError:
            pass  # header row – skip

        for row in reader:
            if len(row) < 3:
                continue
            try:
                row_id = int(row[0])
            except ValueError:
                continue
            data[row_id] = {
                "masked_code":  row[1],
                "ground_truth": row[2].strip(),
            }
    return data


# ---------------------------------------------------------------------------
# Context extraction
# ---------------------------------------------------------------------------

def extract_context(masked_code: str, prediction: str, max_chars: int) -> str:
    """Substitute [MASK] with prediction and return a trimmed snippet.

    For samples with multiple [MASK] tokens only the first is replaced with
    the model prediction; remaining ones are left as [MASK] for context.
    """
    # Position of the first mask in the original code
    mask_pos = masked_code.find("[MASK]")
    if mask_pos == -1:
        # No mask found – just return the code head
        return masked_code[:max_chars]

    code_with_pred = masked_code.replace("[MASK]", prediction, 1)

    half = max_chars // 2
    start = max(0, mask_pos - half)
    end   = min(len(code_with_pred), mask_pos + len(prediction) + half)

    snippet = code_with_pred[start:end]
    if start > 0:
        snippet = "..." + snippet
    if end < len(code_with_pred):
        snippet = snippet + "..."
    return snippet


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are an expert Java code reviewer evaluating the quality of variable names "
    "suggested by an AI model.\n\n"
    "Your task: given a code snippet and a ground-truth variable name, decide "
    "whether the predicted variable name is SEMANTICALLY ACCEPTABLE as a replacement.\n\n"
    "Rules:\n"
    "1. ACCEPTABLE if the prediction conveys the same concept as the ground truth, "
    "even if the exact string differs "
    "(e.g. 'bufSize' vs 'bufferSize' are both fine for a buffer-size variable).\n"
    "2. NOT ACCEPTABLE if the prediction clearly describes a different concept.\n"
    "3. Single-letter names are usually NOT ACCEPTABLE unless obviously correct "
    "(e.g. loop counter 'i', 'j').\n"
    "4. Names that are clearly wrong tokens ('0', 'true', 'MASK', 'EOT', etc.) are NOT ACCEPTABLE.\n"
    "5. Abbreviations that preserve the same meaning ARE ACCEPTABLE.\n\n"
    "You MUST respond with EXACTLY one line, either:\n"
    "    VERDICT: 1\n"
    "or\n"
    "    VERDICT: 0\n\n"
    "Do NOT add any other text."
)


def build_user_prompt(code_context: str, ground_truth: str, prediction: str) -> str:
    return (
        "Code context (the predicted name replaces the masked identifier):\n"
        "```java\n"
        + code_context
        + "\n```\n\n"
        "Ground-truth variable name: `" + ground_truth + "`\n"
        "Predicted variable name:    `" + prediction + "`\n\n"
        "Is the predicted name semantically acceptable given the code context and "
        "the ground truth?\n"
        "Reply with EXACTLY one line: VERDICT: 1   or   VERDICT: 0"
    )


# ---------------------------------------------------------------------------
# Judge model loading & inference
# ---------------------------------------------------------------------------

def load_judge(model_id: str):
    """Load judge tokenizer and model. Returns (tokenizer, model)."""
    print(f"  Loading judge model: {model_id} on {DEVICE}")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16 if DEVICE == "cuda" else torch.float32,
        device_map="auto" if DEVICE == "cuda" else None,
        trust_remote_code=True,
    )
    if DEVICE == "cpu":
        model = model.to("cpu")
    model.eval()
    print(f"  Judge model loaded.")
    return tokenizer, model


def _apply_chat_template(tokenizer, system: str, user: str) -> str:
    """Apply chat template; fall back to a plain format if unavailable.

    enable_thinking=False forces Qwen3-style judges into non-thinking mode
    (otherwise the 32-token budget is spent inside <think> and the VERDICT
    line never appears); other templates simply ignore the extra variable.
    Gemma-2 templates reject the system role, so retry with the system
    prompt merged into the user turn before falling back to plain text.
    """
    messages = [
        {"role": "system", "content": system},
        {"role": "user",   "content": user},
    ]
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except Exception:
        pass
    try:
        merged = [{"role": "user", "content": system + "\n\n" + user}]
        return tokenizer.apply_chat_template(
            merged,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except Exception:
        # Fallback for models without a chat template
        return f"{system}\n\nUser: {user}\nAssistant:"


def parse_verdict(text: str) -> int:
    """Extract the binary verdict from the model output.

    Returns 1, 0, or -1 (parsing failure).
    """
    # Look for "VERDICT: 1" or "VERDICT: 0" (case-insensitive)
    m = re.search(r"VERDICT\s*:\s*([01])", text, re.IGNORECASE)
    if m:
        return int(m.group(1))

    # Fallback: look for a lone 1 or 0 on a line by itself
    for line in text.strip().splitlines():
        line = line.strip()
        if line in ("1", "0"):
            return int(line)
        if line.lower() in ("yes", "acceptable", "correct"):
            return 1
        if line.lower() in ("no", "not acceptable", "incorrect", "wrong"):
            return 0

    return -1  # could not parse


def judge_one(
    tokenizer,
    model,
    code_context: str,
    ground_truth: str,
    prediction: str,
) -> tuple:
    """Run the judge on a single sample.

    Returns
    -------
    verdict : int
        1  = acceptable
        0  = not acceptable
        -1 = parse failure (treated as 0 in scoring)
    raw_output : str
        The raw text generated by the judge.
    """
    user_content = build_user_prompt(code_context, ground_truth, prediction)
    prompt_text  = _apply_chat_template(tokenizer, SYSTEM_PROMPT, user_content)

    inputs = tokenizer(
        prompt_text,
        return_tensors="pt",
        truncation=True,
        max_length=4096,
    ).to(model.device)

    with torch.no_grad():
        out_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            temperature=1.0,
            pad_token_id=(
                tokenizer.eos_token_id
                if tokenizer.eos_token_id is not None
                else (tokenizer.pad_token_id or 0)
            ),
        )

    new_ids  = out_ids[0][inputs.input_ids.shape[1]:]
    raw_text = tokenizer.decode(new_ids, skip_special_tokens=True).strip()
    verdict  = parse_verdict(raw_text)
    return verdict, raw_text


# ---------------------------------------------------------------------------
# Per-file evaluation
# ---------------------------------------------------------------------------

def collect_benchmark_files(
    diffusion_dir: str,
    fim_dir: str,
    input_file: str | None,
) -> list:
    """Return list of (benchmark_type, filepath) tuples to evaluate."""
    files = []
    if input_file:
        path = Path(input_file)
        if not path.exists():
            sys.exit(f"ERROR: Input file not found: {input_file}")
        btype = "diffusion" if "diffusion" in str(path).lower() else "fim"
        files.append((btype, str(path)))
    else:
        for fpath in sorted(Path(diffusion_dir).glob("*.csv")):
            files.append(("diffusion", str(fpath)))
        for fpath in sorted(Path(fim_dir).glob("*.csv")):
            files.append(("fim", str(fpath)))
    return files


def load_existing_results(output_path: str) -> set:
    """Load already-judged sample ids from an existing output CSV."""
    done_ids = set()
    if not os.path.exists(output_path):
        return done_ids
    with open(output_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                done_ids.add(int(row["id"]))
            except (KeyError, ValueError):
                pass
    return done_ids


def evaluate_file(
    benchmark_type: str,
    benchmark_path: str,
    test_data: dict,
    tokenizer,
    model,
    results_dir: str,
    judge_name: str,
    max_samples: int | None,
    resume: bool,
) -> dict:
    """Evaluate all samples in a benchmark CSV and save a results CSV.

    Returns a summary dict.
    """
    benchmark_stem = Path(benchmark_path).stem
    timestamp      = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_judge     = re.sub(r"[/\\]", "_", judge_name)
    out_filename   = f"{benchmark_stem}__judge_{safe_judge}__{timestamp}.csv"
    out_path       = os.path.join(results_dir, out_filename)

    os.makedirs(results_dir, exist_ok=True)

    # Load benchmark results
    csv.field_size_limit(sys.maxsize)
    rows = []
    with open(benchmark_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    if max_samples is not None:
        rows = rows[:max_samples]

    # Resume: find already-judged ids
    done_ids: set = set()
    if resume:
        # Find the most recent output file for this benchmark
        existing = sorted(
            Path(results_dir).glob(f"{benchmark_stem}__judge_{safe_judge}__*.csv"),
            reverse=True,
        )
        if existing:
            out_path = str(existing[0])  # append to latest
            done_ids = load_existing_results(out_path)
            print(f"  Resuming from {out_path} ({len(done_ids)} already done).")

    # Determine fieldnames
    fieldnames = [
        "id", "ground_truth", "prediction", "exact_match",
        "llm_verdict", "llm_raw_output", "error",
    ]

    # Open output (append mode if resuming)
    write_mode = "a" if resume and done_ids else "w"
    out_f = open(out_path, write_mode, newline="", encoding="utf-8")
    writer = csv.DictWriter(out_f, fieldnames=fieldnames)
    if write_mode == "w":
        writer.writeheader()

    # Track stats
    total_judged = 0
    total_verdict_1 = 0
    total_exact = 0
    parse_failures = 0
    errors = 0

    pbar = tqdm(rows, desc=f"  Judging {benchmark_stem[:40]}", unit="sample")
    for row in pbar:
        try:
            sample_id = int(row["id"])
        except (KeyError, ValueError):
            continue

        if sample_id in done_ids:
            continue

        ground_truth = row.get("ground_truth", "").strip()
        prediction   = row.get("prediction",   "").strip()
        exact_match  = str(row.get("correct", "False")).lower() in ("true", "1", "yes")

        # Look up code context from test.csv
        test_entry = test_data.get(sample_id)
        if test_entry is None:
            # Try 0-indexed offset (test.csv rows start at 1)
            test_entry = test_data.get(sample_id + 1)

        if test_entry is None:
            error_msg = f"sample_id {sample_id} not found in test.csv"
            writer.writerow({
                "id": sample_id,
                "ground_truth": ground_truth,
                "prediction": prediction,
                "exact_match": exact_match,
                "llm_verdict": -1,
                "llm_raw_output": "",
                "error": error_msg,
            })
            errors += 1
            continue

        masked_code = test_entry["masked_code"]

        # If exact match is True, verdict is automatically 1 (skip LLM call)
        if exact_match:
            writer.writerow({
                "id": sample_id,
                "ground_truth": ground_truth,
                "prediction": prediction,
                "exact_match": True,
                "llm_verdict": 1,
                "llm_raw_output": "exact_match",
                "error": "",
            })
            total_judged += 1
            total_exact  += 1
            total_verdict_1 += 1
            out_f.flush()
            pbar.set_postfix({"acc_judge": f"{total_verdict_1}/{total_judged}"})
            continue

        # Skip clearly invalid predictions before calling the LLM
        if not prediction or not re.search(r"[a-zA-Z]", prediction):
            writer.writerow({
                "id": sample_id,
                "ground_truth": ground_truth,
                "prediction": prediction,
                "exact_match": False,
                "llm_verdict": 0,
                "llm_raw_output": "invalid_prediction",
                "error": "",
            })
            total_judged += 1
            out_f.flush()
            pbar.set_postfix({"acc_judge": f"{total_verdict_1}/{total_judged}"})
            continue

        # Extract code context snippet
        context = extract_context(masked_code, prediction, MAX_CONTEXT_CHARS)

        # Call LLM judge
        try:
            verdict, raw_out = judge_one(
                tokenizer, model, context, ground_truth, prediction
            )
        except Exception as e:
            verdict = -1
            raw_out = str(e)
            errors += 1

        if verdict == -1:
            parse_failures += 1

        # Treat parse failure as 0 for scoring purposes
        score = max(verdict, 0)
        total_verdict_1 += score
        total_judged    += 1

        writer.writerow({
            "id": sample_id,
            "ground_truth": ground_truth,
            "prediction": prediction,
            "exact_match": exact_match,
            "llm_verdict": verdict,
            "llm_raw_output": raw_out[:200],  # truncate long outputs
            "error": "",
        })
        out_f.flush()
        pbar.set_postfix({"acc_judge": f"{total_verdict_1}/{total_judged}"})

    out_f.close()

    judge_accuracy = total_verdict_1 / total_judged if total_judged else 0.0
    exact_accuracy = total_exact      / total_judged if total_judged else 0.0

    summary = {
        "benchmark_type":  benchmark_type,
        "benchmark_file":  os.path.basename(benchmark_path),
        "judge_model":     judge_name,
        "total_judged":    total_judged,
        "exact_match_acc": f"{exact_accuracy:.4f}",
        "llm_judge_acc":   f"{judge_accuracy:.4f}",
        "verdict_1_count": total_verdict_1,
        "parse_failures":  parse_failures,
        "errors":          errors,
        "output_file":     out_path,
    }
    return summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="LLM-as-a-Judge evaluation for variable renaming benchmark results.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--input", type=str, default=None,
        help="Path to a single benchmark CSV to evaluate."
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Evaluate ALL CSVs in both benchmark directories."
    )
    parser.add_argument(
        "--judge-model", type=str, default=DEFAULT_JUDGE,
        help=(
            "Judge model: either a short name from the registry "
            f"({', '.join(JUDGE_REGISTRY.keys())}) "
            "or a full HF model ID. Default: %(default)s"
        ),
    )
    parser.add_argument(
        "--max-samples", type=int, default=None,
        help="Evaluate only the first N samples per file (useful for quick tests)."
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume a previous run by skipping already-judged samples."
    )
    parser.add_argument(
        "--results-dir", type=str, default=RESULTS_DIR,
        help=f"Directory to save judge output CSVs. Default: {RESULTS_DIR}"
    )
    parser.add_argument(
        "--test-data", type=str, default=DATA_PATH,
        help=f"Path to test.csv (for code context). Default: {DATA_PATH}"
    )
    parser.add_argument(
        "--list-models", action="store_true",
        help="List available judge models and exit."
    )
    parser.add_argument(
        "--diffusion-dir", type=str, default=DIFFUSION_BENCHMARK_DIR,
        help="Directory containing Diffusion benchmark CSVs."
    )
    parser.add_argument(
        "--fim-dir", type=str, default=FIM_BENCHMARK_DIR,
        help="Directory containing FIM benchmark CSVs."
    )

    args = parser.parse_args()

    if args.list_models:
        print("Available judge models:")
        for name, mid in JUDGE_REGISTRY.items():
            print(f"  {name:<30} -> {mid}")
        sys.exit(0)

    if not args.input and not args.all:
        parser.error("Specify --input <file> or --all to evaluate all benchmark files.")

    # Resolve judge model ID
    judge_name = args.judge_model
    if judge_name in JUDGE_REGISTRY:
        judge_model_id = JUDGE_REGISTRY[judge_name]
    else:
        # Assume it's a full HF model ID
        judge_model_id = judge_name
        # Use the last component as a friendly name
        judge_name = judge_name.split("/")[-1]

    print("=" * 70)
    print("  LLM-as-a-Judge: Variable Renaming Evaluation")
    print("=" * 70)
    print(f"  Judge model : {judge_model_id}")
    print(f"  Device      : {DEVICE}")
    print(f"  Test data   : {args.test_data}")
    print(f"  Results dir : {args.results_dir}")
    if args.max_samples:
        print(f"  Max samples : {args.max_samples} per file")
    print()

    # Load test data (code context)
    print("Loading test.csv ...")
    test_data = load_test_data(args.test_data)
    print(f"  Loaded {len(test_data)} samples from test.csv\n")

    # Collect benchmark files
    bench_files = collect_benchmark_files(
        args.diffusion_dir, args.fim_dir, args.input
    )
    if not bench_files:
        sys.exit("ERROR: No benchmark CSV files found.")

    print(f"Found {len(bench_files)} benchmark file(s) to evaluate:")
    for btype, fpath in bench_files:
        print(f"  [{btype:9s}] {os.path.basename(fpath)}")
    print()

    # Load judge model (once, reused across all files)
    tokenizer, model = load_judge(judge_model_id)
    print()

    os.makedirs(args.results_dir, exist_ok=True)

    all_summaries = []
    grand_start   = time.time()

    for btype, fpath in bench_files:
        print(f"\n{'─'*70}")
        print(f"  Evaluating [{btype}]: {os.path.basename(fpath)}")
        print(f"{'─'*70}")

        t0 = time.time()
        summary = evaluate_file(
            benchmark_type=btype,
            benchmark_path=fpath,
            test_data=test_data,
            tokenizer=tokenizer,
            model=model,
            results_dir=args.results_dir,
            judge_name=judge_name,
            max_samples=args.max_samples,
            resume=args.resume,
        )
        elapsed = time.time() - t0

        print(
            f"\n  Exact Match Acc : {summary['exact_match_acc']}"
            f"\n  LLM Judge Acc   : {summary['llm_judge_acc']}"
            f"  ({summary['verdict_1_count']}/{summary['total_judged']} acceptable)"
            f"\n  Parse failures  : {summary['parse_failures']}"
            f"\n  Errors          : {summary['errors']}"
            f"\n  Time            : {elapsed:.1f}s"
            f"\n  Output          : {summary['output_file']}"
        )
        all_summaries.append(summary)

    # Save aggregate summary
    safe_judge = re.sub(r"[/\\]", "_", judge_name)
    summary_path = os.path.join(
        args.results_dir,
        f"summary_judge_{safe_judge}_"
        f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
    )
    if all_summaries:
        with open(summary_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(all_summaries[0].keys()))
            writer.writeheader()
            writer.writerows(all_summaries)

    # Print aggregate summary table
    total_elapsed = time.time() - grand_start
    print(f"\n{'='*70}")
    print("  JUDGE EVALUATION SUMMARY")
    print(f"  Judge model: {judge_model_id}")
    print(f"{'='*70}")
    print(f"  {'File':<50} {'EM Acc':>8}  {'LLM Acc':>8}")
    print(f"  {'-'*68}")
    for s in all_summaries:
        fname = s["benchmark_file"][:50]
        print(
            f"  {fname:<50} {s['exact_match_acc']:>8}  {s['llm_judge_acc']:>8}"
        )
    print(f"\n  Summary saved to: {summary_path}")
    print(f"  Total time: {total_elapsed:.1f}s")

    # Cleanup
    del model
    del tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


if __name__ == "__main__":
    main()
