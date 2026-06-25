"""
Benchmark multiple autoregressive models on the refineID task using Fill-In-the-Middle (FIM).

This script evaluates the identifier renaming ability of various AR code models
by using FIM instead of prompting, making the comparison fair with diffusion LLMs.

For code snippets with multiple [MASK] positions, each mask is filled iteratively
via separate FIM calls -- the first mask is filled, its prediction is inserted back
into the code, and then the next mask is processed.

Data:   data/test.csv  (full 1000-sample test set)
Metric: Exact Match (prediction == ground_truth for the target identifier)

Usage:
    python experiments/benchmark_ar_models_fim.py                        # run all models
    python experiments/benchmark_ar_models_fim.py --model StarCoder2-3B  # single model
    python experiments/benchmark_ar_models_fim.py --max-samples 100      # quick test
    python experiments/benchmark_ar_models_fim.py --hf-repo your-username/your-repo --hf-token your-token
"""

import os
import sys
import csv
import re
import gc
import argparse
import time
from datetime import datetime

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

try:
    from huggingface_hub import HfApi
    HAS_HF_HUB = True
except ImportError:
    HAS_HF_HUB = False

# ---- Configuration ---------------------------------------------------------

DATA_PATH = "data/test.csv"
RESULTS_DIR = "results/ar_fim_benchmark"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_NEW_TOKENS = 20
MAX_INPUT_TOKENS = 8192

# ---- Model Registry --------------------------------------------------------

MODEL_REGISTRY = {
    "StarCoder2-3B":            {"id": "bigcode/starcoder2-3b",                "type": "starcoder",   "max_ctx": 16384},
    "StarCoder2-7B":            {"id": "bigcode/starcoder2-7b",                "type": "starcoder",   "max_ctx": 16384},
    "StarCoder2-15B":           {"id": "bigcode/starcoder2-15b",               "type": "starcoder",   "max_ctx": 16384},
    "DeepSeek-Coder-1.3B-Base": {"id": "deepseek-ai/deepseek-coder-1.3b-base", "type": "deepseek",   "max_ctx": 16384},
    "DeepSeek-Coder-6.7B-Base": {"id": "deepseek-ai/deepseek-coder-6.7b-base", "type": "deepseek",   "max_ctx": 16384},
    "DeepSeek-Coder-33B-Base":  {"id": "deepseek-ai/deepseek-coder-33b-base",  "type": "deepseek",   "max_ctx": 16384},
    "CodeLlama-7B":             {"id": "codellama/CodeLlama-7b-hf",            "type": "codellama",   "max_ctx": 16384},
    "CodeLlama-13B":            {"id": "codellama/CodeLlama-13b-hf",           "type": "codellama",   "max_ctx": 16384},
    "CodeLlama-34B":            {"id": "codellama/CodeLlama-34b-hf",           "type": "codellama",   "max_ctx": 16384},
    "CodeGemma-2B":             {"id": "google/codegemma-2b",                  "type": "codegemma",   "max_ctx": 8192},
    "CodeGemma-7B":             {"id": "google/codegemma-7b",                  "type": "codegemma",   "max_ctx": 8192},
    "Qwen2.5-Coder-1.5B":      {"id": "Qwen/Qwen2.5-Coder-1.5B",             "type": "qwen25coder", "max_ctx": 32768},
    "Qwen2.5-Coder-3B":        {"id": "Qwen/Qwen2.5-Coder-3B",               "type": "qwen25coder", "max_ctx": 32768},
    "Qwen2.5-Coder-7B":        {"id": "Qwen/Qwen2.5-Coder-7B",               "type": "qwen25coder", "max_ctx": 32768},
    "Qwen2.5-Coder-14B":       {"id": "Qwen/Qwen2.5-Coder-14B",              "type": "qwen25coder", "max_ctx": 32768},
}


# ---- FIM Prompt Construction -----------------------------------------------

def build_fim_prompt(prefix, suffix, model_type):
    """Construct a Fill-In-the-Middle prompt for the given model type."""
    if model_type == "starcoder":
        return "<fim_prefix>" + prefix + "<fim_suffix>" + suffix + "<fim_middle>"
    elif model_type == "deepseek":
        return "<｜fim▁begin｜>" + prefix + "<｜fim▁hole｜>" + suffix + "<｜fim▁end｜>"
    elif model_type == "codellama":
        return "<PRE> " + prefix + " <SUF>" + suffix + " <MID>"
    elif model_type in ("codegemma", "qwen25coder"):
        return "<|fim_prefix|>" + prefix + "<|fim_suffix|>" + suffix + "<|fim_middle|>"
    else:
        return "<fim_prefix>" + prefix + "<fim_suffix>" + suffix + "<fim_middle>"


# ---- Prediction Cleaning ---------------------------------------------------

def _get_strip_tokens(model_type):
    """Return a list of special tokens to strip from raw model output."""
    if model_type == "starcoder":
        return ["<fim_prefix>", "<fim_suffix>", "<fim_middle>", "<|endoftext|>", "<file_sep>"]
    elif model_type == "deepseek":
        return ["<｜fim▁begin｜>", "<｜fim▁hole｜>", "<｜fim▁end｜>", "<｜end▁of▁sentence｜>", "<｜begin▁of▁sentence｜>"]
    elif model_type == "codellama":
        return ["<PRE>", "<SUF>", "<MID>"]
    elif model_type in ("codegemma", "qwen25coder"):
        return ["<|fim_prefix|>", "<|fim_suffix|>", "<|fim_middle|>", "<|endoftext|>", "<|end_of_middle|>"]
    else:
        return ["<fim_prefix>", "<fim_suffix>", "<fim_middle>", "<|endoftext|>"]


def clean_prediction(text, model_type):
    """Extract a clean Java identifier from raw model output."""
    for token in _get_strip_tokens(model_type):
        text = text.replace(token, "")

    # Take only the first line, strip quotes and whitespace
    text = text.split("\n")[0].strip('`"\'\' ')

    # Extract the first valid Java identifier
    match = re.search(r'[a-zA-Z_$][a-zA-Z0-9_$]*', text)
    if match:
        return match.group(0)
    return text.strip()


# ---- Data Loading -----------------------------------------------------------

def load_data(data_path, max_samples=None):
    """Load the test CSV (id, masked_code, target)."""
    csv.field_size_limit(sys.maxsize)
    rows = []
    with open(data_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for i, row in enumerate(reader):
            if max_samples is not None and i >= max_samples:
                break
            rows.append({
                "id": row[0],
                "masked_code": row[1],
                "target": row[2].strip(),
            })
    return rows


# ---- Truncation Helper ------------------------------------------------------

def truncate_for_fim(prefix, suffix, tokenizer, max_tokens):
    """Truncate prefix and/or suffix so the FIM prompt fits within max_tokens.
    
    Strategy: keep as much context around the mask as possible.
    Split the budget 60/40 between prefix (tail) and suffix (head).
    """
    prefix_budget = int(max_tokens * 0.6)
    suffix_budget = max_tokens - prefix_budget

    prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
    suffix_ids = tokenizer.encode(suffix, add_special_tokens=False)

    if len(prefix_ids) > prefix_budget:
        prefix_ids = prefix_ids[-prefix_budget:]  # keep tail (closest to mask)
        prefix = tokenizer.decode(prefix_ids, skip_special_tokens=True)

    if len(suffix_ids) > suffix_budget:
        suffix_ids = suffix_ids[:suffix_budget]    # keep head (closest to mask)
        suffix = tokenizer.decode(suffix_ids, skip_special_tokens=True)

    return prefix, suffix


# ---- Single-sample FIM Inference --------------------------------------------

def run_fim_on_sample(masked_code, model, tokenizer, model_type, max_input_tokens):
    """Fill all [MASK] tokens in masked_code via iterative FIM calls.
    
    Returns:
        predictions: list of predicted identifiers (one per mask)
        raw_predictions: list of raw model outputs
        fim_prompts: list of FIM prompts used
        final_code: the code with all masks filled
    """
    current_code = masked_code
    predictions = []
    raw_predictions = []
    fim_prompts = []

    mask_count = current_code.count("[MASK]")

    for _ in range(mask_count):
        # Split at the first remaining [MASK]
        parts = current_code.split("[MASK]", 1)
        prefix = parts[0]
        suffix = parts[1] if len(parts) > 1 else ""

        # Truncate if needed
        prefix_trunc, suffix_trunc = truncate_for_fim(
            prefix, suffix, tokenizer, max_input_tokens
        )

        # Build FIM prompt
        prompt = build_fim_prompt(prefix_trunc, suffix_trunc, model_type)
        fim_prompts.append(prompt)

        # Tokenize and generate
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True,
                           max_length=max_input_tokens).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
                    if tokenizer.eos_token_id is not None
                    else (tokenizer.pad_token_id or 0),
            )

        # Decode only the newly generated tokens
        new_token_ids = outputs[0][inputs.input_ids.shape[1]:]
        raw_pred = tokenizer.decode(new_token_ids, skip_special_tokens=False)
        raw_predictions.append(raw_pred)

        # Clean prediction
        pred = clean_prediction(raw_pred, model_type)
        predictions.append(pred)

        # Replace this mask with the prediction and continue
        current_code = prefix + pred + suffix

    return predictions, raw_predictions, fim_prompts, current_code


# ---- HF Upload Helper -------------------------------------------------------

def upload_to_hf(file_path, repo_id, token, path_in_repo=None):
    """Upload a file to Hugging Face Hub, creating the repo if it doesn't exist."""
    if not HAS_HF_HUB:
        print("    ERROR: 'huggingface_hub' not installed. Skipping upload.")
        return False
    
    if not repo_id:
        return False

    try:
        api = HfApi(token=token)
        
        # Ensure the repository exists
        try:
            api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True)
        except Exception as e:
            # If it's just a permission error on creation but the repo exists, we can still try to upload
            if "already exists" not in str(e).lower():
                print(f"    Note on repo creation: {e}")

        filename = os.path.basename(file_path)
        if path_in_repo is None:
            path_in_repo = f"ar_fim_benchmark/{filename}"
        
        print(f"    Uploading {filename} to HF repo {repo_id}...")
        api.upload_file(
            path_or_fileobj=file_path,
            path_in_repo=path_in_repo,
            repo_id=repo_id,
            token=token,
            repo_type="dataset"
        )
        print("    Upload successful.")
        return True
    except Exception as e:
        print(f"    Upload failed: {e}")
        # Suggest a solution for the common 401 error
        if "401" in str(e) or "Invalid username or password" in str(e):
            print("    HINT: Check if your HF_TOKEN has 'Write' permissions and if the Repo ID is correct.")
        return False


# ---- Main Benchmark ---------------------------------------------------------

def run_benchmark(target_models=None, max_samples=None, hf_repo=None, hf_token=None):
    """Run the FIM benchmark on the refineID dataset."""
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Determine which models to run
    if target_models:
        models_to_run = {}
        for name in target_models:
            if name in MODEL_REGISTRY:
                models_to_run[name] = MODEL_REGISTRY[name]
            else:
                print(f"WARNING: Model '{name}' not found in registry. Available: {list(MODEL_REGISTRY.keys())}")
        if not models_to_run:
            print("ERROR: No valid models specified. Exiting.")
            return
    else:
        models_to_run = MODEL_REGISTRY

    # Load data
    print(f"Loading data from {DATA_PATH}...")
    data = load_data(DATA_PATH, max_samples=max_samples)
    print(f"Loaded {len(data)} samples.")

    # Summary collector
    summary = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    for model_name, meta in models_to_run.items():
        print(f"\n{'='*60}")
        print(f"  Model: {model_name}")
        print(f"  HF ID: {meta['id']}")
        print(f"  Type:  {meta['type']}")
        print(f"{'='*60}")

        # ── Load model ────────────────────────────────────────────────
        t0 = time.time()
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                meta["id"], trust_remote_code=True
            )
            model = AutoModelForCausalLM.from_pretrained(
                meta["id"],
                torch_dtype=torch.bfloat16 if DEVICE == "cuda" else torch.float32,
                device_map="auto" if DEVICE == "cuda" else None,
                trust_remote_code=True,
            )
            if DEVICE == "cpu":
                model = model.to("cpu")
            model.eval()
            load_time = time.time() - t0
            print(f"  Model loaded in {load_time:.1f}s")
        except Exception as e:
            print(f"  FAILED to load model: {e}")
            summary.append({"model": model_name, "error": str(e)})
            continue

        # ── Run inference ─────────────────────────────────────────────
        results = []
        correct = 0
        errors = 0

        for row in tqdm(data, desc=f"  {model_name}"):
            item_id = row["id"]
            masked_code = row["masked_code"]
            ground_truth = row["target"]

            try:
                preds, raw_preds, prompts, final_code = run_fim_on_sample(
                    masked_code, model, tokenizer, meta["type"], MAX_INPUT_TOKENS
                )

                primary_pred = preds[0] if preds else ""
                is_correct = (primary_pred == ground_truth)
                if is_correct:
                    correct += 1

                results.append({
                    "id": item_id,
                    "ground_truth": ground_truth,
                    "prediction": primary_pred,
                    "correct": is_correct,
                    "mask_count": masked_code.count("[MASK]"),
                    "all_predictions": "|".join(preds),
                    "raw_prediction": raw_preds[0] if raw_preds else "",
                    "all_raw_predictions": "|".join(raw_preds),
                })
            except Exception as e:
                errors += 1
                results.append({
                    "id": item_id,
                    "ground_truth": ground_truth,
                    "prediction": "",
                    "correct": False,
                    "error": str(e),
                })
                if errors <= 5:
                    print(f"    Error on sample {item_id}: {e}")
                elif errors == 6:
                    print(f"    ... suppressing further error messages")

        # ── Save per-model results ────────────────────────────────────
        out_file = os.path.join(RESULTS_DIR, f"{model_name}_refineID_fim_{timestamp}.csv")
        with open(out_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "id", "ground_truth", "prediction", "correct",
                "mask_count", "all_predictions",
                "raw_prediction", "all_raw_predictions", "error"
            ])
            writer.writeheader()
            for r in results:
                writer.writerow({k: r.get(k, "") for k in writer.fieldnames})

        accuracy = correct / len(data) if data else 0
        print(f"  Accuracy: {correct}/{len(data)} = {accuracy:.2%}")
        print(f"  Errors:   {errors}")
        print(f"  Results:  {out_file}")

        # ── Auto-upload to HF ────────────────────────────────────────
        if hf_repo:
            upload_to_hf(out_file, hf_repo, hf_token)

        summary.append({
            "model": model_name,
            "hf_id": meta["id"],
            "type": meta["type"],
            "accuracy": f"{accuracy:.4f}",
            "correct": correct,
            "total": len(data),
            "errors": errors,
            "results_file": out_file,
        })

        # ── Cleanup memory ────────────────────────────────────────────
        del model
        del tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    # ── Save summary ──────────────────────────────────────────────────
    summary_file = os.path.join(RESULTS_DIR, f"summary_{timestamp}.csv")
    if summary:
        with open(summary_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
            writer.writeheader()
            writer.writerows(summary)
            
        # Upload summary too
        if hf_repo:
            upload_to_hf(summary_file, hf_repo, hf_token)

    # ── Print summary table ───────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  BENCHMARK SUMMARY")
    print(f"{'='*60}")
    print(f"  {'Model':<30} {'Accuracy':>10} {'Correct':>8} / {'Total':>5} {'Errors':>7}")
    print(f"  {'-'*65}")
    for s in summary:
        if "error" in s and "accuracy" not in s:
            print(f"  {s['model']:<30} LOAD FAILED: {s['error']}")
        else:
            print(f"  {s['model']:<30} {s['accuracy']:>10} {s['correct']:>8} / {s['total']:>5} {s['errors']:>7}")
    print(f"\n  Summary saved to: {summary_file}")


# ---- CLI Entry Point --------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark AR models on refineID using FIM with HF upload"
    )
    parser.add_argument(
        "--model", action="append", default=None,
        help="Model name(s) from MODEL_REGISTRY to run. Can be specified "
             "multiple times. If omitted, all models are run."
    )
    parser.add_argument(
        "--max-samples", type=int, default=None,
        help="Maximum number of test samples to evaluate (for quick tests)."
    )
    parser.add_argument(
        "--hf-repo", type=str, default=None,
        help="Hugging Face Dataset repo ID (e.g. 'username/repo-name') to upload results."
    )
    parser.add_argument(
        "--hf-token", type=str, default=os.environ.get("HF_TOKEN"),
        help="HF Write Token. Recommended to set in HF_TOKEN env var."
    )
    parser.add_argument(
        "--list-models", action="store_true",
        help="List all available models in the registry and exit."
    )

    args = parser.parse_args()

    if args.list_models:
        print("Available models:")
        for name, meta in MODEL_REGISTRY.items():
            print(f"  {name:<30} {meta['id']}")
        sys.exit(0)

    # Check if HF hub is installed if repo is provided
    if args.hf_repo and not HAS_HF_HUB:
        print("WARNING: --hf-repo provided but 'huggingface_hub' is not installed.")
        print("Install it with: pip install huggingface_hub")

    run_benchmark(
        target_models=args.model, 
        max_samples=args.max_samples,
        hf_repo=args.hf_repo,
        hf_token=args.hf_token
    )
