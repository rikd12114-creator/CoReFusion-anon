"""
Benchmark diffusion language models (DiffuCoder, DreamCoder, DreamOn) on the refineID task.

For DiffuCoder / DreamCoder:
    Replace each [MASK] with NUM_MASK_TOKENS <|mask|> tokens and run a single
    diffusion_generate pass over the whole sequence.

For DreamOn (Dream-org/DreamOn-v0-7B):
    DreamOn is a variable-length code infilling model that expects the input
    framed as  BOS + prefix_ids + [mask_id]*N + suffix_ids + EOS.
    Each [MASK] site is therefore processed independently as its own infilling
    prompt. We start with NUM_MASK_TOKENS mask tokens and let DreamOn expand
    /contract the canvas up to DREAMON_MAX_NEW_TOKENS during diffusion.

Data:   data/test.csv (full 1000-sample test set)
Metric: Exact Match (on the first masked site)

Usage:
    python experiments/benchmark_diffusion_models.py                        # run all models
    python experiments/benchmark_diffusion_models.py --model DiffuCoder-7B  # single model
    python experiments/benchmark_diffusion_models.py --model DreamOn-7B     # DreamOn only
    python experiments/benchmark_diffusion_models.py --max-samples 100      # quick test
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
import pandas as pd
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

try:
    from huggingface_hub import HfApi
    HAS_HF_HUB = True
except ImportError:
    HAS_HF_HUB = False

# --- Environment Setting: Mock torchvision ---
# Workaround to prevent import errors with some Diffusion Transformers lacking visual components
class MockModule:
    def __getattr__(self, name): return MockModule()
    def __call__(self, *args, **kwargs): return MockModule()

import sys
sys.modules['torchvision'] = MockModule()
sys.modules['torchvision.ops'] = MockModule()
sys.modules['torchvision.transforms'] = MockModule()
if not hasattr(torch.ops, 'torchvision'):
    class DummyOps:
        def nms(*args, **kwargs): return torch.tensor([])
    torch.ops.torchvision = DummyOps()

# ---- Configuration ---------------------------------------------------------

DATA_PATH = "CoRefusion/data/test.csv"
RESULTS_DIR = "results/diffusion_benchmark"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Standard identifier mask length for continuous substitution
NUM_MASK_TOKENS = 2

# DreamOn-specific: max canvas size after expansion (must be >= NUM_MASK_TOKENS)
DREAMON_MAX_NEW_TOKENS = 32

# ---- Model Registry --------------------------------------------------------

MODEL_REGISTRY = {
    "DiffuCoder-7B":     {"id": "apple/DiffuCoder-7B-Base", "type": "diffucoder", "mask_token": "<|mask|>"},
    "DreamCoder-7B":     {"id": "Dream-org/Dream-Coder-v0-Instruct-7B", "type": "dreamcoder", "mask_token": "<|mask|>"},
    "DreamOn-7B":        {"id": "Dream-org/DreamOn-v0-7B", "type": "dreamon", "mask_token": None},
}


# ---- Prediction Extraction -------------------------------------------------

def extract_all_predictions(full_code, masked_code):
    """
    Extract predictions for ALL [MASK] locations in the snippet.
    Returns:
        list of extracted identifiers matching each [MASK] position.
    """
    parts = masked_code.split("[MASK]")
    if len(parts) <= 1: 
        return []
    
    predictions = []
    current_search_start = 0
    
    for i in range(len(parts) - 1):
        pre = parts[i]
        post = parts[i+1]
        
        # Use short anchors for robustness
        pre_anchor = pre.strip()[-30:] if len(pre.strip()) > 30 else pre.strip()
        post_anchor = post.strip()[:30] if len(post.strip()) > 30 else post.strip()
        
        # 1. Locate pre_anchor
        if pre_anchor:
            idx_start = full_code.find(pre_anchor, current_search_start)
            if idx_start != -1:
                idx_start += len(pre_anchor)
            else:
                idx_start = current_search_start
        else:
            idx_start = current_search_start
            
        # 2. Locate post_anchor
        if post_anchor:
            idx_end = full_code.find(post_anchor, idx_start)
        else:
            idx_end = -1
            
        # 3. Extract the gap content
        if idx_end != -1:
            gap_content = full_code[idx_start:idx_end].strip()
            current_search_start = idx_end
        else:
            # Fallback constraint window
            gap_content = full_code[idx_start : idx_start + 60].strip()
            current_search_start = idx_start + 60
            
        # 4. Extract valid Java identifier
        match = re.search(r'[a-zA-Z_$][a-zA-Z0-9_$]*', gap_content)
        predictions.append(match.group(0) if match else gap_content[:20])
        
    return predictions


# ---- DreamOn Inference ------------------------------------------------------

def build_dreamon_input(prefix, suffix, tokenizer, num_mask_tokens):
    """Build the BOS + prefix + masks + suffix + EOS infilling prompt for DreamOn."""
    bos = tokenizer.bos_token_id
    eos = tokenizer.eos_token_id
    mask_id = tokenizer.mask_token_id
    pre_ids = tokenizer.encode(prefix, add_special_tokens=False)
    suf_ids = tokenizer.encode(suffix, add_special_tokens=False)
    ids = [bos] + pre_ids + [mask_id] * num_mask_tokens + suf_ids + [eos]
    return ids, len(pre_ids) + 1, num_mask_tokens  # ids, infill_start (after BOS+prefix), initial_canvas_len


def extract_dreamon_infill(generated_ids, tokenizer, prefix, suffix):
    """Decode DreamOn output and slice out the segment between prefix and suffix."""
    full = tokenizer.decode(generated_ids, skip_special_tokens=True)
    # Anchor on prefix tail / suffix head to locate the infilled span
    pre_anchor = prefix.strip()[-30:] if len(prefix.strip()) > 30 else prefix.strip()
    suf_anchor = suffix.strip()[:30] if len(suffix.strip()) > 30 else suffix.strip()

    if pre_anchor:
        i = full.rfind(pre_anchor)
        start = i + len(pre_anchor) if i != -1 else 0
    else:
        start = 0
    if suf_anchor:
        j = full.find(suf_anchor, start)
        end = j if j != -1 else min(start + 60, len(full))
    else:
        end = min(start + 60, len(full))
    return full[start:end].strip()


def run_dreamon_per_mask(model, tokenizer, masked_code, num_initial_masks, max_new_tokens, gen_kwargs):
    """For each [MASK] site, run a separate DreamOn infilling pass.

    Returns:
        preds: list[str]  - extracted identifier per [MASK] site
        full_codes: list[str] - decoded outputs (for debugging)
    """
    parts = masked_code.split("[MASK]")
    if len(parts) <= 1:
        return [], []

    preds = []
    full_codes = []
    # Walk masks left-to-right; for each one, prefix = code so far (with PREVIOUSLY
    # predicted identifiers substituted in), suffix = the rest with remaining [MASK]s.
    resolved = parts[0]
    for i in range(len(parts) - 1):
        prefix = resolved
        # Suffix = remainder containing remaining masks (but DreamOn only fills the FIRST gap)
        suffix = "[MASK]".join(parts[i + 1:])

        input_ids, _, _ = build_dreamon_input(prefix, suffix, tokenizer, num_initial_masks)
        input_ids_t = torch.LongTensor([input_ids]).to(model.device)

        with torch.no_grad():
            output = model.diffusion_generate(
                input_ids_t,
                max_new_tokens=max_new_tokens,
                return_dict_in_generate=True,
                output_history=False,
                **gen_kwargs,
            )
        seq = output.sequences[0] if hasattr(output, "sequences") else output[0]
        gen_text = extract_dreamon_infill(seq, tokenizer, prefix, suffix)
        full_codes.append(tokenizer.decode(seq, skip_special_tokens=True))

        # Pull a Java identifier out of the gap text
        m = re.search(r'[a-zA-Z_$][a-zA-Z0-9_$]*', gen_text)
        ident = m.group(0) if m else gen_text[:20]
        preds.append(ident)

        # Splice the predicted identifier in so the next prefix sees consistent code
        resolved = prefix + ident + parts[i + 1]

    return preds, full_codes


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
            if "already exists" not in str(e).lower():
                print(f"    Note on repo creation: {e}")

        filename = os.path.basename(file_path)
        if path_in_repo is None:
            path_in_repo = f"diffusion_benchmark/{filename}"
        
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
        if "401" in str(e) or "Invalid username or password" in str(e):
            print("    HINT: Check if your HF_TOKEN has 'Write' permissions and if the Repo ID is correct.")
        return False


# ---- Main Benchmark ---------------------------------------------------------

def run_benchmark(target_models=None, max_samples=None, hf_repo=None, hf_token=None):
    """Run the benchmark for Diffusion Models on the refineID dataset."""
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
            model = AutoModel.from_pretrained(
                meta["id"],
                torch_dtype=torch.bfloat16 if DEVICE == "cuda" else torch.float32,
                trust_remote_code=True,
            )
            # Send model to device
            model = model.to(DEVICE)
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
                if meta["type"] == "dreamon":
                    # DreamOn: per-mask infilling with BOS/EOS framing
                    dreamon_kwargs = dict(
                        temperature=0.2,
                        alg="entropy",
                        alg_temp=0.0,
                        top_p=0.9,
                        number_transfer_tokens=1,
                    )
                    preds, full_codes = run_dreamon_per_mask(
                        model, tokenizer, masked_code,
                        num_initial_masks=NUM_MASK_TOKENS,
                        max_new_tokens=DREAMON_MAX_NEW_TOKENS,
                        gen_kwargs=dreamon_kwargs,
                    )
                    full_code_len = sum(len(c) for c in full_codes)
                else:
                    # 1. Substitute [MASK] with <|mask|> * NUM_MASK_TOKENS
                    multi_mask = meta["mask_token"] * NUM_MASK_TOKENS
                    input_code = masked_code.replace("[MASK]", multi_mask)

                    # 2. Tokenize
                    inputs = tokenizer(input_code, return_tensors="pt")
                    input_ids = inputs.input_ids.to(model.device)
                    attention_mask = inputs.attention_mask.to(model.device)

                    # 3. Diffusion Generation Step
                    with torch.no_grad():
                        output = model.diffusion_generate(
                            input_ids,
                            attention_mask=attention_mask,
                            max_new_tokens=1,
                            steps=64,
                            temperature=0.3,
                            top_p=0.95,
                            alg="entropy",
                            alg_temp=0.,
                        )

                    generated_ids = output.sequences[0] if hasattr(output, "sequences") else output[0]
                    full_code = tokenizer.decode(generated_ids, skip_special_tokens=True)
                    preds = extract_all_predictions(full_code, masked_code)
                    full_code_len = len(full_code)

                # primary_pred uses the first mask prediction
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
                    "full_code_length": full_code_len,
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
        out_file = os.path.join(RESULTS_DIR, f"{model_name}_refineID_diffusion_{timestamp}.csv")
        with open(out_file, "w", newline="", encoding="utf-8") as f:
            headers = ["id", "ground_truth", "prediction", "correct", 
                       "mask_count", "all_predictions", "full_code_length", "error"]
            writer = csv.DictWriter(f, fieldnames=headers)
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
        description="Benchmark Diffusion Models on refineID with no-prompt setup"
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
