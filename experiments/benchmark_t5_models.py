"""
Benchmark T5-type (encoder-decoder) code models on the refineID task.

T5 models were pretrained with span-masking using sentinel tokens
(<extra_id_0>, <extra_id_1>, ...), which maps naturally onto our [MASK]
identifier-filling task. CodeT5 in particular had "identifier unmasking"
as an explicit pretraining objective (IT -- Identifier Tagging /
MIP -- Masked Identifier Prediction), so we expect strong performance.

Unlike the AR FIM benchmark, a single forward pass fills *all* masks at
once: we replace the i-th [MASK] with <extra_id_i> and parse the decoder
output which alternates sentinels and predicted spans.

Data:   data/test.csv
Metric: Exact Match on the first masked identifier (same as AR benchmark).

Usage:
    python experiments/benchmark_t5_models.py
    python experiments/benchmark_t5_models.py --model CodeT5p-770M
    python experiments/benchmark_t5_models.py --max-samples 100
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
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, T5ForConditionalGeneration
from tqdm import tqdm

try:
    from huggingface_hub import HfApi
    HAS_HF_HUB = True
except ImportError:
    HAS_HF_HUB = False

# ---- Configuration ---------------------------------------------------------

DATA_PATH = "data/test.csv"
# Share the results directory with the AR FIM benchmark so downstream
# analysis / summary loaders can treat T5 rows as just another row type.
RESULTS_DIR = "results/ar_fim_benchmark"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_NEW_TOKENS = 16    # identifiers are short; this also discourages span-mode rambling
MAX_INPUT_TOKENS = 512 # most CodeT5 variants are 512; CodeT5+ allows more

# ---- Model Registry --------------------------------------------------------
# "max_ctx" = encoder input length the model was trained with.
# CodeT5 family == 512; CodeT5+ 770M and larger support up to 2048.

MODEL_REGISTRY = {
    "CodeT5-small":       {"id": "Salesforce/codet5-small",        "type": "codet5",  "max_ctx": 512},
    "CodeT5-base":        {"id": "Salesforce/codet5-base",         "type": "codet5",  "max_ctx": 512},
    "CodeT5-large":       {"id": "Salesforce/codet5-large",        "type": "codet5",  "max_ctx": 512},
    "CodeT5p-220M":       {"id": "Salesforce/codet5p-220m",        "type": "codet5p", "max_ctx": 512},
    "CodeT5p-770M":       {"id": "Salesforce/codet5p-770m",        "type": "codet5p", "max_ctx": 512},
    # Full-size CodeT5+ (2B / 6B / 16B) live in a separate script,
    # benchmark_codet5p_16b.py, because their CodeGen tokenizer + custom
    # remote arch require the prefix-completion pattern from the model
    # card and are incompatible with this script's sentinel protocol.
}


# ---- Prompt Construction ---------------------------------------------------

def build_t5_single_slot_input(prefix, suffix):
    """Build a single-sentinel T5 encoder input that mirrors the AR FIM call.

    The AR benchmark fills one [MASK] at a time: it splits the code at the
    first [MASK] and feeds (prefix, suffix) into the model's native FIM
    interface. We do the exact same thing here, only the "fill-in-the-middle"
    interface is T5's <extra_id_0> sentinel instead of FIM tokens.

    Other unfilled [MASK] strings stay as literal text in `suffix`, just
    like in the AR FIM script -- the model sees them but they don't get
    structured treatment.
    """
    return prefix + "<extra_id_0>" + suffix


# ---- Prediction Parsing ----------------------------------------------------

# Matches <extra_id_N> sentinels in decoder output
_SENTINEL_RE = re.compile(r"<extra_id_(\d+)>")


def parse_t5_first_span(decoded):
    """Extract the span the decoder produced for <extra_id_0>.

    Decoder output looks like:
        <pad> <extra_id_0> <span> <extra_id_1> ... </s>
    We return the slice between <extra_id_0> and the next sentinel (or
    </s>). One slot per call -- the iteration is at the caller level.
    """
    m0 = re.search(r"<extra_id_0>", decoded)
    if not m0:
        return decoded.replace("<pad>", "").replace("</s>", "").strip()
    tail = decoded[m0.end():]
    m_next = _SENTINEL_RE.search(tail)
    span = tail[:m_next.start()] if m_next else tail
    return span.replace("</s>", "").replace("<pad>", "").strip()


def clean_identifier(text):
    """Extract the first valid Java identifier from a span."""
    # Strip leading whitespace / punctuation from the span
    text = text.split("\n")[0].strip('`"\'\t ')
    match = re.search(r'[a-zA-Z_$][a-zA-Z0-9_$]*', text)
    return match.group(0) if match else text.strip()


# ---- Data Loading ----------------------------------------------------------

def load_data(data_path, max_samples=None):
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


# ---- Truncation ------------------------------------------------------------

def truncate_for_fim(prefix, suffix, tokenizer, max_tokens):
    """Identical to the AR FIM script's `truncate_for_fim`.

    Keep as much context around the slot as possible: 60% of the budget
    for the prefix tail, 40% for the suffix head. Reserve ~3 tokens for
    the sentinel itself.
    """
    budget = max_tokens - 3  # leave room for <extra_id_0>
    prefix_budget = int(budget * 0.6)
    suffix_budget = budget - prefix_budget

    prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
    suffix_ids = tokenizer.encode(suffix, add_special_tokens=False)

    if len(prefix_ids) > prefix_budget:
        prefix_ids = prefix_ids[-prefix_budget:]
        prefix = tokenizer.decode(prefix_ids, skip_special_tokens=True)

    if len(suffix_ids) > suffix_budget:
        suffix_ids = suffix_ids[:suffix_budget]
        suffix = tokenizer.decode(suffix_ids, skip_special_tokens=True)

    return prefix, suffix


# ---- Single-sample Inference -----------------------------------------------

def run_t5_on_sample(masked_code, model, tokenizer, max_input_tokens):
    """Iterate over [MASK]s and fill one at a time, mirroring AR's FIM loop.

    For each mask:
      1. Split the *current* code at the first remaining [MASK]
         (other [MASK]s stay as literal text in `suffix`, exactly like AR).
      2. Truncate prefix/suffix into the encoder budget around the slot.
      3. Encode `prefix + <extra_id_0> + suffix`, generate, parse the
         <extra_id_0> span, clean to a Java identifier.
      4. Replace this [MASK] with the prediction and continue.

    Returns: predictions, raw_predictions, prompts, final_code
    -- same shape as `run_fim_on_sample` in benchmark_ar_models_fim.py.
    """
    current_code = masked_code
    predictions = []
    raw_predictions = []
    prompts = []
    mask_count = current_code.count("[MASK]")

    for _ in range(mask_count):
        parts = current_code.split("[MASK]", 1)
        prefix = parts[0]
        suffix = parts[1] if len(parts) > 1 else ""

        prefix_t, suffix_t = truncate_for_fim(prefix, suffix, tokenizer, max_input_tokens)
        t5_input = build_t5_single_slot_input(prefix_t, suffix_t)
        prompts.append(t5_input)

        inputs = tokenizer(
            t5_input,
            return_tensors="pt",
            truncation=True,
            max_length=max_input_tokens,
        ).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                num_beams=1,
            )

        raw_pred = tokenizer.decode(outputs[0], skip_special_tokens=False)
        raw_predictions.append(raw_pred)

        span = parse_t5_first_span(raw_pred)
        pred = clean_identifier(span)
        predictions.append(pred)

        current_code = prefix + pred + suffix

    return predictions, raw_predictions, prompts, current_code


# ---- HF Upload Helper ------------------------------------------------------

def upload_to_hf(file_path, repo_id, token, path_in_repo=None):
    if not HAS_HF_HUB or not repo_id:
        return False
    try:
        api = HfApi(token=token)
        try:
            api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True)
        except Exception as e:
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
            repo_type="dataset",
        )
        print("    Upload successful.")
        return True
    except Exception as e:
        print(f"    Upload failed: {e}")
        return False


# ---- Main Benchmark --------------------------------------------------------

def run_benchmark(target_models=None, max_samples=None, hf_repo=None, hf_token=None, debug=False):
    os.makedirs(RESULTS_DIR, exist_ok=True)

    if target_models:
        models_to_run = {n: MODEL_REGISTRY[n] for n in target_models if n in MODEL_REGISTRY}
        missing = [n for n in target_models if n not in MODEL_REGISTRY]
        for n in missing:
            print(f"WARNING: '{n}' not in registry. Available: {list(MODEL_REGISTRY)}")
        if not models_to_run:
            print("ERROR: No valid models specified.")
            return
    else:
        models_to_run = MODEL_REGISTRY

    print(f"Loading data from {DATA_PATH}...")
    data = load_data(DATA_PATH, max_samples=max_samples)
    print(f"Loaded {len(data)} samples.")

    summary = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    for model_name, meta in models_to_run.items():
        print(f"\n{'='*60}")
        print(f"  Model: {model_name}")
        print(f"  HF ID: {meta['id']}")
        print(f"  Type:  {meta['type']}")
        print(f"{'='*60}")

        t0 = time.time()
        try:
            tokenizer = AutoTokenizer.from_pretrained(meta["id"], trust_remote_code=True)
            model = AutoModelForSeq2SeqLM.from_pretrained(
                meta["id"],
                torch_dtype=torch.bfloat16 if DEVICE == "cuda" else torch.float32,
                device_map="auto" if DEVICE == "cuda" else None,
                trust_remote_code=True,
            )
            if DEVICE == "cpu":
                model = model.to("cpu")
            model.eval()
            print(f"  Model loaded in {time.time() - t0:.1f}s")
        except Exception as e:
            print(f"  FAILED to load model: {e}")
            summary.append({"model": model_name, "error": str(e)})
            continue

        max_ctx = min(meta.get("max_ctx", MAX_INPUT_TOKENS), MAX_INPUT_TOKENS)

        results = []
        correct = 0
        errors = 0
        truncated_count = 0  # how many samples needed encoder truncation

        for sample_idx, row in enumerate(tqdm(data, desc=f"  {model_name}")):
            item_id = row["id"]
            masked_code = row["masked_code"]
            ground_truth = row["target"]

            try:
                preds, raw_preds, prompts, final_code = run_t5_on_sample(
                    masked_code, model, tokenizer, max_ctx
                )
                full_len = len(tokenizer.encode(masked_code, add_special_tokens=False))
                if full_len > max_ctx:
                    truncated_count += 1
                if debug and sample_idx < 3:
                    first_prompt = prompts[0] if prompts else ""
                    first_raw = raw_preds[0] if raw_preds else ""
                    print(f"\n  --- DEBUG sample {sample_idx} (id={item_id}) ---")
                    print(f"  target          : {ground_truth!r}")
                    print(f"  mask_count      : {masked_code.count('[MASK]')}")
                    print(f"  full input toks : {full_len}  (encoder cap = {max_ctx})")
                    print(f"  truncated input : {first_prompt[:300]}...{first_prompt[-300:]}")
                    print(f"  raw output[0]   : {first_raw!r}")
                    print(f"  parsed pred[0]  : {preds[0] if preds else '<empty>'!r}")
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

        # Same filename pattern as the AR FIM benchmark for unified loading.
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
        print(f"  Accuracy:  {correct}/{len(data)} = {accuracy:.2%}")
        print(f"  Errors:    {errors}")
        print(f"  Truncated: {truncated_count}/{len(data)} samples exceeded {max_ctx}-token encoder limit")
        print(f"  Results:   {out_file}")

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

        del model
        del tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    summary_file = os.path.join(RESULTS_DIR, f"summary_{timestamp}.csv")
    if summary:
        with open(summary_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
            writer.writeheader()
            writer.writerows(summary)
        if hf_repo:
            upload_to_hf(summary_file, hf_repo, hf_token)

    print(f"\n{'='*60}")
    print("  T5 BENCHMARK SUMMARY")
    print(f"{'='*60}")
    print(f"  {'Model':<20} {'Accuracy':>10} {'Correct':>8} / {'Total':>5} {'Errors':>7}")
    print(f"  {'-'*60}")
    for s in summary:
        if "error" in s and "accuracy" not in s:
            print(f"  {s['model']:<20} LOAD FAILED: {s['error']}")
        else:
            print(f"  {s['model']:<20} {s['accuracy']:>10} {s['correct']:>8} / {s['total']:>5} {s['errors']:>7}")
    print(f"\n  Summary saved to: {summary_file}")


# ---- CLI -------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark T5-type code models on refineID")
    parser.add_argument("--model", action="append", default=None,
                        help="Model name from MODEL_REGISTRY. Repeatable.")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--hf-repo", type=str, default=None)
    parser.add_argument("--hf-token", type=str, default=os.environ.get("HF_TOKEN"))
    parser.add_argument("--list-models", action="store_true")
    parser.add_argument("--debug", action="store_true",
                        help="Print the first 3 samples' truncated input + raw output for inspection.")

    args = parser.parse_args()

    if args.list_models:
        print("Available T5 models:")
        for name, meta in MODEL_REGISTRY.items():
            print(f"  {name:<20} {meta['id']}")
        sys.exit(0)

    run_benchmark(
        target_models=args.model,
        max_samples=args.max_samples,
        hf_repo=args.hf_repo,
        hf_token=args.hf_token,
        debug=args.debug,
    )
