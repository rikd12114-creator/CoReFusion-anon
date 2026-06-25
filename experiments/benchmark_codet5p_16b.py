"""
Benchmark Salesforce/codet5p-16b (and 2B/6B siblings) on the refineID task.

The full-size CodeT5+ family (2B/6B/16B) uses the CodeGen tokenizer and
ships custom remote code that is incompatible with the sentinel-based
T5 prompting protocol used by smaller CodeT5/CodeT5+ checkpoints. So we
follow the official model-card pattern verbatim:

    # Load (per model card)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        checkpoint,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    ).to(device)

    # Generate (per model card)
    encoding = tokenizer(text, return_tensors="pt").to(device)
    encoding["decoder_input_ids"] = encoding["input_ids"].clone()
    outputs = model.generate(**encoding, max_length=...)

This is prefix-only completion: the encoder gets the code BEFORE the
[MASK] slot, the decoder is primed with the same tokens and continues.
The decoder output starts by echoing the prefix and then emits new
tokens; we trim the echoed prefix and take the first identifier from
the continuation as the prediction.

NOTE on methodology: prefix-only completion is asymmetric vs. the small
CodeT5/CodeT5+ benchmark (which sees both prefix AND suffix via the
<extra_id_0> sentinel) and vs. the AR FIM benchmark (which sees both
sides via FIM tokens). This is unavoidable -- CodeT5+ 16B's stage-2
pretraining was pure CLM and the CodeGen tokenizer has no sentinel.
We keep the result columns identical so summaries can still be
concatenated, but interpret 16B accuracy as a CLM upper-bound rather
than a direct apples-to-apples comparison.

Data:   data/test.csv
Metric: Exact Match on the first masked identifier.

Usage:
    python experiments/benchmark_codet5p_16b.py
    python experiments/benchmark_codet5p_16b.py --model CodeT5p-2B
    python experiments/benchmark_codet5p_16b.py --max-samples 50 --debug
    python experiments/benchmark_codet5p_16b.py --hf-repo your-name/your-repo
"""

import os
import sys
import csv
import re
import gc
import argparse
import time
from datetime import datetime

import json

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers.dynamic_module_utils import get_class_from_dynamic_module
from huggingface_hub import hf_hub_download
from tqdm import tqdm

try:
    from huggingface_hub import HfApi
    HAS_HF_HUB = True
except ImportError:
    HAS_HF_HUB = False

# ---- Configuration ---------------------------------------------------------

DATA_PATH = "data/test.csv"
# Share the results directory with the AR FIM and small T5 benchmarks
# so downstream summary loaders can treat 16B rows as just another row.
RESULTS_DIR = "results/ar_fim_benchmark"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_NEW_TOKENS = 16
MAX_INPUT_TOKENS = 2048  # CodeT5+ 2B/6B/16B encoder supports 2048

# ---- Model Registry --------------------------------------------------------

MODEL_REGISTRY = {
    "CodeT5p-2B":  {"id": "Salesforce/codet5p-2b",  "max_ctx": 2048},
    "CodeT5p-6B":  {"id": "Salesforce/codet5p-6b",  "max_ctx": 2048},
    "CodeT5p-16B": {"id": "Salesforce/codet5p-16b", "max_ctx": 2048},
}
DEFAULT_MODEL = "CodeT5p-16B"


# ---- Prediction Cleaning ---------------------------------------------------

def clean_identifier(text):
    """Extract the first valid Java identifier from a span."""
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

def truncate_prefix(prefix, tokenizer, max_tokens):
    """Keep the *tail* of prefix (closest to the [MASK] slot) within budget.

    For prefix-only completion the relevant context is right before the
    slot, so when we have to drop tokens we drop from the head.
    """
    ids = tokenizer.encode(prefix, add_special_tokens=False)
    if len(ids) > max_tokens:
        ids = ids[-max_tokens:]
        prefix = tokenizer.decode(ids, skip_special_tokens=True)
    return prefix


# ---- Single-sample Inference -----------------------------------------------

def run_codet5p_large_on_sample(masked_code, model, tokenizer, max_input_tokens):
    """Iteratively fill each [MASK] via prefix-only completion.

    Mirrors the multi-mask iteration shape used by the AR FIM and small
    T5 benchmarks (one model call per [MASK], replace, continue). Per
    [MASK] the encoder sees only the *prefix* (everything before the
    slot), unfilled later [MASK]s included as literal text, and the
    decoder is primed with input_ids.clone() per the model card.

    Returns: predictions, raw_predictions, prompts, final_code
    -- same shape as benchmark_t5_models.run_t5_on_sample.
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

        # Reserve room for generated tokens; truncate prefix from the head.
        budget = max(1, max_input_tokens - MAX_NEW_TOKENS - 4)
        prefix_t = truncate_prefix(prefix, tokenizer, budget)
        prompts.append(prefix_t)

        encoding = tokenizer(
            prefix_t,
            return_tensors="pt",
            truncation=True,
            max_length=budget,
        ).to(model.device)
        # Per model card: prime the decoder with the same tokens.
        encoding["decoder_input_ids"] = encoding["input_ids"].clone()

        gen_kwargs = dict(
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            num_beams=1,
        )
        if tokenizer.pad_token_id is not None:
            gen_kwargs["pad_token_id"] = tokenizer.pad_token_id
        elif tokenizer.eos_token_id is not None:
            gen_kwargs["pad_token_id"] = tokenizer.eos_token_id

        with torch.no_grad():
            outputs = model.generate(**encoding, **gen_kwargs)

        # The primed decoder echoes the prefix; only continuation is new.
        # generate() prepends decoder_start_token_id when decoder_input_ids
        # don't already start with it, shifting the echo by one -- locate the
        # echo in the output instead of assuming it sits at position 0.
        prefix_len = encoding["decoder_input_ids"].shape[1]
        out_ids = outputs[0]
        dec_ids = encoding["decoder_input_ids"][0]
        start = prefix_len
        if len(out_ids) > prefix_len and not torch.equal(out_ids[:prefix_len], dec_ids) \
                and torch.equal(out_ids[1:prefix_len + 1], dec_ids):
            start = prefix_len + 1
        new_token_ids = out_ids[start:]
        raw_pred = tokenizer.decode(new_token_ids, skip_special_tokens=False)
        raw_predictions.append(raw_pred)

        cleaned = (
            raw_pred
            .replace("<pad>", "")
            .replace("</s>", "")
            .replace("<|endoftext|>", "")
        )
        pred = clean_identifier(cleaned)
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


# ---- Custom config loader (transformers 4.46+ workaround) ------------------

def load_codet5p_config(checkpoint):
    """Manually instantiate the CodeT5pConfig, bypassing transformers' loader.

    Why this exists:
        In transformers >= 4.46, PretrainedConfig.from_dict() (which
        AutoConfig.from_pretrained and AutoModelForSeq2SeqLM.from_pretrained
        both go through) preprocesses the dict in ways that drop or rename
        certain keys before they reach the custom config's __init__. The
        Salesforce CodeT5pConfig has this guard:

            if "encoder" not in kwargs or "decoder" not in kwargs:
                raise ValueError("Config has to be initialized with "
                                 "encoder and decoder config")

        and those nested keys never make it through. The error you see is:
            "Config has to be initialized with encoder and decoder config"

    Fix:
        Download config.json from the repo, resolve the custom CodeT5pConfig
        class via the repo's auto_map (downloads configuration_codet5p.py
        on demand, the same way trust_remote_code does), and call
        config_class(**config_dict) directly with the raw JSON dict so the
        encoder/decoder sub-dicts survive intact.
    """
    config_file = hf_hub_download(checkpoint, "config.json")
    with open(config_file, "r") as f:
        config_dict = json.load(f)

    auto_map = config_dict.get("auto_map", {})
    config_class_ref = auto_map.get("AutoConfig")
    if not config_class_ref:
        raise RuntimeError(
            f"{checkpoint}/config.json has no auto_map.AutoConfig entry; "
            "cannot resolve the custom CodeT5pConfig class."
        )

    # get_class_from_dynamic_module downloads the .py file referenced in
    # auto_map (e.g. "configuration_codet5p.CodeT5pConfig") and returns the
    # class. This is the same machinery trust_remote_code uses.
    config_class = get_class_from_dynamic_module(config_class_ref, checkpoint)
    return config_class(**config_dict)


def ensure_generate_compatible(model):
    """Disarm the legacy generation-config branch in transformers >= 4.50.

    On transformers 4.57, generate() -> _prepare_generation_config() calls
    config._get_non_default_generation_parameters(), which instantiates
    `self.__class__()` with no kwargs. CodeT5pConfig *asserts* that encoder/
    decoder kwargs are present, and transformers only catches ValueError --
    so every generate() call dies with
        "Config has to be initialized with encoder and decoder config".
    That branch is only entered while generation_config still carries the
    `_from_model_config` flag; clearing the flag skips it entirely. No-op on
    older transformers where generate() never re-derives the config.
    """
    gen_cfg = getattr(model, "generation_config", None)
    if gen_cfg is not None and getattr(gen_cfg, "_from_model_config", False):
        gen_cfg._from_model_config = False
    return model


# ---- Main Benchmark --------------------------------------------------------

def run_benchmark(model_name, max_samples=None, hf_repo=None, hf_token=None, debug=False):
    if model_name not in MODEL_REGISTRY:
        print(f"ERROR: '{model_name}' not in registry. Available: {list(MODEL_REGISTRY)}")
        return

    meta = MODEL_REGISTRY[model_name]
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print(f"Loading data from {DATA_PATH}...")
    data = load_data(DATA_PATH, max_samples=max_samples)
    print(f"Loaded {len(data)} samples.")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print(f"\n{'='*60}")
    print(f"  Model: {model_name}")
    print(f"  HF ID: {meta['id']}")
    print(f"  Mode:  prefix-only completion (model-card pattern)")
    print(f"{'='*60}")

    # ── Load model (model-card pattern + manual config workaround) ──────
    # The official model card says:
    #     model = AutoModelForSeq2SeqLM.from_pretrained(
    #         checkpoint, torch_dtype=torch.float16,
    #         low_cpu_mem_usage=True, trust_remote_code=True,
    #     ).to(device)
    # But on transformers >= 4.46 that path crashes with
    #   "Config has to be initialized with encoder and decoder config"
    # because the loader drops the nested encoder/decoder dicts before
    # they reach CodeT5pConfig.__init__. We pre-build the config manually
    # (load_codet5p_config) and hand it to from_pretrained so it skips
    # config re-instantiation. Everything else stays exactly as the
    # model card prescribes (fp16 + low_cpu_mem_usage + .to(device)).
    t0 = time.time()
    try:
        tokenizer = AutoTokenizer.from_pretrained(meta["id"], trust_remote_code=True)
        config = load_codet5p_config(meta["id"])
        model = AutoModelForSeq2SeqLM.from_pretrained(
            meta["id"],
            config=config,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        ).to(DEVICE)
        model.eval()
        ensure_generate_compatible(model)
        print(f"  Model loaded in {time.time() - t0:.1f}s")
    except Exception as e:
        print(f"  FAILED to load model: {e}")
        # Persist the failure as a one-row summary so the run is auditable.
        summary_file = os.path.join(RESULTS_DIR, f"summary_{model_name}_{timestamp}.csv")
        with open(summary_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["model", "hf_id", "error"])
            writer.writeheader()
            writer.writerow({"model": model_name, "hf_id": meta["id"], "error": str(e)})
        return

    max_ctx = min(meta.get("max_ctx", MAX_INPUT_TOKENS), MAX_INPUT_TOKENS)

    # ── Inference ───────────────────────────────────────────────────────
    results = []
    correct = 0
    errors = 0
    truncated_count = 0

    for sample_idx, row in enumerate(tqdm(data, desc=f"  {model_name}")):
        item_id = row["id"]
        masked_code = row["masked_code"]
        ground_truth = row["target"]

        try:
            preds, raw_preds, prompts, final_code = run_codet5p_large_on_sample(
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
                tail = first_prompt[-300:] if len(first_prompt) > 300 else first_prompt
                print(f"  prefix tail     : ...{tail}")
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

    # ── Save results (same schema as the AR FIM / T5 benchmarks) ────────
    out_file = os.path.join(RESULTS_DIR, f"{model_name}_refineID_fim_{timestamp}.csv")
    with open(out_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "id", "ground_truth", "prediction", "correct",
            "mask_count", "all_predictions",
            "raw_prediction", "all_raw_predictions", "error",
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

    # ── Per-run summary file ────────────────────────────────────────────
    summary_file = os.path.join(RESULTS_DIR, f"summary_{model_name}_{timestamp}.csv")
    with open(summary_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "model", "hf_id", "type", "accuracy", "correct", "total", "errors", "results_file",
        ])
        writer.writeheader()
        writer.writerow({
            "model": model_name,
            "hf_id": meta["id"],
            "type": "codet5p_large",
            "accuracy": f"{accuracy:.4f}",
            "correct": correct,
            "total": len(data),
            "errors": errors,
            "results_file": out_file,
        })
    if hf_repo:
        upload_to_hf(summary_file, hf_repo, hf_token)
    print(f"  Summary:   {summary_file}")

    # ── Cleanup ─────────────────────────────────────────────────────────
    del model
    del tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


# ---- CLI -------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark Salesforce/codet5p-{2,6,16}b on refineID using "
                    "the official model-card prefix-completion pattern."
    )
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL,
                        choices=list(MODEL_REGISTRY.keys()),
                        help=f"Which CodeT5+ large variant to run (default: {DEFAULT_MODEL}).")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Limit number of test samples (for quick smoke tests).")
    parser.add_argument("--hf-repo", type=str, default=None,
                        help="Optional HF dataset repo to upload results to.")
    parser.add_argument("--hf-token", type=str, default=os.environ.get("HF_TOKEN"),
                        help="HF write token (or set $HF_TOKEN).")
    parser.add_argument("--debug", action="store_true",
                        help="Print first 3 samples' truncated input + raw output.")
    parser.add_argument("--list-models", action="store_true")

    args = parser.parse_args()

    if args.list_models:
        print("Available models:")
        for name, meta in MODEL_REGISTRY.items():
            print(f"  {name:<14} {meta['id']}")
        sys.exit(0)

    run_benchmark(
        model_name=args.model,
        max_samples=args.max_samples,
        hf_repo=args.hf_repo,
        hf_token=args.hf_token,
        debug=args.debug,
    )
