"""
Benchmark google/diffusiongemma-26B-A4B-it (block-diffusion dLLM) on refineID.

DiffusionGemma denoises a generation canvas appended AFTER the prompt
(block-autoregressive); it cannot infill <|mask|> in-place like Dream/DreamOn.
There is also no FIM-tokenised base checkpoint -- only the instruct model.
So this engine mirrors the iterative per-site protocol of
benchmark_ar_models_fim.run_fim_on_sample, but through a chat prompt:

    1. take the FIRST remaining [MASK], mark it <FILL_HERE> (later sites stay
       as [MASK], earlier sites already hold their predictions),
    2. ask the model to answer with ONLY the identifier for <FILL_HERE>,
    3. substitute the prediction back and continue with the next site.

Per-site predictions feed the all-sites consistency gate exactly like every
other model in run_all_refineID_unified.py. PROTOCOL CAVEAT for the paper:
this is *prompted* identifier naming, not FIM/infill -- footnote it when
comparing against the base-model rows.

Requires a transformers release that ships the `diffusion_gemma` architecture.
The checkpoint config reports transformers_version 5.8.0.dev0; on the HPC cluster we pin
5.11.0 in a SEPARATE $UMBRELLA/pylibs_dgemma tree (the main env is pinned to
4.57.1 for Dream/DreamOn's custom modeling). See server/setup_dgemma_pylibs.sh.
The canonical class is `DiffusionGemmaForBlockDiffusion`; `.generate()` IS the
block-diffusion sampler (it uses the model's generation_config: canvas_length
256, entropy-bound sampler, <=48 denoising steps). Do NOT call diffusion_generate
(that is the Dream/DiffuCoder API, absent here).

Standalone smoke test (96GB GPU: bf16 weights are ~52GB, A40 will OOM):
    python experiments/benchmark_diffusiongemma.py --max-samples 5 --debug
    python experiments/benchmark_diffusiongemma.py --max-samples 5 --debug \
        --debug-dump results/diffusiongemma_smoke/debug.jsonl
Full runs go through the unified runner:
    python experiments/run_all_refineID_unified.py --only DiffusionGemma-26B-A4B
"""

import os
import sys
import csv
import re
import json
import argparse
import time

# ---- Configuration ---------------------------------------------------------

MODEL_ID = "google/diffusiongemma-26B-A4B-it"
DATA_PATH = "data/test.csv"
RESULTS_DIR = "results/diffusiongemma_smoke"
# Canvas length is 256 (config.json): block-diffusion denoises a full 256-token
# canvas. The single-identifier answer is short; with thinking disabled it lands
# well within one canvas. Keep >= one canvas so the answer is never starved.
MAX_NEW_TOKENS = 256
MAX_INPUT_TOKENS = 16384   # model does 256K; cap for speed, FIM-style window

FILL_MARK = "<FILL_HERE>"

PROMPT_TEMPLATE = (
    "The Java code below has occurrences of ONE identifier masked.\n"
    "Sites marked [MASK] are other occurrences of the same identifier; the "
    "site marked " + FILL_MARK + " is the one to name now.\n"
    "Reply with ONLY the Java identifier for " + FILL_MARK + " -- a single "
    "name, no explanation, no code, no quotes.\n\n"
    "```java\n{code}\n```"
)

# ---- Debug instrumentation --------------------------------------------------
# These are flipped on by the standalone smoke test (--debug). They let us
# capture the REAL generation behaviour from one the HPC cluster GPU run: the runtime type
# of model.generate()'s output, the canvas/sequence shapes, the rendered prompt,
# and the full raw/plain decode. Without this the empty-output cause cannot be
# pinned (the model can't run on the dev Mac).
DEBUG = False
_DEBUG_BUDGET = 6          # print diagnostics for at most this many sites
_DEBUG_SEEN = 0
DEBUG_RECORDS = []         # appended when DEBUG; main() can dump to JSONL


def _dbg(msg):
    if DEBUG:
        print(msg, flush=True)


# ---- Loading ----------------------------------------------------------------


def load_diffusiongemma(hf_id=MODEL_ID, hf_token=None):
    """Load (processor, model). Needs transformers with `diffusion_gemma`.

    `processor` is the full AutoProcessor when its image stack (torchvision) is
    available, else a bare AutoTokenizer carrying the same chat template -- the
    refineID task is text-only, so the tokenizer is sufficient. Callers detect
    which one via hasattr(processor, "tokenizer").
    """
    import transformers
    from transformers import AutoProcessor
    try:
        from transformers import DiffusionGemmaForBlockDiffusion as Cls
        cls_name = "DiffusionGemmaForBlockDiffusion"
    except ImportError:
        try:
            from transformers import AutoModelForMultimodalLM as Cls
            cls_name = "AutoModelForMultimodalLM"
        except ImportError:
            raise ImportError(
                "transformers " + transformers.__version__ + " has neither "
                "DiffusionGemmaForBlockDiffusion nor AutoModelForMultimodalLM. "
                "Install a newer transformers into a SEPARATE dir (do not "
                "touch the Dream/DreamOn-pinned pylibs): "
                "bash server/setup_dgemma_pylibs.sh")

    try:
        processor = AutoProcessor.from_pretrained(hf_id, token=hf_token)
        proc_kind = type(processor).__name__ + " (full processor)"
    except Exception as ex:
        # Gemma4Processor's image processor hard-imports torchvision (tfm 5.x).
        # refineID is text-only: the tokenizer carries the same chat template,
        # so fall back instead of dragging torchvision into the env.
        from transformers import AutoTokenizer
        print("AutoProcessor unavailable (" + str(ex)[:140]
              + ") -> text-only AutoTokenizer fallback", flush=True)
        processor = AutoTokenizer.from_pretrained(hf_id, token=hf_token)
        proc_kind = type(processor).__name__ + " (tokenizer fallback)"

    model = Cls.from_pretrained(
        hf_id, dtype="auto", device_map="auto", token=hf_token)
    model.eval()
    print("loaded %s via %s | processor=%s | transformers=%s"
          % (hf_id, cls_name, proc_kind, transformers.__version__), flush=True)
    return processor, model


# ---- Prompt construction ----------------------------------------------------


def _tokenizer_of(processor):
    return getattr(processor, "tokenizer", processor)


def _is_full_processor(processor):
    # full AutoProcessor wraps a .tokenizer; a bare AutoTokenizer does not.
    return hasattr(processor, "tokenizer")


def truncate_around_mark(code, tokenizer, max_tokens):
    """Token-truncate code around FILL_MARK, 60/40 prefix/suffix like FIM."""
    if FILL_MARK not in code:
        return code
    prefix, suffix = code.split(FILL_MARK, 1)
    prefix_budget = int(max_tokens * 0.6)
    suffix_budget = max_tokens - prefix_budget
    prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
    suffix_ids = tokenizer.encode(suffix, add_special_tokens=False)
    if len(prefix_ids) > prefix_budget:
        prefix = tokenizer.decode(prefix_ids[-prefix_budget:], skip_special_tokens=True)
    if len(suffix_ids) > suffix_budget:
        suffix = tokenizer.decode(suffix_ids[:suffix_budget], skip_special_tokens=True)
    return prefix + FILL_MARK + suffix


def build_messages(code_with_mark, is_processor):
    """Chat messages. Full processors expect typed content parts; a bare
    tokenizer's text chat template expects a plain string -- feeding it the
    list form renders the literal list repr (or nothing) and breaks the prompt.
    """
    prompt = PROMPT_TEMPLATE.format(code=code_with_mark)
    if is_processor:
        return [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
    return [{"role": "user", "content": prompt}]


def _apply_chat(processor, messages):
    """apply_chat_template with thinking disabled when the template supports it
    (keeps the 256-token canvas for the short answer instead of a reasoning
    monologue). Falls back gracefully if the kwarg is unknown."""
    base = dict(add_generation_prompt=True, tokenize=True,
                return_dict=True, return_tensors="pt")
    try:
        return processor.apply_chat_template(messages, enable_thinking=False, **base)
    except TypeError:
        return processor.apply_chat_template(messages, **base)


# ---- Prediction cleaning ----------------------------------------------------

# DiffusionGemma channel format: <|channel>thought\n[reasoning]<channel|>final
_THOUGHT_RE = re.compile(r"<\|channel>thought\n.*?<channel\|>", re.DOTALL)
_IDENT_RE = re.compile(r"[A-Za-z_$][A-Za-z0-9_$]*")
_QUOTED_RE = re.compile(r"[`\"']([A-Za-z_$][A-Za-z0-9_$]*)[`\"']")


def clean_prediction(raw):
    """Extract one Java identifier from raw decoded output."""
    text = _THOUGHT_RE.sub("", raw)
    if "<channel|>" in text:                 # unmatched closing tag
        text = text.split("<channel|>")[-1]
    text = text.replace("<|channel>thought", " ")
    text = re.sub(r"<[^>\n]{0,40}>", " ", text)   # drop remaining tag-likes
    # drop code-fence marker lines (```java etc.) so we don't grab "java"
    lines = [ln for ln in text.splitlines() if not ln.strip().startswith("```")]
    text = "\n".join(lines)
    # instruct models often answer 'the identifier is "x"' -- prefer quoted
    m = _QUOTED_RE.search(text)
    if m:
        return m.group(1)
    for line in lines:
        line = line.strip().strip("`\"'. ")
        if not line:
            continue
        m = _IDENT_RE.search(line)
        if m:
            return m.group(0)
    m = _IDENT_RE.search(text)
    return m.group(0) if m else ""


def _extract(raw, plain):
    """Pick the best identifier from the two decodes.

    When the channel/thought structure is present we trust the channel-aware
    pass on `raw` (delimiters intact); skip_special_tokens=True can merge the
    thought and answer with no delimiter, so `plain` is the fallback there.
    Otherwise prefer the clean `plain` decode.
    """
    if "channel" in raw.lower():
        return clean_prediction(raw) or clean_prediction(plain)
    return clean_prediction(plain) or clean_prediction(raw)


# ---- Single-sample inference (signature mirrors run_fim_on_sample) ----------


def run_dgemma_on_sample(masked_code, model, processor, max_input_tokens=MAX_INPUT_TOKENS):
    """Fill all [MASK] sites via iterative per-site chat prompts.

    Returns (predictions, raw_predictions, prompts, final_code) like
    benchmark_ar_models_fim.run_fim_on_sample.
    """
    import torch
    global _DEBUG_SEEN
    tokenizer = _tokenizer_of(processor)
    is_proc = _is_full_processor(processor)
    current_code = masked_code
    predictions, raw_predictions, prompts = [], [], []
    mask_count = current_code.count("[MASK]")

    for _ in range(mask_count):
        parts = current_code.split("[MASK]", 1)
        prefix = parts[0]
        suffix = parts[1] if len(parts) > 1 else ""

        code_with_mark = truncate_around_mark(
            prefix + FILL_MARK + suffix, tokenizer, max_input_tokens)
        messages = build_messages(code_with_mark, is_proc)
        prompts.append(code_with_mark)

        inputs = _apply_chat(processor, messages).to(model.device)
        input_len = inputs["input_ids"].shape[-1]
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS)

        # Normalize: generate() may return a plain LongTensor (batch, seq) OR a
        # ModelOutput whose .sequences is (batch, seq). Take the sequence row,
        # then slice off the prompt. (The old code did out[0][input_len:], which
        # silently empties to "" when out is a ModelOutput -- the all-empty bug.)
        seq = out.sequences if hasattr(out, "sequences") else out
        if hasattr(seq, "dim") and seq.dim() == 2:
            seq = seq[0]
        new_ids = seq[input_len:]

        raw = processor.decode(new_ids, skip_special_tokens=False)
        plain = processor.decode(new_ids, skip_special_tokens=True)
        raw_predictions.append(raw)

        pred = _extract(raw, plain)
        predictions.append(pred)
        current_code = prefix + pred + suffix

        if DEBUG and _DEBUG_SEEN < _DEBUG_BUDGET:
            _DEBUG_SEEN += 1
            rendered = processor.decode(inputs["input_ids"][0], skip_special_tokens=False)
            rec = {
                "out_type": type(out).__name__,
                "seq_shape": list(getattr(seq, "shape", [])),
                "input_len": int(input_len),
                "new_len": int(new_ids.shape[-1]) if hasattr(new_ids, "shape") else None,
                "is_processor": is_proc,
                "pred": pred,
                "raw": raw,
                "plain": plain,
                "prompt_tail": rendered[-600:],
            }
            DEBUG_RECORDS.append(rec)
            _dbg("  --- dgemma debug -------------------------------------------")
            _dbg("    out_type=%s seq_shape=%s input_len=%d new_len=%s is_proc=%s"
                 % (rec["out_type"], rec["seq_shape"], input_len, rec["new_len"], is_proc))
            _dbg("    prompt_tail=%r" % rec["prompt_tail"])
            _dbg("    raw  =%r" % raw)
            _dbg("    plain=%r" % plain)
            _dbg("    pred =%r" % pred)

    return predictions, raw_predictions, prompts, current_code


# ---- Standalone smoke test ---------------------------------------------------


def load_data(data_path, max_samples=None):
    csv.field_size_limit(2**31 - 1)
    rows = []
    with open(data_path, "r", encoding="utf-8") as f:
        for i, row in enumerate(csv.reader(f)):
            if max_samples is not None and i >= max_samples:
                break
            if len(row) < 3:
                continue
            rows.append({"id": row[0], "masked_code": row[1], "target": row[2].strip()})
    return rows


def main():
    global MAX_NEW_TOKENS, DEBUG
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-id", default=MODEL_ID)
    ap.add_argument("--data", default=DATA_PATH)
    ap.add_argument("--max-samples", type=int, default=None)
    ap.add_argument("--max-new-tokens", type=int, default=None,
                    help="override MAX_NEW_TOKENS (canvas budget per site)")
    ap.add_argument("--hf-token", default=os.environ.get("HF_TOKEN"))
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--debug-dump", default=None,
                    help="write per-site debug records (type/shapes/raw/plain) as JSONL")
    args = ap.parse_args()

    if args.max_new_tokens:
        MAX_NEW_TOKENS = args.max_new_tokens
    if args.debug or args.debug_dump:
        DEBUG = True

    data = load_data(args.data, args.max_samples)
    print(f"Loaded {len(data)} samples from {args.data}", flush=True)

    t0 = time.time()
    processor, model = load_diffusiongemma(args.model_id, hf_token=args.hf_token)
    print(f"Loaded {args.model_id} in {time.time()-t0:.1f}s", flush=True)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, "DiffusionGemma-26B-A4B.csv")
    fields = ["id", "ground_truth", "n_total_masks", "predictions",
              "first_pred", "first_correct", "error"]
    n_ok = 0
    n_nonempty = 0
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in data:
            gt = row["target"]
            n_masks = row["masked_code"].count("[MASK]")
            try:
                preds, raws, _, _ = run_dgemma_on_sample(
                    row["masked_code"], model, processor)
                first = preds[0] if preds else ""
                n_ok += int(first == gt)
                n_nonempty += int(any(p for p in preds))
                w.writerow({"id": row["id"], "ground_truth": gt,
                            "n_total_masks": n_masks,
                            "predictions": "|".join(preds), "first_pred": first,
                            "first_correct": (first == gt), "error": ""})
                if args.debug:
                    print(f"  [{row['id']}] gt={gt!r} preds={preds}", flush=True)
            except Exception as ex:
                w.writerow({"id": row["id"], "ground_truth": gt,
                            "n_total_masks": n_masks, "predictions": "",
                            "first_pred": "", "first_correct": False,
                            "error": str(ex)[:200]})
                print(f"  error on {row['id']}: {ex}", flush=True)
            f.flush()

    if args.debug_dump and DEBUG_RECORDS:
        os.makedirs(os.path.dirname(args.debug_dump) or ".", exist_ok=True)
        with open(args.debug_dump, "w", encoding="utf-8") as df:
            for rec in DEBUG_RECORDS:
                df.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"wrote {len(DEBUG_RECORDS)} debug records -> {args.debug_dump}", flush=True)

    print(f"first-site EM {n_ok}/{len(data)} | samples with any non-empty pred "
          f"{n_nonempty}/{len(data)}  ->  {out_path}", flush=True)
    if n_nonempty == 0 and data:
        print("!! ALL predictions empty. Inspect the debug dump / raw above: if "
              "raw is non-empty the bug is in extraction (_extract/clean_prediction); "
              "if raw is empty the bug is in generation (prompt/template/canvas).",
              flush=True)
        # Non-zero exit = a GATE: in the SLURM chain (dgemma_rq1_all.sh), the
        # afterok dependency then aborts the 24h full run instead of producing
        # another all-empty CSV. The debug output above is already in the log.
        sys.exit(2)


if __name__ == "__main__":
    main()
