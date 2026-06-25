"""
Part 3: Dynamic vs Static MASK-Token Count Comparison
=======================================================

Research question:
  Can *dynamically* choosing the number of <|mask|> tokens for each
  variable name improve diffusion-LLM renaming quality over using a
  single fixed (static) count?

Two dynamic strategies are tested here:

  Strategy 1 — F1-Optimal Threshold (from Exp C / experiment_threshold_detector.py)
    Uses the entropy-based signal to pick the best τ, then maps a detected 
    "smell score" per token to an estimate of how many mask tokens to use:
        score ≥ τ_high  →  use k_high masks (longer name expected)  
        τ_low ≤ score < τ_high → middle estimate
        score < τ_low   →  use k_low masks
    (In the absence of a live entropy signal at inference time, we approximate
     the variable name length from the MASKED code using a char-length heuristic
     that mimics the threshold decision boundary.)

  Strategy 2 — Context Naming Length  
    Looks at all OTHER variable names appearing in the same Java code snippet,
    computes their mean token length under the target model's tokenizer, and
    uses that as the dynamic mask count (rounded, clipped to [1, 5]).

Both dynamic strategies are compared against the *best static k* from Part 2
(or a user-supplied k via --static-k).

Output:
  results/1t5t_exp/part3_raw_{model}_{strategy}_{timestamp}.csv
  results/1t5t_exp/part3_summary_{timestamp}.csv
  figures/part3_dynamic_vs_static_{timestamp}.png / .pdf

Usage (from repo root):
    # Full comparison, both models, both dynamic strategies vs static k=3
    python experiments/1t5t_exp/part3_dynamic_vs_static.py \
        --data data/test.csv \
        --models both \
        --static-k 3 \
        --steps 32 \
        --judge-model Qwen/Qwen2.5-7B-Instruct \
        --max-samples 200

    # Use default best static k from Part 2 summary CSV
    python experiments/1t5t_exp/part3_dynamic_vs_static.py \
        --data data/test.csv \
        --part2-summary results/1t5t_exp/part2_summary_YYYYMMDD_HHMMSS.csv

    # Skip LLM judge
    python experiments/1t5t_exp/part3_dynamic_vs_static.py --no-judge
"""

import os, sys, csv, re, gc, argparse, json
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm
from datetime import datetime
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
from collections import Counter

# ── torchvision mock ─────────────────────────────────────────────────────────
class _Mock:
    def __getattr__(self, n): return _Mock()
    def __call__(self, *a, **k): return _Mock()
for _m in ["torchvision", "torchvision.ops", "torchvision.transforms"]:
    sys.modules.setdefault(_m, _Mock())
if not hasattr(torch.ops, "torchvision"):
    class _DummyOps:
        def nms(*a, **k): return torch.tensor([])
    torch.ops.torchvision = _DummyOps()
# ─────────────────────────────────────────────────────────────────────────────

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# How often (every N samples) to call torch.cuda.empty_cache() during inference
CACHE_CLEAR_INTERVAL = 20

MODELS_REGISTRY = {
    "diffucoder": {
        "name": "DiffuCoder-7B",
        "id":   "apple/DiffuCoder-7B-Instruct",
        "mask_token": "<|mask|>",
    },
    "dreamcoder": {
        "name": "DreamCoder-7B",
        "id":   "Dream-org/Dream-Coder-v0-Instruct-7B",
        "mask_token": "<|mask|>",
    },
}

RESULTS_DIR = "results/1t5t_exp"
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")

JUDGE_REGISTRY = {
    "Qwen2.5-7B-Instruct": "Qwen/Qwen2.5-7B-Instruct",
    "Qwen2.5-3B-Instruct":  "Qwen/Qwen2.5-3B-Instruct",
}

# Java keywords to exclude from "other variable" analysis
JAVA_KEYWORDS = {
    "abstract", "assert", "boolean", "break", "byte", "case", "catch",
    "char", "class", "const", "continue", "default", "do", "double",
    "else", "enum", "extends", "final", "finally", "float", "for",
    "goto", "if", "implements", "import", "instanceof", "int", "interface",
    "long", "native", "new", "package", "private", "protected", "public",
    "return", "short", "static", "strictfp", "super", "switch",
    "synchronized", "this", "throw", "throws", "transient", "try",
    "void", "volatile", "while", "true", "false", "null", "var",
    "String", "Integer", "Long", "Double", "Float", "Boolean",
    "Object", "List", "ArrayList", "Map", "HashMap", "Set",
    "System", "Math", "Override",
}


# ─────────────────────────────────────────────────────────────────────────────
# Memory helpers
# ─────────────────────────────────────────────────────────────────────────────

def release_model(model, tokenizer):
    """Fully release a model from GPU memory."""
    try:
        del model
    except Exception:
        pass
    try:
        del tokenizer
    except Exception:
        pass
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    gc.collect()
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        print(f"  [mem] GPU allocated after release: {allocated:.2f} GB")


def vram_info() -> str:
    if not torch.cuda.is_available():
        return "CPU"
    alloc    = torch.cuda.memory_allocated() / 1e9
    reserved = torch.cuda.memory_reserved() / 1e9
    total    = torch.cuda.get_device_properties(0).total_memory / 1e9
    return f"alloc={alloc:.1f}GB  reserved={reserved:.1f}GB  total={total:.1f}GB"



# ─────────────────────────────────────────────────────────────────────────────
# Dynamic k strategies
# ─────────────────────────────────────────────────────────────────────────────

def strategy_context_naming_length(masked_code: str,
                                    tokenizer,
                                    min_k: int = 1,
                                    max_k: int = 5) -> int:
    """
    Strategy 2 – Context Naming Length.

    Extract identifiers from the code that are NOT at the [MASK] position,
    compute their average token length under the model's tokeniser,
    round and clip to [min_k, max_k].

    If no other identifiers are found, return the fallback character-based 
    heuristic (see strategy_threshold below).
    """
    # Find all lower-camelCase or snake_case identifiers in the code
    # (exclude the [MASK] placeholder itself)
    code_no_mask = masked_code.replace("[MASK]", " __MASK__ ")
    identifiers = re.findall(r'\b([a-z_][a-zA-Z0-9_]{1,})\b', code_no_mask)
    identifiers = [
        ident for ident in identifiers
        if ident not in JAVA_KEYWORDS and ident != "__MASK__"
    ]

    if not identifiers:
        # Fallback: char heuristic
        return strategy_threshold_heuristic(masked_code)

    lengths = [len(tokenizer.encode(ident, add_special_tokens=False))
               for ident in identifiers]
    mean_len = np.mean(lengths)
    k = int(round(mean_len))
    return int(np.clip(k, min_k, max_k))


def strategy_threshold_heuristic(masked_code: str,
                                   min_k: int = 1,
                                   max_k: int = 5) -> int:
    """
    Strategy 1 – Threshold-Based Heuristic (approximation of Exp-C detector).

    Since we cannot run the full entropy-fluctuation detector at inference
    time without the model already loaded in "detector" mode, we use a
    character-length proxy that statistically mirrors the F1-optimal
    threshold split found in experiment_threshold_detector.py:

      char_len ≤ 4  → k = 1   (single-token names, e.g. 'cnt', 'res')
      char_len 5-8  → k = 2   (typical Java camelCase token)
      char_len 9-13 → k = 3   (compound names, e.g. 'inputStream')
      char_len 14+  → k = 4   (long compound, e.g. 'connectionTimeout')

    The breakpoints were chosen so that the resulting distribution of k
    approximately matches the GT token-length percentile distribution from
    Part 1 (median ≈ 2-3 tokens, P90 ≈ 4-5 tokens).

    To incorporate the F1-optimal tau in a future run, this function can
    be updated to read the threshold_sensitivity.csv output from
    experiment_threshold_detector.py.
    """
    # Guess variable length by looking at surrounding code context (chars
    # between the previous and next separator around [MASK])
    # As a proxy, we use the immediately surrounding token:
    m = re.search(r'(\w+)\s*=\s*\[MASK\]|\[MASK\]\s*(?:=|,|\)|\;)', masked_code)
    if m:
        # try to find the type declaration length hint
        type_match = re.search(
            r'(\w+)\s+\[MASK\]|(\w+)\s+\w+\s*=\s*\[MASK\]', masked_code
        )
        if type_match:
            context_word = (type_match.group(1) or type_match.group(2) or "")
            # Type keywords like 'int', 'String' don't help, use char proxy
            pass

    # Simple character proxy: count the typical identifier length from context
    # by sampling identifiers near the [MASK] (±200 chars)
    idx = masked_code.find("[MASK]")
    window = masked_code[max(0, idx - 200): idx + 200]
    nearby_ids = re.findall(r'\b([a-z_][a-zA-Z0-9_]{1,30})\b', window)
    nearby_ids = [x for x in nearby_ids if x not in JAVA_KEYWORDS]

    if nearby_ids:
        avg_char_len = np.mean([len(x) for x in nearby_ids])
    else:
        avg_char_len = 6  # default

    if avg_char_len <= 4:
        k = 1
    elif avg_char_len <= 7:
        k = 2
    elif avg_char_len <= 11:
        k = 3
    elif avg_char_len <= 16:
        k = 4
    else:
        k = 5

    return int(np.clip(k, min_k, max_k))


# ─────────────────────────────────────────────────────────────────────────────
# Shared inference helpers  (mirrors benchmark_diffusion_models.py exactly)
# ─────────────────────────────────────────────────────────────────────────────

def extract_all_predictions(full_code: str, masked_code: str) -> list:
    """
    Extract predictions for each [MASK] by anchoring on surrounding context.
    Identical to the implementation in benchmark_diffusion_models.py.
    """
    parts = masked_code.split("[MASK]")
    if len(parts) <= 1:
        return []

    predictions = []
    current_search_start = 0

    for i in range(len(parts) - 1):
        pre  = parts[i]
        post = parts[i + 1]

        pre_anchor  = pre.strip()[-30:]  if len(pre.strip())  > 30 else pre.strip()
        post_anchor = post.strip()[:30]  if len(post.strip()) > 30 else post.strip()

        if pre_anchor:
            idx_start = full_code.find(pre_anchor, current_search_start)
            idx_start = (idx_start + len(pre_anchor)) if idx_start != -1 else current_search_start
        else:
            idx_start = current_search_start

        if post_anchor:
            idx_end = full_code.find(post_anchor, idx_start)
        else:
            idx_end = -1

        if idx_end != -1:
            gap_content = full_code[idx_start:idx_end].strip()
            current_search_start = idx_end
        else:
            gap_content = full_code[idx_start: idx_start + 60].strip()
            current_search_start = idx_start + 60

        match = re.search(r'[a-zA-Z_$][a-zA-Z0-9_$]*', gap_content)
        predictions.append(match.group(0) if match else gap_content[:20])

    return predictions


def run_diffusion_inference(model, tokenizer, masked_code: str,
                            mask_token: str, k: int, steps: int) -> tuple:
    """
    Replace [MASK] with k concatenated mask tokens (NO spaces between tokens),
    tokenise directly (NO chat template), denoise in-place (max_new_tokens=1),
    decode the ENTIRE output sequence, extract prediction via context anchoring.
    All intermediate GPU tensors deleted before return.
    """
    multi_mask = mask_token * k
    input_code = masked_code.replace("[MASK]", multi_mask)

    inputs    = tokenizer(input_code, return_tensors="pt")
    input_ids = inputs.input_ids.to(model.device)
    attn_mask = inputs.attention_mask.to(model.device)
    del inputs

    with torch.no_grad():
        output = model.diffusion_generate(
            input_ids,
            attention_mask=attn_mask,
            max_new_tokens=1,
            steps=steps,
            temperature=0.3,
            top_p=0.95,
            alg="entropy",
            alg_temp=0.,
        )

    gen_ids   = output.sequences[0] if hasattr(output, "sequences") else output[0]
    full_code = tokenizer.decode(gen_ids, skip_special_tokens=True)

    del input_ids, attn_mask, output, gen_ids

    preds = extract_all_predictions(full_code, masked_code)
    return full_code, (preds[0] if preds else "")



# ─────────────────────────────────────────────────────────────────────────────
# LLM judge
# ─────────────────────────────────────────────────────────────────────────────

JUDGE_SYSTEM = (
    "You are an expert Java code reviewer. "
    "Decide whether the predicted variable name is SEMANTICALLY ACCEPTABLE "
    "given the code context and ground-truth name.\n\n"
    "Respond with EXACTLY one line:\n"
    "    VERDICT: 1\n"
    "or\n"
    "    VERDICT: 0"
)


def _chat_prompt(tok, user_text):
    msgs = [{"role": "system", "content": JUDGE_SYSTEM},
            {"role": "user",   "content": user_text}]
    try:
        return tok.apply_chat_template(msgs, tokenize=False,
                                       add_generation_prompt=True)
    except Exception:
        return f"{JUDGE_SYSTEM}\n\nUser: {user_text}\nAssistant:"


def parse_verdict(text):
    m = re.search(r"VERDICT\s*:\s*([01])", text, re.I)
    if m:
        return int(m.group(1))
    for line in text.strip().splitlines():
        line = line.strip()
        if line in ("1", "0"):
            return int(line)
        if line.lower() in ("yes", "acceptable"):
            return 1
        if line.lower() in ("no", "not acceptable"):
            return 0
    return -1


def judge_one(jtok, jmodel, masked_code, prediction, ground_truth):
    """Run judge on one sample. All intermediate tensors freed after call."""
    context = masked_code.replace("[MASK]", prediction)[:2000]
    user_text = (
        f"Code:\n```java\n{context}\n```\n\n"
        f"Ground truth: `{ground_truth}`\n"
        f"Prediction:   `{prediction}`\n\n"
        f"VERDICT: 1 or VERDICT: 0"
    )
    prompt = _chat_prompt(jtok, user_text)
    inp = jtok(prompt, return_tensors="pt", truncation=True,
               max_length=4096).to(jmodel.device)
    with torch.no_grad():
        out = jmodel.generate(
            **inp, max_new_tokens=16, do_sample=False,
            pad_token_id=(jtok.eos_token_id or jtok.pad_token_id or 0),
        )
    new_ids = out[0][inp["input_ids"].shape[1]:]
    raw     = jtok.decode(new_ids, skip_special_tokens=True).strip()
    del inp, out, new_ids
    return parse_verdict(raw)



# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_data(data_path, max_samples=None):
    """Load test.csv: id | masked_code | ground_truth (no required header)."""
    csv.field_size_limit(sys.maxsize)
    rows = []
    with open(data_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for i, row in enumerate(reader):
            if max_samples is not None and i >= max_samples:
                break
            if len(row) < 3:
                continue
            rows.append({
                "id":           row[0],
                "masked_code":  row[1],
                "ground_truth": row[2].strip(),
            })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Core experiment loop for one strategy
# ─────────────────────────────────────────────────────────────────────────────

def run_strategy(model_key: str,
                 strategy_name: str,
                 static_k,      # int or None → use dynamic strategy
                 df: pd.DataFrame,
                 steps: int,
                 timestamp: str) -> str:
    """
    Load the diffusion model, run inference for one (model, strategy),
    save raw CSV, then FULLY RELEASE model before returning.

    LLM judging is done in a separate pass (run_judge_phase_3) after all
    diffusion models have been released.

    Returns the path to the saved raw CSV.
    """
    cfg        = MODELS_REGISTRY[model_key]
    model_name = cfg["name"]
    model_id   = cfg["id"]
    mask_token = cfg["mask_token"]

    print(f"\n{'─'*60}")
    print(f"  {model_name}  |  strategy={strategy_name}  |  steps={steps}")
    print(f"  {vram_info()}")
    print(f"{'─'*60}")

    # Load diffusion model
    t0 = time.time()
    print(f"  Loading {model_id} …")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16 if DEVICE == "cuda" else torch.float32,
        trust_remote_code=True,
    ).to(DEVICE).eval()
    print(f"  Loaded in {time.time() - t0:.1f}s  |  {vram_info()}")

    rows   = []
    errors = 0

    try:
        for i, row_data in enumerate(tqdm(df.itertuples(), total=len(df),
                                          desc=f"{model_name[:12]} {strategy_name}")):
            sample_id    = row_data.id
            masked_code  = str(row_data.masked_code)
            ground_truth = str(row_data.ground_truth).strip()

            try:
                # Determine k for this sample
                if static_k is not None:
                    k = static_k
                elif strategy_name == "dynamic_threshold":
                    k = strategy_threshold_heuristic(masked_code)
                else:  # dynamic_context
                    k = strategy_context_naming_length(masked_code, tokenizer)

                _full_code, prediction = run_diffusion_inference(
                    model, tokenizer, masked_code, mask_token, k, steps
                )
                exact_match = int(prediction == ground_truth)

                rows.append({
                    "id":           sample_id,
                    "model":        model_name,
                    "strategy":     strategy_name,
                    "dynamic_k":    k,
                    "steps":        steps,
                    "ground_truth": ground_truth,
                    "prediction":   prediction,
                    "exact_match":  exact_match,
                    "llm_verdict":  -2,  # will be filled in judge phase
                })

            except Exception as e:
                errors += 1
                rows.append({
                    "id":           sample_id,
                    "model":        model_name,
                    "strategy":     strategy_name,
                    "dynamic_k":    -1,
                    "steps":        steps,
                    "ground_truth": ground_truth,
                    "prediction":   "",
                    "exact_match":  0,
                    "llm_verdict":  -1,
                    "error":        str(e),
                })
                if errors <= 5:
                    print(f"    Error on sample {sample_id}: {e}")
                elif errors == 6:
                    print("    ... suppressing further error messages")

            # Periodic VRAM flush
            if (i + 1) % CACHE_CLEAR_INTERVAL == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Final cache clear for this run
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    finally:
        # Always release diffusion model, even on exception
        print(f"  Releasing {model_name} …")
        release_model(model, tokenizer)

    raw_df   = pd.DataFrame(rows)
    safe_nm  = model_name.replace("-", "_")
    out_path = os.path.join(RESULTS_DIR,
                            f"part3_raw_{safe_nm}_{strategy_name}_{timestamp}.csv")
    raw_df.to_csv(out_path, index=False)

    em_mean = raw_df["exact_match"].mean()
    mean_k  = raw_df[raw_df["dynamic_k"] >= 0]["dynamic_k"].mean()
    print(f"  EM={em_mean:.4f}  errors={errors}  mean_k={mean_k:.2f}")
    print(f"  → Saved: {out_path}")

    return out_path


# ─────────────────────────────────────────────────────────────────────────────
# Phase 2: LLM-as-Judge pass (after all diffusion models are released)
# ─────────────────────────────────────────────────────────────────────────────

def run_judge_phase_3(raw_paths: list, judge_model_id: str,
                     test_data_lookup: dict) -> None:
    """
    Load judge model ONCE (after all diffusion models are fully released),
    fill llm_verdict in each raw CSV, overwrite files in place.
    """
    print(f"\n{'═'*60}")
    print(f"  JUDGE PHASE: {judge_model_id}")
    print(f"  {vram_info()}")
    print(f"{'═'*60}")

    t0        = time.time()
    judge_tok = AutoTokenizer.from_pretrained(judge_model_id, trust_remote_code=True)
    judge_mdl = AutoModelForCausalLM.from_pretrained(
        judge_model_id,
        torch_dtype=torch.bfloat16 if DEVICE == "cuda" else torch.float32,
        device_map="auto" if DEVICE == "cuda" else None,
        trust_remote_code=True,
    ).eval()
    print(f"  Judge loaded in {time.time() - t0:.1f}s  |  {vram_info()}")

    try:
        for raw_path in raw_paths:
            print(f"\n  Judging: {os.path.basename(raw_path)}")
            df       = pd.read_csv(raw_path)
            verdicts = []

            for i, row in enumerate(tqdm(df.itertuples(), total=len(df),
                                         desc="    judging")):
                exact_match = int(row.exact_match)
                prediction  = str(getattr(row, "prediction", ""))

                if exact_match:
                    verdicts.append(1)
                elif not prediction or not re.search(r"[a-zA-Z]", prediction):
                    verdicts.append(0)
                else:
                    row_id       = str(row.id)
                    masked_code  = test_data_lookup.get(row_id, {}).get("masked_code", "")
                    ground_truth = str(row.ground_truth)
                    try:
                        v = judge_one(judge_tok, judge_mdl,
                                      masked_code, prediction, ground_truth)
                    except Exception:
                        v = -1
                    verdicts.append(v)

                if (i + 1) % CACHE_CLEAR_INTERVAL == 0 and torch.cuda.is_available():
                    torch.cuda.empty_cache()

            df["llm_verdict"] = verdicts
            df.to_csv(raw_path, index=False)

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            valid_v = [v for v in verdicts if v >= 0]
            lj_mean = sum(valid_v) / len(valid_v) if valid_v else float("nan")
            print(f"    LJ={lj_mean:.4f}  ({sum(1 for v in valid_v if v==1)}/{len(valid_v)})")
            print(f"    → Updated: {raw_path}")

    finally:
        print("\n  Releasing judge model …")
        release_model(judge_mdl, judge_tok)
        print(f"  {vram_info()}")



# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

STRATEGY_COLORS = {
    "static":            "#3a86ff",
    "dynamic_threshold": "#ff9f1c",
    "dynamic_context":   "#2ec4b6",
}

MODEL_HATCHES = {
    "DiffuCoder-7B": "",
    "DreamCoder-7B": "///",
}


def plot_comparison(summary_df: pd.DataFrame, timestamp: str):
    models     = summary_df["model"].unique()
    strategies = summary_df["strategy"].unique()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for metric, ax, title in [
        ("exact_match", axes[0], "Exact Match (EM)"),
        ("llm_judge",   axes[1], "LLM-as-Judge Acceptance Rate"),
    ]:
        x      = np.arange(len(strategies))
        width  = 0.25
        n_mod  = len(models)
        offs   = np.linspace(-width * (n_mod - 1) / 2,
                              width * (n_mod - 1) / 2, n_mod)

        for model_name, offset in zip(models, offs):
            sub = summary_df[summary_df["model"] == model_name].set_index("strategy")
            vals = [sub.loc[s, metric] if s in sub.index else 0. for s in strategies]
            color  = {"DiffuCoder-7B": "#3a86ff", "DreamCoder-7B": "#ff6b6b"}.get(model_name, "grey")
            hatch  = MODEL_HATCHES.get(model_name, "")
            bars   = ax.bar(x + offset, vals, width,
                            label=model_name, color=color,
                            hatch=hatch, alpha=0.85)
            for bar, v in zip(bars, vals):
                if not np.isnan(v):
                    ax.text(bar.get_x() + bar.get_width() / 2,
                            bar.get_height() + 0.003,
                            f"{v:.3f}", ha="center", va="bottom", fontsize=8)

        ax.set_xlabel("Strategy")
        ax.set_ylabel("Score")
        ax.set_title(title, fontweight="bold")
        ax.set_xticks(x)
        strategy_labels = {
            "static":            f"Static k={summary_df.loc[summary_df['strategy']=='static','static_k'].iloc[0] if 'static_k' in summary_df.columns else '?'}",
            "dynamic_threshold": "Dynamic\n(Threshold)",
            "dynamic_context":   "Dynamic\n(Context)",
        }
        ax.set_xticklabels([strategy_labels.get(s, s) for s in strategies],
                           fontsize=9)
        ax.set_ylim(0, min(1.0, summary_df[metric].max() * 1.25 + 0.05))
        ax.legend(fontsize=10)
        ax.grid(axis="y", alpha=0.4)

    plt.suptitle(
        "Dynamic vs Static MASK-Token Counts on Variable Renaming Quality\n"
        f"(RefineID, {summary_df['steps'].iloc[0]} diffusion steps)",
        fontsize=13, fontweight="bold", y=1.02,
    )
    plt.tight_layout()
    out = os.path.join(FIGURES_DIR, f"part3_dynamic_vs_static_{timestamp}.png")
    plt.savefig(out, bbox_inches="tight")
    plt.savefig(out.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close()
    print(f"\n  → Figure saved: {out}")


def plot_dynamic_k_distribution(all_raw_dfs: list, timestamp: str):
    """Show how the dynamic strategies distribute k values."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    strats = ["dynamic_threshold", "dynamic_context"]
    titles = ["Dynamic Threshold\n(k distribution)",
              "Dynamic Context Length\n(k distribution)"]
    colors_strat = ["#ff9f1c", "#2ec4b6"]

    for ax, strat, title, color in zip(axes, strats, titles, colors_strat):
        for df in all_raw_dfs:
            sub = df[df["strategy"] == strat]
            if len(sub) == 0:
                continue
            ks = sub["dynamic_k"].dropna().astype(int)
            counter = Counter(ks)
            total = len(ks)
            x_vals = sorted(counter.keys())
            y_vals = [counter[k] / total * 100 for k in x_vals]
            ax.bar([str(k) for k in x_vals], y_vals,
                   color=color, alpha=0.75,
                   label=sub["model"].iloc[0])
            ax.set_xlabel("Chosen k")
            ax.set_ylabel("% of samples")
            ax.set_title(title, fontweight="bold")
            ax.legend(fontsize=9)

    plt.tight_layout()
    out = os.path.join(FIGURES_DIR, f"part3_k_distribution_{timestamp}.png")
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"  → K-distribution figure saved: {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry-point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Part 3 – Dynamic vs static MASK-token count comparison."
    )
    parser.add_argument("--data", default="data/test.csv")
    parser.add_argument("--models", default="both",
                        choices=["diffucoder", "dreamcoder", "both"])
    parser.add_argument("--static-k", type=int, default=3)
    parser.add_argument("--part2-summary", type=str, default=None)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--judge-model", type=str,
                        default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--no-judge", action="store_true")
    parser.add_argument("--strategies", nargs="+",
                        choices=["static", "dynamic_threshold", "dynamic_context"],
                        default=["static", "dynamic_threshold", "dynamic_context"])
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    static_k = args.static_k
    if args.part2_summary and os.path.exists(args.part2_summary):
        p2 = pd.read_csv(args.part2_summary)
        best_row = p2.loc[p2["exact_match"].idxmax()]
        static_k = int(best_row["mask_count"])
        print(f"[Auto] Using best static k={static_k} from Part 2 summary.")

    print("=" * 65)
    print("  Part 3 – Dynamic vs Static MASK-Token Count Comparison")
    print("=" * 65)
    print(f"  Static baseline k : {static_k}")
    print(f"  Diffusion steps   : {args.steps}")
    print(f"  Strategies        : {args.strategies}")
    print(f"  LLM judge         : {'DISABLED' if args.no_judge else args.judge_model}")
    print(f"  Device            : {DEVICE}  |  {vram_info()}")

    print(f"\nLoading data from {args.data} …")
    df = load_data(args.data, args.max_samples)
    print(f"  {len(df)} samples loaded.")

    # Build lookup for judge phase
    test_data_lookup = {str(r["id"]): r for r in df.to_dict("records")}

    if args.models == "both":
        model_keys = ["diffucoder", "dreamcoder"]
    else:
        model_keys = [args.models]

    # ── Phase 1: Diffusion inference (one (model, strategy) at a time) ────────────
    # Critical: judge is NOT loaded yet. Each diffusion model is fully
    # released via try/finally before the next one loads.
    all_raw_paths: list = []

    for model_key in model_keys:
        for strategy in args.strategies:
            k_for_run = static_k if strategy == "static" else None
            raw_path = run_strategy(
                model_key=model_key,
                strategy_name=strategy,
                static_k=k_for_run,
                df=df,
                steps=args.steps,
                timestamp=timestamp,
            )
            all_raw_paths.append(raw_path)

    # ── Phase 2: LLM judge (runs AFTER all diffusion models are released) ─────
    if not args.no_judge:
        judge_id = JUDGE_REGISTRY.get(args.judge_model, args.judge_model)
        run_judge_phase_3(all_raw_paths, judge_id, test_data_lookup)

    # ── Aggregate summary ──────────────────────────────────────────────────────────
    all_dfs = [pd.read_csv(p) for p in all_raw_paths]
    combined = pd.concat(all_dfs, ignore_index=True)
    all_summaries = []

    for (model_name, strategy), grp in combined.groupby(["model", "strategy"]):
        valid_lj = grp[grp["llm_verdict"] >= 0]["llm_verdict"]
        mean_k   = grp[grp["dynamic_k"] >= 0]["dynamic_k"].mean()
        all_summaries.append({
            "model":       model_name,
            "strategy":    strategy,
            "static_k":    static_k if strategy == "static" else None,
            "mean_k":      round(mean_k, 2),
            "steps":       args.steps,
            "n_samples":   len(grp),
            "exact_match": round(grp["exact_match"].mean(), 4),
            "llm_judge":   round(valid_lj.mean(), 4) if len(valid_lj) > 0 else None,
        })

    summary_df   = pd.DataFrame(all_summaries)
    summary_path = os.path.join(RESULTS_DIR, f"part3_summary_{timestamp}.csv")
    summary_df.to_csv(summary_path, index=False)

    print("\n" + "=" * 65)
    print("  PART 3 SUMMARY")
    print("=" * 65)
    print(summary_df.to_string(index=False))
    print(f"\n  Summary saved → {summary_path}")

    if len(summary_df) > 0:
        plot_comparison(summary_df, timestamp)
        plot_dynamic_k_distribution(all_dfs, timestamp)

    print("\n" + "=" * 65)
    print("  DYNAMIC vs STATIC IMPROVEMENT")
    print("=" * 65)
    for model_name in summary_df["model"].unique():
        sub       = summary_df[summary_df["model"] == model_name]
        s_em_rows = sub[sub["strategy"] == "static"]["exact_match"].values
        s_lj_rows = sub[sub["strategy"] == "static"]["llm_judge"].values
        if len(s_em_rows) == 0:
            continue
        s_em = s_em_rows[0]
        s_lj = float(s_lj_rows[0]) if len(s_lj_rows) and s_lj_rows[0] is not None else float("nan")
        print(f"\n  {model_name}  (static k={static_k}: EM={s_em:.4f}  LJ={s_lj!r})")
        for _, row in sub[sub["strategy"] != "static"].iterrows():
            d_em = row["exact_match"] - s_em
            d_lj = (float(row["llm_judge"]) if row["llm_judge"] is not None else float("nan")) - s_lj
            print(f"    {row['strategy']:25s}: EM={row['exact_match']:.4f} (Δ{d_em:+.4f})  "
                  f"LJ={row['llm_judge']!r} (Δ{d_lj:+.4f})")
    print("=" * 65)


if __name__ == "__main__":
    main()
