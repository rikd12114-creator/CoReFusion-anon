"""
Part 2: Static MASK-Token Count Ablation
==========================================

Research question:
  Given that the mask position is known, what is the *optimal number* of
  <|mask|> tokens to place there so that a diffusion language model best
  recovers the original variable name?

Experiment design:
  • Dataset  : data/test.csv  (RefineID, Java variable renaming)
  • Models   : DiffuCoder-7B-Instruct, DreamCoder-7B-Instruct
  • Mask counts tested: [1, 2, 3, 4, 5]
  • Diffusion steps  : 32  (fixed)
  • Evaluation metrics:
      1. Exact Match (EM)  – prediction == ground_truth
      2. LLM-as-Judge (LJ) – Qwen2.5-7B-Instruct judge (binary: 0/1)

Memory management strategy:
  - Diffusion model is loaded, used for ALL k values for that model, then
    fully released before anything else loads.
  - LLM judge is loaded AFTER the diffusion model is released; judging is
    done in a second pass over the saved raw CSV.
  - Every inference call explicitly deletes intermediate tensors and calls
    torch.cuda.empty_cache() after each sample on a configurable cadence.

Usage (from repo root):
    python experiments/1t5t_exp/part2_static_token_ablation.py \\
        --data data/test.csv --models both --mask-counts 1 2 3 4 5 \\
        --steps 32 --judge-model Qwen/Qwen2.5-7B-Instruct --max-samples 200

    # EM-only (no judge, much faster)
    python experiments/1t5t_exp/part2_static_token_ablation.py --no-judge

    # Single model
    python experiments/1t5t_exp/part2_static_token_ablation.py \\
        --models diffucoder --mask-counts 1 2 3 4 5 --no-judge
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
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
from tqdm import tqdm

# ── torchvision mock ──────────────────────────────────────────────────────────
class MockModule:
    def __getattr__(self, name): return MockModule()
    def __call__(self, *args, **kwargs): return MockModule()

sys.modules['torchvision'] = MockModule()
sys.modules['torchvision.ops'] = MockModule()
sys.modules['torchvision.transforms'] = MockModule()
if not hasattr(torch.ops, 'torchvision'):
    class DummyOps:
        def nms(*args, **kwargs): return torch.tensor([])
    torch.ops.torchvision = DummyOps()
# ─────────────────────────────────────────────────────────────────────────────

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODELS_REGISTRY = {
    "diffucoder": {
        "name":       "DiffuCoder-7B",
        "id":         "apple/DiffuCoder-7B-Instruct",
        "mask_token": "<|mask|>",
    },
    "dreamcoder": {
        "name":       "DreamCoder-7B",
        "id":         "Dream-org/Dream-Coder-v0-Instruct-7B",
        "mask_token": "<|mask|>",
    },
}

RESULTS_DIR = "results/1t5t_exp"
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")

JUDGE_REGISTRY = {
    "Qwen2.5-7B-Instruct": "Qwen/Qwen2.5-7B-Instruct",
    "Qwen2.5-3B-Instruct":  "Qwen/Qwen2.5-3B-Instruct",
}

MODEL_COLORS = {
    "DiffuCoder-7B": "#3a86ff",
    "DreamCoder-7B": "#ff6b6b",
}

# How often (every N samples) to call torch.cuda.empty_cache() during inference
CACHE_CLEAR_INTERVAL = 20


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
    alloc = torch.cuda.memory_allocated() / 1e9
    reserved = torch.cuda.memory_reserved() / 1e9
    total = torch.cuda.get_device_properties(0).total_memory / 1e9
    return f"alloc={alloc:.1f}GB  reserved={reserved:.1f}GB  total={total:.1f}GB"


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_data(data_path: str, max_samples: int | None = None) -> list[dict]:
    """Load test.csv: columns are id | masked_code | ground_truth (no header)."""
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
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# Prediction extraction  (mirrors benchmark_diffusion_models.py)
# ─────────────────────────────────────────────────────────────────────────────

def extract_all_predictions(full_code: str, masked_code: str) -> list[str]:
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

        idx_end = full_code.find(post_anchor, idx_start) if post_anchor else -1

        if idx_end != -1:
            gap_content = full_code[idx_start:idx_end].strip()
            current_search_start = idx_end
        else:
            gap_content = full_code[idx_start: idx_start + 60].strip()
            current_search_start = idx_start + 60

        match = re.search(r'[a-zA-Z_$][a-zA-Z0-9_$]*', gap_content)
        predictions.append(match.group(0) if match else gap_content[:20])

    return predictions


# ─────────────────────────────────────────────────────────────────────────────
# Diffusion model inference
# ─────────────────────────────────────────────────────────────────────────────

def run_diffusion_inference(model, tokenizer, masked_code: str,
                            mask_token: str, k: int,
                            steps: int) -> tuple[str, str]:
    """
    Replace [MASK] with k * mask_token (no spaces), tokenise directly (no
    chat template), diffusion_generate in-place (max_new_tokens=1), decode
    the entire output sequence, extract prediction via context anchoring.

    All intermediate tensors are explicitly deleted to minimise VRAM retention.
    """
    multi_mask = mask_token * k
    input_code = masked_code.replace("[MASK]", multi_mask)

    inputs    = tokenizer(input_code, return_tensors="pt")
    input_ids = inputs.input_ids.to(model.device)
    attn_mask = inputs.attention_mask.to(model.device)
    del inputs  # free CPU tensor immediately

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

    # Free GPU tensors immediately
    del input_ids, attn_mask, output, gen_ids
    # (no torch.cuda.empty_cache() here — called on interval in outer loop)

    preds = extract_all_predictions(full_code, masked_code)
    return full_code, (preds[0] if preds else "")


# ─────────────────────────────────────────────────────────────────────────────
# LLM-as-Judge
# ─────────────────────────────────────────────────────────────────────────────

JUDGE_SYSTEM = (
    "You are an expert Java code reviewer evaluating the quality of variable names "
    "suggested by an AI model.\n\n"
    "Your task: given a code snippet and a ground-truth variable name, decide "
    "whether the predicted variable name is SEMANTICALLY ACCEPTABLE as a replacement.\n\n"
    "Rules:\n"
    "1. ACCEPTABLE if the prediction conveys the same concept as the ground truth, "
    "even if the exact string differs "
    "(e.g. 'bufSize' vs 'bufferSize' are both acceptable for a buffer-size variable).\n"
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


def _apply_chat_template(tokenizer, user_text: str) -> str:
    msgs = [{"role": "system", "content": JUDGE_SYSTEM},
            {"role": "user",   "content": user_text}]
    try:
        return tokenizer.apply_chat_template(msgs, tokenize=False,
                                             add_generation_prompt=True)
    except Exception:
        return f"{JUDGE_SYSTEM}\n\nUser: {user_text}\nAssistant:"


def parse_verdict(text: str) -> int:
    m = re.search(r"VERDICT\s*:\s*([01])", text, re.IGNORECASE)
    if m:
        return int(m.group(1))
    for line in text.strip().splitlines():
        line = line.strip()
        if line in ("1", "0"):
            return int(line)
        if line.lower() in ("yes", "acceptable", "correct"):
            return 1
        if line.lower() in ("no", "not acceptable", "incorrect", "wrong"):
            return 0
    return -1


def judge_one(judge_tok, judge_model,
              masked_code: str, prediction: str, ground_truth: str) -> int:
    """Run judge on a single sample. Returns 1/0/-1. All tensors freed after call."""
    mask_pos       = masked_code.find("[MASK]")
    code_with_pred = masked_code.replace("[MASK]", prediction, 1)
    half   = 1000
    start  = max(0, mask_pos - half)
    end    = min(len(code_with_pred), mask_pos + len(prediction) + half)
    context = ("..." if start > 0 else "") + code_with_pred[start:end] + \
              ("..." if end < len(code_with_pred) else "")

    user_text = (
        f"Code context:\n```java\n{context}\n```\n\n"
        f"Ground-truth variable name: `{ground_truth}`\n"
        f"Predicted variable name:    `{prediction}`\n\n"
        f"Reply with EXACTLY one line: VERDICT: 1   or   VERDICT: 0"
    )
    prompt = _apply_chat_template(judge_tok, user_text)
    inp = judge_tok(prompt, return_tensors="pt", truncation=True,
                    max_length=4096).to(judge_model.device)
    with torch.no_grad():
        out = judge_model.generate(
            **inp, max_new_tokens=16, do_sample=False,
            pad_token_id=(judge_tok.eos_token_id or judge_tok.pad_token_id or 0),
        )
    new_ids = out[0][inp["input_ids"].shape[1]:]
    raw     = judge_tok.decode(new_ids, skip_special_tokens=True).strip()

    # Free judge tensors immediately
    del inp, out, new_ids
    # Caller is responsible for periodic empty_cache()

    return parse_verdict(raw)


# ─────────────────────────────────────────────────────────────────────────────
# Phase 1: Diffusion inference for all k values of ONE model
# ─────────────────────────────────────────────────────────────────────────────

def run_diffusion_phase(model_key: str, k_values: list[int],
                        data: list[dict], steps: int,
                        timestamp: str) -> list[str]:
    """
    Load the diffusion model ONCE, run inference for every k in k_values,
    save one raw CSV per k, then fully release the model.

    Returns the list of saved raw CSV paths.
    """
    cfg        = MODELS_REGISTRY[model_key]
    model_name = cfg["name"]
    model_id   = cfg["id"]
    mask_token = cfg["mask_token"]

    print(f"\n{'═'*65}")
    print(f"  DIFFUSION PHASE: {model_name}")
    print(f"  k values: {k_values}  |  steps={steps}")
    print(f"  {vram_info()}")
    print(f"{'═'*65}")

    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16 if DEVICE == "cuda" else torch.float32,
        trust_remote_code=True,
    ).to(DEVICE).eval()
    print(f"  Model loaded in {time.time() - t0:.1f}s  |  {vram_info()}")

    saved_paths = []

    try:
        for k in k_values:
            print(f"\n  ── k={k} {'─'*50}")
            rows    = []
            correct = 0
            errors  = 0

            for i, row in enumerate(tqdm(data, desc=f"  {model_name} k={k}")):
                item_id      = row["id"]
                masked_code  = row["masked_code"]
                ground_truth = row["ground_truth"]

                try:
                    _, prediction = run_diffusion_inference(
                        model, tokenizer, masked_code, mask_token, k, steps
                    )
                    exact_match = int(prediction == ground_truth)
                    if exact_match:
                        correct += 1

                    rows.append({
                        "id":           item_id,
                        "model":        model_name,
                        "mask_count":   k,
                        "steps":        steps,
                        "ground_truth": ground_truth,
                        "prediction":   prediction,
                        "exact_match":  exact_match,
                        "llm_verdict":  -2,  # not judged yet
                    })
                except Exception as e:
                    errors += 1
                    rows.append({
                        "id":           item_id,
                        "model":        model_name,
                        "mask_count":   k,
                        "steps":        steps,
                        "ground_truth": ground_truth,
                        "prediction":   "",
                        "exact_match":  0,
                        "llm_verdict":  -1,
                        "error":        str(e),
                    })
                    if errors <= 5:
                        print(f"    Error on sample {item_id}: {e}")
                    elif errors == 6:
                        print("    ... suppressing further error messages")

                # Periodic VRAM flush
                if (i + 1) % CACHE_CLEAR_INTERVAL == 0 and torch.cuda.is_available():
                    torch.cuda.empty_cache()

            # Final cache clear for this k
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            raw_df   = pd.DataFrame(rows)
            safe_nm  = model_name.replace("-", "_")
            raw_path = os.path.join(RESULTS_DIR,
                                    f"part2_raw_{safe_nm}_{k}tok_{timestamp}.csv")
            raw_df.to_csv(raw_path, index=False)
            saved_paths.append(raw_path)

            em = raw_df["exact_match"].mean()
            print(f"\n  EM={em:.4f}  ({correct}/{len(data)})  errors={errors}")
            print(f"  → Saved: {raw_path}")

    finally:
        # Always release diffusion model — even if an exception occurred midway
        print(f"\n  Releasing {model_name} from GPU …")
        release_model(model, tokenizer)
        print(f"  {vram_info()}")

    return saved_paths


# ─────────────────────────────────────────────────────────────────────────────
# Phase 2: LLM-as-Judge pass over saved raw CSVs
# ─────────────────────────────────────────────────────────────────────────────

def run_judge_phase(raw_paths: list[str], judge_model_id: str,
                    test_data_lookup: dict) -> None:
    """
    Load the judge model ONCE (after diffusion model has been released),
    iterate over saved raw CSVs, fill in llm_verdict, overwrite in place.
    """
    print(f"\n{'═'*65}")
    print(f"  JUDGE PHASE: {judge_model_id}")
    print(f"  {vram_info()}")
    print(f"{'═'*65}")

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
                prediction  = str(row.prediction) if hasattr(row, "prediction") else ""

                if exact_match:
                    verdicts.append(1)   # exact match → automatically acceptable
                elif not prediction or not re.search(r"[a-zA-Z]", prediction):
                    verdicts.append(0)   # invalid prediction
                else:
                    # Fall back to masked_code from test_data_lookup
                    row_id      = str(row.id)
                    masked_code = test_data_lookup.get(row_id, {}).get("masked_code", "")
                    ground_truth = str(row.ground_truth)
                    try:
                        v = judge_one(judge_tok, judge_mdl,
                                      masked_code, prediction, ground_truth)
                    except Exception as e:
                        v = -1
                    verdicts.append(v)

                # Periodic VRAM flush
                if (i + 1) % CACHE_CLEAR_INTERVAL == 0 and torch.cuda.is_available():
                    torch.cuda.empty_cache()

            df["llm_verdict"] = verdicts
            df.to_csv(raw_path, index=False)

            # Final cache clear for this file
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            valid_v = [v for v in verdicts if v >= 0]
            lj_mean = sum(valid_v) / len(valid_v) if valid_v else float("nan")
            print(f"    LJ={lj_mean:.4f}  ({sum(1 for v in valid_v if v==1)}/{len(valid_v)})")
            print(f"    → Updated: {raw_path}")

    finally:
        print(f"\n  Releasing judge model …")
        release_model(judge_mdl, judge_tok)
        print(f"  {vram_info()}")


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

def plot_results(summary_df: pd.DataFrame, timestamp: str):
    models   = summary_df["model"].unique()
    k_vals   = sorted(summary_df["mask_count"].unique())
    x        = np.arange(len(k_vals))
    width    = 0.35
    n_models = len(models)
    offsets  = np.linspace(-width * (n_models - 1) / 2,
                            width * (n_models - 1) / 2, n_models)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for metric, ax, title in [
        ("exact_match", axes[0], "Exact Match (EM)"),
        ("llm_judge",   axes[1], "LLM-as-Judge Acceptance Rate"),
    ]:
        for model_name, offset in zip(models, offsets):
            sub  = summary_df[summary_df["model"] == model_name].set_index("mask_count")
            vals = []
            for k in k_vals:
                v = sub.loc[k, metric] if k in sub.index else 0.
                vals.append(float(v) if not pd.isna(v) else 0.)

            bars = ax.bar(x + offset, vals, width,
                          label=model_name,
                          color=MODEL_COLORS.get(model_name, "grey"),
                          alpha=0.85)
            for bar, v in zip(bars, vals):
                if v > 0:
                    ax.text(bar.get_x() + bar.get_width() / 2,
                            bar.get_height() + 0.002,
                            f"{v:.3f}", ha="center", va="bottom", fontsize=8)

        ax.set_xlabel("Number of <|mask|> tokens per variable (k)", fontsize=11)
        ax.set_ylabel("Score", fontsize=11)
        ax.set_title(title, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([str(k) for k in k_vals])
        max_val = summary_df[metric].dropna().max()
        ax.set_ylim(0, min(1.05, max_val * 1.25 + 0.05) if max_val > 0 else 0.2)
        ax.legend(fontsize=10)
        ax.grid(axis="y", alpha=0.4)

    plt.suptitle(
        "Effect of Static MASK-Token Count on Variable Renaming Quality\n"
        f"(RefineID · {summary_df['steps'].iloc[0]} diffusion steps · "
        f"{summary_df['n_samples'].iloc[0]} samples)",
        fontsize=13, fontweight="bold", y=1.02,
    )
    plt.tight_layout()
    out = os.path.join(FIGURES_DIR, f"part2_ablation_{timestamp}.png")
    plt.savefig(out, bbox_inches="tight")
    plt.savefig(out.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close()
    print(f"\n  → Figure saved: {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry-point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Part 2 – Static MASK-token count ablation (1-5 tokens).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--data",        default="data/test.csv")
    parser.add_argument("--models",      default="both",
                        choices=["diffucoder", "dreamcoder", "both"])
    parser.add_argument("--mask-counts", nargs="+", type=int,
                        default=[1, 2, 3, 4, 5])
    parser.add_argument("--steps",       type=int, default=32)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--judge-model", type=str,
                        default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--no-judge",    action="store_true",
                        help="Skip LLM judge – EM evaluation only.")
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("=" * 65)
    print("  Part 2 – Static MASK-Token Count Ablation")
    print("=" * 65)
    print(f"  Data            : {args.data}")
    print(f"  Diffusion steps : {args.steps}")
    print(f"  Mask counts     : {args.mask_counts}")
    print(f"  Models          : {args.models}")
    print(f"  LLM judge       : {'DISABLED' if args.no_judge else args.judge_model}")
    print(f"  Device          : {DEVICE}  |  {vram_info()}")

    # ── Load data ──────────────────────────────────────────────────────────
    print(f"\nLoading data from {args.data} …")
    data = load_data(args.data, args.max_samples)
    print(f"  {len(data):,} samples loaded.")

    # Build a lookup dict for judge phase (id → row)
    test_data_lookup = {str(r["id"]): r for r in data}

    if args.models == "both":
        model_keys = ["diffucoder", "dreamcoder"]
    else:
        model_keys = [args.models]

    # ── Phase 1: Diffusion inference (one model at a time, ALL k values) ──
    # Key memory insight: load model ONCE per model_key, run all k values,
    # then fully release BEFORE loading the next model or the judge.
    all_raw_paths: list[str] = []

    for model_key in model_keys:
        paths = run_diffusion_phase(
            model_key=model_key,
            k_values=args.mask_counts,
            data=data,
            steps=args.steps,
            timestamp=timestamp,
        )
        all_raw_paths.extend(paths)

    # ── Phase 2: LLM judge (only after ALL diffusion models are released) ─
    if not args.no_judge:
        judge_id = JUDGE_REGISTRY.get(args.judge_model, args.judge_model)
        run_judge_phase(all_raw_paths, judge_id, test_data_lookup)

    # ── Aggregate summary ──────────────────────────────────────────────────
    all_dfs = [pd.read_csv(p) for p in all_raw_paths]
    combined = pd.concat(all_dfs, ignore_index=True)

    summary_rows = []
    for (model_name, k), grp in combined.groupby(["model", "mask_count"]):
        valid_lj = grp[grp["llm_verdict"] >= 0]["llm_verdict"]
        summary_rows.append({
            "model":        model_name,
            "mask_count":   k,
            "steps":        args.steps,
            "n_samples":    len(grp),
            "exact_match":  round(grp["exact_match"].mean(), 4),
            "llm_judge":    round(valid_lj.mean(), 4) if len(valid_lj) > 0 else None,
        })

    summary_df   = pd.DataFrame(summary_rows)
    summary_path = os.path.join(RESULTS_DIR, f"part2_summary_{timestamp}.csv")
    summary_df.to_csv(summary_path, index=False)

    print("\n" + "=" * 65)
    print("  PART 2 SUMMARY")
    print("=" * 65)
    print(summary_df.to_string(index=False))
    print(f"\n  Summary saved → {summary_path}")

    if len(summary_df) > 0:
        plot_results(summary_df, timestamp)

    # ── Print optimal k ──────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  OPTIMAL MASK COUNT (k*)")
    print("=" * 65)
    for model_name in summary_df["model"].unique():
        sub = summary_df[summary_df["model"] == model_name]
        best_em = sub.loc[sub["exact_match"].idxmax()]
        print(f"\n  {model_name}:")
        print(f"    Best EM  → k={int(best_em['mask_count'])}  "
              f"EM={best_em['exact_match']:.4f}")
        lj_col = sub["llm_judge"].dropna()
        if len(lj_col) > 0:
            best_lj = sub.loc[lj_col.idxmax()]
            print(f"    Best LLM → k={int(best_lj['mask_count'])}  "
                  f"LJ={best_lj['llm_judge']:.4f}")
    print("=" * 65)


if __name__ == "__main__":
    main()
