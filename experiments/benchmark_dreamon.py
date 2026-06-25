"""
Standalone DreamOn benchmark for the refineID (variable-naming refactoring) task.

Strictly follows the DreamOn quickstart from
    https://github.com/DreamLM/DreamOn

Key DreamOn-specific notes (from upstream README):
    * Context window is only 2048 tokens (input + output) -- files in our
      test set go up to ~50K chars / ~15K tokens, so we MUST crop a local
      window around the first [MASK] site.
    * Input format: BOS + prefix + [mask_id]*N + suffix + EOS
    * ``max_new_tokens`` is the MAX canvas size (not "additional"); must be
      >= initial number_of_mask. Quickstart uses 4 / 64.
    * ``number_transfer_tokens=1`` (1 token per denoising step).
    * Recommended sampling: temperature=0.2, top_p=0.9, alg='entropy', alg_temp=0.
    * NO attention_mask, NO steps argument.

Strategy on our refineID data:
    All [MASK] occurrences in a sample are the SAME identifier (e.g.
    sample 3 has 61 masks all referring to ``style``). We:

      1. Crop a CONTEXT_CHARS-sized window around the first [MASK].
      2. Within the window, replace ALL [MASK] with <|mask|>*NUM_MASK_PER_SITE
         (multi-site single-pass infilling, so the model uses each
         occurrence as context for the others).
      3. Run a single diffusion_generate call.
      4. Extract the first identifier by taking the K tokens immediately
         after the prefix and regex-matching a Java identifier. This is
         robust to DreamOn's variable-length output padding.

Usage on Colab:
    !pip install transformers==4.46.2 torch==2.5.1 omegaconf tqdm pandas \
                  huggingface_hub
    !python experiments/benchmark_dreamon.py --max-samples 5  --debug
    !python experiments/benchmark_dreamon.py --max-samples 100
    !python experiments/benchmark_dreamon.py
"""

import os
import sys
import csv
import re
import gc
import argparse
import time
from collections import Counter
from datetime import datetime

import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

try:
    from huggingface_hub import HfApi
    HAS_HF_HUB = True
except ImportError:
    HAS_HF_HUB = False


# --- Mock torchvision (some Dream/DreamOn checkpoints try to import it) ---
class _MockModule:
    def __getattr__(self, name): return _MockModule()
    def __call__(self, *args, **kwargs): return _MockModule()

sys.modules.setdefault('torchvision', _MockModule())
sys.modules.setdefault('torchvision.ops', _MockModule())
sys.modules.setdefault('torchvision.transforms', _MockModule())
if not hasattr(torch.ops, 'torchvision'):
    class _DummyOps:
        def nms(*args, **kwargs): return torch.tensor([])
    torch.ops.torchvision = _DummyOps()


# ---- Configuration ---------------------------------------------------------

MODEL_ID = "Dream-org/DreamOn-v0-7B"
DATA_PATH = "data/test.csv"
RESULTS_DIR = "results/dreamon_benchmark"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Window cropping around the first [MASK] (chars).
# Roughly 4 chars/token for code -> 3000 chars ~ 750 tokens of context, well
# within DreamOn's 2048 token budget after we add masks + EOS.
CONTEXT_CHARS = 3000

# How many <|mask|> tokens to put at each [MASK] site initially.
NUM_MASK_PER_SITE = 4

# Max canvas size after DreamOn expansion (per call). Quickstart uses 64.
MAX_NEW_TOKENS = 64

# Cap on the number of [MASK] sites we keep in the window. Sample 3 has 61
# masks; keeping all of them blows up canvas / context budget.
MAX_SITES_IN_WINDOW = 8

# Number of tokens to read AFTER the prefix when extracting the first
# predicted identifier. Java identifiers are typically <= 5 BPE tokens.
EXTRACT_TOKENS = 12

GEN_KWARGS = dict(
    temperature=0.2,
    top_p=0.9,
    alg="entropy",
    alg_temp=0,
    number_transfer_tokens=1,
)


# ---- Window cropping -------------------------------------------------------

_MASK_RE = re.compile(r'\[MASK\]')


def crop_window(code, target_chars=CONTEXT_CHARS, max_sites=MAX_SITES_IN_WINDOW):
    """Crop a window around the FIRST [MASK] in code.

    Kept for backward compatibility / sanity check. ``tile_windows`` is the
    preferred entry point because it covers every [MASK] in the sample.
    """
    first = code.find("[MASK]")
    if first == -1:
        return code, -1

    half = target_chars // 2
    start = max(0, first - half)
    end = min(len(code), first + half)
    window = code[start:end]

    while window.count("[MASK]") > max_sites and end > first + 200:
        end -= 200
        window = code[start:end]

    return window, first


def tile_windows(code, target_chars=CONTEXT_CHARS, max_sites=MAX_SITES_IN_WINDOW):
    """Tile the FULL sample into a list of non-overlapping windows so
    EVERY [MASK] in the sample falls into exactly one window.

    Each window:
        * Contains up to ``max_sites`` [MASK] sites.
        * Has ~target_chars/2 chars of context on each side of the
          contained sites (clamped so adjacent windows don't share a site).
        * Stays small enough to fit comfortably in DreamOn's 2048 token
          context after we substitute <|mask|> tokens.

    Returns:
        list of (window_text, global_site_indices) where
            window_text contains exactly len(global_site_indices) [MASK]s
            global_site_indices is the list of which mask indices (0-based
                in the full sample) live inside this window.
    """
    spans = [m.span() for m in _MASK_RE.finditer(code)]
    if not spans:
        return []

    half = target_chars // 2
    windows = []
    i = 0
    while i < len(spans):
        group = [i]
        # Greedily pack subsequent sites into this window while the span
        # of all grouped sites stays within target_chars.
        while (len(group) < max_sites
               and group[-1] + 1 < len(spans)
               and spans[group[-1] + 1][1] - spans[group[0]][0] <= target_chars):
            group.append(group[-1] + 1)

        first_site_start = spans[group[0]][0]
        last_site_end = spans[group[-1]][1]
        win_start = max(0, first_site_start - half)
        win_end = min(len(code), last_site_end + half)

        # Clip so adjacent external [MASK]s never appear inside this window.
        if group[0] > 0:
            prev_end = spans[group[0] - 1][1]
            win_start = max(win_start, prev_end)
        if group[-1] + 1 < len(spans):
            next_start = spans[group[-1] + 1][0]
            win_end = min(win_end, next_start)

        win_text = code[win_start:win_end]
        # Sanity: window must contain exactly len(group) [MASK]s
        n_in_win = win_text.count("[MASK]")
        assert n_in_win == len(group), (
            f"tile_windows mismatch: window has {n_in_win} [MASK]s but "
            f"group has {len(group)}"
        )

        windows.append((win_text, group))
        i = group[-1] + 1

    return windows


# ---- Multi-site DreamOn prompt ---------------------------------------------

def build_multisite_prompt(window_text, tokenizer, num_mask_per_site=NUM_MASK_PER_SITE):
    """Replace each [MASK] in window_text with num_mask_per_site mask tokens
    and frame with BOS/EOS.

    Returns:
        input_ids: list[int]
        prefix_token_count: int  -- number of tokens of the prefix BEFORE
            the first <|mask|> region (excluding BOS). Use this with
            EXTRACT_TOKENS to read the first denoised identifier.
    """
    parts = window_text.split("[MASK]")
    bos = tokenizer.bos_token_id
    eos = tokenizer.eos_token_id
    mask_id = tokenizer.mask_token_id

    ids = [bos]
    prefix_len = None  # tokens of prefix excluding BOS, computed at first mask
    for i, segment in enumerate(parts):
        seg_ids = tokenizer.encode(segment, add_special_tokens=False)
        ids.extend(seg_ids)
        if i < len(parts) - 1:
            if prefix_len is None:
                prefix_len = len(ids) - 1   # exclude BOS at index 0
            ids.extend([mask_id] * num_mask_per_site)
    ids.append(eos)
    return ids, prefix_len


# ---- Identifier extraction -------------------------------------------------

_IDENT_RE = re.compile(r'[a-zA-Z_$][a-zA-Z0-9_$]*')

def extract_all_idents_from_multisite(seq, tokenizer, window_text):
    """Return one identifier per [MASK] site in window_text.

    DreamOn expand/contract makes ID-position tracking across multiple mask
    sites unreliable (the second canvas's start position depends on what
    happened to the first). String anchors are robust because the surrounding
    text segments never get rewritten.
    """
    full = tokenizer.decode(seq, skip_special_tokens=True)
    parts = window_text.split("[MASK]")  # N+1 segments for N masks
    if len(parts) <= 1:
        return []

    preds = []
    cursor = 0
    for i in range(len(parts) - 1):
        pre = parts[i]
        post = parts[i + 1]
        pre_anchor = pre.strip()[-30:] if len(pre.strip()) > 30 else pre.strip()
        post_anchor = post.strip()[:30] if len(post.strip()) > 30 else post.strip()

        if pre_anchor:
            idx = full.find(pre_anchor, cursor)
            if idx == -1:
                preds.append("")
                continue
            cursor = idx + len(pre_anchor)
        # else: use current cursor

        if post_anchor:
            end_idx = full.find(post_anchor, cursor)
            if end_idx == -1:
                end_idx = min(cursor + 60, len(full))
        else:
            end_idx = min(cursor + 60, len(full))

        canvas_text = full[cursor:end_idx] if end_idx > cursor else full[cursor:cursor + 60]
        m = _IDENT_RE.search(canvas_text)
        preds.append(m.group(0) if m else canvas_text.strip()[:20])
        cursor = end_idx
    return preds


def extract_identifier_from_tokens(seq, tokenizer, prefix_token_count, n_tokens=EXTRACT_TOKENS):
    """Take the n_tokens immediately after the prefix in the generated seq,
    decode them, and pull out the first valid Java identifier.

    This is robust to DreamOn's variable-length output / trailing padding,
    because:
        * BOS is at index 0 (unchanged)
        * prefix_token_count tokens of prefix follow (unchanged -- they
          weren't masked, so the model cannot rewrite them)
        * the FIRST denoised canvas starts at index 1 + prefix_token_count
        * the identifier we want is the very first non-noise token there.
    """
    if torch.is_tensor(seq):
        seq_list = seq.tolist()
    else:
        seq_list = list(seq)

    start = 1 + prefix_token_count        # skip BOS + prefix
    end = min(len(seq_list), start + n_tokens)
    chunk = seq_list[start:end]
    text = tokenizer.decode(chunk, skip_special_tokens=True)

    m = _IDENT_RE.search(text)
    if m:
        return m.group(0), text
    return text.strip()[:20], text


# ---- Single-sample inference -----------------------------------------------

def _run_one_window(model, tokenizer, window_text, num_mask_per_site,
                    max_new_tokens, debug=False):
    """Run one DreamOn forward on a single window. Returns the list of
    predictions, one per [MASK] site in that window."""
    input_ids, prefix_token_count = build_multisite_prompt(
        window_text, tokenizer, num_mask_per_site=num_mask_per_site,
    )
    eff_max_new = max(num_mask_per_site, max_new_tokens)

    input_t = torch.LongTensor([input_ids]).to(model.device)
    with torch.no_grad():
        output = model.diffusion_generate(
            input_t,
            max_new_tokens=eff_max_new,
            return_dict_in_generate=True,
            output_history=False,
            **GEN_KWARGS,
        )
    seq = output.sequences[0] if hasattr(output, "sequences") else output[0]

    preds = extract_all_idents_from_multisite(seq, tokenizer, window_text)

    if debug:
        full = tokenizer.decode(seq, skip_special_tokens=False)
        print(f"  [debug] window_chars={len(window_text)} sites={window_text.count('[MASK]')}")
        print(f"  [debug] preds in window: {preds}")
        nl_repr = chr(92) + "n"   # literal backslash-n kept OUT of the f-string (py3.11)
        print(f"  [debug] full[:300 with specials]: "
              f"{full[:300].replace(chr(10), nl_repr)}")
    return preds


def predict_one(model, tokenizer, masked_code, debug=False,
                context_chars=CONTEXT_CHARS,
                num_mask_per_site=NUM_MASK_PER_SITE,
                max_new_tokens=MAX_NEW_TOKENS,
                max_sites=MAX_SITES_IN_WINDOW):
    """Tile the FULL sample into windows so every [MASK] is covered, run
    multi-site infill on each, and return per-site predictions.

    Returns:
        site_preds: list[str] of length total_masks, one prediction per site
        n_windows: int, how many forward passes were used
    """
    n_total = masked_code.count("[MASK]")
    if n_total == 0:
        return [], 0

    windows = tile_windows(masked_code, target_chars=context_chars, max_sites=max_sites)
    site_preds = [""] * n_total
    for w_idx, (win_text, global_indices) in enumerate(windows):
        preds = _run_one_window(
            model, tokenizer, win_text,
            num_mask_per_site=num_mask_per_site,
            max_new_tokens=max_new_tokens,
            debug=(debug and w_idx == 0),
        )
        for local_i, gi in enumerate(global_indices):
            site_preds[gi] = preds[local_i] if local_i < len(preds) else ""

    return site_preds, len(windows)


# ---- Data loading ----------------------------------------------------------

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


# ---- HF upload helper ------------------------------------------------------

def upload_to_hf(file_path, repo_id, token, path_in_repo=None):
    if not HAS_HF_HUB or not repo_id:
        return False
    try:
        api = HfApi(token=token)
        api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True)
        filename = os.path.basename(file_path)
        if path_in_repo is None:
            path_in_repo = f"dreamon_benchmark/{filename}"
        print(f"    Uploading {filename} to {repo_id}...")
        api.upload_file(path_or_fileobj=file_path, path_in_repo=path_in_repo,
                        repo_id=repo_id, token=token, repo_type="dataset")
        print("    Upload OK.")
        return True
    except Exception as e:
        print(f"    Upload failed: {e}")
        return False


# ---- Main benchmark --------------------------------------------------------

def run(max_samples=None, hf_repo=None, hf_token=None, debug=False,
        context_chars=CONTEXT_CHARS,
        num_mask_per_site=NUM_MASK_PER_SITE,
        max_new_tokens=MAX_NEW_TOKENS,
        max_sites=MAX_SITES_IN_WINDOW):
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print(f"Loading data from {DATA_PATH}...")
    data = load_data(DATA_PATH, max_samples=max_samples)
    print(f"Loaded {len(data)} samples.")

    print(f"\nLoading {MODEL_ID} on {DEVICE}...")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16 if DEVICE == "cuda" else torch.float32,
        trust_remote_code=True,
    ).to(DEVICE).eval()
    print(f"Model loaded in {time.time() - t0:.1f}s")
    print(f"  bos_token_id  = {tokenizer.bos_token_id}")
    print(f"  eos_token_id  = {tokenizer.eos_token_id}")
    print(f"  mask_token_id = {tokenizer.mask_token_id}  ({tokenizer.mask_token!r})")

    # ---- Sanity check ---------------------------------------------------
    if debug:
        print("\n[sanity] Quickstart-style infill on a tiny example:")
        sane_window = "public int add(int a, int b) {\n    return a + [MASK];\n}\n"
        preds, n_windows = predict_one(
            model, tokenizer, sane_window,
            context_chars=10000, num_mask_per_site=num_mask_per_site,
            max_new_tokens=max_new_tokens, max_sites=max_sites, debug=True,
        )
        print(f"[sanity] preds={preds} n_windows={n_windows} (any of a/b/1/2 is fine)\n")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    site_file = os.path.join(RESULTS_DIR, f"DreamOn-7B_per_site_{timestamp}.csv")
    sample_file = os.path.join(RESULTS_DIR, f"DreamOn-7B_per_sample_{timestamp}.csv")

    site_rows = []
    sample_rows = []
    site_correct = 0
    site_total = 0
    sample_majority_correct = 0
    sample_any_correct = 0
    errors = 0

    for idx, row in enumerate(tqdm(data, desc="DreamOn-7B")):
        item_id = row["id"]
        masked_code = row["masked_code"]
        ground_truth = row["target"]
        n_total_masks = masked_code.count("[MASK]")

        try:
            site_preds, n_windows = predict_one(
                model, tokenizer, masked_code,
                context_chars=context_chars,
                num_mask_per_site=num_mask_per_site,
                max_new_tokens=max_new_tokens,
                max_sites=max_sites,
                debug=debug and idx < 2,
            )

            for site_idx, pred in enumerate(site_preds):
                is_c = (pred == ground_truth)
                site_total += 1
                if is_c:
                    site_correct += 1
                site_rows.append({
                    "id": item_id,
                    "site_idx": site_idx,
                    "ground_truth": ground_truth,
                    "prediction": pred,
                    "correct": is_c,
                })

            if site_preds:
                mode_pred, mode_count = Counter(site_preds).most_common(1)[0]
            else:
                mode_pred, mode_count = "", 0
            maj_correct = (mode_pred == ground_truth)
            any_correct = any(p == ground_truth for p in site_preds)
            if maj_correct:
                sample_majority_correct += 1
            if any_correct:
                sample_any_correct += 1

            sample_rows.append({
                "id": item_id,
                "ground_truth": ground_truth,
                "n_total_masks": n_total_masks,
                "n_windows": n_windows,
                "predictions": "|".join(site_preds),
                "majority_pred": mode_pred,
                "majority_count": mode_count,
                "majority_correct": maj_correct,
                "any_correct": any_correct,
            })

        except Exception as e:
            errors += 1
            sample_rows.append({
                "id": item_id, "ground_truth": ground_truth,
                "n_total_masks": n_total_masks, "n_windows": 0,
                "predictions": "", "majority_pred": "", "majority_count": 0,
                "majority_correct": False, "any_correct": False,
                "error": str(e),
            })
            if errors <= 5:
                print(f"  Error on {item_id}: {e}")

    # ---- Save ----------------------------------------------------------
    site_fields = ["id", "site_idx", "ground_truth", "prediction", "correct"]
    with open(site_file, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=site_fields)
        w.writeheader()
        for r in site_rows:
            w.writerow({k: r.get(k, "") for k in site_fields})

    sample_fields = ["id", "ground_truth", "n_total_masks", "n_windows",
                     "predictions", "majority_pred", "majority_count",
                     "majority_correct", "any_correct", "error"]
    with open(sample_file, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=sample_fields)
        w.writeheader()
        for r in sample_rows:
            w.writerow({k: r.get(k, "") for k in sample_fields})

    site_acc = site_correct / site_total if site_total else 0.0
    samp_maj_acc = sample_majority_correct / len(data) if data else 0.0
    samp_any_acc = sample_any_correct / len(data) if data else 0.0

    print(f"\n=== DreamOn-7B (tiled, all-site coverage) ===")
    print(f"Site-level EM:               {site_correct}/{site_total} = {site_acc:.2%}")
    print(f"Sample-level EM (majority):  {sample_majority_correct}/{len(data)} = {samp_maj_acc:.2%}")
    print(f"Sample-level EM (any-site):  {sample_any_correct}/{len(data)} = {samp_any_acc:.2%}")
    print(f"Errors:  {errors}")
    print(f"Per-site:    {site_file}")
    print(f"Per-sample:  {sample_file}")

    if hf_repo:
        upload_to_hf(site_file, hf_repo, hf_token)
        upload_to_hf(sample_file, hf_repo, hf_token)

    summary_file = os.path.join(RESULTS_DIR, f"summary_{timestamp}.csv")
    with open(summary_file, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=[
            "model", "site_acc", "site_correct", "site_total",
            "sample_majority_acc", "sample_any_acc", "samples", "errors",
            "context_chars", "num_mask_per_site", "max_new_tokens", "max_sites",
        ])
        w.writeheader()
        w.writerow({
            "model": "DreamOn-7B",
            "site_acc": f"{site_acc:.4f}",
            "site_correct": site_correct, "site_total": site_total,
            "sample_majority_acc": f"{samp_maj_acc:.4f}",
            "sample_any_acc": f"{samp_any_acc:.4f}",
            "samples": len(data), "errors": errors,
            "context_chars": context_chars,
            "num_mask_per_site": num_mask_per_site,
            "max_new_tokens": max_new_tokens,
            "max_sites": max_sites,
        })
    if hf_repo:
        upload_to_hf(summary_file, hf_repo, hf_token)

    del model, tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


# ---- CLI -------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark DreamOn-7B on refineID.")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--context-chars", type=int, default=CONTEXT_CHARS,
                        help="Window size in chars around the first [MASK].")
    parser.add_argument("--num-mask-per-site", type=int, default=NUM_MASK_PER_SITE,
                        help="Initial <|mask|> tokens per [MASK] site (default 4).")
    parser.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS,
                        help="Max canvas size after expansion (default 64).")
    parser.add_argument("--max-sites", type=int, default=MAX_SITES_IN_WINDOW,
                        help="Cap on number of [MASK] sites kept in the window.")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--hf-repo", type=str, default=None)
    parser.add_argument("--hf-token", type=str, default=os.environ.get("HF_TOKEN"))
    args = parser.parse_args()

    if args.max_new_tokens < args.num_mask_per_site:
        print(f"ERROR: --max-new-tokens ({args.max_new_tokens}) must be "
              f">= --num-mask-per-site ({args.num_mask_per_site}).")
        sys.exit(1)

    run(
        max_samples=args.max_samples,
        hf_repo=args.hf_repo,
        hf_token=args.hf_token,
        debug=args.debug,
        context_chars=args.context_chars,
        num_mask_per_site=args.num_mask_per_site,
        max_new_tokens=args.max_new_tokens,
        max_sites=args.max_sites,
    )
