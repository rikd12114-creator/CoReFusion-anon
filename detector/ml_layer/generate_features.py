"""
End-to-end feature generator for the ML smell-localization head.

Given the RefineID dataset (data/test.csv with `id, masked_code, target`):

  1. For each sample, build TWO versions of the code:
       smell version  → `[MASK]` replaced by a sampled smell identifier
                        (drawn from a configurable severity-mixed vocabulary).
       gt    version  → `[MASK]` replaced by the original `target` name.
  2. Re-mask every Java identifier and run a single 64-step denoising trajectory
     of DiffuCoder. Per token, record:
       * flip_step             (Experiment B)
       * first_confident_step  (Experiment B)
       * entropy series        (Experiment C → mean/max |Δentropy|)
  3. Aggregate per identifier (mean over all occurrences) and emit one row.
  4. Label `is_smell_token` = True iff `identifier_name == injected_smell_name`
     (so the GT version contributes only negatives, providing calibration).

Output:
  detector/ml_layer/features/features_<model_tag>_<timestamp>.csv

Run on GPU. The 64-step denoising at 7B is the expensive bit:
  python detector/ml_layer/generate_features.py --num-samples 200 --runs-per-sample 1
"""

import argparse
import gc
import os
import random
import re
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

# ---- DiffuCoder torchvision shim (matches the rest of the repo) -------------
class _MockMod:
    def __getattr__(self, name): return _MockMod()
    def __call__(self, *a, **k): return _MockMod()


for _name in ("torchvision", "torchvision.ops", "torchvision.transforms"):
    sys.modules[_name] = _MockMod()
if not hasattr(torch.ops, "torchvision"):
    class _DummyOps:
        def nms(*a, **k): return torch.tensor([])
    torch.ops.torchvision = _DummyOps()
# -----------------------------------------------------------------------------

JAVA_KEYWORDS = {
    "public", "static", "int", "if", "return", "void", "class", "for", "new", "boolean",
    "private", "protected", "final", "else", "while", "this", "null", "true", "false",
    "try", "catch", "throw", "throws", "import", "package", "byte", "char", "short", "long",
    "float", "double", "switch", "case", "default", "break", "continue", "interface",
    "extends", "implements",
}

DEFAULT_SMELL_VOCAB = {
    # severity → candidate names (matches detector/code_naming_smell_detector.py)
    "severe":   ["x", "a", "n", "i", "b", "y"],
    "moderate": ["tmp", "val", "foo", "res", "data", "var"],
    "mild":     ["temp1", "val1", "item", "stuff", "obj1"],
}

MODEL_REGISTRY = {
    "DiffuCoder-7B-Instruct": ("apple/DiffuCoder-7B-Instruct", "<|mask|>"),
    "DiffuCoder-7B-Base":     ("apple/DiffuCoder-7B-Base",     "<|mask|>"),
    "DreamCoder-7B":          ("Dream-org/Dream-Coder-v0-Instruct-7B", "<|mask|>"),
}


def get_num_transfer_tokens(mask_index, steps):
    mask_num = mask_index.sum(dim=1, keepdim=True)
    base = mask_num // steps
    remainder = mask_num % steps
    out = torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64) + base
    for i in range(mask_num.size(0)):
        out[i, : remainder[i]] += 1
    return out


def add_gumbel_noise(logits, temperature):
    if temperature == 0:
        return logits
    logits = logits.to(torch.float32)
    noise = torch.rand_like(logits, dtype=torch.float32)
    gumbel = (-torch.log(noise)) ** temperature
    return logits.exp() / gumbel


def find_identifiers(text: str) -> list[tuple[int, int, str]]:
    """Return `(start_byte, end_byte, name)` triples for every non-keyword
    identifier in the source text. Tries tree-sitter first, falls back to regex.
    """
    try:
        from tree_sitter_languages import get_parser
        parser = get_parser("java")
        tree = parser.parse(bytes(text, "utf8"))
        ranges = []
        def visit(node):
            if node.type == "identifier":
                name = text[node.start_byte:node.end_byte]
                if name not in JAVA_KEYWORDS:
                    ranges.append((node.start_byte, node.end_byte, name))
            for c in node.children:
                visit(c)
        visit(tree.root_node)
        return ranges
    except Exception:
        return [(m.start(), m.end(), m.group(0))
                for m in re.finditer(r"\b[A-Za-z_][A-Za-z0-9_]*\b", text)
                if m.group(0) not in JAVA_KEYWORDS]


def map_identifiers_to_tokens(text: str, tokenizer, input_ids):
    """Group token positions by the source-level identifier they belong to."""
    ranges = find_identifiers(text)
    ids = input_ids[0].tolist()
    # Build (start_char, end_char) for every token via a left-prefix decode.
    offsets = []
    for i in range(len(ids)):
        prefix = tokenizer.decode(ids[:i],     skip_special_tokens=False)
        full   = tokenizer.decode(ids[: i + 1], skip_special_tokens=False)
        offsets.append((len(prefix), len(full)))

    mask = torch.zeros(len(ids), dtype=torch.bool)
    groups = []
    for sb, eb, name in ranges:
        idxs = [i for i, (a, b) in enumerate(offsets) if sb <= (a + b) / 2 < eb]
        if idxs:
            for i in idxs:
                mask[i] = True
            groups.append({"name": name, "indices": idxs})
    return mask, groups


def run_probes(model, tokenizer, mask_token_id, code: str, total_steps: int, device: str,
               temperature: float = 0.3, confidence_threshold: float = 0.8):
    """Single denoising trajectory; returns per-token flip_step,
    first_confident_step, and the per-step entropy series.

    This combines Experiments B (flip dynamics) and C (entropy dynamics) into
    one forward-pass loop, halving GPU time vs running them separately.
    """
    inputs = tokenizer(code, return_tensors="pt").to(device)
    input_ids, attn = inputs.input_ids, inputs.attention_mask

    id_mask, id_groups = map_identifiers_to_tokens(code, tokenizer, input_ids)
    if not id_mask.any():
        return None
    id_mask = id_mask.to(device)

    x = input_ids.clone()
    x[0, id_mask] = mask_token_id
    seq_len = x.shape[1]

    n_transfer = get_num_transfer_tokens(id_mask.unsqueeze(0), total_steps)

    flip_step      = {i: -1 for i in range(seq_len)}
    first_conf     = {i: -1 for i in range(seq_len)}
    entropy_series = {i: [] for i in range(seq_len)}

    for step_i in range(total_steps):
        with torch.no_grad():
            cur_mask = (x == mask_token_id)
            if not cur_mask.any():
                break
            out = model(x, attention_mask=attn.bool())
            logits = out.logits

            p_all = F.softmax(logits.float(), dim=-1)
            log_p = torch.log(torch.clamp(p_all, min=1e-10))
            step_entropy = -(p_all * log_p).sum(dim=-1)[0]

            x0 = torch.argmax(add_gumbel_noise(logits, temperature), dim=-1)
            x0_p = torch.gather(p_all, -1, x0.unsqueeze(-1)).squeeze(-1)

            for i in range(seq_len):
                if cur_mask[0, i].item():
                    entropy_series[i].append(step_entropy[i].item())
                    if x0_p[0, i].item() > confidence_threshold and first_conf[i] == -1:
                        first_conf[i] = step_i

            confidence = torch.where(cur_mask, x0_p, torch.tensor(-np.inf, device=device))
            transfer = torch.zeros_like(x0, dtype=torch.bool)
            k = int(n_transfer[0, step_i].item()) if n_transfer.shape[1] > step_i else 0
            k = min(k, int(cur_mask.sum().item()))
            if k > 0:
                _, sel = torch.topk(confidence[0], k=k)
                transfer[0, sel] = True
                x[transfer] = x0[transfer]
                for i in sel.tolist():
                    if flip_step[i] == -1:
                        flip_step[i] = step_i

    return id_groups, flip_step, first_conf, entropy_series


def aggregate_per_identifier(id_groups, flip_step, first_conf, entropy_series, total_steps,
                             injected_smell_name: str | None):
    rows = []
    for g in id_groups:
        name = g["name"]
        idxs = g["indices"]
        flips = [flip_step[i] for i in idxs if flip_step.get(i, -1) != -1]
        confs = [first_conf[i] for i in idxs if first_conf.get(i, -1) != -1]
        avg_flip = sum(flips) / len(flips) if flips else float(total_steps)
        avg_conf = sum(confs) / len(confs) if confs else float(total_steps)

        diffs = []
        for i in idxs:
            ent = entropy_series.get(i, [])
            if len(ent) > 1:
                diffs.extend(np.abs(np.diff(ent)).tolist())
        mean_dh = float(np.mean(diffs)) if diffs else 0.0
        max_dh  = float(np.max(diffs))  if diffs else 0.0

        rows.append({
            "identifier_name": name,
            "is_smell_token":  bool(injected_smell_name is not None and name == injected_smell_name),
            "n_occurrences":   len(idxs),
            "avg_flip_step":         avg_flip,
            "first_confident_step":  avg_conf,
            "mean_entropy_change":   mean_dh,
            "max_entropy_change":    max_dh,
        })
    return rows


def pick_smell_name(target: str, vocab: dict, rng: random.Random) -> str:
    """Sample a smell name distinct from the GT target."""
    pool = [n for sev in vocab for n in vocab[sev] if n != target]
    return rng.choice(pool)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-path", default="data/test.csv")
    p.add_argument("--out-dir",   default="detector/ml_layer/features")
    p.add_argument("--model",     default="DiffuCoder-7B-Instruct",
                   choices=list(MODEL_REGISTRY.keys()))
    p.add_argument("--num-samples", type=int, default=100,
                   help="how many RefineID samples to process")
    p.add_argument("--runs-per-sample", type=int, default=1,
                   help="how many random smell names to inject per sample")
    p.add_argument("--total-steps", type=int, default=64,
                   help="diffusion steps (default 64, matches ABC)")
    p.add_argument("--include-gt-version", action="store_true",
                   help="also probe the GT-named code as all-clean negatives")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("WARNING: no GPU detected — 7B inference on CPU will be impractically slow.",
              file=sys.stderr)

    model_id, mask_tok = MODEL_REGISTRY[args.model]
    print(f"loading {model_id} on {device}")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        model_id,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
    ).to(device).eval()
    mask_token_id = tokenizer.convert_tokens_to_ids(mask_tok)

    df = pd.read_csv(args.data_path, header=None, names=["id", "masked_code", "target"])
    df = df.head(args.num_samples)
    print(f"loaded {len(df)} samples")

    rows = []
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        sample_id = int(row["id"])
        masked_code = str(row["masked_code"])
        target = str(row["target"]).strip()
        if "[MASK]" not in masked_code:
            continue

        try:
            for run_i in range(args.runs_per_sample):
                smell_name = pick_smell_name(target, DEFAULT_SMELL_VOCAB, rng)
                code_smell = masked_code.replace("[MASK]", smell_name)
                result = run_probes(model, tokenizer, mask_token_id, code_smell,
                                    args.total_steps, device)
                if result is None:
                    continue
                id_groups, flip_step, first_conf, ent = result
                for r in aggregate_per_identifier(id_groups, flip_step, first_conf, ent,
                                                  args.total_steps, smell_name):
                    r.update({"sample_id": sample_id, "run_id": run_i,
                              "version": "smell", "injected_name": smell_name,
                              "gt_target": target})
                    rows.append(r)

            if args.include_gt_version:
                code_gt = masked_code.replace("[MASK]", target)
                result = run_probes(model, tokenizer, mask_token_id, code_gt,
                                    args.total_steps, device)
                if result is not None:
                    id_groups, flip_step, first_conf, ent = result
                    for r in aggregate_per_identifier(id_groups, flip_step, first_conf, ent,
                                                      args.total_steps, None):
                        r.update({"sample_id": sample_id, "run_id": -1,
                                  "version": "gt", "injected_name": target,
                                  "gt_target": target})
                        rows.append(r)
        except torch.cuda.OutOfMemoryError:
            print(f"  [OOM] sample {sample_id} skipped", file=sys.stderr)
            torch.cuda.empty_cache(); gc.collect()
            continue

    if not rows:
        print("no rows produced — exiting", file=sys.stderr)
        return

    out_df = pd.DataFrame(rows)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = args.model.replace("/", "_")
    out_path = out_dir / f"features_{tag}_{ts}.csv"
    out_df.to_csv(out_path, index=False)
    pos = int(out_df.is_smell_token.sum())
    print(f"\nwrote {len(out_df):,} rows ({pos} smell, {len(out_df) - pos} clean) "
          f"→ {out_path}")
    print(f"unique samples: {out_df.sample_id.nunique()}")
    print(f"injected names: {sorted(out_df.injected_name.unique().tolist())}")


if __name__ == "__main__":
    main()
