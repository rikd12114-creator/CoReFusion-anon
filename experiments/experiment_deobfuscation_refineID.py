"""
RQ2 — Deobfuscation Experiment on RefineID Dataset

Three modes for evaluating localization–filling asymmetry:

  --mode all-masked   (default)
      Mask ALL identifiers simultaneously. Tests full deobfuscation ability.
      Expected: very low EM — demonstrates filling fails without context.

  --mode target-only
      Obfuscate ALL variables to a,b,c but only mask the TARGET variable.
      The rest of the code retains obfuscated (bad) names as context.
      Directly comparable with RQ1: same single-target fill, degraded context.

  --mode sequential
      Mask and fill one identifier at a time (most-frequent first).
      Each filled identifier is inserted back before the next is attempted.
      Tests whether iterative recovery improves over all-at-once.

Models: DiffuCoder-7B-Base, DreamCoder-7B
Data:   data/test.csv (same 1000 samples as RQ1 benchmark)

Usage:
    python experiments/experiment_deobfuscation_refineID.py --mode target-only --max-samples 50
    python experiments/experiment_deobfuscation_refineID.py --mode all-masked
    python experiments/experiment_deobfuscation_refineID.py --mode sequential --max-samples 100
"""

import os
import sys
import csv
import re
import gc
import json
import argparse
import time
from datetime import datetime
from typing import List, Dict, Tuple, Optional
from collections import Counter

import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

try:
    from huggingface_hub import HfApi
    HAS_HF_HUB = True
except ImportError:
    HAS_HF_HUB = False

# --- Mock torchvision for DiffuCoder/DreamCoder compatibility ---
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

# ---- Configuration ----------------------------------------------------------

DATA_PATH = "data/test.csv"
RESULTS_DIR = "results/deobfuscation_refineID"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

NUM_MASK_TOKENS = 2          # Match RQ1 benchmark
MAX_INPUT_TOKENS = 16384     # DiffuCoder supports 131072; generous limit
DIFFUSION_STEPS = 64         # Match RQ1 benchmark
SENTINEL = "[DEOBF_MASK]"   # Sentinel for template-based extraction

# ---- Model Registry ---------------------------------------------------------

MODEL_REGISTRY = {
    "DiffuCoder-7B": {
        "id": "apple/DiffuCoder-7B-Base",
        "mask_token": "<|mask|>",
    },
    "DreamCoder-7B": {
        "id": "Dream-org/Dream-Coder-v0-Instruct-7B",
        "mask_token": "<|mask|>",
    },
}

# ---- Java Keywords & Types (for identifier filtering) -----------------------

JAVA_KEYWORDS = {
    'abstract', 'assert', 'boolean', 'break', 'byte', 'case', 'catch', 'char',
    'class', 'const', 'continue', 'default', 'do', 'double', 'else', 'enum',
    'extends', 'final', 'finally', 'float', 'for', 'goto', 'if', 'implements',
    'import', 'instanceof', 'int', 'interface', 'long', 'native', 'new',
    'package', 'private', 'protected', 'public', 'return', 'short', 'static',
    'strictfp', 'super', 'switch', 'synchronized', 'this', 'throw', 'throws',
    'transient', 'try', 'void', 'volatile', 'while', 'true', 'false', 'null',
    'var', 'record', 'sealed', 'permits', 'yield',
}

COMMON_TYPES = {
    'String', 'Integer', 'Long', 'Double', 'Float', 'Boolean', 'Character',
    'Object', 'Class', 'System', 'Math', 'List', 'ArrayList', 'Map', 'HashMap',
    'Set', 'HashSet', 'Collection', 'Iterator', 'Exception', 'RuntimeException',
    'Thread', 'Runnable', 'Comparable', 'Serializable', 'Override', 'Deprecated',
    'SuppressWarnings', 'FunctionalInterface', 'Test', 'Before', 'After',
}


# ---- Identifier Extraction --------------------------------------------------

def extract_java_identifiers_regex(code: str) -> List[Dict]:
    """
    Extract Java LOCAL variable/parameter identifiers via regex.
    Skips: import/package lines, qualified-name segments (after '.'),
    method calls (followed by '('), annotations (after '@'),
    single-char identifiers that are ambiguous.
    """
    results = []
    identifier_pattern = re.compile(r'\b([a-z_][a-zA-Z0-9_]*)\b')

    for line_num, line in enumerate(code.split('\n'), 1):
        stripped = line.strip()
        # Skip comments
        if stripped.startswith('//') or stripped.startswith('*') or stripped.startswith('/*'):
            continue
        # Skip import and package declarations — these are not local variables
        if stripped.startswith('import ') or stripped.startswith('package '):
            continue
        # Skip annotation lines
        if stripped.startswith('@'):
            continue

        for match in identifier_pattern.finditer(line):
            name = match.group(1)
            col = match.start()

            if name in JAVA_KEYWORDS or name in COMMON_TYPES:
                continue
            # Skip method calls / declarations: identifier followed by '('
            if match.end() < len(line) and line[match.end()] == '(':
                continue
            # Skip qualified-name segments: identifier preceded by '.'
            # e.g. in "org.apache.dubbo", skip "apache" and "dubbo"
            if col > 0 and line[col - 1] == '.':
                continue
            # Skip identifiers followed by '.' — likely package/class prefix
            # e.g. in "System.out", skip "out" (already caught above)
            # But also skip "org" in "org.apache" (it's followed by '.')
            if match.end() < len(line) and line[match.end()] == '.':
                continue
            # Skip single-char names that are likely loop vars in original code
            # (we only want to obfuscate meaningful multi-char variables)
            # Exception: common single-char params like 'e', 'i', 'j', 'k' ARE variables
            # but they're too ambiguous for deobfuscation testing, skip them
            if len(name) == 1:
                continue

            results.append({'name': name, 'line': line_num, 'col': col})

    seen = set()
    unique = []
    for item in results:
        key = (item['line'], item['col'])
        if key not in seen:
            seen.add(key)
            unique.append(item)
    return sorted(unique, key=lambda x: (x['line'], x['col']))


def extract_identifiers_with_positions(code: str) -> List[Dict]:
    """Find all occurrences of each unique identifier with (line, col) positions."""
    identifiers = extract_java_identifiers_regex(code)
    unique_names = set(item['name'] for item in identifiers)

    results = []
    for line_num, line in enumerate(code.split('\n'), 1):
        for name in unique_names:
            for match in re.finditer(r'\b' + re.escape(name) + r'\b', line):
                results.append({'name': name, 'line': line_num, 'col': match.start()})
    return sorted(results, key=lambda x: (x['line'], x['col']))


# ---- Obfuscation ------------------------------------------------------------

def obfuscate_java_code(code: str) -> Tuple[str, Dict[str, str]]:
    """
    Replace all identifiers with single/double-letter names (a, b, c, ..., aa, ab, ...).
    Returns (obfuscated_code, mapping: {original_name -> obfuscated_name}).
    """
    identifiers = extract_identifiers_with_positions(code)
    if not identifiers:
        return code, {}

    # Unique names in first-appearance order
    unique_names = []
    seen = set()
    for item in identifiers:
        if item['name'] not in seen:
            unique_names.append(item['name'])
            seen.add(item['name'])

    # Build mapping
    obfuscation_map = {}
    for idx, name in enumerate(unique_names):
        if idx < 26:
            obfuscation_map[name] = chr(ord('a') + idx)
        else:
            obfuscation_map[name] = chr(ord('a') + (idx - 26) // 26) + chr(ord('a') + (idx - 26) % 26)

    # Replace in reverse position order to preserve offsets
    lines = code.split('\n')
    for item in sorted(identifiers, key=lambda x: (x['line'], x['col']), reverse=True):
        li = item['line'] - 1
        ci = item['col']
        orig = item['name']
        if li < len(lines):
            line = lines[li]
            lines[li] = line[:ci] + obfuscation_map[orig] + line[ci + len(orig):]

    return '\n'.join(lines), obfuscation_map


# ---- Mask Insertion & Template Building -------------------------------------

def prepare_masked_input(
    obfuscated_code: str,
    obfuscation_map: Dict[str, str],
    mask_token: str,
    num_mask_tokens: int,
) -> Tuple[str, str, List[Dict]]:
    """
    Replace every obfuscated identifier occurrence with <|mask|> tokens.

    Returns:
        masked_code:  string with <|mask|> tokens (fed to model)
        template:     parallel string with [DEOBF_MASK] sentinels (for extraction)
        positions:    list of {obfuscated, original} per replacement site
    """
    reverse_map = {v: k for k, v in obfuscation_map.items()}
    obf_names = sorted(reverse_map.keys(), key=len, reverse=True)

    if not obf_names:
        return obfuscated_code, obfuscated_code, []

    pattern = re.compile(r'\b(' + '|'.join(re.escape(n) for n in obf_names) + r')\b')

    positions = []
    mask_replacement = mask_token * num_mask_tokens

    def replacer_masked(m):
        positions.append({'obfuscated': m.group(0), 'original': reverse_map[m.group(0)]})
        return mask_replacement

    # Reset positions for template pass
    masked_code = pattern.sub(replacer_masked, obfuscated_code)

    # Build template with sentinels (separate pass to get correct string)
    template = pattern.sub(SENTINEL, obfuscated_code)

    return masked_code, template, positions


# ---- Prediction Extraction --------------------------------------------------

def extract_deobfuscation_predictions(full_output: str, template: str) -> List[str]:
    """
    Extract predicted identifiers from model output using anchor-based matching.
    Mirrors benchmark's extract_all_predictions() but uses DEOBF_MASK sentinel.
    """
    parts = template.split(SENTINEL)
    if len(parts) <= 1:
        return []

    predictions = []
    search_start = 0

    for i in range(len(parts) - 1):
        pre = parts[i]
        post = parts[i + 1]

        pre_anchor = pre.strip()[-30:] if len(pre.strip()) > 30 else pre.strip()
        post_anchor = post.strip()[:30] if len(post.strip()) > 30 else post.strip()

        # Locate pre_anchor
        if pre_anchor:
            idx_start = full_output.find(pre_anchor, search_start)
            if idx_start != -1:
                idx_start += len(pre_anchor)
            else:
                idx_start = search_start
        else:
            idx_start = search_start

        # Locate post_anchor
        if post_anchor:
            idx_end = full_output.find(post_anchor, idx_start)
        else:
            idx_end = -1

        # Extract gap
        if idx_end != -1:
            gap = full_output[idx_start:idx_end].strip()
            search_start = idx_end
        else:
            gap = full_output[idx_start:idx_start + 60].strip()
            search_start = idx_start + 60

        # Extract first valid Java identifier from gap
        m = re.search(r'[a-zA-Z_$][a-zA-Z0-9_$]*', gap)
        predictions.append(m.group(0) if m else "")

    return predictions


# ---- Evaluation -------------------------------------------------------------

def evaluate_predictions(
    predictions: List[str],
    position_records: List[Dict],
) -> Dict:
    """
    Evaluate deobfuscation quality using majority-vote per unique identifier.
    """
    if not predictions or not position_records:
        return {
            'identifiers_correct': 0, 'identifiers_total': 0,
            'per_sample_em_rate': 0.0, 'meaningful_rate': 0.0,
            'predicted_obfuscated_rate': 0.0,
            'originals': {}, 'majority_predictions': {},
        }

    n = min(len(predictions), len(position_records))

    # Group predictions by original identifier name
    groups = {}  # original_name -> list of predictions
    for i in range(n):
        orig = position_records[i]['original']
        obf = position_records[i]['obfuscated']
        pred = predictions[i]
        if orig not in groups:
            groups[orig] = {'predictions': [], 'obfuscated': obf}
        groups[orig]['predictions'].append(pred)

    identifiers_correct = 0
    identifiers_total = len(groups)
    meaningful_count = 0
    predicted_obfuscated = 0
    majority_predictions = {}

    for orig_name, info in groups.items():
        preds = info['predictions']
        obf_name = info['obfuscated']

        # Majority vote (most common non-empty prediction)
        counter = Counter(p for p in preds if p)
        if counter:
            majority_pred = counter.most_common(1)[0][0]
        else:
            majority_pred = ""

        majority_predictions[orig_name] = majority_pred

        if majority_pred == orig_name:
            identifiers_correct += 1
        if len(majority_pred) > 1:
            meaningful_count += 1
        if majority_pred == obf_name:
            predicted_obfuscated += 1

    em_rate = identifiers_correct / identifiers_total if identifiers_total else 0.0
    meaningful_rate = meaningful_count / identifiers_total if identifiers_total else 0.0
    obf_rate = predicted_obfuscated / identifiers_total if identifiers_total else 0.0

    return {
        'identifiers_correct': identifiers_correct,
        'identifiers_total': identifiers_total,
        'per_sample_em_rate': em_rate,
        'meaningful_rate': meaningful_rate,
        'predicted_obfuscated_rate': obf_rate,
        'originals': {orig: info['obfuscated'] for orig, info in groups.items()},
        'majority_predictions': majority_predictions,
        # RAW per-site predictions per identifier group, in site order. Needed
        # for the all-sites-consistency metric (analysis/identifier_similarity_
        # metrics.eval_sample): an identifier is "consistent" only if every site
        # emits the SAME non-empty name. Empties are kept on purpose.
        'group_site_predictions': {orig: info['predictions'] for orig, info in groups.items()},
    }


# ---- Sequential Fill --------------------------------------------------------

def _run_sequential_fill(model, tokenizer, obfuscated_code, obfuscation_map,
                         mask_token, num_mask_tokens, diffusion_steps, device,
                         max_input_tokens):
    """
    Fill identifiers one at a time, most-frequent first.
    After each fill, substitute the prediction back into the code for the next round.
    """
    reverse_map = {v: k for k, v in obfuscation_map.items()}
    current_code = obfuscated_code

    # Sort identifiers by occurrence count (most frequent first — gives model most context benefit)
    occ_counts = {}
    for obf_name in reverse_map:
        occ_counts[obf_name] = len(re.findall(r'\b' + re.escape(obf_name) + r'\b', current_code))
    sorted_obf = sorted(reverse_map.keys(), key=lambda n: occ_counts.get(n, 0), reverse=True)

    majority_predictions = {}
    originals = {}

    for obf_name in sorted_obf:
        orig_name = reverse_map[obf_name]
        originals[orig_name] = obf_name

        # Mask only this one identifier
        single_map = {orig_name: obf_name}
        try:
            input_code, template, positions = prepare_masked_input(
                current_code, single_map, mask_token, num_mask_tokens,
            )
        except Exception:
            majority_predictions[orig_name] = ""
            continue

        inputs = tokenizer(input_code, return_tensors="pt")
        input_ids = inputs.input_ids.to(device)
        attention_mask = inputs.attention_mask.to(device)

        if input_ids.shape[1] > max_input_tokens:
            majority_predictions[orig_name] = ""
            continue

        try:
            with torch.no_grad():
                output = model.diffusion_generate(
                    input_ids, attention_mask=attention_mask,
                    max_new_tokens=1, steps=diffusion_steps,
                    temperature=0.3, top_p=0.95, alg="entropy", alg_temp=0.,
                )
            generated_ids = output.sequences[0] if hasattr(output, "sequences") else output[0]
            full_output = tokenizer.decode(generated_ids, skip_special_tokens=True)

            preds = extract_deobfuscation_predictions(full_output, template)
            # Majority vote across occurrences
            counter = Counter(p for p in preds if p)
            pred = counter.most_common(1)[0][0] if counter else ""
        except Exception:
            pred = ""

        majority_predictions[orig_name] = pred

        # Substitute the prediction back into code for next identifier
        if pred:
            pattern = re.compile(r'\b' + re.escape(obf_name) + r'\b')
            current_code = pattern.sub(pred, current_code)

    # Compute metrics
    total = len(majority_predictions)
    correct = sum(1 for orig, pred in majority_predictions.items() if pred == orig)
    meaningful = sum(1 for pred in majority_predictions.values() if len(pred) > 1)
    obf_back = sum(1 for orig in majority_predictions
                   if majority_predictions[orig] == originals.get(orig, ''))

    return {
        'identifiers_correct': correct,
        'identifiers_total': total,
        'per_sample_em_rate': correct / total if total else 0.0,
        'meaningful_rate': meaningful / total if total else 0.0,
        'predicted_obfuscated_rate': obf_back / total if total else 0.0,
        'majority_predictions': majority_predictions,
        'originals': originals,
        'num_occurrences': sum(occ_counts.values()),
        'token_length': 0,
    }


# ---- Data Loading -----------------------------------------------------------

def load_data(data_path, max_samples=None):
    csv.field_size_limit(sys.maxsize)
    rows = []
    with open(data_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for i, row in enumerate(reader):
            if max_samples is not None and i >= max_samples:
                break
            if len(row) < 3:
                continue
            rows.append({"id": row[0], "masked_code": row[1], "target": row[2].strip()})
    return rows


# ---- HF Upload --------------------------------------------------------------

def upload_to_hf(file_path, repo_id, token, path_in_repo=None):
    if not HAS_HF_HUB or not repo_id:
        return False
    try:
        api = HfApi(token=token)
        api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True)
        filename = os.path.basename(file_path)
        if path_in_repo is None:
            path_in_repo = f"deobfuscation_benchmark/{filename}"
        print(f"    Uploading {filename} to {repo_id}...")
        api.upload_file(
            path_or_fileobj=file_path, path_in_repo=path_in_repo,
            repo_id=repo_id, token=token, repo_type="dataset",
        )
        return True
    except Exception as e:
        print(f"    Upload failed: {e}")
        return False


# ---- Main Experiment --------------------------------------------------------

def run_experiment(target_models=None, max_samples=None, hf_repo=None, hf_token=None,
                   mode="all-masked"):
    os.makedirs(RESULTS_DIR, exist_ok=True)

    if target_models:
        models_to_run = {k: MODEL_REGISTRY[k] for k in target_models if k in MODEL_REGISTRY}
        if not models_to_run:
            print(f"ERROR: No valid models. Available: {list(MODEL_REGISTRY.keys())}")
            return
    else:
        models_to_run = MODEL_REGISTRY

    print(f"Loading data from {DATA_PATH}...")
    data = load_data(DATA_PATH, max_samples=max_samples)
    print(f"Loaded {len(data)} samples.")
    print(f"Mode: {mode}\n")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    all_summaries = []

    for model_name, meta in models_to_run.items():
        print(f"{'=' * 60}")
        print(f"  Model: {model_name}  ({meta['id']})")
        print(f"{'=' * 60}")

        # ── Load model ──────────────────────────────────────────────
        t0 = time.time()
        try:
            tokenizer = AutoTokenizer.from_pretrained(meta["id"], trust_remote_code=True)
            model = AutoModel.from_pretrained(
                meta["id"],
                torch_dtype=torch.bfloat16 if DEVICE == "cuda" else torch.float32,
                trust_remote_code=True,
            ).to(DEVICE).eval()
            print(f"  Loaded in {time.time() - t0:.1f}s")
        except Exception as e:
            print(f"  FAILED to load: {e}")
            all_summaries.append({"model": model_name, "error": str(e)})
            continue

        mask_token = meta["mask_token"]

        # ── Inference loop ──────────────────────────────────────────
        results = []
        total_id_correct = 0
        total_id_count = 0
        skipped = 0
        errors = 0

        for row in tqdm(data, desc=f"  {model_name}"):
            item_id = row["id"]
            masked_code_raw = row["masked_code"]
            target = row["target"]

            try:
                # 1. Reconstruct original code
                original_code = masked_code_raw.replace("[MASK]", target)

                # 2. Obfuscate ALL identifiers
                obfuscated_code, obfuscation_map = obfuscate_java_code(original_code)
                if not obfuscation_map:
                    results.append({"id": item_id, "skipped": "no_identifiers"})
                    skipped += 1
                    continue

                # 3. Build input depending on mode
                if mode == "target-only":
                    # Only mask the TARGET variable; keep other obfuscated names
                    # Find the obfuscated name for the target
                    target_obf = obfuscation_map.get(target)
                    if not target_obf:
                        results.append({"id": item_id, "skipped": "target_not_in_map"})
                        skipped += 1
                        continue
                    # Replace only the target's obfuscated name with masks
                    target_mask_map = {target: target_obf}
                    input_code, template, positions = prepare_masked_input(
                        obfuscated_code, target_mask_map, mask_token, NUM_MASK_TOKENS,
                    )
                elif mode == "sequential":
                    # Will be handled below in a separate loop
                    pass
                else:  # all-masked
                    input_code, template, positions = prepare_masked_input(
                        obfuscated_code, obfuscation_map, mask_token, NUM_MASK_TOKENS,
                    )

                # --- Sequential mode: fill one identifier at a time ---
                if mode == "sequential":
                    seq_results = _run_sequential_fill(
                        model, tokenizer, obfuscated_code, obfuscation_map,
                        mask_token, NUM_MASK_TOKENS, DIFFUSION_STEPS, DEVICE,
                        MAX_INPUT_TOKENS,
                    )
                    if seq_results is None:
                        results.append({"id": item_id, "skipped": "seq_failed"})
                        skipped += 1
                        continue
                    metrics = seq_results
                    total_id_correct += metrics['identifiers_correct']
                    total_id_count += metrics['identifiers_total']
                    results.append({
                        "id": item_id,
                        "num_unique_identifiers": metrics['identifiers_total'],
                        "num_occurrences": metrics.get('num_occurrences', 0),
                        "identifiers_correct": metrics['identifiers_correct'],
                        "identifiers_total": metrics['identifiers_total'],
                        "per_sample_em_rate": f"{metrics['per_sample_em_rate']:.4f}",
                        "meaningful_rate": f"{metrics['meaningful_rate']:.4f}",
                        "predicted_obfuscated_rate": f"{metrics['predicted_obfuscated_rate']:.4f}",
                        "predictions_json": json.dumps(metrics['majority_predictions']),
                        "site_predictions_json": json.dumps(metrics.get('group_site_predictions', {})),
                        "originals_json": json.dumps(metrics['originals']),
                        "mapping_json": json.dumps(obfuscation_map),
                        "token_length": metrics.get('token_length', 0),
                    })
                    continue

                # --- Non-sequential modes (all-masked, target-only) ---
                # 4. Tokenize and check length
                inputs = tokenizer(input_code, return_tensors="pt")
                input_ids = inputs.input_ids.to(DEVICE)
                attention_mask = inputs.attention_mask.to(DEVICE)

                if input_ids.shape[1] > MAX_INPUT_TOKENS:
                    results.append({
                        "id": item_id, "skipped": "too_long",
                        "token_length": input_ids.shape[1],
                    })
                    skipped += 1
                    continue

                # 5. Diffusion generation (same params as RQ1 benchmark)
                with torch.no_grad():
                    output = model.diffusion_generate(
                        input_ids,
                        attention_mask=attention_mask,
                        max_new_tokens=1,
                        steps=DIFFUSION_STEPS,
                        temperature=0.3,
                        top_p=0.95,
                        alg="entropy",
                        alg_temp=0.,
                    )

                generated_ids = output.sequences[0] if hasattr(output, "sequences") else output[0]
                full_output = tokenizer.decode(generated_ids, skip_special_tokens=True)

                # 6. Extract predictions
                predictions = extract_deobfuscation_predictions(full_output, template)

                # 7. Evaluate
                metrics = evaluate_predictions(predictions, positions)

                total_id_correct += metrics['identifiers_correct']
                total_id_count += metrics['identifiers_total']

                results.append({
                    "id": item_id,
                    "num_unique_identifiers": metrics['identifiers_total'],
                    "num_occurrences": len(positions),
                    "identifiers_correct": metrics['identifiers_correct'],
                    "identifiers_total": metrics['identifiers_total'],
                    "per_sample_em_rate": f"{metrics['per_sample_em_rate']:.4f}",
                    "meaningful_rate": f"{metrics['meaningful_rate']:.4f}",
                    "predicted_obfuscated_rate": f"{metrics['predicted_obfuscated_rate']:.4f}",
                    "predictions_json": json.dumps(metrics['majority_predictions']),
                    "site_predictions_json": json.dumps(metrics.get('group_site_predictions', {})),
                    "originals_json": json.dumps(metrics['originals']),
                    "mapping_json": json.dumps(obfuscation_map),
                    "token_length": input_ids.shape[1],
                })

            except torch.cuda.OutOfMemoryError:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                results.append({"id": item_id, "error": "CUDA_OOM"})
                errors += 1
            except Exception as e:
                results.append({"id": item_id, "error": str(e)})
                errors += 1
                if errors <= 5:
                    print(f"    Error on {item_id}: {e}")

        # ── Save results ────────────────────────────────────────────
        out_file = os.path.join(RESULTS_DIR, f"{model_name}_{mode}_{timestamp}.csv")
        if results:
            all_keys = set()
            for r in results:
                all_keys.update(r.keys())
            all_keys = sorted(all_keys)
            with open(out_file, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=all_keys)
                writer.writeheader()
                for r in results:
                    writer.writerow({k: r.get(k, "") for k in all_keys})

        overall_em = total_id_correct / total_id_count if total_id_count else 0
        processed = len(data) - skipped - errors

        print(f"\n  --- {model_name} Summary ---")
        print(f"  Processed:  {processed}/{len(data)}  (skipped={skipped}, errors={errors})")
        print(f"  Identifier EM: {total_id_correct}/{total_id_count} = {overall_em:.2%}")
        print(f"  Results: {out_file}\n")

        if hf_repo:
            upload_to_hf(out_file, hf_repo, hf_token)

        all_summaries.append({
            "model": model_name,
            "hf_id": meta["id"],
            "processed": processed,
            "skipped": skipped,
            "errors": errors,
            "identifier_em": f"{overall_em:.4f}",
            "id_correct": total_id_correct,
            "id_total": total_id_count,
        })

        del model, tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    # ── Summary table ───────────────────────────────────────────────
    summary_file = os.path.join(RESULTS_DIR, f"summary_{timestamp}.csv")
    if all_summaries:
        with open(summary_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(all_summaries[0].keys()))
            writer.writeheader()
            writer.writerows(all_summaries)

    print(f"{'=' * 60}")
    print("  DEOBFUSCATION BENCHMARK SUMMARY")
    print(f"{'=' * 60}")
    print(f"  {'Model':<20} {'Identifier EM':>14} {'Processed':>10} {'Skipped':>8}")
    print(f"  {'-' * 56}")
    for s in all_summaries:
        if "error" in s and "identifier_em" not in s:
            print(f"  {s['model']:<20} LOAD FAILED")
        else:
            print(f"  {s['model']:<20} {s['identifier_em']:>14} {s['processed']:>10} {s['skipped']:>8}")
    print(f"\n  Summary: {summary_file}")


# ---- CLI --------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="RQ2 — Deobfuscation benchmark for diffusion LLMs on RefineID",
        allow_abbrev=False,  # prevent --mode being matched as --model prefix
    )
    parser.add_argument(
        "--model", action="append", default=None,
        help="Model name(s) from registry. Repeatable. Default: all.",
    )
    parser.add_argument(
        "--mode", choices=["all-masked", "target-only", "sequential"],
        default="all-masked",
        help="Experiment mode (default: all-masked).",
    )
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--hf-repo", type=str, default=None)
    parser.add_argument("--hf-token", type=str, default=os.environ.get("HF_TOKEN"))
    parser.add_argument("--list-models", action="store_true")

    args = parser.parse_args()

    if args.list_models:
        print("Available models:")
        for k, v in MODEL_REGISTRY.items():
            print(f"  {k:<30} {v['id']}")
        sys.exit(0)

    run_experiment(
        target_models=args.model,
        max_samples=args.max_samples,
        hf_repo=args.hf_repo,
        hf_token=args.hf_token,
        mode=args.mode,
    )
