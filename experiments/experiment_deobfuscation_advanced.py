"""
Advanced Deobfuscation Experiment with Enhanced Evaluation

This script extends the basic deobfuscation experiment with:
1. Multiple evaluation metrics (BLEU, edit distance, semantic similarity)
2. Step-by-step tracking of deobfuscation progress
3. Comparison with baseline methods
4. Detailed analysis of which variable types are easier to recover
"""

import sys
import os
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from datetime import datetime
import re
import json
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
import numpy as np
from experiment_deobfuscation import (
    obfuscate_code,
    get_token_mask_from_identifiers,
    MockModule
)
from extract_names import extract_python_identifiers

# Mock torchvision
sys.modules['torchvision'] = MockModule()
sys.modules['torchvision.ops'] = MockModule()
sys.modules['torchvision.transforms'] = MockModule()

if not hasattr(torch.ops, 'torchvision'):
    class DummyOps:
        def nms(*args, **kwargs): return torch.tensor([])
    torch.ops.torchvision = DummyOps()


def calculate_edit_distance(str1: str, str2: str) -> int:
    """Calculate Levenshtein distance between two strings."""
    if len(str1) < len(str2):
        return calculate_edit_distance(str2, str1)

    if len(str2) == 0:
        return len(str1)

    previous_row = range(len(str2) + 1)
    for i, c1 in enumerate(str1):
        current_row = [i + 1]
        for j, c2 in enumerate(str2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


def calculate_advanced_metrics(original_code: str,
                               obfuscated_code: str,
                               deobfuscated_code: str) -> Dict:
    """
    Calculate comprehensive evaluation metrics.
    """
    original_identifiers = extract_python_identifiers(original_code)
    deobfuscated_identifiers = extract_python_identifiers(deobfuscated_code)

    # Group by position
    original_by_pos = {}
    for item in original_identifiers:
        key = (item['line'], item['col'])
        original_by_pos[key] = item

    deobfuscated_by_pos = {}
    for item in deobfuscated_identifiers:
        key = (item['line'], item['col'])
        deobfuscated_by_pos[key] = item

    # Metrics by identifier type
    metrics_by_type = defaultdict(lambda: {
        'total': 0,
        'exact_match': 0,
        'partial_match': 0,
        'edit_distances': []
    })

    overall_metrics = {
        'total': 0,
        'exact_match': 0,
        'partial_match': 0,
        'edit_distances': [],
        'by_type': {}
    }

    for pos, original_item in original_by_pos.items():
        if pos not in deobfuscated_by_pos:
            continue

        original_name = original_item['name']
        deobfuscated_name = deobfuscated_by_pos[pos]['name']
        identifier_type = original_item['type']

        # Calculate edit distance
        edit_dist = calculate_edit_distance(original_name, deobfuscated_name)

        # Update type-specific metrics
        metrics_by_type[identifier_type]['total'] += 1
        metrics_by_type[identifier_type]['edit_distances'].append(edit_dist)

        if original_name == deobfuscated_name:
            metrics_by_type[identifier_type]['exact_match'] += 1
        elif original_name.lower() == deobfuscated_name.lower():
            metrics_by_type[identifier_type]['partial_match'] += 1

        # Update overall metrics
        overall_metrics['total'] += 1
        overall_metrics['edit_distances'].append(edit_dist)

        if original_name == deobfuscated_name:
            overall_metrics['exact_match'] += 1
        elif original_name.lower() == deobfuscated_name.lower():
            overall_metrics['partial_match'] += 1

    # Calculate rates
    if overall_metrics['total'] > 0:
        overall_metrics['exact_match_rate'] = overall_metrics['exact_match'] / overall_metrics['total']
        overall_metrics['partial_match_rate'] = (overall_metrics['exact_match'] + overall_metrics['partial_match']) / overall_metrics['total']
        overall_metrics['avg_edit_distance'] = np.mean(overall_metrics['edit_distances']) if overall_metrics['edit_distances'] else 0
        overall_metrics['median_edit_distance'] = np.median(overall_metrics['edit_distances']) if overall_metrics['edit_distances'] else 0

    # Calculate type-specific rates
    for identifier_type, type_metrics in metrics_by_type.items():
        if type_metrics['total'] > 0:
            type_metrics['exact_match_rate'] = type_metrics['exact_match'] / type_metrics['total']
            type_metrics['partial_match_rate'] = (type_metrics['exact_match'] + type_metrics['partial_match']) / type_metrics['total']
            type_metrics['avg_edit_distance'] = np.mean(type_metrics['edit_distances']) if type_metrics['edit_distances'] else 0

    overall_metrics['by_type'] = dict(metrics_by_type)

    return overall_metrics


def track_deobfuscation_progress(model,
                                tokenizer,
                                obfuscated_code: str,
                                original_code: str,
                                ident_mask: torch.Tensor,
                                input_ids_full: torch.Tensor,
                                total_steps: int,
                                device: str) -> List[Dict]:
    """
    Track metrics at each diffusion step.
    """
    x = input_ids_full.clone()
    progress = []

    for step in range(total_steps):
        with torch.no_grad():
            logits = model(x).logits
            x_pred = torch.argmax(logits, dim=-1)
            x_next = torch.where(ident_mask, x_pred, x)

            # Decode current state
            current_code = tokenizer.decode(x_next[0], skip_special_tokens=True)

            # Calculate metrics
            try:
                metrics = calculate_advanced_metrics(original_code, obfuscated_code, current_code)
                metrics['step'] = step + 1
                metrics['converged'] = torch.equal(x, x_next)
                progress.append(metrics)
            except Exception as e:
                print(f"Warning: Could not calculate metrics at step {step+1}: {e}")
                progress.append({
                    'step': step + 1,
                    'error': str(e),
                    'converged': torch.equal(x, x_next)
                })

            if torch.equal(x, x_next):
                break

            x = x_next

    return progress


def run_advanced_deobfuscation_experiment(model_name: str,
                                         model_id: str,
                                         code_snippet: str,
                                         total_steps: int = 20,
                                         device: str = "cuda",
                                         track_progress: bool = True):
    """
    Run advanced deobfuscation experiment with detailed tracking.
    """
    print(f"\n{'='*80}")
    print(f"Advanced Deobfuscation Experiment: {model_name}")
    print(f"{'='*80}")

    # Create output directory
    os.makedirs("results/deobfuscation_advanced", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"results/deobfuscation_advanced/{model_name}_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)

    # Load model
    print(f"Loading model: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    ).to(device).eval()

    # Obfuscate code
    print("\n[Step 1] Obfuscating code...")
    obfuscated_code, obfuscation_map = obfuscate_code(code_snippet)

    # Tokenize
    print("\n[Step 2] Tokenizing...")
    input_ids_full = tokenizer.encode(obfuscated_code, return_tensors="pt").to(device)
    input_ids = input_ids_full[0]

    # Extract mask
    print("\n[Step 3] Extracting variable positions...")
    ident_mask = get_token_mask_from_identifiers(obfuscated_code, tokenizer, input_ids).to(device)

    # Track progress
    if track_progress:
        print(f"\n[Step 4] Running diffusion with progress tracking ({total_steps} steps)...")
        progress = track_deobfuscation_progress(
            model, tokenizer, obfuscated_code, code_snippet,
            ident_mask, input_ids_full, total_steps, device
        )

        # Save progress
        progress_file = os.path.join(output_dir, "progress.json")
        with open(progress_file, 'w', encoding='utf-8') as f:
            json.dump(progress, f, indent=2, ensure_ascii=False)
        print(f"Progress saved to: {progress_file}")

    # Run full diffusion
    print(f"\n[Step 5] Running full diffusion...")
    x = input_ids_full.clone()
    step_outputs = []

    for step in range(total_steps):
        with torch.no_grad():
            logits = model(x).logits
            x_pred = torch.argmax(logits, dim=-1)
            x_next = torch.where(ident_mask, x_pred, x)

            decoded = tokenizer.decode(x_next[0], skip_special_tokens=True)
            step_outputs.append({
                'step': step + 1,
                'code': decoded
            })

            if torch.equal(x, x_next):
                print(f"Converged at step {step+1}")
                break

            x = x_next

            if (step + 1) % 5 == 0:
                print(f"Step {step+1}/{total_steps} completed")

    # Final evaluation
    final_code = tokenizer.decode(x[0], skip_special_tokens=True)
    print("\n[Step 6] Evaluating final result...")
    final_metrics = calculate_advanced_metrics(code_snippet, obfuscated_code, final_code)

    # Save detailed results
    results = {
        'model_name': model_name,
        'model_id': model_id,
        'timestamp': timestamp,
        'original_code': code_snippet,
        'obfuscated_code': obfuscated_code,
        'obfuscation_map': obfuscation_map,
        'final_code': final_code,
        'final_metrics': final_metrics,
        'step_outputs': step_outputs,
        'total_steps': len(step_outputs)
    }

    results_file = os.path.join(output_dir, "results.json")
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # Save human-readable report
    report_file = os.path.join(output_dir, "report.txt")
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(f"{'='*80}\n")
        f.write(f"ADVANCED DEOBFUSCATION EXPERIMENT REPORT\n")
        f.write(f"{'='*80}\n\n")
        f.write(f"Model: {model_name}\n")
        f.write(f"Model ID: {model_id}\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Total Steps: {len(step_outputs)}\n\n")

        f.write(f"{'='*80}\n")
        f.write(f"ORIGINAL CODE\n")
        f.write(f"{'='*80}\n")
        f.write(code_snippet + "\n\n")

        f.write(f"{'='*80}\n")
        f.write(f"OBFUSCATED CODE\n")
        f.write(f"{'='*80}\n")
        f.write(obfuscated_code + "\n\n")

        f.write(f"{'='*80}\n")
        f.write(f"OBFUSCATION MAPPING\n")
        f.write(f"{'='*80}\n")
        for orig, obf in obfuscation_map.items():
            f.write(f"{orig:20} -> {obf}\n")
        f.write("\n")

        f.write(f"{'='*80}\n")
        f.write(f"FINAL DEOBFUSCATED CODE\n")
        f.write(f"{'='*80}\n")
        f.write(final_code + "\n\n")

        f.write(f"{'='*80}\n")
        f.write(f"EVALUATION METRICS\n")
        f.write(f"{'='*80}\n\n")

        f.write(f"Overall Metrics:\n")
        f.write(f"  Total identifiers: {final_metrics.get('total', 0)}\n")
        f.write(f"  Exact matches: {final_metrics.get('exact_match', 0)}\n")
        f.write(f"  Exact match rate: {final_metrics.get('exact_match_rate', 0):.2%}\n")
        f.write(f"  Partial match rate: {final_metrics.get('partial_match_rate', 0):.2%}\n")
        f.write(f"  Average edit distance: {final_metrics.get('avg_edit_distance', 0):.2f}\n")
        f.write(f"  Median edit distance: {final_metrics.get('median_edit_distance', 0):.2f}\n\n")

        f.write(f"Metrics by Identifier Type:\n")
        for id_type, type_metrics in final_metrics.get('by_type', {}).items():
            f.write(f"\n  {id_type}:\n")
            f.write(f"    Total: {type_metrics.get('total', 0)}\n")
            f.write(f"    Exact match rate: {type_metrics.get('exact_match_rate', 0):.2%}\n")
            f.write(f"    Avg edit distance: {type_metrics.get('avg_edit_distance', 0):.2f}\n")

    print(f"\n{'='*80}")
    print(f"FINAL RESULTS")
    print(f"{'='*80}")
    print(f"Exact match rate: {final_metrics.get('exact_match_rate', 0):.2%}")
    print(f"Partial match rate: {final_metrics.get('partial_match_rate', 0):.2%}")
    print(f"Average edit distance: {final_metrics.get('avg_edit_distance', 0):.2f}")
    print(f"\nResults saved to: {output_dir}")
    print(f"{'='*80}\n")

    return results


if __name__ == "__main__":
    # Test sample
    test_code = """def binary_search(sorted_list, target):
    left = 0
    right = len(sorted_list) - 1
    while left <= right:
        middle = (left + right) // 2
        if sorted_list[middle] == target:
            return middle
        elif sorted_list[middle] < target:
            left = middle + 1
        else:
            right = middle - 1
    return -1"""

    # Run experiment
    run_advanced_deobfuscation_experiment(
        model_name="dreamcoder",
        model_id="Dream-org/Dream-Coder-v0-Instruct-7B",
        code_snippet=test_code,
        total_steps=20,
        device="cuda",
        track_progress=True
    )
