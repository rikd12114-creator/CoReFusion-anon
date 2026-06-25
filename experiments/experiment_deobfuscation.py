"""
Deobfuscation Experiment using Diffusion Language Model

This experiment tests whether a diffusion model can recover meaningful variable names
from obfuscated code when given the positions of all variables via AST extraction.

Workflow:
1. Load clean code samples with meaningful variable names
2. Obfuscate them by replacing all variables with meaningless names (a, b, c, ...)
3. Use AST to extract all variable positions
4. Apply diffusion model with constrained generation (only update variable tokens)
5. Evaluate the quality of deobfuscated variable names
"""

import sys
import os
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from datetime import datetime
import re
import json
from typing import List, Dict, Tuple
from extract_names import extract_python_identifiers

# --- Mock torchvision for compatibility ---
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
# ------------------------------------------


def obfuscate_code(code: str) -> Tuple[str, Dict[str, str]]:
    """
    Obfuscate Python code by replacing all identifiers with meaningless names.

    Returns:
        obfuscated_code: Code with obfuscated variable names
        mapping: Dictionary mapping original names to obfuscated names
    """
    identifiers = extract_python_identifiers(code)

    # Extract unique identifier names (preserve order of first appearance)
    unique_names = []
    seen = set()
    for item in identifiers:
        name = item['name']
        if name not in seen:
            unique_names.append(name)
            seen.add(name)

    # Create obfuscation mapping: original_name -> obfuscated_name
    # Use single letters first, then double letters (a, b, ..., z, aa, ab, ...)
    obfuscation_map = {}
    for idx, name in enumerate(unique_names):
        if idx < 26:
            obfuscated = chr(ord('a') + idx)
        else:
            first = (idx - 26) // 26
            second = (idx - 26) % 26
            obfuscated = chr(ord('a') + first) + chr(ord('a') + second)
        obfuscation_map[name] = obfuscated

    # Sort identifiers by position (reverse order to avoid offset issues)
    sorted_identifiers = sorted(identifiers, key=lambda x: (x['line'], x['col']), reverse=True)

    # Replace identifiers in code
    lines = code.split('\n')
    for item in sorted_identifiers:
        line_idx = item['line'] - 1
        col_idx = item['col']
        original_name = item['name']
        obfuscated_name = obfuscation_map[original_name]

        line = lines[line_idx]
        # Replace the specific occurrence at this position
        before = line[:col_idx]
        after = line[col_idx + len(original_name):]
        lines[line_idx] = before + obfuscated_name + after

    obfuscated_code = '\n'.join(lines)
    return obfuscated_code, obfuscation_map


def get_token_mask_from_identifiers(code: str, tokenizer, input_ids: torch.Tensor) -> torch.Tensor:
    """
    Maps character-level identifiers to token-level mask.
    Returns a boolean tensor indicating which tokens correspond to identifiers.
    """
    identifiers = extract_python_identifiers(code)

    # Calculate byte offsets for each identifier
    lines = code.split('\n')
    ident_ranges = []
    for item in identifiers:
        line_idx = item['line'] - 1
        col_idx = item['col']
        start_offset = sum(len(l) + 1 for l in lines[:line_idx]) + col_idx
        end_offset = start_offset + len(item['name'])
        ident_ranges.append((start_offset, end_offset))

    # Tokenize and create mask
    tokens_decoded = [tokenizer.decode([tid]) for tid in input_ids]
    is_ident_token = torch.zeros(len(input_ids), dtype=torch.bool)

    current_char_pos = 0

    for i, tok_text in enumerate(tokens_decoded):
        if not tok_text or tok_text in ["<s>", "</s>", "<unk>", "<pad>"]:
            continue

        clean_tok = tok_text.replace('▁', ' ')
        search_text = clean_tok.strip()

        if not search_text:
            continue

        start_find = code.find(search_text, current_char_pos)
        if start_find != -1:
            tok_start = start_find
            tok_end = start_find + len(search_text)
            current_char_pos = tok_end

            # Check overlap with any identifier range
            for i_start, i_end in ident_ranges:
                if max(tok_start, i_start) < min(tok_end, i_end):
                    # Only mask if token is pure identifier (no syntax symbols)
                    pure_content = clean_tok.lstrip(' ')
                    if not re.search(r"[^a-zA-Z0-9_]", pure_content):
                        is_ident_token[i] = True
                    break

    return is_ident_token


def calculate_exact_match(original_map: Dict[str, str],
                         original_code: str,
                         deobfuscated_code: str) -> Dict:
    """
    Calculate exact match accuracy between original and deobfuscated variable names.

    Returns:
        Dictionary with evaluation metrics
    """
    # Extract identifiers from both codes
    original_identifiers = extract_python_identifiers(original_code)
    deobfuscated_identifiers = extract_python_identifiers(deobfuscated_code)

    # Create position-based mapping
    original_names_by_pos = {}
    for item in original_identifiers:
        key = (item['line'], item['col'])
        original_names_by_pos[key] = item['name']

    deobfuscated_names_by_pos = {}
    for item in deobfuscated_identifiers:
        key = (item['line'], item['col'])
        deobfuscated_names_by_pos[key] = item['name']

    # Calculate metrics
    total_positions = len(original_names_by_pos)
    exact_matches = 0
    partial_matches = 0

    for pos, original_name in original_names_by_pos.items():
        if pos in deobfuscated_names_by_pos:
            deobfuscated_name = deobfuscated_names_by_pos[pos]
            if original_name == deobfuscated_name:
                exact_matches += 1
            elif original_name.lower() == deobfuscated_name.lower():
                partial_matches += 1

    return {
        'total_positions': total_positions,
        'exact_matches': exact_matches,
        'partial_matches': partial_matches,
        'exact_match_rate': exact_matches / total_positions if total_positions > 0 else 0,
        'partial_match_rate': (exact_matches + partial_matches) / total_positions if total_positions > 0 else 0
    }


def run_deobfuscation_experiment(model_name: str,
                                model_id: str,
                                code_snippet: str,
                                total_steps: int = 20,
                                device: str = "cuda"):
    """
    Run deobfuscation experiment on a single code snippet.
    """
    print(f"\n{'='*60}")
    print(f"Deobfuscation Experiment: {model_name}")
    print(f"{'='*60}")

    # Create output directory
    os.makedirs("results/deobfuscation", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"results/deobfuscation/{model_name}_{timestamp}.txt"

    # Load model
    print(f"Loading model: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    ).to(device).eval()

    # Step 1: Obfuscate the code
    print("\n[Step 1] Obfuscating code...")
    obfuscated_code, obfuscation_map = obfuscate_code(code_snippet)
    print(f"Obfuscation mapping: {obfuscation_map}")

    # Step 2: Tokenize obfuscated code
    print("\n[Step 2] Tokenizing obfuscated code...")
    input_ids_full = tokenizer.encode(obfuscated_code, return_tensors="pt").to(device)
    input_ids = input_ids_full[0]

    # Step 3: Extract variable positions and create mask
    print("\n[Step 3] Extracting variable positions via AST...")
    ident_mask = get_token_mask_from_identifiers(obfuscated_code, tokenizer, input_ids).to(device)
    num_masked_tokens = ident_mask.sum().item()
    print(f"Total tokens: {len(input_ids)}, Masked tokens: {num_masked_tokens}")

    # Step 4: Run constrained diffusion
    print(f"\n[Step 4] Running constrained diffusion ({total_steps} steps)...")
    x = input_ids_full.clone()

    with open(output_filename, "w", encoding="utf-8") as f:
        # Write header
        f.write(f"{'='*60}\n")
        f.write(f"Deobfuscation Experiment\n")
        f.write(f"{'='*60}\n\n")
        f.write(f"Model: {model_name}\n")
        f.write(f"Model ID: {model_id}\n")
        f.write(f"Diffusion Steps: {total_steps}\n")
        f.write(f"Device: {device}\n\n")

        # Write original code
        f.write(f"{'='*60}\n")
        f.write(f"ORIGINAL CODE (Ground Truth)\n")
        f.write(f"{'='*60}\n")
        f.write(code_snippet + "\n\n")

        # Write obfuscation mapping
        f.write(f"{'='*60}\n")
        f.write(f"OBFUSCATION MAPPING\n")
        f.write(f"{'='*60}\n")
        f.write(json.dumps(obfuscation_map, indent=2) + "\n\n")

        # Write obfuscated code
        f.write(f"{'='*60}\n")
        f.write(f"OBFUSCATED CODE (Input)\n")
        f.write(f"{'='*60}\n")
        f.write(obfuscated_code + "\n\n")

        # Write initial state
        f.write(f"{'='*60}\n")
        f.write(f"DIFFUSION PROCESS\n")
        f.write(f"{'='*60}\n\n")
        f.write(f"--- Step 0 (Initial) ---\n")
        f.write(tokenizer.decode(x[0], skip_special_tokens=False) + "\n\n")

        # Diffusion loop
        for step in range(total_steps):
            with torch.no_grad():
                logits = model(x).logits
                x_pred = torch.argmax(logits, dim=-1)

                # Apply updates ONLY to identifier positions
                x_next = torch.where(ident_mask, x_pred, x)

                # Check convergence
                if torch.equal(x, x_next):
                    f.write(f"--- Step {step+1} ---\n")
                    f.write("Converged (no changes).\n\n")
                    print(f"Converged at step {step+1}")
                    break

                x = x_next
                decoded = tokenizer.decode(x[0], skip_special_tokens=False)

                f.write(f"--- Step {step+1} ---\n")
                f.write(decoded + "\n\n")

                if (step + 1) % 5 == 0:
                    print(f"Step {step+1}/{total_steps} completed")

        # Final deobfuscated code
        final_code = tokenizer.decode(x[0], skip_special_tokens=True)
        f.write(f"{'='*60}\n")
        f.write(f"FINAL DEOBFUSCATED CODE\n")
        f.write(f"{'='*60}\n")
        f.write(final_code + "\n\n")

        # Evaluation
        print("\n[Step 5] Evaluating deobfuscation quality...")
        metrics = calculate_exact_match(obfuscation_map, code_snippet, final_code)

        f.write(f"{'='*60}\n")
        f.write(f"EVALUATION METRICS\n")
        f.write(f"{'='*60}\n")
        f.write(f"Total variable positions: {metrics['total_positions']}\n")
        f.write(f"Exact matches: {metrics['exact_matches']}\n")
        f.write(f"Partial matches (case-insensitive): {metrics['partial_matches']}\n")
        f.write(f"Exact match rate: {metrics['exact_match_rate']:.2%}\n")
        f.write(f"Partial match rate: {metrics['partial_match_rate']:.2%}\n\n")

        print(f"\nEvaluation Results:")
        print(f"  Exact match rate: {metrics['exact_match_rate']:.2%}")
        print(f"  Partial match rate: {metrics['partial_match_rate']:.2%}")

    print(f"\n{'='*60}")
    print(f"Experiment completed. Results saved to: {output_filename}")
    print(f"{'='*60}\n")

    return metrics


def run_batch_deobfuscation_experiment(model_name: str,
                                      model_id: str,
                                      code_samples: List[Dict],
                                      total_steps: int = 20,
                                      device: str = "cuda"):
    """
    Run deobfuscation experiment on multiple code samples.

    Args:
        code_samples: List of dicts with 'name' and 'code' keys
    """
    print(f"\n{'='*80}")
    print(f"BATCH DEOBFUSCATION EXPERIMENT")
    print(f"Model: {model_name}")
    print(f"Total samples: {len(code_samples)}")
    print(f"{'='*80}\n")

    # Create output directory
    os.makedirs("results/deobfuscation", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_filename = f"results/deobfuscation/{model_name}_batch_summary_{timestamp}.json"

    all_metrics = []

    for idx, sample in enumerate(code_samples):
        print(f"\n[Sample {idx+1}/{len(code_samples)}] {sample['name']}")
        print("-" * 80)

        try:
            metrics = run_deobfuscation_experiment(
                model_name=f"{model_name}_sample{idx+1}",
                model_id=model_id,
                code_snippet=sample['code'],
                total_steps=total_steps,
                device=device
            )
            metrics['sample_name'] = sample['name']
            metrics['sample_index'] = idx
            all_metrics.append(metrics)
        except Exception as e:
            print(f"Error processing sample {idx+1}: {e}")
            all_metrics.append({
                'sample_name': sample['name'],
                'sample_index': idx,
                'error': str(e)
            })

    # Calculate aggregate statistics
    successful_metrics = [m for m in all_metrics if 'error' not in m]
    if successful_metrics:
        avg_exact_match = sum(m['exact_match_rate'] for m in successful_metrics) / len(successful_metrics)
        avg_partial_match = sum(m['partial_match_rate'] for m in successful_metrics) / len(successful_metrics)
    else:
        avg_exact_match = 0
        avg_partial_match = 0

    summary = {
        'model_name': model_name,
        'model_id': model_id,
        'total_samples': len(code_samples),
        'successful_samples': len(successful_metrics),
        'failed_samples': len(code_samples) - len(successful_metrics),
        'average_exact_match_rate': avg_exact_match,
        'average_partial_match_rate': avg_partial_match,
        'individual_results': all_metrics,
        'timestamp': timestamp
    }

    # Save summary
    with open(summary_filename, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*80}")
    print(f"BATCH EXPERIMENT SUMMARY")
    print(f"{'='*80}")
    print(f"Total samples: {summary['total_samples']}")
    print(f"Successful: {summary['successful_samples']}")
    print(f"Failed: {summary['failed_samples']}")
    print(f"Average exact match rate: {summary['average_exact_match_rate']:.2%}")
    print(f"Average partial match rate: {summary['average_partial_match_rate']:.2%}")
    print(f"\nSummary saved to: {summary_filename}")
    print(f"{'='*80}\n")

    return summary


if __name__ == "__main__":
    # Test samples with meaningful variable names
    test_samples = [
        {
            'name': 'bubble_sort',
            'code': """def bubble_sort(array):
    length = len(array)
    for i in range(length):
        for j in range(0, length - i - 1):
            if array[j] > array[j + 1]:
                temp = array[j]
                array[j] = array[j + 1]
                array[j + 1] = temp
    return array"""
        },
        {
            'name': 'binary_search',
            'code': """def binary_search(sorted_list, target):
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
        },
        {
            'name': 'fibonacci',
            'code': """def fibonacci(n):
    if n <= 1:
        return n
    previous = 0
    current = 1
    for i in range(2, n + 1):
        next_value = previous + current
        previous = current
        current = next_value
    return current"""
        },
        {
            'name': 'calculate_average',
            'code': """def calculate_average(numbers):
    if not numbers:
        return 0
    total = sum(numbers)
    count = len(numbers)
    average = total / count
    return average"""
        },
        {
            'name': 'find_maximum',
            'code': """def find_maximum(values):
    if not values:
        return None
    maximum = values[0]
    for value in values:
        if value > maximum:
            maximum = value
    return maximum"""
        }
    ]

    # Run single experiment
    print("Running single deobfuscation experiment...")
    run_deobfuscation_experiment(
        model_name="dreamcoder",
        model_id="Dream-org/Dream-Coder-v0-Instruct-7B",
        code_snippet=test_samples[0]['code'],
        total_steps=20,
        device="cuda"
    )

    # Run batch experiment (uncomment to run on all samples)
    # print("\n\nRunning batch deobfuscation experiment...")
    # run_batch_deobfuscation_experiment(
    #     model_name="dreamcoder",
    #     model_id="Dream-org/Dream-Coder-v0-Instruct-7B",
    #     code_samples=test_samples,
    #     total_steps=20,
    #     device="cuda"
    # )
