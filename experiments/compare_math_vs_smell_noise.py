"""
Experiment: What Role Does Code Smell Play as "Noise" in Diffusion Language Models?
====================================================================================

Central Question:
    In traditional diffusion models, noise has a clear mathematical definition:
    at timestep t, the input x_t = √(ᾱ_t) * x_0 + √(1-ᾱ_t) * ε.
    
    If code smell (e.g., bad variable naming) acts as "noise" in diffusion LMs,
    WHERE on the noise schedule does it correspond to? Is it equivalent to
    timestep 50? 100? 200 out of 256?

Experimental Design:
    Phase 1 - Noise Spectrum Calibration:
        Run full diffusion from 100% masked code → clean code.
        At each step t, record the model's metrics at identifier positions:
        - Confidence (P of argmax token)
        - Entropy (H of distribution)
        - Current-token probability (P of what's actually there)
        This builds a "noise spectrum" / calibration curve.

    Phase 2 - Code Smell Probing (SINGLE forward pass, no diffusion):
        Feed the model code with bad identifiers. In ONE forward pass, measure:
        - P(current_bad_token): How likely does the model think `xxx` belongs here?
        - Rank(current_bad_token): Rank of `xxx` in the softmax distribution
        - P(ground_truth_token): How likely does the model think the correct name is?
        - Rank(ground_truth_token): Rank of the correct name
        - Entropy: How uncertain is the model at this position?
        - Argmax prediction: What does the model WANT to put here?
        
    Phase 3 - Clean Code Probing (SINGLE forward pass, control):
        Same as Phase 2, but with clean, correctly-named code.
        This is the baseline: measurements at "noise level = 0".

    Phase 4 - Equivalent Noise Level Mapping:
        For each smell sample, find which step t* in Phase 1's trajectory
        has the closest entropy/confidence to the smell probe.
        t* is the "equivalent noise level" of code smell.

Metrics:
    1. current_token_prob:    P(x_current | context) — model's "satisfaction"
    2. current_token_rank:    Rank of x_current in sorted softmax
    3. gt_token_prob:         P(x_gt | context) — how much model wants GT
    4. gt_token_rank:         Rank of x_gt in sorted softmax  
    5. entropy:               Shannon entropy at target positions
    6. argmax_prediction:     What the model would predict (no noise)
    7. top5_predictions:      The model's top-5 token predictions
    8. kl_divergence:         KL(P_smell || P_clean) — distribution shift

Output:
    - noise_calibration_*.csv:   Per-step metrics during diffusion (Phase 1)
    - smell_probe_*.csv:         Smell probing results (Phase 2)
    - clean_probe_*.csv:         Clean probing results (Phase 3)
    - noise_mapping_*.csv:       Equivalent noise level mapping (Phase 4)
    - summary_*.csv:             Aggregate statistics
"""

import sys
import os
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import csv
import re
import json
import random
from datetime import datetime
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

# --- Mock torchvision for DreamCoder/DiffuCoder compatibility ---
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
# -----------------------------------------------------------------

# ======================== Configuration ==========================
DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'test_filtered_1024.csv')
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results')

MODELS = {
    "diffucoder": {
        "id": "apple/DiffuCoder-7B-Instruct",
        "mask_token": "<|mask|>",
    },
    "dreamcoder": {
        "id": "Dream-org/Dream-Coder-v0-Instruct-7B",
        "mask_token": "<|mask|>",
    },
}

# Bad identifier names representing different severities of code smell
BAD_NAMES_BY_SEVERITY = {
    "severe": ["x", "a", "xx", "a1"],                          # Single-char, meaningless
    "moderate": ["tmp", "val", "foo", "bar", "data"],           # Common placeholder names
    "mild": ["myVar", "temp1", "result1", "value1", "item"],    # At least structured, but vague
}
ALL_BAD_NAMES = [n for names in BAD_NAMES_BY_SEVERITY.values() for n in names]

# Experiment parameters
TOTAL_STEPS = 32
LIMIT = 50
REPEATS = 10
MAX_TOKENS = 1024
TEMPERATURE = 0.3
# =================================================================


def get_num_transfer_tokens(mask_index, steps):
    """Compute the schedule of how many tokens to unmask at each step."""
    mask_num = mask_index.sum(dim=1, keepdim=True)
    base = mask_num // steps
    remainder = mask_num % steps
    num_transfer_tokens = torch.zeros(
        mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64
    ) + base
    for i in range(mask_num.size(0)):
        num_transfer_tokens[i, :remainder[i]] += 1
    return num_transfer_tokens


def add_gumbel_noise(logits, temperature):
    """Add Gumbel noise for sampling."""
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (- torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def find_subsequence_indices(sequence, subsequence):
    """Finds the first occurrence of subsequence in sequence."""
    seq_len = len(sequence)
    sub_len = len(subsequence)
    for i in range(seq_len - sub_len + 1):
        if sequence[i: i + sub_len] == subsequence:
            return i, i + sub_len
    return None


def probe_single_pass(model, tokenizer, input_ids, attention_mask, target_indices,
                      current_tokens, gt_tokens):
    """
    Single forward pass probing: measure the model's opinion about tokens
    at target positions WITHOUT running diffusion.
    
    This is the key metric: "How satisfied is the model with what's currently here?"
    
    Returns a dict with:
        - current_token_prob: avg P(current_token) at target positions
        - current_token_rank: avg rank of current_token in softmax
        - gt_token_prob: avg P(gt_token) at target positions
        - gt_token_rank: avg rank of gt_token in softmax
        - entropy: avg Shannon entropy at target positions
        - argmax_decoded: what the model would predict
        - top5: top-5 predictions with probabilities
        - logit_vector: raw logit values at target positions (for KL computation)
    """
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask.bool())
        logits = outputs.logits  # (1, seq_len, vocab_size)

    target_logits = logits[0, target_indices, :]  # (n_target, vocab_size)
    probs = F.softmax(target_logits.float(), dim=-1)

    # Shannon entropy
    log_probs_all = torch.log(probs + 1e-10)
    entropy = -(probs * log_probs_all).sum(dim=-1)
    avg_entropy = entropy.mean().item()

    # Argmax prediction
    argmax_tokens = torch.argmax(target_logits, dim=-1).tolist()
    argmax_decoded = tokenizer.decode(argmax_tokens).strip()
    argmax_decoded_clean = re.sub(r'[^A-Za-z0-9_]', '', argmax_decoded)

    # Confidence of argmax
    argmax_probs = probs[range(len(target_indices)), argmax_tokens]
    avg_argmax_conf = argmax_probs.mean().item()

    # --- Current token metrics ---
    current_token_probs = []
    current_token_ranks = []
    for i, idx in enumerate(target_indices):
        token_id = current_tokens[i]
        p = probs[i, token_id].item()
        current_token_probs.append(p)
        # Rank: how many tokens have higher probability?
        rank = (probs[i] > probs[i, token_id]).sum().item() + 1
        current_token_ranks.append(rank)

    # --- Ground truth token metrics ---
    gt_token_probs = []
    gt_token_ranks = []
    for i, idx in enumerate(target_indices):
        if i < len(gt_tokens):
            token_id = gt_tokens[i]
            p = probs[i, token_id].item()
            gt_token_probs.append(p)
            rank = (probs[i] > probs[i, token_id]).sum().item() + 1
            gt_token_ranks.append(rank)

    # Top-5 predictions
    top5_vals, top5_ids = torch.topk(probs, k=5, dim=-1)
    # Average across target positions, take the first position's top5 for display
    top5_tokens = [tokenizer.decode([tid.item()]).strip() for tid in top5_ids[0]]
    top5_probs_list = top5_vals[0].tolist()

    return {
        'current_token_prob': np.mean(current_token_probs) if current_token_probs else 0,
        'current_token_rank': np.mean(current_token_ranks) if current_token_ranks else 0,
        'gt_token_prob': np.mean(gt_token_probs) if gt_token_probs else 0,
        'gt_token_rank': np.mean(gt_token_ranks) if gt_token_ranks else 0,
        'entropy': avg_entropy,
        'argmax_confidence': avg_argmax_conf,
        'argmax_decoded': argmax_decoded_clean,
        'top5': json.dumps(list(zip(top5_tokens, top5_probs_list))),
        'logits_at_target': target_logits.cpu(),  # For KL divergence
    }


def run_diffusion_calibration(model, tokenizer, input_ids, attention_mask,
                              target_indices, gt_tokens, mask_token_id,
                              total_steps=256, temperature=0.3):
    """
    Phase 1: Run full diffusion from masked input, recording per-step metrics
    at the target identifier positions.
    
    This builds the "noise spectrum calibration curve":
    step 0 (full mask) → step T (denoised) with metrics at each point.
    
    Returns:
        step_metrics: list of dicts, one per step
    """
    x = input_ids.clone()
    step_metrics = []

    initial_mask_index = (x == mask_token_id)
    num_transfer_tokens = get_num_transfer_tokens(initial_mask_index, total_steps)

    for step_i in range(total_steps):
        with torch.no_grad():
            current_mask_index = (x == mask_token_id)
            outputs = model(x, attention_mask=attention_mask.bool())
            logits = outputs.logits

            target_logits = logits[0, target_indices, :]
            probs = F.softmax(target_logits.float(), dim=-1)

            # Entropy
            log_p = torch.log(probs + 1e-10)
            entropy = -(probs * log_p).sum(dim=-1)
            avg_entropy = entropy.mean().item()

            # Argmax confidence
            argmax_tokens = torch.argmax(target_logits, dim=-1)
            argmax_probs = probs[range(len(target_indices)), argmax_tokens.tolist()]
            avg_argmax_conf = argmax_probs.mean().item()

            # GT token probability at this step
            gt_probs = []
            gt_ranks = []
            for i in range(len(target_indices)):
                if i < len(gt_tokens):
                    p = probs[i, gt_tokens[i]].item()
                    gt_probs.append(p)
                    rank = (probs[i] > probs[i, gt_tokens[i]]).sum().item() + 1
                    gt_ranks.append(rank)

            # Current token at target positions
            current_tokens = x[0, target_indices].tolist()
            is_mask = all(t == mask_token_id for t in current_tokens)
            matches_gt = (current_tokens == gt_tokens)

            # Decoded current identifier
            decoded = tokenizer.decode(current_tokens).strip()
            decoded_clean = re.sub(r'[^A-Za-z0-9_]', '', decoded)

            step_metrics.append({
                'step': step_i,
                'entropy': avg_entropy,
                'argmax_confidence': avg_argmax_conf,
                'gt_token_prob': np.mean(gt_probs) if gt_probs else 0,
                'gt_token_rank': np.mean(gt_ranks) if gt_ranks else 0,
                'is_mask': is_mask,
                'matches_gt': matches_gt,
                'decoded': decoded_clean,
            })

            # --- Diffusion update ---
            if current_mask_index.any():
                logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
                x0 = torch.argmax(logits_with_noise, dim=-1)
                p_all = F.softmax(logits.float(), dim=-1)
                x0_p = torch.squeeze(
                    torch.gather(p_all, dim=-1, index=torch.unsqueeze(x0, -1)), -1
                )
                x0 = torch.where(current_mask_index, x0, x)
                confidence = torch.where(current_mask_index, x0_p,
                                          torch.tensor(-np.inf, device=x.device))
                transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
                if num_transfer_tokens.shape[1] > step_i:
                    k = num_transfer_tokens[0, step_i].item()
                else:
                    k = 0
                if k > 0:
                    _, sel = torch.topk(confidence[0], k=int(k))
                    transfer_index[0, sel] = True
                    x[transfer_index] = x0[transfer_index]

    return step_metrics


def find_equivalent_noise_level(calibration_curve, probe_metrics):
    """
    Phase 4: Map a probe measurement to its equivalent noise level
    on the calibration curve.
    
    Strategy: Find the step t* where the calibration curve's entropy
    is closest to the probe's entropy. Also compute for gt_token_prob.
    
    Returns:
        t_entropy: equivalent step based on entropy matching
        t_gt_prob: equivalent step based on GT probability matching
        t_confidence: equivalent step based on argmax confidence matching
    """
    probe_entropy = probe_metrics['entropy']
    probe_gt_prob = probe_metrics['gt_token_prob']
    probe_conf = probe_metrics['argmax_confidence']

    # Find closest step for each metric
    min_ent_diff = float('inf')
    t_entropy = 0
    min_gt_diff = float('inf')
    t_gt_prob = 0
    min_conf_diff = float('inf')
    t_confidence = 0

    for step_data in calibration_curve:
        # Entropy matching
        diff = abs(step_data['entropy'] - probe_entropy)
        if diff < min_ent_diff:
            min_ent_diff = diff
            t_entropy = step_data['step']

        # GT probability matching
        diff = abs(step_data['gt_token_prob'] - probe_gt_prob)
        if diff < min_gt_diff:
            min_gt_diff = diff
            t_gt_prob = step_data['step']

        # Confidence matching
        diff = abs(step_data['argmax_confidence'] - probe_conf)
        if diff < min_conf_diff:
            min_conf_diff = diff
            t_confidence = step_data['step']

    return t_entropy, t_gt_prob, t_confidence


def run_single_sample(tokenizer, model, mask_token_id, original_code,
                      ground_truth_name, sample_id, run_id, total_steps=256):
    """
    Run all phases for a single data sample.
    
    Returns:
        calibration_data: list of per-step dicts (Phase 1)
        probe_results: list of probe result dicts (Phase 2 & 3)
        mapping_results: list of noise mapping dicts (Phase 4)
    """
    probe_results = []
    mapping_results = []

    # ---- Prepare base tokenizations ----
    clean_code = original_code.replace("[MASK]", ground_truth_name)
    inputs_clean = tokenizer(clean_code, return_tensors="pt").to("cuda")
    clean_ids = inputs_clean.input_ids[0].tolist()

    # Locate ground truth tokens
    gt_token_ids = tokenizer.encode(ground_truth_name, add_special_tokens=False)
    target_result = find_subsequence_indices(clean_ids, gt_token_ids)
    if target_result is None:
        gt_token_ids_space = tokenizer.encode(" " + ground_truth_name, add_special_tokens=False)
        target_result = find_subsequence_indices(clean_ids, gt_token_ids_space)
        if target_result is not None:
            gt_token_ids = gt_token_ids_space
    if target_result is None:
        return [], [], []

    target_start, target_end = target_result
    target_indices = list(range(target_start, target_end))
    gt_tokens = clean_ids[target_start:target_end]

    # ============================================================
    # Phase 1: Noise Spectrum Calibration (full diffusion from mask)
    # ============================================================
    masked_ids = inputs_clean.input_ids.clone()
    for idx in target_indices:
        masked_ids[0, idx] = mask_token_id

    calibration_data = run_diffusion_calibration(
        model, tokenizer, masked_ids, inputs_clean.attention_mask,
        target_indices, gt_tokens, mask_token_id,
        total_steps=total_steps, temperature=TEMPERATURE,
    )

    # ============================================================
    # Phase 2: Code Smell Probing (single forward pass)
    # ============================================================
    for severity, bad_names in BAD_NAMES_BY_SEVERITY.items():
        bad_name = random.choice(bad_names)
        smell_code = original_code.replace("[MASK]", bad_name)
        inputs_smell = tokenizer(smell_code, return_tensors="pt").to("cuda")
        smell_ids = inputs_smell.input_ids[0].tolist()

        # Find bad name tokens in the sequence
        bad_toks = tokenizer.encode(bad_name, add_special_tokens=False)
        smell_target = find_subsequence_indices(smell_ids, bad_toks)
        if smell_target is None:
            bad_toks_space = tokenizer.encode(" " + bad_name, add_special_tokens=False)
            smell_target = find_subsequence_indices(smell_ids, bad_toks_space)
            if smell_target is not None:
                bad_toks = bad_toks_space
        if smell_target is None:
            continue

        s_start, s_end = smell_target
        smell_indices = list(range(s_start, s_end))
        current_bad_tokens = smell_ids[s_start:s_end]

        # We need GT tokens at the same conceptual position.
        # Since tokenization may differ, we find GT tokens for comparison.
        # The GT tokens are what SHOULD be there.
        # Note: the number of tokens may differ between bad_name and gt_name.
        # We compare the first min(len_bad, len_gt) tokens from each.
        gt_for_probe = gt_tokens[:len(smell_indices)] if len(gt_tokens) >= len(smell_indices) else gt_tokens

        probe = probe_single_pass(
            model, tokenizer, inputs_smell.input_ids, inputs_smell.attention_mask,
            smell_indices, current_bad_tokens, gt_for_probe
        )

        # Phase 4: Map to equivalent noise level
        t_ent, t_gt, t_conf = find_equivalent_noise_level(calibration_data, probe)

        probe_results.append({
            'sample_id': sample_id,
            'run_id': run_id,
            'group': 'smell',
            'severity': severity,
            'bad_name': bad_name,
            'ground_truth': ground_truth_name,
            'current_token_prob': probe['current_token_prob'],
            'current_token_rank': probe['current_token_rank'],
            'gt_token_prob': probe['gt_token_prob'],
            'gt_token_rank': probe['gt_token_rank'],
            'entropy': probe['entropy'],
            'argmax_confidence': probe['argmax_confidence'],
            'argmax_decoded': probe['argmax_decoded'],
            'top5': probe['top5'],
        })

        mapping_results.append({
            'sample_id': sample_id,
            'run_id': run_id,
            'group': 'smell',
            'severity': severity,
            'bad_name': bad_name,
            'ground_truth': ground_truth_name,
            'equiv_step_entropy': t_ent,
            'equiv_step_gt_prob': t_gt,
            'equiv_step_confidence': t_conf,
            'probe_entropy': probe['entropy'],
            'probe_gt_prob': probe['gt_token_prob'],
            'probe_confidence': probe['argmax_confidence'],
        })

    # ============================================================
    # Phase 3: Clean Code Probing (single forward pass, control)
    # ============================================================
    probe_clean = probe_single_pass(
        model, tokenizer, inputs_clean.input_ids, inputs_clean.attention_mask,
        target_indices, gt_tokens, gt_tokens
    )

    t_ent_c, t_gt_c, t_conf_c = find_equivalent_noise_level(calibration_data, probe_clean)

    probe_results.append({
        'sample_id': sample_id,
        'run_id': run_id,
        'group': 'control',
        'severity': 'none',
        'bad_name': '',
        'ground_truth': ground_truth_name,
        'current_token_prob': probe_clean['current_token_prob'],
        'current_token_rank': probe_clean['current_token_rank'],
        'gt_token_prob': probe_clean['gt_token_prob'],
        'gt_token_rank': probe_clean['gt_token_rank'],
        'entropy': probe_clean['entropy'],
        'argmax_confidence': probe_clean['argmax_confidence'],
        'argmax_decoded': probe_clean['argmax_decoded'],
        'top5': probe_clean['top5'],
    })

    mapping_results.append({
        'sample_id': sample_id,
        'run_id': run_id,
        'group': 'control',
        'severity': 'none',
        'bad_name': '',
        'ground_truth': ground_truth_name,
        'equiv_step_entropy': t_ent_c,
        'equiv_step_gt_prob': t_gt_c,
        'equiv_step_confidence': t_conf_c,
        'probe_entropy': probe_clean['entropy'],
        'probe_gt_prob': probe_clean['gt_token_prob'],
        'probe_confidence': probe_clean['argmax_confidence'],
    })

    # ============================================================
    # Phase 2b: Mask Probing (for reference: what does step 0 look like?)
    # ============================================================
    probe_mask = probe_single_pass(
        model, tokenizer, masked_ids, inputs_clean.attention_mask,
        target_indices, [mask_token_id] * len(target_indices), gt_tokens
    )

    probe_results.append({
        'sample_id': sample_id,
        'run_id': run_id,
        'group': 'mask',
        'severity': 'full',
        'bad_name': '',
        'ground_truth': ground_truth_name,
        'current_token_prob': probe_mask['current_token_prob'],
        'current_token_rank': probe_mask['current_token_rank'],
        'gt_token_prob': probe_mask['gt_token_prob'],
        'gt_token_rank': probe_mask['gt_token_rank'],
        'entropy': probe_mask['entropy'],
        'argmax_confidence': probe_mask['argmax_confidence'],
        'argmax_decoded': probe_mask['argmax_decoded'],
        'top5': probe_mask['top5'],
    })

    return calibration_data, probe_results, mapping_results


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Investigate code smell's role as noise in diffusion LMs"
    )
    parser.add_argument("--model", type=str, default="diffucoder",
                        choices=list(MODELS.keys()))
    parser.add_argument("--limit", type=int, default=LIMIT)
    parser.add_argument("--repeats", type=int, default=REPEATS)
    parser.add_argument("--steps", type=int, default=TOTAL_STEPS)
    args = parser.parse_args()

    total_steps = args.steps
    limit = args.limit
    repeats = args.repeats

    os.makedirs(RESULTS_DIR, exist_ok=True)

    # ---- Load Model ----
    model_cfg = MODELS[args.model]
    model_id = model_cfg["id"]
    print(f"Loading model: {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, trust_remote_code=True
    ).to("cuda").eval()
    mask_token_id = tokenizer.convert_tokens_to_ids(model_cfg["mask_token"])
    print(f"Model loaded. Mask token ID: {mask_token_id}")

    # ---- Load Data ----
    print(f"Loading data from {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH, header=None, names=['id', 'X', 'y'], nrows=limit)
    print(f"Loaded {len(df)} samples.")

    # ---- Setup Output ----
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    calibration_csv = os.path.join(RESULTS_DIR, f"noise_calibration_{args.model}_{timestamp}.csv")
    probe_csv = os.path.join(RESULTS_DIR, f"noise_probe_{args.model}_{timestamp}.csv")
    mapping_csv = os.path.join(RESULTS_DIR, f"noise_mapping_{args.model}_{timestamp}.csv")
    summary_csv = os.path.join(RESULTS_DIR, f"noise_summary_{args.model}_{timestamp}.csv")

    # Initialize CSVs
    calib_fields = ['sample_id', 'run_id', 'step', 'entropy', 'argmax_confidence',
                    'gt_token_prob', 'gt_token_rank', 'is_mask', 'matches_gt', 'decoded']
    probe_fields = ['sample_id', 'run_id', 'group', 'severity', 'bad_name', 'ground_truth',
                    'current_token_prob', 'current_token_rank',
                    'gt_token_prob', 'gt_token_rank',
                    'entropy', 'argmax_confidence', 'argmax_decoded', 'top5']
    mapping_fields = ['sample_id', 'run_id', 'group', 'severity', 'bad_name', 'ground_truth',
                      'equiv_step_entropy', 'equiv_step_gt_prob', 'equiv_step_confidence',
                      'probe_entropy', 'probe_gt_prob', 'probe_confidence']

    for path, fields in [(calibration_csv, calib_fields), 
                          (probe_csv, probe_fields),
                          (mapping_csv, mapping_fields)]:
        with open(path, 'w', newline='') as f:
            csv.DictWriter(f, fieldnames=fields).writeheader()

    print(f"\n{'='*70}")
    print(f" Experiment: Code Smell's Noise Role in Diffusion LMs")
    print(f" Model:      {args.model} ({model_id})")
    print(f" Samples:    {len(df)}, Repeats: {repeats}, Steps: {total_steps}")
    print(f" Outputs:")
    print(f"   Calibration: {calibration_csv}")
    print(f"   Probing:     {probe_csv}")
    print(f"   Mapping:     {mapping_csv}")
    print(f"{'='*70}\n")

    # ---- Run Experiment ----
    all_probe_results = []
    all_mapping_results = []

    for run_id in tqdm(range(repeats), desc="Runs"):
        for _, row in tqdm(df.iterrows(), total=len(df),
                           desc=f"Run {run_id+1}/{repeats}", leave=False):
            sample_id = row['id']
            X = str(row['X'])
            y = str(row['y']).strip()

            if '[MASK]' not in X:
                continue

            # Token length check
            tokens = tokenizer.encode(X.replace('[MASK]', y), add_special_tokens=False)
            if len(tokens) > MAX_TOKENS:
                continue

            try:
                calib_data, probe_data, mapping_data = run_single_sample(
                    tokenizer, model, mask_token_id,
                    X, y, sample_id, run_id + 1,
                    total_steps=total_steps
                )

                # Write calibration data
                if calib_data:
                    with open(calibration_csv, 'a', newline='') as f:
                        writer = csv.DictWriter(f, fieldnames=calib_fields)
                        for row_data in calib_data:
                            row_data['sample_id'] = sample_id
                            row_data['run_id'] = run_id + 1
                            writer.writerow(row_data)

                # Write probe data
                if probe_data:
                    with open(probe_csv, 'a', newline='') as f:
                        writer = csv.DictWriter(f, fieldnames=probe_fields)
                        for row_data in probe_data:
                            writer.writerow(row_data)
                    all_probe_results.extend(probe_data)

                # Write mapping data
                if mapping_data:
                    with open(mapping_csv, 'a', newline='') as f:
                        writer = csv.DictWriter(f, fieldnames=mapping_fields)
                        for row_data in mapping_data:
                            writer.writerow(row_data)
                    all_mapping_results.extend(mapping_data)

            except Exception as e:
                print(f"\n  Error on sample {sample_id}: {e}")
                import traceback
                traceback.print_exc()
                continue

    # ============================================================
    # Generate Summary
    # ============================================================
    if all_probe_results and all_mapping_results:
        df_probe = pd.DataFrame(all_probe_results)
        df_mapping = pd.DataFrame(all_mapping_results)

        summary_lines = []
        summary_lines.append("=" * 70)
        summary_lines.append(" EXPERIMENT RESULTS: Code Smell's Noise Role")
        summary_lines.append("=" * 70)

        # --- Probe Summary ---
        summary_lines.append("\n" + "─" * 70)
        summary_lines.append(" PHASE 2/3: Single-Pass Probing Results")
        summary_lines.append("─" * 70)

        for group in ['mask', 'smell', 'control']:
            gdf = df_probe[df_probe['group'] == group]
            if gdf.empty:
                continue

            summary_lines.append(f"\n  ┌─ Group: {group.upper()}")
            if group == 'smell':
                for sev in ['severe', 'moderate', 'mild']:
                    sdf = gdf[gdf['severity'] == sev]
                    if sdf.empty:
                        continue
                    summary_lines.append(f"  │  Severity: {sev} (n={len(sdf)})")
                    summary_lines.append(f"  │    Current Token Prob:  {sdf['current_token_prob'].astype(float).mean():.6f}")
                    summary_lines.append(f"  │    Current Token Rank:  {sdf['current_token_rank'].astype(float).mean():.1f}")
                    summary_lines.append(f"  │    GT Token Prob:       {sdf['gt_token_prob'].astype(float).mean():.6f}")
                    summary_lines.append(f"  │    GT Token Rank:       {sdf['gt_token_rank'].astype(float).mean():.1f}")
                    summary_lines.append(f"  │    Entropy:             {sdf['entropy'].astype(float).mean():.4f}")
            else:
                summary_lines.append(f"  │  Samples: {len(gdf)}")
                summary_lines.append(f"  │  Current Token Prob:  {gdf['current_token_prob'].astype(float).mean():.6f}")
                summary_lines.append(f"  │  Current Token Rank:  {gdf['current_token_rank'].astype(float).mean():.1f}")
                summary_lines.append(f"  │  GT Token Prob:       {gdf['gt_token_prob'].astype(float).mean():.6f}")
                summary_lines.append(f"  │  GT Token Rank:       {gdf['gt_token_rank'].astype(float).mean():.1f}")
                summary_lines.append(f"  │  Entropy:             {gdf['entropy'].astype(float).mean():.4f}")
            summary_lines.append(f"  └─")

        # --- Noise Mapping Summary ---
        summary_lines.append("\n" + "─" * 70)
        summary_lines.append(" PHASE 4: Equivalent Noise Level Mapping")
        summary_lines.append("   (Which diffusion step does code smell correspond to?)")
        summary_lines.append("   (Step 0 = full mask/noise, Step 255 = fully denoised)")
        summary_lines.append("─" * 70)

        for group in ['smell', 'control']:
            gdf = df_mapping[df_mapping['group'] == group]
            if gdf.empty:
                continue
            summary_lines.append(f"\n  ┌─ Group: {group.upper()}")

            if group == 'smell':
                for sev in ['severe', 'moderate', 'mild']:
                    sdf = gdf[gdf['severity'] == sev]
                    if sdf.empty:
                        continue
                    summary_lines.append(f"  │  Severity: {sev} (n={len(sdf)})")
                    summary_lines.append(f"  │    Equiv Step (entropy):    {sdf['equiv_step_entropy'].astype(float).mean():.1f} / {total_steps}")
                    summary_lines.append(f"  │    Equiv Step (GT prob):    {sdf['equiv_step_gt_prob'].astype(float).mean():.1f} / {total_steps}")
                    summary_lines.append(f"  │    Equiv Step (confidence): {sdf['equiv_step_confidence'].astype(float).mean():.1f} / {total_steps}")
                    # Express as percentage of noise schedule
                    pct_ent = (1 - sdf['equiv_step_entropy'].astype(float).mean() / total_steps) * 100
                    pct_gt = (1 - sdf['equiv_step_gt_prob'].astype(float).mean() / total_steps) * 100
                    summary_lines.append(f"  │    ≈ Noise Level:           {pct_ent:.1f}% (entropy), {pct_gt:.1f}% (GT prob)")
            else:
                summary_lines.append(f"  │  Samples: {len(gdf)}")
                summary_lines.append(f"  │  Equiv Step (entropy):    {gdf['equiv_step_entropy'].astype(float).mean():.1f} / {total_steps}")
                summary_lines.append(f"  │  Equiv Step (GT prob):    {gdf['equiv_step_gt_prob'].astype(float).mean():.1f} / {total_steps}")
                summary_lines.append(f"  │  Equiv Step (confidence): {gdf['equiv_step_confidence'].astype(float).mean():.1f} / {total_steps}")
                pct_ent = (1 - gdf['equiv_step_entropy'].astype(float).mean() / total_steps) * 100
                pct_gt = (1 - gdf['equiv_step_gt_prob'].astype(float).mean() / total_steps) * 100
                summary_lines.append(f"  │  ≈ Noise Level:           {pct_ent:.1f}% (entropy), {pct_gt:.1f}% (GT prob)")
            summary_lines.append(f"  └─")

        # --- Key Finding ---
        smell_all = df_mapping[df_mapping['group'] == 'smell']
        ctrl_all = df_mapping[df_mapping['group'] == 'control']
        if not smell_all.empty and not ctrl_all.empty:
            smell_equiv = smell_all['equiv_step_entropy'].astype(float).mean()
            ctrl_equiv = ctrl_all['equiv_step_entropy'].astype(float).mean()
            summary_lines.append("\n" + "─" * 70)
            summary_lines.append(" KEY FINDING")
            summary_lines.append("─" * 70)
            summary_lines.append(f"  Code Smell maps to ~step {smell_equiv:.0f}/{total_steps} on the noise schedule")
            summary_lines.append(f"  Clean Code maps to ~step {ctrl_equiv:.0f}/{total_steps} on the noise schedule")
            summary_lines.append(f"  Noise level difference: {abs(smell_equiv - ctrl_equiv):.1f} steps")
            if smell_equiv < ctrl_equiv:
                noise_pct = (1 - smell_equiv / total_steps) * 100
                summary_lines.append(f"  → Code smell ≈ {noise_pct:.1f}% noise level in the diffusion process")
                summary_lines.append(f"  → The model sees bad naming as '{noise_pct:.0f}% corrupt',")
                summary_lines.append(f"    not fully random (100%) but also not clean (0%).")

        summary_lines.append("\n" + "=" * 70)
        summary_report = "\n".join(summary_lines)
        print(summary_report)

        # Save summary
        with open(summary_csv, 'w') as f:
            f.write(summary_report)
        print(f"\nSummary saved to: {summary_csv}")

        # Also save a compact numeric summary for plotting
        summary_data = []
        for group in ['mask', 'smell', 'control']:
            gdf = df_probe[df_probe['group'] == group]
            if gdf.empty:
                continue
            if group == 'smell':
                for sev in ['severe', 'moderate', 'mild']:
                    sdf = gdf[gdf['severity'] == sev]
                    if sdf.empty:
                        continue
                    mdf = df_mapping[(df_mapping['group'] == 'smell') & (df_mapping['severity'] == sev)]
                    summary_data.append({
                        'group': f'smell_{sev}',
                        'n': len(sdf),
                        'current_token_prob': sdf['current_token_prob'].astype(float).mean(),
                        'current_token_rank': sdf['current_token_rank'].astype(float).mean(),
                        'gt_token_prob': sdf['gt_token_prob'].astype(float).mean(),
                        'gt_token_rank': sdf['gt_token_rank'].astype(float).mean(),
                        'entropy': sdf['entropy'].astype(float).mean(),
                        'equiv_step_entropy': mdf['equiv_step_entropy'].astype(float).mean() if not mdf.empty else -1,
                        'equiv_step_gt_prob': mdf['equiv_step_gt_prob'].astype(float).mean() if not mdf.empty else -1,
                    })
            else:
                mdf = df_mapping[df_mapping['group'] == group] if group != 'mask' else pd.DataFrame()
                summary_data.append({
                    'group': group,
                    'n': len(gdf),
                    'current_token_prob': gdf['current_token_prob'].astype(float).mean(),
                    'current_token_rank': gdf['current_token_rank'].astype(float).mean(),
                    'gt_token_prob': gdf['gt_token_prob'].astype(float).mean(),
                    'gt_token_rank': gdf['gt_token_rank'].astype(float).mean(),
                    'entropy': gdf['entropy'].astype(float).mean(),
                    'equiv_step_entropy': mdf['equiv_step_entropy'].astype(float).mean() if not mdf.empty else -1,
                    'equiv_step_gt_prob': mdf['equiv_step_gt_prob'].astype(float).mean() if not mdf.empty else -1,
                })

        numeric_csv = os.path.join(RESULTS_DIR, f"noise_numeric_summary_{args.model}_{timestamp}.csv")
        pd.DataFrame(summary_data).to_csv(numeric_csv, index=False)
        print(f"Numeric summary saved to: {numeric_csv}")
    else:
        print("\nNo results generated.")

    # Cleanup
    del model
    del tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    import gc
    gc.collect()


if __name__ == "__main__":
    main()
