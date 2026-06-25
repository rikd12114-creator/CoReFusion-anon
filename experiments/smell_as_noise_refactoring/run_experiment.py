import os
import torch
import pandas as pd
import numpy as np
import random
import csv
import argparse
from tqdm import tqdm
from datetime import datetime
from utils import load_model, load_data, find_token_indices, BAD_NAMES

# Configuration
MODEL_NAME = "diffucoder"
DATA_PATH = "data/test.csv"
OUTPUT_DIR = "experiments/smell_as_noise_refactoring/results"
NOISE_LEVELS = [0.1, 0.3, 0.5, 0.7, 0.9]
TRIALS_PER_LEVEL = 3  # Reduce trials to save time, increase for final run

def apply_random_mask(input_ids, mask_token_id, noise_level):
    """
    Randomly mask `noise_level` fraction of tokens.
    """
    seq_len = input_ids.shape[1]
    num_mask = int(seq_len * noise_level)
    if num_mask == 0:
        return input_ids.clone(), []

    # Create mask indices
    indices = torch.randperm(seq_len)[:num_mask]
    masked_input = input_ids.clone()
    masked_input[0, indices] = mask_token_id
    
    return masked_input, indices.tolist()

def run_experiment():
    parser = argparse.ArgumentParser(description="Run Refactoring Threshold Experiment")
    parser.add_argument("--debug", action="store_true", help="Run with only 20 samples for verification")
    args = parser.parse_args()

    limit = 20 if args.debug else None
    print(f"Running experiment with limit={limit} (Debug mode: {args.debug})")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load model
    model, tokenizer, cfg = load_model(MODEL_NAME)
    mask_token_id = tokenizer.convert_tokens_to_ids(cfg['mask_token'])
    device = model.device
    print(f"Running on device: {device}")
    
    # Load data
    df = load_data(DATA_PATH, limit=limit)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(OUTPUT_DIR, f"retention_rate_{timestamp}.csv")
    
    # Prepare CSV
    fieldnames = [
        'sample_id', 'condition', 'noise_level', 'trial_id', 
        'original_name', 'target_name', 'predicted_name', 
        'is_restored', 'is_refactored', 'target_masked'
    ]
    
    with open(results_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing samples"):
            sample_id = row['id']
            code_template = row['X']
            ground_truth = row['y']
            
            # Select a random bad name for this sample
            bad_name = random.choice(BAD_NAMES)
            
            # Truncate code if too long, centered around [MASK]
            mask_pos = code_template.find('[MASK]')
            if mask_pos != -1:
                # Approximate 1024 tokens ~ 3000 chars. Use 2500 to be safe.
                WINDOW_SIZE = 2500
                code_len = len(code_template)
                
                if code_len > WINDOW_SIZE:
                    start = max(0, mask_pos - WINDOW_SIZE // 2)
                    end = min(code_len, start + WINDOW_SIZE)
                    
                    # Adjust if window is too small (e.g. mask at very end)
                    if end - start < WINDOW_SIZE and start > 0:
                        start = max(0, end - WINDOW_SIZE)
                    
                    # Apply truncation
                    code_template = code_template[start:end]
                    # Note: We don't update ground_truth or bad_name, just the context
            
            # Prepare two conditions: Smelly and Clean
            conditions = [
                ('smelly', code_template.replace('[MASK]', bad_name), bad_name, ground_truth),
                ('clean', code_template.replace('[MASK]', ground_truth), ground_truth, ground_truth) # Target is itself
            ]
            
            for cond_name, code, current_name, target_refactor_name in conditions:
                # Tokenize
                inputs = tokenizer(code, return_tensors="pt", truncation=True, max_length=1024)
                input_ids = inputs.input_ids.to(device)
                attention_mask = inputs.attention_mask.to(device)
                
                # Identify where the variable of interest is
                # Note: Tokenization might split the name into multiple tokens.
                # We look for the sequence of tokens corresponding to current_name.
                target_indices_list = find_token_indices(tokenizer, input_ids[0], current_name)
                
                if not target_indices_list:
                    print(f"Skipping sample {sample_id} ({cond_name}): Target '{current_name}' not found in tokens.")
                    continue
                
                # We only care about the first occurrence (where [MASK] was)
                # In a real scenario, we might want all, but [MASK] is unique usually.
                target_range = target_indices_list[0] # (start, end)
                target_tokens = input_ids[0, target_range[0]:target_range[1]].tolist()
                
                for noise_level in NOISE_LEVELS:
                    for trial in range(TRIALS_PER_LEVEL):
                        # Apply random mask
                        masked_input, masked_indices = apply_random_mask(input_ids, mask_token_id, noise_level)
                        
                        # Check if our target variable was masked
                        # We consider it "masked" if ANY of its tokens were masked.
                        is_target_masked = any(idx in masked_indices for idx in range(target_range[0], target_range[1]))
                        
                        # Run generation (inpainting)
                        # We use the model to fill in the masks.
                        with torch.no_grad():
                            # DiffuCoder generate
                            output = model.diffusion_generate(
                                masked_input,
                                attention_mask=attention_mask,
                                max_length=input_ids.shape[1] + 1, # Keep length roughly same
                                steps=32, # Fast generation for experiment
                                temperature=0.0, # Deterministic filling preferred for stability check
                                top_p=1.0,
                                top_k=50, 
                                alg="entropy", # Explicitly set algorithm to entropy to be safe
                                alg_temp=0.,
                            )
                        
                        # Decode output
                        seqs = output.sequences if hasattr(output, "sequences") else output
                        predicted_ids = seqs[0]
                        
                        # Extract the predicted tokens at the target position
                        # Note: The length might have changed if we used autoregressive generation, 
                        # but DiffuCoder is usually length-preserving or we can align.
                        # For simplicity, we assume length preservation or just look at the text.
                        # Wait, diffusion_generate might change length? 
                        # DiffuCoder is non-autoregressive (mostly), but if it's based on LLaMA it might be AR.
                        # Let's check the output text directly.
                        
                        decoded_text = tokenizer.decode(predicted_ids, skip_special_tokens=True)
                        
                        # Heuristic: We need to find what replaced the target.
                        # Since we know the context, we can try to locate it.
                        # Or simpler: just check if the `target_refactor_name` or `current_name` is present 
                        # in the approximate location.
                        
                        # Better approach: Compare tokens if length is preserved.
                        if predicted_ids.shape[0] == input_ids.shape[1]:
                            pred_target_tokens = predicted_ids[target_range[0]:target_range[1]].tolist()
                            predicted_name = tokenizer.decode(pred_target_tokens, skip_special_tokens=True).strip()
                        else:
                            # Fallback: String matching in the whole text? 
                            # This is risky. Let's try to align or just search.
                            # For this experiment, let's assume we can find it.
                            # We will use a simple string containment check around the neighborhood?
                            # No, let's just use the decoded text and regex if possible.
                            # Actually, for "stability", exact token match is best.
                            # If lengths differ, we mark as "structure_changed".
                            predicted_name = "LENGTH_MISMATCH"
                        
                        # Metrics
                        is_restored = (predicted_name == current_name)
                        is_refactored = (predicted_name == target_refactor_name)
                        
                        writer.writerow({
                            'sample_id': sample_id,
                            'condition': cond_name,
                            'noise_level': noise_level,
                            'trial_id': trial,
                            'original_name': current_name,
                            'target_name': target_refactor_name,
                            'predicted_name': predicted_name,
                            'is_restored': is_restored,
                            'is_refactored': is_refactored,
                            'target_masked': is_target_masked
                        })
                        f.flush()

    print(f"Experiment finished. Results saved to {results_file}")

if __name__ == "__main__":
    run_experiment()
