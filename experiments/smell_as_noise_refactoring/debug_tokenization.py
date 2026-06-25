import torch
import pandas as pd
from transformers import AutoTokenizer
from utils import load_data, find_token_indices, BAD_NAMES
import random

MODEL_ID = "apple/DiffuCoder-7B-Instruct"

def debug_tokenization():
    print(f"Loading tokenizer: {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    
    # Load a few samples
    df = load_data("data/test.csv", limit=5)
    
    for idx, row in df.iterrows():
        print(f"\n{'='*40}")
        print(f"Sample {row['id']}")
        print(f"{'='*40}")
        
        code_template = row['X']
        ground_truth = row['y']
        bad_name = "data" # Fix a bad name for debugging
        
        conditions = [
            ('smelly', code_template.replace('[MASK]', bad_name), bad_name),
            ('clean', code_template.replace('[MASK]', ground_truth), ground_truth)
        ]
        
        for cond_name, code, target_name in conditions:
            print(f"\n--- Condition: {cond_name} ---")
            print(f"Target Name: '{target_name}'")
            
            # Tokenize full code
            inputs = tokenizer(code, return_tensors="pt", truncation=True, max_length=1024)
            input_ids = inputs.input_ids[0]
            
            print(f"Code Length: {len(code)} chars")
            print(f"Token Count: {len(input_ids)}")
            
            # Check if target is likely in the truncated part
            target_pos = code.find(target_name)
            print(f"Target '{target_name}' position in code: {target_pos}")
            
            # Rough estimation: 1 token approx 3-4 chars
            estimated_token_pos = target_pos / 3.5
            print(f"Estimated token position: ~{int(estimated_token_pos)}")
            
            if len(input_ids) == 1024 and estimated_token_pos > 900:
                print("⚠️ WARNING: Target likely truncated!")

            
            # Tokenize target name in isolation
            target_ids_raw = tokenizer.encode(target_name, add_special_tokens=False)
            target_tokens_raw = [tokenizer.decode([tid]) for tid in target_ids_raw]
            
            target_ids_space = tokenizer.encode(" " + target_name, add_special_tokens=False)
            target_tokens_space = [tokenizer.decode([tid]) for tid in target_ids_space]
            
            print(f"Target '{target_name}' tokens (isolated): {target_tokens_raw} (IDs: {target_ids_raw})")
            print(f"Target ' {target_name}' tokens (isolated): {target_tokens_space} (IDs: {target_ids_space})")
            
            # Try to find in full code
            indices = find_token_indices(tokenizer, input_ids, target_name)
            
            # Get raw tokens for debugging
            raw_tokens = tokenizer.convert_ids_to_tokens(input_ids)
            
            if indices:
                print(f"✅ Found at indices: {indices}")
                for start, end in indices:
                    found_tokens = raw_tokens[start:end]
                    print(f"   Matched raw tokens: {found_tokens}")
            else:
                print(f"❌ NOT FOUND")
                print("Dumping first 200 raw tokens to debug:")
                for i, tok in enumerate(raw_tokens[:200]):
                    # Simple heuristic: print if it looks like part of the target
                    clean_tok = tok.replace(' ', '')
                    if any(part in clean_tok for part in target_name.split(' ')) or \
                       any(part in target_name for part in clean_tok if len(clean_tok) > 2):
                        print(f"   {i}: {repr(tok)} (ID: {input_ids[i]})")
                
                # Also verify if the code even contains the string
                if target_name not in code:
                    print(f"CRITICAL: Target '{target_name}' NOT in source code string!")
                else:
                    print(f"Target '{target_name}' IS present in source code string.")

if __name__ == "__main__":
    debug_tokenization()
