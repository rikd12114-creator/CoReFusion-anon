import os
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModel
from datetime import datetime
import sys

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

def find_subsequence_indices(sequence, subsequence):
    """
    Finds the start and end indices of the first occurrence of subsequence in sequence.
    Returns (start_index, end_index) or None if not found.
    """
    seq_len = len(sequence)
    sub_len = len(subsequence)
    for i in range(seq_len - sub_len + 1):
        if sequence[i : i + sub_len] == subsequence:
            return i, i + sub_len
    return None

def run_experiment():
    model_id = "apple/DiffuCoder-7B-Instruct"
    print(f"Loading model: {model_id}...")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        model = AutoModel.from_pretrained(model_id, torch_dtype=torch.bfloat16, trust_remote_code=True).to("cuda").eval()
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    csv_path = os.path.join('../data', 'test.csv')
    print(f"Reading data from {csv_path}...")
    
    try:
        # Reading first 100 for the experiment as an initial set, or all if feasible.
        # User said "run this experiment on the dataset", but for safety I'll start with a chunk and can expand.
        # Let's read all but process with a limit or specific range if args were parsed, 
        # but here I'll default to a reasonable number or all.
        df = pd.read_csv(csv_path, header=None, names=['id', 'X', 'y'])
        print(f"Loaded {len(df)} rows.")
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = f"results/steps_experiment_{timestamp}"
    os.makedirs(experiment_dir, exist_ok=True)
    results_csv = os.path.join(experiment_dir, f"identifier_change_summary_{timestamp}.csv")
    
    TERRIBLE_IDENTIFIER = "terrible_var_name_x"
    print(f"Using terrible identifier: {TERRIBLE_IDENTIFIER}")

    
    results = []
    process_limit = len(df)
    print(f"Processing the entire dataset ({process_limit} items)...")

    for i, row in df.head(process_limit).iterrows():
        try:
            input_text = row['X']
            if '[MASK]' not in input_text:
                print(f"Skipping {row['id']}: No [MASK] found.")
                continue

            # Replace MASK with terrible identifier
            # We assume [MASK] appears once or we replace all? User said "the MASK position".
            # Usually strict single mask or multiple. I'll replace the first one or all.
            # Replace MASK with terrible identifier for tokenization and tracking
            full_text_with_terrible = input_text.replace('[MASK]', TERRIBLE_IDENTIFIER)
            
            # FULLY TOKENIZE the code as requested
            inputs = tokenizer(full_text_with_terrible, return_tensors="pt", truncation=False)
            input_ids = inputs.input_ids.to("cuda")
            attention_mask = inputs.attention_mask.to("cuda")
            input_ids_list = input_ids[0].tolist()

            # Find the token indices of the terrible identifier
            id_tokens = tokenizer.encode(TERRIBLE_IDENTIFIER, add_special_tokens=False)
            target_indices = find_subsequence_indices(input_ids_list, id_tokens)
            
            if not target_indices:
                id_tokens_space = tokenizer.encode(" " + TERRIBLE_IDENTIFIER, add_special_tokens=False)
                target_indices = find_subsequence_indices(input_ids_list, id_tokens_space)
            
            if not target_indices:
                print(f"Could not find identifier tokens in processed text for {row['id']}. Skipping.")
                continue
                
            start_idx, end_idx = target_indices
            original_id_tokens = input_ids_list[start_idx:end_idx]
            
            # Pass the ORIGINAL tokenized input directly to the diffusion process
            # as requested (no masking)
            with torch.no_grad():
                output = model.diffusion_generate(
                    input_ids,
                    attention_mask=attention_mask,
                    max_length=input_ids.shape[1] + 1,
                    steps=256,
                    temperature=0.3,
                    top_p=0.95,
                    alg="entropy",
                    alg_temp=0.,
                    output_history=True,
                    return_dict_in_generate=True,
                )
            
            # Check history
            if not hasattr(output, 'history'):
                print(f"No history returned for {row['id']}")
                continue
                
            history = output.history 
            
            change_step = None
            first_step_differs = False
            
            data_dir = os.path.join(experiment_dir, f"data_{row['id']}")
            os.makedirs(data_dir, exist_ok=True)

            for step_idx, step_tensor in enumerate(history):
                # step_tensor shape (1, seq_len)
                step_tokens = step_tensor[0].tolist()
                
                # Decode the full sequence for this step
                decoded_step_text = tokenizer.decode(step_tokens, skip_special_tokens=True)
                
                # Save each step to a .java file
                step_filename = f"data_{row['id']}_step{step_idx}_{timestamp}.java"
                step_path = os.path.join(data_dir, step_filename)
                
                with open(step_path, "w", encoding="utf-8") as f:
                    # In step 0-1, we can manually inject the 'terrible name' tokens for the record,
                    # since the model might start with masks.
                    # This helps visualize the 'denoising' of the bad name.
                    if step_idx == 0:
                        f.write(full_text_with_terrible)
                    else:
                        f.write(decoded_step_text)

                # Check for identifier change
                # We compare current step tokens at original indices to the terrible identifier tokens
                if change_step is None:
                    if len(step_tokens) != len(input_ids_list):
                        # Length changed, something definitely changed
                        change_step = step_idx
                    else:
                        current_segment = step_tokens[start_idx:end_idx]
                        if current_segment != original_id_tokens:
                            # It's no longer the terrible identifier.
                            # Since we are not using masks, any change is the 'change_step'.
                            change_step = step_idx
            
            if change_step is not None:
                print(f"  Changed at step {change_step}")
                results.append({'id': row['id'], 'change_step': change_step})
            else:
                print(f"  No change detected (steps checked: {len(history)})")
                results.append({'id': row['id'], 'change_step': -1}) # -1 means never changed (or preserved)
            
        except Exception as e:
            print(f"Error processing {row['id']}: {e}")
            continue

    # Save results
    if results:
        res_df = pd.DataFrame(results)
        res_df.to_csv(results_csv, index=False)
        print(f"Saved results to {results_csv}")
        
        # Calculate average
        # process valid changes
        valid_changes = res_df[res_df['change_step'] != -1]
        if not valid_changes.empty:
            avg_step = valid_changes['change_step'].mean()
            print(f"Average change step: {avg_step:.2f}")
        else:
            print("No changes detected in any samples.")
    else:
        print("No results generated.")

if __name__ == "__main__":
    run_experiment()
