import os
import torch
import sys
import re
import pandas as pd
import csv
from io import BytesIO
from transformers import AutoTokenizer, AutoModel
from datetime import datetime
from tqdm import tqdm
from huggingface_hub import HfApi

# --- Environment Setting: Mock torchvision ---
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

# --------------------------------------------------

def get_java_identifier_metadata(text, tokenizer, input_ids_tensor):
    """
    Returns (mask, identifier_groups)
    Each group: {'name': str, 'indices': list of token indices, 'range': (start_byte, end_byte)}
    """
    input_ids = input_ids_tensor[0].tolist()
    
    java_keywords = {
        "public", "static", "int", "if", "return", "void", "class", "for", "new", "boolean",
        "private", "protected", "final", "else", "while", "this", "null", "true", "false",
        "long", "double", "float", "char", "byte", "short", "import", "package", "try", "catch", "throw"
    }
    
    id_ranges = []
    try:
        from tree_sitter_languages import get_parser
        parser = get_parser('java')
        tree = parser.parse(bytes(text, "utf8"))
        def traverse(node):
            if node.type == 'identifier':
                name = text[node.start_byte:node.end_byte]
                if name not in java_keywords:
                    id_ranges.append((node.start_byte, node.end_byte, name))
            for child in node.children:
                traverse(child)
        traverse(tree.root_node)
    except Exception as e:
        # Regex Fallback
        for m in re.finditer(r'\b[A-Za-z_][A-Za-z0-9_]*\b', text):
            if m.group(0) not in java_keywords:
                id_ranges.append((m.start(), m.end(), m.group(0)))

    # Manual Token Offset Calculation
    token_offsets = []
    for i in range(len(input_ids)):
        prefix = tokenizer.decode(input_ids[:i], skip_special_tokens=False)
        full = tokenizer.decode(input_ids[:i+1], skip_special_tokens=False)
        token_offsets.append((len(prefix), len(full)))

    identifier_groups = []
    mask = torch.zeros(len(input_ids), dtype=torch.bool)
    
    for start_byte, end_byte, id_name in id_ranges:
        group_indices = []
        for i, (t_start, t_end) in enumerate(token_offsets):
            t_mid = (t_start + t_end) / 2
            if start_byte <= t_mid < end_byte:
                group_indices.append(i)
                mask[i] = True
        if group_indices:
            identifier_groups.append({
                'name': id_name, 
                'indices': group_indices,
                'range': (start_byte, end_byte)
            })
            
    return mask, identifier_groups

def run_tracking(tokenizer, model, code_snippet, target_range, mask_token_id):
    """
    Runs diffusion and calculates avg_step for the identifier in target_range.
    """
    inputs = tokenizer(code_snippet, return_tensors="pt")
    input_ids = inputs.input_ids.to("cuda")
    
    # 1. Get identity mask
    identifier_mask, id_groups = get_java_identifier_metadata(code_snippet, tokenizer, input_ids)
    identifier_mask = identifier_mask.to("cuda")
    
    # 2. Find ONLY the target identifier group
    target_group = None
    if target_range:
        m_start, m_end = target_range
        for group in id_groups:
            g_start, g_end = group['range']
            # Intersection check
            if max(m_start, g_start) < min(m_end, g_end):
                target_group = group
                break
    
    if not target_group:
        return None, None, None

    # 3. Mask target in input
    constrained_input_ids = input_ids.clone()
    constrained_input_ids[0, identifier_mask] = mask_token_id
    
    # 4. Diffusion (DreamCoder parameters)
    with torch.no_grad():
        output = model.diffusion_generate(
            constrained_input_ids,
            attention_mask=inputs.attention_mask.to("cuda"),
            max_length=input_ids.shape[1] + 1,
            steps=768,
            output_history=True,
            return_dict_in_generate=True,
            temperature=0.1
        )
    
    history = output.history
    indices = target_group['indices']
    orig_name = target_group['name']
    
    # Step Analytics for target
    fill_step = -1
    final_toks = []
    for s_idx, h in enumerate(history):
        current = [h[0, i].item() for i in indices]
        if mask_token_id not in current:
            fill_step = s_idx
            final_toks = current
            break
            
    if fill_step != -1:
        res_raw = tokenizer.decode(final_toks).strip()
        res_clean = re.sub(r'[^A-Za-z0-9_]', '', res_raw)
        return float(fill_step), res_clean, orig_name
    return None, None, orig_name

def upload_to_hf(file_path, repo_id="anonymous/Dreamcoder_naming_flip"):
    """Helper to upload files to Hugging Face."""
    try:
        api = HfApi()
        api.upload_file(
            path_or_fileobj=file_path,
            path_in_repo=f"naming_exp/{os.path.basename(file_path)}",
            repo_id=repo_id,
            repo_type="dataset"
        )
    except Exception as e:
        print(f"\n[Warning] HF Upload failed: {e}")

def main():
    LIMIT = 50 
    REPEATS = 50
    MAX_TOKENS = 1024 # Skip data points with more than 1024 tokens
    model_id = "Dream-org/Dream-Coder-v0-Instruct-7B"
    hf_repo = "anonymous/Dreamcoder_naming_flip"
    
    print(f"Loading model: {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_id, torch_dtype=torch.bfloat16, trust_remote_code=True).to("cuda").eval()
    mask_token_id = tokenizer.convert_tokens_to_ids('<|mask|>')
    
    csv_path = '../data/test.csv'
    print(f"Reading dataset: {csv_path} (Limit: {LIMIT}, Repeats: {REPEATS}, Token Limit: {MAX_TOKENS})")
    df = pd.read_csv(csv_path, header=None, names=['id', 'X', 'y'], nrows=LIMIT)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_csv = f"../results/dreamcoder_scale_naming_exp_{timestamp}.csv"
    os.makedirs('../results', exist_ok=True)

    # Prepare CSV header
    fields = ['run_id', 'id', 'ground_truth', 'step_good', 'result_good', 'step_bad', 'result_bad']
    with open(out_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()

    print(f"Starting experiment. Results will be saved to {out_csv} and uploaded to HF.")

    # Top-level Progress Bar for Runs
    pbar_runs = tqdm(total=REPEATS, desc="Total Runs")
    
    for run_idx in range(REPEATS):
        # Middle Progress Bar for Items in this Run
        pbar_items = tqdm(total=len(df), desc=f"Run {run_idx+1}/{REPEATS}", leave=False)
        
        for idx, row in df.iterrows():
            entry_id = row['id']
            X = row['X']
            y = str(row['y'])
            
            # Token length check
            tokens = tokenizer.encode(X, add_special_tokens=False)
            if len(tokens) > MAX_TOKENS:
                pbar_items.write(f"Skip {entry_id}: Exceeds {MAX_TOKENS} tokens (Actual: {len(tokens)}).")
                pbar_items.update(1)
                continue

            mask_start = X.find("[MASK]")
            if mask_start == -1:
                pbar_items.write(f"Skip {entry_id}: No [MASK] found.")
                pbar_items.update(1)
                continue
                
            # Inner Progress Bar for the two cases (Good/Bad)
            pbar_cases = tqdm(total=2, desc=f"Run {run_idx+1} ID {entry_id}", leave=False)
            
            # --- GOOD CASE ---
            code_good = X.replace("[MASK]", y)
            target_range_good = (mask_start, mask_start + len(y))
            step_good, val_good, _ = run_tracking(tokenizer, model, code_good, target_range_good, mask_token_id)
            pbar_cases.update(1)
            
            # --- BAD CASE ---
            bad_name = "terrible_var_xyz"
            code_bad = X.replace("[MASK]", bad_name)
            target_range_bad = (mask_start, mask_start + len(bad_name))
            step_bad, val_bad, _ = run_tracking(tokenizer, model, code_bad, target_range_bad, mask_token_id)
            pbar_cases.update(1)
            
            pbar_cases.close()
            
            # Collect result
            res_data = {
                'run_id': run_idx + 1,
                'id': entry_id,
                'ground_truth': y,
                'step_good': step_good,
                'result_good': val_good,
                'step_bad': step_bad,
                'result_bad': val_bad
            }
            
            # Write to CSV immediately
            with open(out_csv, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fields)
                writer.writerow(res_data)
                
            # Upload to HF after each item iteration to be safe
            upload_to_hf(out_csv, repo_id=hf_repo)
            
            pbar_items.update(1)
            
        pbar_items.close()
        pbar_runs.update(1)

    print("\n" + "="*50)
    print(f"DreamCoder Experiment Finished! Results saved to: {out_csv}")
    print("="*50)

if __name__ == "__main__":
    main()
