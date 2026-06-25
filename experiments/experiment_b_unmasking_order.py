import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from tqdm import tqdm
from datetime import datetime
from transformers import AutoTokenizer, AutoModel
import re
import matplotlib.pyplot as plt
import seaborn as sns

# ==============================================================================
# Environment Setting: Mock torchvision for DiffuCoder compatibility
# ==============================================================================
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
# ==============================================================================

# Configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "apple/DiffuCoder-7B-Instruct"
DATA_PATH = "CoRefusion/data/test.csv"
RESULTS_DIR = "results"
TOTAL_STEPS = 64
CONFIDENCE_THRESHOLD = 0.8  # Threshold for "first confident step"

def get_num_transfer_tokens(mask_index, steps):
    mask_num = mask_index.sum(dim=1, keepdim=True)
    base = mask_num // steps
    remainder = mask_num % steps
    num_transfer_tokens = torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64) + base
    for i in range(mask_num.size(0)):
        num_transfer_tokens[i, :remainder[i]] += 1
    return num_transfer_tokens

def add_gumbel_noise(logits, temperature):
    if temperature == 0:
        return logits
    logits = logits.to(torch.float32)
    noise = torch.rand_like(logits, dtype=torch.float32)
    gumbel_noise = (- torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise

def get_java_identifier_metadata(text, tokenizer, input_ids_tensor):
    input_ids = input_ids_tensor[0].tolist()
    
    java_keywords = {
        "public", "static", "int", "if", "return", "void", "class", "for", "new", "boolean",
        "private", "protected", "final", "else", "while", "this", "null", "true", "false",
        "try", "catch", "throw", "throws", "import", "package", "byte", "char", "short", "long",
        "float", "double", "switch", "case", "default", "break", "continue", "interface", "extends", "implements"
    }
    
    try:
        from tree_sitter_languages import get_parser
        parser = get_parser('java')
        tree = parser.parse(bytes(text, "utf8"))
        id_ranges = []
        def traverse(node):
            if node.type == 'identifier':
                name = text[node.start_byte:node.end_byte]
                if name not in java_keywords:
                    id_ranges.append((node.start_byte, node.end_byte, name))
            for child in node.children:
                traverse(child)
        traverse(tree.root_node)
    except Exception as e:
        id_ranges = []
        for m in re.finditer(r'\b[A-Za-z_][A-Za-z0-9_]*\b', text):
            if m.group(0) not in java_keywords:
                id_ranges.append((m.start(), m.end(), m.group(0)))

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
            identifier_groups.append({'name': id_name, 'indices': group_indices})
            
    return mask, identifier_groups

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    print(f"Loading tokenizer and model: {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        MODEL_NAME, 
        torch_dtype=torch.bfloat16, 
        trust_remote_code=True
    ).to(DEVICE).eval()
    mask_token_id = tokenizer.convert_tokens_to_ids("<|mask|>")
    
    print(f"Loading data from {DATA_PATH}...")
    try:
        df = pd.read_csv(DATA_PATH, header=None, names=['id', 'masked_code', 'target'])
    except Exception as e:
        print(f"Dataset load error: {e}. Ensure path is correct.")
        return
    
    df = df.head(20)
    
    token_results = []

    print("Running Unmasking Order Analysis...")
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        try:
            sample_id = row['id']
            masked_code = str(row['masked_code'])

            if '[MASK]' not in masked_code:
                continue

            DUMMY_SMELL_ID = "SMELL_DUMMY_TOKEN"
            code_str = masked_code.replace("[MASK]", DUMMY_SMELL_ID)

            inputs = tokenizer(code_str, return_tensors="pt").to(DEVICE)
            input_ids = inputs.input_ids
            attention_mask = inputs.attention_mask

            identifier_mask, id_groups = get_java_identifier_metadata(code_str, tokenizer, input_ids)
            identifier_mask = identifier_mask.to(DEVICE)

            if not identifier_mask.any():
                continue

            x = input_ids.clone()
            x[0, identifier_mask] = mask_token_id

            num_transfer_tokens = get_num_transfer_tokens(identifier_mask.unsqueeze(0), TOTAL_STEPS)

            seq_len = x.shape[1]
            flip_steps = {i: -1 for i in range(seq_len)}
            first_confident_steps = {i: -1 for i in range(seq_len)}

            for step_i in range(TOTAL_STEPS):
                with torch.no_grad():
                    current_mask_index = (x == mask_token_id)
                    if not current_mask_index.any():
                        break

                    diff_outputs = model(x, attention_mask=attention_mask.bool())
                    logits = diff_outputs.logits
                    logits_with_noise = add_gumbel_noise(logits, temperature=0.3)
                    x0 = torch.argmax(logits_with_noise, dim=-1)

                    p_all = F.softmax(logits.float(), dim=-1)
                    x0_p = torch.squeeze(torch.gather(p_all, dim=-1, index=torch.unsqueeze(x0, -1)), -1)

                    for i in range(seq_len):
                        if current_mask_index[0, i].item():
                            if x0_p[0, i].item() > CONFIDENCE_THRESHOLD and first_confident_steps[i] == -1:
                                first_confident_steps[i] = step_i

                    confidence = torch.where(current_mask_index, x0_p, torch.tensor(-np.inf, device=DEVICE))

                    transfer_index = torch.zeros_like(x0, dtype=torch.bool)
                    k_val = num_transfer_tokens[0, step_i].item() if num_transfer_tokens.shape[1] > step_i else 0
                    if k_val > 0:
                        k_val = min(k_val, current_mask_index.sum().item())
                        _, sel = torch.topk(confidence[0], k=int(k_val))
                        transfer_index[0, sel] = True
                        x[transfer_index] = x0[transfer_index]

                        for i in sel.tolist():
                            if flip_steps[i] == -1:
                                flip_steps[i] = step_i

            for group in id_groups:
                name = group['name']
                indices = group['indices']
                is_smell = (name == DUMMY_SMELL_ID)

                valid_flips = [flip_steps[i] for i in indices if flip_steps.get(i, -1) != -1]
                valid_conf = [first_confident_steps[i] for i in indices if first_confident_steps.get(i, -1) != -1]

                avg_flip = sum(valid_flips) / len(valid_flips) if valid_flips else TOTAL_STEPS
                avg_conf = sum(valid_conf) / len(valid_conf) if valid_conf else TOTAL_STEPS

                token_results.append({
                    'sample_id': sample_id,
                    'is_smell_token': is_smell,
                    'identifier_name': name,
                    'avg_flip_step': avg_flip,
                    'first_confident_step': avg_conf
                })

        except Exception as e:
            if 'out of memory' in str(e).lower() or 'oom' in str(e).lower():
                print(f'\n[OOM] Skipping sample {row.get("id", "unknown")}...')
                torch.cuda.empty_cache()
                import gc; gc.collect()
                continue
            else:
                raise e

    df_results = pd.DataFrame(token_results)
    
    if df_results.empty:
        print("No results to save.")
        return
        
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_csv = f"{RESULTS_DIR}/unmasking_order_{ts}.csv"
    df_results.to_csv(out_csv, index=False)
    print(f"\nExperiment complete. Data saved to {out_csv}")
    
    print("\n--- Summary Statistics ---")
    summary = df_results.groupby('is_smell_token')[['avg_flip_step', 'first_confident_step']].agg(['mean', 'std', 'count'])
    print(summary)
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    sns.boxplot(data=df_results, x='is_smell_token', y='first_confident_step', palette='Set2')
    plt.title('First Confident Step: Smell vs Non-Smell')
    plt.ylabel('Step (0 to 64)')
    
    plt.subplot(1, 2, 2)
    sns.boxplot(data=df_results, x='is_smell_token', y='avg_flip_step', palette='Set2')
    plt.title('Flip Step: Smell vs Non-Smell')
    plt.ylabel('Step (0 to 64)')
    
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/unmasking_order_boxplot_{ts}.png")
    plt.close()

if __name__ == "__main__":
    main()
