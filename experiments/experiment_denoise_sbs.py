
import sys
import os
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from datetime import datetime
from huggingface_hub import HfApi
from transformers import AutoTokenizer, AutoModel

# --- Mock torchvision for DreamCoder compatibility ---
class MockModule:
    def __getattr__(self, name):
        return MockModule()
    def __call__(self, *args, **kwargs):
        return MockModule()

sys.modules['torchvision'] = MockModule()
sys.modules['torchvision.ops'] = MockModule()
sys.modules['torchvision.transforms'] = MockModule()

import torch
if not hasattr(torch.ops, 'torchvision'):
    class DummyOps:
        def nms(*args, **kwargs): return torch.tensor([])
    torch.ops.torchvision = DummyOps()
# -----------------------------------------------------

MODELS = {
    "dreamcoder": "Dream-org/Dream-Coder-v0-Instruct-7B",
    "llada": "GSAI-ML/LLaDA-8B-Instruct",
    "diffucoder": "apple/DiffuCoder-7B-Instruct"
}



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
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (- torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise

def run_diffusion_sbs(model_name, model_id, code_snippet, output_dir, total_steps=20):
    print(f"Loading {model_name} ({model_id})...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_id, torch_dtype=torch.bfloat16, trust_remote_code=True).to("cuda").eval()
    
    # 1. Tokenize Data Point (No Prompt / Unconditional interpretation matching 'on it')
    inputs = tokenizer(code_snippet, return_tensors="pt").to("cuda")
    input_ids = inputs.input_ids
    attention_mask = inputs.attention_mask
    
    # 2. Setup Logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Naming format: 模型_MASK_日期_时间
    # Using 'MASK' literal as requested, though we could specify mask ratio.
    result_filename = f"{model_name}_MASK_{timestamp}.txt"
    result_path = os.path.join(output_dir, result_filename)
    
    with open(result_path, "w", encoding="utf-8") as f:
        f.write(f"Model: {model_name}\n")
        f.write(f"Model ID: {model_id}\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write("-" * 40 + "\n\n")
        
        f.write("Original Code:\n")
        f.write(code_snippet + "\n")
        f.write("-" * 40 + "\n\n")

        # 3. Initialize Input (Treating original data as the noisy state)
        # User requested to remove automatic noise adding / entropy masking logic.
        x = input_ids.clone()
        
        # Ensure we have the correct mask token id. 
        mask_token_id = tokenizer.mask_token_id
        if mask_token_id is None:
             # Fallback known ID for these LLaMA-based diffusion models
            mask_token_id = 126336 
        
        f.write("Initial Input (No automated masking):\n")
        f.write(tokenizer.decode(x[0], skip_special_tokens=False) + "\n")
        f.write("-" * 40 + "\n\n")
        
        # 4. Diffusion Loop
        initial_mask_index = (x == mask_token_id)
        num_transfer_tokens = get_num_transfer_tokens(initial_mask_index, total_steps)
        temperature = 0.1 # Low temp for reconstruction
        
        # Store intermediate steps
        steps_data = []
        
        for i in range(total_steps):
            with torch.no_grad():
                current_mask_index = (x == mask_token_id)
                
                # Predict
                logits = model(x, attention_mask=attention_mask.bool()).logits
                
                # Sample
                logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
                x0 = torch.argmax(logits_with_noise, dim=-1)
                
                # Low confidence remasking strategy
                p = F.softmax(logits, dim=-1)
                x0_p = torch.squeeze(torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1)
                
                x0 = torch.where(current_mask_index, x0, x)
                confidence = torch.where(current_mask_index, x0_p, -np.inf)
                
                # Select tokens to unmask
                transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
                if num_transfer_tokens.shape[1] > i:
                    k_transfer = num_transfer_tokens[0, i]
                else:
                    k_transfer = 0
                
                if k_transfer > 0:
                    _, select_index = torch.topk(confidence[0], k=k_transfer)
                    transfer_index[0, select_index] = True
                    x[transfer_index] = x0[transfer_index]
                
                # Log Step
                params_to_save = {
                    "step": i + 1,
                    "tokens": x[0].cpu().numpy().tolist(),
                    "decoded": tokenizer.decode(x[0], skip_special_tokens=True)
                }
                steps_data.append(params_to_save)
                
                f.write(f"Step {i+1}:\n")
                f.write(params_to_save["decoded"].strip() + "\n")
                f.write("-" * 20 + "\n")

    return result_path

def main():
    # 0. Output Dir
    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Read Data Point
    csv_path = "data/test.csv"
    try:
        df = pd.read_csv(csv_path)
        # Assuming no header or specific header. The 'head' command showed: 0, "content..."
        # If headers are missing, pandas might use first row. 
        # Safest is to read with header=None if it looks like that, but '0' looks like an index.
        # Let's assume standard reading works or inspect row 0.
        # Actually '0,"/* ...' suggests the first row IS data if header is missing, or header is 0.
        # But '0' is a digit. Usually headers are strings.
        # Let's try reading and see.
        pass
    except Exception as e:
        print(f"Error reading csv: {e}")
        return

    # Let's just pick the first valid code snippet.
    # We'll read the first row's 2nd column (index 1) as the content.
    # If using pd.read_csv without header logic, we might need to be careful.
    # The 'head' output: 0,"/* ...
    # This implies no header, just data.
    df = pd.read_csv(csv_path, header=None, nrows=5)
    data_point = df.iloc[0, 1] # First row, second column
    
    print(f"Selected Data Point (Length: {len(data_point)} chars)")
    
    # 2. Iterate Models
    
    for model_name, model_id in MODELS.items():
        try:
            print(f"\nProcessing {model_name}...")
            result_file = run_diffusion_sbs(model_name, model_id, data_point, output_dir)
            
            # 3. Upload
            repo_id = "anonymous/Denoising_SBS"
            print(f"Uploading {result_file} to {repo_id}...")
            api = HfApi()
            try:
                api.upload_file(
                    path_or_fileobj=result_file,
                    path_in_repo=os.path.basename(result_file),
                    repo_id=repo_id,
                    repo_type="dataset", 
                    token="" # Will use env or cached token
                )
                print("Upload successful!")
            except Exception as e:
                print(f"Upload failed: {e}")
                
            # Cleanup
            import gc
            torch.cuda.empty_cache()
            gc.collect()
            
        except Exception as e:
            print(f"Failed to run {model_name}: {e}")

if __name__ == "__main__":
    main()
