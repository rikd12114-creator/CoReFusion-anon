import sys
import os
import torch
import torch.nn.functional as F
import numpy as np
from datetime import datetime
from huggingface_hub import HfApi

# 1. FORCE MOCK torchvision BEFORE ANYTHING ELSE
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

from transformers import AutoTokenizer, AutoModel

def calculate_token_entropy(model, input_ids, attention_mask):
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        probs = F.softmax(logits, dim=-1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-9), dim=-1)
    return entropy

def identify_bad_names(entropy, top_k_percent=0.15):
    ent = entropy.squeeze(0)
    num_to_mask = int(len(ent) * top_k_percent)
    if num_to_mask < 1: num_to_mask = 1
    _, indices = torch.topk(ent, k=num_to_mask)
    return indices

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

def run_step_by_step_refactor(code_snippet, total_steps=20, output_every=5):
    # Setup logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"dreamcoder_sbs_{timestamp}.txt"
    log_file = open(log_filename, "w", encoding="utf-8")
    
    def log_and_print(msg):
        print(msg)
        log_file.write(msg + "\n")

    model_id = "Dream-org/Dream-Coder-v0-Instruct-7B"
    log_and_print(f"Loading {model_id}...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_id, torch_dtype=torch.bfloat16, trust_remote_code=True).to("cuda").eval()
    
    inputs = tokenizer(code_snippet, return_tensors="pt").to("cuda")
    input_ids = inputs.input_ids
    attention_mask = inputs.attention_mask
    
    # Identify Noise
    entropy = calculate_token_entropy(model, input_ids, attention_mask.bool())
    mask_indices = identify_bad_names(entropy)
    
    x = input_ids.clone()
    mask_token_id = tokenizer.mask_token_id if tokenizer.mask_token_id is not None else 126336
    
    # Apply initial masks
    for idx in mask_indices:
        x[0, idx] = mask_token_id

    log_and_print("\nInitial Masked Code:")
    log_and_print(tokenizer.decode(x[0], skip_special_tokens=False))
    log_and_print("-" * 30)

    # Diffusion Loop (Adapted from LLaDA)
    initial_mask_index = (x == mask_token_id)
    num_transfer_tokens = get_num_transfer_tokens(initial_mask_index, total_steps)
    
    temperature = 0.1
    
    for i in range(total_steps):
        with torch.no_grad():
            current_mask_index = (x == mask_token_id)
            
            # Predict
            logits = model(x, attention_mask=attention_mask.bool()).logits
            
            # Sample
            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1)
            
            # Confidence for remasking (low confidence first)
            p = F.softmax(logits, dim=-1)
            x0_p = torch.squeeze(torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1)
            
            # We only care about tokens that are currently masked
            x0 = torch.where(current_mask_index, x0, x)
            confidence = torch.where(current_mask_index, x0_p, -np.inf)
            
            # Select which tokens to "unmask" (transfer)
            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
            # Batch size is 1 here
            _, select_index = torch.topk(confidence[0], k=num_transfer_tokens[0, i])
            transfer_index[0, select_index] = True
            
            # Update x
            x[transfer_index] = x0[transfer_index]
            
            # Visualization
            if (i + 1) % output_every == 0 or (i + 1) == total_steps:
                current_code = tokenizer.decode(x[0], skip_special_tokens=True)
                log_and_print(f"Step {i+1}/{total_steps}:")
                log_and_print(current_code.strip())
                log_and_print("-" * 20)

    final_code = tokenizer.decode(x[0], skip_special_tokens=True).strip()
    log_and_print("\nFinal Refactored Code:")
    log_and_print(final_code)
    
    log_file.close()

    # Upload to Hugging Face
    repo_id = "anonymous/Denoising_SBS_DreamCoder"
    hf_token = ""
    print(f"\nUploading {log_filename} to Hugging Face: {repo_id}...")
    try:
        api = HfApi()
        api.upload_file(
            path_or_fileobj=log_filename,
            path_in_repo=log_filename,
            repo_id=repo_id,
            repo_type="dataset", # Assuming it's a dataset repo, change if it's a model repo
            token=hf_token
        )
        print("Upload successful!")
    except Exception as e:
        print(f"Upload failed: {e}")

if __name__ == "__main__":
    bad_code = """
def process(a, b):
    # a is radius, b is pi
    res = b * (a ** 2)
    return res

list_1 = [1, 2, 3]
v_sum = 0
for i in list_1:
    v_sum += i
"""
    run_step_by_step_refactor(bad_code, total_steps=20, output_every=5)