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
        # Add epsilon to prevent log(0)
        entropy = -torch.sum(probs * torch.log(probs + 1e-9), dim=-1)
    return entropy

def identify_bad_names(entropy, top_k_percent=0.15):
    # For this experiment, we might want manual control, but entropy is a good start
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

def expand_sequence_with_masks(input_ids, mask_indices, mask_token_id, expansion_factor=3):
    """
    Expands the input_ids sequence by replacing tokens at mask_indices
    with multiple mask_token_ids.
    """
    orig_list = input_ids[0].tolist()
    new_list = []
    
    # Create set for faster lookup
    mask_indices_set = set(mask_indices.tolist())
    
    for i, token in enumerate(orig_list):
        if i in mask_indices_set:
            # Expand this position
            new_list.extend([mask_token_id] * expansion_factor)
        else:
            new_list.append(token)
            
    # Create new tensors
    new_input_ids = torch.tensor([new_list], device=input_ids.device)
    new_attention_mask = torch.ones_like(new_input_ids, device=input_ids.device)
    
    return new_input_ids, new_attention_mask

def run_multi_token_noise_experiment(code_snippet, total_steps=20, expansion_factor=3):
    # Setup logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"multi_token_noise_exp_{timestamp}.txt"
    # Ensure results directory exists
    os.makedirs("results", exist_ok=True)
    log_path = os.path.join("results", log_filename)
    
    log_file = open(log_path, "w", encoding="utf-8")
    
    def log_and_print(msg):
        print(msg)
        log_file.write(msg + "\n")

    model_id = "Dream-org/Dream-Coder-v0-Instruct-7B"
    log_and_print(f"Loading {model_id}...")
    log_and_print(f"Experiment Configuration:")
    log_and_print(f"  Expansion Factor: {expansion_factor} mask tokens per target")
    log_and_print(f"  Total Steps: {total_steps}")
    
    # Device detection
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    log_and_print(f"Using device: {device}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_id, torch_dtype=torch.bfloat16, trust_remote_code=True).to(device).eval()
    
    # Initial Tokenization
    inputs = tokenizer(code_snippet, return_tensors="pt").to(device)
    input_ids = inputs.input_ids
    attention_mask = inputs.attention_mask
    
    # Identify candidates for masking (using entropy for now, as in previous scripts)
    log_and_print("\nCalcalating Entropy to identify masked tokens...")
    entropy = calculate_token_entropy(model, input_ids, attention_mask.bool())
    mask_indices = identify_bad_names(entropy, top_k_percent=0.1) # Mask top 10%
    
    # Display what we are masking
    log_and_print(f"Identified {len(mask_indices)} tokens to mask/expand.")
    masked_words = [tokenizer.decode([input_ids[0][idx]]) for idx in mask_indices]
    log_and_print(f"Tokens to be expanded: {masked_words}")

    # Re-construct input with expanded masks
    mask_token_id = tokenizer.mask_token_id if tokenizer.mask_token_id is not None else 126336
    
    x, attention_mask = expand_sequence_with_masks(input_ids, mask_indices, mask_token_id, expansion_factor)
    
    log_and_print("\nExpanded Masked Code (Initial State):")
    log_and_print(tokenizer.decode(x[0], skip_special_tokens=False))
    log_and_print("-" * 30)

    # Diffusion Loop
    initial_mask_index = (x == mask_token_id)
    num_transfer_tokens = get_num_transfer_tokens(initial_mask_index, total_steps)
    
    temperature = 0.1
    
    for i in range(total_steps):
        with torch.no_grad():
            current_mask_index = (x == mask_token_id)
            
            # Predict
            logits = model(x, attention_mask=attention_mask.bool()).logits
            
            # Sample (with some noise)
            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1)
            
            # Confidence for remasking
            p = F.softmax(logits, dim=-1)
            x0_p = torch.squeeze(torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1)
            
            # We only care about tokens that are currently masked
            x0 = torch.where(current_mask_index, x0, x)
            confidence = torch.where(current_mask_index, x0_p, -np.inf)
            
            # Select which tokens to "unmask" (transfer)
            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
            
            if num_transfer_tokens.shape[1] > i:
                 k_transfer = num_transfer_tokens[0, i]
            else:
                 k_transfer = 0

            if k_transfer > 0:
                _, select_index = torch.topk(confidence[0], k=k_transfer)
                transfer_index[0, select_index] = True
                x[transfer_index] = x0[transfer_index]
            
            # Visualization
            if (i + 1) % 5 == 0 or (i + 1) == total_steps:
                current_code = tokenizer.decode(x[0], skip_special_tokens=True)
                log_and_print(f"Step {i+1}/{total_steps}:")
                log_and_print(current_code.strip())
                log_and_print("-" * 20)

    final_code = tokenizer.decode(x[0], skip_special_tokens=True).strip()
    log_and_print("\nFinal Refactored Code:")
    log_and_print(final_code)
    
    log_file.close()

if __name__ == "__main__":
    # Example code snippet with some potentially bad identifiers
    bad_code = """
def p(a, b):
    # a is radius, b is pi
    r = b * (a ** 2)
    return r
    
lst = [1, 2, 3]
s = 0
for i in lst:
    s += i
"""
    # 5 noise tokens per mask to really give it space
    run_multi_token_noise_experiment(bad_code, total_steps=20, expansion_factor=5)
