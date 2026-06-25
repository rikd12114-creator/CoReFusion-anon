import sys
import os

# 1. FORCE MOCK torchvision BEFORE ANYTHING ELSE
class MockModule:
    def __getattr__(self, name):
        return MockModule()
    def __call__(self, *args, **kwargs):
        return MockModule()

# This prevents any attempt to load the real (broken) torchvision
sys.modules['torchvision'] = MockModule()
sys.modules['torchvision.ops'] = MockModule()
sys.modules['torchvision.transforms'] = MockModule()

import torch
import torch.nn.functional as F
# Mock the specific torch operator that is failing
if not hasattr(torch.ops, 'torchvision'):
    class DummyOps:
        def nms(*args, **kwargs): return torch.tensor([])
    torch.ops.torchvision = DummyOps()

from transformers import AutoTokenizer, AutoModel

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.append(project_root)

def calculate_token_entropy(model, input_ids, attention_mask):
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        probs = F.softmax(logits, dim=-1)
        # Use simple entropy
        entropy = -torch.sum(probs * torch.log(probs + 1e-9), dim=-1)
    return entropy

def identify_bad_names(entropy, top_k_percent=0.15):
    ent = entropy.squeeze(0)
    num_to_mask = int(len(ent) * top_k_percent)
    if num_to_mask < 1: num_to_mask = 1
    _, indices = torch.topk(ent, k=num_to_mask)
    return indices

def run_refactor_showcase(code_snippet):
    model_id = "Dream-org/Dream-Coder-v0-Instruct-7B"
    print(f"Loading {model_id} (Denoising Expert)...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_id, torch_dtype=torch.bfloat16, trust_remote_code=True).to("cuda").eval()
    
    inputs = tokenizer(code_snippet, return_tensors="pt").to("cuda")
    input_ids = inputs.input_ids
    attention_mask = inputs.attention_mask
    
    # Identify Noise
    entropy = calculate_token_entropy(model, input_ids, attention_mask.bool())
    mask_indices = identify_bad_names(entropy)
    
    masked_input_ids = input_ids.clone()
    mask_token_id = 126336 # Standard for LLaDA base
    if tokenizer.mask_token_id is not None:
        mask_token_id = tokenizer.mask_token_id

    # Apply masks to 'noisy' tokens
    for idx in mask_indices:
        masked_input_ids[0, idx] = mask_token_id

    print("Noise identification complete. Starting Diffusion Denoising...")
    
    with torch.no_grad():
        # Fill the masks
        refactored_ids = model.diffusion_generate(
            masked_input_ids,
            attention_mask=attention_mask.bool(),
            steps=128,
            temperature=0.1,
            top_p=0.95,
            alg="entropy"
        )
    
    final_ids = refactored_ids.sequences if hasattr(refactored_ids, "sequences") else refactored_ids
    refactored_code = tokenizer.decode(final_ids[0], skip_special_tokens=True).strip()
    
    print("\n" + "="*30)
    print("ORIGINAL (Noisy Code):")
    print(code_snippet.strip())
    print("\nREFACTORED (Denoised Code):")
    print(refactored_code)
    print("="*30)

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
    try:
        run_refactor_showcase(bad_code)
    except Exception as e:
        import traceback
        traceback.print_exc()
