
import sys
import os
import torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoTokenizer, AutoModel

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

def run_nomask_test(model_id, code_snippet, total_steps=10, update_percent=0.1):
    print(f"\n{'='*20} Testing Hypothesis Mini-Loop (No Masks) {'='*20}")
    print(f"Model: {model_id}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_id, torch_dtype=torch.bfloat16, trust_remote_code=True).to("cuda").eval()
    
    inputs = tokenizer(code_snippet, return_tensors="pt").to("cuda")
    x = inputs.input_ids.clone()
    attention_mask = inputs.attention_mask
    
    print("\nOriginal Code:")
    print(tokenizer.decode(x[0], skip_special_tokens=True))
    print("-" * 30)

    for step in range(total_steps):
        with torch.no_grad():
            # Get logits for current tokens
            outputs = model(x, attention_mask=attention_mask.bool())
            logits = outputs.logits # [batch, seq_len, vocab_size]
            
            # 1. Prediction (what model thinks should be there)
            x0 = torch.argmax(logits, dim=-1)
            
            # 2. Confidence of current tokens in x
            probs = F.softmax(logits, dim=-1)
            # Gather probability of the tokens currently in x
            current_probs = torch.gather(probs, dim=-1, index=x.unsqueeze(-1)).squeeze(-1)
            
            # 3. Identify lowest confidence tokens to "diffuse"
            num_to_update = max(1, int(x.size(1) * update_percent))
            # We want to replace tokens where the model is LEAST confident about the CURRENT token
            _, low_conf_indices = torch.topk(-current_probs[0], k=num_to_update)
            
            # 4. Update those tokens with model's prediction
            x_new = x.clone()
            x_new[0, low_conf_indices] = x0[0, low_conf_indices]
            
            # Check if anything changed
            if torch.equal(x, x_new):
                print(f"Step {step+1}: No more changes (Converged).")
                break
            
            x = x_new
            
            print(f"Step {step+1} (Updated {num_to_update} low-confidence tokens):")
            print(tokenizer.decode(x[0], skip_special_tokens=True))
            print("-" * 20)

    print("\nFinal Result:")
    print(tokenizer.decode(x[0], skip_special_tokens=True))

if __name__ == "__main__":
    terrible_sort = """
def a(b):
    c = len(b)
    for d in range(c):
        for e in range(0, c - d - 1):
            if b[e] > b[e + 1]:
                f = b[e]
                b[e] = b[e + 1]
                b[e + 1] = f
    return b
"""
    # Running for DiffuCoder
    run_nomask_test("apple/DiffuCoder-7B-Instruct", terrible_sort)
