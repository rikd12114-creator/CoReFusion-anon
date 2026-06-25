
import sys
import os
import torch
from transformers import AutoTokenizer, AutoModel
from datetime import datetime

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

def run_direct_diffusion_test(model_name, model_id, code_snippet, total_steps=20):
    print(f"\nProcessing {model_name}...")
    
    # Setup results directory
    os.makedirs("results", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"results/{model_name}_direct_test_{timestamp}.txt"
    
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_id, torch_dtype=torch.bfloat16, trust_remote_code=True).to("cuda").eval()
    
    # Tokenize input (No Mask Tokens)
    inputs = tokenizer(code_snippet, return_tensors="pt").to("cuda")
    x = inputs.input_ids.clone()
    attention_mask = inputs.attention_mask
    
    with open(output_filename, "w", encoding="utf-8") as f:
        f.write(f"Model: {model_name} ({model_id})\n")
        f.write(f"Original Code:\n{code_snippet}\n")
        f.write("="*50 + "\n\n")
        
        # Iterative Denoising Loop
        for step in range(total_steps):
            with torch.no_grad():
                # Direct prediction from current input (even if not masked)
                logits = model(x, attention_mask=attention_mask.bool()).logits
                
                # Update ALL tokens based on model's top prediction for each position
                x_next = torch.argmax(logits, dim=-1)
                
                # Check for convergence
                if torch.equal(x, x_next):
                    f.write(f"Step {step+1}: Converged (No further changes).\n")
                    break
                
                x = x_next
                decoded = tokenizer.decode(x[0], skip_special_tokens=True)
                
                # Log step results
                f.write(f"Step {step+1}:\n{decoded}\n")
                f.write("-" * 30 + "\n")
                
                if (step + 1) % 5 == 0:
                    print(f"Step {step+1}/{total_steps} recorded.")

        f.write("\nFinal Output:\n" + tokenizer.decode(x[0], skip_special_tokens=True))
    
    print(f"Results saved to: {output_filename}")

if __name__ == "__main__":
    terrible_sort = """def a(b):
    c = len(b)
    for d in range(c):
        for e in range(0, c - d - 1):
            if b[e] > b[e + 1]:
                f = b[e]
                b[e] = b[e + 1]
                b[e + 1] = f
    return b"""

    # Choose one to run or iterate
    # You can change to DiffuCoder: "apple/DiffuCoder-7B-Instruct"
    run_direct_diffusion_test("dreamcoder", "Dream-org/Dream-Coder-v0-Instruct-7B", terrible_sort)
    
    # Optional cleanup to run the next model
    import gc
    torch.cuda.empty_cache()
    gc.collect()
    
    # run_direct_diffusion_test("diffucoder", "apple/DiffuCoder-7B-Instruct", terrible_sort)
