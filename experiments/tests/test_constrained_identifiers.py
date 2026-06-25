
import sys
import os
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from datetime import datetime
import ast
from asttokens import ASTTokens

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

def get_manual_offsets(tokenizer, code):
    """
    Fallback for tokenizers that don't support return_offsets_mapping.
    This is an approximation for piece-wise tokenization.
    """
    tokens = tokenizer.convert_ids_to_tokens(tokenizer.encode(code, add_special_tokens=False))
    offsets = []
    current_pos = 0
    
    for token in tokens:
        # Llama-style tokenizers often use ' ' (representative of space)
        clean_token = token.replace(' ', ' ').replace('<0x0A>', '\n')
        # This is a simplified heuristic: find the token in the string starting from current_pos
        # Note: This might not be 100% perfect for all edge cases but works for most code.
        found_pos = code.find(clean_token.strip() if clean_token.strip() else clean_token, current_pos)
        if found_pos != -1:
            start = found_pos
            end = found_pos + len(clean_token.strip() if clean_token.strip() else clean_token)
            offsets.append((start, end))
            current_pos = end
        else:
            offsets.append((0, 0))
    return offsets

def get_identifier_token_mask(code, tokenizer, input_ids):
    """
    Uses Python's built-in AST + ASTTokens to find identifier byte ranges.
    Handles both Fast and Slow Tokenizers.
    """
    try:
        atok = ASTTokens(code, parse=True)
    except Exception as e:
        print(f"AST Parsing Error: {e}")
        return torch.zeros(input_ids.size(1), dtype=torch.bool)
    
    ident_ranges = []
    for node in ast.walk(atok.tree):
        if isinstance(node, (ast.Name, ast.arg)):
            start, end = atok.get_text_range(node)
            ident_ranges.append((start, end))

    # Try to get offset mapping
    is_identifier_token = torch.zeros(input_ids.size(1), dtype=torch.bool)
    
    try:
        # Try Fast Tokenizer feature
        encoding = tokenizer(code, return_offsets_mapping=True, return_tensors="pt", add_special_tokens=True)
        offset_mapping = encoding.offset_mapping[0]
    except NotImplementedError:
        print("Fast Tokenizer not available. Using manual offset mapping heuristic...")
        # Fallback logic
        raw_offsets = get_manual_offsets(tokenizer, code)
        # Pad with 0,0 for special tokens like BOS/EOS if present
        # Most slow tokenizers add BOS at 0.
        input_ids_list = input_ids[0].tolist()
        # We need to align raw_offsets with input_ids_list
        # This is a rough alignment:
        offset_mapping = [(0, 0)] * len(input_ids_list)
        # Assuming BOS/EOS are the only special tokens
        enc_full = tokenizer.encode(code, add_special_tokens=True)
        enc_no_special = tokenizer.encode(code, add_special_tokens=False)
        
        offset_idx = 0
        for i, idx in enumerate(enc_full):
            if idx in enc_no_special and offset_idx < len(raw_offsets):
                offset_mapping[i] = raw_offsets[offset_idx]
                offset_idx += 1

    for i, (start, end) in enumerate(offset_mapping):
        if start == 0 and end == 0: continue
        for i_start, i_end in ident_ranges:
            if max(start, i_start) < min(end, i_end):
                is_identifier_token[i] = True
                break
                
    return is_identifier_token

def run_constrained_diffusion(model_name, model_id, code_snippet, total_steps=20):
    print(f"\n{'='*20} Constrained Diffusion: {model_name} {'='*20}")
    os.makedirs("results", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"results/{model_name}_ast_constrained_{timestamp}.txt"
    
    # Try to use fast tokenizer if available
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, use_fast=True)
    model = AutoModel.from_pretrained(model_id, torch_dtype=torch.bfloat16, trust_remote_code=True).to("cuda").eval()
    
    inputs = tokenizer(code_snippet, return_tensors="pt").to("cuda")
    x = inputs.input_ids.clone()
    attention_mask = inputs.attention_mask
    
    print("Extracting identifier locations...")
    ident_mask = get_identifier_token_mask(code_snippet, tokenizer, x).to("cuda")
    
    num_idents = ident_mask.sum().item()
    print(f"Total tokens identified as identifiers: {num_idents} / {x.size(1)}")

    with open(output_filename, "w", encoding="utf-8") as f:
        f.write(f"Model: {model_name}\nOriginal Code:\n{code_snippet}\n" + "="*40 + "\n\n")
        
        for step in range(total_steps):
            with torch.no_grad():
                logits = model(x, attention_mask=attention_mask.bool()).logits
                x_pred = torch.argmax(logits, dim=-1)
                
                # Constrain update
                x_next = torch.where(ident_mask, x_pred, x)
                
                if torch.equal(x, x_next):
                    f.write(f"Step {step+1}: Converged.\n")
                    break
                
                x = x_next
                decoded = tokenizer.decode(x[0], skip_special_tokens=True)
                
                f.write(f"--- Step {step+1} ---\n{decoded}\n\n")
                if (step + 1) % 5 == 0:
                    print(f"Step {step+1} recorded.")

        f.write("\nFinal Output:\n" + tokenizer.decode(x[0], skip_special_tokens=True))

    print(f"Finished. Results: {output_filename}")

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
    run_constrained_diffusion("dreamcoder", "Dream-org/Dream-Coder-v0-Instruct-7B", terrible_sort)
