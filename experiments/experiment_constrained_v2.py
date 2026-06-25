
import sys
import os
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from datetime import datetime
import re
from extract_names import extract_python_identifiers

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

def get_token_mask_from_identifiers(code, tokenizer, input_ids):
    """
    Maps character-level identifiers to token-level mask.
    Specifically handles the case where tokenizers merge symbols (like '(b' or '[e').
    """
    identifiers = extract_python_identifiers(code)
    
    # We need the character offsets for each identifier.
    # extract_python_identifiers only gives line/col now, let's modify it or calculate offsets.
    # Actually, a better way is to use the same ASTtokens logic inside or update extract_names.
    # But for now, let's calculate byte offsets from line/col for simple cases
    lines = code.split('\n')
    ident_ranges = []
    for item in identifiers:
        # line is 1-indexed
        line_idx = item['line'] - 1
        col_idx = item['col']
        # Find start offset
        start_offset = sum(len(l) + 1 for l in lines[:line_idx]) + col_idx
        end_offset = start_offset + len(item['name'])
        ident_ranges.append((start_offset, end_offset))

    # Tokenize with added special tokens (Full sequence)
    tokens_decoded = [tokenizer.decode([tid]) for tid in input_ids]
    is_ident_token = torch.zeros(len(input_ids), dtype=torch.bool)
    
    current_char_pos = 0
    print("\n--- Token Alignment Debug ---")
    
    # Regex to check if a token contains syntax symbols that should NOT be masked
    SYNTAX_GUARD = r"[()\[\]:,\.\s\n\+\-\*/=<>!]"

    for i, tok_text in enumerate(tokens_decoded):
        if not tok_text or tok_text in ["<s>", "</s>", "<unk>"]:
            continue
            
        clean_tok = tok_text.replace(' ', ' ')
        
        # We need to find this token in the code. 
        # Tokenizers often strip leading/trailing whitespace or handle it specially.
        # This search logic needs to be robust.
        search_text = clean_tok.strip()
        if not search_text:
            # Token is pure whitespace
            continue

        start_find = code.find(search_text, current_char_pos)
        if start_find != -1:
            tok_start = start_find
            tok_end = start_find + len(search_text)
            current_char_pos = tok_end
            
            # Check overlap with any identifier range
            for i_start, i_end in ident_ranges:
                # If the token is fully or partially inside an identifier range
                if max(tok_start, i_start) < min(tok_end, i_end):
                    # SAFETY CHECK: If this token contains syntax symbols, DO NOT MASK it.
                    # Because masking a token like '(b' will lose the '(' structure.
                    pure_content = clean_tok.lstrip(' ')
                    if not re.search(r"[^a-zA-Z0-9_]", pure_content):
                        is_ident_token[i] = True
                        print(f"Token {i:2}: '{tok_text}' -> MASKED (Matches '{code[i_start:i_end]}')")
                    else:
                        print(f"Token {i:2}: '{tok_text}' -> GUARDED (Identifier merged with syntax)")
                    break
    
    return is_ident_token

def run_constrained_diffusion_v2(model_name, model_id, code_snippet, total_steps=20):
    print(f"\n{'='*20} Constrained Diffusion V2: {model_name} {'='*20}")
    os.makedirs("results", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"results/{model_name}_constrained_v2_{timestamp}.txt"
    
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_id, torch_dtype=torch.bfloat16, trust_remote_code=True).to("cuda").eval()
    
    # Tokenize
    input_ids_full = tokenizer.encode(code_snippet, return_tensors="pt").to("cuda")
    input_ids = input_ids_full[0]
    
    # Get mask
    ident_mask = get_token_mask_from_identifiers(code_snippet, tokenizer, input_ids).to("cuda")
    
    # Initial state for diffusion (we can start from noisy if we want, but user asked for direct input)
    # The user wants "constrained", meaning we only update the identifier tokens.
    x = input_ids_full.clone()
    
    with open(output_filename, "w", encoding="utf-8") as f:
        f.write(f"Model: {model_name}\n")
        f.write(f"Mode: Constrained Diffusion V2 (Identified via extract_names.py)\n")
        f.write(f"Original Code:\n{code_snippet}\n" + "="*40 + "\n\n")
        
        # Step 0: Print initial state (before any updates)
        f.write(f"--- Step 0 (Initial) ---\n")
        f.write(tokenizer.decode(x[0], skip_special_tokens=False) + "\n\n")
        
        # Diffusion Loop
        # In constrained diffusion, we allow the model to suggest tokens for ALL positions,
        # but we ONLY apply the changes to the identifier positions.
        for step in range(total_steps):
            with torch.no_grad():
                logits = model(x).logits
                x_pred = torch.argmax(logits, dim=-1)
                
                # Apply updates ONLY to identifier masks
                x_next = torch.where(ident_mask, x_pred, x)
                
                if torch.equal(x, x_next):
                    f.write(f"Step {step+1}: Converged.\n")
                    break
                
                x = x_next
                decoded = tokenizer.decode(x[0], skip_special_tokens=False)
                
                f.write(f"--- Step {step+1} ---\n{decoded}\n\n")
                if (step + 1) % 5 == 0:
                    print(f"Step {step+1} recorded.")

        f.write("\nFinal Output:\n" + tokenizer.decode(x[0], skip_special_tokens=True))

    print(f"Experiment finished. Results: {output_filename}")

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
    
    # You can change to "apple/DiffuCoder-7B-Instruct" as well
    run_constrained_diffusion_v2("dreamcoder", "Dream-org/Dream-Coder-v0-Instruct-7B", terrible_sort)
