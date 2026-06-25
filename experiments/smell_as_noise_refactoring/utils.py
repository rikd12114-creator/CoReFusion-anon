import sys
import torch
import pandas as pd
import os
import re
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

MODELS = {
    "diffucoder": {
        "id": "apple/DiffuCoder-7B-Instruct",
        "mask_token": "<|mask|>",
    },
    "dreamcoder": {
        "id": "Dream-org/Dream-Coder-v0-Instruct-7B",
        "mask_token": "<|mask|>",
    }
}

BAD_NAMES = ["x", "temp", "data", "obj", "var", "foo", "bar", "val"]

def load_model(model_name="diffucoder"):
    if model_name not in MODELS:
        raise ValueError(f"Unknown model: {model_name}")
    
    cfg = MODELS[model_name]
    print(f"Loading {model_name} ({cfg['id']})...")
    
    # Determine device
    if torch.cuda.is_available():
        device = "cuda"
        dtype = torch.bfloat16
    elif torch.backends.mps.is_available():
        device = "mps"
        dtype = torch.float16 # MPS often prefers float16
    else:
        device = "cpu"
        dtype = torch.float32 # CPU might not support half precision well
        
    print(f"Using device: {device}")
    
    tokenizer = AutoTokenizer.from_pretrained(cfg['id'], trust_remote_code=True)
    model = AutoModel.from_pretrained(cfg['id'], torch_dtype=dtype, trust_remote_code=True).to(device).eval()
    return model, tokenizer, cfg

def load_data(path, limit=None):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Data file not found: {path}")
    
    df = pd.read_csv(path, header=None, names=['id', 'X', 'y'])
    if limit:
        df = df.head(limit)
    return df

def find_token_indices(tokenizer, input_ids, target_str):
    """
    Find the indices of the tokens corresponding to target_str in input_ids.
    Returns a list of (start, end) tuples.
    
    Robust strategy:
    1. Decode all tokens individually.
    2. Search for a contiguous sequence of tokens that reconstructs target_str.
    """
    if not target_str:
        return []
        
    input_list = input_ids.tolist() if torch.is_tensor(input_ids) else input_ids
    
    # Decode each token individually to see what characters they contain
    # Note: We use a special function to handle the Llama tokenizer's SPIECE_UNDERLINE ( )
    token_texts = [tokenizer.decode([tid], skip_special_tokens=False) for tid in input_list]
    
    indices = []
    n = len(token_texts)
    
    # We search for a subsequence token_texts[i:j] such that "".join(token_texts[i:j]).replace(" ", "") == target_str
    # But wait, exact reconstruction is tricky because of spaces.
    # The target_str usually does NOT contain spaces (it's a variable name).
    # But tokens might contain leading spaces.
    
    # Optimization: Only start checking if the token contains the start of target_str
    # This is O(N^2) in worst case but N is small (1024) and max token length for a var is small (~5).
    
    target_clean = target_str.strip()
    
    for i in range(n):
        # Quick check: first char match
        # The token text might start with ' ' (U+2581)
        curr_text = token_texts[i].lstrip(' ') # Remove leading SPIECE_UNDERLINE
        if not curr_text: continue
        
        # If the token doesn't even contain the start of our target (or isn't contained in it), skip
        # Actually, a token like "get" might be the start of "getValue".
        
        current_reconstruction = ""
        for j in range(i, min(i + 10, n)): # Assume variable name is not split into more than 10 tokens
            # Add current token text
            # We need to be careful about spaces. 
            # If we are reconstructing a variable name, it should NOT have internal spaces.
            # Llama tokens: " int", " my", "Var" -> " int", " myVar"
            
            part = token_texts[j]
            # If it's the first token in our sequence, we ignore leading space
            # If it's a subsequent token, it should NOT have a leading space if it's part of the same word.
            # E.g. "my" + "Var". "Var" usually doesn't have space if it's "myVar".
            # BUT: " decoded" + "Capacity". "Capacity" usually has no space.
            
            if j == i:
                part_clean = part.lstrip(' ')
            else:
                if part.startswith(' '): 
                    # If a subsequent token starts with space, it usually means a new word started.
                    # So we should probably stop, UNLESS target_str actually has spaces (unlikely for var name).
                    break
                part_clean = part
            
            current_reconstruction += part_clean
            
            if current_reconstruction == target_clean:
                indices.append((i, j + 1))
                break
            
            # Optimization: if reconstruction is already not a prefix of target, stop
            if not target_clean.startswith(current_reconstruction):
                break
                
    return sorted(list(set(indices)))
