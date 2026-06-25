import sys
import os
import json
from transformers import AutoTokenizer
import collections

def check_token_adhesion(model_id):
    print(f"\n{'='*20} Checking {model_id} {'='*20}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    except Exception as e:
        print(f"Error loading tokenizer for {model_id}: {e}")
        return None

    vocab = tokenizer.get_vocab()
    print(f"Vocabulary size: {len(vocab)}")

    sorted_vocab = sorted(vocab.items(), key=lambda x: x[1])

    glued_tokens = []
    symbol_counts = collections.defaultdict(int)
    
    # Symbols set
    symbols = set("()[]{}.:;,+=-*/&|^%<>!~@#$")
    
    for token_str, token_id in sorted_vocab:
        if not isinstance(token_str, str):
            continue
            
        # skip special tokens
        if token_str.startswith('<') and token_str.endswith('>'):
            continue
        if token_str.startswith('[') and token_str.endswith(']'):
             if token_str[1:-1].isupper():
                continue

        # Clean up token string for display/check if possible
        clean_token = token_str.lstrip('Ġ ')
        
        has_alpha = any(c.isalnum() for c in clean_token)
        has_symbol = any(c in symbols for c in clean_token)
        
        if has_alpha and has_symbol:
            glued_tokens.append(token_str)
            # Count which symbols are present
            for char in clean_token:
                if char in symbols:
                    symbol_counts[char] += 1

    print(f"Found {len(glued_tokens)} potential 'glued' tokens.")
    
    return {
        "model_id": model_id,
        "vocab_size": len(vocab),
        "glued_count": len(glued_tokens),
        "symbol_counts": symbol_counts,
        "glued_tokens_sample": glued_tokens[:100] # Save a sample
    }


if __name__ == "__main__":
    models = [
        "apple/DiffuCoder-7B-Instruct",
        "Dream-org/Dream-Coder-v0-Instruct-7B"
    ]
    
    results = []
    for m in models:
        res = check_token_adhesion(m)
        if res:
            results.append(res)
            
    output_file = os.path.join(os.path.dirname(__file__), "token_adhesion_results.json")
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_file}")
