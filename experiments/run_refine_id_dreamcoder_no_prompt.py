import os
import torch
import sys
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
import time
from datetime import datetime
import re

# --- Environment Setting: Mock torchvision ---
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

# Configuration
DATA_PATH = "data/test_filtered_1024.csv"
RESULTS_DIR = "results"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Model configuration for DreamCoder (No-Prompt Testing)
# Note: Using the Instruct model but in base/no-prompt mode (direct in-filling)
MODEL_NAME = "Dream-org/Dream-Coder-v0-Instruct-7B"
MASK_TOKEN = "<|mask|>"
NUM_MASK_TOKENS = 4  # Use 4 mask tokens to replace [MASK]

def extract_prediction(full_code, masked_code):
    """
    More robust extraction for multi-token identifiers.
    Finds the content between anchors and takes the first identifier found.
    """
    parts = masked_code.split("[MASK]")
    if len(parts) < 2: return "ERROR_MASK_NOT_FOUND"
    
    pre, post = parts[0], parts[1]
    
    # Use shorter anchors for higher match probability
    pre_anchor = pre.strip()[-30:] if len(pre.strip()) > 30 else pre.strip()
    post_anchor = post.strip()[:30] if len(post.strip()) > 30 else post.strip()
    
    if not pre_anchor and not post_anchor:
        match = re.search(r'[a-zA-Z_][a-zA-Z0-9_]*', full_code)
        return match.group(0) if match else ""

    # Locate the gap
    idx_start = full_code.find(pre_anchor)
    if idx_start != -1:
        idx_start += len(pre_anchor)
    else:
        idx_start = 0

    idx_end = full_code.find(post_anchor, idx_start)
    
    if idx_end != -1:
        gap_content = full_code[idx_start:idx_end].strip()
    else:
        gap_content = full_code[idx_start : idx_start + 60].strip()

    # Extract the first valid identifier word
    match = re.search(r'[a-zA-Z_][a-zA-Z0-9_]*', gap_content)
    if match:
        return match.group(0)
        
    return "EXTRACTION_FAILED"

def run_experiment():
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)

    print(f"Loading data from {DATA_PATH}...")
    try:
        # Assuming CSV format: id, masked_code, target
        df = pd.read_csv(DATA_PATH, header=None, names=['id', 'masked_code', 'target'])
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return

    print(f"\n{'='*50}")
    print(f"Running No-Prompt Experiment for: DreamCoder")
    print(f"Model ID: {MODEL_NAME}")
    print(f"Mode: Native Continuous In-filling")
    print(f"Mask Tokens: {NUM_MASK_TOKENS} (Continuous)")
    print(f"{'='*50}")

    try:
        # Load Model and Tokenizer
        print("Loading tokenizer and model...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
        model = AutoModel.from_pretrained(
            MODEL_NAME, 
            torch_dtype=torch.bfloat16 if DEVICE == "cuda" else torch.float32,
            trust_remote_code=True
        )
        model = model.to(DEVICE)
        model.eval()
        
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    results = []
    
    # Process samples
    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Testing DreamCoder Base/NoPrompt"):
        item_id = row['id']
        masked_code = str(row['masked_code'])
        ground_truth = str(row['target']).strip()

        try:
            # CRITICAL: DreamCoder often expects [MASK] or <|mask|> depending on the task.
            # For no-prompt in-filling, we use the continuous mask substitution.
            multi_mask = MASK_TOKEN * NUM_MASK_TOKENS
            input_code = masked_code.replace("[MASK]", multi_mask)
            
            # Tokenize input
            inputs = tokenizer(input_code, return_tensors="pt")
            input_ids = inputs.input_ids.to(model.device)
            attention_mask = inputs.attention_mask.to(model.device)
            
            # Run diffusion generation
            # DreamCoder might benefit from more steps (e.g. 128-256) compared to DiffuCoder.
            # But we'll start with a reasonable value.
            GEN_STEPS = 64
            
            with torch.no_grad():
                output = model.diffusion_generate(
                    input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=1, 
                    steps=GEN_STEPS,
                    temperature=0.3, # DreamCoder often uses lower temp like 0.1, but 0.3 is standard
                    top_p=0.95,
                    alg="entropy",
                    alg_temp=0.
                )
            
            # Decode the full denoised sequence
            generated_ids = output.sequences[0] if hasattr(output, "sequences") else output[0]
            full_code = tokenizer.decode(generated_ids, skip_special_tokens=True)
            
            # Extract the prediction from the denoised code
            prediction = extract_prediction(full_code, masked_code)
            
            results.append({
                "id": item_id,
                "ground_truth": ground_truth,
                "prediction": prediction,
                "full_code": full_code,
                "correct": (prediction == ground_truth)
            })

            # Debug printing for the first sample
            if len(results) == 1:
                print(f"\n--- Sample {item_id} Debug ---")
                print(f"Ground Truth: {ground_truth}")
                print(f"Prediction:   {prediction}")
                print(f"Match:        {prediction == ground_truth}")
                print(f"--- End Debug ---\n")

        except Exception as e:
            print(f"Error on sample {item_id}: {e}")
            results.append({"id": item_id, "error": str(e)})

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(RESULTS_DIR, f"DreamCoder_noPrompt_{timestamp}.csv")
    pd.DataFrame(results).to_csv(output_file, index=False)
    
    accuracy = sum(1 for r in results if r.get('correct', False)) / len(results) if results else 0
    print(f"\nResults saved to {output_file}")
    print(f"Accuracy: {accuracy:.2%}")

    # Cleanup
    del model
    del tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    import gc
    gc.collect()

if __name__ == "__main__":
    run_experiment()
