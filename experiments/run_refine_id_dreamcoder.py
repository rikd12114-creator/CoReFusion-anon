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

# Model configuration for DreamCoder
MODEL_NAME = "Dream-org/Dream-Coder-v0-Instruct-7B"
MASK_TOKEN = "<|mask|>"
NUM_MASK_TOKENS = 4  # Use 4 mask tokens to replace [MASK]

def clean_prediction(text):
    """Extracts a clean identifier from model output."""
    # Remove whitespace and newlines
    text = text.strip().split('\n')[0].strip('`"\' ')
    # Match first valid Java identifier found
    match = re.search(r'[a-zA-Z_][a-zA-Z0-9_]*', text)
    if match:
        return match.group(0)
    return text

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
    print(f"Running Experiment for: DreamCoder")
    print(f"Model ID: {MODEL_NAME}")
    print(f"Using {NUM_MASK_TOKENS} mask tokens to replace [MASK]")
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
        if DEVICE == "cuda":
            model = model.to("cuda")
        else:
            model = model.to("cpu")
        model.eval()
        
        # Get mask token id
        mask_token_id = tokenizer.convert_tokens_to_ids(MASK_TOKEN)
        print(f"Mask token: {MASK_TOKEN}, ID: {mask_token_id}")
        
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    results = []
    
    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Testing DreamCoder"):
        item_id = row['id']
        masked_code = str(row['masked_code'])
        ground_truth = str(row['target']).strip()

        try:
            # Replace [MASK] with 4 mask tokens
            # Create the input with 4 mask tokens instead of [MASK]
            multi_mask = " ".join([MASK_TOKEN] * NUM_MASK_TOKENS)
            input_code = masked_code.replace("[MASK]", multi_mask)
            
            # Tokenize input
            inputs = tokenizer(input_code, return_tensors="pt")
            input_ids = inputs.input_ids.to(model.device)
            attention_mask = inputs.attention_mask.to(model.device)
            
            # DreamCoder parameters following official API
            TOKEN_PER_STEP = 1
            MAX_NEW_TOKENS = 20  # Limit new tokens for identifier prediction
            
            # Run diffusion generation (following official DreamCoder API)
            with torch.no_grad():
                output = model.diffusion_generate(
                    input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=MAX_NEW_TOKENS,
                    steps=MAX_NEW_TOKENS // TOKEN_PER_STEP,
                    output_history=False,
                    return_dict_in_generate=True,
                    temperature=0.1,
                    top_p=0.95,
                    alg="entropy",
                    alg_temp=0.
                )
            
            # Decode the generated sequence (following official output processing)
            generated_ids = output.sequences[0]
            # Decode only the new tokens (skip input)
            generated_text = tokenizer.decode(
                generated_ids[len(input_ids[0]):].tolist()
            )
            # Remove potential padding tokens
            if '<|dlm_pad|>' in generated_text:
                generated_text = generated_text.split('<|dlm_pad|>')[0]
            
            # Extract the filled identifier
            # The generated_text should contain the filled identifier
            prediction = clean_prediction(generated_text)
            
            # Reconstruct full code with prediction
            full_code = masked_code.replace("[MASK]", prediction)
            
            results.append({
                "id": item_id,
                "ground_truth": ground_truth,
                "prediction": prediction,
                "full_code": full_code,
                "correct": (prediction == ground_truth)
            })

        except Exception as e:
            print(f"Error on sample {item_id}: {e}")
            results.append({"id": item_id, "error": str(e)})

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(RESULTS_DIR, f"DreamCoder_refineID_results_{timestamp}.csv")
    pd.DataFrame(results).to_csv(output_file, index=False)
    
    accuracy = sum(1 for r in results if r.get('correct', False)) / len(results) if results else 0
    print(f"\nResults saved to {output_file}")
    print(f"Accuracy: {accuracy:.2%}")

    # Cleanup memory
    del model
    del tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    import gc
    gc.collect()

if __name__ == "__main__":
    run_experiment()
