
import sys
import os
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModel
from datetime import datetime

# --- Mock torchvision for compatibility with certain model code ---
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
# -----------------------------------------------------------------

def run_experiment():
    model_id = "apple/DiffuCoder-7B-Instruct"
    print(f"Loading model and tokenizer: {model_id}...")
    
    # Load model and tokenizer
    # Using trust_remote_code=True as required by DiffuCoder
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_id, torch_dtype=torch.bfloat16, trust_remote_code=True).to("cuda").eval()

    # Read the first 10 data points from test.csv
    csv_path = os.path.join('data', 'test.csv')
    print(f"Reading data from {csv_path}...")
    try:
        # Based on run_inference.py, the columns are [id, X, y]
        df = pd.read_csv(csv_path, header=None, names=['id', 'X', 'y'], nrows=10)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Mask token for DiffuCoder
    mask_token = '<|mask|>'
    
    for i, row in df.iterrows():
        print(f"Processing data point {i} (id: {row['id']})...")

        input_text = row['X']
        noisy_text = input_text.replace('[MASK]', '<|mask|>')
        
        inference_text = noisy_text.replace('<|mask|>', mask_token)
        
        # Tokenize
        # Use a large max_length to avoid truncation as requested.
        # diffucoder-7bcontext is usually 2048 or 4096.
        inputs = tokenizer(inference_text, return_tensors="pt", truncation=False)
        input_ids = inputs.input_ids.to("cuda")
        attention_mask = inputs.attention_mask.to("cuda")
        
        # Check original length
        # original_length = input_ids.shape[1]
        
        print(f"  Input sequence length: {input_ids.shape[1]} tokens")
        
        # Diffusion generation
        # Use only max_length instead of max_new_tokens=0 to avoid the preference-based 
        # recalculation that triggers the ValueError.
        with torch.no_grad():
            output = model.diffusion_generate(
                input_ids,
                attention_mask=attention_mask,
                max_length=input_ids.shape[1] + 1,
                steps=256,
                temperature=0.3,
                top_p=0.95,
                alg="entropy",
                alg_temp=0.,
            )
            
        # Decode the full sequence
        # We want the complete input code (denoised)
        seqs = output.sequences if hasattr(output, "sequences") else output
        decoded_output = tokenizer.decode(seqs[0], skip_special_tokens=True)
        
        # Prepare output filename: data_序号_日期_时间.py
        output_filename = f"data_{i}_{timestamp}.py"
        
        # Save to file
        with open(output_filename, "w", encoding="utf-8") as f:
            f.write(decoded_output)
            
        print(f"  Result saved to {output_filename}")

    print("\nExperiment completed successfully.")

if __name__ == "__main__":
    run_experiment()
