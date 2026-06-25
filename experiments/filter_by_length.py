import pandas as pd
from transformers import AutoTokenizer
from tqdm import tqdm
import os

def filter_dataset_by_tokens(input_csv, output_csv, model_id, max_tokens=1024):
    print(f"Loading tokenizer for {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    
    print(f"Reading dataset: {input_csv}")
    # Read the dataset (no headers according to previous scripts)
    df = pd.read_csv(input_csv, header=None, names=['id', 'X', 'y'])
    
    filtered_rows = []
    total_count = len(df)
    
    print(f"Filtering {total_count} rows...")
    
    for idx, row in tqdm(df.iterrows(), total=total_count, desc="Calculating Token Lengths"):
        X_text = str(row['X'])
        
        # Calculate token length
        tokens = tokenizer.encode(X_text, add_special_tokens=False)
        
        if len(tokens) <= max_tokens:
            filtered_rows.append(row)
            
    # Create new dataframe
    filtered_df = pd.DataFrame(filtered_rows)
    
    # Save to new CSV
    filtered_df.to_csv(output_csv, index=False, header=False)
    
    print("\n" + "="*50)
    print(f"Filtering Complete!")
    print(f"Original Count: {total_count}")
    print(f"Filtered Count (<= {max_tokens} tokens): {len(filtered_df)}")
    print(f"New dataset saved to: {output_csv}")
    print("="*50)

if __name__ == "__main__":
    # Settings
    INPUT_FILE = '/path/to/CoReFusion/data/test.csv'
    OUTPUT_FILE = '/path/to/CoReFusion/data/test_filtered_1024.csv'
    
    # Using DreamCoder tokenizer as default (or DiffuCoder, both should be similar for length)
    MODEL_ID = "Dream-org/Dream-Coder-v0-Instruct-7B"
    
    filter_dataset_by_tokens(INPUT_FILE, OUTPUT_FILE, MODEL_ID, max_tokens=1024)
