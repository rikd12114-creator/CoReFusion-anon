import os
import csv
import json
import argparse
import pandas as pd
from tqdm import tqdm

try:
    from openai import OpenAI
except ImportError:
    print("Please install the openai package: pip install openai")
    exit(1)

def load_test_data(test_csv_path):
    """
    Load the test dataset which contains the full code context.
    Returns a dictionary mapping ID to the full code.
    """
    print(f"Loading context data from {test_csv_path}...")
    # pandas is typically faster, but some CSVs might be large.
    # Read the id and the code column. Assuming the columns are 'id', 'full_code', 'ground_truth'
    # Actually, looking at the file it seems 'id', 'code' or similar.
    df = pd.read_csv(test_csv_path, usecols=[0, 1], header=None, names=['id', 'code'])
    df['id'] = df['id'].astype(int)
    id_to_code = dict(zip(df['id'], df['code']))
    return id_to_code

def evaluate_with_llm(client, model, code_context, ground_truth, prediction):
    prompt = f"""You are an expert Java software engineer. I am evaluating an AI model that predicts variable names.

Below is a snippet of Java code where a specific identifier has been replaced with `[MASK]`.
{'-'*40}
{code_context[:3000]} # Truncated to avoid context limits if too long
{'-'*40}

The original developer named it (Ground Truth): `{ground_truth}`
The AI model predicted: `{prediction}`

Your task is to act as a judge and determine if the AI's prediction is an acceptable and good name in this context. 
A name is acceptable if it is semantically correct, adheres to Java naming conventions, and makes sense in the context, even if it is not an exact match to the ground truth.
If the prediction is completely generic (like `x`, `tmp`, `val`) when a specific name is required, or if it changes the semantic meaning incorrectly, it should be rejected.

Please provide a brief reasoning, and a final boolean verdict.
Respond strictly in the following JSON format:
{{
    "reasoning": "your explanation here",
    "is_acceptable": true/false
}}
"""
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful and strict code review assistant that outputs JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,
            response_format={ "type": "json_object" }
        )
        content = response.choices[0].message.content.strip()
        if content.startswith("```json"):
            content = content[7:]
        if content.startswith("```"):
            content = content[3:]
        if content.endswith("```"):
            content = content[:-3]
        result = json.loads(content)
        return result
    except Exception as e:
        print(f"Error calling LLM: {e}")
        return {"reasoning": str(e), "is_acceptable": False}

def process_file(file_path, id_to_code, client, model, output_dir, limit=None):
    print(f"Processing {file_path}...")
    df = pd.read_csv(file_path)
    
    if limit:
        df = df.head(limit)
        
    results = []
    
    # We will iterate row by row
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Evaluating"):
        task_id = int(row['id'])
        gt = row['ground_truth']
        pred = row['prediction']
        
        # If prediction is NaN or not string
        if pd.isna(pred):
            pred = ""
            
        exact_match = (str(gt) == str(pred))
        
        # We can optimize by skipping exact matches if we assume exact match is always acceptable
        if exact_match:
            is_acceptable = True
            reasoning = "Exact match."
        else:
            context = id_to_code.get(task_id, "[Context Missing]")
            llm_res = evaluate_with_llm(client, model, context, gt, pred)
            is_acceptable = llm_res.get('is_acceptable', False)
            reasoning = llm_res.get('reasoning', '')
            
        results.append({
            'id': task_id,
            'ground_truth': gt,
            'prediction': pred,
            'exact_match': exact_match,
            'llm_acceptable': is_acceptable,
            'llm_reasoning': reasoning
        })
        
    out_df = pd.DataFrame(results)
    base_name = os.path.basename(file_path)
    out_path = os.path.join(output_dir, f"judged_{base_name}")
    out_df.to_csv(out_path, index=False)
    
    acc_exact = out_df['exact_match'].mean() * 100
    acc_llm = out_df['llm_acceptable'].mean() * 100
    print(f"Results for {base_name}: Exact Match = {acc_exact:.2f}%, LLM Acceptable = {acc_llm:.2f}%")
    print(f"Saved judged file to {out_path}\n")

def main():
    parser = argparse.ArgumentParser(description="LLM as a Judge for Code Naming")
    parser.add_argument("--test-data", default="/path/to/CoReFusion/data/test.csv", help="Path to test.csv containing context")
    parser.add_argument("--diffusion-dir", default="/path/to/CoReFusion/data/benchmark_ReFineID_Diffusion/diffusion_benchmark")
    parser.add_argument("--fim-dir", default="/path/to/CoReFusion/data/benchmark_ReFineID_FIM/ar_fim_benchmark")
    parser.add_argument("--output-dir", default="/path/to/CoReFusion/detector/llm_judgement_results")
    parser.add_argument("--model", default="gpt-4o-mini", help="OpenAI model to use")
    parser.add_argument("--api-key", required=False, help="OpenAI API key (or set OPENAI_API_KEY env var)")
    parser.add_argument("--base-url", required=False, help="Custom base URL for the API (e.g., http://localhost:8000/v1)")
    parser.add_argument("--limit", type=int, default=None, help="Number of rows to evaluate per file (for cost/time saving)")
    
    args = parser.parse_args()
    
    api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Please provide an OpenAI API key via --api-key or OPENAI_API_KEY environment variable.")
        return
        
    client = OpenAI(api_key=api_key, base_url=args.base_url)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    id_to_code = load_test_data(args.test_data)
    
    targets = []
    
    # Collect files from diffusion dir
    if os.path.exists(args.diffusion_dir):
        for f in os.listdir(args.diffusion_dir):
            if f.endswith('.csv'):
                targets.append(os.path.join(args.diffusion_dir, f))
                
    # Collect files from fim dir
    if os.path.exists(args.fim_dir):
        for f in os.listdir(args.fim_dir):
            if f.endswith('.csv'):
                targets.append(os.path.join(args.fim_dir, f))
                
    if not targets:
        print("No CSV files found in the provided directories.")
        return
        
    print(f"Found {len(targets)} files to evaluate.")
    
    for f in targets:
        process_file(f, id_to_code, client, args.model, args.output_dir, args.limit)

if __name__ == "__main__":
    main()
