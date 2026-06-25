import csv
import os

def calculate_exact_match_accuracy(file_path):
    if not os.path.exists(file_path):
        print(f"Error: {file_path} not found.")
        return None
    
    try:
        with open(file_path, mode='r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames or 'ground_truth' not in reader.fieldnames or 'prediction' not in reader.fieldnames:
                print(f"Error: {file_path} must contain 'ground_truth' and 'prediction' columns.")
                return None
            
            total = 0
            matches = 0
            for row in reader:
                gt = (row['ground_truth'] or "").strip()
                pred = (row['prediction'] or "").strip()
                
                if gt == pred:
                    matches += 1
                total += 1
            
            if total == 0:
                return 0.0
            return matches / total
            
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None

if __name__ == "__main__":
    # TODO: Add your result filenames here
    results_to_analyze = [
        "results/DreamCoder32_noPrompt_20260206_094435.csv",
        "results/DreamCoder64_noPrompt_20260206_093622.csv",
        "results/DiffuCoder_Base_noPrompt_20260206_091137.csv",
        "results/DeepSeek-Coder-6.7B-Instruct_refineID_results(1).csv",
        "results/Llama-3.1-8B-Instruct_refineID_results(1).csv",
        "results/Qwen2.5-Coder-7B-Instruct_refineID_results(1).csv",
    ]
    
    print(f"{'Filename':<60} | {'Exact Match Accuracy':<20}")
    print("-" * 83)
    
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    for relative_path in results_to_analyze:
        file_path = os.path.join(repo_root, relative_path)
        
        accuracy = calculate_exact_match_accuracy(file_path)
        if accuracy is not None:
            filename = os.path.basename(file_path)
            print(f"{filename:<60} | {accuracy:.4%}")
