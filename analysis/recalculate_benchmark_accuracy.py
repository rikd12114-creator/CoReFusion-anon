import pandas as pd
import os
import glob

def calculate_accuracy(df):
    if len(df) == 0:
        return 0.0
    # Strip whitespace for fair comparison
    # Fillna with empty string to avoid comparison with NaN
    gt = df['ground_truth'].fillna('').astype(str).str.strip()
    pred = df['prediction'].fillna('').astype(str).str.strip()
    matches = (gt == pred).sum()
    return matches / len(df)

def main():
    # Paths (Absolute paths derived from presumed repo structure)
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DIFF_DIR = os.path.join(repo_root, "data/benchmark_ReFineID_Diffusion/diffusion_benchmark/")
    FIM_DIR = os.path.join(repo_root, "data/benchmark_ReFineID_FIM/ar_fim_benchmark/")

    if not os.path.exists(DIFF_DIR):
        print(f"Error: Diffusion directory not found at {DIFF_DIR}")
        return
    if not os.path.exists(FIM_DIR):
        print(f"Error: FIM directory not found at {FIM_DIR}")
        return

    # 1. Find common valid IDs from all Diffusion files
    diff_files = sorted(glob.glob(os.path.join(DIFF_DIR, "*.csv")))
    common_valid_ids = None

    print(f"Filtering based on {len(diff_files)} Diffusion result files...")
    for f in diff_files:
        df = pd.read_csv(f)
        # Identify rows without errors
        valid_rows = df[df["error"].isna()]
        valid_ids = set(valid_rows["id"])
        
        if common_valid_ids is None:
            common_valid_ids = valid_ids
        else:
            common_valid_ids = common_valid_ids.intersection(valid_ids)

    if common_valid_ids is None:
        print("No Diffusion files found.")
        return

    print(f"Number of common valid IDs across Diffusion models: {len(common_valid_ids)} (out of original data)")

    # 2. Re-calculate accuracy for Diffusion models
    print("\n" + "="*80)
    print(f"{'Diffusion Model Filename':<60} | {'Filtered EM Acc':<15}")
    print("-" * 80)
    for f in diff_files:
        df = pd.read_csv(f)
        filtered_df = df[df["id"].isin(common_valid_ids)]
        acc = calculate_accuracy(filtered_df)
        print(f"{os.path.basename(f):<60} | {acc:.4%}")

    # 3. Re-calculate accuracy for FIM models
    print("\n" + "="*100)
    print(f"{'FIM Model Filename':<60} | {'Old EM Acc':<12} -> {'Filtered EM Acc':<15}")
    print("-" * 100)
    fim_files = sorted(glob.glob(os.path.join(FIM_DIR, "*.csv")))
    
    results = []
    for f in fim_files:
        df = pd.read_csv(f)
        
        # Check if 'id' column exists
        if "id" not in df.columns:
            print(f"Warning: {os.path.basename(f)} missing 'id' column. Skipping.")
            continue
            
        old_acc = calculate_accuracy(df)
        filtered_df = df[df["id"].isin(common_valid_ids)]
        new_acc = calculate_accuracy(filtered_df)
        
        print(f"{os.path.basename(f):<60} | {old_acc:>10.4%} -> {new_acc:>15.4%}")
        results.append({
            'filename': os.path.basename(f),
            'old_acc': old_acc,
            'new_acc': new_acc,
            'delta': new_acc - old_acc
        })

if __name__ == "__main__":
    main()
