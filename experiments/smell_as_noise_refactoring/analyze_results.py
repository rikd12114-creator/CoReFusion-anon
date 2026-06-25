import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob

# Set style
sns.set(style="whitegrid")
plt.rcParams.update({'font.size': 12})

RESULTS_DIR = "experiments/smell_as_noise_refactoring/results"

def load_latest_results():
    files = glob.glob(os.path.join(RESULTS_DIR, "retention_rate_*.csv"))
    if not files:
        raise FileNotFoundError("No results found.")
    latest_file = max(files, key=os.path.getctime)
    print(f"Loading results from {latest_file}")
    return pd.read_csv(latest_file)

def analyze_and_plot():
    df = load_latest_results()
    
    # Calculate metrics
    # Group by condition and noise_level
    grouped = df.groupby(['condition', 'noise_level']).agg({
        'is_restored': 'mean',
        'is_refactored': 'mean',
        'target_masked': 'mean' # Just to check mask coverage
    }).reset_index()
    
    print("Summary Statistics:")
    print(grouped)
    
    # Plot 1: Retention Rate (Stability)
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x='noise_level', y='is_restored', hue='condition', marker='o', ci=95)
    plt.title("Code Stability vs Noise Level\n(Hypothesis: Smelly code is less stable/more noise-like)")
    plt.ylabel("Retention Rate (Probability of keeping original name)")
    plt.xlabel("Noise Level (Mask Ratio)")
    plt.ylim(0, 1.1)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "stability_curve.png"))
    plt.show()
    
    # Plot 2: Refactoring Success Rate
    # Only for smelly condition
    smelly_df = df[df['condition'] == 'smelly']
    if not smelly_df.empty:
        plt.figure(figsize=(10, 6))
        sns.lineplot(data=smelly_df, x='noise_level', y='is_refactored', marker='s', color='green', ci=95)
        plt.title("Refactoring Success Rate vs Noise Level\n(Can noise induce refactoring?)")
        plt.ylabel("Refactoring Rate (Probability of changing to clean name)")
        plt.xlabel("Noise Level (Mask Ratio)")
        plt.ylim(0, 1.1)
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, "refactoring_curve.png"))
        plt.show()

if __name__ == "__main__":
    analyze_and_plot()
