import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.ticker import PercentFormatter

def extract_model_name(filename):
    """
    Extracts the clean model name from the benchmark file name.
    Example: 'DiffuCoder-7B_refineID_diffusion_20260226_125550.csv' -> 'DiffuCoder-7B'
             'Qwen2.5-Coder-7B_refineID_fim_20260226_014318.csv' -> 'Qwen2.5-Coder-7B'
    """
    if "_refineID_" in filename:
        return filename.split("_refineID_")[0]
    return filename.replace(".csv", "")

def main():
    # Setup paths
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    summary_file = os.path.join(repo_root, "results", "llm_judge_results", "llm_judge", "summary_judge_Qwen2.5-7B-Instruct_20260304_130047.csv")
    output_dir = os.path.join(repo_root, "results", "llm_judge_results")
    
    if not os.path.exists(summary_file):
        print(f"Error: Could not find {summary_file}")
        return
        
    # Read the summary data
    print(f"Loading data from {summary_file}...")
    df = pd.read_csv(summary_file)
    
    # Process model names
    df['model_name'] = df['benchmark_file'].apply(extract_model_name)
    
    # Aggregate data (take mean for repeated runs of the same model)
    agg_df = df.groupby(['benchmark_type', 'model_name']).agg({
        'exact_match_acc': 'mean',
        'llm_judge_acc': 'mean'
    }).reset_index()
    
    # Convert to percentages for plotting
    agg_df['exact_match_pct'] = agg_df['exact_match_acc'] * 100
    agg_df['llm_judge_pct'] = agg_df['llm_judge_acc'] * 100
    
    # Calculate the gap/improvement
    agg_df['improvement'] = agg_df['llm_judge_pct'] - agg_df['exact_match_pct']
    
    # Sort for better visualization: 
    # First split by benchmark_type, then sort by LLM judge accuracy descending
    agg_df = agg_df.sort_values(by=['benchmark_type', 'llm_judge_pct'], ascending=[True, False])
    
    # Output statistics to console
    print("\n" + "="*80)
    print(f"{'Model':<30} | {'Type':<10} | {'Exact Match':<12} | {'LLM Acceptable':<15}")
    print("-" * 80)
    for _, row in agg_df.iterrows():
        b_type = row['benchmark_type'].upper()
        em = row['exact_match_pct']
        llm = row['llm_judge_pct']
        print(f"{row['model_name']:<30} | {b_type:<10} | {em:>5.1f}%      | {llm:>6.1f}% (+{llm-em:.1f}%)")
    print("="*80 + "\n")
    
    # ---- PLOTTING ----
    print("Generating visualizations...")
    
    # Set style
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({'font.size': 12, 'font.family': 'sans-serif'})
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    x = np.arange(len(agg_df))
    width = 0.35
    
    # Define colors
    color_em = '#5B84B1FF' # Diffused Blue
    color_llm = '#FC766AFF' # Fiery Coral
    
    # Create grouped bar chart
    rects1 = ax.bar(x - width/2, agg_df['exact_match_pct'], width, label='Exact Match (Strict)', color=color_em, edgecolor='white', linewidth=1)
    rects2 = ax.bar(x + width/2, agg_df['llm_judge_pct'], width, label='LLM Judged Acceptable (Semantic)', color=color_llm, edgecolor='white', linewidth=1)
    
    # Add values on top of bars
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.1f}%',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9, rotation=0)
            
    autolabel(rects1)
    autolabel(rects2)
    
    # Customise axes
    ax.set_ylabel('Accuracy (%)', fontsize=14, fontweight='bold')
    ax.set_title('Variable Naming Success Rate: Exact Match vs. Semantic Acceptability\n(N=1000 per dataset, Judge: Qwen2.5-7B-Instruct)', fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(x)
    
    # Create labels with type annotations
    labels = []
    for _, row in agg_df.iterrows():
        prefix = "[DIFF] " if row['benchmark_type'] == 'diffusion' else "[FIM] "
        labels.append(f"{prefix}{row['model_name']}")
        
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.yaxis.set_major_formatter(PercentFormatter())
    
    # Legend
    ax.legend(loc='upper left', fontsize=12, frameon=True, shadow=True)
    
    # Add subtle vertical lines to separate diffusion from FIM
    diffusion_count = len(agg_df[agg_df['benchmark_type'] == 'diffusion'])
    if 0 < diffusion_count < len(agg_df):
        ax.axvline(x=diffusion_count - 0.5, color='gray', linestyle='--', alpha=0.7, linewidth=2)
        
        # Add text annotations for groups
        ax.text(diffusion_count / 2 - 0.5, ax.get_ylim()[1] * 0.95, 
                'DIFFUSION MODELS', ha='center', va='top', 
                fontsize=14, fontweight='bold', color='gray', 
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
                
        ax.text(diffusion_count + (len(agg_df) - diffusion_count) / 2 - 0.5, ax.get_ylim()[1] * 0.95, 
                'AUTOREGRESSIVE (FIM) MODELS', ha='center', va='top', 
                fontsize=14, fontweight='bold', color='gray',
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

    plt.tight_layout()
    
    # Save the plot
    output_path = os.path.join(output_dir, 'naming_accuracy_comparison.png')
    output_path_pdf = os.path.join(output_dir, 'naming_accuracy_comparison.pdf')
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_path_pdf, bbox_inches='tight')
    print(f"Visualizations saved to:\n- {output_path}\n- {output_path_pdf}")
    
    # SECOND PLOT: Just the improvement gap (Delta)
    plt.figure(figsize=(12, 6))
    
    # Sort by improvement
    gap_df = agg_df.sort_values('improvement', ascending=True)
    y_pos = np.arange(len(gap_df))
    
    colors = ['#4CAF50' if t == 'diffusion' else '#2196F3' for t in gap_df['benchmark_type']]
    
    bars = plt.barh(y_pos, gap_df['improvement'], color=colors, edgecolor='white')
    
    # Add labels
    plt.yticks(y_pos, gap_df['model_name'])
    plt.xlabel('Absolute Improvement (% points)', fontsize=12, fontweight='bold')
    plt.title('Hidden Performance: LLM Acceptable - Exact Match\n(How much exact match underestimates the actual semantic capability)', fontsize=14, fontweight='bold')
    
    # Add values on bars
    for i, bar in enumerate(bars):
        width = bar.get_width()
        plt.text(width + 0.5, bar.get_y() + bar.get_height()/2, f'+{width:.1f}%', 
                 ha='left', va='center', fontsize=10)
                 
    # Custom legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#4CAF50', label='Diffusion Models'),
        Patch(facecolor='#2196F3', label='FIM Models')
    ]
    plt.legend(handles=legend_elements, loc='lower right')
    
    plt.tight_layout()
    
    gap_output_path = os.path.join(output_dir, 'accuracy_hidden_gap.png')
    plt.savefig(gap_output_path, dpi=300, bbox_inches='tight')
    print(f"- {gap_output_path}")

if __name__ == "__main__":
    main()
