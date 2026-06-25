
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

def create_visualizations(csv_file):
    # Set the aesthetic style
    sns.set_theme(style="whitegrid", palette="muted")
    plt.rcParams.update({
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 16,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
        'figure.titlesize': 18,
        'figure.dpi': 300
    })

    # Load data
    df = pd.read_csv(csv_file)
    
    # Preprocessing: Cleanup
    # Ensure step columns are numeric
    df['step_good'] = pd.to_numeric(df['step_good'], errors='coerce')
    df['step_bad'] = pd.to_numeric(df['step_bad'], errors='coerce')
    
    # Drop rows with NaN steps if they exist
    df = df.dropna(subset=['step_good', 'step_bad'])
    
    # Filter for the 50 datapoints (repeated ~10 times)
    # The user mentioned多余数据需要删除, we can keep only first 10 runs per id if needed
    df = df.sort_values(['id', 'run_id'])
    df = df.groupby('id').head(10).reset_index(drop=True)

    # Calculate Aggregate Statistics for Labels
    mean_good = df['step_good'].mean()
    mean_bad = df['step_bad'].mean()
    std_good = df['step_good'].std()
    std_bad = df['step_bad'].std()
    
    # Perform t-test for significance
    t_stat, p_val = stats.ttest_rel(df['step_good'], df['step_bad'])

    # Create a figure with subplots - 2x2 grid for the 4 core analyses
    fig = plt.figure(figsize=(18, 14))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.25)

    # --- Plot 1: Step Density Comparison (KDE) ---
    ax1 = fig.add_subplot(gs[0, 0])
    sns.kdeplot(df['step_good'], fill=True, label='Well-Named', color="#4C72B0", alpha=0.5, ax=ax1)
    sns.kdeplot(df['step_bad'], fill=True, label='Poorly-Named', color="#C44E52", alpha=0.5, ax=ax1)
    ax1.set_title("Step Density Comparison (Distribution)", fontweight='bold')
    ax1.set_xlabel("Diffusion Step")
    ax1.legend()

    # --- Plot 2: Cumulative Success Rate ---
    ax2 = fig.add_subplot(gs[0, 1])
    max_step = int(max(df['step_good'].max(), df['step_bad'].max()))
    x_range = np.linspace(0, max_step, 100)
    
    ecdf_good = [sum(df['step_good'] <= x) / len(df) for x in x_range]
    ecdf_bad = [sum(df['step_bad'] <= x) / len(df) for x in x_range]
    
    ax2.plot(x_range, ecdf_good, label='Well-Named', color="#4C72B0", lw=2.5)
    ax2.plot(x_range, ecdf_bad, label='Poorly-Named', color="#C44E52", lw=2.5)
    ax2.fill_between(x_range, ecdf_good, alpha=0.1, color="#4C72B0")
    ax2.fill_between(x_range, ecdf_bad, alpha=0.1, color="#C44E52")
    ax2.set_title("Cumulative Convergence Rate", fontweight='bold')
    ax2.set_xlabel("Diffusion Step")
    ax2.set_ylabel("Fraction Converged")
    ax2.set_ylim(0, 1.05)
    ax2.legend(loc='lower right')

    # --- Plot 3: Stability Comparison (STD across runs) ---
    ax3 = fig.add_subplot(gs[1, 0])
    id_stds = df.groupby('id')[['step_good', 'step_bad']].std().reset_index()
    std_plot_df = pd.melt(id_stds[['step_good', 'step_bad']], var_name='Condition', value_name='Step STD')
    std_plot_df['Condition'] = std_plot_df['Condition'].map({'step_good': 'Well-Named', 'step_bad': 'Poorly-Named'})
    
    sns.boxplot(data=std_plot_df, x='Condition', y='Step STD', palette=["#4C72B0", "#C44E52"], hue='Condition', legend=False, ax=ax3)
    sns.stripplot(data=std_plot_df, x='Condition', y='Step STD', color=".3", size=5, alpha=0.5, jitter=True, ax=ax3)
    ax3.set_title("Convergence Stability (Lower is Better)", fontweight='bold')
    ax3.set_ylabel("STD of Steps (across 10 Runs per ID)")

    # --- Plot 4: Statistics Summary Table ---
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis('off')
    
    # Calculate stability means for the table
    stability_good = id_stds['step_good'].mean()
    stability_bad = id_stds['step_bad'].mean()
    
    stats_text = (
        f"       Summary Statistics\n"
        f"       ==================\n\n"
        f" * Average Convergence Step:\n"
        f"   - Well-Named:   {mean_good:.1f} ± {std_good:.1f}\n"
        f"   - Poorly-Named: {mean_bad:.1f} ± {std_bad:.1f}\n\n"
        f" * Performance Delta:\n"
        f"   - Mean Delay:   +{mean_bad - mean_good:.1f} steps\n"
        f"   - Relative:     {((mean_bad/mean_good)-1)*100:.1f}% slower\n\n"
        f" * Statistical Significance:\n"
        f"   - Paired t-test: p = {p_val:.2e}\n\n"
    )
    ax4.text(0.05, 0.5, stats_text, family='monospace', fontsize=15, verticalalignment='center', 
             bbox=dict(facecolor='none', edgecolor='#CCCCCC', boxstyle='round,pad=1'))
    ax4.set_title("Quantitative Analysis", fontweight='bold')

    plt.suptitle("Impact of Identifier Naming on Diffusion Convergence (DiffuCoder)", fontsize=28, fontweight='bold', y=0.98)
    
    output_filename = csv_file.replace('.csv', '_publication_plot_v2.png')
    plt.savefig(output_filename, bbox_inches='tight', dpi=300)
    plt.savefig(output_filename.replace('.png', '.pdf'), bbox_inches='tight')
    print(f"Publication-ready plots saved to {output_filename}")
    plt.close()

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        csv_path = sys.argv[1]
    else:
        csv_path = "/path/to/CoReFusion/results/dreamcoder_scale_naming_exp_20260122_053530.csv"
    
    create_visualizations(csv_path)
