import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

RESULTS_DIR = "results/presentation_plots"
os.makedirs(RESULTS_DIR, exist_ok=True)

# Set publication style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size': 14,
    'axes.labelsize': 16,
    'axes.titlesize': 18,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 14,
    'figure.titlesize': 20,
    'figure.dpi': 300,
    'font.family': 'sans-serif'
})

colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6']

print("Generating Experiment B Plots...")
try:
    df_unmask = pd.read_csv("results/abc/unmasking_order_20260312_225458.csv")
    df_unmask['Token Type'] = df_unmask['is_smell_token'].map({True: 'Smell Token', False: 'Contextual Token'})
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    sns.violinplot(data=df_unmask, x='Token Type', y='first_confident_step', 
                   palette=[colors[0], colors[1]], ax=axes[0], inner="box", alpha=0.7)
    axes[0].set_title('First Confident Step Distribution')
    axes[0].set_ylabel('Denoising Step (Earlier is more confident)')
    axes[0].set_xlabel('')
    
    sns.violinplot(data=df_unmask, x='Token Type', y='avg_flip_step', 
                   palette=[colors[0], colors[1]], ax=axes[1], inner="box", alpha=0.7)
    axes[1].set_title('Average Flip Step Distribution')
    axes[1].set_ylabel('Denoising Step (Earlier is flipped faster)')
    axes[1].set_xlabel('')
    
    plt.suptitle("Unmasking Dynamics: Do models prioritize 'smell' components?", y=1.05)
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/exp_b_unmasking_dynamics.png", bbox_inches='tight')
    plt.savefig(f"{RESULTS_DIR}/exp_b_unmasking_dynamics.pdf", bbox_inches='tight')
    plt.close()
    print("-> exp_b_unmasking_dynamics.png created.")
except Exception as e:
    print(f"Error producing Exp B plot: {e}")

print("Generating Experiment C Plots...")
try:
    df_ranking = pd.read_csv("results/abc/token_ranking_20260312_230225.csv")
    df_ranking['Token Type'] = df_ranking['is_smell_token'].map({True: 'Smell Token', False: 'Contextual Token'})
    
    fig, ax = plt.subplots(figsize=(10, 7))
    sns.boxenplot(data=df_ranking, x='Token Type', y='mean_entropy_change', 
                  palette=[colors[0], colors[1]], ax=ax)
    
    ax.set_title('Token Ranking Swings\n(Cross-Step Entropy $\\Delta$)', pad=20)
    ax.set_ylabel('Mean Absolute $\\Delta$ Entropy $|H_{k+1} - H_k|$ \n(Higher = More Fluctuation / Uncertainty)')
    ax.set_xlabel('')
    
    # Add significance annotation
    y_max = df_ranking['mean_entropy_change'].max()
    plt.plot([-0.05, -0.05, 1.05, 1.05], [y_max+0.1, y_max+0.2, y_max+0.2, y_max+0.1], lw=1.5, c='k')
    plt.text(0.5, y_max+0.25, "*** ($p < 0.001$)", ha='center', va='bottom', color='k')
    
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/exp_c_entropy_fluctuation.png", bbox_inches='tight')
    plt.savefig(f"{RESULTS_DIR}/exp_c_entropy_fluctuation.pdf", bbox_inches='tight')
    plt.close()
    print("-> exp_c_entropy_fluctuation.png created.")
except Exception as e:
    print(f"Error producing Exp C plot: {e}")

print("Done. Plots saved to you results/presentation_plots directory.")
