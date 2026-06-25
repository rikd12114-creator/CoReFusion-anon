import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

RESULTS_DIR = "results/edit_signal"
os.makedirs(RESULTS_DIR, exist_ok=True)

# Publication style
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

def extract_num(val, delimiter='_'):
    try:
        return int(val.split(delimiter)[-1])
    except:
        return 0

print("Generating Layer-wise Trajectory Plot...")
try:
    df_layer = pd.read_csv(f"{RESULTS_DIR}/edit_signals_layer_wise.csv")
    layer_summary = df_layer.groupby('step')[['cosine_sim', 'l2_dist']].mean().reset_index()
    layer_summary['layer_num'] = layer_summary['step'].apply(extract_num)
    layer_summary = layer_summary.sort_values('layer_num')
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    ax1.set_xlabel('Transformer Layer Depth (0 to 31)')
    ax1.set_ylabel('L2 Distance (Magnitude)', color=colors[0], fontweight='bold')
    ax1.plot(layer_summary['layer_num'], layer_summary['l2_dist'], color=colors[0], marker='o', linewidth=2.5, markersize=8, label='L2 Feature Distance')
    ax1.tick_params(axis='y', labelcolor=colors[0])
    
    ax2 = ax1.twinx()  
    ax2.set_ylabel('Cosine Similarity (Direction)', color=colors[1], fontweight='bold')
    ax2.plot(layer_summary['layer_num'], layer_summary['cosine_sim'], color=colors[1], marker='s', linewidth=2.5, markersize=8, linestyle='--', label='Cosine Similarity')
    ax2.tick_params(axis='y', labelcolor=colors[1])
    
    plt.title('Internal Edit Signal Trajectory Across Model Layers', pad=20)
    fig.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/layer_trajectory_plot.png", bbox_inches='tight')
    plt.savefig(f"{RESULTS_DIR}/layer_trajectory_plot.pdf", bbox_inches='tight')
    plt.close()
    print("-> layer_trajectory_plot generated.")
except Exception as e:
    print(f"Error producing Layer plot: {e}")

print("Generating Diffusion Step-wise Trajectory Plot...")
try:
    df_diff = pd.read_csv(f"{RESULTS_DIR}/edit_signals_diffusion_steps.csv")
    diff_summary = df_diff.groupby('step')[['cosine_sim', 'l2_dist']].mean().reset_index()
    diff_summary['step_num'] = diff_summary['step'].apply(extract_num)
    diff_summary = diff_summary.sort_values('step_num')
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    ax1.set_xlabel('Diffusion / Denoising Step (T=0 to T=32)')
    ax1.set_ylabel('L2 Distance (Magnitude)', color=colors[0], fontweight='bold')
    ax1.plot(diff_summary['step_num'], diff_summary['l2_dist'], color=colors[0], marker='o', linewidth=2.5, markersize=8, label='L2 Feature Distance')
    ax1.tick_params(axis='y', labelcolor=colors[0])
    
    # Fill under for area focus
    ax1.fill_between(diff_summary['step_num'], diff_summary['l2_dist'], alpha=0.1, color=colors[0])
    
    ax2 = ax1.twinx()  
    ax2.set_ylabel('Cosine Similarity (Direction)', color=colors[1], fontweight='bold')
    ax2.plot(diff_summary['step_num'], diff_summary['cosine_sim'], color=colors[1], marker='s', linewidth=2.5, markersize=8, linestyle='--', label='Cosine Similarity')
    ax2.tick_params(axis='y', labelcolor=colors[1])
    
    # Highlight the alignment phase
    ax1.axvspan(20, 32, color='grey', alpha=0.15)
    plt.text(26, ax2.get_ylim()[0] + 0.05 * (ax2.get_ylim()[1] - ax2.get_ylim()[0]), "Alignment Phase", ha='center', fontsize=12, fontweight='bold', color='#333333')
    
    plt.title('Internal Edit Signal Converging Across Diffusion Steps', pad=20)
    fig.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/diffusion_trajectory_plot.png", bbox_inches='tight')
    plt.savefig(f"{RESULTS_DIR}/diffusion_trajectory_plot.pdf", bbox_inches='tight')
    plt.close()
    print("-> diffusion_trajectory_plot generated.")
except Exception as e:
    print(f"Error producing Diffusion Step plot: {e}")

print("Plotting complete.")
