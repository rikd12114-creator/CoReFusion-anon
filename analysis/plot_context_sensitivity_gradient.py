import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# ==============================================================================
# 1. Setup and Data Loading
# ==============================================================================
os.makedirs("docs/figures", exist_ok=True)

file_path = "results/DiffuCoder-7B-Base_20260303_100019.csv"
print(f"Loading data from: {file_path}")
df = pd.read_csv(file_path)

# Filter for the critical regime we want to highlight
df_rare = df[df['regime'] == 'CONFIDENT_RARE']

# Calculate mean probe_rank for each probe type at each alpha
grouped = df_rare.groupby(['alpha', 'probe_type'])['probe_rank'].mean().reset_index()

# Define presentation mapping
probe_mapping = {
    'gt': 'Ground Truth (Original Highly Specific)',
    'smell_severe': "Severe Smell (e.g., 'x', 'i')",
    'smell_moderate': "Moderate Smell (e.g., 'tmp', 'val')",
    'smell_mild': "Mild Smell (e.g., 'result1')"
}
grouped['Probe Type'] = grouped['probe_type'].map(probe_mapping)

colors = {
    "Ground Truth (Original Highly Specific)": "#2A9D8F",  # Nice Teal/Green
    "Severe Smell (e.g., 'x', 'i')": "#E76F51",  # Strong Coral/Red
    "Moderate Smell (e.g., 'tmp', 'val')": "#F4A261", # Orange
    "Mild Smell (e.g., 'result1')": "#E9C46A"  # Yellow-orange
}

# ==============================================================================
# 2. Plotting Setup
# ==============================================================================
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("paper", font_scale=1.5)
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Inter', 'Arial', 'Helvetica']
plt.rcParams['axes.edgecolor'] = '#333333'
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['grid.alpha'] = 0.5
plt.rcParams['grid.color'] = '#e0e0e0'
plt.rcParams['legend.frameon'] = True
plt.rcParams['legend.edgecolor'] = '#cccccc'
plt.rcParams['legend.fontsize'] = 12

fig, ax = plt.subplots(figsize=(11, 7), dpi=300)

# Draw lines
sns.lineplot(
    data=grouped,
    x='alpha',
    y='probe_rank',
    hue='Probe Type',
    palette=colors,
    linewidth=4,
    marker="o",
    markersize=9,
    ax=ax
)

# Invert Y-axis so rank 1 (highest confidence) is at the top
ax.invert_yaxis()
ax.set_ylim(25000, 0) # Max rank goes down to ~24000, top is 0

# X-axis as percentage for clearer intuitive reading
ax.set_xlim(-0.02, 1.02)
import matplotlib.ticker as ticker
ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f"{int(x*100)}%"))

# Customize axes labels
ax.set_xlabel("Context Masking Ratio (α)\n← Full Context --------------------------------------- No Context (Blind) →", 
              fontsize=14, weight='bold', labelpad=12)
ax.set_ylabel("Mean Token Rank (Linear Scale)\n↑ Higher on chart = Higher Confidence (Top is Rank 1)", 
              fontsize=14, weight='bold')

ax.set_title("The Degradation Gradient: Model Confidence vs. Context Loss", 
             fontsize=16, weight='bold', pad=25)

# Formatting Y-axis
ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, pos: f"{int(y):,}"))

# Add specific annotations
# Highlight Severe Smell diving to top rank at alpha 1.0
end_severe = grouped[(grouped['alpha'] == 1.0) & (grouped['probe_type'] == 'smell_severe')]['probe_rank'].values[0]
ax.annotate(
    f'Blind Confidence:\nRank rockets to {int(end_severe)}',
    xy=(1.0, end_severe),
    xytext=(0.7, 3000),
    arrowprops=dict(facecolor='#E76F51', shrink=0.05, width=2, headwidth=8, alpha=0.9),
    fontsize=12,
    color='#E76F51',
    weight='bold',
    ha='center',
    bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#E76F51", alpha=0.9)
)

# Highlight GT dropping to bottom rank at alpha 1.0
end_gt = grouped[(grouped['alpha'] == 1.0) & (grouped['probe_type'] == 'gt')]['probe_rank'].values[0]
ax.annotate(
    f'Complete Uncertainty:\nRank plummets to {int(end_gt):,}',
    xy=(1.0, end_gt),
    xytext=(0.7, 21000),
    arrowprops=dict(facecolor='#2A9D8F', shrink=0.05, width=2, headwidth=8, alpha=0.9),
    fontsize=12,
    color='#2A9D8F',
    weight='bold',
    ha='center',
    bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#2A9D8F", alpha=0.9)
)

# Enhance legend
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, labels, title='Probe Type (Evaluated in Confident Rare Regime)', 
          title_fontsize=12, loc='center left', bbox_to_anchor=(0.02, 0.45))

sns.despine(top=False, bottom=True, right=True) # Because we inverted Y, top spine is the "0" line
plt.tight_layout()

output_path = "docs/figures/context_degradation_gradient.png"
plt.savefig(output_path, bbox_inches='tight')
plt.savefig(output_path.replace('.png', '.pdf'), bbox_inches='tight')

print(f"Plots successfully generated and saved to:")
print(f"  - {output_path}")
print(f"  - {output_path.replace('.png', '.pdf')}")
