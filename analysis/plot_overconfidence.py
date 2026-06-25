import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# ==============================================================================
# 1. Setup and Data Loading
# ==============================================================================
# Create output directory
os.makedirs("docs/figures", exist_ok=True)

file_path = "results/DiffuCoder-7B-Base_20260303_085659.csv"
print(f"Loading data from: {file_path}")
df = pd.read_csv(file_path)

# Calculate mean base_rank by regime and probe type
grouped = df.groupby(['regime', 'probe_type'])['base_rank'].mean().unstack('probe_type')

# Ordering for the plot
regime_order = ["OVERCONFIDENT", "UNCERTAIN", "CONFIDENT_RARE"]
regime_labels = ["Overconfident\n(GT already generic)", "Uncertain\n(Middle Ground)", "Confident Rare\n(GT highly specific)"]
probe_order = ["gt", "smell_severe", "smell_moderate", "smell_mild"]
probe_labels = ["Ground Truth (Original)", "Severe Smell (e.g., 'x')", "Moderate Smell (e.g., 'tmp')", "Mild Smell (e.g., 'result1')"]

# Filter to available data just in case
regime_order = [r for r in regime_order if r in grouped.index]

# Rearrange data into a flat format suitable for seaborn
plot_data = []
for i, regime in enumerate(regime_order):
    for j, probe in enumerate(probe_order):
        plot_data.append({
            "Regime": regime_labels[i],
            "Probe Type": probe_labels[j],
            "Mean Rank": grouped.loc[regime, probe]
        })

df_plot = pd.DataFrame(plot_data)

# ==============================================================================
# 2. Plotting Setup (Publication Style)
# ==============================================================================
# Style settings
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

colors = {
    "Ground Truth (Original)": "#2A9D8F",  # Nice Teal/Green
    "Severe Smell (e.g., 'x')": "#E76F51",  # Strong Coral/Red
    "Moderate Smell (e.g., 'tmp')": "#F4A261", # Orange
    "Mild Smell (e.g., 'result1')": "#E9C46A"  # Yellow-orange
}

fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

# Draw grouped barplot
barplot = sns.barplot(
    data=df_plot,
    x="Regime",
    y="Mean Rank",
    hue="Probe Type",
    palette=colors,
    edgecolor="black",
    linewidth=1,
    alpha=0.9,
    ax=ax
)

# Remove Log Scale to use linear scale
# ax.set_yscale('log')

# Important: INVERT the y-axis so Rank=0 (highest confidence) is at the TOP
ax.invert_yaxis()
# Set limits for linear scale (Max rank ~10.4k)
ax.set_ylim(11500, 0)

# Customize axes
ax.set_ylabel("Mean Token Rank (Linear Scale)\n↑ Higher on chart = Higher Confidence (Top is Rank 0)", fontsize=14, weight='bold')
ax.set_xlabel("Identifier Regime", fontsize=14, weight='bold', labelpad=15)
ax.set_title("The Over-confidence Trap in LLMs during Code Refactoring", fontsize=16, weight='bold', pad=20)

# Clean up y-axis to show exact integer ticks
import matplotlib.ticker as ticker
ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, pos: ('{:g}'.format(y))))

# Add value labels on top (now relative to inverted axis, meaning below the line physically)
for p in ax.patches:
    val = p.get_height()
    if pd.isna(val) or val <= 0:
        continue
    # Position adjustments: we want it at the 'end' of the bar. For inverted axis, that means y_pos should be 
    # slightly less than the value numerically (which puts it physically "above" the end of the bar)
    x_pos = p.get_x() + p.get_width() / 2.
    y_pos = val - 150  # Offset for inverted linear scale
    
    # Format number: e.g. 10410.7 -> 10.4k; 73.8 -> 74
    if val >= 1000:
        text = f"{val/1000:.1f}k"
    else:
        text = f"{int(val)}"
        
    ax.text(x_pos, y_pos, text, 
            ha='center', va='bottom', fontsize=11, color='#333333', weight='bold')

# Enhance legend
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, labels, title='Probe / Identifier Type', title_fontsize=13,
          loc='lower left', bbox_to_anchor=(0.02, 0.02))

# Add a subtle annotation explaining the "Trap"
#if "Confident Rare\n(GT highly specific)" in [d['Regime'] for d in plot_data]:
    # Highlight the drop in rank (increase in confidence) for severe smell
    # Note: coordinates must be updated for inverted y-axis
    # ax.annotate(
    #     "Semantic Degradation Trap:\nModel is 6x more confident\nabout severe smell 'x' than\nthe correct specific identifier.",
    #     xy=(2.05, 1750), # Coordinates roughly pointing to the Severe bar in Confident Rare
    #     xytext=(0.8, -300), # Higher up geometrically (lower numerically)
    #     arrowprops=dict(facecolor='#E76F51', shrink=0.08, width=2, headwidth=8, alpha=0.8),
    #     fontsize=12,
    #     color='#E76F51',
    #     weight='bold',
    #     bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#E76F51", alpha=0.9)
    # )

sns.despine(top=False, bottom=True, right=True) # Move spine adjustment due to inversion
plt.tight_layout()

output_path = "docs/figures/overconfidence_trap.png"
plt.savefig(output_path, bbox_inches='tight')
plt.savefig(output_path.replace('.png', '.pdf'), bbox_inches='tight')

print(f"Plots successfully generated and saved to:")
print(f"  - {output_path}")
print(f"  - {output_path.replace('.png', '.pdf')}")
