"""
Visualization & Analysis for the Code Smell Noise Role Experiment
==================================================================

Takes the output CSVs from compare_math_vs_smell_noise.py and generates:
1. Noise Calibration Curve (entropy / confidence / GT prob over diffusion steps)
2. Code Smell Position on the Noise Spectrum (where does smell land?)
3. Token Rank Comparison (how does model rank bad vs good names?)
4. Severity Gradient Analysis (severe → moderate → mild)
5. Statistical Tests
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker
from scipy import stats

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans'],
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'figure.figsize': (14, 8),
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

COLORS = {
    'mask': '#E74C3C',
    'smell_severe': '#E67E22',
    'smell_moderate': '#F1C40F',
    'smell_mild': '#2ECC71',
    'control': '#3498DB',
    'calibration': '#95A5A6',
}

SEVERITY_COLORS = {
    'severe': '#E74C3C',
    'moderate': '#F39C12',
    'mild': '#2ECC71',
}


def plot_noise_calibration_with_mapping(calib_df, probe_df, mapping_df, output_dir, total_steps=256):
    """
    THE key visualization: Noise calibration curve with code smell positions overlaid.
    
    Shows the diffusion noise spectrum (entropy/confidence over steps)
    and marks WHERE code smell of different severities falls on this spectrum.
    """
    fig, axes = plt.subplots(1, 3, figsize=(20, 7))

    metrics = [
        ('entropy', 'Shannon Entropy (nats)', axes[0]),
        ('argmax_confidence', 'Argmax Confidence', axes[1]),
        ('gt_token_prob', 'P(Ground Truth)', axes[2]),
    ]

    for metric_name, ylabel, ax in metrics:
        # Plot calibration curve (average across all samples + runs)
        calib_grouped = calib_df.groupby('step')[metric_name].agg(['mean', 'std']).reset_index()
        steps = calib_grouped['step'].values
        mean_vals = calib_grouped['mean'].values
        std_vals = calib_grouped['std'].values

        ax.plot(steps, mean_vals, color=COLORS['calibration'], linewidth=2,
                label='Diffusion Calibration Curve', zorder=1)
        ax.fill_between(steps, mean_vals - std_vals, mean_vals + std_vals,
                        color=COLORS['calibration'], alpha=0.15, zorder=1)

        # Overlay probe measurements as horizontal lines + markers
        # Mask probe
        mask_probe = probe_df[probe_df['group'] == 'mask']
        if not mask_probe.empty:
            mask_val = mask_probe[metric_name].astype(float).mean()
            ax.axhline(y=mask_val, color=COLORS['mask'], linestyle='--', alpha=0.7,
                       linewidth=1.5, label=f'Mask (step 0): {mask_val:.3f}')
            ax.plot(0, mask_val, 'o', color=COLORS['mask'], markersize=10, zorder=5)

        # Control probe
        ctrl_probe = probe_df[probe_df['group'] == 'control']
        if not ctrl_probe.empty:
            ctrl_val = ctrl_probe[metric_name].astype(float).mean()
            ax.axhline(y=ctrl_val, color=COLORS['control'], linestyle='--', alpha=0.7,
                       linewidth=1.5, label=f'Clean Code: {ctrl_val:.3f}')

        # Smell probes by severity
        for sev in ['severe', 'moderate', 'mild']:
            sev_probe = probe_df[(probe_df['group'] == 'smell') & (probe_df['severity'] == sev)]
            if sev_probe.empty:
                continue
            sev_val = sev_probe[metric_name].astype(float).mean()

            # Find equivalent step from mapping
            sev_map = mapping_df[(mapping_df['group'] == 'smell') & (mapping_df['severity'] == sev)]
            if not sev_map.empty:
                if metric_name == 'entropy':
                    equiv_step = sev_map['equiv_step_entropy'].astype(float).mean()
                elif metric_name == 'argmax_confidence':
                    equiv_step = sev_map['equiv_step_confidence'].astype(float).mean()
                else:
                    equiv_step = sev_map['equiv_step_gt_prob'].astype(float).mean()
            else:
                equiv_step = 0

            color = SEVERITY_COLORS[sev]
            ax.axhline(y=sev_val, color=color, linestyle=':', alpha=0.6, linewidth=1.5)
            ax.plot(equiv_step, sev_val, 's', color=color, markersize=12,
                    zorder=5, markeredgecolor='black', markeredgewidth=1)
            ax.annotate(f'{sev}\n(step ≈{equiv_step:.0f})',
                        xy=(equiv_step, sev_val),
                        xytext=(equiv_step + total_steps * 0.05, sev_val),
                        fontsize=9, fontweight='bold', color=color,
                        arrowprops=dict(arrowstyle='->', color=color, lw=1.5))

        ax.set_xlabel(f'Diffusion Step (0=full noise → {total_steps}=denoised)')
        ax.set_ylabel(ylabel)
        ax.legend(loc='best', fontsize=8)
        ax.grid(alpha=0.3)

    axes[0].set_title('Entropy along Noise Spectrum')
    axes[1].set_title('Confidence along Noise Spectrum')
    axes[2].set_title('P(Ground Truth) along Noise Spectrum')

    fig.suptitle('Code Smell Position on the Diffusion Noise Spectrum', fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    path = os.path.join(output_dir, 'noise_spectrum_mapping.png')
    plt.savefig(path)
    plt.close()
    print(f"  Saved: {path}")


def plot_token_rank_comparison(probe_df, output_dir):
    """
    Bar chart comparing current_token_rank and gt_token_rank across groups.
    Lower rank = model is more "satisfied" with that token.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    groups = []
    labels = []
    current_ranks = []
    gt_ranks = []
    colors = []

    # Mask
    mask_df = probe_df[probe_df['group'] == 'mask']
    if not mask_df.empty:
        groups.append('mask')
        labels.append('Full Mask\n(Step 0)')
        current_ranks.append(mask_df['current_token_rank'].astype(float).mean())
        gt_ranks.append(mask_df['gt_token_rank'].astype(float).mean())
        colors.append(COLORS['mask'])

    # Smell by severity
    for sev in ['severe', 'moderate', 'mild']:
        sdf = probe_df[(probe_df['group'] == 'smell') & (probe_df['severity'] == sev)]
        if sdf.empty:
            continue
        groups.append(f'smell_{sev}')
        labels.append(f'Smell\n({sev})')
        current_ranks.append(sdf['current_token_rank'].astype(float).mean())
        gt_ranks.append(sdf['gt_token_rank'].astype(float).mean())
        colors.append(SEVERITY_COLORS[sev])

    # Control
    ctrl_df = probe_df[probe_df['group'] == 'control']
    if not ctrl_df.empty:
        groups.append('control')
        labels.append('Clean Code\n(Control)')
        current_ranks.append(ctrl_df['current_token_rank'].astype(float).mean())
        gt_ranks.append(ctrl_df['gt_token_rank'].astype(float).mean())
        colors.append(COLORS['control'])

    # Plot 1: Current Token Rank
    bars1 = ax1.bar(labels, current_ranks, color=colors, alpha=0.8,
                    edgecolor='white', linewidth=1.5)
    for bar, rank in zip(bars1, current_ranks):
        ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
                 f'{rank:.0f}', ha='center', va='bottom', fontweight='bold', fontsize=10)

    ax1.set_ylabel('Average Rank (lower = model more satisfied)')
    ax1.set_title('Model\'s Satisfaction with Current Token\n'
                  '(Rank of current token in softmax distribution)')
    ax1.grid(axis='y', alpha=0.3)

    # Plot 2: GT Token Rank (how much model wants the correct name)
    bars2 = ax2.bar(labels, gt_ranks, color=colors, alpha=0.8,
                    edgecolor='white', linewidth=1.5)
    for bar, rank in zip(bars2, gt_ranks):
        ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
                 f'{rank:.0f}', ha='center', va='bottom', fontweight='bold', fontsize=10)

    ax2.set_ylabel('Average Rank (lower = model prefers GT more)')
    ax2.set_title('How Much Model Wants the Ground Truth Name\n'
                  '(Rank of GT token in softmax distribution)')
    ax2.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, 'token_rank_comparison.png')
    plt.savefig(path)
    plt.close()
    print(f"  Saved: {path}")


def plot_equivalent_noise_level_distribution(mapping_df, output_dir, total_steps=256):
    """
    Distribution of equivalent noise levels for each severity.
    This is THE thesis figure: where does code smell fall on the noise spectrum?
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    map_cols = [
        ('equiv_step_entropy', 'Equiv. Step (Entropy)', axes[0]),
        ('equiv_step_gt_prob', 'Equiv. Step (GT Prob)', axes[1]),
        ('equiv_step_confidence', 'Equiv. Step (Confidence)', axes[2]),
    ]

    for col, title, ax in map_cols:
        data_list = []
        label_list = []
        color_list = []

        for sev in ['severe', 'moderate', 'mild']:
            sdf = mapping_df[(mapping_df['group'] == 'smell') & (mapping_df['severity'] == sev)]
            if sdf.empty:
                continue
            vals = sdf[col].astype(float).values
            data_list.append(vals)
            label_list.append(f'Smell\n({sev})')
            color_list.append(SEVERITY_COLORS[sev])

        ctrl = mapping_df[mapping_df['group'] == 'control']
        if not ctrl.empty:
            data_list.append(ctrl[col].astype(float).values)
            label_list.append('Control')
            color_list.append(COLORS['control'])

        if data_list:
            bp = ax.boxplot(data_list, labels=label_list, patch_artist=True, widths=0.6,
                            showmeans=True, meanprops=dict(marker='D', markerfacecolor='white', markersize=8))
            for patch, color in zip(bp['boxes'], color_list):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)

        ax.set_ylabel(f'Equivalent Diffusion Step\n(0=full noise, {total_steps}=clean)')
        ax.set_title(title)
        ax.grid(axis='y', alpha=0.3)

        # Add reference lines
        ax.axhline(y=0, color=COLORS['mask'], linestyle='--', alpha=0.4, label='Full Noise (step 0)')
        ax.axhline(y=total_steps, color=COLORS['control'], linestyle='--', alpha=0.4, label='Clean (step 256)')

    fig.suptitle('Equivalent Noise Level Distribution by Code Smell Severity\n'
                 '(Where does code smell fall on the diffusion noise schedule?)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.92])

    path = os.path.join(output_dir, 'equivalent_noise_distribution.png')
    plt.savefig(path)
    plt.close()
    print(f"  Saved: {path}")


def plot_severity_gradient(probe_df, mapping_df, output_dir):
    """
    Show the gradient: severe → moderate → mild → control
    across multiple metrics. If there's a clear gradient, it supports
    the "code smell = noise with different σ" interpretation.
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    categories = ['severe', 'moderate', 'mild', 'control']
    category_labels = ['Severe\nSmell', 'Moderate\nSmell', 'Mild\nSmell', 'Clean\nCode']
    category_colors = [SEVERITY_COLORS['severe'], SEVERITY_COLORS['moderate'],
                       SEVERITY_COLORS['mild'], COLORS['control']]

    probe_metrics = [
        ('current_token_prob', 'P(Current Token)', axes[0, 0]),
        ('current_token_rank', 'Current Token Rank', axes[0, 1]),
        ('entropy', 'Entropy (nats)', axes[0, 2]),
    ]

    mapping_metrics = [
        ('equiv_step_entropy', 'Equiv. Step (Entropy)', axes[1, 0]),
        ('equiv_step_gt_prob', 'Equiv. Step (GT Prob)', axes[1, 1]),
        ('equiv_step_confidence', 'Equiv. Step (Confidence)', axes[1, 2]),
    ]

    # Probe metrics
    for col, ylabel, ax in probe_metrics:
        vals = []
        for cat in categories:
            if cat == 'control':
                cdf = probe_df[probe_df['group'] == 'control']
            else:
                cdf = probe_df[(probe_df['group'] == 'smell') & (probe_df['severity'] == cat)]
            if not cdf.empty:
                vals.append(cdf[col].astype(float).mean())
            else:
                vals.append(0)

        bars = ax.bar(category_labels, vals, color=category_colors, alpha=0.8,
                      edgecolor='white', linewidth=1.5)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                    f'{v:.3f}' if v < 100 else f'{v:.0f}',
                    ha='center', va='bottom', fontsize=9, fontweight='bold')

        ax.set_ylabel(ylabel)
        ax.grid(axis='y', alpha=0.3)

    # Mapping metrics
    for col, ylabel, ax in mapping_metrics:
        vals = []
        for cat in categories:
            if cat == 'control':
                cdf = mapping_df[mapping_df['group'] == 'control']
            else:
                cdf = mapping_df[(mapping_df['group'] == 'smell') & (mapping_df['severity'] == cat)]
            if not cdf.empty:
                vals.append(cdf[col].astype(float).mean())
            else:
                vals.append(0)

        bars = ax.bar(category_labels, vals, color=category_colors, alpha=0.8,
                      edgecolor='white', linewidth=1.5)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                    f'{v:.1f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

        ax.set_ylabel(ylabel)
        ax.grid(axis='y', alpha=0.3)

    fig.suptitle('Code Smell Severity Gradient: Is There a Monotonic Trend?\n'
                 '(If yes → code smell = noise with varying σ)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.93])

    path = os.path.join(output_dir, 'severity_gradient.png')
    plt.savefig(path)
    plt.close()
    print(f"  Saved: {path}")


def run_statistical_tests(probe_df, mapping_df, output_dir):
    """Statistical analysis with focus on the noise-level interpretation."""
    lines = []
    lines.append("=" * 70)
    lines.append(" STATISTICAL ANALYSIS: Code Smell as Noise")
    lines.append("=" * 70)

    # --- Test 1: Is the severity gradient significant? ---
    lines.append("\n--- Test 1: Severity Gradient (Spearman correlation) ---")
    lines.append("  H1: More severe smell → lower equiv noise step (= more noisy)")

    # Map severity to ordinal: severe=3, moderate=2, mild=1, control=0
    sev_mapping = {'severe': 3, 'moderate': 2, 'mild': 1}

    smell_map = mapping_df[mapping_df['group'] == 'smell'].copy()
    if not smell_map.empty:
        smell_map['severity_ord'] = smell_map['severity'].map(sev_mapping)
        for col in ['equiv_step_entropy', 'equiv_step_gt_prob', 'equiv_step_confidence']:
            vals = smell_map[col].astype(float).values
            sev_ord = smell_map['severity_ord'].values
            if len(vals) > 2:
                rho, p = stats.spearmanr(sev_ord, vals)
                lines.append(f"\n  {col}:")
                lines.append(f"    Spearman ρ = {rho:.4f}, p = {p:.6f} "
                             f"{'***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'}")
                if rho < 0:
                    lines.append(f"    → More severe smell = earlier noise step = MORE noisy ✓")
                else:
                    lines.append(f"    → Direction opposite to hypothesis")

    # --- Test 2: Is smell different from control? ---
    lines.append("\n\n--- Test 2: Code Smell vs. Clean Code ---")
    lines.append("  H1: Model is less satisfied with smell tokens than clean tokens")

    for metric in ['current_token_prob', 'current_token_rank', 'entropy']:
        smell_vals = probe_df[probe_df['group'] == 'smell'][metric].astype(float).values
        ctrl_vals = probe_df[probe_df['group'] == 'control'][metric].astype(float).values

        if len(smell_vals) > 1 and len(ctrl_vals) > 1:
            t_stat, t_p = stats.ttest_ind(smell_vals, ctrl_vals, equal_var=False)
            u_stat, u_p = stats.mannwhitneyu(smell_vals, ctrl_vals, alternative='two-sided')
            pooled_std = np.sqrt((np.var(smell_vals) + np.var(ctrl_vals)) / 2)
            d = (np.mean(smell_vals) - np.mean(ctrl_vals)) / pooled_std if pooled_std > 0 else 0

            lines.append(f"\n  {metric}:")
            lines.append(f"    Smell: mean={np.mean(smell_vals):.4f}, std={np.std(smell_vals):.4f}, n={len(smell_vals)}")
            lines.append(f"    Control: mean={np.mean(ctrl_vals):.4f}, std={np.std(ctrl_vals):.4f}, n={len(ctrl_vals)}")
            lines.append(f"    Welch's t: t={t_stat:.4f}, p={t_p:.6f} "
                         f"{'***' if t_p < 0.001 else '**' if t_p < 0.01 else '*' if t_p < 0.05 else 'ns'}")
            lines.append(f"    Mann-Whitney U: U={u_stat:.0f}, p={u_p:.6f} "
                         f"{'***' if u_p < 0.001 else '**' if u_p < 0.01 else '*' if u_p < 0.05 else 'ns'}")
            lines.append(f"    Cohen's d: {d:.4f} "
                         f"({'large' if abs(d) > 0.8 else 'medium' if abs(d) > 0.5 else 'small' if abs(d) > 0.2 else 'negligible'})")

    # --- Test 3: Equivalent noise level comparison ---
    lines.append("\n\n--- Test 3: Equivalent Noise Level ---")
    lines.append("  H1: Smell's equiv step < Control's equiv step (smell is noisier)")

    for col in ['equiv_step_entropy', 'equiv_step_gt_prob', 'equiv_step_confidence']:
        smell_vals = mapping_df[mapping_df['group'] == 'smell'][col].astype(float).values
        ctrl_vals = mapping_df[mapping_df['group'] == 'control'][col].astype(float).values

        if len(smell_vals) > 1 and len(ctrl_vals) > 1:
            t_stat, t_p = stats.ttest_ind(smell_vals, ctrl_vals, equal_var=False)
            u_stat, u_p = stats.mannwhitneyu(smell_vals, ctrl_vals, alternative='less')
            d = (np.mean(smell_vals) - np.mean(ctrl_vals)) / np.sqrt((np.var(smell_vals) + np.var(ctrl_vals)) / 2)

            lines.append(f"\n  {col}:")
            lines.append(f"    Smell: mean={np.mean(smell_vals):.1f}")
            lines.append(f"    Control: mean={np.mean(ctrl_vals):.1f}")
            lines.append(f"    Diff: {np.mean(smell_vals) - np.mean(ctrl_vals):.1f} steps")
            lines.append(f"    Mann-Whitney U (one-sided): p={u_p:.6f} "
                         f"{'***' if u_p < 0.001 else '**' if u_p < 0.01 else '*' if u_p < 0.05 else 'ns'}")
            lines.append(f"    Cohen's d: {d:.4f}")

    lines.append(f"\n{'='*70}")
    lines.append("Significance: *** p<0.001, ** p<0.01, * p<0.05, ns = not significant")

    report = "\n".join(lines)
    print(report)

    path = os.path.join(output_dir, 'statistical_analysis.txt')
    with open(path, 'w') as f:
        f.write(report)
    print(f"\n  Saved: {path}")


def main():
    parser = argparse.ArgumentParser(description="Visualize noise role experiment results")
    parser.add_argument("--probe", type=str, required=True,
                        help="Path to noise_probe_*.csv")
    parser.add_argument("--calibration", type=str, default=None,
                        help="Path to noise_calibration_*.csv")
    parser.add_argument("--mapping", type=str, default=None,
                        help="Path to noise_mapping_*.csv")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory for plots")
    parser.add_argument("--steps", type=int, default=256,
                        help="Total diffusion steps used in experiment")
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = os.path.join(os.path.dirname(args.probe), 'noise_role_plots')
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading probe results from: {args.probe}")
    probe_df = pd.read_csv(args.probe)
    print(f"  Loaded {len(probe_df)} probe records.")

    calib_df = None
    if args.calibration and os.path.exists(args.calibration):
        print(f"Loading calibration data from: {args.calibration}")
        calib_df = pd.read_csv(args.calibration)
        print(f"  Loaded {len(calib_df)} calibration records.")

    mapping_df = None
    if args.mapping and os.path.exists(args.mapping):
        print(f"Loading mapping data from: {args.mapping}")
        mapping_df = pd.read_csv(args.mapping)
        print(f"  Loaded {len(mapping_df)} mapping records.")

    print(f"\nGroups in probe data: {probe_df['group'].value_counts().to_dict()}")

    print("\nGenerating visualizations...")

    # 1. Noise Calibration with Mapping
    if calib_df is not None and mapping_df is not None:
        plot_noise_calibration_with_mapping(calib_df, probe_df, mapping_df,
                                            args.output_dir, args.steps)

    # 2. Token Rank Comparison
    plot_token_rank_comparison(probe_df, args.output_dir)

    # 3. Equivalent Noise Level Distribution
    if mapping_df is not None:
        plot_equivalent_noise_level_distribution(mapping_df, args.output_dir, args.steps)

    # 4. Severity Gradient
    if mapping_df is not None:
        plot_severity_gradient(probe_df, mapping_df, args.output_dir)

    # 5. Statistical Tests
    if mapping_df is not None:
        print("\nRunning statistical tests...")
        run_statistical_tests(probe_df, mapping_df, args.output_dir)

    print(f"\nAll outputs saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
