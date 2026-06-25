"""
Visualization and Analysis Script for Deobfuscation Experiments

This script provides tools to:
1. Visualize deobfuscation progress over diffusion steps
2. Compare performance across different code samples
3. Analyze which variable types are easier to recover
4. Generate summary plots and tables
"""

import json
import os
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict
from pathlib import Path


def load_experiment_results(results_dir: str) -> List[Dict]:
    """Load all experiment results from a directory."""
    results = []
    results_path = Path(results_dir)

    if not results_path.exists():
        print(f"Directory not found: {results_dir}")
        return results

    # Load individual experiment results
    for subdir in results_path.iterdir():
        if subdir.is_dir():
            results_file = subdir / "results.json"
            if results_file.exists():
                with open(results_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    data['experiment_dir'] = str(subdir)
                    results.append(data)

    # Load batch summaries
    for file in results_path.glob("*_batch_summary_*.json"):
        with open(file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            data['is_batch'] = True
            results.append(data)

    return results


def load_progress_data(experiment_dir: str) -> List[Dict]:
    """Load progress tracking data from an experiment."""
    progress_file = Path(experiment_dir) / "progress.json"
    if not progress_file.exists():
        return []

    with open(progress_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def plot_deobfuscation_progress(progress_data: List[Dict], output_file: str = None):
    """
    Plot how deobfuscation metrics improve over diffusion steps.
    """
    if not progress_data:
        print("No progress data available")
        return

    steps = [p['step'] for p in progress_data if 'error' not in p]
    exact_match_rates = [p.get('exact_match_rate', 0) for p in progress_data if 'error' not in p]
    avg_edit_distances = [p.get('avg_edit_distance', 0) for p in progress_data if 'error' not in p]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Plot exact match rate
    ax1.plot(steps, exact_match_rates, marker='o', linewidth=2, markersize=6)
    ax1.set_xlabel('Diffusion Step', fontsize=12)
    ax1.set_ylabel('Exact Match Rate', fontsize=12)
    ax1.set_title('Deobfuscation Accuracy Over Steps', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 1.05])

    # Plot average edit distance
    ax2.plot(steps, avg_edit_distances, marker='s', color='orange', linewidth=2, markersize=6)
    ax2.set_xlabel('Diffusion Step', fontsize=12)
    ax2.set_ylabel('Average Edit Distance', fontsize=12)
    ax2.set_title('Edit Distance Over Steps', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Progress plot saved to: {output_file}")
    else:
        plt.show()

    plt.close()


def plot_metrics_by_type(results: Dict, output_file: str = None):
    """
    Plot performance metrics grouped by identifier type.
    """
    final_metrics = results.get('final_metrics', {})
    by_type = final_metrics.get('by_type', {})

    if not by_type:
        print("No type-specific metrics available")
        return

    types = list(by_type.keys())
    exact_match_rates = [by_type[t].get('exact_match_rate', 0) for t in types]
    avg_edit_distances = [by_type[t].get('avg_edit_distance', 0) for t in types]
    totals = [by_type[t].get('total', 0) for t in types]

    # Clean up type names for display
    display_names = [t.replace('_', ' ').title() for t in types]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Plot exact match rates
    bars1 = ax1.bar(display_names, exact_match_rates, color='steelblue', alpha=0.8)
    ax1.set_ylabel('Exact Match Rate', fontsize=12)
    ax1.set_title('Deobfuscation Accuracy by Identifier Type', fontsize=14, fontweight='bold')
    ax1.set_ylim([0, 1.05])
    ax1.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for bar, rate, total in zip(bars1, exact_match_rates, totals):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{rate:.1%}\n(n={total})',
                ha='center', va='bottom', fontsize=10)

    # Plot average edit distances
    bars2 = ax2.bar(display_names, avg_edit_distances, color='coral', alpha=0.8)
    ax2.set_ylabel('Average Edit Distance', fontsize=12)
    ax2.set_title('Edit Distance by Identifier Type', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for bar, dist in zip(bars2, avg_edit_distances):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{dist:.2f}',
                ha='center', va='bottom', fontsize=10)

    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Type metrics plot saved to: {output_file}")
    else:
        plt.show()

    plt.close()


def plot_batch_comparison(batch_results: Dict, output_file: str = None):
    """
    Compare performance across multiple code samples in a batch experiment.
    """
    individual_results = batch_results.get('individual_results', [])

    if not individual_results:
        print("No individual results in batch")
        return

    # Filter out failed samples
    successful = [r for r in individual_results if 'error' not in r]

    if not successful:
        print("No successful samples to plot")
        return

    sample_names = [r['sample_name'] for r in successful]
    exact_match_rates = [r.get('exact_match_rate', 0) for r in successful]
    partial_match_rates = [r.get('partial_match_rate', 0) for r in successful]

    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(sample_names))
    width = 0.35

    bars1 = ax.bar(x - width/2, exact_match_rates, width, label='Exact Match', color='steelblue', alpha=0.8)
    bars2 = ax.bar(x + width/2, partial_match_rates, width, label='Partial Match', color='lightblue', alpha=0.8)

    ax.set_xlabel('Code Sample', fontsize=12)
    ax.set_ylabel('Match Rate', fontsize=12)
    ax.set_title('Deobfuscation Performance Across Samples', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(sample_names, rotation=45, ha='right')
    ax.legend(fontsize=11)
    ax.set_ylim([0, 1.05])
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1%}',
                   ha='center', va='bottom', fontsize=9)

    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Batch comparison plot saved to: {output_file}")
    else:
        plt.show()

    plt.close()


def generate_summary_table(results: List[Dict], output_file: str = None):
    """
    Generate a summary table of all experiments.
    """
    if not results:
        print("No results to summarize")
        return

    print("\n" + "="*100)
    print("EXPERIMENT SUMMARY TABLE")
    print("="*100)
    print(f"{'Model':<20} {'Sample':<20} {'Exact Match':<15} {'Partial Match':<15} {'Avg Edit Dist':<15}")
    print("-"*100)

    summary_data = []

    for result in results:
        if result.get('is_batch'):
            # Batch result
            model_name = result.get('model_name', 'Unknown')
            avg_exact = result.get('average_exact_match_rate', 0)
            avg_partial = result.get('average_partial_match_rate', 0)
            print(f"{model_name:<20} {'[BATCH AVG]':<20} {avg_exact:<15.2%} {avg_partial:<15.2%} {'-':<15}")

            for ind_result in result.get('individual_results', []):
                if 'error' not in ind_result:
                    sample_name = ind_result.get('sample_name', 'Unknown')
                    exact = ind_result.get('exact_match_rate', 0)
                    partial = ind_result.get('partial_match_rate', 0)
                    print(f"{'  └─':<20} {sample_name:<20} {exact:<15.2%} {partial:<15.2%} {'-':<15}")
        else:
            # Individual result
            model_name = result.get('model_name', 'Unknown')
            final_metrics = result.get('final_metrics', {})
            exact = final_metrics.get('exact_match_rate', 0)
            partial = final_metrics.get('partial_match_rate', 0)
            avg_edit = final_metrics.get('avg_edit_distance', 0)
            print(f"{model_name:<20} {'[Single]':<20} {exact:<15.2%} {partial:<15.2%} {avg_edit:<15.2f}")

            summary_data.append({
                'model': model_name,
                'exact_match': exact,
                'partial_match': partial,
                'avg_edit_distance': avg_edit
            })

    print("="*100 + "\n")

    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("Experiment Summary\n")
            f.write("="*100 + "\n\n")
            for data in summary_data:
                f.write(f"Model: {data['model']}\n")
                f.write(f"  Exact Match Rate: {data['exact_match']:.2%}\n")
                f.write(f"  Partial Match Rate: {data['partial_match']:.2%}\n")
                f.write(f"  Avg Edit Distance: {data['avg_edit_distance']:.2f}\n\n")
        print(f"Summary table saved to: {output_file}")


def analyze_experiment(experiment_dir: str):
    """
    Comprehensive analysis of a single experiment.
    """
    print(f"\nAnalyzing experiment: {experiment_dir}")
    print("="*80)

    # Load results
    results_file = Path(experiment_dir) / "results.json"
    if not results_file.exists():
        print(f"Results file not found: {results_file}")
        return

    with open(results_file, 'r', encoding='utf-8') as f:
        results = json.load(f)

    # Create output directory for plots
    plots_dir = Path(experiment_dir) / "plots"
    plots_dir.mkdir(exist_ok=True)

    # Plot progress if available
    progress_data = load_progress_data(experiment_dir)
    if progress_data:
        plot_deobfuscation_progress(
            progress_data,
            output_file=str(plots_dir / "progress.png")
        )

    # Plot metrics by type
    plot_metrics_by_type(
        results,
        output_file=str(plots_dir / "metrics_by_type.png")
    )

    print(f"\nAnalysis complete. Plots saved to: {plots_dir}")


def analyze_all_experiments(results_dir: str = "results/deobfuscation_advanced"):
    """
    Analyze all experiments in the results directory.
    """
    results_path = Path(results_dir)

    if not results_path.exists():
        print(f"Results directory not found: {results_dir}")
        return

    print(f"\nAnalyzing all experiments in: {results_dir}")
    print("="*80)

    # Find all experiment directories
    experiment_dirs = [d for d in results_path.iterdir() if d.is_dir()]

    if not experiment_dirs:
        print("No experiment directories found")
        return

    print(f"Found {len(experiment_dirs)} experiments\n")

    # Analyze each experiment
    for exp_dir in experiment_dirs:
        analyze_experiment(str(exp_dir))

    # Load and summarize all results
    all_results = load_experiment_results(results_dir)
    if all_results:
        summary_file = results_path / "summary_table.txt"
        generate_summary_table(all_results, output_file=str(summary_file))


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        # Analyze specific experiment directory
        experiment_dir = sys.argv[1]
        analyze_experiment(experiment_dir)
    else:
        # Analyze all experiments
        analyze_all_experiments("results/deobfuscation_advanced")

        # Also check for batch results in basic deobfuscation
        basic_results = load_experiment_results("results/deobfuscation")
        if basic_results:
            print("\n" + "="*80)
            print("BASIC DEOBFUSCATION RESULTS")
            print("="*80)
            generate_summary_table(basic_results)
