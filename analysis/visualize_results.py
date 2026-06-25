import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set aesthetic style
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.sans-serif'] = ['Arial']

def visualize_results(csv_path, output_dir='plots'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Load data
    df = pd.read_csv(csv_path)
    
    # Filter out PascalCase as requested
    df = df[df['dominant_style'] != 'PascalCase']
    
    # 1. Overall Success Rate
    plt.figure(figsize=(8, 8))
    success_counts = df['success'].value_counts()
    plt.pie(success_counts, labels=success_counts.index, autopct='%1.1f%%', 
            colors=['#ff9999','#66b3ff'], startangle=140, explode=(0.05, 0))
    plt.title('Overall Style Consistency Success Rate', fontsize=16)
    plt.savefig(os.path.join(output_dir, 'overall_success_rate.png'), dpi=300)
    plt.close()

    # 2. Success Rate by Dominant Style
    plt.figure(figsize=(10, 6))
    style_success = df.groupby('dominant_style')['success'].mean().sort_values(ascending=False) * 100
    sns.barplot(x=style_success.index, y=style_success.values, hue=style_success.index, palette='viridis', legend=False)
    plt.axhline(y=df['success'].mean() * 100, color='r', linestyle='--', label=f"Avg: {df['success'].mean()*100:.1f}%")
    plt.title('Success Rate by Target Style', fontsize=16)
    plt.ylabel('Success Rate (%)')
    plt.xlabel('Dominant Style (Target)')
    plt.ylim(0, 100)
    for i, v in enumerate(style_success.values):
        plt.text(i, v + 2, f"{v:.1f}%", ha='center', fontweight='bold')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'success_by_style.png'), dpi=300)
    plt.close()

    # 3. Style Confusion Matrix
    plt.figure(figsize=(12, 10))
    # Filter only the most common styles to keep it clean
    common_styles = df['dominant_style'].value_counts().nlargest(5).index
    df_filtered = df[df['dominant_style'].isin(common_styles)]
    
    confusion = pd.crosstab(df_filtered['dominant_style'], df_filtered['fixed_style'], normalize='index') * 100
    sns.heatmap(confusion, annot=True, fmt=".1f", cmap='Blues', cbar_kws={'label': 'Percentage (%)'})
    plt.title('Style Conversion Heatmap\n(Rows: Target Style, Columns: Predicted Style)', fontsize=16)
    plt.xlabel('Generated Style')
    plt.ylabel('Target Style (Context)')
    plt.savefig(os.path.join(output_dir, 'style_confusion_matrix.png'), dpi=300)
    plt.close()

    # 4. Consistency across Repetitions
    plt.figure(figsize=(10, 6))
    repeat_success = df.groupby('repeat_idx')['success'].mean() * 100
    sns.lineplot(x=repeat_success.index, y=repeat_success.values, marker='o', linewidth=2.5, color='#2ecc71')
    plt.title('Performance Stability Across Repetitions', fontsize=16)
    plt.ylabel('Success Rate (%)')
    plt.xlabel('Repetition Index')
    plt.xticks(repeat_success.index)
    plt.ylim(0, 100)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(os.path.join(output_dir, 'stability_over_repeats.png'), dpi=300)
    plt.close()

    # 5. Case-by-case analysis (Top 10 most failing cases)
    plt.figure(figsize=(12, 8))
    case_failure = df.groupby('case_idx')['success'].mean().sort_values().head(10) * 100
    sns.barplot(x=case_failure.index.astype(str), y=case_failure.values, palette='magma', hue=case_failure.index.astype(str), legend=False)
    plt.title('Top 10 Most Challenging Test Cases (Lowest Success Rate)', fontsize=16)
    plt.ylabel('Success Rate (%)')
    plt.xlabel('Case Index')
    plt.ylim(0, 100)
    plt.savefig(os.path.join(output_dir, 'challenging_cases.png'), dpi=300)
    plt.close()

    print(f"Visualizations saved to {output_dir}/")
    print(f"Summary Statistics:")
    print(df.groupby('dominant_style')['success'].agg(['mean', 'count']))

if __name__ == "__main__":
    csv_file = '/path/to/CoReFusion/results/style_consistency_results_0123_1006.csv'
    visualize_results(csv_file)
