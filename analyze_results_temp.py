import pandas as pd
import numpy as np

print("="*50)
print("Experiment A: Internal Representation Edit Signals")
print("="*50)
try:
    df_layer = pd.read_csv("results/abc/edit_signals_layer_wise.csv")
    print("\n--- Layer-wise Edit Signals ---")
    print(df_layer.groupby('step')[['cosine_sim', 'l2_dist']].mean().head())
    print("...")
    print(df_layer.groupby('step')[['cosine_sim', 'l2_dist']].mean().tail())
except Exception as e:
    print("Error reading layer_wise:", e)

try:
    df_diff = pd.read_csv("results/abc/edit_signals_diffusion_steps.csv")
    print("\n--- Diffusion Step-wise Edit Signals ---")
    print(df_diff.groupby('step')[['cosine_sim', 'l2_dist']].mean().head())
    print("...")
    print(df_diff.groupby('step')[['cosine_sim', 'l2_dist']].mean().tail())
except Exception as e:
    print("Error reading diffusion_steps:", e)

print("\n" + "="*50)
print("Experiment B: Unmasking Order (First Confident & Flip Step)")
print("="*50)
try:
    df_unmask = pd.read_csv("results/abc/unmasking_order_20260312_225458.csv")
    print(df_unmask.groupby('is_smell_token')[['avg_flip_step', 'first_confident_step']].agg(['mean', 'std', 'count']))
except Exception as e:
    print("Error reading unmasking order:", e)

print("\n" + "="*50)
print("Experiment C: Token Ranking Change (Entropy Differential)")
print("="*50)
try:
    df_ranking = pd.read_csv("results/abc/token_ranking_20260312_230225.csv")
    print(df_ranking.groupby('is_smell_token')[['mean_entropy_change', 'max_entropy_change']].agg(['mean', 'std', 'count']))
    
    from scipy import stats
    smell_data = df_ranking[df_ranking['is_smell_token'] == True]['mean_entropy_change']
    non_smell_data = df_ranking[df_ranking['is_smell_token'] == False]['mean_entropy_change']
    if len(smell_data) > 0 and len(non_smell_data) > 0:
        stat, p_value = stats.mannwhitneyu(smell_data, non_smell_data, alternative='less')
        print(f"\nMann-Whitney U Test (H1: Smell tokens have SMALLER mean entropy change):")
        print(f"Statistic (U): {stat}, p-value: {p_value:.4e}")
        if p_value < 0.05:
            print("=> Result: Significant! Smell positions show significantly lower 'certainty of change'.")
        else:
            print("=> Result: Not significant.")
except Exception as e:
    print("Error reading token ranking:", e)
