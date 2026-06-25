import pandas as pd
import numpy as np

def analyze_csv(filepath, title):
    print(f"\n{'='*20} {title} {'='*20}")
    try:
        df = pd.read_csv(filepath)
        
        # Sort properly by numerical value in the step
        def extract_num(x):
            try:
                # e.g. "layer_0" -> 0, "diff_step_0" -> 0
                return int(x.split('_')[-1])
            except:
                return 0
                
        # Calculate means
        summary = df.groupby('step')[['cosine_sim', 'l2_dist']].mean().reset_index()
        summary['sort_key'] = summary['step'].apply(extract_num)
        summary = summary.sort_values('sort_key').drop('sort_key', axis=1).set_index('step')
        
        print(summary.head(10))
        print("...")
        print(summary.tail(10))
        
        first = summary.iloc[0]
        last = summary.iloc[-1]
        peak_cos = summary['cosine_sim'].max()
        peak_cos_step = summary['cosine_sim'].idxmax()
        
        print(f"\n--- Summary stats for {title} ---")
        print(f"Total samples recorded: {df['sample_id'].nunique()}")
        print(f"Initial Cosine Similarity: {first['cosine_sim']:.4f}")
        print(f"Final Cosine Similarity: {last['cosine_sim']:.4f}")
        print(f"Peak Cosine Similarity: {peak_cos:.4f} at {peak_cos_step}")
        print(f"Initial L2 Distance: {first['l2_dist']:.4f}")
        print(f"Final L2 Distance: {last['l2_dist']:.4f}")
        print(f"Average L2 Distance overall: {summary['l2_dist'].mean():.4f}")
        
    except Exception as e:
        print(f"Error analyzing {filepath}: {e}")

analyze_csv("results/edit_signal/edit_signals_layer_wise.csv", "Layer-wise Analysis (Internal Error Signal)")
analyze_csv("results/edit_signal/edit_signals_diffusion_steps.csv", "Diffusion Step-wise Analysis")

