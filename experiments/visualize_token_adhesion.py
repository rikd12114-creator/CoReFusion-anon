import json
import matplotlib.pyplot as plt
import os
import numpy as np

def visualize_results():
    input_file = os.path.join(os.path.dirname(__file__), "token_adhesion_results.json")
    if not os.path.exists(input_file):
        print(f"File not found: {input_file}")
        return

    with open(input_file, 'r') as f:
        data = json.load(f)

    # Prepare data for plotting
    models = [d['model_id'].split('/')[-1] for d in data] # Shorten model names
    total_vocabs = [d['vocab_size'] for d in data]
    glued_counts = [d['glued_count'] for d in data]
    
    # 1. Bar chart: Total vs Glued Tokens
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(models))
    width = 0.35

    rects1 = ax.bar(x - width/2, total_vocabs, width, label='Total Vocabulary')
    rects2 = ax.bar(x + width/2, glued_counts, width, label='Adhered Tokens')

    ax.set_ylabel('Count')
    ax.set_title('Vocabulary Size vs Adhered Tokens')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend()

    ax.bar_label(rects1, padding=3)
    ax.bar_label(rects2, padding=3)

    plt.tight_layout()
    output_path1 = os.path.join(os.path.dirname(__file__), "vocab_adhesion_comparison.png")
    plt.savefig(output_path1)
    print(f"Saved {output_path1}")
    plt.close()

    # 2. Breakdown by Symbol (Top 10 symbols)
    for d in data:
        model_name = d['model_id'].split('/')[-1]
        symbol_counts = d['symbol_counts']
        # Sort by count desc
        sorted_symbols = sorted(symbol_counts.items(), key=lambda x: x[1], reverse=True)[:15]
        
        syms, counts = zip(*sorted_symbols)
        
        plt.figure(figsize=(12, 6))
        plt.bar(syms, counts, color='orange')
        plt.xlabel('Symbol')
        plt.ylabel('Count of Tokens containing Symbol')
        plt.title(f'Top 15 Symbols present in Adhered Tokens - {model_name}')
        
        output_path2 = os.path.join(os.path.dirname(__file__), f"symbol_breakdown_{model_name}.png")
        plt.savefig(output_path2)
        print(f"Saved {output_path2}")
        plt.close()

if __name__ == "__main__":
    visualize_results()
