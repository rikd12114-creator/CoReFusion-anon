import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from tqdm import tqdm
from datetime import datetime
from transformers import AutoTokenizer, AutoModel
import umap
from sklearn.metrics.pairwise import cosine_similarity
import re

# ==============================================================================
# Environment Setting: Mock torchvision for DiffuCoder compatibility
# ==============================================================================
class MockModule:
    def __getattr__(self, name): return MockModule()
    def __call__(self, *args, **kwargs): return MockModule()

sys.modules['torchvision'] = MockModule()
sys.modules['torchvision.ops'] = MockModule()
sys.modules['torchvision.transforms'] = MockModule()

if not hasattr(torch.ops, 'torchvision'):
    class DummyOps:
        def nms(*args, **kwargs): return torch.tensor([])
    torch.ops.torchvision = DummyOps()
# ==============================================================================

# Configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "apple/DiffuCoder-7B-Instruct"
DATA_PATH = "data/test.csv" # Or data/test_filtered_1024.csv based on your preference
RESULTS_DIR = "results"

# Code Smell parameters
SMELL_NAME = "tmp" # The "smelly code" identifier name used to replace [MASK]

# For Diffusion Trajectory
TOTAL_STEPS = 32

def find_subsequence_indices(sequence, subsequence):
    """Finds the boundary indices of a tokenized sequence inside a larger sequence."""
    seq_len = len(sequence)
    sub_len = len(subsequence)
    for i in range(seq_len - sub_len + 1):
        if sequence[i: i + sub_len] == subsequence:
            return i, i + sub_len
    return None

def get_num_transfer_tokens(mask_index, steps):
    mask_num = mask_index.sum(dim=1, keepdim=True)
    base = mask_num // steps
    remainder = mask_num % steps
    num_transfer_tokens = torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64) + base
    for i in range(mask_num.size(0)):
        num_transfer_tokens[i, :remainder[i]] += 1
    return num_transfer_tokens

def add_gumbel_noise(logits, temperature):
    if temperature == 0:
        return logits
    logits = logits.to(torch.float32)
    noise = torch.rand_like(logits, dtype=torch.float32)
    gumbel_noise = (- torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise

def compute_umap_and_similarity(results_df, output_prefix):
    """Helper snippet to compute UMAP projection and summarize edit signals."""
    if results_df.empty: return
    
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # 1. UMAP over Internal Representations
    features = np.stack(results_df['hidden_state'].values)
    
    # UMAP: lower n_neighbors -> focus on local structure; metric='cosine' matches hidden state analysis
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=min(15, len(features) - 1),
        min_dist=0.1,
        metric='cosine',
        random_state=42
    )
    features_umap = reducer.fit_transform(features)
    results_df = results_df.copy()
    results_df['umap_1'] = features_umap[:, 0]
    results_df['umap_2'] = features_umap[:, 1]
    
    # Color map: smell=red, gt=blue  
    palette = {'smelly': '#e74c3c', 'gt': '#3498db'}
    markers  = {'smelly': 'X', 'gt': 'o'}
    
    plt.figure(figsize=(11, 7))
    for cond, grp in results_df.groupby('condition'):
        plt.scatter(
            grp['umap_1'], grp['umap_2'],
            label=cond,
            c=palette.get(cond, 'grey'),
            marker=markers.get(cond, 'o'),
            alpha=0.7, s=60, edgecolors='white', linewidth=0.4
        )
    plt.title(f'UMAP of Internal Representations — {output_prefix}\n'
              f'(Smell Token vs. GT Token Hidden States)', fontsize=14)
    plt.xlabel('UMAP Dimension 1')
    plt.ylabel('UMAP Dimension 2')
    plt.legend(title='Condition', fontsize=11)
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/umap_{output_prefix}.png", dpi=300)
    plt.savefig(f"{RESULTS_DIR}/umap_{output_prefix}.pdf", bbox_inches='tight')
    plt.close()
    print(f"  -> UMAP plot saved: umap_{output_prefix}.png")

    # 2. Extract "Edit Signal" Distance (L2 norm) or Cosine Similarity tracking
    edit_signals = []
    for sample_id in results_df['sample_id'].unique():
        sample_df = results_df[results_df['sample_id'] == sample_id]
        for step in sample_df['step'].unique():
            hs_smell = sample_df[(sample_df['step'] == step) & (sample_df['condition'] == 'smelly')]['hidden_state']
            hs_gt = sample_df[(sample_df['step'] == step) & (sample_df['condition'] == 'gt')]['hidden_state']
            
            if not hs_smell.empty and not hs_gt.empty:
                v_smell = hs_smell.values[0]
                v_gt = hs_gt.values[0]
                cos_sim = cosine_similarity([v_smell], [v_gt])[0][0]
                l2_dist = np.linalg.norm(v_smell - v_gt)
                edit_signals.append({'sample_id': sample_id, 'step': step, 'cosine_sim': cos_sim, 'l2_dist': l2_dist})
                
    if edit_signals:
        df_sig = pd.DataFrame(edit_signals)
        df_sig.to_csv(f"{RESULTS_DIR}/edit_signals_{output_prefix}.csv", index=False)

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    print(f"Loading tokenizer and model: {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        MODEL_NAME, 
        torch_dtype=torch.bfloat16, 
        trust_remote_code=True
    ).to(DEVICE).eval()
    mask_token_id = tokenizer.convert_tokens_to_ids("<|mask|>")
    
    print(f"Loading data from {DATA_PATH}...")
    try:
        df = pd.read_csv(DATA_PATH, header=None, names=['id', 'masked_code', 'target'])
    except Exception as e:
        print(f"Dataset load error: {e}. Ensure path is correct.")
        return
    
    # Run test on smaller subset (e.g. 100 samples)
    df = df.head(20)
    all_hidden_states = []

    print("Extracting Internal Representations...")
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        try:
            sample_id = row['id']
            masked_code = str(row['masked_code'])
            target_name = str(row['target']).strip()

            if '[MASK]' not in masked_code:
                continue

            code_smell = masked_code.replace("[MASK]", SMELL_NAME)
            code_gt = masked_code.replace("[MASK]", target_name)

            conditions = [('smelly', code_smell, SMELL_NAME), ('gt', code_gt, target_name)]

            for cond_label, code_str, token_name in conditions:
                # Tokenize Condition Text
                inputs = tokenizer(code_str, return_tensors="pt").to(DEVICE)
                input_ids = inputs.input_ids
                attention_mask = inputs.attention_mask

                # Locate condition token
                target_toks = tokenizer.encode(token_name, add_special_tokens=False)
                target_pos = find_subsequence_indices(input_ids[0].tolist(), target_toks)

                if target_pos is None:
                    # Add a leading space if initial tokenization missed
                    target_toks_space = tokenizer.encode(" " + token_name, add_special_tokens=False)
                    target_pos = find_subsequence_indices(input_ids[0].tolist(), target_toks_space)

                if target_pos is None:
                    continue

                start_idx, end_idx = target_pos

                # -----------------------------------------------------------------
                # 内部表征分析 Method 1: Cross-Layer activations (Zero Diffusion) 
                #   Often 'denoising steps' are confused with layers in standard LM representations.
                # -----------------------------------------------------------------
                with torch.no_grad():
                    outputs = model(input_ids, attention_mask=attention_mask.bool(), output_hidden_states=True)
                    # Average embedded token representations at target position
                    layers_hs = [hs[0, start_idx:end_idx, :].mean(dim=0).float().cpu().numpy() for hs in outputs.hidden_states]

                for layer_i, hs_val in enumerate(layers_hs):
                    all_hidden_states.append({
                        'sample_id': sample_id,
                        'condition': cond_label,
                        'step': f"layer_{layer_i}",
                        'hidden_state': hs_val
                    })

                # -----------------------------------------------------------------
                # 内部表征分析 Method 2: Denoising Step Trajectory (Discrete Diffusion)
                #   We fix the smell/GT token position as input condition, and fully
                #   MASK the surrounding context. Over T steps, as the context is 
                #   denoised, we capture the internal state of the smell/GT token.
                # -----------------------------------------------------------------
                x = input_ids.clone()
                context_mask = torch.ones_like(x, dtype=torch.bool)
                context_mask[0, start_idx:end_idx] = False # keep smell/GT fixed!
                # Do NOT mask special tokens if applicable, but for code sequences it's fine.
                x[context_mask] = mask_token_id 

                num_transfer_tokens = get_num_transfer_tokens(context_mask, TOTAL_STEPS)

                for step_i in range(TOTAL_STEPS):
                    with torch.no_grad():
                        current_mask_index = (x == mask_token_id)
                        diff_outputs = model(x, attention_mask=attention_mask.bool(), output_hidden_states=True)

                        # Capture the Condition Token's LAST LAYER hidden state at step T
                        step_token_hs = diff_outputs.hidden_states[-1][0, start_idx:end_idx, :].mean(dim=0).float().cpu().numpy()

                        all_hidden_states.append({
                            'sample_id': sample_id,
                            'condition': cond_label,
                            'step': f"diff_step_{step_i}",
                            'hidden_state': step_token_hs
                        })

                        # Compute next step via discrete Gumbel diffusion
                        if current_mask_index.any():
                            logits = diff_outputs.logits
                            logits_with_noise = add_gumbel_noise(logits, temperature=0.3)
                            x0 = torch.argmax(logits_with_noise, dim=-1)
                            p_all = F.softmax(logits.float(), dim=-1)
                            x0_p = torch.squeeze(torch.gather(p_all, dim=-1, index=torch.unsqueeze(x0, -1)), -1)
                            x0 = torch.where(current_mask_index, x0, x)
                            confidence = torch.where(current_mask_index, x0_p, torch.tensor(-np.inf, device=DEVICE))

                            transfer_index = torch.zeros_like(x0, dtype=torch.bool)
                            k = num_transfer_tokens[0, step_i].item() if num_transfer_tokens.shape[1] > step_i else 0
                            if k > 0:
                                _, sel = torch.topk(confidence[0], k=int(k))
                                transfer_index[0, sel] = True
                                x[transfer_index] = x0[transfer_index]

        except Exception as e:
            if 'out of memory' in str(e).lower() or 'oom' in str(e).lower():
                print(f'\n[OOM] Skipping sample {row.get("id", "unknown")}...')
                torch.cuda.empty_cache()
                import gc; gc.collect()
                continue
            else:
                raise e

    # Analysis Export
    print("\nSaving results and creating PCA/Similarity metrics...")
    df_results = pd.DataFrame(all_hidden_states)
    
    # Save raw hidden states if needed (warning: can be large)
    # df_results.to_pickle(f"{RESULTS_DIR}/internal_representations.pkl")
    
    # 1. Analyze Layer-wise edit signals
    df_layers = df_results[df_results['step'].str.contains('layer_')].copy()
    compute_umap_and_similarity(df_layers, "layer_wise")
    
    # 2. Analyze Diffusion-Step-wise edit signals
    df_diff = df_results[df_results['step'].str.contains('diff_step_')].copy()
    compute_umap_and_similarity(df_diff, "diffusion_steps")
    
    print("\nExperiment completed! You can check the PCA and Cosine Similarity outputs in the 'results' directory.")

if __name__ == "__main__":
    main()
