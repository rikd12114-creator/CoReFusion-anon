# =============================================================================
# COLAB NOTEBOOK: Partial Masking Noise Mapping (Supplementary Experiment)
# =============================================================================
# Copy each ### CELL ### block into a separate Google Colab cell.
# Runtime: GPU (L4 or A100 recommended). Estimated time: ~1-2h for 50 samples.
# =============================================================================


# ─────────────────────── ### CELL 1: Install & Setup ### ─────────────────────
"""Run this cell first."""

# Install dependencies
import subprocess
subprocess.run(["pip", "install", "-q", "transformers", "accelerate",
                "pandas", "numpy", "scipy", "matplotlib", "tqdm",
                "huggingface_hub"], check=True)

import os, sys, json, re, random, csv, types
from datetime import datetime
from pathlib import Path

# Create directory structure
for d in ["data", "CoRefusion/results/noise_mapping_supplementary",
          "CoRefusion/results/paper_figures_supplement"]:
    Path(d).mkdir(parents=True, exist_ok=True)

# GPU check
import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\nUsing device: {DEVICE}")


# ─────────────────── ### CELL 2: Upload Dataset ### ──────────────────────────
"""
Upload your test_filtered_1024.csv file to Colab.
Either use the Files panel on the left, or run the cell below to use Drive.
"""

# Option A: Upload directly
from google.colab import files
print("Please upload test_filtered_1024.csv ...")
uploaded = files.upload()
# Move to data/
for fname in uploaded:
    os.rename(fname, f"CoRefusion/data/{fname}")
print("Upload complete.")

# ---------- OR ----------

# Option B: Mount Google Drive (comment out Option A if using this)
# from google.colab import drive
# drive.mount('/content/drive')
# DATA_PATH = "/content/drive/MyDrive/YOUR_FOLDER/test_filtered_1024.csv"

DATA_PATH = "CoRefusion/data/test_filtered_1024.csv"
assert os.path.exists(DATA_PATH), f"Dataset not found at {DATA_PATH}"
print(f"Dataset ready: {DATA_PATH}")


# ──────────────── ### CELL 3: Configuration ### ──────────────────────────────
"""Adjust LIMIT and REPEATS based on your available GPU time."""

# ── Model ─────────────────────────────────────────────────────────────────────
MODEL_ID   = "apple/DiffuCoder-7B-Instruct"
MASK_TOKEN = "<|mask|>"
HF_TOKEN   = ""   # ← paste your HuggingFace token here if the repo is gated

# ── Experiment scale ──────────────────────────────────────────────────────────
LIMIT      = 50    # number of code snippets  (50 → ~1-2h on A100)
REPEATS    = 3     # repeats per snippet       (3 gives good variance estimate)
SEED       = 42
MAX_TOKS   = 512

# ── α grid: context masking fractions ────────────────────────────────────────
# Finer grid in the 0–0.3 range where smell→clean differences are expected
ALPHA_GRID = [0.00, 0.05, 0.10, 0.15, 0.20, 0.30,
              0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00]

# ── Smell token sets ──────────────────────────────────────────────────────────
BAD_NAMES = {
    "severe":   ["x", "a", "n", "i"],
    "moderate": ["tmp", "val", "foo", "res"],
    "mild":     ["myVar", "temp1", "result1", "value1"],
}

# Output
TS         = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_CSV = f"CoRefusion/results/noise_mapping_supplementary/partial_masking_{TS}.csv"

print(f"Config set. Output will be saved to: {OUTPUT_CSV}")
print(f"Total forward passes ≈ {LIMIT * REPEATS * (3+1) * len(ALPHA_GRID):,}")


# ──────────────── ### CELL 4: Load Model ### ─────────────────────────────────
"""This cell downloads and loads DiffuCoder-7B. Takes ~5-10 min on first run."""

from transformers import AutoTokenizer, AutoModel

# Mock torchvision (prevents import errors in some model configs)
sys.modules.setdefault("torchvision", types.ModuleType("torchvision"))

print(f"Loading tokenizer from {MODEL_ID} ...")
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID,
    trust_remote_code=True,
    token=HF_TOKEN or None,
)

print(f"Loading model ...")
model = AutoModel.from_pretrained(
    MODEL_ID,
    trust_remote_code=True,
    torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
    token=HF_TOKEN or None,
).to(DEVICE).eval()

mask_id = tokenizer.convert_tokens_to_ids(MASK_TOKEN)
print(f"Model loaded. Mask token '{MASK_TOKEN}' → id={mask_id}")
print(f"Vocab size: {tokenizer.vocab_size:,}")


# ──────────────── ### CELL 5: Core Experiment Functions ### ──────────────────
"""Helper functions – run once, then run Cell 6 to start the experiment."""

import numpy as np
import pandas as pd
from tqdm.notebook import tqdm   # Colab-friendly progress bar


def single_forward(input_ids, attn_mask):
    """One forward pass → logits [1, seq, vocab]."""
    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=attn_mask)
    if hasattr(out, "logits"):
        return out.logits
    raise AttributeError("Cannot extract logits — check model output format")


def metrics_at(logits, pos, current_id, gt_id):
    """Per-position metrics from a logits tensor."""
    lp   = logits[0, pos, :].float()
    prob = torch.softmax(lp, dim=-1)
    lprob= torch.log(prob + 1e-12)

    entropy = -(prob * lprob).sum().item()

    sorted_ids = torch.argsort(prob, descending=True)
    rank_map   = {int(t): r+1 for r, t in enumerate(sorted_ids)}

    vocab = len(prob)
    kl_uniform = float((prob*(lprob - (-np.log(vocab)))).sum())

    return {
        "entropy":            entropy,
        "current_token_rank": rank_map.get(int(current_id), vocab),
        "gt_token_rank":      rank_map.get(int(gt_id),      vocab),
        "gt_token_prob":      float(prob[gt_id]),
        "kl_vs_uniform":      kl_uniform,
        "argmax_token":       int(sorted_ids[0]),
    }


def apply_context_mask(input_ids, mask_id, target_idx, alpha, rng):
    """
    Randomly mask α fraction of context tokens (NOT the target position).
    Returns a new tensor with masked positions set to mask_id.
    """
    ids = input_ids.clone()
    seq = ids[0].tolist()
    n   = len(seq)
    # Eligible positions: everything except position 0 (BOS), n-1 (EOS), target
    eligible = [i for i in range(1, n-1) if i != target_idx]
    k = int(round(alpha * len(eligible)))
    for i in rng.sample(eligible, k) if k else []:
        ids[0, i] = mask_id
    return ids


def find_target_position(tokenizer, full_code, mask_char_pos, seq_len):
    """
    Estimate which token index corresponds to the identifier start position.
    Uses prefix tokenization as a heuristic.
    """
    prefix    = full_code[:mask_char_pos]
    prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
    # + 1 for BOS token that AutoTokenizer usually prepends
    return min(len(prefix_ids) + 1, seq_len - 2)


def probe_sample(tokenizer, mask_id, original_code, gt_name,
                 mask_start_char, sample_id, run_id, rng):
    """
    Run the full α-sweep for one sample across all severity levels.
    Returns a list of result dicts.
    """
    enc = tokenizer(
        original_code, return_tensors="pt",
        truncation=True, max_length=MAX_TOKS,
    ).to(DEVICE)
    input_ids   = enc["input_ids"]
    attn_mask   = enc["attention_mask"]
    seq_len     = input_ids.shape[1]

    target_idx = find_target_position(tokenizer, original_code, mask_start_char, seq_len)

    # Ground-truth token
    gt_tok_ids = tokenizer.encode(gt_name, add_special_tokens=False)
    gt_tok     = gt_tok_ids[0] if gt_tok_ids else tokenizer.unk_token_id

    # Build condition list: (group_key, severity, token_str, token_id)
    conditions = [("control", "none", gt_name, gt_tok)]
    for sev, names in BAD_NAMES.items():
        bad_name   = rng.choice(names)
        bad_tok_ids = tokenizer.encode(bad_name, add_special_tokens=False)
        bad_tok    = bad_tok_ids[0] if bad_tok_ids else tokenizer.unk_token_id
        conditions.append((f"smell_{sev}", sev, bad_name, bad_tok))

    results = []
    for group, severity, token_str, token_id in conditions:
        # Set target position to the smell/clean token
        probe_ids = input_ids.clone()
        probe_ids[0, target_idx] = token_id

        for alpha in ALPHA_GRID:
            masked_ids = apply_context_mask(probe_ids, mask_id, target_idx, alpha, rng)
            try:
                logits  = single_forward(masked_ids, attn_mask)
                m       = metrics_at(logits, target_idx, token_id, gt_tok)
            except Exception as e:
                m = {"entropy": None, "current_token_rank": None,
                     "gt_token_rank": None, "gt_token_prob": None,
                     "kl_vs_uniform": None, "argmax_token": None}
            results.append({
                "sample_id": sample_id, "run_id": run_id,
                "group": group, "severity": severity,
                "token": token_str, "gt_name": gt_name,
                "alpha": alpha,
                **m,
            })
    return results


print("Helper functions defined ✓")


# ──────────────── ### CELL 6: Run the Experiment ### ─────────────────────────
"""
Main loop. Saves results incrementally so you don't lose data if Colab disconnects.
Expected time: ~1-2h on A100 for LIMIT=50, REPEATS=3.
"""

rng_main = random.Random(SEED)

df_data = pd.read_csv(DATA_PATH, header=None, names=["id", "X", "y"],
                      nrows=LIMIT)
print(f"Loaded {len(df_data)} samples from dataset.")

FIELDNAMES = ["sample_id", "run_id", "group", "severity", "token", "gt_name",
              "alpha", "entropy", "current_token_rank", "gt_token_rank",
              "gt_token_prob", "kl_vs_uniform", "argmax_token"]

total_written = 0
with open(OUTPUT_CSV, "w", newline="") as fout:
    writer = csv.DictWriter(fout, fieldnames=FIELDNAMES, extrasaction="ignore")
    writer.writeheader()

    for run_id in tqdm(range(REPEATS), desc="Runs"):
        for _, row in tqdm(df_data.iterrows(), total=len(df_data),
                           desc=f"Run {run_id+1}/{REPEATS}", leave=False):
            code_with_mask = str(row["X"])
            gt_name        = str(row["y"])
            sample_id      = row["id"]

            mask_pos = code_with_mask.find("[MASK]")
            if mask_pos == -1:
                continue
            original_code = code_with_mask.replace("[MASK]", gt_name, 1)

            try:
                batch = probe_sample(
                    tokenizer, mask_id,
                    original_code, gt_name, mask_pos,
                    sample_id, run_id, rng_main,
                )
                writer.writerows(batch)
                total_written += len(batch)
                fout.flush()
            except Exception as e:
                print(f"\n[Skip] sample={sample_id} run={run_id}: {e}")

print(f"\nDone! {total_written:,} rows → {OUTPUT_CSV}")


# ──── Optional: save to Google Drive ──────────────────────────────────────────
# from google.colab import drive
# drive.mount('/content/drive')
# import shutil
# shutil.copy(OUTPUT_CSV, "/content/drive/MyDrive/YOUR_FOLDER/")
# print("Saved to Drive.")


# ──────────────── ### CELL 7: Quick Sanity Check ### ─────────────────────────
"""Verify data looks reasonable before running full analysis."""

df_result = pd.read_csv(OUTPUT_CSV)
print(f"Loaded {len(df_result):,} rows")
print(f"Groups: {df_result['group'].unique().tolist()}")
print(f"Alphas: {sorted(df_result['alpha'].unique().tolist())}")
print()
print("Mean entropy by group × alpha (pivoted):")
pivot = df_result.groupby(["group", "alpha"])["entropy"].mean().unstack("alpha")
print(pivot.round(3).to_string())


# ──────────────── ### CELL 8: Analysis & Paper Figures ### ───────────────────
"""Full analysis: α-entropy curves, KL decay, α* mapping, stats. Saves PDFs."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy import stats as scipy_stats
from scipy.interpolate import interp1d

OUTDIR_FIG = "CoRefusion/results/paper_figures_supplement"
DPI = 300

matplotlib.rcParams.update({
    "font.family":       "serif",
    "font.serif":        ["Times New Roman", "DejaVu Serif"],
    "font.size":         8,
    "axes.titlesize":    8,
    "axes.labelsize":    8,
    "xtick.labelsize":   7,
    "ytick.labelsize":   7,
    "legend.fontsize":   7,
    "axes.spines.top":  False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.linewidth":    0.4,
    "grid.alpha":        0.4,
})

C = {
    "smell_severe":   "#FF7F0E",
    "smell_moderate": "#BCBD22",
    "smell_mild":     "#2CA02C",
    "control":        "#1F77B4",
}
MARKERS = {"smell_severe": "^", "smell_moderate": "s",
           "smell_mild":   "D", "control":        "o"}
LABELS  = {
    "smell_severe":   "Severe smell (x, a)",
    "smell_moderate": "Moderate smell (tmp, val)",
    "smell_mild":     "Mild smell (myVar)",
    "control":        "Clean code (GT)",
}

# ── helpers ──────────────────────────────────────────────────────────────────
df = pd.read_csv(OUTPUT_CSV)
agg_ent = (df.groupby(["group", "alpha"])["entropy"]
             .agg(mean="mean", se=lambda x: x.sem())
             .reset_index())
agg_kl  = (df.groupby(["group", "alpha"])["kl_vs_uniform"]
             .agg(mean="mean", se=lambda x: x.sem())
             .reset_index())

GROUP_ORDER = ["control", "smell_mild", "smell_moderate", "smell_severe"]

# ── FIG S1: α-Entropy curves ─────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(5.0, 3.2))

ref_at_1 = agg_ent[agg_ent["alpha"] == 1.0]["mean"].mean()  # full-mask reference

for grp in GROUP_ORDER:
    sub = agg_ent[agg_ent["group"] == grp].sort_values("alpha")
    if sub.empty:
        continue
    x, y, ye = sub["alpha"].values, sub["mean"].values, sub["se"].values
    ax.plot(x, y, color=C[grp], lw=1.4, marker=MARKERS[grp],
            markersize=4.5, label=LABELS[grp], zorder=5)
    ax.fill_between(x, y-ye, y+ye, color=C[grp], alpha=0.12)

ax.axhline(ref_at_1, color="#888", ls=":", lw=1.1,
           label=f"Full-mask entropy ({ref_at_1:.3f} nats)")

# Mark crossing points α*
alpha_stars = {}
for grp in GROUP_ORDER:
    if grp == "control":
        continue
    sub = agg_ent[agg_ent["group"] == grp].sort_values("alpha")
    if sub.empty or len(sub) < 3:
        continue
    x, y = sub["alpha"].values, sub["mean"].values
    try:
        f_interp   = interp1d(x, y, kind="linear", fill_value="extrapolate")
        alphas_fine = np.linspace(0, 1, 10000)
        ys_fine    = f_interp(alphas_fine)
        idx    = np.argmin(np.abs(ys_fine - ref_at_1))
        a_star = float(alphas_fine[idx])
        alpha_stars[grp] = a_star
        ax.axvline(a_star, color=C[grp], ls="--", lw=0.8, alpha=0.65)
        ax.annotate(f"α*={a_star:.2f}", xy=(a_star, ref_at_1),
                    xytext=(a_star+0.02, ref_at_1*0.95),
                    fontsize=6, color=C[grp], fontweight="bold")
    except Exception:
        pass

ax.set_xlabel("Context masking fraction α  (0 = no masking,  1 = fully masked)")
ax.set_ylabel("Shannon entropy at target position (nats)")
ax.set_title("(S1)  Partial masking bridges the two noise regimes\n"
             "α* = context masking fraction where smell entropy = mask entropy",
             fontsize=8, fontweight="bold")
ax.legend(loc="upper left", frameon=False, fontsize=7)
ax.set_xlim(-0.02, 1.02)
ax.set_ylim(bottom=0)
fig.tight_layout(pad=0.4)
fig.savefig(f"{OUTDIR_FIG}/figS1_alpha_entropy.pdf", bbox_inches="tight")
fig.savefig(f"{OUTDIR_FIG}/figS1_alpha_entropy.png", bbox_inches="tight", dpi=DPI)
plt.show()
print("✓ figS1_alpha_entropy")

# ── FIG S2: KL decay ─────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(5.0, 3.0))
for grp in GROUP_ORDER:
    sub = agg_kl[agg_kl["group"] == grp].sort_values("alpha")
    if sub.empty:
        continue
    x, y, ye = sub["alpha"].values, sub["mean"].values, sub["se"].values
    ax.plot(x, y, color=C[grp], lw=1.4, marker=MARKERS[grp],
            markersize=4.5, label=LABELS[grp])
    ax.fill_between(x, y-ye, y+ye, color=C[grp], alpha=0.12)
ax.set_xlabel("Context masking fraction α")
ax.set_ylabel("KL divergence from uniform (nats)\n↑ more confident   ↓ more noise-like")
ax.set_title("(S2)  Model confidence decays as context is masked\n"
             "Smell tokens converge toward mask-token uncertainty",
             fontsize=8, fontweight="bold")
ax.legend(loc="upper right", frameon=False, fontsize=7)
ax.set_xlim(-0.02, 1.02)
fig.tight_layout(pad=0.4)
fig.savefig(f"{OUTDIR_FIG}/figS2_kl_decay.pdf", bbox_inches="tight")
fig.savefig(f"{OUTDIR_FIG}/figS2_kl_decay.png", bbox_inches="tight", dpi=DPI)
plt.show()
print("✓ figS2_kl_decay")

# ── FIG S3: α* bar chart ─────────────────────────────────────────────────────
if alpha_stars:
    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    sev_order  = ["smell_severe", "smell_moderate", "smell_mild"]
    xlabels    = ["Severe\nSmell", "Moderate\nSmell", "Mild\nSmell"]
    vals       = [alpha_stars.get(k, np.nan) for k in sev_order]
    cols       = [C[k] for k in sev_order]

    bars = ax.bar(range(3), vals, color=cols, width=0.55,
                  edgecolor="white", linewidth=0.6, alpha=0.82)
    for bar, val in zip(bars, vals):
        if not np.isnan(val):
            ax.text(bar.get_x() + bar.get_width()/2,
                    val + 0.02, f"α*={val:.2f}",
                    ha="center", va="bottom", fontsize=7, fontweight="bold")

    ax.axhline(1.0, color="#888", ls=":", lw=0.9, label="α=1 (fully masked)")
    ax.set_xticks(range(3))
    ax.set_xticklabels(xlabels)
    ax.set_ylabel("α*  (↓ = more noise-like at 0 context)")
    ax.set_title("(S3)  Equivalent noise masking fraction α*\n"
                 "Lower α* → code smell is more inherently noise-like",
                 fontsize=8, fontweight="bold")
    ax.set_ylim(0, 1.2)
    ax.legend(frameon=False, fontsize=6.5)
    fig.tight_layout(pad=0.4)
    fig.savefig(f"{OUTDIR_FIG}/figS3_alpha_star.pdf", bbox_inches="tight")
    fig.savefig(f"{OUTDIR_FIG}/figS3_alpha_star.png", bbox_inches="tight", dpi=DPI)
    plt.show()
    print("✓ figS3_alpha_star")

# ── Print α* interpretation ───────────────────────────────────────────────────
print("\n" + "="*60)
print("NOISE MAPPING RESULTS — α* Crossing Points")
print("="*60)
print(f"Reference entropy (full mask, α=1.0): {ref_at_1:.4f} nats\n")
for grp, a_star in sorted(alpha_stars.items(), key=lambda x: x[1]):
    noise_pct = a_star * 100
    print(f"  {LABELS[grp]:<35}: α* = {a_star:.3f}  "
          f"= {noise_pct:.1f}% context masking needed")
print()
print("Interpretation:")
print("  Lower α* → smell token already 'behaves like noise' with less context help")
print("  Higher α* → smell token needs more destruction to reach full-mask uncertainty")


# ──────────────── ### CELL 9: Statistical Tests ### ──────────────────────────
"""Pairwise tests at each α level."""

print("Statistical tests across α levels\n" + "="*55)

report_rows = []
for alpha in sorted(df["alpha"].unique()):
    sub  = df[df["alpha"] == alpha]
    ctrl = sub[sub["group"] == "control"]["entropy"].dropna()
    for grp in ["smell_severe", "smell_moderate", "smell_mild"]:
        g = sub[sub["group"] == grp]["entropy"].dropna()
        if len(g) < 3 or len(ctrl) < 3:
            continue
        t, p = scipy_stats.ttest_ind(g, ctrl, equal_var=False)
        d    = (g.mean() - ctrl.mean()) / (
                    np.sqrt((g.std()**2 + ctrl.std()**2) / 2) + 1e-12)
        sig  = "***" if p < 0.001 else "**" if p<0.01 else "*" if p<0.05 else "ns"
        report_rows.append({
            "alpha": alpha, "group": grp,
            "mean_smell": g.mean(), "mean_control": ctrl.mean(),
            "delta": g.mean()-ctrl.mean(),
            "cohen_d": d, "p": p, "sig": sig,
        })

stats_df = pd.DataFrame(report_rows)
print(stats_df.to_string(index=False))

stats_df.to_csv(f"CoRefusion/{OUTDIR_FIG}/stats_supplement.csv", index=False)
print(f"\nStats saved → {OUTDIR_FIG}/stats_supplement.csv")


# ──────────────── ### CELL 10: Download Results ### ──────────────────────────
"""Download all results to your local machine."""

import shutil, zipfile
from google.colab import files

zip_path = "supplementary_results.zip"
with zipfile.ZipFile(zip_path, "w") as zf:
    for d in [OUTDIR_FIG, "CoRefusion/results/noise_mapping_supplementary"]:
        for f_path in Path(d).rglob("*"):
            if f_path.is_file():
                zf.write(f_path)

files.download(zip_path)
print(f"Downloaded: {zip_path}")
