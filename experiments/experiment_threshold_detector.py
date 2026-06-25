"""
Threshold Detector: Entropy-based Code Smell Signal for dLLM Refactoring

Based on Experiment C findings:
- Smell tokens show ~2x higher mean entropy fluctuation (ΔH) vs context tokens
- Effect size d=0.81 (large), p < 8.14e-22

This script:
1. Loads pre-computed token_ranking CSV from Experiment C
2. Fits a threshold (τ) using the natural separation in ΔEntropy distributions
3. Evaluates detection performance (Precision, Recall, F1, AUROC)
4. Generates a "Smell Heatmap" showing which tokens would be flagged
5. Demonstrates what this threshold would look like as a decision tool
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.metrics import (
    roc_auc_score, roc_curve, precision_recall_curve,
    classification_report, confusion_matrix
)
from sklearn.mixture import GaussianMixture
import os

# ======================================================
# Config: point this to your Experiment C output CSV
# ======================================================
RANKING_CSV = "results/abc/token_ranking_20260312_230225.csv"
RESULTS_DIR = "results/threshold_analysis"
os.makedirs(RESULTS_DIR, exist_ok=True)

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({'font.size': 13, 'axes.titlesize': 15, 'figure.dpi': 200})
SMELL_COLOR   = '#e74c3c'
NORMAL_COLOR  = '#3498db'

print("="*60)
print("  Entropy-based Smell Threshold Construction")
print("="*60)

# ─────────────────────────────────────────────────────────────
# Step 1: Load Data
# ─────────────────────────────────────────────────────────────
df = pd.read_csv(RANKING_CSV)
df['label'] = df['is_smell_token'].astype(int)  # 1=smell, 0=context
feature = 'mean_entropy_change'

smell_vals   = df[df['label'] == 1][feature].values
context_vals = df[df['label'] == 0][feature].values

print(f"\n[Data] Smell tokens: {len(smell_vals):,}  |  Context tokens: {len(context_vals):,}")
print(f"  Smell   mean ΔH: {smell_vals.mean():.4f}  ± {smell_vals.std():.4f}")
print(f"  Context mean ΔH: {context_vals.mean():.4f}  ± {context_vals.std():.4f}")

# ─────────────────────────────────────────────────────────────
# Step 2: Threshold Construction — 3 Methods
# ─────────────────────────────────────────────────────────────

# Method A: Percentile-based (95th percentile of context distribution)
tau_percentile = np.percentile(context_vals, 80)
print(f"\n[Method A] Percentile Threshold (80th pct of context): τ = {tau_percentile:.4f}")

# Method B: Midpoint between means
tau_midpoint = (smell_vals.mean() + context_vals.mean()) / 2
print(f"[Method B] Midpoint-of-Means Threshold:                τ = {tau_midpoint:.4f}")

# Method C: GMM-based decision boundary (fit 2-component GMM on all data)
X = df[[feature]].values
gmm = GaussianMixture(n_components=2, random_state=42)
gmm.fit(X)
means = gmm.means_.flatten()
gmm_boundary = means.mean()  # midpoint of two GMM centers
print(f"[Method C] GMM Decision Boundary (2-component):        τ = {gmm_boundary:.4f}")

# Use Method B (midpoint) as our primary threshold
tau = tau_midpoint

# ─────────────────────────────────────────────────────────────
# Step 3: Evaluate Threshold Performance
# ─────────────────────────────────────────────────────────────
y_true = df['label'].values
y_score = df[feature].values
y_pred  = (y_score >= tau).astype(int)

print(f"\n[Evaluation] Using Primary Threshold τ = {tau:.4f}")
print(classification_report(y_true, y_pred, target_names=['Context', 'Smell']))

auroc = roc_auc_score(y_true, y_score)
print(f"  AUROC: {auroc:.4f}")

# ─────────────────────────────────────────────────────────────
# Step 4: Visualizations
# ─────────────────────────────────────────────────────────────

# --- Plot 1: Distribution + Threshold Overlay ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
ax.hist(context_vals, bins=80, color=NORMAL_COLOR, alpha=0.5, label='Context Token', density=True)
ax.hist(smell_vals,   bins=40, color=SMELL_COLOR,  alpha=0.6, label='Smell Token',   density=True)
ax.axvline(tau_percentile, color='orange',  ls='--', lw=2, label=f'Method A τ={tau_percentile:.3f}')
ax.axvline(tau_midpoint,   color='green',   ls='--', lw=2, label=f'Method B τ={tau_midpoint:.3f}')
ax.axvline(gmm_boundary,   color='purple',  ls='--', lw=2, label=f'Method C τ={gmm_boundary:.3f}')
ax.set_title('Entropy ΔH Distribution + Decision Thresholds')
ax.set_xlabel('Mean Absolute ΔEntropy')
ax.set_ylabel('Density')
ax.legend(fontsize=9)
ax.set_xlim(0, min(3.0, np.percentile(df[feature], 99)))

# --- Plot 2: ROC Curve ---
ax2 = axes[1]
fpr, tpr, _ = roc_curve(y_true, y_score)
ax2.plot(fpr, tpr, color=SMELL_COLOR, lw=2, label=f'ROC (AUC = {auroc:.3f})')
ax2.plot([0,1],[0,1], 'k--', lw=1)
ax2.set_xlabel('False Positive Rate')
ax2.set_ylabel('True Positive Rate')
ax2.set_title('ROC Curve — Entropy ΔH as Smell Detector')
ax2.legend()

plt.tight_layout()
plt.savefig(f"{RESULTS_DIR}/threshold_distributions_roc.png", bbox_inches='tight')
plt.savefig(f"{RESULTS_DIR}/threshold_distributions_roc.pdf", bbox_inches='tight')
plt.close()
print(f"\n  -> Saved: threshold_distributions_roc.png")

# --- Plot 3: Precision-Recall Curve ---
plt.figure(figsize=(7, 5))
prec, rec, pr_thresh = precision_recall_curve(y_true, y_score)
plt.plot(rec, prec, color=SMELL_COLOR, lw=2)
plt.axhline(df['label'].mean(), color='grey', ls='--', label='Baseline (prevalence)')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve — Smell Token Detection')
plt.legend()
plt.tight_layout()
plt.savefig(f"{RESULTS_DIR}/precision_recall_curve.png", bbox_inches='tight')
plt.close()
print(f"  -> Saved: precision_recall_curve.png")

# --- Plot 4: Threshold Sensitivity Analysis (Precision/Recall vs τ) ---
thresholds = np.linspace(0, 1.5, 200)
precisions, recalls, f1s = [], [], []
for t in thresholds:
    preds = (y_score >= t).astype(int)
    tp = ((preds == 1) & (y_true == 1)).sum()
    fp = ((preds == 1) & (y_true == 0)).sum()
    fn = ((preds == 0) & (y_true == 1)).sum()
    p = tp / (tp + fp) if (tp + fp) > 0 else 0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0
    f = 2*p*r/(p+r) if (p+r) > 0 else 0
    precisions.append(p)
    recalls.append(r)
    f1s.append(f)

best_tau_idx = np.argmax(f1s)
best_tau = thresholds[best_tau_idx]

plt.figure(figsize=(10, 5))
plt.plot(thresholds, precisions, label='Precision', color='#2ecc71', lw=2)
plt.plot(thresholds, recalls,    label='Recall',    color='#3498db', lw=2)
plt.plot(thresholds, f1s,        label='F1 Score',  color='#e74c3c', lw=2)
plt.axvline(best_tau, color='black', ls='--', lw=2, label=f'Best F1 τ = {best_tau:.3f}')
plt.axvline(tau_midpoint, color='green', ls=':', lw=2, label=f'Midpoint τ = {tau_midpoint:.3f}')
plt.xlabel('Threshold τ on Mean ΔEntropy')
plt.ylabel('Score')
plt.title('Threshold Sensitivity: Precision, Recall & F1 vs τ')
plt.legend()
plt.tight_layout()
plt.savefig(f"{RESULTS_DIR}/threshold_sensitivity.png", bbox_inches='tight')
plt.savefig(f"{RESULTS_DIR}/threshold_sensitivity.pdf", bbox_inches='tight')
plt.close()
print(f"  -> Saved: threshold_sensitivity.png")

print(f"\n[Best Threshold] τ* = {best_tau:.4f} (maximizes F1 = {max(f1s):.4f})")
print(f"  At τ*: Precision={precisions[best_tau_idx]:.3f}  Recall={recalls[best_tau_idx]:.3f}")

# ─────────────────────────────────────────────────────────────
# Step 5: Final Summary
# ─────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("  THRESHOLD CONSTRUCTION SUMMARY")
print("="*60)
print(f"  Method A (80th Pctile of Context):  τ = {tau_percentile:.4f}")
print(f"  Method B (Midpoint of Means):       τ = {tau_midpoint:.4f}")
print(f"  Method C (GMM Boundary):            τ = {gmm_boundary:.4f}")
print(f"  Method D (Optimal F1):              τ* = {best_tau:.4f}  [RECOMMENDED]")
print(f"\n  AUROC of raw ΔEntropy feature: {auroc:.4f}")
print(f"\n  Interpretation:")
print(f"  -> Tokens with ΔH >= τ* = {best_tau:.3f} are flagged as potential SMELL tokens")
print(f"  -> These are positions where the dLLM internally 'debates' its prediction,")
print(f"     indicating awareness of semantic incorrectness (naming quality issue).")
print("="*60)
