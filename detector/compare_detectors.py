"""
Bad Naming Detector — Comparative Evaluation Framework
=======================================================

Compares four detection strategies for identifying "bad" variable names
(i.e., names that a well-named ground-truth identifier would not be predicted
by the model, or names that are generic / context-free):

  1. RANDOM GUESSING
     Baseline: randomly assigns SMELL / CLEAN with a fixed prior.

  2. ML CLASSIFIER (heuristic-feature-based)
     Uses hand-crafted lexical & structural features (name length, char-class
     mix, entropy, CamelCase/snake_case ratio, smell-token membership …)
     fed into a Logistic Regression / Random Forest trained on the benchmark
     ground-truth labels (correct=False → bad name, correct=True → good name).

  3. AR MODEL (FIM logit-based detector)
     Uses any AR model from the benchmark CSVs as a proxy:
     if the model's prediction == ground_truth  → CLEAN
     else                                        → SMELL
     This uses the pre-computed benchmark results, so no extra inference is
     needed on your machine.

  4. OUR METHOD (Over-confidence + Context Sensitivity, optimised)
     Combines:
       • OC probe  – rank of gt vs smell probes in the diffusion model's
                     softmax at the masked position (requires model)
       • CTX probe – entropy rise ΔH as masking fraction increases
     The optimised version adds:
       • Calibrated per-regime score functions
       • Soft thresholds with sigmoid interpolation
       • Optional cross-model ensembling

Usage (no model – methods 1-3 only, uses pre-computed benchmark CSVs):
    python detector/compare_detectors.py --mode offline

Usage (full – all 4 methods, needs diffusion model access):
    python detector/compare_detectors.py --mode full \\
        --model DiffuCoder-7B-Base \\
        --max-samples 200

Usage (evaluate just AR baselines):
    python detector/compare_detectors.py --mode ar-only

Output:
    detector/comparison_results/
        comparison_metrics.csv
        comparison_metrics.json
        roc_curves.png
        pr_curves.png
        confusion_matrices.png
        bar_chart.png
"""

from __future__ import annotations

import os
import sys
import csv
import json
import math
import random
import argparse
import warnings
import re
import ast
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── Optional heavy imports ────────────────────────────────────────────────────
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from matplotlib.patches import Patch
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("[WARN] matplotlib not found – plots disabled.")

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import cross_val_score, StratifiedKFold
    from sklearn.metrics import (
        roc_auc_score, average_precision_score,
        f1_score, precision_score, recall_score, accuracy_score,
        roc_curve, precision_recall_curve, confusion_matrix
    )
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    print("[WARN] scikit-learn not found – ML classifier disabled.")

try:
    import torch
    from transformers import AutoTokenizer, AutoModel
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

# ══════════════════════════════════════════════════════════════════════════════
# Paths
# ══════════════════════════════════════════════════════════════════════════════

ROOT_DIR     = Path(__file__).resolve().parent.parent
DIFFUSION_BENCH = ROOT_DIR / "data" / "benchmark_ReFineID_Diffusion" / "diffusion_benchmark"
FIM_BENCH       = ROOT_DIR / "data" / "benchmark_ReFineID_FIM" / "ar_fim_benchmark"
TEST_CSV        = ROOT_DIR / "data" / "test.csv"
OUT_DIR         = Path(__file__).resolve().parent / "comparison_results"

# Smell token sets (mirrors code_naming_smell_detector.py)
SMELL_SEVERE   = {"x", "a", "n", "i", "b", "c", "k", "v", "s", "e", "t"}
SMELL_MODERATE = {"tmp", "val", "foo", "res", "obj", "data", "temp", "var",
                  "item", "elem", "node", "info"}
SMELL_MILD     = {"myVar", "temp1", "result1", "value1", "myObj", "newObj",
                  "helper", "handler", "manager", "processor", "wrapper"}
ALL_SMELLS     = SMELL_SEVERE | SMELL_MODERATE | SMELL_MILD

# Diffusion model config (for method 4)
DEVICE   = "cuda" if (HAS_TORCH and torch.cuda.is_available()) else "cpu"
MAX_TOKS = 512
LEFT_CTX = MAX_TOKS // 2
NUM_STEPS = 32

MODEL_REGISTRY = {
    "DiffuCoder-7B-Base":     {"id": "apple/DiffuCoder-7B-Base",                 "mask_token": "<|mask|>"},
    "DiffuCoder-7B-Instruct": {"id": "apple/DiffuCoder-7B-Instruct",             "mask_token": "<|mask|>"},
    "DreamCoder-7B":          {"id": "Dream-org/Dream-Coder-v0-Instruct-7B",     "mask_token": "<|mask|>"},
}

SMELL_PROBES = {
    "severe":   ["x", "a", "n", "i"],
    "moderate": ["tmp", "val", "foo", "res"],
    "mild":     ["myVar", "temp1", "result1", "value1"],
}

# Calibrated thresholds
THRESH_OC          = 200
THRESH_RARE        = 1000
OC_SMELL_RANK_THR  = 2000
OC_TRAP_RATIO_THR  = 2.0
CTX_DELTA_H_LOW    = 1.5
CTX_DELTA_H_HIGH   = 3.0
ALPHA_GRID_FAST    = [0.0, 0.4, 0.8]
ALPHA_GRID_FULL    = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
W_OC, W_CTX        = 0.4, 0.6
VERDICT_SMELL_THR  = 0.65
VERDICT_SUSP_THR   = 0.35

# ══════════════════════════════════════════════════════════════════════════════
# Data Structures
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class Sample:
    sample_id: int
    ground_truth: str
    masked_code: str          # full code with [MASK]
    is_bad_name: bool         # ground-truth label: True = bad naming

@dataclass
class DetectorResult:
    method: str
    sample_id: int
    ground_truth: str
    is_bad_name: bool         # true label
    prediction: bool          # predicted: True = SMELL
    score: float              # continuous smell score [0, 1]
    meta: dict = field(default_factory=dict)

# ══════════════════════════════════════════════════════════════════════════════
# Dataset Loading
# ══════════════════════════════════════════════════════════════════════════════

def load_benchmark_with_code(
    bench_dir: Path,
    test_csv: Path,
    max_samples: Optional[int] = None,
) -> list[Sample]:
    """
    Load the first benchmark CSV found in bench_dir and join with test.csv
    to get masked_code.  Label: correct=False → bad name (is_bad_name=True).
    """
    csv_files = sorted(bench_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSVs in {bench_dir}")

    bench_file = csv_files[0]
    print(f"  [data] Benchmark file : {bench_file.name}")

    bench_df = pd.read_csv(bench_file)
    bench_df = bench_df.dropna(subset=["ground_truth", "prediction"])
    if max_samples:
        bench_df = bench_df.head(max_samples)

    # Load masked code from test.csv
    print(f"  [data] Loading test.csv …")
    csv.field_size_limit(sys.maxsize)
    id_to_code: dict[int, str] = {}
    with open(test_csv, "r", encoding="utf-8") as f:
        for row in csv.reader(f):
            if len(row) >= 2:
                try:
                    id_to_code[int(row[0])] = row[1]
                except ValueError:
                    pass

    samples: list[Sample] = []
    for _, row in bench_df.iterrows():
        sid = int(row["id"])
        gt  = str(row["ground_truth"]).strip()
        pred = str(row["prediction"]).strip()
        correct = bool(row.get("correct", pred == gt))
        masked_code = id_to_code.get(sid, "")
        samples.append(Sample(
            sample_id=sid,
            ground_truth=gt,
            masked_code=masked_code,
            is_bad_name=not correct,   # bad name ↔ model got it wrong (proxy label)
        ))

    print(f"  [data] {len(samples)} samples loaded. "
          f"Bad-name rate: {sum(s.is_bad_name for s in samples)/len(samples)*100:.1f}%")
    return samples


def load_developer_dataset(csv_path: Path, max_samples: Optional[int] = None) -> list[Sample]:
    """Load the developer dataset where all variables are assumed originally from repos (no 'bad' proxy)."""
    if not csv_path.exists():
        raise FileNotFoundError(f"Developer dataset not found at {csv_path}")

    print(f"  [data] Loading Developer dataset from {csv_path.name} …")
    df = pd.read_csv(csv_path)
    if max_samples:
        df = df.head(max_samples)

    samples = []
    for idx, row in df.iterrows():
        tv_str = str(row.get("Target-Var", ""))
        if not tv_str.strip() or pd.isna(row.get("Target-Var")):
            continue
        try:
            tv = ast.literal_eval(tv_str)
            name = list(tv.keys())[0]
        except Exception:
            continue

        code = str(row.get("methodBody", ""))
        
        # Replace whole-word exact matches with [MASK]
        masked_code = re.sub(rf'\b{re.escape(name)}\b', "[MASK]", code)

        samples.append(Sample(
            sample_id=int(idx),
            ground_truth=name,
            masked_code=masked_code,
            is_bad_name=False,  # Treat all as "Clean / Good Name" since they are real code
        ))

    print(f"  [data] {len(samples)} Developer samples loaded.")
    return samples

def load_all_ar_predictions(fim_dir: Path) -> dict[str, pd.DataFrame]:
    """Load all AR-FIM benchmark CSVs, keyed by model name."""
    result = {}
    for f in sorted(fim_dir.glob("*.csv")):
        key = f.stem.split("_refineID")[0]
        df = pd.read_csv(f)
        result[key] = df
    return result

# ══════════════════════════════════════════════════════════════════════════════
# METHOD 1 — Random Guessing
# ══════════════════════════════════════════════════════════════════════════════

def run_random_detector(
    samples: list[Sample],
    bad_name_prior: float = 0.5,
    seed: int = 42,
) -> list[DetectorResult]:
    """
    Random baseline: sample a Bernoulli(bad_name_prior) prediction for each sample.
    Score = uniform random in [0, 1].
    """
    rng = random.Random(seed)
    results = []
    for s in samples:
        score = rng.random()
        pred  = score >= (1 - bad_name_prior)
        results.append(DetectorResult(
            method="Random",
            sample_id=s.sample_id,
            ground_truth=s.ground_truth,
            is_bad_name=s.is_bad_name,
            prediction=pred,
            score=score,
            meta={"prior": bad_name_prior},
        ))
    return results

# ══════════════════════════════════════════════════════════════════════════════
# METHOD 2 — ML Classifier (heuristic features)
# ══════════════════════════════════════════════════════════════════════════════

def _extract_features(name: str, masked_code: str = "") -> np.ndarray:
    """
    Hand-crafted lexical + structural features for a variable name.

    Features:
      0  name length (chars)
      1  token length (# camelCase / snake_case parts)
      2  is single char
      3  is all lowercase
      4  is all uppercase
      5  has digits
      6  camelCase (has upper after lower)
      7  snake_case (has underscore)
      8  char-type entropy (Shannon over {upper, lower, digit, other})
      9  in SMELL_SEVERE
      10 in SMELL_MODERATE
      11 in SMELL_MILD
      12 name char entropy (Shannon over character set)
      13 context length (clamped to 2000)
      14 fraction of name chars that are uppercase
      15 starts with common smell prefix ('get','set','my','new','tmp','val','res')
    """
    n = name.strip()
    if not n:
        return np.zeros(16, dtype=float)

    # Part count: split by CamelCase + underscore
    parts_camel = re.findall(r'[A-Z]?[a-z]+|[A-Z]+(?=[A-Z]|$)|\d+', n)
    parts_snake = n.split("_")
    num_parts = max(len(parts_camel), len(parts_snake))

    # Char class counts
    u = sum(1 for c in n if c.isupper())
    l = sum(1 for c in n if c.islower())
    d = sum(1 for c in n if c.isdigit())
    o = len(n) - u - l - d
    total = max(len(n), 1)

    # Char-type entropy
    def _h(cnt): return 0.0 if cnt == 0 else -(cnt/total)*math.log2(cnt/total + 1e-12)
    char_entropy = _h(u) + _h(l) + _h(d) + _h(o)

    # Name char entropy (over individual characters)
    freq = {}
    for c in n:
        freq[c] = freq.get(c, 0) + 1
    name_char_ent = -sum((v/total)*math.log2(v/total + 1e-12) for v in freq.values())

    has_upper_after_lower = bool(re.search(r'[a-z][A-Z]', n))
    has_underscore        = "_" in n

    SMELL_PREFIXES = {"get", "set", "my", "new", "tmp", "val", "res", "temp", "data"}
    starts_smell   = any(n.lower().startswith(p) for p in SMELL_PREFIXES)

    ctx_len = min(len(masked_code), 2000) if masked_code else 0

    feat = np.array([
        len(n),                               # 0
        num_parts,                            # 1
        float(len(n) == 1),                   # 2
        float(n.islower()),                   # 3
        float(n.isupper()),                   # 4
        float(any(c.isdigit() for c in n)),   # 5
        float(has_upper_after_lower),         # 6
        float(has_underscore),               # 7
        char_entropy,                         # 8
        float(n in SMELL_SEVERE),             # 9
        float(n in SMELL_MODERATE),           # 10
        float(n in SMELL_MILD),              # 11
        name_char_ent,                        # 12
        ctx_len / 2000.0,                     # 13
        u / total,                            # 14
        float(starts_smell),                  # 15
    ], dtype=float)
    return feat


def run_ml_classifier(
    samples: list[Sample],
    seed: int = 42,
    n_splits: int = 5,
) -> list[DetectorResult]:
    """
    Train a Gradient Boosting classifier on heuristic features via cross-val.
    Returns out-of-fold predictions so there's no data leakage.
    """
    if not HAS_SKLEARN:
        print("[WARN] sklearn not available – returning random scores for ML method.")
        return run_random_detector(samples, seed=seed + 1)

    X = np.array([_extract_features(s.ground_truth, s.masked_code) for s in samples])
    y = np.array([int(s.is_bad_name) for s in samples])

    clf = Pipeline([
        ("scaler", StandardScaler()),
        ("model", GradientBoostingClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.1,
            random_state=seed,
        )),
    ])

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    scores = np.full(len(y), 0.5)
    preds  = np.zeros(len(y), dtype=bool)

    for fold_i, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        clf.fit(X[train_idx], y[train_idx])
        proba = clf.predict_proba(X[val_idx])[:, 1]
        scores[val_idx] = proba
        preds[val_idx]  = proba >= 0.5
        print(f"    Fold {fold_i+1}/{n_splits} done.")

    results = []
    for i, s in enumerate(samples):
        results.append(DetectorResult(
            method="ML-Classifier",
            sample_id=s.sample_id,
            ground_truth=s.ground_truth,
            is_bad_name=s.is_bad_name,
            prediction=bool(preds[i]),
            score=float(scores[i]),
            meta={"n_folds": n_splits},
        ))
    return results

# ══════════════════════════════════════════════════════════════════════════════
# METHOD 3 — AR Model (FIM predictions as detector)
# ══════════════════════════════════════════════════════════════════════════════

def run_ar_detector(
    samples: list[Sample],
    ar_predictions: dict[str, pd.DataFrame],
    ensemble: bool = True,
) -> list[DetectorResult]:
    """
    Use pre-computed AR-FIM benchmark predictions as a bad-name detector.

    Strategy: for sample i, if model's prediction != ground_truth → SMELL.
    Score = (# models that got it wrong) / (# models that attempted it).

    With ensemble=False, picks the single best-performing model by exact match.
    """
    if not ar_predictions:
        raise ValueError("No AR predictions loaded.")

    # Build per-sample id→{model: correct} lookup
    id_correct: dict[int, list[bool]] = {}
    for model_name, df in ar_predictions.items():
        df2 = df.dropna(subset=["ground_truth", "prediction"])
        for _, row in df2.iterrows():
            sid = int(row["id"])
            correct = bool(row.get("correct", str(row["prediction"]).strip() == str(row["ground_truth"]).strip()))
            id_correct.setdefault(sid, []).append(correct)

    # Compute accuracy per model to pick best
    model_acc: dict[str, float] = {}
    for model_name, df in ar_predictions.items():
        df2 = df.dropna(subset=["correct"])
        model_acc[model_name] = df2["correct"].mean()
    best_model = max(model_acc, key=model_acc.get)
    print(f"  [AR] Best AR model: {best_model} (acc={model_acc[best_model]*100:.1f}%)")
    print(f"  [AR] Ensembling {len(ar_predictions)} AR models: {ensemble}")

    # Build id→correct from best single model
    best_df = ar_predictions[best_model]
    id_best: dict[int, bool] = {}
    for _, row in best_df.dropna(subset=["ground_truth", "prediction"]).iterrows():
        sid = int(row["id"])
        id_best[sid] = bool(row.get("correct", False))

    results = []
    for s in samples:
        sid = s.sample_id
        if ensemble:
            corrects = id_correct.get(sid, [])
            if corrects:
                wrong_frac = 1.0 - (sum(corrects) / len(corrects))
            else:
                wrong_frac = 0.5   # unknown
            score = wrong_frac
            pred  = wrong_frac >= 0.5
        else:
            correct = id_best.get(sid, True)
            score   = 0.1 if correct else 0.9
            pred    = not correct

        results.append(DetectorResult(
            method="AR-FIM",
            sample_id=sid,
            ground_truth=s.ground_truth,
            is_bad_name=s.is_bad_name,
            prediction=pred,
            score=float(score),
            meta={"best_model": best_model, "ensemble": ensemble},
        ))
    return results

# ══════════════════════════════════════════════════════════════════════════════
# METHOD 4 — Our Method (OC + CTX, optimised)
# ══════════════════════════════════════════════════════════════════════════════

# ── Diffusion model utils (duplicated here for self-containment) ──────────────

def _load_diffusion_model(model_name: str):
    meta = MODEL_REGISTRY[model_name]
    print(f"  [OC+CTX] Loading {model_name} …")
    tokenizer = AutoTokenizer.from_pretrained(meta["id"], trust_remote_code=True)
    dtype = torch.bfloat16 if DEVICE == "cuda" else torch.float32
    model = AutoModel.from_pretrained(
        meta["id"], trust_remote_code=True, torch_dtype=dtype
    ).to(DEVICE).eval()
    mask_id = tokenizer.convert_tokens_to_ids(meta["mask_token"])
    if hasattr(model, "generation_config") and hasattr(model.generation_config, "steps"):
        model.generation_config.steps = NUM_STEPS
    return tokenizer, model, mask_id


def _forward(model, input_ids):
    with torch.no_grad():
        try:
            out = model(input_ids=input_ids, attention_mask=None, num_steps=NUM_STEPS)
        except TypeError:
            out = model(input_ids=input_ids, attention_mask=None)
    return out.logits if hasattr(out, "logits") else (out[0] if isinstance(out, tuple) else out)


def _build_window(tokenizer, masked_code: str, gt_name: str, mask_id: int):
    mask_char = masked_code.find("[MASK]")
    if mask_char == -1:
        raise ValueError("No [MASK] token in code.")
    prefix = masked_code[:mask_char]
    suffix = masked_code[mask_char + len("[MASK]"):]
    prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
    suffix_ids = tokenizer.encode(suffix, add_special_tokens=False)
    right_ctx  = MAX_TOKS - LEFT_CTX - 1
    prefix_win = prefix_ids[-LEFT_CTX:]
    suffix_win = suffix_ids[:right_ctx]
    bos = ([tokenizer.bos_token_id] if getattr(tokenizer, "bos_token_id", None) else [])
    token_seq  = bos + prefix_win + [mask_id] + suffix_win
    target_idx = len(bos) + len(prefix_win)
    input_ids  = torch.tensor([token_seq], dtype=torch.long).to(DEVICE)
    return input_ids, target_idx


def _softmax_stats(logits, position: int, token_id: int) -> dict:
    lp    = logits[0, position, :].float()
    probs = torch.softmax(lp, dim=-1)
    lprob = torch.log(probs + 1e-12)
    entropy = float(-(probs * lprob).sum())
    sorted_ids = torch.argsort(probs, descending=True)
    rank_map   = {int(t): r + 1 for r, t in enumerate(sorted_ids)}
    rank = rank_map.get(int(token_id), len(rank_map))
    return {"entropy": entropy, "prob": float(probs[token_id]), "rank": rank}


# ── Optimised smell score (v2) ────────────────────────────────────────────────

def _sigmoid(x: float, center: float, slope: float) -> float:
    """Soft threshold via sigmoid: 0 = clean, 1 = smell."""
    return 1.0 / (1.0 + math.exp(-slope * (x - center)))


def compute_smell_score_v2(oc: dict, ctx: dict) -> tuple[float, str]:
    """
    Optimised smell score:
    - Uses sigmoid soft thresholds instead of hard step functions.
    - Regime-aware weighting: OVERCONFIDENT samples get ctx signal discounted.
    - Adds bonus for extreme trap_ratio.
    """
    # ── OC score ──────────────────────────────────────────────────────────────
    regime  = oc.get("regime", "UNCERTAIN")
    trap_r  = oc.get("trap_ratio", 1.0)
    sev_r   = oc.get("smell_severe_rank", 9999)

    if regime == "OVERCONFIDENT":
        # gt token is already generic → treat as soft smell
        oc_score = _sigmoid(trap_r, center=1.5, slope=2.0) * 0.9
    elif regime == "CONFIDENT_RARE":
        # Real trap: specific name but model prefers generic tokens
        if sev_r < OC_SMELL_RANK_THR:
            oc_score = _sigmoid(trap_r, center=OC_TRAP_RATIO_THR, slope=0.8)
        else:
            oc_score = 0.05 + _sigmoid(trap_r, center=4.0, slope=0.5) * 0.4
    else:  # UNCERTAIN
        oc_score = 0.1 + _sigmoid(trap_r, center=1.5, slope=1.0) * 0.4

    # ── CTX score ─────────────────────────────────────────────────────────────
    dh = ctx.get("delta_h", 0.0)
    # High ΔH → clean, low ΔH → smell  (inverted sigmoid)
    ctx_score = 1.0 - _sigmoid(dh, center=(CTX_DELTA_H_LOW + CTX_DELTA_H_HIGH) / 2, slope=1.2)

    # ── Dynamic weights: trust OC more for OVERCONFIDENT, CTX more for RARE ──
    if regime == "OVERCONFIDENT":
        w_oc, w_ctx = 0.55, 0.45
    elif regime == "CONFIDENT_RARE":
        w_oc, w_ctx = 0.35, 0.65
    else:
        w_oc, w_ctx = W_OC, W_CTX

    # Handle errors gracefully
    if "error" in oc or "error" in ctx:
        w_ctx = 0.0 if "error" in ctx else w_ctx
        w_oc  = 0.0 if "error" in oc  else w_oc
        total_w = max(w_oc + w_ctx, 1e-6)
        score = (w_oc * oc_score + w_ctx * ctx_score) / total_w
    else:
        score = w_oc * oc_score + w_ctx * ctx_score

    # ── Extreme trap override ─────────────────────────────────────────────────
    if oc_score >= 0.88:
        score = max(score, 0.72)

    score = float(np.clip(score, 0.0, 1.0))

    if score >= VERDICT_SMELL_THR:
        verdict = "SMELL"
    elif score >= VERDICT_SUSP_THR:
        verdict = "SUSPICIOUS"
    else:
        verdict = "CLEAN"

    return round(score, 4), verdict


def _probe_sample(sample: Sample, tokenizer, model, mask_id: int,
                  rng: random.Random, alpha_grid: list) -> dict:
    """Run OC + CTX on a single sample. Returns combined score dict."""
    try:
        base_ids, tgt = _build_window(tokenizer, sample.masked_code, sample.ground_truth, mask_id)
    except Exception as e:
        return {"score": 0.5, "verdict": "UNKNOWN", "error": str(e)}

    # GT token id
    gt_toks = tokenizer.encode(sample.ground_truth, add_special_tokens=False)
    gt_id   = gt_toks[0] if gt_toks else (tokenizer.unk_token_id or 0)

    # Smell probe ids
    smell_ids: dict[str, tuple] = {}
    for sev, names in SMELL_PROBES.items():
        candidates = list(names); rng.shuffle(candidates)
        for cand in candidates:
            t = tokenizer.encode(cand, add_special_tokens=False)
            cid = t[0] if t else (tokenizer.unk_token_id or 0)
            if cid != gt_id:
                smell_ids[sev] = (cand, cid)
                break

    try:
        logits = _forward(model, base_ids)
    except Exception as e:
        return {"score": 0.5, "verdict": "UNKNOWN", "error": str(e)}

    # OC probe
    gt_stats = _softmax_stats(logits, tgt, gt_id)
    gt_rank  = gt_stats["rank"]
    smell_ranks = {sev: _softmax_stats(logits, tgt, sid)["rank"] for sev, (_, sid) in smell_ids.items()}

    if gt_rank <= THRESH_OC:
        regime = "OVERCONFIDENT"
    elif gt_rank <= THRESH_RARE:
        regime = "UNCERTAIN"
    else:
        regime = "CONFIDENT_RARE"

    sev_rank   = smell_ranks.get("severe", gt_rank)
    trap_ratio = gt_rank / max(sev_rank, 1)
    oc = {
        "gt_rank": gt_rank, "gt_prob": gt_stats["prob"],
        "regime": regime,
        "smell_severe_rank":   smell_ranks.get("severe", -1),
        "smell_moderate_rank": smell_ranks.get("moderate", -1),
        "smell_mild_rank":     smell_ranks.get("mild", -1),
        "trap_ratio": trap_ratio,
    }

    # CTX probe
    h_by_alpha = {}
    for alpha in alpha_grid:
        if alpha == 0.0:
            ids = base_ids
        else:
            ids = base_ids.clone()
            eligible = [i for i in range(ids.shape[1]) if i != tgt]
            k = int(round(alpha * len(eligible)))
            for i in rng.sample(eligible, k) if k else []:
                ids[0, i] = mask_id
        try:
            lg  = _forward(model, ids)
            lp  = lg[0, tgt, :].float()
            pr  = torch.softmax(lp, dim=-1)
            h_by_alpha[alpha] = float(-(pr * torch.log(pr + 1e-12)).sum())
        except Exception:
            h_by_alpha[alpha] = 0.0

    h_0  = h_by_alpha.get(0.0, 0.0)
    h_08 = h_by_alpha.get(0.8, h_by_alpha.get(max(alpha_grid), 0.0))
    ctx = {"h_at_0": h_0, "h_at_08": h_08, "delta_h": h_08 - h_0}

    score, verdict = compute_smell_score_v2(oc, ctx)
    return {"score": score, "verdict": verdict, "oc": oc, "ctx": ctx}


def run_our_method(
    samples: list[Sample],
    model_name: str = "DiffuCoder-7B-Base",
    fast_mode: bool = True,
    seed: int = 42,
) -> list[DetectorResult]:
    """OC + CTX method (optimised v2). Requires diffusion model."""
    if not HAS_TORCH:
        raise RuntimeError("PyTorch not available for OC+CTX method.")

    tokenizer, model, mask_id = _load_diffusion_model(model_name)
    alpha_grid = ALPHA_GRID_FAST if fast_mode else ALPHA_GRID_FULL
    rng = random.Random(seed)

    results = []
    for i, s in enumerate(samples):
        if (i + 1) % 50 == 0:
            print(f"    [{i+1}/{len(samples)}] probing …")
        probe = _probe_sample(s, tokenizer, model, mask_id, rng, alpha_grid)
        is_smell = probe.get("verdict", "CLEAN") in ("SMELL", "SUSPICIOUS")
        results.append(DetectorResult(
            method="OC+CTX (Ours)",
            sample_id=s.sample_id,
            ground_truth=s.ground_truth,
            is_bad_name=s.is_bad_name,
            prediction=is_smell,
            score=probe.get("score", 0.5),
            meta=probe,
        ))
    return results

# ══════════════════════════════════════════════════════════════════════════════
# Evaluation metrics
# ══════════════════════════════════════════════════════════════════════════════

def compute_metrics(results: list[DetectorResult]) -> dict:
    y_true  = np.array([int(r.is_bad_name) for r in results])
    y_pred  = np.array([int(r.prediction) for r in results])
    y_score = np.array([r.score for r in results])

    metrics = {
        "method":    results[0].method,
        "n":         len(results),
        "n_bad":     int(y_true.sum()),
        "n_good":    int((1 - y_true).sum()),
        "accuracy":  round(float(accuracy_score(y_true, y_pred) if HAS_SKLEARN else np.mean(y_true == y_pred)), 4),
    }
    if HAS_SKLEARN:
        metrics.update({
            "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
            "recall":    round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
            "f1":        round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
            "roc_auc":   round(float(roc_auc_score(y_true, y_score) if len(np.unique(y_true)) > 1 else 0.5), 4),
            "avg_prec":  round(float(average_precision_score(y_true, y_score) if len(np.unique(y_true)) > 1 else 0.5), 4),
        })
    else:
        metrics.update({"precision": None, "recall": None, "f1": None, "roc_auc": None, "avg_prec": None})

    return metrics


def print_metrics_table(all_metrics: list[dict]):
    header = f"{'Method':<22} {'Acc':>6} {'Prec':>6} {'Rec':>6} {'F1':>6} {'AUC':>6} {'AP':>6}"
    sep    = "-" * len(header)
    print(f"\n{sep}\n{header}\n{sep}")
    for m in all_metrics:
        def fmt(v): return f"{v:.4f}" if isinstance(v, float) else "  N/A"
        print(f"  {m['method']:<20} {fmt(m['accuracy'])} {fmt(m['precision'])} "
              f"{fmt(m['recall'])} {fmt(m['f1'])} {fmt(m['roc_auc'])} {fmt(m['avg_prec'])}")
    print(sep)

# ══════════════════════════════════════════════════════════════════════════════
# Visualisation
# ══════════════════════════════════════════════════════════════════════════════

METHOD_COLORS = {
    "Random":         "#9e9e9e",
    "ML-Classifier":  "#42a5f5",
    "AR-FIM":         "#66bb6a",
    "OC+CTX (Ours)":  "#ef5350",
}


def plot_comparison(all_results: dict[str, list[DetectorResult]],
                    all_metrics: list[dict],
                    out_dir: Path):
    if not HAS_MPL or not HAS_SKLEARN:
        print("[WARN] matplotlib/sklearn not available – skipping plots.")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "figure.facecolor": "#0f0f1a",
        "axes.facecolor": "#1a1a2e",
        "text.color": "#e0e0e0",
        "axes.labelcolor": "#e0e0e0",
        "xtick.color": "#e0e0e0",
        "ytick.color": "#e0e0e0",
        "axes.edgecolor": "#444",
        "grid.color": "#333",
        "legend.facecolor": "#1a1a2e",
        "legend.edgecolor": "#444",
    })

    # ── 1. Bar chart of key metrics ───────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.patch.set_facecolor("#0f0f1a")
    fig.suptitle("Bad Naming Detector — Method Comparison", color="#e0e0e0",
                 fontsize=16, fontweight="bold", y=1.02)

    metric_keys  = ["accuracy", "f1", "roc_auc"]
    metric_names = ["Accuracy", "F1 Score", "ROC-AUC"]

    for ax, mk, mn in zip(axes, metric_keys, metric_names):
        methods = [m["method"] for m in all_metrics]
        vals    = [m.get(mk) or 0.0 for m in all_metrics]
        colors  = [METHOD_COLORS.get(m, "#888") for m in methods]

        bars = ax.bar(methods, vals, color=colors, edgecolor="#222", linewidth=0.8, width=0.5)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel(mn, fontsize=11)
        ax.set_title(mn, fontsize=12, color="#e0e0e0", pad=10)
        ax.tick_params(axis="x", rotation=15)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=9, color="#e0e0e0")
        # Baseline line at 0.5
        ax.axhline(0.5, color="#888", linestyle="--", linewidth=0.8, alpha=0.5, label="Random baseline")

    plt.tight_layout()
    out_path = out_dir / "bar_chart.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="#0f0f1a")
    plt.close()
    print(f"  [plot] {out_path}")

    # ── 2. ROC Curves ────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 6))
    fig.patch.set_facecolor("#0f0f1a")
    ax.set_facecolor("#1a1a2e")
    ax.plot([0, 1], [0, 1], "w--", alpha=0.3, label="Random")

    for method_name, res_list in all_results.items():
        y_true  = np.array([int(r.is_bad_name) for r in res_list])
        y_score = np.array([r.score for r in res_list])
        if len(np.unique(y_true)) < 2:
            continue
        fpr, tpr, _ = roc_curve(y_true, y_score)
        auc = roc_auc_score(y_true, y_score)
        ax.plot(fpr, tpr, label=f"{method_name} (AUC={auc:.3f})",
                color=METHOD_COLORS.get(method_name, "#888"), linewidth=2)

    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate", fontsize=11)
    ax.set_title("ROC Curves — Bad Naming Detection", color="#e0e0e0", fontsize=13)
    ax.legend(fontsize=9)
    plt.tight_layout()
    out_path = out_dir / "roc_curves.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="#0f0f1a")
    plt.close()
    print(f"  [plot] {out_path}")

    # ── 3. Precision–Recall Curves ───────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 6))
    fig.patch.set_facecolor("#0f0f1a")
    ax.set_facecolor("#1a1a2e")

    for method_name, res_list in all_results.items():
        y_true  = np.array([int(r.is_bad_name) for r in res_list])
        y_score = np.array([r.score for r in res_list])
        if len(np.unique(y_true)) < 2:
            continue
        prec, rec, _ = precision_recall_curve(y_true, y_score)
        ap = average_precision_score(y_true, y_score)
        ax.plot(rec, prec, label=f"{method_name} (AP={ap:.3f})",
                color=METHOD_COLORS.get(method_name, "#888"), linewidth=2)

    ax.set_xlabel("Recall", fontsize=11)
    ax.set_ylabel("Precision", fontsize=11)
    ax.set_title("Precision–Recall Curves", color="#e0e0e0", fontsize=13)
    ax.legend(fontsize=9)
    plt.tight_layout()
    out_path = out_dir / "pr_curves.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="#0f0f1a")
    plt.close()
    print(f"  [plot] {out_path}")

    # ── 4. Confusion matrices ────────────────────────────────────────────────
    n_methods = len(all_results)
    fig, axes = plt.subplots(1, n_methods, figsize=(4 * n_methods, 4))
    fig.patch.set_facecolor("#0f0f1a")
    if n_methods == 1:
        axes = [axes]
    fig.suptitle("Confusion Matrices", color="#e0e0e0", fontsize=14, y=1.02)

    for ax, (method_name, res_list) in zip(axes, all_results.items()):
        ax.set_facecolor("#1a1a2e")
        y_true = np.array([int(r.is_bad_name) for r in res_list])
        y_pred = np.array([int(r.prediction) for r in res_list])
        cm = confusion_matrix(y_true, y_pred)
        im = ax.imshow(cm, cmap="Blues", aspect="auto")
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(["CLEAN", "SMELL"])
        ax.set_yticklabels(["CLEAN", "SMELL"])
        ax.set_xlabel("Predicted", fontsize=10)
        ax.set_ylabel("Actual", fontsize=10)
        ax.set_title(method_name, color=METHOD_COLORS.get(method_name, "#e0e0e0"),
                     fontsize=11, fontweight="bold")
        for (i, j), val in np.ndenumerate(cm):
            ax.text(j, i, str(val), ha="center", va="center",
                    color="white" if val > cm.max() / 2 else "#aaa", fontsize=13)

    plt.tight_layout()
    out_path = out_dir / "confusion_matrices.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="#0f0f1a")
    plt.close()
    print(f"  [plot] {out_path}")

    # ── 5. Score distributions ────────────────────────────────────────────────
    fig, axes = plt.subplots(1, n_methods, figsize=(4.5 * n_methods, 4), sharey=False)
    fig.patch.set_facecolor("#0f0f1a")
    if n_methods == 1:
        axes = [axes]
    fig.suptitle("Score Distributions (Bad vs Good Names)", color="#e0e0e0", fontsize=14, y=1.02)

    for ax, (method_name, res_list) in zip(axes, all_results.items()):
        ax.set_facecolor("#1a1a2e")
        color = METHOD_COLORS.get(method_name, "#888")
        bad_scores  = [r.score for r in res_list if r.is_bad_name]
        good_scores = [r.score for r in res_list if not r.is_bad_name]
        ax.hist(bad_scores,  bins=20, alpha=0.7, color="#ef5350", label="Bad name",  density=True)
        ax.hist(good_scores, bins=20, alpha=0.7, color="#66bb6a", label="Good name", density=True)
        ax.axvline(0.5, color="white", linestyle="--", linewidth=1.0, alpha=0.5)
        ax.set_title(method_name, color=color, fontsize=11, fontweight="bold")
        ax.set_xlabel("Smell Score", fontsize=9)
        ax.set_ylabel("Density", fontsize=9)
        ax.legend(fontsize=8)

    plt.tight_layout()
    out_path = out_dir / "score_distributions.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="#0f0f1a")
    plt.close()
    print(f"  [plot] {out_path}")

# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Compare bad-naming detectors: Random, ML, AR-FIM, OC+CTX"
    )
    parser.add_argument("--mode", choices=["offline", "full", "ar-only"], default="offline",
        help=(
            "offline = methods 1-3 (no diffusion model needed, default)\n"
            "full    = all 4 methods (requires diffusion model)\n"
            "ar-only = only AR-FIM baseline"
        ))
    parser.add_argument("--model", default="DiffuCoder-7B-Base",
        choices=list(MODEL_REGISTRY.keys()),
        help="Diffusion model to use for OC+CTX (method 4, full mode only).")
    parser.add_argument("--fast", action="store_true",
        help="Use 3-step alpha grid for CTX probe (faster).")
    parser.add_argument("--max-samples", type=int, default=None,
        help="Limit samples for quick testing.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-plots", action="store_true",
        help="Skip generating plots.")
    parser.add_argument("--out-dir", type=str, default=str(OUT_DIR),
        help="Output directory for results.")
    parser.add_argument("--dataset", type=str, choices=["refineid", "developer"], default="refineid",
        help="Dataset to use. 'refineid' uses benchmark proxy, 'developer' evaluates Test-Dev.csv")

    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 66)
    print("  BAD NAMING DETECTOR — COMPARATIVE EVALUATION")
    print("=" * 66)

    # ── Load samples ──────────────────────────────────────────────────────────
    if args.dataset == "refineid":
        print(f"\n[1] Loading benchmark data …")
        samples = load_benchmark_with_code(DIFFUSION_BENCH, TEST_CSV, args.max_samples)
    else:
        print(f"\n[1] Loading Developer data …")
        samples = load_developer_dataset(ROOT_DIR / "data" / "Developer" / "Test-Dev.csv", args.max_samples)

    # ── Load AR predictions ───────────────────────────────────────────────────
    if args.dataset == "refineid":
        print(f"\n[2] Loading AR-FIM predictions …")
        ar_preds = load_all_ar_predictions(FIM_BENCH)
        print(f"    {len(ar_preds)} AR models: {list(ar_preds.keys())}")
    else:
        ar_preds = None

    all_results: dict[str, list[DetectorResult]] = {}
    all_metrics: list[dict] = []

    # ── Method 1: Random ─────────────────────────────────────────────────────
    if args.mode in ("offline", "full") and args.dataset == "refineid":
        print(f"\n[3] Running METHOD 1 — Random Guessing …")
        bad_prior = sum(s.is_bad_name for s in samples) / len(samples)
        res = run_random_detector(samples, bad_name_prior=bad_prior, seed=args.seed)
        all_results["Random"] = res
        m = compute_metrics(res)
        all_metrics.append(m)
        print(f"    Acc={m['accuracy']:.4f}  F1={m['f1']}  AUC={m['roc_auc']}")

    # ── Method 2: ML Classifier ──────────────────────────────────────────────
    if args.mode in ("offline", "full") and args.dataset == "refineid":
        print(f"\n[4] Running METHOD 2 — ML Classifier (cross-validation) …")
        res = run_ml_classifier(samples, seed=args.seed)
        all_results["ML-Classifier"] = res
        m = compute_metrics(res)
        all_metrics.append(m)
        print(f"    Acc={m['accuracy']:.4f}  F1={m['f1']}  AUC={m['roc_auc']}")

    # ── Method 3: AR-FIM ─────────────────────────────────────────────────────
    if args.mode in ("offline", "full", "ar-only") and ar_preds and args.dataset == "refineid":
        print(f"\n[5] Running METHOD 3 — AR Model (FIM, ensemble) …")
        res = run_ar_detector(samples, ar_preds, ensemble=True)
        all_results["AR-FIM"] = res
        m = compute_metrics(res)
        all_metrics.append(m)
        print(f"    Acc={m['accuracy']:.4f}  F1={m['f1']}  AUC={m['roc_auc']}")

    # ── Method 4: OC + CTX ───────────────────────────────────────────────────
    if args.mode == "full":
        print(f"\n[6] Running METHOD 4 — OC+CTX (Ours, {args.model}) …")
        res = run_our_method(samples, model_name=args.model,
                             fast_mode=args.fast, seed=args.seed)
        all_results["OC+CTX (Ours)"] = res
        m = compute_metrics(res)
        all_metrics.append(m)
        print(f"    Acc={m['accuracy']:.4f}  F1={m['f1']}  AUC={m['roc_auc']}")

    # ── Summary table ─────────────────────────────────────────────────────────
    print(f"\n{'=' * 66}\n  FINAL COMPARISON TABLE\n{'=' * 66}")
    print_metrics_table(all_metrics)

    # ── Save results ──────────────────────────────────────────────────────────
    metrics_csv = out_dir / "comparison_metrics.csv"
    pd.DataFrame(all_metrics).to_csv(metrics_csv, index=False)
    print(f"\n  [save] {metrics_csv}")

    metrics_json = out_dir / "comparison_metrics.json"
    with open(metrics_json, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"  [save] {metrics_json}")

    # Per-method detailed CSVs
    for method_name, res_list in all_results.items():
        safe_name = method_name.replace(" ", "_").replace("+", "_").replace("(", "").replace(")", "")
        detail_path = out_dir / f"results_{safe_name}.csv"
        rows = []
        for r in res_list:
            rows.append({
                "sample_id": r.sample_id,
                "ground_truth": r.ground_truth,
                "is_bad_name": r.is_bad_name,
                "prediction": r.prediction,
                "score": r.score,
            })
        pd.DataFrame(rows).to_csv(detail_path, index=False)
        print(f"  [save] {detail_path}")

    # ── Plots ─────────────────────────────────────────────────────────────────
    if not args.no_plots and args.dataset == "refineid":
        print(f"\n[7] Generating plots …")
        plot_comparison(all_results, all_metrics, out_dir)
    elif args.dataset == "developer":
        print("\n[7] Skipping plots: Developer dataset has no true labels for AUC/Accuracy curves")

    print(f"\n{'=' * 66}")
    print(f"  Done. Results saved to: {out_dir}")
    print(f"{'=' * 66}\n")


if __name__ == "__main__":
    main()
