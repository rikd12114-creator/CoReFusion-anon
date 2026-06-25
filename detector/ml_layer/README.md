# ML Smell-Localization Layer

A lightweight supervised head that turns dLLM probe signals into a calibrated
"smell probability" per identifier, used to **rank** candidate refactoring
locations in a Java source.

## Why this exists

The paper's UMAP plots show that the hidden states of injected smell tokens
sit in a partly-separable cloud from those of the GT names — but separation
is not perfect, and a hand-tuned threshold on a single feature (e.g.
`mean_entropy_change > 0.4`) cannot exploit the multi-feature structure that
drives that separation. This layer learns the linear/nonlinear combination
explicitly and turns "filling-finding asymmetry" (RQ2) into a working
localizer.

## Workflow

```
┌─────────────────────────────────────────────────────────────────────┐
│  1) generate_features.py    GPU      RefineID + dLLM → features.csv │
│  2) train_smell_classifier  CPU      features.csv   → model.joblib  │
│  3) demo_e2e.py / localize  CPU/GPU  source code    → ranked smells │
└─────────────────────────────────────────────────────────────────────┘
```

### 1. Feature generation (GPU, slow)

For each RefineID sample, build two views — `[MASK]` replaced with a sampled
smell name, and (optionally) `[MASK]` replaced with the GT target — then run
a single 64-step DiffuCoder denoising trajectory on each. Per identifier:

| Feature                | Source                                  |
|------------------------|-----------------------------------------|
| `avg_flip_step`        | Experiment B: when the token is committed |
| `first_confident_step` | Experiment B: when softmax peak ≥ 0.8     |
| `mean_entropy_change`  | Experiment C: avg \|H(k+1)−H(k)\|         |
| `max_entropy_change`   | Experiment C: max \|H(k+1)−H(k)\|         |

```
python detector/ml_layer/generate_features.py \
    --num-samples 200 \
    --runs-per-sample 1 \
    --include-gt-version \
    --model DiffuCoder-7B-Instruct
# → detector/ml_layer/features/features_DiffuCoder-7B-Instruct_<ts>.csv
```

The smell vocabulary defaults to a severity-mixed pool (`x`, `a`, `i`, `tmp`,
`val`, `foo`, `data`, …). Each call samples one name per `(sample, run)`,
guaranteed distinct from the GT target.

### 2. Train the head (CPU, fast)

```
python detector/ml_layer/train_smell_classifier.py \
    --features detector/ml_layer/features/features_<...>.csv
# → detector/ml_layer/runs/<ts>/{model_logreg,model_mlp,model_gbdt}.joblib
```

Trains LogReg, MLP, GBDT under leave-one-sample-out CV (one fold per
unique `sample_id`). Reports both per-token classification metrics
(ROC-AUC, PR-AUC) and per-(sample,run) localization metrics
(MRR, R@1, R@3, R@5, R@10) against a random and heuristic baseline.

### 3. Localize on new code

**CSV path** — replay precomputed features (no GPU):

```
python detector/ml_layer/demo_e2e.py \
    --features detector/ml_layer/features/features_<...>.csv \
    --model-path detector/ml_layer/runs/<ts>/model_gbdt.joblib \
    --sample-id 17 --honest --top-k 10
```

`--honest` re-fits the classifier with this sample held out, giving a
leakage-free score. Without it the saved model has been trained on the
queried sample and will look unrealistically perfect.

**GPU end-to-end** — full pipeline including dLLM probing on a Java file:

```
python detector/ml_layer/demo_e2e.py --gpu \
    --model-path detector/ml_layer/runs/<ts>/model_gbdt.joblib \
    --sample-id 13 --smell-name tmp \
    --dllm DiffuCoder-7B-Instruct
```

Replaces every `[MASK]` in RefineID sample 13 with `tmp`, runs the 64-step
probe on DiffuCoder, scores every identifier, and prints the ranked list
together with the GT target so you can verify the injection point lands at
rank 1 (or close to it).

## What the current numbers say

Using the legacy ABC features (20 unique RefineID samples,
`SMELL_DUMMY_TOKEN` as the only injection), under LOSO CV:

| Model       | ROC-AUC | MRR    | R@1   | R@5   |
|-------------|---------|--------|-------|-------|
| **gbdt**    | 0.669   | 0.126  | 0.05  | 0.20  |
| logreg      | 0.654   | 0.019  | 0.00  | 0.00  |
| mlp         | 0.463   | 0.027  | 0.00  | 0.05  |
| heuristic   | 0.720   | 0.020  | 0.00  | 0.00  |
| random      | —       | 0.002  | 0.002 | 0.010 |

GBDT is **~60× random on MRR and 20× random on R@5**. The heuristic threshold
beats it on per-token ROC-AUC but cannot rank within a single file (R@5 = 0).

## Limitations and next steps

* **Sample size**: only 20 source files in the legacy dataset. Re-running
  `generate_features.py --num-samples 500 --runs-per-sample 3` is the
  cheapest way to drive these numbers up.
* **Smell vocabulary**: the legacy data used a single literal token,
  `SMELL_DUMMY_TOKEN`. The new generator samples from a realistic pool —
  switching to it should make the model robust to unseen names.
* **Feature richness**: the four scalars are summary statistics. The deep
  hidden state at the identifier's position (Experiment A) carries far more
  signal — extending `generate_features.py` to also pickle the last-layer
  vector and adding a logistic probe over it should push ROC-AUC well above
  0.85.
* **Beyond entropy**: integrating the over-confidence probe outputs from
  `detector/code_naming_smell_detector.py` (`gt_rank`, `delta_h`,
  `trap_ratio`) would add a second, complementary signal stream.

## Files

| File                          | Role                                |
|-------------------------------|-------------------------------------|
| `generate_features.py`        | GPU. RefineID → labeled feature CSV |
| `train_smell_classifier.py`   | CPU. Feature CSV → trained models   |
| `localize.py`                 | Inference: code/CSV → ranked smells |
| `demo_e2e.py`                 | One-sample workflow demo            |
| `runs/<ts>/`                  | Model + metrics + plots per run     |
| `features/`                   | Feature CSVs from `generate_…`      |
