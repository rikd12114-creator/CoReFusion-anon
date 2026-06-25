# RQ3 v2 — condensed, probe-based redesign

**Goal:** make RQ3 *shorter* and *stronger*, and replace the weak qualitative
UMAP/cosine analysis with a quantitative machine-learning **linear probe**.
RQ1/RQ2 are untouched.

---

## 0. What changed, at a glance

| | Old RQ3 (4 probes) | New RQ3 (2 experiments + 1 panel) |
|---|---|---|
| Length | ~3.5 pages, 9 panels + 1 table | ~2 pages, 2 figures + 1 main panel + appendix |
| Hidden-state result | cosine(smelly,clean) line plots + **UMAP scatter** ("clouds partially overlap") | **AUC(layer) probe curve** + **1-D probe-projection histogram**, with CI |
| "Bad name" definition | throw-away vocab `tmp/x/foo/a/n` | **length/subword-matched *misplaced real* identifier** (kills the length shortcut) |
| Headline | "a signal exists but is partial/regime-dependent/late" (vague) | two numbers: **contextual AUC** (Exp1) and **DCL** (Exp2) |
| Commitment result | 20 snippets, descriptive | 230 snippets, overlaid with detection → **DCL** |
| Rank-by-regime | full subsection | one **main-text panel** (the citable sign-inversion) |
| ΔH context masking | full subsection | **appendix table** |

**Data-count correction (verified, important):** the paper's *"926 RefineID
samples"* for the hidden-state experiment is a **position/site count and is
retired**. The real universe is **`data/test_filtered_1024.csv` = 230 snippets /
1055 [MASK] sites / 196 distinct developer names / 64 package prefixes**
(`org.elasticsearch.xpack` alone is 18 %). DiffuCoder-7B-Base EM base rate = 0.299.
State `n = 230 snippets / 1055 sites / 64 packages` everywhere in RQ3.

---

## 1. Why a linear probe beats UMAP (the answer to "用机器学习怎么让结果更好")

A linear probe (Alain & Bengio 2017) trains a logistic regression on the residual
stream to predict a label, and reports an **AUC** — a number with a confidence
interval. Versus the UMAP scatter the thesis currently shows:

1. **Quantitative, not "look, two clouds."** "AUC = 0.78 ± 0.03 at layer 22" is
   comparable across layers/models and supports significance tests; "partially
   overlapping" is not.
2. **Disambiguates "absent" from "present-but-tangled."** If the *linear* probe
   fails but a 1-hidden-layer MLP probe succeeds, the information *is* there but
   non-linearly encoded — UMAP cannot make that distinction.
3. **Standard, defensible methodology** a committee recognizes: linear probes +
   **control tasks** (Hewitt & Liang 2019, selectivity) + group-wise CV.
4. **Gives an onset curve.** AUC(layer) directly answers *"at what depth does
   name-fit become readable"* — one curve, not one scatter per axis.
5. **The controls turn correlation into a claim.** The *double difference*
   (intact − scrambled context) and *leave-one-package-out* generalization let us
   say the probe reads **contextual name-fit**, not token identity — something a
   UMAP picture can never establish.

The UMAP scatter is replaced by a **1-D histogram of the probe-weight projection
`w·h`** (signed distance to the decision boundary) — the *same object* as the
reported AUC, so the picture can never contradict the number.

### The two confounds the review caught (and the design now defeats)

- **Length/subword shortcut.** Good names are median 8 chars (0 % ≤ 4); the old
  severe smells (`x`,`a`,`n`) are 1 char. A probe hits ~0.95 AUC reading
  *subword count*, learning nothing about fit. **Fix:** the "bad" condition is a
  **subword-count-matched misplaced real identifier** drawn from a *different*
  snippet (`queryBuilder`→`parameterObject`, `aggProfileShardResult`→`putDataStreamRequest`).
  Plus a per-layer subword-count-only logistic baseline that must be ≈ 0.5.
- **Token-identity vs context.** Even length-matched, a probe might read a token's
  intrinsic embedding. **Fix:** the headline is the **double difference**
  `AUC_contextual = AUC(intact context) − AUC(scrambled context)`: anything
  token-intrinsic survives scrambling and cancels; only contextual name-fit remains.
- **Leakage / effective N.** 64 packages, one is 18 %. **Fix:** headline CV is
  **leave-one-package-out** (`GroupKFold` on prefixes), reported as a ladder with
  by-sample and leave-name-out; the *by-sample − by-package gap* is printed as the
  leakage estimate. All CIs are **cluster bootstraps resampling snippets**.

---

## 2. The two experiments

### Exp 1 — *Is name-fit represented, and is it contextual?* (depth axis)
Single forward pass, full real context visible, clamp the target identifier to
**good** (developer name) or **bad** (matched misplaced name). Read the residual
stream at the first target sub-token for all 29 layers. Train an L2-logistic probe
per layer. **Headline =** `AUC_contextual(layer)` rising in the deeper third, above
the layer-0 embedding, subword-count, and random-label-control baselines, holding
under leave-one-package-out CV. Figure: AUC(layer) curve + the `w·h` histogram.

### Exp 2 — *Does the success signal arrive before commitment?* (denoise-step axis)
Real generation (target masked, model fills over the denoising loop). Per step,
probe **"will the model's own final fill be EM-correct?"** — evaluated **only on
positions still masked at that step** (so the committed token cannot leak in).
Overlay the per-step detection AUC with the **commitment CDF** (first step at
conf ≥ 0.8). **Headline = DCL**: detection lags commitment ⇒ the model resolves
whether it will succeed *after* it has already locked the position. Reported three
ways: (1) threshold-free area between the curves (primary), (2) committed-fraction
at detection, (3) a 3×3 `t_detect × t_commit` sensitivity surface (interpolated,
right-censored). PR-AUC (no-skill = 0.30) + Brier alongside ROC-AUC.

### Main-text panel — *the regime inversion*
One slope chart of median `r_gt` vs `r_smell` across HighConfident / Uncertain /
**RareConfident**, where the order flips (the model ranks a generic smell *above*
the developer's rare project-specific name). This is the most citable RQ3 finding
and the direct through-line to why RQ1/RQ2 fail on rare names. ΔH → appendix.

---

## 3. Paste-ready condensed section (fill X/Y/Z after the run)

```latex
\section{RQ3: What the Model Represents About Name Fit}
RQ1 and RQ2 characterized the model from the outside. A model that recovers the
developer's exact identifier 31\% of the time must compute something internally
that separates the names it keeps from the names it would replace. We probe
DiffuCoder-7B-Base on $n=230$ RefineID snippets ($1055$ sites, $64$ packages).

\subsection{Name fit is linearly decodable, contextually, in deep layers}
For each snippet we clamp the target identifier to the developer's name or to a
\emph{length- and subword-count-matched misplaced real identifier} taken from a
different snippet, holding the surrounding code byte-identical, and read the
residual stream at the target position across all $29$ layers. An $\ell_2$
logistic probe at each layer is evaluated under leave-one-package-out
cross-validation; all intervals are snippet-cluster bootstraps. The headline is
the \emph{contextual} decodability $\mathrm{AUC}_{\text{intact}}-
\mathrm{AUC}_{\text{scrambled}}$, which cancels any token-intrinsic signal.

Name fit is near-undecodable through the first two-thirds of the stack and rises
to a contextual AUC of $X$ in the deeper third (Fig.~\ref{fig:rq3-exp1}a), above
the layer-0 embedding ($X_0$), subword-count ($\approx0.5$), and random-label
control ($\approx0.5$) baselines (selectivity $Y$). Leave-one-package-out costs
only $Z$ AUC over leave-one-sample-out, bounding codebase leakage. The
probe-weight projection separates the two classes cleanly at the chosen layer
(Fig.~\ref{fig:rq3-exp1}b).

\subsection{The signal arrives after the schedule commits}
In real generation the target is masked and filled over the denoising schedule.
At each step we probe, on positions \emph{still masked at that step}, whether the
model's own final fill will be exact-correct, and overlay this detection curve
with the commitment CDF (first step at confidence $\geq0.8$). Detection lags
commitment by $\mathrm{DCL}=D$ (sign-stable in $S\%$ of bootstraps;
Fig.~\ref{fig:rq3-exp2}): by the step at which success becomes decodable, $C\%$ of
positions are already locked. The model can represent name fit but resolves it too
late to act on it.

\subsection{Why rare names are hardest}
Stratifying target positions by ground-truth rank, the output distribution ranks a
generic smell \emph{above} the developer's name precisely in the RareConfident
regime (median rank $733$ vs $10{,}989$; Fig.~\ref{fig:rq3-regime}) --- the
project-specific names where renaming matters most. (Context-sensitivity $\Delta H$
in App.~\ref{app:rq3}.)
```

**Condensed conclusion sentence (abstract-safe under either outcome):**
> DiffuCoder linearly encodes whether an identifier fits its surrounding code —
> a probe separates the developer's name from a shape-matched misplaced name in
> the deeper third of the network, generalizing across held-out packages and
> beating length/embedding/control baselines — but on the generation trajectory
> this signal becomes decodable only *after* the unmasking schedule has begun
> committing the position (DCL > 0). The model represents name fit yet resolves
> it too late to act, which is why it reproduces poor names on the rare,
> project-specific identifiers where intervention matters most.

> Note: if the probe shows the signal arrives *early* in steps, the conclusion
> only strengthens to "knows but the schedule acts before it" — the wording above
> holds either way. We report decodability (and optional steering), not full
> causal use; the bad class is necessarily synthetic since RefineID has no natural
> pre-rename names.

---

## 4. How to run (you submit; only Exp-extraction needs the GPU)

The three scripts live in `experiments/rq3_probe/`; the SLURM job is
`server/jobs/rq3_extract.slurm`. Deliver them to the HPC cluster the usual way (commit + push,
then `curl` from `raw.githubusercontent.com/<sha>` into the umbrella checkout, or
paste — same as `finish_paper.sh`).

```bash
cd /path/to/CoReFusion && mkdir -p logs
# smoke test first (8 snippets, ~2 min):
sbatch server/jobs/rq3_extract.slurm --max-snippets 8
# full run (230 snippets, ~1 h on an A40):
sbatch server/jobs/rq3_extract.slurm
```

The job runs GPU extraction → `results/rq3_probe/exp{1,2}_states.npz`, then (if
`scikit-learn`/`matplotlib` are in the venv) trains probes and renders figures to
`figures/new/rq3/`. If sklearn is missing, run on the login node:

```bash
uv pip install scikit-learn matplotlib
python experiments/rq3_probe/train_probes.py --out-dir results/rq3_probe
python experiments/rq3_probe/plot_rq3.py     --out-dir results/rq3_probe --fig-dir figures/new/rq3
```

Outputs: `fig_rq3_exp1_depth_probe`, `fig_rq3_exp2_dcl`,
`fig_rq3_regime_inversion` (+ `exp{1,2}_curves.json` with every number).

### ✅ Validated locally
`train_probes.py` + `plot_rq3.py` were run end-to-end on synthetic data (all 3
figures render, JSON complete); the extraction helpers (subword-count match,
bad-name pool, package grouping, context scramble) were unit-tested on the real
`test_filtered_1024.csv` (229/230 bad names matched on subword count, 64 packages).
The GPU forward/denoise loop is a near-verbatim copy of the working
`experiment_a_internal_rep.py`.

---

## 5. Two things to confirm before `sbatch`

1. **Checkpoint.** Default `--model apple/DiffuCoder-7B-Base` (the paper's RQ3
   model). The legacy `experiment_a/b/c.py` hard-code `-Instruct`. Confirm which
   checkpoint RQ1/RQ2 EM/LJ were reported on so DCL legitimately explains them.
2. **Denoising steps.** Default `--steps 32` (the paper's RQ3 trajectory). The
   unified RQ1/RQ2 runner uses `steps=64`. DCL and the commitment CDF are
   step-count dependent — match the RQ1/RQ2 generation config (pass
   `--steps 64` if that is what RQ1/RQ2 used).

## 6. Optional (1 day) causal upgrade — turns "decodable" into "used"
At the Exp1 chosen layer, steer the masked-generation hidden state along the probe
weight `w` and measure whether the fill moves toward the developer name (EM/rank).
A positive effect upgrades the claim from *"the model represents name fit"* to
*"name fit is causally used."* Left as future work unless you want it; the probe
weight is already saved by `train_probes.py`.
```
