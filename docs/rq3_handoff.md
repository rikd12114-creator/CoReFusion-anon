# RQ3 redesign — complete handoff for the LaTeX-writing agent

This is the single source of truth for rewriting RQ3 in the thesis. Everything
below is final and verified. Ready-made LaTeX (figures, section, table, captions)
is in **`docs/rq3_paper_ready.tex`** (blocks A–H); this file gives the full
context, all numbers, and the exact OLD→NEW replacements for the main text.
Only RQ3 is in scope (RQ1/RQ2 untouched).

---

## 0. TL;DR of what changed

The old RQ3 was **four loosely-coupled probes** (output-rank by regime;
context-masking ΔH; cosine + **UMAP** of smelly-vs-clean hidden states; commitment
step) ending in a vague conclusion ("a signal exists but is partial,
regime-dependent, late"). The redesign replaces it with **two quantitative
linear-probe experiments + one kept panel**, with proper statistics, controls,
and a cross-model replication:

- **Exp1 (depth):** a linear probe shows *name-fit is contextually decodable but
  only in the deepest layers* — this **replaces the UMAP/cosine** analysis with a
  number (contextual AUC) + CIs.
- **Exp2 (trajectory):** a per-step probe shows *the model has no early readable
  signal of its own success* before it commits a fill.
- **Regime panel kept:** the RareConfident rank **inversion** (most citable old
  finding) stays as one main-text figure; ΔH → appendix.
- **Replicated on DreamCoder-7B**, so the finding is not single-checkpoint.

Net: 4 subsections / ~9 panels / vague prose → **2 experiments + 1 panel / 4
figures / 1 table / two quantitative headlines**.

---

## 1. Dataset / universe facts (use these everywhere; the old "926" is WRONG)

- Probed split: `data/test_filtered_1024.csv` = **230 snippets / 1055 [MASK] sites
  / 196 distinct developer names / 64 source-package prefixes** (one package,
  `org.elasticsearch.xpack`, is 18%). The thesis's "**926 samples**" was a stale
  position-level count and must be **retired** — say *230 snippets / 1055 sites /
  64 packages*.
- Models: **DiffuCoder-7B-Base** (primary) and **DreamCoder-7B**
  (`Dream-org/Dream-Coder-v0-Instruct-7B`, replication). Both masked discrete
  diffusion, 28 layers (29 hidden states), hidden dim 3584, `<|mask|>`, 32
  denoising steps, k=2 canvas.
- The "bad name" class is **necessarily synthetic** (RefineID has no natural
  pre-rename names); this is stated as a limitation, not hidden.

---

## 2. Exp1 — "Is name-fit represented, contextually, and where?" (depth axis)

**Setup.** Per snippet, build a good/bad pair differing only at the target
identifier, with byte-identical surrounding code:
- **good** = developer's name clamped at the target;
- **bad** = a **length/sub-word-count–matched MISPLACED real identifier** taken
  from a *different* snippet (e.g. `queryBuilder`→`parameterObject`,
  `aggProfileShardResult`→`putDataStreamRequest`). This is the crucial design
  choice: it turns "real word vs short throw-away" into "right name vs equally-
  shaped wrong name in this context." (229/230 matched on sub-word count.)

Read the residual stream at the target's first sub-token for layers 0–28; train an
ℓ2 logistic probe per layer. **Headline = the double difference**
`AUC_contextual(ℓ) = AUC_intact(ℓ) − AUC_scrambled(ℓ)`, where *scrambled* clamps
the same tokens but permutes the surrounding code — token-intrinsic separability
cancels, leaving only context-dependent name-fit. Evaluation: **leave-one-package-
out GroupKFold** (headline grouping; gap to snippet-grouping = leakage estimate);
all CIs are **snippet-cluster bootstraps**. Controls: sub-word-count-only logistic
baseline, layer-0 embedding baseline, Hewitt–Liang random-label control task
(selectivity = AUC − control).

**Results (DiffuCoder-7B-Base):**
- Intact AUC **0.956** (95% CI [0.939, 0.972]); scrambled **0.693**;
  **contextual AUC = 0.263** at the output layer (28).
- Shape: contextual is **flat ≈0.10 through the first two-thirds** (early-third
  mean 0.103) then **rises monotonically over the final third** (late-third mean
  0.186) to peak 0.263. The *scrambled* curve falls from 0.87 mid-stack to 0.69 at
  the output → deep layers increasingly depend on real context.
- Controls all pass: sub-word-count baseline **0.500**, random-label control
  **0.534** (selectivity **0.422**), layer-0 embedding **0.301**, leakage gap
  (snippet − package CV) **0.006**.
- The old throw-away smell vocab (tmp/x/foo) hits **AUC 1.000** — the length
  shortcut the matched design removes (appendix contrast).

**One-line claim:** *Whether an identifier fits its context is linearly decodable,
but the genuinely contextual component emerges only in the final third of the
layers (AUC 0.26), above all baselines and robust to held-out packages.*

The UMAP scatter is replaced by a **1-D histogram of the probe's out-of-fold
decision score** (good vs bad) at the chosen layer (AUC 0.96) — same object as the
reported AUC, so the picture cannot contradict the number.

---

## 3. Exp2 — "Does the success signal arrive before commitment?" (step axis)

**Setup.** Normal generation (target masked, filled over 32 steps via the
validated `diffusion_generate`; EM base rate **0.367**). At each step, capture the
hidden state at each target site and a *still-masked* flag; train a per-step probe
for "will the model's **own** final fill be exact," **evaluated only on positions
still masked at that step** (so the committed token cannot leak in — anti-
tautology). Overlay with the commitment CDF (first step at confidence ≥0.8).
Report ROC-AUC + PR-AUC (no-skill = 0.367) + Brier, snippet-clustered.

**Results:**
- Commitment concentrates in the **final third**: CDF 25% by step 21, 50% by step
  26, 75% by step 29.
- Detection is **near chance throughout**: ROC rises only to a **peak 0.63 at step
  22** and never reaches 0.70; PR-AUC sits at the 0.37 base rate. By the detection
  peak, **31% of positions are already committed**.
- **Robust to label noise (CIS check):** the strict-EM label is depressed by the
  k=2 canvas duplicating short names (`request`→`requestRequest`). Re-scoring
  success with the project's **CoReFusion Identifier Score** (CIS, mean of
  lev/nw/subtok similarity) lifts the peak only **0.63 → 0.67** (CIS≥0.5) and
  leaves it at the **same late step 22**. So the null is *not* an EM artifact.
- The earlier `DCL_area` metric is a normalization artifact when detection ≈ chance
  and was **dropped**; report the honest quantities (peak ROC & step, commitment
  median step, % committed at detection peak).

**One-line claim:** *During generation the model carries no early, internally-
readable signal of whether its own fill will be correct: commitment completes in
the final third of the schedule while a success probe stays near chance (peak ROC
0.63) until then — the model commits fills "blind."*

> ⚠️ FRAMING (important): Exp2 is an **honest null** — "no early success signal,"
> NOT the optimistic "knows but acts too late." Do not overclaim. It is consistent
> with Exp1 (the discriminative signal lives in deep layers / final steps, too late
> to act) and with the abstract's "after the schedule has confirmed its predictions."

---

## 4. Cross-model replication (DreamCoder-7B) — answers Mali's only RQ3 point

Re-ran Exp1 unchanged on DreamCoder-7B. **It replicates, slightly stronger:**

| | DiffuCoder-7B-Base | DreamCoder-7B |
|---|---|---|
| contextual AUC (peak, output) | 0.263 | **0.297** |
| intact / scrambled | 0.956 / 0.693 | 0.978 / 0.681 |
| early-third → late-third | 0.103 → 0.186 | 0.110 → 0.199 |
| length baseline | 0.500 | 0.500 |
| selectivity | 0.422 | 0.511 |
| leakage gap | 0.006 | 0.006 |
| layer-0 embedding | 0.301 | 0.303 |

The two contextual-AUC(layer) curves nearly overlap (flat early, rising in the
deep third). → *the deep-layer name-fit representation is a property of the
masked-diffusion family, not one checkpoint.*

---

## 5. Regime panel (kept in main text, ~3 sentences)

Stratify target positions by ground-truth rank: **HighConfident** (median r_gt 77
vs r_smell 492), **Uncertain** (524 vs 676), **RareConfident** — the ordering
**inverts**, generic smell at 733 vs the developer's project-specific name at
**10,989**. These rare names are exactly where the late, weak fit signal cannot
help — the through-line to RQ2. (ΔH context-sensitivity → appendix: HighConfident
2.01, RareConfident 2.23, Uncertain 2.42; 0.41-bit spread.)

---

## 6. Figures (all rendered, PNG+PDF in `figures/new/rq3/`)

| file | what it shows | where |
|---|---|---|
| `fig_rq3_exp1_depth_probe.pdf` | (a) contextual-AUC(layer) + intact/scrambled/baselines/control, deep-third shaded; (b) 1-D OOF probe-projection histogram good vs bad (AUC 0.96) — **replaces UMAP** | §Exp1 |
| `fig_rq3_exp2_dcl.pdf` | per-step detection ROC/PR + commitment CDF; annotation "peak ROC 0.63@22, committed 31%, does NOT precede commitment" | §Exp2 |
| `fig_rq3_regime_inversion.pdf` | r_gt vs r_smell across 3 regimes, RareConfident crossing | §regime |
| `fig_rq3_compare_models.pdf` | DiffuCoder vs DreamCoder contextual-AUC(depth), two near-overlapping curves | §cross-model |

Ready `\begin{figure}` blocks + captions: **`docs/rq3_paper_ready.tex` block C**.

---

## 7. Consolidated number sheet (every value the LaTeX needs)

```
Universe:        230 snippets / 1055 sites / 196 names / 64 packages; DiffuCoder-7B-Base + DreamCoder-7B; 32 steps; k=2.
Exp1 DiffuCoder: intact 0.956 [0.939,0.972] | scrambled 0.693 | CONTEXTUAL 0.263 @layer28
                 early-3rd 0.103 | late-3rd 0.186 | len-base 0.500 | control 0.534 (selectivity 0.422)
                 layer0 0.301 | leakage gap 0.006 | smell-vocab ceiling 1.000 | hist AUC 0.96
Exp1 DreamCoder: intact 0.978 | scrambled 0.681 | CONTEXTUAL 0.297 @layer28 | selectivity 0.511 | leakage 0.006
Exp2:            EM base 0.367 | commit CDF 25%@21,50%@26,75%@29 | detection peak ROC 0.631@step22 (never>=0.70)
                 mean ROC 0.517 | mean PR 0.344 (no-skill 0.367) | committed@peak 31%
                 CIS robustness: EM 0.631 -> CIS>=0.5 0.671 -> CIS>=0.7 0.624, all @step22
Regime:          High 77/492 | Uncertain 524/676 | Rare 10989/733 (inverts) | dH: 2.01/2.42/2.23
```

---

## 8. MAIN-TEXT PATCHES (exact OLD → NEW; OLD quoted verbatim from the current PDF)

The thesis source is in Overleaf. Apply these find→replace edits. New text is also
in `docs/rq3_paper_ready.tex` (blocks A/B/E/F/G).

### 8.1 Abstract  (replace the RQ3 sentence)
**OLD:**
> Probing the internal states of DiffuCoder-7B shows why: the signal that tells a
> bad name from a good one appears only in the last few layers and the last few
> denoising steps, after the unmasking schedule has already confirmed most of its
> predictions. Providing the rename positions as masks bypasses this timing
> problem, which is why dLLMs work as filling engines but not as standalone
> refactoring agents.

**NEW:** use `rq3_paper_ready.tex` **block A**. (A linear probe quantifies it:
contextual name-fit AUC 0.26 emerging only in deep layers, replicated on
DreamCoder; during generation no early success signal — commitment in the final
third while a success probe stays near chance.)
> NOTE (out of RQ3 scope but adjacent): Mali flags the earlier abstract clause
> "When the same dLLMs must instead **find the positions on their own**" as an
> overclaim — RQ2 still supplies positions as masks. If softening the abstract,
> change to "must rely on uninformative/adversarial context." Leave to the RQ2
> pass; noted here only for consistency.

### 8.2 RQ3 section (Section VI)  — REPLACE WHOLE SECTION
Delete the four-probe Section VI ("VI. RQ3: INTERNAL-STATE DIFFERENCES…" through
its Analysis subsection) and insert `rq3_paper_ready.tex` **block B** (the new
two-experiment section), with figures from **block C** and Table from **block D**.

### 8.3 Methodology (Section III-D3, "RQ3 protocol")  — REPLACE
Replace the old four-experiment protocol paragraph with `rq3_paper_ready.tex`
**block E** (depth probe + schedule probe protocol).

### 8.4 Conclusion (Section IX)  (replace the RQ3 sentence)
**OLD:**
> Inside DiffuCoder-7B-Base, four experiments show that a representation
> distinguishing well-named from poorly-named identifiers exists but is partial,
> regime-dependent, and concentrated in the final third of both the layer stack
> and the denoising trajectory (RQ3).

**NEW:** use `rq3_paper_ready.tex` **block F**.

### 8.5 Threats to validity  (3 edits)
**External — OLD:**
> …and the internal-state experiments in Section VI use DiffuCoder-7B-Base alone.
> The findings characterize this specific scope;…

**External — NEW:** "…the depth probe replicates on DreamCoder-7B
(Fig.~\ref{fig:rq3-compare}), so the deep-layer name-fit signal is not specific to
DiffuCoder-7B-Base; the internal-state findings still use two masked-diffusion code
models on 230 snippets (64 packages), and other languages or generation
configurations may differ."

**Conclusion-validity — OLD:**
> The Welch t-test and Mann–Whitney U test in Section VI are robust to the unequal
> sample sizes (158 vs. 19,838) we use.

**Conclusion-validity — NEW** (also fixes Mali's "Welch t-test appears nowhere"):
"RQ3 reports per-layer and per-step AUCs with simultaneous snippet-cluster
bootstrap bands and a pre-registered late-vs-early-third contrast rather than a
single selected cell; grouping is leave-one-package-out to avoid optimistic
intervals."

**Construct — OLD:**
> The RQ3 smelly class is operationalized through a curated vocabulary (tmp, x,
> foo, single-letter aliases) and applies to the obvious-smell regime rather than
> to subtler real-world smells.

**Construct — NEW:** "The RQ3 bad class is a length- and sub-word-count–matched
misplaced real identifier (the curated tmp/x/foo vocabulary is retained only as an
appendix length-shortcut ceiling); since RefineID has no natural pre-rename names
the bad class is necessarily synthetic, and the headline is the contextual double
difference (intact − scrambled) rather than raw AUC."

---

## 9. Mali feedback → status (RQ3 only)

| Mali point (RQ3) | status |
|---|---|
| "rests mainly on DiffuCoder-7B-Base; replicate on DreamCoder or make central caveat" | **DONE** — replicated on DreamCoder-7B (§4), now a result not a caveat |
| "statistical support can be better" | **DONE** — bootstrap CIs, leave-one-package-out CV, 3 control baselines, selectivity |
| "claims may stretch beyond experiments" | **DONE** — Exp2 honest-null wording; abstract/conclusion rewritten to quantified, non-overreaching claims |
| (minor) "Welch t-test appears nowhere" | **FIXED** — redesign removes t-tests; threats 8.5 rewritten |

---

## 10. Source artifacts (for the LaTeX agent to pull from)
- `docs/rq3_paper_ready.tex` — ready blocks A–H (abstract, section, figures, table,
  methods, conclusion, threats, appendix pointer). **Primary LaTeX source.**
- `docs/rq3_redesign.md` — design rationale + why-probe-beats-UMAP.
- Figures: `figures/new/rq3/*.{pdf,png}` (4).
- Numbers JSON: `results/rq3_probe/exp1_curves.json`, `exp2_curves.json`,
  `exp2_cis_curves.json`, `results/rq3_probe_dreamcoder/exp1_curves.json`.
- Scripts (reproduce everything): `experiments/rq3_probe/{extract_probe_states,
  train_probes,plot_rq3,exp2_cis_label,plot_compare_models}.py`.
