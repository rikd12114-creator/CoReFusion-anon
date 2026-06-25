# CoReFusion ICSE27 — Revision Plan

Target paper file: `/home/user/Desktop/CoReFusion_ICSE27/conference_101719.tex`
New artifacts root: `/path/to/CoReFusion/` (`figures/new/`, `figures/rebuilt/`, `results/`, `docs/`)
Plan written: 2026-06-14.

---

## 1. Summary of what changed

Five independent workstreams have produced new material that must be folded into the submission. None of them touch RQ1's headline benchmark (Table II EM/LJ values, the stratification result, the ablations) — RQ1 is stable and only gains a new metric column and figure restyle.

1. **RQ2 re-run (June 2026-06-12).** The deobfuscation experiments were re-run and the section rewritten (`docs/rq2_deobfuscation_rewrite.tex`). The numbers shift slightly: all-masked target-EM 12.2/13.9 → **12.4/13.5**; target-only 3.1/4.1 → **2.8/4.7**; the copy-bias breakdown moves to 931 wrong / 97.2% / 671 (72.1%) short-copy / 189 (20.3%) long / 71 (7.6%) empty; one Table VI cell flips (`bytes` prediction `b`→`f`); per-bucket Fig 8(b) is regenerated; and a previously-stated "declines monotonically" claim is **deleted** because it was never monotone for DiffuCoder. RQ1 clean-context column (31.1/33.2) is unchanged. The abstract's "drops to about 3%" remains correct (2.8 ≈ 3).

2. **RQ3 redesign (partial).** RQ3 is being shrunk from 4 qualitative probes (cosine-trajectory, UMAP, entropy-regime, commitment-step + a rank table) to **2 quantitative linear-probe experiments + a main-text regime-inversion slope panel** (`docs/rq3_redesign.md`). **Exp 1 (depth probe) is done and strong** (contextual AUC ≈ 0.263 at layer 28; all four sanity controls pass; `results/rq3_probe/exp1_curves.json` + `figures/new/rq3/fig_rq3_exp1_depth_probe.{pdf,png}`). **Exp 2 (DCL) is BLOCKED** — the extraction is degenerate (EM-correct 2/1055), so `exp2_curves.json` and `fig_rq3_exp2_dcl` do not exist and **no DCL number can be quoted yet**. The regime-inversion figure exists but is rendered from **paper Table VII constants**, not a fresh run.

3. **New evaluation metric (CIS).** A new training-free metric, the **CoReFusion Identifier Score = mean of 4 similarity metrics** (IdBench `lev`+`nw`, subtoken `jacc`+`fuzzy`), is defined in `analysis/identifier_similarity_metrics.py` / `analysis/metric_eval_final.py`. It tracks the expensive LLM-judge far better than EM (AUC→LJ 0.9588 vs EM 0.7708; leaderboard Spearman 0.9718 vs 0.7267; judge noise ceiling 0.924). A full benchmark leaderboard with EM/LJ/CIS for 22 models exists (`figures/new/leaderboard_benchmark_cis.csv`) plus two figures (scatter + rank-bump). Two intrinsic "quality" metrics (M3) are defined but deliberately excluded from CIS.

4. **Restyled figure set.** All 13 active paper figures have been recolored/rebuilt to a consistent house style as `fig01`–`fig13` in `figures/rebuilt/`. These are drop-in replacements (with three double-`.pdf.pdf` filename traps and one 2-up→1-file UMAP caveat).

5. **Richer dataset table + DreamOn-Java supplement.** A new `docs/refineid_table.tex` re-derives the RefineID stats with a different declaration-site classifier (headline shift: Local variable 84.9%→48.3%, Formal parameter 10.3%→37.6%, plus re-bucketing and new scale rows), but it still has **4 unfilled `\refineidTBD{}` placeholders**. A standalone DreamOn-Java EM+LJ benchmark figure (`figures/new/fig4_dreamon_java.png`) exists but is not yet referenced anywhere.

---

## 2. Change-map (the core)

Line numbers refer to `conference_101719.tex`. Effort: S = mechanical (<15 min), M = paragraph-scale rewrite, L = section-scale rewrite or blocked-on-data.

| # | Paper location (section / table / figure / line) | Current content (1-line) | New result | Required edit | Evidence (file) | Effort | Risk |
|---|---|---|---|---|---|---|---|
| **— FIGURE SWAPS (mechanical) —** | | | | | | | |
| F1 | Fig `fig:DLM_denoising`, L128/130, `fig_dllm_demo.png` | original denoising demo | exact recolor | replace src with `fig01_dllm_denoising.pdf` | `figures/rebuilt/fig01_dllm_denoising.pdf` | S | low |
| F2 | Fig `fig:dreamon_canvas`, L153/155, `fig_dreamon_canvas.png` | DreamOn canvas | exact recolor | replace src with `fig02_dreamon_canvas.pdf` | `figures/rebuilt/fig02_dreamon_canvas.pdf` | S | low |
| F3 | Fig `fig:fixed_canvas`, L181/185, `fig_fixed_canvas.png` | fixed canvas | exact recolor | replace src with `fig03_fixed_canvas.pdf` | `figures/rebuilt/fig03_fixed_canvas.pdf` | S | low |
| F4 | Fig `fig:rq2_example`, L314/318, `fig_rq2_example.pdf` | RQ2 obfuscation example | exact recolor | replace src with `fig04_rq2_example.pdf` | `figures/rebuilt/fig04_rq2_example.pdf` | S | low |
| F5 | Fig `fig:single_multi`, L495/497, `fig_em_stratified_by_smell_positions.png` | EM by cardinality | rebuilt from Table III | replace src with `fig05_em_by_cardinality.pdf` | `figures/rebuilt/fig05_em_by_cardinality.pdf` | S | low |
| F6 | Fig `fig:mask_ablation`, L522/524, `fig_mask_ablation.pdf` | mask-count ablation | exact numbers | replace src with `fig06_mask_count_ablation.pdf` | `figures/rebuilt/fig06_mask_count_ablation.pdf` | S | low |
| F7 | Fig `fig:step_robustness`, L600/602, **`fig_step_robustness.pdf.pdf`** (double ext) | diffusion-step robustness | recreated from Table IV | replace src with `fig07_diffusion_steps.pdf` AND **fix the double `.pdf.pdf` in the `\includegraphics` path** | `figures/rebuilt/fig07_diffusion_steps.pdf` | S | med (filename trap) |
| F8 | Fig `fig:rq2_deobfuscation`, L735/737, `fig_rq2_deobfuscation.pdf` | RQ2 results | **two choices** (see Open Decision 1): `fig08_rq2_results.pdf` (thesis numbers) OR `figures/new/fig8_rq2_em_june.pdf` (June re-run, matches the rewrite) | replace src — **prefer the June file** to stay consistent with the rewritten Table V | `figures/new/fig8_rq2_em_june.pdf` ; `figures/rebuilt/fig08_rq2_results.pdf` | S | med (number consistency) |
| F9 | Fig `fig:rq3_context_sensitivity`, L891/893, **`fig_rq3_context_sensitivity.pdf.pdf`** | RQ3 context sensitivity | reproduced; **demoted to appendix** in redesign | move to appendix OR delete (see RQ3 rows); if kept, `fig09_context_sensitivity.pdf` + fix double ext | `figures/rebuilt/fig09_context_sensitivity.pdf` | M | med |
| F10 | Fig `fig:rq3_edit_signal`, L905/907, `fig_rq3_edit_signal.pdf` | cosine edit-signal | **REPLACED** by Exp1 depth probe | delete; replace section with `fig_rq3_exp1_depth_probe.pdf` | `figures/new/rq3/fig_rq3_exp1_depth_probe.pdf` | M | med |
| F11 | Fig `fig:rq3_umap`, L914+915/917, `fig_rq3_umap_layers.pdf` + `fig_rq3_umap_steps.pdf` (2-up) | UMAP scatter | **REPLACED** by 1-D probe-projection histogram | delete both; covered by Exp1 panel (b) | `figures/new/rq3/fig_rq3_exp1_depth_probe.pdf` (panel b) | M | med |
| F12 | Fig `fig:rq3_entropy_regime`, L926/928, **`fig_rq3_entropy_regime.pdf.pdf`** | entropy change | **MOVED to appendix** | move ΔH content + figure to `app:rq3`; if rebuilt-style wanted use `fig12_entropy_change.pdf` + fix double ext | `figures/rebuilt/fig12_entropy_change.pdf` | M | med |
| F13 | Fig `fig:rq3_unmasking`, L942/944, `fig_rq3_unmasking.pdf` | commitment/unmasking step | **KEPT but recast** into the commitment-CDF inside Exp2 DCL figure (BLOCKED) | hold until Exp2 data fixed; interim keep `fig13_commitment_step.pdf` | `figures/new/rq3/fig_rq3_exp2_dcl.*` (**MISSING**) ; `figures/rebuilt/fig13_commitment_step.pdf` | L | **high (blocked)** |
| **— METRICS / CIS —** | | | | | | | |
| M1 | Table II `tab:models`, L387–425, columns Model/Arch/Params/EM/LJ | 21-row EM+LJ leaderboard | new CIS column for 22 models | **add a `CIS (%)` column** between EM and LJ (do not replace); transcribe rounded values | `figures/new/leaderboard_benchmark_cis.csv` | M | med (partial denominators, see Risk R7) |
| M2 | Evaluation Metrics §, L355–365 (only EM + LJ defined) | EM (eq:em) + LJ defined | 4 sim metrics + CIS + (excluded) M3 quality | **add a new subsection**: all-sites consistency gate, M1 `lev`/`nw` (IdBench ICSE'21), M2 `jacc`/`fuzzy` (Wong et al. ESEM'25), define **CIS = mean of 4 sims**; note M3 quality defined-but-excluded | `analysis/identifier_similarity_metrics.py`, `analysis/metric_eval_final.py` | M | low |
| M3 | New Results subsection (after RQ1 or RQ2) | none | CIS-vs-LJ proxy validation | **add short subsection** "Is CIS a cheaper proxy for the judge?": AUC(CIS→LJ)=0.9588 vs ceiling 0.924 vs EM 0.7708; leaderboard Spearman 0.9718 vs 0.7267; MASK contamination 42.8%; anchor with the two new figures | `figures/new/benchmark_cis_scatter.pdf`, `figures/new/benchmark_metric_ranks.pdf`, `results/identifier_metrics/metric_eval_final.json` | M | low |
| **— RQ2 SECTION (V) —** | | | | | | | |
| R2-1 | Table IV `tab:rq2_deobfuscation`, L717–731 | all-masked 12.2/13.9; target-only 3.1/4.1 | **12.4/13.5** and **2.8/4.7** | replace 4 cells (RQ1 clean ctx 31.1/33.2 unchanged) | `docs/rq2_deobfuscation_rewrite.tex` Table V; `results/deobfuscation_refineID/reproduced/rq2_numbers.json` | S | low |
| R2-2 | Table V `tab:copypaste`, L758–778 | row id=9 `bytes` → prediction `b` | prediction **`f`** | change that one cell; 8 other rows + † rows {3,12,13} unchanged | `docs/rq2_deobfuscation_rewrite.tex` Table VI | S | low |
| R2-3 | Fig `fig:rq2_deobfuscation`, L735/737 | thesis-number RQ2 figure | June re-run figure | swap to `figures/new/fig8_rq2_em_june.pdf` (see F8) | `figures/new/fig8_rq2_em_june.pdf` | S | med |
| R2-4 | §rq2_allmasked prose L713–715 | 12.2/13.9; "EM declines as identifier count grows" | 12.4/13.5; crowded files ≈2%; **monotonic claim removed** | update numbers; delete the "declines as count grows / robust-to-sites" monotonicity framing | `docs/rq2_deobfuscation_rewrite.tex` Sec V | M | med (claim deletion) |
| R2-5 | §rq2_targetonly prose L743–745 | 3.1/4.1; "9–10% below all-masked" | 2.8/4.7; **"9.6 and 8.8 pp below"** | update numbers | rewrite Sec V | S | low |
| R2-6 | §rq2_copypaste prose L748–783 | 928 wrong; 96.9%; 662/71.3%; 189/20.4%; 77/8.3%; example `parseInt` | **931 wrong; 97.2%; 671/72.1%; 189/20.3%; 71/7.6%; example `randomGenerator`** | update all six numbers + the example token (`fConfig` kept); copy-classifier `^[a-z]{1,2}$` now stated explicitly | rewrite Sec V-C; `analysis/reproduce_rq2_deobfuscation.py` L88 | M | low |
| R2-7 | §rq2_summary prose L787–793 | "3–4%"; "71.3%"; "declines monotonically" | **"2.8–4.7%"; "72.1%"; monotonic claim deleted** | update numbers; remove the monotone-decline sentence | rewrite Sec V-D | M | med |
| R2-8 | §rq2 typos L671 "co-referrin", L743 "keep them" | typos | — | fix while editing | — | S | low |
| **— RQ3 SECTION (VI) —** | | | | | | | |
| R3-1 | §rq3 intro L800 "We design four experiments" | four-experiment framing | two experiments + regime panel | rewrite intro: 2 linear-probe experiments + main-text regime-inversion panel; ΔH demoted to appendix | `docs/rq3_redesign.md` | M | med |
| R3-2 | Subsec A `sec:rq3_confidence` + Table `tab:rq3_overconfidence` L834–862, 847–860 | full rank-by-regime table + prose (77/492, 524/676, 10989/733) | **compressed** to one slope panel | replace the 4-row table/subsection with `fig:rq3-regime` slope chart (same constants 77/492, 524/676, 10989/733) | `figures/new/rq3/fig_rq3_regime_inversion.pdf` | M | med |
| R3-3 | Subsec B `sec:rq3_context` + Fig `fig:rq3_context_sensitivity` L884–898 | ΔH context-sensitivity subsection | **MOVED to appendix `app:rq3`** | cut from main text; place ΔH numbers (2.01/2.23/2.42, spread 0.41) + figure in appendix | `figures/rebuilt/fig12_*`/`fig09_*` (appendix) | M | med |
| R3-4 | Subsec C `sec:rq3_trajectory` (cosine + UMAP + entropy) L900–932 | cosine trajectory, UMAP, entropy-regime | **REPLACED** by Exp 1 depth probe | delete cosine/UMAP/entropy prose+figs; write Exp 1: contextual AUC 0.263 @ layer 28, intact 0.956/scrambled 0.693, length baseline 0.50, selectivity 0.422, leakage 0.006, late>early; figure = `fig:rq3-exp1` | `results/rq3_probe/exp1_curves.json`; `figures/new/rq3/fig_rq3_exp1_depth_probe.pdf`; `docs/rq3_redesign.md` | L | med |
| R3-5 | Subsec D `sec:rq3_unmasking` + Fig `fig:rq3_unmasking` L935–947 | commitment-step (20 snippets; 13.6 vs 11.0; MWU) | **RECAST** into Exp 2 DCL (commitment CDF + detection AUC), n=230/1055 | rewrite as Exp 2 + DCL metric — **BLOCKED: Exp2 data degenerate, no DCL value, `fig_rq3_exp2_dcl` missing**. Interim: keep old 13.6/11.0 prose OR mark as TODO | `figures/new/rq3/fig_rq3_exp2_dcl.*` (**MISSING**); `results/rq3_probe/exp2_curves.json` (**MISSING**) | L | **high (blocked)** |
| R3-6 | §rq3 Analysis `sec:rq3_synthesis` L1011–1027 | synthesis of 4 probes incl. 86% entropy, 0.35–0.65 cosine, 2.6-step gap | new synthesis from 2 experiments | rewrite around contextual AUC + (pending) DCL; drop cosine/UMAP/86% claims; keep regime-inversion through-line | `docs/rq3_redesign.md` | M | med (depends on R3-5) |
| **— DATASET TABLE (I) —** | | | | | | | |
| D1 | Table I `tab:refineid_stats`, L237–275 | Local var 84.9%, Formal param 10.3%, field 4.8%, method 0%; camelCase 59.9%; buckets 2–5/6–10/11–20/21+ | re-derived: **Local 48.3%, Param 37.6%, field 3.8%, method 1.2%, usage-only 9.1%**; segmented surface forms; re-bucketed 2–4/5–10/>10; +7,712 occurrences, 847 distinct, package mix, max-length 40, P95/max file size | **replace with `docs/refineid_table.tex`** AFTER filling the 4 `\refineidTBD{}` cells (mining tool, commit window, #commits, clean/mixed split) and reconciling `\label` (`tab:refineid` vs `tab:refineid_stats`) | `docs/refineid_table.tex` | L | **high (placeholders + classifier change)** |
| **— ABSTRACT / INTRO —** | | | | | | | |
| A1 | Abstract L52–54 | "about 3%" target-EM; "more than ten points" vs CodeT5-large | 2.8% ≈ 3% (OK); intro/abstract baseline-count inconsistency pre-existing | "about 3%" stays; **resolve the abstract-vs-intro baseline framing** (CodeT5-large vs "23.5% of seventeen"); add CIS mention if it becomes a headline | paper:front extract §1 | M | med |
| A2 | Intro RQ2 finding L82 | "71.3% of wrong predictions copy lexical style" | **72.1%** | update 71.3→72.1 | `docs/rq2_deobfuscation_rewrite.tex` | S | low |
| A3 | Intro RQ3 finding L82 | "four probes ... final third / final third" | two experiments; "final third" still holds for depth (layer 28/29); DCL pending | rewrite to "linear probes" framing; keep "final third" for depth only; do not assert DCL number | `docs/rq3_redesign.md` | M | med |
| **— CONCLUSION (IX) —** | | | | | | | |
| C1 | Conclusion L1097 | "target-EM drops to 3–4%"; "71.3% copy"; "four experiments show ... final third of both" | **"2.8–4.7%"** (or keep "3–4%"); **72.1%**; "linear probes show ... final third of the layer stack" (drop "both axes" unless DCL lands) | update numbers + RQ3 framing; keep T=2/k=2 97.1%/32× claim | rewrite docs | M | med |
| C2 | Threats `sec:threats` L1069–1075 | n=158 vs 19,838; thresholds; "≥10% gaps" | RQ3 internal-state n changes to 230 snippets/1055 sites; new "bad class" = matched misplaced name | update internal-validity + conclusion-validity paragraphs to the linear-probe design and new n | `docs/rq3_redesign.md`; `results/rq3_probe/extract_meta.json` | M | med |
| C3 | Future Work `sec:future_work` L1080–1086 | DreamOn 55.9% LJ; "propagate signal to earlier layers"; T=2/k=2 hybrid | mostly stable; could cite DreamOn-Java supplement | optional: add DreamOn-Java pointer (Open Decision 4) | `figures/new/fig4_dreamon_java.png` | S | low |

---

## 3. Sequenced edit plan

Work in batches that recompile cleanly at each checkpoint. Copy needed PDFs from `figures/rebuilt/` and `figures/new/**` into `/home/user/Desktop/CoReFusion_ICSE27/figures/` first.

### Batch A — Figure swaps (mechanical, low-risk) — rows F1–F8
- **Files:** `conference_101719.tex` (`\includegraphics` lines 128,153,181,314,495,522,600,735); `figures/` dir.
- **Do:** Drop in `fig01`–`fig08`. For F7 fix the `\includegraphics{figures/fig_step_robustness.pdf.pdf}` double extension. For F8 use `fig8_rq2_em_june.pdf` (June re-run) so the figure matches the rewritten Table V — **do not** use the thesis-number `fig08_rq2_results.pdf` unless Open Decision 1 says otherwise.
- **Verify:** `latexmk -pdf` compiles with no missing-file or "unknown extension" warnings; visually confirm 8 figures render. Defer the three RQ3 double-ext files (F9/F12) to Batch C.

### Batch B — RQ2 section replace — rows R2-1…R2-8, A2
- **Files:** `conference_101719.tex` Section V (L668–795); source of truth `docs/rq2_deobfuscation_rewrite.tex`.
- **Do:** Replace Table IV cells (12.4/13.5, 2.8/4.7), Table V `bytes`→`f`, and the four prose subsections wholesale with the rewrite text. Delete the monotonic-decline sentences. Fix the two typos. Update intro L82 71.3→72.1.
- **Verify:** Recompile; cross-check every number against `results/deobfuscation_refineID/reproduced/rq2_numbers.json` (931/97.2%/671/189/71; 72.1%). Confirm no dangling `\ref` to deleted content.

### Batch C — RQ3 redesign — rows R3-1…R3-6, F9–F13, C2
- **Files:** `conference_101719.tex` Section VI (L796–1027) + appendix; sources `docs/rq3_redesign.md`, `results/rq3_probe/exp1_curves.json`.
- **Do:**
  1. Rewrite intro (R3-1) to 2 experiments + regime panel.
  2. Replace Subsec A table with regime-inversion slope figure (R3-2).
  3. Move Subsec B ΔH to appendix `app:rq3` (R3-3, F9/F12).
  4. Replace Subsec C with Exp 1 depth-probe prose + figure (R3-4, F10/F11) using exact JSON values (0.263, layer 28, 0.956/0.693, 0.50, 0.422, 0.006).
  5. **Exp 2 / DCL (R3-5, F13): BLOCKED.** Either (a) leave the old commitment-step subsection in place as a stopgap with a TODO, or (b) write the Exp 2 prose with `\TODO{DCL=...}` placeholders. Do **not** invent a DCL number.
  6. Rewrite Analysis (R3-6) and Threats (C2) to the new design.
- **Verify:** Recompile; Exp1 figure renders; appendix label resolves; no reference to deleted UMAP/cosine figures remains; confirm Exp2 placeholders are visually obvious so they are not shipped by accident.

### Batch D — Metrics / CIS — rows M1, M2, M3
- **Files:** `conference_101719.tex` Evaluation Metrics § (L355–365), Table II (L387–425), new Results subsection; sources `analysis/metric_eval_final.py`, `figures/new/leaderboard_benchmark_cis.csv`, the two scatter/rank PDFs.
- **Do:** Add CIS definition subsection (M2); add CIS column to Table II (M1); add the proxy-validation subsection with both figures (M3). Copy `benchmark_cis_scatter.pdf` and `benchmark_metric_ranks.pdf` into `figures/`.
- **Verify:** Recompile; Table II column alignment intact; CIS values rounded to 1 dp match the CSV; note in text or footnote the partial-denominator caveat (Risk R7).

### Batch E — Dataset Table I — row D1
- **Files:** `conference_101719.tex` Table I (L237–275); source `docs/refineid_table.tex`.
- **Do:** Only after the 4 `\refineidTBD{}` cells are filled from mining records and the `\label` is reconciled. Swap the table; update any in-text references to the old percentages (esp. "84.9% local variable" if cited elsewhere).
- **Verify:** Recompile; grep the paper for any prose that still quotes 84.9% / 10.3% / the old bucket boundaries and update them. **Do not** ship `\refineidTBD` red placeholders.

### Batch F — Propagate numbers to abstract / intro / conclusion — rows A1, A3, C1, C3
- **Files:** Abstract (L51–55), Intro (L75–100), Conclusion (L1087–1097), Future Work (L1077–1086).
- **Do:** Update RQ2 (72.1%, 2.8–4.7%) and RQ3 (linear-probe framing, "final third" for depth only) everywhere they appear. Resolve the pre-existing abstract-vs-intro baseline inconsistency (Open Decision 5). Optionally add DreamOn-Java + CIS mentions.
- **Verify:** Final full recompile; do a global grep for stale numbers (3.1, 4.1, 12.2, 13.9, 71.3, 928, 96.9, "four experiments", 84.9%) and confirm each remaining hit is intentional.

---

## 4. Open decisions for the author (the author)

1. **RQ2 figure source.** Use the June re-run figure (`figures/new/fig8_rq2_em_june.pdf`, matches the rewritten Table V 12.4/13.5/2.8/4.7) or the restyled thesis-number figure (`figures/rebuilt/fig08_rq2_results.pdf`, which encodes the OLD 12.2/13.9/3.1/4.1)? **Recommended default: the June file** — the rebuilt fig08 intentionally uses thesis numbers and would contradict the rewritten table.

2. **RQ3 Exp 2 / DCL — ship or defer?** The DCL experiment is blocked (extraction degenerate, EM-correct 2/1055; `exp2_curves.json` and `fig_rq3_exp2_dcl` absent). Options: (a) re-run the extraction with a working `diffusion_generate` before camera-ready and report DCL; (b) keep the old commitment-step subsection (13.6 vs 11.0, n=20) as the RQ3 step-axis result and present only Exp 1 + regime-inversion as "new"; (c) ship Exp 1 + regime-inversion only and drop the step axis entirely. **Recommended default: (a) if the re-run lands in time, else (b)** — never quote a DCL number until `exp2_curves.json` exists.

3. **Old RQ3 figures — delete or keep as appendix?** The redesign deletes cosine (`fig:rq3_edit_signal`) and UMAP (`fig:rq3_umap`) and moves entropy-regime (`fig:rq3_entropy_regime`) + ΔH context-sensitivity to the appendix. Confirm: are cosine/UMAP fully cut, or kept as appendix evidence? **Recommended default: cut cosine + UMAP entirely; keep ΔH + entropy-regime in appendix `app:rq3`** (matches `docs/rq3_redesign.md`).

4. **CIS in the headline — augment or replace LJ?** Recommended is to **add CIS as a third column** to Table II alongside EM and LJ (cheap, training-free proxy), not replace LJ. Confirm you want all three columns. **Recommended default: augment (EM | CIS | LJ), keep LJ as the semantic ground truth.**

5. **Abstract/intro baseline framing (pre-existing inconsistency).** Abstract says best non-dLLM is CodeT5-large beaten "by more than ten points"; intro says "23.5% of seventeen non-dLLM baselines" (= CodeT5+ 16B); a commented variant says "20.6% CodeT5-large." Table II confirms CodeT5+ 16B = 23.5% is the strongest non-dLLM. **Recommended default: standardize on "23.5% (CodeT5+ 16B), strongest of seventeen non-dLLM baselines" in both abstract and intro**, and fix the Seq2Seq count (abstract/intro say "five", Methods says "three" then lists six — Table II shows 6 Seq2Seq rows).

6. **Dataset Table I swap — now or camera-ready?** `docs/refineid_table.tex` has 4 unfilled `\refineidTBD{}` cells and a materially different classifier (Local var 84.9%→48.3%). Swapping it changes a headline dataset statistic. **Recommended default: hold until the placeholders are filled and the classifier change is verified/justified in text; do not ship a half-filled table.** If it cannot be completed, keep the current `tab:refineid_stats`.

7. **DreamOn-Java supplement — subsection, appendix, or omit?** `figures/new/fig4_dreamon_java.png` exists but is unreferenced and has no rebuilt counterpart. **Recommended default: appendix or a one-line Future-Work pointer**, not a main-text subsection, given page budget.

8. **Missing artifacts to confirm.** `exp2_curves.json`, `figures/new/rq3/fig_rq3_exp2_dcl.*`, and per-sim model-level columns for Table II are absent. The per-sim values can be re-aggregated from `results/identifier_metrics/fusion_long.csv` if needed. **Confirm whether you need per-sim columns** (default: no — report only the fused CIS).

---

## 5. Risks / inconsistencies found

- **R1 — RQ3 Exp 2 / DCL has no data.** `results/rq3_probe/exp2_meta.csv` is degenerate (EM-correct 2/1055, base rate 0.0019; predictions are mostly `)` or NaN). `train_probes.py` crashes (`ValueError: only one class`). Therefore `exp2_curves.json` and `fig_rq3_exp2_dcl.{png,pdf}` **do not exist** (confirmed on disk: `figures/new/rq3/` holds only `fig_rq3_exp1_depth_probe.*` and `fig_rq3_regime_inversion.*`). This is the known "Exp2 diffusion_generate fills garbage" bug (commits 9870c8f / c46f246). **No DCL/sign-stability/committed-fraction number may be quoted.** RQ3's denoise-step axis is blocked until re-extraction succeeds.

- **R2 — Regime-inversion figure is not from a fresh run.** `fig_rq3_regime_inversion.pdf` is rendered from `plot_rq3.py`'s hard-coded `PAPER_REGIMES` (= thesis Table VII constants 77/492, 524/676, 10989/733). It is internally consistent with the paper but is **not new evidence** — frame it as a re-presentation, not a re-measurement.

- **R3 — RQ2 figure/number mismatch risk.** The restyled `figures/rebuilt/fig08_rq2_results.pdf` encodes OLD thesis numbers (12.2/13.9/3.1/4.1), while the rewritten Table V uses June numbers (12.4/13.5/2.8/4.7). Using the rebuilt fig08 would contradict the table. Use `figures/new/fig8_rq2_em_june.pdf` instead (see Open Decision 1).

- **R4 — Two RQ2 metrics in play (do not cross-wire).** The headline rewrite uses **per-target EM** (12.4/13.5/2.8/4.7) and a per-position copy rate of **72.1%**. The `figures/new/fig1/fig2/fig3` family uses a stricter **all-sites consistency EM** (all-masked 1.8/1.9%, target-only 1.5/2.6%) and a consistency-based copy rate of **42–46%**. These are different metrics; only the per-target/`fig8_rq2_em_june` numbers belong in the rewritten section.

- **R5 — Dataset table contradicts current headline stat.** `docs/refineid_table.tex` re-derives **Local variable 84.9% → 48.3%** and **Formal parameter 10.3% → 37.6%** with a different declaration-site classifier, re-buckets sites-per-sample (2–4/5–10/>10 vs 2–5/6–10/11–20/21+), and reclassifies method names from "0% excluded" to 1.2%. This is a substantive change to a dataset description, not cosmetic. It also has **4 unfilled `\refineidTBD{}` placeholders** and a different `\label`. Must be reconciled and justified before swapping; check the paper does not assert "method names excluded" elsewhere.

- **R6 — Three double-`.pdf.pdf` filenames are real on disk** (`fig_step_robustness.pdf.pdf`, `fig_rq3_context_sensitivity.pdf.pdf`, `fig_rq3_entropy_regime.pdf.pdf` confirmed present in `/home/user/Desktop/CoReFusion_ICSE27/figures/`). The active `\includegraphics` lines (600, 891, 926) point at them. When swapping in rebuilt figures, either rename to the exact double-ext target or fix the path — otherwise LaTeX silently keeps using the old file.

- **R7 — CIS leaderboard rests on partial denominators.** Per `figures/new/MAPPING.md`, the RQ1 leaderboard predictions were not cleanly regenerated: DiffuCoder/DreamCoder are truncated (n=692/673), CodeT5p-2B=909, CodeT5p-6B=193, and DiffusionGemma-26B is all-empty/absent. The CIS column inherits these partial `n`. Also the headline leaderboard's EM (25.6/27.2 for the dLLMs) does **not** match Table II's EM (31.1/33.2) because of the different gated denominator — do not present the leaderboard EM as the Table II EM. Re-run the clean benchmark to n=1000 before camera-ready.

- **R8 — Two leaderboards must not be confused.** `figures/new/leaderboard_benchmark_cis.csv` (percentages, Qwen2.5-7B-Instruct judge, the paper file) vs `results/identifier_metrics/leaderboard_cis.csv` (fractions, multi-judge consensus, internal-validation only, much smaller n). Use the former for Table II.

- **R9 — CIS definition has two versions in the repo.** `metric_eval_and_fusion.py` explored a *trained* LogReg/GBDT fusion also called "CoReFusion Identifier Score"; the **adopted** definition is the unweighted **mean of the 4 sims** in `metric_eval_final.py` (which supersedes it; trained model gives no cross-family edge: 0.8161 vs 0.8153). Cite only the simple-mean definition.

- **R10 — Pre-existing internal inconsistencies (independent of new results), worth fixing in passing:** Seq2Seq count "five" (abstract/intro) vs "three" then six enumerated (Methods L232) — Table II has 6 rows; the abstract "more than ten points vs CodeT5-large" vs intro "23.5% of seventeen"; the k=2 justification wording "less than two sub-words" (should be "two or fewer"); the regime label `HighConfidence` (prose) vs `HighConfident` (table); the active-prose throughput claim "16×" (L609) vs table/L605 "32×" for T=2.

- **R11 — RQ3 universe count change.** Old RQ3 cited "926 RefineID samples" / "7,800 target positions" / "200 snippets"; the redesign standardizes on **230 snippets / 1055 sites / 64 packages** (`results/rq3_probe/extract_meta.json`). Every RQ3 n in main text, threats (158 vs 19,838), and captions must be updated to the new universe — and the old 926/7,800/200 numbers retired wherever they survive.
