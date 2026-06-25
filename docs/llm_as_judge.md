# LLM-as-a-Judge (LJ) — Setup and Reproduction

This document is the repository-side companion to the paper appendix
*"LLM-as-a-judge Setup"*. It records the exact configuration used to compute the
**LJ** column in the main results table, and maps every claim in the appendix to
the script, function, and line where it is implemented, so that the metric can be
reproduced from the released code alone (no paid API). The judge's raw per-sample
output, the leaderboard CSVs, and the scripts below constitute the supplementary
material for LJ.

---

## 0. Where it lives in the repo

| Component | File | Key symbols |
|---|---|---|
| Judge primitives (models, prompt, decoding, parsing) | `experiments/llm_judge_variable_naming.py` | `JUDGE_REGISTRY`, `SYSTEM_PROMPT`, `build_user_prompt`, `load_judge`, `judge_one`, `parse_verdict`, `extract_context` |
| Consistency-gated runner (produces the LJ in the main table) | `experiments/run_llm_judge_unified.py` | `is_consistent`, `judge_model_csv`, `summarize`, `write_combined` |
| Standalone judge over the separate Diffusion/FIM benchmark CSVs | `experiments/llm_judge_variable_naming.py` | `evaluate_file`, `main` |
| Leaderboard assembly (Table-II style, one column per judge) | `analysis/make_leaderboard.py` | `JUDGES`, `JUDGE_SHORT`, `lj_table` |
| Single-judge convenience wrapper (default Qwen2.5-7B) | `analysis/llm_judge.py` | `LLMJudge` |
| Cross-judge agreement (the 0.924 figure context) | `analysis/revision_major_stats.py` | `M3 CROSS-JUDGE AGREEMENT` |
| Judge ceiling on ground-truth-as-prediction | `experiments/judge_refineid_ground_truth.py`, `analysis/analyze_gt_judge_ceiling.py` | — |
| Heatmap / leaderboard visualisation | `analysis/viz_heatmap.py`, `analysis/viz_leaderboard.py` | — |

> **Two entry points, one judge.** `run_llm_judge_unified.py` *imports* the
> primitives from `llm_judge_variable_naming.py` (same `SYSTEM_PROMPT`, same
> `judge_one`, same `parse_verdict`) and adds the all-sites **consistency gate**
> plus the two acceptance views. The LJ reported in the main table is the
> **gated** view from `run_llm_judge_unified.py`. The standalone
> `llm_judge_variable_naming.py --all` path judges the per-benchmark CSVs without
> the consistency gate and is kept for the earlier Diffusion/FIM benchmarks.

---

## 1. Judge models and generation configuration

To keep LJ from depending on any single evaluator, the reported panel is **five
open-weights judges** spanning three families (Qwen, Mistral, Gemma) and a
7B–32B size range. The panel is fixed in `analysis/make_leaderboard.py`
(`JUDGES` / `JUDGE_SHORT`):

| Judge | HF model ID | Params | Table column | Role |
|---|---|---|---|---|
| Qwen2.5-7B-Instruct | `Qwen/Qwen2.5-7B-Instruct` | 7B | `LJ_Q7` | **primary** — the single LJ quoted in the main text |
| Qwen2.5-14B-Instruct | `Qwen/Qwen2.5-14B-Instruct` | 14B | `LJ_Q14` | size ladder |
| Qwen2.5-32B-Instruct | `Qwen/Qwen2.5-32B-Instruct` | 32B | `LJ_Q32` | size ladder |
| Mistral-Small-24B | `mistralai/Mistral-Small-24B-Instruct-2501` | 24B | `LJ_M24` | cross-family |
| Gemma-2-27B-It | `google/gemma-2-27b-it` | 27B | `LJ_G27` | cross-family |

The default/primary judge is set in code as
`DEFAULT_JUDGE = "Qwen2.5-7B-Instruct"`
(`experiments/llm_judge_variable_naming.py:91`). The full `JUDGE_REGISTRY`
(`llm_judge_variable_naming.py:76`) additionally exposes other judges
(Qwen2.5-3B, Mistral-7B, Gemma-2-9B, Llama-3.1-8B, Phi-4, Qwen3-32B) for
ablation; only the five above are reported.

**Loading** (`load_judge`, `llm_judge_variable_naming.py:212`):
each judge is loaded off the shelf via HuggingFace `transformers`
(`AutoModelForCausalLM.from_pretrained`) at **`torch.bfloat16`** precision, with
**no fine-tuning and no quantization**, `trust_remote_code=True`, and put in
`eval()` mode.

**Decoding** (`judge_one`, `llm_judge_variable_naming.py:287`):
deterministic — **greedy, `do_sample=False`** — with **`max_new_tokens=32`**
(`MAX_NEW_TOKENS`, line 71), which is enough for the one-line verdict the prompt
requests. The prompt is tokenised with **`truncation=True, max_length=4096`**
(line 312). A `temperature=1.0` argument is also passed but is **inert** under
`do_sample=False` (greedy decoding ignores it); output is fully deterministic.

**Hardware.** `device_map="auto"` is used on GPU (`load_judge`, line 219). The 7B
and 14B judges fit a single 80 GB card; the 24B/27B/32B judges (bf16 weights need
a ≥80 GB card) are sharded with `device_map="auto"` onto a 96 GB card. Registry
comments (lines 83–89) note the per-size memory footprints. Everything uses
open weights so the evaluation replicates without a paid API.

---

## 2. Prompt template

The judge receives a fixed two-part template: a `system` prompt setting the
reviewer role and acceptance rubric, and a `user` prompt carrying the code
context, the ground-truth name, and the model prediction. The **same** template
is applied to every prediction, every model, and every judge — unguided (no
in-prompt demonstration), and unchanged during the evaluation.

`SYSTEM_PROMPT` (`llm_judge_variable_naming.py:172`):

```
You are an expert Java code reviewer evaluating the quality of variable names suggested by an AI model.

Your task: given a code snippet and a ground-truth variable name, decide whether the predicted variable name is SEMANTICALLY ACCEPTABLE as a replacement.

Rules:
1. ACCEPTABLE if the prediction conveys the same concept as the ground truth, even if the exact string differs (e.g. 'bufSize' vs 'bufferSize' are both fine for a buffer-size variable).
2. NOT ACCEPTABLE if the prediction clearly describes a different concept.
3. Single-letter names are usually NOT ACCEPTABLE unless obviously correct (e.g. loop counter 'i', 'j').
4. Names that are clearly wrong tokens ('0', 'true', 'MASK', 'EOT', etc.) are NOT ACCEPTABLE.
5. Abbreviations that preserve the same meaning ARE ACCEPTABLE.

You MUST respond with EXACTLY one line, either:
    VERDICT: 1
or
    VERDICT: 0

Do NOT add any other text.
```

User prompt (`build_user_prompt`, `llm_judge_variable_naming.py:194`):

```
Code context (the predicted name replaces the masked identifier):

```java
{code_context}
```

Ground-truth variable name: `{ground_truth}`
Predicted variable name:    `{prediction}`

Is the predicted name semantically acceptable given the code context and the ground truth?
Reply with EXACTLY one line: VERDICT: 1   or   VERDICT: 0
```

**Per-judge chat template** (`_apply_chat_template`,
`llm_judge_variable_naming.py:229`): each judge's own chat template is applied
automatically with `add_generation_prompt=True`. Two compatibility handlers:
- `enable_thinking=False` forces Qwen3-style judges into non-thinking mode (so
  the 32-token budget is not consumed inside `<think>` and the `VERDICT` line
  appears); templates that don't know the flag ignore it.
- Templates that reject a `system` role (e.g. **Gemma-2**) are retried with the
  system prompt **merged into the user turn**; a plain-text format is the final
  fallback.

**Code context** (`extract_context`, `llm_judge_variable_naming.py:142`):
a window of up to **`MAX_CONTEXT_CHARS = 2000`** characters (line 68) centred on
the masked site — `half = max_chars // 2` characters on each side — with the
masked identifier replaced by the (consistency-agreed) prediction at that site;
`...` ellipses mark truncation. Any further mask tokens inside the window remain
as literal `[MASK]` context.

---

## 3. Verdict parsing

`parse_verdict` (`llm_judge_variable_naming.py:264`) maps the judge's reply to
`1`, `0`, or `-1`:

1. Primary: regex `VERDICT\s*:\s*([01])`, case-insensitive.
2. Fallback: a lone `1`/`0` on its own line, or `yes/acceptable/correct` → 1,
   `no/not acceptable/incorrect/wrong` → 0.
3. Otherwise `-1` (parse failure).

In scoring, **anything from which a `1` cannot be parsed is a rejection**: a
`-1` is folded to `0` (`score = max(verdict, 0)` in the standalone path,
`llm_judge_variable_naming.py:528`; `verdict == 1` counted otherwise).

---

## 4. Scoring logic (consistency gate + short-circuits + two views)

Implemented in `experiments/run_llm_judge_unified.py`. Each prediction first
passes the **all-sites consistency gate** — byte-identical to the metric
pipeline's `identifier_similarity_metrics.eval_sample`:

```python
# is_consistent (run_llm_judge_unified.py:83)
return len(preds) > 0 and preds[0] != "" and len(set(preds)) == 1
```

A sample is **usable** only if every masked site emitted the identical non-empty
identifier (otherwise the renamed program would not compile). Among usable
samples, three short-circuits avoid wasted judge calls
(`judge_model_csv`, lines 164–192):

| Condition | Verdict | LLM call? |
|---|---|---|
| Inconsistent / empty sites | `0` (`"inconsistent"`) | no |
| Agreed name == ground truth (exact match) | `1` (`"exact_match"`) | no |
| Agreed name has no alphabetic char | `0` (`"invalid_prediction"`) | no |
| Otherwise | judge's `VERDICT` | **yes**, in context |

The per-sample CSV (`FIELDS`, line 79) records
`consistent`, `agreed_pred`, `exact_match`, `llm_verdict`, `llm_raw_output`.
`summarize` (line 101) then produces **two acceptance views**:

- **`judge_acc_consistent`** — `verdict==1` over the consistent (compilable)
  samples only.
- **`judge_acc_gated`** — `verdict==1` over **all** samples, with inconsistent
  ones scored `0`.

It also reports `em_gated` (strict all-sites EM) and `consistency_rate`.
**The LJ column in the main table is the gated view**; `analysis/make_leaderboard.py`
(`lj_table`, line 56) reads the latest per-(model, judge) CSV and computes
`v1 / n` over all rows — i.e. the gated acceptance — into columns
`LJ_Q7 / LJ_Q14 / LJ_Q32 / LJ_M24 / LJ_G27`.

**LJ equation.** LJ is the mean of the per-sample gated outcomes over the
model's `N` predictions, as a percentage:

```
LJ = (100 / |D|) · Σ  1[ verdict(x̂) == 1 ]   %
```

For a model with full coverage (`N = 1000`), LJ = 20.0% corresponds to 200
accepting verdicts over the 1000 samples. (`make_leaderboard.py` flags any model
or judge whose coverage `n < 1000`.)

---

## 5. Human validation

To check that the verdicts track human judgment, a random sample of judged
predictions — covering both accepted (`VERDICT: 1`) and rejected (`VERDICT: 0`)
outcomes and spanning the judge families above — was manually inspected, and the
binary verdicts confirmed reasonable in context. The judged per-sample CSVs that
back this inspection are released under `results/unified_refineID/llm_judge/`
(one row per sample, with `agreed_pred`, `llm_verdict`, and `llm_raw_output`).

---

## 6. Known biases and limitations

- **Positional bias** does not apply: each judge is asked about a *single*
  prediction, never a pairwise comparison.
- **Self-enhancement bias** is mitigated by excluding every benchmarked model
  from the judge role — none of the five judges appears among the benchmarked
  models. Because the benchmark contains Qwen2.5-Coder models, two judges from
  different pretraining lineages (Mistral-Small-24B, Gemma-2-27B) are reported so
  no conclusion rests on a single family. In practice the judges agree closely
  (inter-judge agreement ≈ **0.924**; computed in
  `analysis/revision_major_stats.py`, the `M3 CROSS-JUDGE AGREEMENT` block).
  EM is reported alongside LJ throughout, as independent evidence.
- **What LJ measures.** LJ measures agreement with the developer's chosen name,
  not absolute naming quality: a prediction as good as the developer's but
  lexically different is often rejected. Read LJ as an approximation of how often
  the model recovers the developer's intent, not as an estimate of how often it
  produces a good name. A separate **judge ceiling** (feeding the ground-truth
  name back as the prediction) is computed in
  `experiments/judge_refineid_ground_truth.py` /
  `analysis/analyze_gt_judge_ceiling.py`.

---

## 7. Reproduce LJ end to end

All paths are repo-relative; `<...>` are placeholders. Outputs land under
`results/` so a single archive captures everything.

```bash
# 0) Per-model, per-site predictions (one CSV per model under predictions/)
python experiments/run_all_refineID_unified.py        # writes results/unified_refineID/predictions/<Model>.csv

# 1) Consistency-gated judging with the full five-judge panel
python experiments/run_llm_judge_unified.py \
    --judge-model Qwen2.5-7B-Instruct \
    --judge-model Qwen2.5-14B-Instruct \
    --judge-model Qwen2.5-32B-Instruct \
    --judge-model Mistral-Small-24B \
    --judge-model Gemma-2-27B-It
#   -> results/unified_refineID/llm_judge/<Model>__judge_<judge>__<ts>.csv   (per sample)
#   -> results/unified_refineID/llm_judge/judge_leaderboard_<judge>_<ts>.csv (per model)
#   -> ..._COMBINED_<ts>.csv  (+ a model x judge acceptance matrix, printed)

# 2) Assemble the Table-II-style leaderboard (EM/consistency/M1-3 + LJ_* columns)
python analysis/make_leaderboard.py
#   -> figures/new/leaderboard_full.csv  (gated LJ per judge)
```

Useful flags on `run_llm_judge_unified.py`:

| Flag | Effect |
|---|---|
| `--only <Model>` (repeatable) | judge a subset of models |
| `--judge-model <name|HF-id>` (repeatable) | one or many judges in one process (load → judge all → unload → next) |
| `--max-samples N` | quick smoke test |
| `--resume` | append to the latest output per (model, judge), skipping done ids |
| `--combine-only` | no judging — rebuild the COMBINED leaderboard + matrix from existing per-sample CSVs (use this to merge one-GPU-per-judge jobs) |
| `--list-models` | print the judge registry |

For one-GPU-per-judge parallelism, submit one job per `--judge-model` and merge
afterwards with `--combine-only`.

**Standalone judge** (no consistency gate; over the separate Diffusion/FIM
benchmark CSVs):

```bash
python experiments/llm_judge_variable_naming.py --all --judge-model Qwen2.5-7B-Instruct
python experiments/llm_judge_variable_naming.py --list-models
```

---

## 8. Inputs and key constants (quick reference)

| Item | Value | Source |
|---|---|---|
| Test data (code context) | `data/test.csv` (`id | masked_code | ground_truth`) | `DATA_PATH`, `llm_judge_variable_naming.py:60` |
| Unified predictions in | `results/unified_refineID/predictions/` | `run_llm_judge_unified.py:75` |
| Judge output out | `results/unified_refineID/llm_judge/` | `run_llm_judge_unified.py:76` |
| Context window | 2000 chars, centred on mask | `MAX_CONTEXT_CHARS`, line 68 |
| Max new tokens | 32 | `MAX_NEW_TOKENS`, line 71 |
| Prompt truncation | 4096 tokens | `judge_one`, line 312 |
| Precision | bfloat16 (GPU) | `load_judge`, line 218 |
| Decoding | greedy, `do_sample=False` | `judge_one`, line 319 |
| Reported view | gated acceptance | `summarize`, line 129; `make_leaderboard.lj_table` |

> Line numbers are accurate as of this artifact; if the files are edited, locate
> by the function/symbol names instead.
