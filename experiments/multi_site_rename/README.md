# Multi-Site / Multi-Identifier Rename Benchmark

A self-contained benchmark that measures how well **small open-source code LLMs**
(DeepSeek-Coder, Qwen2.5-Coder) can perform a **multiple-identifier, multiple-site
rename refactoring** on real Java code drawn from well-known open-source repos.

This complements the existing RefineID benchmark in the repo, which masks a
**single** identifier per file (`data/test.csv`). Here we mask **several distinct
identifiers per file at once**, each replaced consistently at *all* of its usage
sites, and ask the model to recover every original name. This is the realistic
"rename a handful of badly-named variables in this method" scenario.

```
masked source (placeholders IDENT_0..IDENT_k at every site)
        │
        ▼
   small code LLM  ──►  {"IDENT_0": "name0", "IDENT_1": "name1", ...}
        │
        ▼
   metrics: per-identifier EM · joint all-correct · edit-sim · site-consistency
```

## Why this is harder than single-site filling

* The model must keep names **mutually consistent** across many holes.
* Recovering `IDENT_2` may depend on first inferring what `IDENT_0`/`IDENT_1` are
  (coupled identifiers, e.g. `left`/`right`, `nameStart`/`valueStart`).
* No position hint is given for *which* token to focus on — every site is masked.

## Layout

| file | purpose |
|------|---------|
| `repos.json`        | curated list of famous OSS Java files (pinned to release tags) |
| `build_dataset.py`  | fetch Java files → `javalang` AST → mask K identifiers at all sites → `data/multisite_rename.jsonl` |
| `models.py`         | model registry + prompt builders + output parsing (instruct & FIM) |
| `metrics.py`        | exact-match, joint accuracy, edit similarity, sub-token F1, consistency |
| `run_benchmark.py`  | load each model, run the rename task, write per-sample + summary results |
| `colab.ipynb`       | one-click Colab driver (GPU runtime) |
| `data/sample_java/` | tiny offline-fallback Java file so the pipeline runs with no network |

## Quick start (Colab)

Open `colab.ipynb`, set runtime to **GPU** (T4 is enough for the 1.3–1.5B models;
use A100 for 6.7–7B), then run all cells. Or from a terminal:

```bash
pip install -r experiments/multi_site_rename/requirements.txt

# 1) Build the dataset from famous repos (skips any file that fails to fetch/parse)
python experiments/multi_site_rename/build_dataset.py \
    --repos experiments/multi_site_rename/repos.json \
    --out   experiments/multi_site_rename/data/multisite_rename.jsonl \
    --scope method --num-identifiers 3 --min-sites 2 --instances-per-file 6

# 2) Run the benchmark for the small models
python experiments/multi_site_rename/run_benchmark.py \
    --data   experiments/multi_site_rename/data/multisite_rename.jsonl \
    --models deepseek-1.3b-instruct qwen2.5-1.5b-instruct \
    --out-dir experiments/multi_site_rename/results \
    --limit 100
```

A pre-built dataset (`data/multisite_rename.jsonl`, 90 instances from 17 OSS
files) is already checked in, so you can skip step 1 and run the benchmark
directly. Re-run step 1 to refresh or change the masking.

### Dataset construction (`build_dataset.py`)

* `--scope method` (default) carves **one instance per method** — short, coherent
  "rename the locals in this method" snippets. `--scope file` masks across the
  whole file instead.
* Identifiers are chosen by `javalang` AST: only **local variables / parameters**
  (never class/method/field names), each with `>= --min-sites` occurrences and
  `>= --min-len` characters, top-`--num-identifiers` by occurrence count.
* `--strip-comments` blanks out comments first, so prose can't restate (leak) a
  masked name — a stricter setting for clean accuracy numbers
  (`data/multisite_rename_nocomments.jsonl` is the pre-built strict variant).
* Files that 404 or fail to parse (e.g. Java records / text blocks `javalang`
  can't handle) are skipped; the bundled `data/sample_java/` file is always
  available so the pipeline runs with no network (`--no-fetch`).

## Models

`--models` accepts the keys defined in `models.py`. Defaults are Colab-friendly:

| key | HF id | params | protocol |
|-----|-------|--------|----------|
| `deepseek-1.3b-instruct` | `deepseek-ai/deepseek-coder-1.3b-instruct` | 1.3B | json |
| `deepseek-6.7b-instruct` | `deepseek-ai/deepseek-coder-6.7b-instruct` | 6.7B | json |
| `qwen2.5-0.5b-instruct`  | `Qwen/Qwen2.5-Coder-0.5B-Instruct`        | 0.5B | json |
| `qwen2.5-1.5b-instruct`  | `Qwen/Qwen2.5-Coder-1.5B-Instruct`        | 1.5B | json |
| `qwen2.5-3b-instruct`    | `Qwen/Qwen2.5-Coder-3B-Instruct`          | 3B   | json |
| `qwen2.5-7b-instruct`    | `Qwen/Qwen2.5-Coder-7B-Instruct`          | 7B   | json |
| `deepseek-1.3b-base`     | `deepseek-ai/deepseek-coder-1.3b-base`    | 1.3B | fim  |
| `qwen2.5-1.5b-base`      | `Qwen/Qwen2.5-Coder-1.5B`                  | 1.5B | fim  |

Add your own by editing the `MODELS` dict in `models.py`.

## Protocols

* **`json` (default, instruct models)** — the masked file is shown in full and the
  model returns a JSON map `placeholder -> identifier`. Site-consistency is exact
  by construction (one answer per identifier).
* **`fim` (base models)** — each identifier's first masked site is filled via the
  model's native FIM format; the rest of the file (including the other
  placeholders) is left as context. Consistency across an identifier's sites is
  measured, not enforced.

## Metrics (written to `results/<model>_summary.json`)

* `em` — mean per-identifier exact match (case-sensitive).
* `em_ci` — case-insensitive variant.
* `joint_acc` — fraction of files where **every** masked identifier is recovered exactly.
* `edit_sim` — mean normalized edit similarity (partial credit).
* `subtoken_f1` — mean F1 over camelCase/snake_case sub-tokens.
* `coverage` — fraction of placeholders the model produced any prediction for.
