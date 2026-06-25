# CoReFusion

Anonymized code artifact accompanying the paper
**"CoReFusion: Diffusion-LLM-based Code Refactoring Localization."**

This repository is released for **double-blind review**. All author, institution,
and account identifiers have been removed. Paths to private clusters, home
directories, and personal Hugging Face repositories have been replaced with
neutral placeholders (e.g. `/path/to/CoReFusion`, `/home/user`, `anonymous/...`).

## What this is

The project studies whether diffusion language models (dLLMs) can *localize*
code-refactoring edits — specifically variable-naming smells in Java — and
compares them against autoregressive code models using exact-match, consistency,
a Context-Insensitivity Score (CIS), and an LLM-as-Judge panel.

## Repository layout

```
analysis/      Metric computation, evaluation, and plotting
               (identifier-similarity metrics, metric fusion / CIS,
                RQ2 deobfuscation reproduction, heatmaps, leaderboards)
detector/      Naming-smell detectors and the LLM-as-Judge implementation
               (ml_layer/ holds the ML-based detector)
experiments/   Benchmark drivers and probes
               1t5t_exp/                  1-token vs 5-token study
               multi_site_rename/         multi-site rename benchmark
               rq3_probe/                 detect-before-commit / RQ3 probes
               smell_as_noise_refactoring/
               tests/                     unit-style checks
models/        Thin loader wrappers per model family
               (qwen, llama, deepseek, mistral, llada, dreamcoder, diffucoder)
notebooks/     Colab notebooks for running benchmarks
docs/          Design notes, runbooks, and paper-section drafts (.md / .tex)
run_inference.py, unified_framework.py, analyze_*.py, generate_*_plots.py
               Top-level entry points
```

## Setup

```bash
pip install -r requirements.txt
```

Install a PyTorch build matching your CUDA version separately
(https://pytorch.org/get-started/locally/); it is intentionally not pinned here.

## Notes for reviewers

- **Data, model weights, generated results, and figures are not included** in
  this artifact (they are large and/or generated). The code that produces them
  is provided in full.
- HPC/SLURM launch scripts were intentionally excluded; the Python code they
  invoke is included and can be run directly.
- A few Hugging Face dataset/model identifiers appear as `anonymous/...`
  placeholders where they previously pointed to author-owned repositories.

## License

Released under the MIT License (see [LICENSE](LICENSE)).
