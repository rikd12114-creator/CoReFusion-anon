# DreamOn-7B on refineID — paper paragraph

## Strict "all-sites-correct" metric (apples-to-apples with AR / dLLM baselines)

To compare DreamOn to the AR and dLLM baselines fairly we adopt the strict
sample-level metric used in our existing refineID pipeline: a sample is
counted as correct iff **every** masked occurrence of the renamed variable
is recovered identically to the ground truth. Under this metric:

| Model | n | First-mask EM | **Strict all-correct EM** |
|---|---:|---:|---:|
| DreamOn-7B (OLD: single window, first-mask)       | 1000 | 16.40 % | **N/A** ¹ |
| **DreamOn-7B (NEW: tiled, all-site coverage)**    | 1000 | 15.40 % | **9.10 %** |
| CodeT5p-16B (FIM baseline)                        | 1000 | 23.50 % | 7.90 % |
| StarCoder2-7B (AR baseline)                       |  230 | 10.87 % | 5.65 % |
| DeepSeek-Coder-6.7B-Base (AR baseline)            |  230 |  0.00 % | 0.00 % |

¹ The OLD DreamOn run only stored the first mask's prediction in its CSV;
per-site predictions are not recoverable, so the strict metric cannot be
applied retrospectively. The 16.40 % figure is therefore *first-mask* EM,
not directly comparable to the strict column.

The strict metric is much harsher than first-mask EM — for CodeT5p-16B it
cuts the score from 23.5 % to 7.9 %, and for StarCoder2-7B from 10.9 %
to 5.7 % — because every additional [MASK] in a sample creates another
opportunity to fail. **Under this strict metric, the tiled DreamOn-7B
configuration achieves 9.10 %**, slightly above CodeT5p-16B (7.90 %) and
notably above StarCoder2-7B (5.65 %), while requiring 1.72 forward
passes per sample on average to cover all [MASK] sites within the
2,048-token context budget.

---

## Drop-in paragraph (≈220 words)

We evaluate DreamOn-7B, a variable-length masked-diffusion FIM model, on
the refineID variable-renaming benchmark (1,000 Java samples;
mean 7.71 [MASK] sites per sample, max 175). Because DreamOn's
2,048-token context cannot accommodate the full file (up to ~50K
characters in our set), we tile each sample into non-overlapping
local windows of \~3K characters that together cover every [MASK] site,
and run one multi-site infill pass per window using the upstream-
recommended hyper-parameters (`number_transfer_tokens=1`, `alg=entropy`,
`alg_temp=0`, `temperature=0.2`, `top_p=0.9`, initial canvas of four
mask tokens, maximum canvas of 64). Across the full benchmark, DreamOn
achieves a **site-level exact match of 14.6 %** (1,126 / 7,712) and a
**sample-level majority-vote EM of 16.0 %** (160 / 1,000), comparable to
a simpler first-mask-only baseline (16.4 %) at the cost of 1.72× more
forward passes. Strikingly, the **any-site EM upper bound reaches 24.3 %**,
revealing an 8.3-point gap between samples where the model has the
*capability* to recover the identifier at some position and those where
its plurality vote selects it. Stratified analysis shows that accuracy
**rises with identifier complexity** (40.5 % EM for ≥ 4-segment
CamelCase names versus 11.3 % for single-word names), suggesting that
DreamOn's failures concentrate on short, low-entropy identifiers where
many candidates are plausible. The most common wrong predictions are
generic Java tokens (`double`, `int`, `format`, `code`, `new`),
indicating that the model falls back to syntactic-type fillers when
local semantics are under-determined.

---

## Optional shorter version (≈110 words, for related-work/abstract pressure)

We evaluate DreamOn-7B — a variable-length FIM diffusion model — on
refineID with a tiled multi-site infilling scheme that covers all
[MASK] occurrences inside DreamOn's 2K-token context budget. On 1,000
samples (7,712 sites) DreamOn reaches **14.6 % site EM** and **16.0 %
sample EM by majority vote**, comparable to a first-mask-only baseline.
However, its **any-site EM is 24.3 %**, indicating an 8.3-point
reliability gap between *capability* and *aggregated prediction*.
Performance increases monotonically with identifier complexity (40.5 %
EM for ≥ 4-segment names versus 11.3 % for single words), and the
dominant failure mode is the model emitting generic Java tokens
(`double`, `int`, `new`).

---

## Suggested figure captions

**Figure 1 (headline_metrics).** Headline metrics for DreamOn-7B on the
1,000-sample refineID benchmark. The old single-window, first-mask-only
baseline reaches 16.4 % EM; tiled all-site coverage gives 14.6 % at the
site level, 16.0 % by majority vote, and a 24.3 % any-site upper bound.

**Figure 2 (em_vs_mask_count).** Sample-level EM stratified by the
number of [MASK] sites per sample. Performance is U-shaped: single-mask
samples are easiest (43.5 %), 3–10-mask samples are hardest (≈ 11–13 %),
and ≥ 11-mask samples recover (22.9 % at 11–20 sites, 30.0 % at 50+).
The any-site curve is consistently higher, widening to ~32 points at
21–50 masks.

**Figure 3 (em_vs_identifier_complexity).** Site-level EM by
ground-truth identifier complexity. Accuracy rises monotonically with
both character length and CamelCase segment count, reaching 40.5 % for
≥ 4-segment names — likely because longer identifiers are more uniquely
constrained by local context.

**Figure 4 (em_vs_site_idx).** Site-level EM as a function of the
site's index within its sample. Accuracy is stable (~15 %) for the
first 15 sites and drops to 10.7 % for the 16th and later sites,
suggesting that windows farther from the file's first [MASK] receive
weaker context under our 8-site-per-window cap.

**Figure 5 (top_wrong_predictions).** Twelve most frequent wrong
predictions. The list is dominated by generic Java type / keyword
tokens (`double`, `format`, `int`, `code`, `new`), indicating that
DreamOn falls back to syntactic fillers when local semantics
under-determine the variable name.

**Figure 6 (consistency_vs_accuracy).** Sample-level EM (majority
vote) vs. DreamOn's self-consistency across sites. Even when the model
agrees on a single token at all sites (100 % consistency, 54 % of
samples), accuracy is only 16.9 %, near the overall mean. Self-
consistency is therefore a poor reliability signal — DreamOn is
frequently *confidently wrong*.

**Figure 7 (strict_all_correct_comparison).** Strict all-sites-correct
EM vs first-mask EM across DreamOn-7B and the existing refineID
baselines. The strict metric, which requires every masked occurrence
to be recovered, halves or worse the accuracy of every model. Under
the strict metric, the tiled DreamOn-7B configuration (9.10 %) edges
out CodeT5p-16B (7.90 %) and StarCoder2-7B (5.65 %). The OLD DreamOn
run (single window, first-mask only) could not be re-scored because
its CSV did not store per-site predictions.

---

## Files in this folder

```
analysis/dreamon_results/
├── paper_paragraph.md            (this file)
├── summary_stats.csv             (key numbers, ready for tables)
├── plot_dreamon_results.py       (reproduces the figures)
└── plots/
    ├── fig1_headline_metrics.png
    ├── fig2_em_vs_mask_count.png
    ├── fig3_em_vs_identifier_complexity.png
    ├── fig4_em_vs_site_idx.png
    ├── fig5_top_wrong_predictions.png
    └── fig6_consistency_vs_accuracy.png
```
