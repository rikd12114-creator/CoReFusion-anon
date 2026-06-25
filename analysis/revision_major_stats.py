"""Stats for supervisor revision:
  M2 = bootstrap 95% CIs on EM (Table II headline + Table III/Fig5 |S| buckets)
  M3 = cross-judge agreement (paper judge Qwen2.5-7B-Instruct vs other families)

Canonical EM source = the same per-position benchmark CSVs that
analysis/plot_em_stratified.py uses (one row per masked position; a sample is
EM-correct iff every position is correct). Table II EM uses denominator 1000
(samples whose mask_count is unparseable count as wrong); Table III buckets use
the parseable subset, giving |S|=1 = 23 samples, exactly as in the paper.

Run: python3 analysis/revision_major_stats.py
"""
import glob
import os
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, kendalltau

np.random.seed(42)
ROOT = "/path/to/CoReFusion"
DIFF_DIR = f"{ROOT}/data/benchmark_ReFineID_Diffusion/diffusion_benchmark"
FIM_DIR = f"{ROOT}/data/benchmark_ReFineID_FIM/ar_fim_benchmark"
JUDGE = f"{ROOT}/results/unified_refineID/llm_judge"
TOTAL = 1000  # RefineID test split size (denominator for Table II EM)


def boot_ci(x, n_boot=20000, alpha=0.05):
    x = np.asarray(x, float)
    n = len(x)
    if n == 0:
        return np.nan, np.nan, np.nan, 0
    means = x[np.random.randint(0, n, size=(n_boot, n))].mean(axis=1)
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return x.mean(), lo, hi, n


def boot_diff_ci(x, y, n_boot=20000, alpha=0.05):
    """95% CI of mean(x)-mean(y) by independent resampling; also P(x>y)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    bx = x[np.random.randint(0, len(x), size=(n_boot, len(x)))].mean(1)
    by = y[np.random.randint(0, len(y), size=(n_boot, len(y)))].mean(1)
    d = bx - by
    lo, hi = np.percentile(d, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return d.mean(), lo, hi, (d > 0).mean()


def load_sample_em(path):
    """-> DataFrame indexed by id with columns em(bool), mc(|S|); plus n_total.

    EM is TRUE all-sites: a sample is correct iff EVERY '|'-split site in
    all_predictions equals ground_truth (empty/NaN output counts as wrong).
    NOTE: the precomputed `correct` column is target-position only (first site),
    so it must NOT be used for all-sites EM -- see all_predictions instead.
    """
    df = pd.read_csv(path)
    n_total = df["id"].nunique()
    gt = df["ground_truth"].fillna("").astype(str).tolist()
    ap = df["all_predictions"].fillna("").astype(str).tolist()
    df["em"] = [bool(a) and all(p == g for p in a.split("|")) for a, g in zip(ap, gt)]  # all-sites
    df["mc"] = pd.to_numeric(df["mask_count"], errors="coerce")
    df = df.dropna(subset=["mc"])
    g = df.groupby("id").agg(em=("em", "first"), mc=("mc", "first"))
    return g, n_total


def latest(d, pat):
    fs = sorted(glob.glob(os.path.join(d, pat)))
    return fs[-1] if fs else None


def em_full_to_1000(g, n_total):
    """EM array padded so unparseable samples count as wrong (len = n_total)."""
    arr = g["em"].astype(int).values
    pad = max(0, n_total - len(arr))
    return np.concatenate([arr, np.zeros(pad, int)])


def bucket(s):
    if s == 1: return "=1"
    if s == 2: return "=2"
    if s <= 5: return "3-5"
    if s <= 10: return "6-10"
    return ">=11"


MODELS = {  # display -> (dir, glob)
    "DreamCoder-7B": (DIFF_DIR, "DreamCoder-7B_refineID_diffusion_*.csv"),
    "DiffuCoder-7B": (DIFF_DIR, "DiffuCoder-7B_refineID_diffusion_*.csv"),
    "CodeLlama-13B": (FIM_DIR, "CodeLlama-13B_refineID_fim_*.csv"),
    "CodeLlama-7B": (FIM_DIR, "CodeLlama-7B_refineID_fim_*.csv"),
    "StarCoder2-15B": (FIM_DIR, "StarCoder2-15B_refineID_fim_*.csv"),
    "Qwen2.5-Coder-7B": (FIM_DIR, "Qwen2.5-Coder-7B_refineID_fim_*.csv"),
}
order = ["=1", "=2", "3-5", "6-10", ">=11"]

print("=" * 74)
print("M2  OVERALL EM + 95% bootstrap CI  (denominator 1000 -> matches Table II)")
print("=" * 74)
loaded = {}
for m, (d, pat) in MODELS.items():
    f = latest(d, pat)
    if not f:
        print(f"  MISSING {m}"); continue
    g, nt = load_sample_em(f)
    loaded[m] = (g, nt)
    emarr = em_full_to_1000(g, TOTAL)
    mean, lo, hi, n = boot_ci(emarr)
    print(f"  {m:18s} EM={100*mean:5.1f}  95% CI [{100*lo:4.1f}, {100*hi:4.1f}]"
          f"  (+-{100*(hi-lo)/2:.1f})  parseable={len(g)}/{nt}")

print()
print("=" * 74)
print("M2  EM by |S| bucket + 95% CI   (Table III / Fig 5)")
print("=" * 74)
for m, (g, nt) in loaded.items():
    g = g.copy()
    g["bk"] = g["mc"].apply(bucket)
    cells = []
    for bk in order:
        sub = g[g.bk == bk]["em"].astype(int).values
        mean, lo, hi, n = boot_ci(sub)
        cells.append(f"|S|{bk:>4}: {100*mean:5.1f} [{100*lo:5.1f},{100*hi:5.1f}] n={n}")
    print(f"  {m}\n    " + "\n    ".join(cells))

g0 = list(loaded.values())[0][0].copy()
g0["bk"] = g0["mc"].apply(bucket)
print("\n  BUCKET SIZES:", g0["bk"].value_counts().reindex(order).to_dict(),
      "TOTAL parseable", len(g0))

print()
print("=" * 74)
print("M2  |S|=1 head-to-head: is the FIM-AR 'win' inside noise? (n=23)")
print("=" * 74)
b1 = {}
for m, (g, nt) in loaded.items():
    b1[m] = g[g["mc"] == 1]["em"].astype(int).values
for fim in ["CodeLlama-13B", "CodeLlama-7B", "StarCoder2-15B", "Qwen2.5-Coder-7B"]:
    for dl in ["DreamCoder-7B", "DiffuCoder-7B"]:
        d, lo, hi, p = boot_diff_ci(b1[fim], b1[dl])
        print(f"  {fim:16s} - {dl:14s}: dEM={100*d:+5.1f}pp  95% CI [{100*lo:+5.1f},{100*hi:+5.1f}]"
              f"  P({fim[:8]}>{dl[:8]})={p:.2f}")

# ============================ M3 ============================
print()
print("=" * 74)
print("M3  CROSS-JUDGE AGREEMENT  (paper judge Qwen2.5-7B-Instruct vs others)")
print("=" * 74)
JUDGES = ["Qwen2.5-7B-Instruct", "Qwen2.5-14B-Instruct", "Qwen2.5-32B-Instruct",
          "Gemma-2-27B-It", "Mistral-Small-24B"]
PAPER = "Qwen2.5-7B-Instruct"
NONQWEN = {"Gemma-2-27B-It", "Mistral-Small-24B"}
paper_models = ["DreamCoder-7B", "DiffuCoder-7B", "DreamOn-7B", "CodeLlama-13B", "CodeLlama-7B",
                "StarCoder2-15B", "StarCoder2-7B", "StarCoder2-3B", "DeepSeek-Coder-6.7B",
                "DeepSeek-Coder-1.3B", "Qwen2.5-Coder-14B", "Qwen2.5-Coder-7B", "Qwen2.5-Coder-3B",
                "Qwen2.5-Coder-1.5B", "CodeGemma-7B", "CodeGemma-2B", "CodeT5p-16B", "CodeT5p-6B",
                "CodeT5p-2B", "CodeT5-large", "CodeT5-base", "CodeT5-small"]


def jlatest(model, judge):
    fs = sorted(glob.glob(f"{JUDGE}/{model}__judge_{judge}__*.csv"))
    return fs[-1] if fs else None


def jverdicts(model, judge):
    f = jlatest(model, judge)
    if not f:
        return None
    d = pd.read_csv(f)
    return d.set_index("id")[["llm_verdict", "consistent"]]


def kappa(a, b):
    a, b = np.asarray(a, int), np.asarray(b, int)
    po = (a == b).mean()
    pa, pb = a.mean(), b.mean()
    pe = pa * pb + (1 - pa) * (1 - pb)
    return ((po - pe) / (1 - pe) if (1 - pe) > 0 else np.nan), po


print("\n  -- per-sample verdict agreement vs paper judge (pooled over models) --")
for other in [j for j in JUDGES if j != PAPER]:
    A, B, Ac, Bc, nmods = [], [], [], [], 0
    for m in paper_models:
        va, vb = jverdicts(m, PAPER), jverdicts(m, other)
        if va is None or vb is None:
            continue
        nmods += 1
        j = va.join(vb, lsuffix="_p", rsuffix="_o", how="inner").dropna()
        A += j.llm_verdict_p.astype(int).tolist(); B += j.llm_verdict_o.astype(int).tolist()
        jc = j[j.consistent_p == 1]
        Ac += jc.llm_verdict_p.astype(int).tolist(); Bc += jc.llm_verdict_o.astype(int).tolist()
    if A:
        k, po = kappa(A, B); kc, poc = kappa(Ac, Bc)
        fam = "NON-QWEN" if other in NONQWEN else "qwen-fam"
        print(f"  [{fam}] vs {other:22s}: ALL agree={100*po:5.1f}% k={k:.3f} (N={len(A)},{nmods}m)"
              f" | consistent-only agree={100*poc:5.1f}% k={kc:.3f} (N={len(Ac)})")

print("\n  -- dLLMs only (DreamCoder+DiffuCoder) --")
for other in [j for j in JUDGES if j != PAPER]:
    A, B = [], []
    for m in ["DreamCoder-7B", "DiffuCoder-7B"]:
        va, vb = jverdicts(m, PAPER), jverdicts(m, other)
        if va is None or vb is None:
            continue
        j = va.join(vb, lsuffix="_p", rsuffix="_o", how="inner").dropna()
        A += j.llm_verdict_p.astype(int).tolist(); B += j.llm_verdict_o.astype(int).tolist()
    if A:
        k, po = kappa(A, B)
        print(f"  vs {other:22s}: agree={100*po:5.1f}% kappa={k:.3f} (N={len(A)})")

print("\n  -- model RANKING stability across judges (vs paper judge) --")
acc = {}
for judge in JUDGES:
    acc[judge] = {m: v.llm_verdict.mean() for m in paper_models
                  if (v := jverdicts(m, judge)) is not None}
common = [m for m in paper_models if all(m in acc[j] for j in JUDGES)]
print(f"  (n_models common to all 5 judges = {len(common)})")
base = [acc[PAPER][m] for m in common]
for other in [j for j in JUDGES if j != PAPER]:
    o = [acc[other][m] for m in common]
    print(f"  vs {other:22s}: spearman={spearmanr(base,o)[0]:.3f}  kendall={kendalltau(base,o)[0]:.3f}")

print("\n  -- sanity: dLLM mean verdict under paper judge (Table II LJ = 66.2/64.1) --")
for m in ["DreamCoder-7B", "DiffuCoder-7B"]:
    v = jverdicts(m, PAPER)
    print(f"    {m}: all={100*v.llm_verdict.mean():.1f}%  "
          f"consistent-only={100*v[v.consistent==1].llm_verdict.mean():.1f}%")
