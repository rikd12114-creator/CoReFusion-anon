"""
Build the canonical per-sample identifier-evaluation dataset for the
metric-reasonableness study and the metric-fusion model.

For every LLM-as-Judge CSV in results/unified_refineID/llm_judge/ we have, per
identifier group (sample):
    id, ground_truth, n_sites, consistent, agreed_pred, exact_match, llm_verdict
We RECOMPUTE the new identifier metrics (M1 IdBench lev/nw, M2 subtoken
jaccard/fuzzy, M3 quality char/word) on the SAME (agreed_pred, ground_truth)
pair the judge scored, so every feature, EM, and LJ refer to the identical
prediction -> a fully self-consistent supervised dataset:

    features : em, lev_sim, nw_sim, subtok_jaccard, subtok_fuzzy,
               qual_char, qual_word  (+ gt_qual_char, len_gt, len_pred)
    label    : lj = llm_verdict (per-judge) ; lj_consensus = mean over judges

Outputs
-------
results/identifier_metrics/fusion_long.csv   one row per (model, judge, sample)
results/identifier_metrics/fusion_consensus.csv  one row per (model, sample),
        metrics + EM + per-judge LJ columns + lj_mean/lj_majority over judges
"""
import os, re, csv, sys, glob
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
import identifier_similarity_metrics as M

JUDGE_DIR = "results/unified_refineID/llm_judge"
OUT_DIR = "results/identifier_metrics"
os.makedirs(OUT_DIR, exist_ok=True)

# model -> (family, arch); family from lj_vs_em_table, arch from analysis_data rq1
FAMILY = {
    "DiffuCoder-7B": ("dLLM", "dLLM (fixed-canvas)"),
    "DreamCoder-7B": ("dLLM", "dLLM (fixed-canvas)"),
    "DreamOn-7B": ("dLLM", "dLLM (variable canvas)"),
    "DiffusionGemma-26B-A4B": ("dLLM", "dLLM (block-AR)"),
    "CodeLlama-7B": ("FIM", "Decoder-only (FIM)"),
    "CodeLlama-13B": ("FIM", "Decoder-only (FIM)"),
    "DeepSeek-Coder-1.3B": ("FIM", "Decoder-only (FIM)"),
    "DeepSeek-Coder-6.7B": ("FIM", "Decoder-only (FIM)"),
    "Qwen2.5-Coder-1.5B": ("FIM", "Decoder-only (FIM)"),
    "Qwen2.5-Coder-3B": ("FIM", "Decoder-only (FIM)"),
    "Qwen2.5-Coder-7B": ("FIM", "Decoder-only (FIM)"),
    "Qwen2.5-Coder-14B": ("FIM", "Decoder-only (FIM)"),
    "StarCoder2-3B": ("FIM", "Decoder-only (FIM)"),
    "StarCoder2-7B": ("FIM", "Decoder-only (FIM)"),
    "StarCoder2-15B": ("FIM", "Decoder-only (FIM)"),
    "CodeGemma-2B": ("FIM", "Decoder-only (FIM)"),
    "CodeGemma-7B": ("FIM", "Decoder-only (FIM)"),
    "CodeT5p-6B": ("FIM", "Decoder-only (FIM)"),
    "CodeT5-small": ("Seq2Seq", "Encoder-decoder"),
    "CodeT5-base": ("Seq2Seq", "Encoder-decoder"),
    "CodeT5-large": ("Seq2Seq", "Encoder-decoder"),
    "CodeT5p-2B": ("Seq2Seq", "Encoder-decoder"),
    "CodeT5p-16B": ("Seq2Seq", "Encoder-decoder"),
}

FNAME_RE = re.compile(r"^(?P<model>.+?)__judge_(?P<judge>.+?)__(?P<ts>\d{8}_\d{6})\.csv$")
METRIC_KEYS = M.METRIC_KEYS  # lev_sim, nw_sim, subtok_jaccard, subtok_fuzzy, qual_char, qual_word


def pick_files():
    """One file per (model, judge): max rows, then latest timestamp."""
    best = {}
    for path in glob.glob(os.path.join(JUDGE_DIR, "*.csv")):
        fn = os.path.basename(path)
        if fn.startswith("judge_leaderboard"):
            continue
        m = FNAME_RE.match(fn)
        if not m:
            continue
        model, judge, ts = m["model"], m["judge"], m["ts"]
        with open(path, encoding="utf-8") as f:
            n = sum(1 for _ in f) - 1
        key = (model, judge)
        cur = best.get(key)
        if cur is None or (n, ts) > (cur[1], cur[2]):
            best[key] = (path, n, ts)
    return best


def main():
    dictionary = M.load_dictionary()
    print(f"dictionary: {len(dictionary)} words + {len(M.ABBREVIATIONS)} abbrevs")
    best = pick_files()
    judges = sorted({j for (_, j) in best})
    models = sorted({m for (m, _) in best})
    print(f"{len(best)} (model,judge) files | {len(models)} models | judges: {judges}")

    metric_cache = {}  # (pred, gt) -> dict of metrics+em
    def metrics_for(pred, gt):
        k = (pred, gt)
        if k not in metric_cache:
            d = {}
            d.update(M.metric_idbench(pred, gt))
            d.update(M.metric_subtoken(pred, gt))
            d.update(M.metric_quality(pred, dictionary))
            d["em_recomp"] = float(pred == gt)
            metric_cache[k] = d
        return metric_cache[k]

    long_rows = []
    mismatch = 0
    val = defaultdict(lambda: {"n": 0, "em": 0, "lj": 0})  # (model,judge) validation

    for (model, judge), (path, n, ts) in sorted(best.items()):
        fam, arch = FAMILY.get(model, ("?", "?"))
        with open(path, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                gt = (r.get("ground_truth") or "").strip()
                pred = (r.get("agreed_pred") or "").strip()
                cons = str(r.get("consistent", "")).strip() in ("1", "1.0", "True", "true")
                em_stored = str(r.get("exact_match", "")).strip().lower() in ("1", "true", "1.0")
                try:
                    lj = int(float(r.get("llm_verdict", "")))
                except (ValueError, TypeError):
                    lj = None
                try:
                    n_sites = int(float(r.get("n_sites", "0") or 0))
                except (ValueError, TypeError):
                    n_sites = 0

                row = {
                    "model": model, "family": fam, "arch": arch, "judge": judge,
                    "id": r.get("id", ""), "ground_truth": gt, "agreed_pred": pred,
                    "n_sites": n_sites, "consistent": int(cons),
                    "lj": lj,
                }
                if cons and pred:
                    mm = metrics_for(pred, gt)
                    for k in METRIC_KEYS:
                        row[k] = mm[k]
                    row["em"] = mm["em_recomp"]
                    row["gt_qual_char"] = M.metric_quality(gt, dictionary)["qual_char"]
                    row["len_gt"] = len(gt)
                    row["len_pred"] = len(pred)
                    if int(mm["em_recomp"]) != int(em_stored):
                        mismatch += 1
                else:
                    for k in METRIC_KEYS:
                        row[k] = 0.0
                    row["em"] = 0.0
                    row["gt_qual_char"] = M.metric_quality(gt, dictionary)["qual_char"] if gt else 0.0
                    row["len_gt"] = len(gt)
                    row["len_pred"] = len(pred)
                long_rows.append(row)

                v = val[(model, judge)]
                v["n"] += 1
                v["em"] += int(em_stored)
                if lj is not None:
                    v["lj"] += lj

    # ---- write long ----
    cols = ["model", "family", "arch", "judge", "id", "ground_truth", "agreed_pred",
            "n_sites", "consistent", "em", "lj"] + METRIC_KEYS + ["gt_qual_char", "len_gt", "len_pred"]
    long_path = os.path.join(OUT_DIR, "fusion_long.csv")
    with open(long_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(long_rows)
    print(f"\nwrote {len(long_rows)} rows -> {long_path}")
    print(f"EM recompute vs stored exact_match mismatches: {mismatch}")

    # ---- consensus (one row per model,sample; LJ across judges) ----
    by_ms = defaultdict(dict)  # (model,id) -> {judge: lj, ...feature row}
    feat = {}
    for r in long_rows:
        key = (r["model"], r["id"])
        if key not in feat:
            feat[key] = {k: r[k] for k in ["model", "family", "arch", "id", "ground_truth",
                                           "agreed_pred", "n_sites", "consistent", "em",
                                           "gt_qual_char", "len_gt", "len_pred"] + METRIC_KEYS}
        if r["lj"] is not None:
            by_ms[key][r["judge"]] = r["lj"]

    cons_rows = []
    for key, fr in feat.items():
        votes = by_ms.get(key, {})
        row = dict(fr)
        for j in judges:
            row[f"lj_{j}"] = votes.get(j, "")
        vlist = [v for v in votes.values()]
        row["lj_n_judges"] = len(vlist)
        row["lj_mean"] = sum(vlist) / len(vlist) if vlist else ""
        row["lj_majority"] = int(sum(vlist) >= (len(vlist) / 2.0)) if vlist else ""
        cons_rows.append(row)

    ccols = (["model", "family", "arch", "id", "ground_truth", "agreed_pred", "n_sites",
              "consistent", "em"] + METRIC_KEYS + ["gt_qual_char", "len_gt", "len_pred"]
             + [f"lj_{j}" for j in judges] + ["lj_n_judges", "lj_mean", "lj_majority"])
    cons_path = os.path.join(OUT_DIR, "fusion_consensus.csv")
    with open(cons_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=ccols)
        w.writeheader()
        w.writerows(cons_rows)
    print(f"wrote {len(cons_rows)} consensus rows -> {cons_path}")

    # ---- validation vs lj_vs_em_table (judge = Qwen2.5-7B-Instruct) ----
    print("\n== Qwen2.5-7B-Instruct subset: gated EM% / LJ% per model (validate vs lj_vs_em_table) ==")
    print(f"{'model':<22}{'n':>6}{'EM%':>8}{'LJ%':>8}")
    for (model, judge), v in sorted(val.items()):
        if judge != "Qwen2.5-7B-Instruct":
            continue
        em_pct = 100 * v["em"] / v["n"] if v["n"] else 0
        lj_pct = 100 * v["lj"] / v["n"] if v["n"] else 0
        print(f"{model:<22}{v['n']:>6}{em_pct:>8.1f}{lj_pct:>8.1f}")


if __name__ == "__main__":
    main()
