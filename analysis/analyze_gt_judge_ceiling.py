"""
Analyse the LLM-as-Judge "ground-truth ceiling" experiment.

Reads the per-sample CSVs produced by experiments/judge_refineid_ground_truth.py
(results/refineid_groundtruth_judge/groundtruth__judge_<judge>__<ts>.csv), where
each RefineID gold identifier was fed to the judge AS the prediction. Answers:

  * How often does each judge ACCEPT the dataset's own ground truth?  -> the
    empirical ceiling of LJ-acceptance, and a validation of the exact-match
    auto-accept shortcut used in the main benchmark.
  * Which gold names get rejected, and do judges agree on them?
  * Does the residual-[MASK] context (first-only fill) depress the ceiling?

Canonical run per judge = the LATEST per-sample CSV that has 1000 unique ids
(de-duped by id, so a duplicated --resume append is handled).

Outputs:
  results/refineid_groundtruth_judge/ceiling_summary.csv     (one row per judge)
  results/refineid_groundtruth_judge/ceiling_rejected.csv    (samples any judge rejected)
  figures/new/gt_judge_ceiling.png

Run from the repo root:  python analysis/analyze_gt_judge_ceiling.py
"""
import os
import re
import csv
import glob
import sys

csv.field_size_limit(2**31 - 1)

RES_DIR = "results/refineid_groundtruth_judge"
FIG_DIR = "figures/new"
BLUE, ORANGE, GREY = "#0076C2", "#FF8000", "#888888"

# size ladder, for stable ordering in the table / plot
JUDGE_ORDER = ["Qwen2.5-7B-Instruct", "Qwen2.5-14B-Instruct", "Mistral-Small-24B",
               "Gemma-2-27B-It", "Qwen2.5-32B-Instruct"]
_JRE = re.compile(r"^groundtruth__judge_(?P<j>.+)__(?P<ts>\d{8}_\d{6})\.csv$")


def _accept(row):
    return 1 if str(row.get("llm_verdict")).strip() == "1" else 0


def _truthy(x):
    return str(x or "").strip().lower() in ("1", "true", "yes")


def load_dedup(path):
    """id -> row, keeping the last occurrence (de-dups a resumed/appended file)."""
    out = {}
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            out[str(r.get("id"))] = r
    return out


def canonical_runs():
    """judge -> (path, {id: row}); latest CSV that de-dups to 1000 ids."""
    by_judge = {}
    for p in glob.glob(os.path.join(RES_DIR, "groundtruth__judge_*__*.csv")):
        m = _JRE.match(os.path.basename(p))
        if m:
            by_judge.setdefault(m.group("j"), []).append((m.group("ts"), p))
    canon = {}
    for j, lst in by_judge.items():
        lst.sort()
        chosen = None
        for _ts, p in reversed(lst):           # newest first
            d = load_dedup(p)
            if len(d) == 1000:
                chosen = (p, d)
                break
        if chosen is None:                      # fallback: newest whatever
            _ts, p = lst[-1]
            chosen = (p, load_dedup(p))
        canon[j] = chosen
    return canon


def summarize_judge(rows):
    n = len(rows)
    acc = sum(_accept(r) for r in rows)
    ss = [r for r in rows if str(r.get("n_sites")).strip() == "1"]
    ms = [r for r in rows if str(r.get("n_sites")).strip() not in ("1", "", None)]
    clean = [r for r in rows if r.get("ctx_has_residual_mask") not in (None, "")
             and not _truthy(r.get("ctx_has_residual_mask"))]
    resid = [r for r in rows if _truthy(r.get("ctx_has_residual_mask"))]

    def rate(sub):
        return (sum(_accept(r) for r in sub) / len(sub)) if sub else None
    return {
        "n": n, "accept": acc, "reject": n - acc,
        "gt_accept_rate": acc / n if n else 0.0,
        "single_site_n": len(ss), "single_site_rate": rate(ss),
        "multi_site_n": len(ms), "multi_site_rate": rate(ms),
        "clean_ctx_n": len(clean), "clean_ctx_rate": rate(clean),
        "residual_mask_n": len(resid), "residual_mask_rate": rate(resid),
    }


def main():
    canon = canonical_runs()
    if not canon:
        sys.exit(f"No per-sample CSVs in {RES_DIR}. Pull the results first.")
    judges = [j for j in JUDGE_ORDER if j in canon] + \
             [j for j in sorted(canon) if j not in JUDGE_ORDER]

    # ---- per-judge summary ----
    summaries = {j: summarize_judge(list(canon[j][1].values())) for j in judges}

    print("=" * 100)
    print("  RefineID GROUND-TRUTH acceptance by LLM-as-Judge   (gt fed as prediction)")
    print("=" * 100)
    print(f"{'judge':<22}{'n':>5}{'accept':>8}{'reject':>7}{'gt_accept%':>12}"
          f"{'1site%':>9}{'multi%':>9}{'clean%':>9}{'residMASK%':>12}")
    print("  " + "-" * 96)

    def pc(x):
        return "   n/a" if x is None else f"{x*100:6.2f}%"
    for j in judges:
        s = summaries[j]
        print(f"{j:<22}{s['n']:>5}{s['accept']:>8}{s['reject']:>7}"
              f"{s['gt_accept_rate']*100:>11.2f}%{pc(s['single_site_rate']):>9}"
              f"{pc(s['multi_site_rate']):>9}{pc(s['clean_ctx_rate']):>9}"
              f"{pc(s['residual_mask_rate']):>12}")
    macro = sum(summaries[j]["gt_accept_rate"] for j in judges) / len(judges)
    print("  " + "-" * 96)
    print(f"  panel mean gt_accept_rate = {macro*100:.2f}%   "
          f"(residual-[MASK] contexts: {summaries[judges[0]]['residual_mask_n']}/1000)")

    os.makedirs(RES_DIR, exist_ok=True)
    spath = os.path.join(RES_DIR, "ceiling_summary.csv")
    cols = ["judge", "file", "n", "accept", "reject", "gt_accept_rate",
            "single_site_n", "single_site_rate", "multi_site_n", "multi_site_rate",
            "clean_ctx_n", "clean_ctx_rate", "residual_mask_n", "residual_mask_rate"]
    with open(spath, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for j in judges:
            row = {"judge": j, "file": os.path.basename(canon[j][0]), **summaries[j]}
            w.writerow({k: row.get(k) for k in cols})
    print(f"\n  per-judge summary -> {spath}")

    # ---- cross-judge: which gold names did any judge reject? ----
    all_ids = sorted({i for j in judges for i in canon[j][1]}, key=lambda x: int(x))
    rej_rows = []
    for i in all_ids:
        verdicts = {j: _accept(canon[j][1][i]) for j in judges if i in canon[j][1]}
        n_acc = sum(verdicts.values())
        if n_acc < len(verdicts):                  # at least one judge rejected
            any_row = next(canon[j][1][i] for j in judges if i in canon[j][1])
            rej_rows.append({
                "id": i,
                "ground_truth": any_row.get("ground_truth", ""),
                "n_sites": any_row.get("n_sites", ""),
                "ctx_has_residual_mask": any_row.get("ctx_has_residual_mask", ""),
                "n_judges": len(verdicts),
                "n_accept": n_acc,
                "n_reject": len(verdicts) - n_acc,
                "rejected_by": ";".join(j for j in judges if verdicts.get(j) == 0),
            })
    rpath = os.path.join(RES_DIR, "ceiling_rejected.csv")
    with open(rpath, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["id", "ground_truth", "n_sites",
                           "ctx_has_residual_mask", "n_judges", "n_accept",
                           "n_reject", "rejected_by"])
        w.writeheader()
        w.writerows(rej_rows)

    print(f"\n  Gold names rejected by >=1 judge: {len(rej_rows)} "
          f"(out of 1000)  -> {rpath}")
    for r in rej_rows:
        print(f"    id={r['id']:>4}  gt={r['ground_truth']:<22} sites={r['n_sites']:>3} "
              f"resid={str(r['ctx_has_residual_mask']):<5} "
              f"rejected_by={r['rejected_by']} ({r['n_reject']}/{r['n_judges']})")

    # ---- figure: per-judge acceptance bars + panel-mean ceiling line ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        os.makedirs(FIG_DIR, exist_ok=True)
        rates = [summaries[j]["gt_accept_rate"] * 100 for j in judges]
        short = {"Qwen2.5-7B-Instruct": "Qwen2.5-7B", "Qwen2.5-14B-Instruct": "Qwen2.5-14B",
                 "Qwen2.5-32B-Instruct": "Qwen2.5-32B", "Mistral-Small-24B": "Mistral-24B",
                 "Gemma-2-27B-It": "Gemma2-27B"}
        labels = [short.get(j, j) for j in judges]
        fig, ax = plt.subplots(figsize=(8, 4.5))
        bars = ax.bar(labels, rates, color=BLUE, edgecolor="k", zorder=3)
        for b, j in zip(bars, judges):
            s = summaries[j]
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() - 0.6,
                    f"{s['accept']}/{s['n']}", ha="center", va="top",
                    color="white", fontsize=8, zorder=4)
        ax.axhline(macro * 100, color=ORANGE, ls="--", lw=1.5, zorder=2,
                   label=f"panel mean = {macro*100:.2f}%")
        ax.set_ylim(98.0, 100.2)
        ax.set_ylabel("ground-truth acceptance (%)")
        ax.set_title("LLM-as-Judge accepts RefineID gold identifiers ~always\n"
                     "(judge ceiling; gt fed as the prediction, n=1000)")
        ax.legend(loc="lower right", fontsize=9)
        ax.grid(axis="y", ls=":", alpha=0.5, zorder=0)
        fig.tight_layout()
        fpath = os.path.join(FIG_DIR, "gt_judge_ceiling.png")
        fig.savefig(fpath, dpi=150)
        print(f"\n  figure -> {fpath}")
    except Exception as e:                          # noqa: BLE001
        print(f"\n  [warn] figure skipped: {e}")


if __name__ == "__main__":
    main()
