"""
DiffusionGemma-26B-A4B RQ1 analysis report.

Run AFTER you have downloaded the fixed DiffusionGemma prediction + judge CSVs
and rebuilt the leaderboard:

    python analysis/make_leaderboard.py          # rebuilds figures/new/leaderboard_full.csv
    python analysis/dgemma_report.py             # this report

It (1) sanity-checks the prediction CSV is non-empty (the old bug), (2) prints
DiffusionGemma's headline metrics, (3) ranks it among all models by all-sites EM
and by mean LLM-judge acceptance, and (4) compares it against the other dLLMs.
Stdlib only.
"""
import csv
import os
import statistics

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LB = os.path.join(ROOT, "figures/new/leaderboard_full.csv")
PRED = os.path.join(ROOT, "results/unified_refineID/predictions/DiffusionGemma-26B-A4B.csv")
CIS_CANDIDATES = [
    os.path.join(ROOT, "figures/new/leaderboard_cis.csv"),
    os.path.join(ROOT, "results/identifier_metrics/leaderboard_cis.csv"),
    os.path.join(ROOT, "figures/new/metric_eval/leaderboard_cis.csv"),
]
MODEL = "DiffusionGemma-26B-A4B"
LJ = ["LJ_Q7", "LJ_Q14", "LJ_Q32", "LJ_M24", "LJ_G27"]
JUDGE_LABEL = {"LJ_Q7": "Qwen2.5-7B", "LJ_Q14": "Qwen2.5-14B", "LJ_Q32": "Qwen2.5-32B",
               "LJ_M24": "Mistral-24B", "LJ_G27": "Gemma-2-27B"}


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return 0.0


def sanity_check_predictions():
    if not os.path.exists(PRED):
        print(f"!! prediction CSV not found: {PRED}\n   download it from the HPC cluster first.")
        return False
    csv.field_size_limit(2**31 - 1)
    n = n_nonempty = n_err = 0
    with open(PRED, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            n += 1
            if (r.get("first_pred") or "").strip():
                n_nonempty += 1
            if (r.get("error") or "").strip():
                n_err += 1
    print("== prediction sanity ==")
    print(f"   rows={n}  non-empty first_pred={n_nonempty} ({100*n_nonempty/max(n,1):.1f}%)  errors={n_err}")
    if n_nonempty == 0:
        print("   !! STILL ALL EMPTY -- this is the stale/broken CSV. Re-download the fixed one.")
    elif n_nonempty < n * 0.5:
        print("   ~ many empties; generation works but extraction may be lossy on some sites.")
    else:
        print("   OK: model produced real identifiers.")
    print()
    return n_nonempty > 0


def load_leaderboard():
    if not os.path.exists(LB):
        print(f"!! {LB} not found. Run: python analysis/make_leaderboard.py")
        return None
    rows = list(csv.DictReader(open(LB, encoding="utf-8")))
    for r in rows:
        r["_em"] = _f(r.get("em_gated"))
        r["_cons"] = _f(r.get("consistency"))
        r["_emc"] = _f(r.get("em_consistent"))
        r["_lj"] = statistics.mean([_f(r.get(j)) for j in LJ])
    return rows


def load_cis():
    for p in CIS_CANDIDATES:
        if os.path.exists(p):
            d = {}
            for r in csv.DictReader(open(p, encoding="utf-8")):
                m = r.get("model") or r.get("Model")
                cis = r.get("cis") or r.get("CIS") or r.get("cis_score")
                if m:
                    d[m] = _f(cis)
            return d, p
    return {}, None


def pct(x):
    return f"{100*x:5.1f}"


def main():
    sanity_check_predictions()
    rows = load_leaderboard()
    if not rows:
        return
    cis, cis_path = load_cis()

    dg = next((r for r in rows if r["model"] == MODEL), None)
    if dg is None:
        print(f"!! {MODEL} not in {LB}. Did make_leaderboard pick up the new CSV?")
        return
    if dg["_em"] == 0 and dg["_lj"] == 0:
        print(f"!! {MODEL} row is all zeros in the leaderboard -> still the empty run. "
              "Re-download the fixed CSVs and re-run make_leaderboard.")
        return

    n = len(rows)
    by_em = sorted(rows, key=lambda r: -r["_em"])
    by_lj = sorted(rows, key=lambda r: -r["_lj"])
    rank_em = [r["model"] for r in by_em].index(MODEL) + 1
    rank_lj = [r["model"] for r in by_lj].index(MODEL) + 1

    print("== DiffusionGemma-26B-A4B headline (RQ1, RefineID n=%s) ==" % dg.get("n", "?"))
    print(f"   all-sites EM (strict) : {pct(dg['_em'])}%   -> rank {rank_em}/{n}")
    print(f"   site-consistency rate : {pct(dg['_cons'])}%")
    print(f"   EM on consistent set  : {pct(dg['_emc'])}%")
    if MODEL in cis:
        print(f"   CIS                   : {pct(cis[MODEL])}%   (from {os.path.relpath(cis_path, ROOT)})")
    print(f"   LLM-judge (mean of 5) : {pct(dg['_lj'])}%   -> rank {rank_lj}/{n}")
    for j in LJ:
        print(f"       {JUDGE_LABEL[j]:<14}: {pct(_f(dg.get(j)))}%")
    print()

    # dLLM sub-block comparison
    dllms = sorted([r for r in rows if "dLLM" in (r.get("arch") or "")],
                   key=lambda r: -r["_em"])
    print("== dLLM block (sorted by all-sites EM) ==")
    print(f"   {'model':<26}{'arch':<24}{'EM':>7}{'cons':>7}{'EM_c':>7}{'LJ̄':>7}")
    for r in dllms:
        star = "  <--" if r["model"] == MODEL else ""
        print(f"   {r['model']:<26}{(r.get('arch') or ''):<24}"
              f"{pct(r['_em']):>7}{pct(r['_cons']):>7}{pct(r['_emc']):>7}{pct(r['_lj']):>7}{star}")
    print()

    # verbal summary vs the other dLLMs
    others = [r for r in dllms if r["model"] != MODEL]
    best = others[0] if others else None
    print("== read-out ==")
    if best:
        d_em = dg["_em"] - best["_em"]
        rel = "above" if d_em >= 0 else "below"
        print(f"   - On strict all-sites EM, DiffusionGemma is {pct(dg['_em'])}% "
              f"({rel} the best prior dLLM {best['model']} at {pct(best['_em'])}%).")
    print(f"   - It ranks {rank_em}/{n} on EM and {rank_lj}/{n} on LLM-judge acceptance overall.")
    print( "   - PROTOCOL CAVEAT: DiffusionGemma uses prompted per-site naming (block-AR, no")
    print( "     in-place FIM/infill), so its row is a dLLM end-task data point, not directly")
    print( "     comparable to the base-model FIM rows. Footnote this in Table II.")
    print()
    print("   Next: python analysis/viz_leaderboard.py && python analysis/viz_heatmap.py")
    print("   (DiffusionGemma now auto-appears; the all-zero exclusion no longer triggers.)")


if __name__ == "__main__":
    main()
