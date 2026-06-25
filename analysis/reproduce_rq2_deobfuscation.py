"""
Authoritative, reproducible recomputation of every RQ2 (deobfuscation) number
in the thesis, sourced *exclusively* from the latest 2026-06-12 runs.

It reproduces, from the raw per-file CSVs + the ground-truth targets:

  * Table V  / Fig 8(a) -- target-position EM under RQ1 (clean), all-masked,
                           and target-only, for DiffuCoder-7B and DreamCoder-7B.
  * Fig 8(b)            -- all-masked mean per-sample EM, stratified by the
                           number of distinct identifiers in the smell set S.
  * Table VI            -- DiffuCoder-7B target-only example wrong predictions.
  * Sec V-C breakdown   -- wrong predictions split into short-copy / long-
                           meaningful / empty, using the paper's *exact*
                           classifier (a prediction "copies the obfuscated
                           style" iff it matches ^[a-z]{1,2}$). This rule was
                           verified to reproduce the paper's 662/928 = 71.3%
                           on the run the paper was written against.

It also writes the `target-only` summary CSV that the experiment driver never
emitted (only the all-masked summaries were saved for the June runs).

WHY target-position EM needs its own script: neither the experiment summary
(which reports majority-vote *all-site* EM, ~7.4%) nor the consistency
leaderboard (which reports the strict consistency-gated EM, ~1.8%) emits the
per-target EM that Table V/Fig 8 report (12.4% / 2.8% ...). It is computed here
by joining each run's `predictions_json` with the RefineID target.

Inputs (pinned to the latest = 6/12 runs):
    data/test.csv                                          ground truth, col 2 = target
    results/deobfuscation_refineID/<Model>_<mode>_<ts>.csv June timestamps

Outputs -> results/deobfuscation_refineID/reproduced/
    table5_target_em.csv
    fig8b_idcount_buckets.csv
    table6_examples.csv
    secVC_wrong_breakdown.csv
    summary_target-only_20260612.csv     (the missing summary)
    rq2_numbers.json                      (everything, machine-readable)
    figures/new/fig8_rq2_em_june.{png,pdf}   (unless --no-figure)

Usage:
    python analysis/reproduce_rq2_deobfuscation.py
    python analysis/reproduce_rq2_deobfuscation.py --no-figure
"""

import os
import re
import csv
import sys
import json
import argparse

csv.field_size_limit(2**31 - 1)

# --- Repo-relative paths -----------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
DEOBF_DIR = os.path.join(REPO, "results", "deobfuscation_refineID")
OUT_DIR = os.path.join(DEOBF_DIR, "reproduced")
FIG_DIR = os.path.join(REPO, "figures", "new")
TEST_CSV = os.path.join(REPO, "data", "test.csv")

# --- Pinned June (latest) runs ----------------------------------------------
# Both modes for both models. These are the files Table V / Fig 8 must source.
RUNS = {
    ("DiffuCoder-7B", "all-masked"):  "DiffuCoder-7B_all-masked_20260612_102623.csv",
    ("DreamCoder-7B", "all-masked"):  "DreamCoder-7B_all-masked_20260612_102631.csv",
    ("DiffuCoder-7B", "target-only"): "DiffuCoder-7B_target-only_20260612_102623.csv",
    ("DreamCoder-7B", "target-only"): "DreamCoder-7B_target-only_20260612_102631.csv",
}
MODELS = ["DiffuCoder-7B", "DreamCoder-7B"]
HF_ID = {
    "DiffuCoder-7B": "apple/DiffuCoder-7B-Base",
    "DreamCoder-7B": "Dream-org/Dream-Coder-v0-Instruct-7B",
}

# RQ1 clean-context EM is produced by the *separate* RQ1 benchmark, not the
# deobfuscation runs. It is pinned here to the same RQ1 table the thesis reports
# so Fig 8(a) can show all three bars. Update these if RQ1 is re-pinned.
RQ1_CLEAN_EM = {"DiffuCoder-7B": 31.1, "DreamCoder-7B": 33.2}

# Stratification buckets for Fig 8(b), by #distinct identifiers in S.
BUCKETS = [(1, 10, "1-10"), (11, 20, "11-20"), (21, 50, "21-50"),
           (51, 100, "51-100"), (101, 10**9, "100+")]

# Paper's exact "copies the obfuscated single-letter style" classifier.
# Verified to reproduce 662/928 = 71.3% on the run the paper used.
SHORT_COPY = re.compile(r"^[a-z]{1,2}$")

# The nine example rows shown in Table VI (DiffuCoder-7B, target-only).
TABLE6_IDS = ["0", "2", "3", "6", "9", "11", "12", "13", "15"]


# --- Helpers -----------------------------------------------------------------
def load_targets():
    targets = {}
    with open(TEST_CSV, encoding="utf-8") as f:
        for row in csv.reader(f):
            if len(row) >= 3:
                targets[row[0]] = row[2].strip()
    return targets


def load_rows(name_or_path):
    path = name_or_path if os.path.isabs(name_or_path) else os.path.join(DEOBF_DIR, name_or_path)
    with open(path, newline="", encoding="utf-8") as f:
        return {r["id"]: r for r in csv.DictReader(f)}


def jloads(s):
    try:
        return json.loads(s) if s else {}
    except Exception:
        return {}


def is_skipped(r):
    return bool((r.get("skipped") or "").strip())


def bucket_label(n):
    for lo, hi, lab in BUCKETS:
        if lo <= n <= hi:
            return lab
    return BUCKETS[-1][2]


# --- Metric 1: target-position EM (Table V / Fig 8a) -------------------------
def target_position_em(rows, targets):
    """EM evaluated ONLY at the RefineID target identifier's prediction."""
    correct = scored = 0
    for sid, r in rows.items():
        if is_skipped(r):
            continue
        tgt = targets.get(sid)
        preds = jloads(r.get("predictions_json"))
        if tgt in preds:
            scored += 1
            if preds[tgt] == tgt:
                correct += 1
    em = 100.0 * correct / scored if scored else float("nan")
    return correct, scored, em


# --- Metric 2: Fig 8(b) stratified all-masked per-sample EM ------------------
def fig8b_buckets(rows):
    agg = {lab: [] for _, _, lab in BUCKETS}
    for r in rows.values():
        if is_skipped(r):
            continue
        try:
            n = int(r.get("num_unique_identifiers") or 0)
            em = float(r.get("per_sample_em_rate") or 0.0)
        except (TypeError, ValueError):
            continue
        if n > 0:
            agg[bucket_label(n)].append(em)
    out = []
    for _, _, lab in BUCKETS:
        v = agg[lab]
        out.append({"bucket": lab, "n_samples": len(v),
                    "mean_em_pct": round(100.0 * sum(v) / len(v), 2) if v else float("nan")})
    return out


# --- Metric 3: Sec V-C wrong-prediction breakdown ----------------------------
def wrong_breakdown(rows, targets):
    scored = correct = wrong = 0
    short = longm = empty = 0
    for sid, r in rows.items():
        if is_skipped(r):
            continue
        tgt = targets.get(sid)
        preds = jloads(r.get("predictions_json"))
        if tgt not in preds:
            continue
        scored += 1
        p = preds[tgt]
        if p == tgt:
            correct += 1
            continue
        wrong += 1
        if p == "":
            empty += 1
        elif SHORT_COPY.match(p):       # paper's exact "1-2 lowercase letters" rule
            short += 1
        else:
            longm += 1
    pct = lambda x: round(100.0 * x / wrong, 1) if wrong else float("nan")
    return {
        "scored": scored, "correct": correct, "wrong": wrong,
        "wrong_pct_of_scored": round(100.0 * wrong / scored, 1) if scored else float("nan"),
        "short_copy": short, "short_copy_pct": pct(short),
        "long_meaningful": longm, "long_meaningful_pct": pct(longm),
        "empty": empty, "empty_pct": pct(empty),
    }


# --- Metric 4: Table VI examples ---------------------------------------------
def table6(rows, targets, ids):
    out = []
    for sid in ids:
        r = rows.get(sid, {})
        tgt = targets.get(sid)
        preds = jloads(r.get("predictions_json"))
        origs = jloads(r.get("originals_json"))   # orig_name -> obf token
        obf = origs.get(tgt, "")
        pred = preds.get(tgt, "")
        out.append({"id": sid, "target": tgt, "obf": obf, "prediction": pred,
                    "matches_obf": "yes" if (pred and pred == obf) else ""})
    return out


# --- Writers -----------------------------------------------------------------
def write_csv(path, fieldnames, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def write_target_only_summary(rows_by_model):
    """Reproduce the experiment-style summary the driver never wrote for the
    June target-only runs. id_correct/id_total are summed from the CSV columns,
    exactly as run_experiment() would have aggregated them."""
    out_rows = []
    for model in MODELS:
        rows = rows_by_model[(model, "target-only")]
        id_correct = id_total = skipped = errors = 0
        for r in rows.values():
            if is_skipped(r):
                skipped += 1
                continue
            if (r.get("error") or "").strip():
                errors += 1
                continue
            id_correct += int(r.get("identifiers_correct") or 0)
            id_total += int(r.get("identifiers_total") or 0)
        processed = len(rows) - skipped - errors
        em = id_correct / id_total if id_total else 0.0
        out_rows.append({
            "model": model, "hf_id": HF_ID[model], "processed": processed,
            "skipped": skipped, "errors": errors,
            "identifier_em": f"{em:.4f}", "id_correct": id_correct, "id_total": id_total,
        })
    path = os.path.join(OUT_DIR, "summary_target-only_20260612.csv")
    write_csv(path, ["model", "hf_id", "processed", "skipped", "errors",
                     "identifier_em", "id_correct", "id_total"], out_rows)
    return out_rows


# --- Figure (Fig 8 a+b) ------------------------------------------------------
def make_figure(table5, fig8b):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"  [figure skipped: matplotlib unavailable: {e}]")
        return None

    os.makedirs(FIG_DIR, exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))

    # (a) grouped bars: 3 conditions x 2 models
    conds = ["RQ1\nclean", "RQ2\nall-masked", "RQ2\ntarget-only"]
    x = range(len(conds))
    w = 0.38
    diffu = [RQ1_CLEAN_EM["DiffuCoder-7B"], table5["DiffuCoder-7B"]["all-masked"],
             table5["DiffuCoder-7B"]["target-only"]]
    dream = [RQ1_CLEAN_EM["DreamCoder-7B"], table5["DreamCoder-7B"]["all-masked"],
             table5["DreamCoder-7B"]["target-only"]]
    b1 = ax1.bar([i - w / 2 for i in x], diffu, w, label="DiffuCoder-7B")
    b2 = ax1.bar([i + w / 2 for i in x], dream, w, label="DreamCoder-7B")
    for bars in (b1, b2):
        ax1.bar_label(bars, fmt="%.1f", padding=2, fontsize=9)
    ax1.set_xticks(list(x))
    ax1.set_xticklabels(conds)
    ax1.set_ylabel("Target-position Exact Match (%)")
    ax1.set_title("(a) Target-position EM")
    ax1.legend()
    ax1.set_ylim(0, max(diffu + dream) * 1.25)

    # (b) all-masked per-sample EM by #identifiers bucket
    labels = [b["bucket"] for b in fig8b["DiffuCoder-7B"]]
    dvals = [b["mean_em_pct"] for b in fig8b["DiffuCoder-7B"]]
    rvals = [b["mean_em_pct"] for b in fig8b["DreamCoder-7B"]]
    x2 = range(len(labels))
    c1 = ax2.bar([i - w / 2 for i in x2], dvals, w, label="DiffuCoder-7B")
    c2 = ax2.bar([i + w / 2 for i in x2], rvals, w, label="DreamCoder-7B")
    for bars in (c1, c2):
        ax2.bar_label(bars, fmt="%.1f", padding=2, fontsize=8)
    ax2.set_xticks(list(x2))
    ax2.set_xticklabels(labels)
    ax2.set_xlabel("Number of distinct identifiers in S")
    ax2.set_ylabel("Mean per-sample EM (%)")
    ax2.set_title("(b) All-masked EM by identifier count")
    ax2.legend()

    fig.tight_layout()
    png = os.path.join(FIG_DIR, "fig8_rq2_em_june.png")
    pdf = os.path.join(FIG_DIR, "fig8_rq2_em_june.pdf")
    fig.savefig(png, dpi=200)
    fig.savefig(pdf)
    plt.close(fig)
    return png, pdf


# --- Main --------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Reproduce all RQ2 deobfuscation numbers from the 6/12 runs.")
    ap.add_argument("--no-figure", action="store_true", help="skip Fig 8 generation")
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    targets = load_targets()

    # Load all four pinned runs, verifying they exist.
    rows_by_model = {}
    for key, fname in RUNS.items():
        path = os.path.join(DEOBF_DIR, fname)
        if not os.path.exists(path):
            sys.exit(f"MISSING pinned run: {path}")
        rows_by_model[key] = load_rows(path)

    bundle = {"source_runs": {f"{m}/{mode}": RUNS[(m, mode)] for (m, mode) in RUNS}}

    # ---- Table V: target-position EM ----
    print("=" * 72)
    print("TABLE V  -- target-position EM (%)   [source: 6/12 runs]")
    print(f"{'Model':<16}{'RQ1 clean':>11}{'all-masked':>12}{'target-only':>13}")
    table5 = {m: {} for m in MODELS}
    table5_rows = []
    for m in MODELS:
        am_c, am_n, am = target_position_em(rows_by_model[(m, "all-masked")], targets)
        to_c, to_n, to = target_position_em(rows_by_model[(m, "target-only")], targets)
        table5[m] = {"all-masked": round(am, 2), "target-only": round(to, 2),
                     "RQ1_clean": RQ1_CLEAN_EM[m]}
        print(f"{m:<16}{RQ1_CLEAN_EM[m]:>11.1f}{am:>12.2f}{to:>13.2f}")
        table5_rows.append({"model": m, "rq1_clean_em": RQ1_CLEAN_EM[m],
                            "all_masked_target_em": round(am, 2), "all_masked_correct": am_c, "all_masked_scored": am_n,
                            "target_only_target_em": round(to, 2), "target_only_correct": to_c, "target_only_scored": to_n})
    write_csv(os.path.join(OUT_DIR, "table5_target_em.csv"),
              ["model", "rq1_clean_em", "all_masked_target_em", "all_masked_correct", "all_masked_scored",
               "target_only_target_em", "target_only_correct", "target_only_scored"], table5_rows)
    bundle["table5"] = table5

    # ---- Fig 8(b): stratified all-masked EM ----
    print("\n" + "=" * 72)
    print("FIG 8(b) -- all-masked mean per-sample EM (%) by #identifiers bucket")
    fig8b = {}
    fig8b_rows = []
    for m in MODELS:
        b = fig8b_buckets(rows_by_model[(m, "all-masked")])
        fig8b[m] = b
        cells = "  ".join(f"{x['bucket']}:{x['mean_em_pct']:.1f}(n={x['n_samples']})" for x in b)
        print(f"  {m:<16}{cells}")
        for x in b:
            fig8b_rows.append({"model": m, **x})
    write_csv(os.path.join(OUT_DIR, "fig8b_idcount_buckets.csv"),
              ["model", "bucket", "n_samples", "mean_em_pct"], fig8b_rows)
    bundle["fig8b"] = fig8b

    # ---- Sec V-C: wrong-prediction breakdown ----
    print("\n" + "=" * 72)
    print("SEC V-C -- target-only wrong-prediction breakdown (paper rule ^[a-z]{1,2}$)")
    secvc = {}
    secvc_rows = []
    for m in MODELS:
        wb = wrong_breakdown(rows_by_model[(m, "target-only")], targets)
        secvc[m] = wb
        print(f"  {m}: wrong={wb['wrong']} ({wb['wrong_pct_of_scored']}% of {wb['scored']})  "
              f"short-copy={wb['short_copy']} ({wb['short_copy_pct']}%)  "
              f"long={wb['long_meaningful']} ({wb['long_meaningful_pct']}%)  "
              f"empty={wb['empty']} ({wb['empty_pct']}%)")
        secvc_rows.append({"model": m, **wb})
    write_csv(os.path.join(OUT_DIR, "secVC_wrong_breakdown.csv"),
              list(secvc_rows[0].keys()), secvc_rows)
    bundle["secVC"] = secvc

    # ---- Table VI: example wrong predictions (DiffuCoder target-only) ----
    print("\n" + "=" * 72)
    print("TABLE VI -- DiffuCoder-7B target-only examples")
    t6 = table6(rows_by_model[("DiffuCoder-7B", "target-only")], targets, TABLE6_IDS)
    for r in t6:
        dag = " (matches obf)" if r["matches_obf"] else ""
        print(f"  id={r['id']:<3} target={str(r['target']):<18} obf={r['obf']:<3} -> {r['prediction']}{dag}")
    write_csv(os.path.join(OUT_DIR, "table6_examples.csv"),
              ["id", "target", "obf", "prediction", "matches_obf"], t6)
    bundle["table6"] = t6

    # ---- Missing target-only summary ----
    print("\n" + "=" * 72)
    print("Writing the target-only summary the driver never emitted ...")
    summ = write_target_only_summary(rows_by_model)
    for s in summ:
        print(f"  {s['model']:<16} EM={s['identifier_em']}  ({s['id_correct']}/{s['id_total']}, "
              f"processed={s['processed']}, skipped={s['skipped']})")
    bundle["target_only_summary"] = summ

    # ---- machine-readable bundle ----
    with open(os.path.join(OUT_DIR, "rq2_numbers.json"), "w", encoding="utf-8") as f:
        json.dump(bundle, f, indent=2)

    # ---- figure ----
    if not args.no_figure:
        print("\n" + "=" * 72)
        res = make_figure(table5, fig8b)
        if res:
            print(f"  wrote {res[0]}\n        {res[1]}")

    print("\nAll outputs -> " + os.path.relpath(OUT_DIR, REPO))


if __name__ == "__main__":
    main()
