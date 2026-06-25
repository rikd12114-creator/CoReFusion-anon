"""
Score the RQ2 deobfuscation runs with the ALL-SITES-CONSISTENCY metric.

The deobfuscation experiment (experiment_deobfuscation_refineID.py) renames
every local variable to a single-letter token and asks the dLLM to recover the
names. Each identifier occurs at several sites. The thesis scored this with
majority-vote EM per identifier group. This re-scores the SAME runs with the
consistency-gated metric we use for RQ1 (analysis/identifier_similarity_metrics.
eval_sample): an identifier is USABLE only if EVERY one of its sites emits the
SAME non-empty name (else the rename does not compile); only then is the single
agreed name scored against the original (clean) name.

Per identifier group it reports, aggregated by (model, mode):
  n_groups            total identifier groups scored
  consistency_rate    groups whose sites all agree on one non-empty name
  em_gated            STRICT all-sites EM: consistent AND agreed == original
                      (inconsistent groups count as wrong) -- the headline
  em_consistent       EM among the consistent groups only
  lev_sim / nw_sim    M1 IdBench string similarity (consistent subset)
  subtok_fuzzy        M2 sub-token similarity        (consistent subset)
  qual_char           M3 identifier quality          (consistent subset)
  copy_obf_rate       agreed name copies the obfuscated single-letter style
                      (== the obf label, or a 1-2 char lowercase token) --
                      the all-sites analogue of the thesis's 71.3% copy bias

Input  : results/deobfuscation_refineID/<Model>_<mode>_<ts>.csv
         (needs the site_predictions_json column added to the experiment)
Output : results/deobfuscation_refineID/consistency_leaderboard_<ts>.csv

Usage (pure CPU; run after the GPU jobs finish):
    python experiments/score_deobfuscation_consistency.py
    python experiments/score_deobfuscation_consistency.py --dict /usr/share/dict/words
    python experiments/score_deobfuscation_consistency.py --glob 'results/deobfuscation_refineID/*target-only*.csv'
"""

import os
import re
import sys
import csv
import json
import glob
import argparse
from datetime import datetime

csv.field_size_limit(2**31 - 1)

_HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(_HERE)
for _p in (_HERE, os.path.join(REPO, "analysis")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import identifier_similarity_metrics as M   # eval_sample + M1/M2/M3 + consistency gate

DEFAULT_GLOB = os.path.join(REPO, "results", "deobfuscation_refineID", "*.csv")
_SHORT_OBF = re.compile(r"^[a-z]{1,2}$")     # single-letter obfuscation style


def parse_run_name(path):
    """`<Model>_<mode>_<timestamp>.csv` -> (model, mode). mode is one of the
    known RQ2 modes; everything before it is the model name."""
    base = os.path.basename(path)
    stem = base[:-4] if base.endswith(".csv") else base
    for mode in ("all-masked", "target-only", "sequential"):
        tag = f"_{mode}_"
        if tag in stem:
            return stem.split(tag)[0], mode
    return stem, "unknown"


def score_file(path, dictionary):
    """Yield one scored row per identifier group in the file."""
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if "site_predictions_json" not in (reader.fieldnames or []):
            return  # old run without per-site predictions -> nothing to score
        model, mode = parse_run_name(path)
        for r in reader:
            raw = r.get("site_predictions_json") or ""
            obf_map = {}
            try:
                obf_map = json.loads(r.get("originals_json") or "{}")  # orig -> obf label
            except Exception:
                obf_map = {}
            if not raw:
                continue
            try:
                groups = json.loads(raw)              # {orig_name: [per-site preds]}
            except Exception:
                continue
            for orig, preds in groups.items():
                preds = [(_p or "").strip() for _p in preds]
                ev = M.eval_sample(preds, orig, dictionary)
                agreed = ev.get("agreed_pred", "")
                obf_label = (obf_map.get(orig) or "").strip()
                copies_obf = bool(agreed) and (
                    agreed == obf_label or bool(_SHORT_OBF.match(agreed)))
                yield {
                    "model": model, "mode": mode, "id": r.get("id", ""),
                    "identifier": orig, "n_sites": ev["n_sites"],
                    "consistent": ev["consistent"], "agreed_pred": agreed,
                    "em": ev["em"], "lev_sim": ev["lev_sim"], "nw_sim": ev["nw_sim"],
                    "subtok_fuzzy": ev["subtok_fuzzy"], "qual_char": ev["qual_char"],
                    "copies_obf": float(copies_obf),
                }


def aggregate(rows):
    """rows -> per-(model,mode) aggregate dict."""
    buckets = {}
    for r in rows:
        buckets.setdefault((r["model"], r["mode"]), []).append(r)
    out = []
    for (model, mode), rs in sorted(buckets.items()):
        n = len(rs)
        cons = [r for r in rs if r["consistent"] >= 1.0]
        nc = len(cons)
        mean = lambda key, src: (sum(x[key] for x in src) / len(src)) if src else 0.0
        out.append({
            "model": model, "mode": mode, "n_groups": n,
            "consistency_rate": nc / n if n else 0.0,
            "em_gated": mean("em", rs),               # strict all-sites EM (headline)
            "em_consistent": (sum(x["em"] for x in cons) / nc) if nc else 0.0,
            "lev_sim_cons": mean("lev_sim", cons),
            "nw_sim_cons": mean("nw_sim", cons),
            "subtok_fuzzy_cons": mean("subtok_fuzzy", cons),
            "qual_char_cons": mean("qual_char", cons),
            "copy_obf_rate": mean("copies_obf", rs),
        })
    return out


COLS = ["model", "mode", "n_groups", "consistency_rate", "em_gated",
        "em_consistent", "lev_sim_cons", "nw_sim_cons", "subtok_fuzzy_cons",
        "qual_char_cons", "copy_obf_rate"]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--glob", default=DEFAULT_GLOB)
    ap.add_argument("--dict", default="/usr/share/dict/words")
    ap.add_argument("--per-group-out", default=None,
                    help="optional path to dump every scored identifier group")
    args = ap.parse_args()

    dictionary = M.load_dictionary(args.dict)
    files = sorted(p for p in glob.glob(args.glob)
                   if not os.path.basename(p).startswith(("summary_", "consistency_")))
    if not files:
        sys.exit(f"No deobfuscation CSVs matched {args.glob}")

    all_rows = []
    for p in files:
        rows = list(score_file(p, dictionary))
        if rows:
            print(f"  scored {len(rows):>5} groups from {os.path.basename(p)}")
            all_rows.extend(rows)
        else:
            print(f"  [skip] {os.path.basename(p)} (no site_predictions_json -- rerun the experiment)")

    if not all_rows:
        sys.exit("No scorable rows found. Re-run experiment_deobfuscation_refineID.py "
                 "(patched to emit site_predictions_json) first.")

    if args.per_group_out:
        with open(args.per_group_out, "w", newline="", encoding="utf-8") as f:
            keys = list(all_rows[0].keys())
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(all_rows)
        print(f"  per-group dump -> {os.path.relpath(args.per_group_out, REPO)}")

    summary = aggregate(all_rows)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = os.path.join(REPO, "results", "deobfuscation_refineID",
                       f"consistency_leaderboard_{ts}.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLS, extrasaction="ignore")
        w.writeheader()
        w.writerows(summary)

    print(f"\n{'='*96}")
    print("  RQ2 DEOBFUSCATION -- ALL-SITES-CONSISTENCY EVALUATION")
    print(f"{'='*96}")
    print(f"{'model':<16}{'mode':<13}{'#grp':>7}{'consist%':>10}{'EM_gated':>10}"
          f"{'EM_cons':>9}{'fuzzy':>7}{'qualC':>7}{'copyObf%':>10}")
    print("  " + "-" * 92)
    for s in summary:
        print(f"{s['model']:<16}{s['mode']:<13}{s['n_groups']:>7}"
              f"{s['consistency_rate']*100:>9.1f}%{s['em_gated']:>10.3f}"
              f"{s['em_consistent']:>9.3f}{s['subtok_fuzzy_cons']:>7.3f}"
              f"{s['qual_char_cons']:>7.3f}{s['copy_obf_rate']*100:>9.1f}%")
    print(f"\n  Leaderboard -> {os.path.relpath(out, REPO)}")


if __name__ == "__main__":
    main()
