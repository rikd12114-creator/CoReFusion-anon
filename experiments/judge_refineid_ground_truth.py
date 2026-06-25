"""
LLM-as-Judge on the RefineID GROUND TRUTH itself (judge "ceiling" / sanity check)
=================================================================================

Question this answers
---------------------
Our benchmark scores model predictions with an open-source LLM-as-Judge
(experiments/llm_judge_variable_naming.py).  In that pipeline an EXACT match
(prediction == ground_truth) is auto-accepted (verdict 1) WITHOUT ever asking
the judge.  This script removes that assumption and measures it directly:

    "How often does the judge ACCEPT the RefineID dataset's own ground-truth
     identifier, shown in real code context?"

i.e. it feeds the **ground-truth name as the prediction** for every one of the
1,000 RefineID samples and records the judge's verdict.  The resulting
acceptance rate is the empirical ceiling of LJ-acceptance for this benchmark:
no model prediction can be judged "more correct" than the gold answer, so this
is the upper bound any model could reach under this judge, and a validation of
the exact-match auto-accept shortcut.

What is kept IDENTICAL to the original LJ run
---------------------------------------------
Everything about the judge is imported and reused verbatim from
experiments/llm_judge_variable_naming.py — so the model and the way the model
is called are byte-for-byte the same:
  * judge model        : Qwen/Qwen2.5-7B-Instruct (DEFAULT_JUDGE); same registry
  * SYSTEM_PROMPT      : reused as-is
  * user prompt        : build_user_prompt(context, ground_truth, prediction)
  * code context       : extract_context() — first [MASK] -> prediction, the rest
                         stay [MASK] (exactly the original behaviour), centred
                         window of MAX_CONTEXT_CHARS = 2000 chars
  * generation         : judge_one() — greedy, max_new_tokens=32, same chat
                         template / parse_verdict()
  * test.csv loading   : load_test_data()

Because here prediction == ground_truth, the user prompt shows the same string
in both the "Ground-truth variable name" and "Predicted variable name" fields —
this is the faithful consequence of keeping the prompt unchanged, and is exactly
the (gt, gt) pair whose acceptance the exact-match shortcut assumes is 1.

The ONLY behavioural change vs the original judge: we do NOT short-circuit exact
matches to verdict 1 — we actually send every sample to the LLM (that is the
whole point).  The original "invalid prediction" guard (no letters) is kept for
fidelity but essentially never fires on real identifiers.

Multi-judge: --judge-model is repeatable (same semantics as
run_llm_judge_unified.py); judges run sequentially in one process
(load -> judge all 1,000 -> unload -> next judge). To reproduce the SAME 5-judge
panel as the main benchmark (Qwen2.5-7B / Qwen2.5-14B / Mistral-Small-24B /
Gemma-2-27B-It / Qwen2.5-32B), run one GPU job per judge with
server/jobs/gt_judge_multi.sh, then merge with --combine-only on the login node.
The big three (24B/27B/32B, bf16 47-65GB) need a 96GB card; Gemma-2-27B-It is
gated -> set HF_TOKEN.

Outputs (under results/ so one zip captures everything):
  results/refineid_groundtruth_judge/groundtruth__judge_<judge>__<ts>.csv   per-sample
  results/refineid_groundtruth_judge/groundtruth_summary_<ts>.csv           per-judge

Usage (the HPC cluster, from the repo root after sourcing server/env_cluster.sh):
  python experiments/judge_refineid_ground_truth.py                       # default judge
  python experiments/judge_refineid_ground_truth.py --judge-model Qwen2.5-14B-Instruct
  python experiments/judge_refineid_ground_truth.py \
      --judge-model Qwen2.5-7B-Instruct --judge-model Qwen2.5-14B-Instruct # multi-judge
  python experiments/judge_refineid_ground_truth.py --max-samples 50      # quick smoke
  python experiments/judge_refineid_ground_truth.py --resume              # continue a run
  python experiments/judge_refineid_ground_truth.py --mask-fill all       # clean-ctx upper bound
  python experiments/judge_refineid_ground_truth.py --list-models

Residual-[MASK] note: with the faithful default (--mask-fill first) the judged
context still contains literal [MASK] tokens in ~90% of multi-site rows (the
original judge replaces only the first site). That is the FAIR ceiling -- every
benchmarked model was judged under the same condition. The per-sample CSV records
`ctx_has_residual_mask` and the summary stratifies acceptance by single-/multi-site
and clean-/residual-mask context so the effect is measured, not assumed. Run
`--mask-fill all` for the clean-context upper bound.
"""

import os
import sys
import csv
import re
import gc
import glob
import time
import argparse
from datetime import datetime

# This file lives in experiments/; the canonical judge module is a sibling.
_HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(_HERE)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import llm_judge_variable_naming as judge       # reuse EXACT judge model/prompt/calls

csv.field_size_limit(2**31 - 1)

DATA_PATH = os.path.join(REPO, "data", "test.csv")
OUT_DIR = os.path.join(REPO, "results", "refineid_groundtruth_judge")

FIELDS = ["id", "ground_truth", "prediction", "n_sites", "exact_match",
          "ctx_has_residual_mask", "llm_verdict", "llm_raw_output", "error"]


def load_done_ids(path):
    done = set()
    if os.path.exists(path):
        with open(path, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                v = str(r.get("id", "")).strip()
                if v:
                    done.add(v)
    return done


def _truthy(x):
    return str(x or "").strip().lower() in ("1", "true", "yes")


def summarize(out_path):
    """Recompute the summary from the FULL per-sample CSV (so --resume is exact).

    Robust to a truncated trailing row (a kill mid-writerow leaves DictReader
    yielding None for the missing cells -- exactly the file --resume recovers).
    Also reports two diagnostic strata for the residual-[MASK] context effect:
    single- vs multi-site, and clean- vs residual-mask context.
    """
    n = v1 = v0 = parse_fail = errors = 0
    ss_n = ss_v1 = ms_n = ms_v1 = 0          # single- / multi-site
    clean_n = clean_v1 = cont_n = cont_v1 = 0  # clean- / residual-mask context
    with open(out_path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            n += 1
            try:
                v = int(float(r.get("llm_verdict") or -1))
            except (ValueError, TypeError):
                v = -1
            err = str(r.get("error") or "").strip()
            acc = 1 if v == 1 else 0
            if v == 1:
                v1 += 1
            elif v == 0:
                v0 += 1
            if v == -1 and not err:          # genuine parse failure (disjoint from errors)
                parse_fail += 1
            if err:
                errors += 1
            # stratum: usage-site count
            try:
                ns = int(float(r.get("n_sites") or 0))
            except (ValueError, TypeError):
                ns = 0
            if ns == 1:
                ss_n += 1; ss_v1 += acc
            elif ns >= 2:
                ms_n += 1; ms_v1 += acc
            # stratum: did the judged context still contain a literal [MASK]?
            rm = str(r.get("ctx_has_residual_mask") or "").strip()
            if rm != "":                     # invalid-guard rows (no ctx) are excluded
                if _truthy(rm):
                    cont_n += 1; cont_v1 += acc
                else:
                    clean_n += 1; clean_v1 += acc

    def rate(a, b):
        return (a / b) if b else None

    return {
        "n": n,
        "accept": v1,
        "reject": v0,
        "parse_fail": parse_fail,        # -1 verdicts w/o exception (count as reject in rate)
        "errors": errors,                # exception rows (disjoint from parse_fail)
        # acceptance rate over ALL samples; parse failures count as NOT accepted
        # (identical scoring convention to the original judge: score = max(v,0)).
        "gt_accept_rate": (v1 / n) if n else 0.0,
        # diagnostics for the residual-[MASK] context bias (see --mask-fill):
        "single_site_n": ss_n, "single_site_rate": rate(ss_v1, ss_n),
        "multi_site_n": ms_n, "multi_site_rate": rate(ms_v1, ms_n),
        "clean_ctx_n": clean_n, "clean_ctx_rate": rate(clean_v1, clean_n),
        "residual_mask_n": cont_n, "residual_mask_rate": rate(cont_v1, cont_n),
    }


def build_ctx(masked, pred, mask_fill):
    """Code context shown to the judge.

    mask_fill='first' (default, FAITHFUL to the original judge): reuse
    judge.extract_context verbatim -- only the FIRST [MASK] becomes the
    prediction, the rest stay [MASK]. This is the fair apples-to-apples ceiling
    because every benchmarked model was judged under this exact condition.

    mask_fill='all' (clean-context upper bound): also fill the remaining sites
    with the ground truth before windowing, so no literal [MASK] noise remains
    (the judge's SYSTEM_PROMPT rule 4 calls 'MASK' a wrong token). Windowing +
    first-mask substitution still go through judge.extract_context, so the only
    difference vs 'first' is that the trailing masks are filled.
    """
    if mask_fill == "all":
        pos = masked.find("[MASK]")
        if pos != -1:
            cut = pos + len("[MASK]")
            masked = masked[:cut] + masked[cut:].replace("[MASK]", pred)
    return judge.extract_context(masked, pred, judge.MAX_CONTEXT_CHARS)


def judge_ground_truth(test_data, tok, model, judge_name,
                       max_samples=None, resume=False, mask_fill="first"):
    from tqdm import tqdm
    os.makedirs(OUT_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe = re.sub(r"[/\\]", "_", judge_name)
    out_path = os.path.join(OUT_DIR, f"groundtruth__judge_{safe}__{ts}.csv")

    done = set()
    if resume:
        prev = sorted(glob.glob(os.path.join(OUT_DIR, f"groundtruth__judge_{safe}__*.csv")))
        if prev:
            out_path = prev[-1]          # append to the latest run for this judge
            done = load_done_ids(out_path)
            print(f"  [resume] {os.path.basename(out_path)} ({len(done)} already judged)")

    mode = "a" if (resume and done) else "w"
    f = open(out_path, mode, newline="", encoding="utf-8")
    w = csv.DictWriter(f, fieldnames=FIELDS)
    if mode == "w":
        w.writeheader()

    # Iterate the RefineID dataset itself, in id order (0..999).
    items = sorted(test_data.items(), key=lambda kv: kv[0])
    if max_samples is not None:
        items = items[:max_samples]

    for sid, entry in tqdm(items, desc=f"  gt-judge [{judge_name[:24]}]", unit="s"):
        if str(sid).strip() in done:
            continue

        gt = entry["ground_truth"]
        masked = entry["masked_code"]
        pred = gt                              # judge the GROUND TRUTH itself
        n_sites = masked.count("[MASK]")       # metadata only; does NOT gate the verdict
        error = ""
        residual = ""                          # "" for guarded rows (no context built)

        # Faithful to the original judge's pre-LLM guard (effectively never fires
        # on a real identifier). We deliberately do NOT short-circuit the
        # exact match here — the whole point is to ask the LLM about (gt, gt).
        if not pred or not re.search(r"[a-zA-Z]", pred):
            verdict, raw = 0, "invalid_prediction"
        else:
            ctx = build_ctx(masked, pred, mask_fill)
            residual = "[MASK]" in ctx         # leftover mask noise in the judged window
            try:
                verdict, raw = judge.judge_one(tok, model, ctx, gt, pred)
            except Exception as ex:            # noqa: BLE001
                verdict, raw, error = -1, "", str(ex)[:200]

        w.writerow({
            "id": sid, "ground_truth": gt, "prediction": pred,
            "n_sites": n_sites, "exact_match": True,
            "ctx_has_residual_mask": residual,
            "llm_verdict": verdict, "llm_raw_output": str(raw)[:200],
            "error": error,
        })
        f.flush()

    f.close()
    summ = summarize(out_path)
    summ["judge_model"] = judge_name
    summ["out_file"] = os.path.relpath(out_path, REPO)
    return summ


SUMM_COLS = ["judge_model", "mask_fill", "n", "accept", "reject", "parse_fail",
             "errors", "gt_accept_rate",
             "single_site_n", "single_site_rate",
             "multi_site_n", "multi_site_rate",
             "clean_ctx_n", "clean_ctx_rate",
             "residual_mask_n", "residual_mask_rate", "out_file"]


_PER_SAMPLE_RE = re.compile(r"^groundtruth__judge_(?P<judge>.+)__(?P<ts>\d{8}_\d{6})\.csv$")


def collect_existing_summaries():
    """--combine-only: latest per-sample CSV per judge, re-summarized.

    Lets the one-card-per-judge launcher (server/jobs/gt_judge_multi.sh) run each
    judge in its own GPU job, then merge them into one cross-judge table on the
    login node (no GPU). Mirrors run_llm_judge_unified.collect_existing_summaries.
    """
    latest = {}
    for p in glob.glob(os.path.join(OUT_DIR, "groundtruth__judge_*__*.csv")):
        m = _PER_SAMPLE_RE.match(os.path.basename(p))
        if not m:
            continue
        j, ts = m.group("judge"), m.group("ts")
        if j not in latest or ts > latest[j][0]:
            latest[j] = (ts, p)
    out = []
    for j, (_ts, p) in sorted(latest.items()):
        s = summarize(p)
        s["judge_model"] = j
        s["mask_fill"] = ""              # not recoverable from the per-sample CSV
        s["out_file"] = os.path.relpath(p, REPO)
        out.append(s)
    return out


def _pct(x):
    return "   n/a" if x is None else f"{x*100:5.2f}%"


def write_summary(summaries):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(OUT_DIR, f"groundtruth_summary_{ts}.csv")
    rows = sorted(summaries, key=lambda s: s["gt_accept_rate"], reverse=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=SUMM_COLS, extrasaction="ignore")
        wr.writeheader()
        wr.writerows(rows)

    print(f"\n{'='*86}")
    print("  RefineID GROUND-TRUTH acceptance by LLM-as-Judge  (gt fed as prediction)")
    print(f"{'='*86}")
    print(f"{'judge':<22}{'fill':>6}{'n':>6}{'accept':>8}{'reject':>8}"
          f"{'parse?':>7}{'gt_accept%':>12}")
    print("  " + "-" * 82)
    for s in rows:
        print(f"{s['judge_model']:<22}{s.get('mask_fill','first'):>6}{s['n']:>6}"
              f"{s['accept']:>8}{s['reject']:>8}{s['parse_fail']:>7}"
              f"{s['gt_accept_rate']*100:>11.2f}%")
    print("\n  diagnostics (residual-[MASK] context effect):")
    print(f"  {'judge':<22}{'1-site':>9}{'(n)':>6}{'multi':>9}{'(n)':>6}"
          f"{'clean':>9}{'(n)':>6}{'residMASK':>11}{'(n)':>6}")
    for s in rows:
        print(f"  {s['judge_model']:<22}{_pct(s['single_site_rate']):>9}{s['single_site_n']:>6}"
              f"{_pct(s['multi_site_rate']):>9}{s['multi_site_n']:>6}"
              f"{_pct(s['clean_ctx_rate']):>9}{s['clean_ctx_n']:>6}"
              f"{_pct(s['residual_mask_rate']):>11}{s['residual_mask_n']:>6}")
    print(f"\n  Summary -> {os.path.relpath(path, REPO)}")
    return path


def _free_judge(tok, model):
    del model, tok
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    gc.collect()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--judge-model", action="append", default=None,
                    help="registry name or full HF id; REPEATABLE for multi-judge runs "
                         f"(registry: {', '.join(judge.JUDGE_REGISTRY)}; "
                         f"default: {judge.DEFAULT_JUDGE})")
    ap.add_argument("--max-samples", type=int, default=None,
                    help="judge only the first N samples (smoke test)")
    ap.add_argument("--resume", action="store_true",
                    help="skip already-judged ids (append to latest output per judge)")
    ap.add_argument("--mask-fill", choices=["first", "all"], default="first",
                    help="'first' (default, FAITHFUL to the original judge): only the "
                         "first [MASK] becomes the gt, others stay [MASK]. 'all': also "
                         "fill the remaining sites with the gt for a clean-context upper "
                         "bound (removes residual-[MASK] noise).")
    ap.add_argument("--combine-only", action="store_true",
                    help="no judging: rebuild the cross-judge summary from existing "
                         "per-sample CSVs (run after one-card-per-judge jobs finish)")
    ap.add_argument("--data", default=DATA_PATH, help="path to RefineID test.csv")
    ap.add_argument("--list-models", action="store_true")
    args = ap.parse_args()

    if args.list_models:
        for nm, mid in judge.JUDGE_REGISTRY.items():
            print(f"  {nm:<24} -> {mid}")
        return

    if args.combine_only:
        rows = collect_existing_summaries()
        if not rows:
            sys.exit(f"No per-sample judge CSVs found in {OUT_DIR}.")
        print(f"  found {len(rows)} judge result set(s)")
        write_summary(rows)
        return

    # resolve judge ids/names exactly like llm_judge_variable_naming.main()
    judges_arg = args.judge_model or [judge.DEFAULT_JUDGE]
    resolved = []
    for j in judges_arg:
        if j in judge.JUDGE_REGISTRY:
            resolved.append((judge.JUDGE_REGISTRY[j], j))
        else:
            resolved.append((j, j.split("/")[-1]))

    os.makedirs(OUT_DIR, exist_ok=True)
    print("=" * 78)
    print("  LLM-as-Judge on the RefineID GROUND TRUTH (judge ceiling)")
    print(f"  judges={[n for _, n in resolved]}  device={judge.DEVICE}")
    print(f"  mask_fill={args.mask_fill}  data={os.path.relpath(args.data, REPO)}")
    print(f"  out -> {os.path.relpath(OUT_DIR, REPO)}")
    print("=" * 78)

    print("\nLoading RefineID test.csv ...")
    test_data = judge.load_test_data(args.data)
    print(f"  {len(test_data)} samples loaded.")
    if not test_data:
        sys.exit(f"No samples loaded from {args.data}.")

    all_summaries, t0 = [], time.time()
    for j_idx, (model_id, judge_name) in enumerate(resolved):
        print(f"\n{'#'*78}\n  JUDGE {j_idx + 1}/{len(resolved)}: {judge_name}  ({model_id})\n{'#'*78}")
        tj = time.time()
        tok, model = judge.load_judge(model_id)     # loaded ONCE per judge
        s = judge_ground_truth(test_data, tok, model, judge_name,
                               max_samples=args.max_samples, resume=args.resume,
                               mask_fill=args.mask_fill)
        s["mask_fill"] = args.mask_fill
        all_summaries.append(s)
        print(f"  gt_accept_rate={s['gt_accept_rate']:.4f}  "
              f"accept={s['accept']}/{s['n']}  reject={s['reject']}  "
              f"parse_fail={s['parse_fail']}  errors={s['errors']}")
        print(f"  judge {judge_name} done in {time.time() - tj:.1f}s")
        _free_judge(tok, model)                     # release VRAM before next judge

    write_summary(all_summaries)
    print(f"\n  Total time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
