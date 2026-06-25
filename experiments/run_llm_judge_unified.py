"""
Consistency-aware LLM-as-Judge over the UNIFIED refineID predictions.

Reads results/unified_refineID/predictions/<Model>.csv -- the per-site
predictions written by run_all_refineID_unified.py for every AR (FIM) /
Seq2Seq (T5, CodeT5+) / dLLM model -- and runs the SAME open-source judge as
experiments/llm_judge_variable_naming.py, but with the ALL-SITES CONSISTENCY
GATE applied (identical rule to analysis/identifier_similarity_metrics.eval_sample):

  * A sample is USABLE only if EVERY [MASK] site emitted the SAME non-empty
    identifier (else the renamed code won't compile). The judge is shown that
    single agreed name in context.
  * Inconsistent / empty samples -> automatic verdict 0 (NEVER sent to the LLM).
  * Exact match (agreed == ground truth) -> verdict 1 without an LLM call
    (same exact-match shortcut as the original judge).

We reuse the EXACT judge primitives (SYSTEM_PROMPT, chat template, Qwen2.5-7B
default, parse_verdict, judge_one) by importing llm_judge_variable_naming -- the
only thing new here is the consistency gate + two acceptance views.

MULTI-JUDGE: --judge-model is repeatable. Judges run sequentially in one
process (load -> judge every model -> unload -> next judge), each writing its
own per-judge leaderboard; with >1 judge a COMBINED leaderboard + a
model x judge acceptance matrix are emitted as well. For one-GPU-per-judge
parallelism submit one job per judge instead and merge afterwards with
--combine-only (scans existing per-sample CSVs, takes the latest run per
model+judge pair).

Outputs (under results/ so a single zip captures everything):
  results/unified_refineID/llm_judge/<Model>__judge_<judge>__<ts>.csv   per-sample
  results/unified_refineID/llm_judge/judge_leaderboard_<judge>_<ts>.csv per-model
  results/unified_refineID/llm_judge/judge_leaderboard_COMBINED_<ts>.csv

Two acceptance views (mirroring the metric script's two views):
  judge_acc_consistent : verdict==1 among the consistent (compilable) samples
  judge_acc_gated      : verdict==1 over ALL samples (inconsistent scored 0)
  em_gated             : strict all-sites EM (consistent AND agreed == gt)

Usage (the HPC cluster, after the per-model jobs filled predictions/):
  python experiments/run_llm_judge_unified.py                          # all models
  python experiments/run_llm_judge_unified.py --only Qwen2.5-Coder-7B --only CodeT5p-6B
  python experiments/run_llm_judge_unified.py --judge-model Qwen2.5-14B-Instruct
  # multi-judge (sizes ladder) in one job:
  python experiments/run_llm_judge_unified.py \
      --judge-model Qwen2.5-7B-Instruct --judge-model Qwen2.5-14B-Instruct \
      --judge-model Qwen2.5-32B-Instruct
  python experiments/run_llm_judge_unified.py --combine-only           # merge existing runs
  python experiments/run_llm_judge_unified.py --max-samples 50         # quick smoke
  python experiments/run_llm_judge_unified.py --resume
  python experiments/run_llm_judge_unified.py --list-models
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

# import paths: this file lives in experiments/; analysis/ is a sibling.
_HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(_HERE)
for p in (_HERE, os.path.join(REPO, "analysis")):
    if p not in sys.path:
        sys.path.insert(0, p)

import llm_judge_variable_naming as judge       # reuse EXACT judge prompt/model/parse
import identifier_similarity_metrics as M       # reuse read_pred_rows + consistency rule

csv.field_size_limit(2**31 - 1)

PRED_DIR = os.path.join(REPO, "results", "unified_refineID", "predictions")
OUT_DIR = os.path.join(REPO, "results", "unified_refineID", "llm_judge")
DATA_PATH = os.path.join(REPO, "data", "test.csv")

FIELDS = ["id", "ground_truth", "n_sites", "consistent", "agreed_pred",
          "exact_match", "llm_verdict", "llm_raw_output", "error"]


def is_consistent(preds):
    """ALL-SITES consistency gate -- byte-identical to
    identifier_similarity_metrics.eval_sample: every site emits the same
    NON-EMPTY identifier (else the renamed code does not compile)."""
    return len(preds) > 0 and preds[0] != "" and len(set(preds)) == 1


def load_done_ids(path):
    done = set()
    if os.path.exists(path):
        with open(path, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                v = str(r.get("id", "")).strip()
                if v:
                    done.add(v)
    return done


def summarize(out_path):
    """Recompute the summary from the FULL per-sample CSV (so --resume is exact)."""
    n = nc = v1_all = v1_c = em = errors = 0
    with open(out_path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            n += 1
            c = str(r.get("consistent", "0")).strip() in ("1", "1.0", "True", "true")
            e = str(r.get("exact_match", "False")).strip() in ("1", "True", "true")
            try:
                v = int(float(r.get("llm_verdict", "-1")))
            except ValueError:
                v = -1
            if c:
                nc += 1
            if v == 1:
                v1_all += 1
                if c:
                    v1_c += 1
            if e:
                em += 1
            if v == -1:
                errors += 1
    return {
        "n": n,
        "n_consistent": nc,
        "consistency_rate": (nc / n) if n else 0.0,
        "em_gated": (em / n) if n else 0.0,
        "judge_acc_consistent": (v1_c / nc) if nc else 0.0,
        "judge_acc_gated": (v1_all / n) if n else 0.0,
        "errors": errors,
    }


def judge_model_csv(name, pred_path, test_data, tok, model, judge_name,
                    max_samples=None, resume=False):
    from tqdm import tqdm
    os.makedirs(OUT_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe = re.sub(r"[/\\]", "_", judge_name)
    out_path = os.path.join(OUT_DIR, f"{name}__judge_{safe}__{ts}.csv")

    done = set()
    if resume:
        prev = sorted(glob.glob(os.path.join(OUT_DIR, f"{name}__judge_{safe}__*.csv")))
        if prev:
            out_path = prev[-1]              # append to the latest run for this model+judge
            done = load_done_ids(out_path)
            print(f"  [resume] {os.path.basename(out_path)} ({len(done)} already judged)")

    mode = "a" if (resume and done) else "w"
    f = open(out_path, mode, newline="", encoding="utf-8")
    w = csv.DictWriter(f, fieldnames=FIELDS)
    if mode == "w":
        w.writeheader()

    rows = list(M.read_pred_rows(pred_path))
    if max_samples is not None:
        rows = rows[:max_samples]

    for sid, gt, preds in tqdm(rows, desc=f"  judge {name[:28]}", unit="s"):
        if str(sid).strip() in done:
            continue

        consistent = is_consistent(preds)
        agreed = preds[0] if consistent else ""
        n_sites = len(preds)
        em = bool(consistent and agreed == gt)
        error = ""

        if not consistent:
            verdict, raw = 0, "inconsistent"          # won't compile -> hard fail
        elif em:
            verdict, raw = 1, "exact_match"           # shortcut, no LLM call
        elif not re.search(r"[a-zA-Z]", agreed):
            verdict, raw = 0, "invalid_prediction"    # junk token -> fail
        else:
            # look up code context from test.csv (1-indexed fallback like the orig judge)
            entry = None
            try:
                key = int(sid)
                entry = test_data.get(key) or test_data.get(key + 1)
            except (TypeError, ValueError):
                entry = None
            if entry is None:
                verdict, raw, error = -1, "", f"id {sid} not in test.csv"
            else:
                ctx = judge.extract_context(entry["masked_code"], agreed,
                                            judge.MAX_CONTEXT_CHARS)
                try:
                    verdict, raw = judge.judge_one(tok, model, ctx, gt, agreed)
                except Exception as ex:               # noqa: BLE001
                    verdict, raw, error = -1, "", str(ex)[:200]

        w.writerow({
            "id": sid, "ground_truth": gt, "n_sites": n_sites,
            "consistent": int(consistent), "agreed_pred": agreed,
            "exact_match": em, "llm_verdict": verdict,
            "llm_raw_output": str(raw)[:200], "error": error,
        })
        f.flush()

    f.close()
    summ = summarize(out_path)
    summ["model"] = name
    summ["judge_model"] = judge_name
    summ["out_file"] = os.path.relpath(out_path, REPO)
    return summ


LEAD_COLS = ["model", "judge_model", "n", "n_consistent", "consistency_rate",
             "em_gated", "judge_acc_consistent", "judge_acc_gated", "errors", "out_file"]


def write_judge_leaderboard(summaries, judge_name):
    """Per-judge leaderboard CSV + console table. Returns the CSV path."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_judge = re.sub(r"[/\\]", "_", judge_name)
    lead = os.path.join(OUT_DIR, f"judge_leaderboard_{safe_judge}_{ts}.csv")
    rows = sorted(summaries, key=lambda s: s["judge_acc_gated"], reverse=True)
    with open(lead, "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=LEAD_COLS, extrasaction="ignore")
        wr.writeheader()
        wr.writerows(rows)

    print(f"\n{'='*92}")
    print(f"  LLM-AS-JUDGE LEADERBOARD  (judge={judge_name}, sorted by gated acceptance)")
    print(f"{'='*92}")
    print(f"{'model':<22}{'consist%':>9}{'EM_gat':>8}{'judge_cons':>12}{'judge_gat':>11}{'err':>6}")
    print("  " + "-" * 88)
    for s in rows:
        print(f"{s['model']:<22}{s['consistency_rate']*100:>8.1f}%{s['em_gated']:>8.3f}"
              f"{s['judge_acc_consistent']:>12.3f}{s['judge_acc_gated']:>11.3f}{s['errors']:>6}")
    print(f"\n  Leaderboard -> {os.path.relpath(lead, REPO)}")
    return lead


def write_combined(all_summaries):
    """Cross-judge long-format CSV + model x judge acceptance matrix."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    lead = os.path.join(OUT_DIR, f"judge_leaderboard_COMBINED_{ts}.csv")
    rows = sorted(all_summaries, key=lambda s: (s["model"], s["judge_model"]))
    with open(lead, "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=LEAD_COLS, extrasaction="ignore")
        wr.writeheader()
        wr.writerows(rows)

    judges_l = sorted({s["judge_model"] for s in all_summaries})
    models_l = sorted({s["model"] for s in all_summaries})
    acc = {(s["model"], s["judge_model"]): s["judge_acc_gated"] for s in all_summaries}

    print(f"\n{'='*100}")
    print("  COMBINED judge_acc_gated  (rows = benchmarked model, cols = judge)")
    print(f"{'='*100}")
    header = "model".ljust(22) + "".join(j[:17].rjust(19) for j in judges_l)
    print(header)
    print("  " + "-" * (max(len(header) - 2, 40)))
    for m_ in models_l:
        line = m_[:21].ljust(22)
        for j_ in judges_l:
            v = acc.get((m_, j_))
            line += ("--".rjust(19) if v is None else f"{v:.3f}".rjust(19))
        print(line)
    print(f"\n  Combined leaderboard -> {os.path.relpath(lead, REPO)}")
    return lead


_PER_SAMPLE_RE = re.compile(
    r"^(?P<model>.+?)__judge_(?P<judge>.+)__(?P<ts>\d{8}_\d{6})\.csv$")


def collect_existing_summaries():
    """--combine-only: latest per-sample CSV per (model, judge), re-summarized."""
    latest = {}
    for p in glob.glob(os.path.join(OUT_DIR, "*__judge_*__*.csv")):
        m_ = _PER_SAMPLE_RE.match(os.path.basename(p))
        if not m_:
            continue
        key = (m_.group("model"), m_.group("judge"))
        if key not in latest or m_.group("ts") > latest[key][0]:
            latest[key] = (m_.group("ts"), p)
    out = []
    for (model_name, judge_name), (_ts, p) in sorted(latest.items()):
        s = summarize(p)
        s["model"] = model_name
        s["judge_model"] = judge_name
        s["out_file"] = os.path.relpath(p, REPO)
        out.append(s)
    return out


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
    ap.add_argument("--only", action="append", default=None,
                    help="model name(s) (CSV stem in predictions/), repeatable")
    ap.add_argument("--judge-model", action="append", default=None,
                    help="registry name or full HF id; REPEATABLE for multi-judge runs "
                         f"(registry: {', '.join(judge.JUDGE_REGISTRY)}; "
                         f"default: {judge.DEFAULT_JUDGE})")
    ap.add_argument("--max-samples", type=int, default=None)
    ap.add_argument("--resume", action="store_true",
                    help="skip already-judged ids (append to latest output per model+judge)")
    ap.add_argument("--combine-only", action="store_true",
                    help="no judging: rebuild COMBINED leaderboard from existing per-sample CSVs")
    ap.add_argument("--predictions-dir", default=PRED_DIR)
    ap.add_argument("--data", default=DATA_PATH)
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
        print(f"  found {len(rows)} (model, judge) result sets")
        write_combined(rows)
        return

    pred_dir = args.predictions_dir
    all_csvs = sorted(glob.glob(os.path.join(pred_dir, "*.csv")))
    if not all_csvs:
        sys.exit(f"No prediction CSVs in {pred_dir}. Run the per-model jobs first.")
    names = [os.path.splitext(os.path.basename(p))[0] for p in all_csvs]
    if args.only:
        want = set(args.only)
        picked = [(n, p) for n, p in zip(names, all_csvs) if n in want]
        missing = want - {n for n, _ in picked}
        if missing:
            print(f"  [warn] not found in predictions/: {', '.join(sorted(missing))}")
    else:
        picked = list(zip(names, all_csvs))
    if not picked:
        sys.exit("No matching prediction CSVs selected. Use --list-models / check --only names.")

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
    print("  Consistency-gated LLM-as-Judge over unified refineID predictions")
    print(f"  judges={[n for _, n in resolved]}  device={judge.DEVICE}")
    print(f"  models={len(picked)}  predictions={pred_dir}")
    print(f"  out -> {os.path.relpath(OUT_DIR, REPO)}")
    print("=" * 78)

    print("\nLoading test.csv for code context ...")
    test_data = judge.load_test_data(args.data)
    print(f"  {len(test_data)} samples loaded.")

    all_summaries, t0 = [], time.time()
    for j_idx, (model_id, judge_name) in enumerate(resolved):
        print(f"\n{'#'*78}\n  JUDGE {j_idx + 1}/{len(resolved)}: {judge_name}  ({model_id})\n{'#'*78}")
        tj = time.time()
        tok, model = judge.load_judge(model_id)     # loaded ONCE, reused across models

        summaries = []
        for name, path in picked:
            print(f"\n{'-'*78}\n  {name}  [judge={judge_name}]\n{'-'*78}")
            s = judge_model_csv(name, path, test_data, tok, model, judge_name,
                                max_samples=args.max_samples, resume=args.resume)
            summaries.append(s)
            print(f"  consist={s['consistency_rate']:.1%}  EM_gated={s['em_gated']:.3f}  "
                  f"judge_acc(consistent)={s['judge_acc_consistent']:.3f}  "
                  f"judge_acc(gated)={s['judge_acc_gated']:.3f}  errors={s['errors']}")

        write_judge_leaderboard(summaries, judge_name)
        all_summaries.extend(summaries)
        print(f"  judge {judge_name} done in {time.time() - tj:.1f}s")

        _free_judge(tok, model)                     # fully release VRAM before next judge

    if len(resolved) > 1:
        write_combined(all_summaries)
    print(f"\n  Total time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
