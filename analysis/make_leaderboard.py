"""Per-model leaderboard, Table-II style (grouped by architecture), with the
full metric suite + one column per LLM-as-Judge.

Self-contained: EM / consistency / M1 / M2 / M3 are recomputed from
predictions/ via identifier_similarity_metrics.eval_sample (does NOT trust the
possibly-stale metrics/ folder). LJ columns come from llm_judge/ (latest run
per model+judge), acceptance = verdict==1 over all samples (gated).

Usage:
    python analysis/make_leaderboard.py
    python analysis/make_leaderboard.py --results results/unified_refineID
"""
import os, sys, csv, glob, re, argparse
csv.field_size_limit(2**31 - 1)
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import identifier_similarity_metrics as M

ARCH_ORDER = ["dLLM (fixed-canvas)", "dLLM (variable canvas)", "dLLM (block-AR)",
              "Decoder-only (FIM)", "Encoder-decoder"]
ARCH = {}
for m in ["DreamCoder-7B", "DiffuCoder-7B"]: ARCH[m] = "dLLM (fixed-canvas)"
for m in ["DreamOn-7B"]: ARCH[m] = "dLLM (variable canvas)"
for m in ["DiffusionGemma-26B-A4B"]: ARCH[m] = "dLLM (block-AR)"
for m in ["CodeLlama-13B","CodeLlama-7B","StarCoder2-15B","StarCoder2-7B","StarCoder2-3B",
          "DeepSeek-Coder-6.7B","DeepSeek-Coder-1.3B","Qwen2.5-Coder-14B","Qwen2.5-Coder-7B",
          "Qwen2.5-Coder-3B","Qwen2.5-Coder-1.5B","CodeGemma-7B","CodeGemma-2B"]:
    ARCH[m] = "Decoder-only (FIM)"
for m in ["CodeT5p-16B","CodeT5p-6B","CodeT5p-2B","CodeT5-large","CodeT5-base","CodeT5-small"]:
    ARCH[m] = "Encoder-decoder"

JUDGES = ["Qwen2.5-7B-Instruct", "Qwen2.5-14B-Instruct", "Qwen2.5-32B-Instruct",
          "Mistral-Small-24B", "Gemma-2-27B-It"]
JUDGE_SHORT = {"Qwen2.5-7B-Instruct":"LJ_Q7", "Qwen2.5-14B-Instruct":"LJ_Q14",
               "Qwen2.5-32B-Instruct":"LJ_Q32", "Mistral-Small-24B":"LJ_M24",
               "Gemma-2-27B-It":"LJ_G27"}
_PAT = re.compile(r"^(?P<m>.+?)__judge_(?P<j>.+)__(?P<ts>\d{8}_\d{6})\.csv$")


def em_metrics(pred_path, dictionary):
    n = nc = 0
    em = emc = lev = nw = fuzzy = qual = 0.0
    for sid, gt, preds in M.read_pred_rows(pred_path):
        r = M.eval_sample(preds, gt, dictionary)
        n += 1
        em += r["em"]
        if r["consistent"] >= 1.0:
            nc += 1; emc += r["em"]; lev += r["lev_sim"]; nw += r["nw_sim"]
            fuzzy += r["subtok_fuzzy"]; qual += r["qual_char"]
    if not n: return None
    f = lambda x: x / nc if nc else 0.0
    return {"n": n, "consistency": nc/n, "em_gated": em/n, "em_consistent": f(emc),
            "lev": f(lev), "nw": f(nw), "fuzzy": f(fuzzy), "qual": f(qual)}


def lj_table(lj_dir):
    latest = {}
    for p in glob.glob(os.path.join(lj_dir, "*__judge_*__*.csv")):
        mm = _PAT.match(os.path.basename(p))
        if not mm: continue
        k = (mm["m"], mm["j"])
        if k not in latest or mm["ts"] > latest[k][0]: latest[k] = (mm["ts"], p)
    out = {}
    for (model, judge), (_, p) in latest.items():
        n = v1 = 0
        for r in csv.DictReader(open(p)):
            n += 1
            try:
                if int(float(r.get("llm_verdict", "-1"))) == 1: v1 += 1
            except ValueError: pass
        out.setdefault(model, {})[judge] = (v1/n if n else 0.0, n)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=os.path.join(os.path.dirname(_HERE), "results", "unified_refineID"))
    ap.add_argument("--dict", default="/usr/share/dict/words")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(_HERE), "figures", "new"))
    args = ap.parse_args()
    dictionary = M.load_dictionary(args.dict)
    lj = lj_table(os.path.join(args.results, "llm_judge"))

    rows = []
    for p in sorted(glob.glob(os.path.join(args.results, "predictions", "*.csv"))):
        model = os.path.basename(p)[:-4]
        if model.endswith("_colab") or model not in ARCH: continue
        em = em_metrics(p, dictionary)
        if not em: continue
        row = {"model": model, "arch": ARCH[model], **em}
        flags = []
        if em["n"] < 1000: flags.append(f"pred n={em['n']}")
        for j in JUDGES:
            cell = lj.get(model, {}).get(j)
            row[JUDGE_SHORT[j]] = cell[0] if cell else None
            if cell is None: flags.append(f"no {JUDGE_SHORT[j]}")
            elif cell[1] < em["n"]: flags.append(f"{JUDGE_SHORT[j]} n={cell[1]}")
        row["flags"] = "; ".join(flags)
        rows.append(row)

    # CSV
    cols = ["model","arch","n","em_gated","em_consistent","consistency","lev","nw",
            "fuzzy","qual"] + [JUDGE_SHORT[j] for j in JUDGES] + ["flags"]
    os.makedirs(args.out, exist_ok=True)
    csv_path = os.path.join(args.out, "leaderboard_full.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{r[k]:.4f}" if isinstance(r.get(k), float) else r.get(k, "")) for k in cols})

    # Markdown grouped by arch, sorted by em_gated within group
    def pct(x): return "--" if x is None else f"{x*100:.1f}"
    md = ["# Single-model leaderboard (Table II style)\n",
          "EM_strict = all-sites em_gated · EM_cons = EM on consistent subset · "
          "cons% = consistency rate · M1 lev/nw · M2 fuzzy · M3 qual · "
          "LJ_* = judge acceptance % (Q7/Q14/Q32 Qwen2.5, M24 Mistral-Small-24B, G27 Gemma-2-27B)\n"]
    hdr = ("| model | EM_strict | EM_cons | cons% | lev | nw | fuzzy | qual | "
           "LJ_Q7 | LJ_Q14 | LJ_Q32 | LJ_M24 | LJ_G27 | n |")
    sep = "|" + "---|" * 14
    for arch in ARCH_ORDER:
        grp = sorted([r for r in rows if r["arch"] == arch], key=lambda r: -r["em_gated"])
        if not grp: continue
        md.append(f"\n### {arch}\n\n{hdr}\n{sep}")
        for r in grp:
            md.append("| {m} | {eg} | {ec} | {c} | {lev} | {nw} | {fz} | {ql} | "
                      "{q7} | {q14} | {q32} | {m24} | {g27} | {n} |".format(
                m=r["model"], eg=pct(r["em_gated"]), ec=pct(r["em_consistent"]),
                c=pct(r["consistency"]), lev=pct(r["lev"]), nw=pct(r["nw"]),
                fz=pct(r["fuzzy"]), ql=pct(r["qual"]), q7=pct(r["LJ_Q7"]),
                q14=pct(r["LJ_Q14"]), q32=pct(r["LJ_Q32"]), m24=pct(r["LJ_M24"]),
                g27=pct(r["LJ_G27"]), n=r["n"]))
    md_path = os.path.join(args.out, "leaderboard_full.md")
    open(md_path, "w").write("\n".join(md) + "\n")

    # console: report data gaps
    print(f"models: {len(rows)}  ->  {md_path}\n         {csv_path}\n")
    gaps = [r for r in rows if r["flags"]]
    if gaps:
        print("DATA GAPS still present:")
        for r in gaps: print(f"  {r['model']:<24} {r['flags']}")
    else:
        print("No gaps: every model has 1000 predictions + all 5 judges.")
    print("\n" + "\n".join(md[3:]))


if __name__ == "__main__":
    main()
