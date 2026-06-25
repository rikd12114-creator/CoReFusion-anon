"""
LLM-as-a-judge evaluation for the full-size CodeT5+ family on RefineID.

Sibling of `detector/llm_naming_judge_codet5.py`. Same judge model, same
prompt, same parsing, same per-row CSV schema -- only the target benchmark
files differ. We score the 2B / 6B / 16B CodeT5+ checkpoints, which were
run via the prefix-only completion protocol (model-card pattern), so the
LJ column for them in Table~\\ref{tab:models} comes from this script.

Per-row CSV schema (one file per benchmark CSV evaluated):
    id, ground_truth, prediction, exact_match,
    relationship, explanation, decision, llm_acceptable, judge_raw

Usage
-----
    # Score the three full-size CodeT5+ result CSVs in one go (default).
    python detector/llm_naming_judge_codet5p_large.py \
        --result-file results/codeT5results/CodeT5p-16B_refineID_fim_20260509_134747.csv \
        --result-file "/home/user/Downloads/ar_fim_benchmark 5/CodeT5p-6B_refineID_fim_20260512_143508.csv" \
        --result-file "/home/user/Downloads/ar_fim_benchmark 5/CodeT5p-2B_refineID_fim_20260512_075946.csv" \
        --test-data data/test.csv

    # Or point at a directory and let it pick up all matching CSVs.
    python detector/llm_naming_judge_codet5p_large.py \
        --fim-dir "/home/user/Downloads/ar_fim_benchmark 5"

    # Quick smoke-test (first 20 rows per file).
    python detector/llm_naming_judge_codet5p_large.py --fim-dir ... --limit 20
"""

import os
import re
import csv
import sys
import gc
import glob
import time
import argparse
from datetime import datetime

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM


# ---- Configuration (matches Appendix app:llm_judge) ------------------------

JUDGE_MODEL_ID   = "Qwen/Qwen2.5-7B-Instruct"
JUDGE_DTYPE      = torch.bfloat16
GEN_TEMPERATURE  = 0.0
GEN_TOP_P        = 1.0
GEN_MAX_NEW_TOK  = 200

# Full-size CodeT5+ tags recognised in the FIM benchmark directory(ies).
CODET5P_LARGE_TAGS = ("CodeT5p-2B", "CodeT5p-6B", "CodeT5p-16B")


# ---- Prompt template (verbatim from Appendix app:llm_judge_prompt) ---------

PROMPT_TEMPLATE = """You are evaluating a code identifier-infilling task
(RefineID). A specific identifier (variable, method, or type name) was
**masked** in the original Java code, and a model predicted a replacement.
===================================================
GROUND TRUTH     : "{ground_truth}"
MODEL PREDICTION : "{prediction}"
===================================================
FULL CODE CONTEXT (the masked location is where the identifier should
appear):
```java
{full_code}
```
Evaluate the prediction by answering:
1. What is the relationship between the prediction and the ground truth
   in this exact context?
2. Would using the prediction instead of the ground truth preserve the
   program's correctness and readability?

Respond ONLY in this exact format (3 lines, no extra text):
Relationship: [Identical | Semantically Equivalent | Related but Different | Incorrect | Syntactically Invalid]
Explanation: [max 20 words]
Decision: [Accept | Reject]
"""

SYSTEM_PROMPT = "You are a strict code review assistant. Respond only in the requested 3-line format."

# Same truncation budget as the small-CodeT5 / AR judge runs so prompts
# stay comparable across the table.
CODE_CONTEXT_CHARS = 3000


# ---- Output parsing --------------------------------------------------------

_REL_RE = re.compile(r"^\s*Relationship\s*:\s*(.+?)\s*$", re.IGNORECASE | re.MULTILINE)
_EXP_RE = re.compile(r"^\s*Explanation\s*:\s*(.+?)\s*$",  re.IGNORECASE | re.MULTILINE)
_DEC_RE = re.compile(r"^\s*Decision\s*:\s*(Accept|Reject)\b", re.IGNORECASE | re.MULTILINE)


def parse_judge_output(text):
    """Extract (relationship, explanation, decision, accepted) from a judge reply.

    Per the appendix, anything other than an explicit `Accept` is treated
    as a rejection. Unparseable outputs therefore fail closed.
    """
    rel = _REL_RE.search(text)
    exp = _EXP_RE.search(text)
    dec = _DEC_RE.search(text)
    relationship = rel.group(1).strip() if rel else ""
    explanation  = exp.group(1).strip() if exp else ""
    decision     = dec.group(1).strip().capitalize() if dec else ""
    accepted     = (decision == "Accept")
    return relationship, explanation, decision, accepted


# ---- Data loading ----------------------------------------------------------

def load_test_contexts(test_csv_path):
    """Map id -> full masked Java snippet from data/test.csv."""
    print(f"Loading code contexts from {test_csv_path}...")
    csv.field_size_limit(sys.maxsize)
    id_to_code = {}
    with open(test_csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            try:
                rid = int(row[0])
            except ValueError:
                continue
            id_to_code[rid] = row[1]
    print(f"  loaded {len(id_to_code)} contexts")
    return id_to_code


def collect_codet5p_large_files(args):
    """Resolve the list of result CSVs to score."""
    files = []
    if args.result_file:
        files.extend(args.result_file)
    if args.fim_dir:
        for d in args.fim_dir:
            for path in sorted(glob.glob(os.path.join(d, "*.csv"))):
                base = os.path.basename(path)
                if base.startswith("summary_"):
                    continue
                if any(tag in base for tag in CODET5P_LARGE_TAGS):
                    files.append(path)
    # de-dup, preserve order
    seen = set()
    unique = []
    for p in files:
        ap = os.path.abspath(p)
        if ap not in seen:
            seen.add(ap)
            unique.append(ap)
    return unique


# ---- Judge wrapper ---------------------------------------------------------

class QwenJudge:
    """Thin wrapper around Qwen2.5-7B-Instruct chat generation."""

    def __init__(self, model_id=JUDGE_MODEL_ID, dtype=JUDGE_DTYPE):
        print(f"Loading judge model {model_id}...")
        t0 = time.time()
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=dtype,
            device_map="auto",
            trust_remote_code=True,
        )
        self.model.eval()
        print(f"  loaded in {time.time() - t0:.1f}s "
              f"(device map: {getattr(self.model, 'hf_device_map', 'auto')})")

    @torch.no_grad()
    def judge(self, ground_truth, prediction, code_context):
        ctx = code_context if len(code_context) <= CODE_CONTEXT_CHARS \
              else code_context[:CODE_CONTEXT_CHARS]
        user_prompt = PROMPT_TEMPLATE.format(
            ground_truth=ground_truth,
            prediction=prediction,
            full_code=ctx,
        )
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_prompt},
        ]
        chat_text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(chat_text, return_tensors="pt",
                                truncation=True, max_length=8192).to(self.model.device)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=GEN_MAX_NEW_TOK,
            do_sample=False,         # temperature=0
            top_p=GEN_TOP_P,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        gen_ids = outputs[0][inputs.input_ids.shape[1]:]
        return self.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()


# ---- Per-file scoring ------------------------------------------------------

def score_file(file_path, id_to_code, judge, output_dir, limit=None):
    base = os.path.basename(file_path)
    print(f"\n=== {base} ===")
    df = pd.read_csv(file_path)

    if limit:
        df = df.head(limit)

    rows = []
    n_em = 0
    n_lj = 0
    n_total = len(df)

    for _, r in tqdm(df.iterrows(), total=n_total, desc=f"  judging {base}"):
        rid = int(r["id"])
        gt  = "" if pd.isna(r["ground_truth"]) else str(r["ground_truth"]).strip()
        pred = "" if pd.isna(r["prediction"]) else str(r["prediction"]).strip()

        exact_match = (gt == pred and gt != "")
        if exact_match:
            n_em += 1
            n_lj += 1
            rows.append({
                "id": rid,
                "ground_truth": gt,
                "prediction": pred,
                "exact_match": True,
                "relationship": "Identical",
                "explanation": "Exact lexical match.",
                "decision": "Accept",
                "llm_acceptable": True,
                "judge_raw": "",
            })
            continue

        ctx = id_to_code.get(rid, "")
        if not ctx:
            # No context available -- the judge cannot decide; record as Reject.
            rows.append({
                "id": rid, "ground_truth": gt, "prediction": pred,
                "exact_match": False, "relationship": "",
                "explanation": "Context missing.", "decision": "Reject",
                "llm_acceptable": False, "judge_raw": "",
            })
            continue

        try:
            raw = judge.judge(gt, pred, ctx)
            rel, exp, dec, acc = parse_judge_output(raw)
        except Exception as e:
            raw = f"<judge-error: {e}>"
            rel, exp, dec, acc = "", str(e), "Reject", False

        if acc:
            n_lj += 1

        rows.append({
            "id": rid,
            "ground_truth": gt,
            "prediction": pred,
            "exact_match": False,
            "relationship": rel,
            "explanation": exp,
            "decision": dec,
            "llm_acceptable": acc,
            "judge_raw": raw,
        })

    out_df = pd.DataFrame(rows, columns=[
        "id", "ground_truth", "prediction", "exact_match",
        "relationship", "explanation", "decision", "llm_acceptable", "judge_raw",
    ])
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"judged_{base}")
    out_df.to_csv(out_path, index=False)

    em_pct = 100.0 * n_em / n_total if n_total else 0.0
    lj_pct = 100.0 * n_lj / n_total if n_total else 0.0
    print(f"  EM = {em_pct:.2f}%   LJ = {lj_pct:.2f}%   ({n_em}/{n_lj}/{n_total})")
    print(f"  saved -> {out_path}")
    return {
        "results_file": file_path,
        "judged_file":  out_path,
        "n":            n_total,
        "exact_match":  n_em,
        "em_pct":       round(em_pct, 4),
        "llm_accept":   n_lj,
        "lj_pct":       round(lj_pct, 4),
    }


# ---- Main ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Run Qwen2.5-7B-Instruct LJ on full-size CodeT5+ RefineID results."
    )
    parser.add_argument("--test-data", default="data/test.csv",
                        help="Path to RefineID test.csv (id, masked_code, target).")
    parser.add_argument("--fim-dir", action="append", default=None,
                        help="Directory of *_refineID_fim_*.csv result files; "
                             "repeatable. All matching CodeT5p-{2B,6B,16B} CSVs "
                             "are picked up.")
    parser.add_argument("--result-file", action="append", default=None,
                        help="Explicit per-model result CSV(s) to score (repeatable).")
    parser.add_argument("--output-dir", default="results/llm_judge_codet5p_large",
                        help="Where to write judged_*.csv and the LJ summary.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Smoke-test: only judge the first N rows of each file.")
    args = parser.parse_args()

    files = collect_codet5p_large_files(args)
    if not files:
        print("ERROR: no result files matched. Pass --fim-dir or --result-file.")
        print(f"  recognised tags: {CODET5P_LARGE_TAGS}")
        sys.exit(1)

    print(f"Will judge {len(files)} file(s):")
    for f in files:
        print(f"  - {f}")

    id_to_code = load_test_contexts(args.test_data)
    judge = QwenJudge()

    summaries = []
    for f in files:
        try:
            summaries.append(score_file(f, id_to_code, judge, args.output_dir, limit=args.limit))
        except Exception as e:
            print(f"  FAILED on {f}: {e}")
            summaries.append({
                "results_file": f, "judged_file": "", "n": 0,
                "exact_match": 0, "em_pct": 0.0,
                "llm_accept": 0, "lj_pct": 0.0,
            })

    # Summary table
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(args.output_dir, exist_ok=True)
    summary_path = os.path.join(args.output_dir, f"lj_summary_{timestamp}.csv")
    pd.DataFrame(summaries).to_csv(summary_path, index=False)

    print("\n" + "="*72)
    print("  LJ summary (Qwen2.5-7B-Instruct, temperature=0)")
    print("="*72)
    print(f"  {'file':<55} {'EM%':>6} {'LJ%':>6}")
    print("  " + "-"*70)
    for s in summaries:
        name = os.path.basename(s["results_file"])[:54]
        print(f"  {name:<55} {s['em_pct']:>6.2f} {s['lj_pct']:>6.2f}")
    print(f"\n  summary -> {summary_path}")

    # Cleanup
    del judge
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


if __name__ == "__main__":
    main()
