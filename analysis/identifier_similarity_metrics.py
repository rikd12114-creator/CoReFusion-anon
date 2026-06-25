"""
Identifier-name evaluation metrics for the refineID variable-naming task,
implemented from three papers, as a softer replacement for plain Exact Match.

Metrics (one family per paper)
------------------------------
M1  IdBench string-distance similarity   [Wainakh, Rauf, Pradel, ICSE 2021]
      lev_sim : 1 - Levenshtein(pred, gt) / max(len)          (their "LV")
      nw_sim  : normalised Needleman-Wunsch global-alignment   (their "NW")
                score (match +1 / mismatch -1 / gap -1).
      -> character-level lexical closeness of pred vs gt, in [0,1].

M2  Subtoken similarity                  [Wong et al., "Identifier Name
                                          Similarities", ESEM 2025]
      That paper is a 7-category *taxonomy* of name similarity (no formula);
      its categories are all built on shared sub-tokens / words (abbreviated,
      derivational, type-descriptive, ... variants). We operationalise it as
      sub-token overlap after splitting camelCase / snake_case / digits:
      subtok_jaccard : |A n B| / |A u B| on the sub-token SETS  (headline)
      subtok_fuzzy   : same, but two sub-tokens also match when one is a
                       >=3-char prefix of the other or char-sim >= 0.8
                       (captures Abbreviated / Derivational variants).
      -> word/concept-level closeness of pred vs gt, in [0,1].

M3  Identifier quality                   [Lawrie, Feild, Binkley,
                                          "Quantifying identifier quality",
                                          Empir. Softw. Eng. 2007]
      Split into hard words then soft words; quality = fraction of the
      identifier that is made of real dictionary words / known abbreviations
      (their "on a list" notion). This scores the prediction ITSELF (no gt) --
      i.e. is the produced name a readable, meaningful identifier vs cryptic?
      qual_char : chars covered by on-list words / total chars   (headline)
      qual_word : on-list soft words / total soft words
      -> intrinsic readability of pred, in [0,1].

Consistency is a HARD GATE (not an average)
-------------------------------------------
Every [MASK] in a refineID sample is the SAME variable, so the model must emit
the IDENTICAL identifier at every site -- otherwise the renamed code does not
compile and the output is worthless. We therefore:

  1. Gate on consistency: a sample is USABLE only if every site produced the
     same non-empty identifier (``consistent``). Inconsistent / empty samples
     do not compile -> they are failures, never partially credited.
  2. Only on the usable (consistent) samples do we score the single regenerated
     identifier with M1/M2/M3 against the ground truth.

We report two views:
  * "consistent subset": mean metric over the samples that actually compile
    (quality of the regenerated name, given it is consistent).
  * "gated / all": mean over ALL samples with inconsistent ones scored 0
    (usable quality across the whole test set). ``em_gated`` == consistent AND
    name correct == the strict all-sites EM.

Usage:
    python analysis/identifier_similarity_metrics.py \
        --input results/dreamon/DreamOn-7B_per_sample_20260509_134411.csv
    # several files (per-model comparison table):
    python analysis/identifier_similarity_metrics.py --input a.csv --input b.csv
    # self-test on toy pairs:
    python analysis/identifier_similarity_metrics.py --selftest
"""

import os
import re
import csv
import sys
import argparse
from collections import Counter

csv.field_size_limit(2**31 - 1)


# ---------------------------------------------------------------------------
# Tokenisation
# ---------------------------------------------------------------------------

_SPLIT_RE = re.compile(r'[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|[0-9]+')


def split_subtokens(name):
    """camelCase / snake_case / digits -> lowercased sub-token list.
    'maxEscapedBytes' -> [max, escaped, bytes]; 'COUNT_2' -> [count, 2]."""
    if not name:
        return []
    return [t.lower() for t in _SPLIT_RE.findall(name)]


def split_hard_words(name):
    """Split only on explicit/camel markers into hard words (keep digits)."""
    if not name:
        return []
    return _SPLIT_RE.findall(name)


# ---------------------------------------------------------------------------
# M1 - IdBench string distance functions
# ---------------------------------------------------------------------------

def levenshtein(a, b):
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def lev_sim(a, b):
    if not a and not b:
        return 1.0
    m = max(len(a), len(b))
    return 1.0 - levenshtein(a, b) / m if m else 1.0


def nw_sim(a, b, match=1, mismatch=-1, gap=-1):
    """Needleman-Wunsch global alignment score, normalised to [0,1]."""
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    n, m = len(a), len(b)
    prev = [k * gap for k in range(m + 1)]
    for i in range(1, n + 1):
        cur = [i * gap]
        ai = a[i - 1]
        for j in range(1, m + 1):
            diag = prev[j - 1] + (match if ai == b[j - 1] else mismatch)
            cur.append(max(diag, prev[j] + gap, cur[j - 1] + gap))
        prev = cur
    score = prev[-1]
    L = max(n, m)
    return max(0.0, (score + L) / (2.0 * L))   # all-match -> 1, worst -> 0


def metric_idbench(pred, gt):
    return {"lev_sim": lev_sim(pred, gt), "nw_sim": nw_sim(pred, gt)}


# ---------------------------------------------------------------------------
# M2 - sub-token similarity (Exploratory Study taxonomy, operationalised)
# ---------------------------------------------------------------------------

def _fuzzy_match(x, y):
    if x == y:
        return True
    if len(x) >= 3 and len(y) >= 3 and (x.startswith(y) or y.startswith(x)):
        return True
    return lev_sim(x, y) >= 0.8


def subtok_jaccard(pred, gt):
    A, B = set(split_subtokens(pred)), set(split_subtokens(gt))
    if not A and not B:
        return 1.0
    if not A or not B:
        return 0.0
    return len(A & B) / len(A | B)


def subtok_fuzzy(pred, gt):
    A, B = split_subtokens(pred), split_subtokens(gt)
    if not A and not B:
        return 1.0
    if not A or not B:
        return 0.0
    sA, sB = set(A), set(B)
    inter = 0
    usedB = set()
    for x in sA:
        for y in sB:
            if y not in usedB and _fuzzy_match(x, y):
                inter += 1
                usedB.add(y)
                break
    union = len(sA) + len(sB) - inter
    return inter / union if union else 1.0


def metric_subtoken(pred, gt):
    return {"subtok_jaccard": subtok_jaccard(pred, gt),
            "subtok_fuzzy": subtok_fuzzy(pred, gt)}


# ---------------------------------------------------------------------------
# M3 - identifier quality (Lawrie/Feild/Binkley)
# ---------------------------------------------------------------------------

# Small, frozen list of well-known programming/domain abbreviations
# (the paper used ~200; this is a representative subset).
ABBREVIATIONS = {
    "buf", "buff", "ptr", "idx", "len", "str", "num", "msg", "tmp", "temp",
    "val", "var", "obj", "ctx", "cfg", "conf", "config", "init", "impl",
    "env", "arg", "args", "cmd", "db", "conn", "req", "res", "resp", "err",
    "char", "int", "bool", "dir", "dest", "src", "pos", "prev", "cur", "curr",
    "next", "min", "max", "avg", "sum", "cnt", "count", "id", "ids", "info",
    "btn", "img", "doc", "elem", "attr", "param", "params", "func", "fn",
    "ref", "addr", "byte", "bytes", "bit", "hex", "dec", "bin", "regex",
    "iter", "idx", "lst", "arr", "vec", "mat", "col", "cols", "row", "rows",
    "win", "ui", "io", "os", "url", "uri", "uuid", "json", "xml", "html",
    "css", "sql", "api", "cpu", "gpu", "ram", "kb", "mb", "gb", "px",
}

_DICT_CACHE = {}


def load_dictionary(path="/usr/share/dict/words"):
    if path in _DICT_CACHE:
        return _DICT_CACHE[path]
    words = set()
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                w = line.strip().lower()
                if len(w) >= 2 and w.isalpha():
                    words.add(w)
    except FileNotFoundError:
        print(f"  [warn] dictionary {path} not found; quality uses abbrevs only.")
    _DICT_CACHE[path] = words
    return words


def _on_list(word, dictionary):
    w = word.lower()
    return w in ABBREVIATIONS or (len(w) >= 2 and w in dictionary)


def _max_covered_chars(word, dictionary):
    """DP: max #chars of a (letters-only) hard word coverable by on-list words."""
    w = word.lower()
    n = len(w)
    dp = [0] * (n + 1)
    for i in range(1, n + 1):
        dp[i] = dp[i - 1]                       # leave char i-1 uncovered
        for j in range(i - 1, -1, -1):
            if _on_list(w[j:i], dictionary):
                dp[i] = max(dp[i], dp[j] + (i - j))
    return dp[n]


def identifier_quality(name, dictionary):
    """Return (qual_char, qual_word) in [0,1]."""
    hard = split_hard_words(name)
    if not hard:
        return 0.0, 0.0
    total_chars = covered_chars = 0
    soft_total = soft_on = 0
    for hw in hard:
        total_chars += len(hw)
        if hw.isdigit():
            soft_total += 1                     # digit run = 1 "other" soft word
            continue
        covered_chars += _max_covered_chars(hw, dictionary)
        # word-level: is the hard word (or its segments) on a list?
        if _on_list(hw, dictionary):
            soft_total += 1
            soft_on += 1
        else:
            # count covered fraction as partial soft-word credit
            cov = _max_covered_chars(hw, dictionary)
            soft_total += 1
            if cov >= max(2, 0.6 * len(hw)):
                soft_on += 1
    qual_char = covered_chars / total_chars if total_chars else 0.0
    qual_word = soft_on / soft_total if soft_total else 0.0
    return qual_char, qual_word


def metric_quality(pred, dictionary):
    qc, qw = identifier_quality(pred, dictionary)
    return {"qual_char": qc, "qual_word": qw}


# ---------------------------------------------------------------------------
# Per-sample evaluation with all-sites aggregation
# ---------------------------------------------------------------------------

METRIC_KEYS = ["lev_sim", "nw_sim", "subtok_jaccard", "subtok_fuzzy",
               "qual_char", "qual_word"]


def eval_sample(preds, gt, dictionary):
    """preds: list of per-site predictions (empties kept); gt: ground truth.

    Consistency gate first: the renamed code only compiles if EVERY site emits
    the same non-empty identifier. Metrics are computed only for that single
    agreed name; inconsistent samples are failures (metrics = 0)."""
    n_sites = len(preds)
    consistent = (n_sites > 0 and preds[0] != "" and len(set(preds)) == 1)

    row = {"n_sites": n_sites, "consistent": float(consistent)}
    if consistent:
        agreed = preds[0]
        m = {}
        m.update(metric_idbench(agreed, gt))
        m.update(metric_subtoken(agreed, gt))
        m.update(metric_quality(agreed, dictionary))
        for k in METRIC_KEYS:
            row[k] = m[k]
        row["em"] = float(agreed == gt)
        row["agreed_pred"] = agreed
    else:
        for k in METRIC_KEYS:
            row[k] = 0.0
        row["em"] = 0.0
        row["agreed_pred"] = ""
    row["gt_qual_char"] = metric_quality(gt, dictionary)["qual_char"]
    return row


def read_pred_rows(path):
    """Yield (id, gt, [per-site preds]) from a per_sample/all_predictions CSV."""
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        cols = reader.fieldnames or []
        pred_col = "predictions" if "predictions" in cols else (
            "all_predictions" if "all_predictions" in cols else None)
        for r in reader:
            gt = (r.get("ground_truth") or "").strip()
            if pred_col:
                raw = r.get(pred_col) or ""
                # keep empties: a missing site means the code won't compile
                preds = [p.strip() for p in raw.split("|")] if raw else [""]
            else:
                preds = [(r.get("prediction") or "").strip()]
            yield r.get("id", ""), gt, preds


def evaluate_csv(path, dictionary, max_samples=None):
    rows = []
    for i, (sid, gt, preds) in enumerate(read_pred_rows(path)):
        if max_samples is not None and i >= max_samples:
            break
        r = eval_sample(preds, gt, dictionary)
        r["id"] = sid
        rows.append(r)
    if not rows:
        return None

    n = len(rows)
    usable = [r for r in rows if r["consistent"] == 1.0]
    n_usable = len(usable)

    summary = {
        "file": os.path.basename(path),
        "n": n,
        "n_consistent": n_usable,
        "consistency_rate": n_usable / n,
        "gt_qual_char": sum(r["gt_qual_char"] for r in rows) / n,
    }
    # (A) consistent subset: quality of the regenerated identifier
    for k in ["em"] + METRIC_KEYS:
        summary[f"{k}_consistent"] = (sum(r[k] for r in usable) / n_usable) if n_usable else 0.0
    # (B) gated over ALL samples: inconsistent (won't compile) scored 0
    for k in ["em"] + METRIC_KEYS:
        summary[f"{k}_gated"] = sum(r[k] for r in rows) / n
    return summary, rows


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_summary(summaries):
    print(f"\n{'='*100}")
    print("  (A) CONSISTENT SUBSET  -- quality of the regenerated identifier among samples that COMPILE")
    print(f"{'='*100}")
    print(f"{'file':<46}{'consist%':>9}{'n_ok':>6} | {'EM':>6}{'lev':>6}{'nw':>6}"
          f"{'jacc':>6}{'fuzzy':>6}{'qualC':>7}{'gtQ':>6}")
    print("  " + "-" * 96)
    for s in summaries:
        print(f"{s['file'][:46]:<46}{s['consistency_rate']*100:>8.1f}%{s['n_consistent']:>6} | "
              f"{s['em_consistent']:>6.3f}{s['lev_sim_consistent']:>6.3f}{s['nw_sim_consistent']:>6.3f}"
              f"{s['subtok_jaccard_consistent']:>6.3f}{s['subtok_fuzzy_consistent']:>6.3f}"
              f"{s['qual_char_consistent']:>7.3f}{s['gt_qual_char']:>6.3f}")

    print(f"\n{'='*100}")
    print("  (B) GATED OVER ALL SAMPLES -- inconsistent (non-compiling) scored 0   [em_gated == strict all-sites EM]")
    print(f"{'='*100}")
    print(f"{'file':<46}{'EM_gated':>9}{'lev':>7}{'nw':>7}{'jacc':>7}{'fuzzy':>7}{'qualC':>8}")
    print("  " + "-" * 96)
    for s in summaries:
        print(f"{s['file'][:46]:<46}{s['em_gated']:>9.3f}{s['lev_sim_gated']:>7.3f}"
              f"{s['nw_sim_gated']:>7.3f}{s['subtok_jaccard_gated']:>7.3f}"
              f"{s['subtok_fuzzy_gated']:>7.3f}{s['qual_char_gated']:>8.3f}")
    print(f"\n  Legend: consist%=fraction all-sites-consistent (compilable) · n_ok=#consistent samples ·"
          f"\n  lev/nw=IdBench(M1) · jacc/fuzzy=subtoken(M2) · qualC=quality(M3) · gtQ=quality of ground truth")


def selftest():
    d = load_dictionary()
    pairs = [("bufferSize", "bufSize"), ("size", "length"), ("count", "total"),
             ("isTrackExcluded", "canExclude"), ("decodedCapacity", "maxEscapedBytes"),
             ("tmp", "temporaryBuffer"), ("x", "index"), ("getUserName", "getUserId"),
             ("foo123", "fooBar"), ("style", "style")]
    print(f"{'pred':<18}{'gt':<18}{'lev':>6}{'nw':>6}{'jacc':>6}{'fuzzy':>7}{'qcPred':>8}{'qcGt':>6}")
    for p, g in pairs:
        m = {}
        m.update(metric_idbench(p, g)); m.update(metric_subtoken(p, g))
        print(f"{p:<18}{g:<18}{m['lev_sim']:>6.2f}{m['nw_sim']:>6.2f}"
              f"{m['subtok_jaccard']:>6.2f}{m['subtok_fuzzy']:>7.2f}"
              f"{identifier_quality(p,d)[0]:>8.2f}{identifier_quality(g,d)[0]:>6.2f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", action="append", default=None,
                    help="per_sample / all_predictions CSV (repeatable).")
    ap.add_argument("--dict", default="/usr/share/dict/words")
    ap.add_argument("--max-samples", type=int, default=None)
    ap.add_argument("--out", default=None, help="write per-sample metric CSV here.")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        selftest()
        return
    if not args.input:
        ap.error("give --input <csv> (repeatable), or --selftest")

    dictionary = load_dictionary(args.dict)
    print(f"  dictionary: {len(dictionary)} words + {len(ABBREVIATIONS)} abbreviations")

    summaries = []
    for path in args.input:
        if not os.path.exists(path):
            print(f"  [skip] missing: {path}")
            continue
        res = evaluate_csv(path, dictionary, max_samples=args.max_samples)
        if res is None:
            print(f"  [skip] no rows: {path}")
            continue
        summary, rows = res
        summaries.append(summary)
        if args.out and len(args.input) == 1:
            keys = list(rows[0].keys())
            with open(args.out, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=keys)
                w.writeheader()
                w.writerows(rows)
            print(f"  wrote per-sample metrics -> {args.out}")

    if summaries:
        print_summary(summaries)


if __name__ == "__main__":
    main()
