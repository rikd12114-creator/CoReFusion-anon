"""Metrics for the multi-identifier / multi-site rename benchmark.

Everything operates on two dicts keyed by placeholder (IDENT_0, IDENT_1, ...):
    gt:   {placeholder -> original identifier}
    pred: {placeholder -> model-predicted identifier (may be missing/empty)}
"""
import difflib
import re


def split_subtokens(name):
    """Split camelCase / snake_case / digits into lowercase sub-tokens."""
    if not name:
        return []
    parts = re.split(r"[_\W]+", name)
    out = []
    for p in parts:
        # camelCase + ACRONYMBoundary + trailing digits
        for m in re.findall(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|\d+", p):
            out.append(m.lower())
    return out


def edit_similarity(a, b):
    """Normalized similarity in [0,1] (1.0 == identical)."""
    if not a and not b:
        return 1.0
    return difflib.SequenceMatcher(None, a or "", b or "").ratio()


def subtoken_f1(gt_name, pred_name):
    g = split_subtokens(gt_name)
    p = split_subtokens(pred_name)
    if not g and not p:
        return 1.0
    if not g or not p:
        return 0.0
    gs, ps = set(g), set(p)
    inter = len(gs & ps)
    if inter == 0:
        return 0.0
    prec = inter / len(ps)
    rec = inter / len(gs)
    return 2 * prec * rec / (prec + rec)


def score_instance(gt, pred):
    """Score one masked file. Returns a dict of aggregate + per-placeholder info."""
    per = {}
    em_hits, emci_hits, edit_sum, f1_sum, covered = 0, 0, 0.0, 0.0, 0
    for ph, gold in gt.items():
        guess = (pred or {}).get(ph, "") or ""
        guess = guess.strip()
        is_em = guess == gold
        is_emci = guess.lower() == gold.lower()
        es = edit_similarity(gold, guess)
        f1 = subtoken_f1(gold, guess)
        per[ph] = {
            "gold": gold, "pred": guess,
            "em": is_em, "em_ci": is_emci,
            "edit_sim": round(es, 4), "subtoken_f1": round(f1, 4),
        }
        em_hits += int(is_em)
        emci_hits += int(is_emci)
        edit_sum += es
        f1_sum += f1
        covered += int(bool(guess))
    k = len(gt) or 1
    return {
        "per_identifier": per,
        "n_identifiers": len(gt),
        "em": em_hits / k,
        "em_ci": emci_hits / k,
        "edit_sim": edit_sum / k,
        "subtoken_f1": f1_sum / k,
        "coverage": covered / k,
        "joint_correct": em_hits == len(gt) and len(gt) > 0,
    }


def aggregate(instance_scores):
    """Aggregate per-instance scores into a single summary dict."""
    n = len(instance_scores)
    if n == 0:
        return {"n_instances": 0}
    total_ids = sum(s["n_identifiers"] for s in instance_scores)

    def micro(field):
        # weight by number of identifiers (per-identifier mean)
        return sum(s[field] * s["n_identifiers"] for s in instance_scores) / max(total_ids, 1)

    return {
        "n_instances": n,
        "n_identifiers": total_ids,
        "em": round(micro("em"), 4),
        "em_ci": round(micro("em_ci"), 4),
        "edit_sim": round(micro("edit_sim"), 4),
        "subtoken_f1": round(micro("subtoken_f1"), 4),
        "coverage": round(micro("coverage"), 4),
        "joint_acc": round(sum(int(s["joint_correct"]) for s in instance_scores) / n, 4),
    }
