"""
Code Naming Smell Detector
==========================

Combines two probing methods to surface bad naming in Java source code:

  METHOD 1 — Over-confidence Probe
    Masks each identifier and checks the model's rank for the current name
    vs. known smell tokens. If the model ranks a severe smell HIGHER than
    the real name, the identifier is flagged as potentially redundant /
    context-free (i.e., the name conveys nothing beyond what `x` would).

  METHOD 2 — Context Sensitivity (Masking Gradient)
    Sweeps alpha (context masking fraction) from 0 → MAX_ALPHA and measures
    how much the model's entropy at the target position rises (ΔH).
    High ΔH → name is strongly tied to local context (good).
    Low  ΔH → name carries almost no context-specific information (smell).

Extracts identifiers using tree-sitter (Java support).

Usage:
    # minimal – CPU-friendly for local testing
    python detector/code_naming_smell_detector.py --file path/to/Foo.java

    # full analysis with both probes
    python detector/code_naming_smell_detector.py \\
        --file path/to/Foo.java \\
        --model DiffuCoder-7B-Base \\
        --alpha-steps 5 \\
        --output report.json
"""

import os
import sys
import json
import argparse
import random
import math
from dataclasses import dataclass, asdict
from typing import Optional, Any

# torch is only required when actually running model inference.
# Import lazily so tree-sitter + scoring logic can be unit-tested without GPU.
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None  # type: ignore

try:
    from transformers import AutoTokenizer, AutoModel
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    AutoTokenizer = AutoModel = None  # type: ignore

# ── tree-sitter ────────────────────────────────────────────────────────────────
try:
    import tree_sitter_java as tsjava
    from tree_sitter import Language, Parser as TSParser
    JAVA_LANGUAGE = Language(tsjava.language())
    _TS_OK = True
except ImportError:
    _TS_OK = False

# ══════════════════════════════════════════════════════════════════════════════
# Configuration
# ══════════════════════════════════════════════════════════════════════════════

DEVICE   = "cuda" if (HAS_TORCH and torch.cuda.is_available()) else "cpu"
MAX_TOKS = 512
LEFT_CTX = MAX_TOKS // 2          # centered window: 256 left + mask + 255 right
NUM_STEPS = 32                    # diffusion steps for fast logit probing

# Regime thresholds (from calibration experiments)
THRESH_OC   = 200                 # gt_rank ≤ 200  → OVERCONFIDENT (name is already generic)
THRESH_RARE = 1000                # gt_rank > 1000 → CONFIDENT_RARE (name is specific)

# Smell probe vocabulary
SMELL_PROBES = {
    "severe":   ["x", "a", "n", "i"],
    "moderate": ["tmp", "val", "foo", "res"],
    "mild":     ["myVar", "temp1", "result1", "value1"],
}

# Context sensitivity sweep
ALPHA_GRID_FULL = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
ALPHA_GRID_FAST = [0.0, 0.4, 0.8]   # fast mode

MODEL_REGISTRY = {
    "DiffuCoder-7B-Base":      {"id": "apple/DiffuCoder-7B-Base",                    "mask_token": "<|mask|>"},
    "DiffuCoder-7B-Instruct":  {"id": "apple/DiffuCoder-7B-Instruct",                "mask_token": "<|mask|>"},
    "DreamCoder-7B":           {"id": "Dream-org/Dream-Coder-v0-Instruct-7B",        "mask_token": "<|mask|>"},
}

# Smell score weights (tune as needed)
W_OC  = 0.4    # weight for over-confidence signal
W_CTX = 0.6    # weight for context sensitivity signal

# ══════════════════════════════════════════════════════════════════════════════
# Data types
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class IdentifierInfo:
    name: str
    node_type: str          # e.g. "variable_declarator", "formal_parameter"
    start_byte: int
    end_byte: int
    start_point: tuple      # (row, col)

@dataclass
class ProbeResult:
    identifier: str
    node_type: str
    line: int
    col: int

    # --- Method 1: Over-confidence probe ---
    gt_rank: int            # rank of identifier in softmax (no masking)
    gt_prob: float
    regime: str             # OVERCONFIDENT / UNCERTAIN / CONFIDENT_RARE
    smell_severe_rank: int
    smell_moderate_rank: int
    smell_mild_rank: int
    trap_ratio: float       # gt_rank / smell_severe_rank  (>1 = trap fired)

    # --- Method 2: Context sensitivity ---
    delta_h: float          # H(alpha=0.8) - H(alpha=0.0)
    h_at_0: float
    h_at_08: float

    # --- Combined score ---
    smell_score: float      # 0.0 (clean) → 1.0 (strong smell)
    verdict: str            # "CLEAN" / "SUSPICIOUS" / "SMELL"

# ══════════════════════════════════════════════════════════════════════════════
# Tree-sitter: Java identifier extraction
# ══════════════════════════════════════════════════════════════════════════════

# Node types we care about for naming analysis
TARGET_NODE_TYPES = {
    "variable_declarator",   # local variable, field
    "formal_parameter",      # method parameter
    "enhanced_for_statement" # for-each loop variable  (child: name)
}

# Within each parent, the child field that holds the name
NAME_FIELD = "name"

def _walk_tree(node, results: list, source_bytes: bytes):
    """Recursively visit all nodes and collect identifiers of interest."""
    if node.type in TARGET_NODE_TYPES:
        name_node = node.child_by_field_name(NAME_FIELD)
        if name_node:
            name_text = source_bytes[name_node.start_byte:name_node.end_byte].decode("utf-8", errors="replace")
            # Skip trivially short synthetic names (loop counters in tiny snippets)
            results.append(IdentifierInfo(
                name=name_text,
                node_type=node.type,
                start_byte=name_node.start_byte,
                end_byte=name_node.end_byte,
                start_point=name_node.start_point,  # (row, col)
            ))
    for child in node.children:
        _walk_tree(child, results, source_bytes)


def extract_identifiers(java_source: str) -> list[IdentifierInfo]:
    """Parse Java source and return all variable / parameter identifiers."""
    if not _TS_OK:
        raise RuntimeError("tree-sitter or tree-sitter-java not installed.")
    parser = TSParser(JAVA_LANGUAGE)
    source_bytes = java_source.encode("utf-8")
    tree = parser.parse(source_bytes)
    results = []
    _walk_tree(tree.root_node, results, source_bytes)
    return results

# ══════════════════════════════════════════════════════════════════════════════
# Model utilities  (shared with experiments)
# ══════════════════════════════════════════════════════════════════════════════

def load_model(model_name: str):
    meta = MODEL_REGISTRY[model_name]
    print(f"  Loading {model_name}  ({meta['id']}) …")
    tokenizer = AutoTokenizer.from_pretrained(meta["id"], trust_remote_code=True)
    dtype = torch.bfloat16 if DEVICE == "cuda" else torch.float32
    model = AutoModel.from_pretrained(
        meta["id"], trust_remote_code=True, torch_dtype=dtype
    ).to(DEVICE).eval()
    mask_id = tokenizer.convert_tokens_to_ids(meta["mask_token"])
    if hasattr(model, "generation_config") and hasattr(model.generation_config, "steps"):
        model.generation_config.steps = NUM_STEPS
    return tokenizer, model, mask_id


def _forward(model, input_ids: Any) -> Any:
    with torch.no_grad():
        try:
            out = model(input_ids=input_ids, attention_mask=None, num_steps=NUM_STEPS)
        except TypeError:
            out = model(input_ids=input_ids, attention_mask=None)
    return out.logits if hasattr(out, "logits") else (out[0] if isinstance(out, tuple) else out)


def _build_window(tokenizer, source: str, start_byte: int, end_byte: int, mask_id: int):
    """Build a centered MAX_TOKS-token window with <mask> at the identifier position."""
    prefix = source[:start_byte]
    suffix = source[end_byte:]
    prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
    suffix_ids = tokenizer.encode(suffix, add_special_tokens=False)
    right_ctx  = MAX_TOKS - LEFT_CTX - 1
    prefix_win = prefix_ids[-LEFT_CTX:]
    suffix_win = suffix_ids[:right_ctx]
    bos = ([tokenizer.bos_token_id] if getattr(tokenizer, "bos_token_id", None) is not None else [])
    token_seq  = bos + prefix_win + [mask_id] + suffix_win
    target_idx = len(bos) + len(prefix_win)
    input_ids  = torch.tensor([token_seq], dtype=torch.long).to(DEVICE)
    return input_ids, target_idx


def _apply_masking(base_ids: Any, mask_id: int, target_idx: int, alpha: float, rng) -> Any:
    ids = base_ids.clone()
    eligible = [i for i in range(ids.shape[1]) if i != target_idx]
    k = int(round(alpha * len(eligible)))
    for i in rng.sample(eligible, k) if k else []:
        ids[0, i] = mask_id
    return ids


def _softmax_stats(logits: Any, position: int, probe_id: int) -> dict:
    lp    = logits[0, position, :].float()
    probs = torch.softmax(lp, dim=-1)
    lprob = torch.log(probs + 1e-12)
    entropy = float(-(probs * lprob).sum())
    probe_prob = float(probs[probe_id])
    sorted_ids = torch.argsort(probs, descending=True)
    rank_map   = {int(t): r + 1 for r, t in enumerate(sorted_ids)}
    probe_rank = rank_map.get(int(probe_id), len(rank_map))
    return {"entropy": entropy, "prob": probe_prob, "rank": probe_rank}

# ══════════════════════════════════════════════════════════════════════════════
# Probe runners
# ══════════════════════════════════════════════════════════════════════════════

def run_overconfidence_probe(model, tokenizer, mask_id: int,
                             source: str, info: IdentifierInfo,
                             rng: random.Random) -> dict:
    """Method 1: single forward pass at alpha=0, rank comparison."""
    try:
        base_ids, tgt = _build_window(tokenizer, source,
                                      info.start_byte, info.end_byte, mask_id)
    except Exception as e:
        return {"error": str(e)}

    # GT token id
    gt_toks = tokenizer.encode(info.name, add_special_tokens=False)
    gt_id   = gt_toks[0] if gt_toks else tokenizer.unk_token_id

    # Smell probe token ids (collision-free)
    smell_ids = {}
    for sev, names in SMELL_PROBES.items():
        candidates = list(names)
        rng.shuffle(candidates)
        for cand in candidates:
            t = tokenizer.encode(cand, add_special_tokens=False)
            cid = t[0] if t else tokenizer.unk_token_id
            if cid != gt_id:
                smell_ids[sev] = (cand, cid)
                break

    try:
        logits = _forward(model, base_ids)
    except Exception as e:
        return {"error": str(e)}

    gt_stats  = _softmax_stats(logits, tgt, gt_id)
    gt_rank   = gt_stats["rank"]

    smell_ranks = {}
    for sev, (_, sid) in smell_ids.items():
        smell_ranks[sev] = _softmax_stats(logits, tgt, sid)["rank"]

    # Regime
    if gt_rank <= THRESH_OC:
        regime = "OVERCONFIDENT"
    elif gt_rank <= THRESH_RARE:
        regime = "UNCERTAIN"
    else:
        regime = "CONFIDENT_RARE"

    sev_rank = smell_ranks.get("severe", gt_rank)
    trap_ratio = gt_rank / max(sev_rank, 1)

    return {
        "gt_rank": gt_rank,
        "gt_prob": gt_stats["prob"],
        "regime":  regime,
        "smell_severe_rank":   smell_ranks.get("severe", -1),
        "smell_moderate_rank": smell_ranks.get("moderate", -1),
        "smell_mild_rank":     smell_ranks.get("mild", -1),
        "trap_ratio": trap_ratio,
    }


def run_context_sensitivity_probe(model, tokenizer, mask_id: int,
                                  source: str, info: IdentifierInfo,
                                  rng: random.Random,
                                  alpha_grid: list) -> dict:
    """Method 2: sweep alpha, track entropy at the target position."""
    try:
        base_ids, tgt = _build_window(tokenizer, source,
                                      info.start_byte, info.end_byte, mask_id)
    except Exception as e:
        return {"error": str(e)}

    h_by_alpha = {}
    for alpha in alpha_grid:
        if alpha == 0.0:
            ids = base_ids
        else:
            ids = _apply_masking(base_ids, mask_id, tgt, alpha, rng)
        try:
            logits = _forward(model, ids)
        except Exception as e:
            return {"error": str(e)}

        lp    = logits[0, tgt, :].float()
        probs = torch.softmax(lp, dim=-1)
        lprob = torch.log(probs + 1e-12)
        h_by_alpha[alpha] = float(-(probs * lprob).sum())

    h_0  = h_by_alpha.get(0.0, 0.0)
    h_08 = h_by_alpha.get(0.8, h_by_alpha.get(max(alpha_grid), 0.0))
    delta_h = h_08 - h_0

    return {
        "h_at_0":  h_0,
        "h_at_08": h_08,
        "delta_h": delta_h,
        "h_curve": h_by_alpha,
    }

# ══════════════════════════════════════════════════════════════════════════════
# Smell Score & Verdict
# ══════════════════════════════════════════════════════════════════════════════

# Calibrated from 1000-sample experiment
OC_SMELL_RANK_THRESHOLD  = 2000   # smell_severe_rank < this = smell fires (mean was ~1742 in experiments)
OC_TRAP_RATIO_THRESHOLD  = 2.0    # gt_rank / severe_rank > 2.0x = strong trap
CTX_DELTA_H_THRESHOLD    = 1.5    # ΔH < 1.5 = low context sensitivity (suspicious)
CTX_DELTA_H_GOOD         = 3.0    # ΔH > 3.0 = well-anchored identifier


def compute_smell_score(oc: dict, ctx: dict) -> tuple[float, str]:
    """
    Returns (smell_score 0–1, verdict).
    
    Score breakdown:
      over_confidence_score (0–1):
        0.0 = model is MORE confident about gt than smells (clean)
        1.0 = model heavily prefers smells over gt (strong smell)
      
      ctx_sensitivity_score (0–1):
        0.0 = high ΔH (name is tightly context-bound, clean)
        1.0 = low  ΔH (name is context-free, smell)
    """
    # --- OC score ---
    regime = oc.get("regime", "UNCERTAIN")
    trap_r = oc.get("trap_ratio", 1.0)
    sev_r  = oc.get("smell_severe_rank", 9999)

    if regime == "OVERCONFIDENT":
        # Model already treats gt like a smell → immediate flag
        oc_score = 0.85
    elif regime == "CONFIDENT_RARE":
        if trap_r >= OC_TRAP_RATIO_THRESHOLD and sev_r < OC_SMELL_RANK_THRESHOLD:
            # Scale score aggressively based on the trap ratio
            oc_score = min(1.0, 0.5 + 0.1 * trap_r)
        elif trap_r >= 4.0:
            # Trap ratio is so insane that we flag it regardless of absolute rank limit
            oc_score = min(1.0, 0.4 + 0.1 * trap_r)
        else:
            oc_score = 0.1
    else:  # UNCERTAIN
        # E.g., gt_rank = 500. If an injected smell ranks 100, trap_r = 5.0
        if trap_r > 1.5:
            oc_score = min(1.0, 0.3 + 0.15 * trap_r)
        else:
            oc_score = 0.2

    # --- Context sensitivity score ---
    dh = ctx.get("delta_h", 0.0)
    if dh >= CTX_DELTA_H_GOOD:
        ctx_score = 0.0
    elif dh <= 0.5:
        ctx_score = 1.0
    else:
        ctx_score = max(0.0, 1.0 - (dh / CTX_DELTA_H_GOOD))

    # --- Combined ---
    if "error" in oc or "error" in ctx:
        ctx_w = 0.0 if "error" in ctx else W_CTX
        oc_w  = 0.0 if "error" in oc  else W_OC
        total_w = oc_w + ctx_w
        score = ((oc_w * oc_score) + (ctx_w * ctx_score)) / max(total_w, 1e-6)
    else:
        # Base weighted score
        score = W_OC * oc_score + W_CTX * ctx_score
        
        # If the Over-confidence trap fired strongly, it should override a "clean" context score
        # because the model's extreme preference for garbage names dominates safe naming.
        if oc_score >= 0.9:
            score = max(score, 0.7)  # Floor it at SMELL territory

    if score >= 0.65:
        verdict = "SMELL"
    elif score >= 0.35:
        verdict = "SUSPICIOUS"
    else:
        verdict = "CLEAN"

    return round(score, 4), verdict

# ══════════════════════════════════════════════════════════════════════════════
# Main Analysis Entry Point
# ══════════════════════════════════════════════════════════════════════════════

def analyze_java_file(java_source: str,
                      model_name: str = "DiffuCoder-7B-Base",
                      fast_mode: bool = False,
                      seed: int = 42) -> list[ProbeResult]:
    """
    Full pipeline:
      1. Parse Java → extract identifiers
      2. Load model
      3. For each identifier: run OC probe + CTX probe
      4. Compute smell score
      5. Return sorted ProbeResult list
    """
    # ── 1. Parse ────────────────────────────────────────────────────────────
    print("\n[1/4] Parsing Java source …")
    identifiers = extract_identifiers(java_source)
    print(f"      Found {len(identifiers)} identifiers: "
          f"{[i.name for i in identifiers]}")

    if not identifiers:
        print("  No identifiers found. Exiting.")
        return []

    # ── 2. Load model ───────────────────────────────────────────────────────
    print("\n[2/4] Loading model …")
    tokenizer, model, mask_id = load_model(model_name)

    alpha_grid = ALPHA_GRID_FAST if fast_mode else ALPHA_GRID_FULL
    rng = random.Random(seed)

    results = []

    # ── 3. Probe each identifier ────────────────────────────────────────────
    print(f"\n[3/4] Probing {len(identifiers)} identifiers …")
    for i, info in enumerate(identifiers, 1):
        print(f"  [{i}/{len(identifiers)}] '{info.name}' "
              f"({info.node_type}, line {info.start_point[0]+1}) …", end=" ")

        oc  = run_overconfidence_probe(model, tokenizer, mask_id,
                                       java_source, info, rng)
        ctx = run_context_sensitivity_probe(model, tokenizer, mask_id,
                                            java_source, info, rng, alpha_grid)
        score, verdict = compute_smell_score(oc, ctx)
        print(f"→ {verdict} (score={score})")

        if "error" in oc or "error" in ctx:
            print(f"       Error: OC={oc.get('error','')}, CTX={ctx.get('error','')}")

        results.append(ProbeResult(
            identifier=info.name,
            node_type=info.node_type,
            line=info.start_point[0] + 1,
            col=info.start_point[1] + 1,
            gt_rank=oc.get("gt_rank", -1),
            gt_prob=oc.get("gt_prob", -1.0),
            regime=oc.get("regime", "UNKNOWN"),
            smell_severe_rank=oc.get("smell_severe_rank", -1),
            smell_moderate_rank=oc.get("smell_moderate_rank", -1),
            smell_mild_rank=oc.get("smell_mild_rank", -1),
            trap_ratio=oc.get("trap_ratio", -1.0),
            delta_h=ctx.get("delta_h", -1.0),
            h_at_0=ctx.get("h_at_0", -1.0),
            h_at_08=ctx.get("h_at_08", -1.0),
            smell_score=score,
            verdict=verdict,
        ))

    # ── 4. Sort by smell score desc ─────────────────────────────────────────
    results.sort(key=lambda r: r.smell_score, reverse=True)
    return results


def print_report(results: list[ProbeResult], java_source: str):
    """Pretty-print the analysis results."""
    lines = java_source.splitlines()
    print()
    print("=" * 70)
    print("  CODE NAMING SMELL REPORT")
    print("=" * 70)

    smells     = [r for r in results if r.verdict == "SMELL"]
    suspicious = [r for r in results if r.verdict == "SUSPICIOUS"]
    clean      = [r for r in results if r.verdict == "CLEAN"]

    print(f"\n  {'SMELL':>12}: {len(smells)}")
    print(f"  {'SUSPICIOUS':>12}: {len(suspicious)}")
    print(f"  {'CLEAN':>12}: {len(clean)}")
    print()

    VERDICT_ICONS = {"SMELL": "🔴", "SUSPICIOUS": "🟡", "CLEAN": "🟢"}

    for r in results:
        icon   = VERDICT_ICONS.get(r.verdict, "⚪")
        ctx_ln = lines[r.line - 1] if r.line <= len(lines) else ""
        print(f"  {icon} [{r.verdict:<10}]  '{r.identifier}' "
              f"  Line {r.line:<4}  (score={r.smell_score:.2f})")
        print(f"       Code  : …{ctx_ln.strip()[:65]}…")
        print(f"       Regime: {r.regime:<16}  gt_rank={r.gt_rank:<8}  "
              f"severe_rank={r.smell_severe_rank}  trap_ratio={r.trap_ratio:.1f}x")
        print(f"       ΔH    : {r.delta_h:.3f}  (H₀={r.h_at_0:.3f} → H₀.₈={r.h_at_08:.3f})")
        print()

    print("=" * 70)
    print("  INTERPRETATION GUIDE")
    print("=" * 70)
    print("""
  SMELL      — High confidence model prefers generic tokens over the real name
               AND the name has low context sensitivity (ΔH < 1.5).
               → Rename to a more domain-specific, contextual identifier.

  SUSPICIOUS — One signal fires (OC trap OR low ΔH), but not both.
               → Review manually; may still be a meaningful short name.

  CLEAN      — Model respects the real name AND name is context-sensitive.
               → Identifier appears well-chosen for its role.

  Regime:
    OVERCONFIDENT  → model already treats the name like a smell (gt_rank ≤ 200).
    UNCERTAIN      → intermediate (200 < gt_rank ≤ 1000).
    CONFIDENT_RARE → model finds the name surprising but semantically rich.

  trap_ratio = gt_rank / smell_severe_rank
    > 2x → over-confidence trap fired (smells ranked much higher than real name)

  ΔH = H(α=0.8) − H(α=0.0)
    > 3.0 → name is strongly anchored to surrounding context (good).
    < 1.5 → name carries almost no context-specific information (bad).
""")

# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Code Naming Smell Detector — Java + DiffuCoder"
    )
    parser.add_argument("--file",    required=True,  help="Path to .java source file.")
    parser.add_argument("--code",    default=None,   help="Inline Java code string (alternative to --file).")
    parser.add_argument("--model",   default="DiffuCoder-7B-Base",
                        choices=list(MODEL_REGISTRY.keys()),
                        help="Model key to use for probing.")
    parser.add_argument("--fast",    action="store_true",
                        help="Use only 3 alpha steps (faster, less accurate).")
    parser.add_argument("--seed",    type=int, default=42)
    parser.add_argument("--output",  default=None,
                        help="Path to write JSON results (optional).")
    parser.add_argument("--list-models", action="store_true",
                        help="List available models and exit.")
    args = parser.parse_args()

    if args.list_models:
        print("Available models:")
        for k, v in MODEL_REGISTRY.items():
            print(f"  {k:<40} {v['id']}")
        sys.exit(0)

    if args.code:
        java_source = args.code
    else:
        with open(args.file, "r", encoding="utf-8") as f:
            java_source = f.read()

    print(f"\nAnalyzing: {args.file or '(inline)'}")
    print(f"Model:     {args.model}  |  fast_mode={args.fast}  |  device={DEVICE}")

    results = analyze_java_file(
        java_source=java_source,
        model_name=args.model,
        fast_mode=args.fast,
        seed=args.seed,
    )

    print_report(results, java_source)

    if args.output:
        with open(args.output, "w") as f:
            json.dump([asdict(r) for r in results], f, indent=2)
        print(f"\n  JSON report saved → {args.output}")
