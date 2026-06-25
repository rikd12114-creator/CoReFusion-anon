#!/usr/bin/env python3
"""
Quick unit-test for the smell detector without loading the model.
Validates:
  1. tree-sitter Java parsing
  2. smell score & verdict calculation with synthetic data
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from detector.code_naming_smell_detector import (
    extract_identifiers, compute_smell_score, print_report, ProbeResult
)

# ── 1. Test Java parsing ───────────────────────────────────────────────────────
JAVA_SRC = """
public class UserService {
    private String userAuthToken;

    public void validateSession(int x, String sessionKey) {
        String tmp = getUserData();
        int retryCount = 0;
        for (int i = 0; i < 5; i++) {
            boolean res = attemptLogin(tmp);
            if (res) break;
            retryCount++;
        }
    }
}
"""

print("="*70)
print("  TEST 1: tree-sitter Java parsing")
print("="*70)
identifiers = extract_identifiers(JAVA_SRC)
print(f"  Extracted {len(identifiers)} identifiers:")
for info in identifiers:
    print(f"    '{info.name:<20}' node={info.node_type:<28} line={info.start_point[0]+1}")
print()

# ── 2. Test smell score ────────────────────────────────────────────────────────
print("="*70)
print("  TEST 2: Smell score & verdict calculation")
print("="*70)

test_cases = [
    # (name, oc_dict, ctx_dict, expected_verdict)
    {
        "name": "x",
        "oc":  {"gt_rank": 12, "gt_prob": 0.45, "regime": "OVERCONFIDENT",
                "smell_severe_rank": 8, "smell_moderate_rank": 50, "smell_mild_rank": 120, "trap_ratio": 1.5},
        "ctx": {"h_at_0": 0.5, "h_at_08": 0.9, "delta_h": 0.4},
        "expected": "SMELL",
        "note": "OVERCONFIDENT + very low ΔH → should be SMELL"
    },
    {
        "name": "tmp",
        "oc":  {"gt_rank": 95, "gt_prob": 0.05, "regime": "OVERCONFIDENT",
                "smell_severe_rank": 40, "smell_moderate_rank": 70, "smell_mild_rank": 200, "trap_ratio": 2.4},
        "ctx": {"h_at_0": 1.2, "h_at_08": 1.8, "delta_h": 0.6},
        "expected": "SMELL",
        "note": "Moderate generic name, trap fired"
    },
    {
        "name": "userAuthToken",
        "oc":  {"gt_rank": 8200, "gt_prob": 0.0001, "regime": "CONFIDENT_RARE",
                "smell_severe_rank": 900, "smell_moderate_rank": 4000, "smell_mild_rank": 3500, "trap_ratio": 9.1},
        "ctx": {"h_at_0": 1.1, "h_at_08": 5.3, "delta_h": 4.2},
        "expected": "SMELL",
        "note": "CONFIDENT_RARE but trap_ratio=9x AND ΔH is high (context-specific name ironically flagged by OC)"
    },
    {
        "name": "retryCount",
        "oc":  {"gt_rank": 5500, "gt_prob": 0.00005, "regime": "CONFIDENT_RARE",
                "smell_severe_rank": 3200, "smell_moderate_rank": 6100, "smell_mild_rank": 5800, "trap_ratio": 1.7},
        "ctx": {"h_at_0": 1.4, "h_at_08": 5.0, "delta_h": 3.6},
        "expected": "CLEAN",
        "note": "CONFIDENT_RARE, trap_ratio < 2x, ΔH high → CLEAN descriptor"
    },
    {
        "name": "sessionKey",
        "oc":  {"gt_rank": 650, "gt_prob": 0.0009, "regime": "UNCERTAIN",
                "smell_severe_rank": 300, "smell_moderate_rank": 800, "smell_mild_rank": 900, "trap_ratio": 2.2},
        "ctx": {"h_at_0": 1.5, "h_at_08": 2.8, "delta_h": 1.3},
        "expected": "SUSPICIOUS",
        "note": "UNCERTAIN zone + borderline ΔH"
    },
]

all_ok = True
for tc in test_cases:
    score, verdict = compute_smell_score(tc["oc"], tc["ctx"])
    ok = "✅" if verdict == tc["expected"] else "❌"
    if verdict != tc["expected"]:
        all_ok = False
    print(f"  {ok}  '{tc['name']:<20}'  score={score:.3f}  verdict={verdict:<10} | expected={tc['expected']}")
    print(f"       {tc['note']}")
    print()

print("="*70)
print(f"  All tests: {'PASSED ✅' if all_ok else 'SOME FAILED ❌'}")
print("="*70)
