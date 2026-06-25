"""
Generate the CodeT5+ 16B paper-ready figures from the refineID results CSV.

Inputs:
    results/codeT5results/CodeT5p-16B_refineID_fim_20260509_134747.csv
    results/refine_id_benchmark/llm_judge_results/llm_judge/
        summary_judge_Qwen2.5-7B-Instruct_20260304_130047.csv   (peer EM)

Outputs (all under results/codet5p_16b_analysis/):
    fig_em_peer_comparison.{pdf,png}     -- horizontal bar chart of EM vs peers
    fig_em_by_mask_count.{pdf,png}       -- EM stratified by # of [MASK] slots
    fig_em_by_camel_pieces.{pdf,png}     -- EM stratified by # of camelCase pieces
    fig_em_by_gt_length.{pdf,png}        -- EM stratified by ground-truth length
    stratification.csv                   -- the underlying numbers in CSV form
"""

import csv
import os
import re
import sys
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = "/path/to/CoReFusion"
RESULTS_CSV = f"{ROOT}/results/codeT5results/CodeT5p-16B_refineID_fim_20260509_134747.csv"
PEER_CSV = f"{ROOT}/results/refine_id_benchmark/llm_judge_results/llm_judge/summary_judge_Qwen2.5-7B-Instruct_20260304_130047.csv"
OUT_DIR = f"{ROOT}/results/codet5p_16b_analysis"

os.makedirs(OUT_DIR, exist_ok=True)
csv.field_size_limit(sys.maxsize)

plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 110,
})

OURS = "CodeT5+ 16B"
OURS_COLOR = "#d62728"
PEER_COLOR = "#4c78a8"


# ---- Load data -------------------------------------------------------------

def load_rows(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def camel_pieces(s):
    return len(re.findall(r"[A-Z]?[a-z]+|[A-Z]+(?=[A-Z]|$)", s))


def save(fig, name):
    for ext in ("pdf", "png"):
        out = os.path.join(OUT_DIR, f"{name}.{ext}")
        fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {name}.{{pdf,png}}")


# ---- Figure 1: peer comparison --------------------------------------------

def fig_peer_comparison():
    peers = []  # (label, em, protocol)
    with open(PEER_CSV) as f:
        for row in csv.DictReader(f):
            fn = row["benchmark_file"]
            em = float(row["exact_match_acc"])
            bt = row["benchmark_type"]
            # Strip "_refineID_<protocol>_<timestamp>.csv"
            label = re.sub(r"_refineID_(diffusion|fim)_\d+_\d+\.csv$", "", fn)
            peers.append((label, em, bt))

    # Deduplicate (keep highest score per model)
    best = {}
    for label, em, bt in peers:
        if label not in best or em > best[label][0]:
            best[label] = (em, bt)

    # Our result
    ours_em = 0.235
    rows = [(OURS, ours_em, "prefix-completion")] + [
        (l, e, bt) for l, (e, bt) in best.items()
    ]
    rows.sort(key=lambda x: x[1])

    labels = [r[0] for r in rows]
    ems = [r[1] for r in rows]
    bts = [r[2] for r in rows]

    colors = []
    for l, b in zip(labels, bts):
        if l == OURS:
            colors.append(OURS_COLOR)
        elif b == "diffusion":
            colors.append("#2ca02c")
        else:
            colors.append(PEER_COLOR)

    fig, ax = plt.subplots(figsize=(7.5, 0.32 * len(labels) + 1.2))
    y = list(range(len(labels)))
    ax.barh(y, ems, color=colors, edgecolor="black", linewidth=0.4)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Exact-Match accuracy")
    ax.set_xlim(0, max(ems) * 1.18)
    ax.grid(axis="x", linestyle=":", alpha=0.5)

    for yi, em in zip(y, ems):
        ax.text(em + 0.004, yi, f"{em:.3f}", va="center", fontsize=9)

    # Legend
    from matplotlib.patches import Patch
    handles = [
        Patch(color=OURS_COLOR, label="CodeT5+ 16B (this work)"),
        Patch(color="#2ca02c", label="Diffusion (masked)"),
        Patch(color=PEER_COLOR, label="Autoregressive (FIM)"),
    ]
    ax.legend(handles=handles, loc="lower right", frameon=False)
    ax.set_title("RefineID Exact-Match: CodeT5+ 16B vs. AR-FIM and diffusion baselines")
    save(fig, "fig_em_peer_comparison")


# ---- Stratification figures -----------------------------------------------

def stratify_and_plot():
    rows = load_rows(RESULTS_CSV)

    # Coarse mask_count buckets
    def mc_bucket(n):
        if n == 1: return "1"
        if n == 2: return "2"
        if n == 3: return "3"
        if 4 <= n <= 6: return "4-6"
        if 7 <= n <= 10: return "7-10"
        if 11 <= n <= 20: return "11-20"
        return "21+"
    order_mc = ["1", "2", "3", "4-6", "7-10", "11-20", "21+"]

    mc_total = Counter(); mc_correct = Counter()
    for r in rows:
        b = mc_bucket(int(r["mask_count"] or 0))
        mc_total[b] += 1
        if r["correct"] == "True":
            mc_correct[b] += 1

    # camelCase pieces
    cp_total = Counter(); cp_correct = Counter()
    for r in rows:
        p = camel_pieces(r["ground_truth"])
        key = str(p) if p <= 3 else "4+"
        cp_total[key] += 1
        if r["correct"] == "True":
            cp_correct[key] += 1
    order_cp = ["1", "2", "3", "4+"]

    # GT length
    def len_bucket(L):
        if L <= 8:   return "5-8"
        if L <= 12:  return "9-12"
        if L <= 20:  return "13-20"
        return "21+"
    order_len = ["5-8", "9-12", "13-20", "21+"]
    L_total = Counter(); L_correct = Counter()
    for r in rows:
        b = len_bucket(len(r["ground_truth"]))
        L_total[b] += 1
        if r["correct"] == "True":
            L_correct[b] += 1

    def plot_strat(name, title, xlabel, order, total, correct):
        em = [correct[k] / total[k] if total[k] else 0 for k in order]
        n = [total[k] for k in order]

        fig, ax = plt.subplots(figsize=(6.0, 3.6))
        bars = ax.bar(order, em, color=OURS_COLOR, edgecolor="black", linewidth=0.5)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Exact-Match accuracy")
        ax.set_ylim(0, max(em) * 1.25 if em else 0.5)
        ax.grid(axis="y", linestyle=":", alpha=0.5)
        ax.set_title(title)
        for bar, e, ni in zip(bars, em, n):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f"{e:.2%}\n(n={ni})", ha="center", va="bottom", fontsize=9)
        # Overall reference line
        overall = sum(1 for r in rows if r["correct"]=="True") / len(rows)
        ax.axhline(overall, ls="--", c="grey", lw=0.8)
        ax.text(0.99, overall + 0.005, f"overall = {overall:.2%}",
                transform=ax.get_yaxis_transform(), ha="right", color="grey", fontsize=9)
        save(fig, name)
        return list(zip(order, [total[k] for k in order],
                        [correct[k] for k in order], em))

    rows_mc = plot_strat(
        "fig_em_by_mask_count",
        "CodeT5+ 16B: EM by number of [MASK] slots in sample",
        "Number of [MASK] slots", order_mc, mc_total, mc_correct,
    )
    rows_cp = plot_strat(
        "fig_em_by_camel_pieces",
        "CodeT5+ 16B: EM by camelCase complexity of the target identifier",
        "Number of camelCase pieces in ground-truth identifier",
        order_cp, cp_total, cp_correct,
    )
    rows_len = plot_strat(
        "fig_em_by_gt_length",
        "CodeT5+ 16B: EM by ground-truth identifier length",
        "Ground-truth identifier length (chars)",
        order_len, L_total, L_correct,
    )

    # Save numbers to CSV
    out = os.path.join(OUT_DIR, "stratification.csv")
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["stratification", "bucket", "n", "correct", "em"])
        for label, dataset in [
            ("mask_count", rows_mc),
            ("camel_pieces", rows_cp),
            ("gt_length", rows_len),
        ]:
            for bucket, n, c, em in dataset:
                w.writerow([label, bucket, n, c, f"{em:.4f}"])
    print(f"  wrote stratification.csv")


def main():
    print(f"Writing to {OUT_DIR}")
    fig_peer_comparison()
    stratify_and_plot()
    print("done.")


if __name__ == "__main__":
    main()
