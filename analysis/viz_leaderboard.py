import csv
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

BLUE, ORANGE = "#0076C2", "#FF8000"
OUT = "/path/to/CoReFusion/figures/new"
CSV = f"{OUT}/leaderboard_full.csv"
LJ = ["LJ_Q7","LJ_Q14","LJ_Q32","LJ_M24","LJ_G27"]

rows = []
for r in csv.DictReader(open(CSV)):
    f = lambda k: float(r[k]) * 100 if r.get(k) not in (None, "", "None") else 0.0
    lj = [f(j) for j in LJ]
    em = f("em_gated")
    # Skip rows with no signal at all (e.g. a model whose predictions are all
    # empty -> EM=0 and every judge 0). Real models clear this bar; this makes
    # DiffusionGemma auto-appear once its predictions are non-empty, instead of
    # a hardcoded name exclusion.
    if em == 0 and max(lj) == 0:
        continue
    rows.append({"model": r["model"], "arch": r["arch"], "em": em,
                 "lj_mean": sum(lj)/len(lj), "lj_lo": min(lj), "lj_hi": max(lj),
                 "is_dllm": "dLLM" in r["arch"]})
rows.sort(key=lambda r: r["em"])           # ascending -> highest on top in barh
models = [r["model"] for r in rows]
colors = [ORANGE if r["is_dllm"] else BLUE for r in rows]
y = np.arange(len(rows))

fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 8.5), sharey=True,
                               gridspec_kw={"wspace": 0.05})
# (a) strict EM
axL.barh(y, [r["em"] for r in rows], color=colors, edgecolor="white", height=0.74)
for i, r in enumerate(rows):
    axL.text(r["em"] + 0.4, i, f"{r['em']:.1f}", va="center", fontsize=8.5)
axL.set_yticks(y); axL.set_yticklabels(models, fontsize=9)
axL.set_xlabel("Exact Match (strict all-sites, %)")
axL.set_title("(a) Strict EM", fontweight="bold")
axL.set_xlim(0, max(r["em"] for r in rows) * 1.16)
axL.grid(axis="x", alpha=0.25); axL.set_axisbelow(True)

# (b) LJ mean + min–max whisker across the 5 judges
ljm = [r["lj_mean"] for r in rows]
err = [[r["lj_mean"]-r["lj_lo"] for r in rows], [r["lj_hi"]-r["lj_mean"] for r in rows]]
axR.barh(y, ljm, color=colors, edgecolor="white", height=0.74)
axR.errorbar(ljm, y, xerr=err, fmt="none", ecolor="#444", elinewidth=1, capsize=2.5)
for i, r in enumerate(rows):
    axR.text(r["lj_hi"] + 0.9, i, f"{r['lj_mean']:.1f}", va="center", fontsize=8.5)
axR.set_xlabel("LLM-as-Judge acceptance (%)")
axR.set_title("(b) Semantic acceptance — mean of 5 judges (whisker = min–max)", fontweight="bold")
axR.set_xlim(0, max(r["lj_hi"] for r in rows) * 1.16)
axR.grid(axis="x", alpha=0.25); axR.set_axisbelow(True)

leg = [Line2D([0],[0], marker="s", color="w", markerfacecolor=ORANGE, markersize=12, label="dLLM (diffusion)"),
       Line2D([0],[0], marker="s", color="w", markerfacecolor=BLUE, markersize=12, label="AR (FIM) / Encoder-decoder")]
axR.legend(handles=leg, loc="lower right", fontsize=9.5, framealpha=0.95)
fig.suptitle(f"CoReFusion identifier-renaming leaderboard   ({len(rows)} models · RefineID · n=1000)",
             fontweight="bold", fontsize=13)
fig.text(0.5, 0.01, "Sorted by strict EM. Models with no signal (EM=0 and all judges 0) are omitted.",
         ha="center", fontsize=8.5, color="#777")
fig.subplots_adjust(left=0.16, right=0.985, top=0.92, bottom=0.10)
fig.savefig(f"{OUT}/fig6_leaderboard_bars.png", bbox_inches="tight", dpi=150)
print("wrote", f"{OUT}/fig6_leaderboard_bars.png")
