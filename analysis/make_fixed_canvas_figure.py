"""
Generate fig_fixed_canvas.{pdf,png}: visual illustration of the
fixed-canvas constraint in standard masked diffusion LMs.

The figure makes one claim explicit:

    > the number k of <|mask|> tokens allocated for a single identifier
    > site must be FIXED before inference begins.

Three rows pin down the same Java declaration  `int <name> = 0;`  for
three pre-inference choices of k.  The middle row picks k = k*, where
k* is the number of BPE sub-tokens in the ground-truth identifier
`activeCount`  (= [`active`, `Count`], so k* = 2).  Each row shows:

    [k label]   [pre-inference canvas]   →   [post-inference result]   [outcome badge]

The two failure modes the prose talks about are visualised directly:

    k = 1 < k*  : single mask cannot fit a 2-BPE-token name; the model
                  must commit to a shorter sub-word — here `count` —
                  losing the qualifying prefix in the target name.

    k = 2 = k*  : canvas matches target BPE tokenization exactly; both
                  sub-tokens land on `active` + `Count` (green pills).

    k = 4 > k*  : excess masks force the model to fabricate filler
                  sub-tokens not in the ground truth — here `User` and
                  `Idx` (orange pills) — yielding an over-generated
                  identifier `activeUserCountIdx`.

Style sympathies with analysis/make_dreamon_canvas_figure.py and
analysis/make_dllm_demo_figure.py: IEEE figure* width, Times serif
text, monospace code, rounded panel backgrounds, pastel pills on
sentinel/sub-token spans.
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

# ─────────────────────────────── paths ──────────────────────────────────────
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
OUTDIR = os.path.join(ROOT, "docs", "figures")
os.makedirs(OUTDIR, exist_ok=True)

# ─────────────────────────────── colours ────────────────────────────────────
C_MASK = "#E63946"  # red — <|mask|>
C_MATCH = "#06A77D"  # green — sub-tokens that match the target name
C_DEVIATE = "#F77F00"  # orange — truncated / filler sub-tokens
C_CODE = "#1B1B1B"
C_KW = "#0F4F9C"  # blue keywords
C_PANEL_BG = "#FCFCFC"
C_PANEL_BORDER = "#D0D0D0"
C_ARROW = "#9A9A9A"
C_LABEL = "#1F1F1F"
C_LABEL_SUB = "#5A5A5A"

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 9,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


# ─────────────────────────────── content ────────────────────────────────────
# Each case: pre-inference canvas, post-inference result, outcome badge.
# kind:
#   "plain"   regular code
#   "kw"      Java keyword
#   "mask"    <|mask|>           red pill, white bold text
#   "match"   sub-token that matches target identifier   green pill (low alpha)
#   "deviate" truncated / filler sub-token              orange pill (low alpha)

CASES = [
    {
        "k": 1,
        "before": [
            ("int", "kw"),
            (" ", "plain"),
            ("<|mask|>", "mask"),
            (" = 0;", "plain"),
        ],
        "after": [
            ("int", "kw"),
            (" ", "plain"),
            ("count", "deviate"),
            (" = 0;", "plain"),
        ],
        "tag": "k < k*    underflow",
        "sub": "1 mask mostly can't fit a 2-token name",
        "kind": "deviate",
    },
    {
        "k": 2,
        "before": [
            ("int", "kw"),
            (" ", "plain"),
            ("<|mask|>", "mask"),
            ("<|mask|>", "mask"),
            (" = 0;", "plain"),
        ],
        "after": [
            ("int", "kw"),
            (" ", "plain"),
            ("active", "match"),
            ("Count", "match"),
            (" = 0;", "plain"),
        ],
        "tag": "k = k*    oracle",
        "sub": "canvas size matches target tokens",
        "kind": "match",
    },
    {
        "k": 4,
        "before": [
            ("int", "kw"),
            (" ", "plain"),
            ("<|mask|>", "mask"),
            ("<|mask|>", "mask"),
            ("<|mask|>", "mask"),
            ("<|mask|>", "mask"),
            (" = 0;", "plain"),
        ],
        "after": [
            ("int", "kw"),
            (" ", "plain"),
            ("active", "match"),
            ("User", "deviate"),
            ("Count", "match"),
            ("Idx", "deviate"),
            (" = 0;", "plain"),
        ],
        "tag": "k > k*    overflow",
        "sub": "excess masks → filler unnecessary sub-tokens",
        "kind": "deviate",
    },
]


# ────────────────────────── rendering helpers ───────────────────────────────
def get_style(kind):
    """Return (text_color, pill_bg, pill_alpha, is_pill, bold)."""
    palette = {
        "kw": (C_KW, None, 0, False, False),
        "mask": ("white", C_MASK, 0.95, True, True),
        "match": (C_CODE, C_MATCH, 0.32, True, False),
        "deviate": (C_CODE, C_DEVIATE, 0.42, True, False),
        "plain": (C_CODE, None, 0, False, False),
    }
    return palette.get(kind, palette["plain"])


def draw_code_tokens(ax, tokens, y, font_size, x_start):
    PAD_X = 0.18
    HALF_H = 0.32
    x = x_start
    for text, kind in tokens:
        color, bg, alpha, pill, bold = get_style(kind)
        w = len(text)
        if pill:
            rect = FancyBboxPatch(
                (x - PAD_X, y - HALF_H),
                w + 2 * PAD_X,
                2 * HALF_H,
                boxstyle="round,pad=0.0,rounding_size=0.18",
                facecolor=bg,
                edgecolor="none",
                alpha=alpha,
                zorder=1,
            )
            ax.add_patch(rect)
        ax.text(
            x,
            y,
            text,
            family="monospace",
            fontsize=font_size,
            color=color,
            ha="left",
            va="center",
            zorder=3,
            fontweight="bold" if bold else "normal",
        )
        x += w
    return x - x_start


def tune_font_for_axes(fig, ax, fontsize_init):
    """Return the font size that makes 1 monospace char = 1 data unit on ax."""
    N = 40
    probe = ax.text(
        0,
        0,
        "X" * N,
        family="monospace",
        fontsize=fontsize_init,
        ha="left",
        va="bottom",
        alpha=0.0,
    )
    fig.canvas.draw()
    bbox = probe.get_window_extent(renderer=fig.canvas.get_renderer())
    probe.remove()
    inv = ax.transData.inverted()
    x0 = inv.transform((bbox.x0, bbox.y0))[0]
    x1 = inv.transform((bbox.x1, bbox.y0))[0]
    observed = x1 - x0
    return fontsize_init * (N / observed)


# Each code box internal char-width.  Sized so the widest content fits
# with comfortable padding:
#   before max: "int <|mask|><|mask|><|mask|><|mask|> = 0;"  = 41 chars
#   after  max: "int activeUserCountIdx = 0;"               = 27 chars
BEFORE_COLS = 44
AFTER_COLS = 33  # chosen so 1 char in BEFORE and AFTER axes is the same
# physical width (BEFORE_W/44 == AFTER_W/33), giving a
# uniform code font size across both columns


def draw_code_box(fig, x0, y0, w, h, tokens, n_cols, code_font):
    ax = fig.add_axes([x0, y0, w, h])
    ax.set_xlim(-0.4, n_cols + 0.4)
    ax.set_ylim(-0.10, 2.10)
    ax.axis("off")

    # rounded background panel
    bg_patch = FancyBboxPatch(
        (0.0, 0.30),
        n_cols,
        1.45,
        boxstyle="round,pad=0.0,rounding_size=0.32",
        facecolor=C_PANEL_BG,
        edgecolor=C_PANEL_BORDER,
        linewidth=0.6,
        zorder=0,
    )
    ax.add_patch(bg_patch)

    # code line, centred horizontally
    code_chars = sum(len(t[0]) for t in tokens)
    x_start = max(0.6, (n_cols - code_chars) / 2)
    draw_code_tokens(ax, tokens, 1.03, code_font, x_start=x_start)


def draw_arrow(fig, x0, y0, w, h):
    ax = fig.add_axes([x0, y0, w, h])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    arrow = FancyArrowPatch(
        (0.05, 0.52),
        (0.95, 0.52),
        arrowstyle="-|>,head_length=4.5,head_width=2.8",
        color=C_ARROW,
        linewidth=1.0,
        mutation_scale=6,
    )
    ax.add_patch(arrow)
    ax.text(
        0.5,
        0.78,
        "inference",
        fontsize=6.6,
        fontstyle="italic",
        color="#888",
        ha="center",
        va="center",
    )


def draw_status_badge(fig, x0, y0, w, h, case):
    ax = fig.add_axes([x0, y0, w, h])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    tag_color = C_MATCH if case["kind"] == "match" else C_DEVIATE

    # soft colored pill behind the tag text
    badge = FancyBboxPatch(
        (0.04, 0.56),
        0.92,
        0.30,
        boxstyle="round,pad=0.0,rounding_size=0.20",
        facecolor=tag_color,
        edgecolor="none",
        alpha=0.32,
        zorder=1,
    )
    ax.add_patch(badge)
    ax.text(
        0.5,
        0.71,
        case["tag"],
        fontsize=8.2,
        fontweight="bold",
        color=C_LABEL,
        ha="center",
        va="center",
        family="serif",
        zorder=2,
    )

    # italic sub-caption underneath
    ax.text(
        0.5,
        0.24,
        case["sub"],
        fontsize=6.6,
        fontstyle="italic",
        color=C_LABEL_SUB,
        ha="center",
        va="center",
        family="serif",
    )


# ─────────────────────────────── figure ─────────────────────────────────────
FIG_W = 7.16  # IEEE figure*
FIG_H = 3.85

fig = plt.figure(figsize=(FIG_W, FIG_H), dpi=200)

# layout (fig-fraction units)
LABEL_X = 0.005
LABEL_W = 0.060
BEFORE_X = 0.072
BEFORE_W = 0.360
ARROW_X = 0.438
ARROW_W = 0.050
AFTER_X = 0.495
AFTER_W = 0.270
STATUS_X = 0.775
STATUS_W = 0.220

HEADER_Y = 0.945
SUB_Y = 0.910

TOP = 0.860
BOT = 0.040
N_ROWS = 3
GAP_V = 0.030
ROW_H = (TOP - BOT - GAP_V * (N_ROWS - 1)) / N_ROWS

# column headers
fig.text(
    BEFORE_X + BEFORE_W / 2,
    HEADER_Y,
    "Pre-inference canvas",
    fontsize=9.4,
    fontweight="bold",
    ha="center",
    va="top",
    color=C_LABEL,
)
fig.text(
    BEFORE_X + BEFORE_W / 2,
    SUB_Y,
    "(k chosen before inference begins)",
    fontsize=7.2,
    fontstyle="italic",
    ha="center",
    va="top",
    color="#666",
)

fig.text(
    AFTER_X + AFTER_W / 2,
    HEADER_Y,
    "Post-inference result",
    fontsize=9.4,
    fontweight="bold",
    ha="center",
    va="top",
    color=C_LABEL,
)
fig.text(
    AFTER_X + AFTER_W / 2,
    SUB_Y,
    "(target  activeCount  →  [active, Count],  k* = 2)",
    fontsize=7.2,
    fontstyle="italic",
    ha="center",
    va="top",
    color="#666",
)

# tune font once on a sample BEFORE-shaped axes; reuse for AFTER too
# (BEFORE_W/BEFORE_COLS == AFTER_W/AFTER_COLS so the physical char size is
# identical in both columns)
_sample = fig.add_axes([0, 0, BEFORE_W, ROW_H])
_sample.set_xlim(-0.4, BEFORE_COLS + 0.4)
_sample.set_ylim(-0.10, 2.10)
_sample.axis("off")
CODE_FONT = tune_font_for_axes(fig, _sample, 9.0)
fig.delaxes(_sample)

# rows
for i, case in enumerate(CASES):
    y_top = TOP - i * (ROW_H + GAP_V)
    y0 = y_top - ROW_H

    # left "k = X" label
    lax = fig.add_axes([LABEL_X, y0, LABEL_W, ROW_H])
    lax.set_xlim(0, 1)
    lax.set_ylim(0, 1)
    lax.axis("off")
    lax.text(
        0.5,
        0.55,
        f"$k = {case['k']}$",
        fontsize=12.0,
        fontweight="bold",
        color=C_LABEL,
        ha="center",
        va="center",
    )

    # pre-inference canvas
    draw_code_box(
        fig, BEFORE_X, y0, BEFORE_W, ROW_H, case["before"], BEFORE_COLS, CODE_FONT
    )

    # arrow
    draw_arrow(fig, ARROW_X, y0, ARROW_W, ROW_H)

    # post-inference result
    draw_code_box(
        fig, AFTER_X, y0, AFTER_W, ROW_H, case["after"], AFTER_COLS, CODE_FONT
    )

    # outcome badge + sub-caption
    draw_status_badge(fig, STATUS_X, y0, STATUS_W, ROW_H, case)


# save
out_pdf = os.path.join(OUTDIR, "fig_fixed_canvas.pdf")
out_png = os.path.join(OUTDIR, "fig_fixed_canvas.png")
fig.savefig(out_pdf, pad_inches=0.04)
fig.savefig(out_png, pad_inches=0.04, dpi=300)
plt.close(fig)
print(f"Wrote {out_pdf}")
print(f"Wrote {out_png}")
