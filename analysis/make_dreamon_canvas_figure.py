"""
Generate fig_dreamon_canvas.{pdf,png}: a diagrammatic explanation of
DreamOn's variable-length canvas mechanism (Wu et al., 2025), the
trick that frees masked diffusion language models from a fixed-size
canvas.

Two side-by-side trajectories illustrate the two new sentinel tokens,
applied to a tiny Java identifier-renaming infilling task that matches
the surrounding thesis context (Java variable-naming smells):

    <|expand|>  — predicted from <|mask|>, then deterministically
                  rewritten into TWO <|mask|> tokens at the same
                  position.  Effect: canvas grows by 1 per occurrence.
    <|delete|>  — predicted from <|mask|>, then deterministically
                  removed from the sequence.
                  Effect: canvas shrinks by 1 per occurrence.

Both substitutions are applied AFTER each denoising step, treating the
two new tokens as regular vocabulary items the model can predict —
which is exactly what makes the scheme architecture-agnostic.

Style matches analysis/make_dllm_demo_figure.py: IEEE figure*
(7.16 in) width, Times serif text, monospace code, soft pastel pills
on sentinel tokens, downward arrows between snapshots.
"""

import os
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

# ─────────────────────────────── paths ──────────────────────────────────────
HERE   = os.path.dirname(os.path.abspath(__file__))
ROOT   = os.path.abspath(os.path.join(HERE, ".."))
OUTDIR = os.path.join(ROOT, "docs", "figures")
os.makedirs(OUTDIR, exist_ok=True)

# ─────────────────────────────── colours ────────────────────────────────────
C_MASK         = "#E63946"   # red   — <|mask|>           (matches sibling fig)
C_EXPAND       = "#F77F00"   # orange — <|expand|>         (matches step 2 orange)
C_DELETE       = "#4B7FB5"   # slate blue — <|delete|>      (distinct from type purple)
C_REVEAL       = "#06A77D"   # green — denoised regular     (matches step 3 green)
C_CODE         = "#1B1B1B"
C_KW           = "#0F4F9C"
C_TYPE         = "#6B2D8C"
C_PANEL_BG     = "#FCFCFC"
C_PANEL_BORDER = "#D0D0D0"
C_ARROW        = "#9A9A9A"
C_LABEL_SUB    = "#5A5A5A"
C_BADGE        = "#3C3C3C"

plt.rcParams.update({
    "font.family":  "serif",
    "font.serif":   ["Times New Roman", "DejaVu Serif"],
    "font.size":    9,
    "pdf.fonttype": 42,
    "ps.fonttype":  42,
})


# ─────────────────────────────── content ────────────────────────────────────
# Each panel: dict with
#   "step"   : badge text (top of panel)
#   "sub"    : italic caption (bottom of panel)
#   "tokens" : list of (text, kind) — rendered left-to-right in monospace.
# kind:
#   "plain"  — regular code
#   "kw"     — Java keyword
#   "mask"   — <|mask|>     red pill, white bold text
#   "expand" — <|expand|>   orange pill
#   "delete" — <|delete|>   blue pill
#   "reveal" — denoised regular token, low-alpha green pill

EXPAND_TRAJ = [
    {
        "step": "t = T  ·  initial canvas",
        "sub":  "only 1 mask reserved — target needs 2 BPE tokens",
        "tokens": [("int", "kw"), (" ", "plain"),
                   ("<|mask|>", "mask"),
                   (" = 0;", "plain")],
    },
    {
        "step": "step k  ·  model prediction",
        "sub":  "the mask is predicted as <|expand|>",
        "tokens": [("int", "kw"), (" ", "plain"),
                   ("<|expand|>", "expand"),
                   (" = 0;", "plain")],
    },
    {
        "step": "step k  ·  after substitution rule",
        "sub":  "rule:  <|expand|>  →  two <|mask|>    (canvas + 1)",
        "tokens": [("int", "kw"), (" ", "plain"),
                   ("<|mask|>", "mask"), ("<|mask|>", "mask"),
                   (" = 0;", "plain")],
    },
    {
        "step": "t = 0  ·  converged",
        "sub":  "the two masks denoise to regular BPE tokens",
        "tokens": [("int", "kw"), (" ", "plain"),
                   ("active", "reveal"), ("Count", "reveal"),
                   (" = 0;", "plain")],
    },
]

DELETE_TRAJ = [
    {
        "step": "t = T  ·  initial canvas",
        "sub":  "3 masks reserved — target needs only 1 BPE token",
        "tokens": [("int", "kw"), (" ", "plain"),
                   ("<|mask|>", "mask"), ("<|mask|>", "mask"), ("<|mask|>", "mask"),
                   (" = 0;", "plain")],
    },
    {
        "step": "step k  ·  model prediction",
        "sub":  "one mask  →  “n”;   two masks  →  <|delete|>",
        "tokens": [("int", "kw"), (" ", "plain"),
                   ("n", "reveal"),
                   ("<|delete|>", "delete"), ("<|delete|>", "delete"),
                   (" = 0;", "plain")],
    },
    {
        "step": "step k  ·  after substitution rule",
        "sub":  "rule:  <|delete|>  →  removed    (canvas − 2)",
        "tokens": [("int", "kw"), (" ", "plain"),
                   ("n", "reveal"),
                   (" = 0;", "plain")],
    },
    {
        "step": "t = 0  ·  converged",
        "sub":  "canvas length now matches the target",
        "tokens": [("int", "kw"), (" ", "plain"),
                   ("n", "reveal"),
                   (" = 0;", "plain")],
    },
]


# ─────────────────────────────── helpers ────────────────────────────────────
def get_style(kind):
    """Return (text_color, pill_bg_color, pill_alpha, is_pill, bold)."""
    palette = {
        "kw":     (C_KW,    None,     0,    False, False),
        "type":   (C_TYPE,  None,     0,    False, False),
        "mask":   ("white", C_MASK,   0.95, True,  True),
        "expand": ("white", C_EXPAND, 0.95, True,  True),
        "delete": ("white", C_DELETE, 0.95, True,  True),
        "reveal": (C_CODE,  C_REVEAL, 0.32, True,  False),
        "plain":  (C_CODE,  None,     0,    False, False),
    }
    return palette.get(kind, palette["plain"])


def draw_code_line(ax, tokens, y, font_size, x_start):
    """Render a list of (text, kind) tokens horizontally at row y."""
    PAD_X  = 0.18
    HALF_H = 0.32
    x = x_start
    for text, kind in tokens:
        color, bg, alpha, pill, bold = get_style(kind)
        w = len(text)
        if pill:
            rect = FancyBboxPatch(
                (x - PAD_X, y - HALF_H), w + 2 * PAD_X, 2 * HALF_H,
                boxstyle="round,pad=0.0,rounding_size=0.18",
                facecolor=bg, edgecolor='none', alpha=alpha, zorder=1,
            )
            ax.add_patch(rect)
        ax.text(x, y, text, family='monospace', fontsize=font_size,
                color=color, ha='left', va='center', zorder=3,
                fontweight='bold' if bold else 'normal')
        x += w
    return x - x_start


def tune_font_for_axes(fig, ax, fontsize_init):
    """Return the monospace font size that makes 1 char = 1 data unit on ax."""
    N = 40
    probe = ax.text(0, 0, "X" * N, family='monospace', fontsize=fontsize_init,
                    ha='left', va='bottom', alpha=0.0)
    fig.canvas.draw()
    bbox = probe.get_window_extent(renderer=fig.canvas.get_renderer())
    probe.remove()
    inv = ax.transData.inverted()
    x0 = inv.transform((bbox.x0, bbox.y0))[0]
    x1 = inv.transform((bbox.x1, bbox.y0))[0]
    observed = (x1 - x0)
    return fontsize_init * (N / observed)


# Each panel reserves PANEL_COLS chars of horizontal data space.
# 38 cols comfortably fits the widest line we render — namely
#   "int <|mask|><|mask|><|mask|> = 0;" = 33 chars
PANEL_COLS = 38


def draw_panel(fig, x0, y0, w, h, panel, accent, code_font):
    ax = fig.add_axes([x0, y0, w, h])
    ax.set_xlim(-0.4, PANEL_COLS + 0.4)
    ax.set_ylim(-0.10, 2.10)
    ax.axis('off')

    # background
    bg_patch = FancyBboxPatch(
        (0.0, 0.05), PANEL_COLS, 1.92,
        boxstyle="round,pad=0.0,rounding_size=0.35",
        facecolor=C_PANEL_BG, edgecolor=C_PANEL_BORDER, linewidth=0.6,
        zorder=0,
    )
    ax.add_patch(bg_patch)

    # left-edge accent stripe (matches sibling figure)
    stripe = mpatches.Rectangle(
        (0.10, 0.15), 0.30, 1.75,
        facecolor=accent, edgecolor='none', alpha=0.85, zorder=2,
    )
    ax.add_patch(stripe)

    # step badge (top, bold serif) with thin underline divider
    ax.text(0.95, 1.66, panel["step"],
            fontsize=7.6, color=C_BADGE, ha='left', va='center',
            fontweight='bold', family='serif')
    # divider line between badge and code area
    ax.plot([0.95, PANEL_COLS - 0.5], [1.42, 1.42],
            color=C_PANEL_BORDER, linewidth=0.4, zorder=1)

    # code line, centred
    code_chars = sum(len(t[0]) for t in panel["tokens"])
    x_start = max(1.0, (PANEL_COLS - code_chars) / 2)
    draw_code_line(ax, panel["tokens"], 0.95, code_font, x_start=x_start)

    # italic sub-caption (bottom)
    ax.text(PANEL_COLS / 2, 0.32, panel["sub"],
            fontsize=6.9, fontstyle='italic', color=C_LABEL_SUB,
            ha='center', va='center')


# ─────────────────────────────── figure ─────────────────────────────────────
FIG_W = 7.16   # IEEE figure*
FIG_H = 4.95   # title strip removed — figure goes straight from column headers to panels

fig = plt.figure(figsize=(FIG_W, FIG_H), dpi=200)

# column headers (no title above)
fig.text(0.252, 0.970,
         "EXPANSION   too few masks   →   grow canvas",
         fontsize=9.7, fontweight='bold', ha='center', va='top', color=C_EXPAND)
fig.text(0.752, 0.970,
         "DELETION   too many masks   →   shrink canvas",
         fontsize=9.7, fontweight='bold', ha='center', va='top', color=C_DELETE)

# panel placement
LEFT_X   = 0.025
RIGHT_X  = 0.515
COL_W    = 0.460
TOP_Y    = 0.935
BOT_Y    = 0.110
N_PANELS = 4
GAP_V    = 0.034
H_PANEL  = (TOP_Y - BOT_Y - GAP_V * (N_PANELS - 1)) / N_PANELS

# tune monospace font size ONCE — all 8 panels share the same axes geometry
_sample = fig.add_axes([0, 0, COL_W, H_PANEL])
_sample.set_xlim(-0.4, PANEL_COLS + 0.4)
_sample.set_ylim(-0.10, 2.10)
_sample.axis('off')
CODE_FONT = tune_font_for_axes(fig, _sample, 9.0)
fig.delaxes(_sample)

# draw all 8 panels and remember anchors for the arrow layer
panel_anchors = []
for i in range(N_PANELS):
    y_top = TOP_Y - i * (H_PANEL + GAP_V)
    y0    = y_top - H_PANEL
    panel_anchors.append((y_top, y0))
    draw_panel(fig, LEFT_X,  y0, COL_W, H_PANEL,
               EXPAND_TRAJ[i], C_EXPAND, CODE_FONT)
    draw_panel(fig, RIGHT_X, y0, COL_W, H_PANEL,
               DELETE_TRAJ[i], C_DELETE, CODE_FONT)

# downward arrows between row pairs, labelled with the operation that
# bridges the two snapshots
ARROW_LABELS = ["predict", "rule", "denoise"]

arrow_ax = fig.add_axes([0, 0, 1, 1])
arrow_ax.set_xlim(0, 1); arrow_ax.set_ylim(0, 1)
arrow_ax.axis('off'); arrow_ax.patch.set_alpha(0)

for col_center in (LEFT_X + COL_W / 2, RIGHT_X + COL_W / 2):
    for i in range(N_PANELS - 1):
        y_top_p = panel_anchors[i][1]      # bottom of panel i
        y_bot_p = panel_anchors[i + 1][0]  # top of panel i+1
        arrow = FancyArrowPatch(
            (col_center, y_top_p - 0.001),
            (col_center, y_bot_p + 0.001),
            arrowstyle='-|>,head_length=2.6,head_width=1.5',
            color=C_ARROW, linewidth=0.7, mutation_scale=7,
        )
        arrow_ax.add_patch(arrow)
        arrow_ax.text(col_center + 0.014,
                      (y_top_p + y_bot_p) / 2,
                      ARROW_LABELS[i],
                      fontsize=7.2, color="#5A5A5A",
                      ha='left', va='center', fontstyle='italic')

# legend
LEG_Y = 0.040
def legend_chip(x_left, color, alpha, label, label_dx=0.022):
    cax = fig.add_axes([x_left, LEG_Y - 0.012, 0.018, 0.024])
    cax.axis('off')
    cax.set_xlim(0, 1); cax.set_ylim(0, 1)
    cax.add_patch(FancyBboxPatch(
        (0.04, 0.10), 0.92, 0.80,
        boxstyle="round,pad=0.0,rounding_size=0.22",
        facecolor=color, alpha=alpha, edgecolor='none',
    ))
    fig.text(x_left + label_dx, LEG_Y, label,
             fontsize=7.4, color="#333", ha='left', va='center')

legend_chip(0.080, C_MASK,   0.95, "<|mask|>")
legend_chip(0.220, C_EXPAND, 0.95, "<|expand|>   →   two <|mask|>")
legend_chip(0.490, C_DELETE, 0.95, "<|delete|>   →   removed")
legend_chip(0.740, C_REVEAL, 0.32, "denoised token")

# save
out_pdf = os.path.join(OUTDIR, "fig_dreamon_canvas.pdf")
out_png = os.path.join(OUTDIR, "fig_dreamon_canvas.png")
fig.savefig(out_pdf, pad_inches=0.04)
fig.savefig(out_png, pad_inches=0.04, dpi=300)
plt.close(fig)
print(f"Wrote {out_pdf}")
print(f"Wrote {out_png}")
