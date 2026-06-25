"""
Generate fig_rq2_example.{pdf,png}: visual illustration of the RQ2
construction — obfuscating a RefineID sample into an alphabetic-token
form to drive multi-identifier multi-site renaming.

Top panel: the clean reference x*.  The RefineID-target identifier is
highlighted in green; every other lowercase-initial, non-single-char,
non-type, non-method renameable identifier is highlighted in yellow.
Together the green and yellow positions form the smell set S that the
all-masked condition targets in RQ2.

Bottom panel: the obfuscated x̃, where every renameable identifier is
replaced by a short alphabetic token (a, b, c, ...), applied
consistently across all of its occurrences.  Types, method names, and
keywords are preserved verbatim so that control flow and library
structure remain intact while every meaningful local name is gone.

Style sympathetic to analysis/make_dllm_demo_figure.py: IEEE figure*
(7.16 in) width, Times serif text, monospace code, rounded panel
backgrounds, pastel pills on identifier spans, and a slim downward
"obfuscate" arrow in the left margin between the two panels.
"""

import os
import re
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
C_TARGET       = "#06A77D"   # green — RefineID target identifier
C_OTHER        = "#FFD166"   # warm yellow — other renameable identifiers in S
C_CODE         = "#1B1B1B"
C_KW           = "#0F4F9C"   # blue keywords
C_TYPE         = "#6B2D8C"   # purple types
C_PANEL_BG     = "#FCFCFC"
C_PANEL_BORDER = "#D0D0D0"
C_ARROW        = "#9A9A9A"
C_LABEL        = "#1F1F1F"
C_LABEL_SUB    = "#5A5A5A"

plt.rcParams.update({
    "font.family":      "serif",
    "font.serif":       ["Times New Roman", "DejaVu Serif"],
    "font.size":        9,
    "mathtext.fontset": "dejavuserif",
    "pdf.fonttype":     42,
    "ps.fonttype":      42,
})

KEYWORDS = {"public", "new", "for", "if", "return", "int", "while",
            "void", "private", "static", "true", "false"}
TYPES    = {"List", "Integer", "ArrayList", "String", "Map", "User",
            "Double", "Boolean"}

# ─────────────────────────────── content ────────────────────────────────────
# (placeholder_tag, clean_name, obfuscated_token, is_target)
# RENAMES enumerates every renameable identifier in the example method.
# The middle row is the RefineID target — drawn in green in both panels.
RENAMES = [
    ("USERS",  "users",  "a", False),
    ("MIN",    "min",    "b", False),
    ("COUNT",  "count",  "c", True),    # RefineID target
    ("USER",   "user",   "d", False),
    ("SCORE",  "score",  "e", False),
]

NAME_BY_TAG = {tag: name for tag, name, _, _ in RENAMES}
OBF_BY_TAG  = {tag: obf  for tag, _, obf, _ in RENAMES}
IS_TARGET   = {tag: t    for tag, _, _, t in RENAMES}

# A small, fully-renderable Java snippet with exactly five renameable
# identifiers, plus types / method names / keywords that the obfuscator
# explicitly skips.  Max line length when expanded: 52 chars.
TEMPLATE = [
    "public int countActive(List<User> {USERS}, int {MIN}) {",
    "    int {COUNT} = 0;",
    "    for (User {USER} : {USERS}) {",
    "        int {SCORE} = {USER}.getScore();",
    "        if ({USER}.isActive() && {SCORE} > {MIN}) {COUNT}++;",
    "    }",
    "    return {COUNT};",
    "}",
]


# ────────────────────────── rendering helpers ───────────────────────────────
def render_panel(use_obf):
    """Substitute placeholders.  Return (rendered_lines, highlight_spans).

    highlight_spans : list[(line_idx, col_start, col_end, color)]
    """
    rendered, highlights = [], []
    for li, line in enumerate(TEMPLATE):
        out, col, pos = [], 0, 0
        while pos < len(line):
            m = re.search(r"\{([A-Z_]+)\}", line[pos:])
            if not m:
                out.append(line[pos:])
                break
            pre = line[pos:pos + m.start()]
            out.append(pre)
            col += len(pre)
            tag = m.group(1)
            sub = OBF_BY_TAG[tag] if use_obf else NAME_BY_TAG[tag]
            color = C_TARGET if IS_TARGET[tag] else C_OTHER
            highlights.append((li, col, col + len(sub), color))
            out.append(sub)
            col += len(sub)
            pos += m.end()
        rendered.append("".join(out))
    return rendered, highlights


def tokenize_for_color(line):
    """Yield (text, color) groups for word-level syntax highlighting."""
    for p in re.findall(r"\w+|\W+", line):
        if not p:
            continue
        if p in KEYWORDS:
            yield p, C_KW
        elif p in TYPES:
            yield p, C_TYPE
        else:
            yield p, C_CODE


CODE_COLS = 60   # internal char-width of each code panel (max line ≈ 52)


def tune_font_for_axes(fig, ax, fontsize_init):
    """Return the font size that makes 1 monospace char = 1 data unit on ax."""
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


def draw_code_panel(ax, use_obf, font_size):
    rendered, highlights = render_panel(use_obf)
    num_lines = len(TEMPLATE)

    ax.set_xlim(-0.6, CODE_COLS + 0.5)
    ax.set_ylim(num_lines + 0.05, -0.05)
    ax.axis("off")

    # rounded background panel
    panel = FancyBboxPatch(
        (-0.3, 0.06), CODE_COLS + 0.4, num_lines - 0.12,
        boxstyle="round,pad=0.02,rounding_size=0.30",
        facecolor=C_PANEL_BG, edgecolor=C_PANEL_BORDER, linewidth=0.6,
        zorder=0,
    )
    ax.add_patch(panel)

    # identifier highlights
    for li, c0, c1, color in highlights:
        rect = mpatches.Rectangle(
            (c0 - 0.04, li + 0.10), (c1 - c0) + 0.08, 0.80,
            facecolor=color, edgecolor='none', alpha=0.45, zorder=1,
        )
        ax.add_patch(rect)

    # code lines, token-by-token for cheap syntax colouring
    for li, line in enumerate(rendered):
        x_col = 0
        for tok, color in tokenize_for_color(line):
            ax.text(x_col, li + 0.5, tok,
                    family='monospace', fontsize=font_size,
                    color=color, ha='left', va='center', zorder=3)
            x_col += len(tok)


# ─────────────────────────────── figure ─────────────────────────────────────
FIG_W = 7.16     # IEEE figure*
FIG_H = 5.00

fig = plt.figure(figsize=(FIG_W, FIG_H), dpi=200)

# layout (fig-fraction units)
LEFT      = 0.005
LABEL_W   = 0.115
GAP_LR    = 0.015
PANEL_L   = LEFT + LABEL_W + GAP_LR
PANEL_R   = 0.985

TOP_Y         = 0.935    # top of clean panel
PANEL_H_FRAC  = 0.285    # per panel
GAP_V         = 0.075    # vertical gap between the two panels

PANEL1_TOP = TOP_Y
PANEL1_BOT = TOP_Y - PANEL_H_FRAC
PANEL2_TOP = PANEL1_BOT - GAP_V
PANEL2_BOT = PANEL2_TOP - PANEL_H_FRAC

# tune monospace font once on a sample axes (both code panels share size)
_sample = fig.add_axes([0, 0, PANEL_R - PANEL_L, PANEL_H_FRAC])
_sample.set_xlim(-0.6, CODE_COLS + 0.5)
_sample.set_ylim(len(TEMPLATE) + 0.05, -0.05)
_sample.axis('off')
CODE_FONT = tune_font_for_axes(fig, _sample, 9.0)
fig.delaxes(_sample)


def draw_panel_with_label(fig, y0, y_top, math_label, sub_label,
                          accent, use_obf):
    """Draw one labelled code panel and its accent stripe."""
    h = y_top - y0

    # code axes
    ax = fig.add_axes([PANEL_L, y0, PANEL_R - PANEL_L, h])
    ax.set_xlim(-0.6, CODE_COLS + 0.5)
    ax.set_ylim(len(TEMPLATE) + 0.05, -0.05)
    draw_code_panel(ax, use_obf=use_obf, font_size=CODE_FONT)

    # left label strip
    lax = fig.add_axes([LEFT, y0, LABEL_W, h])
    lax.set_xlim(0, 1)
    lax.set_ylim(0, 1)
    lax.axis('off')

    # accent stripe
    stripe = mpatches.Rectangle((0.92, 0.08), 0.06, 0.84,
                                facecolor=accent, edgecolor='none', alpha=0.9)
    lax.add_patch(stripe)

    lax.text(0.06, 0.62, math_label,
             fontsize=15.5, fontweight='bold',
             color=C_LABEL, ha='left', va='center')
    lax.text(0.06, 0.36, sub_label,
             fontsize=7.6, fontstyle='italic',
             color=C_LABEL_SUB, ha='left', va='center')


draw_panel_with_label(fig, PANEL1_BOT, PANEL1_TOP,
                      r"$\mathbf{x}^{\ast}$",
                      "clean reference",
                      C_TARGET, use_obf=False)

draw_panel_with_label(fig, PANEL2_BOT, PANEL2_TOP,
                      r"$\tilde{\mathbf{x}}$",
                      "obfuscated",
                      C_OTHER, use_obf=True)

# slim downward "obfuscate" arrow in the left margin between the two panels
arrow_ax = fig.add_axes([0, 0, 1, 1])
arrow_ax.set_xlim(0, 1)
arrow_ax.set_ylim(0, 1)
arrow_ax.axis('off')
arrow_ax.patch.set_alpha(0)
arrow_x   = LEFT + LABEL_W / 2
arrow_top = PANEL1_BOT - 0.003
arrow_bot = PANEL2_TOP + 0.003
arrow = FancyArrowPatch(
    (arrow_x, arrow_top), (arrow_x, arrow_bot),
    arrowstyle='-|>,head_length=2.8,head_width=1.6',
    color=C_ARROW, linewidth=0.7, mutation_scale=6,
)
arrow_ax.add_patch(arrow)
arrow_ax.text(arrow_x + 0.012, (arrow_top + arrow_bot) / 2,
              "obfuscate",
              fontsize=7.0, fontstyle='italic', color="#666",
              ha='left', va='center')

# renaming map below the bottom panel
fig.text(0.50, 0.180,
         (r"renaming map:"
          r"     users $\rightarrow$ a"
          r"     min $\rightarrow$ b"
          r"     count $\rightarrow$ c "
          r"$\mathit{(target)}$"
          r"     user $\rightarrow$ d"
          r"     score $\rightarrow$ e"),
         fontsize=8.6, ha='center', va='center', color=C_CODE,
         family='serif')

# bottom legend
LEG_Y = 0.062
def chip(x_center, color, label, alpha=0.50):
    cax = fig.add_axes([x_center - 0.012, LEG_Y - 0.010, 0.024, 0.020])
    cax.axis("off")
    cax.set_xlim(0, 1); cax.set_ylim(0, 1)
    cax.add_patch(mpatches.Rectangle((0, 0), 1, 1,
                                     facecolor=color, alpha=alpha,
                                     edgecolor='none'))
    fig.text(x_center + 0.016, LEG_Y, label, fontsize=7.6,
             color="#333", ha='left', va='center')

chip(0.205, C_TARGET, "RefineID target identifier")
chip(0.510, C_OTHER,  r"other renameables in smell set $\mathcal{S}$")

# save
out_pdf = os.path.join(OUTDIR, "fig_rq2_example.pdf")
out_png = os.path.join(OUTDIR, "fig_rq2_example.png")
fig.savefig(out_pdf, pad_inches=0.04)
fig.savefig(out_png, pad_inches=0.04, dpi=300)
plt.close(fig)
print(f"Wrote {out_pdf}")
print(f"Wrote {out_png}")
