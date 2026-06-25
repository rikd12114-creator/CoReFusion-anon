"""
Generate fig_dllm_demo.pdf: Iterative denoising in a diffusion LLM
applied to multi-site variable renaming.

Four snapshots track a single Java method whose five <|mask|> tokens
are all occurrences of the SAME final identifier — `count` — appearing
in five different syntactic roles inside `countActive(List<User>)`:

    S1 declaration              int <|mask|> = 0;
    S2 increment                if (u.isActive()) <|mask|>++;
    S3 guard condition          if (<|mask|> > 0) ...
    S4 println argument         ... System.out.println(<|mask|>);
    S5 return value             return <|mask|>;

State lifecycle for each site:

    <|mask|>    →   low-confidence prediction   (pale yellow)
                →   high-confidence prediction  (orange)
                →   committed                   (green, locked-in)

Crucially, the *content* of the predicted token at low- and high-conf
states is independent of the final committed answer:

    low_conf    : prediction may differ from the final answer
                  (often does — the model is just guessing).
    high_conf   : prediction usually matches the final answer, but
                  with low probability can still differ.
    committed   : prediction is GUARANTEED to be the final answer.

The PREDICTED_AT_STEP dict below realises this: at t=T/3 DECL guesses
`cnt` and INCR guesses `n`; at t=2T/3 PRINT (low) guesses `tot` and
RETURN (high) confidently misfires on `ctr`; by t=T all five sites
have committed to `count`.
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
C_MASK         = "#E63946"   # red for <|mask|>
C_LOW_CONF     = "#FFD166"   # pale warm yellow for low-confidence predictions
C_HIGH_CONF    = "#F77F00"   # orange for high-confidence predictions
C_COMMITTED    = "#06A77D"   # green for committed (locked-in) tokens
C_CODE         = "#1B1B1B"
C_KW           = "#0F4F9C"   # blue keywords
C_TYPE         = "#6B2D8C"   # purple types
C_STR_LITERAL  = "#A05A2C"
C_PANEL_BG     = "#FCFCFC"
C_PANEL_BORDER = "#D0D0D0"
C_LABEL        = "#1F1F1F"
C_LABEL_SUB    = "#5A5A5A"
C_CONF         = "#444444"
C_ARROW        = "#9A9A9A"

# Per-state colour + pill alpha — lower alpha for low-conf to read as
# "tentative", higher for high-conf to read as "confident"
STATE_COLORS = {
    "mask":      C_MASK,
    "low_conf":  C_LOW_CONF,
    "high_conf": C_HIGH_CONF,
    "committed": C_COMMITTED,
}
STATE_ALPHA = {
    "mask":      0.55,
    "low_conf":  0.50,
    "high_conf": 0.70,
    "committed": 0.55,
}

KEYWORDS = {"public", "new", "for", "if", "return", "int", "while", "void", "private", "static"}
TYPES    = {"List", "Integer", "ArrayList", "String", "Map", "User"}

plt.rcParams.update({
    "font.family":  "serif",
    "font.serif":   ["Times New Roman", "DejaVu Serif"],
    "font.size":    9,
    "pdf.fonttype": 42,
    "ps.fonttype":  42,
})

# ─────────────────────────────── content ────────────────────────────────────
# All five <|mask|> tokens resolve to the SAME final identifier.
TARGET_NAME = "count"

IDS = {
    "DECL":   TARGET_NAME,
    "INCR":   TARGET_NAME,
    "COND":   TARGET_NAME,
    "PRINT":  TARGET_NAME,
    "RETURN": TARGET_NAME,
}

# A short syntactic-role label shown for each site in the state board.
SITE_LABELS = {
    "DECL":   "decl",
    "INCR":   "++",
    "COND":   "cond",
    "PRINT":  "print",
    "RETURN": "return",
}

# State of each site at each snapshot.
# Index 0 = t=0, 1 = t=T/3, 2 = t=2T/3, 3 = t=T.
STATE_AT_STEP = {
    "DECL":   ["mask", "low_conf", "committed", "committed"],
    "INCR":   ["mask", "low_conf", "committed", "committed"],
    "COND":   ["mask", "mask",     "high_conf", "committed"],
    "PRINT":  ["mask", "mask",     "low_conf",  "committed"],
    "RETURN": ["mask", "low_conf", "high_conf", "committed"],
}

# Predicted name at each snapshot.  Low-/high-conf predictions can differ
# from TARGET_NAME (low: often does; high: rarely does); committed must
# equal TARGET_NAME.  None when the site is still masked.
PREDICTED_AT_STEP = {
    "DECL":   [None, "cnt",   "count", "count"],
    "INCR":   [None, "n",     "count", "count"],
    "COND":   [None, None,    "count", "count"],
    "PRINT":  [None, None,    "tot",   "count"],
    "RETURN": [None, "count", "ctr",   "count"],
}

# Predicted probability for each site at each snapshot.
CONFIDENCE_AT_STEP = {
    "DECL":   [None, 0.62, 0.95, 0.95],
    "INCR":   [None, 0.58, 0.93, 0.93],
    "COND":   [None, None, 0.78, 0.88],
    "PRINT":  [None, None, 0.55, 0.84],
    "RETURN": [None, 0.54, 0.81, 0.91],
}

# Display order of sites in the side state board (matches their visual
# top-to-bottom order in the source code).
SIDE_ORDER = ["DECL", "INCR", "COND", "PRINT", "RETURN"]

TEMPLATE = [
    "public int countActive(List<User> users) {",
    "    int {DECL} = 0;",
    "    for (User u : users)",
    "        if (u.isActive()) {INCR}++;",
    "    if ({COND} > 0) System.out.println({PRINT});",
    "    return {RETURN};",
    "}",
]

MASK_TOKEN     = "<|mask|>"
FONT_SIZE_INIT = 9.0   # auto-tuned so 1 monospace char = 1 data unit


# ────────────────────────── rendering helpers ───────────────────────────────
def render_state_at_step(step):
    """Return (rendered_lines, highlight_spans).

    highlight_spans : list[(line_idx, col_start, col_end,
                            state, color, conf, tag)]
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
            tag   = m.group(1)
            state = STATE_AT_STEP[tag][step]
            conf  = CONFIDENCE_AT_STEP[tag][step]
            sub   = MASK_TOKEN if state == "mask" else PREDICTED_AT_STEP[tag][step]
            color = STATE_COLORS[state]
            highlights.append((li, col, col + len(sub), state, color, conf, tag))
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


# Sized so the widest rendered line — "    if (<|mask|> > 0) System.out.println(<|mask|>);"
# = 51 chars — fits comfortably with whitespace to spare.
CODE_COLS  = 60
SIDE_COLS  = 18
GAP_COLS   = 2.0


def tune_font_for_axes(fig, ax, fontsize_init):
    """Return the font size that makes 1 monospace char = 1 data unit."""
    N = 40
    probe = ax.text(0, 0, "X" * N, family='monospace', fontsize=fontsize_init,
                    ha='left', va='bottom', alpha=0.0)
    fig.canvas.draw()
    bbox = probe.get_window_extent(renderer=fig.canvas.get_renderer())
    probe.remove()
    inv = ax.transData.inverted()
    x0 = inv.transform((bbox.x0, bbox.y0))[0]
    x1 = inv.transform((bbox.x1, bbox.y0))[0]
    observed_chars = (x1 - x0)
    return fontsize_init * (N / observed_chars)


def draw_code_panel(ax, step, font_size):
    rendered, highlights = render_state_at_step(step)
    num_lines = len(TEMPLATE)
    total_cols = CODE_COLS + GAP_COLS + SIDE_COLS

    ax.set_xlim(-0.6, total_cols + 0.5)
    ax.set_ylim(num_lines + 0.05, -0.05)
    ax.axis("off")

    # rounded panel background (covers code + side area)
    panel = FancyBboxPatch(
        (-0.3, 0.06), total_cols + 0.4, num_lines - 0.12,
        boxstyle="round,pad=0.02,rounding_size=0.30",
        facecolor=C_PANEL_BG, edgecolor=C_PANEL_BORDER, linewidth=0.6,
        zorder=0,
    )
    ax.add_patch(panel)

    # thin separator between code and side panel
    sep_x = CODE_COLS + GAP_COLS / 2
    ax.plot([sep_x, sep_x], [0.20, num_lines - 0.20],
            color=C_PANEL_BORDER, linewidth=0.5, zorder=0)

    # highlight backgrounds behind each identifier span
    for li, c0, c1, state, color, _conf, _tag in highlights:
        rect = mpatches.Rectangle(
            (c0 - 0.04, li + 0.10), (c1 - c0) + 0.08, 0.80,
            facecolor=color, edgecolor='none',
            alpha=STATE_ALPHA[state], zorder=1,
        )
        ax.add_patch(rect)

    # code lines — render token-by-token for cheap syntax colouring
    for li, line in enumerate(rendered):
        x_col = 0
        for tok, color in tokenize_for_color(line):
            ax.text(x_col, li + 0.5, tok,
                    family='monospace', fontsize=font_size,
                    color=color, ha='left', va='center', zorder=3)
            x_col += len(tok)

    # ── side panel: state board for all 5 occurrences ───────────────────────
    side_x = CODE_COLS + GAP_COLS
    side_w = SIDE_COLS

    # two-column header
    ax.text(side_x, 0.42, "site state",
            family='serif', fontsize=6.8, fontstyle='italic',
            color="#666", ha='left', va='center')
    ax.text(side_x + side_w - 0.3, 0.42, "p",
            family='serif', fontsize=6.8, fontstyle='italic',
            color="#666", ha='right', va='center')

    # one row per site, in source order
    n_rows = len(SIDE_ORDER)
    row_h  = (num_lines - 1.1) / n_rows
    for i, tag in enumerate(SIDE_ORDER):
        state = STATE_AT_STEP[tag][step]
        conf  = CONFIDENCE_AT_STEP[tag][step]
        color = STATE_COLORS[state]
        role  = SITE_LABELS[tag]
        row_y = 1.05 + i * row_h + row_h / 2

        chip = mpatches.Rectangle(
            (side_x, row_y - 0.18), 0.8, 0.36,
            facecolor=color, edgecolor='none',
            alpha=min(STATE_ALPHA[state] + 0.20, 0.95), zorder=2,
        )
        ax.add_patch(chip)

        ax.text(side_x + 1.3, row_y, role,
                family='monospace', fontsize=font_size - 1.5,
                color=C_CODE, ha='left', va='center')

        conf_text = f"{conf:.2f}" if conf is not None else "—"
        ax.text(side_x + side_w - 0.3, row_y, conf_text,
                family='serif', fontsize=7.4, fontstyle='italic',
                color=C_CONF, ha='right', va='center')


# ─────────────────────────────── figure ─────────────────────────────────────
FIG_W = 7.16   # IEEE figure*
FIG_H = 5.40   # no title strip — panels go straight from the top of the figure

fig = plt.figure(figsize=(FIG_W, FIG_H), dpi=200)

LEFT       = 0.005
LABEL_W    = 0.150
GAP_LR     = 0.015
PANEL_L    = LEFT + LABEL_W + GAP_LR
PANEL_R    = 0.995
TOP        = 0.970   # panels start near the top of the figure (no title)
BOT        = 0.080
N_PANELS   = 4
GAP_V      = 0.020
TOTAL_H    = TOP - BOT
PANEL_H    = (TOTAL_H - GAP_V * (N_PANELS - 1)) / N_PANELS

# Each panel: (step_idx, main_label, accent_color)
# Accent colour tracks the *novel* state introduced at that snapshot.
PANELS = [
    (0, "t = 0",     C_MASK),
    (1, "t = T/3",   C_LOW_CONF),
    (2, "t = 2T/3",  C_HIGH_CONF),
    (3, "t = T",     C_COMMITTED),
]

panel_anchors = []
for i, (step, lbl_main, accent) in enumerate(PANELS):
    top = TOP - i * (PANEL_H + GAP_V)
    bot = top - PANEL_H
    panel_anchors.append((top, bot))

    # code panel axes
    ax = fig.add_axes([PANEL_L, bot, PANEL_R - PANEL_L, PANEL_H])
    total_cols = CODE_COLS + GAP_COLS + SIDE_COLS
    ax.set_xlim(-0.6, total_cols + 0.5)
    ax.set_ylim(len(TEMPLATE) + 0.05, -0.05)
    font_size = tune_font_for_axes(fig, ax, FONT_SIZE_INIT)
    draw_code_panel(ax, step, font_size)

    # label axes (left strip)
    lax = fig.add_axes([LEFT, bot, LABEL_W, PANEL_H])
    lax.axis("off")
    lax.set_xlim(0, 1)
    lax.set_ylim(0, 1)

    # accent stripe
    stripe = mpatches.Rectangle((0.93, 0.06), 0.055, 0.88,
                                facecolor=accent, edgecolor='none', alpha=0.9)
    lax.add_patch(stripe)

    # main label only (sub-label removed per spec)
    lax.text(0.06, 0.50, lbl_main, fontsize=10.5, fontweight='bold',
             color=C_LABEL, ha='left', va='center')

# slim downward "denoising" arrows in the left margin between panels
arrow_ax = fig.add_axes([0, 0, 1, 1])
arrow_ax.set_xlim(0, 1)
arrow_ax.set_ylim(0, 1)
arrow_ax.axis("off")
arrow_ax.patch.set_alpha(0)
arrow_x = LEFT + LABEL_W / 2
for i in range(N_PANELS - 1):
    y_top = panel_anchors[i][1]
    y_bot = panel_anchors[i + 1][0]
    arrow = FancyArrowPatch(
        (arrow_x, y_top - 0.002),
        (arrow_x, y_bot + 0.002),
        arrowstyle='-|>,head_length=2.8,head_width=1.6',
        color=C_ARROW, linewidth=0.7, mutation_scale=6,
    )
    arrow_ax.add_patch(arrow)
arrow_ax.text(arrow_x + 0.012,
              (panel_anchors[0][1] + panel_anchors[1][0]) / 2,
              "denoising", fontsize=6.3, color=C_ARROW,
              ha='left', va='center', fontstyle='italic', rotation=0)

# bottom legend strip
LEG_Y = 0.030
def chip(x_center, color, label, alpha=0.55):
    chip_ax = fig.add_axes([x_center - 0.012, LEG_Y - 0.010, 0.024, 0.020])
    chip_ax.axis("off")
    chip_ax.set_xlim(0, 1); chip_ax.set_ylim(0, 1)
    chip_ax.add_patch(mpatches.Rectangle((0, 0), 1, 1,
                                         facecolor=color, alpha=alpha,
                                         edgecolor='none'))
    fig.text(x_center + 0.016, LEG_Y, label, fontsize=7.6,
             color="#333", ha='left', va='center')

chip(0.07, C_MASK,      "<|mask|>")
chip(0.22, C_LOW_CONF,  "low confidence")
chip(0.42, C_HIGH_CONF, "high confidence")
chip(0.63, C_COMMITTED, "committed")
fig.text(0.86, LEG_Y, "p = predicted probability",
         fontsize=7.6, color="#333", ha='center', va='center',
         fontstyle='italic')

# save
out_pdf = os.path.join(OUTDIR, "fig_dllm_demo.pdf")
out_png = os.path.join(OUTDIR, "fig_dllm_demo.png")
fig.savefig(out_pdf, pad_inches=0.04)
fig.savefig(out_png, pad_inches=0.04, dpi=300)
plt.close(fig)
print(f"Wrote {out_pdf}")
print(f"Wrote {out_png}")
