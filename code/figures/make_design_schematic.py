#!/usr/bin/env python3
"""Evaluation-design schematic: three settings on one real item (MedMCQA
"vasodilation", key C), real GPT-5.6 Luna outputs. Each setting is its own
bordered panel; the withheld block is the visual focus of each card.
Explanatory text (source item, what each 'hidden' means) lives in the caption."""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
from matplotlib.transforms import Bbox

OUT = os.path.expanduser("~/Downloads/paper-figures/creative")
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({
    "pdf.fonttype": 42, "ps.fonttype": 42, "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 8, "figure.dpi": 300, "savefig.dpi": 300,
    "savefig.bbox": "standard", "hatch.linewidth": 0.6,
})
INK, MUT, LINE = "#151B21", "#5C6B75", "#CBD4DA"
TEAL, AMBER, PLUM = "#0F6E8C", "#C77A30", "#7E5A86"          # setting identities
ANS = {TEAL: "#E9F2F5", AMBER: "#F7EADB", PLUM: "#EFE8F1"}

QLINES = ["Which of the following", "causes vasodilation?"]
OPTS = ["Thromboxane A2", "Prostaglandin E2", "Histamine", "Serotonin"]
COLX, CW = [0.018, 0.345, 0.672], 0.310

PT, PB, HEAD_H, DIV = 0.900, 0.292, 0.072, 0.492            # panel geometry

fig = plt.figure(figsize=(6.6, 2.40))
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

# ---- source bar + connectors ----
BAR_Y, BAR_H = 0.926, 0.058
ax.add_patch(FancyBboxPatch((0.205, BAR_Y), 0.590, BAR_H,
                            boxstyle="round,pad=0.004,rounding_size=0.018",
                            fc="#F1F4F6", ec=LINE, lw=0.8))
byc = BAR_Y + BAR_H / 2
ax.text(0.372, byc, "2,773 questions", ha="right", va="center",
        fontsize=8.2, fontweight="bold", color=INK)
ax.text(0.372, byc, "   ·   MedQA-USMLE   ·   MedMCQA   ·   PubMedQA",
        ha="left", va="center", fontsize=7.8, color=MUT)
ax.plot([COLX[0] + CW / 2, COLX[2] + CW / 2], [0.912, 0.912], color=LINE, lw=0.9)
for x0 in COLX:
    ax.plot([x0 + CW / 2, x0 + CW / 2], [PT, BAR_Y], color=LINE, lw=0.9)

def redact(x, y, w, h, label, col):
    """Prominent 'withheld' block: the focal point of the card. Single bold
    label; the meaning of 'hidden' is spelled out in the caption."""
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.012",
                                fc=col, ec="none", alpha=0.13, zorder=3))
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.012",
                                fc="none", ec=col, lw=1.4, ls=(0, (4, 2)), zorder=4,
                                hatch="////"))
    pw, ph = w * 0.86, 0.052
    ax.add_patch(FancyBboxPatch((x + (w - pw) / 2, y + h / 2 - ph / 2), pw, ph,
                                boxstyle="round,pad=0.004,rounding_size=0.012",
                                fc="white", ec="none", zorder=5))
    ax.text(x + w / 2, y + h / 2, label, ha="center", va="center",
            fontsize=8.2, fontweight="bold", color=col, zorder=6)

def opts_block(bx):
    oy = 0.678
    for lab, txt in zip("ABCD", OPTS):
        hot = lab == "C"
        ax.add_patch(Rectangle((bx, oy - 0.013), 0.026, 0.026,
                               fc=TEAL if hot else "#DEE6EA", ec="none", zorder=3))
        ax.text(bx + 0.013, oy, lab, ha="center", va="center", fontsize=5.8,
                color="white" if hot else MUT, zorder=4)
        ax.text(bx + 0.038, oy, txt, ha="left", va="center", fontsize=6.6,
                color=INK, fontweight="bold" if hot else "normal", zorder=4)
        oy -= 0.049

def column(x0, accent, num, name, measures, stem, opts, ans):
    xc = x0 + CW / 2
    # ---- outer panel (the box) ----
    ax.add_patch(FancyBboxPatch((x0, PB), CW, PT - PB,
                                boxstyle="round,pad=0,rounding_size=0.020",
                                fc="white", ec=accent, lw=1.4, zorder=1))
    ax.add_patch(FancyBboxPatch((x0 + 0.008, PB + 0.008), CW - 0.016, DIV - PB - 0.014,
                                boxstyle="round,pad=0,rounding_size=0.014",
                                fc=ANS[accent], ec="none", zorder=2))
    # heading bar
    ax.add_patch(FancyBboxPatch((x0 + 0.009, PT - 0.009 - HEAD_H), CW - 0.018, HEAD_H,
                                boxstyle="round,pad=0,rounding_size=0.012",
                                fc=accent, ec="none", zorder=3))
    hy = PT - 0.009 - HEAD_H / 2
    ax.text(x0 + 0.020, hy, f"Setting {num}", ha="left", va="center",
            fontsize=9.6, fontweight="bold", color="white", zorder=4)
    ax.text(x0 + CW - 0.020, hy, name, ha="right", va="center",
            fontsize=8.5, color="white", zorder=4)

    bx = x0 + 0.026
    # ---- QUESTION region (enlarged) ----
    if stem:
        yy = 0.788
        for ln in QLINES:
            ax.text(bx, yy, ln, ha="left", va="center", fontsize=7.3, color=INK, zorder=4)
            yy -= 0.042
    else:
        redact(bx - 0.004, 0.720, CW - 0.044, 0.082, "Question hidden", accent)
    # ---- OPTIONS region ----
    if opts:
        opts_block(bx)
    else:
        redact(bx - 0.004, 0.516, CW - 0.044, 0.180, "Answer choices hidden", accent)

    # divider + flow chevron
    ax.plot([x0 + 0.030, x0 + CW - 0.030], [DIV, DIV], color=accent, lw=0.8,
            alpha=0.55, zorder=4)
    ax.scatter([xc], [DIV], marker="v", s=26, color=accent, ec="white", lw=0.6, zorder=5)

    # ---- answer ----
    ax.text(xc, 0.456, "The model answers", ha="center", va="center",
            fontsize=6.3, color=accent, fontweight="bold", zorder=4)
    n = len(ans); dy = 0.052
    y = 0.366 + (n - 1) * dy / 2
    for ln, bold in ans:
        ax.text(xc, y, ln, ha="center", va="center",
                fontsize=7.2 if bold else 6.5, color=INK,
                fontweight="bold" if bold else "normal", zorder=4)
        y -= dy

column(COLX[0], TEAL, "1", "Multiple choice", "measures recognition", True, True,
       [("(C)  Histamine", True), ("with a short rationale", False)])
column(COLX[1], AMBER, "2", "Generative", "measures recall", True, False,
       [("“Nitric oxide”", True), ("a vasodilator — but not among the choices", False)])
column(COLX[2], PLUM, "3", "Reconstruction", "measures familiarity", False, True,
       [("“Which mediator promotes platelet", True),
        ("aggregation and vasoconstriction?”", True)])

# crop to the content box (the axes fills the figure, so trim the empty
# band below the captions with an explicit inch-space bbox)
W, H = 6.6, 2.40
crop = Bbox.from_extents(0.006 * W, 0.284 * H, 0.994 * W, 0.988 * H)
fig.savefig(os.path.join(OUT, "Schematic_evaluation_design.pdf"), bbox_inches=crop)
fig.savefig(os.path.join(OUT, "Schematic_evaluation_design.png"), dpi=190, bbox_inches=crop)
print("wrote Schematic_evaluation_design")
