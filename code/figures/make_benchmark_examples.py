#!/usr/bin/env python3
"""One real example question from each of the three benchmarks.

Companion to Table 1: the table gives the counts, this gives the flavour of
each question type. Every string below is verbatim from the evaluation set --
MedQA-USMLE test split, the seed-0 MedMCQA validation sample, and the PubMedQA
expert-annotated test split (PMID 18719011). The PubMedQA abstract is the only
abbreviated field, marked with an ellipsis.

Benchmarks are deliberately NOT coloured teal/amber/plum: those three identify
the three evaluation settings elsewhere in the paper. Green marks the key only.

Two things keep this figure tight. Lines are wrapped against a *measured* text
width rather than broken by hand, so no line stops short of the panel edge; and
panel heights are computed from the resulting line counts in points before any
conversion to figure fractions, so no panel can silently squeeze another.
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
from matplotlib.transforms import Bbox

OUT = os.path.expanduser("~/Downloads/paper-figures")
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({
    "pdf.fonttype": 42, "ps.fonttype": 42, "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 8, "figure.dpi": 300, "savefig.dpi": 300,
    "savefig.bbox": "standard",
})
INK, MUT, LINE = "#151B21", "#5C6B75", "#D6DEE3"
HEAD, KEY, PALE = "#33454F", "#2E7D5B", "#F3F6F7"

# ------------------------------------------------- type scale (points)
# One proportional ladder, applied identically in all three panels:
#   name > stem > choices = abstract > meta = kind > letter badge
FS_NAME, FS_STEM, FS_BODY, FS_META, FS_BADGE = 9.2, 8.0, 7.2, 7.0, 6.2

# --------------------------------------------------------------- the examples
BENCH = [
    dict(
        name="MedQA-USMLE",
        meta="US medical licensing examination   ·   four answer choices",
        kind="A clinical scenario describing one patient, followed by a question about "
             "diagnosis, management, or mechanism.",
        stem="A 72-year-old anthropologist with long-standing hypertension visits your office "
             "for a routine exam. You notice an abnormality on his laboratory results caused by "
             "his regimen of captopril and triamterene. What abnormality did you most likely "
             "find?",
        opts=[("A", "Hyperkalemia"), ("B", "Hypernatremia"),
              ("C", "Thrombocytopenia"), ("D", "Anemia")],
        key="A",
    ),
    dict(
        name="MedMCQA",
        meta="Indian medical entrance examinations   ·   four answer choices",
        kind="A single recalled fact from one of 21 subjects, phrased tersely and without a case.",
        stem="Cells most commonly affected in glaucomatous optic atrophy?",
        opts=[("A", "Amacrine cells"), ("B", "Bipolar cells"),
              ("C", "Ganglion cells"), ("D", "Rods and cones")],
        key="C",
    ),
    dict(
        name="PubMedQA",
        meta="PubMed abstracts   ·   yes / no / maybe",
        kind="A research question answered from the abstract of the study that asked it.",
        ctx="To compare growth curves of body mass index from children to adolescents, and then "
            "to young adults, in Japanese girls and women in birth cohorts born from 1930 to "
            "1999. … More recent cohorts were more overweight as children but thinner as young "
            "women.",
        stem="Do overweight children necessarily make overweight adults?",
        opts=[("", "yes"), ("", "no"), ("", "maybe")],
        key="no",
    ),
]

# ------------------------------------------------------- measured word wrap
_probe = plt.figure(figsize=(4, 4))
_pax = _probe.add_axes([0, 0, 1, 1])
_rend = _probe.canvas.get_renderer()


def text_width_in(s, fs, style="normal", weight="normal"):
    t = _pax.text(0, 0, s, fontsize=fs, style=style, fontweight=weight)
    w = t.get_window_extent(renderer=_rend).width / _probe.dpi
    t.remove()
    return w


def wrap(s, fs, max_in, **kw):
    """Greedy wrap against the real rendered width, not a character count."""
    words, lines, cur = s.split(), [], ""
    for w in words:
        trial = w if not cur else cur + " " + w
        if cur and text_width_in(trial, fs, **kw) > max_in:
            lines.append(cur); cur = w
        else:
            cur = trial
    if cur:
        lines.append(cur)
    return lines


W = 6.6
X0, X1 = 0.010, 0.990
PAD_X = 0.013                                    # panel side padding (fraction)
TEXT_IN = (X1 - X0 - 2 * PAD_X) * W              # usable text width, inches
CTX_IN = TEXT_IN - 0.020 * W                     # abstract box is inset further

for s in BENCH:
    s["kind_l"] = wrap(s["kind"], FS_META, TEXT_IN, style="italic")
    s["stem_l"] = wrap(s["stem"], FS_STEM, TEXT_IN)
    s["ctx_l"] = wrap(s["ctx"], FS_BODY, CTX_IN) if s.get("ctx") else []

plt.close(_probe)

# ------------------------------------------------------- vertical metrics (pt)
HEAD_PT, KIND_PT, STEM_PT, CTX_PT = 14.0, 10.0, 10.5, 9.6
CTXPAD_PT, GAP_PT, CHOICE_PT, PAD_PT, BETWEEN = 4.5, 5.0, 12.0, 6.5, 6.0

for s in BENCH:
    h = (HEAD_PT + 3.0 + KIND_PT * len(s["kind_l"]) + 3.0
         + STEM_PT * len(s["stem_l"]) + GAP_PT + CHOICE_PT + PAD_PT)
    if s["ctx_l"]:
        h += CTX_PT * len(s["ctx_l"]) + 2 * CTXPAD_PT + 5.0
    s["h_pt"] = h

H = (sum(s["h_pt"] for s in BENCH) + BETWEEN * (len(BENCH) - 1) + 3.0) / 72.0

fig = plt.figure(figsize=(W, H))
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")


def pt(x):
    """Point offset -> figure fraction, so spacing survives a height change."""
    return x / 72.0 / H


top = 1.0 - pt(1.5)

for spec in BENCH:
    bot = top - pt(spec["h_pt"])
    hh = pt(HEAD_PT)

    # ---- panel shell + heading strip ----
    ax.add_patch(FancyBboxPatch((X0, bot), X1 - X0, top - bot,
                                boxstyle="round,pad=0,rounding_size=0.010",
                                fc="white", ec=LINE, lw=0.8, zorder=1))
    ax.add_patch(FancyBboxPatch((X0, top - hh), X1 - X0, hh,
                                boxstyle="round,pad=0,rounding_size=0.010",
                                fc=HEAD, ec="none", zorder=2))
    ax.add_patch(Rectangle((X0, top - hh), X1 - X0, pt(4), fc=HEAD, ec="none", zorder=2))
    hy = top - hh / 2
    ax.text(X0 + PAD_X, hy, spec["name"], ha="left", va="center",
            fontsize=FS_NAME, fontweight="bold", color="white", zorder=3)
    ax.text(X1 - PAD_X, hy, spec["meta"], ha="right", va="center",
            fontsize=FS_META, color="#C2D0D7", zorder=3)

    y = top - hh - pt(3.0)
    for ln in spec["kind_l"]:
        y -= pt(KIND_PT / 2)
        ax.text(X0 + PAD_X, y, ln, ha="left", va="center",
                fontsize=FS_META, color=MUT, style="italic", zorder=3)
        y -= pt(KIND_PT / 2)
    y -= pt(3.0)

    # ---- abstract excerpt (PubMedQA only) ----
    if spec["ctx_l"]:
        box_h = pt(CTX_PT * len(spec["ctx_l"]) + 2 * CTXPAD_PT)
        ax.add_patch(FancyBboxPatch((X0 + PAD_X - 0.004, y - box_h), X1 - X0 - 2 * PAD_X + 0.008,
                                    box_h, boxstyle="round,pad=0,rounding_size=0.008",
                                    fc=PALE, ec="none", zorder=2))
        yy = y - pt(CTXPAD_PT + CTX_PT / 2)
        for ln in spec["ctx_l"]:
            ax.text(X0 + PAD_X + 0.006, yy, ln, ha="left", va="center",
                    fontsize=FS_BODY, color="#3A4750", zorder=3)
            yy -= pt(CTX_PT)
        y = y - box_h - pt(5.0)

    # ---- question stem ----
    for ln in spec["stem_l"]:
        y -= pt(STEM_PT / 2)
        ax.text(X0 + PAD_X, y, ln, ha="left", va="center",
                fontsize=FS_STEM, color=INK, zorder=3)
        y -= pt(STEM_PT / 2)

    # ---- answer choices, one row ----
    y -= pt(GAP_PT + CHOICE_PT / 2)
    if spec["opts"][0][0]:                                   # lettered A-D
        cw = (X1 - X0 - 2 * PAD_X) / len(spec["opts"])
        for i, (lab, txt) in enumerate(spec["opts"]):
            cx = X0 + PAD_X + i * cw
            hot = lab == spec["key"]
            ax.add_patch(Rectangle((cx, y - pt(4.5)), 0.021, pt(9.0),
                                   fc=KEY if hot else "#E2E9EC", ec="none", zorder=3))
            ax.text(cx + 0.0105, y, lab, ha="center", va="center", fontsize=FS_BADGE,
                    color="white" if hot else MUT, zorder=4)
            ax.text(cx + 0.030, y, txt, ha="left", va="center", fontsize=FS_BODY,
                    color=INK if hot else MUT,
                    fontweight="bold" if hot else "normal", zorder=4)
    else:                                                    # yes / no / maybe
        cx = X0 + PAD_X
        for _, txt in spec["opts"]:
            hot = txt == spec["key"]
            bw = text_width_in(txt, FS_BODY) / W + 0.026
            ax.add_patch(FancyBboxPatch((cx, y - pt(6)), bw, pt(12),
                                        boxstyle="round,pad=0,rounding_size=0.007",
                                        fc=KEY if hot else "#E2E9EC", ec="none", zorder=3))
            ax.text(cx + bw / 2, y, txt, ha="center", va="center", fontsize=FS_BODY,
                    color="white" if hot else MUT,
                    fontweight="bold" if hot else "normal", zorder=4)
            cx += bw + 0.010

    top = bot - pt(BETWEEN)

fig.savefig(os.path.join(OUT, "Main_Figure0_benchmark_examples.pdf"),
            bbox_inches=Bbox.from_extents(0, 0, W, H))
fig.savefig(os.path.join(OUT, "Main_Figure0_benchmark_examples.png"), dpi=190,
            bbox_inches=Bbox.from_extents(0, 0, W, H))
print("wrote Main_Figure0_benchmark_examples  (%.2f x %.2f in)" % (W, H))
for s in BENCH:
    print("   %-14s kind=%d  stem=%d  ctx=%d lines" %
          (s["name"], len(s["kind_l"]), len(s["stem_l"]), len(s["ctx_l"])))
