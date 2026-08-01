#!/usr/bin/env python3
"""Candidate figures that would replace tables in the reanalysis manuscript.

  Main_FigureB_memorization.pdf        replaces tab:rq3 + tab:rq9
  Main_FigureD_reconstructable_pools.pdf   replaces tab:rq5   (RQ4)
  Main_FigureC_ensemble_size.pdf       replaces tab:ensemble

Same spec as the main figure set: 6.6 in wide, vector PDF, Helvetica Type-42.
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

OUT = os.path.expanduser("~/Downloads/paper-figures")
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "legend.fontsize": 7, "legend.frameon": False,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 300, "savefig.dpi": 300, "savefig.bbox": "standard",
    "mathtext.fontset": "custom",
    "mathtext.rm": "Helvetica", "mathtext.it": "Helvetica:italic",
})
W = 6.6
ACC, NEG, POS, MUT, LINE = "#0F6E8C", "#B24630", "#2E7D5B", "#586873", "#CAD5DB"
SOFT, WARM = "#9CC6D6", "#E5BAB1"

def save(fig, name):
    p = os.path.join(OUT, name)
    fig.savefig(p + ".pdf"); fig.savefig(p + ".png", dpi=200); plt.close(fig)
    print("wrote", p + ".pdf")

def panel_tag(ax, s):
    ax.text(-0.02, 1.06, s, transform=ax.transAxes, fontsize=9,
            fontweight="bold", va="bottom", ha="right")

# ---------------------------------------------------------------- shared data
NAMES = ["Qwen 3.7 Plus", "GPT-5.6 Luna", "DeepSeek V4 Flash", "Mistral Large 3",
         "DeepSeek V4 Pro", "Llama 4 Maverick", "GPT-OSS 20B", "Qwen 3.5 9B",
         "Ministral 8B", "Llama 3.1 8B"]
SHORT = ["Qwen 3.7", "Luna", "DS Flash", "Mistral L3", "DS Pro",
         "Maverick", "GPT-OSS", "Qwen 3.5", "Ministral", "Llama 3.1"]
REC_MEDQA = [30.24, 28.99, 23.88, 24.35, 22.15, 18.07, 18.22, 15.32, 12.33, 9.27]
REC_MEDMC = [21.50, 21.00, 17.90, 15.20, 17.70, 14.50, 11.90, 11.50, 9.00, 8.90]
REC_RIGHT = [27.8, 26.6, 22.7, 21.5, 21.0, 17.3, 17.8, 14.5, 12.0, 10.7]
REC_WRONG = [13.9, 16.5, 10.9, 13.6, 14.1, 11.7, 7.4, 9.1, 7.7, 6.3]
UNREC_N   = [1672, 1687, 1782, 1810, 1809, 1896, 1916, 1958, 2022, 2064]
TOT_ITEMS = 2273
MC_UNREC  = [88.2, 88.6, 87.1, 83.9, 88.2, 84.4, 75.9, 83.2, 72.7, 62.4]
MC_REC    = [94.7, 93.4, 94.2, 90.0, 92.4, 89.6, 89.5, 89.4, 81.4, 74.9]
GEN_UNREC = [58.0, 63.0, 56.5, 53.9, 56.9, 51.5, 46.5, 45.0, 39.8, 29.7]
GEN_REC   = [68.0, 73.6, 68.1, 72.7, 71.2, 64.5, 68.9, 60.0, 58.3, 44.9]
y = np.arange(len(NAMES))[::-1]

N_MEDQA, N_MEDMC = 1273, 1000          # items per benchmark

def wilson(p_pct, n, z=1.96):
    """95% Wilson score interval, returned as (lower_err, upper_err) in points."""
    p = np.asarray(p_pct, float) / 100.0
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    lo, hi = (c - h) * 100, (c + h) * 100
    return np.vstack([np.asarray(p_pct) - lo, hi - np.asarray(p_pct)])


# ================================================ FIGURE B  (tab:rq3 + tab:rq9)
fig, axs = plt.subplots(1, 2, figsize=(W, 3.9), layout="constrained")

ax = axs[0]; bh = 0.38
ekw = dict(ecolor="#151B21", elinewidth=0.7, capsize=1.4, capthick=0.7)
ax.barh(y + bh/2, REC_MEDQA, bh, color=ACC, label="MedQA-USMLE", zorder=3,
        xerr=wilson(REC_MEDQA, N_MEDQA), error_kw=ekw)
ax.barh(y - bh/2, REC_MEDMC, bh, color=SOFT, label="MedMCQA", zorder=3,
        xerr=wilson(REC_MEDMC, N_MEDMC), error_kw=ekw)
ax.set_yticks(y); ax.set_yticklabels(NAMES)
ax.set_xlabel("Question-reconstruction fidelity (%)")
ax.set_xlim(0, 35); ax.legend(loc="lower right")
ax.grid(axis="x", color=LINE, lw=0.5, zorder=0); ax.set_axisbelow(True)
ax.set_title("How often the hidden question can be reconstructed", pad=6)
panel_tag(ax, "A")

ax = axs[1]
# each model's items split four ways: reconstructable? x answered correctly?
seg_uo, seg_ro, seg_rw, seg_uw = [], [], [], []
for i in range(len(NAMES)):
    pu = UNREC_N[i] / TOT_ITEMS; pr = 1 - pu
    seg_uo.append(pu * MC_UNREC[i]); seg_ro.append(pr * MC_REC[i])
    seg_rw.append(pr * (100 - MC_REC[i])); seg_uw.append(pu * (100 - MC_UNREC[i]))
segs = [(seg_uo, SOFT, "Cannot reconstruct  ·  correct"),
        (seg_ro, ACC,  "Can reconstruct  ·  correct"),
        (seg_rw, NEG,  "Can reconstruct  ·  wrong"),
        (seg_uw, WARM, "Cannot reconstruct  ·  wrong")]
left = np.zeros(len(NAMES))
for vals, col, lab in segs:
    vals = np.array(vals)
    ax.barh(y, vals, 0.62, left=left, color=col, label=lab,
            edgecolor="white", linewidth=0.5, zorder=3)
    left = left + vals
ax.set_yticks(y); ax.set_yticklabels([])
ax.set_xlim(0, 100); ax.set_xticks([0, 25, 50, 75, 100])
ax.set_xlabel("Share of MedQA-USMLE + MedMCQA questions (%)")
ax.grid(axis="x", color="white", lw=0.5, zorder=4)
ax.set_axisbelow(False)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2,
          columnspacing=1.3, handlelength=1.1, handletextpad=0.5)
ax.set_title("Most correct answers come from questions\nthe model cannot reconstruct", pad=6)
panel_tag(ax, "B")
save(fig, "Main_FigureB_memorization")

# ============================== FIGURE D  (tab:rq5, RQ4) — C and D combined
fig, ax = plt.subplots(figsize=(W, 3.8), layout="constrained")
for yi, gu, gr, mu, mr in zip(y, GEN_UNREC, GEN_REC, MC_UNREC, MC_REC):
    ax.plot([gu, gr], [yi, yi], color=NEG, lw=1.5, alpha=.45, zorder=2,
            solid_capstyle="round")
    ax.plot([mu, mr], [yi, yi], color=ACC, lw=1.5, alpha=.45, zorder=2,
            solid_capstyle="round")
    for xv, col in ((gu, NEG), (mu, ACC)):                       # cannot reconstruct
        ax.scatter(xv, yi, s=26, facecolor="white", edgecolor=col,
                   linewidth=1.1, zorder=3)
    for xv, col in ((gr, NEG), (mr, ACC)):                       # can reconstruct
        ax.scatter(xv, yi, s=26, color=col, zorder=3)
ax.set_yticks(y); ax.set_yticklabels(NAMES)
ax.set_xlim(26, 99); ax.set_ylim(-0.7, 9.7); ax.set_xlabel("Accuracy (%)")
ax.grid(axis="x", color=LINE, lw=0.5, zorder=0); ax.set_axisbelow(True)
handles = [
    Line2D([], [], marker="o", ls="none", markerfacecolor="white",
           markeredgecolor=ACC, markeredgewidth=1.1, ms=5,
           label="Multiple choice — cannot reconstruct"),
    Line2D([], [], marker="o", ls="none", color=ACC, ms=5,
           label="Multiple choice — can reconstruct"),
    Line2D([], [], marker="o", ls="none", markerfacecolor="white",
           markeredgecolor=NEG, markeredgewidth=1.1, ms=5,
           label="Generative — cannot reconstruct"),
    Line2D([], [], marker="o", ls="none", color=NEG, ms=5,
           label="Generative — can reconstruct")]
ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.30),
          ncol=2, columnspacing=1.6, handletextpad=0.4)
ax.set_title("Accuracy on the items a model can and cannot reconstruct", pad=6)
save(fig, "Main_FigureD_reconstructable_pools")

# ================================================ FIGURE C  (tab:ensemble)
K = np.arange(1, 11)
VOTE = [88.39, 88.39, 89.36, 89.54, 89.25, 89.36, 88.93, 89.04, 88.60, 88.24]
ANY1 = [88.39, 91.74, 93.40, 94.48, 95.17, 95.60, 95.89, 96.07, 96.18, 96.21]
BEST1 = 88.39
fig, ax = plt.subplots(figsize=(W, 3.2), layout="constrained")
ax.plot(K, ANY1, "-o", color=POS, lw=1.7, ms=4, label="At least one model correct", zorder=3)
ax.plot(K, VOTE, "-o", color=ACC, lw=1.7, ms=4, label="Majority vote", zorder=3)
ax.axhline(BEST1, color=MUT, lw=0.8, ls="--", zorder=2)
ax.text(10, BEST1 + 0.35, "best single model 88.4", ha="right", fontsize=6.8, color=MUT)
ax.set_xticks(K); ax.set_xlabel("Models in the ensemble")
ax.set_ylabel("Accuracy (%)"); ax.set_ylim(82, 98)
ax.legend(loc="upper left", bbox_to_anchor=(0.005, 0.88))
ax.grid(color=LINE, lw=0.5, zorder=0); ax.set_axisbelow(True)
ax.set_title("The pool knows far more than voting can extract", pad=6)
save(fig, "Main_FigureC_ensemble_size")
