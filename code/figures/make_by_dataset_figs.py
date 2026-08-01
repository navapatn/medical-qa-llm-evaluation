#!/usr/bin/env python3
"""Per-dataset versions, consistent palette (green/red reserved for correctness):
  Main_FigureC_ensemble_by_dataset.pdf   3 panels; teal vote / amber at-least-one
  Main_FigureD_pools_by_dataset.pdf       4 zoomed strips: {MedQA,MedMCQA} x {MC,Gen}"""
import os, json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

SD = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.expanduser("~/Downloads/paper-figures")
D = json.load(open(SD + "/by_dataset.json"))
figC, figD, MODELS = D["figC"], D["figD"], D["models"]

plt.rcParams.update({
    "pdf.fonttype": 42, "ps.fonttype": 42, "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8.5,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "legend.frameon": False, "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "ytick.major.size": 0,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 300, "savefig.dpi": 300, "savefig.bbox": "standard",
    "mathtext.fontset": "custom", "mathtext.rm": "Helvetica",
})
INK, MUT, LINE = "#151B21", "#5C6B75", "#D6DEE3"
TEAL, AMBER, GREY = "#0F6E8C", "#C77A30", "#7F94A0"     # MC / generative / neutral

# ============================================= FIGURE C : ensemble, 3 datasets
K = np.arange(1, 11)
fig, axs = plt.subplots(1, 3, figsize=(6.9, 2.9), sharey=True)
for ax, dset in zip(axs, ["MedQA-USMLE", "MedMCQA", "PubMedQA"]):
    d = figC[dset]
    ax.plot(K, d["anyone"], "-o", color=AMBER, lw=1.6, ms=3.4,
            label="at least one correct", zorder=3)
    ax.plot(K, d["vote"], "-o", color=TEAL, lw=1.6, ms=3.4,
            label="majority vote", zorder=3)
    ax.axhline(d["best_single"], color=MUT, lw=0.8, ls=(0, (4, 3)), zorder=2)
    ax.set_title(dset, pad=5)
    ax.set_xticks([1, 4, 7, 10]); ax.set_xlabel("models in the committee")
    ax.grid(color=LINE, lw=0.5, zorder=0); ax.set_axisbelow(True)
axs[0].set_ylim(78, 100); axs[0].set_yticks([80, 85, 90, 95, 100])
axs[0].set_ylabel("Accuracy (%)")
axs[2].text(10, figC["PubMedQA"]["best_single"] - 1.4, "best single model",
            ha="right", fontsize=6, color=MUT)
axs[1].legend(loc="lower center", bbox_to_anchor=(0.5, -0.46), ncol=2, columnspacing=1.8)
fig.suptitle("Ensembling does not recover the gap on any benchmark",
             fontsize=9.5, fontweight="bold", y=1.0)
fig.subplots_adjust(top=0.84, bottom=0.28, left=0.075, right=0.985, wspace=0.12)
fig.savefig(os.path.join(OUT, "Main_FigureC_ensemble_by_dataset.pdf"))
fig.savefig(os.path.join(OUT, "Main_FigureC_ensemble_by_dataset.png"), dpi=190)
plt.close(fig); print("wrote Main_FigureC_ensemble_by_dataset")

# ============================================= FIGURE D : 4 zoomed strips
yy = np.arange(len(MODELS))[::-1]
fig = plt.figure(figsize=(6.9, 3.5))
fig.text(0.5, 0.972, "Accuracy on reconstructable vs. un-reconstructable items",
         ha="center", va="top", fontsize=9.6, fontweight="bold", color=INK)

axn = fig.add_axes([0.0, 0.205, 0.135, 0.60]); axn.axis("off")
axn.set_ylim(-0.7, 9.7); axn.set_xlim(0, 1)
for y, nm in zip(yy, MODELS):
    axn.text(0.98, y, nm, ha="right", va="center", fontsize=6.8, color=INK)

def strip(pos, dset, cond, color, keys):
    ax = fig.add_axes(pos)
    lo = min(figD[dset][m][keys[0]] for m in MODELS)
    hi = max(figD[dset][m][keys[1]] for m in MODELS)
    pad = max(2.0, (hi - lo) * 0.08)
    x0, x1 = lo - pad, hi + pad
    for y, nm in zip(yy, MODELS):
        u, r = figD[dset][nm][keys[0]], figD[dset][nm][keys[1]]
        ax.plot([u, r], [y, y], color=color, lw=1.6, alpha=.5, zorder=2, solid_capstyle="round")
        ax.scatter(u, y, s=20, facecolor="white", edgecolor=color, lw=1.1, zorder=3)  # cannot
        ax.scatter(r, y, s=20, color=color, zorder=3)                                  # can
    ax.set_xlim(x0, x1); ax.set_ylim(-0.7, 9.7)
    ticks = [t for t in range(0, 101, 10) if x0 + 1 < t < x1 - 1]
    if len(ticks) > 3: ticks = ticks[::2]
    ax.set_xticks(ticks); ax.tick_params(labelsize=6.2)
    ax.set_yticks([]); ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color=LINE, lw=0.5, zorder=0); ax.set_axisbelow(True)
    ax.set_title(cond, fontsize=7.2, color=color, fontweight="bold", pad=3)
    return ax

# layout: [MedQA MC][MedQA Gen]  gap  [MedMCQA MC][MedMCQA Gen]
W = 0.176
strip([0.150, 0.205, W, 0.60], "MedQA-USMLE", "MC", TEAL, ("mc_unrec", "mc_rec"))
strip([0.335, 0.205, W, 0.60], "MedQA-USMLE", "Generative", AMBER, ("gen_unrec", "gen_rec"))
strip([0.590, 0.205, W, 0.60], "MedMCQA", "MC", TEAL, ("mc_unrec", "mc_rec"))
strip([0.775, 0.205, W, 0.60], "MedMCQA", "Generative", AMBER, ("gen_unrec", "gen_rec"))
# dataset labels spanning each pair
fig.text(0.150 + W + 0.0045, 0.845, "MedQA-USMLE", ha="center", va="bottom",
         fontsize=8.2, fontweight="bold", color=INK)
fig.text(0.590 + W + 0.0045, 0.845, "MedMCQA", ha="center", va="bottom",
         fontsize=8.2, fontweight="bold", color=INK)
fig.text(0.5, 0.088, "Accuracy (%).  Each strip is zoomed to its own range; "
         "note the x-axes differ.", ha="center", va="center", fontsize=6.4, color=MUT)
leg = [Line2D([], [], marker="o", ls="none", color=INK, ms=6, label="can reconstruct"),
       Line2D([], [], marker="o", ls="none", markerfacecolor="white", markeredgecolor=INK,
              markeredgewidth=1.1, ms=6, label="cannot reconstruct")]
fig.legend(handles=leg, loc="lower center", ncol=2, bbox_to_anchor=(0.5, 0.010),
           columnspacing=2.0, handletextpad=0.4)
fig.savefig(os.path.join(OUT, "Main_FigureD_pools_by_dataset.pdf"))
fig.savefig(os.path.join(OUT, "Main_FigureD_pools_by_dataset.png"), dpi=190)
plt.close(fig); print("wrote Main_FigureD_pools_by_dataset")
