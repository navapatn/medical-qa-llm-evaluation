#!/usr/bin/env python3
"""Regenerate the figures the paper actually uses, for twelve models.

Only the eight data figures change: the schematic and the benchmark-example
figure carry no per-model data, and the supplementary prompt/reasoning figures
are unaffected. Everything is driven by the recomputed per-question outcomes,
not by the previously published summary numbers.

Palette follows the project convention: teal = multiple choice, amber =
generative, plum = reconstruction, green/red reserved for correct/wrong.
"""
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

A = os.path.expanduser("~/Documents/Research/CellPress/analysis12")
OUT = os.path.expanduser("~/Downloads/paper-figures/twelve-model")
os.makedirs(OUT, exist_ok=True)
R = json.load(open(os.path.join(A, "stats12.json")))
E = json.load(open(os.path.join(A, "ensemble12.json")))
PM = R["per_model"]

plt.rcParams.update({
    "pdf.fonttype": 42, "ps.fonttype": 42, "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8.5,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "legend.fontsize": 7.5, "legend.frameon": False, "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 300, "savefig.dpi": 300, "savefig.bbox": "standard",
    "mathtext.fontset": "custom", "mathtext.rm": "Helvetica",
    "mathtext.it": "Helvetica:italic",
})
W = 6.6
ACC, NEG, POS, MUT, LINE = "#0F6E8C", "#B24630", "#2E7D5B", "#586873", "#CAD5DB"
SMALL_C, SOFT, WARM = "#9CC6D6", "#9CC6D6", "#E5BAB1"
ORDER = sorted(PM, key=lambda m: -PM[m]["overall"]["acc"])
PAIRS = [tuple(p) for p in json.load(open(os.path.join(A, "per_question_12.json")))["pairs"]]


def save(fig, name):
    fig.savefig(os.path.join(OUT, name + ".pdf"))
    fig.savefig(os.path.join(OUT, name + ".png"), dpi=200)
    plt.close(fig)
    print("  wrote", name)


# ============================== Fig 1: by provider, smaller vs larger
PROV = [("OpenAI", "GPT-OSS 20B", "GPT-5.6 Luna"), ("DeepSeek", "DeepSeek V4 Flash", "DeepSeek V4 Pro"),
        ("Qwen", "Qwen 3.5 9B", "Qwen 3.7 Plus"), ("Meta", "Llama 3.1 8B", "Llama 4 Maverick"),
        ("Mistral", "Ministral 8B", "Mistral Large 3"), ("Google", "MedGemma 4B", "MedGemma 27B")]
HUMAN = {"MedQA-USMLE": 87.0, "MedMCQA": 90.0, "PubMedQA": 78.0}
fig, axes = plt.subplots(1, 3, figsize=(W, 3.0), sharey=True, layout="constrained")
x = np.arange(len(PROV)); bw = 0.36
for ax, b in zip(axes, ("MedQA-USMLE", "MedMCQA", "PubMedQA")):
    sm = [PM[p[1]][b]["acc"] for p in PROV]
    lg = [PM[p[2]][b]["acc"] for p in PROV]
    ax.bar(x - bw/2, sm, bw, color=SMALL_C, edgecolor="white", linewidth=0.4,
           label="Smaller model", zorder=3)
    ax.bar(x + bw/2, lg, bw, color=ACC, edgecolor="white", linewidth=0.4,
           label="Larger model", zorder=3)
    ax.axhline(HUMAN[b], color=NEG, lw=0.9, ls="--", zorder=4, label="Human expert")
    ax.set_title(b, pad=4)
    ax.set_xticks(x); ax.set_xticklabels([p[0] for p in PROV], rotation=30, ha="right")
    ax.set_ylim(50, 100); ax.set_yticks([50, 60, 70, 80, 90, 100])
    ax.grid(axis="y", color=LINE, lw=0.5, zorder=0); ax.set_axisbelow(True)
axes[0].set_ylabel("Accuracy (%)")
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc="outside upper center", ncol=3, handlelength=1.6, columnspacing=1.4)
save(fig, "Fig1_generational_gains")

# ============================== Fig 2: open-ended penalty
fig, ax = plt.subplots(figsize=(W, 4.0), layout="constrained")
y = np.arange(len(ORDER))[::-1]
for yi, m in zip(y, ORDER):
    mc, gen = PM[m]["mc4"], PM[m]["gen4"]
    ax.plot([gen, mc], [yi, yi], color=LINE, lw=1.6, zorder=2, solid_capstyle="round")
    ax.scatter(mc, yi, s=26, color=ACC, zorder=3)
    ax.scatter(gen, yi, s=26, color=NEG, zorder=3)
    ax.text(gen - 1.6, yi, "%+.1f" % (gen - mc), va="center", ha="right",
            fontsize=6.8, color=MUT)
ax.scatter([], [], s=26, color=ACC, label="Multiple-choice answers")
ax.scatter([], [], s=26, color=NEG, label="Generative answers")
ax.set_yticks(y); ax.set_yticklabels(ORDER)
ax.set_xlim(22, 99); ax.set_xlabel("Accuracy on MedQA-USMLE + MedMCQA (%)")
ax.grid(axis="x", color=LINE, lw=0.5, zorder=0); ax.set_axisbelow(True)
# below the axes: with twelve rows the lower-right corner is occupied by data
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2,
          columnspacing=1.6, handletextpad=0.4)
save(fig, "Fig2_openended_penalty")

# ============================== Fig 3: reconstruction vs accuracy
fig, ax = plt.subplots(figsize=(W, 3.4), layout="constrained")
xs = np.array([PM[m]["overall"]["acc"] for m in ORDER])
ys = np.array([PM[m]["rec"]["acc"] for m in ORDER])
ax.scatter(xs, ys, s=34, color=ACC, zorder=3)
k, b = np.polyfit(xs, ys, 1)
xr = np.linspace(xs.min() - 2, xs.max() + 2, 10)
ax.plot(xr, k * xr + b, ls="--", color=MUT, lw=0.9, zorder=2)
# Twelve points crowd the upper right, so a few labels are placed by hand:
# (dx, dy, horizontal alignment). Defaults sit to the right of the marker.
OFF = {"Mistral Large 3":   (-0.6,  0.0, "right"),
       "GPT-OSS 20B":       (-0.6,  0.0, "right"),
       "DeepSeek V4 Flash": ( 0.6,  0.35, "left"),
       "DeepSeek V4 Pro":   ( 0.6, -0.45, "left"),
       "MedGemma 27B":      ( 0.6,  0.25, "left"),
       "Llama 4 Maverick":  ( 0.6, -0.25, "left")}
for m, xx, yy in zip(ORDER, xs, ys):
    dx, dy, ha = OFF.get(m, (0.6, 0.0, "left"))
    ax.annotate(m, (xx + dx, yy + dy), fontsize=6.6, color=MUT, va="center", ha=ha)
ax.text(0.03, 0.95, "$r$ = %.2f" % R["twelve_corr"], transform=ax.transAxes,
        fontsize=7.5, va="top")
ax.set_xlabel("Multiple-choice accuracy, all benchmarks (%)")
ax.set_ylabel("Question-reconstruction fidelity (%)")
ax.set_xlim(62, 95); ax.set_ylim(5, 31)
ax.grid(color=LINE, lw=0.5, zorder=0); ax.set_axisbelow(True)
save(fig, "Fig3_reconstruction_vs_accuracy")

# ============================== Fig 4: memorization, two panels
fig, axs = plt.subplots(1, 2, figsize=(W, 4.3), layout="constrained")
RO = sorted(PM, key=lambda m: -PM[m]["rec"]["acc"])
ax = axs[0]; bh = 0.38
yy = np.arange(len(RO))[::-1]
ax.barh(yy + bh/2, [PM[m]["rec_MedQA-USMLE"] for m in RO], bh, color=ACC,
        label="MedQA-USMLE", zorder=3)
ax.barh(yy - bh/2, [PM[m]["rec_MedMCQA"] for m in RO], bh, color=SOFT,
        label="MedMCQA", zorder=3)
ax.set_yticks(yy); ax.set_yticklabels(RO)
ax.set_xlabel("Question-reconstruction fidelity (%)"); ax.set_xlim(0, 35)
ax.legend(loc="lower right")
ax.grid(axis="x", color=LINE, lw=0.5, zorder=0); ax.set_axisbelow(True)
ax.set_title("How often the hidden question can be reconstructed", pad=6)
ax.text(-0.02, 1.06, "A", transform=ax.transAxes, fontsize=9, fontweight="bold",
        va="bottom", ha="right")

ax = axs[1]
segs = {"uo": [], "ro": [], "rw": [], "uw": []}
for m in RO:
    p = R["pools"][m]
    pu = p["unrec_pct"] / 100; pr = 1 - pu
    segs["uo"].append(pu * p["mc_unrec"]); segs["ro"].append(pr * p["mc_rec"])
    segs["rw"].append(pr * (100 - p["mc_rec"])); segs["uw"].append(pu * (100 - p["mc_unrec"]))
left = np.zeros(len(RO))
for key, c, lab in (("uo", SOFT, "Cannot reconstruct  ·  correct"),
                    ("ro", ACC, "Can reconstruct  ·  correct"),
                    ("rw", NEG, "Can reconstruct  ·  wrong"),
                    ("uw", WARM, "Cannot reconstruct  ·  wrong")):
    v = np.array(segs[key])
    ax.barh(yy, v, 0.62, left=left, color=c, label=lab, edgecolor="white",
            linewidth=0.5, zorder=3)
    left += v
ax.set_yticks(yy); ax.set_yticklabels([])
ax.set_xlim(0, 100); ax.set_xticks([0, 25, 50, 75, 100])
ax.set_xlabel("Share of MedQA-USMLE + MedMCQA questions (%)")
ax.grid(axis="x", color="white", lw=0.5, zorder=4); ax.set_axisbelow(False)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2,
          columnspacing=1.3, handlelength=1.1, handletextpad=0.5)
ax.set_title("Most correct answers come from questions\nthe model cannot reconstruct", pad=6)
ax.text(-0.02, 1.06, "B", transform=ax.transAxes, fontsize=9, fontweight="bold",
        va="bottom", ha="right")
save(fig, "Fig4_memorization")

# ============================== Fig 5: reconstructable pools
fig, ax = plt.subplots(figsize=(W, 4.2), layout="constrained")
y = np.arange(len(ORDER))[::-1]
for yi, m in zip(y, ORDER):
    p = R["pools"][m]
    ax.plot([p["gen_unrec"], p["gen_rec"]], [yi, yi], color=NEG, lw=1.5, alpha=.45,
            zorder=2, solid_capstyle="round")
    ax.plot([p["mc_unrec"], p["mc_rec"]], [yi, yi], color=ACC, lw=1.5, alpha=.45,
            zorder=2, solid_capstyle="round")
    for xv, c in ((p["gen_unrec"], NEG), (p["mc_unrec"], ACC)):
        ax.scatter(xv, yi, s=26, facecolor="white", edgecolor=c, linewidth=1.1, zorder=3)
    for xv, c in ((p["gen_rec"], NEG), (p["mc_rec"], ACC)):
        ax.scatter(xv, yi, s=26, color=c, zorder=3)
ax.set_yticks(y); ax.set_yticklabels(ORDER)
ax.set_xlim(26, 99); ax.set_xlabel("Accuracy (%)")
ax.grid(axis="x", color=LINE, lw=0.5, zorder=0); ax.set_axisbelow(True)
handles = [
    Line2D([], [], marker="o", ls="none", markerfacecolor="white", markeredgecolor=ACC,
           markeredgewidth=1.1, ms=5, label="Multiple choice — cannot reconstruct"),
    Line2D([], [], marker="o", ls="none", color=ACC, ms=5,
           label="Multiple choice — can reconstruct"),
    Line2D([], [], marker="o", ls="none", markerfacecolor="white", markeredgecolor=NEG,
           markeredgewidth=1.1, ms=5, label="Generative — cannot reconstruct"),
    Line2D([], [], marker="o", ls="none", color=NEG, ms=5,
           label="Generative — can reconstruct")]
ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.26), ncol=2,
          columnspacing=1.6, handletextpad=0.4)
save(fig, "Fig5_reconstructable_pools")

# ============================== Fig 7: ensemble by tier
small = [p[0] for p in PAIRS]; large = [p[1] for p in PAIRS]
fig, axes = plt.subplots(1, 3, figsize=(W, 2.9), sharey=True, layout="constrained")
for ax, (title, key, pool) in zip(axes, [
        ("All twelve models", "twelve", ORDER),
        ("Larger model of each pair", "larger-6", large),
        ("Smaller model of each pair", "smaller-6", small)]):
    curve = E[key]["curve"]; best = E[key]["best_single"]
    ks = np.arange(1, len(curve) + 1)
    ax.plot(ks, curve, "-o", color=ACC, lw=1.6, ms=3.4, zorder=3)
    ax.axhline(best, color=MUT, lw=0.8, ls="--", zorder=2)
    ax.fill_between(ks, best, curve, where=np.array(curve) >= best, color=POS,
                    alpha=0.14, zorder=1)
    ax.fill_between(ks, best, curve, where=np.array(curve) < best, color=NEG,
                    alpha=0.12, zorder=1)
    ax.set_title(title, pad=4); ax.set_xlabel("Models in the committee")
    ax.set_xticks(ks[::2] if len(ks) > 6 else ks)
    ax.grid(color=LINE, lw=0.5, zorder=0); ax.set_axisbelow(True)
axes[0].set_ylabel("Majority-vote accuracy (%)")
axes[0].set_ylim(83, 91)
save(fig, "Fig7_ensemble_by_tier")

# ============================== Fig 9: item difficulty
HIST = R["twelve_difficulty"]["hist"]; NTOT = R["twelve_difficulty"]["n"]
fig, ax = plt.subplots(figsize=(W, 2.9), layout="constrained")
cols = [NEG] + [SMALL_C] * (len(HIST) - 2) + [POS]
ax.bar(range(len(HIST)), HIST, color=cols, edgecolor="white", linewidth=0.4, zorder=3)
for i, v in enumerate(HIST):
    ax.text(i, v + 26, "%,d".replace(",", ",") % v if False else format(v, ","),
            ha="center", fontsize=6.4, color=MUT)
ax.set_xticks(range(len(HIST)))
ax.set_xlabel("Number of models answering the question correctly (of 12)")
ax.set_ylabel("Questions"); ax.set_ylim(0, 1500)
ax.grid(axis="y", color=LINE, lw=0.5, zorder=0); ax.set_axisbelow(True)
ax.annotate("missed by all twelve", xy=(0, HIST[0]), xytext=(1.4, 600), fontsize=7,
            color=NEG, arrowprops=dict(arrowstyle="-", color=NEG, lw=0.7))
axr = ax.secondary_yaxis("right", functions=(lambda c: 100 * c / NTOT,
                                             lambda p: p * NTOT / 100))
axr.set_ylabel("Share of all questions (%)"); axr.set_yticks([0, 10, 20, 30, 40, 50])
axr.spines["right"].set_visible(True); axr.spines["right"].set_linewidth(0.6)
save(fig, "Fig9_item_difficulty")

print("\nAll twelve-model figures written to", OUT)
