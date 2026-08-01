#!/usr/bin/env python3
"""Publication-ready figures for the medical-benchmark reanalysis.

Output: vector PDF, TrueType (Type 42) embedded fonts, sized for a 7in text
block so that figures included at 0.95\\linewidth render at ~1.0 scale and the
8 pt figure text stays 8 pt on the printed page.
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = os.path.expanduser("~/Downloads/paper-figures")
os.makedirs(OUT, exist_ok=True)

# ---------------------------------------------------------------- style
plt.rcParams.update({
    "pdf.fonttype": 42, "ps.fonttype": 42,          # embed editable TrueType
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 8,
    "axes.labelsize": 8, "axes.titlesize": 8.5,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "legend.fontsize": 7.5, "legend.frameon": False,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 300, "savefig.dpi": 300,
    "savefig.bbox": "standard",
    "mathtext.fontset": "custom",
    "mathtext.rm": "Helvetica", "mathtext.it": "Helvetica:italic",
})
W = 6.6                                   # inches ~= 0.95 * 7in text block
ACC, NEG, POS, MUT, LINE = "#0F6E8C", "#B24630", "#2E7D5B", "#586873", "#CAD5DB"
SMALL_C, LARGE_C = "#9CC6D6", ACC

def save(fig, name):
    p = os.path.join(OUT, name)
    fig.savefig(p + ".pdf")
    fig.savefig(p + ".png", dpi=200)      # preview only
    plt.close(fig)
    print("wrote", p + ".pdf")

# ================================================ Figure 1 (NEW): by provider
PROV = ["OpenAI", "DeepSeek", "Qwen", "Meta", "Mistral"]
SMALL = {"MedQA-USMLE": [85.31, 93.95, 91.91, 67.48, 77.22],
         "MedMCQA":     [68.80, 81.80, 74.10, 58.40, 69.30],
         "PubMedQA":    [75.00, 77.80, 80.80, 77.80, 77.00]}
LARGE = {"MedQA-USMLE": [94.66, 94.49, 94.66, 89.87, 88.69],
         "MedMCQA":     [83.80, 83.23, 83.80, 79.50, 80.70],
         "PubMedQA":    [78.00, 76.00, 81.60, 80.80, 77.40]}
# prior-generation anchors reproduced from Lievin et al. (2024)
PRIOR = {"MedQA-USMLE": {"best 2023": 86.5, "human expert": 87.0},
         "MedMCQA":     {"best 2023": 73.7, "human expert": 90.0},
         "PubMedQA":    {"best 2023": 81.2, "human expert": 78.0}}

fig, axes = plt.subplots(1, 3, figsize=(W, 2.95), sharey=True, layout="constrained")
x = np.arange(len(PROV)); bw = 0.36
for ax, bm in zip(axes, ["MedQA-USMLE", "MedMCQA", "PubMedQA"]):
    ax.bar(x - bw/2, SMALL[bm], bw, color=SMALL_C, edgecolor="white", linewidth=0.4,
           label="Smaller model", zorder=3)
    ax.bar(x + bw/2, LARGE[bm], bw, color=LARGE_C, edgecolor="white", linewidth=0.4,
           label="Larger model", zorder=3)
    he = ax.axhline(PRIOR[bm]["human expert"], color=NEG, lw=0.9, ls="--",
                    zorder=4, label="Human expert")
    ax.set_title(bm, pad=4)
    ax.set_xticks(x); ax.set_xticklabels(PROV, rotation=30, ha="right")
    ax.set_ylim(50, 100); ax.set_yticks([50, 60, 70, 80, 90, 100])
    ax.grid(axis="y", color=LINE, lw=0.5, zorder=0)
    ax.set_axisbelow(True)
axes[0].set_ylabel("Accuracy (%)")
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc="outside upper center", ncol=3,
           handlelength=1.6, columnspacing=1.4)
save(fig, "Main_Figure1_generational_gains")

# ================================================ Figure 2: open-ended penalty
M2 = [("GPT-5.6 Luna", 89.9, 65.5), ("Qwen 3.7 Plus", 89.9, 60.6),
      ("DeepSeek V4 Pro", 89.5, 59.7), ("DeepSeek V4 Flash", 88.6, 58.8),
      ("Llama 4 Maverick", 85.3, 53.7), ("Mistral Large 3", 85.2, 57.7),
      ("Qwen 3.5 9B", 84.1, 46.9), ("GPT-OSS 20B", 78.0, 49.8),
      ("Ministral 8B", 73.7, 41.8), ("Llama 3.1 8B", 63.5, 31.1)]
fig, ax = plt.subplots(figsize=(W, 3.3), layout="constrained")
y = np.arange(len(M2))[::-1]
for yi, (n, mc, gen) in zip(y, M2):
    ax.plot([gen, mc], [yi, yi], color=LINE, lw=1.6, zorder=2, solid_capstyle="round")
    ax.scatter(mc, yi, s=26, color=ACC, zorder=3)
    ax.scatter(gen, yi, s=26, color=NEG, zorder=3)
    ax.text(gen - 1.6, yi, f"{gen - mc:+.1f}", va="center", ha="right",
            fontsize=6.8, color=MUT)
ax.scatter([], [], s=26, color=ACC, label="Multiple-choice answers")
ax.scatter([], [], s=26, color=NEG, label="Generative answers")
ax.set_yticks(y); ax.set_yticklabels([m[0] for m in M2])
ax.set_xlim(22, 96); ax.set_xlabel("Accuracy on MedQA-USMLE + MedMCQA (%)")
ax.grid(axis="x", color=LINE, lw=0.5, zorder=0); ax.set_axisbelow(True)
ax.legend(loc="lower right", ncol=1)
save(fig, "Main_Figure2_openended_penalty")

# ================================================ Figure 3: reconstruction vs acc
M3 = [("Qwen 3.7 Plus", 88.39, 26.40), ("GPT-5.6 Luna", 87.74, 25.47),
      ("DeepSeek V4 Pro", 87.11, 20.19), ("DeepSeek V4 Flash", 86.66, 21.25),
      ("Llama 4 Maverick", 84.49, 16.50), ("Mistral Large 3", 83.77, 20.33),
      ("Qwen 3.5 9B", 83.48, 13.64), ("GPT-OSS 20B", 77.50, 15.44),
      ("Ministral 8B", 74.32, 10.87), ("Llama 3.1 8B", 66.07, 9.11)]
xs = np.array([m[1] for m in M3]); ys = np.array([m[2] for m in M3])
fig, ax = plt.subplots(figsize=(W, 3.4), layout="constrained")
b, a = np.polyfit(xs, ys, 1)
xf = np.linspace(xs.min() - 2, xs.max() + 2, 50)
ax.plot(xf, b * xf + a, color=MUT, lw=0.9, ls="--", zorder=2)
ax.scatter(xs, ys, s=34, color=ACC, zorder=3, edgecolor="white", linewidth=0.5)
off = {"Qwen 3.7 Plus":     (-0.7,  0.6, "right"),
       "GPT-5.6 Luna":      (-0.7, -1.7, "right"),
       "DeepSeek V4 Flash": (-0.7,  1.0, "right"),
       "DeepSeek V4 Pro":   ( 0.7, -0.9, "left"),
       "Mistral Large 3":   (-0.7, -1.7, "right"),
       "Llama 4 Maverick":  ( 0.7,  0.0, "left"),
       "Qwen 3.5 9B":       ( 0.7, -0.3, "left")}
for n, xx, yy in M3:
    dx, dy, ha = off.get(n, (0.7, 0.0, "left"))
    ax.annotate(n, (xx + dx, yy + dy), fontsize=6.6, color=MUT,
                ha=ha, va="center")
r = np.corrcoef(xs, ys)[0, 1]
ax.text(0.03, 0.95, f"$r$ = {r:.2f}, $p$ = .001", transform=ax.transAxes,
        fontsize=7.5, va="top")
ax.set_xlabel("Multiple-choice accuracy, all benchmarks (%)")
ax.set_ylabel("Question-reconstruction fidelity (%)")
ax.set_xlim(62, 95); ax.set_ylim(5, 31)
ax.grid(color=LINE, lw=0.5, zorder=0); ax.set_axisbelow(True)
save(fig, "Main_Figure3_reconstruction_vs_accuracy")

# ================================================ Figure 4: larger-model wins
PAIR = [("OpenAI (GPT-5.6 Luna)",      25.6, 17.0),
        ("DeepSeek (V4 Pro)",          20.3, 15.6),
        ("Qwen (3.7 Plus)",            26.5, 18.0),
        ("Meta (Llama 4 Maverick)",    16.5, 13.5),
        ("Mistral (Large 3)",          20.4, 14.4)]
fig, ax = plt.subplots(figsize=(W, 2.8), layout="constrained")
y = np.arange(len(PAIR))[::-1]
for yi, (n, base, wins) in zip(y, PAIR):
    ax.plot([wins, base], [yi, yi], color=LINE, lw=1.6, zorder=2, solid_capstyle="round")
    ax.scatter(base, yi, s=30, color=ACC, zorder=3)
    ax.scatter(wins, yi, s=30, color=NEG, zorder=3)
    ax.text(wins - 0.55, yi, f"{wins - base:+.1f}", va="center", ha="right",
            fontsize=6.8, color=NEG)
ax.scatter([], [], s=30, color=ACC, label="All items the larger model answers")
ax.scatter([], [], s=30, color=NEG, label="Only the items its smaller sibling got wrong")
ax.set_yticks(y); ax.set_yticklabels([p[0] for p in PAIR])
ax.set_xlim(9.5, 29.5)
ax.set_xlabel("Question-reconstruction fidelity of the larger model (%)")
ax.grid(axis="x", color=LINE, lw=0.5, zorder=0); ax.set_axisbelow(True)
ax.legend(loc="lower right")
ax.text(0.985, 0.60, "the larger model's extra wins are\nconsistently $\\it{less}$ reconstructable\npaired $t$(4) = $-$5.67, $p$ = .005",
        transform=ax.transAxes, ha="right", va="top", fontsize=6.8, color=MUT, linespacing=1.5)
save(fig, "Main_Figure4_large_only_right")

# ================================================ Figure 5: item difficulty
HIST = [105, 62, 51, 51, 69, 83, 103, 153, 241, 460, 1395]
NTOT = 2773                                    # fixed pool, so counts map 1:1 to %
fig, ax = plt.subplots(figsize=(W, 2.9), layout="constrained")
cols = [NEG] + [SMALL_C] * 9 + [POS]
ax.bar(range(11), HIST, color=cols, edgecolor="white", linewidth=0.4, zorder=3)
for i, v in enumerate(HIST):
    ax.text(i, v + 28, f"{v:,}", ha="center", fontsize=6.6, color=MUT)
ax.set_xticks(range(11))
ax.set_xlabel("Number of models answering the question correctly (of 10)")
ax.set_ylabel("Questions")
ax.set_ylim(0, 1560)
ax.grid(axis="y", color=LINE, lw=0.5, zorder=0); ax.set_axisbelow(True)
ax.annotate("missed by all ten", xy=(0, 105), xytext=(1.4, 620),
            fontsize=7, color=NEG,
            arrowprops=dict(arrowstyle="-", color=NEG, lw=0.7))
# right-hand axis: the same bars read as a share of the 2,773-question pool
axr = ax.secondary_yaxis("right", functions=(lambda c: 100 * c / NTOT,
                                             lambda p: p * NTOT / 100))
axr.set_ylabel("Share of all questions (%)")
axr.set_yticks([0, 10, 20, 30, 40, 50])
axr.spines["right"].set_visible(True)
axr.spines["right"].set_linewidth(0.6)
save(fig, "Main_Figure5_item_difficulty")

print("\nAll figures written to", OUT)
