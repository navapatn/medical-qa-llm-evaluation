#!/usr/bin/env python3
"""Finding 6: cost against accuracy, twelve models.

Setting 2 cost is put on an equal-prompt basis (Setting-1 prompt cost +
Setting-2 completion cost) so the two panels are comparable, as in the
ten-model version. All twelve models are drawn identically and coloured by
size tier. MedGemma cost comes from token counts at its provider's list
rates rather than a metered per-request charge, which the table caption
records.
"""
import json, os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

A = json.load(open(os.path.expanduser("~/Documents/Research/CellPress/analysis12/costadj12.json")))
S = json.load(open(os.path.expanduser("~/Documents/Research/CellPress/analysis12/stats12.json")))["per_model"]
OUT = os.path.expanduser("~/Downloads/paper-figures/twelve-model"); os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"pdf.fonttype":42,"ps.fonttype":42,"font.family":"sans-serif",
 "font.sans-serif":["Helvetica","Arial","DejaVu Sans"],"font.size":8,"axes.linewidth":0.6,
 "legend.fontsize":6.6,"legend.frameon":False,"figure.dpi":300,"savefig.dpi":300,
 "savefig.bbox":"standard","mathtext.fontset":"custom","mathtext.rm":"Helvetica"})
INK,MUT,LINE="#151B21","#5C6B75","#D6DEE3"
SMALL_C,LARGE_C,HUM,FR="#C77A30","#0F6E8C","#151B21","#9AA7B0"
TIER = {"Qwen 3.7 Plus":"L","GPT-5.6 Luna":"L","DeepSeek V4 Pro":"L","Llama 4 Maverick":"L",
        "Mistral Large 3":"L","MedGemma 27B":"L","DeepSeek V4 Flash":"S","Qwen 3.5 9B":"S",
        "GPT-OSS 20B":"S","Ministral 8B":"S","Llama 3.1 8B":"S","MedGemma 4B":"S"}
MCACC  = {m: S[m]["mc4"]  for m in A}
# Twelve points crowd the upper right of both panels, so label placement is
# manual: (dx, dy in points, horizontal alignment).
OFF={"A":{"Qwen 3.7 Plus":(0,12,"center"),"GPT-5.6 Luna":(13,5,"left"),
 "DeepSeek V4 Pro":(11,-9,"left"),"DeepSeek V4 Flash":(0,12,"center"),
 "Llama 4 Maverick":(-4,12,"right"),"Mistral Large 3":(14,3,"left"),
 "Qwen 3.5 9B":(-2,-12,"right"),"GPT-OSS 20B":(-9,4,"right"),
 "Ministral 8B":(0,-12,"center"),"Llama 3.1 8B":(11,-2,"left"),
 "MedGemma 27B":(0,-14,"center"),"MedGemma 4B":(0,-13,"center")},
 "B":{"GPT-5.6 Luna":(0,12,"center"),"Qwen 3.7 Plus":(0,12,"center"),
 "DeepSeek V4 Pro":(13,-3,"left"),"DeepSeek V4 Flash":(0,12,"center"),
 "Mistral Large 3":(0,12,"center"),"Llama 4 Maverick":(0,-12,"center"),
 "Qwen 3.5 9B":(11,2,"left"),"GPT-OSS 20B":(-9,5,"right"),
 "Ministral 8B":(0,-12,"center"),"Llama 3.1 8B":(11,-2,"left"),
 "MedGemma 27B":(0,-13,"center"),"MedGemma 4B":(0,-13,"center")}}
GENACC = {m: S[m]["gen4"] for m in A}

def frontier(pts):
    keep=[]; best=-1
    for n,c,a in sorted(pts,key=lambda x:x[1]):
        if a>best: keep.append(n); best=a
    return set(keep)

fig,axes=plt.subplots(1,2,figsize=(6.9,3.6))
for ax,tag in zip(axes,["A","B"]):
    if tag=="A":
        pts=[(n,A[n][0],MCACC[n]) for n in A]; title="Multiple choice (Setting 1)"
        ylim=(60,93); ticks=[65,70,75,80,85,90]; hum=86.5
    else:
        pts=[(n,A[n][2],GENACC[n]) for n in A]; title="Answer choices withheld (Setting 2)"
        ylim=(26,72); ticks=[30,40,50,60,70]; hum=None
    fset=frontier(pts)
    fr=sorted([p for p in pts if p[0] in fset],key=lambda x:x[1])
    ax.plot([p[1] for p in fr],[p[2] for p in fr],color=FR,lw=1.3,ls="--",zorder=2)
    if hum:
        ax.axhline(hum,color=HUM,lw=1.1,ls=(0,(3,2)),zorder=2)
        ax.text(0.035,hum+0.9,"Human expert (86.5)",fontsize=6.2,color=HUM,
                fontweight="bold",va="bottom")
    for n,c,a in pts:
        col=SMALL_C if TIER[n]=="S" else LARGE_C
        ax.scatter(c,a,s=105,color=col,edgecolor="white",lw=1.3,zorder=6)
    ax.set_xscale("log"); ax.set_ylim(*ylim); ax.set_yticks(ticks)
    if tag=="A":
        ax.set_xlim(0.032,12.0); tk=[0.05,0.1,0.25,0.5,1,2,4,8]
        lb=["0.05","0.10","0.25","0.50","1.00","2.00","4.00","8.00"]
    else:
        ax.set_xlim(0.03,11.0); tk=[0.05,0.1,0.3,1,3]
        lb=["0.05","0.10","0.30","1.00","3.00"]
    ax.set_xticks(tk); ax.set_xticklabels(lb); ax.minorticks_off()
    ax.set_xlabel("Cost per 1,000 questions (USD, log)",fontsize=7.6,color=INK,labelpad=2)
    ax.tick_params(labelsize=6.8,colors=MUT,length=0)
    ax.grid(color=LINE,lw=0.5,zorder=0); ax.set_axisbelow(True)
    ax.set_title(title,fontsize=8.3,fontweight="bold",color=INK,pad=6)
    for sp in ("top","right"): ax.spines[sp].set_visible(False)
    ax.spines["left"].set_color(LINE); ax.spines["bottom"].set_color(LINE)
    # labels placed after all points exist, nudged to avoid the marker
    for n,c,a in pts:
        dx,dy,ha=OFF[tag][n]
        ax.annotate(n,(c,a),xytext=(dx,dy),
                    textcoords="offset points",ha=ha,va="center",fontsize=5.9,
                    color=INK if n in fset else MUT,
                    fontweight="bold" if n in fset else "normal",zorder=7)
axes[0].set_ylabel("Accuracy on MedQA-USMLE + MedMCQA (%)",fontsize=7.6,color=INK)
handles=[Line2D([],[],marker="o",ls="none",color=LARGE_C,ms=6,label="Larger model"),
         Line2D([],[],marker="o",ls="none",color=SMALL_C,ms=6,label="Smaller model")]
fig.legend(handles=handles,loc="lower center",ncol=2,bbox_to_anchor=(0.5,-0.01),
           columnspacing=2.0,handletextpad=0.4)
fig.tight_layout(rect=(0,0.06,1,1))
fig.savefig(os.path.join(OUT,"Fig6_cost_benefit.pdf"))
fig.savefig(os.path.join(OUT,"Fig6_cost_benefit.png"),dpi=200)
print("wrote Fig6_cost_benefit")
