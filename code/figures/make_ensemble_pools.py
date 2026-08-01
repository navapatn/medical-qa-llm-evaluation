#!/usr/bin/env python3
"""Ensemble by model tier: all ten, larger-only, smaller-only."""
import os,json
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
OUT=os.path.expanduser("~/Downloads/paper-figures"); os.makedirs(OUT,exist_ok=True)
plt.rcParams.update({"pdf.fonttype":42,"ps.fonttype":42,"font.family":"sans-serif",
 "font.sans-serif":["Helvetica","Arial","DejaVu Sans"],"font.size":8,"axes.linewidth":0.6,
 "legend.fontsize":7,"legend.frameon":False,"figure.dpi":300,"savefig.dpi":300,
 "savefig.bbox":"standard","mathtext.fontset":"custom","mathtext.rm":"Helvetica"})
INK,MUT,LINE="#151B21","#5C6B75","#D6DEE3"
ALL_C,LG_C,SM_C="#5C6B75","#0F6E8C","#C77A30"
D=json.load(open("/tmp/ens.json"))
BEST=88.39
fig,axes=plt.subplots(1,2,figsize=(6.9,3.3),sharex=True)
for ax,key,title,ylim in [(axes[0],"vote","Majority vote",(83,97)),
                          (axes[1],"any","At least one model correct",(83,97))]:
    for tag,col,lab,lw,ms,z in [("all",ALL_C,"All ten models",3.4,6.2,3),
                                ("large",LG_C,"Larger models only",1.6,3.8,5),
                                ("small",SM_C,"Smaller models only",1.6,3.8,5)]:
        y=D[tag][key]; x=np.arange(1,len(y)+1)
        ax.plot(x,y,"-o",color=col,lw=lw,ms=ms,zorder=z,alpha=0.45 if tag=="all" else 1.0,
                markeredgecolor="white",markeredgewidth=0.7,label=lab)
    ax.axhline(BEST,color=INK,lw=1.0,ls=(0,(3,2)),zorder=2)
    ax.set_title(title,fontsize=8.6,fontweight="bold",color=INK,pad=6)
    ax.set_xlabel("Models in the committee ($k$)",fontsize=7.8,color=INK)
    ax.set_xticks(range(1,11)); ax.set_xlim(0.6,10.4); ax.set_ylim(*ylim)
    ax.set_yticks([85,90,95])
    ax.tick_params(labelsize=7,colors=MUT,length=0)
    ax.grid(color=LINE,lw=0.5,zorder=0); ax.set_axisbelow(True)
    for sp in ("top","right"): ax.spines[sp].set_visible(False)
    ax.spines["left"].set_color(LINE); ax.spines["bottom"].set_color(LINE)
axes[0].set_ylabel("Accuracy (%)",fontsize=8.2,color=INK)
axes[0].text(10.2,BEST-0.75,"best single model (88.4)",ha="right",va="top",
             fontsize=6.3,color=INK,fontweight="bold")
axes[0].annotate("peak 89.5",xy=(4,89.54),xytext=(4.6,92.4),textcoords="data",
                 fontsize=6.4,color=LG_C,fontweight="bold",
                 arrowprops=dict(arrowstyle="-",color=LG_C,lw=0.8,shrinkA=0,shrinkB=3))
axes[0].annotate("voting never helps",xy=(3,85.86),xytext=(4.4,84.0),textcoords="data",
                 fontsize=6.4,color=SM_C,fontweight="bold",
                 arrowprops=dict(arrowstyle="-",color=SM_C,lw=0.8,shrinkA=0,shrinkB=3))
h,l=axes[0].get_legend_handles_labels()
h.append(Line2D([],[],color=INK,lw=1.0,ls=(0,(3,2)))); l.append("Best single model")
axes[0].text(1.15, 91.1, "optimal committees at k ≤ 4\nare all larger models", fontsize=6.2, color=MUT, va="center", linespacing=1.5)
fig.legend(h,l,loc="lower center",ncol=4,bbox_to_anchor=(0.5,-0.015),
           columnspacing=1.7,handletextpad=0.5)
fig.subplots_adjust(left=0.083,right=0.987,top=0.90,bottom=0.245,wspace=0.13)
fig.savefig(os.path.join(OUT,"Main_Figure7_ensemble_by_tier.pdf"))
fig.savefig(os.path.join(OUT,"Main_Figure7_ensemble_by_tier.png"),dpi=190)
print("wrote Main_Figure7_ensemble_by_tier")
