#!/usr/bin/env python3
"""Ensemble by pool: all ten, larger only, smaller only. Majority vote against
each pool's own best single member. Panel widths are proportional to pool size,
so one unit of k is the same distance in every panel."""
import os,json
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
OUT=os.path.expanduser("~/Downloads/paper-figures"); os.makedirs(OUT,exist_ok=True)
plt.rcParams.update({"pdf.fonttype":42,"ps.fonttype":42,"font.family":"sans-serif",
 "font.sans-serif":["Helvetica","Arial","DejaVu Sans"],"font.size":8,"axes.linewidth":0.6,
 "legend.fontsize":7,"legend.frameon":False,"figure.dpi":300,"savefig.dpi":300,
 "savefig.bbox":"standard","mathtext.fontset":"custom","mathtext.rm":"Helvetica"})
INK,MUT,LINE="#151B21","#5C6B75","#D6DEE3"
ALL_C,LG,SM="#4A5A64","#0F6E8C","#C77A30"
UP,DOWN="#CFE2EA","#F2E0CC"
D=json.load(open("/tmp/ens.json"))
BEST={"all":88.39,"large":88.39,"small":86.66}
PANELS=[("all",ALL_C,"All ten models","10 models"),
        ("large",LG,"Larger models only","5 models"),
        ("small",SM,"Smaller models only","5 models")]
fig=plt.figure(figsize=(6.9,2.98))
gs=fig.add_gridspec(1,3,width_ratios=[1.45,1,1],wspace=0.13,
                    left=0.085,right=0.988,top=0.845,bottom=0.185)
axes=[fig.add_subplot(gs[0,i]) for i in range(3)]
for ax,(tag,col,title,sub) in zip(axes,PANELS):
    v=np.array(D[tag]["vote"]); x=np.arange(1,len(v)+1); b=BEST[tag]
    ax.fill_between(x,b,v,where=(v>=b),color=UP,zorder=2,lw=0,interpolate=True)
    ax.fill_between(x,b,v,where=(v<b),color=DOWN,zorder=2,lw=0,interpolate=True)
    ax.axhline(b,color=INK,lw=1.0,ls=(0,(2,2)),zorder=4)
    ax.plot(x,v,"-o",color=col,lw=1.9,ms=4.6,zorder=6,
            markeredgecolor="white",markeredgewidth=0.9)
    i=int(np.argmax(v)); d=v[i]-b
    ax.annotate(f"{v[i]:.1f}", xy=(x[i],v[i]), xytext=(0,10),
                textcoords="offset points", ha="center", fontsize=6.5,
                color=col, fontweight="bold")
    ax.scatter([x[-1]],[v[-1]],s=52,facecolor="white",edgecolor=col,lw=1.5,zorder=7)
    ax.annotate(f"{v[-1]:.1f}", xy=(x[-1],v[-1]), xytext=(0,-13),
                textcoords="offset points", ha="center", fontsize=6.5,
                color=col, fontweight="bold")
    ax.set_title(title,fontsize=8.6,fontweight="bold",color=col,pad=13)
    ax.text(0.5,1.015,sub,transform=ax.transAxes,ha="center",va="bottom",
            fontsize=6.3,color=MUT)
    n=len(v)
    ax.set_xticks(range(1,n+1)); ax.set_xlim(0.65,n+0.35)
    ax.set_ylim(83.4,91.2); ax.set_yticks([84,86,88,90])
    ax.tick_params(labelsize=7,colors=MUT,length=0)
    ax.grid(axis="y",color=LINE,lw=0.5,zorder=0); ax.set_axisbelow(True)
    for sp in ("top","right"): ax.spines[sp].set_visible(False)
    ax.spines["left"].set_color(LINE); ax.spines["bottom"].set_color(LINE)
    if ax is not axes[0]:
        ax.set_yticklabels([])
        ax.spines["left"].set_visible(False)
axes[0].set_ylabel("Majority-vote accuracy (%)",fontsize=8.2,color=INK)
axes[1].set_xlabel("Models in the committee ($k$)",fontsize=8.0,color=INK,labelpad=2)
axes[0].annotate("best single model (88.4)",xy=(6.5,BEST["all"]),xytext=(6.5,86.9),
                 ha="center",va="top",fontsize=6.4,color=INK,fontweight="bold",
                 arrowprops=dict(arrowstyle="-",color=INK,lw=0.7,shrinkA=2,shrinkB=0))
axes[2].text(2.6,84.5,"voting never beats\nthe best member",ha="center",va="center",
             fontsize=6.4,color=SM,fontweight="bold",linespacing=1.4)
leg=[Line2D([],[],color=INK,lw=1.9,marker="o",ms=5,markeredgecolor="white",label="Majority vote"),
     Line2D([],[],color=INK,lw=1.0,ls=(0,(2,2)),label="Pool's best single model"),
     Patch(facecolor=UP,label="Voting gains"),Patch(facecolor=DOWN,label="Voting loses")]
fig.legend(handles=leg,loc="lower center",ncol=4,bbox_to_anchor=(0.5,0.012),
           columnspacing=1.7,handletextpad=0.5)
fig.savefig(os.path.join(OUT,"Main_Figure7_ensemble_v4.pdf"))
fig.savefig(os.path.join(OUT,"Main_Figure7_ensemble_v4.png"),dpi=190)
print("wrote v4")
