#!/usr/bin/env python3
"""Finding 6: cost vs accuracy. Setting 2 cost is put on an equal-prompt basis
(Setting-1 prompt cost + Setting-2 completion cost) so the two are comparable."""
import os,json
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
OUT=os.path.expanduser("~/Downloads/paper-figures"); os.makedirs(OUT,exist_ok=True)
plt.rcParams.update({"pdf.fonttype":42,"ps.fonttype":42,"font.family":"sans-serif",
 "font.sans-serif":["Helvetica","Arial","DejaVu Sans"],"font.size":8,"axes.linewidth":0.6,
 "legend.fontsize":7,"legend.frameon":False,"figure.dpi":300,"savefig.dpi":300,
 "savefig.bbox":"standard","mathtext.fontset":"custom","mathtext.rm":"Helvetica"})
INK,MUT,LINE="#151B21","#5C6B75","#D6DEE3"
SMALL_C,LARGE_C,HUM,FR="#C77A30","#0F6E8C","#151B21","#9AA7B0"
A=json.load(open("/tmp/costadj.json"))   # model -> [mc $/1kq, gen observed, gen adjusted]
MCACC={"Qwen 3.7 Plus":89.9,"GPT-5.6 Luna":89.9,"DeepSeek V4 Pro":89.5,"DeepSeek V4 Flash":88.6,
 "Llama 4 Maverick":85.3,"Mistral Large 3":85.2,"Qwen 3.5 9B":84.1,"GPT-OSS 20B":78.0,
 "Ministral 8B":73.7,"Llama 3.1 8B":63.5}
GENACC={"GPT-5.6 Luna":65.5,"Qwen 3.7 Plus":60.6,"DeepSeek V4 Pro":59.7,"DeepSeek V4 Flash":58.8,
 "Llama 4 Maverick":53.7,"Mistral Large 3":57.7,"Qwen 3.5 9B":46.9,"GPT-OSS 20B":49.8,
 "Ministral 8B":41.8,"Llama 3.1 8B":31.1}
TIER={"Qwen 3.7 Plus":"L","GPT-5.6 Luna":"L","DeepSeek V4 Pro":"L","Llama 4 Maverick":"L",
 "Mistral Large 3":"L","DeepSeek V4 Flash":"S","Qwen 3.5 9B":"S","GPT-OSS 20B":"S",
 "Ministral 8B":"S","Llama 3.1 8B":"S"}
def frontier(p):
    o=[];b=-1
    for n,c,a in sorted(p,key=lambda x:x[1]):
        if a>b: o.append(n); b=a
    return set(o)
OFF={"A":{"Qwen 3.7 Plus":(0,12,"center"),"GPT-5.6 Luna":(0,-12,"center"),
 "DeepSeek V4 Pro":(9,-4,"left"),"DeepSeek V4 Flash":(0,12,"center"),
 "Llama 4 Maverick":(-9,-6,"right"),"Mistral Large 3":(0,-12,"center"),
 "Qwen 3.5 9B":(9,4,"left"),"GPT-OSS 20B":(9,-3,"left"),
 "Ministral 8B":(9,-3,"left"),"Llama 3.1 8B":(10,-3,"left")},
 "B":{"GPT-5.6 Luna":(0,11,"center"),"Qwen 3.7 Plus":(0,-12,"center"),
 "DeepSeek V4 Pro":(-10,4,"right"),"DeepSeek V4 Flash":(0,12,"center"),
 "Mistral Large 3":(9,-6,"left"),"Llama 4 Maverick":(9,-5,"left"),
 "Qwen 3.5 9B":(9,5,"left"),"GPT-OSS 20B":(-9,5,"right"),
 "Ministral 8B":(9,-4,"left"),"Llama 3.1 8B":(10,-3,"left")}}
fig,axes=plt.subplots(1,2,figsize=(6.9,3.35))
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
        ax.text(0.035,hum+0.9,"Human expert (86.5)",fontsize=6.2,color=HUM,fontweight="bold",va="bottom")
    if tag=="B":   # ghost markers = observed (short-prompt) cost, arrow to adjusted
        for n in A:
            ax.annotate("",xy=(A[n][2],GENACC[n]),xytext=(A[n][1],GENACC[n]),
                        arrowprops=dict(arrowstyle="-",color="#C9D2D8",lw=0.9),zorder=1)
            ax.scatter(A[n][1],GENACC[n],s=26,facecolor="none",edgecolor="#B7C2CA",lw=0.9,zorder=3)
    for n,c,a in pts:
        col=SMALL_C if TIER[n]=="S" else LARGE_C
        ax.scatter(c,a,s=105,color=col,edgecolor="white",lw=1.3,zorder=6)
        dx,dy,ha=OFF[tag][n]
        ax.annotate(n,(c,a),xytext=(dx,dy),textcoords="offset points",ha=ha,va="center",
                    fontsize=6.0,color=INK if n in fset else MUT,
                    fontweight="bold" if n in fset else "normal",zorder=7)
    ax.set_xscale("log"); ax.set_ylim(*ylim); ax.set_yticks(ticks)
    if tag=="A":
        ax.set_xlim(0.032,7.0); tk=[0.05,0.1,0.25,0.5,1,2,4]
        lb=["0.05","0.10","0.25","0.50","1.00","2.00","4.00"]
    else:
        ax.set_xlim(0.006,7.0); tk=[0.01,0.03,0.1,0.3,1,3]
        lb=["0.01","0.03","0.10","0.30","1.00","3.00"]
    ax.set_xticks(tk); ax.set_xticklabels(lb)
    ax.minorticks_off()
    ax.set_xlabel("Cost per 1,000 questions (USD, log)",fontsize=7.6,color=INK,labelpad=2)
    ax.tick_params(labelsize=6.8,colors=MUT,length=0)
    ax.grid(color=LINE,lw=0.5,zorder=0); ax.set_axisbelow(True)
    ax.set_title(title,fontsize=8.3,fontweight="bold",color=INK,pad=6)
    for sp in ("top","right"): ax.spines[sp].set_visible(False)
    ax.spines["left"].set_color(LINE); ax.spines["bottom"].set_color(LINE)
axes[0].set_ylabel("Accuracy (%)",fontsize=8.2,color=INK)
axes[1].text(0.0068,70.5,"hollow = as run (short prompt)\nfilled = equal-prompt basis",
             fontsize=6.1,color=MUT,va="top",linespacing=1.5)
leg=[Line2D([],[],marker="o",ls="none",markerfacecolor=SMALL_C,markeredgecolor="white",
            markeredgewidth=1.3,ms=8,label="Smaller model"),
     Line2D([],[],marker="o",ls="none",markerfacecolor=LARGE_C,markeredgecolor="white",
            markeredgewidth=1.3,ms=8,label="Larger model"),
     Line2D([],[],color=FR,lw=1.3,ls="--",label="Cost-accuracy frontier"),
     Line2D([],[],color=HUM,lw=1.1,ls=(0,(3,2)),label="Human expert")]
fig.legend(handles=leg,loc="lower center",ncol=4,bbox_to_anchor=(0.5,0.012),
           columnspacing=1.6,handletextpad=0.5)
fig.subplots_adjust(left=0.082,right=0.987,top=0.91,bottom=0.185,wspace=0.16)
fig.savefig(os.path.join(OUT,"Main_Figure6_cost_benefit.pdf"))
fig.savefig(os.path.join(OUT,"Main_Figure6_cost_benefit.png"),dpi=190)
print("wrote Main_Figure6_cost_benefit (equal-prompt basis)")
