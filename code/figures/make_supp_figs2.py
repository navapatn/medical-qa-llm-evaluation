#!/usr/bin/env python3
"""Supplementary figures S4-S5: paired reasoning (small vs large, wrong vs right)
and questions every model answered incorrectly. All text verbatim from artifacts."""
import os, textwrap
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Ellipse
from matplotlib.transforms import Bbox
OUT=os.path.expanduser("~/Downloads/paper-figures/supp"); os.makedirs(OUT,exist_ok=True)
plt.rcParams.update({"pdf.fonttype":42,"ps.fonttype":42,"font.family":"sans-serif",
 "font.sans-serif":["Helvetica","Arial","DejaVu Sans"],"font.size":8,
 "figure.dpi":300,"savefig.dpi":300,"savefig.bbox":"standard"})
INK,MUT,LINE="#151B21","#5C6B75","#D3DBE0"
TEAL,AMBER,PLUM="#0F6E8C","#C77A30","#7E5A86"
OK,BAD="#2E7D5B","#B0563C"          # correct / incorrect  (reserved use)
def pt(x,H): return x/72.0/H

class Sheet:
    def __init__(self,W,H):
        self.W,self.H=W,H
        self.fig=plt.figure(figsize=(W,H))
        self.ax=self.fig.add_axes([0,0,1,1]); self.ax.set_xlim(0,1); self.ax.set_ylim(0,1)
        self.ax.axis("off"); self.LH=pt(8.4,H)
    def disc(self,x,y,rx,fc,z=6):
        self.ax.add_patch(Ellipse((x,y),2*rx,2*rx*self.W/self.H,fc=fc,ec="white",lw=1.0,zorder=z))
    def title(self,y,txt,fs=9.4):
        self.ax.text(0.02,y,txt,fontsize=fs,fontweight="bold",color=INK,va="top")
        return y-pt(21,self.H)
    def save(self,name,ybot):
        if ybot<0.01: print(f"  !! {name}: overflow (y={ybot:.3f})")
        crop=Bbox.from_extents(0.0,max(0.0,ybot-pt(6,self.H))*self.H,self.W,self.H)
        self.fig.savefig(os.path.join(OUT,name+".pdf"),bbox_inches=crop)
        self.fig.savefig(os.path.join(OUT,name+".png"),dpi=180,bbox_inches=crop)
        plt.close(self.fig); print("wrote",name)

# ==================================================== S4: paired reasoning
Q="The retrocuspid papilla is found approx 1mm below the free gingiva on the attached gingiva lingual to:"
CH=[("A","Mandibular canine.",True),("B","Maxillary canine.",False),
    ("C","Mandibular premolars",False),("D","Maxillary premolars.",False)]
SMALL=("Qwen 3.5 9B","smaller model","B",False,
 "Explanation: The retrocuspid papilla is a specific anatomical landmark found in the oral cavity. "
 "It is a small, distinct elevation of the attached gingiva located on the lingual aspect of the "
 "maxillary canine. It is situated approximately 1 mm below the free gingival margin. This feature "
 "is unique to the maxillary canine region and is not typically found on mandibular teeth or "
 "premolars. Therefore, the retrocuspid papilla is associated with the maxillary canine.\n"
 "Answer: (B)")
LARGE=("Qwen 3.7 Plus","larger model","A",True,
 "Explanation: The retrocuspid papilla is a small, circumscribed, and elevated area of normal "
 "gingival tissue. It is classically located on the attached gingiva, approximately 1 mm below the "
 "free gingival margin, specifically on the lingual aspect of the mandibular canine. While it is a "
 "normal anatomical landmark, it can occasionally become inflamed or hypertrophic.\n"
 "Answer: (A)")
s=Sheet(6.9,4.5); H=s.H; ax=s.ax
y=s.title(0.975,"Two models from the same provider on one question")
# question card full width
qlines=textwrap.wrap(Q,104)
qh=pt(16,H)+len(qlines)*pt(10,H)+len(CH)*pt(11.5,H)+pt(8,H)
ax.add_patch(FancyBboxPatch((0.02,y-qh),0.96,qh,
            boxstyle="round,pad=0,rounding_size=0.008",fc="#F7F9FA",ec=LINE,lw=0.9))
ty=y-pt(10,H)
ax.text(0.034,ty,"MedMCQA question",fontsize=6.2,color=MUT,fontweight="bold",va="center"); ty-=pt(12,H)
for ln in qlines:
    ax.text(0.034,ty,ln,fontsize=7.2,color=INK,va="center"); ty-=pt(10,H)
ty-=pt(2,H)
for lab,txt,key in CH:
    ax.add_patch(FancyBboxPatch((0.034,ty-pt(4.2,H)),0.020,pt(8.6,H),boxstyle="square,pad=0",
                fc=OK if key else "#DDE5E9",ec="none"))
    ax.text(0.044,ty,lab,ha="center",va="center",fontsize=5.6,color="white" if key else MUT)
    ax.text(0.060,ty,txt,ha="left",va="center",fontsize=6.8,color=INK,
            fontweight="bold" if key else "normal")
    if key: ax.text(0.215,ty,"keyed answer",ha="left",va="center",fontsize=5.9,color=OK)
    ty-=pt(11.5,H)
y=y-qh-pt(12,H)
# two columns
colw=0.468
for i,(nm,role,ans,ok,txt) in enumerate([SMALL,LARGE]):
    x=0.02+i*(colw+0.024)
    accent=OK if ok else BAD
    ax.text(x,y-pt(6,H),nm,ha="left",va="center",fontsize=8.0,fontweight="bold",color=INK)
    tag="CORRECT" if ok else "INCORRECT"
    ax.text(x+colw,y-pt(6,H),tag,ha="right",va="center",fontsize=6.6,
            fontweight="bold",color=accent)
    ax.text(x,y-pt(17,H),f"{role}  ·  answered ({ans})",ha="left",va="center",
            fontsize=6.3,color=MUT)
    yy=y-pt(26,H)
    lines=[]
    for para in txt.split("\n"):
        lines += textwrap.wrap(para,58) or [""]
    bh=pt(8,H)+len(lines)*s.LH
    ax.add_patch(FancyBboxPatch((x,yy-bh),colw,bh,boxstyle="round,pad=0,rounding_size=0.007",
                fc="#F6F9F7" if ok else "#FBF4F2",ec=accent,lw=0.9,zorder=2))
    ax.add_patch(FancyBboxPatch((x,yy-bh),0.0035,bh,boxstyle="square,pad=0",fc=accent,
                ec="none",zorder=3))
    tyy=yy-pt(6,H)-s.LH/2
    for ln in lines:
        w="bold" if ln.startswith("Answer:") else "normal"
        c=accent if ln.startswith("Answer:") else INK
        ax.text(x+0.012,tyy,ln,ha="left",va="center",color=c,zorder=4,
                family="monospace",fontsize=5.9,fontweight=w)
        tyy-=s.LH
    if i==0: ybot=yy-bh
    else: ybot=min(ybot,yy-bh)
y=ybot-pt(12,H)
ax.text(0.02,y,"Both models were given the identical prompt at temperature 0. The smaller model "
        "locates the landmark on the maxillary canine; the larger one on the mandibular canine.",
        fontsize=6.3,color=MUT,va="top")
s.save("FigS4_paired_reasoning",y-pt(14,H))

# ==================================================== S5: unanimous failures
HARD=[
 ("MedMCQA","Wave patterns of EEF, ECG and EMG are depicted below. The B pattern belongs to "
  "(Figure was not provided in the exam):",
  [("A","NREM sleep",False,False),("B","REM sleep",False,True),
   ("C","Wakefulness",True,False),("D","Quiet wakefulness",False,False)],
  "10 of 10","The stimulus the question refers to is absent, so the item cannot be answered from "
  "text alone. The question itself records that the figure was not provided."),
 ("MedMCQA","The police has brought an unresponsive patient to you. What is the first thing you will do?",
  [("A","Sta chest compressions immediately",False,False),("B","Check carotid pulse",True,False),
   ("C","Check for response and call help",False,True),("D","Sta rescue breaths",False,False)],
  "10 of 10","Every model selects the assess-and-summon-help step; the key selects the pulse check. "
  "Two choice strings are also truncated (\u201cSta\u201d for \u201cStart\u201d)."),
 ("MedMCQA","In mandibular primary second molar true statement is",
  [("A","ML is largest cusp and distobuccal is smallest",False,True),
   ("B","All buccal cusp are fo same size> all lingual cusp are of same size",True,False),
   ("C","DB is largest cusp",False,False),("D","All of the above",False,False)],
  "8 of 10","The keyed choice is typographically corrupted (\u201cfo\u201d for \u201cof\u201d, a stray "
  "\u201c>\u201d), leaving no well-formed correct choice."),
]
s=Sheet(6.9,7.4); H=s.H; ax=s.ax
y=s.title(0.982,"Questions that every model answered incorrectly")
ax.text(0.02,y,"Three of the 105 questions missed by all ten models, one per recurring pattern. "
        "Green marks the recorded key; red marks the answer the models converged on.",
        fontsize=6.6,color=MUT,va="top"); y-=pt(22,H)
for i,(dset,q,ch,cons,note) in enumerate(HARD,1):
    qlines=textwrap.wrap(q,100)
    nlines=textwrap.wrap(note,104)
    card=pt(15,H)+len(qlines)*pt(10,H)+pt(4,H)+len(ch)*pt(11.5,H)+pt(6,H)+len(nlines)*pt(9,H)+pt(10,H)
    ax.add_patch(FancyBboxPatch((0.02,y-card),0.96,card,
                boxstyle="round,pad=0,rounding_size=0.008",fc="#F8F9FA",ec=LINE,lw=0.9))
    ty=y-pt(11,H)
    ax.text(0.034,ty,f"Example {i}  ·  {dset}",fontsize=6.4,color=MUT,fontweight="bold",va="center")
    ax.text(0.966,ty,f"models agreeing on one non-key answer: {cons}",ha="right",va="center",
            fontsize=6.2,color=MUT); ty-=pt(13,H)
    for ln in qlines:
        ax.text(0.034,ty,ln,fontsize=7.1,color=INK,va="center"); ty-=pt(10,H)
    ty-=pt(4,H)
    for lab,txt,key,con in ch:
        fc=OK if key else (BAD if con else "#DDE5E9")
        ax.add_patch(FancyBboxPatch((0.034,ty-pt(4.2,H)),0.020,pt(8.6,H),boxstyle="square,pad=0",
                    fc=fc,ec="none"))
        ax.text(0.044,ty,lab,ha="center",va="center",fontsize=5.6,
                color="white" if (key or con) else MUT)
        ax.text(0.060,ty,txt,ha="left",va="center",fontsize=6.8,color=INK,
                fontweight="bold" if (key or con) else "normal")
        if key: ax.text(0.60,ty,"recorded key",ha="left",va="center",fontsize=5.9,color=OK)
        if con: ax.text(0.60,ty,"model consensus",ha="left",va="center",fontsize=5.9,color=BAD)
        ty-=pt(11.5,H)
    ty-=pt(4,H)
    for ln in nlines:
        ax.text(0.034,ty,ln,fontsize=6.3,color=MUT,va="center",style="italic"); ty-=pt(9,H)
    y=y-card-pt(11,H)
ax.text(0.02,y,"We describe these questions as warranting review rather than as confirmed errors: "
        "no independent clinician adjudication was obtained.",fontsize=6.3,color=MUT,va="top")
s.save("FigS5_hard_questions",y-pt(14,H))
