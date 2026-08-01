#!/usr/bin/env python3
"""Supplementary figures S1-S3: prompt templates, judge templates, reasoning trace.
All text is verbatim from the archived run artifacts. Vertical spacing is derived
from the figure height in inches so line spacing never collapses."""
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
SYS,USR,AST="#5C6B75","#0F6E8C","#2E7D5B"
MFS=6.0                                   # mono font size
def pt(x,H): return x/72.0/H              # points -> axes fraction

class Sheet:
    def __init__(self,W,H):
        self.W,self.H=W,H
        self.fig=plt.figure(figsize=(W,H))
        self.ax=self.fig.add_axes([0,0,1,1])
        self.ax.set_xlim(0,1); self.ax.set_ylim(0,1); self.ax.axis("off")
        self.LH=pt(8.6,H)                  # mono line height
    def title(self,y,txt,fs=9.4):
        self.ax.text(0.02,y,txt,fontsize=fs,fontweight="bold",color=INK,va="top")
        return y-pt(20,self.H)
    def band(self,y,txt,accent,note=""):
        h=pt(15,self.H)
        self.ax.add_patch(FancyBboxPatch((0.02,y-h),0.96,h,
            boxstyle="round,pad=0,rounding_size=0.006",fc=accent,ec="none",zorder=3))
        self.ax.text(0.033,y-h/2,txt,ha="left",va="center",fontsize=8.0,
                     fontweight="bold",color="white",zorder=4)
        if note: self.ax.text(0.966,y-h/2,note,ha="right",va="center",fontsize=6.2,
                              color="white",zorder=4)
        return y-h-pt(8,self.H)
    def role(self,y,role,note=""):
        col={"system":SYS,"user":USR,"assistant":AST}[role]
        self.ax.text(0.035,y,role.upper(),ha="left",va="center",fontsize=5.8,
                     color=col,fontweight="bold",zorder=5)
        if note:
            off=0.030+0.0072*len(role)
            self.ax.text(0.035+off,y,note,ha="left",va="center",fontsize=5.8,
                         color=MUT,zorder=5)
        return y-pt(9,self.H)
    def bubble(self,y,text,role,wrapn=92,x=0.035,w=0.925):
        col={"system":SYS,"user":USR,"assistant":AST}[role]
        tint={"system":"#F2F4F6","user":"#EDF4F7","assistant":"#F1F7F3"}[role]
        lines=[]
        for para in text.split("\n"):
            lines += textwrap.wrap(para,wrapn) or [""]
        padv=pt(6,self.H)
        h=self.LH*len(lines)+2*padv
        self.ax.add_patch(FancyBboxPatch((x,y-h),w,h,
            boxstyle="round,pad=0,rounding_size=0.006",fc=tint,ec=col,lw=0.8,zorder=3))
        self.ax.add_patch(FancyBboxPatch((x,y-h),0.0035,h,boxstyle="square,pad=0",
            fc=col,ec="none",zorder=4))
        ty=y-padv-self.LH/2
        for ln in lines:
            self.ax.text(x+0.012,ty,ln,ha="left",va="center",color=INK,zorder=5,
                         family="monospace",fontsize=MFS)
            ty-=self.LH
        return y-h-pt(8,self.H)
    def foot(self,y,txt):
        self.ax.text(0.02,y,txt,fontsize=6.3,color=MUT,va="top")
        return y-pt(14,self.H)
    def save(self,name,ybot):
        if ybot < 0.01:
            print(f"  !! {name}: content overflows canvas (final y={ybot:.3f}); increase height")
        crop=Bbox.from_extents(0.0,max(0.0,ybot-pt(6,self.H))*self.H,self.W,self.H)
        self.fig.savefig(os.path.join(OUT,name+".pdf"),bbox_inches=crop)
        self.fig.savefig(os.path.join(OUT,name+".png"),dpi=180,bbox_inches=crop)
        plt.close(self.fig); print("wrote",name)

def disc(ax,x,y,rx,fc,W,H,z=6):
    ax.add_patch(Ellipse((x,y),2*rx,2*rx*W/H,fc=fc,ec="white",lw=1.0,zorder=z))

def badge(ax,x,y,r,fc,mono,label=None,fs=6.8,sub=None,H=1.0,W=6.9):
    disc(ax,x,y,r,fc,W,H)
    ax.text(x,y,mono,ha="center",va="center",fontsize=fs,color="white",
            fontweight="bold",zorder=7)
    if label: ax.text(x+r+0.010,y,label,ha="left",va="center",fontsize=7.4,
                      color=INK,fontweight="bold",zorder=7)
    if sub: ax.text(x+r+0.010,y-pt(11,H),sub,ha="left",va="center",fontsize=6.2,
                    color=MUT,zorder=7)

SYS_P=("As a healthcare professional, provide an expert response to each question. "
       "Exactly one answer option is the most correct.")
MC_U="Question: {question}\n(A) {choice A}   (B) {choice B}   (C) {choice C}   (D) {choice D}"
MC_A="Explanation: {worked rationale}\nAnswer: Therefore, the answer is ({letter}) {choice text}."
GEN_U=("Answer the following medical question without seeing answer choices. Give a concise "
       "answer and end with exactly `Answer: <answer phrase>`.\n\nQuestion: {question}")
REC_U=("Below are answer choices from a medical multiple-choice item, but the question stem is "
       "hidden. Reconstruct one most likely medical question for which these choices are plausible."
       "\n\nItem genre: {dataset-specific genre hint}. Reconstruct the minimal stem supported by "
       "the choices; do not invent unsupported patient details.\n\nOutput only the reconstructed "
       "question, ending with a question mark.\n\nChoices:\nA. {choice A}\nB. {choice B}\n"
       "C. {choice C}\nD. {choice D}")

# ---------------------------------------------------------------- S1
s=Sheet(6.9,8.3)
y=s.title(0.984,"Prompt templates, verbatim from the run configuration")
for name,accent,msgs,note,tags in [
 ("Setting 1  ·  Multiple choice",TEAL,
  [("system",SYS_P),("user",MC_U),("assistant",MC_A),("user",MC_U)],
  "five worked exemplars precede the target question",
  {1:"exemplar 1 of 5",2:"exemplar response",3:"target question"}),
 ("Setting 2  ·  Answer choices withheld",AMBER,
  [("system",SYS_P),("user",GEN_U)],"no exemplars",{}),
 ("Setting 3  ·  Question reconstruction",PLUM,
  [("user",REC_U)],"no exemplars, no system message",{})]:
    y=s.band(y,name,accent,note)
    for k,(role,txt) in enumerate(msgs):
        y=s.role(y,role,tags.get(k,""))
        y=s.bubble(y,txt,role)
        if name.startswith("Setting 1") and k==2:
            s.ax.text(0.5,y,"$\\vdots$   exemplars 2–5 omitted   $\\vdots$",ha="center",
                      va="center",fontsize=6.4,color=MUT); y-=pt(16,s.H)
    y-=pt(6,s.H)
y=s.foot(y,"Braces mark fields substituted per question. The five-shot exemplars in Setting 1 are "
            "fixed across models and datasets.")
s.save("FigS1_prompt_templates",y)

# ---------------------------------------------------------------- S2
J1S=("You are an impartial evaluator of free-form medical answers. Return only the requested "
     "JSON object and no chain-of-thought.")
J1U=("Determine whether the model response gives the same substantive answer as the reference "
     "answer. Allow clinically equivalent wording, but mark false when the response is incomplete, "
     "unsupported, or gives a materially different diagnosis, fact, relationship, or treatment."
     "\n\nQUESTION:\n{question stem}\n\nREFERENCE CORRECT ANSWER:\n{keyed choice text}\n\n"
     "MODEL RESPONSE:\n{free-text answer}\n\n"
     'Return JSON only: {"semantic_correct": <true or false>, "rationale": "<maximum 15 words>"}')
J2S=("You are an impartial evaluator of reconstructed medical questions. Return only the requested "
     "JSON object and no chain-of-thought.")
J2U=("Evaluate whether the generated question recovers the same medical knowledge target as the "
     "reference question. Allow paraphrases. Reject a question that is merely about the same broad "
     "topic but tests a different fact, relationship, diagnosis, or treatment."
     "\n\nREFERENCE QUESTION:\n{original stem}\n\nGENERATED QUESTION:\n{reconstructed stem}\n\n"
     'Return JSON only: {"target_match": <true or false>, "fidelity": <integer 1-5>, '
     '"rationale": "<maximum 20 words>"}')
s=Sheet(6.9,7.2)
y=s.title(0.980,"Judge templates, verbatim from the run configuration")
s.ax.text(0.02,y-pt(5,s.H),"Judge: Claude Sonnet 5",ha="left",va="center",
          fontsize=8.0,fontweight="bold",color=INK)
s.ax.text(0.02,y-pt(16,s.H),"temperature 0  ·  one request per judgment, no reasoning trace",
          ha="left",va="center",fontsize=6.3,color=MUT)
y-=pt(32,s.H)
for name,accent,msgs in [("Grading a generative answer (Setting 2)",AMBER,
                          [("system",J1S),("user",J1U)]),
                         ("Grading a reconstructed question (Setting 3)",PLUM,
                          [("system",J2S),("user",J2U)])]:
    y=s.band(y,name,accent)
    for role,txt in msgs:
        y=s.role(y,role); y=s.bubble(y,txt,role)
    y-=pt(6,s.H)
y=s.foot(y,"The judge never sees the keyed letter when grading a generative answer, and returns a "
            "single JSON verdict.")
s.save("FigS2_judge_templates",y)

# ---------------------------------------------------------------- S3
Q="Cold agglutinin is"
OPTS=[("A","IgG",False),("B","IgM",True),("C","IgA",False),("D","IgD",False)]
TRACE=[(1,"Cold agglutinins are antibodies that cause clumping (agglutination) of red blood"),
       (1,"cells at low temperatures, typically below normal body temperature."),
       (2,"These antibodies are most active at temperatures around 4°C but can also react at"),
       (2,"higher temperatures (up to 30–37°C in some cases)."),
       (3,"Cold agglutinins are primarily associated with autoimmune hemolytic anemia and"),
       (3,"certain infections, such as Mycoplasma pneumoniae and Epstein-Barr virus."),
       (4,"Immunoglobulin classes differ in structure and function. IgM is a pentameric"),
       (4,"antibody, making it highly efficient at agglutinating cells by binding multiple"),
       (4,"antigens simultaneously."),
       (5,"IgG, IgA, and IgD are not typically associated with cold agglutination. IgG is more"),
       (5,"commonly involved in warm autoimmune hemolytic anemia."),
       (6,"Therefore, cold agglutinins are predominantly of the IgM class.")]
s=Sheet(6.9,4.6); H=s.H; ax=s.ax
y=s.title(0.972,"A chain-of-thought trace under Setting 1  (Multiple choice)")
# question card, height fitted to its contents
card_h=pt(30,H)+len(OPTS)*pt(13,H)
ax.add_patch(FancyBboxPatch((0.02,y-card_h),0.46,card_h,
            boxstyle="round,pad=0,rounding_size=0.008",fc="#F7F9FA",ec=LINE,lw=0.9))
ax.text(0.034,y-pt(9,H),"MedMCQA question",fontsize=6.2,color=MUT,fontweight="bold",va="center")
ax.text(0.034,y-pt(21,H),Q,fontsize=7.6,color=INK,va="center")
oy=y-pt(34,H)
for lab,txt,hot in OPTS:
    ax.add_patch(FancyBboxPatch((0.034,oy-pt(4.4,H)),0.021,pt(9,H),boxstyle="square,pad=0",
                fc=TEAL if hot else "#DDE5E9",ec="none"))
    ax.text(0.0445,oy,lab,ha="center",va="center",fontsize=5.6,color="white" if hot else MUT)
    ax.text(0.062,oy,txt,ha="left",va="center",fontsize=6.9,color=INK,
            fontweight="bold" if hot else "normal")
    if hot: ax.text(0.100,oy,"keyed answer",ha="left",va="center",fontsize=5.9,color=TEAL)
    oy-=pt(13,H)
ax.text(0.520,y-pt(7,H),"Mistral Large 3",ha="left",va="center",fontsize=8.0,
        fontweight="bold",color=INK)
ax.text(0.520,y-pt(18,H),"temperature 0  ·  reasoning emitted as visible output",
        ha="left",va="center",fontsize=6.3,color=MUT)
y=y-card_h-pt(12,H)
# output panel, fitted
out_h=pt(16,H)+s.LH*len(TRACE)+pt(30,H)
ax.add_patch(FancyBboxPatch((0.02,y-out_h),0.96,out_h,
            boxstyle="round,pad=0,rounding_size=0.008",fc="#F1F7F3",ec=AST,lw=0.9,zorder=2))
ax.add_patch(FancyBboxPatch((0.02,y-out_h),0.0035,out_h,boxstyle="square,pad=0",
            fc=AST,ec="none",zorder=3))
ty=y-pt(11,H)
ax.text(0.036,ty,"Step-by-step reasoning:",fontsize=6.4,color=INK,fontweight="bold",
        zorder=4,family="monospace"); ty-=pt(13,H)
prev=None
for st,ln in TRACE:
    if st!=prev:
        disc(ax,0.044,ty,0.0062,AST,s.W,H,z=5)
        ax.text(0.044,ty,str(st),ha="center",va="center",fontsize=5.0,color="white",
                fontweight="bold",zorder=6); prev=st
    ax.text(0.060,ty,ln,ha="left",va="center",color=INK,zorder=4,
            family="monospace",fontsize=MFS)
    ty-=s.LH
ty-=pt(3,H)
ax.plot([0.036,0.964],[ty,ty],color="#C9DED2",lw=0.8,zorder=4); ty-=pt(13,H)
ax.text(0.036,ty,"Answer: (B)",fontsize=7.2,color=AST,fontweight="bold",zorder=4,
        family="monospace")
ax.text(0.152,ty,"parsed as B  ·  matches the key",fontsize=6.2,color=MUT,zorder=4)
y=y-out_h-pt(10,H)
y=s.foot(y,"The final line is the only part the multiple-choice parser reads; the numbered steps "
            "are the model's own formatting.")
s.save("FigS3_reasoning_example",y)
