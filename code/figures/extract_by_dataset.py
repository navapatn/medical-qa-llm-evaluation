#!/usr/bin/env python3
"""Build per-dataset data for the ensemble-size figure (C) and the
reconstructable-pool figure (D). Validates against the pooled paper numbers."""
import json, glob, collections, itertools, os
import numpy as np

RUNS = ("<HOME>/Documents/Research/GRAIL-internship/cell-paper-replication/"
        "paper-replicator/codex_version/papers/cellpress-medical-reasoning/runs")
SD = os.path.dirname(os.path.abspath(__file__))

MAIN = {
 "GPT-5.6 Luna": "paper_gpt56_luna_main_k1_medium_t0_v3",
 "GPT-OSS 20B": "paper_gpt_oss20_main_k1_medium_t0_v1",
 "DeepSeek V4 Flash": "paper_deepseek_v4_flash_main_k1_high_t0_v1",
 "Llama 4 Maverick": "paper_llama4_maverick_main_k1_t0_v1",
 "Llama 3.1 8B": "paper_llama31_8b_main_k1_t0_v1",
 "Ministral 8B": "paper_ministral3_8b_main_k1_t0_v1",
 "Mistral Large 3": "paper_mistral_large3_main_k1_t0_v1",
 "Qwen 3.5 9B": "paper_qwen35_9b_main_k1_native_reasoning_t0_v5"}
JUDGE = {
 "GPT-5.6 Luna": "paper_claude_sonnet5_judge_gpt56_luna_short_reason_templateB_v1",
 "GPT-OSS 20B": "paper_claude_sonnet5_judge_gpt_oss20_short_reason_templateB_v1",
 "DeepSeek V4 Flash": "paper_claude_sonnet5_judge_deepseek_v4_flash_short_reason_v1",
 "DeepSeek V4 Pro": "paper_claude_sonnet5_judge_deepseek_v4_pro_short_reason_v1",
 "Llama 4 Maverick": "paper_claude_sonnet5_judge_llama4_maverick_short_reason_v1",
 "Llama 3.1 8B": "paper_claude_sonnet5_judge_llama31_8b_short_reason_v1",
 "Ministral 8B": "paper_claude_sonnet5_judge_ministral3_8b_short_reason_v1",
 "Mistral Large 3": "paper_claude_sonnet5_judge_mistral_large3_short_reason_v1",
 "Qwen 3.5 9B": "paper_claude_sonnet5_judge_qwen35_9b_short_reason_v1",
 "Qwen 3.7 Plus": "paper_claude_sonnet5_judge_qwen37_plus_short_reason_v1"}
MODELS = ["Qwen 3.7 Plus", "GPT-5.6 Luna", "DeepSeek V4 Pro", "DeepSeek V4 Flash",
          "Llama 4 Maverick", "Mistral Large 3", "Qwen 3.5 9B", "GPT-OSS 20B",
          "Ministral 8B", "Llama 3.1 8B"]

def tf(v): return v in (True, "true", "True", 1)

# ---- MC: parsed answer + correctness + dataset, per item, per model ----
mc_correct = collections.defaultdict(dict)      # model -> eid -> bool
mc_pred = collections.defaultdict(dict)          # model -> eid -> letter
gold, ds = {}, {}
def load_mc(model, path):
    for line in open(path):
        r = json.loads(line)
        if r.get("sample_index", 0) != 0: continue
        e = r["example_id"]
        mc_correct[model][e] = bool(r.get("correct"))
        mc_pred[model][e] = r.get("parsed_answer")
        gold[e] = r.get("expected_letter"); ds[e] = r.get("dataset")
for m, d in MAIN.items():
    load_mc(m, f"{RUNS}/{d}/combined_predictions.jsonl")
for f in glob.glob(f"{RUNS}/openrouter_deepseek_v4_pro_qwen37_full_k1/shards/*/predictions.jsonl"):
    m = "DeepSeek V4 Pro" if "/deepseek_" in f else "Qwen 3.7 Plus"
    load_mc(m, f)

# ---- judge verdicts: reconstruction target_match, generative judge_correct ----
recon = collections.defaultdict(dict)            # model -> eid -> bool (can rebuild)
gen_correct = collections.defaultdict(dict)      # model -> eid -> bool
for m, d in JUDGE.items():
    for line in open(f"{RUNS}/{d}/canonical_judgments.jsonl"):
        r = json.loads(line); e = r["example_id"]
        if r["task"] == "question_reconstruction_fidelity":
            if r.get("target_match") is not None: recon[m][e] = tf(r.get("target_match"))
        elif r["task"] == "free_answer_semantic_correctness":
            if r.get("judge_correct") is not None: gen_correct[m][e] = tf(r.get("judge_correct"))

items = sorted(gold)
FOUR = ["MedQA-USMLE", "MedMCQA"]
print("items:", len(items), "| datasets:", collections.Counter(ds.values()))

# ============================ FIGURE D data (per dataset, MedQA & MedMCQA) ====
figD = {}
print("\n=== Figure D: pools per dataset (validate rec/unrec MC & Gen) ===")
for dset in FOUR:
    figD[dset] = {}
    eids = [e for e in items if ds[e] == dset]
    for m in MODELS:
        rec_e = [e for e in eids if e in recon[m]]
        can = [e for e in rec_e if recon[m][e]]
        cannot = [e for e in rec_e if not recon[m][e]]
        def acc(pool, tbl): return 100*np.mean([tbl[m][e] for e in pool if e in tbl[m]]) if pool else float("nan")
        figD[dset][m] = dict(mc_rec=acc(can, mc_correct), mc_unrec=acc(cannot, mc_correct),
                             gen_rec=acc(can, gen_correct), gen_unrec=acc(cannot, gen_correct),
                             n_can=len(can), n_cannot=len(cannot))
# pooled validation vs tab:rq5 (Luna 88.6/93.4 unrec/rec MC ; 63.0/73.6 gen)
def pooled(m, key):
    tot=cor=0
    for dset in FOUR:
        d=figD[dset][m]
    # recompute pooled directly
    rec_e=[e for e in items if ds[e] in FOUR and e in recon[m]]
    can=[e for e in rec_e if recon[m][e]]; cannot=[e for e in rec_e if not recon[m][e]]
    f={'mc_rec':(can,mc_correct),'mc_unrec':(cannot,mc_correct),'gen_rec':(can,gen_correct),'gen_unrec':(cannot,gen_correct)}
    pool,tbl=f[key]; return 100*np.mean([tbl[m][e] for e in pool if e in tbl[m]])
for m in ["GPT-5.6 Luna","Qwen 3.7 Plus","Llama 3.1 8B"]:
    print(f"  {m:16} MC unrec {pooled(m,'mc_unrec'):.1f} rec {pooled(m,'mc_rec'):.1f} | "
          f"Gen unrec {pooled(m,'gen_unrec'):.1f} rec {pooled(m,'gen_rec'):.1f}")

# ============================ FIGURE C data (ensemble by size, per dataset) ===
codes={}
def c(x): return -1 if x is None else codes.setdefault(str(x).strip().upper(), len(codes))
figC = {}
print("\n=== Figure C: best vote / best at-least-one by size, per dataset ===")
for dset in ["MedQA-USMLE", "MedMCQA", "PubMedQA"]:
    eids=[e for e in items if ds[e]==dset]
    G=np.array([c(gold[e]) for e in eids])
    P=np.array([[c(mc_pred[m].get(e)) for e in eids] for m in MODELS])
    corr=(P==G[None,:])
    acc=[corr[i].mean() for i in range(len(MODELS))]
    NA=len(codes)
    oh=np.zeros((len(MODELS),len(eids),NA),dtype=np.int16)
    for a in range(len(MODELS)):
        for j in range(NA): oh[a,:,j]=(P[a]==j)
    def vote(idx):
        cnt=oh[list(idx)].sum(0); mx=cnt.max(1); tied=(cnt==mx[:,None]).sum(1)>1
        ok=(cnt.argmax(1)==G)&(mx>0); best=max(idx,key=lambda a:acc[a])
        return 100*np.where(tied,(P[best]==G),ok).mean()
    bestvote,bestany=[],[]
    for k in range(1,11):
        bv=max(vote(s) for s in itertools.combinations(range(10),k))
        ba=max(100*corr[list(s)].any(0).mean() for s in itertools.combinations(range(10),k))
        bestvote.append(round(bv,2)); bestany.append(round(ba,2))
    figC[dset]=dict(vote=bestvote, anyone=bestany, best_single=round(100*max(acc),2))
    print(f"  {dset:13} vote {bestvote[0]:.1f}->{bestvote[3]:.1f}(peak?)->{bestvote[-1]:.1f} | "
          f"any {bestany[0]:.1f}->{bestany[-1]:.1f} | best single {100*max(acc):.1f}")

json.dump({"figC":figC,"figD":figD,"models":MODELS}, open(SD+"/by_dataset.json","w"), indent=1)
print("\nsaved by_dataset.json")
