#!/usr/bin/env python3
"""Unify the ten deposited models with the two new MedGemma runs.

Everything the paper reports is recomputed from per-question outcomes rather
than from the previously published summary numbers, so the twelve-model
figures cannot silently disagree with the ten-model ones.

The join key is example_id (medqa_usmle_0, medmcqa_<uuid>, pubmedqa_<pmid>),
which both the deposited runs and the MedGemma runs carry.
"""
import collections
import json
import os

DEP = os.path.expanduser("~/Documents/Research/CellPress/zenodo-deposit/outputs")
MG = os.path.expanduser("~/Documents/Research/CellPress/medgemma-runs")
OUT = os.path.expanduser("~/Documents/Research/CellPress/analysis12")
os.makedirs(OUT, exist_ok=True)

TEN = {
    "Qwen 3.7 Plus": "Qwen_3.7_Plus", "GPT-5.6 Luna": "GPT-5.6_Luna",
    "DeepSeek V4 Pro": "DeepSeek_V4_Pro", "DeepSeek V4 Flash": "DeepSeek_V4_Flash",
    "Llama 4 Maverick": "Llama_4_Maverick", "Mistral Large 3": "Mistral_Large_3",
    "Qwen 3.5 9B": "Qwen_3.5_9B", "GPT-OSS 20B": "GPT-OSS_20B",
    "Ministral 8B": "Ministral_8B", "Llama 3.1 8B": "Llama_3.1_8B",
}
MG_MODELS = {"MedGemma 27B": "medgemma-27b-it", "MedGemma 4B": "medgemma-4b-it"}

# provider pairs: (smaller, larger)
PAIRS = [("GPT-OSS 20B", "GPT-5.6 Luna"), ("DeepSeek V4 Flash", "DeepSeek V4 Pro"),
         ("Qwen 3.5 9B", "Qwen 3.7 Plus"), ("Llama 3.1 8B", "Llama 4 Maverick"),
         ("Ministral 8B", "Mistral Large 3"), ("MedGemma 4B", "MedGemma 27B")]

mc = collections.defaultdict(dict)     # model -> example_id -> bool
gen = collections.defaultdict(dict)
rec = collections.defaultdict(dict)
ds = {}                                # example_id -> dataset


def add(store, model, eid, val, dataset=None):
    store[model][eid] = val
    if dataset:
        ds[eid] = dataset


# ---------------------------------------------------------------- ten models
for label, d in TEN.items():
    base = os.path.join(DEP, d, "setting1_multiple_choice")
    files = [os.path.join(base, "combined_predictions.jsonl")]
    if not os.path.isfile(files[0]):
        sh = os.path.join(base, "shards")
        files = [os.path.join(sh, x, "predictions.jsonl") for x in sorted(os.listdir(sh))
                 if os.path.isfile(os.path.join(sh, x, "predictions.jsonl"))]
    # A resumed shard can hold several attempts at the same question. Reproduce
    # the paper's latest_prediction_rows precedence exactly: prefer a usable
    # attempt (no error, not truncated) over an unusable one, and among equally
    # usable attempts take the chronologically latest. Taking the last row in
    # file order instead silently scores retried questions as failures.
    best = {}
    for f in files:
        for idx, line in enumerate(open(f)):
            r = json.loads(line)
            eid = r.get("example_id")
            if eid is None:
                continue
            usable = int(not r.get("api_error") and not r.get("truncated", False))
            cand = (usable, str(r.get("created_at", "")), idx)
            if eid not in best or cand >= best[eid][0]:
                best[eid] = (cand, r)
    for eid, (_c, r) in best.items():
        if r.get("api_error"):
            continue
        add(mc, label, eid, bool(r.get("correct")), r.get("dataset"))
    j = os.path.join(DEP, d, "judge_verdicts", "canonical_judgments.jsonl")
    for line in open(j):
        r = json.loads(line)
        if r.get("api_error"):
            continue
        eid = r.get("example_id")
        if r["task"] == "free_answer_semantic_correctness":
            add(gen, label, eid, bool(r.get("semantic_correct") or r.get("judge_correct")),
                r.get("dataset"))
        else:
            add(rec, label, eid, bool(r.get("target_match")), r.get("dataset"))

# ------------------------------------------------------------- two new models
for label, m in MG_MODELS.items():
    for line in open(os.path.join(MG, "%s_setting1" % m, "predictions.jsonl")):
        r = json.loads(line)
        if r.get("api_error"):
            continue
        add(mc, label, r["item_id"], r["parsed"] == r["gold"], r["dataset"])
    for line in open(os.path.join(MG, "%s_judge" % m, "canonical_judgments.jsonl")):
        r = json.loads(line)
        if r.get("api_error"):
            continue
        eid = r["example_id"]
        if r["task"] == "free_answer_semantic_correctness":
            add(gen, label, eid, bool(r.get("semantic_correct")), r["dataset"])
        else:
            add(rec, label, eid, bool(r.get("target_match")), r["dataset"])

MODELS = list(TEN) + list(MG_MODELS)
print("=== coverage check (expect mc 2773, gen 2773, rec 2273) ===")
bad = 0
for m in MODELS:
    a, b, c = len(mc[m]), len(gen[m]), len(rec[m])
    flag = "" if (a, b, c) == (2773, 2773, 2273) else "  <-- MISMATCH"
    bad += bool(flag)
    print("  %-18s mc=%-5d gen=%-5d rec=%-5d%s" % (m, a, b, c, flag))
print("mismatches:", bad)

json.dump({"mc": {m: mc[m] for m in MODELS}, "gen": {m: gen[m] for m in MODELS},
           "rec": {m: rec[m] for m in MODELS}, "ds": ds,
           "models": MODELS, "pairs": PAIRS},
          open(os.path.join(OUT, "per_question_12.json"), "w"))
print("\nwrote", os.path.join(OUT, "per_question_12.json"))
