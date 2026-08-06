#!/usr/bin/env python3
"""Run MedGemma 4B / 27B through the dr7.ai API on the paper's evaluation sets.

Design constraints, so the two new models are comparable to the existing ten:

  * the message sequence is rebuilt from the paper's released prompt templates,
    including the same system prompt and the same five exemplars;
  * the same question sets are used (MedQA-USMLE test, the seed-0 MedMCQA
    1,000-question validation sample, PubMedQA expert test);
  * answers are extracted by the paper's own parse_answer, extended only to
    accept \\boxed{X}. That extension is verified against the ten existing
    models before use: exactly 2 of ~25,000 stored rows contain "boxed", so it
    cannot move any published number.

Operational notes specific to this endpoint:
  * Cloudflare 1010-bans the default Python-urllib client signature, so an
    explicit User-Agent is required;
  * finish_reason is reported as "stop" even when the completion is cut off at
    max_tokens, so truncation is detected by comparing completion_tokens
    against the cap rather than by trusting the field.

Resumable:each request is appended to predictions.jsonl and skipped on rerun.
"""
import argparse
import json
import os
import random
import re
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

KEY = os.environ["DR7_API_KEY"]
URL = "https://dr7.ai/api/v1/medical/chat/completions"
UA = "cellpress-medical-eval/1.0"
ROOT = os.path.expanduser("~/Documents/Research/GRAIL-internship/cell-paper-replication/"
                          "paper-replicator/codex_version/papers/cellpress-medical-reasoning")
DATA = os.path.join(ROOT, "data")
OUT = os.path.expanduser("~/Documents/Research/CellPress/medgemma-runs")

SYSTEM = ("As a healthcare professional, provide an expert response to each question. "
          "Exactly one answer option is the most correct.")
TAIL = ("\nThink through the question step by step. "
        "End your response with exactly `Answer: (X)`, replacing X with the option letter.")
SHOTS = [
 ("Question: A 22-year-old male marathon runner presents with right-sided rib pain when he runs long distances. Examination shows normal heart and lung findings and an exhalation dysfunction at ribs 4-5 on the right. Which muscle or muscle group will be most useful in correcting this dysfunction with a direct method?\nChoices: (A) Anterior scalene (B) Latissimus dorsi (C) Pectoralis minor (D) Quadratus lumborum.",
  "Explanation: Among the options, the pectoralis minor originates from the outer surfaces of the third through fifth ribs, so it can be used to help correct this rib dysfunction.\nAnswer: Therefore, the answer is (C) Pectoralis minor."),
 ("Question: A 36-year-old man has 3 weeks of low back pain. Prone examination reveals a deep sacral sulcus on the left, a posterior inferior lateral angle on the right, and a lumbosacral junction that springs freely on compression. Which diagnosis is most likely?\nChoices: (A) Left-on-left sacral torsion (B) Left-on-right sacral torsion (C) Right unilateral sacral flexion (D) Right-on-right sacral torsion.",
  "Explanation: A deep sulcus on the left, posterior ILA on the right, and a negative spring test indicate a right-on-right sacral torsion.\nAnswer: Therefore, the answer is (D) Right-on-right sacral torsion."),
 ("Question: A 44-year-old man has a 3-day history of sore throat, nonproductive cough, runny nose, and frontal headache. He is afebrile, has erythematous nares and posterior pharyngeal lymphoid hyperplasia, no cervical adenopathy, and clear lungs. Which cause is most likely?\nChoices: (A) Allergic rhinitis (B) Epstein-Barr virus (C) Mycoplasma pneumonia (D) Rhinovirus.",
  "Explanation: The acute upper-respiratory symptoms, absence of cervical adenopathy, and clear lungs are most consistent with a rhinovirus infection.\nAnswer: Therefore, the answer is (D) Rhinovirus."),
 ("Question: A previously healthy 32-year-old woman has sadness, poor appetite, and insomnia 8 months after her husband died. She has also developed new repetitive checking and counting rituals. Pharmacotherapy should target which neurotransmitter?\nChoices: (A) Dopamine (B) Glutamate (C) Norepinephrine (D) Serotonin.",
  "Explanation: Her depressive symptoms and new obsessive-compulsive symptoms are both treated with serotonergic medications, making serotonin the best target.\nAnswer: Therefore, the answer is (D) Serotonin."),
 ("Question: A 42-year-old man is preparing for adrenalectomy after a 10-cm adrenal mass, hypertension, and elevated metanephrines were found. His current blood pressure is 170/95 mm Hg. Which treatment should be started first?\nChoices: (A) Labetalol (B) A loading dose of potassium chloride (C) Nifedipine (D) Phenoxybenzamine.",
  "Explanation: The adrenal mass, metanephrine elevation, and hypertension indicate pheochromocytoma. Preoperative preparation begins with alpha blockade using phenoxybenzamine before any beta blockade.\nAnswer: Therefore, the answer is (D) Phenoxybenzamine."),
]

# ------------------------------------------------------------------ parsing
# The paper's own two patterns, verbatim, plus a \boxed{} form for MedGemma 27B.
# Built with an f-string rather than .format(), because the patterns contain
# literal braces ({0,120} and \{) that .format() would try to interpret.
def _patterns(labels):
    return [
        rf"\\boxed\{{\(?([{labels}])\)?\}}",
        rf"(?i)(?:final\s+)?answer\s*(?::|is)?\s*\(?([{labels}])\)?",
        rf"(?i)therefore[^\n]{{0,120}}?answer[^\n]{{0,40}}?\(?([{labels}])\)?",
    ]


def parse_answer(output, labels):
    for pat in _patterns(labels):
        m = re.findall(pat, output or "")
        if m:
            return m[-1].upper()
    return None


# ------------------------------------------------------------------ datasets
# The canonical question set is taken from the frozen selected_questions.jsonl
# that all ten existing models were scored on (identical across every model,
# sha256 of ids c3ea2bac5d88d934). Re-deriving the MedMCQA seed-0 sample from
# the raw file does NOT reproduce it, so the frozen file is the only correct
# source: MedGemma must see byte-identical questions, choices, and ordering.
SELECTED = os.path.expanduser(
    "~/Documents/Research/CellPress/zenodo-deposit/outputs/Llama_4_Maverick/"
    "setting2_generative_setting3_reconstruction/selected_questions.jsonl")


def load(bench):
    out = []
    for line in open(SELECTED):
        r = json.loads(line)
        if r["dataset"] != bench:
            continue
        out.append({"id": r["id"], "q": r["question"], "choices": r["choices"],
                    "gold": "ABCD"[r["answer_index"]]})
    return out


def build(item, setting):
    labels = "ABCD"[:len(item["choices"])]
    ch = " ".join("(%s) %s" % (l, c) for l, c in zip(labels, item["choices"]))
    if setting == 1:
        msgs = [{"role": "system", "content": SYSTEM}]
        for u, a in SHOTS:
            msgs += [{"role": "user", "content": u}, {"role": "assistant", "content": a}]
        msgs.append({"role": "user",
                     "content": "Question: %s\nChoices: %s.%s" % (item["q"], ch, TAIL)})
        return msgs
    if setting == 2:                                        # zero-shot, choices withheld
        return [{"role": "system", "content": SYSTEM},
                {"role": "user", "content":
                 "Question: %s\nAnswer the question directly and concisely. "
                 "Do not list options." % item["q"]}]
    return [{"role": "system", "content": SYSTEM},          # setting 3: reconstruct
            {"role": "user", "content":
             "The following are the answer choices to a single question from a medical "
             "examination:\n%s\n\nWrite the question these choices belong to." % ch}]


# ------------------------------------------------------------------ transport
LOCK = threading.Lock()
STATS = {"ok": 0, "err": 0, "trunc": 0, "cost_tok_in": 0, "cost_tok_out": 0}
# Circuit breaker. Running out of credit looks like an unbroken wall of the
# same error, and retrying through it would spend hours achieving nothing.
# After CB_LIMIT consecutive failures we stop the stage, leave the completed
# rows intact, and write STOPPED.json so the caller can tell "out of credit"
# apart from "finished". Relaunching after a top-up resumes exactly here.
CB = {"streak": 0, "last": None, "tripped": False}
CB_LIMIT = 25


def ask(model, msgs, max_tokens, retries=8):
    body = json.dumps({"model": model, "messages": msgs,
                       "max_tokens": max_tokens, "temperature": 0}).encode()
    for a in range(retries):
        try:
            req = urllib.request.Request(URL, data=body, headers={
                "Authorization": "Bearer " + KEY, "Content-Type": "application/json",
                "User-Agent": UA})
            t0 = time.time()
            with urllib.request.urlopen(req, timeout=180) as r:
                d = json.loads(r.read())
            d["_latency"] = round(time.time() - t0, 3)
            return d
        except urllib.error.HTTPError as e:
            if e.code in (401, 402, 403):        # auth / payment / forbidden
                return {"_error": "HTTP %d (hard stop)" % e.code, "_hard": True}
            if e.code in (429, 500, 502, 503, 504) and a < retries - 1:
                # 429 is common here; back off long and jitter so parallel
                # workers do not retry in lockstep.
                time.sleep(min(90, 4 * 2 ** a) * (0.6 + random.random() * 0.8))
                continue
            return {"_error": "HTTP %d" % e.code}
        except Exception as e:                                       # noqa: BLE001
            if a < retries - 1:
                time.sleep(min(90, 4 * 2 ** a) * (0.6 + random.random() * 0.8))
                continue
            return {"_error": str(e)[:200]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--setting", type=int, required=True, choices=(1, 2, 3))
    ap.add_argument("--benchmarks", default="MedQA-USMLE,MedMCQA,PubMedQA")
    ap.add_argument("--max-tokens", type=int, default=6144)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    tag = "%s_setting%d" % (args.model.replace("/", "_"), args.setting)
    outdir = os.path.join(OUT, tag)
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, "predictions.jsonl")
    # Only a row with a real response counts as done. A row recorded with an
    # api_error (e.g. a 429 that outlived its retries) must be re-attempted on
    # resume, otherwise rate-limited items would be silently dropped.
    done = set()
    if os.path.exists(path):
        keep = []
        for l in open(path):
            try:
                r = json.loads(l)
            except Exception:
                continue
            if r.get("api_error") or not (r.get("response") or "").strip():
                continue
            done.add(r["item_id"]); keep.append(l)
        with open(path, "w") as fh:          # drop failed rows so the file stays clean
            fh.writelines(keep)

    items = []
    for b in args.benchmarks.split(","):
        if args.setting == 3 and b == "PubMedQA":
            continue                       # identical choices carry nothing to reconstruct
        for it in load(b):
            it["bench"] = b
            items.append(it)
    if args.limit:
        items = items[:args.limit]
    todo = [i for i in items if i["id"] not in done]
    print("%s | %d items, %d already done, %d to run" % (tag, len(items), len(done), len(todo)),
          flush=True)

    fh = open(path, "a")
    t_start = time.time()

    def work(item):
        if CB["tripped"]:
            return
        labels = "ABCD"[:len(item["choices"])]
        d = ask(args.model, build(item, args.setting), args.max_tokens)
        txt = ""
        if "choices" in d:
            txt = (d["choices"][0].get("message") or {}).get("content") or ""
        u = d.get("usage") or {}
        trunc = u.get("completion_tokens", 0) >= args.max_tokens
        row = {
            "item_id": item["id"], "dataset": item["bench"], "setting": args.setting,
            "model": args.model, "provider": "dr7.ai", "temperature": 0,
            "gold": item["gold"], "response": txt,
            "parsed": parse_answer(txt, labels) if args.setting == 1 else None,
            "usage": u, "latency": d.get("_latency"), "response_id": d.get("id"),
            "echoed_model": d.get("model"),
            "finish_reason": (d.get("choices", [{}])[0] or {}).get("finish_reason"),
            "truncated": trunc, "api_error": d.get("_error"),
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        with LOCK:
            fh.write(json.dumps(row) + "\n"); fh.flush()
            if d.get("_error"):
                CB["streak"] += 1; CB["last"] = d["_error"]
                if d.get("_hard") or CB["streak"] >= CB_LIMIT:
                    CB["tripped"] = True
            else:
                CB["streak"] = 0
            STATS["ok" if not d.get("_error") else "err"] += 1
            STATS["trunc"] += int(trunc)
            STATS["cost_tok_in"] += u.get("prompt_tokens", 0)
            STATS["cost_tok_out"] += u.get("completion_tokens", 0)
            n = STATS["ok"] + STATS["err"]
            if n % 50 == 0:
                el = time.time() - t_start
                print("  %5d/%d  ok=%d err=%d trunc=%d  %.1f req/s  eta=%.0f min"
                      % (n, len(todo), STATS["ok"], STATS["err"], STATS["trunc"],
                         n / el, (len(todo) - n) / max(n / el, 1e-9) / 60), flush=True)

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        list(ex.map(work, todo))
    fh.close()

    rate = {"medgemma-4b-it": (0.001, 0.002), "medgemma-27b-it": (0.003, 0.006)}
    ri, ro = rate.get(args.model, (0, 0))
    cost = STATS["cost_tok_in"] / 1000 * ri + STATS["cost_tok_out"] / 1000 * ro
    json.dump({"model": args.model, "setting": args.setting, "stats": STATS,
               "estimated_cost_usd": round(cost, 2),
               "elapsed_s": round(time.time() - t_start, 1),
               "max_tokens": args.max_tokens, "endpoint": URL},
              open(os.path.join(outdir, "run_manifest.json"), "w"), indent=1)
    if CB["tripped"]:
        json.dump({"stopped": True, "reason": CB["last"], "completed_rows": STATS["ok"],
                   "stopped_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                   "resume": "rerun the same command after topping up; completed rows "
                             "are kept and failed rows are retried"},
                  open(os.path.join(outdir, "STOPPED.json"), "w"), indent=1)
        print("STOPPED: %s after %d consecutive failures (last: %s)"
              % (tag, CB_LIMIT, CB["last"]), flush=True)
    elif os.path.exists(os.path.join(outdir, "STOPPED.json")):
        os.remove(os.path.join(outdir, "STOPPED.json"))
    print("done: %s  ok=%d err=%d trunc=%d  est_cost=$%.2f"
          % (tag, STATS["ok"], STATS["err"], STATS["trunc"], cost), flush=True)


if __name__ == "__main__":
    main()
