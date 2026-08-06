#!/usr/bin/env python3
"""Score the MedGemma Settings 2 and 3 outputs with the paper's frozen judge.

The grading instrument must be identical to the one used on the other ten
models or the new rows are not comparable, so:

  * grader is Claude Sonnet 5 at temperature 0, via OpenRouter, one request
    per judgment, exactly as before;
  * the two message templates are reproduced byte-for-byte from the frozen
    judge_items.jsonl of an existing run -- this is TEMPLATE B. Template A was
    ~21 points more lenient and must never be used (see the project note on
    template standardisation);
  * verdict fields keep the same names, so the existing analysis code reads
    these files without modification.

Each judgment is appended to canonical_judgments.jsonl and skipped on rerun,
so the pass is resumable. Rows that error are re-attempted on resume.
"""
import argparse
import json
import os
import re
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

KEY = os.environ["OPENROUTER_API_KEY"]
URL = "https://openrouter.ai/api/v1/chat/completions"
JUDGE_MODEL = "anthropic/claude-sonnet-5"
RUNS = os.path.expanduser("~/Documents/Research/CellPress/medgemma-runs")
SELECTED = os.path.expanduser(
    "~/Documents/Research/CellPress/zenodo-deposit/outputs/Llama_4_Maverick/"
    "setting2_generative_setting3_reconstruction/selected_questions.jsonl")

SYS_FREE = ("You are an impartial evaluator of free-form medical answers. Return only "
            "the requested JSON object and no chain-of-thought.")
SYS_RECON = ("You are an impartial evaluator of reconstructed medical questions. Return "
             "only the requested JSON object and no chain-of-thought.")
USR_FREE = (
    "Determine whether the model response gives the same substantive answer as the "
    "reference answer. Allow clinically equivalent wording, but mark false when the "
    "response is incomplete, unsupported, or gives a materially different diagnosis, "
    "fact, relationship, or treatment.\n\nQUESTION:\n{q}\n\nREFERENCE CORRECT ANSWER:\n"
    "{ref}\n\nMODEL RESPONSE:\n{resp}\n\nReturn JSON only: "
    '{{"semantic_correct": <true or false>, "rationale": "<maximum 15 words>"}}')
USR_RECON = (
    "Evaluate whether the generated question recovers the same medical knowledge target "
    "as the reference question. Allow paraphrases. Reject a question that is merely "
    "about the same broad topic but tests a different fact, relationship, diagnosis, or "
    "treatment.\n\nREFERENCE QUESTION:\n{q}\n\nGENERATED QUESTION:\n{resp}\n\n"
    'Return JSON only: {{"target_match": <true or false>, "fidelity": <integer 1-5>, '
    '"rationale": "<maximum 20 words>"}}')

REF = {}
for _l in open(SELECTED):
    _r = json.loads(_l)
    REF[_r["id"]] = _r

LOCK = threading.Lock()
STATS = {"ok": 0, "err": 0, "cost": 0.0, "true": 0}


def ask(messages, retries=6):
    body = json.dumps({"model": JUDGE_MODEL, "messages": messages,
                       "temperature": 0, "max_tokens": 300,
                       "usage": {"include": True}}).encode()
    for a in range(retries):
        try:
            req = urllib.request.Request(URL, data=body, headers={
                "Authorization": "Bearer " + KEY, "Content-Type": "application/json",
                "HTTP-Referer": "https://github.com/navapatn/medical-qa-llm-evaluation",
                "X-Title": "cellpress-medical-eval"})
            with urllib.request.urlopen(req, timeout=180) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            if e.code in (401, 402, 403):
                return {"_error": "HTTP %d (hard stop)" % e.code, "_hard": True}
            if a < retries - 1:
                time.sleep(min(60, 3 * 2 ** a)); continue
            return {"_error": "HTTP %d" % e.code}
        except Exception as e:                                   # noqa: BLE001
            if a < retries - 1:
                time.sleep(min(60, 3 * 2 ** a)); continue
            return {"_error": str(e)[:200]}


def parse_verdict(text, task):
    """The rubric asks for JSON only; tolerate a stray code fence."""
    if not text:
        return {}
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return {}
    try:
        d = json.loads(m.group(0))
    except Exception:
        return {}
    return d if isinstance(d, dict) else {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)          # medgemma-4b-it / -27b-it
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    items = []
    for setting, task in ((2, "free_answer_semantic_correctness"),
                          (3, "question_reconstruction_fidelity")):
        f = os.path.join(RUNS, "%s_setting%d" % (args.model, setting), "predictions.jsonl")
        for line in open(f):
            r = json.loads(line)
            if r.get("api_error"):
                continue
            ref = REF[r["item_id"]]
            resp = (r.get("response") or "").strip()
            if setting == 2:
                msgs = [{"role": "system", "content": SYS_FREE},
                        {"role": "user", "content": USR_FREE.format(
                            q=ref["question"], ref=ref["answer_text"], resp=resp)}]
            else:
                msgs = [{"role": "system", "content": SYS_RECON},
                        {"role": "user", "content": USR_RECON.format(
                            q=ref["question"], resp=resp)}]
            items.append({"task": task, "dataset": r["dataset"],
                          "example_id": r["item_id"], "source_model": args.model,
                          "source_response_id": r.get("response_id"), "messages": msgs,
                          "gold_option": "ABCD"[ref["answer_index"]]})
    if args.limit:
        items = items[:args.limit]

    outdir = os.path.join(RUNS, "%s_judge" % args.model)
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, "canonical_judgments.jsonl")
    done = set()
    if os.path.exists(path):
        keep = []
        for l in open(path):
            try:
                r = json.loads(l)
            except Exception:
                continue
            if r.get("api_error"):
                continue
            done.add((r["task"], r["example_id"])); keep.append(l)
        with open(path, "w") as fh:
            fh.writelines(keep)
    todo = [i for i in items if (i["task"], i["example_id"]) not in done]
    print("%s judge | %d items, %d done, %d to run"
          % (args.model, len(items), len(done), len(todo)), flush=True)

    fh = open(path, "a")
    t0 = time.time()

    def work(it):
        d = ask(it["messages"])
        txt = ""
        if "choices" in d:
            txt = (d["choices"][0].get("message") or {}).get("content") or ""
        v = parse_verdict(txt, it["task"])
        u = d.get("usage") or {}
        row = {
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "provider": "openrouter", "model": JUDGE_MODEL,
            "model_label": "Claude Sonnet 5 short-reasoning boolean judge (OpenRouter)",
            "task": it["task"], "dataset": it["dataset"], "example_id": it["example_id"],
            "source_model": it["source_model"],
            "source_response_id": it["source_response_id"],
            "gold_option": it["gold_option"], "output": txt,
            "target_match": v.get("target_match"),
            "semantic_correct": v.get("semantic_correct"),
            "fidelity": v.get("fidelity"), "rationale": v.get("rationale"),
            "parse_error": None if v else "no JSON verdict",
            "usage": u, "api_error": d.get("_error"),
        }
        with LOCK:
            fh.write(json.dumps(row) + "\n"); fh.flush()
            if d.get("_error"):
                STATS["err"] += 1
            else:
                STATS["ok"] += 1
                STATS["cost"] += float(u.get("cost") or 0)
                STATS["true"] += bool(v.get("target_match") or v.get("semantic_correct"))
            n = STATS["ok"] + STATS["err"]
            if n % 200 == 0:
                el = time.time() - t0
                print("  %5d/%d ok=%d err=$%d cost=$%.2f %.1f/s eta=%.0f min"
                      % (n, len(todo), STATS["ok"], STATS["err"], STATS["cost"],
                         n / el, (len(todo) - n) / max(n / el, 1e-9) / 60), flush=True)

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        list(ex.map(work, todo))
    fh.close()
    json.dump({"model": args.model, "judge": JUDGE_MODEL, "template": "B",
               "stats": STATS, "elapsed_s": round(time.time() - t0, 1)},
              open(os.path.join(outdir, "judge_run_manifest.json"), "w"), indent=1)
    print("done: %s judge  ok=%d err=%d cost=$%.2f"
          % (args.model, STATS["ok"], STATS["err"], STATS["cost"]), flush=True)


if __name__ == "__main__":
    main()
