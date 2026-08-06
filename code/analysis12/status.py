#!/usr/bin/env python3
"""Progress and spend for the MedGemma runs. Safe to run any time, including
while the runs are in flight -- it only reads the append-only prediction logs.

Spend is computed from the token counts the API actually returned, priced at
dr7.ai's published rates. It is an estimate: the endpoint does not return a
cost field, so this is tokens x list price, not a billed amount.
"""
import glob
import json
import os
import subprocess
import time

RUNS = os.path.expanduser("~/Documents/Research/CellPress/medgemma-runs")
RATE = {"medgemma-4b-it": (0.001, 0.002), "medgemma-27b-it": (0.003, 0.006)}
TARGET = {1: 2773, 2: 2773, 3: 2273}          # PubMedQA has no reconstruction condition
JUDGE_PER_MODEL = 5046                        # 2,773 free-answer + 2,273 reconstruction


def main():
    alive = subprocess.run(["pgrep", "-f", "run_medgemma.py"],
                           capture_output=True, text=True).stdout.split()
    alive += subprocess.run(["pgrep", "-f", "judge_medgemma.py"],
                            capture_output=True, text=True).stdout.split()
    stopped = sorted(glob.glob(os.path.join(RUNS, "*_setting*", "STOPPED.json")))
    print("=" * 78)
    print("MedGemma progress  ", time.strftime("%Y-%m-%d %H:%M:%S"),
          "  |  worker processes running:", len(alive))
    if stopped:
        print()
        for sp in stopped:
            d = json.load(open(sp))
            print("  !! STOPPED: %-28s after %d rows -- %s"
                  % (os.path.basename(os.path.dirname(sp)),
                     d.get("completed_rows", 0), d.get("reason")))
        print("  !! likely out of credit. Top up, then rerun the same command to resume.")
    print("=" * 78)

    grand_cost = grand_done = grand_target = 0
    rows_out = []
    for d in sorted(glob.glob(os.path.join(RUNS, "*_setting*"))):
        f = os.path.join(d, "predictions.jsonl")
        if not os.path.isfile(f):
            continue
        name = os.path.basename(d)
        model = name.rsplit("_setting", 1)[0]
        setting = int(name.rsplit("_setting", 1)[1])
        tin = tout = ok = err = trunc = correct = scored = 0
        for line in open(f):
            try:
                r = json.loads(line)
            except Exception:
                continue
            if r.get("api_error"):
                err += 1
                continue
            ok += 1
            u = r.get("usage") or {}
            tin += u.get("prompt_tokens", 0)
            tout += u.get("completion_tokens", 0)
            trunc += bool(r.get("truncated"))
            if setting == 1 and r.get("parsed"):
                scored += 1
                correct += r["parsed"] == r["gold"]
        ri, ro = RATE.get(model, (0, 0))
        cost = tin / 1000 * ri + tout / 1000 * ro
        tgt = TARGET.get(setting, 0)
        grand_cost += cost
        grand_done += ok
        grand_target += tgt
        acc = ("%.1f%%" % (100 * correct / scored)) if scored else "--"
        rows_out.append((model, setting, ok, tgt, 100 * ok / tgt if tgt else 0,
                         acc, trunc, err, tin, tout, cost))

    hdr = "%-17s %-4s %-14s %-7s %-7s %-6s %-6s %-9s" % (
        "model", "set", "rows done", "pct", "acc", "trunc", "errs", "est cost")
    print(hdr)
    print("-" * 78)
    for m, s, ok, tgt, pct, acc, tr, er, ti, to, c in rows_out:
        print("%-17s %-4d %6d/%-7d %6.1f%% %-7s %-6d %-6d $%-8.2f"
              % (m, s, ok, tgt, pct, acc, tr, er, c))
    print("-" * 78)
    pct = 100 * grand_done / grand_target if grand_target else 0
    print("%-17s      %6d/%-7d %6.1f%%%31s$%.2f"
          % ("TOTAL (models)", grand_done, grand_target, pct, "", grand_cost))

    # ---------------------------------------------------------------- judge
    jdirs = sorted(glob.glob(os.path.join(RUNS, "*_judge")))
    if jdirs:
        print()
        print("JUDGE PASS  (Claude Sonnet 5, temperature 0, Template B, via OpenRouter)")
        print("-" * 78)
        jt = jc = 0
        for d in jdirs:
            f = os.path.join(d, "canonical_judgments.jsonl")
            if not os.path.isfile(f):
                continue
            model = os.path.basename(d).rsplit("_judge", 1)[0]
            ok = err = cost = 0
            by = {"free_answer_semantic_correctness": [0, 0],
                  "question_reconstruction_fidelity": [0, 0]}
            for line in open(f):
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                if r.get("api_error"):
                    err += 1
                    continue
                ok += 1
                cost += float((r.get("usage") or {}).get("cost") or 0)
                b = by.get(r.get("task"))
                if b:
                    b[1] += 1
                    b[0] += bool(r.get("semantic_correct") or r.get("target_match"))
            jt += ok
            jc += cost
            fa, rc = by["free_answer_semantic_correctness"], by["question_reconstruction_fidelity"]
            print("  %-17s %5d/%-5d  generative %-14s reconstruction %-14s errs=%-4d $%.2f"
                  % (model, ok, JUDGE_PER_MODEL,
                     ("%.1f%% (%d/%d)" % (100 * fa[0] / fa[1], fa[0], fa[1])) if fa[1] else "--",
                     ("%.1f%% (%d/%d)" % (100 * rc[0] / rc[1], rc[0], rc[1])) if rc[1] else "--",
                     err, cost))
        print("-" * 78)
        print("  %-17s %5d/%-5d%45s$%.2f"
              % ("TOTAL (judge)", jt, 2 * JUDGE_PER_MODEL, "", jc))
        print()
        print("  Spend: $%.2f dr7.ai (models) + $%.2f OpenRouter (judge) = $%.2f"
              % (grand_cost, jc, grand_cost + jc))
    else:
        print()
        print("Judge pass not yet run   : %d verdicts (2 models x %d), ~$16 on OpenRouter"
              % (2 * JUDGE_PER_MODEL, JUDGE_PER_MODEL))
    remaining = grand_target - grand_done
    if remaining:
        print()
        print("Remaining model requests : %d" % remaining)


if __name__ == "__main__":
    main()
