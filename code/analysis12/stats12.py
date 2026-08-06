#!/usr/bin/env python3
"""Recompute every number the paper reports, for twelve models.

Validates itself against the published ten-model figures first: if the
ten-model subset does not reproduce 82.8 -> 52.6 and the other headline
values, the join is wrong and the twelve-model numbers cannot be trusted.
"""
import itertools
import json
import math
import os
from collections import Counter, defaultdict

A = os.path.expanduser("~/Documents/Research/CellPress/analysis12")
D = json.load(open(os.path.join(A, "per_question_12.json")))
MC, GEN, REC, DS = D["mc"], D["gen"], D["rec"], D["ds"]
MODELS, PAIRS = D["models"], [tuple(p) for p in D["pairs"]]
TEN = MODELS[:10]
FOUR = ("MedQA-USMLE", "MedMCQA")
BENCH = ("MedQA-USMLE", "MedMCQA", "PubMedQA")


def wilson(c, n, z=1.96):
    if not n:
        return (0.0, 0.0)
    p = c / n; d = 1 + z * z / n
    ctr = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (100 * (ctr - h), 100 * (ctr + h))


def acc(store, m, benches=BENCH):
    c = n = 0
    for eid, ok in store[m].items():
        if DS.get(eid) in benches:
            n += 1; c += ok
    return c, n, (100 * c / n if n else 0.0)


def paired_t(diffs):
    n = len(diffs); mean = sum(diffs) / n
    if n < 2:
        return mean, 0.0
    sd = math.sqrt(sum((d - mean) ** 2 for d in diffs) / (n - 1))
    return mean, (mean / (sd / math.sqrt(n)) if sd else float("inf"))


R = {}
# ---------------------------------------------------------------- per model
R["per_model"] = {}
for m in MODELS:
    row = {}
    for b in BENCH:
        c, n, p = acc(MC, m, (b,))
        row[b] = {"acc": p, "ci": wilson(c, n), "c": c, "n": n}
    c, n, p = acc(MC, m); row["overall"] = {"acc": p, "ci": wilson(c, n)}
    mc4 = acc(MC, m, FOUR)[2]; g4 = acc(GEN, m, FOUR)[2]
    row["mc4"], row["gen4"], row["delta4"] = mc4, g4, g4 - mc4
    row["gen_pubmed"] = acc(GEN, m, ("PubMedQA",))[2]
    row["mc_pubmed"] = acc(MC, m, ("PubMedQA",))[2]
    rc, rn, rp = acc(REC, m, FOUR)
    row["rec"] = {"acc": rp, "ci": wilson(rc, rn), "c": rc, "n": rn}
    for b in FOUR:
        row["rec_" + b] = acc(REC, m, (b,))[2]
    R["per_model"][m] = row

# --------------------------------------------------- headline drop, 10 vs 12
for tag, pool in (("ten", TEN), ("twelve", MODELS)):
    mcs = [R["per_model"][m]["mc4"] for m in pool]
    gens = [R["per_model"][m]["gen4"] for m in pool]
    mean_d, t = paired_t([g - mm for g, mm in zip(gens, mcs)])
    R[tag + "_drop"] = {"mc_mean": sum(mcs) / len(mcs), "gen_mean": sum(gens) / len(gens),
                        "delta": mean_d, "t": t, "df": len(pool) - 1}
    R[tag + "_pubmed_rise"] = sum(R["per_model"][m]["gen_pubmed"] - R["per_model"][m]["mc_pubmed"]
                                  for m in pool) / len(pool)

# ------------------------------------------------------------- scale pairing
for tag, pairs in (("ten", PAIRS[:5]), ("twelve", PAIRS)):
    d_mc = [acc(MC, lg)[2] - acc(MC, sm)[2] for sm, lg in pairs]
    d_gen = [acc(GEN, lg, FOUR)[2] - acc(GEN, sm, FOUR)[2] for sm, lg in pairs]
    d_rec = [acc(REC, lg, FOUR)[2] - acc(REC, sm, FOUR)[2] for sm, lg in pairs]
    R[tag + "_scale"] = {
        "mc": paired_t(d_mc), "gen": paired_t(d_gen), "rec": paired_t(d_rec),
        "per_pair": {"%s->%s" % p: (a, b, c) for p, a, b, c in zip(pairs, d_mc, d_gen, d_rec)}}

# ------------------------------------------------------------ pools by model
R["pools"] = {}
for m in MODELS:
    p = {True: [0, 0, 0, 0], False: [0, 0, 0, 0]}
    for eid, canrec in REC[m].items():
        b = p[canrec]
        if eid in MC[m]: b[1] += 1; b[0] += MC[m][eid]
        if eid in GEN[m]: b[3] += 1; b[2] += GEN[m][eid]
    r_, u_ = p[True], p[False]
    R["pools"][m] = {
        "unrec_n": u_[1], "unrec_pct": 100 * u_[1] / (u_[1] + r_[1]),
        "mc_unrec": 100 * u_[0] / max(u_[1], 1), "mc_rec": 100 * r_[0] / max(r_[1], 1),
        "gen_unrec": 100 * u_[2] / max(u_[3], 1), "gen_rec": 100 * r_[2] / max(r_[3], 1)}
for tag, pool in (("ten", TEN), ("twelve", MODELS)):
    R[tag + "_pool_gap"] = {
        "mc": paired_t([R["pools"][m]["mc_rec"] - R["pools"][m]["mc_unrec"] for m in pool]),
        "gen": paired_t([R["pools"][m]["gen_rec"] - R["pools"][m]["gen_unrec"] for m in pool])}

# ------------------------------------------------------- reconstruction corr
for tag, pool in (("ten", TEN), ("twelve", MODELS)):
    xs = [acc(MC, m)[2] for m in pool]; ys = [acc(REC, m, FOUR)[2] for m in pool]
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den = math.sqrt(sum((x - mx) ** 2 for x in xs) * sum((y - my) ** 2 for y in ys))
    R[tag + "_corr"] = num / den if den else 0.0

# ------------------------------------------------------------ item difficulty
for tag, pool in (("ten", TEN), ("twelve", MODELS)):
    hist = Counter()
    perbench = defaultdict(lambda: [0, 0, 0])
    for eid in MC[pool[0]]:
        k = sum(MC[m].get(eid, False) for m in pool)
        hist[k] += 1
        b = perbench[DS[eid]]; b[0] += 1
        b[1] += (k == len(pool)); b[2] += (k == 0)
    R[tag + "_difficulty"] = {"hist": [hist.get(i, 0) for i in range(len(pool) + 1)],
                              "per_bench": {k: v for k, v in perbench.items()},
                              "n": len(MC[pool[0]])}

# ------------------------------------------------------------------ ensemble
def vote(committee, order):
    right = 0
    for eid in MC[committee[0]]:
        votes = Counter()
        for m in committee:
            votes[MC[m].get(eid, False)] += 1
        top = max(votes.values())
        win = [v for v, c in votes.items() if c == top]
        right += (True in win) if len(win) == 1 else MC[order[0]].get(eid, False)
    return 100 * right / len(MC[committee[0]])


for tag, pool in (("ten", TEN), ("twelve", MODELS)):
    order = sorted(pool, key=lambda m: -acc(MC, m)[2])
    best_single = acc(MC, order[0])[2]
    curve = []
    for k in range(1, len(pool) + 1):
        best = max(vote(list(c), sorted(c, key=lambda m: -acc(MC, m)[2]))
                   for c in itertools.combinations(pool, k))
        curve.append(best)
    R[tag + "_ensemble"] = {"curve": curve, "best_single": best_single,
                            "full_vote": vote(order, order),
                            "n_subsets": 2 ** len(pool) - 1}

json.dump(R, open(os.path.join(A, "stats12.json"), "w"), indent=1, default=str)

# ------------------------------------------------------------------- report
print("=== VALIDATION: ten-model subset vs published paper ===")
t = R["ten_drop"]
print("  four-choice MC mean  %.1f%%   (paper 82.8)" % t["mc_mean"])
print("  four-choice Gen mean %.1f%%   (paper 52.6)" % t["gen_mean"])
print("  drop                 %.1f pts (paper -30.2)  t(%d)=%.1f" % (t["delta"], t["df"], t["t"]))
print("  scale gap MC         %+.1f pts (paper +8.7)" % R["ten_scale"]["mc"][0])
print("  pool gap MC          %+.1f pts (paper +7.5)" % R["ten_pool_gap"]["mc"][0])
print("  pool gap Gen         %+.1f pts (paper +14.9)" % R["ten_pool_gap"]["gen"][0])
print("  recon correlation    %.2f     (paper 0.86)" % R["ten_corr"])
print("  PubMedQA gen rise    %+.1f pts (paper +5.6)" % R["ten_pubmed_rise"])
d = R["ten_difficulty"]
print("  solved by all / none %d / %d   (paper 1395 / 105)" % (d["hist"][-1], d["hist"][0]))
print("  best single / vote   %.1f / %.1f (paper 88.4 / 88.2)"
      % (R["ten_ensemble"]["best_single"], R["ten_ensemble"]["full_vote"]))

print("\n=== TWELVE-MODEL VALUES ===")
t = R["twelve_drop"]
print("  four-choice MC mean  %.1f%%  -> Gen %.1f%%   drop %.1f pts  t(%d)=%.1f"
      % (t["mc_mean"], t["gen_mean"], t["delta"], t["df"], t["t"]))
print("  scale gap MC %+.1f  Gen %+.1f  Recon %+.1f (6 pairs)"
      % (R["twelve_scale"]["mc"][0], R["twelve_scale"]["gen"][0], R["twelve_scale"]["rec"][0]))
print("  pool gap MC %+.1f   Gen %+.1f" % (R["twelve_pool_gap"]["mc"][0],
                                           R["twelve_pool_gap"]["gen"][0]))
print("  recon correlation %.2f" % R["twelve_corr"])
print("  PubMedQA gen rise %+.1f pts" % R["twelve_pubmed_rise"])
d = R["twelve_difficulty"]
print("  solved by all 12: %d (%.1f%%)   missed by all 12: %d (%.1f%%)"
      % (d["hist"][-1], 100 * d["hist"][-1] / d["n"], d["hist"][0], 100 * d["hist"][0] / d["n"]))
print("  per benchmark (n, all-correct, all-wrong):")
for b, v in d["per_bench"].items():
    print("     %-13s %4d  %4d (%.0f%%)  %3d (%.1f%%)" % (b, v[0], v[1], 100*v[1]/v[0], v[2], 100*v[2]/v[0]))
e = R["twelve_ensemble"]
print("  ensemble best single %.1f  full vote %.1f  peak %.1f at k=%d  (%d subsets)"
      % (e["best_single"], e["full_vote"], max(e["curve"]),
         e["curve"].index(max(e["curve"])) + 1, e["n_subsets"]))
print("\nwrote", os.path.join(A, "stats12.json"))
