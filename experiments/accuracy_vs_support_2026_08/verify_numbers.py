"""Recompute every number quoted in MEETING_2026_08_18.md from the raw data.

Run it and it prints one line per check. Any BAD line is a document defect.
This exists because the write-up asserts about 150 numbers and a reader should not
have to trust any of them.

Every chk() call recomputes its left side from the result files. The right side is the
digit the document prints, transcribed by hand. A few checks divide two printed numbers
instead, which is the only way to catch arithmetic done in prose.

  uv run --with numpy,scipy python verify_numbers.py
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from statistics import mean, median, stdev
from math import comb

import numpy as np
from scipy.stats import spearmanr

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "regret_solver_2026_07"))

# A is the fixed 1800-episode sweep, B the same level counts at 360 episodes each
# C is five levels plus two sealed dead ends
# V is A rerun under CURVES=1, since the committed A carries no per-round curve
# G and E are the 2026-08-06 five-level experiment that Parts 3 and 4 quote
A = json.load(open(HERE / "accuracy_vs_support.json"))
B = json.load(open(HERE / "accuracy_perlevel_budget.json"))
C = json.load(open(HERE / "clamp_safety.json"))
V = json.load(open(HERE / "accuracy_vs_support_curves.json"))
G = json.load(open(HERE.parent / "student_curriculum_2026_07" / "student_curriculum.json"))
E = json.load(open(HERE.parent / "student_curriculum_2026_07" / "extra_experiments.json"))

FAIL: list[str] = []
N = 0


def chk(label, got, want, tol=0.051):
    """One recomputed number against the one the document prints.

    Default tol is half a unit of the last decimal the document shows, so a value it
    rounded to one place passes and a changed digit fails. Callers tighten it for numbers
    quoted to more places and pass tol=0 for exact identities. Nothing raises here, so one
    run reports every defect at once.
    """
    global N
    N += 1
    ok = abs(got - want) <= tol
    if not ok:
        FAIL.append(f"{label}: doc {want}, computed {got}")
    print(f"{'OK ' if ok else 'BAD'} {label:<50} doc {want:<8} computed {got:.4g}")


def est_runs(src):
    """Group by method, minus the methods that estimate no ceiling.

    uniform samples levels evenly and reports nothing, so run_one writes its v_hat as null.
    What is left is the five estimate sources Parts 3 and 4 tabulate, which is where the 25
    cells below come from.
    """
    d = {}
    for r in src:
        if r.get("v_hat"):
            d.setdefault(r["method"], []).append(r)
    return d


def paired(src, L, a="oracle", b="shuffle"):
    """Per-run score difference in points, matched on (maze, seed).

    A matched pair differs only in the ranking of the reported ceilings. Maze difficulty
    spans 13 to 46 points at L=30 and would swamp an unmatched comparison.
    """
    O = {(r["maze"], r["seed"]): r for r in src if r["n_levels"] == L and r["method"] == a}
    S = {(r["maze"], r["seed"]): r for r in src if r["n_levels"] == L and r["method"] == b}
    return {k: 100 * (O[k]["mean"] - S[k]["mean"]) for k in O if k in S}


# ---------------------------------------------------------------- Part 1, glossary
chk("13x13 squares", 13 * 13, 169, 0)
chk("100 rounds x 18 episodes", 100 * 18, 1800, 0)
# a d-step goal pays gamma^(d-1) on arrival, so the d=14 level tops out at 0.97^13
chk("0.97^13 = d=14 ceiling", 0.97 ** 13, 0.673, 0.0005)
chk("4 mazes x 3 seeds", 4 * 3, 12, 0)
# build_maze reads TARGETS off the module at call time and rejects a layout that misses
# one of those distances, so the 13x13 values go in before the call below
import mcts_vs_ppo_regret as base  # noqa: E402
base.TARGETS = [2, 5, 8, 11, 14]
# coords is the open-square list, so this is the free-cell count at wall_p 0.22
# 129 to 146 across these six seeds, against a glossary that quotes one round number
chk("about 136 open squares",
    mean(len(base.build_maze(s, H=13, W=13)["coords"]) for s in range(6)), 136, 3)

# ---------------------------------------------------------------- Parts 3 and 4
by = est_runs(G["runs"])
# the 2026-08-06 diagonal: mean final score per method and level against the share of runs
# whose reported ceiling was nonzero there
# a single binary per cell reproduces the whole score table to within half a point
gaps = [abs(100 * mean(r["final"][i] for r in by[m])
            - 100 * sum(1 for r in by[m] if r["v_hat"][i] != 0) / len(by[m]))
        for m in by for i in range(5)]
chk("25 method-level cells", len(gaps), 25, 0)
chk("mean distance from diagonal", mean(gaps), 0.06, 0.005)
chk("worst distance from diagonal", max(gaps), 0.5, 0.05)
chk("mcts_root d=11 estimate", mean(r["v_hat"][3] for r in by["mcts_root"]), 0.0128, 0.0005)
chk("mcts_root d=11 truth", mean(r["v_opt"][3] for r in by["mcts_root"]), 0.7374, 0.0005)
# both ratios divide numbers the document prints and each operand is checked above, so a
# digit mistyped in the prose fails here while the data stays clean
chk("58x wrong at d=11", 0.7374 / 0.0128, 58, 0.6)
chk("269x wrong at d=14", 0.6730 / 0.0025, 269, 1.5)
# e1_clamp holds one worst-level score per run, so the mean is over raw numbers here and
# over run records on the line below
chk("clamped PPO worst level", 100 * mean(E["e1_clamp"]["ppo"]), 99.5, 0.06)
chk("unclamped PPO worst level", 100 * mean(r["worst"] for r in by["ppo"]), 8.3, 0.06)
chk("guided worst level", 100 * mean(r["worst"] for r in by["mcts_g"]), 99.5, 0.06)
chk("guided sub-100 run", 100 * min(r["worst"] for r in by["mcts_g"]), 94.1, 0.02)
# index 4 is the d=14 level, the last entry of TARGETS
# a reported 0 leaves clipped regret at 0 forever, so vanilla MCTS never samples that level
for m, w in [("oracle", 0.673), ("mcts_g", 0.635), ("mcts_root", 0.0025),
             ("ppo", 0.053), ("mcts", 0.0)]:
    chk(f"fig1 {m} reported ceiling", mean(r["v_hat"][4] for r in by[m]), w, 0.0006)
# a curve row holds episodes so far then scores then p, so [2][4] is the last round's
# sampling probability on the d=14 level
chk("fig1 root final p", mean(r["curve"][-1][2][4] for r in by["mcts_root"]), 0.917, 0.001)
chk("fig1 exact final p", mean(r["curve"][-1][2][4] for r in by["oracle"]), 0.667, 0.001)
# the 2026-08-06 loop recorded a point after the final round too, so its curves hold 101
# entries against the 100 in V
# 10 numbers per entry is five scores plus five probabilities
chk("fig1 values drawn", 5 * 12 * len(by["oracle"][0]["curve"]), 6060, 0)
chk("Aug 6 curve numbers", len(G["runs"]) * 101 * 10, 72720, 0)

# ---------------------------------------------------------------- Part 5
EP = V["config"]["ep_per_round"]
for L, wc, we in [(5, 2.08, 8.64), (15, 8.92, 2.02), (30, 19.67, 0.92)]:
    rs = [r for r in V["runs"] if r["n_levels"] == L and r["method"] == "oracle"]
    # a probability here is either exactly 0 from floored regret or above 1e-3, so any
    # cutoff in between counts the levels still in play
    e = mean(sum(1 for x in r["curve"][-1][2] if x > 1e-9) for r in rs)
    chk(f"L={L} competing at the end", e, wc, 0.02)
    chk(f"L={L} episodes each per round", EP / e, we, 0.02)
for L, w in [(5, 2.4), (15, 10.0), (30, 22.6)]:
    rs = [r for r in V["runs"] if r["n_levels"] == L and r["method"] == "shuffle"]
    chk(f"L={L} shuffle competing", mean(sum(1 for x in r["curve"][-1][2] if x > 1e-9)
                                         for r in rs), w, 0.06)
# rs is left over from the loop above
# every V run records the same 100 rounds, so which group is in hand does not matter
chk("fig2 values drawn", 3 * 12 * len(rs[0]["curve"]), 3600, 0)

# ---------------------------------------------------------------- Part 6
for m, wm, wz, wb in [("oracle", 126, 228, 6), ("shuffle", 101, 245, 14), ("uniform", 104, 0, 0)]:
    # one final score per level per run, so 12 runs by 30 levels
    # 0.999 is the mastery test run_one applies, and a score at or under 0.001 means the
    # greedy student never reached that goal inside the horizon
    cells = [v for r in A["runs"] if r["n_levels"] == 30 and r["method"] == m for v in r["final"]]
    chk(f"{m} mastered of 360", sum(1 for v in cells if v >= 0.999), wm, 0)
    # the document splits the remainder for oracle and shuffle only, so uniform stops here
    if wz:
        chk(f"{m} never reached", sum(1 for v in cells if v <= 0.001), wz, 0)
        chk(f"{m} partly solved", sum(1 for v in cells if 0.001 < v < 0.999), wb, 0)
chk("360 cells per panel", 12 * 30, 360, 0)
for L, rows in {5: [(35, 64.5), (34, 64.5), (36, 61.6)],
                15: [(73, 45.3), (70, 39.9), (62, 34.4)],
                30: [(126, 36.6), (101, 31.7), (104, 29.4)]}.items():
    for m, (wc, ws) in zip(["oracle", "shuffle", "uniform"], rows):
        rs = [r for r in A["runs"] if r["n_levels"] == L and r["method"] == m]
        chk(f"L={L} {m} mastered", sum(1 for r in rs for v in r["final"] if v >= 0.999), wc, 0)
        chk(f"L={L} {m} mean score", 100 * mean(r["mean"] for r in rs), ws, 0.06)

# ---------------------------------------------------------------- Part 7
# the L=5 row is identical in both tables, since 1800 total and 360 per level are the same
# thing at five levels
for tag, src, vals in [
        ("fixed", A["runs"], {5: (0.01, 4.32, 0.0, 5, 3, 0.500),
                              15: (5.33, 2.31, 6.3, 11, 0, 0.003),
                              30: (4.84, 1.74, 3.4, 10, 0, 0.019)}),
        ("control", B["runs"], {5: (0.01, 4.32, 0.0, 5, 3, 0.500),
                                15: (0.16, 1.21, 0.0, 5, 5, 0.227),
                                30: (2.29, 2.14, 1.8, 6, 1, 0.500)})]:
    for L, (wm, wse, wmd, ww, wt, wp) in vals.items():
        d = list(paired(src, L).values())
        w = sum(1 for x in d if x > 0)
        # a tie is an exact float 0, since the paired runs end on the same per-level scores
        # comparing with a tolerance would change nz and move every p in the table
        t = sum(1 for x in d if x == 0)
        nz = len(d) - t
        # one-sided sign test as the document defines it at 1.29: chance of at least w wins
        # from nz coin flips, with ties dropped from n
        p = sum(comb(nz, i) for i in range(w, nz + 1)) / 2 ** nz if nz else 1.0
        chk(f"{tag} L={L} mean", mean(d), wm, 0.011)
        chk(f"{tag} L={L} SE", stdev(d) / len(d) ** 0.5, wse, 0.011)
        chk(f"{tag} L={L} median", median(d), wmd, 0.06)
        chk(f"{tag} L={L} wins", w, ww, 0)
        chk(f"{tag} L={L} ties", t, wt, 0)
        chk(f"{tag} L={L} p", p, wp, 0.0006)
# chk takes scalars, so the twelve differences go in as a list at the one decimal the
# document prints
# the raw values carry float noise like -19.999999999999996
d5 = sorted(round(x, 1) for x in paired(A["runs"], 5).values())
want5 = [-21.2, -20.0, -20.0, -1.2, 0.0, 0.0, 0.0, 1.2, 1.2, 18.9, 20.0, 21.2]
N += 1
print(f"{'OK ' if d5 == want5 else 'BAD'} {'L=5 twelve paired differences':<50} {d5}")
if d5 != want5:
    FAIL.append("L=5 twelve paired differences")
for L, w in [(5, 64.5), (15, 63.6), (30, 65.3)]:
    chk(f"control L={L} exact score", 100 * mean(
        r["mean"] for r in B["runs"] if r["n_levels"] == L and r["method"] == "oracle"), w, 0.06)
# pm is the per-maze mean of the paired difference and loo drops one maze at a time
# maze 3 runs about three times the others, so loo is the row that carries the claim
pm = {15: [-0.8, 4.6, 6.4, 11.1], 30: [3.7, 3.4, 0.1, 12.2]}
loo = {15: [7.36, 5.58, 4.97, 3.40], 30: [5.23, 5.32, 6.41, 2.40]}
for L in (15, 30):
    d = paired(A["runs"], L)
    for mz in range(4):
        chk(f"L={L} maze {mz} mean", mean(v for k, v in d.items() if k[0] == mz), pm[L][mz], 0.06)
        chk(f"L={L} drop maze {mz}", mean(v for k, v in d.items() if k[0] != mz), loo[L][mz], 0.006)
# v_opt is 0.97^(d-1), so the log inverts it back to the goal distance
# [0] takes one seed of three, since v_opt depends on the maze and the goal set only
for L, lo, hi in [(5, 19, 23), (30, 22, 27)]:
    d = []
    for mz in range(4):
        r = [x for x in A["runs"] if x["n_levels"] == L and x["maze"] == mz
             and x["method"] == "oracle"][0]
        d.append(max(round(math.log(v) / math.log(0.97)) + 1 for v in r["v_opt"]))
    chk(f"L={L} hardest distance low", min(d), lo, 0)
    chk(f"L={L} hardest distance high", max(d), hi, 0)

# ---------------------------------------------------------------- Part 8
# seven levels per run and two of them are sealed dead ends with V* exactly 0
# score and mastery count cover the five learnable levels, and wasted_alloc is the share of
# episodes spent on the other two
for m, ws, ww, wmast in [("oracle", 64.7, 0.0, 3.00), ("mcts_clamp", 61.5, 39.5, 2.92),
                         ("ppo_clamp", 58.1, 39.9, 2.75), ("mcts_g", 50.0, 0.0, 2.50),
                         ("ppo", 34.9, 2.2, 1.67), ("mcts", 28.2, 5.0, 1.33)]:
    rs = [r for r in C["runs"] if r["method"] == m]
    chk(f"sealed {m} score", 100 * mean(r["mean"] for r in rs), ws, 0.06)
    chk(f"sealed {m} wasted", 100 * mean(r["wasted_alloc"] for r in rs), ww, 0.06)
    chk(f"sealed {m} mastered of 5", mean(r["n_mastered"] for r in rs), wmast, 0.006)
chk("504 cells", 6 * 12 * 7, 504, 0)
for b, ws, ww in [("ppo", 23.2, 37.7), ("mcts", 33.2, 34.5)]:
    P = {(r["maze"], r["seed"]): r for r in C["runs"] if r["method"] == b}
    Q = {(r["maze"], r["seed"]): r for r in C["runs"] if r["method"] == b + "_clamp"}
    chk(f"{b} clamp score gain", mean(100 * (Q[k]["mean"] - P[k]["mean"]) for k in P), ws, 0.06)
    chk(f"{b} clamp waste added",
        mean(100 * (Q[k]["wasted_alloc"] - P[k]["wasted_alloc"]) for k in P), ww, 0.06)
# waste is exactly 0 or near 0.30 in these runs, so 0.001 separates them without trusting
# float equality
w = [r for r in C["runs"] if r["method"] in ("ppo", "mcts") and r["wasted_alloc"] > 0.001]
chk("unclamped runs that wasted anything", len(w), 3, 0)
# 2 of 7 is what uniform sampling sends to the sealed levels
# the two divisions rebuild the table's waste averages from the three runs above
chk("2 of 7 as a percentage", 100 * 2 / 7, 28.6, 0.06)
chk("26.9/12", 26.9 / 12, 2.24, 0.005)
chk("(30.1+29.9)/12", (30.1 + 29.9) / 12, 5.0, 0.005)

# ---------------------------------------------------------------- Part 8b
want = {(15, "oracle"): (1.00, 10.82, 10), (15, "noisy_0.3"): (0.56, 7.09, 8),
        (15, "noisy_1.0"): (0.36, 5.46, 8), (15, "flat"): (0.00, 6.01, 8),
        (15, "shuffle"): (0.03, 5.49, 8), (30, "oracle"): (1.00, 7.14, 11),
        (30, "noisy_0.3"): (0.57, 3.48, 9), (30, "noisy_1.0"): (0.28, 4.07, 9),
        (30, "flat"): (0.00, 3.79, 8), (30, "shuffle"): (-0.04, 2.30, 7)}
for (L, m), (wr, wd, wb) in want.items():
    U = {(r["maze"], r["seed"]): r["mean"] for r in A["runs"]
         if r["n_levels"] == L and r["method"] == "uniform"}
    rs = [r for r in A["runs"] if r["n_levels"] == L and r["method"] == m]
    # flat reports one constant and spearmanr returns nan on constant input
    # the test is std < 1e-12 because those values differ in the last bit, so an equality
    # test would let the nan through and poison the mean
    rho = mean(0.0 if np.asarray(r["v_hat"]).std() < 1e-12
               else spearmanr(r["v_hat"], r["v_opt"]).correlation for r in rs)
    dl = [100 * (r["mean"] - U[(r["maze"], r["seed"])]) for r in rs]
    chk(f"L={L} {m} rank correlation", rho, wr, 0.006)
    chk(f"L={L} {m} vs no curriculum", mean(dl), wd, 0.006)
    chk(f"L={L} {m} runs better", sum(1 for x in dl if x > 0), wb, 0)
# the spread across the 12 oracle runs, which is what pairing in Part 7 removes
sc = [100 * r["mean"] for r in A["runs"] if r["n_levels"] == 30 and r["method"] == "oracle"]
chk("30-level exact span low", min(sc), 13, 0.5)
chk("30-level exact span high", max(sc), 46, 0.5)

print()
print(f"{N} checks, {len(FAIL)} failures")
for f in FAIL:
    print("  ", f)
sys.exit(1 if FAIL else 0)
