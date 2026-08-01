"""Does regret-estimate ACCURACY matter, once levels compete for a scarce budget?

The 2026-08-06 result (../student_curriculum_2026_07) found that a single binary
per (run, level) -- did the solver report a nonzero ceiling -- predicts the whole
25-cell score table to within 0.5 points. Accuracy bought nothing. But with five
levels and 1800 episodes, levels are mastered one at a time and never compete for
sampling mass, and clipped regret concentrates the curriculum automatically on
whatever is left. So accuracy CANNOT show up in that design.

This isolates accuracy from support. The corrupted oracles below all keep support
at 100% (every reported ceiling is strictly positive) and differ only in how well
they rank the levels. If `shuffle` matches `oracle` at every level count, accuracy
truly does not matter and "which solver" is a dead question.

The axis is the number of levels L, with the episode budget HELD FIXED, so mass
becomes scarce as L grows.

  uv run --with numpy,scipy python accuracy_vs_support.py oracle    # part A
  uv run --with numpy,scipy python accuracy_vs_support.py perlevel  # part A control
  uv run --with numpy,scipy python accuracy_vs_support.py clamp     # part B
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "regret_solver_2026_07"))
import mcts_vs_ppo_regret as base  # noqa: E402
from mcts_vs_ppo_regret import (  # noqa: E402
    GAMMA, HORIZON, build_maze, maze_oracle, opt_policy, policy_value,
    solve_ppo, solve_mcts, solve_mcts_guided,
)

OUT = Path(__file__).parent
# 13x13 gives about 130 reachable cells, enough for 30 quantile slots to stay populated
H = W = 13
# solver queries per level, the matched budget the base module's solvers count against
B_SOLVER = 16000
N_ROUNDS = 100
EP_PER_ROUND = 18          # 1800 episodes total, FIXED for every L
EPS_GREEDY = 0.2
CLAMP = 0.01               # the 2026-08-06 one-line fix
# CURVES=1 writes the *_curves.json set so a rerun leaves the committed JSON alone
TAG = "_curves" if os.environ.get("CURVES") else ""
# 5 is the 2026-08-06 level count, kept so part A brackets that design
LEVEL_COUNTS = [5, 15, 30]
# 4x3 is the 12 paired runs behind each sign test
N_MAZES, N_SEEDS = 4, 3

ORACLE_METHODS = ["oracle", "shuffle", "noisy_0.3", "noisy_1.0", "flat", "uniform"]
# every entry takes (M, goal, budget, seed), so the root readout needs the wrapper
SOLVER_FN = {"ppo": solve_ppo, "mcts": solve_mcts, "mcts_g": solve_mcts_guided,
             "mcts_root": lambda M, g, b, s: solve_mcts(M, g, b, s, readout="root")}


# ------------------------------------------------------------------ levels
def pick_goals(M, n_levels, rng):
    """n_levels goal cells spread over the reachable distance range (>=2).

    A level is a goal cell, not a target distance, so the level set scales past
    the number of distinct distances the maze happens to contain."""
    cand = np.array([s for s in range(M["S"]) if M["dist"][s] >= 2])
    d = M["dist"][cand]
    order = np.argsort(d, kind="stable")
    cand, d = cand[order], d[order]
    # stratify: n_levels quantile slots over the distance range, one goal each
    edges = np.linspace(0, len(cand), n_levels + 1).astype(int)
    goals = []
    for i in range(n_levels):
        # a collapsed slot (n_levels past the candidate count) would make integers(lo, lo) raise
        lo, hi = edges[i], max(edges[i + 1], edges[i] + 1)
        goals.append(int(cand[rng.integers(lo, min(hi, len(cand)))]))
    return goals


def seal_dead_ends(M, k, rng):
    """Make k levels genuinely unlearnable, so V* is exactly 0 and honest regret
    on them is 0 forever.

    wall_p 0.22 on 13x13 always yields a connected maze, so there are no
    naturally unreachable cells. Seal DEAD ENDS (cells with a single distinct
    neighbour): nothing routes through a dead end, so removing its incoming
    transitions cannot lengthen any other level's optimal path. The sealed maze
    is shared by every method, and v_opt is recomputed after sealing."""
    ends = []
    for s in range(M["S"]):
        # trans[s, a] == s is a wall bounce, so removing s leaves the real neighbours
        nb = {int(M["trans"][s, a]) for a in range(4)} - {s}
        # the >= 2 floor matches pick_goals, so a sealed cell is a goal the picker could have drawn
        if len(nb) == 1 and M["dist"][s] >= 2:
            ends.append(s)
    if len(ends) < k:
        # return before any mutation so the maze is left intact for the caller to skip
        return []
    chosen = [int(x) for x in rng.choice(ends, size=k, replace=False)]
    for g in chosen:                              # nothing can enter g any more
        for s in range(M["S"]):
            for a in range(4):
                if M["trans"][s, a] == g:
                    M["trans"][s, a] = s
    return chosen


# ------------------------------------------------------------------ student
def student_greedy(Q, rng):
    S = Q.shape[0]
    pi = np.empty(S, dtype=int)
    for s in range(S):
        q = Q[s]
        # argmax would take action 0 in every state while Q is all zeros
        pi[s] = rng.choice(np.flatnonzero(q == q.max()))
    return pi


def run_episode(M, Q, goal, rng):
    s, steps = M["s0"], 0
    for _ in range(HORIZON):
        if rng.random() < EPS_GREEDY:
            a = int(rng.integers(4))
        else:
            q = Q[s]
            a = int(rng.choice(np.flatnonzero(q == q.max())))
        ns = M["trans"][s, a]
        steps += 1
        # reward 1 on entering the goal and nothing anywhere else, as in value_iteration
        if ns == goal:
            Q[s, a] = 1.0
            return steps
        # transitions are deterministic, so one visit is the exact backup and alpha is 1
        Q[s, a] = GAMMA * Q[ns].max()
        s = ns
    return steps


def eval_student(M, Qs, goals, v_opt):
    """Exact normalized score per level. An unreachable level (v_opt == 0) has
    nothing to achieve, so it scores 1.0 and is excluded from the metrics."""
    scores = np.empty(len(goals))
    # fresh rng(0) every call so greedy ties break the same way for every method and round
    rng = np.random.default_rng(0)
    for j, g in enumerate(goals):
        if v_opt[j] <= 0:
            scores[j] = 1.0
            continue
        pi = student_greedy(Qs[j], rng)
        # exact policy evaluation, so a score difference between methods carries no rollout noise
        scores[j] = policy_value(M, pi, g)[M["s0"]] / v_opt[j]
    return scores


# ------------------------------------------------------------------ ceilings
def estimate_ceiling(method, M, goals, v_opt, seed):
    """The ONE thing that differs between methods. Every corrupted oracle keeps
    support at 100%: each reported ceiling is strictly positive."""
    rng = np.random.default_rng(7_000 + seed)
    # None is the sentinel for no ceiling at all, which run_one turns into the uniform control
    if method == "uniform":
        return None
    if method == "oracle":
        return v_opt.copy()
    if method == "shuffle":                       # same magnitudes, ranking destroyed
        return rng.permutation(v_opt)
    if method.startswith("noisy_"):
        sigma = float(method.split("_")[1])
        # sigma is in log units, so the corruption degrades the ordering while keeping the scale
        # 1e-3 holds support at 100% and 1.0 is the largest value a discounted return can take
        return np.clip(v_opt * rng.lognormal(0.0, sigma, size=len(v_opt)), 1e-3, 1.0)
    if method == "flat":                          # zero information, full support
        pos = v_opt[v_opt > 0]
        # an empty pos would make mean() nan and poison p, so fall back to the floor
        return np.full(len(v_opt), float(pos.mean()) if len(pos) else CLAMP)
    clamp = method.endswith("_clamp")
    # 6 is len("_clamp")
    fn = SOLVER_FN[method[:-6] if clamp else method]
    # the *100 stride keeps per-level solver streams disjoint while L stays under 100
    # solvers return (estimate, queries used), so index 0
    v_hat = np.array([fn(M, g, B_SOLVER, 10_000 + seed * 100 + j)[0]
                      for j, g in enumerate(goals)])
    return np.maximum(v_hat, CLAMP) if clamp else v_hat


def run_one(M, goals, method, seed, v_opt, ep_per_round=EP_PER_ROUND):
    rng = np.random.default_rng(seed)
    L = len(goals)
    v_hat = estimate_ceiling(method, M, goals, v_opt, seed)
    # one Q table per level with no sharing, so allocation is the only channel the curriculum has
    Qs = [np.zeros((M["S"], 4)) for _ in range(L)]
    alloc = np.zeros(L)
    p_prev = np.full(L, 1.0 / L)
    curve = []                                    # (episodes so far, scores, p) per round
    episodes = 0
    for _ in range(N_ROUNDS):
        scores = eval_student(M, Qs, goals, v_opt)
        if v_hat is None:
            p = np.full(L, 1.0 / L)
        else:
            # scores come back normalized by v_opt, so multiply back to compare in value units
            # the clip at zero drops a level for good once its ceiling is under the achieved value
            # that is the failure the floor at CLAMP exists to prevent
            regret = np.clip(v_hat - scores * v_opt, 0.0, None)
            # every level at or above its reported ceiling zeroes the sum, so hold the last p
            p = regret / regret.sum() if regret.sum() > 1e-9 else p_prev
            p_prev = p
        # recorded before the round's episodes, so the first entry is the untrained student at 0
        curve.append((episodes, [round(float(x), 4) for x in scores],
                      [round(float(x), 4) for x in p]))
        for level in rng.choice(L, size=ep_per_round, p=p):
            run_episode(M, Qs[level], goals[level], rng)
            alloc[level] += 1
            episodes += 1
    final = eval_student(M, Qs, goals, v_opt)
    # unlearnable is decided by v_opt, so a sealed cell drawn as a normal level still counts right
    learn = v_opt > 0                             # metrics over learnable levels only
    return {
        "method": method, "seed": seed, "n_levels": L,
        "worst": float(final[learn].min()), "mean": float(final[learn].mean()),
        # 0.999 is float slack on an exactly optimal score of 1.0
        "n_mastered": int((final[learn] >= 0.999).sum()), "n_learnable": int(learn.sum()),
        "n_dead_levels": int((~learn).sum()),
        "wasted_alloc": float(alloc[~learn].sum() / alloc.sum()) if (~learn).any() else 0.0,
        "final": final.tolist(), "alloc": (alloc / alloc.sum()).tolist(),
        "v_hat": None if v_hat is None else np.asarray(v_hat).tolist(),
        "v_opt": v_opt.tolist(),
        "curve": curve,
    }


def mazes():
    out, ms = [], 0
    while len(out) < N_MAZES:
        M = build_maze(ms, H=H, W=W)
        ms += 1
        # maze_oracle returns (ceiling, v, p_star, val) over the base module's own 5 goals
        # val is the LP minimax regret, and under 0.1 no curriculum can separate from uniform
        if maze_oracle(M)[3] > 0.1:
            out.append((ms - 1, M))
    return out


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "oracle"
    MZ = mazes()

    if which == "oracle":
        # Part A: accuracy at fixed support, swept over level count.
        runs = []
        for L in LEVEL_COUNTS:
            for mi, (mseed, M) in enumerate(MZ):
                # the goal set follows the maze seed, so raising N_MAZES leaves old sets alone
                goals = pick_goals(M, L, np.random.default_rng(500 + mseed))
                # v_opt[j] is GAMMA**(d-1) for a goal at distance d, so ceilings run 0.9 down to 0.5
                v_opt = np.array([opt_policy(M, g)[1][M["s0"]] for g in goals])
                print(f"L={L} maze {mi}: distances "
                      f"{sorted(int(M['dist'][g]) for g in goals)[:8]}...")
                for method in ORACLE_METHODS:
                    for sd in range(N_SEEDS):
                        r = run_one(M, goals, method, sd, v_opt)
                        r["maze"] = mi
                        runs.append(r)
                    # last seed only, the aggregation is in verify_numbers.py
                    last = runs[-1]
                    print(f"  {method:10s} worst {last['worst']:.3f} mean {last['mean']:.3f} "
                          f"mastered {last['n_mastered']}/{last['n_learnable']}")
        cfg = dict(level_counts=LEVEL_COUNTS, methods=ORACLE_METHODS, n_rounds=N_ROUNDS,
                   ep_per_round=EP_PER_ROUND, eps_greedy=EPS_GREEDY, H=H, W=W,
                   n_mazes=N_MAZES, n_seeds=N_SEEDS, clamp=CLAMP)
        (OUT / f"accuracy_vs_support{TAG}.json").write_text(
            json.dumps({"runs": runs, "config": cfg}))
        print("wrote accuracy_vs_support.json")

    elif which == "perlevel":
        # Control for Part A. Same level counts, but the budget scales with L so
        # every level gets the SAME number of episodes as the L=5 condition
        # (1800/5 = 360). Scarcity is removed while the level count is kept, so
        # if the accuracy gap disappears here it was competition for a scarce
        # budget rather than the level count or the difficulty range.
        runs = []
        for L in LEVEL_COUNTS:
            ep = EP_PER_ROUND * L // LEVEL_COUNTS[0]           # 18, 54, 108
            for mi, (mseed, M) in enumerate(MZ):
                goals = pick_goals(M, L, np.random.default_rng(500 + mseed))
                v_opt = np.array([opt_policy(M, g)[1][M["s0"]] for g in goals])
                print(f"L={L} maze {mi}: {ep * N_ROUNDS} episodes "
                      f"({ep * N_ROUNDS // L} per level)")
                for method in ["oracle", "shuffle", "uniform"]:
                    for sd in range(N_SEEDS):
                        r = run_one(M, goals, method, sd, v_opt, ep_per_round=ep)
                        r["maze"] = mi
                        r["episodes"] = ep * N_ROUNDS
                        runs.append(r)
                    last = runs[-1]
                    print(f"  {method:10s} mean {last['mean']:.3f} "
                          f"mastered {last['n_mastered']}/{last['n_learnable']}")
        # 360 below is EP_PER_ROUND * N_ROUNDS / 5 written by hand, so it goes stale if either moves
        (OUT / f"accuracy_perlevel_budget{TAG}.json").write_text(json.dumps(
            {"runs": runs, "config": {"level_counts": LEVEL_COUNTS,
                                      "episodes_per_level": 360,
                                      "n_rounds": N_ROUNDS, "n_mazes": N_MAZES,
                                      "n_seeds": N_SEEDS}}))
        print("wrote accuracy_perlevel_budget.json")

    elif which == "clamp":
        # Part B: does the clamp waste budget on genuinely unlearnable levels?
        runs = []
        for mi, (mseed, M) in enumerate(MZ):
            rng = np.random.default_rng(900 + mseed)
            # mutates M in place, so no other branch may reuse these mazes in the same process
            dead = seal_dead_ends(M, 2, rng)
            if not dead:
                print(f"maze {mi}: fewer than 2 dead ends, skipped")
                continue
            # 5 learnable plus 2 sealed, so the only change from 2026-08-06 is the true zeros
            goals = pick_goals(M, 5, np.random.default_rng(500 + mseed)) + dead
            v_opt = np.array([opt_policy(M, g)[1][M["s0"]] for g in goals])
            print(f"maze {mi}: {len(dead)} unlearnable levels, v_opt {v_opt.round(3).tolist()}")
            # each solver appears with and without the floor, so the cost is a paired difference
            for method in ["oracle", "ppo", "ppo_clamp", "mcts", "mcts_clamp", "mcts_g"]:
                for sd in range(N_SEEDS):
                    r = run_one(M, goals, method, sd, v_opt)
                    r["maze"] = mi
                    runs.append(r)
                last = runs[-1]
                print(f"  {method:10s} worst {last['worst']:.3f} "
                      f"wasted_alloc {last['wasted_alloc']:.3f}")
        (OUT / f"clamp_safety{TAG}.json").write_text(
            json.dumps({"runs": runs, "config": {"clamp": CLAMP, "b_solver": B_SOLVER}}))
        print("wrote clamp_safety.json")


if __name__ == "__main__":
    main()
