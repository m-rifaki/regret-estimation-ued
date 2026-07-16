"""Train a student under each curriculum and score it against the exact optimum.

Six sources for the ceiling V*, everything else shared. See README.md for the
result and for the three designs that failed before this one (9x9 too easy, a
uniform fallback that rescued the misled generators, and a readout mismatch
between guided and vanilla MCTS that forced the mcts_root ablation).

The six sources differ on support before they differ on accuracy. A ceiling
reported as exactly 0 clips that level's regret to 0 and the generator stops
drawing the level for the rest of the run.

  uv run --with numpy,scipy,matplotlib python student_curriculum.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "regret_solver_2026_07"))
import mcts_vs_ppo_regret as base  # noqa: E402
# build_maze reads TARGETS off the base module at call time, so the assignment has
# to precede both the import below and every build_maze call
base.TARGETS = [2, 5, 8, 11, 14]   # 13x13 regime: far levels need concentrated samples
from mcts_vs_ppo_regret import (  # noqa: E402
    GAMMA, HORIZON, N_LEVELS, TARGETS,
    build_maze, maze_oracle, opt_policy, policy_value,
    solve_ppo, solve_mcts, solve_mcts_guided,
)


def solve_mcts_root(M, goal, budget, seed):
    """Vanilla MCTS with the root-value readout (max root Q = W/N) in place of
    the achieved-return readout. Same search, different estimand: the readout
    ablation the estimand question demands."""
    return solve_mcts(M, goal, budget, seed, readout="root")

OUT = Path(__file__).parent
H = W = 13                         # 13x13: random-walk first success on distance 14 is 0.7-2.3%
B_SOLVER = 16000                   # guided estimates all 5 ceilings; the greedy-readout baselines put 0 on distance 14
N_ROUNDS = 100                     # outer curriculum rounds
EP_PER_ROUND = 18                  # student episodes per round (tight budget: allocation matters)
EPS_GREEDY = 0.2
METHODS = ["oracle", "mcts_g", "ppo", "mcts", "mcts_root", "uniform"]
SOLVER_FN = {"ppo": solve_ppo, "mcts": solve_mcts, "mcts_g": solve_mcts_guided,
             "mcts_root": solve_mcts_root}


def student_greedy(Q, rng):
    """Greedy policy with random tie-breaking (before the first success all
    Q are zero, so behavior is a uniform random walk)."""
    S = Q.shape[0]
    pi = np.empty(S, dtype=int)
    for s in range(S):
        q = Q[s]
        best = np.flatnonzero(q == q.max())
        pi[s] = rng.choice(best)
    return pi


def run_episode(M, Q, goal, rng):
    """One eps-greedy Q-learning episode. Deterministic transitions, so lr=1
    (each backup is exact). Returns env steps consumed."""
    s, steps = M["s0"], 0
    for _ in range(HORIZON):
        if rng.random() < EPS_GREEDY:
            a = int(rng.integers(4))
        else:
            q = Q[s]
            a = int(rng.choice(np.flatnonzero(q == q.max())))
        # walls self-loop in trans, so a bump costs a step like any other move
        ns = M["trans"][s, a]
        steps += 1
        if ns == goal:
            # reward 1 on entering the goal and no bootstrap past it, matching
            # value_iteration's terminal case
            Q[s, a] = 1.0
            return steps
        Q[s, a] = GAMMA * Q[ns].max()
        s = ns
    # reached only when the goal was missed, so this always returns HORIZON
    return steps


def eval_student(M, Qs, v_opt):
    """Exact normalized score per level: value of the greedy policy (computed by
    policy evaluation, no rollout noise) over the exact optimum."""
    scores = np.empty(N_LEVELS)
    rng = np.random.default_rng(0)                # tie-break only; value is exact given pi
    for j, g in enumerate(M["goals"]):
        pi = student_greedy(Qs[j], rng)
        # goals come from the BFS-reachable set so v_opt is positive here
        scores[j] = policy_value(M, pi, g)[M["s0"]] / v_opt[j]
    return scores


def estimate_ceiling(method, M, seed):
    """The ONE thing that differs between methods."""
    if method == "oracle":
        return np.array([opt_policy(M, g)[1][M["s0"]] for g in M["goals"]]), 0.0
    if method == "uniform":
        return None, None
    fn = SOLVER_FN[method]
    # the 10_000 offset keeps these searches off the student rng seeded 0 to 2
    v_hat = np.array([fn(M, g, B_SOLVER, 10_000 + seed * 100 + j)[0]
                      for j, g in enumerate(M["goals"])])
    # for the record only, since feeding this back would make every method the oracle
    v_exact = np.array([opt_policy(M, g)[1][M["s0"]] for g in M["goals"]])
    return v_hat, float(np.mean(np.abs(v_hat - v_exact)))


def run_one(M, method, seed, floor):
    rng = np.random.default_rng(seed)
    # scoring is always against the exact optimum, and only the curriculum sees v_hat
    v_opt = np.array([opt_policy(M, g)[1][M["s0"]] for g in M["goals"]])
    v_hat, ceil_err = estimate_ceiling(method, M, seed)
    # one Q table per level, so the allocation is the only channel between levels
    Qs = [np.zeros((M["S"], 4)) for _ in range(N_LEVELS)]
    samples, alloc = 0, np.zeros(N_LEVELS)
    p_prev = np.full(N_LEVELS, 1.0 / N_LEVELS)
    curve = []                                    # (samples, per-level scores, p)
    for _ in range(N_ROUNDS):
        scores = eval_student(M, Qs, v_opt)
        if method == "uniform":
            p = np.full(N_LEVELS, 1.0 / N_LEVELS)
        else:
            # clipping at 0 caps a level at the first policy to beat the reported
            # ceiling, so an underestimate is never corrected
            regret = np.clip(v_hat - scores * v_opt, 0.0, None)
            # zero regret everywhere = zero gradient for a regret-maximizing
            # generator: the distribution stays where it is (no reset)
            p = regret / regret.sum() if regret.sum() > 1e-9 else p_prev
            # p_prev holds the unmixed p, so a frozen generator does not remix the
            # floor every round and drift to uniform
            p_prev = p
            p = (1 - floor) * p + floor / N_LEVELS
        # recorded before the round's episodes, so scores and samples are the state going in
        curve.append((samples, scores.tolist(), p.tolist()))
        for level in rng.choice(N_LEVELS, size=EP_PER_ROUND, p=p):
            samples += run_episode(M, Qs[level], M["goals"][level], rng)
            alloc[level] += 1
    final = eval_student(M, Qs, v_opt)
    # p here is the previous round's, since only the scores are recomputed after the loop
    curve.append((samples, final.tolist(), p.tolist()))
    return {"method": method, "seed": seed, "ceil_err": ceil_err,
            "v_hat": None if v_hat is None else v_hat.tolist(), "v_opt": v_opt.tolist(),
            "final": final.tolist(), "worst": float(final.min()), "mean": float(final.mean()),
            "alloc": (alloc / alloc.sum()).tolist(), "curve": curve}


def main():
    n_mazes, n_seeds = 4, 3
    mazes, ms = [], 0
    while len(mazes) < n_mazes:
        M = build_maze(ms, H=H, W=W)
        ms += 1
        # index 3 of maze_oracle is the LP minimax value, near zero when a single
        # student covers every level
        if maze_oracle(M)[3] > 0.1:               # real regret structure only
            mazes.append((ms - 1, M))
    # both variants run the same four mazes, so the floor is the only difference
    # between the two json files
    for floor, tag in [(0.0, ""), (0.10, "_floor")]:
        runs = []
        for mi, (mseed, M) in enumerate(mazes):
            dists = [int(M["dist"][g]) for g in M["goals"]]
            print(f"[floor={floor}] maze {mi} (seed {mseed}) goal distances {dists}")
            for method in METHODS:
                for sd in range(n_seeds):
                    r = run_one(M, method, sd, floor)
                    r["maze"] = mi
                    runs.append(r)
                    print(f"  {method:8s} seed {sd}  worst {r['worst']:.2f}  mean {r['mean']:.2f}"
                          f"  ceil_err {r['ceil_err'] if r['ceil_err'] is not None else float('nan'):.3f}"
                          f"  alloc {[f'{a:.2f}' for a in r['alloc']]}")
        out = {"runs": runs, "methods": METHODS, "targets": TARGETS, "b_solver": B_SOLVER,
               "n_rounds": N_ROUNDS, "ep_per_round": EP_PER_ROUND, "floor": floor,
               "eps_greedy": EPS_GREEDY, "n_mazes": n_mazes, "n_seeds": n_seeds}
        (OUT / f"student_curriculum{tag}.json").write_text(json.dumps(out))
        print("wrote", OUT / f"student_curriculum{tag}.json")


if __name__ == "__main__":
    main()
