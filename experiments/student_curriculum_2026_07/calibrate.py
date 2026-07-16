"""Pick the regime: maze size, targets, solver budget, episode budget.

Scouting run behind student_curriculum.py. It writes no file and reads no json,
so re-running it cannot disturb a finished experiment. The base module's 9x9
default left every method at 1.00 with nothing to separate them. These prints
are what moved the experiment to 13x13 goals at 2/5/8/11/14 and B_SOLVER 16000.

EP_PER_ROUND is never printed here. The random-walk rate at the end is what
bounds it.
"""
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "regret_solver_2026_07"))
import mcts_vs_ppo_regret as base

# both imports resolve to one module object in sys.modules, so this write is
# what the already-bound build_maze reads at call time
# the list has to equal the one in student_curriculum.py or the calibration
# describes a regime the experiment never runs
base.TARGETS = [2, 5, 8, 11, 14]
from mcts_vs_ppo_regret import build_maze, opt_policy, solve_ppo, solve_mcts, solve_mcts_guided, maze_oracle, GAMMA, HORIZON

# the same maze selection as the experiment: first 4 with minimax value > 0.1
kept, ms = [], 0
while len(kept) < 4:
    M = build_maze(ms, H=13, W=13)
    # index 3 of maze_oracle is the LP minimax value. near zero means one student
    # covers every level and there is no regret game for a curriculum to play
    if maze_oracle(M)[3] > 0.1:
        kept.append((ms, M))
    ms += 1
for ms, M in kept:
    val = maze_oracle(M)[3]
    # build_maze accepts a goal within 2 steps of its target, so the real
    # difficulty of a level comes off dist and TARGETS is only the request
    dists = [int(M["dist"][g]) for g in M["goals"]]
    # reward is paid once on entering the goal, so each entry is GAMMA ** (dist - 1)
    # and the printed row can be checked against dists by hand
    v_exact = np.array([opt_policy(M, g)[1][M["s0"]] for g in M["goals"]])
    # ms is the build seed, which student_curriculum.py logs as mseed. the maze
    # index it prints is the position in this list
    print(f"maze {ms}: dists {dists}  minimax val {val:.3f}  v_exact {np.round(v_exact,2)}")
    # the two candidates for B_SOLVER. 16000 was taken because guided holds every
    # ceiling within 0.12 there while the other two report distance 14 as 0
    for B in (8000, 16000):
        # solve_mcts takes its default greedy readout, so the root-value arm the
        # experiment later had to add is uncalibrated
        for name, fn in (("ppo", solve_ppo), ("mcts", solve_mcts), ("g", solve_mcts_guided)):
            # 7 + j is per level and shared by the three solvers, so the rows are
            # paired on the same draws. solve_mcts_guided ignores its seed entirely
            vh = np.array([fn(M, g, B, 7 + j)[0] for j, g in enumerate(M["goals"])])
            # a level reported as 0 contributes its whole ceiling here, so this
            # column moves with support before it moves with accuracy
            print(f"   B={B:6d} {name:4s} v_hat {np.round(vh,2)}  err {np.abs(vh - v_exact).mean():.3f}")
    # first-success rate of a pure random walk (the student before any success), per level
    # one stream for all five levels, so the per-level rates are independent draws.
    # a fresh 0 per maze keeps the maze rows comparable
    rng = np.random.default_rng(0)
    for j, g in enumerate(M["goals"]):
        succ = 0
        # 400 walks resolves a rate to 0.0025, so the 0.007 quoted for distance 14
        # is three hits and carries about one significant figure
        for _ in range(400):
            s = M["s0"]
            # HORIZON is the same 60-step cap run_episode uses, so this is the
            # per-episode success probability a cold student faces
            for _ in range(HORIZON):
                # uniform over four actions matches the student with an all-zero Q,
                # where student_greedy resolves the tie uniformly
                # a wall self-loops in trans, so a bump costs a step of the walk
                s = M["trans"][s, rng.integers(4)]
                if s == g:
                    succ += 1
                    break
        # the experiment runs 1800 episodes. a uniform curriculum spends 360 of them
        # on distance 14 and buys 2 to 8 first successes at the rates printed here
        print(f"   level {j} (d={dists[j]}): random-walk success {succ/400:.3f}")
