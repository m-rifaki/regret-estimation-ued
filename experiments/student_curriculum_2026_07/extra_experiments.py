"""Three follow-ups on the support result. Reuses student_curriculum.py.

E1 clamp: report max(v_hat, 0.01) instead of a raw zero. Does support alone
   rescue the greedy-readout curricula?
E2 floor sweep: how much blind exploration floor does it take to rescue a
   zeroed estimate? (0.10 already failed.)
E3 support vs budget: at what solver budget does PPO / vanilla-greedy stop
   reporting the distance-14 ceiling as exactly 0?

  uv run --with numpy,scipy,matplotlib python extra_experiments.py
"""
import json
from pathlib import Path

import numpy as np

import student_curriculum as sc

OUT = Path(__file__).parent


def get_mazes():
    mazes, ms = [], 0
    while len(mazes) < 4:
        M = sc.build_maze(ms, H=sc.H, W=sc.W)
        ms += 1
        if sc.maze_oracle(M)[3] > 0.1:
            mazes.append(M)
    return mazes


def main():
    mazes = get_mazes()
    out = {}

    # E1: clamp the reported ceiling away from zero
    real_estimate = sc.estimate_ceiling

    def clamped(method, M, seed):
        v_hat, err = real_estimate(method, M, seed)
        return np.maximum(v_hat, 0.01), err

    sc.estimate_ceiling = clamped
    e1 = {}
    for m in ["ppo", "mcts"]:
        ws = []
        for M in mazes:
            for sd in range(3):
                r = sc.run_one(M, m, sd, 0.0)
                ws.append(r["worst"])
                print(f"E1 clamp {m} worst {r['worst']:.2f}")
        e1[m] = ws
    sc.estimate_ceiling = real_estimate
    out["e1_clamp"] = e1

    # E2: floor sweep, no clamp
    e2 = {}
    for f in [0.02, 0.10, 0.20, 0.30, 0.50]:
        for m in ["ppo", "mcts"]:
            ws = []
            for M in mazes:
                for sd in range(3):
                    ws.append(sc.run_one(M, m, sd, f)["worst"])
            e2[f"{m}@{f}"] = ws
            print(f"E2 floor {f} {m}: mean worst {np.mean(ws)*100:.1f} ({sum(w >= 0.99 for w in ws)}/12 at 100)")
    out["e2_floor"] = e2

    # E3: does the solver report distance-14 > 0, per budget (solver only)
    e3 = {}
    for m in ["ppo", "mcts"]:
        fn = sc.SOLVER_FN[m]
        for B in [16000, 32000, 64000, 128000, 256000]:
            hits = []
            for mi, M in enumerate(mazes):
                g = M["goals"][-1]
                for sd in range(3):
                    v = fn(M, g, B, 10_000 + sd * 100 + 4)[0]
                    hits.append(v > 0)
            e3[f"{m}@{B}"] = float(np.mean(hits))
            print(f"E3 {m} B={B}: support {np.mean(hits)*100:.0f}%")
    out["e3_support"] = e3

    (OUT / "extra_experiments.json").write_text(json.dumps(out))
    print("wrote extra_experiments.json")


if __name__ == "__main__":
    main()
