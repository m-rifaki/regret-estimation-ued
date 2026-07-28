# MCTS-UED: tabular toy + best-response experiments (June to July 2026)

These are the two experiments that came before the regret-solver comparison in
`experiments/regret_solver_2026_07`. They were run in a working directory that was
wiped between sessions, so the exact data files (the 5-partner Overcooked cross-play
matrix) and the base Overcooked modules are not bundled here. The code and the
results are recorded below; the follow-up in `regret_solver_2026_07` is fully
self-contained and runnable.

## 1. The tabular MMD toy (Eugene 6/11): solve the easy problem first

Freeze the antagonist so the partner game is a 2-player zero-sum game (student mix
vs generator over partners, payoff = clipped regret), plot the dynamics on the
2-simplex, and check convergence to the minimax point. Two dynamics, one MMD update
(Sokota-Lanctot: entropy magnet + KL to the previous step, both annealed).

- Result on the real 5-partner cross-play data: on the 3-partner subset that carries
  the mass, vanilla Hedge did NOT converge (cycled, landed on the wrong vertex,
  TV 0.75 from the LP minimax). Annealed MMD converged to the exact LP minimax
  `p* = [0, 0.25, 0.75]`, TV 0.021.
- `mmd_simplex_toy.py` here reproduces the METHOD on a self-contained illustrative
  3-partner regret game (the real cross-play matrix was on the wiped machine). It is
  runnable and produces `mmd_simplex_toy.png`: Hedge cycles, MMD reaches the LP
  minimax. `minimax_regret.py` is the exact LP oracle used throughout.

## 2. The best-response estimator fix (July)

Carrying the MMD generator into the real 5-partner loop with the antagonist unfrozen
still landed on the wrong partner. The cause was the REGRET ESTIMATE, not the update:
the co-trained antagonist scored below the student (`v_ant < v_pro`), so the measured
regret went negative and mis-ranked the partners.

- Fix: train a real best-response per partner for a clean ceiling `V*(j)`.
  `V* = [66, 54, 120, 59, 59]`; regret `= max(0, V*(j) - v_pro)`.
- With the corrected regret the generator settled cleanly (entropy 1.61 -> 0.72,
  no cycling) onto partner 2, TV 0.18 from the minimax of the trained-V* regret game
  `[0, 0, 1, 0, 0]`.
- This is the finding that motivated the July regret-solver comparison: a correct
  solver produces a correct regret, which controls whether the curriculum converges.
- `pipeline_bestresponse.py` is the driver. It needs the base Overcooked modules
  (`ippo`, `ippo_corrected`, `pipeline_corrected`) that lived in the working
  directory; the result and figure are recorded in `docs/`.

## Where to look next
- `experiments/regret_solver_2026_07/`  - the fully-runnable follow-up: PPO vs vanilla
  UCT vs value-guided MCTS, showing value-guided MCTS reaches the minimax at ~4x less
  compute. This is the experiment to build on.
- `docs/meeting-2026-07-16.md`  - the meeting where this became a paper direction.
