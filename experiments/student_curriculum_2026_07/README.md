# Student curriculum runs: does the regret estimate reach the final agent?

Last week's result (`../regret_solver_2026_07`) stops at the generator's distribution.
This closes the loop: train an actual student under each curriculum and score it
against the exact optimum. The board sketch from the 7/30 meeting is exactly this
plot (test performance under curriculum, method 1 vs method 2).

## Result

| ceiling source | worst level | mean |
| --- | --- | --- |
| exact oracle | 100.0 (12/12 at 100) | 100.0 |
| guided MCTS | 99.5 (11/12) | 99.9 |
| vanilla MCTS, root readout | 90.7 (9/12) | 98.1 |
| uniform | 83.3 (10/12) | 96.7 |
| PPO (REINFORCE) | 8.3 (1/12) | 76.7 |
| vanilla MCTS, greedy readout | 0.0 (0/12) | 61.7 |

The split is on one condition. Pool the 60 non-uniform runs (five V* sources x 12)
and split on "did the solver report the distance-14 ceiling as exactly 0":
zero -> 24/24 runs end at 0 on that level; nonzero -> 36/36 end at 94 to 100.
Reported ceiling 0 means clipped regret 0 means sampling probability 0 forever
(the generator is frozen on zero gradient). The failure is silent and worse than
no curriculum (uniform saves 10/12).

The readout ablation matters: the same vanilla search read out by a greedy rollout
reports 0.000 and fails completely; read out by its root value it reports ~0.003,
off by two orders of magnitude from the true 0.673, and recovers to 90.7, because
once easy levels are mastered their regret clips away and any leftover regret
dominates. What the estimate must get right is a nonzero value on every learnable
level; accuracy is second order.

Guided's one sub-100 run is 94.1 = 0.97^2: a two-step detour that beat the slightly
low estimated ceiling (0.633 vs 0.673), so regret clipped and the detour froze in.
An underestimate caps the level at whatever policy first beats it. Tabular students
hide most of this (all-or-nothing mastery); a value network should feel the cap
much harder. That is the prediction for the jaxued run.

## What broke before this worked

1. 9x9, fat budget, 10% floor: every method at 1.00, nothing to compare. The floor
   plus the budget were doing the curriculum's job.
2. Dropped the floor, still ~1.00: my zero-regret fallback reset the generator to
   uniform, which retrained the starved levels for free. A regret maximizer with
   zero gradient stays put; fixing that exposed the starvation.
3. First full 13x13 run read guided out by its internal value but vanilla by a
   greedy rollout, two different estimands. The root-value readout of the same
   search flips vanilla's outcome, so both readouts are in the grid (fresh seeds).

Also checked and killed: "2x sample efficiency vs uniform" (1.6x at the 80%
crossing, partly driven by uniform's two failed runs); an error-vs-outcome
correlation plot (root readout has the largest error and recovers, wrong axis);
"the floor rescues them" (forced 10% floor, greedy-readout methods still fail
23/24, see student_curriculum_floor.json).

## Setup

13x13 random mazes, level = goal at distance 2/5/8/11/14, 4 mazes x 3 seeds,
1800 episodes per run. Random walk reaches distance 14 in 0.7-2.3% of episodes
(calibrate.py, measured on the experiment's own mazes); at B=16000 guided has
every ceiling within 0.12 while PPO/vanilla-greedy report distance 14 as 0.
Six V* sources: oracle (value iteration), guided MCTS, PPO-style REINFORCE
(tabular policy gradient, no clip/critic), vanilla UCT greedy readout, vanilla UCT
root readout, uniform. Shared: tabular Q-learning student (lr 1, deterministic
env, eps 0.2), p proportional to clipped regret, generator frozen on zero total
regret. The student's value is computed exactly by policy evaluation, so the
ceiling is the only estimated quantity in the loop.

## Follow-ups (extra_experiments.py, 2026-08-06)

- One-line clamp (never report a ceiling below 0.01): PPO 8.3 -> 99.5, vanilla
  greedy 0.0 -> 98.6. The runs that miss 100 land at 94.1 and 88.5 = 0.97^2 and
  0.97^4, the detour cap again. Support alone restores survival; the residue is
  the accuracy cost.
- Floor sweep (2/10/20/30/50%): blind exploration barely helps. Even a 50% floor
  rescues only 2-3 runs of 12. Exploration cannot repair a broken estimate.
- Support vs solver budget: PPO/vanilla stop reporting 0 for distance 14 at
  about 8x the budget (0-8% support at 16k, ~half at 64k, 100% at 128k). What a
  better solver buys first is a nonzero estimate.

## Files

- `student_curriculum.py` runs the grid (~5 min), writes `student_curriculum.json`
  and the `_floor` variant
- `calibrate.py` picks the regime (maze size, budget, targets)
- `make_figures.py` writes the 5 figures + `summary.txt`
- `extra_experiments.py` runs the clamp / floor-sweep / support-vs-budget
  follow-ups, writes `extra_experiments.json`

```
uv run --with numpy,scipy,matplotlib python student_curriculum.py
uv run --with numpy,matplotlib python make_figures.py
```
