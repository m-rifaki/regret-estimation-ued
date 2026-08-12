# Does regret-estimate accuracy matter? Isolating accuracy from support

The 2026-08-06 result (`../student_curriculum_2026_07`) found that one binary per
(run, level), did the solver report a nonzero ceiling, predicts the whole 25-cell
score table to within 0.5 points. Accuracy bought nothing, and a one-line clamp
at 0.01 took PPO from 8.3 to 99.5. Taken at face value that kills the solver
question: if a free clamp buys the same score as a 4x cheaper solver, nobody
needs a better solver.

This experiment shows why that design could not see accuracy, and finds the two
regimes where it reappears.

Why accuracy was invisible: five levels and 1800 episodes means levels are
mastered one at a time and never compete for sampling mass, and clipped regret
concentrates the curriculum on whatever is left regardless of magnitude. The
2026-08-06 data shows this directly: the root readout reports 0.0025 against a
true 0.673, off by 270x, and gives the hardest level 84.7% of all episodes, MORE
than the exact oracle's 64.9%.

## Part A: accuracy at fixed support, swept over level count

Support is held at 100% by construction. `shuffle` permutes the exact ceilings
across levels, so magnitudes and support are preserved and only the ranking is
destroyed. The distance range and the 1800-episode budget are fixed, so the only
thing changing is how many levels compete for that budget.

| levels | exact | shuffled | no curriculum | paired gap | median | favours exact | p (sign) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 5  | 64.5% | 64.5% | 61.6% | +0.01 +- 4.32 | +0.0 | 5/12 (3 ties) | 0.50 |
| 15 | 45.3% | 39.9% | 34.4% | +5.33 +- 2.31 | +6.3 | 11/12 | 0.003 |
| 30 | 36.6% | 31.7% | 29.4% | +4.85 +- 1.74 | +3.4 | 10/12 | 0.019 |

At five levels, shuffling the ceiling is a coin flip: three runs lose 20 points,
three gain 20, and the mean is +0.01. Accuracy is not neutral there, it is noise.
At fifteen and thirty levels it is a consistent gain, and at thirty the exact
ceiling masters 10.50 of 30 levels against the shuffle's 8.42.

So accuracy does matter, and the 2026-08-06 design was one level short of a
budget where ranking has any consequence.

### Control: is it scarcity, or just more and harder levels?

Levels are picked by sorting squares by distance, cutting the list into as many
equal slices as there are levels, and drawing one goal per slice, so the hardest
level creeps up with the level count (distance 19 to 23 at L=5, 22 to 27 at L=30).
The control scales the budget with the level count instead, giving every level the
same 360 episodes as the L=5 condition.

| levels | total budget fixed at 1800 | budget scales, 360 per level |
| --- | --- | --- |
| 5 | +0.01 pts (p 0.50) | +0.01 pts (p 0.50) |
| 15 | +5.33 pts (p 0.003) | +0.16 pts (p 0.23) |
| 30 | +4.85 pts (p 0.019) | +2.29 pts (p 0.50) |

The gap disappears. Under the scaled budget the exact ceiling also scores the same
at every level count (64.5%, 63.6%, 65.3%), which is the check that per-level budget
is the right normalisation. The cause is scarcity, not level count and not
difficulty.

## Part B: what the clamp costs when a level genuinely cannot be solved

Every level in the 2026-08-06 setup is learnable, so a floor of 0.01 is free.
Here two levels are sealed dead ends with `V* = 0` exactly, where honest regret
is zero forever and any sampling is waste. wall_p 0.22 on 13x13 always yields a
connected maze, so the dead ends are created by removing their incoming
transitions; nothing routes through a dead end, so no other level's optimal path
changes, and `v_opt` is recomputed after sealing.

| ceiling source | budget on unsolvable levels | mean score | levels mastered |
| --- | --- | --- | --- |
| exact oracle | 0.0% | 64.7% | 3.00/5 |
| guided MCTS | 0.0% | 50.0% | 2.50/5 |
| PPO | 2.2% | 34.9% | 1.67/5 |
| PPO + clamp | 39.9% | 58.1% | 2.75/5 |
| vanilla MCTS | 5.0% | 28.2% | 1.33/5 |
| vanilla MCTS + clamp | 39.5% | 61.5% | 2.92/5 |

Paired, the clamp adds +37.7 points of wasted budget for +23.2 points of score on
PPO, and +34.5 for +33.2 on vanilla MCTS. So the clamp is still net positive with
two of seven levels unsolvable, because the failure it prevents is worse than the
waste it causes. But it fabricates support indiscriminately: it cannot tell a
level whose ceiling was underestimated from one whose ceiling is genuinely zero,
and it spends about 40% of training on the latter.

Report honestly: clamped vanilla MCTS (61.5%) beats guided MCTS (50.0%) on score
here while wasting 40% of the budget. Nothing dominates except the oracle, which
wins on both axes. Guided MCTS wins on waste alone.

## What this says about the solver question

The solver's value is not accuracy on its own and not support on its own. It is
support that respects a true zero. The clamp buys support by asserting no level
is ever worthless; a good solver knows which levels are. That distinction is
invisible in the 2026-08-06 design (every level learnable, five of them) and it
is the reason to care which solver runs the inner loop.

The design this suggests: a floor applied only where the solver is uncertain,
not where it confidently reports zero. Untested.

## Files

- `accuracy_vs_support.py oracle` runs Part A (~4 min, no solver calls)
- `accuracy_vs_support.py perlevel` runs the control (~25 min, budget scales with L)
- `accuracy_vs_support.py clamp` runs Part B (~12 min, real solvers at B=16000)
- `make_figures.py` writes five figures, each titled with the question it answers:
  `fig_mechanism_trajectories.png` (did a bigger estimate concentrate the curriculum
  more?), `fig_competition.png` (why does a short budget make the ranking matter?),
  `fig_mastery_matrix.png` (does knowing the ranking master more levels?),
  `fig_ranking_matters.png` (and only when the budget is short?), and
  `fig_where_training_went.png` (what happens when a level cannot be solved?), and
  `fig_partial_ordering.png` (does a partly correct ordering help, or only an exactly
  correct one?).
  Design rule: one mark = one observation. Absolute score against level count is
  deliberately not plotted, because those lines all fall to the right for the trivial
  reason that a fixed budget spread over more levels gives each level fewer episodes.
- `MEETING_2026_08_18.md` is the full write-up: every term defined before use, every
  figure walked through mark by mark, and every number traced to its file.
- config is recorded in each JSON under `config`

```
uv run --with numpy,scipy python accuracy_vs_support.py oracle
uv run --with numpy,scipy python accuracy_vs_support.py perlevel
uv run --with numpy,scipy python accuracy_vs_support.py clamp
uv run --with numpy,matplotlib,seaborn python make_figures.py
```
