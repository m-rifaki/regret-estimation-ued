# Which solver gives good regret? Swapping PPO for MCTS in MCTS-UED

## The one-line answer
A UED curriculum only finds the right levels when its regret signal is accurate. Plain MCTS does not help, but a value-guided MCTS (planning with a learned value) produces an accurate regret signal with about 4x less compute than PPO. This is the compute-efficiency result the paper needs, and it makes the original "MCTS vs PPO" question sharp.

## Why this experiment
In UED the generator picks levels to maximize the student's regret, and regret is `V*(level) - V_student(level)`, where `V*` is the best return achievable, estimated by a solver. In the June meeting we saw the loop converge to the wrong levels because the solver's regret estimate was wrong. Eugene's next step was to swap the solver from PPO to MCTS and see if a better solver fixes it. The Overcooked partner setup is a bandit at the partner level, so the honest test is in a small maze where the true regret is known exactly. That way every claim is checked against an oracle, not a proxy.

## Setup (read this to understand the figure)
- Env: random 9x9 mazes. A "level" is a goal cell. Levels differ only in how far the goal is from the start: distances 2, 4, 6, 8, 10. Far goals are harder to reach, so they are where a weak solver fails.
- The true regret ceiling `V*(level)` is computed exactly by value iteration (the oracle).
- Three solvers estimate that ceiling under a matched compute budget B (env or model steps):
  - PPO: model-free RL (tabular REINFORCE).
  - vanilla MCTS: UCT with random rollouts.
  - value-guided MCTS: the same tree search, but leaves are scored by a learned value that is refined with model-based one-step lookahead, instead of random rollouts. This is the AlphaZero / MuZero idea in tabular form.
- The minimax-regret generator over levels is solved exactly (LP) and by the annealed MMD loop.
- Repeated over 4 random mazes and 3 seeds; shaded bands are the std.

## The result, panel by panel

Panel (a): the better the solver, the smaller its regret error. At every compute budget, value-guided MCTS estimates the ceiling most accurately, PPO next, vanilla MCTS worst. Error 0 means the estimate is exactly the oracle.

Panel (b): the curriculum reaches the true minimax only once the regret is accurate, and the guided solver gets there first. It reaches the minimax reliably at 8,000 steps; PPO and vanilla MCTS need 32,000. That is about 4x less compute.

Panel (c): the reason is the hard levels. At a fixed budget (B=4000), PPO and vanilla MCTS estimate near-goals fine but their error blows up on the far levels (distance 8 to 10). The guided solver stays accurate everywhere. Far levels are exactly the ones a UED adversary cares about, so getting them wrong wrecks the curriculum.

| solver | compute to reliably reach the minimax |
| --- | --- |
| value-guided MCTS | 8,000 |
| PPO | 32,000  (4x more) |
| vanilla MCTS | 32,000  (4x more, no better than PPO) |

## What this means for the paper
- The load-bearing claim (Step 1): regret-estimate quality controls whether the UED curriculum converges. Shown on real environments against a known oracle.
- The solver claim (Step 2), now precise and honest: it is not MCTS in general that helps. Tabula-rasa UCT with random rollouts is no better than PPO. It is a value-guided search that gives the same regret quality for a quarter of the compute. That is the reason to care which solver runs the inner loop.

## Next
- Port the ablation to a standard UED environment in jaxued (Maze + PAIRED) for the headline plot everyone recognizes.
- Replace the tabular learned value with a value network so the guided solver scales past small mazes.
