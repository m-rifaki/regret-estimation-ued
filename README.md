# Regret estimation for unsupervised environment design

What an inner solver's quality contributes to a student under unsupervised environment design,
measured in a setting where the exact ceiling is available.

## Install

Python 3.11 or later, with [uv](https://docs.astral.sh/uv/).

    uv sync --extra dev
    uv run pytest tests -q

The maze experiments run on one CPU core and need no accelerator. Only the Overcooked strand pulls
JAX and JaxMARL.

## Layout

    src/mcts_ued/        the loop, the student, the three solvers, the partner spaces
    experiments/         one directory per experiment, each holding its driver, its figures and its recorded results
    tests/               30 test functions over the estimators, the search and the loop
    results/             the Overcooked runs and the checkpoint evaluations

## The experiments

Each directory carries its driver, the JSON it wrote, and the script that turns that JSON into
figures. The JSON holds the runs beside the configuration block that produced them, so a rerun is
reproducible from the file alone.

| directory | what it measures | recorded file |
|---|---|---|
| `regret_solver_2026_07` | solver quality against the exact minimax distribution across seven query budgets | `mcts_vs_ppo_regret.json` |
| `student_curriculum_2026_07` | the closed loop onto a trained student, one score per ceiling source and level | `student_curriculum.json`, `extra_experiments.json`, `student_curriculum_floor.json` |
| `accuracy_vs_support_2026_08` | the ordering against a shuffled ordering at three level counts, and the price of a clamp that manufactures support | `accuracy_vs_support.json`, `accuracy_perlevel_budget.json`, `clamp_safety.json`, and the `_curves` variant of each |
| `tabular_and_bestresponse_2026_07` | the exact minimax by linear programming, the mirror-descent toy, the partner-level ceiling | `mmd_simplex_toy.json`, and `minimax_regret.json` under `results/eugene_tasks/` |

Run one from the repository root. Each driver writes its figures as well as its JSON, so the
figure stack comes with it:

    uv run --with numpy,scipy,matplotlib,seaborn python experiments/regret_solver_2026_07/mcts_vs_ppo_regret.py

## The partner strand

The project began on partner policies in Overcooked, where the generator proposes a collaborator
rather than a maze. `src/mcts_ued/partners/parametrize.py` holds four parameterizations of that
space and `src/mcts_ued/env/_ippo.py` holds an IPPO trainer written from the published algorithm.
One outer loop over five frozen checkpoints collapsed onto the strongest one, which is what moved
the work to mazes where the ceiling can be computed. None of the four parameterizations was compared
against the IPPO student, and the UCB1 search with progressive widening over the continuous box
produces no reported number.

## License

MIT. The environment comes from [JaxMARL](https://github.com/FLAIROx/JaxMARL) at commit
`c793d095`, and prioritized level replay, its robust variant and the learnability score are
reimplementations from the published descriptions with none of the authors' released code used.
