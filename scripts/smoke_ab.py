"""End-to-end smoke A/B: MCTS vs PPO vs DR adversary on the MockStudent.

This runs all three adversaries against the same MockStudent peak +
budget so we can verify the PAIRED outer loop works before the real
JaxMARL student is wired. Output is a 3-row summary table.

    python scripts/smoke_ab.py [--steps 200] [--seed 0]

The three rows are not compute matched. An MCTS propose spends
`n_simulations` student calls before it answers and a PPO propose spends
none, so at the defaults the first row buys its numbers with 6400 extra
train_step calls.

The dr row is a frozen Gaussian centered on the box at std 1.0. Clipping
folds 62 percent of its coordinates onto a face, which scores an order of
magnitude under the uniform draw `BoxPartnerSpace.sample` makes.
"""
from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path
from typing import Iterable

import numpy as np

# covers `python scripts/smoke_ab.py` with no install. under `uv run` from the
# repo root the editable install already resolves mcts_ued and this is a no-op
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from mcts_ued.paired import PairedConfig, PairedStep, train_paired  # noqa: E402
from mcts_ued.partners.parametrize import BoxPartnerSpace  # noqa: E402
from mcts_ued.solvers.mcts import MCTSAdversary, MCTSConfig  # noqa: E402
from mcts_ued.solvers.ppo import PPOAdversary, PPOAdversaryConfig  # noqa: E402
from mcts_ued.student import MockStudent  # noqa: E402


def _summarise(label: str, steps: Iterable[PairedStep]) -> dict[str, object]:
    step_list = list(steps)
    regrets = [s.regret for s in step_list]
    # max() covers --steps under 5 where the floor division gives 0 and the
    # slice regrets[-0:] would hand back the whole run as its own last quintile
    last_q = regrets[-max(1, len(regrets) // 5):]
    return {
        "adversary": label,
        "n_steps": len(step_list),
        "regret_mean": statistics.fmean(regrets),
        "regret_last_quintile_mean": statistics.fmean(last_q),
        # the landscape below caps regret at 0.6, so this column reads against a
        # known ceiling and an arm near it has stood on the peak
        "regret_max": max(regrets),
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    # 200 is what the whole table costs under a second on the mock. the lr_mu
    # below is picked against this count and does not carry to a longer run
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    space = BoxPartnerSpace()
    # only axis 0 is off the box center. 0.85 holds the maximum off the wall so a
    # solver that walks its mean onto the face at 1.0 overshoots visibly
    peak = np.array([0.85, 0.5, 0.5, 0.5, 0.5], dtype=np.float32)
    student = MockStudent(
        peak=peak,
        width=0.25,
        # the difference of the two strengths is the regret ceiling, so 0.6 is the
        # most any row can print and the center of the box already scores 0.225
        student_strength=0.4,
        antagonist_strength=1.0,
        # deterministic landscape, so a row is one run and --seed moves only the
        # solvers. repeated --seed values are the error bar this table has
        noise_std=0.0,
    )

    def leaf_value(spec):
        # MCTS scores a leaf by the student return. on MockStudent that is the
        # regret up to a positive scale so the argmax agrees, and the agreement
        # ends as soon as the antagonist value moves with the spec
        # a fixed seed across leaves does nothing while noise_std is 0
        return float(student.train_step(spec, seed=args.seed))

    mcts = MCTSAdversary(
        # 32 leaf evaluations per propose is 6400 student calls over a 200-step
        # run against 200 for either other arm
        cfg=MCTSConfig(n_simulations=32, seed=args.seed),
        space=space,
        leaf_value_fn=leaf_value,
    )
    # 6x the config default. over 200 steps the default 0.05 walks mu[0] to 0.60
    # and 0.3 walks it to the 1.0 face, so neither run parks on the 0.85 peak
    ppo = PPOAdversary(PPOAdversaryConfig(seed=args.seed, lr_mu=0.3), space)
    # the same class with both learning rates at zero, so the sampler never moves.
    # init_log_std 0.0 overrides the -0.5 default and widens sigma to 1.0, which is
    # what puts almost every draw on a face of the box
    dr = PPOAdversary(
        PPOAdversaryConfig(
            seed=args.seed,
            init_log_std=0.0,
            lr_mu=0.0,
            lr_log_std=0.0,
        ),
        space,
    )

    rows = []
    for label, adv in [("mcts", mcts), ("ppo", ppo), ("dr", dr)]:
        # the three arms share one student instance and one seed, so they draw the
        # same rollout-seed stream and the second arm searches the landscape the
        # first one left untouched
        cfg = PairedConfig(n_outer_steps=args.steps, seed=args.seed)
        rows.append(_summarise(label, train_paired(adv, student, cfg)))

    print(f"{'adversary':<10} {'mean':>8} {'last20%':>10} {'max':>8}")
    for r in rows:
        print(
            f"{r['adversary']:<10} {r['regret_mean']:>8.3f} "
            f"{r['regret_last_quintile_mean']:>10.3f} {r['regret_max']:>8.3f}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
