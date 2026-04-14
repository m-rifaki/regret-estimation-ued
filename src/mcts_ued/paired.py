"""PAIRED outer loop, adversary-agnostic.

Drives a `UEDAdversary` (MCTS or PPO) against a focal `Student`,
yielding one `PairedStep` per outer iteration. Swapping solvers is a
constructor-time choice; everything downstream is identical, which
keeps the MCTS-vs-PPO A/B clean.
"""
from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np

from mcts_ued.partners.parametrize import PartnerSpec
from mcts_ued.regret import RegretEstimator, positive_value_loss_regret
from mcts_ued.solvers.base import UEDAdversary
from mcts_ued.student import Student


@dataclass
class PairedConfig:
    n_outer_steps: int = 100
    # the rollout seeds come off this and nothing else. both solvers carry their
    # own seed in MCTSConfig or PPOAdversaryConfig, so two runs at one cfg.seed
    # still search differently
    seed: int = 0


@dataclass
class PairedStep:
    step: int
    spec: PartnerSpec
    student_value: float
    antagonist_value: float
    regret: float


def train_paired(
    adversary: UEDAdversary,
    student: Student,
    cfg: PairedConfig,
    regret_estimator: RegretEstimator | None = None,
) -> Iterator[PairedStep]:
    """Yield one `PairedStep` per outer iteration.

    Nothing runs until the caller iterates. `update` is called before the step
    is yielded, so a caller that stops early has already fed back the regret on
    the proposal it stopped at.
    """
    # MaxMCRegret does not fit this call: its regret() wants a level key and a
    # running max held across steps. v1 keeps the replay buffer out of the A/B
    estimator = regret_estimator or positive_value_loss_regret
    rng = np.random.default_rng(cfg.seed)
    for step in range(cfg.n_outer_steps):
        spec = adversary.propose()
        # separate draws so the student and antagonist returns carry independent
        # noise. one seed for both cancels the MockStudent noise term out of
        # their difference exactly
        # 2**31 - 1 keeps the draw inside int32, which is what env/_ippo.py
        # hands to jax.random.PRNGKey
        s_seed = int(rng.integers(0, 2**31 - 1))
        a_seed = int(rng.integers(0, 2**31 - 1))
        # cast because a jax or numpy scalar from a Student would reach
        # json.dumps in train.py and raise there
        student_value = float(student.train_step(spec, seed=s_seed))
        # the estimated ceiling for this spec, from the fixed strong recipe in
        # OvercookedConfig.antagonist_*. an experimental condition changes what
        # the ceiling is and nothing else in this loop moves
        antagonist_value = float(student.antagonist_value(spec, seed=a_seed))
        regret = float(estimator(student_value, antagonist_value))
        adversary.update(spec, regret)
        yield PairedStep(
            step=step,
            spec=spec,
            student_value=student_value,
            antagonist_value=antagonist_value,
            regret=regret,
        )
