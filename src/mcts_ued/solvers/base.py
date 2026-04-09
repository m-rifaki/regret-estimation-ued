"""Adversary protocol shared by MCTS and PPO solvers."""
from __future__ import annotations

from typing import Protocol

from mcts_ued.partners.parametrize import PartnerSpec


class UEDAdversary(Protocol):
    """The adversary side of the UED game (PAIRED outer loop).

    Implementations search over a `PartnerSpace` and return one
    `PartnerSpec` per outer-loop step; the loop feeds back the regret
    observed on that proposal via `update`.

    `train_paired` calls `propose` then `update` once each per step and
    passes back the same `PartnerSpec` object it was handed. `PPOAdversary`
    finds its pending record by the exact params vector, so a caller that
    re-decodes or perturbs the spec between the two calls gets a silent
    no-op where it expected a gradient step.

    Structural only: neither `MCTSAdversary` nor `PPOAdversary` subclasses
    this. It is not `runtime_checkable`, so an isinstance test against it
    raises TypeError instead of answering.
    """

    def propose(self) -> PartnerSpec:
        """Return one partner-policy proposal for the next student rollout.

        The returned spec carries the adversary's own [0, 1] search
        coordinates in `params`. A stateful adversary matches the regret in
        `update` back to the point it sampled by that vector.
        """
        # no arguments: what an implementation knows about the student arrives
        # through its constructor, the way MCTS takes `leaf_value_fn`

    def update(self, spec: PartnerSpec, regret: float) -> None:
        """Feed back the observed regret for the proposed partner."""
        # regret is whatever estimator `train_paired` was built with, so
        # `signed_regret` can deliver a negative value here
        # a no-op is a conforming implementation: MCTS v1 rebuilds its tree
        # inside propose and carries nothing across outer steps
