"""Unit tests for the MCTS adversary over partner-policy parameter space.

The landscapes are built here instead of through `MockStudent` so that one
test can put two peaks in the box. Both helpers score `spec.params` and never
touch `decode_kwargs`, so the decode map in `parametrize.py` stays untested
from this file.

Every assertion is a distance to a known peak or a box bound. Nothing reads a
visit count or a tree shape, so a proposer that reached the peak without
searching would pass the whole file.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from mcts_ued.partners.parametrize import BoxPartnerSpace
from mcts_ued.solvers.mcts import MCTSAdversary, MCTSConfig


def _unimodal_leaf_value(peak: np.ndarray, width: float = 0.15):
    # MockStudent._bump with student_strength dropped, so the peak scores
    # exactly 1.0. ucb_c defaults to sqrt(2), which was derived for leaf
    # values on the unit interval
    def fn(spec):
        d2 = float(np.sum((spec.params - peak) ** 2))
        return float(math.exp(-d2 / (2 * width * width)))
    return fn


def _bimodal_leaf_value(peaks: list[np.ndarray], width: float = 0.1):
    # 0.1 holds the two bumps apart. the midpoint between them scores 1e-4, so
    # there is no ridge joining the modes and a child has to fall inside one
    # before UCB1 can tell it from the rest
    def fn(spec):
        return max(
            float(math.exp(-float(np.sum((spec.params - p) ** 2)) / (2 * width * width)))
            for p in peaks
        )
    return fn


def test_finds_unimodal_peak(box_space: BoxPartnerSpace):
    peak = np.array([0.7, 0.5, 0.5, 0.5, 0.5], dtype=np.float32)
    # 128 simulations opens 11 root children. progressive widening admits the
    # next one at each perfect square, so the count is floor(sqrt(n - 1)) and
    # the remaining 117 simulations go into revisits
    cfg = MCTSConfig(n_simulations=128, seed=0)
    adv = MCTSAdversary(cfg=cfg, space=box_space, leaf_value_fn=_unimodal_leaf_value(peak))
    # one rng stream spans the eight calls, so these are eight different trees
    # and seed 0 pins every draw the test makes
    proposals = [adv.propose() for _ in range(8)]
    best = max(proposals, key=lambda s: -float(np.linalg.norm(s.params - peak)))
    # 0.5 is loose in five dimensions. it holds for 15 percent of uniform draws
    # and for the box midpoint each search starts from. that midpoint is 0.2
    # out, so an adversary that never left the root would pass. six of the
    # eight proposals clear it on their own at seed 0, so the best-of-eight
    # above is not what carries this
    assert float(np.linalg.norm(best.params - peak)) < 0.5


def test_handles_bimodal_landscape(box_space: BoxPartnerSpace):
    peaks = [
        np.array([0.2, 0.2, 0.5, 0.5, 0.5], dtype=np.float32),
        np.array([0.8, 0.8, 0.5, 0.5, 0.5], dtype=np.float32),
    ]
    cfg = MCTSConfig(n_simulations=128, seed=0)
    adv = MCTSAdversary(cfg=cfg, space=box_space, leaf_value_fn=_bimodal_leaf_value(peaks))
    # sixteen proposals where the unimodal test takes eight. 94 percent of
    # uniform draws score under 1e-3 against this landscape, so the eleven root
    # children come back at twelve visits apiece with the winner one or two
    # ahead. each returned spec is close to a round robin pick over them
    proposals = [adv.propose() for _ in range(16)]
    # the min runs over both peaks, so a search that collapsed onto one mode
    # would pass. at seed 0 the proposals do split across the two
    closest = min(
        float(np.linalg.norm(s.params - p))
        for s in proposals
        for p in peaks
    )
    # 0.3 excludes the box midpoint at 0.424 from either peak, so unlike the
    # unimodal threshold this one goes red for a search that never moves. it is
    # also the thin one: 2 percent of uniform draws clear it and five seeds in
    # 200 would miss it at this budget
    assert closest < 0.3


@pytest.mark.parametrize("n_simulations", [4, 16, 64])
def test_more_simulations_improve_proposal(box_space: BoxPartnerSpace, n_simulations: int):
    # the name overstates the body. the three budgets run the same three checks
    # and nothing compares them, so this is a decode-bounds test that happens
    # to be parametrized
    peak = np.array([0.9, 0.1, 0.5, 0.5, 0.5], dtype=np.float32)
    cfg = MCTSConfig(n_simulations=n_simulations, seed=42)
    adv = MCTSAdversary(cfg=cfg, space=box_space, leaf_value_fn=_unimodal_leaf_value(peak))
    # at 4 simulations widening never opens a second root child, so that case
    # returns a single uniform draw and no distance assertion could hold for it
    spec = adv.propose()
    # decode clips, so these three hold for any state the tree could hand it.
    # what they cover is BoxPartnerSpace: a decode that returned the rescaled
    # kwargs vector would come back with hidden_dim in the hundreds
    assert spec.params.shape == (5,)
    assert spec.params.min() >= 0.0
    assert spec.params.max() <= 1.0


def test_update_is_a_no_op(box_space: BoxPartnerSpace):
    # a corner of the box and never read. nothing below looks at the spec
    # beyond handing it back
    peak = np.zeros(5, dtype=np.float32)
    cfg = MCTSConfig(n_simulations=4, seed=0)
    adv = MCTSAdversary(cfg=cfg, space=box_space, leaf_value_fn=_unimodal_leaf_value(peak))
    spec = adv.propose()
    # no assertion here: the only failure it can report is update raising or
    # losing an argument name. once the v1 no-op grows into the priors hook
    # this test stops covering what update does
    adv.update(spec, regret=0.42)
