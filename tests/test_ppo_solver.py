"""Unit tests for the PPO partner-policy adversary.

Each test drives propose and update by hand, so the pending-key round trip that
train_paired would hide is what they cover. The same solver under the outer loop
is in test_paired.py.
"""
from __future__ import annotations

import math

import numpy as np

from mcts_ued.partners.parametrize import BoxPartnerSpace
from mcts_ued.solvers.ppo import PPOAdversary, PPOAdversaryConfig


def test_propose_returns_decoded_spec(box_space: BoxPartnerSpace):
    adv = PPOAdversary(PPOAdversaryConfig(seed=0), box_space)
    spec = adv.propose()
    assert spec.params.shape == (5,)
    # the seed-0 draw is already inside the box before propose clips it, so both
    # bounds below hold even for a propose that returned its raw Gaussian sample.
    # what they catch is a decode that rescaled params to the natural ranges
    assert spec.params.min() >= 0.0
    assert spec.params.max() <= 1.0
    # _ippo.py indexes these five names on every partner-training call. a renamed
    # axis raises there minutes into a run and here in a second
    assert set(spec.decode_kwargs.keys()) == {
        "hidden_dim", "n_layers", "lr", "train_steps", "action_temperature",
    }


def test_mu_drifts_toward_high_regret_region(box_space: BoxPartnerSpace):
    """Reward proposals near params[0] = 0.9; adversary's mean should drift up."""
    # 2x the config default. the default walks mu[0] onto the same face over these
    # 200 steps, so the raised rate shortens the walk and decides nothing here
    adv = PPOAdversary(PPOAdversaryConfig(seed=0, lr_mu=0.1), box_space)
    initial_mu0 = float(adv.mu[0])
    for _ in range(200):
        spec = adv.propose()
        # only axis 0 enters the reward. the other four still take a step on every
        # update because the same advantage multiplies their per-axis gradient, and
        # mu[4] ends near 0.05 with nothing behind it
        regret = float(math.exp(-((spec.params[0] - 0.9) ** 2) / (2 * 0.1**2)))
        adv.update(spec, regret)
    # mu[0] crosses this by step 7 and is pinned at the 1.0 face past the 0.9 peak
    # for more than half the run. the threshold asks for the direction only
    assert float(adv.mu[0]) > initial_mu0 + 0.05


def test_update_without_matching_pending_is_safe(box_space: BoxPartnerSpace):
    adv = PPOAdversary(PPOAdversaryConfig(seed=0), box_space)
    # decoded off the space, so no propose ever registered this vector. without the
    # early return in update, rec.params raises AttributeError and the test fails
    # before it reaches the assertion
    fake = box_space.decode(np.array([0.5, 0.5, 0.5, 0.5, 0.5], dtype=np.float32))
    adv.update(fake, regret=1.0)
    # the fake is the box center and mu starts there too, so an update that did find
    # a record would compute a zero mean gradient and pass this line anyway. the
    # crash is what the test holds
    assert float(adv.mu[0]) == 0.5


def test_log_std_stays_inside_bounds(box_space: BoxPartnerSpace):
    # lr_log_std is 1000x the config default and the two bounds are tighter than the
    # defaults, so one step is enough to reach a clip
    cfg = PPOAdversaryConfig(seed=1, lr_log_std=10.0,
                             log_std_min=-1.0, log_std_max=0.2)
    adv = PPOAdversary(cfg, box_space)
    for _ in range(50):
        spec = adv.propose()
        # a constant regret makes the trailing baseline equal it from the second
        # update on. the advantage is then zero and 49 of these 50 iterations move
        # nothing, so the loop count is decoration
        adv.update(spec, regret=1.0)
    # clip holds every sample within 0.5 of the starting mean and sigma opens at
    # 0.61, so |z| tops out at 0.83 and z**2 - 1 is negative on all five axes. this
    # seed reaches the floor on its one live step and never approaches the ceiling
    assert float(adv.log_std.min()) >= cfg.log_std_min - 1e-6
    # float32 rounds 0.2 up to 0.20000000298 and update casts back to float32 after
    # np.clip, so an exact comparison against log_std_max fails on the cast alone
    assert float(adv.log_std.max()) <= cfg.log_std_max + 1e-6


def test_zero_lr_keeps_distribution_uniform(box_space: BoxPartnerSpace):
    """The degenerate domain-randomization config should not move mu."""
    # the dr arm in scripts/smoke_ab.py is built from these same four values. the
    # name says uniform but a sigma of 1.0 on a box one unit wide clips 39 of the 40
    # proposals below onto a face
    cfg = PPOAdversaryConfig(seed=0, lr_mu=0.0, lr_log_std=0.0, init_log_std=0.0)
    adv = PPOAdversary(cfg, box_space)
    initial_mu = adv.mu.copy()
    initial_log_std = adv.log_std.copy()
    for _ in range(40):
        spec = adv.propose()
        # the generator is rebuilt from seed 0 inside the loop, so every call returns
        # 0.6369616873214543 and the regret is constant. one generator outside the
        # loop is what would vary it
        adv.update(spec, regret=float(np.random.default_rng(0).uniform()))
    # with both rates at zero the clip and the float32 cast in update are the one
    # path left that could still move an in-box mean
    assert np.allclose(adv.mu, initial_mu)
    assert np.allclose(adv.log_std, initial_log_std)
