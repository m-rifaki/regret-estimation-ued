"""Unit tests for the partner-policy parametrization.

The space has no inverse map, so nothing here recovers params from decode_kwargs.
Every expected value is read out of the same BoxConfig the decode reads, which
pins these tests to the shape of the map and leaves them blind to the bounds.
"""
from __future__ import annotations

import numpy as np
import pytest

from mcts_ued.partners.parametrize import BoxConfig, BoxPartnerSpace


def test_dim_is_five():
    # five is what the solvers size mu and sigma by and what MockStudent.peak has
    # to match. decode indexes p[0] through p[4] by hand, so a sixth axis added to
    # BoxConfig would decode to nothing and leave this green
    space = BoxPartnerSpace()
    assert space.dim == 5


def test_clip_bounds_to_unit_interval():
    space = BoxPartnerSpace()
    p = np.array([-1.0, 0.5, 2.0, 0.5, 1.5], dtype=np.float32)
    clipped = space.clip(p)
    # the dtype line is the one with teeth. PPOAdversary keys _pending by
    # tuple(spec.params), so a float64 vector reaching it builds keys that never
    # match the ones its update pops
    assert clipped.dtype == np.float32
    # the two in-range 0.5 entries are never read back, so np.zeros_like would
    # satisfy both bounds below
    assert clipped.min() >= 0.0
    assert clipped.max() <= 1.0


def test_decode_at_zero_yields_minima():
    cfg = BoxConfig()
    space = BoxPartnerSpace(cfg)
    spec = space.decode(np.zeros(5, dtype=np.float32))
    assert spec.decode_kwargs["hidden_dim"] == cfg.hidden_dim_min
    assert spec.decode_kwargs["n_layers"] == cfg.n_layers_min
    assert spec.decode_kwargs["train_steps"] == cfg.train_steps_min
    # the three integer axes above come out of round() exact at a corner. lr does
    # not, because exp of log of lr_min comes back about one ulp off
    assert spec.decode_kwargs["lr"] == pytest.approx(cfg.lr_min, rel=1e-5)
    # exact here too, since decode adds 0.0 times the width to temp_min. the
    # approx is unused slack
    assert spec.decode_kwargs["action_temperature"] == pytest.approx(cfg.temp_min)


def test_decode_at_one_yields_maxima():
    cfg = BoxConfig()
    space = BoxPartnerSpace(cfg)
    # 1.0 survives the float32 cast in clip, so the top of every axis is reachable
    # by decode. it is the one point sample never draws, since rng.uniform is half
    # open on the upper end
    spec = space.decode(np.ones(5, dtype=np.float32))
    assert spec.decode_kwargs["hidden_dim"] == cfg.hidden_dim_max
    assert spec.decode_kwargs["n_layers"] == cfg.n_layers_max
    assert spec.decode_kwargs["train_steps"] == cfg.train_steps_max
    assert spec.decode_kwargs["lr"] == pytest.approx(cfg.lr_max, rel=1e-5)
    assert spec.decode_kwargs["action_temperature"] == pytest.approx(cfg.temp_max)


def test_lr_is_log_uniform_at_midpoint():
    cfg = BoxConfig()
    space = BoxPartnerSpace(cfg)
    spec = space.decode(np.array([0.5, 0.5, 0.5, 0.5, 0.5], dtype=np.float32))
    # a linear lr decode passes both corner tests above, because the endpoints
    # agree. the midpoint is the only place the two maps separate
    expected_lr = (cfg.lr_min * cfg.lr_max) ** 0.5  # log-uniform midpoint
    # 5.5e-4 here against 1.55e-3 for a linear midpoint. the two are a factor of
    # 2.8 apart so rel=1e-5 has room to spare
    assert spec.decode_kwargs["lr"] == pytest.approx(expected_lr, rel=1e-5)


def test_sample_inside_box():
    space = BoxPartnerSpace()
    rng = np.random.default_rng(0)
    # sample decodes what it draws and decode clips, so neither bound below can
    # report anything. shape is what is left for the loop to catch
    # 64 draws off seed 0 stay inside the box and never reach the corners the two
    # decode tests pin
    for _ in range(64):
        spec = space.sample(rng)
        assert spec.params.shape == (5,)
        assert spec.params.min() >= 0.0
        assert spec.params.max() <= 1.0
