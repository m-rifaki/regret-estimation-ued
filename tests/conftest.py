"""Shared test fixtures for the mcts_ued unit tests.

Neither fixture is parametrized and both hold the package defaults. A
test that needs a different bound or a different strength builds its own
object instead of overriding here.
"""
from __future__ import annotations

import numpy as np
import pytest

from mcts_ued.partners.parametrize import BoxConfig, BoxPartnerSpace
from mcts_ued.student import MockStudent


@pytest.fixture
def box_space() -> BoxPartnerSpace:
    # the space holds no per-test state. decode and clip each return a fresh
    # array and the solvers mutate only their own mu and log_std, so a session
    # scope here would read the same
    # every test that takes this reads spec.params or the key set of
    # decode_kwargs, so the BoxConfig bounds below never reach an assertion.
    # test_parametrize.py is where a changed bound shows up and it builds its
    # own space
    return BoxPartnerSpace(BoxConfig())


@pytest.fixture
def mock_student() -> MockStudent:
    # no test requests this one. test_paired.py rebuilds the same five values in
    # its own _student helper and the convergence test there widens width to
    # 0.25, because at 0.15 a uniform draw from the 5-D box scores 6e-5 of the
    # peak and a cold start has no signal to follow
    return MockStudent(
        # only axis 0 is off the box center, so the whole distance to the peak
        # is carried by one coordinate and a solver parked at the center shows
        # up in mu[0]
        peak=np.array([0.7, 0.5, 0.5, 0.5, 0.5], dtype=np.float32),
        width=0.15,
        # student under antagonist is what holds positive value loss above zero
        # away from the peak. flipping the two floors the regret to zero over
        # the whole box and a landscape assertion would pass on nothing
        student_strength=0.55,
        antagonist_strength=1.0,
        # overrides the 0.02 dataclass default. with the noise term gone both
        # returns are functions of params alone and a regret assertion can be
        # an equality
        noise_std=0.0,
    )
