"""Focal RL student protocol + a deterministic mock for tests/smoke runs."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from mcts_ued.partners.parametrize import PartnerSpec


class Student(Protocol):
    """RL student that responds to partner-spec proposals.

    `train_step(spec, seed)` returns the focal-student's mean return
    after one inner-loop training pass with `spec` as the partner.
    `antagonist_value(spec, seed)` returns the antagonist's mean return
    on the same partner, used as the PAIRED regret anchor.

    The two values reach the adversary only through their difference in
    `mcts_ued.regret`, so an offset common to both is invisible to it.
    `train_paired` draws a fresh seed per call and carries no student
    state across outer steps, so an implementation may retrain from
    scratch on every call. OvercookedIPPOStudent does exactly that.
    """

    def train_step(self, partner: PartnerSpec, *, seed: int) -> float: ...

    def antagonist_value(self, partner: PartnerSpec, *, seed: int) -> float: ...


@dataclass
class MockStudent:
    """Deterministic synthetic student for tests + outer-loop smoke runs.

    Builds a known regret landscape so adversary-side unit tests can
    verify convergence in O(seconds). Not used in real training; the
    real student lives in `mcts_ued.env.overcooked.OvercookedIPPOStudent`.

    The landscape is one Gaussian bump over the normalized params.
    Positive value loss reduces to a scaled copy of that bump with its
    single maximum at `peak`, which is the answer the solver tests
    check their proposals against.
    """

    # in the adversary's normalized [0, 1] view of the box, never the
    # decoded kwargs. length has to match `PartnerSpace.dim`
    peak: np.ndarray
    # normalized param units. at 0.15 the median uniform draw from the 5-D
    # box scores 6e-5 of the peak, so the convergence tests and smoke_ab
    # widen it to 0.25 to give a first-order signal from a cold start
    width: float = 0.15
    # student below antagonist is what puts the regret maximum on the peak.
    # flipping the two floors the positive value loss to zero over the whole
    # box. the pair stands in for the real capacity split, where the
    # antagonist trains with its own hidden width and step budget
    student_strength: float = 0.55
    antagonist_strength: float = 1.0
    # every test constructs with 0.0 so the landscape assertions stay exact.
    # this default is the outer-loop smoke path only
    noise_std: float = 0.02

    def _bump(self, params: np.ndarray) -> float:
        # one width for all five axes and no read of decode_kwargs, so a mock
        # run leaves the decode map in parametrize.py untested
        d2 = float(np.sum((params - self.peak) ** 2))
        # unnormalized gaussian, so the value at the peak is exactly 1.0 and
        # train_step tops out at student_strength
        return float(math.exp(-d2 / (2 * self.width * self.width)))

    def train_step(self, partner: PartnerSpec, *, seed: int) -> float:
        # a generator built per call keeps (spec, seed) reproducible. one held
        # on the dataclass would make the return depend on how many times it
        # had been drawn from before
        rng = np.random.default_rng(seed)
        # the draw comes from seed alone. params never enter it, so a caller
        # holding the seed fixed shifts the whole landscape by one constant
        # and still ranks specs exactly. train.py's leaf_value_fn does that
        return self.student_strength * self._bump(partner.params) + float(
            rng.normal(0.0, self.noise_std)
        )

    def antagonist_value(self, partner: PartnerSpec, *, seed: int) -> float:
        # paired.py draws a second seed for this call, so the two noise terms
        # are independent and the regret carries noise_std * sqrt(2). handing
        # both calls the same seed would cancel the noise instead
        rng = np.random.default_rng(seed)
        return self.antagonist_strength * self._bump(partner.params) + float(
            rng.normal(0.0, self.noise_std)
        )


def build_mock_student(cfg: dict[str, Any]) -> MockStudent:
    """Build a `MockStudent` from the `student:` block of a train.py yaml."""
    # only axis 0 is off the box center, so the distance to the peak is carried
    # by that axis and a solver parked at the center is visible in mu[0] alone.
    # float32 matches BoxPartnerSpace.clip so the subtraction in _bump holds
    peak = np.asarray(cfg.get("peak", [0.7, 0.5, 0.5, 0.5, 0.5]), dtype=np.float32)
    # the defaults are repeated from the dataclass so a yaml student block can
    # name one key and leave the rest out
    return MockStudent(
        peak=peak,
        width=float(cfg.get("width", 0.15)),
        student_strength=float(cfg.get("student_strength", 0.55)),
        antagonist_strength=float(cfg.get("antagonist_strength", 1.0)),
        noise_std=float(cfg.get("noise_std", 0.02)),
    )
