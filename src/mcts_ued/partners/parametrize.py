"""Partner-policy uncertainty set parametrization.

UED-over-partners requires an explicit parametrization of the partner-
policy uncertainty set. This module defines a `PartnerSpec` (one point
in the set) and a `PartnerSpace` (the set + a sampling/decoding
interface). The choice of parametrization is the open research
question of the project; see `docs/parametrization.md`.

v1 uses a 5-D Box over (hidden_dim, n_layers, lr, train_steps,
action_temperature) so the end-to-end A/B between MCTS and PPO
solvers is runnable. Alternative parametrizations (population-index,
latent-z, behavior-prior) plug in via the same `PartnerSpace` protocol.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass(frozen=True)
class PartnerSpec:
    """One concrete partner specification, sampled or proposed.

    `params` is the canonical numpy vector the adversary searches over
    (each component is in [0, 1] for the v1 Box space). `decode_kwargs`
    is the python-level kwargs the partner-training routine consumes;
    decoding lives on `PartnerSpace` so the adversary stays abstract.
    """

    params: np.ndarray
    decode_kwargs: dict


class PartnerSpace(Protocol):
    """Abstract over the partner-policy uncertainty set."""

    @property
    def dim(self) -> int: ...

    def sample(self, rng: np.random.Generator) -> PartnerSpec: ...

    def decode(self, params: np.ndarray) -> PartnerSpec: ...

    # on the protocol because PPOAdversary pushes its own mean through it on
    # every update. a space that only bounded proposals would let mu drift out
    def clip(self, params: np.ndarray) -> np.ndarray: ...


@dataclass
class BoxConfig:
    """5-D Box parametrization: (hidden_dim, n_layers, lr, train_steps, action_temperature).

    Each axis is normalized to [0, 1] for the adversary's view; decode
    rescales to the per-axis natural range. `lr` is decoded log-uniform.

    The bounds come from the cramped_room engagement note: a 4x5x30
    observation makes 32 to 256 hidden units enough. None of the five
    axes has been checked against OvercookedIPPOStudent. Every run so
    far scored specs with MockStudent, which reads `params` and never
    touches decode_kwargs.
    """

    hidden_dim_min: int = 32
    hidden_dim_max: int = 256
    n_layers_min: int = 1
    n_layers_max: int = 4
    # decoded log-uniform because the range spans a factor of 30. a linear
    # decode would put half the box above 1.5e-3, five times the student lr
    lr_min: float = 1e-4
    lr_max: float = 3e-3
    # _run_ippo sizes the update count by the larger of the focal and partner
    # budgets. the focal side is already 50k in OvercookedConfig so the top of
    # this axis buys no extra env steps
    train_steps_min: int = 5_000
    train_steps_max: int = 50_000
    # temperature divides the partner's logits in ActorCritic. the neutral 1.0
    # decodes from p = 1/3 so two uniform draws in three give a flatter partner
    temp_min: float = 0.5
    temp_max: float = 2.0


class BoxPartnerSpace:
    """5-D continuous Box partner-policy parametrization."""

    DIM = 5

    def __init__(self, cfg: BoxConfig | None = None):
        self.cfg = cfg or BoxConfig()

    @property
    def dim(self) -> int:
        return self.DIM

    def clip(self, params: np.ndarray) -> np.ndarray:
        # decode has to be total over R^5. both solvers clip their own states
        # already, so what this catches is a hand-built vector reaching the IPPO
        # trainer as a hidden_dim above 256
        return np.clip(np.asarray(params, dtype=np.float32), 0.0, 1.0).astype(np.float32)

    def sample(self, rng: np.random.Generator) -> PartnerSpec:
        params = rng.uniform(0.0, 1.0, size=self.DIM).astype(np.float32)
        return self.decode(params)

    def decode(self, params: np.ndarray) -> PartnerSpec:
        # np.clip returns a new array, so the spec below owns its vector and a
        # solver mutating its own state in place cannot rewrite a spec it
        # already handed out
        p = self.clip(params)
        c = self.cfg
        hidden_dim = int(round(c.hidden_dim_min + p[0] * (c.hidden_dim_max - c.hidden_dim_min)))
        # round() over 4 integers gives 1 and 4 half the width of 2 and 3, so a
        # uniform p is not a uniform depth
        n_layers = int(round(c.n_layers_min + p[1] * (c.n_layers_max - c.n_layers_min)))
        lr = math.exp(
            math.log(c.lr_min) + p[2] * (math.log(c.lr_max) - math.log(c.lr_min))
        )
        train_steps = int(round(
            c.train_steps_min + p[3] * (c.train_steps_max - c.train_steps_min)
        ))
        temp = c.temp_min + p[4] * (c.temp_max - c.temp_min)
        return PartnerSpec(
            # this is the clipped vector and not the caller's input. PPOAdversary
            # pops _pending by tuple(spec.params) so any rescaling here would turn
            # every update into a silent no-op
            params=p,
            decode_kwargs={
                "hidden_dim": hidden_dim,
                "n_layers": n_layers,
                "lr": float(lr),
                "train_steps": train_steps,
                "action_temperature": float(temp),
            },
        )
