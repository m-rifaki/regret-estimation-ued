"""PPO-style partner-policy adversary (PAIRED baseline).

PAIRED's classic adversary is a gradient-trained policy over level
parameters with regret as the reward. Here the "environment" the
adversary controls is partner-policy parameters.

The adversary is a diagonal Gaussian policy over the `PartnerSpace`'s
[0, 1]^dim Box. Each `propose` samples one point; `update(spec, regret)`
applies one PPO-clipped REINFORCE step at that point. Numpy-only:
PPO clipping on a 1-step Gaussian bandit has no benefit from JAX/torch
auto-diff, and analytic gradients keep the implementation auditable.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass

import numpy as np

from mcts_ued.partners.parametrize import PartnerSpace, PartnerSpec


@dataclass
class PPOAdversaryConfig:
    """Hyperparameters for the PPO adversary.

    lr_mu and lr_log_std at zero freeze the sampling distribution. scripts/smoke_ab.py
    builds the domain-randomization control arm out of this same class that way.
    """

    init_log_std: float = -0.5      # exp(-0.5) ~ 0.61 std on [0, 1]
    lr_mu: float = 0.05
    lr_log_std: float = 0.01
    clip_eps: float = 0.2
    # exp(-2.0) ~ 0.14 and exp(0.5) ~ 1.65 std on a unit-width axis. the floor keeps
    # the proposal distribution off a point mass and the ceiling stops sigma from
    # outgrowing the box where clip would stack samples on 0 and 1
    log_std_min: float = -2.0
    log_std_max: float = 0.5
    # advantage baseline is the trailing mean of the last 32 regrets. a fixed window
    # follows drift in the regret scale where a lifetime mean would keep subtracting
    # a stale level
    baseline_window: int = 32
    seed: int = 0


@dataclass
class _Pending:
    """One proposal held from propose until its regret arrives.

    mu and log_std are copied at sampling time because the PPO ratio needs the
    distribution the sample came from and the current one may have moved.
    """

    params: np.ndarray
    mu: np.ndarray
    log_std: np.ndarray
    log_prob: float


def _gaussian_log_prob(x: np.ndarray, mu: np.ndarray, log_std: np.ndarray) -> float:
    """Log density of the diagonal Gaussian N(mu, diag(exp(log_std))**2) at x.

    The 2 pi normalizer cancels inside the PPO ratio and is kept here so the return
    value is a real log density.
    """
    sigma = np.exp(log_std)
    z = (x - mu) / sigma
    return float(
        -0.5 * np.sum(z * z)
        - np.sum(log_std)
        - 0.5 * x.size * math.log(2.0 * math.pi)
    )


class PPOAdversary:
    """Diagonal-Gaussian PPO-clipped 1-step bandit over PartnerSpace."""

    def __init__(self, cfg: PPOAdversaryConfig, space: PartnerSpace):
        self.cfg = cfg
        self.space = space
        self.rng = np.random.default_rng(cfg.seed)
        self.mu = np.full(space.dim, 0.5, dtype=np.float32)
        self.log_std = np.full(space.dim, cfg.init_log_std, dtype=np.float32)
        self._baseline: deque = deque(maxlen=cfg.baseline_window)
        # keyed by the proposed vector so any number of proposals can be outstanding and
        # update can arrive out of order. train_paired keeps exactly one live
        self._pending: dict[tuple, _Pending] = {}

    def propose(self) -> PartnerSpec:
        sigma = np.exp(self.log_std)
        eps = self.rng.normal(0.0, 1.0, size=self.space.dim).astype(np.float32)
        sample = self.space.clip(self.mu + sigma * eps)
        # the unclipped Gaussian density read at the clipped point. clip folds tail
        # mass onto the faces of the box and neither this log prob nor the one in
        # update accounts for it
        log_prob = _gaussian_log_prob(sample, self.mu, self.log_std)
        # decode re-clips an already clipped float32 vector so spec.params comes back
        # bit-identical and this key matches the pop in update. a space that rescaled
        # params here would turn every update into a silent no-op
        key = tuple(np.asarray(sample).tolist())
        self._pending[key] = _Pending(
            params=sample.copy(),
            mu=self.mu.copy(),
            log_std=self.log_std.copy(),
            log_prob=log_prob,
        )
        return self.space.decode(sample)

    def update(self, spec: PartnerSpec, regret: float) -> None:
        key = tuple(np.asarray(spec.params).tolist())
        rec = self._pending.pop(key, None)
        # no gradient for a spec this instance never proposed or already popped. the
        # alternative moves mu at a point that was never sampled
        if rec is None:
            return
        # baseline reads the window as it stood before this regret. folding the regret
        # into its own baseline would shrink the advantage toward zero
        baseline = float(np.mean(self._baseline)) if self._baseline else 0.0
        advantage = float(regret) - baseline
        self._baseline.append(float(regret))

        # train_paired updates each proposal before the next propose so mu and log_std
        # have not moved and this ratio is exactly 1. the clip below does something
        # only for a caller that keeps several proposals outstanding
        new_log_prob = _gaussian_log_prob(rec.params, self.mu, self.log_std)
        ratio = math.exp(new_log_prob - rec.log_prob)
        clip_low = 1.0 - self.cfg.clip_eps
        clip_high = 1.0 + self.cfg.clip_eps
        if (advantage >= 0 and ratio > clip_high) or (advantage < 0 and ratio < clip_low):
            return  # PPO clip: trust region exited, drop the gradient

        # Gradient of log pi(x; mu, log_std) wrt mu and log_std,
        # evaluated at the *old* policy (the sampling distribution).
        old_sigma = np.exp(rec.log_std)
        diff = (rec.params - rec.mu) / old_sigma
        # d_mu is (x - mu) / sigma**2 and diff already carries one of the two divisions.
        # d_log_std is z**2 - 1 for that same standardized z
        d_mu = diff / old_sigma
        d_log_std = diff * diff - 1.0
        scale = ratio * advantage

        # left unclipped the mean walks outside the box and every later sample clips
        # onto the same face. the adversary would then repeat one spec forever
        self.mu = self.space.clip(self.mu + self.cfg.lr_mu * scale * d_mu)
        self.log_std = np.clip(
            self.log_std + self.cfg.lr_log_std * scale * d_log_std,
            self.cfg.log_std_min,
            self.cfg.log_std_max,
        ).astype(np.float32)
