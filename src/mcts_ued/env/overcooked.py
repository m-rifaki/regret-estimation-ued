"""Overcooked v2 (JaxMARL) IPPO student.

Wires a JaxMARL IPPO student against a partner whose architecture +
training config is decoded from the `PartnerSpace` (the v1 Box).
Antagonist value is computed by training the same partner with a
fixed strong recipe (max train_steps, mid temperature) so the
`positive_value_loss` regret estimator has a meaningful anchor.

JaxMARL imports are deferred to `_ensure_jax`; the rest of the
package + tests run with numpy only. Install the JAX stack via
`pip install '.[jax]'` (or `uv pip install -e '.[jax]'`).

Note: this v0 wires the env+student structure end-to-end. The IPPO
gradient loop lives in `_ippo.py` and is invoked from `train_step`
and `antagonist_value`. Smoke-running it requires JAX/Flax/Optax and
a JaxMARL install; without those the module imports fine but the
methods raise a clear RuntimeError pointing at the install command.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from mcts_ued.partners.parametrize import PartnerSpec


@dataclass
class OvercookedConfig:
    # cramped_room only until the v1 A/B is done so solver behavior is not
    # confounded by layout-distribution shift (docs/decisions.md 2026-05-16)
    layout: str = "cramped_room"
    # _ippo scores these with argmax actions. greedy measured 0.00 on this layout
    # against 35.00 stochastic at 200k env steps, so both sides of the regret
    # come back compressed
    n_eval_episodes: int = 8
    student_hidden: int = 64
    student_n_layers: int = 2
    student_lr: float = 3e-4
    # unit is env steps: 50_000 becomes 48 updates at _ippo's fixed 16 envs x 64 steps
    # both agents take the larger of this and the partner's train_steps, so at the
    # Box maximum of 50_000 the partner's own budget never binds
    student_train_steps: int = 50_000
    # unread: _ippo bounds an episode with env.max_steps and collects in 64-step chunks
    rollout_horizon: int = 400
    # Antagonist recipe: a fixed strong partner used as the PAIRED
    # regret anchor. Values match the upper end of the v1 Box so the
    # antagonist's training budget tracks the largest partner the
    # adversary can propose.
    # only the focal recipe changes between the two roles and the partner spec is
    # the same object, so regret prices the extra focal capacity
    # positive_value_loss floors at zero, so an antagonist weaker than the student
    # hands the adversary a flat signal over the whole box
    antagonist_hidden: int = 256
    antagonist_n_layers: int = 4
    # the one axis held under its Box maximum of 3e-3
    antagonist_lr: float = 1e-3
    antagonist_train_steps: int = 50_000
    # unread: _ippo applies action_temperature to the partner net only, so the focal
    # keeps ActorCritic's default of 1.0
    antagonist_temperature: float = 1.0


class OvercookedIPPOStudent:
    """Student on overcooked v2 (JaxMARL).

    agent_0 is the focal seat inside `_ippo` and agent_1 is the partner, so
    both roles here train the same seat under a different recipe.
    """

    def __init__(self, cfg: OvercookedConfig):
        self.cfg = cfg
        self._jax_ctx: dict[str, Any] | None = None

    @classmethod
    def from_yaml(cls, raw: dict[str, Any]) -> OvercookedIPPOStudent:
        # kind is the dispatch key in train.build_student, so it never reaches
        # OvercookedConfig
        return cls(OvercookedConfig(**{k: v for k, v in raw.items() if k != "kind"}))

    def _ensure_jax(self) -> dict[str, Any]:
        if self._jax_ctx is not None:
            return self._jax_ctx
        try:
            import jax  # type: ignore[import-not-found]
            import jaxmarl  # type: ignore[import-not-found]
        except ImportError as e:
            # these two stand in for the whole [jax] extra since _ippo's flax
            # and optax arrive from the same install
            raise RuntimeError(
                "OvercookedIPPOStudent needs the [jax] extras: "
                "uv pip install -e '.[jax]' (or pip install -e '.[jax]')."
            ) from e
        env = jaxmarl.make("overcooked_v2", layout_name=self.cfg.layout)
        # one env instance serves every train_step and antagonist_value call: _run_ippo
        # vmaps reset and step and holds no state on it
        # cfg.layout is read once here, so a later change to it never reaches the env
        self._jax_ctx = {"jax": jax, "jaxmarl": jaxmarl, "env": env}
        return self._jax_ctx

    def train_step(self, partner: PartnerSpec, *, seed: int) -> float:
        ctx = self._ensure_jax()
        # local import: _ippo pulls flax and optax at module scope and tests/test_env.py
        # asserts this module imports without them
        from mcts_ued.env._ippo import train_one_ippo
        # each call trains a fresh pair since _run_ippo initializes both nets inside itself
        # nothing accumulates across outer steps, so this is per-proposal value
        return train_one_ippo(ctx, partner, self.cfg, seed=seed, role="student")

    def antagonist_value(self, partner: PartnerSpec, *, seed: int) -> float:
        ctx = self._ensure_jax()
        from mcts_ued.env._ippo import train_one_ippo
        # paired.py draws a separate seed for this call, so the anchor shares no env
        # resets or minibatch order with the student run
        return train_one_ippo(ctx, partner, self.cfg, seed=seed, role="antagonist")
