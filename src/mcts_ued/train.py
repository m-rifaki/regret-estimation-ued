"""Train one PAIRED-style run from a yaml config.

usage:
    python -m mcts_ued.train --config experiments/cramped_room_mcts.yaml
    python -m mcts_ued.train --config experiments/cramped_room_ppo.yaml

The two configs differ only in their adversary block. That block is the
whole MCTS-vs-PPO A/B, and cramped_room_domain_rand.yaml runs the null
adversary as a ppo block with both learning rates at zero.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml

from mcts_ued.paired import PairedConfig, train_paired
from mcts_ued.partners.parametrize import BoxConfig, BoxPartnerSpace, PartnerSpace
from mcts_ued.regret import positive_value_loss_regret
from mcts_ued.solvers.base import UEDAdversary
from mcts_ued.solvers.mcts import MCTSAdversary, MCTSConfig
from mcts_ued.solvers.ppo import PPOAdversary, PPOAdversaryConfig
from mcts_ued.student import Student, build_mock_student


def build_space(cfg: dict[str, Any]) -> PartnerSpace:
    kind = cfg.get("kind", "box")
    if kind == "box":
        return BoxPartnerSpace(BoxConfig(**cfg.get("box", {})))
    raise ValueError(f"unknown partner-space kind: {kind}")


def build_student(cfg: dict[str, Any]) -> Student:
    kind = cfg.get("kind", "overcooked_ippo")
    if kind == "overcooked_ippo":
        # OvercookedIPPOStudent defers its jax import to the first train_step, so
        # a missing [jax] extra raises after the outer loop has already started
        from mcts_ued.env.overcooked import OvercookedIPPOStudent
        return OvercookedIPPOStudent.from_yaml(cfg)
    if kind == "mock":
        return build_mock_student(cfg)
    raise ValueError(f"unknown student kind: {kind}")


def build_adversary(
    cfg: dict[str, Any], space: PartnerSpace, student: Student
) -> UEDAdversary:
    """Build the adversary named by the config block.

    Both kinds take the student because MCTS evaluates a leaf by training
    it, and PPO learns from the regret handed back through `update`. The
    two arms are not compute matched: an MCTS propose costs
    `n_simulations` extra student trainings per outer step and a PPO
    propose costs none.
    """
    kind = cfg["kind"]
    if kind == "mcts":
        # leaf value is the student return. the regret estimator runs in the
        # outer loop and never reaches the tree
        def leaf_value_fn(spec):
            # seed 0 on every leaf so sibling values differ only by their partner
            # spec. train_step trains a fresh IPPO pair per call, so these
            # evaluations leave no state on the student the outer loop measures
            return float(student.train_step(spec, seed=0))

        return MCTSAdversary(
            cfg=MCTSConfig(**cfg.get("mcts", {})),
            space=space,
            leaf_value_fn=leaf_value_fn,
        )
    if kind == "ppo":
        return PPOAdversary(
            cfg=PPOAdversaryConfig(**cfg.get("ppo", {})),
            space=space,
        )
    raise ValueError(f"unknown adversary kind: {kind}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True,
                   help="path to a yaml config under experiments/")
    p.add_argument("--out", type=Path, default=Path("results/run.jsonl"),
                   help="output jsonl with per-step records")
    args = p.parse_args(argv)

    cfg = yaml.safe_load(args.config.read_text())
    space = build_space(cfg["partner_space"])
    student = build_student(cfg["student"])
    adv = build_adversary(cfg["adversary"], space=space, student=student)
    paired_cfg = PairedConfig(**cfg.get("paired", {}))

    # results/ is gitignored so the default --out has no parent on a fresh clone
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        # train_paired yields one record per outer step, so this handle stays
        # open for the length of the run
        for step in train_paired(adv, student, paired_cfg, positive_value_loss_regret):
            f.write(json.dumps({
                "step": step.step,
                "regret": step.regret,
                "student_value": step.student_value,
                "antagonist_value": step.antagonist_value,
                # decoded kwargs put the record in natural units. rounding
                # inside BoxPartnerSpace.decode makes spec.params unrecoverable
                # from this field
                "spec": step.spec.decode_kwargs,
            }) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
