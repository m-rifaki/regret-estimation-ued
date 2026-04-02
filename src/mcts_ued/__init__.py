"""MCTS-UED: UED over partner policies in overcooked v2.

Compare MCTS and PPO as solvers of the partner-policy minimax-regret
game; see README.md for the project framing.

Every name below imports with stdlib and numpy alone. The overcooked
IPPO student needs the jax extra before it can train, so it stays
behind `mcts_ued.env.overcooked` and is pulled in where train.py
builds the student from a config.
"""
from mcts_ued.paired import PairedConfig, PairedStep, train_paired
from mcts_ued.partners.parametrize import (
    BoxConfig,
    BoxPartnerSpace,
    PartnerSpace,
    PartnerSpec,
)
from mcts_ued.regret import (
    MaxMCRegret,
    RegretEstimator,
    positive_value_loss_regret,
    signed_regret,
)
from mcts_ued.solvers.base import UEDAdversary
from mcts_ued.solvers.mcts import MCTSAdversary, MCTSConfig
from mcts_ued.solvers.ppo import PPOAdversary, PPOAdversaryConfig
from mcts_ued.student import MockStudent, Student, build_mock_student

# train.py hardwires positive_value_loss_regret into train_paired, so a yaml run
# never reaches signed_regret or MaxMCRegret. MockStudent is the opposite case:
# `kind: mock` selects it from a config, and that path drives the outer loop with
# no jax stack installed.
__all__ = [
    "BoxConfig",
    "BoxPartnerSpace",
    "MaxMCRegret",
    "MCTSAdversary",
    "MCTSConfig",
    "MockStudent",
    "PairedConfig",
    "PairedStep",
    "PartnerSpace",
    "PartnerSpec",
    "PPOAdversary",
    "PPOAdversaryConfig",
    "RegretEstimator",
    "Student",
    "UEDAdversary",
    "build_mock_student",
    "positive_value_loss_regret",
    "signed_regret",
    "train_paired",
]
