"""Integration test: the PAIRED outer loop with both solvers + MockStudent.

Everything here scores specs with MockStudent, which reads `params` and
never opens `decode_kwargs`. The loop and both solvers are covered.
Nothing in this file reaches the decode map in parametrize.py or the
OvercookedIPPOStudent path that a real run takes.
"""
from __future__ import annotations

import numpy as np

from mcts_ued.paired import PairedConfig, train_paired
from mcts_ued.partners.parametrize import BoxPartnerSpace
from mcts_ued.solvers.mcts import MCTSAdversary, MCTSConfig
from mcts_ued.solvers.ppo import PPOAdversary, PPOAdversaryConfig
from mcts_ued.student import MockStudent


def _student(peak):
    # the same five values as the mock_student fixture in conftest.py, which no
    # test requests. only the peak is parametrized here
    return MockStudent(
        peak=np.asarray(peak, dtype=np.float32),
        # a uniform draw from the 5-D box scores 6e-5 of the peak at this width,
        # so the two loop tests below run on an all but flat regret signal
        width=0.15,
        # antagonist above student holds the regret at 0.45 of the bump, so
        # positive_value_loss_regret never reaches its clip in these tests
        student_strength=0.55,
        antagonist_strength=1.0,
        # overrides the 0.02 dataclass default. with the noise term gone both
        # returns are functions of params alone and the rollout seeds stop
        # changing anything
        noise_std=0.0,
    )


def test_loop_runs_with_mcts(box_space: BoxPartnerSpace):
    peak = [0.7, 0.5, 0.5, 0.5, 0.5]
    student = _student(peak)
    adv = MCTSAdversary(
        # progressive widening at k=1 and alpha=0.5 gives the root two children
        # over eight simulations, so six of the eight leaf calls rescore a state
        # the tree already holds
        cfg=MCTSConfig(n_simulations=8, seed=0),
        space=box_space,
        # the leaf value is the student's return. MCTSAdversary.update is a no-op
        # so the regret train_paired computes never enters the tree. under
        # MockStudent the return and the regret are positive multiples of one
        # bump and rank specs the same way, which a real student would not do
        leaf_value_fn=lambda s: float(student.train_step(s, seed=0)),
    )
    # train_paired is a generator and nothing has called the student yet
    steps = list(train_paired(adv, student, PairedConfig(n_outer_steps=10, seed=0)))
    assert len(steps) == 10
    # the max() inside positive_value_loss_regret makes this true for any pair of
    # returns. what it would catch is train_paired defaulting to signed_regret
    assert all(s.regret >= 0.0 for s in steps)


def test_loop_runs_with_ppo(box_space: BoxPartnerSpace):
    peak = [0.7, 0.5, 0.5, 0.5, 0.5]
    student = _student(peak)
    adv = PPOAdversary(PPOAdversaryConfig(seed=0), box_space)
    # nine of the ten regrets come back under 1e-5 against a box maximum of 0.45
    # and mu ends 6e-5 from its 0.5 start, so what runs here is an exception
    # check on the PPO arm
    steps = list(train_paired(adv, student, PairedConfig(n_outer_steps=10, seed=0)))
    assert len(steps) == 10
    assert all(s.regret >= 0.0 for s in steps)


def test_ppo_outer_loop_concentrates_on_peak(box_space: BoxPartnerSpace):
    """After many outer steps, PPO's mean partner-spec drifts toward the peak.

    Only the mean moves. sigma ends near 0.59 against an init of 0.61, so the
    proposal distribution never concentrates and mu[0] finishes at 0.998 pinned
    on the box face past the peak at 0.85.
    """
    peak = np.array([0.85, 0.5, 0.5, 0.5, 0.5], dtype=np.float32)
    student = MockStudent(
        peak=peak,
        # wider than the helper's 0.15, where the median uniform draw scores
        # 6e-5 of the peak and a cold mu at 0.5 has no first-order signal
        width=0.25,
        # the regret is 0.6 of the bump here against the helper's 0.45, which
        # scales up the advantage every REINFORCE step multiplies through
        student_strength=0.4,
        antagonist_strength=1.0,
        noise_std=0.0,
    )
    # six times the default lr_mu. the default reaches 0.81 at this seed and
    # clears the assertion too, so the raise only buys speed
    adv = PPOAdversary(PPOAdversaryConfig(seed=0, lr_mu=0.3), box_space)
    list(train_paired(adv, student, PairedConfig(n_outer_steps=800, seed=0)))
    # a drift of 0.1 past the 0.5 start and a long way short of the peak. axes 1
    # through 4 begin on the peak's own 0.5 and carry no gradient at all
    assert float(adv.mu[0]) > 0.6


def test_swap_solvers_one_line_difference(box_space: BoxPartnerSpace):
    """Same student, same config, different solver - both produce 5 steps cleanly.

    The two step lists are never compared to each other. A params round-trip
    broken between propose and update would stay invisible here, because
    PPOAdversary.update returns silently on a key it cannot find.
    """
    # the peak is the box center, which is where PPO's mu starts and where the
    # MCTS root state is. max_depth 1 keeps the root itself from being decoded
    student = _student([0.5] * 5)

    def leaf_value(spec):
        return float(student.train_step(spec, seed=0))

    # four simulations give the root one child, since the widening cap is still
    # int(3**0.5) at the last chance to add a second. this arm proposes one
    # uniform draw per outer step and is domain randomization
    mcts = MCTSAdversary(MCTSConfig(n_simulations=4, seed=0), box_space, leaf_value)
    ppo = PPOAdversary(PPOAdversaryConfig(seed=0), box_space)
    n = 5
    s_mcts = list(train_paired(mcts, student, PairedConfig(n_outer_steps=n, seed=0)))
    s_ppo = list(train_paired(ppo, student, PairedConfig(n_outer_steps=n, seed=0)))
    # both lengths come off range(n_outer_steps) in train_paired and hold for any
    # adversary whose propose returns a spec
    assert len(s_mcts) == n
    assert len(s_ppo) == n
