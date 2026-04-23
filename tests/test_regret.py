"""Unit tests for regret estimators.

Every call below names its arguments, so nothing here pins the order of the
two parameters. Swapping them in `regret.py` leaves this file green while
`train_paired` calls the estimator positionally and inverts the sign.
`test_ppo_outer_loop_concentrates_on_peak` is what catches that.

`MaxMCRegret` has no other coverage. The v1 loop hardwires
`positive_value_loss_regret`, so these assertions are the whole contract the
replay-buffer estimator carries until v2 wires it in.
"""
from __future__ import annotations

from mcts_ued.regret import MaxMCRegret, positive_value_loss_regret, signed_regret


def test_positive_value_loss_floor_at_zero():
    # student above antagonist. the same pair reaches signed_regret below, so
    # the two estimators are told apart only on inputs of this sign
    assert positive_value_loss_regret(student_value=1.0, antagonist_value=0.5) == 0.0


def test_positive_value_loss_positive_gap():
    # 0.9 - 0.2 rounds to the double nearest 0.7 so exact equality holds here.
    # 0.3 - 0.1 would not, and the estimator rounds nothing on its own
    assert positive_value_loss_regret(student_value=0.2, antagonist_value=0.9) == 0.7


def test_signed_regret_can_go_negative():
    # nothing downstream consumes a negative regret. the MCTS backup runs UCB1
    # at c = sqrt(2), which was derived for values between zero and one
    assert signed_regret(student_value=1.0, antagonist_value=0.5) == -0.5


def test_maxmc_first_visit_is_zero():
    tracker = MaxMCRegret()
    # this call inserts lvl-a as it reads through the defaultdict, so a tracker
    # that was only ever queried still hands levels to all_levels()
    assert tracker.regret(key="lvl-a", current_value=0.3) == 0.0


def test_maxmc_tracks_ceiling():
    tracker = MaxMCRegret()
    tracker.observe(key="lvl-a", value=0.8)
    # 0.5 after 0.8 is the point of the test: an observe that kept the last
    # value instead of the running max would report 0.1 and a ceiling of 0.5
    tracker.observe(key="lvl-a", value=0.5)
    assert tracker.regret(key="lvl-a", current_value=0.4) == 0.4
    assert tracker.ceiling(key="lvl-a") == 0.8


def test_maxmc_ceiling_zero_for_unseen():
    tracker = MaxMCRegret()
    # without this, a ceiling that forgot the sentinel would hand -inf to a
    # caller and poison every sum it entered. the 0.0 returned instead cannot
    # be told from a level whose best return really was 0.0
    assert tracker.ceiling(key="never") == 0.0


def test_maxmc_levels_collected():
    tracker = MaxMCRegret()
    tracker.observe(key="lvl-a", value=0.5)
    # one value for both levels, so nothing in the value path can turn this
    # test red. comparing as a set leaves the order all_levels() returns
    # unpinned for a later replay sampler to choose
    tracker.observe(key="lvl-b", value=0.5)
    assert set(tracker.all_levels()) == {"lvl-a", "lvl-b"}
