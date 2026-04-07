"""Regret estimators for UED.

`positive_value_loss_regret` is the classic PAIRED estimator
(Dennis et al. 2020): max(0, V_antagonist - V_student).

`MaxMCRegret` is the max-Monte-Carlo regret used in ACCEL-style
replay buffers (Parker-Holder et al. 2022; refined in
Rutherford et al. 2024, arxiv 2402.12284). It tracks the running max
return per level and reports (max_seen - current) on each visit, so
levels whose ceilings have already been hit produce zero regret.

`signed_regret` is the same difference with the clip removed. An
antagonist weaker than the student sends it negative on the levels the
student has mastered, which the clipped form reports as zero.

Section 3 of the report carries these as eq:pvl and eq:maxmc.
"""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Hashable

# argument order is (student, antagonist) and train_paired calls it positionally
RegretEstimator = Callable[[float, float], float]


def positive_value_loss_regret(student_value: float, antagonist_value: float) -> float:
    """PAIRED's positive value loss: max(0, V_antagonist - V_student).

    Report eq:pvl. The clip erases the sign, so a level the student has
    mastered and a level nobody can solve both report zero.
    """
    return max(antagonist_value - student_value, 0.0)


def signed_regret(student_value: float, antagonist_value: float) -> float:
    """V_antagonist - V_student, no floor. Diagnostic only.

    Kept out of the loop because `MCTSAdversary` backs leaf values up
    through UCB1 at c = sqrt(2), a constant derived for values between
    zero and one.
    """
    return antagonist_value - student_value


class MaxMCRegret:
    """Running max-Monte-Carlo regret estimator.

    `observe(key, value)` updates the per-level ceiling.
    `regret(key, value)` returns max(0, ceiling - value); 0 for unseen.

    Levels are identified by any hashable key (e.g. a tuple of the
    partner-spec params for partner-policy UED, or a layout id for
    scene-level UED).

    The running max is a lower bound on V*. A level whose ceiling has
    never been approached reports near-zero regret and stops being
    sampled, which is what keeps it from being approached.
    """

    def __init__(self) -> None:
        # sentinel is -inf so the first observation becomes the ceiling even
        # on a level whose returns are all negative
        self._max: dict[Hashable, float] = defaultdict(lambda: float("-inf"))

    def observe(self, key: Hashable, value: float) -> None:
        self._max[key] = max(self._max[key], value)

    def regret(self, key: Hashable, current_value: float) -> float:
        # reading through the defaultdict inserts key, so a level that was
        # only ever queried still turns up in all_levels()
        prev = self._max[key]
        if prev == float("-inf"):
            # unseen is zero by contract, and -inf minus a finite value
            # already clips to that same zero in the max below
            return 0.0
        return max(prev - current_value, 0.0)

    def all_levels(self) -> list[Hashable]:
        return list(self._max.keys())

    def ceiling(self, key: Hashable) -> float:
        v = self._max[key]
        # zero here reads the same as a level whose best return really was zero
        # so this cannot be used to test whether a level was ever observed
        return float(v) if v != float("-inf") else 0.0
