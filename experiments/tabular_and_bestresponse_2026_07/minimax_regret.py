"""Tabular minimax-regret oracle for the MCTS-UED formulation.

Eugene's Q3: empirically check whether the UED generator distribution over
partners matches the minimax-regret-optimal one.

Setup. We are handed a cross-play return matrix ``R[i][j]`` = expected return of
student ``i`` paired with partner ``j``. Rows are students, columns are partners.
The matrix is NOT symmetric; we never symmetrize it.

Given a per-partner best-response value ``V*(j)`` (the best return any student in
scope can achieve against partner ``j``), the regret of student ``i`` against
partner ``j`` is  ``Reg[i][j] = max(V*(j) - R[i][j], 0)``.

The minimax-regret student is the row player's optimal mixed strategy in the
zero-sum matrix game with payoff ``Reg``: the student picks a distribution ``x``
over students to minimize the worst-case expected regret; the generator picks a
distribution ``p`` over partners to maximize it. We solve the row player's LP with
``scipy.optimize.linprog`` and recover the generator distribution ``p*`` from the
duals. ``p*`` is the distribution the UED adversary should converge to if the
formulation is solved exactly.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.optimize import linprog


@dataclass
class MinimaxRegretSolution:
    """Output of the LP. ``generator_dist`` is p*, the distribution a UED generator is
    scored against.

    ``game_value`` and ``regret_matrix`` carry the units of ``R``. That is Overcooked
    episode return, with per-partner ceilings from 54 to 120 on the 5-partner run.
    """
    student_mix: np.ndarray
    generator_dist: np.ndarray
    game_value: float
    regret_matrix: np.ndarray
    best_response_value: np.ndarray
    pure_student_idx: int
    pure_student_worst_regret: float
    per_partner_regret_at_mix: np.ndarray
    support_partners: list[int] = field(default_factory=list)

    def as_dict(self) -> dict:
        # dataclasses.asdict hands the arrays back untouched and json.dumps refuses them
        return {
            "student_mix": self.student_mix.tolist(),
            "generator_dist": self.generator_dist.tolist(),
            "game_value": float(self.game_value),
            "regret_matrix": self.regret_matrix.tolist(),
            "best_response_value": self.best_response_value.tolist(),
            "pure_student_idx": int(self.pure_student_idx),
            "pure_student_worst_regret": float(self.pure_student_worst_regret),
            "per_partner_regret_at_mix": self.per_partner_regret_at_mix.tolist(),
            "support_partners": [int(j) for j in self.support_partners],
        }


def best_response_value(R: np.ndarray, external: Optional[np.ndarray] = None) -> np.ndarray:
    """Per-partner best-response value V*(j). Default: column max (best any
    in-scope student gets against partner j). Pass ``external`` to override.

    The column max only ever sees students already in the pool, so it is a lower bound on
    the true ceiling. pipeline_bestresponse trains a fresh protagonist against each frozen
    partner and passes those values in. mmd_simplex_toy passes zeros so that a caller
    already holding a regret matrix can feed it as -R.
    """
    R = np.asarray(R, dtype=float)
    if external is not None:
        v = np.asarray(external, dtype=float)
        if v.shape != (R.shape[1],):
            raise ValueError(f"external V* shape {v.shape} != (n_partners={R.shape[1]},)")
        return v
    return R.max(axis=0)


def regret_matrix(R: np.ndarray, external_v_star: Optional[np.ndarray] = None,
                  clip: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Build Reg[i][j] = V*(j) - R[i][j]. Returns (Reg, V*). clip -> negatives to 0."""
    R = np.asarray(R, dtype=float)
    v_star = best_response_value(R, external_v_star)
    reg = v_star[None, :] - R
    if clip:
        # under the column-max ceiling reg is nonnegative by construction and this is a no-op
        # it earns its place on an external V*: a best-response estimate coming in below the
        # pool max turns the regret negative and a negative regret mis-ranks the partners
        reg = np.clip(reg, 0.0, None)
    return reg, v_star


def solve_minimax_regret(R: np.ndarray, external_v_star: Optional[np.ndarray] = None,
                         pure_score: Optional[np.ndarray] = None, clip: bool = True,
                         support_tol: float = 1e-6) -> MinimaxRegretSolution:
    """Solve the zero-sum minimax-regret game by LP. Variables [x_0..x_{n-1}, v];
    minimize v s.t. Reg^T x <= v for every partner column, sum x = 1, x >= 0.
    p* is read from the duals of the column constraints (fallback: column LP)."""
    R = np.asarray(R, dtype=float)
    n_students, n_partners = R.shape
    reg, v_star = regret_matrix(R, external_v_star, clip=clip)

    n_var = n_students + 1
    c = np.zeros(n_var); c[-1] = 1.0
    A_ub = np.zeros((n_partners, n_var)); A_ub[:, :n_students] = reg.T; A_ub[:, -1] = -1.0
    b_ub = np.zeros(n_partners)
    A_eq = np.zeros((1, n_var)); A_eq[0, :n_students] = 1.0; b_eq = np.array([1.0])
    # v stays free because an unclipped regret game can have a negative value and a
    # (0, None) bound would report 0 for it
    bounds = [(0.0, None)] * n_students + [(None, None)]
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
    if not res.success:
        raise RuntimeError(f"linprog failed: {res.message}")

    # HiGHS meets the simplex equality to its own tolerance and clipping a small negative
    # moves the sum off 1 again. TV and KL downstream want an exact distribution
    x_star = np.clip(np.array(res.x[:n_students], dtype=float), 0.0, None)
    x_star = x_star / x_star.sum()
    game_value = float(res.x[-1])
    p_star = _generator_from_dual(res, reg, support_tol)
    per_partner = reg.T @ x_star

    if pure_score is not None:
        pure_score = np.asarray(pure_score, dtype=float)
    elif n_students == n_partners:
        # a square R means row i and column i came off the same seed, so the diagonal is
        # self-play and its argmax is the student a self-play-only run would hand you
        pure_score = np.diag(R)
    else:
        # no diagonal to read on a rectangular R, so the mean over partners stands in
        pure_score = R.mean(axis=1)
    pure_idx = int(np.argmax(pure_score))
    # the pure student is feasible for the LP above, so game_value can never exceed this
    # number and the two together are the whole claim that mixing over students buys anything
    pure_worst_regret = float(reg[pure_idx, :].max())
    # 1e-6 clears the dust the LP leaves on partners it dropped and stays under the
    # lightest live weight the real runs produced, which was 0.25
    support = [j for j in range(n_partners) if p_star[j] > support_tol]

    return MinimaxRegretSolution(
        student_mix=x_star, generator_dist=p_star, game_value=game_value,
        regret_matrix=reg, best_response_value=v_star, pure_student_idx=pure_idx,
        pure_student_worst_regret=pure_worst_regret, per_partner_regret_at_mix=per_partner,
        support_partners=support)


def _generator_from_dual(res, reg: np.ndarray, support_tol: float) -> np.ndarray:
    """Column player's mix, read off the duals of the ``reg^T x <= v`` rows.

    Stationarity in v forces those multipliers to sum to 1, so the duals already are a
    distribution over partners and the division at the end is tolerance cleanup.
    """
    n_partners = reg.shape[1]
    duals = None
    if getattr(res, "ineqlin", None) is not None:
        # marginals are d(objective)/d(b_ub) and come back <= 0 for a <= row in a minimization
        cand = -np.asarray(res.ineqlin.marginals, dtype=float)
        if cand.shape == (n_partners,) and cand.sum() > support_tol:
            duals = cand
    if duals is None:
        # no usable dual came back, so pay for a second LP and solve the generator side
        duals = _solve_column_player(reg)
    duals = np.clip(duals, 0.0, None)
    total = duals.sum()
    if total <= support_tol:
        # dividing by total here would put NaN into every TV and KL taken against p*
        return np.full(n_partners, 1.0 / n_partners)
    return duals / total


def _solve_column_player(reg: np.ndarray) -> np.ndarray:
    """The generator's own LP: maximize v subject to ``(reg p)_i >= v`` for every student row.

    Reached only when the primal duals come back unusable. At the five partners this ran
    on the second solve costs nothing.
    """
    n_students, n_partners = reg.shape
    n_var = n_partners + 1
    c = np.zeros(n_var); c[-1] = -1.0
    A_ub = np.zeros((n_students, n_var)); A_ub[:, :n_partners] = -reg; A_ub[:, -1] = 1.0
    b_ub = np.zeros(n_students)
    A_eq = np.zeros((1, n_var)); A_eq[0, :n_partners] = 1.0; b_eq = np.array([1.0])
    bounds = [(0.0, None)] * n_partners + [(None, None)]
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
    if not res.success:
        raise RuntimeError(f"column-player linprog failed: {res.message}")
    return np.clip(np.asarray(res.x[:n_partners], dtype=float), 0.0, None)


def kl_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    # q is normally the LP p* and a vertex solution carries exact zeros, so without the
    # floor this is infinite wherever the generator kept mass the LP dropped
    p = np.clip(np.asarray(p, float), eps, None); p = p / p.sum()
    q = np.clip(np.asarray(q, float), eps, None); q = q / q.sum()
    return float(np.sum(p * np.log(p / q)))


def total_variation(p: np.ndarray, q: np.ndarray) -> float:
    # the number every result in the writeup quotes, since it stays finite against a
    # p* that has exact zeros in it
    p = np.asarray(p, float) / np.asarray(p, float).sum()
    q = np.asarray(q, float) / np.asarray(q, float).sum()
    return float(0.5 * np.abs(p - q).sum())


def load_R(topology_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # the topology json is phase B of pipeline_bestresponse. SE is the per-cell eval
    # standard error and no part of the LP reads it, so a regret difference under SE is noise
    data = json.loads(Path(topology_path).read_text())
    return (np.asarray(data["R"], float), np.asarray(data["SE"], float),
            np.asarray(data["selfplay_diag"], float))


def load_ued_generator(comparison_path: Path) -> np.ndarray:
    # the recorded probs are the MMD iterate at the final outer step, so they still carry
    # the residual entropy of the anneal (0.72 nats at the end of the 5-partner run)
    p = np.asarray(json.loads(Path(comparison_path).read_text())["final_generator_probs"], float)
    return p / p.sum()
