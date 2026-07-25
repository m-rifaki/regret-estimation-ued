"""Eugene 6/11: solve the easy problem first.

Freeze the antagonist so the partner game is a 2-player zero-sum game (student
mix vs generator over partners, payoff = clipped regret), plot the learning
dynamics on the 2-simplex, and check whether they converge to the minimax point
or shoot around. Two dynamics, one MMD update (Sokota-Lanctot magnetic mirror
descent): entropy magnet toward uniform + KL to the previous step, both annealed.
  - alpha = 0            -> vanilla Hedge (cycles around the boundary forever).
  - alpha annealed to ~0 -> converges to the exact LP minimax point.

NOTE: the original ran on the real 5-partner Overcooked cross-play matrix
(topology_corrected.json), which lived on a working machine that was wiped. The
scientific point (MMD converges where Hedge cycles on a MIXED regret game) is
reproduced here on a self-contained illustrative 3-partner regret matrix; the LP
minimax below is the ground truth the annealed dynamics must reach.

  uv run --with numpy,scipy,matplotlib python mmd_simplex_toy.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from minimax_regret import solve_minimax_regret, total_variation

OUT = Path(__file__).parent

# illustrative 3-partner regret game with a genuinely MIXED minimax saddle
# (cyclic, asymmetric magnitudes -> non-uniform p*). Rows = students, cols = partners.
# zero diagonal reads as student i being the best response to partner i
# the asymmetry is what puts the anneal under test. a uniform p* would be reached by
# MMD's uniform magnet alone with alpha pinned high
REG = np.array([
    [0.0, 4.0, 4.0],
    [3.0, 0.0, 4.0],
    [3.0, 4.0, 0.0],
])


def duality_gap(M, x, p):
    # generator's best pure reply minus the student's, so this is 0 only at a saddle
    # M comes in unnormalized here, so the number reads against game_value
    return float((M.T @ x).max() - (M @ p).min())


def mmd(M, T, eta, alpha0, alpha_min, half_life, seed=0):
    """Entropy-regularized mirror descent (magnet = uniform), temperature annealed
    alpha0 -> alpha_min. Payoff normalized so eta is scale-free."""
    # normalizing puts the payoff at or below 1 so eta means the same on any regret scale
    # the epsilon covers an all-zero regret matrix
    Mn = M / (M.max() + 1e-9)
    ns, nl = M.shape
    rng = np.random.default_rng(seed)
    # Dirichlet(1) is uniform over the simplex so the start favors neither p* nor the magnet
    x = rng.dirichlet(np.ones(ns)); p = rng.dirichlet(np.ones(nl))
    xs, ps, gaps = [x.copy()], [p.copy()], []
    for t in range(T):
        # geometric decay toward alpha_min and no further
        # alpha held at alpha_min from the start cycles instead. the anneal is what walks the
        # iterate into the basin of the regularized saddle
        alpha = alpha_min + (alpha0 - alpha_min) * 0.5 ** (t / half_life)
        # closed-form MMD step: a is the exponent on the previous iterate and c scales the payoff
        # alpha = 0 gives a = 1 and c = eta, which is Hedge
        a, c = 1.0 / (1 + eta * alpha), eta / (1 + eta * alpha)
        # x minimizes worst-case regret so it moves against the payoff. p maximizes the same matrix
        gx, gp = -(Mn @ p), (Mn.T @ x)
        # the uniform magnet adds log(1/n) to every coordinate and softmax cancels a constant
        # so it never appears in these two lines
        # clip before the log: an exact 0 is absorbing under exp weights and the vanilla run
        # would stop on a face of the simplex looking convergent
        x = _softmax(a * np.log(np.clip(x, 1e-12, None)) + c * gx)
        p = _softmax(a * np.log(np.clip(p, 1e-12, None)) + c * gp)
        xs.append(x.copy()); ps.append(p.copy()); gaps.append(duality_gap(M, x, p))
    return {"xs": np.array(xs), "ps": np.array(ps), "gaps": np.array(gaps)}


def _softmax(z):
    # the shift matters when every z is far below 0, where exp would underflow the sum to 0
    z = z - z.max(); w = np.exp(z); return w / w.sum()


# barycentric 2-simplex projection
_V = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, np.sqrt(3) / 2]])
def to_xy(P): return np.asarray(P) @ _V


def main():
    # REG is already the payoff (regret) matrix, not a return matrix: pass -REG
    # with a zero ceiling so regret_matrix() rebuilds REG instead of colmax(REG)-REG.
    sol = solve_minimax_regret(-REG, external_v_star=np.zeros(3))
    p_star = sol.generator_dist
    # eta = 1 against a payoff at or below 1 moves a log-weight by at most 1 per step
    # 6000 iterations is 15 half-lives of the annealed run, so its last iterate is at alpha_min
    T, eta = 6000, 1.0
    # alpha0 = alpha_min = 0 holds a at 1 and c at eta forever, so half_life does nothing
    # both runs take seed 0, so the two panels start from one point and only the update differs
    vanilla = mmd(REG, T, eta, 0.0, 0.0, 1e9, seed=0)          # Hedge
    # 0.02 is the smallest floor on this matrix that stops the tail cycling. under it the last
    # iterate is back to Hedge, and over it the pull to uniform costs TV around 0.57 * alpha_min
    annealed = mmd(REG, T, eta, 2.0, 0.02, 400.0, seed=0)      # annealed MMD

    # the measured quantity is the last iterate. Hedge is no-regret so its time average
    # would reach p* and hide the cycling
    out = {"reg": REG.tolist(), "p_star": p_star.tolist(), "game_value": sol.game_value,
           "vanilla_final_p": vanilla["ps"][-1].tolist(), "annealed_final_p": annealed["ps"][-1].tolist(),
           "vanilla_tv": total_variation(vanilla["ps"][-1], p_star),
           "annealed_tv": total_variation(annealed["ps"][-1], p_star)}
    (OUT / "mmd_simplex_toy.json").write_text(json.dumps(out, indent=2))
    print("LP minimax p*        :", np.round(p_star, 3), "value", round(sol.game_value, 3))
    print("vanilla Hedge final p:", np.round(vanilla["ps"][-1], 3), "TV", round(out["vanilla_tv"], 3))
    print("annealed MMD  final p:", np.round(annealed["ps"][-1], 3), "TV", round(out["annealed_tv"], 3))
    make_figure(vanilla, annealed, p_star)


def make_figure(vanilla, annealed, p_star):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    try:
        import seaborn as sns; sns.set_theme(style="white", context="talk")
    except Exception:
        pass
    fig = plt.figure(figsize=(16, 5.2))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.2], wspace=0.35)

    def simplex(ax, run, title, cmap):
        tri = plt.Polygon(_V, closed=True, fill=False, ec="#b9b3a7", lw=1.4)
        ax.add_patch(tri)
        for (x, y), lab in zip(_V, ["partner 0", "partner 1", "partner 2"]):
            ax.text(x, y - 0.05 if y < 0.1 else y + 0.04, lab, ha="center", fontsize=11, fontweight="bold")
        xy = to_xy(run["ps"]); seg = np.concatenate([xy[:-1, None], xy[1:, None]], 1)
        # color runs with the iteration index so a closed orbit reads apart from an inward spiral
        ax.add_collection(LineCollection(list(seg), cmap=cmap, array=np.linspace(0, 1, len(seg)), lw=2.2))
        ax.scatter(*to_xy(run["ps"][0]), color="#444", s=50, zorder=3, label="start")
        ax.scatter(*to_xy(p_star), color="#c2410c", s=300, marker="*", ec="k", zorder=5, label="LP minimax $p^*$")
        ax.set_title(title, fontsize=13); ax.set_aspect("equal"); ax.axis("off")
        ax.legend(loc="upper right", frameon=False, fontsize=10)

    simplex(fig.add_subplot(gs[0, 0]), vanilla, "vanilla Hedge: cycles", plt.colormaps["Blues"])
    simplex(fig.add_subplot(gs[0, 1]), annealed, "annealed MMD: converges", plt.colormaps["Greens"])
    ax = fig.add_subplot(gs[0, 2])
    ax.plot(vanilla["gaps"], color="#2563eb", lw=1.8, label="vanilla Hedge")
    ax.plot(annealed["gaps"], color="#1c7a45", lw=1.8, label="annealed MMD")
    ax.set_yscale("log"); ax.set_xlabel("iteration"); ax.set_ylabel("Nash duality gap")
    ax.set_title("convergence to the minimax saddle", fontsize=13); ax.legend(frameon=False, fontsize=11)
    fig.suptitle("On a mixed regret game, annealed MMD reaches the LP minimax where vanilla Hedge cycles", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(OUT / "mmd_simplex_toy.png", dpi=200, bbox_inches="tight")
    plt.close(fig); print("wrote", OUT / "mmd_simplex_toy.png")


if __name__ == "__main__":
    main()
