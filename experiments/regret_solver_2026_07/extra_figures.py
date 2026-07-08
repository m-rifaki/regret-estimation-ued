"""Six extra, genuinely-relevant figures for the regret-solver result.

  1 fig_environment   : the maze, its levels, and the exact value landscape a solver must estimate.
  2 fig_generators    : the actual UED output, the level distribution each solver's regret produces vs the true minimax.
  3 fig_mechanism     : WHY guided MCTS wins, its learned value vs what PPO actually visits, on a hard level.
  4 fig_hard_level    : sample efficiency on the hardest level, ceiling estimate vs compute.
  5 fig_speedup       : compute needed to reach a target regret accuracy, and the guided speedup.
  6 fig_robustness    : the ordering holds across every maze, not just on average.

  fig_speedup is the only one that reads mcts_vs_ppo_regret.json, so main() in
  mcts_vs_ppo_regret has to have run first. The other five recompute their own
  solver runs from scratch.

  The directory in the line below no longer exists. Outputs resolve from
  __file__, so running the file from its own directory is enough.

  cd /Users/mrifaki/Projects/regret-estimation-ued-work && \
    uv run --with numpy,scipy,matplotlib,seaborn python extra_figures.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

import mcts_vs_ppo_regret as X

OUT = Path(__file__).parent
GAMMA, HORIZON = X.GAMMA, X.HORIZON
COL = {"ppo": "#d1682f", "mcts": "#8b909b", "mcts_g": "#1f7a4d", "oracle": "#3a5bbf"}
LAB = {"ppo": "PPO", "mcts": "vanilla MCTS", "mcts_g": "guided MCTS", "oracle": "oracle"}
sns.set_theme(style="whitegrid", context="paper", font_scale=1.35,
              rc={"axes.edgecolor": "#c4c8d0", "grid.color": "#edeff3", "font.family": "sans-serif"})


def grid_of(M, values, fill=np.nan):
    """Map per-state values onto the HxW grid; walls = fill.

    nan by default because imshow leaves nan cells out of its autoscale, so a
    wall cannot stretch the color range of a value map.
    """
    g = np.full((M["H"], M["W"]), fill, float)
    for s, (r, c) in enumerate(M["coords"]):
        g[r, c] = values[s]
    return g


# --- trace variants that expose the internal state (for the mechanism figure) ---
def guided_V(M, goal, budget, seed):
    """solve_mcts_guided with the whole V table returned in place of V[s0].

    Held in step with that solver by hand. If the search rule there moves, the
    mechanism panel stops showing the solver that produced the numbers.
    """
    # rng is never drawn from here or in solve_mcts_guided
    # the guided search takes no random draw at all, so a seed average over it is an average of one number
    rng = np.random.default_rng(seed); trans, S = M["trans"], M["S"]
    V = np.zeros(S); Nsa = np.zeros((S, 4)); steps = 0
    def qvals(s):
        # all four model queries are charged to the budget, so a backup costs what an expansion costs
        nonlocal steps
        q = np.empty(4)
        for a in range(4):
            ns = trans[s, a]; steps += 1
            q[a] = 1.0 if ns == goal else GAMMA * V[ns]
        return q
    while steps < budget:
        s, visited = M["s0"], []
        for _ in range(HORIZON):
            q = qvals(s); tot = Nsa[s].sum()
            # the +1 terms keep the first visit to a state out of log(0) and a zero divide
            a = int(np.argmax(q + 1.0 * np.sqrt(np.log(tot + 1) / (Nsa[s] + 1))))
            Nsa[s, a] += 1; visited.append(s)
            if trans[s, a] == goal:
                break
            s = trans[s, a]
        # backward over the trajectory so one episode carries value from the goal end to s0
        # a forward sweep would need one episode per step of goal distance
        for ps in reversed(visited):
            V[ps] = qvals(ps).max()
    return V


def ppo_visits(M, goal, budget, seed):
    """The state visitation of solve_ppo with the policy update dropped.

    Dropping it changes nothing on this level. PPO never reaches the goal here,
    so every advantage is zero. The entropy term shifts a state's four logits by
    one amount and leaves the softmax uniform. Measured on maze 0 goal 4 at
    B=3000: the visit distributions of the two loops agree to 0 in TV.
    """
    rng = np.random.default_rng(seed); theta = np.zeros((M["S"], 4)); visits = np.zeros(M["S"]); steps = 0
    while steps < budget:
        s = M["s0"]
        for _ in range(HORIZON):
            p = np.exp(theta[s] - theta[s].max()); p /= p.sum()
            a = rng.choice(4, p=p); visits[s] += 1; steps += 1
            if trans_step(M, s, a) == goal:
                break
            s = trans_step(M, s, a)
    return visits


def trans_step(M, s, a):
    return M["trans"][s, a]


# ============================================================ 1. environment
def fig_environment():
    M = X.build_maze(0)
    gmid = M["goals"][3]                      # a far goal (distance 8) for the value map
    V = X.value_iteration(M, gmid)
    fig, ax = plt.subplots(1, 2, figsize=(11.5, 5.2))

    # (a) layout
    wall = M["wall"].astype(float)
    # vmax above 1 holds the walls at mid gray instead of black
    # the black d= labels are drawn half a cell up and can land on a wall
    ax[0].imshow(wall, cmap="Greys", vmin=0, vmax=1.6)
    sr, sc = M["coords"][M["s0"]]
    ax[0].scatter([sc], [sr], marker="s", s=170, color="#111", zorder=5, label="start")
    pal = sns.color_palette("flare", len(M["goals"]))
    for j, g in enumerate(M["goals"]):
        r, c = M["coords"][g]
        ax[0].scatter([c], [r], s=150, color=pal[j], edgecolor="k", lw=0.8, zorder=6)
        ax[0].text(c, r - 0.5, "d=%d" % M["dist"][g], ha="center", va="bottom", fontsize=9.5, fontweight="bold")
    ax[0].set_title("(a)  The environment: one maze, five levels\n(a level = a goal at distance 2..10)",
                    loc="left", fontsize=12.5, fontweight="bold")
    ax[0].set_xticks([]); ax[0].set_yticks([]); ax[0].legend(loc="lower right", frameon=True, fontsize=10)

    # (b) exact value landscape for the far goal
    # value_iteration pins V*(goal) to 0, so the star marks the darkest cell of the map
    hm = grid_of(M, V)
    im = ax[1].imshow(hm, cmap="viridis")
    r, c = M["coords"][gmid]; ax[1].scatter([c], [r], marker="*", s=280, color="w", edgecolor="k", lw=1, zorder=6)
    ax[1].scatter([sc], [sr], marker="s", s=120, color="w", edgecolor="k", zorder=6)
    ax[1].set_title("(b)  The exact value V*(s) for the far goal\n(this is what a solver must estimate)",
                    loc="left", fontsize=12.5, fontweight="bold")
    ax[1].set_xticks([]); ax[1].set_yticks([])
    fig.colorbar(im, ax=ax[1], shrink=0.8, label="V*(s)")
    fig.suptitle("The far levels have a long, thin value gradient, hard to find by random exploration",
                 fontsize=13.5, weight="bold", y=1.02)
    fig.tight_layout(); save(fig, "fig_environment")


# ============================================================ 2. the UED output
def fig_generators():
    # a maze whose minimax needs a far level, and a budget where guided has
    # converged but the baselines have not (so the contrast is real).
    # seeds 0 and 2 put the whole minimax on the two nearest levels where every solver agrees, so seed 3 is the first usable maze
    # if none of the 20 qualify M keeps whatever seed 19 gave and the panel is no longer the intended comparison
    for ms in range(20):
        M = X.build_maze(ms); ceiling, v, p_star, val = X.maze_oracle(M)
        if p_star[3] > 0.1 or p_star[4] > 0.1:
            break
    B = 8000; dist = [int(M["dist"][g]) for g in M["goals"]]
    gens = {"oracle": p_star}
    for soft, fn in X.SOLVERS:
        # the seed average goes in before the LP, so this is one curriculum per solver instead of the mean of three
        # sd * 100 + j gives every level its own seed and keeps the three repeats disjoint
        chat = np.array([np.mean([fn(M, g, B, sd * 100 + j)[0] for sd in range(3)]) for j, g in enumerate(M["goals"])])
        # clipped at 0 because an estimated ceiling under the student value would hand the LP a negative regret entry
        gens[soft] = X.minimax_regret(np.clip(chat[None, :] - v, 0, None))[0]
    fig, ax = plt.subplots(figsize=(10, 5.4))
    parts = np.arange(len(dist)); w = 0.2
    for i, key in enumerate(["oracle", "ppo", "mcts", "mcts_g"]):
        ax.bar(parts + (i - 1.5) * w, gens[key], w, label=LAB[key], color=COL[key],
               edgecolor="white", lw=0.5)
    ax.set_xticks(parts); ax.set_xticklabels(["level %d\nd=%d" % (j, d) for j, d in enumerate(dist)])
    ax.set_ylabel("generator probability over levels"); ax.set_ylim(0, max(p_star) * 1.35)
    ax.set_title("The actual UED curriculum each solver produces (B=%d)\nonly guided MCTS recovers the true minimax over levels" % B,
                 loc="left", fontsize=13, fontweight="bold")
    ax.legend(frameon=False, fontsize=10.5, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.0))
    ax.text(0.015, 0.97, "TV to the true minimax:\nguided %.2f   PPO %.2f   vanilla %.2f"
            % (X.tv(gens["mcts_g"], p_star), X.tv(gens["ppo"], p_star), X.tv(gens["mcts"], p_star)),
            transform=ax.transAxes, ha="left", va="top", fontsize=10.5,
            bbox=dict(boxstyle="round", fc="white", ec="0.7"))
    fig.tight_layout(); save(fig, "fig_generators")


# ============================================================ 3. mechanism
def fig_mechanism():
    M = X.build_maze(0); g = M["goals"][4]          # hardest level, distance 10
    B = 3000
    Vg = guided_V(M, g, B, 1)
    vis = ppo_visits(M, g, B, 1)
    Vstar = X.value_iteration(M, g)
    fig, ax = plt.subplots(1, 3, figsize=(15.5, 5.0))
    gr, gc = M["coords"][g]; sr, sc = M["coords"][M["s0"]]
    def deco(a):
        a.scatter([gc], [gr], marker="*", s=240, color="w", edgecolor="k", lw=1, zorder=6)
        a.scatter([sc], [sr], marker="s", s=110, color="w", edgecolor="k", zorder=6)
        a.set_xticks([]); a.set_yticks([])
    # (a) is pinned to the oracle range, since an autoscaled partial value map reads as converged beside (c)
    im0 = ax[0].imshow(grid_of(M, Vg), cmap="viridis", vmin=0, vmax=Vstar.max()); deco(ax[0])
    ax[0].set_title("(a)  guided MCTS: learned value\nafter %d steps" % B, loc="left", fontsize=12, fontweight="bold")
    fig.colorbar(im0, ax=ax[0], shrink=0.72)
    # log visits because 70% of the walk's steps stay within distance 4 of the start
    vv = np.log1p(vis); im1 = ax[1].imshow(grid_of(M, vv, fill=np.nan), cmap="rocket_r"); deco(ax[1])
    ax[1].set_title("(b)  PPO: states actually visited\n(it never reaches the goal)", loc="left", fontsize=12, fontweight="bold")
    fig.colorbar(im1, ax=ax[1], shrink=0.72, label="log visits")
    im2 = ax[2].imshow(grid_of(M, Vstar), cmap="viridis"); deco(ax[2])
    ax[2].set_title("(c)  oracle value V*(s)\n(the target)", loc="left", fontsize=12, fontweight="bold")
    fig.colorbar(im2, ax=ax[2], shrink=0.72)
    fig.suptitle("Why guided MCTS wins: it plans, so it reconstructs the value landscape from the goal; PPO only sees where it randomly wandered",
                 fontsize=13, weight="bold", y=1.03)
    fig.tight_layout(); save(fig, "fig_mechanism")


# ============================================================ 4. hardest level sample efficiency
def fig_hard_level():
    M = X.build_maze(0); g = M["goals"][4]; Vstar = X.value_iteration(M, g)[M["s0"]]
    budgets = [500, 1000, 2000, 4000, 8000, 16000, 32000]
    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    # every readout is a lower bound on V*, so the three curves can only approach the dashed line from below
    # PPO and vanilla MCTS report an achieved rollout return and the guided backups climb from V=0
    for soft, fn in X.SOLVERS:
        m = [np.mean([fn(M, g, B, s)[0] for s in range(5)]) for B in budgets]
        ax.plot(budgets, m, "-o", ms=6, lw=2.6, color=COL[soft], label=LAB[soft])
    ax.axhline(Vstar, color="#3a5bbf", ls="--", lw=1.5, label="true ceiling V*")
    ax.set_xscale("log"); ax.set_xlabel("solver compute (env / model steps)")
    ax.set_ylabel("estimated ceiling for the far level (d=10)")
    ax.set_title("On the hardest level, only guided MCTS finds the\nbest response quickly; the baselines lag or fail",
                 loc="left", fontsize=13, fontweight="bold")
    ax.legend(frameon=False, fontsize=11)
    fig.tight_layout(); save(fig, "fig_hard_level")


# ============================================================ 5. speedup to a target accuracy
def fig_speedup():
    d = json.loads((OUT / "mcts_vs_ppo_regret.json").read_text())
    B = np.array(d["budgets"], float)
    thr = [0.30, 0.20, 0.10, 0.05]
    def budget_to(soft, t):
        # cerr is the mean and std pair main() wrote over mazes x seeds, so [0] is the mean
        m = np.array([r["cerr"][0] for r in d["rows"][soft]])
        # first crossing, which agrees with the last one only while the curve is monotone
        # it is monotone for all three solvers in the committed json
        below = np.where(m <= t)[0]
        # nan for a threshold never reached, which drops the bar and skips the speedup below
        return B[below[0]] if len(below) else np.nan
    fig, ax = plt.subplots(1, 2, figsize=(13, 5.0))
    x = np.arange(len(thr)); w = 0.26
    for i, soft in enumerate(["mcts", "ppo", "mcts_g"]):
        vals = [budget_to(soft, t) for t in thr]
        ax[0].bar(x + (i - 1) * w, vals, w, label=LAB[soft], color=COL[soft])
    ax[0].set_xticks(x); ax[0].set_xticklabels(["<=%.2f" % t for t in thr])
    ax[0].set_yscale("log"); ax[0].set_xlabel("target regret estimate error")
    ax[0].set_ylabel("compute needed (log)")
    ax[0].set_title("(a)  Compute to reach a target regret accuracy", loc="left", fontsize=12.5, fontweight="bold")
    ax[0].legend(frameon=False, fontsize=10.5)
    # the budget grid doubles, so every ratio here is a power of two and the 4x is the resolution of that grid
    sp = [budget_to("ppo", t) / budget_to("mcts_g", t) for t in thr]
    ax[1].plot(x, sp, "-o", ms=9, lw=2.6, color="#1f7a4d")
    for xi, s in zip(x, sp):
        if np.isfinite(s):
            ax[1].text(xi, s + 0.3, "%.0fx" % s, ha="center", fontweight="bold", fontsize=12, color="#1f7a4d")
    ax[1].set_xticks(x); ax[1].set_xticklabels(["<=%.2f" % t for t in thr])
    # the + [4] keeps max() alive and the axis open when no threshold is reached and every speedup is nan
    ax[1].axhline(1, color="#bbb", ls=":", lw=1); ax[1].set_ylim(0, max([s for s in sp if np.isfinite(s)] + [4]) + 1.5)
    ax[1].set_xlabel("target regret estimate error"); ax[1].set_ylabel("guided-MCTS speedup over PPO")
    ax[1].set_title("(b)  Guided MCTS needs several times less compute\nas the accuracy bar rises", loc="left", fontsize=12.5, fontweight="bold")
    fig.tight_layout(); save(fig, "fig_speedup")


# ============================================================ 6. robustness across mazes
def fig_robustness():
    budgets = [1000, 2000, 4000, 8000, 16000, 32000]
    mazes, ms = [], 0
    # same val > 0.1 filter and seed order as main(), so maze k here is maze k of the headline figure
    # seed 1 drops out with a degenerate minimax, so a label counts position in this list instead of the seed
    while len(mazes) < 4:
        M = X.build_maze(ms); ms += 1
        ceiling, v, p_star, val = X.maze_oracle(M)
        if val > 0.1:
            mazes.append((M, ceiling, v, p_star))
    conv = {s: [] for s, _ in X.SOLVERS}
    for (M, ceiling, v, p_star) in mazes:
        for soft, fn in X.SOLVERS:
            b = None
            for B in budgets:
                tvs = []
                for sd in range(3):
                    chat = np.array([fn(M, g, B, sd * 100 + j)[0] for j, g in enumerate(M["goals"])])
                    tvs.append(X.tv(X.minimax_regret(np.clip(chat[None, :] - v, 0, None))[0], p_star))
                # 0.10 is the TV threshold main() counts its reached-the-minimax fraction against
                if np.mean(tvs) <= 0.10:
                    b = B; break
            # a solver that never gets there is parked at twice the largest budget, off the measured grid
            conv[soft].append(b if b else budgets[-1] * 2)
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    x = np.arange(len(mazes)); w = 0.26
    for i, soft in enumerate(["mcts", "ppo", "mcts_g"]):
        ax.bar(x + (i - 1) * w, conv[soft], w, label=LAB[soft], color=COL[soft])
    ax.set_xticks(x); ax.set_xticklabels(["maze %d" % (i + 1) for i in range(len(mazes))])
    ax.set_yscale("log"); ax.set_ylabel("compute to reach the minimax (log)")
    ax.set_title("Guided MCTS reaches the minimax with the least\ncompute on every maze, not just on average",
                 loc="left", fontsize=13, fontweight="bold")
    ax.legend(frameon=False, fontsize=11)
    fig.tight_layout(); save(fig, "fig_robustness")


def save(fig, name):
    fig.savefig(OUT / (name + ".png"), dpi=210, bbox_inches="tight", facecolor="white")
    plt.close(fig); print("wrote", name + ".png")


if __name__ == "__main__":
    fig_environment()
    fig_generators()
    fig_mechanism()
    fig_hard_level()
    fig_speedup()
    fig_robustness()
