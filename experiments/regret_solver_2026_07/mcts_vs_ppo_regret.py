"""Regret-solver comparison for MCTS-UED (Eugene 7/16 task), finished version.

Question: does a better best-response / regret SOLVER give better regret estimates
and thus a UED curriculum that reaches the true minimax? Tested in clean tabular
mazes where the true regret is known exactly (value iteration), so every claim is
checked against an oracle. Robust across several random mazes and seeds.

Three solvers estimate the per-level best-response ceiling V*(level) under a
matched budget B (environment / model queries):
  PPO           tabular REINFORCE (model-free RL).
  vanilla MCTS  UCT with the simulator and random rollouts.
  guided MCTS   UCB-directed trajectories with model-based Bellman backups of a
                learned value (the AlphaZero / MuZero family idea, tabular).

Regret(level) = V*(level) - V_student(level); the minimax-regret generator over
levels is solved by LP (true minimax) and run through annealed MMD (the loop).

The three are not scored the same way. PPO and vanilla MCTS report the return their
greedy policy achieves from s0 while guided MCTS reports its learned root value,
which can only approach V* from below. Budget B counts one unit per environment or
model transition. The guided solver is charged 4 units for every state it expands,
and the compute numbers are quoted after that premium.

Run:
  cd /Users/mrifaki/Projects/regret-estimation-ued-work && \
    uv run --with numpy,scipy,matplotlib,seaborn python mcts_vs_ppo_regret.py
"""
from __future__ import annotations

import json
from collections import deque
from pathlib import Path

import numpy as np
from scipy.optimize import linprog

# reward is 1 only on entering the goal, so V*(level) is exactly GAMMA ** (dist - 1)
# and every oracle number in the output can be checked by hand
GAMMA = 0.97
# six times the farthest target, so a truncated rollout is never what limits a solver
HORIZON = 60
N_LEVELS = 5
TARGETS = [2, 4, 6, 8, 10]           # goal distances from start (easy -> hard)
ACTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
OUT = Path(__file__).parent


# --------------------------------------------------------------- maze (random, solvable)
def build_maze(seed=0, H=9, W=9, wall_p=0.22):
    """Random maze plus one goal cell per entry of TARGETS.

    Redraws until every target distance is matched to within 2 steps. Level j then
    means the same difficulty in every maze, which is what lets panel (c) pool the
    per-level error across mazes."""
    rng = np.random.default_rng(seed)
    for _ in range(500):
        wall = rng.random((H, W)) < wall_p
        # the start cell has to be free or the idx lookup for s0 below raises
        wall[0, 0] = False
        free = [(r, c) for r in range(H) for c in range(W) if not wall[r, c]]
        idx = {cell: i for i, cell in enumerate(free)}
        S = len(free)
        trans = np.zeros((S, 4), int)
        for cell, s in idx.items():
            for a, (dr, dc) in enumerate(ACTIONS):
                nr, nc = cell[0] + dr, cell[1] + dc
                trans[s, a] = idx[(nr, nc)] if (0 <= nr < H and 0 <= nc < W and not wall[nr, nc]) else s
        s0 = idx[(0, 0)]
        dist = np.full(S, -1); dist[s0] = 0; q = deque([s0])
        while q:
            s = q.popleft()
            for a in range(4):
                ns = trans[s, a]
                if dist[ns] < 0:
                    dist[ns] = dist[s] + 1; q.append(ns)
        goals = []
        ok = True
        for td in TARGETS:
            # cells BFS never reached keep dist -1, so >= 2 drops them together with
            # s0 and its immediate neighbors
            cand = [s for s in range(S) if dist[s] >= 2 and s not in goals]
            if not cand:
                ok = False; break
            g = min(cand, key=lambda s: (abs(dist[s] - td), dist[s]))
            if abs(dist[g] - td) > 2:
                ok = False; break
            goals.append(g)
        if ok:
            return dict(S=S, trans=trans, s0=s0, goals=goals, dist=dist,
                        wall=wall, coords=free, H=H, W=W)
    raise RuntimeError("no solvable maze found")


# --------------------------------------------------------------- exact DP
def value_iteration(M, goal):
    S, trans = M["S"], M["trans"]
    V = np.zeros(S)
    for _ in range(400):
        Q = np.stack([(trans[:, a] == goal) + GAMMA * np.where(trans[:, a] == goal, 0.0, V[trans[:, a]])
                      for a in range(4)], 1)
        # zero the goal row, since re-entering the goal would pay 1 again and send
        # V* to a geometric series
        Q[goal] = 0.0
        newV = Q.max(1)
        if np.max(np.abs(newV - V)) < 1e-10:
            return newV
        V = newV
    return V


def policy_value(M, pi, goal):
    """Exact discounted value of the deterministic policy pi against `goal`.

    pi is optimal for some other level in this experiment, so the call is the
    cross-evaluation entry v[i][j] that the regret matrix is built from."""
    S, trans = M["S"], M["trans"]
    V = np.zeros(S)
    for _ in range(400):
        ns = trans[np.arange(S), pi]
        newV = (ns == goal) + GAMMA * np.where(ns == goal, 0.0, V[ns])
        newV[goal] = 0.0
        if np.max(np.abs(newV - V)) < 1e-10:
            return V
        V = newV
    return V


def opt_policy(M, goal):
    S, trans = M["S"], M["trans"]
    V = value_iteration(M, goal)
    Q = np.stack([(trans[:, a] == goal) + GAMMA * np.where(trans[:, a] == goal, 0.0, V[trans[:, a]])
                  for a in range(4)], 1)
    return Q.argmax(1), V


def rollout_return(M, act_fn, goal):
    """Return of one deterministic rollout from s0, GAMMA ** (t - 1) when the goal is
    entered on step t.

    A rollout that never enters the goal scores exactly 0, so a starved solver reports
    a ceiling of 0 instead of a slightly low one. That collapse is what panel (c) shows
    on the far levels."""
    s, disc = M["s0"], 1.0
    for _ in range(HORIZON):
        ns = M["trans"][s, act_fn(s)]
        if ns == goal:
            return disc
        disc *= GAMMA; s = ns
    return 0.0


# --------------------------------------------------------------- solvers
def solve_ppo(M, goal, budget, seed):
    """Tabular REINFORCE with an EMA baseline, the model-free arm of the comparison.

    steps counts environment transitions at one per action, so the budget is the same
    currency the two searches spend. The reported number is the achieved return of the
    greedy policy."""
    rng = np.random.default_rng(seed)
    # lr 0.5 is safe because theta is tabular, so a step at one state cannot disturb
    # any other state
    theta = np.zeros((M["S"], 4)); lr, steps, baseline = 0.5, 0, 0.0
    while steps < budget:
        s, traj, disc, ret = M["s0"], [], 1.0, 0.0
        for _ in range(HORIZON):
            p = np.exp(theta[s] - theta[s].max()); p /= p.sum()
            a = rng.choice(4, p=p); ns = M["trans"][s, a]
            traj.append((s, a, disc)); steps += 1
            if ns == goal:
                ret += disc; break
            disc *= GAMMA; s = ns
        baseline = 0.9 * baseline + 0.1 * ret; adv = ret - baseline
        for (s, a, disc) in traj:
            p = np.exp(theta[s] - theta[s].max()); p /= p.sum()
            grad = -p; grad[a] += 1.0
            # disc carries the GAMMA ** t weight of the discounted policy gradient
            # before the first goal hit ret and baseline are both 0, so adv is exactly 0
            # and the 0.01 entropy term is the only thing moving theta
            theta[s] += lr * (adv * disc * grad + 0.01 * (-p * (np.log(p + 1e-12) + 1)))
    greedy = theta.argmax(1)
    return rollout_return(M, lambda s: greedy[s], goal), steps


def solve_mcts(M, goal, budget, seed, readout="greedy"):
    """readout="greedy": return of one greedy-policy rollout (the achieved-return
    convention, same as solve_ppo). readout="root": max root Q value W/N (the
    AlphaZero-style value readout).

    N and W are keyed by (state, action) with no path in the key, so transpositions
    share statistics. Return-to-go depends only on (s, a) once the goal is fixed, so
    the merge is sound and the backup divides the depth back out.

    SOLVERS takes the greedy default. readout="root" is the arm that the sibling
    accuracy_vs_support experiment calls."""
    rng = np.random.default_rng(seed); trans = M["trans"]; N, W = {}, {}; steps = 0
    while steps < budget:
        s, path, disc, depth, ret = M["s0"], [], 1.0, 0, 0.0
        while depth < HORIZON:
            unvisited = [a for a in range(4) if (s, a) not in N]
            if unvisited:
                a = int(unvisited[rng.integers(len(unvisited))]); expand = True
            else:
                tot = sum(N[(s, aa)] for aa in range(4))
                # 1.4 is the usual sqrt(2) UCT constant and W/N here is already in [0, 1]
                a = max(range(4), key=lambda aa: W[(s, aa)] / N[(s, aa)] + 1.4 * np.sqrt(np.log(tot) / N[(s, aa)]))
                expand = False
            path.append((s, a)); ns = trans[s, a]; steps += 1
            if ns == goal:
                ret = disc; break
            disc *= GAMMA; s = ns; depth += 1
            if expand:
                rs, rdisc = s, disc
                for _ in range(HORIZON - depth):
                    # rollout transitions are charged to the same budget as tree steps
                    rns = trans[rs, int(rng.integers(4))]; steps += 1
                    if rns == goal:
                        ret = rdisc; break
                    rdisc *= GAMMA; rs = rns
                break
        # backup return-to-go, discount recovered from position in the path
        # ret is discounted from the root, so ret / GAMMA ** i is the value at depth i
        for i, (ps, pa) in enumerate(path):
            k = (ps, pa); N[k] = N.get(k, 0) + 1
            W[k] = W.get(k, 0.0) + ret / (GAMMA ** i)
    # a state the tree never visited defaults to action 0, so a greedy rollout that
    # leaves the visited set repeats one move to the horizon and scores 0
    def pv(s):
        cand = [(W[(s, a)] / N[(s, a)], a) for a in range(4) if (s, a) in N]
        return max(cand)[1] if cand else 0
    if readout == "root":
        s0 = M["s0"]
        root_q = [W[(s0, a)] / N[(s0, a)] for a in range(4) if (s0, a) in N]
        return (max(root_q) if root_q else 0.0), steps
    return rollout_return(M, pv, goal), steps


def solve_mcts_guided(M, goal, budget, seed):
    """UCB-directed trajectories with model-based Bellman backups of a tabular V.

    V starts at zero and every write is a max over one-step lookaheads, so the estimate
    climbs toward V* from below and never reports a ceiling that is too high. A starved
    run understates regret instead of inflating it.

    No randomness is drawn. The seed argument only matches the signature of the other
    two solvers, so the three seeds per level are one run repeated and the spread on
    the guided band comes from the mazes alone."""
    rng = np.random.default_rng(seed); trans, S = M["trans"], M["S"]
    V = np.zeros(S); Nsa = np.zeros((S, 4)); steps = 0

    # one lookahead reads all four successors and is charged 4 units of budget, where
    # the other two solvers pay 1 per transition
    def qvals(s):
        nonlocal steps
        q = np.empty(4)
        for a in range(4):
            ns = trans[s, a]; steps += 1
            q[a] = 1.0 if ns == goal else GAMMA * V[ns]
        return q

    # the budget is only checked between trajectories, so one trajectory can overshoot
    # by up to 8 * HORIZON queries
    # at B=500 that is about 960 queries spent against 500 asked
    while steps < budget:
        s, visited = M["s0"], []
        for _ in range(HORIZON):
            q = qvals(s); tot = Nsa[s].sum()
            # the +1 in both terms keeps the first visit to a state finite
            a = int(np.argmax(q + 1.0 * np.sqrt(np.log(tot + 1) / (Nsa[s] + 1))))
            Nsa[s, a] += 1; visited.append(s); ns = trans[s, a]
            if ns == goal:
                break
            s = ns
        # backwards so one sweep carries the goal value the whole way to s0
        # a forward sweep would move it one step per trajectory
        for ps in reversed(visited):
            V[ps] = qvals(ps).max()
    # reports the learned root value where the other two report an achieved return
    return float(V[M["s0"]]), steps


SOLVERS = (("ppo", solve_ppo), ("mcts", solve_mcts), ("mcts_g", solve_mcts_guided))


# --------------------------------------------------------------- minimax + MMD
def minimax_regret(Reg):
    """Minimax regret over mixtures of students, plus the level distribution attaining it.

    The LP is written for the student: minimize t subject to Reg.T @ x <= t and
    sum(x) == 1. The generator's p* comes out of the duals on the level rows, so one
    solve gives both sides of the game."""
    ns, nl = Reg.shape
    c = np.zeros(ns + 1); c[-1] = 1.0
    res = linprog(c, np.hstack([Reg.T, -np.ones((nl, 1))]), np.zeros(nl),
                  np.hstack([np.ones((1, ns)), [[0.0]]]), [1.0],
                  [(0, None)] * ns + [(None, None)], method="highs")
    # HiGHS returns non-positive duals on <= rows so the sign flip makes them weights
    # the clip removes solver noise on levels that carry no mass
    p = np.clip(-np.asarray(res.ineqlin.marginals), 0, None)
    # a solver that reports 0 on every level leaves Reg all zero with no dual mass
    # uniform keeps the TV metric defined instead of dividing by 0
    p = p / p.sum() if p.sum() > 1e-9 else np.full(nl, 1 / nl)
    return p, float(res.x[-1])


def mmd_generator(Reg, T=4000, eta=1.0, alpha0=2.0, alpha_min=0.02, half_life=400):
    """Annealed magnetic mirror descent on the normalized regret matrix.

    x is the student and p is the generator. The update is the closed-form MMD step of
    Sokota et al. 2023 with the uniform magnet folded into the exponent a on log x.

    alpha stops at alpha_min and never reaches 0, which leaves the last iterate at a
    regularized equilibrium. On maze 0 that is TV 0.19 away from the exact minimax even
    when the regret matrix is exact, and it is the floor under tv_mmd in the output.
    Panel (b) reads tv_lp for that reason while tv_mmd tracks the loop a real UED run
    would use."""
    # normalize so eta 1.0 means the same thing on mazes of different regret scale
    # the 1e-9 keeps an all-zero Reg out of a divide by zero
    ns, nl = Reg.shape; Mn = Reg / (Reg.max() + 1e-9)
    x = np.full(ns, 1 / ns); p = np.full(nl, 1 / nl)
    for t in range(T):
        alpha = alpha_min + (alpha0 - alpha_min) * 0.5 ** (t / half_life)
        a, cc = 1.0 / (1 + eta * alpha), eta / (1 + eta * alpha)
        x = np.exp(a * np.log(np.clip(x, 1e-12, None)) - cc * (Mn @ p)); x /= x.sum()
        p = np.exp(a * np.log(np.clip(p, 1e-12, None)) + cc * (Mn.T @ x)); p /= p.sum()
    return p


# half the L1 distance, so the 0.10 cut in main is 10% of the level mass misplaced
def tv(a, b):
    return 0.5 * float(np.abs(np.asarray(a) - np.asarray(b)).sum())


# --------------------------------------------------------------- experiment
def maze_oracle(M):
    """Exact ceilings, the cross-evaluation matrix and the true minimax over levels.

    The student population is the N_LEVELS optimal policies so the game matrix is square
    by construction. A policy aimed at a far goal collects any nearer goal it steps over,
    which makes v triangular and leaves the near levels binding. On maze 0 the minimax
    generator puts all of its mass on the two nearest levels and none on the far three."""
    goals = M["goals"]
    ceiling = np.zeros(N_LEVELS); pis = []
    for j, g in enumerate(goals):
        pi, V = opt_policy(M, g); ceiling[j] = V[M["s0"]]; pis.append(pi)
    v = np.array([[policy_value(M, pis[i], g)[M["s0"]] for g in goals] for i in range(N_LEVELS)])
    # clipping at 0 keeps every entry a regret, since float noise can put a student a
    # hair above the ceiling
    Reg = np.clip(ceiling[None, :] - v, 0, None)
    p_star, val = minimax_regret(Reg)
    return ceiling, v, p_star, val


def main():
    # budgets double, so a converge-budget ratio is resolved only to a factor of 2
    budgets = [500, 1000, 2000, 4000, 8000, 16000, 32000]
    # 4 mazes x 3 seeds is 12 runs per point, so frac moves in twelfths and the 0.95 cut
    # below means all 12
    solver_seeds = range(3)
    N_MAZES = 4
    keys = ["cerr", "tv_lp", "tv_mmd"]
    agg = {s: {k: [[] for _ in budgets] for k in keys} for s, _ in SOLVERS}

    # collect mazes with REAL regret structure (minimax value > 0.1); a maze where
    # a single student covers every level has no regret game and a degenerate p*.
    mazes, ms = [], 0
    while len(mazes) < N_MAZES:
        M = build_maze(ms); ms += 1
        ceiling, v, p_star, val = maze_oracle(M)
        if val > 0.1:
            mazes.append((M, ceiling, v, p_star, val))
    M0, ceiling0, v0, p_star0, _ = mazes[0]
    # 4000 is below every solver's convergence budget, so panel (c) still shows where
    # each one loses a level
    B_c = 4000                                   # fixed budget for the error-by-difficulty panel
    per_dist = {s: [] for s, _ in SOLVERS}       # per-level |ceiling error| at B_c, over mazes x seeds

    for (M, ceiling, v, p_star, val) in mazes:
        print("maze dist=%s  p*=%s val %.3f" % ([int(M["dist"][g]) for g in M["goals"]],
                                                np.round(p_star, 2), val))
        for bi, B in enumerate(budgets):
            for sd in solver_seeds:
                for soft, fn in SOLVERS:
                    # the seed is per level and shared across solvers, so the three
                    # curves are paired on the same draws
                    # a bigger budget re-runs from scratch instead of continuing the
                    # smaller one
                    chat = np.array([fn(M, g, B, sd * 100 + j)[0] for j, g in enumerate(M["goals"])])
                    Reg = np.clip(chat[None, :] - v, 0, None)
                    agg[soft]["cerr"][bi].append(float(np.mean(np.abs(chat - ceiling))))
                    agg[soft]["tv_lp"][bi].append(tv(minimax_regret(Reg)[0], p_star))
                    agg[soft]["tv_mmd"][bi].append(tv(mmd_generator(Reg), p_star))
                    if B == B_c:
                        per_dist[soft].append(np.abs(chat - ceiling))
    rows = {s: [{k: (float(np.mean(agg[s][k][bi])), float(np.std(agg[s][k][bi]))) for k in keys}
                for bi in range(len(budgets))] for s, _ in SOLVERS}

    # per-level ceilings on maze 0 at a starved budget (mechanism panel)
    B_show = 2000
    per_level = {"oracle": ceiling0.tolist()}
    # mean over seeds, so a middling entry is a mix of exact hits and zeros instead of
    # one mediocre policy
    for soft, fn in SOLVERS:
        per_level[soft] = [float(np.mean([fn(M0, g, B_show, sd * 100 + j)[0] for sd in solver_seeds]))
                           for j, g in enumerate(M0["goals"])]

    # reliability: fraction of runs (mazes x seeds) that reached the true minimax by budget B
    frac = {s: [float(np.mean(np.array(agg[s]["tv_lp"][bi]) <= 0.10)) for bi in range(len(budgets))]
            for s, _ in SOLVERS}
    # budget at which ~all runs reach the minimax (fraction >= 0.95): the honest
    # "reliably converges" budget (the 50% point is dominated by easy mazes).
    conv = {}
    # a solver that never reaches 0.95 reports the last budget, so a 32000 here has to
    # be read against frac[-1] before it counts as convergence
    for soft, _ in SOLVERS:
        conv[soft] = next((budgets[bi] for bi in range(len(budgets)) if frac[soft][bi] >= 0.95), budgets[-1])
    per_dist_stat = {s: {"mean": np.mean(per_dist[s], 0).tolist(), "std": np.std(per_dist[s], 0).tolist()}
                     for s, _ in SOLVERS}
    out = {"budgets": budgets, "rows": rows, "frac": frac, "per_dist": per_dist_stat, "dist": TARGETS,
           "B_c": B_c, "per_level": per_level, "B_show": B_show,
           "p_star_maze0": p_star0.tolist(), "goal_dist_maze0": [int(M0["dist"][g]) for g in M0["goals"]],
           "converge_budget": conv, "n_mazes": N_MAZES, "n_seeds": len(list(solver_seeds))}
    (OUT / "mcts_vs_ppo_regret.json").write_text(json.dumps(out, indent=2))
    # the 0.05 in the label is stale, frac above cuts at 0.10
    print("converge budget (tv<=0.05):", conv)
    make_figures(out)


def make_figures(out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.4,
                  rc={"axes.edgecolor": "#c4c8d0", "grid.color": "#edeff3", "grid.linewidth": 0.8,
                      "font.family": "sans-serif"})
    B = np.array(out["budgets"]); dist = np.array(out["dist"])
    COL = {"ppo": "#d1682f", "mcts": "#8b909b", "mcts_g": "#1f7a4d"}
    LAB = {"ppo": "PPO (model-free RL)", "mcts": "vanilla MCTS (random rollouts)",
           "mcts_g": "value-guided MCTS (planning)"}
    # worst first so the guided curve draws over the other two at equal zorder
    order = ["mcts", "ppo", "mcts_g"]
    handles = []

    fig, ax = plt.subplots(1, 3, figsize=(17.5, 5.4))

    # ---- (a) regret estimate error vs compute ----
    for soft in order:
        m = np.array([r["cerr"][0] for r in out["rows"][soft]])
        s = np.array([r["cerr"][1] for r in out["rows"][soft]])
        h, = ax[0].plot(B, m, "-o", ms=6, lw=2.7, color=COL[soft], zorder=3, label=LAB[soft])
        # the error is non-negative, so an unclipped band would read as a signed error
        ax[0].fill_between(B, np.clip(m - s, 0, None), m + s, color=COL[soft], alpha=0.1, zorder=1)
        handles.append(h)
    ax[0].set_xscale("log"); ax[0].set_xlim(B[0] * 0.85, B[-1] * 1.15)
    ax[0].set_xlabel("solver compute  (env / model steps per level)")
    ax[0].set_ylabel(r"regret estimate error   $|\hat V - V^\ast|$")
    ax[0].set_title("(a)  More compute buys a better regret\nestimate  (0 = exactly right)",
                    loc="left", fontsize=13, fontweight="bold")

    # ---- (b) reliability: fraction of runs reaching the true minimax, + the compute gap ----
    for soft in order:
        f = np.array(out["frac"][soft]) * 100
        ax[1].plot(B, f, "-o", ms=6, lw=2.7, color=COL[soft], zorder=3)
    # the y range leaves room for the budget labels at -4.5 and the arrow label at 104
    ax[1].set_xscale("log"); ax[1].set_xlim(B[0] * 0.85, B[-1] * 1.15); ax[1].set_ylim(-6, 112)
    ax[1].set_xlabel("solver compute  (env / model steps per level)")
    ax[1].set_ylabel("runs that reach the true minimax  (%)")
    ax[1].set_title("(b)  The curriculum reaches the minimax only\nwhen regret is good; guided MCTS gets there first",
                    loc="left", fontsize=13, fontweight="bold")
    g, p = out["converge_budget"]["mcts_g"], out["converge_budget"]["ppo"]
    # a tie drops the arrow while the suptitle below still quotes p / g
    if g and p and p > g:
        ax[1].axvline(g, color=COL["mcts_g"], ls="--", lw=1.3, alpha=0.8)
        ax[1].axvline(p, color=COL["ppo"], ls="--", lw=1.3, alpha=0.8)
        ax[1].annotate("", xy=(p, 100), xytext=(g, 100),
                       arrowprops=dict(arrowstyle="<->", color="#222", lw=1.8))
        ax[1].text(np.sqrt(g * p), 104, "%.0fx less compute" % (p / g), ha="center", va="bottom",
                   fontsize=12.5, fontweight="bold", color="#111")
        ax[1].text(g, -4.5, "%dk" % (g / 1000), ha="center", va="top", color=COL["mcts_g"], fontsize=10.5, fontweight="bold")
        ax[1].text(p, -4.5, "%dk" % (p / 1000), ha="center", va="top", color=COL["ppo"], fontsize=10.5, fontweight="bold")

    # ---- (c) error by level difficulty: where the baselines fail ----
    for soft in order:
        m = np.array(out["per_dist"][soft]["mean"])
        s = np.array(out["per_dist"][soft]["std"])
        ax[2].plot(dist, m, "-o", ms=7, lw=2.7, color=COL[soft], zorder=3)
        ax[2].fill_between(dist, np.clip(m - s, 0, None), m + s, color=COL[soft], alpha=0.1)
    ax[2].set_xticks(dist); ax[2].set_xlim(dist[0] - 0.5, dist[-1] + 0.5)
    ax[2].set_xlabel("level difficulty  (goal distance from start)")
    ax[2].set_ylabel(r"regret estimate error at B=%d" % out["B_c"])
    ax[2].set_title("(c)  At a fixed budget the baselines fail on the\nfar levels; guided MCTS stays accurate",
                    loc="left", fontsize=13, fontweight="bold")
    # xy and xytext are data coordinates pinned to the current error scale
    ax[2].annotate("baselines blow up\non far levels", xy=(9, 0.6), xytext=(5.4, 0.72),
                   fontsize=10.5, color="#777", arrowprops=dict(arrowstyle="->", color="#aaa", lw=1.2))

    fig.suptitle("A UED curriculum needs a good regret estimate, and value-guided MCTS gets one with about %.0fx less compute than PPO or plain MCTS"
                 % (p / g), fontsize=14.5, weight="bold", y=1.05)
    fig.legend(handles=handles, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.005),
               frameon=False, fontsize=12)
    fig.text(0.5, -0.02, "%d random mazes x %d seeds; shaded band = std.  A level is a maze goal at the marked distance; the regret ceiling V* is exact (value iteration)."
             % (out["n_mazes"], out["n_seeds"]), ha="center", fontsize=10.5, color="#666")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(OUT / "mcts_vs_ppo_regret.png", dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", OUT / "mcts_vs_ppo_regret.png")


if __name__ == "__main__":
    main()
