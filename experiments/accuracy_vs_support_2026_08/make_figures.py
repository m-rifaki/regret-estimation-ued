"""Every figure here answers one question, and that question is its title.

Design rule: one mark = one observation. Nothing is reduced to an average unless the
individual observations are drawn alongside it. Absolute score against level count is
deliberately absent: those lines all fall to the right because a fixed budget spread
over more levels gives each level fewer episodes, which is arithmetic rather than a
result, and it buries the finding under a slope nobody asked about.
"""
from __future__ import annotations
import json
import os
from pathlib import Path
from statistics import mean

# Agg before pyplot binds a backend, so this renders on the HPC with no display
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.colors import ListedColormap, BoundaryNorm

OUT = Path(__file__).parent
# DPI=150 renders a lighter set for embedding as base64 inside the markdown
DPI = int(os.environ.get("FIG_DPI", "200"))
AUG6 = OUT.parent / "student_curriculum_2026_07" / "student_curriculum.json"
sns.set_theme(style="white", context="talk", font_scale=0.8)
GREEN, RED, GREY = "#1f6f4a", "#b3452c", "#6b6b6b"


def fig_mechanism():
    """Question: why did the SIZE of the estimate not change the curriculum?

    One thin line per run: the probability the curriculum gave the hardest level,
    round by round. 5 panels x 12 runs x 101 rounds = 6,060 plotted values."""
    d = json.load(open(AUG6))
    by = {}
    for r in d["runs"]:
        # uniform is the one method that carries no estimate, so this drops its 12 runs
        if r.get("v_hat"):
            by.setdefault(r["method"], []).append(r)
    # the counts in these labels are runs whose line touches the top at some round
    # by round 100 oracle holds 8 of 12 and guided 7 of 12
    order = [("oracle", "exact best-possible score\n11 of 12 runs concentrate"),
             ("mcts_g", "guided MCTS\n11 of 12 runs concentrate"),
             ("mcts_root", "vanilla MCTS, root readout\n11 of 12 concentrate, and stay highest"),
             ("ppo", "PPO\n1 of 12 concentrates"),
             ("mcts", "vanilla MCTS\n0 of 12, flat at zero")]
    fig, ax = plt.subplots(1, 5, figsize=(18.5, 6.2), sharey=True, sharex=True)
    for k, (m, lab) in enumerate(order):
        rs = by[m]
        for r in rs:
            # row[2] is the sampling distribution over the 5 levels, ordered easiest first
            # so index 4 is target 14, the hardest
            p14 = [row[2][4] for row in r["curve"]]
            ax[k].plot(range(len(p14)), p14, lw=1.0, color="#1f6f4a", alpha=0.55)
        meanp = np.mean([[row[2][4] for row in r["curve"]] for r in rs], axis=0)
        ax[k].plot(range(len(meanp)), meanp, lw=3.0, color="black")
        ax[k].axhline(0.2, color="#999", lw=1.0, ls=":")
        ax[k].set_title(lab, fontsize=13.5)
        ax[k].set_xlabel("curriculum round")
        # the -0.05 margin keeps vanilla MCTS visible, since its line is exactly 0.0 at every round
        ax[k].set_ylim(-0.05, 1.05); ax[k].set_xlim(0, 100)
        ax[k].grid(alpha=0.25)
    ax[0].set_ylabel("chance the curriculum picks\nthe hardest level")
    ax[0].text(52, 0.25, "an even split\nwould be 0.2", fontsize=11, color="#666")
    fig.suptitle("Did a bigger estimate concentrate the curriculum more?"
                 "   (one thin line = one run, black = mean of 12)", fontsize=16.5, y=0.985)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(OUT / "fig_mechanism_trajectories.png", dpi=DPI)
    print("wrote fig_mechanism_trajectories.png")


def fig_mastery_matrix():
    """Question: with the same budget, does knowing the order matter?

    One cell per (run, level): 12 runs x 30 levels x 2 conditions = 720 cells.
    Levels are ordered easiest to hardest within each run, so columns are comparable
    across mazes."""
    d = json.load(open(OUT / "accuracy_vs_support.json"))
    runs = [r for r in d["runs"] if r["n_levels"] == 30]
    fig, ax = plt.subplots(1, 3, figsize=(21.0, 5.4), sharey=True)
    cmap = ListedColormap(["#f2f2f2", "#f4b48a", "#1f6f4a"])
    # three bins so exactly 0 and exactly 1 keep their own color and everything between takes the middle
    # 2 percent of cells are in that middle band and a level at 0.94 is drawn as partly there
    norm = BoundaryNorm([0, 0.001, 0.999, 1.001], cmap.N)
    panels = [("oracle", "exact order"), ("shuffle", "shuffled order"),
              ("uniform", "no curriculum at all")]
    for k, (m, lab) in enumerate(panels):
        # this sort is what makes row k the same maze and seed in all three panels
        rs = sorted([r for r in runs if r["method"] == m], key=lambda r: (r["maze"], r["seed"]))
        grid = []
        for r in rs:
            o = np.argsort(-np.asarray(r["v_opt"]))        # easiest (highest ceiling) first
            grid.append(np.asarray(r["final"])[o])
        grid = np.asarray(grid)
        # nearest keeps the three colors exact, since the default resampling invents shades the legend does not carry
        ax[k].imshow(grid, aspect="auto", cmap=cmap, norm=norm, interpolation="nearest")
        for y in range(grid.shape[0] + 1):
            ax[k].axhline(y - 0.5, color="white", lw=1.6)
        # every mastered level in this file is exactly 1.0, so 0.999 counts the same cells as the top bin
        tot = int((grid >= 0.999).sum())
        ax[k].set_title(f"{lab}: {tot} of 360 levels learned", fontsize=15)
        ax[k].set_xlabel("the 30 levels, easiest on the left")
        ax[k].set_xticks([0, 9, 19, 29]); ax[k].set_xticklabels(["easiest", "10th", "20th", "hardest"])
        ax[k].set_yticks(range(len(rs)))
        ax[k].set_yticklabels([f"maze {r['maze']}, seed {r['seed']}" for r in rs], fontsize=10)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in ["#1f6f4a", "#f4b48a", "#f2f2f2"]]
    ax[2].legend(handles, ["learned (100%)", "partly there", "never got there (0%)"],
                 loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False, fontsize=12)
    # 1800 in the title is ep_per_round 18 times n_rounds 100 and is typed in here
    fig.suptitle("With the same 1800 episodes, does knowing the order matter?"
                 "   (one square = one level in one run; rows are the same run in all three panels)",
                 fontsize=15.5, y=0.985)
    fig.tight_layout(rect=(0, 0, 0.95, 0.93))
    fig.savefig(OUT / "fig_mastery_matrix.png", dpi=DPI)
    print("wrote fig_mastery_matrix.png")


def fig_where_training_went():
    """Question: when two levels cannot be solved, where does the training go?

    One cell per (run, level) per method: 6 methods x 12 runs x 7 levels = 504 cells.
    The two right-hand columns of every panel are the sealed levels, where the true
    ceiling is exactly zero and every episode is thrown away."""
    d = json.load(open(OUT / "clamp_safety.json"))
    runs = d["runs"]
    order = [("oracle", "exact best-possible score"), ("mcts_g", "guided MCTS"),
             ("ppo", "PPO"), ("mcts", "vanilla MCTS"),
             ("ppo_clamp", "PPO + clamp"), ("mcts_clamp", "vanilla MCTS + clamp")]
    fig, ax = plt.subplots(2, 3, figsize=(16.5, 8.2), sharey=True)
    ax = ax.ravel()
    for k, (m, lab) in enumerate(order):
        rs = sorted([r for r in runs if r["method"] == m], key=lambda r: (r["maze"], r["seed"]))
        grid, dead_cols = [], None
        for r in rs:
            v = np.asarray(r["v_opt"]); a = np.asarray(r["alloc"])
            o = np.concatenate([np.argsort(-v[v > 0].round(6).argsort() * 0 - v[:5]),
                                np.arange(5, len(v))])
            # sorting only the first 5 keeps the sealed pair in columns 5 and 6 of every panel
            o = np.concatenate([np.argsort(-v[:5]), np.arange(5, len(v))])
            # alloc is a share of that run's episodes and sums to 1, so this is percent
            grid.append(a[o] * 100)
            # read off v[o] so it indexes the reordered grid
            # all 72 runs seal the same two levels, so the last assignment stands for every row
            dead_cols = [i for i in range(len(v)) if v[o][i] <= 0]
        grid = np.asarray(grid)
        # 60 clips the runaway columns: PPO and vanilla MCTS reach 100% on one level while the clamped runs top out at 48%
        im = ax[k].imshow(grid, aspect="auto", cmap="rocket_r", vmin=0, vmax=60,
                          interpolation="nearest")
        # imshow centers column i at x=i, so the divider goes half a cell left of the first sealed column
        ax[k].axvline(min(dead_cols) - 0.5, color="#2f6fb3", lw=3.0)
        for y in range(grid.shape[0] + 1):
            ax[k].axhline(y - 0.5, color="white", lw=1.2)
        waste = grid[:, dead_cols].sum(1).mean()
        ax[k].set_title(f"{lab}\n{waste:.0f}% of practice on the two impossible levels",
                        fontsize=13.5)
        ax[k].set_xticks(range(grid.shape[1]))
        if k >= 3:                                    # labels on the bottom row only
            ax[k].set_xticklabels(["easiest", "2nd", "3rd", "4th", "hardest",
                                   "impossible", "impossible"], fontsize=11, rotation=45, ha="right")
        else:
            ax[k].set_xticklabels([])
        if k % 3 == 0:
            ax[k].set_yticks(range(len(rs)))
            ax[k].set_yticklabels([f"m{r['maze']} s{r['seed']}" for r in rs], fontsize=9)
    fig.subplots_adjust(hspace=0.42, wspace=0.10)
    cb = fig.colorbar(im, ax=ax.tolist(), fraction=0.022, pad=0.02)
    cb.set_label("share of that run's episodes (%)", fontsize=12)
    fig.suptitle("When two levels are impossible, where does the practice go?"
                 "   (one square = one level in one run)", fontsize=16, y=0.99)
    # this figure spaces itself, so the crop happens at save time instead of through tight_layout
    fig.savefig(OUT / "fig_where_training_went.png", dpi=DPI, bbox_inches="tight")  # noqa
    print("wrote fig_where_training_went.png")


def fig_competition():
    """Question: why does a short budget make the ranking matter?

    Because of how many levels are competing for it at once. One thin line per run:
    the number of levels the curriculum still gives a nonzero chance, round by round.
    3 panels x 12 runs x 100 rounds = 3,600 plotted values."""
    d = json.load(open(OUT / "accuracy_vs_support_curves.json"))
    runs = d["runs"]
    EP = d["config"]["ep_per_round"]
    Ls = d["config"]["level_counts"]
    fig, ax = plt.subplots(1, 3, figsize=(15.5, 5.2), sharex=True)
    for k, L in enumerate(Ls):
        rs = [r for r in runs if r["n_levels"] == L and r["method"] == "oracle"]
        series = []
        for r in rs:
            # the sampler clamps at 0.01, so nothing exists between zero and this threshold
            n = [sum(1 for x in row[2] if x > 1e-9) for row in r["curve"]]
            series.append(n)
            ax[k].plot(range(len(n)), n, lw=1.0, color="#1f6f4a", alpha=0.55)
        m = np.mean(series, axis=0)
        ax[k].plot(range(len(m)), m, lw=3.0, color="black")
        end = m[-1]
        # at 30 levels this is under one episode per level per round
        ax[k].set_title(f"{L} levels: ends with {end:.0f} in play\n"
                        f"{EP / end:.1f} episodes each per round", fontsize=14)
        ax[k].set_xlabel("curriculum round")
        # each panel scales to its own L, so heights are not comparable across the three
        ax[k].set_ylim(0, L * 1.05); ax[k].set_xlim(0, 100)
        ax[k].grid(alpha=0.25)
    ax[0].set_ylabel("levels the curriculum still\ngives any chance")
    fig.suptitle(f"How many levels are still in play, with the same {EP} episodes "
                 f"per round throughout", fontsize=16, y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(OUT / "fig_competition.png", dpi=DPI)
    print("wrote fig_competition.png")


def paired(runs, L, a="oracle", b="shuffle"):
    """One difference per (maze, seed), so maze difficulty cancels out of every point."""
    A = {(r["maze"], r["seed"]): r for r in runs if r["n_levels"] == L and r["method"] == a}
    B = {(r["maze"], r["seed"]): r for r in runs if r["n_levels"] == L and r["method"] == b}
    # a run with no partner on the other side is dropped instead of compared across mazes
    return [100 * (A[k]["mean"] - B[k]["mean"]) for k in A if k in B]


def fig_ranking():
    """Question: does knowing which levels are hardest make the student better?

    One dot per run: the student's score with the exact ceiling minus its score with
    the ceilings shuffled between levels, on the same maze with the same seed. Above
    zero means knowing the ranking helped."""
    a = json.load(open(OUT / "accuracy_vs_support.json"))
    b = json.load(open(OUT / "accuracy_perlevel_budget.json"))
    Ls = a["config"]["level_counts"]

    fig, ax = plt.subplots(1, 2, figsize=(12.6, 5.4), sharey=True)
    panels = [(a["runs"], "Each level gets fewer episodes as levels are added",
               "360, 120, 60 episodes per level"),
              (b["runs"], "Each level keeps the same episodes",
               "360 episodes per level throughout")]

    for k, (runs, title, sub) in enumerate(panels):
        for i, L in enumerate(Ls):
            d = paired(runs, L)
            # offsets are by index, so a rebuild puts every dot back where it was
            xs = [i + (j - len(d) / 2) * 0.030 for j in range(len(d))]
            ax[k].scatter(xs, d, s=64, color=GREEN, alpha=0.75, zorder=3)
            ax[k].hlines(mean(d), i - 0.26, i + 0.26, color="black", lw=3, zorder=4)
            # 27 clears the widest run at 21 points, so the mean label never covers a dot
            ax[k].text(i, 27, f"{mean(d):+.1f}", ha="center", fontsize=15)
        ax[k].axhline(0, color=GREY, lw=1.4, zorder=1)
        ax[k].set_xticks(range(len(Ls)))
        ax[k].set_xticklabels([f"{L} levels" for L in Ls])
        ax[k].set_title(title, fontsize=15)
        ax[k].set_xlabel(sub, fontsize=13)
        ax[k].set_ylim(-27, 32)
    ax[0].set_ylabel("exact order minus shuffled order\n(points of score)")
    fig.suptitle("Does knowing which levels are hardest make the student better?",
                 fontsize=18, y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(OUT / "fig_ranking_matters.png", dpi=DPI)
    print("wrote fig_ranking_matters.png")


def fig_partial_ordering():
    """Question: does a PARTLY correct order help, or only an exactly correct one?

    One dot per run. x is how well that run's estimated ceilings ranked the levels
    (Spearman correlation against the true ceilings). y is that run's score minus the
    score of the no-curriculum run on the SAME maze with the SAME seed, so maze-to-maze
    variation cancels. 3 panels x 5 estimate sources x 12 runs = 180 dots.

    Paired, because unpaired the per-run spread swamps the effect entirely: at 30 levels
    the correlation-1.0 runs alone range from 13% to 46% score.

    `flat` reports one identical number for every level, so its ranking carries no
    information and its correlation is undefined. It is drawn at 0 by that convention."""
    # scipy is needed by this figure alone, so the other five render without it
    from scipy.stats import spearmanr
    d = json.load(open(OUT / "accuracy_vs_support.json"))
    Ls = d["config"]["level_counts"]
    # uniform is the baseline being subtracted, so it would be a row of zeros at an undefined x
    srcs = ["oracle", "noisy_0.3", "noisy_1.0", "flat", "shuffle"]
    fig, ax = plt.subplots(1, 3, figsize=(16.5, 5.6), sharey=True)
    for k, L in enumerate(Ls):
        U = {(r["maze"], r["seed"]): r["mean"] for r in d["runs"]
             if r["n_levels"] == L and r["method"] == "uniform"}
        for m in srcs:
            rs = [r for r in d["runs"] if r["n_levels"] == L and r["method"] == m]
            rho, delta = [], []
            for r in rs:
                vh = np.asarray(r["v_hat"]); vo = np.asarray(r["v_opt"])
                # spearmanr returns nan on a flat estimate and matplotlib drops the dot with no error
                rho.append(0.0 if vh.std() < 1e-12 else spearmanr(vh, vo).correlation)
                delta.append(100 * (r["mean"] - U[(r["maze"], r["seed"])]))
            # oracle is exactly 1.0 in all 12 runs, so it needs its own color to read as a column
            col = "#2f6fb3" if m == "oracle" else GREEN
            ax[k].scatter(rho, delta, s=62, color=col, alpha=0.7, zorder=3)
            ax[k].hlines(mean(delta), mean(rho) - 0.07, mean(rho) + 0.07,
                         color="black", lw=3, zorder=4)
        ax[k].axhline(0, color=RED, lw=2.0, ls="--", zorder=2)
        ax[k].set_title(f"{L} levels", fontsize=15)
        if k == 2:
            h = [plt.Line2D([], [], marker="o", ls="", color=c, markersize=10)
                 for c in ("#2f6fb3", GREEN)]
            ax[k].legend(h, ["exact order", "damaged order"], frameon=False,
                         fontsize=12, loc="lower right")
        ax[k].set_xlabel("how well the estimate put the levels in order\n"
                         "(0 = no information, 1 = exactly right)")
        # the left limit clears shuffle at -0.28 and leaves room for the note at -0.42
        ax[k].set_xlim(-0.45, 1.2)
        ax[k].grid(alpha=0.25)
    ax[0].set_ylabel("score minus the no-curriculum score\non the same maze and seed (points)")
    ax[0].text(-0.42, -7.5, "at this line, no better than\nno curriculum at all",
               fontsize=11, color=RED)
    fig.suptitle("Does a partly correct order help, or only an exactly correct one?"
                 "   (one dot = one run, black bar = mean of 12)", fontsize=16, y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(OUT / "fig_partial_ordering.png", dpi=DPI)
    print("wrote fig_partial_ordering.png")


def _main():
    # partial_ordering first, so a missing scipy stops this before five figures are written
    fig_partial_ordering()
    fig_mechanism()
    fig_competition()
    fig_mastery_matrix()
    fig_ranking()
    fig_where_training_went()


if __name__ == "__main__":
    _main()
