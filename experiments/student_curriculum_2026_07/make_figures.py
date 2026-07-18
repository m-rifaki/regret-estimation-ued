"""Figures for the student-curriculum runs. Reads student_curriculum.json
(and extra_experiments.json for the follow-up figure).

Both logs are final. Every count written into an annotation below was read off them
by hand and has to be rechecked if the runs are ever regenerated.
"""
import json
from pathlib import Path

import numpy as np
import matplotlib

# pyplot binds the backend at import time, so Agg has to be chosen above the import below
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(__file__).parent
plt.rcParams.update({"figure.dpi": 150, "savefig.bbox": "tight", "font.size": 11})

data = json.loads((OUT / "student_curriculum.json").read_text())
# 12 runs per method, from 4 mazes x 3 seeds shared across all six methods
by = {}
for r in data["runs"]:
    by.setdefault(r["method"], []).append(r)

colors = {"oracle": "tab:blue", "mcts_g": "tab:green", "uniform": "tab:orange",
          "ppo": "tab:red", "mcts": "tab:gray", "mcts_root": "tab:cyan"}
names = {"oracle": "oracle", "mcts_g": "guided MCTS", "uniform": "uniform",
         "ppo": "PPO (REINFORCE)", "mcts": "vanilla MCTS",
         "mcts_root": "vanilla MCTS (root value)"}
# mcts_root stays out of main5 because it re-reads the mcts searches instead of running its own
main5 = ["oracle", "mcts_g", "uniform", "ppo", "mcts"]
# goal distances in cells and ascending, so index -1 everywhere below is the d14 level
dists = data["targets"]


def fig_test_performance():
    """Heatmap: methods x levels, mean final score printed in every cell."""
    # mcts_root is included here and put next to mcts so the two readouts read against each other
    order = ["oracle", "mcts_g", "mcts_root", "uniform", "ppo", "mcts"]
    M = np.array([[np.mean([r["final"][j] for r in by[m]]) * 100
                   for j in range(5)] for m in order])
    fig, ax = plt.subplots(figsize=(8, 4.6))
    # 0-100 pinned instead of autoscaled, so a color means the same score in every version
    im = ax.imshow(M, cmap="RdYlGn", vmin=0, vmax=100, aspect="auto")
    for i in range(len(order)):
        for j in range(5):
            v = M[i, j]
            # an exact 0 or 100 is all 12 runs agreeing, and a printed .0 would read as rounding
            # 35 is where RdYlGn goes dark enough that black text stops being readable
            ax.text(j, i, f"{v:.0f}" if v in (0, 100) else f"{v:.1f}",
                    ha="center", va="center",
                    color="white" if v < 35 else "black", fontsize=10)
    ax.set_xticks(range(5))
    ax.set_xticklabels([f"d={d}" for d in dists])
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([names[m] for m in order])
    ax.set_xlabel("level (goal distance)")
    ax.set_title("Final score per level, mean of 12 runs (% of optimal)")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(OUT / "fig_test_performance.png")
    plt.close(fig)


def fig_knife_edge():
    """Reported d14 ceiling vs final d14 score. Log x; exact zeros get their
    own slot on the left (marked by the vertical line)."""
    fig, ax = plt.subplots(figsize=(8.5, 5))
    # a log axis has no 0, so exact zeros park at 2e-5 under the smallest real estimate of 0.00087
    ZERO_X = 2e-5
    # coincident dots get fanned sideways so every run stays visible
    pts = []
    # uniform is absent because it never calls the solver and its v_hat is null in the log
    for m in ["oracle", "mcts_g", "ppo", "mcts", "mcts_root"]:
        for r in by[m]:
            x = r["v_hat"][-1] if r["v_hat"][-1] > 0 else ZERO_X
            pts.append((m, x, r["final"][-1] * 100))
    # collision test in plot space: log10 for the x axis and whole percent for y
    groups = {}
    for m, x, y in pts:
        groups.setdefault((round(np.log10(x), 2), round(y)), []).append((m, x, y))
    seen = set()
    for _, g in groups.items():
        n = len(g)
        # sorted so the fan order is stable, and multiplicative because the x axis is logged
        for k, (m, x, y) in enumerate(sorted(g)):
            fx = x * (1.09 ** (k - (n - 1) / 2))
            lab = names[m] if m not in seen else None
            seen.add(m)
            ax.plot(fx, y, "o", color=colors[m], ms=7, alpha=0.85, label=lab,
                    mec="white", mew=0.5)
    ax.set_xscale("log")
    ax.set_xlim(8e-6, 1.2)
    ax.axvline(1e-4, color="k", lw=0.8, ls="--")
    # 0.673 is the true d14 ceiling and the oracle reports it in all 12 of its runs
    ax.set_xticks([ZERO_X, 1e-3, 1e-2, 1e-1, 0.673])
    ax.set_xticklabels(["exactly 0", "0.001", "0.01", "0.1", "0.673\n(true)"])
    # 24 and 36 count the five methods plotted above and go stale if that list changes
    ax.text(ZERO_X, 22, "24 runs here,\nall score 0", ha="center", fontsize=9.5)
    ax.text(0.02, 80, "36 runs with any nonzero estimate:\nall score 94-100", fontsize=9.5, ha="center")
    ax.set_xlabel("ceiling the solver reported for the distance-14 level")
    ax.set_ylabel("student's final score on that level (%)")
    ax.set_ylim(-8, 108)
    ax.set_title("What the solver reported vs what the student achieved")
    ax.legend(fontsize=8.5, loc="center right")
    fig.tight_layout()
    fig.savefig(OUT / "fig_knife_edge.png")
    plt.close(fig)


def fig_hardest_level():
    """Cumulative expected episodes sent to the distance-14 level."""
    fig, ax = plt.subplots(figsize=(8, 4.8))
    # c[2] is a sampling distribution over levels, so share x 18 episodes is an expected count
    ep = data["ep_per_round"]
    for m in main5:
        # the last checkpoint carries a distribution that training stopped before spending
        cum = np.mean([np.cumsum([c[2][-1] * ep for c in r["curve"][:-1]])
                       for r in by[m]], axis=0)
        ax.plot(cum, color=colors[m], lw=2, label=names[m])
    ax.set_xlabel("curriculum round")
    ax.set_ylabel("cumulative episodes on the d14 level (expected)")
    ax.set_title("Episodes the d14 level received during training")
    # vanilla reported exactly 0 for d14 in all 12 runs so its line stays on the axis
    # PPO's mean is one run spread over 12 because the other 11 also reported 0
    ax.annotate("vanilla: 0 episodes\nPPO line = its single run with a nonzero estimate",
                xy=(85, 8), xytext=(40, 520), fontsize=9.5,
                arrowprops=dict(arrowstyle="->", lw=0.9))
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "fig_hardest_level.png")
    plt.close(fig)


def fig_readout():
    """Paired per-run comparison: same (maze, seed), two readouts."""
    fig, ax = plt.subplots(figsize=(6.5, 5))
    # both methods read the same searches, so a missing key below would mean the pairing broke
    g = {(r["maze"], r["seed"]): r["worst"] * 100 for r in by["mcts"]}
    rt = {(r["maze"], r["seed"]): r["worst"] * 100 for r in by["mcts_root"]}
    for k in g:
        ax.plot([0, 1], [g[k], rt[k]], "-", color="0.6", lw=1)
        ax.plot(0, g[k], "o", color=colors["mcts"], ms=7)
        ax.plot(1, rt[k], "o", color=colors["mcts_root"], ms=7)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["greedy rollout readout\n(reports 0.000)",
                        "root value readout\n(reports ~0.003)"])
    ax.set_xlim(-0.35, 1.35)
    ax.set_ylim(-6, 106)
    ax.set_ylabel("worst-level score (%)")
    ax.set_title("One vanilla MCTS search read out two ways (a line per run)")
    # the right column overlaps at three values so the counts are written in by hand
    ax.text(1.06, 97, "9 runs", fontsize=9)
    ax.text(1.06, 91.5, "2 runs at 94", fontsize=9)
    ax.text(1.06, 2, "1 run", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "fig_readout.png")
    plt.close(fig)


def fig_learning_curves():
    """Small multiples, every run visible plus the mean."""
    fig, axes = plt.subplots(1, 5, figsize=(14, 3.4), sharey=True)
    # 58000 is the longest run rounded down from 58077 env steps
    grid = np.linspace(0, 58000, 200)
    for ax, m in zip(axes, main5):
        cs = []
        for r in by[m]:
            # np.interp holds the final value past the end of a run, so a run that stopped at
            # 15k still counts toward the mean out at 58k
            c = np.interp(grid, [c[0] for c in r["curve"]],
                          [min(c[1]) * 100 for c in r["curve"]])
            cs.append(c)
            ax.plot(grid / 1000, c, color=colors[m], lw=0.8, alpha=0.35)
        ax.plot(grid / 1000, np.mean(cs, axis=0), color="k", lw=2)
        ax.set_title(names[m], fontsize=10)
        ax.set_xlabel("steps (k)")
    axes[0].set_ylabel("worst-level score (%)")
    # 1.04 clears the tight_layout box, which does not reserve room for a suptitle
    fig.suptitle("Worst-level score during training. Thin lines = the 12 runs, black = mean", y=1.04)
    fig.tight_layout()
    fig.savefig(OUT / "fig_learning_curves.png")
    plt.close(fig)


def fig_followups():
    """The three follow-ups: support vs solver budget, and floor sweep vs clamp."""
    e = json.loads((OUT / "extra_experiments.json").read_text())
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))

    ax = axes[0]
    # 16000 is the solver budget of the main experiment and the rest are doublings of it
    # e3_support stores a fraction of 12 runs, so x100 puts it on the percent axis
    budgets = [16000, 32000, 64000, 128000, 256000]
    for m, mk in [("ppo", "o-"), ("mcts", "s-")]:
        ax.plot([b / 1000 for b in budgets],
                [e["e3_support"][f"{m}@{b}"] * 100 for b in budgets],
                mk, color=colors[m], label=names[m])
    ax.set_xscale("log")
    ax.set_xticks([16, 32, 64, 128, 256])
    ax.set_xticklabels(["16k", "32k", "64k", "128k", "256k"])
    ax.xaxis.set_minor_locator(plt.NullLocator())
    ax.axvline(16, color="k", lw=0.8, ls=":")
    ax.text(17, 55, "budget of the\nmain experiment", fontsize=8.5)
    ax.set_xlabel("solver budget (env steps)")
    ax.set_ylabel("runs where reported d14 ceiling > 0 (%)")
    # 8x is the 128k budget over 16k, the first point where every run reports a nonzero ceiling
    ax.set_title("Support can be bought with compute: about 8x")
    ax.legend(fontsize=9)

    ax = axes[1]
    # floor 0 is missing from the follow-up file because the main experiment ran at floor 0.0
    floors = [0, 2, 10, 20, 30, 50]
    base = {"ppo": np.mean([r["worst"] for r in by["ppo"]]) * 100,
            "mcts": np.mean([r["worst"] for r in by["mcts"]]) * 100}
    for m, mk in [("ppo", "o-"), ("mcts", "s-")]:
        # keys were written with Python float repr, so 2 is "0.02" and 10 is "0.1"
        ys = [base[m]] + [np.mean(e["e2_floor"][f"{m}@{f/100}"]) * 100 for f in floors[1:]]
        ax.plot(floors, ys, mk, color=colors[m], label=names[m] + ", floor only")
    # e1_clamp holds the same 12 seeds rerun with the reported ceiling floored at 0.01
    clamp = {m: np.mean(e["e1_clamp"][m]) * 100 for m in ["ppo", "mcts"]}
    ax.axhline(clamp["ppo"], color=colors["ppo"], ls="--", lw=1)
    ax.axhline(clamp["mcts"], color=colors["mcts"], ls="--", lw=1)
    ax.text(1, clamp["mcts"] - 8, f"one-line clamp (v*>=0.01): {clamp['ppo']:.1f} / {clamp['mcts']:.1f}",
            fontsize=9)
    ax.set_xlabel("uniform exploration floor (% of sampling)")
    ax.set_ylabel("worst-level score, mean of 12 runs (%)")
    ax.set_ylim(-5, 108)
    ax.set_title("Exploration barely helps; the clamp fixes it")
    ax.legend(fontsize=9, loc="center right")

    fig.tight_layout()
    fig.savefig(OUT / "fig_followups.png")
    plt.close(fig)


if __name__ == "__main__":
    fig_test_performance()
    fig_knife_edge()
    fig_hardest_level()
    fig_readout()
    fig_learning_curves()
    fig_followups()
    print("wrote 6 figures")
