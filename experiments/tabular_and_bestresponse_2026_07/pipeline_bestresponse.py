"""Best-response estimator fix (Eugene 6/11) + MMD generator.

RECORD of the 2026-07 experiment. The co-trained antagonist in the corrected
Overcooked loop scored below the student (v_ant < v_pro), so the measured regret
went negative and mis-ranked the partners, and the generator settled on the wrong
partner. Fix: replace the co-trained antagonist with a REAL best-response per
partner (train a fresh protagonist to convergence against each frozen partner ->
true V*(j)). Regret = max(0, V*(j) - v_pro), the exact "frozen antagonist =
oracle" 2p-zero-sum game Eugene described. Generator uses the toy-validated MMD
update. Result: the generator settles cleanly onto partner 2, and is TV 0.18 from
the minimax of the trained-V* regret game.

DEPENDENCY NOTE: this drives the Overcooked/JaxMARL pipeline via `ippo`,
`ippo_corrected`, and `pipeline_corrected`, which lived in the working directory
that was wiped between sessions. The exact numbers are recorded in
docs/meeting-2026-07-16.md and the figure fig_estimate_story.png. This file is
kept as the experiment record; to re-run, drop it next to those modules.

  PUED_OUT=results_br PUED_BR_UPDATES=200 python pipeline_bestresponse.py
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from ippo import Config, eval_pair, make_train_selfplay, make_train_vs_frozen
from ippo_corrected import build_inner_step
from minimax_regret import solve_minimax_regret, total_variation
from pipeline_corrected import (
    PartnerGenerator, SEEDS, EVAL_EPS, eval_pair_sym, stack_pool, log, build_init_params,
)

OUT = Path(os.environ.get("PUED_OUT", str(Path.home() / "Projects/regret-estimation-ued-work/results_br")))
OUT.mkdir(parents=True, exist_ok=True)
SP_UPDATES = int(os.environ.get("PUED_SP_UPDATES", "75"))
# 200 against the pool's 75 because V*(j) has to be converged to be a ceiling: a best
# response still climbing reads low and phase D clips that partner to zero regret
BR_UPDATES = int(os.environ.get("PUED_BR_UPDATES", "200"))
# N_OUTER is also the MMD annealing horizon, so a shorter run anneals faster instead
# of stopping early
N_OUTER = int(os.environ.get("PUED_N_OUTER", "30"))
K_INNER = int(os.environ.get("PUED_K_INNER", "4"))
N_LEVELS = len(SEEDS)


def main() -> None:
    t0 = time.time()
    n = len(SEEDS)

    # PHASE A: partner pool (deterministic, same as results_corrected/_mmd).
    log("=" * 60); log(f"PHASE A: selfplay pool of {n} seeds, {SP_UPDATES} updates")
    train_sp = make_train_selfplay(Config(num_updates=SP_UPDATES))
    pool = {}
    for s in SEEDS:
        params, _ = jax.block_until_ready(train_sp(jax.random.PRNGKey(1000 + s)))
        pool[s] = params
        m, _ = eval_pair_sym(params["agent_0"], params["agent_1"], jax.random.PRNGKey(9000 + s))
        log(f"  seed {s}: selfplay {m:.1f}")

    # PHASE B: cross-play topology (gives the LP p* target).
    log("=" * 60); log("PHASE B: cross-play topology")
    R = np.zeros((n, n)); SE = np.zeros((n, n))
    for i, si in enumerate(SEEDS):
        for j, sj in enumerate(SEEDS):
            # 10 * i + j gives one key per ordered pair, and that stays collision free
            # only while len(SEEDS) is under 10
            m, se = eval_pair_sym(pool[si]["agent_0"], pool[sj]["agent_1"], jax.random.PRNGKey(7000 + 10 * i + j))
            R[i, j] = m; SE[i, j] = se
    # selfplay_diag is phase B's own draw at these keys, so it will not reproduce the
    # phase A numbers logged above exactly
    (OUT / "topology_corrected.json").write_text(json.dumps({
        "R": R.tolist(), "SE": SE.tolist(), "selfplay_diag": [R[i, i] for i in range(n)],
        "mean_crossplay": float((R.sum() - np.trace(R)) / (n * n - n)),
    }, indent=2))
    # no external V* passed, so this LP takes the column max over the pool as the ceiling
    sol = solve_minimax_regret(R); p_star = sol.generator_dist
    log(f"  LP minimax p* = {np.round(p_star, 3)}  value {sol.game_value:.3f}")

    # PHASE C: real best-response per partner -> true V*(j).
    log("=" * 60); log(f"PHASE C: best-response per partner ({BR_UPDATES} updates each)")
    train_br = make_train_vs_frozen(Config(num_updates=BR_UPDATES))
    vstar = np.zeros(n)
    for j, sj in enumerate(SEEDS):
        partner = pool[sj]["agent_1"]
        # a fresh protagonist per partner: one protagonist shared across partners would
        # be the co-trained antagonist this experiment exists to remove
        p0, _ = jax.block_until_ready(train_br(jax.random.PRNGKey(3000 + sj), partner))
        # V*(j) is read against the same frozen partner in the seat the best response
        # trained in, so it is a ceiling for that pairing only
        m, _, _, _ = eval_pair(p0, partner, jax.random.PRNGKey(8000 + sj), EVAL_EPS)
        vstar[j] = m
        log(f"  partner {sj}: V* (best-response) {m:.1f}")
    (OUT / "vstar.json").write_text(json.dumps({SEEDS[j]: float(vstar[j]) for j in range(n)}, indent=2))

    # PHASE D: protagonist-only outer loop; regret = max(0, V*(j) - v_pro[j]); MMD generator.
    log("=" * 60); log("PHASE D: protagonist-only loop, V*-based regret, MMD generator")
    inner_step = build_inner_step(Config(num_updates=K_INNER))
    # the generator's indices address pool_stacked while regret[j] is measured on
    # partner_list[j], so both have to stay in SEEDS order
    pool_stacked = stack_pool(pool, None)
    partner_list = [pool[s]["agent_1"] for s in SEEDS]
    # zero logits start the generator uniform, and rule mmd is the annealed update from
    # mmd_simplex_toy where vanilla Hedge cycled on a mixed regret game
    gen = PartnerGenerator(logits=np.zeros(n), rule="mmd")
    key = jax.random.PRNGKey(424242)
    k_init, key = jax.random.split(key)
    # the protagonist is initialized once and threaded through the loop, so each outer
    # step continues its training instead of restarting it
    pro = build_init_params(k_init)
    rows = []
    for step in range(N_OUTER):
        key, ks, kp, ke = jax.random.split(key, 4)
        # one partner index per env, so the generator distribution enters as the mix
        # inside a single inner step
        partner_idx = jnp.asarray(gen.sample_indices(np.asarray(ks), Config().num_envs))
        pro, _ = jax.block_until_ready(inner_step(kp, pool_stacked, partner_idx, pro))
        v_pro = np.zeros(n); regret = np.zeros(n)
        # every partner is evaluated each step, since the MMD update needs the full regret
        # vector and a sampled subset would starve the partners it stopped choosing
        for j in range(n):
            kej, ke = jax.random.split(ke)
            v, _ = eval_pair_sym(pro, partner_list[j], kej)
            v_pro[j] = v
            # the clip is the fix: the co-trained antagonist scored under the student
            # and those negative regrets mis-ranked the partners
            regret[j] = max(0.0, vstar[j] - v)
        gen.update(regret, step, N_OUTER)
        probs = gen.probs()
        rows.append({"step": step, "v_pro": v_pro.tolist(), "regret": regret.tolist(),
                     "probs": probs.tolist(), "entropy": gen.entropy()})
        if step % 5 == 0 or step == N_OUTER - 1:
            log(f"  step {step:3d}  reg={[f'{r:.1f}' for r in regret]}  probs={[f'{p:.2f}' for p in probs]}")

    p_final = gen.probs()
    # tv_to_pstar scores the generator against the phase B cross-play p*, whose ceiling
    # is the column max over the pool instead of the trained vstar above
    (OUT / "comparison_br.json").write_text(json.dumps({
        "final_generator_probs": p_final.tolist(), "p_star": p_star.tolist(),
        "tv_to_pstar": total_variation(p_final, p_star), "vstar": vstar.tolist(),
        "n_outer": N_OUTER, "K_inner": K_INNER, "br_updates": BR_UPDATES}, indent=2))
    (OUT / "ued_log_br.json").write_text(json.dumps(rows, indent=2))
    log(f"  MMD generator final: {np.round(p_final, 3)}  TV to cross-play p* {total_variation(p_final, p_star):.3f}")
    log(f"ALL DONE in {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
