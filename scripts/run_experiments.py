"""Run ALL experiments from the Notion task board:
1. Train IPPO to convergence on cramped_room (~5M steps)
2. Save 5 partner checkpoints at intervals
3. Wire discrete categorical adversary
4. Topology probe (regret per partner)
5. First A/B: MCTS vs PPO vs DR over the 5-partner pool
6. Generate all plots + curves
7. Save everything to /tmp/exp_results/

The list is the board and three of its items read differently in the code.
Training runs 1500 * 32 * 128 = 6,144,000 env steps against the ~5M of item 1.
Item 3 builds no network: ab_comparison picks among the five saved checkpoints
with a numpy bandit and nothing categorical is wired.
results/experiments/ holds the copy of /tmp/exp_results that survived. Its three
JSON files match what main writes here. Its training_curve.jsonl carries
sim_steps and return keys and came from another script.
"""
from __future__ import annotations
import json, math, os, pickle, time
from pathlib import Path
from collections import defaultdict

import flax.linen as nn
import jax
import jax.numpy as jnp
import jaxmarl
import numpy as np
import optax

# /tmp is cleared by the OS, so a run's artifacts are lost unless they are copied out
OUT = Path("/tmp/exp_results")
# this runs on import, so pulling one helper out of this module has a filesystem effect
OUT.mkdir(parents=True, exist_ok=True)

# ---- categorical helpers (no distrax) ----
# distrax 0.1.5 imports jax.typeof and the pinned jax 0.4.38 removed it
def _lsm(logits):
    return logits - jax.scipy.special.logsumexp(logits, axis=-1, keepdims=True)

def _cat_lp(logits, a):
    return jnp.take_along_axis(_lsm(logits), a[..., None], axis=-1).squeeze(-1)

def _cat_ent(logits):
    lsm = _lsm(logits)
    return -jnp.sum(jnp.exp(lsm) * lsm, axis=-1)

# ---- network ----
class AC(nn.Module):
    n_act: int; h: int = 64; nl: int = 2
    @nn.compact
    def __call__(self, x):
        # obs is an int32 (4, 5, 30) grid on cramped_room so the cast has to precede Dense
        z = x.reshape((x.shape[0], -1)).astype(jnp.float32)
        for _ in range(self.nl):
            z = nn.tanh(nn.Dense(self.h, kernel_init=nn.initializers.orthogonal(jnp.sqrt(2.0)))(z))
        # 0.01 gain leaves the initial logits nearly flat so the first updates see a
        # near-uniform policy over the six actions
        logits = nn.Dense(self.n_act, kernel_init=nn.initializers.orthogonal(0.01))(z)
        v = nn.Dense(1, kernel_init=nn.initializers.orthogonal(1.0))(z).squeeze(-1)
        return logits, v

# ---- training ----
def train_ippo(
    env, *, n_updates=1500, n_envs=32, n_steps=128, gamma=0.99, lam=0.95,
    clip_eps=0.2, ent_coef=0.01, vf_coef=0.5, lr=2.5e-4, n_epochs=4,
    n_mb=4, h=64, nl=2, seed=0, shaped_coef=1.0,
    ckpt_at=None, ckpt_dir=None,
):
    """IPPO self-play with one network per agent and no weight sharing.

    shaped_coef never anneals, so `blend` in the curve stays above `raw` for all
    1500 updates and only `raw` compares with a published delivery number.
    ckpt_at is matched against the update index. Its last entry has to be
    n_updates - 1 or the final checkpoint and the returned params disagree.
    """
    key = jax.random.PRNGKey(seed)
    obs_shape = env.observation_space().shape
    n_act = env.action_space().n
    nets = {a: AC(n_act=n_act, h=h, nl=nl) for a in env.agents}
    key, k0, k1 = jax.random.split(key, 3)
    dummy = jnp.zeros((1,) + tuple(obs_shape), dtype=jnp.float32)
    params = {"agent_0": nets["agent_0"].init(k0, dummy),
              "agent_1": nets["agent_1"].init(k1, dummy)}
    # the norm clip is upstream of adam, so 0.5 bounds the raw gradient and not the step
    # 1e-5 is the eps the PPO implementations use where optax defaults to 1e-8
    tx = optax.chain(optax.clip_by_global_norm(0.5), optax.adam(lr, eps=1e-5))
    # one tx serves both agents only because they share lr; _ippo.py builds one per agent
    opt = {a: tx.init(params[a]) for a in env.agents}

    @jax.jit
    def reset(k): return jax.vmap(env.reset)(jax.random.split(k, n_envs))
    @jax.jit
    def step(k, s, a): return jax.vmap(env.step)(jax.random.split(k, n_envs), s, a)

    def sel(net, p, o, k):
        lg, v = net.apply(p, o)
        a = jax.random.categorical(k, lg, axis=-1)
        return a, _cat_lp(lg, a), v

    key, kr = jax.random.split(key)
    obs, state = reset(kr)

    # state and obs carry across all 1500 updates with no reset in between, so the env
    # has to autoreset on done for the carried state to stay valid
    @jax.jit
    def collect(params, state, obs, key):
        def body(c, _):
            s, o, k = c
            k, ka0, ka1, ks = jax.random.split(k, 4)
            a0, lp0, v0 = sel(nets["agent_0"], params["agent_0"], o["agent_0"], ka0)
            a1, lp1, v1 = sel(nets["agent_1"], params["agent_1"], o["agent_1"], ka1)
            o2, s2, r, d, info = step(ks, s, {"agent_0": a0, "agent_1": a1})
            # sparse delivery alone gave no signal at this budget per
            # docs/overcooked_engagement.md, so the shaped channel is what moves the policy
            # r is carried beside it so the logged raw number stays free of shaping
            bl = {a: r[a] + shaped_coef * info["shaped_reward"][a] for a in env.agents}
            return (s2, o2, k), (o, {"agent_0": a0, "agent_1": a1},
                                   {"agent_0": lp0, "agent_1": lp1},
                                   {"agent_0": v0, "agent_1": v1}, bl, d, r)
        (sf, of, _), traj = jax.lax.scan(body, (state, obs, key), None, length=n_steps)
        lv = {a: nets[a].apply(params[a], of[a])[1] for a in env.agents}
        return traj, lv, sf, of

    def gae(rews, dones, vals, lv):
        T = rews.shape[0]
        # scan only runs forward, so ti walks the index back from the end and the stacked
        # output comes out reversed
        def body(c, t):
            g, nv = c; ti = T-1-t
            # cramped_room has no terminal state and every done is the 400-step limit
            # zeroing the bootstrap there teaches the critic a value that decays to 0
            # at the horizon
            nt = 1.0 - dones[ti]
            d = rews[ti] + gamma * nv * nt - vals[ti]
            g = d + gamma * lam * nt * g
            return (g, vals[ti]), g
        _, adv = jax.lax.scan(body, (jnp.zeros_like(lv), lv), jnp.arange(T))
        return adv[::-1], adv[::-1] + vals

    def mk_upd(net):
        def loss_fn(p, o, a, olp, ad, ret, ov):
            lg, v = net.apply(p, o)
            lp = _cat_lp(lg, a); r = jnp.exp(lp - olp)
            # advantages are standardized per minibatch, so n_mb moves the policy-loss scale
            an = (ad - ad.mean()) / (ad.std() + 1e-8)
            pg = -jnp.mean(jnp.minimum(r*an, jnp.clip(r, 1-clip_eps, 1+clip_eps)*an))
            # clip_eps bounds the probability ratio and the per-update value move alike
            # past 0.2 the maximum below picks the clipped term whose gradient is zero, so
            # the value head moves at most 0.2 per update while one delivery is worth 40
            vc = ov + jnp.clip(v-ov, -clip_eps, clip_eps)
            vl = jnp.mean(0.5*jnp.maximum((v-ret)**2, (vc-ret)**2))
            return pg + vf_coef*vl - ent_coef*jnp.mean(_cat_ent(lg))
        gf = jax.value_and_grad(loss_fn)
        # os here is the optimizer state and shadows the os module imported at the top
        @jax.jit
        def upd(p, os, o, a, olp, ad, ret, ov, k):
            # bs is n_steps * n_envs and has to divide by n_mb or the reshape below raises
            bs = o.shape[0]; mb = bs // n_mb
            def ep(c, _):
                p, os, k = c; k, kp = jax.random.split(k)
                ix = jax.random.permutation(kp, bs).reshape(n_mb, mb)
                def step(c, ix):
                    p, os = c
                    _, g = gf(p, o[ix], a[ix], olp[ix], ad[ix], ret[ix], ov[ix])
                    u, os = tx.update(g, os, p); p = optax.apply_updates(p, u)
                    return (p, os), None
                (p, os), _ = jax.lax.scan(step, (p, os), ix)
                return (p, os, k), None
            (p, os, _), _ = jax.lax.scan(ep, (p, os, k), None, length=n_epochs)
            return p, os
        return upd

    upd_fns = {a: mk_upd(nets[a]) for a in env.agents}
    curve = []
    ckpts_saved = {}
    t0 = time.time()
    print(f"training: {n_updates} updates, {n_envs} envs, {n_steps} steps = {n_updates*n_envs*n_steps:,} env steps")
    for u in range(n_updates):
        key, kc = jax.random.split(key)
        traj, lv, state, obs = collect(params, state, obs, kc)
        o_t, a_t, lp_t, v_t, bl_t, d_t, r_t = traj
        for ag in env.agents:
            ad, ret = gae(bl_t[ag], d_t["__all__"].astype(jnp.float32), v_t[ag], lv[ag])
            fl = lambda x: x.reshape((-1,)+x.shape[2:])
            key, ku = jax.random.split(key)
            params[ag], opt[ag] = upd_fns[ag](params[ag], opt[ag],
                fl(o_t[ag]), fl(a_t[ag]), fl(lp_t[ag]), fl(ad), fl(ret), fl(v_t[ag]), ku)
        # both agents are paid the same delivery reward and both are added, so a delivery
        # counts 40 in every number this file reports
        raw = float(r_t["agent_0"].sum() + r_t["agent_1"].sum()) / n_envs
        blend = float(bl_t["agent_0"].sum() + bl_t["agent_1"].sum()) / n_envs
        # the window is 128 steps against a 400-step episode, so this runs below the
        # episode return that the make_plots y label claims
        curve.append({"u": u, "steps": (u+1)*n_envs*n_steps, "raw": raw, "blend": blend})
        if u % 50 == 0 or u == n_updates-1:
            el = time.time()-t0
            print(f"  u={u:5d}  steps={curve[-1]['steps']:>8,}  raw={raw:7.2f}  blend={blend:7.2f}  ({el:.0f}s)")
        if ckpt_at and u in ckpt_at and ckpt_dir:
            fp = Path(ckpt_dir) / f"ckpt_u{u:05d}.pkl"
            with open(fp, "wb") as f:
                # np.asarray first so the pickle holds host arrays and no jax device handle
                pickle.dump({"params": jax.tree.map(np.asarray, params), "update": u,
                             "env_steps": (u+1)*n_envs*n_steps}, f)
            ckpts_saved[u] = str(fp)
            print(f"    -> saved {fp.name}")
    print(f"done in {time.time()-t0:.0f}s")
    return params, curve, ckpts_saved


def eval_policy(env, params, nets, n_eps, key, stoch=False):
    """Stepped one env at a time outside jit. The 5x5 probe at 8 episodes per cell
    dispatches on the order of 80,000 single-env calls this way.

    With stoch=False the only randomness left is env.reset. The committed run returned
    480.0 on all 16 greedy episodes at std 0, so cramped_room resets deterministically
    and n_eps buys nothing on that branch.
    """
    rets = []
    for _ in range(n_eps):
        key, ks = jax.random.split(key)
        o, s = env.reset(ks)
        tot = 0.0
        # the horizon comes from the env and no caller argument can shorten it
        for _ in range(env.max_steps):
            acts = {}
            for ag in env.agents:
                lg, _ = nets[ag].apply(params[ag], o[ag][None])
                if stoch:
                    key, k = jax.random.split(key)
                    acts[ag] = jax.random.categorical(k, lg, axis=-1)[0]
                else:
                    acts[ag] = jnp.argmax(lg, axis=-1)[0]
            key, ks = jax.random.split(key)
            o, s, r, d, _ = env.step(ks, s, acts)
            # each float() forces a device sync, which is most of the cost of this loop
            tot += float(r["agent_0"]) + float(r["agent_1"])
            # the episode always runs the full 400 steps, so this only fires at the horizon
            if bool(d["__all__"]): break
        rets.append(tot)
    return {"mean": float(np.mean(rets)), "max": float(np.max(rets)),
            "std": float(np.std(rets)), "returns": [float(r) for r in rets]}


def topology_probe(env, ckpt_paths, nets, n_eval=8):
    """Return of the focal's agent_0 half paired with the partner's agent_1 half.

    The halves are never swapped, so the matrix is asymmetric by construction. In the
    committed run 1199 against 1499 reads 345 while its transpose reads 315.
    The inner loop reopens pj on every pass, so five files are unpickled 25 times.
    """
    key = jax.random.PRNGKey(42)
    # the names are unpadded, so sorted() gives 1199 1499 299 599 899 and neither this
    # matrix nor the heatmap built from it follows training order
    names = sorted(ckpt_paths.keys())
    matrix = {}
    for i, ni in enumerate(names):
        with open(ckpt_paths[ni], "rb") as f:
            pi = pickle.load(f)["params"]
        pi = jax.tree.map(jnp.asarray, pi)
        for j, nj in enumerate(names):
            with open(ckpt_paths[nj], "rb") as f:
                pj = pickle.load(f)["params"]
            pj = jax.tree.map(jnp.asarray, pj)
            mixed = {"agent_0": pi["agent_0"], "agent_1": pj["agent_1"]}
            key, k = jax.random.split(key)
            ev = eval_policy(env, mixed, nets, n_eval, k, stoch=True)
            matrix[f"{ni}_vs_{nj}"] = ev["mean"]
            print(f"  focal={ni} partner={nj} -> {ev['mean']:.1f}")
    return matrix


def ab_comparison(env, ckpt_paths, nets, n_outer=50, seed=99):
    """Three adversary strategies choosing among the five saved checkpoints.

    What they maximize is the spread of the focal policy's own stochastic return under
    one partner. `best_ret` below is the max of the same four episodes the mean came
    from, so no partner can be preferred for being hard and the committed run separates
    the three by less than one delivery: 41.6 for mcts_ucb against 44.0 and 43.6.
    """
    key = jax.random.PRNGKey(seed)
    names = sorted(ckpt_paths.keys())
    all_params = []
    for n in names:
        with open(ckpt_paths[n], "rb") as f:
            all_params.append(jax.tree.map(jnp.asarray, pickle.load(f)["params"]))

    results = {}
    for strategy in ["mcts_ucb", "ppo_gradient", "domain_rand"]:
        regrets = []
        partner_choices = []
        # 1e-8 keeps the first division finite and hands an unpulled arm a bonus near 1e4
        arm_values = np.zeros(len(names))
        arm_counts = np.zeros(len(names)) + 1e-8
        # every entry equal means the first ppo_gradient draw is uniform whatever the value
        mu = np.full(len(names), 0.5)

        for step in range(n_outer):
            if strategy == "domain_rand":
                # the generator is rebuilt per step, so this sequence follows seed alone
                # and never the global numpy state
                idx = int(np.random.RandomState(seed + step).randint(len(names)))
            elif strategy == "mcts_ucb":
                # log(1) is 0 so step 0 ties every arm and the next four steps pull the
                # unvisited arms in index order
                # 1.4 is the sqrt(2) UCT constant and the bonus it buys stays under 3
                # the arms differ by tens of reward, so this is greedy after those five
                ucb = arm_values / arm_counts + 1.4 * np.sqrt(np.log(step+1) / arm_counts)
                idx = int(np.argmax(ucb))
            elif strategy == "ppo_gradient":
                probs = np.exp(mu) / np.exp(mu).sum()
                idx = int(np.random.RandomState(seed + step).choice(len(names), p=probs))

            partner_p = all_params[idx]
            # names sort lexicographically, so all_params[-1] is the update-899 checkpoint
            # and partner_1499 never plays the focal role
            focal_p = all_params[-1]  # strongest (last ckpt) as focal
            mixed = {"agent_0": focal_p["agent_0"], "agent_1": partner_p["agent_1"]}
            key, k = jax.random.split(key)
            ev = eval_policy(env, mixed, nets, 4, k, stoch=True)

            # topology is never read here. the anchor is the max of the same four episodes
            # the mean came from, so this is a within-pairing spread and the max below can
            # never fire
            best_ret = max(ev["returns"])
            regret = max(0, best_ret - ev["mean"])
            regrets.append(regret)
            partner_choices.append(names[idx])

            # all three strategies write these and only mcts_ucb reads them back
            arm_values[idx] += regret
            arm_counts[idx] += 1
            if strategy == "ppo_gradient":
                grad = np.zeros(len(names))
                # regrets already holds this step's value, so the baseline contains the
                # sample it is subtracted from and step 0 gives a gradient of exactly zero
                grad[idx] = regret - np.mean(regrets)
                # 0.1 times a regret in the tens moves mu by whole numbers, so the clip
                # below is reached within a few outer steps
                mu += 0.1 * grad
                # this caps the ratio between the most and least likely arm at e**6
                mu = np.clip(mu, -3, 3)

        results[strategy] = {
            "mean_regret": float(np.mean(regrets)),
            # unary minus binds before the floor divide, so -50 // 4 is -13
            # the last-quarter window here is 13 outer steps wide
            "last_quarter_regret": float(np.mean(regrets[-n_outer//4:])),
            "partner_distribution": {n: partner_choices.count(n) for n in names},
            "regrets": [float(r) for r in regrets],
        }
        print(f"  {strategy:15s}  mean_regret={results[strategy]['mean_regret']:.2f}  "
              f"last_q={results[strategy]['last_quarter_regret']:.2f}  "
              f"dist={results[strategy]['partner_distribution']}")
    return results


def make_plots(curve, topo, ab, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    # fonttype 42 embeds TrueType so the pdf keeps selectable text
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42})

    # 1. Training curve
    xs = [c["steps"] for c in curve]
    raw = [c["raw"] for c in curve]
    fig, ax = plt.subplots(figsize=(5.4, 3.0))
    ax.plot(xs, raw, color="#324A5F", linewidth=1.0, alpha=0.3, label="per-update")
    # 50 updates is 204,800 env steps of smoothing, which is the entire budget of the
    # earlier run in docs/overcooked_engagement.md
    w = 50
    # under 51 updates the smoothed line is dropped and only the faint raw line draws
    if len(raw) > w:
        ra = np.convolve(raw, np.ones(w)/w, mode="valid")
        ax.plot(xs[w-1:], ra, color="#324A5F", linewidth=1.4, label=f"{w}-update avg")
    ax.set_xlabel("env steps")
    ax.set_ylabel("episode return (raw delivery)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    for ext in ("pdf", "png"): fig.savefig(out / f"training_curve.{ext}")
    plt.close(fig)

    # 2. Topology heatmap
    # the split holds while no partner name contains _vs_ and partner_{u} never does
    names = sorted(set(k.split("_vs_")[0] for k in topo))
    # a missing pair reads 0 and a real zero return reads 0, so a hole is invisible here
    mat = np.array([[topo.get(f"{i}_vs_{j}", 0) for j in names] for i in names])
    fig, ax = plt.subplots(figsize=(4, 3.5))
    im = ax.imshow(mat, cmap="magma")
    ax.set_xticks(range(len(names))); ax.set_yticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels(names, fontsize=7)
    ax.set_xlabel("partner"); ax.set_ylabel("focal")
    for i in range(len(names)):
        for j in range(len(names)):
            # magma is near-black at the low end, so the light text goes on the low cells
            ax.text(j, i, f"{mat[i,j]:.0f}", ha="center", va="center",
                    color="white" if mat[i,j] < mat.max()*0.6 else "black", fontsize=7)
    fig.colorbar(im, ax=ax, shrink=0.7)
    fig.tight_layout()
    for ext in ("pdf", "png"): fig.savefig(out / f"topology_heatmap.{ext}")
    plt.close(fig)

    # 3. A/B regret comparison
    fig, ax = plt.subplots(figsize=(5.4, 3.0))
    colors = {"mcts_ucb": "#324A5F", "ppo_gradient": "#B07A50", "domain_rand": "#888"}
    for strat, d in ab.items():
        regs = d["regrets"]
        w2 = min(10, len(regs))
        ra2 = np.convolve(regs, np.ones(w2)/w2, mode="valid")
        # valid convolution drops the first w2-1 points, so x=0 here is outer step 9
        ax.plot(range(len(ra2)), ra2, color=colors.get(strat, "#000"),
                linewidth=1.4, label=strat)
    ax.set_xlabel("outer step"); ax.set_ylabel("regret (smoothed)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    for ext in ("pdf", "png"): fig.savefig(out / f"ab_comparison.{ext}")
    plt.close(fig)

    print(f"plots saved to {out}")


def main():
    print("="*60)
    print("PARTNER UED: FULL EXPERIMENT SUITE")
    print("="*60)

    env = jaxmarl.make("overcooked_v2", layout="cramped_room")
    n_act = env.action_space().n
    # h and nl take their defaults here while train_ippo takes its own, so passing either
    # to train_ippo writes checkpoints that these two modules cannot apply
    nets = {a: AC(n_act=n_act) for a in env.agents}

    ckpt_dir = OUT / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)

    n_updates = 1500
    # every 300 updates. the last entry has to stay n_updates - 1 or the final checkpoint
    # and the params train_ippo returns are two different policies
    ckpt_updates = [299, 599, 899, 1199, 1499]

    print("\n[1/6] IPPO self-play training to convergence")
    params, curve, ckpts = train_ippo(
        env, n_updates=n_updates, n_envs=32, n_steps=128, seed=0,
        ckpt_at=set(ckpt_updates), ckpt_dir=str(ckpt_dir),
    )
    (OUT / "training_curve.jsonl").write_text("\n".join(json.dumps(c) for c in curve))

    print("\n[2/6] eval converged policy")
    key = jax.random.PRNGKey(77)
    ev_stoch = eval_policy(env, params, nets, 16, key, stoch=True)
    key, k = jax.random.split(key)
    # greedy read 0.00 against 35.00 stochastic at 200 updates in
    # docs/overcooked_engagement.md and the order inverts here: 480 greedy against 365
    ev_greedy = eval_policy(env, params, nets, 16, k, stoch=False)
    summary = {"stochastic": ev_stoch, "greedy": ev_greedy,
               # 32 and 128 are repeated from the call above, so a different rollout shape
               # there mislabels this field without raising
               "total_env_steps": n_updates * 32 * 128,
               "n_updates": n_updates}
    (OUT / "eval_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"  stochastic mean={ev_stoch['mean']:.1f}  greedy mean={ev_greedy['mean']:.1f}")

    print("\n[3/6] 5 partner checkpoints saved")
    ckpt_paths = {}
    for u in sorted(ckpts.keys()):
        # the pickle file pads the update to five digits and this name does not, so every
        # sorted() downstream puts 1199 ahead of 299
        name = f"partner_{u}"
        ckpt_paths[name] = ckpts[u]
        print(f"  {name} -> {ckpts[u]}")

    print("\n[4/6] topology probe (5x5 focal-partner matrix)")
    topo = topology_probe(env, ckpt_paths, nets, n_eval=8)
    (OUT / "topology.json").write_text(json.dumps(topo, indent=2))

    print("\n[5/6] A/B: MCTS vs PPO vs DR over 5-partner pool")
    ab = ab_comparison(env, ckpt_paths, nets, n_outer=50, seed=99)
    (OUT / "ab_comparison.json").write_text(json.dumps(ab, indent=2))

    print("\n[6/6] plots")
    make_plots(curve, topo, ab, OUT)

    print("\n" + "="*60)
    print("ALL EXPERIMENTS COMPLETE")
    print(f"artifacts: {OUT}")
    print("="*60)


if __name__ == "__main__":
    main()
