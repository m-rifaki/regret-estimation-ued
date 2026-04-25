"""Play with overcooked v2 - real env load, random rollouts, and a
minimal self-play IPPO trainer. Five files per run, all under --out:

    random_baseline.json
    ippo_curve.jsonl
    ippo_eval.json
    episode_trace.json
    summary.json

The --out default is under /tmp and nothing here writes to
results/overcooked_v2. The copies committed there were placed by hand from
the 200-update seed-0 run that docs/overcooked_engagement.md quotes.

Run:
    python play_overcooked.py --layout cramped_room
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import flax.linen as nn
import jax
import jax.numpy as jnp
import jaxmarl
import numpy as np
import optax


# ----- categorical helpers --------------------------------------------------

# distrax-0.1.5 imports jax.typeof and the pinned jax 0.4.38 removed it, so the
# categorical is hand-rolled on logsumexp plus jax.random.categorical
def log_softmax(logits):
    return logits - jax.scipy.special.logsumexp(logits, axis=-1, keepdims=True)


def cat_log_prob(logits, action):
    lsm = log_softmax(logits)
    return jnp.take_along_axis(lsm, action[..., None], axis=-1).squeeze(-1)


def cat_entropy(logits):
    # the sum runs over all six actions, so the bonus is the exact policy entropy
    # and carries no sampling noise into the gradient
    lsm = log_softmax(logits)
    probs = jnp.exp(lsm)
    return -jnp.sum(probs * lsm, axis=-1)


def cat_sample(logits, key):
    return jax.random.categorical(key, logits, axis=-1)


# ----- network --------------------------------------------------------------

class ActorCritic(nn.Module):
    n_actions: int
    hidden: int = 64
    n_layers: int = 2

    @nn.compact
    def __call__(self, x):
        # obs is an int32 (4, 5, 30) grid on cramped_room, so the cast has to come
        # before the first Dense. flattening to 600 inputs drops the neighborhood
        z = x.reshape((x.shape[0], -1)).astype(jnp.float32)
        for _ in range(self.n_layers):
            # sqrt(2) is the orthogonal gain that holds tanh activation scale across layers
            z = nn.tanh(nn.Dense(self.hidden, kernel_init=nn.initializers.orthogonal(jnp.sqrt(2.0)))(z))
        # 0.01 flattens the initial logits so the first rollouts sample near-uniformly
        logits = nn.Dense(self.n_actions, kernel_init=nn.initializers.orthogonal(0.01))(z)
        value = nn.Dense(1, kernel_init=nn.initializers.orthogonal(1.0))(z).squeeze(-1)
        return logits, value


# ----- vectorised env -------------------------------------------------------

def make_vec(env, n_envs: int):
    # n_envs is closed over by both, so reset and step are valid only as the pair
    # returned here. one key goes in and n_envs streams come out, so no two envs
    # share a draw
    @jax.jit
    def reset(key):
        keys = jax.random.split(key, n_envs)
        return jax.vmap(env.reset)(keys)

    @jax.jit
    def step(key, state, actions):
        keys = jax.random.split(key, n_envs)
        return jax.vmap(env.step)(keys, state, actions)

    return reset, step


# ----- random baseline ------------------------------------------------------

def random_baseline(env, n_episodes: int, key) -> dict:
    # unjitted env.step driven from python: 16 episodes of 400 dispatches each. this is
    # the null arm so the cost never shows against training
    # the committed run scores 0.0 on all 16 episodes at length 400, so a random pair
    # never delivers on cramped_room and the horizon is the only end
    returns = []
    lens = []
    for ep in range(n_episodes):
        key, ks = jax.random.split(key)
        obs, state = env.reset(ks)
        total = 0.0
        steps = 0
        for t in range(env.max_steps):
            key, ks, ka1, ka2 = jax.random.split(key, 4)
            actions = {
                "agent_0": env.action_space().sample(ka1),
                "agent_1": env.action_space().sample(ka2),
            }
            obs, state, reward, done, info = env.step(ks, state, actions)
            total += float(reward["agent_0"]) + float(reward["agent_1"])
            steps += 1
            if bool(done["__all__"]):
                break
        returns.append(total)
        lens.append(steps)
    return {
        "n_episodes": n_episodes,
        "mean_return": float(np.mean(returns)),
        "median_return": float(np.median(returns)),
        "max_return": float(np.max(returns)),
        "mean_length": float(np.mean(lens)),
        "returns": [float(r) for r in returns],
    }


# ----- IPPO -----------------------------------------------------------------

# the defaults for n_updates and n_envs never run: main passes its own on every call.
# 16 envs by 64 steps over 200 updates is the shape behind docs/overcooked_engagement.md
def train_ippo(
    env,
    layout: str,
    *,
    n_updates: int = 80,
    n_envs: int = 32,
    n_steps: int = 64,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    clip_eps: float = 0.2,
    ent_coef: float = 0.01,
    vf_coef: float = 0.5,
    lr: float = 2.5e-4,
    n_epochs: int = 4,
    n_minibatches: int = 4,
    hidden: int = 64,
    n_layers: int = 2,
    seed: int = 0,
    shaped_coef: float = 1.0,
):
    # shaped_coef stays at 1.0 for every update since main never passes it. the sparse
    # channel alone gives no signal at this budget, and the blended curve keeps a term
    # the eval never pays
    key = jax.random.PRNGKey(seed)
    obs_shape = env.observation_space().shape
    n_actions = env.action_space().n

    # separate params per agent is the I in IPPO. the two nets are built identically so
    # the init key and each net's own gradients are the whole difference between the seats
    nets = {agent: ActorCritic(n_actions=n_actions, hidden=hidden, n_layers=n_layers) for agent in env.agents}

    key, *init_keys = jax.random.split(key, 1 + len(env.agents))
    params = {}
    for agent, ikey in zip(env.agents, init_keys):
        params[agent] = nets[agent].init(ikey, jnp.zeros((1,) + tuple(obs_shape), dtype=jnp.float32))

    # chain order puts the 0.5 norm clip on the raw gradient upstream of adam's rescaling.
    # eps 1e-5 is what the PPO implementations use and optax's own default is 1e-8
    tx = optax.chain(optax.clip_by_global_norm(0.5), optax.adam(lr, eps=1e-5))
    # one tx serves both agents since an optax transform holds no state of its own
    opt_states = {agent: tx.init(params[agent]) for agent in env.agents}

    reset_fn, step_fn = make_vec(env, n_envs)
    key, kr = jax.random.split(key)
    obs, state = reset_fn(kr)

    # ----- per-agent rollout, GAE, update ---------------------------------

    def select_action(net, p, o, k):
        logits, val = net.apply(p, o)
        a = cat_sample(logits, k)
        logp = cat_log_prob(logits, a)
        return a, logp, val

    @jax.jit
    def collect(params, state, obs, key):
        # params comes in as an argument so the update loop can hand over new weights
        # without retracing collect
        def body(carry, _):
            state, obs, key = carry
            key, *ak = jax.random.split(key, 1 + len(env.agents))
            actions, logps, vals = {}, {}, {}
            for agent, k in zip(env.agents, ak):
                a, lp, v = select_action(nets[agent], params[agent], obs[agent], k)
                actions[agent] = a
                logps[agent] = lp
                vals[agent] = v
            key, kstep = jax.random.split(key)
            obs2, state2, reward, done, info = step_fn(kstep, state, actions)
            # GAE runs on the blend and the raw channel goes to logging alone, so the
            # logged curve and the eval numbers stay in the same units
            blended = {
                agent: reward[agent] + shaped_coef * info["shaped_reward"][agent]
                for agent in env.agents
            }
            return (state2, obs2, key), (obs, actions, logps, vals, blended, done, reward)

        (state_f, obs_f, key_f), traj = jax.lax.scan(
            body, (state, obs, key), None, length=n_steps
        )

        # bootstrap value at final obs. env.step auto-resets, so obs_f after a done
        # already belongs to the next episode and the nonterm factor in compute_gae is
        # what keeps this value on the right side of the boundary
        last_vals = {}
        for agent in env.agents:
            _, v = nets[agent].apply(params[agent], obs_f[agent])
            last_vals[agent] = v
        return traj, last_vals, state_f, obs_f, key_f

    # nothing calls this. collect bootstraps from nets[agent].apply inline
    def _apply_value(p, obs, net):
        _, v = net.apply(p, obs)
        return v

    def compute_gae(rewards, dones, values, last_value):
        # rewards, dones, values shape: (T, n_envs)
        # dones is __all__ for both agents and the only done on this layout is the
        # 400-step timeout, so nonterm zeroes a bootstrap a truncation should have kept
        T = rewards.shape[0]
        advantages = jnp.zeros_like(rewards)
        gae = jnp.zeros_like(last_value)

        def body(carry, t):
            gae, next_val = carry
            t_idx = T - 1 - t
            nonterm = 1.0 - dones[t_idx]
            delta = rewards[t_idx] + gamma * next_val * nonterm - values[t_idx]
            gae = delta + gamma * gae_lambda * nonterm * gae
            return (gae, values[t_idx]), gae

        _, advs = jax.lax.scan(body, (gae, last_value), jnp.arange(T))
        # the scan walks t backward from T-1 so advs comes out reversed. the flip has to
        # precede the add below or advantages and values disagree about time
        advs = advs[::-1]
        returns = advs + values
        return advs, returns

    def make_update_fn(net):
        def loss_fn(params, obs, actions, old_logps, advs, returns, old_vals):
            logits, vals = net.apply(params, obs)
            logps = cat_log_prob(logits, actions)
            # old_logps were recorded under the params collect ran with, so the first
            # minibatch of the first epoch has ratio exactly 1
            ratio = jnp.exp(logps - old_logps)
            # mean and std come from the 256 samples of this minibatch. the full
            # 1024-sample batch is never normalized as a whole
            adv_norm = (advs - advs.mean()) / (advs.std() + 1e-8)
            unclipped = ratio * adv_norm
            clipped = jnp.clip(ratio, 1 - clip_eps, 1 + clip_eps) * adv_norm
            pg_loss = -jnp.mean(jnp.minimum(unclipped, clipped))
            # clip_eps does double duty. the same 0.2 bounds a probability ratio above
            # and a value in return units here
            v_clipped = old_vals + jnp.clip(vals - old_vals, -clip_eps, clip_eps)
            v_loss = jnp.mean(0.5 * jnp.maximum((vals - returns) ** 2, (v_clipped - returns) ** 2))
            ent = jnp.mean(cat_entropy(logits))
            loss = pg_loss + vf_coef * v_loss - ent_coef * ent
            return loss, (pg_loss, v_loss, ent)

        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)

        @jax.jit
        def update_agent(params, opt_state, obs, actions, old_logps, advs, returns, old_vals, key):
            # batch_size has to divide by n_minibatches or the permutation reshape
            # raises. 16 envs over 64 steps into 4 minibatches gives 256
            batch_size = obs.shape[0]
            mb_size = batch_size // n_minibatches

            def epoch_body(carry, _):
                params, opt_state, key = carry
                key, kp = jax.random.split(key)
                idx_e = jax.random.permutation(kp, batch_size).reshape(n_minibatches, mb_size)

                def mb_body(carry, mb):
                    params, opt_state = carry
                    (loss, (pg, vl, en)), grads = grad_fn(
                        params,
                        obs[mb], actions[mb], old_logps[mb], advs[mb], returns[mb], old_vals[mb],
                    )
                    updates, opt_state = tx.update(grads, opt_state, params)
                    params = optax.apply_updates(params, updates)
                    return (params, opt_state), (loss, pg, vl, en)

                (params, opt_state), losses = jax.lax.scan(mb_body, (params, opt_state), idx_e)
                return (params, opt_state, key), losses

            (params, opt_state, _), losses = jax.lax.scan(
                epoch_body, (params, opt_state, key), None, length=n_epochs
            )
            # the mean covers both scan axes, so one logged number averages 4 epochs of
            # 4 minibatches taken at different params
            return params, opt_state, jax.tree.map(jnp.mean, losses)

        return update_agent

    update_fns = {agent: make_update_fn(nets[agent]) for agent in env.agents}

    curve = []
    t_start = time.time()
    print(f"training IPPO on {layout}: n_updates={n_updates} n_envs={n_envs} n_steps={n_steps}")
    for upd in range(n_updates):
        key, kc = jax.random.split(key)
        traj, last_vals, state, obs, _ = collect(params, state, obs, kc)
        obs_t, actions_t, logps_t, vals_t, blended_t, dones_t, raw_t = traj
        per_agent_loss_info = {}
        for agent in env.agents:
            adv, ret = compute_gae(
                blended_t[agent],
                dones_t["__all__"].astype(jnp.float32),
                vals_t[agent],
                last_vals[agent],
            )
            flat = lambda x: x.reshape((-1,) + x.shape[2:])  # noqa: E731
            o = flat(obs_t[agent])
            a = flat(actions_t[agent])
            lp = flat(logps_t[agent])
            ad = flat(adv)
            re = flat(ret)
            v = flat(vals_t[agent])
            key, ku = jax.random.split(key)
            params[agent], opt_states[agent], losses = update_fns[agent](
                params[agent], opt_states[agent],
                o, a, lp, ad, re, v, ku,
            )
            per_agent_loss_info[agent] = {k: float(v) for k, v in zip(["loss", "pg", "v", "ent"], losses)}

        # a 64-step window summed over the batch and this is not an episode return.
        # scale by 400 / 64 to read it against the eval numbers
        # one delivery anywhere in the batch moves ep_raw by 40 / 16, so every recorded
        # value is a multiple of 2.5
        ep_raw = float(raw_t["agent_0"].sum() + raw_t["agent_1"].sum()) / n_envs
        ep_blend = float(blended_t["agent_0"].sum() + blended_t["agent_1"].sum()) / n_envs
        curve.append({
            "update": upd,
            "env_steps": (upd + 1) * n_envs * n_steps,
            "rollout_raw_return": ep_raw,
            "rollout_blended_return": ep_blend,
            "agent_0_loss": per_agent_loss_info["agent_0"]["loss"],
            "agent_1_loss": per_agent_loss_info["agent_1"]["loss"],
        })
        if upd % 10 == 0 or upd == n_updates - 1:
            elapsed = time.time() - t_start
            print(f"  upd={upd:4d}  env_steps={curve[-1]['env_steps']:6d}  raw={ep_raw:6.2f}  blend={ep_blend:6.2f}  "
                  f"loss[a0]={per_agent_loss_info['agent_0']['loss']:.3f}  "
                  f"loss[a1]={per_agent_loss_info['agent_1']['loss']:.3f}  ({elapsed:.1f}s)")
    print(f"training done in {time.time() - t_start:.1f}s")
    return params, curve


def eval_policy(env, params, nets, n_episodes, key, *, greedy: bool = True) -> dict:
    # greedy scored 0.00 on the committed seed while the same params sampled 35.00.
    # argmax parks both agents on the stay action for all 400 steps of episode_trace.json
    returns = []
    for _ep in range(n_episodes):
        key, ks = jax.random.split(key)
        obs, state = env.reset(ks)
        total = 0.0
        for _t in range(env.max_steps):
            actions = {}
            for agent in env.agents:
                # nothing vmaps on this path, so the obs needs a batch axis for the
                # reshape inside ActorCritic
                logits, _ = nets[agent].apply(params[agent], obs[agent][None])
                if greedy:
                    a = jnp.argmax(logits, axis=-1)[0]
                else:
                    key, k = jax.random.split(key)
                    a = cat_sample(logits, k)[0]
                actions[agent] = a
            key, ks = jax.random.split(key)
            obs, state, reward, done, _info = env.step(ks, state, actions)
            # reward is shared, so one delivery pays 20 to each agent and 40 into this
            # sum. the committed stochastic returns are all 0 or 40 or 80
            total += float(reward["agent_0"]) + float(reward["agent_1"])
            if bool(done["__all__"]):
                break
        returns.append(total)
    return {
        "n_episodes": n_episodes,
        "mean_return": float(np.mean(returns)),
        "median_return": float(np.median(returns)),
        "max_return": float(np.max(returns)),
        "returns": [float(r) for r in returns],
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--layout", default="cramped_room")
    p.add_argument("--n-random", type=int, default=16)
    # the committed run used 200 updates for 204800 env steps. the default 80 stops
    # at 81920
    p.add_argument("--n-updates", type=int, default=80)
    p.add_argument("--n-envs", type=int, default=16)
    p.add_argument("--n-steps", type=int, default=64)
    p.add_argument("--n-eval", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    # /tmp by default, so a run leaves nothing in the repo until the files are copied
    p.add_argument("--out", type=Path, default=Path("/tmp/overcooked_results"))
    args = p.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    env = jaxmarl.make("overcooked_v2", layout=args.layout)
    key = jax.random.PRNGKey(args.seed)

    print(f"=== overcooked_v2 / {args.layout} ===")
    print(f"  agents: {env.agents}")
    print(f"  obs shape: {env.observation_space().shape}")
    print(f"  actions: {env.action_space().n}")
    print(f"  max_steps: {env.max_steps}")

    print("\n=== random baseline ===")
    key, k = jax.random.split(key)
    rand_baseline = random_baseline(env, args.n_random, k)
    print(f"  random mean return = {rand_baseline['mean_return']:.3f}")
    (args.out / "random_baseline.json").write_text(json.dumps(rand_baseline, indent=2))

    print("\n=== IPPO self-play ===")
    params, curve = train_ippo(
        env, args.layout,
        n_updates=args.n_updates, n_envs=args.n_envs, n_steps=args.n_steps,
        seed=args.seed,
    )
    # the join leaves no trailing newline, so wc -l reports 199 on the committed
    # 200-record file
    (args.out / "ippo_curve.jsonl").write_text(
        "\n".join(json.dumps(c) for c in curve)
    )

    print("\n=== IPPO eval ===")
    # hidden and n_layers here have to match what train_ippo ran with or apply raises
    # on the param shapes. both sides are on ActorCritic's defaults today
    nets = {a: ActorCritic(n_actions=env.action_space().n) for a in env.agents}
    key, k = jax.random.split(key)
    eval_greedy = eval_policy(env, params, nets, args.n_eval, k, greedy=True)
    key, k = jax.random.split(key)
    eval_stoch = eval_policy(env, params, nets, args.n_eval, k, greedy=False)
    print(f"  greedy mean return = {eval_greedy['mean_return']:.3f}")
    print(f"  stoch  mean return = {eval_stoch['mean_return']:.3f}")
    (args.out / "ippo_eval.json").write_text(json.dumps({
        "greedy": eval_greedy, "stochastic": eval_stoch
    }, indent=2))

    # one greedy episode step by step. every a0 and a1 in the committed file is 5
    key, k = jax.random.split(key)
    obs, state = env.reset(k)
    trace = []
    total = 0.0
    for t in range(env.max_steps):
        actions = {}
        for agent in env.agents:
            logits, _ = nets[agent].apply(params[agent], obs[agent][None])
            actions[agent] = int(jnp.argmax(logits, axis=-1)[0])
        key, k = jax.random.split(key)
        obs, state, reward, done, _info = env.step(k, state, actions)
        r0 = float(reward["agent_0"]); r1 = float(reward["agent_1"])
        total += r0 + r1
        trace.append({
            "t": t,
            "a0": actions["agent_0"],
            "a1": actions["agent_1"],
            "r0": r0,
            "r1": r1,
        })
        if bool(done["__all__"]):
            break
    (args.out / "episode_trace.json").write_text(json.dumps({
        "total_return": total,
        "length": len(trace),
        "steps": trace,
    }, indent=2))
    print(f"  greedy episode trace: len={len(trace)} return={total:.3f}")

    summary = {
        "layout": args.layout,
        "random_mean": rand_baseline["mean_return"],
        "ippo_greedy_mean": eval_greedy["mean_return"],
        "ippo_stoch_mean": eval_stoch["mean_return"],
        # greedy is the arm that collapses, so this key reads 0.0 on the committed run
        # while the stochastic arm reached 35.0
        "lift_over_random_greedy": eval_greedy["mean_return"] - rand_baseline["mean_return"],
        "n_updates": args.n_updates,
        "n_envs": args.n_envs,
        "n_steps": args.n_steps,
        "total_env_steps": args.n_updates * args.n_envs * args.n_steps,
        "seed": args.seed,
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
    print("\n=== summary ===")
    for k, v in summary.items():
        print(f"  {k} = {v}")


if __name__ == "__main__":
    main()
