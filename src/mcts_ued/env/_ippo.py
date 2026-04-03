"""IPPO training loop for one partner-spec proposal on overcooked v2.

Self-contained: builds two independent actor-critic networks (one per
agent), collects vectorised rollouts via `jax.lax.scan`, computes GAE,
and applies PPO-clipped updates with entropy bonus and value loss.
The PartnerSpec's decode_kwargs drives the *partner's* network width,
depth, learning rate, training budget, and action-sampling temperature.
The focal-side recipe is fixed by the cfg.

The advantage is computed on the sparse delivery reward plus JaxMARL's
`info["shaped_reward"]` channel. The number this module hands back to
the regret estimator is the sparse channel alone.

Why not distrax: the current pinned jax (0.4.38+) drops `jax.typeof`
which the released distrax-0.1.5 still imports. We use a raw
categorical distribution implemented with `jax.scipy.special.logsumexp`
+ `jax.random.categorical`.
"""
from __future__ import annotations

from typing import Any, Literal

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax

from mcts_ued.partners.parametrize import PartnerSpec

# ---------- categorical helpers (distrax-free) -----------------------------

def _log_softmax(logits):
    return logits - jax.scipy.special.logsumexp(logits, axis=-1, keepdims=True)


def _cat_log_prob(logits, action):
    lsm = _log_softmax(logits)
    return jnp.take_along_axis(lsm, action[..., None], axis=-1).squeeze(-1)


def _cat_entropy(logits):
    lsm = _log_softmax(logits)
    probs = jnp.exp(lsm)
    return -jnp.sum(probs * lsm, axis=-1)


# ---------- network ---------------------------------------------------------

class ActorCritic(nn.Module):
    n_actions: int
    hidden: int = 64
    n_layers: int = 2
    action_temperature: float = 1.0

    @nn.compact
    def __call__(self, x):
        # obs is an int32 grid (4, 5, 30) on cramped_room with a leading batch axis
        # so the cast has to happen before the first Dense
        z = x.reshape((x.shape[0], -1)).astype(jnp.float32)
        for _ in range(self.n_layers):
            z = nn.tanh(
                nn.Dense(self.hidden, kernel_init=nn.initializers.orthogonal(jnp.sqrt(2.0)))(z)
            )
        # 0.01 gain keeps the initial logits flat so the first updates see a near-uniform policy
        logits = nn.Dense(self.n_actions, kernel_init=nn.initializers.orthogonal(0.01))(z)
        # tempering inside the net makes it part of the policy PPO optimizes
        # the surrogate ratio and the entropy bonus both read the tempered logits
        logits = logits / self.action_temperature
        value = nn.Dense(1, kernel_init=nn.initializers.orthogonal(1.0))(z).squeeze(-1)
        return logits, value


# ---------- public entry ----------------------------------------------------

def train_one_ippo(
    ctx: dict[str, Any],
    partner: PartnerSpec,
    cfg: Any,
    *,
    seed: int,
    role: Literal["student", "antagonist"],
) -> float:
    """Train one (student | antagonist) IPPO run paired with the given partner.

    Returns the mean episode return over `cfg.n_eval_episodes` greedy
    evaluation rollouts.

    The role picks the focal recipe only. Both roles face the partner the
    spec decodes to, which is what leaves the difference between the two
    returned values readable as regret. `cfg.antagonist_temperature`
    never reaches a network. The partner's temperature always comes from
    the spec.
    """
    env = ctx["env"]
    if role == "student":
        focal_hidden = cfg.student_hidden
        focal_n_layers = cfg.student_n_layers
        focal_lr = cfg.student_lr
        focal_train_steps = cfg.student_train_steps
    elif role == "antagonist":
        focal_hidden = cfg.antagonist_hidden
        focal_n_layers = cfg.antagonist_n_layers
        focal_lr = cfg.antagonist_lr
        focal_train_steps = cfg.antagonist_train_steps
    else:
        raise ValueError(f"unknown role: {role}")

    partner_kw = partner.decode_kwargs
    return _run_ippo(
        env,
        focal_hidden=focal_hidden,
        focal_n_layers=focal_n_layers,
        focal_lr=focal_lr,
        focal_train_steps=focal_train_steps,
        partner_hidden=partner_kw["hidden_dim"],
        partner_n_layers=partner_kw["n_layers"],
        partner_lr=partner_kw["lr"],
        partner_train_steps=partner_kw["train_steps"],
        partner_temperature=partner_kw["action_temperature"],
        # 16 x 64 is the rollout shape behind the numbers in docs/overcooked_engagement.md
        n_envs=16,
        n_steps=64,
        n_eval_episodes=cfg.n_eval_episodes,
        seed=seed,
    )


# ---------- inner trainer ---------------------------------------------------

def _run_ippo(
    env,
    *,
    focal_hidden: int,
    focal_n_layers: int,
    focal_lr: float,
    focal_train_steps: int,
    partner_hidden: int,
    partner_n_layers: int,
    partner_lr: float,
    partner_train_steps: int,
    partner_temperature: float,
    n_envs: int,
    n_steps: int,
    n_eval_episodes: int,
    seed: int,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    clip_eps: float = 0.2,
    ent_coef: float = 0.01,
    vf_coef: float = 0.5,
    n_epochs: int = 4,
    n_minibatches: int = 4,
    shaped_coef: float = 1.0,
) -> float:
    """The actual IPPO trainer. agent_0 is the focal, agent_1 is the partner.

    The PPO constants below match `scripts/play_overcooked.py`, which
    produced the curve in `docs/overcooked_engagement.md`.
    """
    # both agents update on every iteration, so the shorter budget of the two gets overrun
    # at the default cfg the focal budget already equals the Box maximum of 50k
    # which leaves the partner's train_steps axis with nothing to move
    train_steps_total = max(focal_train_steps, partner_train_steps)
    # train_steps is an env-step budget and one update consumes n_envs * n_steps of it
    # without the floor a budget under n_envs * n_steps runs zero updates and evaluates the initialization
    n_updates = max(1, train_steps_total // (n_envs * n_steps))

    key = jax.random.PRNGKey(seed)
    obs_shape = env.observation_space().shape
    n_actions = env.action_space().n

    nets = {
        "agent_0": ActorCritic(
            n_actions=n_actions, hidden=focal_hidden, n_layers=focal_n_layers,
        ),
        "agent_1": ActorCritic(
            n_actions=n_actions, hidden=partner_hidden, n_layers=partner_n_layers,
            action_temperature=partner_temperature,
        ),
    }
    lrs = {"agent_0": focal_lr, "agent_1": partner_lr}

    key, k0, k1 = jax.random.split(key, 3)
    dummy = jnp.zeros((1,) + tuple(obs_shape), dtype=jnp.float32)
    params = {
        "agent_0": nets["agent_0"].init(k0, dummy),
        "agent_1": nets["agent_1"].init(k1, dummy),
    }
    # chain order puts the 0.5 norm clip on the raw gradient, upstream of adam's rescaling
    # adam eps is the 1e-5 the PPO implementations use and optax's own default is 1e-8
    txs = {a: optax.chain(optax.clip_by_global_norm(0.5), optax.adam(lrs[a], eps=1e-5))
           for a in env.agents}
    opt_states = {a: txs[a].init(params[a]) for a in env.agents}

    @jax.jit
    def reset_fn(key):
        return jax.vmap(env.reset)(jax.random.split(key, n_envs))

    @jax.jit
    def step_fn(key, state, actions):
        return jax.vmap(env.step)(jax.random.split(key, n_envs), state, actions)

    def select_action(net, p, o, k):
        logits, val = net.apply(p, o)
        a = jax.random.categorical(k, logits, axis=-1)
        logp = _cat_log_prob(logits, a)
        return a, logp, val

    key, kr = jax.random.split(key)
    obs, state = reset_fn(kr)

    # nothing in the scan resets, so the env has to autoreset on done for the carried
    # state to stay valid across updates
    @jax.jit
    def collect(params, state, obs, key):
        def body(carry, _):
            state, obs, key = carry
            key, ka0, ka1, kstep = jax.random.split(key, 4)
            a0, lp0, v0 = select_action(nets["agent_0"], params["agent_0"], obs["agent_0"], ka0)
            a1, lp1, v1 = select_action(nets["agent_1"], params["agent_1"], obs["agent_1"], ka1)
            actions = {"agent_0": a0, "agent_1": a1}
            logps = {"agent_0": lp0, "agent_1": lp1}
            vals = {"agent_0": v0, "agent_1": v1}
            obs2, state2, reward, done, info = step_fn(kstep, state, actions)
            # sparse delivery reward alone gave no learning signal at this budget
            # (docs/overcooked_engagement.md), so the shaped channel is what moves the policy
            # no committed run turns it off so the effect of shaped_coef is unmeasured
            blended = {
                a: reward[a] + shaped_coef * info["shaped_reward"][a]
                for a in env.agents
            }
            return (state2, obs2, key), (obs, actions, logps, vals, blended, done, reward)

        (state_f, obs_f, _), traj = jax.lax.scan(
            body, (state, obs, key), None, length=n_steps
        )
        # value at the segment boundary. an episode runs 400 steps on cramped_room against
        # n_steps of 64 so most segments end mid-episode and the bootstrap carries the rest
        last_vals = {}
        for a in env.agents:
            _, v = nets[a].apply(params[a], obs_f[a])
            last_vals[a] = v
        return traj, last_vals, state_f, obs_f

    def compute_gae(rewards, dones, values, last_value):
        T = rewards.shape[0]

        # scan only runs forward, so t_idx walks back from the end and the stack comes out reversed
        def body(carry, t):
            gae, next_val = carry
            t_idx = T - 1 - t
            # done at t_idx cuts the bootstrap because the env has already reset by then
            # so next_val would otherwise be the value of a fresh episode
            nonterm = 1.0 - dones[t_idx]
            delta = rewards[t_idx] + gamma * next_val * nonterm - values[t_idx]
            gae = delta + gamma * gae_lambda * nonterm * gae
            return (gae, values[t_idx]), gae

        _, advs = jax.lax.scan(body, (jnp.zeros_like(last_value), last_value), jnp.arange(T))
        advs = advs[::-1]
        return advs, advs + values

    def make_update_fn(net, tx):
        def loss_fn(params, obs, actions, old_logps, advs, returns, old_vals):
            logits, vals = net.apply(params, obs)
            logps = _cat_log_prob(logits, actions)
            ratio = jnp.exp(logps - old_logps)
            # advantages are standardized per minibatch here, so n_minibatches moves the scale
            # the eps covers a minibatch whose advantages come back all equal
            adv_norm = (advs - advs.mean()) / (advs.std() + 1e-8)
            unclipped = ratio * adv_norm
            clipped = jnp.clip(ratio, 1 - clip_eps, 1 + clip_eps) * adv_norm
            pg_loss = -jnp.mean(jnp.minimum(unclipped, clipped))
            # clipped value loss from the PPO baselines, reusing clip_eps from the policy ratio
            # the max keeps the larger error so a value jump outside the trust region still pays
            v_clipped = old_vals + jnp.clip(vals - old_vals, -clip_eps, clip_eps)
            v_loss = jnp.mean(0.5 * jnp.maximum((vals - returns) ** 2, (v_clipped - returns) ** 2))
            ent = jnp.mean(_cat_entropy(logits))
            return pg_loss + vf_coef * v_loss - ent_coef * ent, (pg_loss, v_loss, ent)

        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)

        @jax.jit
        def update_agent(params, opt_state, obs, actions, old_logps, advs, returns, old_vals, key):
            batch = obs.shape[0]
            # n_envs * n_steps has to divide by n_minibatches or the reshape below raises
            mb = batch // n_minibatches

            def epoch_body(carry, _):
                params, opt_state, key = carry
                key, kp = jax.random.split(key)
                idx_e = jax.random.permutation(kp, batch).reshape(n_minibatches, mb)

                def mb_body(carry, ix):
                    params, opt_state = carry
                    (_loss, _aux), grads = grad_fn(
                        params,
                        obs[ix], actions[ix], old_logps[ix],
                        advs[ix], returns[ix], old_vals[ix],
                    )
                    updates, opt_state = tx.update(grads, opt_state, params)
                    params = optax.apply_updates(params, updates)
                    return (params, opt_state), _loss

                (params, opt_state), _ = jax.lax.scan(mb_body, (params, opt_state), idx_e)
                return (params, opt_state, key), None

            (params, opt_state, _), _ = jax.lax.scan(
                epoch_body, (params, opt_state, key), None, length=n_epochs
            )
            return params, opt_state

        return update_agent

    update_fns = {a: make_update_fn(nets[a], txs[a]) for a in env.agents}

    for _ in range(n_updates):
        key, kc = jax.random.split(key)
        traj, last_vals, state, obs = collect(params, state, obs, kc)
        obs_t, actions_t, logps_t, vals_t, blended_t, dones_t, _ = traj
        # each agent updates from the same rollout and treats the other's actions as environment
        for agent in env.agents:
            # the joint done masks both agents, which holds while the pair terminates together
            adv, ret = compute_gae(
                blended_t[agent], dones_t["__all__"].astype(jnp.float32),
                vals_t[agent], last_vals[agent],
            )
            # GAE has to run before this because axis 0 is time and the flatten folds it away
            flat = lambda x: x.reshape((-1,) + x.shape[2:])  # noqa: E731
            key, ku = jax.random.split(key)
            params[agent], opt_states[agent] = update_fns[agent](
                params[agent], opt_states[agent],
                flat(obs_t[agent]), flat(actions_t[agent]), flat(logps_t[agent]),
                flat(adv), flat(ret), flat(vals_t[agent]),
                ku,
            )

    # ---- evaluation: greedy, n_eval_episodes ----
    # greedy here contradicts the 2026-05-18 entry in docs/decisions.md, which puts the
    # headline number on stochastic rollouts. greedy read 0.00 against 35.00 stochastic
    # at 200 updates on seed 0, so this can report a collapsed policy as a flat zero
    # both sides of the regret difference are read this way so the zero cancels
    returns = []
    for _ in range(n_eval_episodes):
        key, ke = jax.random.split(key)
        obs_e, state_e = env.reset(ke)
        total = 0.0
        # the horizon comes from the env and cfg.rollout_horizon never reaches this function
        for _ in range(env.max_steps):
            actions = {}
            for agent in env.agents:
                logits, _ = nets[agent].apply(params[agent], obs_e[agent][None])
                # argmax ignores action_temperature so that axis of the Box reaches this
                # number only through the weights it produced
                actions[agent] = jnp.argmax(logits, axis=-1)[0]
            key, ks = jax.random.split(key)
            obs_e, state_e, reward, done, _ = env.step(ks, state_e, actions)
            # the shaped channel is left out here so the returned value is delivery reward only
            total += float(reward["agent_0"]) + float(reward["agent_1"])
            if bool(done["__all__"]):
                break
        returns.append(total)
    return float(np.mean(returns))
