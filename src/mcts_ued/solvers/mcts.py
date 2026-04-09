"""MCTS as a UED solver over partner-policy parameter space.

Leaves are `PartnerSpec`s decoded from a `PartnerSpace`
(`partners/parametrize.py`). The A/B against `PPOAdversary` in the
same `train_paired` loop is the core experimental question.

v1 design:

* depth 1: root expands directly to leaves. Deeper trees can be added
  once we have a hierarchical partner parametrization.
* UCB1 selection. PUCT needs a prior policy; that gets added once
  there is a population-based partner space to anchor it.
* progressive widening for the continuous Box (Cazenave-style):
  cap = max(1, k * visits**alpha).
* tree rebuilt per `propose()`: matches PAIRED's per-step adversary
  independence and keeps the A/B comparison clean.
"""
from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from mcts_ued.partners.parametrize import PartnerSpace, PartnerSpec


@dataclass
class MCTSConfig:
    n_simulations: int = 32         # one leaf_value_fn call each, so 32 student train_steps per propose()
    ucb_c: float = math.sqrt(2.0)   # UCB1's constant from Auer et al. 2002 and calibrated for unit-range values
    max_depth: int = 1              # at depth 1 the jitter branch of _expand is unreachable
    progressive_widening_alpha: float = 0.5   # cap = k * visits**alpha, so root children grow as sqrt(simulations)
    progressive_widening_k: float = 1.0
    action_jitter: float = 0.15     # std in normalized box units, so 0.15 is 15 percent of every axis range
    seed: int = 0


@dataclass
class Node:
    """One node in the search tree.

    `state` is the normalized [0, 1]^dim parameter vector. Decoding to
    a `PartnerSpec` happens at the leaf so the tree never carries
    partner kwargs.
    """

    state: np.ndarray
    depth: int
    parent: Node | None = None
    children: list[Node] = field(default_factory=list)
    visits: int = 0
    total_value: float = 0.0

    @property
    def mean_value(self) -> float:
        # averaging over repeat visits buys variance reduction only when leaf_value_fn is
        # stochastic. train.py pins the student seed to 0 so a revisited state scores the same
        return self.total_value / self.visits if self.visits > 0 else 0.0

    def ucb_score(self, c: float, parent_visits: int) -> float:
        if self.visits == 0:
            # a child expanded but not yet backed up would score on a mean of 0 and lose to any
            # arm with positive value
            return float("inf")
        # the clamp keeps math.log away from 0 when a child is scored before its parent's first
        # backup
        return self.mean_value + c * math.sqrt(math.log(max(1, parent_visits)) / self.visits)


class MCTSAdversary:
    """MCTS over partner-policy parameter space.

    `leaf_value_fn(spec) -> float` is the leaf evaluation: PAIRED's
    regret estimate from one student-vs-antagonist rollout with the
    decoded partner. The function is injected so this file does not
    depend on JaxMARL or any concrete student.
    """

    def __init__(
        self,
        cfg: MCTSConfig,
        space: PartnerSpace,
        leaf_value_fn: Callable[[PartnerSpec], float],
    ):
        self.cfg = cfg
        self.space = space
        self.leaf_value_fn = leaf_value_fn
        # one stream for the whole run, so a proposal depends on every draw the earlier searches made
        self.rng = np.random.default_rng(cfg.seed)
        # nothing reads this in v1. it is the counter a priors hook would key on
        self._step_count = 0

    def propose(self) -> PartnerSpec:
        # box midpoint, decoded only when max_depth is 0 because simulation one expands the root
        root = Node(state=np.full(self.space.dim, 0.5, dtype=np.float32), depth=0)
        for _ in range(self.cfg.n_simulations):
            self._simulate(root)
        if not root.children:
            # n_simulations=0 leaves the root childless and the max() below would raise on it
            spec = self.space.sample(self.rng)
        else:
            # selection is on visit count. one lucky leaf value can leave a single-visit child
            # holding the best mean
            best = max(root.children, key=lambda c: c.visits)
            spec = self.space.decode(best.state)
        self._step_count += 1
        return spec

    def update(self, spec: PartnerSpec, regret: float) -> None:
        """v1: tree rebuilt every propose so update is a no-op (priors hook).

        The outer loop's regret never reaches the tree. The search
        signal is `leaf_value_fn`, which calls the student directly.
        """
        del spec, regret
        return

    # ---- internal ----

    def _simulate(self, root: Node) -> float:
        node = root
        while node.depth < self.cfg.max_depth:
            if self._should_expand(node):
                node = self._expand(node)
                break
            node = self._select_child(node)
        spec = self.space.decode(node.state)
        # leaf_value_fn stands in for the random playout, so a fresh child is scored where it was
        # created and one simulation costs one student train_step
        value = float(self.leaf_value_fn(spec))
        self._backup(node, value)
        return value

    def _should_expand(self, node: Node) -> bool:
        # int() floors the cap. with k=1 and alpha=0.5 a node gains its next child only when its
        # visit count reaches the next perfect square
        # both clamps put cap at 1 on the root's first call where visits is still 0
        cap = max(
            1,
            int(self.cfg.progressive_widening_k * (max(1, node.visits) ** self.cfg.progressive_widening_alpha)),
        )
        return len(node.children) < cap

    def _expand(self, node: Node) -> Node:
        if node.depth == 0:
            # root children are uniform over the box. the jitter branch below is unreachable
            # while max_depth is 1
            new_state = self.rng.uniform(0.0, 1.0, size=self.space.dim).astype(np.float32)
        else:
            jitter = self.rng.normal(0.0, self.cfg.action_jitter, size=self.space.dim).astype(np.float32)
            # clip pins an out-of-box proposal to the face, so a node near an edge puts weight
            # on the boundary
            new_state = self.space.clip(node.state + jitter)
        child = Node(state=new_state, depth=node.depth + 1, parent=node)
        node.children.append(child)
        return child

    def _select_child(self, node: Node) -> Node:
        # a node with children has been backed up at least once, so this clamp only covers a
        # caller that selects on an unexpanded node
        parent_visits = max(1, node.visits)
        return max(node.children, key=lambda c: c.ucb_score(self.cfg.ucb_c, parent_visits))

    def _backup(self, leaf: Node, value: float) -> None:
        # no sign flip per level: partner search is single player and every node maximizes the
        # same leaf value
        # the walk reaches the root every simulation, which is the count _should_expand reads
        cur: Node | None = leaf
        while cur is not None:
            cur.visits += 1
            cur.total_value += value
            cur = cur.parent
