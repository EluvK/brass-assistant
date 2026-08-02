"""ISMCTS (Information-Set MCTS) over the Rust engine, guided by the network.

Design notes
------------
* One determinization of the root state per search (`state.determinize()`);
  the whole tree lives in that sampled world. (The Rust `mcts_ai` re-samples
  per simulation; we start single-world to validate that a learned prior/value
  beats the heuristic. Per-sim determinization can be added later.)
* Tree identity is the **policy slot** (`move_slots`): legal moves that differ
  only in resource-source / card choice collapse to one child, matching the
  Rust tree and the policy head.
* Leaf value is a MaxN vector over all players (one batched forward of
  `player_count` perspective encodings). At a terminal node the true
  normalized final VP is used.
* Selection: the player to move at a node maximizes their OWN Q + PUCT prior
  (Brass is non-zero-sum; opponents do not form a coalition against us).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from . import build_input
from .net import PolicyValueNet


@dataclass
class MCTSConfig:
    c_puct: float = 1.5
    max_depth: int = 10
    dirichlet_alpha: float = 0.3
    dirichlet_weight: float = 0.25
    temperature: float = 1.0  # root sampling temp for self-play


@dataclass
class SearchResult:
    best: str | None  # canonical of the max-visits child (None if no legal)
    visits: dict = field(default_factory=dict)  # slot -> visit count (root)
    canon_by_slot: dict = field(default_factory=dict)  # slot -> canonical


@dataclass
class Node:
    state: object
    legal: list = field(default_factory=list)  # [(slot, canonical, describe)]
    prior: np.ndarray = None  # aligned with children
    visits: int = 0
    value_sum: np.ndarray = None  # per player (MaxN)
    children: list = field(default_factory=list)
    n_players: int = 4

    @property
    def is_expanded(self) -> bool:
        return bool(self.children)

    def q(self, player: int) -> float:
        if self.visits == 0:
            return 0.0
        return self.value_sum[player] / self.visits


class ISMCTS:
    def __init__(
        self,
        net: PolicyValueNet,
        cfg: MCTSConfig | None = None,
        device: str | None = None,
    ):
        self.net = net
        self.cfg = cfg or MCTSConfig()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if self.device != "cpu":
            self.net.to(self.device)

    # ------------------------------------------------------------------ core
    def search(self, root_state, sims: int, add_root_noise: bool = False) -> SearchResult:
        """Run `sims` simulations from a determinized root.

        Returns a SearchResult with the best canonical move, the root visit
        distribution by slot (training policy target) and the slot->canonical
        map (needed to replay any chosen slot under temperature sampling).
        """
        root = Node(state=root_state.determinize())
        root.n_players = root_state.player_count
        root.value_sum = np.zeros(root.n_players, dtype=np.float64)

        if sims <= 0:
            return SearchResult(None)

        # First simulation expands the root (creating children + priors).
        self._simulate(root)

        if add_root_noise and root.children:
            d = np.random.dirichlet([self.cfg.dirichlet_alpha] * len(root.children))
            root.prior = self.cfg.dirichlet_weight * d + (1 - self.cfg.dirichlet_weight) * root.prior

        for _ in range(sims - 1):
            self._simulate(root)

        if not root.children:
            return SearchResult(None)

        best = max(root.children, key=lambda c: c.visits)
        canon_by_slot = {slot: canon for (slot, canon, _) in root.legal}
        best_canonical = canon_by_slot[
            next(slot for (slot, _, _), c in zip(root.legal, root.children) if c is best)
        ]
        visits = {slot: c.visits for (slot, _, _), c in zip(root.legal, root.children)}
        return SearchResult(best=best_canonical, visits=visits, canon_by_slot=canon_by_slot)

    def _simulate(self, root: Node):
        path = [root]
        node = root
        depth = 0
        while True:
            if node.state.game_over:
                value = self._terminal_value(node)
                break
            if depth >= self.cfg.max_depth:
                value = self._eval_value(node)
                break
            if not node.is_expanded:
                # Expands (populating node.legal) and returns a value; if no
                # legal moves it falls back to a leaf evaluation.
                value = self._expand(node)
                break
            node = self._select(node)
            path.append(node)
            depth += 1

        for n in path:
            n.visits += 1
            n.value_sum += value

    # --------------------------------------------------------------- select
    def _select(self, node: Node) -> Node:
        pid = node.state.current_player_id
        parent_visits = max(1.0, float(node.visits))
        best_c, best_uct = None, float("-inf")
        for c, p in zip(node.children, node.prior):
            explore = self.cfg.c_puct * p * np.sqrt(parent_visits) / (1.0 + c.visits)
            uct = c.q(pid) + explore
            if uct > best_uct:
                best_uct, best_c = uct, c
        return best_c

    # --------------------------------------------------------------- expand
    def _expand(self, node: Node) -> np.ndarray:
        """Create children for each legal slot; evaluate priors (to-move player)
        and the MaxN value vector (all players) in one batched forward.

        Slots whose every canonical move fails to execute are dropped (the
        engine can enumerate double-rail coal choices that do not actually
        apply; we match the Rust MCTS by skipping them)."""
        groups = self._group_legal(node.state)
        node.legal = []
        children = []
        for slot, (describe, canonicals) in groups.items():
            child_state = self._make_child(node.state, canonicals)
            if child_state is None:
                continue
            child = Node(state=child_state, n_players=node.n_players)
            child.value_sum = np.zeros(node.n_players, dtype=np.float64)
            children.append(child)
            node.legal.append((slot, canonicals[0], describe))

        if not children:
            return self._eval_value(node)

        slots = [s for s, _, _ in node.legal]

        pids = list(range(node.n_players))
        batch = self._encode_perspectives(node.state, pids)
        logits, values = self.net.policy_value(batch)

        pid = node.state.current_player_id
        logits_p = logits[pids.index(pid)]

        mask = np.zeros(len(logits_p), dtype=bool)
        for s in slots:
            mask[s] = True
        full = self._masked_softmax(logits_p.detach().cpu().numpy(), mask)
        # Per-child priors aligned with `legal`/`children` (only legal slots).
        node.prior = full[slots]
        node.children = children
        return values.detach().cpu().numpy()

    @staticmethod
    def _group_legal(state):
        """Group legal moves by policy slot, keeping all canonicals per slot
        (some are not executable; we try them in order at child creation)."""
        groups: dict = {}
        for slot, canonical, describe in state.legal_moves():
            groups.setdefault(slot, (describe, []))[1].append(canonical)
        return groups

    @staticmethod
    def _make_child(state, canonicals):
        """Clone `state` and apply the first executable canonical; None if none."""
        for canonical in canonicals:
            c = state.clone()
            try:
                c.apply_move(canonical)
                return c
            except ValueError:
                continue
        return None

    # -------------------------------------------------------------- evaluate
    def _eval_value(self, node: Node) -> np.ndarray:
        if node.state.game_over:
            return self._terminal_value(node)
        pids = list(range(node.n_players))
        batch = self._encode_perspectives(node.state, pids)
        _, values = self.net.policy_value(batch)
        return values.detach().cpu().numpy()

    @staticmethod
    def _terminal_value(node: Node) -> np.ndarray:
        """z_p = (vp_p - mean) / max(std, eps); all-zero if no spread."""
        vps = np.asarray(node.state.player_vps(), dtype=np.float64)
        std = vps.std()
        if std < 1e-6:
            return np.zeros_like(vps)
        return (vps - vps.mean()) / std

    # --------------------------------------------------------------- helpers
    def _encode_perspectives(self, state, pids):
        rows = [state.state_to_tensor(perspective=p) for p in pids]
        batch = build_input.encode_arrays(*zip(*rows))
        if self.device != "cpu":
            batch = {k: v.to(self.device) for k, v in batch.items()}
        return batch

    @staticmethod
    def _masked_softmax(logits: np.ndarray, mask: np.ndarray) -> np.ndarray:
        masked = np.where(mask, logits, -1e9)
        m = masked - masked.max()
        e = np.exp(m)
        return e / e.sum()
