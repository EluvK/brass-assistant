"""Rust ISMCTS adapter.

The search tree lives entirely in Rust (`GameState.search_net` in the
`brass_engine` extension); the network is queried through a batched Python
callback. This is the supported search implementation for self-play and
evaluation. Its `search(...) -> SearchResult` contract exposes `.best`,
`.visits`, and `.canon_by_slot`.

Callback contract (Rust side builds the arrays, ONE row per request = the
request's current-player perspective):
  board   (rows, BOARD_PLANES*BOARD_CELLS)   float32
  links   (rows, LINK_PLANES*LINK_CELLS)     float32
  global_ (rows, GLOBAL_LEN)                 float32
  own     (rows, HAND_LEN)                   float32
  opp     (rows, 3*HAND_LEN)                 float32
returns (type_logits (rows, 7), goal_logits (rows, policy_table_size),
         values (rows, 4)). The Rust search merges the branched priors
(logit(s) = type[t(s)] + goal[s]) and uses the 4-player value vector directly.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

import brass_engine as be
from .net import PolicyValueNet


def make_net_fn(net: PolicyValueNet, device: str = "cuda"):
    """Build the Python callback the Rust search calls for batched inference."""
    def net_fn(board, links, global_vec, own_hand, opp_hands):
        batch = {
            "board": torch.from_numpy(np.asarray(board, dtype=np.float32)).reshape(-1, be.BOARD_PLANES, be.BOARD_CELLS),
            "links": torch.from_numpy(np.asarray(links, dtype=np.float32)).reshape(-1, be.LINK_PLANES, be.LINK_CELLS),
            "global": torch.from_numpy(np.asarray(global_vec, dtype=np.float32)),
            "own_hand": torch.from_numpy(np.asarray(own_hand, dtype=np.float32)),
            "opp_hands": torch.from_numpy(np.asarray(opp_hands, dtype=np.float32)),
        }
        if device != "cpu":
            batch = {k: v.to(device) for k, v in batch.items()}
        type_logits, goal_logits, values = net.policy_value(batch)
        return (
            type_logits.detach().cpu().numpy(),
            goal_logits.detach().cpu().numpy(),
            values.detach().cpu().numpy(),
        )

    return net_fn


@dataclass
class RustMCTSConfig:
    c_puct: float = 2.5
    max_depth: int = 10
    dirichlet_alpha: float = 0.3
    dirichlet_weight: float = 0.15
    batch_size: int = 64
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class SearchResult:
    best: str | None = None
    visits: dict = field(default_factory=dict)
    canon_by_slot: dict = field(default_factory=dict)


class RustISMCTS:
    """Drop-in replacement for `brass_ai.mcts.ISMCTS` backed by the Rust search."""

    def __init__(self, net: PolicyValueNet, cfg: RustMCTSConfig | None = None):
        self.cfg = cfg or RustMCTSConfig()
        if self.cfg.device != "cpu":
            net.to(self.cfg.device)
        self.net_fn = make_net_fn(net, self.cfg.device)

    def search(self, state, sims: int, add_root_noise: bool = False) -> SearchResult:
        best, children, _legal = state.search_net(
            self.net_fn,
            sims,
            self.cfg.c_puct,
            self.cfg.max_depth,
            self.cfg.dirichlet_alpha,
            self.cfg.dirichlet_weight,
            add_root_noise,
            self.cfg.batch_size,
        )
        visits = {slot: count for slot, _canon, count in children}
        canon_by_slot = {slot: canon for slot, canon, _count in children}
        return SearchResult(best=best, visits=visits, canon_by_slot=canon_by_slot)
