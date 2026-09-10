"""Rust ISMCTS adapter.

The search tree lives entirely in Rust (`GameState.search_net` in the
`brass_ai._engine` extension); the network is queried through a batched Python
callback. This is the supported search implementation for self-play and
evaluation. Its `search(...) -> SearchResult` contract exposes `.best`,
`.visits`, and `.canon_by_candidate`.

Callback contract (Rust side builds the arrays, ONE row per request, framed
from that request's acting player):
  cells      (rows, BOARD_CELLS*F_CELL)         float32
  links      (rows, LINK_CELLS*F_LINK)          float32
  merchants  (rows, MERCHANT_COUNT*F_MERCHANT)  float32
  seats      (rows, SEAT_COUNT*F_SEAT)          float32
  global_    (rows, F_GLOBAL)                   float32
Padded action rows and a candidate mask come along with them; the callback
returns ``(candidate_logits (rows,max_candidates), values (rows,4))``. Rust
masks by each request's real candidate length.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from . import _engine as be
from .net import PolicyValueNet, state_batch


def make_net_fn(net: PolicyValueNet, device: str = "cuda"):
    """Build the Python callback the Rust search calls for batched inference."""
    def net_fn(cells, links, merchants, seats, global_vec, candidates, candidate_mask):
        rows = np.asarray(candidate_mask).shape[0]
        batch = state_batch((
            torch.from_numpy(np.asarray(cells, dtype=np.float32)).reshape(-1, be.BOARD_CELLS, be.F_CELL),
            torch.from_numpy(np.asarray(links, dtype=np.float32)).reshape(-1, be.LINK_CELLS, be.F_LINK),
            torch.from_numpy(np.asarray(merchants, dtype=np.float32)).reshape(-1, be.MERCHANT_COUNT, be.F_MERCHANT),
            torch.from_numpy(np.asarray(seats, dtype=np.float32)).reshape(-1, be.SEAT_COUNT, be.F_SEAT),
            torch.from_numpy(np.asarray(global_vec, dtype=np.float32)).reshape(-1, be.F_GLOBAL),
        ))
        if batch["cells"].shape[0] != rows:
            raise ValueError("state batch and candidate batch sizes differ")
        if device != "cpu":
            batch = {k: v.to(device) for k, v in batch.items()}
        action_features = torch.from_numpy(np.asarray(candidates, dtype=np.float32)).reshape(
            rows, np.asarray(candidate_mask).shape[1], net.cfg.action_features
        )
        mask = torch.from_numpy(np.asarray(candidate_mask, dtype=np.float32) > 0)
        if device != "cpu":
            action_features = action_features.to(device)
            mask = mask.to(device)
        out = net.policy_value(batch, action_features, mask)
        return (
            out["candidate_logits"].detach().cpu().numpy(),
            out["value"].detach().cpu().numpy(),
        )

    return net_fn


@dataclass
class RustMCTSConfig:
    c_puct: float = 2.5
    max_depth: int = 10
    dirichlet_alpha: float = 0.3
    dirichlet_weight: float = 0.15
    batch_size: int = 64
    # Zero expands every concrete legal move; positive values enable the
    # optional heuristic shortlist for controlled experiments.
    candidate_k: int = 0
    # Prior top-K pruning: score all legal candidates, then search only the K
    # highest-prior ones. Zero keeps every legal move. Needed at this branching
    # factor — see `NnMctsConfig::prior_top_k` in engine/src/ai/nn_mcts.rs.
    prior_top_k: int = 0
    # First-play urgency for unvisited children (never treat them as worthless).
    fpu: bool = True
    fpu_reduction: float = 0.0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class SearchResult:
    best: str | None = None
    visits: dict = field(default_factory=dict)
    canon_by_candidate: dict = field(default_factory=dict)
    # Simulations whose selected child could not be executed in the current
    # determinization (stale hand index in a reused tree node). Diagnostics
    # only; see `engine/src/ai/nn_mcts.rs` `descend`.
    failed_applies: int = 0
    # Simulations that executed a stored child whose hand index now names a
    # different card (rules that only bound-check the index accept it), so the
    # search silently played a card the move did not enumerate.
    rewritten_applies: int = 0


class RustISMCTS:
    """ISMCTS search driven by the Rust engine with a Python network callback."""

    def __init__(self, net: PolicyValueNet, cfg: RustMCTSConfig | None = None):
        self.cfg = cfg or RustMCTSConfig()
        if self.cfg.device != "cpu":
            net.to(self.cfg.device)
        self.net_fn = make_net_fn(net, self.cfg.device)

    def search(self, state, sims: int, add_root_noise: bool = False) -> SearchResult:
        best, children, _legal, failed_applies, rewritten_applies = state.search_net(
            self.net_fn,
            sims,
            self.cfg.c_puct,
            self.cfg.max_depth,
            self.cfg.dirichlet_alpha,
            self.cfg.dirichlet_weight,
            add_root_noise,
            self.cfg.batch_size,
            self.cfg.candidate_k,
            self.cfg.prior_top_k,
            self.cfg.fpu,
            self.cfg.fpu_reduction,
        )
        visits = {candidate_id: count for candidate_id, _canon, count in children}
        canon_by_candidate = {candidate_id: canon for candidate_id, canon, _count in children}
        return SearchResult(
            best=best,
            visits=visits,
            canon_by_candidate=canon_by_candidate,
            failed_applies=int(failed_applies),
            rewritten_applies=int(rewritten_applies),
        )
