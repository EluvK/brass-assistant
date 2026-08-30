"""Candidate-scoring Policy-Value network for Brass: Birmingham (head set v4).

The engine supplies a variable-size set of concrete legal moves. The policy
scores those candidates conditional on the state and never learns legality.

Head set (v4, see docs/archived/ai-encoding-v4-design.md §4):
* policy: FiLM-modulated action embedding + masked-mean candidate-set context,
  scored per candidate (O(N), no candidate self-attention).
* rank head: per-seat normalized final rank (rank/n, MSE) — comparable across
  games and order-preserving within one game; replaces the per-game VP z-score.
* winner head: official winner one-hot (softmax CE); the search value for a
  seat is `1 - rank` (higher = better), matching the Rust terminal backup.
* econ: era-split heads (canal / rail), 2 outputs each.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from . import _engine as be

N_ACTIONS = 7
N_PLAYERS = 4


@dataclass
class NetConfig:
    board_emb: int = 128
    links_emb: int = 64
    graph_layers: int = 3
    trunk: int = 256
    action_emb: int = 128
    action_features: int = getattr(be, "ACTION_FEATURE_DIM", 301)
    global_len: int = be.GLOBAL_LEN
    hand_len: int = be.HAND_LEN
    opp_hands_len: int = be.HAND_LEN * 3


class PolicyValueNet(nn.Module):
    """Score concrete legal candidate moves and predict multi-player value."""

    def __init__(self, cfg: NetConfig | None = None):
        super().__init__()
        self.cfg = cfg or NetConfig()
        self.board_enc = nn.Sequential(nn.Linear(be.BOARD_PLANES, self.cfg.board_emb), nn.ReLU())
        self.links_enc = nn.Sequential(nn.Linear(be.LINK_PLANES, self.cfg.links_emb), nn.ReLU())
        self.node_position = nn.Embedding(be.LOCATION_COUNT, self.cfg.board_emb)
        self.edge_position = nn.Embedding(be.LINK_CELLS, self.cfg.links_emb)
        self.edge_updates = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.cfg.links_emb + 3 * self.cfg.board_emb, self.cfg.links_emb),
                nn.ReLU(),
            )
            for _ in range(self.cfg.graph_layers)
        ])
        self.node_updates = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.cfg.board_emb + self.cfg.links_emb, self.cfg.board_emb),
                nn.ReLU(),
            )
            for _ in range(self.cfg.graph_layers)
        ])
        cell_locations = torch.as_tensor(be.BOARD_CELL_LOCATIONS, dtype=torch.long)
        endpoints = torch.as_tensor(be.CONNECTION_ENDPOINTS, dtype=torch.long).reshape(be.LINK_CELLS, 2)
        via_farms = torch.as_tensor(be.CONNECTION_VIA_FARMS, dtype=torch.long)
        if cell_locations.numel() != be.BOARD_CELLS or endpoints.shape != (be.LINK_CELLS, 2):
            raise RuntimeError("engine returned invalid state-graph topology")
        self.register_buffer("cell_locations", cell_locations)
        self.register_buffer("edge_endpoints", endpoints)
        self.register_buffer("edge_via_farms", via_farms)
        trunk_in = (
            2 * self.cfg.board_emb + 2 * self.cfg.links_emb + self.cfg.global_len
            + self.cfg.hand_len + self.cfg.opp_hands_len
        )
        self.trunk = nn.Sequential(
            nn.Linear(trunk_in, self.cfg.trunk), nn.ReLU(),
            nn.Linear(self.cfg.trunk, self.cfg.trunk), nn.ReLU(),
        )
        self.action_encoder = nn.Sequential(
            nn.Linear(self.cfg.action_features, self.cfg.action_emb), nn.ReLU(),
            nn.Linear(self.cfg.action_emb, self.cfg.action_emb), nn.ReLU(),
        )
        # FiLM: the state modulates every action embedding (multiplicative
        # interaction), then each candidate also sees the set context — the
        # masked mean of all candidates — so scores depend on the candidate
        # SET, not only on the single action.
        self.film = nn.Linear(self.cfg.trunk, 2 * self.cfg.action_emb)
        self.action_score = nn.Sequential(
            nn.Linear(3 * self.cfg.action_emb, self.cfg.trunk), nn.ReLU(),
            nn.Linear(self.cfg.trunk, 1),
        )
        self.rank_head = nn.Linear(self.cfg.trunk, N_PLAYERS)
        self.winner_head = nn.Linear(self.cfg.trunk, N_PLAYERS)
        self.econ_canal_head = nn.Linear(self.cfg.trunk, 2)
        self.econ_rail_head = nn.Linear(self.cfg.trunk, 2)

    @staticmethod
    def _scatter_mean(values: torch.Tensor, indices: torch.Tensor, size: int) -> torch.Tensor:
        """Mean-pool `(B,N,D)` values into `size` graph nodes."""
        batch, _, dim = values.shape
        out = values.new_zeros((batch, size, dim))
        expanded = indices.view(1, -1, 1).expand(batch, -1, dim)
        out.scatter_add_(1, expanded, values)
        counts = values.new_zeros((batch, size, 1))
        counts.scatter_add_(1, indices.view(1, -1, 1).expand(batch, -1, 1),
                            values.new_ones((batch, indices.numel(), 1)))
        return out / counts.clamp_min(1.0)

    def encode_state(self, batch: dict) -> torch.Tensor:
        board_cells = self.board_enc(batch["board"].transpose(1, 2))
        node = self._scatter_mean(board_cells, self.cell_locations, be.LOCATION_COUNT)
        node = node + self.node_position.weight.unsqueeze(0)
        edge = self.links_enc(batch["links"].transpose(1, 2))
        edge = edge + self.edge_position.weight.unsqueeze(0)

        a, b = self.edge_endpoints[:, 0], self.edge_endpoints[:, 1]
        via_valid = self.edge_via_farms < be.LOCATION_COUNT
        via = self.edge_via_farms.clamp_max(be.LOCATION_COUNT - 1)
        for edge_update, node_update in zip(self.edge_updates, self.node_updates):
            via_node = node[:, via] * via_valid.view(1, -1, 1)
            edge = edge_update(torch.cat([edge, node[:, a], node[:, b], via_node], dim=-1))
            # An edge informs both endpoints and its brewery farm when present.
            incident = torch.cat([a, b, via[via_valid]], dim=0)
            messages = torch.cat([edge, edge, edge[:, via_valid]], dim=1)
            node = node_update(torch.cat([
                node, self._scatter_mean(messages, incident, be.LOCATION_COUNT)
            ], dim=-1))

        board = torch.cat([node.mean(dim=1), node.max(dim=1).values], dim=1)
        links = torch.cat([edge.mean(dim=1), edge.max(dim=1).values], dim=1)
        return self.trunk(torch.cat(
            [board, links, batch["global"], batch["own_hand"], batch["opp_hands"]], dim=1
        ))

    def forward(self, batch: dict, action_features: torch.Tensor,
                candidate_mask: torch.Tensor | None = None) -> dict:
        """Evaluate candidates shaped ``(B,N,D)`` with optional padding mask."""
        if action_features.ndim == 2:
            action_features = action_features.unsqueeze(0)
        if action_features.ndim != 3 or action_features.shape[-1] != self.cfg.action_features:
            raise ValueError(
                f"action_features must have shape (B,N,{self.cfg.action_features})"
            )
        state = self.encode_state(batch)
        if state.shape[0] != action_features.shape[0]:
            raise ValueError("state batch and action batch sizes differ")

        if candidate_mask is None:
            candidate_mask = torch.ones(
                action_features.shape[:2], dtype=torch.bool, device=action_features.device
            )
        else:
            candidate_mask = candidate_mask.to(device=action_features.device, dtype=torch.bool)
            if candidate_mask.shape != action_features.shape[:2]:
                raise ValueError("candidate_mask must have shape (B,N)")
        if (~candidate_mask).all(dim=1).any():
            raise ValueError("each state must contain at least one legal candidate")

        actions = self.action_encoder(action_features)
        gamma, beta = self.film(state).chunk(2, dim=-1)
        modulated = gamma.unsqueeze(1) * actions + beta.unsqueeze(1)  # (B,N,E)
        weights = candidate_mask.unsqueeze(-1).to(modulated.dtype)
        ctx = (modulated * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)  # (B,E)
        ctx = ctx.unsqueeze(1).expand(-1, modulated.shape[1], -1)
        logits = self.action_score(
            torch.cat([modulated, ctx, modulated * ctx], dim=-1)
        ).squeeze(-1)

        rank = self.rank_head(state)
        log_probs = torch.log_softmax(logits.masked_fill(~candidate_mask, float("-inf")), dim=1)
        econ = torch.cat([self.econ_canal_head(state), self.econ_rail_head(state)], dim=-1)
        return {
            "candidate_logits": logits,
            "candidate_log_probs": log_probs,
            "candidate_mask": candidate_mask,
            "rank": rank,                       # per-seat normalized final rank
            "value": 1.0 - rank,                # search scale (higher = better)
            "winner_logits": self.winner_head(state),
            "econ": econ,                       # (B,4): canal head | rail head
        }

    def policy_value(self, batch: dict, action_features: torch.Tensor,
                     candidate_mask: torch.Tensor | None = None) -> dict:
        was_training = self.training
        self.eval()
        try:
            with torch.no_grad():
                return self.forward(batch, action_features, candidate_mask)
        finally:
            self.train(was_training)
