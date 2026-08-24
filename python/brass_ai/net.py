"""Candidate-scoring Policy-Value network for Brass: Birmingham.

The engine supplies a variable-size set of concrete legal moves. The policy
scores those candidates conditional on the state and never learns legality.
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
    trunk: int = 256
    action_emb: int = 128
    action_features: int = getattr(be, "ACTION_FEATURE_DIM", 235)
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
        self.action_score = nn.Sequential(
            nn.Linear(self.cfg.trunk + self.cfg.action_emb, self.cfg.trunk), nn.ReLU(),
            nn.Linear(self.cfg.trunk, 1),
        )
        self.action_type_head = nn.Linear(self.cfg.trunk, N_ACTIONS)
        self.value_head = nn.Linear(self.cfg.trunk, N_PLAYERS)
        self.econ_head = nn.Linear(self.cfg.trunk, 2)

    def encode_state(self, batch: dict) -> torch.Tensor:
        board = self.board_enc(batch["board"].transpose(1, 2))
        board = torch.cat([board.mean(dim=1), board.max(dim=1).values], dim=1)
        links = self.links_enc(batch["links"].transpose(1, 2))
        links = torch.cat([links.mean(dim=1), links.max(dim=1).values], dim=1)
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
        actions = self.action_encoder(action_features)
        state_per_action = state.unsqueeze(1).expand(-1, actions.shape[1], -1)
        logits = self.action_score(torch.cat([state_per_action, actions], dim=-1)).squeeze(-1)
        if candidate_mask is None:
            candidate_mask = torch.ones_like(logits, dtype=torch.bool)
        else:
            candidate_mask = candidate_mask.to(device=logits.device, dtype=torch.bool)
            if candidate_mask.shape != logits.shape:
                raise ValueError("candidate_mask must have shape (B,N)")
        if (~candidate_mask).all(dim=1).any():
            raise ValueError("each state must contain at least one legal candidate")
        log_probs = torch.log_softmax(logits.masked_fill(~candidate_mask, float("-inf")), dim=1)
        return {
            "type_logits": self.action_type_head(state),
            "candidate_logits": logits,
            "candidate_log_probs": log_probs,
            "candidate_mask": candidate_mask,
            "value": self.value_head(state),
            "econ": self.econ_head(state),
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
