"""Candidate-scoring Policy-Value network for Brass: Birmingham.

Implements the contract in `docs/ai-action-encoding.md`: the engine supplies a
variable-size set of concrete legal moves, the network scores each one against
the state tokens of the same position, and it never learns legality.

The state is a token sequence (49 cells + 39 links + 9 merchants + 4 seats +
1 global) processed by a small transformer. An action is a type plus references
into that token sequence, so "which mine did I drain, and what was on it" is a
structural lookup rather than a hand-written feature.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import _engine as be
from .hierarchical_policy import (
    ACTION_KIND_COUNT,
    ACTION_NUMBERS,
    REF_CARD,
    REF_CELL,
    REF_INDUSTRY,
    REF_KIND_COUNT,
    REF_LINK,
    REF_MERCHANT,
    STATE_GLOBAL,
    STATE_GROUPS,
    TOKEN_TYPES,
    split_action_rows,
)

N_PLAYERS = 4


@dataclass
class NetConfig:
    d_model: int = 192
    layers: int = 4
    heads: int = 6
    dropout: float = 0.0
    action_features: int = be.ACTION_FEATURE_DIM
    action_ref_cap: int = be.ACTION_REF_CAP


class PolicyValueNet(nn.Module):
    """Score concrete legal candidate moves and predict multi-player value."""

    def __init__(self, cfg: NetConfig | None = None):
        super().__init__()
        self.cfg = cfg or NetConfig()
        d = self.cfg.d_model

        self.group_proj = nn.ModuleDict({
            name: nn.Linear(width, d) for name, (_count, width) in STATE_GROUPS.items()
        })
        self.global_proj = nn.Linear(STATE_GLOBAL[1], d)
        self.type_embed = nn.Embedding(len(TOKEN_TYPES), d)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d,
            nhead=self.cfg.heads,
            dim_feedforward=4 * d,
            dropout=self.cfg.dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.cfg.layers)
        self.state_norm = nn.LayerNorm(d)

        # Action side: type, slot, scalars, and the referenced entities.
        self.kind_embed = nn.Embedding(ACTION_KIND_COUNT, d)
        self.slot_embed = nn.Embedding(4, d)
        self.numbers_proj = nn.Linear(ACTION_NUMBERS, d)
        self.industry_embed = nn.Embedding(be.INDUSTRY_COUNT, d)
        self.card_embed = nn.Embedding(be.CARD_SEMANTIC_COUNT, d)
        self.action_norm = nn.LayerNorm(d)

        self.score_head = nn.Sequential(
            nn.Linear(3 * d, 2 * d), nn.GELU(), nn.Linear(2 * d, 1)
        )
        self.q_head = nn.Sequential(
            nn.Linear(3 * d, 2 * d), nn.GELU(), nn.Linear(2 * d, 1)
        )
        self.value_head = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, N_PLAYERS))
        self.winner_head = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, N_PLAYERS))
        self.econ_canal_head = nn.Linear(d, 2)
        self.econ_rail_head = nn.Linear(d, 2)

        self.register_buffer("ref_offset", self._ref_offsets())

    @staticmethod
    def _ref_offsets() -> torch.Tensor:
        """Start index of each reference kind inside the entity table."""
        cells = be.BOARD_CELLS
        links = cells + be.LINK_CELLS
        merchants = links + be.MERCHANT_COUNT
        industry = merchants + be.INDUSTRY_COUNT
        card = industry + be.CARD_SEMANTIC_COUNT
        offsets = torch.zeros(REF_KIND_COUNT, dtype=torch.long)
        offsets[REF_CELL] = 0
        offsets[REF_LINK] = cells
        offsets[REF_MERCHANT] = links
        offsets[REF_INDUSTRY] = industry
        offsets[REF_CARD] = card
        return offsets

    def encode_state(self, batch: dict) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (token sequence (B,T,d), state summary (B,d))."""
        tokens = []
        for i, name in enumerate(TOKEN_TYPES):
            if name == "global":
                vec = self.global_proj(batch["global"]).unsqueeze(1)
            else:
                vec = self.group_proj[name](batch[name])
            tokens.append(vec + self.type_embed.weight[i].view(1, 1, -1))
        seq = torch.cat(tokens, dim=1)
        seq = self.encoder(seq)
        # The global token sits last; the mean gives a set-level summary that is
        # invariant to board size while the sequence keeps per-entity detail.
        summary = self.state_norm(seq[:, -1] + seq.mean(dim=1))
        return seq, summary

    def encode_actions(self, seq: torch.Tensor, actions: torch.Tensor,
                       mask: torch.Tensor) -> torch.Tensor:
        """Return a per-candidate action embedding ``(B,N,d)``."""
        parts = split_action_rows(actions)
        batch, n, _ = actions.shape
        mask2 = mask.unsqueeze(-1).to(seq.dtype)
        parts = {k: (v * mask2 if v.dtype.is_floating_point else v) for k, v in parts.items()}

        entity = torch.cat([
            seq[:, :be.BOARD_CELLS],
            seq[:, be.BOARD_CELLS:be.BOARD_CELLS + be.LINK_CELLS],
            seq[:, be.BOARD_CELLS + be.LINK_CELLS:
                be.BOARD_CELLS + be.LINK_CELLS + be.MERCHANT_COUNT],
            self.industry_embed.weight.unsqueeze(0).expand(batch, -1, -1),
            self.card_embed.weight.unsqueeze(0).expand(batch, -1, -1),
        ], dim=1)

        ref_index = (self.ref_offset[parts["ref_kind"]] + parts["ref_id"]).clamp(
            0, entity.shape[1] - 1
        )
        gathered = entity.gather(
            1, ref_index.reshape(batch, n * self.cfg.action_ref_cap, 1).expand(-1, -1, entity.shape[2])
        ).reshape(batch, n, self.cfg.action_ref_cap, entity.shape[2])
        weight = parts["ref_weight"].unsqueeze(-1)
        pooled = (gathered * weight).sum(dim=2) / weight.sum(dim=2).clamp_min(1e-6)

        action = (
            self.kind_embed(parts["kind"].clamp(0, ACTION_KIND_COUNT - 1))
            + self.slot_embed(parts["slot"].clamp(0, 3))
            + self.numbers_proj(parts["numbers"])
            + pooled
        )
        return self.action_norm(action)

    def forward(self, batch: dict, action_features: torch.Tensor,
                candidate_mask: torch.Tensor | None = None) -> dict:
        """Evaluate candidates shaped ``(B,N,ACTION_FEATURE_DIM)``."""
        if action_features.ndim == 2:
            action_features = action_features.unsqueeze(0)
        if (action_features.ndim != 3
                or action_features.shape[-1] != self.cfg.action_features):
            raise ValueError(
                f"action_features must have shape (B,N,{self.cfg.action_features})"
            )
        if candidate_mask is None:
            candidate_mask = torch.ones(
                action_features.shape[:2], dtype=torch.bool, device=action_features.device
            )
        else:
            candidate_mask = candidate_mask.to(
                device=action_features.device, dtype=torch.bool
            )
            if candidate_mask.shape != action_features.shape[:2]:
                raise ValueError("candidate_mask must have shape (B,N)")
        if (~candidate_mask).all(dim=1).any():
            raise ValueError("each state must contain at least one legal candidate")

        seq, summary = self.encode_state(batch)
        if seq.shape[0] != action_features.shape[0]:
            raise ValueError("state batch and action batch sizes differ")
        action = self.encode_actions(seq, action_features, candidate_mask)

        joint = torch.cat([action, summary.unsqueeze(1).expand_as(action), action * summary.unsqueeze(1)], dim=-1)
        logits = self.score_head(joint).squeeze(-1)
        candidate_value = self.q_head(joint).squeeze(-1)

        log_probs = torch.log_softmax(
            logits.masked_fill(~candidate_mask, float("-inf")), dim=1
        )
        econ = torch.cat([self.econ_canal_head(summary), self.econ_rail_head(summary)], dim=-1)
        return {
            "candidate_logits": logits,
            "candidate_log_probs": log_probs,
            "candidate_mask": candidate_mask,
            "value": self.value_head(summary),          # (B,4), me first
            "candidate_value": candidate_value,          # (B,N) action-conditioned value
            "winner_logits": self.winner_head(summary),  # (B,4), me first (softmax CE)
            "econ": econ,                                # (B,4): canal head | rail head
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


def state_batch(tensors) -> dict:
    """Assemble the network's state batch dict from engine token groups."""
    cells, links, merchants, seats, global_vec = tensors
    return {
        "cells": cells.float() if cells.dtype == torch.uint8 else cells,
        "links": links.float() if links.dtype == torch.uint8 else links,
        "merchants": merchants.float() if merchants.dtype == torch.uint8 else merchants,
        "seats": seats.float() if seats.dtype == torch.uint8 else seats,
        "global": global_vec.float() if global_vec.dtype == torch.uint8 else global_vec,
    }


def candidate_value_loss(predicted: torch.Tensor, target_value: torch.Tensor,
                         action_index: torch.Tensor) -> torch.Tensor:
    """MSE of the action-conditioned value on the move actually played.

    ``target_value`` is the acting player's terminal utility (index 0 of the
    perspective-rotated value target) and ``action_index`` is the played
    candidate's position, or -1 when the sample has no recorded move.
    """
    valid = action_index >= 0
    if not bool(valid.any()):
        return predicted.sum() * 0.0
    index = action_index.clamp_min(0)
    picked = predicted.gather(1, index.unsqueeze(1)).squeeze(1)
    loss = F.mse_loss(picked[valid], target_value[valid])
    return loss
