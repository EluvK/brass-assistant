"""Policy-Value network for Brass: Birmingham.

Structure (AlphaZero-style, flat cell encoding since the map is a fixed graph
rather than a grid):

  board (B,17,49) --shared Linear(17->H)--> cell embeddings --mean+max pool--> (B,2H)
  links (B,6,39)  --shared Linear(6->H2)--> cell embeddings --mean+max pool--> (B,2H2)
  [board_emb, links_emb, global(50), own_hand(35), opp_hands(105)] -> trunk MLP

  Policy (branched head, 2026-08 redesign):
    type_head Linear(256 -> 7)      # build / network / develop / sell / loan /
                                    #   scout / pass action-type marginals
    goal_head Linear(256 -> 1316)   # per-slot goal logits over the policy table
    logit(slot s) = type[t(s)] + goal[s]   # merged on the Rust side over legal
                                    #   slots; t(s) from policy::slot_type
  Value  (redesigned): Linear(256 -> 4) predicts the normalized final VP
    z-vector over ALL 4 players from a SINGLE perspective, NO tanh. The global
    encoding already carries per-player money/income/vp (indices 8..39) and
    opp_hands carries every opponent's hand, so one viewpoint is enough to
    predict all four finals (removes the 4x perspective encode in the search).

The value target is the 4-player normalized VP vector
z = (vp - mean)/std over the players of that game (see selfplay.py).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

import brass_engine as be

POLICY_SIZE = be.policy_table_size
N_ACTIONS = 7  # build / network / develop / sell / loan / scout / pass
N_PLAYERS = 4

# slot -> action-type band, cached from the Rust binding (single source of truth).
_SLOT_TYPES: np.ndarray | None = None


def slot_types() -> np.ndarray:
    global _SLOT_TYPES
    if _SLOT_TYPES is None:
        _SLOT_TYPES = np.asarray(be.slot_types, dtype=np.int64)
    return _SLOT_TYPES


@dataclass
class NetConfig:
    board_emb: int = 128
    links_emb: int = 64
    trunk: int = 256
    policy_size: int = POLICY_SIZE
    global_len: int = be.GLOBAL_LEN
    hand_len: int = be.HAND_LEN
    opp_hands_len: int = be.HAND_LEN * 3


class PolicyValueNet(nn.Module):
    def __init__(self, cfg: NetConfig | None = None):
        super().__init__()
        self.cfg = cfg or NetConfig()

        self.board_enc = nn.Sequential(
            nn.Linear(be.BOARD_PLANES, self.cfg.board_emb), nn.ReLU()
        )
        self.links_enc = nn.Sequential(
            nn.Linear(be.LINK_PLANES, self.cfg.links_emb), nn.ReLU()
        )

        trunk_in = (
            2 * self.cfg.board_emb
            + 2 * self.cfg.links_emb
            + self.cfg.global_len
            + self.cfg.hand_len
            + self.cfg.opp_hands_len
        )
        self.trunk = nn.Sequential(
            nn.Linear(trunk_in, self.cfg.trunk),
            nn.ReLU(),
            nn.Linear(self.cfg.trunk, self.cfg.trunk),
            nn.ReLU(),
        )
        self.type_head = nn.Linear(self.cfg.trunk, N_ACTIONS)
        self.goal_head = nn.Linear(self.cfg.trunk, self.cfg.policy_size)
        self.value_head = nn.Linear(self.cfg.trunk, N_PLAYERS)

    def forward(self, batch: dict):
        """batch keys: board (B,17,49), links (B,6,39), global/own_hand/opp_hands (B,*).
        Returns (type_logits (B,7), goal_logits (B,P), value (B,4))."""
        # board: (B,17,49) -> (B,49,17) -> (B,49,H)
        b = batch["board"].transpose(1, 2)
        b = self.board_enc(b)
        b = torch.cat([b.mean(dim=1), b.max(dim=1).values], dim=1)  # (B,2H)

        l = batch["links"].transpose(1, 2)
        l = self.links_enc(l)
        l = torch.cat([l.mean(dim=1), l.max(dim=1).values], dim=1)  # (B,2H2)

        x = torch.cat(
            [b, l, batch["global"], batch["own_hand"], batch["opp_hands"]], dim=1
        )
        x = self.trunk(x)
        type_logits = self.type_head(x)
        goal_logits = self.goal_head(x)
        value = self.value_head(x)  # (B,4), no tanh
        return type_logits, goal_logits, value

    def merge_logits(self, type_logits, goal_logits) -> torch.Tensor:
        """logit(s) = type[t(s)] + goal[s] over the full policy table -> (B,P)."""
        st = torch.from_numpy(slot_types()).to(goal_logits.device)
        return goal_logits + type_logits.index_select(1, st)

    def policy_value(self, batch: dict):
        """Convenience for MCTS: returns (type (B,7), goal (B,P), value (B,4))
        under eval mode + no-grad, restoring the previous train/eval state."""
        was_training = self.training
        self.eval()
        try:
            with torch.no_grad():
                return self.forward(batch)
        finally:
            self.train(was_training)
