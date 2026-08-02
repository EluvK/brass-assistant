"""Policy-Value network for Brass: Birmingham.

Structure (AlphaZero-style, flat cell encoding since the map is a fixed graph
rather than a grid):

  board (B,17,49) --shared Linear(17->H)--> cell embeddings --mean+max pool--> (B,2H)
  links (B,6,39)  --shared Linear(6->H2)--> cell embeddings --mean+max pool--> (B,2H2)
  [board_emb, links_emb, global(50), own_hand(35), opp_hands(105)] -> trunk MLP
  policy head: 256 -> policy_table_size (1316) logits   (masked externally)
  value head : 256 -> 1 -> tanh                       (normalized VP estimate)

The value target is the perspective player's normalized final VP
z = (vp - mean)/std over the players of that game (see selfplay.py).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

import brass_engine as be

POLICY_SIZE = be.policy_table_size


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
        self.policy_head = nn.Linear(self.cfg.trunk, self.cfg.policy_size)
        self.value_head = nn.Linear(self.cfg.trunk, 1)

    def forward(self, batch: dict):
        """batch keys: board (B,17,49), links (B,6,39), global/own_hand/opp_hands (B,*).
        Returns (policy_logits (B,P), value (B,))."""
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
        policy = self.policy_head(x)
        value = torch.tanh(self.value_head(x).squeeze(-1))
        return policy, value

    def policy_value(self, batch: dict):
        """Convenience for MCTS: returns (logits (B,P), value (B,)) under
        eval mode + no-grad, restoring the previous train/eval state."""
        was_training = self.training
        self.eval()
        try:
            with torch.no_grad():
                return self.forward(batch)
        finally:
            self.train(was_training)
