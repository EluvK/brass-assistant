"""Assemble PyTorch tensors from the Rust `state_to_tensor` numpy arrays.

Input layout (see src/engine/src/encode.rs):
  board    (B, 17, 49)  47 city slots + 2 farms, 17 planes
  links    (B,  6, 39)  39 connections, 6 planes
  global   (B, 50)
  own_hand (B, 35)
  opp_hands(B, 105)     3 opponents x 35

`encode_state` / `encode_states` return a dict of tensors; `net.py` consumes it.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import torch

import brass_engine as be


def tensor_spec() -> dict:
    return {
        "board": (be.BOARD_PLANES, be.BOARD_CELLS),
        "links": (be.LINK_PLANES, be.LINK_CELLS),
        "global": (be.GLOBAL_LEN,),
        "own_hand": (be.HAND_LEN,),
        "opp_hands": (be.HAND_LEN * 3,),
    }


def feature_dim() -> int:
    s = tensor_spec()
    return (
        s["board"][0] * s["board"][1]
        + s["links"][0] * s["links"][1]
        + s["global"][0]
        + s["own_hand"][0]
        + s["opp_hands"][0]
    )


def encode_state(state) -> dict:
    """Encode one GameState into a dict of batched (leading dim 1) tensors."""
    board, links, global_vec, own_hand, opp_hands = state.state_to_tensor()
    return encode_arrays([board], [links], [global_vec], [own_hand], [opp_hands])


def encode_states(states: Iterable) -> dict:
    """Encode a sequence of GameState objects into a dict of batched tensors."""
    boards, links, globals_, own_hands, opp_hands = [], [], [], [], []
    for s in states:
        b, l, g, oh, op = s.state_to_tensor()
        boards.append(b)
        links.append(l)
        globals_.append(g)
        own_hands.append(oh)
        opp_hands.append(op)
    if not boards:
        raise ValueError("encode_states requires at least one state")
    return encode_arrays(boards, links, globals_, own_hands, opp_hands)


def encode_arrays(boards, links, globals_, own_hands, opp_hands) -> dict:
    b = np.stack([np.asarray(x) for x in boards]).astype(np.float32)
    l = np.stack([np.asarray(x) for x in links]).astype(np.float32)
    g = np.stack([np.asarray(x) for x in globals_]).astype(np.float32)
    o = np.stack([np.asarray(x) for x in own_hands]).astype(np.float32)
    p = np.stack([np.asarray(x) for x in opp_hands]).astype(np.float32)
    return {
        "board": torch.from_numpy(b),
        "links": torch.from_numpy(l),
        "global": torch.from_numpy(g),
        "own_hand": torch.from_numpy(o),
        "opp_hands": torch.from_numpy(p),
    }


def move_tensors_to_device(batch: dict, device):
    return {k: v.to(device) for k, v in batch.items()}
