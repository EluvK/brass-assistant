import numpy as np
import torch

import brass_engine as be

from brass_ai import build_input
from brass_ai.hierarchical_policy import encode_legal_candidates, pad_candidate_features
from brass_ai.net import PolicyValueNet


def test_spec_matches_binding():
    spec = build_input.tensor_spec()
    assert spec["board"] == (be.BOARD_PLANES, be.BOARD_CELLS)
    assert spec["links"] == (be.LINK_PLANES, be.LINK_CELLS)
    assert spec["global"] == (be.GLOBAL_LEN,)
    assert spec["own_hand"] == (be.HAND_LEN,)
    assert spec["opp_hands"] == (be.HAND_LEN * 3,)


def test_encode_state_shapes():
    g = be.GameState(seed=1, players=4)
    b = build_input.encode_state(g)
    assert b["board"].shape == (1, 17, 49)
    assert b["links"].shape == (1, 6, 39)
    assert b["global"].shape == (1, 50)
    assert b["own_hand"].shape == (1, 35)
    assert b["opp_hands"].shape == (1, 105)
    for v in b.values():
        assert v.dtype == torch.float32
        assert bool((v >= 0).all()) and bool((v <= 1).all())


def test_encode_states_batching():
    states = [be.GameState(seed=s, players=4) for s in range(3)]
    b = build_input.encode_states(states)
    assert b["board"].shape == (3, 17, 49)
    assert b["global"].shape == (3, 50)


def test_net_forward():
    torch.manual_seed(0)
    net = PolicyValueNet()
    g = be.GameState(seed=5, players=4)
    batch = build_input.encode_states([g, g, g, g])
    _, candidates = encode_legal_candidates(g)
    features, mask = pad_candidate_features([candidates] * 4)
    out = net(batch, features, mask)
    assert out["type_logits"].shape == (4, 7)
    assert out["candidate_logits"].shape == mask.shape
    assert out["value"].shape == (4, 4)
    assert out["econ"].shape == (4, 2)
    assert bool(torch.isfinite(out["value"]).all())
    assert bool(torch.isfinite(out["econ"]).all())
