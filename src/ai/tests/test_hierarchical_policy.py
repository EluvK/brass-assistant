import torch

import brass_engine as be

from brass_ai import build_input
from brass_ai.hierarchical_policy import (
    ACTION_FEATURE_SCHEMA_VERSION,
    encode_legal_candidates,
    encode_teacher_candidates,
    pad_candidate_features,
)
from brass_ai.net import PolicyValueNet


def test_engine_candidates_feed_variable_candidate_policy():
    states = [be.GameState(seed=11, players=4), be.GameState(seed=12, players=4)]
    candidates = [encode_legal_candidates(state) for state in states]
    canonical, rows = zip(*candidates)
    features, mask = pad_candidate_features(list(rows))
    out = PolicyValueNet()(build_input.encode_states(states), features, mask)

    assert len(canonical[0]) > 0
    assert features.shape[-1] == be.ACTION_FEATURE_DIM
    assert out["candidate_logits"].shape == mask.shape
    assert torch.allclose(out["candidate_log_probs"].exp().sum(1), torch.ones(2))


def test_engine_candidate_features_match_declared_schema():
    state = be.GameState(seed=21, players=4)
    canonical, features = encode_legal_candidates(state)
    assert len(canonical) == len(features)
    assert features.shape[1] == be.ACTION_FEATURE_DIM
    assert torch.all(features[:, :7].sum(dim=1) == 1)


def test_teacher_candidates_are_engine_aligned_and_schema_versioned():
    state = be.GameState(seed=31, players=4)
    features, scores, canonical, selected, score = encode_teacher_candidates(state)
    assert be.ACTION_FEATURE_SCHEMA_VERSION == ACTION_FEATURE_SCHEMA_VERSION
    assert canonical
    assert 0 <= selected < len(features)
    assert len(features) <= 14  # top-4 Build + top-4 Network + six other action classes
    assert features.shape == (len(features), be.ACTION_FEATURE_DIM)
    assert scores.shape == (len(features),)
    assert torch.isfinite(features).all()
    assert isinstance(score, float)
