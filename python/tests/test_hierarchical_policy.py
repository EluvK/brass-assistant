import numpy as np
import pytest
import torch

from brass_ai import _engine as be

from brass_ai.hierarchical_policy import (
    ACTION_FEATURE_SCHEMA_VERSION,
    coalesce_equivalent_policy,
    compress_candidate_features,
    encode_legal_candidates,
    encode_teacher_candidates,
    pad_candidate_features,
    teacher_equivalence_policy,
)
from brass_ai.net import PolicyValueNet
from brass_ai.selfplay import Sample, materialize_sample


def test_engine_candidate_features_match_declared_schema():
    state = be.GameState(seed=21, players=4)
    canonical, features = encode_legal_candidates(state)
    assert len(canonical) == len(features)
    assert features.shape[1] == be.ACTION_FEATURE_DIM
    assert torch.all(features[:, :7].sum(dim=1) == 1)


def test_teacher_candidates_are_engine_aligned_and_schema_versioned():
    state = be.GameState(seed=31, players=4)
    features, scores, _card_scores, canonical, selected, score, _card_score = encode_teacher_candidates(state)
    assert be.ACTION_FEATURE_SCHEMA_VERSION == ACTION_FEATURE_SCHEMA_VERSION
    assert canonical
    assert 0 <= selected < len(features)
    assert len(features) <= 14  # top-4 Build + top-4 Network + six other action classes
    assert features.shape == (len(features), be.ACTION_FEATURE_DIM)
    assert scores.shape == (len(features),)
    assert torch.isfinite(features).all()
    assert isinstance(score, float)


def test_candidate_feature_compression_is_lossless():
    state = be.GameState(seed=41, players=4)
    _canonical, features = encode_legal_candidates(state)
    packed = compress_candidate_features(features.numpy())
    restored = packed.astype(np.float32) / 4.0
    np.testing.assert_array_equal(restored, features.numpy())


def test_snapshot_replay_materializes_current_full_legal_candidates():
    state = be.GameState(seed=51, players=4)
    teacher, _, _ = state.choose_heuristic()
    sample = Sample(
        pid=state.current_player_id, era=state.era, rank=np.zeros(4, dtype=np.float32), winner=np.zeros(4, dtype=np.float32),
        econ=np.zeros(2, dtype=np.float32), snapshot=bytes(state.snapshot()),
        teacher_canonical=teacher,
    )
    restored = materialize_sample(sample)
    canonical, features = encode_legal_candidates(state)
    assert restored.candidates.dtype == np.uint8  # Rust-side quarter-step packing
    assert restored.candidates.shape == features.numpy().shape
    np.testing.assert_array_equal(
        restored.candidates.astype(np.float32) / 4.0, features.numpy()
    )
    assert restored.policy.sum() == 1.0
    teacher_index = canonical.index(teacher)
    equivalent = np.all(features.numpy() == features.numpy()[teacher_index], axis=1)
    np.testing.assert_array_equal(restored.policy > 0, equivalent)
    np.testing.assert_allclose(restored.policy[equivalent], 1.0 / equivalent.sum())


def test_teacher_policy_spreads_mass_over_identical_action_features():
    features = np.zeros((4, be.ACTION_FEATURE_DIM), dtype=np.float32)
    features[1, 7] = 1.0
    features[2, 7] = 1.0
    features[3, 8] = 1.0
    policy = teacher_equivalence_policy(features, 1)
    np.testing.assert_array_equal(policy, np.asarray([0.0, 0.5, 0.5, 0.0], dtype=np.float32))


def test_policy_coalescing_preserves_equivalence_class_mass():
    features = np.zeros((3, be.ACTION_FEATURE_DIM), dtype=np.float32)
    features[1, 7] = features[2, 7] = 1.0
    policy = coalesce_equivalent_policy(features, np.asarray([0.2, 0.3, 0.5], dtype=np.float32))
    np.testing.assert_allclose(policy, np.asarray([0.2, 0.4, 0.4], dtype=np.float32))


def test_full_state_snapshot_is_independent_and_not_history_growth():
    state = be.GameState(seed=61, players=4)
    sizes = []
    for _ in range(6):
        sizes.append(len(state.snapshot()))
        state.apply_move(state.choose_heuristic()[0])
    assert max(sizes) - min(sizes) < 512
    with pytest.raises(ValueError, match="unsupported GameState snapshot version"):
        be.GameState.from_snapshot(b"BASS\x01" + b"legacy")
