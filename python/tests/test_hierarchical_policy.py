import numpy as np
import pytest
import torch

from brass_ai import _engine as be

from brass_ai.hierarchical_policy import (
    ACTION_KIND_COUNT,
    ACTION_SCHEMA_VERSION,
    REF_ID_BOUND,
    coalesce_equivalent_policy,
    encode_legal_candidates,
    encode_teacher_candidates,
    pad_candidate_features,
    rotate_to_perspective,
    split_action_rows,
    teacher_equivalence_policy,
)
from brass_ai.selfplay import Sample, materialize_sample


def test_engine_candidate_rows_match_declared_schema():
    state = be.GameState(seed=21, players=4)
    canonical, features = encode_legal_candidates(state)
    assert len(canonical) == len(features)
    assert features.shape[1] == be.ACTION_FEATURE_DIM
    rows = split_action_rows(features)
    assert (rows["kind"] < ACTION_KIND_COUNT).all()
    assert (rows["slot"] < 4).all()
    # Every declared reference must be in range for its kind, and the reference
    # mask must agree with the encoded count.
    assert (rows["ref_kind"] < be.REF_KIND_COUNT).all()
    for kind, bound in REF_ID_BOUND.items():
        selected = rows["ref_id"][rows["ref_mask"] & (rows["ref_kind"] == kind)]
        if selected.numel():
            assert int(selected.max()) < bound
    counted = features[:, be.ACTION_OFF_REF_COUNT]
    assert (counted >= 0).all()
    assert (counted <= be.ACTION_REF_CAP).all()
    assert rows["ref_mask"].sum(dim=1).tolist() == counted.long().tolist()


def test_teacher_candidates_are_engine_aligned_and_schema_versioned():
    state = be.GameState(seed=31, players=4)
    features, scores, _card_scores, canonical, selected, score, _card_score = encode_teacher_candidates(state)
    assert be.ACTION_SCHEMA_VERSION == ACTION_SCHEMA_VERSION
    assert canonical
    assert 0 <= selected < len(features)
    # Generator v4 emits up to SOURCE_VARIANTS (=2) source-identity variants per
    # geometry, so the per-class bound no longer holds: measured mean ~12.4,
    # upper bound 22 (docs/ai-action-encoding.md §6).
    assert len(features) <= 22
    assert features.shape == (len(features), be.ACTION_FEATURE_DIM)
    assert scores.shape == (len(features),)
    assert torch.isfinite(features).all()
    assert isinstance(score, float)


def test_snapshot_replay_materializes_current_full_legal_candidates():
    state = be.GameState(seed=51, players=4)
    teacher, _, _ = state.choose_heuristic_round()
    sample = Sample(
        pid=state.current_player_id, era=state.era,
        value=np.zeros(4, dtype=np.float32), winner=np.zeros(4, dtype=np.float32),
        econ=np.zeros(2, dtype=np.float32), snapshot=bytes(state.snapshot()),
        teacher_canonical=teacher,
    )
    restored = materialize_sample(sample)
    canonical, features = encode_legal_candidates(state)
    assert restored.candidates.dtype == np.float32
    assert restored.candidates.shape == features.numpy().shape
    np.testing.assert_array_equal(restored.candidates, features.numpy())
    assert restored.policy.sum() == 1.0
    teacher_index = canonical.index(teacher)
    equivalent = np.all(features.numpy() == features.numpy()[teacher_index], axis=1)
    np.testing.assert_array_equal(restored.policy > 0, equivalent)
    np.testing.assert_allclose(restored.policy[equivalent], 1.0 / equivalent.sum())


def test_teacher_policy_spreads_mass_over_identical_action_rows():
    features = np.zeros((4, be.ACTION_FEATURE_DIM), dtype=np.float32)
    features[1, 0] = 1.0
    features[2, 0] = 1.0
    features[3, 0] = 2.0
    policy = teacher_equivalence_policy(features, 1)
    np.testing.assert_array_equal(policy, np.asarray([0.0, 0.5, 0.5, 0.0], dtype=np.float32))


def test_policy_coalescing_preserves_equivalence_class_mass():
    features = np.zeros((3, be.ACTION_FEATURE_DIM), dtype=np.float32)
    features[1, 0] = features[2, 0] = 1.0
    policy = coalesce_equivalent_policy(features, np.asarray([0.2, 0.3, 0.5], dtype=np.float32))
    np.testing.assert_allclose(policy, np.asarray([0.2, 0.4, 0.4], dtype=np.float32))


def test_pad_candidate_features_masks_padding():
    rows = [torch.zeros(2, be.ACTION_FEATURE_DIM), torch.zeros(3, be.ACTION_FEATURE_DIM)]
    features, mask = pad_candidate_features(rows)
    assert features.shape == (2, 3, be.ACTION_FEATURE_DIM)
    assert mask.tolist() == [[True, True, False], [True, True, True]]
    with pytest.raises(ValueError):
        pad_candidate_features([torch.zeros(0, be.ACTION_FEATURE_DIM)])


def test_rotate_to_perspective_puts_the_actor_first():
    targets = torch.arange(8, dtype=torch.float32).reshape(2, 4)
    pid = torch.tensor([0, 3])
    rotated = rotate_to_perspective(targets, pid)
    torch.testing.assert_close(rotated[0], targets[0])
    torch.testing.assert_close(rotated[1], torch.tensor([7.0, 4.0, 5.0, 6.0]))


def test_full_state_snapshot_is_independent_and_not_history_growth():
    state = be.GameState(seed=61, players=4)
    sizes = []
    for _ in range(6):
        sizes.append(len(state.snapshot()))
        state.apply_move(state.choose_heuristic_round()[0])
    assert max(sizes) - min(sizes) < 512
    with pytest.raises(ValueError, match="unsupported GameState snapshot version"):
        be.GameState.from_snapshot(b"BASS\x01" + b"legacy")
