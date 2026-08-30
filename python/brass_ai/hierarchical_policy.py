"""Engine-candidate adapter.

Owns conversion of Engine-produced legal candidates into network inputs:
schema validation, equivalence-class policy targets, uint8 compression and
padded batching.
"""

from __future__ import annotations

import numpy as np
import torch

from . import _engine as be

ACTION_FEATURE_SCHEMA_VERSION = 4
ACTION_FEATURE_DIM = 301
ACTION_FEATURE_SCALE = 4


def _feature_width() -> int:
    version = getattr(be, "ACTION_FEATURE_SCHEMA_VERSION", ACTION_FEATURE_SCHEMA_VERSION)
    if version != ACTION_FEATURE_SCHEMA_VERSION:
        raise RuntimeError(f"unsupported action-feature schema version: {version}")
    return getattr(be, "ACTION_FEATURE_DIM", ACTION_FEATURE_DIM)


def encode_legal_candidates(state) -> tuple[list[str], torch.Tensor]:
    """Return concrete canonical moves and their Engine-owned feature rows."""
    candidates = state.legal_candidates()
    if not candidates:
        raise ValueError("state has no legal candidates")
    canonical, features = zip(*candidates)
    array = np.asarray(features, dtype=np.float32)
    if array.ndim != 2 or array.shape[1] != _feature_width():
        raise ValueError("engine returned an invalid action-feature schema")
    return list(canonical), torch.from_numpy(array)


def encode_teacher_candidates(state) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, str, int, float, float]:
    """Return compact Rust-aligned teacher features and ranking scores."""
    flat_features, scores, card_scores, canonical, teacher_index, score, card_score, count = state.heuristic_candidates()
    width = _feature_width()
    array = np.asarray(flat_features, dtype=np.float32)
    if count <= 0 or array.size != count * width:
        raise ValueError("engine returned an invalid teacher action schema")
    if len(card_scores) != count:
        raise ValueError("engine returned invalid teacher card scores")
    if not 0 <= teacher_index < count:
        raise ValueError("engine returned an invalid teacher candidate index")
    return (
        torch.from_numpy(array.reshape(count, width)),
        torch.from_numpy(np.asarray(scores, dtype=np.float32)),
        torch.from_numpy(np.asarray(card_scores, dtype=np.float32)),
        str(canonical),
        int(teacher_index),
        float(score),
        float(card_score),
    )


def compress_candidate_features(features: np.ndarray) -> np.ndarray:
    """Losslessly pack quarter-step action features into uint8 replay data."""
    array = np.asarray(features, dtype=np.float32)
    if array.ndim != 2 or array.shape[1] != _feature_width():
        raise ValueError("invalid action features for compression")
    scaled = array * ACTION_FEATURE_SCALE
    if not np.all(np.isfinite(scaled)) or not np.allclose(scaled, np.rint(scaled)):
        raise ValueError("action features contain values outside the quarter-step schema")
    if scaled.min() < 0 or scaled.max() > 255:
        raise ValueError("action features do not fit uint8 storage")
    return np.rint(scaled).astype(np.uint8)


def pad_candidate_features(rows: list[torch.Tensor], device=None) -> tuple[torch.Tensor, torch.Tensor]:
    """Pad variable candidate sets into ``(B,max_N,D)`` plus a boolean mask."""
    if not rows:
        raise ValueError("cannot pad an empty candidate batch")
    width = _feature_width()
    if any(row.ndim != 2 or row.shape[0] == 0 or row.shape[1] != width for row in rows):
        raise ValueError("all candidate rows must have shape (N, ACTION_FEATURE_DIM), N > 0")
    max_n = max(row.shape[0] for row in rows)
    features = torch.zeros(len(rows), max_n, width, dtype=torch.float32, device=device)
    mask = torch.zeros(len(rows), max_n, dtype=torch.bool, device=device)
    for i, row in enumerate(rows):
        n = row.shape[0]
        features[i, :n] = row if device is None else row.to(device)
        mask[i, :n] = True
    return features, mask


def coalesce_equivalent_policy(features: np.ndarray, policy: np.ndarray) -> np.ndarray:
    """Spread each concrete-policy mass uniformly across equal feature rows."""
    array = np.asarray(features, dtype=np.float32)
    target = np.asarray(policy, dtype=np.float32)
    if array.ndim != 2 or array.shape[1] != _feature_width():
        raise ValueError("invalid candidate features for policy target")
    if target.shape != (len(array),) or not np.isfinite(target).all() or (target < 0).any():
        raise ValueError("invalid policy target")
    total = float(target.sum())
    if total <= 0:
        raise ValueError("policy target must contain positive mass")
    out = np.zeros(len(array), dtype=np.float32)
    pending = np.flatnonzero(target > 0.0)
    # MCTS visit targets have at most `sims` non-zero rows and shortlist
    # targets are small. Work only on those classes instead of constructing a
    # structured np.unique key for every full-legal candidate.
    while len(pending):
        index = pending[0]
        equivalent = np.all(array == array[index], axis=1)
        out[equivalent] = target[equivalent].sum() / equivalent.sum() / total
        pending = pending[~equivalent[pending]]
    return out


def teacher_equivalence_policy(features: np.ndarray, teacher_index: int) -> np.ndarray:
    """Target the complete v4-observable equivalence class of a teacher move.

    v4 intentionally omits execution-only identities such as the index of an
    otherwise identical card.  Several concrete legal moves can therefore
    share one exact feature row.  A one-hot target for one arbitrary canonical
    move is contradictory: the candidate scorer receives identical inputs and
    must emit identical logits.  The teacher mass is consequently distributed
    uniformly across that equivalence class.
    """
    array = np.asarray(features, dtype=np.float32)
    if array.ndim != 2 or array.shape[1] != _feature_width():
        raise ValueError("invalid candidate features for teacher target")
    if not 0 <= teacher_index < len(array):
        raise ValueError("teacher index is outside candidate features")
    # This is intentionally not routed through `coalesce_equivalent_policy`:
    # teacher imitation has one non-zero source, so sorting every candidate row
    # with np.unique only creates large temporary structured arrays.
    equivalent = np.all(array == array[teacher_index], axis=1)
    policy = equivalent.astype(np.float32)
    return policy / policy.sum()
