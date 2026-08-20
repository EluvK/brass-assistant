"""Candidate-policy helpers.

The implementation lives in :mod:`brass_ai.net`; this module owns conversion
of Engine-produced legal candidates to a padded batch.
"""

from __future__ import annotations

import numpy as np
import torch

import brass_engine as be

ACTION_FEATURE_SCHEMA_VERSION = 2
ACTION_FEATURE_DIM = 235


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


def encode_teacher_candidates(state) -> tuple[torch.Tensor, torch.Tensor, str, int, float]:
    """Return compact Rust-aligned teacher features and ranking scores."""
    flat_features, scores, canonical, teacher_index, score, count = state.heuristic_candidates()
    width = _feature_width()
    array = np.asarray(flat_features, dtype=np.float32)
    if count <= 0 or array.size != count * width:
        raise ValueError("engine returned an invalid teacher action schema")
    if not 0 <= teacher_index < count:
        raise ValueError("engine returned an invalid teacher candidate index")
    return (
        torch.from_numpy(array.reshape(count, width)),
        torch.from_numpy(np.asarray(scores, dtype=np.float32)),
        str(canonical),
        int(teacher_index),
        float(score),
    )


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
