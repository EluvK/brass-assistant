"""Versioned on-disk replay shards for the Rust-search training pipeline."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from .hierarchical_policy import (
    ACTION_FEATURE_DIM,
    ACTION_FEATURE_SCHEMA_VERSION,
    pad_candidate_features,
)

from .selfplay import Sample


def save_samples(path: str | Path, samples: list[Sample]) -> int:
    """Write one complete self-play batch as a compressed NumPy shard."""
    if not samples:
        raise ValueError("refusing to write an empty replay shard")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    candidates, candidate_mask = pad_candidate_features([
        torch.from_numpy(s.candidates) for s in samples
    ])
    policy = np.zeros(candidate_mask.shape, dtype=np.float32)
    for i, sample in enumerate(samples):
        policy[i, :len(sample.policy)] = sample.policy
    arrays = {
        "pid": np.asarray([s.pid for s in samples], dtype=np.int8),
        "era": np.asarray([s.era for s in samples], dtype=np.int8),
        "board": np.stack([s.board for s in samples]).astype(np.float32),
        "links": np.stack([s.links for s in samples]).astype(np.float32),
        "global_vec": np.stack([s.global_vec for s in samples]).astype(np.float32),
        "own_hand": np.stack([s.own_hand for s in samples]).astype(np.float32),
        "opp_hands": np.stack([s.opp_hands for s in samples]).astype(np.float32),
        "candidates": candidates.numpy().astype(np.float32),
        "candidate_mask": candidate_mask.numpy().astype(np.bool_),
        "policy": policy,
        "value": np.stack([s.value for s in samples]).astype(np.float32),
        "econ": np.stack([s.econ for s in samples]).astype(np.float32),
        "action_feature_dim": np.asarray(ACTION_FEATURE_DIM, dtype=np.int16),
        "action_feature_schema_version": np.asarray(
            ACTION_FEATURE_SCHEMA_VERSION, dtype=np.int16
        ),
    }
    np.savez_compressed(path, **arrays)
    return len(samples)


def load_samples(path: str | Path) -> list[Sample]:
    """Load a replay shard written by :func:`save_samples`."""
    with np.load(path, allow_pickle=False) as data:
        try:
            schema_version = int(data["action_feature_schema_version"])
            feature_dim = int(data["action_feature_dim"])
        except KeyError as exc:
            raise ValueError("replay shard has no action-feature schema metadata") from exc
        if schema_version != ACTION_FEATURE_SCHEMA_VERSION or feature_dim != ACTION_FEATURE_DIM:
            raise ValueError(
                "incompatible replay action-feature schema: "
                f"got version={schema_version}, dim={feature_dim}; expected "
                f"version={ACTION_FEATURE_SCHEMA_VERSION}, dim={ACTION_FEATURE_DIM}"
            )
        n = len(data["pid"])
        return [
            Sample(
                pid=int(data["pid"][i]), era=int(data["era"][i]),
                board=data["board"][i], links=data["links"][i],
                global_vec=data["global_vec"][i], own_hand=data["own_hand"][i],
                opp_hands=data["opp_hands"][i],
                candidates=data["candidates"][i, data["candidate_mask"][i]],
                policy=data["policy"][i, data["candidate_mask"][i]],
                value=data["value"][i], econ=data["econ"][i],
            )
            for i in range(n)
        ]
