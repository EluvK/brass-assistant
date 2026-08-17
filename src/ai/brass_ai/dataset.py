"""Versioned on-disk replay shards for the Rust-search training pipeline."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .selfplay import Sample


def save_samples(path: str | Path, samples: list[Sample]) -> int:
    """Write one complete self-play batch as a compressed NumPy shard."""
    if not samples:
        raise ValueError("refusing to write an empty replay shard")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {
        "pid": np.asarray([s.pid for s in samples], dtype=np.int8),
        "era": np.asarray([s.era for s in samples], dtype=np.int8),
        "board": np.stack([s.board for s in samples]).astype(np.float32),
        "links": np.stack([s.links for s in samples]).astype(np.float32),
        "global_vec": np.stack([s.global_vec for s in samples]).astype(np.float32),
        "own_hand": np.stack([s.own_hand for s in samples]).astype(np.float32),
        "opp_hands": np.stack([s.opp_hands for s in samples]).astype(np.float32),
        "policy": np.stack([s.policy for s in samples]).astype(np.float32),
        "value": np.stack([s.value for s in samples]).astype(np.float32),
        "econ": np.stack([s.econ for s in samples]).astype(np.float32),
        "legal": np.stack([s.legal for s in samples]).astype(np.bool_),
    }
    np.savez_compressed(path, **arrays)
    return len(samples)


def load_samples(path: str | Path) -> list[Sample]:
    """Load a replay shard written by :func:`save_samples`."""
    with np.load(path, allow_pickle=False) as data:
        n = len(data["pid"])
        return [
            Sample(
                pid=int(data["pid"][i]), era=int(data["era"][i]),
                board=data["board"][i], links=data["links"][i],
                global_vec=data["global_vec"][i], own_hand=data["own_hand"][i],
                opp_hands=data["opp_hands"][i], policy=data["policy"][i],
                value=data["value"][i], econ=data["econ"][i],
                legal=data["legal"][i],
            )
            for i in range(n)
        ]
