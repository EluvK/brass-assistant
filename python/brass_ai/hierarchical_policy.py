"""Engine-candidate adapter.

Owns everything Python knows about the Rust-side schema of
`docs/ai-action-encoding.md`: schema validation, action-reference splitting,
state-token grouping, padded batching, and equivalence-class policy targets.
No other Python module may hardcode a feature offset or width.
"""

from __future__ import annotations

import numpy as np
import torch

from . import _engine as be

# --- schema -----------------------------------------------------------------

ACTION_SCHEMA_VERSION = 1
STATE_TOKEN_SCHEMA_VERSION = 2
ACTION_FEATURE_DIM = 55


def _check_schema() -> None:
    if be.ACTION_SCHEMA_VERSION != ACTION_SCHEMA_VERSION:
        raise RuntimeError(
            f"unsupported action schema: engine={be.ACTION_SCHEMA_VERSION} "
            f"python={ACTION_SCHEMA_VERSION}"
        )
    if be.STATE_TOKEN_SCHEMA_VERSION != STATE_TOKEN_SCHEMA_VERSION:
        raise RuntimeError(
            f"unsupported state token schema: engine={be.STATE_TOKEN_SCHEMA_VERSION} "
            f"python={STATE_TOKEN_SCHEMA_VERSION}"
        )
    if be.ACTION_FEATURE_DIM != ACTION_FEATURE_DIM:
        raise RuntimeError(
            f"action width mismatch: engine={be.ACTION_FEATURE_DIM} "
            f"python={ACTION_FEATURE_DIM}"
        )


_check_schema()

#: State token groups: name -> (token count, feature width).
STATE_GROUPS: dict[str, tuple[int, int]] = {
    "cells": (be.BOARD_CELLS, be.F_CELL),
    "links": (be.LINK_CELLS, be.F_LINK),
    "merchants": (be.MERCHANT_COUNT, be.F_MERCHANT),
    "seats": (be.SEAT_COUNT, be.F_SEAT),
}
#: The global token is a single vector, not a group.
STATE_GLOBAL = ("global", be.F_GLOBAL)
TOKEN_TYPES = ("cells", "links", "merchants", "seats", "global")

# --- action references ------------------------------------------------------

ACTION_REF_CAP = be.ACTION_REF_CAP
ACTION_KIND_COUNT = be.ACTION_KIND_COUNT
ACTION_NUMBERS = be.ACTION_NUMBERS
REF_KIND_COUNT = be.REF_KIND_COUNT
REF_CELL = be.ACTION_REF_CELL
REF_LINK = be.ACTION_REF_LINK
REF_MERCHANT = be.ACTION_REF_MERCHANT
REF_INDUSTRY = be.ACTION_REF_INDUSTRY
REF_CARD = be.ACTION_REF_CARD

#: Id bound per reference kind (industry and card ids index embedding tables).
REF_ID_BOUND = {
    REF_CELL: be.BOARD_CELLS,
    REF_LINK: be.LINK_CELLS,
    REF_MERCHANT: be.MERCHANT_COUNT,
    REF_INDUSTRY: be.INDUSTRY_COUNT,
    REF_CARD: be.CARD_SEMANTIC_COUNT,
}

#: Token group a reference kind resolves against (industry/card use static
#: embeddings instead of state tokens).
REF_TOKEN_GROUP = {
    REF_CELL: "cells",
    REF_LINK: "links",
    REF_MERCHANT: "merchants",
}


def _check_action_rows(array: np.ndarray, what: str) -> np.ndarray:
    array = np.ascontiguousarray(array, dtype=np.float32)
    if array.ndim != 2 or array.shape[1] != ACTION_FEATURE_DIM:
        raise ValueError(f"{what} must have shape (N, {ACTION_FEATURE_DIM})")
    if array.shape[0] == 0:
        raise ValueError(f"{what} must contain at least one candidate")
    return array


def split_action_rows(rows: torch.Tensor) -> dict[str, torch.Tensor]:
    """Split flat action rows into the reference tensors the network consumes."""
    kind = rows[..., be.ACTION_OFF_KIND].long()
    slot = rows[..., be.ACTION_OFF_SLOT].long()
    numbers = rows[..., be.ACTION_OFF_NUMBERS:be.ACTION_OFF_NUMBERS + ACTION_NUMBERS]
    count = rows[..., be.ACTION_OFF_REF_COUNT].long()
    refs = rows[..., be.ACTION_OFF_REFS:].reshape(*rows.shape[:-1], ACTION_REF_CAP, 3)
    ref_kind = refs[..., 0].long()
    ref_id = refs[..., 1].long()
    ref_weight = refs[..., 2]
    # Unused slots are zeroed on the Rust side; make the mask explicit so a
    # padded candidate can never contribute a reference.
    position = torch.arange(ACTION_REF_CAP, device=rows.device)
    ref_mask = position < count.unsqueeze(-1)
    return {
        "kind": kind,
        "slot": slot,
        "numbers": numbers,
        "ref_kind": ref_kind,
        "ref_id": ref_id,
        "ref_weight": ref_weight * ref_mask.to(ref_weight.dtype),
        "ref_mask": ref_mask,
    }


def encode_legal_candidates(state, device=None) -> tuple[list[str], torch.Tensor]:
    """Return concrete canonical moves and their Engine-owned reference rows."""
    canonical, features = state.legal_candidates()
    array = _check_action_rows(np.asarray(features, dtype=np.float32), "legal candidates")
    return list(canonical), torch.from_numpy(array).to(device) if device else torch.from_numpy(array)


def encode_teacher_candidates(state) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, str, int, float, float]:
    """Return compact Rust-aligned teacher features and ranking scores."""
    features, scores, card_scores, canonical, teacher_index, score, card_score, count = (
        state.heuristic_candidates()
    )
    array = _check_action_rows(np.asarray(features, dtype=np.float32), "teacher candidates")
    if count <= 0 or array.shape != (count, ACTION_FEATURE_DIM):
        raise ValueError("engine returned an invalid teacher action schema")
    scores = np.asarray(scores, dtype=np.float32)
    card_scores = np.asarray(card_scores, dtype=np.float32)
    if len(card_scores) != count:
        raise ValueError("engine returned invalid teacher card scores")
    if not 0 <= teacher_index < count:
        raise ValueError("engine returned an invalid teacher candidate index")
    return (
        torch.from_numpy(array),
        torch.from_numpy(scores),
        torch.from_numpy(card_scores),
        str(canonical),
        int(teacher_index),
        float(score),
        float(card_score),
    )


def pad_candidate_features(rows: list[torch.Tensor], device=None) -> tuple[torch.Tensor, torch.Tensor]:
    """Pad variable candidate sets into ``(B,max_N,D)`` plus a boolean mask."""
    if not rows:
        raise ValueError("cannot pad an empty candidate batch")
    if any(row.ndim != 2 or row.shape[0] == 0 or row.shape[1] != ACTION_FEATURE_DIM
           for row in rows):
        raise ValueError(
            f"all candidate rows must have shape (N, {ACTION_FEATURE_DIM}), N > 0"
        )
    max_n = max(row.shape[0] for row in rows)
    features = torch.zeros(len(rows), max_n, ACTION_FEATURE_DIM, dtype=torch.float32, device=device)
    mask = torch.zeros(len(rows), max_n, dtype=torch.bool, device=device)
    for i, row in enumerate(rows):
        n = row.shape[0]
        features[i, :n] = row if device is None else row.to(device)
        mask[i, :n] = True
    return features, mask


def coalesce_equivalent_policy(features: np.ndarray, policy: np.ndarray) -> np.ndarray:
    """Spread each concrete-policy mass uniformly across equal feature rows.

    Delegates to the Rust engine (`_engine.coalesce_equivalent_policy`): the
    self-play hot path coalesces a full-legal candidate matrix every move and
    the numpy implementation allocates a boolean class mask per class.
    """
    array = np.ascontiguousarray(features, dtype=np.float32)
    target = np.ascontiguousarray(policy, dtype=np.float32)
    return be.coalesce_equivalent_policy(array, target)


def teacher_equivalence_policy(features: np.ndarray, teacher_index: int) -> np.ndarray:
    """Target the complete observable equivalence class of a teacher move.

    The action schema intentionally omits execution-only identities, so several
    concrete legal moves can share one exact reference row. A one-hot target for
    one arbitrary canonical move would be contradictory: the candidate scorer
    receives identical inputs and must emit identical logits. The teacher mass
    is therefore distributed uniformly across that equivalence class.
    """
    array = np.asarray(features, dtype=np.float32)
    if array.ndim != 2 or array.shape[1] != ACTION_FEATURE_DIM:
        raise ValueError("invalid candidate features for teacher target")
    if not 0 <= teacher_index < len(array):
        raise ValueError("teacher index is outside candidate features")
    # This is intentionally not routed through `coalesce_equivalent_policy`:
    # teacher imitation has one non-zero source, so sorting every candidate row
    # with np.unique only creates large temporary structured arrays.
    equivalent = np.all(array == array[teacher_index], axis=1)
    policy = equivalent.astype(np.float32)
    return policy / policy.sum()


def rotate_to_perspective(targets: torch.Tensor, pid: torch.Tensor) -> torch.Tensor:
    """Rotate per-seat targets so index 0 is the acting player.

    Observations are encoded from the acting player's perspective, so the value
    and winner heads predict "me first"; the stored targets are in absolute seat
    order and have to be rotated the same way.
    """
    seats = torch.arange(targets.shape[1], device=targets.device)
    index = (seats.unsqueeze(0) + pid.unsqueeze(1)) % targets.shape[1]
    return targets.gather(1, index)
