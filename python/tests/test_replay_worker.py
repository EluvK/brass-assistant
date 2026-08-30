"""Tests for the replay-web strategy worker protocol logic."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from brass_ai import _engine as be  # noqa: E402
from brass_ai.hierarchical_policy import encode_legal_candidates  # noqa: E402
from brass_ai.replay_worker import WorkerConfig, handle_request, load_net, root_forward  # noqa: E402
from brass_ai.rust_mcts import RustISMCTS, RustMCTSConfig  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
CKPT = Path(
    os.environ.get("BRASS_TEST_CKPT", REPO_ROOT / "checkpoints" / "bootstrap-0830-smoke.pt")
)

needs_ckpt = pytest.mark.skipif(
    not CKPT.exists(), reason=f"checkpoint not available: {CKPT}"
)


def _snapshot_and_legal(seed: int = 7, players: int = 2, steps: int = 3):
    state = be.GameState(seed=seed, players=players)
    for _ in range(steps):
        canonical, _, _ = state.choose_heuristic()
        state.apply_move(canonical)
    return bytes(state.snapshot()), [row[1] for row in state.legal_moves()]


@needs_ckpt
def test_load_net_matches_engine_schema():
    net = load_net(WorkerConfig(ckpt=str(CKPT)))
    assert net.cfg.action_features == be.ACTION_FEATURE_DIM


@needs_ckpt
def test_handle_request_policy_mode_returns_legal_choice():
    net = load_net(WorkerConfig(ckpt=str(CKPT), device="cpu"))
    snapshot, legal = _snapshot_and_legal()
    cfg = WorkerConfig(ckpt=str(CKPT), mode="policy", device="cpu")
    answer = handle_request(net, None, cfg, snapshot, legal)
    assert answer["canonical"] in legal
    evidence = answer["evidence"]
    assert evidence["mode"] == "policy"
    assert set(evidence["policy"]) >= set(legal)
    assert abs(sum(evidence["policy"].values()) - 1.0) < 1e-4
    assert evidence["visits"] == {}
    assert -1.0 <= evidence["root_value"] <= 1.0


@needs_ckpt
def test_handle_request_mcts_mode_returns_legal_choice():
    net = load_net(WorkerConfig(ckpt=str(CKPT), device="cpu"))
    snapshot, legal = _snapshot_and_legal()
    cfg = WorkerConfig(ckpt=str(CKPT), mode="mcts", sims=8, device="cpu")
    mcts = RustISMCTS(net, RustMCTSConfig(device="cpu"))
    answer = handle_request(net, mcts, cfg, snapshot, legal)
    assert answer["canonical"] in legal
    evidence = answer["evidence"]
    assert evidence["mode"] == "mcts"
    assert sum(evidence["visits"].values()) >= 8
    assert set(evidence["visits"]).issubset(set(legal))


@needs_ckpt
def test_root_forward_covers_all_legal_candidates():
    net = load_net(WorkerConfig(ckpt=str(CKPT), device="cpu"))
    state = be.GameState(seed=7, players=2)
    policy, value = root_forward(net, state, "cpu")
    canonicals, _ = encode_legal_candidates(state)
    assert set(policy) == set(canonicals)
    assert -1.0 <= value <= 1.0


@needs_ckpt
def test_handle_request_rejects_empty_legal_set():
    net = load_net(WorkerConfig(ckpt=str(CKPT), device="cpu"))
    snapshot, _legal = _snapshot_and_legal()
    cfg = WorkerConfig(ckpt=str(CKPT), mode="policy", device="cpu")
    with pytest.raises(ValueError):
        handle_request(net, None, cfg, snapshot, [])
