import numpy as np
import pytest
import torch

from brass_ai import _engine as be

from brass_ai.net import PolicyValueNet
from brass_ai.rust_mcts import RustISMCTS, RustMCTSConfig
from brass_ai.selfplay import SelfPlayConfig, play_game


def _make():
    torch.manual_seed(0)
    torch.set_num_threads(2)
    net = PolicyValueNet()
    return RustISMCTS(net, RustMCTSConfig(device="cpu", max_depth=6, batch_size=8))


def test_rust_search_returns_executable_move_and_visits():
    mcts = _make()
    game = be.GameState(seed=7, players=4)
    result = mcts.search(game, sims=12)
    assert result.best is not None
    copy = game.clone()
    copy.apply_move(result.best)
    assert sum(result.visits.values()) == 12
    legal = {canonical for _, canonical, _ in game.legal_moves()}
    assert set(result.canon_by_candidate.values()) <= legal


def test_rust_selfplay_produces_complete_game_samples():
    samples, vps = play_game(
        _make(), SelfPlayConfig(sims=2, max_moves=600, seed=5, temperature=0.0)
    )
    assert samples
    assert len(vps) == 4
    for sample in samples:
        assert np.isclose(sample.policy.sum(), 1.0)
        assert sample.rank.shape == (4,)
        assert sample.econ.shape == (2,)
        assert np.isfinite(sample.rank).all()


def test_truncated_selfplay_is_rejected():
    with pytest.raises(RuntimeError, match="samples discarded"):
        play_game(_make(), SelfPlayConfig(sims=1, max_moves=1, seed=5))
