import numpy as np
import pytest
import torch

from brass_ai import _engine as be

from brass_ai import selfplay as selfplay_module
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


def test_small_search_budget_uses_leaf_feedback_before_choosing():
    outputs = []
    for sign in (-1, 1):
        calls = []

        def callback(cells, links, merchants, seats, glob, actions, mask):
            rows, width = np.asarray(mask).shape
            calls.append(rows)
            values = np.zeros((rows, 4), dtype=np.float32)
            if len(calls) > 1:
                values[:] = sign * np.arange(1, 5, dtype=np.float32)
            logits = np.broadcast_to(-0.01 * np.arange(width, dtype=np.float32), (rows, width)).copy()
            return logits, values, np.zeros((rows, width), dtype=np.float32)

        state = be.GameState(seed=42, players=4)
        before = bytes(state.snapshot())
        result = state.search_net(callback, 64, 0.25, 10, 0.3, 0.15,
                                  False, 64, 0, 0, False, 0.0, False)
        assert bytes(state.snapshot()) == before
        assert len(calls) > 2
        assert sum(child[2] for child in result[1]) == 64
        outputs.append({child[1]: child[2] for child in result[1]})
    assert outputs[0] != outputs[1]


def test_rust_selfplay_produces_complete_game_samples():
    samples, vps = play_game(
        _make(),
        SelfPlayConfig(sims=2, max_moves=600, seed=5, temperature=0.0,
                       store_snapshots=False),
    )
    assert samples
    assert len(vps) == 4
    for sample in samples:
        assert np.isclose(sample.policy.sum(), 1.0)
        assert sample.value.shape == (4,)
        assert sample.econ.shape == (2,)
        assert np.isfinite(sample.value).all()
        assert np.isclose(float(sample.value.sum()), 0.0, atol=1e-5)


def test_truncated_selfplay_is_rejected():
    with pytest.raises(RuntimeError, match="samples discarded"):
        play_game(_make(), SelfPlayConfig(sims=1, max_moves=1, seed=5))


def test_game_log_replays_the_recorded_final_scores(tmp_path):
    import json
    from brass_ai.rust_mcts import heuristic_search
    from brass_ai.selfplay import play_game_with_roles

    _, vps = play_game_with_roles([heuristic_search] * 4,
        SelfPlayConfig(seed=23, temperature=0, game_log_dir=str(tmp_path)), collect=set())
    log = json.loads((tmp_path / "game-23.json").read_text(encoding="utf-8"))
    replay = be.GameState(seed=log["seed"], players=log["players"])
    for pid, action in log["actions"]:
        assert replay.current_player_id == pid
        replay.apply_move(action)
    assert replay.game_over and log["complete"]
    assert replay.player_vps() == log["vps"] == vps
    assert replay.final_ranking() == log["ranking"]


def test_search_reports_tree_reuse_diagnostics():
    mcts = _make()
    game = be.GameState(seed=11, players=4)
    result = mcts.search(game, sims=12)
    # Both counters are surfaced so a training loop can monitor how often the
    # tree reuses a node expanded under a different determinization.
    assert result.failed_applies >= 0
    assert result.rewritten_applies >= 0


def test_search_expands_every_legal_candidate_without_pruning():
    torch.manual_seed(0)
    net = PolicyValueNet()
    mcts = RustISMCTS(net, RustMCTSConfig(device="cpu", max_depth=4, batch_size=8))
    game = be.GameState(seed=11, players=4)
    legal = len(game.legal_candidates()[0])
    result = mcts.search(game, sims=8)
    assert len(result.visits) == legal


def test_prior_top_k_prunes_the_searched_branching_factor():
    torch.manual_seed(0)
    net = PolicyValueNet()
    mcts = RustISMCTS(
        net,
        RustMCTSConfig(device="cpu", max_depth=4, batch_size=8, prior_top_k=6),
    )
    game = be.GameState(seed=11, players=4)
    legal = len(game.legal_candidates()[0])
    assert legal > 6
    result = mcts.search(game, sims=8)
    # Only the highest-prior children survive, so the visit vector is bounded by
    # the shortlist even though every legal move was scored for the prior.
    assert 0 < len(result.visits) <= 6


def _sampled_opponent_hands(state):
    """Opponent hand bags as encoded for the acting player's perspective."""
    seats = state.state_tokens()[3]
    lo, hi = be.SEAT_HAND_SAMPLED, be.SEAT_HAND_SAMPLED + be.CARD_SEMANTIC_COUNT
    return seats[1:, lo:hi]


def test_selfplay_observation_is_determinized_by_default():
    mcts = _make()
    state = be.GameState(seed=5, players=4)
    true_opp = _sampled_opponent_hands(state)
    samples, _ = play_game(
        mcts, SelfPlayConfig(sims=2, seed=5, temperature=0.0, store_snapshots=False)
    )
    # The first recorded decision is the opening position, whose true opponent
    # hands are known; training must not see them.
    assert samples
    recorded = samples[0].seats[1:, be.SEAT_HAND_SAMPLED:be.SEAT_HAND_SAMPLED + be.CARD_SEMANTIC_COUNT]
    assert not np.allclose(recorded, true_opp)


def test_selfplay_observation_can_use_the_true_state():
    mcts = _make()
    state = be.GameState(seed=5, players=4)
    true_opp = _sampled_opponent_hands(state)
    samples, _ = play_game(
        mcts,
        SelfPlayConfig(sims=2, seed=5, temperature=0.0, determinize_observation=False,
                       store_snapshots=False),
    )
    assert samples
    recorded = samples[0].seats[1:, be.SEAT_HAND_SAMPLED:be.SEAT_HAND_SAMPLED + be.CARD_SEMANTIC_COUNT]
    np.testing.assert_allclose(recorded, true_opp)


def test_selfplay_stats_report_reuse_counters():
    mcts = _make()
    stats: dict = {}
    play_game(mcts, SelfPlayConfig(sims=2, seed=9, temperature=0.0), stats=stats)
    assert {"failed_applies", "rewritten_applies", "moves", "final_ranking", "zero_vp_players"} <= set(stats)
    assert stats["moves"] > 0
    assert stats["failed_applies"] >= 0
    assert stats["rewritten_applies"] >= 0


def test_play_batch_derives_a_distinct_seed_per_game(monkeypatch):
    seen = []

    def fake_play_game(_mcts, cfg):
        seen.append(cfg.seed)
        return [], np.zeros(4)

    monkeypatch.setattr(selfplay_module, "play_game", fake_play_game)
    selfplay_module.play_batch(None, 3, SelfPlayConfig(seed=100))
    assert seen == [100, 101, 102]
