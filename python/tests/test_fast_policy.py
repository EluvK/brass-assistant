"""Tests for fast pure-policy rollout engine."""

import pytest
import torch
import numpy as np

import brass_ai._engine as be
from brass_ai.net import PolicyValueNet
from brass_ai.fast_policy import (
    FastPolicyPlayer,
    play_game_fast,
    trajectory_to_samples,
    choose_policy_action,
    VectorizedSelfPlay,
)


def test_choose_policy_action_greedy_and_sample():
    logits = torch.tensor([1.0, 5.0, 2.0, -1.0])
    mask = torch.tensor([True, True, True, False])

    # Greedy should always pick index 1
    idx, log_prob, probs = choose_policy_action(logits, mask, greedy=True)
    assert idx == 1
    assert probs[1] > probs[0]
    assert probs[3] == 0.0

    # Temperature sampling should strictly observe mask
    idx, log_prob, probs = choose_policy_action(logits, mask, temperature=1.0, greedy=False)
    assert idx in (0, 1, 2)
    assert probs[3] == 0.0


def test_fast_policy_player_single_step():
    net = PolicyValueNet()
    state = be.GameState(seed=42, players=4)
    player = FastPolicyPlayer(net, device="cpu", temperature=1.0)

    canon, idx, log_prob, probs = player.step(state)
    assert isinstance(canon, str)
    assert idx >= 0
    assert len(probs) > 0
    assert np.isclose(probs.sum(), 1.0, atol=1e-4)


def test_play_game_fast_produces_valid_trajectory():
    net = PolicyValueNet()
    # Play a short game
    traj = play_game_fast(net, seed=100, device="cpu", temperature=0.8, max_moves=20)
    assert traj.seed == 100
    assert len(traj.steps) == 20
    assert len(traj.vps) == 4

    samples = trajectory_to_samples(traj)
    assert len(samples) == 20
    for s in samples:
        assert s.snapshot is not None
        assert s.abs_vp is not None
        assert len(s.abs_vp) == 4
        assert len(s.value) == 4


def test_vectorized_selfplay_multiple_games():
    net = PolicyValueNet()
    runner = VectorizedSelfPlay(net, env_count=2, device="cpu", temperature=0.8)
    trajectories = runner.run_games(start_seed=500, n_games=2, max_moves_per_game=15)
    assert len(trajectories) == 2
    for t in trajectories:
        assert len(t.steps) == 15
        assert len(t.vps) == 4
        samples = trajectory_to_samples(t)
        assert len(samples) == 15


def test_vectorized_selfplay_with_heuristic_teachers():
    net = PolicyValueNet()
    anchor_net = PolicyValueNet()
    runner = VectorizedSelfPlay(
        net, env_count=2, device="cpu", temperature=0.8,
        heuristic_prob=1.0, anchor_net=anchor_net
    )
    trajectories = runner.run_games(start_seed=600, n_games=2, max_moves_per_game=20)
    assert len(trajectories) == 2
    for t in trajectories:
        assert len(t.vps) == 4
        assert hasattr(t, "learner_seats")
        # With heuristic_prob=1.0, exactly 1 seat is learner and 3 seats are heuristic teachers
        assert len(t.learner_seats) == 1
        samples = trajectory_to_samples(t, use_advantage=True)
        assert len(samples) > 0
        for s in samples:
            assert s.pid in t.learner_seats
            assert hasattr(s, "weight")
            assert s.weight > 0.0
            assert s.anchor_probs is not None
            assert len(s.anchor_probs) > 0


def test_trajectory_to_samples_advantage_weighting():
    net = PolicyValueNet()
    traj = play_game_fast(net, seed=700, device="cpu", temperature=0.8, max_moves=20)
    # Assign distinct artificial VPs
    traj.vps = np.array([140.0, 110.0, 90.0, 60.0])
    traj.final_ranking = [0, 1, 2, 3]

    samples = trajectory_to_samples(traj, min_vp_filter=50.0, use_advantage=True)
    assert len(samples) == 20
    # Winner (seat 0, 140 VP) should have significantly higher weight than lowest (seat 3, 60 VP)
    w_winner = [s.weight for s in samples if s.pid == 0][0]
    w_loser = [s.weight for s in samples if s.pid == 3][0]
    assert w_winner > w_loser
    assert w_winner > 1.0

