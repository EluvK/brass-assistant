"""Tests for RL league and gatekeeper arena."""

import pytest
import torch
import numpy as np

from brass_ai.net import PolicyValueNet
from brass_ai.rl_league import (
    evaluate_vs_heuristic_teachers,
    compute_kl_loss,
    GatekeeperResult,
)


def test_compute_kl_loss():
    # Identical distributions should have KL = 0
    probs = torch.tensor([[0.2, 0.5, 0.3, 0.0]])
    log_probs = torch.log(probs.clamp_min(1e-10))
    mask = torch.tensor([[True, True, True, False]])

    loss = compute_kl_loss(log_probs, probs, mask)
    assert torch.isclose(loss, torch.tensor(0.0), atol=1e-5)

    # Different distribution should have positive KL
    divergent_log_probs = torch.log(torch.tensor([[0.5, 0.2, 0.3, 0.0]]).clamp_min(1e-10))
    loss_divergent = compute_kl_loss(divergent_log_probs, probs, mask)
    assert loss_divergent.item() > 0.0


def test_gatekeeper_smoke_evaluates_against_teachers():
    net = PolicyValueNet()
    # Smoke test: 1 short game
    res = evaluate_vs_heuristic_teachers(
        net, n_games=1, candidate_seat=0, device="cpu", min_win_rate=0.0, min_avg_vp=0.0
    )
    assert isinstance(res, GatekeeperResult)
    assert res.games == 1
    assert len(res.details) == 1
    assert 0 <= res.win_rate <= 1.0
