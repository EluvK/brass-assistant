"""Reinforcement learning league and gatekeeper arena for Brass: Birmingham.

Includes:
1. Gatekeeper Arena: Evaluates candidate networks directly against 3 Rust heuristic
   teachers. Only models achieving win_rate >= 35% and mean_vp >= 120 are promoted.
2. KL-regularized Policy Loss: Prevents policy collapse during high-throughput self-play.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F

import brass_ai._engine as be
from .fast_policy import FastPolicyPlayer
from .net import PolicyValueNet


@dataclass
class GatekeeperResult:
    games: int
    wins: int
    win_rate: float
    candidate_avg_vp: float
    teacher_avg_vp: float
    passed: bool
    details: list[dict]


class HeuristicRoundPlayer:
    """Heuristic player that plans the whole round with 2-ply lookahead
    via `choose_heuristic_round` and caches the second action.
    """

    def __init__(self):
        self.pending: list[str] = []

    def step(self, state: be.GameState) -> str:
        if self.pending:
            return self.pending.pop(0)
        first, second, _ = state.choose_heuristic_round()
        if second is not None:
            self.pending.append(second)
        return first

    def reset(self):
        self.pending.clear()


def evaluate_vs_heuristic_teachers(
    net: PolicyValueNet,
    n_games: int = 40,
    candidate_seat: int = 0,
    device: str = "cpu",
    temperature: float = 0.2,
    greedy: bool = False,
    min_win_rate: float = 0.35,
    min_avg_vp: float = 120.0,
) -> GatekeeperResult:
    """Pit the candidate neural network against 3 Rust heuristic teachers.

    The candidate plays `candidate_seat` (default 0), and all other seats are
    driven by full 2-action round planning via `choose_heuristic_round()`.
    """
    player = FastPolicyPlayer(net, device=device, temperature=temperature, greedy=greedy)
    wins = 0
    candidate_vps = []
    teacher_vps = []
    details = []

    for seed in range(1, n_games + 1):
        state = be.GameState(seed=seed, players=4)
        teachers = {p: HeuristicRoundPlayer() for p in range(4)}
        moves = 0
        while not state.game_over and moves < 600:
            moves += 1
            actor = state.current_player_id
            if actor == candidate_seat:
                canon, _, _, _ = player.step(state)
            else:
                canon = teachers[actor].step(state)
            state.apply_move(canon)

        vps = state.player_vps()
        ranking = list(state.final_ranking())
        is_winner = ranking[0] == candidate_seat
        if is_winner:
            wins += 1

        cand_vp = vps[candidate_seat]
        t_vps = [vps[i] for i in range(4) if i != candidate_seat]

        candidate_vps.append(cand_vp)
        teacher_vps.extend(t_vps)

        details.append({
            "seed": seed,
            "candidate_vp": cand_vp,
            "teacher_vps": t_vps,
            "ranking": ranking,
            "winner": ranking[0],
            "is_win": is_winner,
        })

    win_rate = wins / n_games if n_games > 0 else 0.0
    cand_mean = float(np.mean(candidate_vps)) if candidate_vps else 0.0
    teacher_mean = float(np.mean(teacher_vps)) if teacher_vps else 0.0
    passed = (win_rate >= min_win_rate) and (cand_mean >= min_avg_vp)

    return GatekeeperResult(
        games=n_games,
        wins=wins,
        win_rate=win_rate,
        candidate_avg_vp=cand_mean,
        teacher_avg_vp=teacher_mean,
        passed=passed,
        details=details,
    )


def compute_kl_loss(
    log_probs_pred: torch.Tensor,
    anchor_probs: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Compute masked KL divergence: D_KL(anchor || pred).

    Anchor probabilities act as target distribution to bound policy drift.
    """
    valid = mask & (anchor_probs > 1e-10)
    safe_pred = log_probs_pred.clamp_min(-50.0)
    safe_anchor = anchor_probs.clamp_min(1e-10)
    kl_per_candidate = torch.where(
        valid,
        anchor_probs * (torch.log(safe_anchor) - safe_pred),
        torch.zeros_like(anchor_probs),
    )
    return kl_per_candidate.sum(dim=-1).mean()
