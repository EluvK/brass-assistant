"""Fast pure-policy rollout engine for Brass: Birmingham.

Eliminates MCTS tree overhead during self-play data generation by directly
sampling or selecting greedy actions from the neural network's policy logits.
Supports vectorized batched environments to saturate GPU throughput.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
import torch
import torch.nn.functional as F

import brass_ai._engine as be
from .hierarchical_policy import pad_candidate_features
from .net import PolicyValueNet, state_batch
from .selfplay import Sample, _value_targets


@dataclass
class RolloutStep:
    pid: int
    era: int
    action_index: int
    canonical: str
    log_prob: float
    action_probs: np.ndarray  # policy distribution over legal candidates
    snapshot: bytes


@dataclass
class GameTrajectory:
    seed: int
    steps: list[RolloutStep]
    vps: np.ndarray
    final_ranking: list[int]
    canal_econ: list[tuple[int, int]]
    final_econ: list[tuple[int, int]]


def choose_policy_action(
    logits: torch.Tensor,
    mask: torch.Tensor,
    temperature: float = 1.0,
    greedy: bool = False,
) -> tuple[int, float, np.ndarray]:
    """Sample or pick greedy action from masked candidate logits.

    Returns:
        (chosen_index, log_prob_of_chosen, full_prob_distribution)
    """
    masked_logits = logits.masked_fill(~mask, float("-inf"))
    if greedy or temperature <= 1e-4:
        probs = F.softmax(masked_logits, dim=-1)
        idx = int(torch.argmax(probs).item())
        prob_arr = probs.detach().cpu().numpy()
        log_prob = float(torch.log(probs[idx].clamp_min(1e-10)).item())
        return idx, log_prob, prob_arr

    scaled_logits = masked_logits / temperature
    probs = F.softmax(scaled_logits, dim=-1)
    dist = torch.distributions.Categorical(probs=probs)
    action_idx = int(dist.sample().item())
    prob_arr = probs.detach().cpu().numpy()
    log_prob = float(torch.log(probs[action_idx].clamp_min(1e-10)).item())
    return action_idx, log_prob, prob_arr


class FastPolicyPlayer:
    """Zero-search policy player that interacts with a single GameState."""

    def __init__(
        self,
        net: PolicyValueNet,
        device: str = "cpu",
        temperature: float = 0.8,
        greedy: bool = False,
    ):
        self.net = net
        self.device = device
        self.temperature = temperature
        self.greedy = greedy
        self.net.eval()

    def step(self, state: be.GameState) -> tuple[str, int, float, np.ndarray]:
        """Evaluate current state and return (canonical_move, action_idx, log_prob, probs)."""
        cells, links, merchants, seats, g = state.state_tokens()
        batch = state_batch((
            torch.from_numpy(np.asarray(cells, dtype=np.float32)).unsqueeze(0).to(self.device),
            torch.from_numpy(np.asarray(links, dtype=np.float32)).unsqueeze(0).to(self.device),
            torch.from_numpy(np.asarray(merchants, dtype=np.float32)).unsqueeze(0).to(self.device),
            torch.from_numpy(np.asarray(seats, dtype=np.float32)).unsqueeze(0).to(self.device),
            torch.from_numpy(np.asarray(g, dtype=np.float32)).unsqueeze(0).to(self.device),
        ))
        canonical, features = state.legal_candidates()
        actions = torch.from_numpy(np.asarray(features, dtype=np.float32)).unsqueeze(0).to(self.device)
        mask = torch.ones(1, actions.shape[1], dtype=torch.bool, device=self.device)

        with torch.no_grad():
            out = self.net(batch, actions, mask)
            logits = out["candidate_logits"][0]
            mask_1d = out["candidate_mask"][0]

        idx, log_prob, probs = choose_policy_action(
            logits, mask_1d, temperature=self.temperature, greedy=self.greedy
        )
        return canonical[idx], idx, log_prob, probs


def play_game_fast(
    net: PolicyValueNet,
    seed: int,
    device: str = "cpu",
    temperature: float = 0.8,
    greedy: bool = False,
    max_moves: int = 600,
) -> GameTrajectory:
    """Play a complete self-play game using pure policy sampling."""
    state = be.GameState(seed=seed, players=4)
    player = FastPolicyPlayer(net, device=device, temperature=temperature, greedy=greedy)

    steps: list[RolloutStep] = []
    canal_econ: list[tuple[int, int]] = []
    seen_canal_end = False

    for _ in range(max_moves):
        if state.game_over:
            break

        pid = state.current_player_id
        era = state.era
        snapshot_bytes = bytes(state.snapshot())

        canon, idx, log_prob, probs = player.step(state)
        steps.append(RolloutStep(
            pid=pid,
            era=era,
            action_index=idx,
            canonical=canon,
            log_prob=log_prob,
            action_probs=probs,
            snapshot=snapshot_bytes,
        ))

        state.apply_move(canon)

    canal_econ = state.canal_econ()
    final_econ = state.final_econ()
    vps = np.asarray(state.player_vps(), dtype=np.float64)
    final_ranking = list(state.final_ranking())

    return GameTrajectory(
        seed=seed,
        steps=steps,
        vps=vps,
        final_ranking=final_ranking,
        canal_econ=canal_econ,
        final_econ=final_econ,
    )


def trajectory_to_samples(traj: GameTrajectory) -> list[Sample]:
    """Convert a completed GameTrajectory into training Samples."""
    value, winner = _value_targets(traj.vps, traj.final_ranking, 4)
    abs_vp = ((traj.vps - 100.0) / 50.0).astype(np.float32)

    samples: list[Sample] = []
    for step in traj.steps:
        econ_pair = traj.canal_econ[step.pid] if step.era == 0 else traj.final_econ[step.pid]
        econ = np.asarray(econ_pair, dtype=np.float32)
        samples.append(Sample(
            pid=step.pid,
            era=step.era,
            value=value,
            abs_vp=abs_vp,
            winner=winner,
            econ=econ,
            snapshot=step.snapshot,
            teacher_canonical=step.canonical,
        ))
    return samples


class VectorizedSelfPlay:
    """Vectorized batched environment runner for fast neural network self-play.

    Maintains `env_count` simultaneous GameStates, packs their tokens and legal
    candidate features into a single GPU/CPU batch per step, and executes moves
    in lockstep. When an environment terminates, it collects the trajectory and
    resets with a new seed until `n_games` are completed.
    """

    def __init__(
        self,
        net: PolicyValueNet,
        env_count: int = 16,
        device: str = "cpu",
        temperature: float = 0.8,
    ):
        self.net = net
        self.env_count = max(1, env_count)
        self.device = device
        self.temperature = temperature
        self.net.eval()

    def run_games(
        self,
        start_seed: int,
        n_games: int,
        max_moves_per_game: int = 600,
    ) -> list[GameTrajectory]:
        completed_trajectories: list[GameTrajectory] = []
        next_seed = start_seed

        # Initialize slots
        active_states: list[be.GameState | None] = []
        active_steps: list[list[RolloutStep]] = []
        active_seeds: list[int] = []

        for _ in range(min(self.env_count, n_games)):
            active_states.append(be.GameState(seed=next_seed, players=4))
            active_steps.append([])
            active_seeds.append(next_seed)
            next_seed += 1

        while any(s is not None for s in active_states) and len(completed_trajectories) < n_games:
            active_indices = [i for i, s in enumerate(active_states) if s is not None and not s.game_over]
            if not active_indices:
                break

            # Pack state tokens and candidates for all active environments
            cells_list, links_list, merch_list, seats_list, g_list = [], [], [], [], []
            candidate_features_list = []
            canonical_lists = []
            metadata = []

            for idx in active_indices:
                st = active_states[idx]
                pid = st.current_player_id
                era = st.era
                snap = bytes(st.snapshot())

                c, l, m, s, g = st.state_tokens()
                cells_list.append(torch.from_numpy(np.asarray(c, dtype=np.float32)))
                links_list.append(torch.from_numpy(np.asarray(l, dtype=np.float32)))
                merch_list.append(torch.from_numpy(np.asarray(m, dtype=np.float32)))
                seats_list.append(torch.from_numpy(np.asarray(s, dtype=np.float32)))
                g_list.append(torch.from_numpy(np.asarray(g, dtype=np.float32)))

                canons, feats = st.legal_candidates()
                canonical_lists.append(canons)
                candidate_features_list.append(
                    torch.from_numpy(np.ascontiguousarray(feats, dtype=np.float32))
                )
                metadata.append((idx, pid, era, snap))

            # Batch forward pass on device
            batch = state_batch((
                torch.stack(cells_list).to(self.device),
                torch.stack(links_list).to(self.device),
                torch.stack(merch_list).to(self.device),
                torch.stack(seats_list).to(self.device),
                torch.stack(g_list).to(self.device),
            ))
            padded_actions, mask = pad_candidate_features(candidate_features_list, device=self.device)

            with torch.no_grad():
                out = self.net(batch, padded_actions, mask)
                logits = out["candidate_logits"]
                masks = out["candidate_mask"]

            # Step each active environment
            for b_idx, (env_idx, pid, era, snap) in enumerate(metadata):
                st = active_states[env_idx]
                canons = canonical_lists[b_idx]
                act_idx, log_prob, probs = choose_policy_action(
                    logits[b_idx], masks[b_idx], temperature=self.temperature
                )
                canon_move = canons[act_idx]
                active_steps[env_idx].append(RolloutStep(
                    pid=pid,
                    era=era,
                    action_index=act_idx,
                    canonical=canon_move,
                    log_prob=log_prob,
                    action_probs=probs[:len(canons)],
                    snapshot=snap,
                ))

                st.apply_move(canon_move)

                # Check game over or move limit
                if st.game_over or len(active_steps[env_idx]) >= max_moves_per_game:
                    traj = GameTrajectory(
                        seed=active_seeds[env_idx],
                        steps=active_steps[env_idx],
                        vps=np.asarray(st.player_vps(), dtype=np.float64),
                        final_ranking=list(st.final_ranking()),
                        canal_econ=st.canal_econ(),
                        final_econ=st.final_econ(),
                    )
                    completed_trajectories.append(traj)

                    # Reset environment if more games needed
                    if next_seed - start_seed < n_games:
                        active_states[env_idx] = be.GameState(seed=next_seed, players=4)
                        active_steps[env_idx] = []
                        active_seeds[env_idx] = next_seed
                        next_seed += 1
                    else:
                        active_states[env_idx] = None

        return completed_trajectories
