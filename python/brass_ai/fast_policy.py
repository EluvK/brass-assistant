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
    anchor_probs: np.ndarray | None = None


@dataclass
class GameTrajectory:
    seed: int
    steps: list[RolloutStep]
    vps: np.ndarray  # float64 array of shape (4,)
    final_ranking: list[int]
    canal_econ: list[tuple[int, int]]
    final_econ: list[tuple[int, int]]
    learner_seats: set[int] | None = None

    def __post_init__(self):
        if self.learner_seats is None:
            self.learner_seats = {0, 1, 2, 3}


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


def trajectory_to_samples(
    traj: GameTrajectory,
    min_vp_filter: float = 0.0,
    use_advantage: bool = True,
    winner_boost: float = 1.5,
) -> list[Sample]:
    """Convert a completed GameTrajectory into training Samples with optional
    quality filtering and advantage-weighted importance.
    """
    if min_vp_filter > 0.0 and float(np.min(traj.vps)) < min_vp_filter:
        return []

    value, winner = _value_targets(traj.vps, traj.final_ranking, 4)
    abs_vp = ((traj.vps - 100.0) / 50.0).astype(np.float32)

    # Advantage per player: Adv_p = (VP_p - mean(VP)) / 50.0
    mean_vp = float(np.mean(traj.vps))
    player_weights = {}
    for pid in range(4):
        if use_advantage:
            adv = float((traj.vps[pid] - mean_vp) / 50.0)
            w = float(np.clip(np.exp(adv), 0.25, 3.0))
            if traj.final_ranking and traj.final_ranking[0] == pid:
                w *= winner_boost
            player_weights[pid] = w
        else:
            player_weights[pid] = 1.0

    samples: list[Sample] = []
    learners = getattr(traj, "learner_seats", None) or {0, 1, 2, 3}
    for step in traj.steps:
        if step.pid not in learners:
            continue
        econ_pair = traj.canal_econ[step.pid] if step.era == 0 else traj.final_econ[step.pid]
        econ = np.asarray(econ_pair, dtype=np.float32)
        w = player_weights.get(step.pid, 1.0)
        anchor_p = getattr(step, "anchor_probs", None)
        if anchor_p is None:
            anchor_p = getattr(step, "action_probs", None)
        samples.append(Sample(
            pid=step.pid,
            era=step.era,
            value=value,
            abs_vp=abs_vp,
            winner=winner,
            econ=econ,
            snapshot=step.snapshot,
            teacher_canonical=step.canonical,
            weight=w,
            anchor_probs=anchor_p,
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
        heuristic_prob: float = 0.0,
        anchor_net: PolicyValueNet | None = None,
    ):
        self.net = net
        self.env_count = max(1, env_count)
        self.device = device
        self.temperature = temperature
        self.heuristic_prob = max(0.0, min(1.0, float(heuristic_prob)))
        self.anchor_net = anchor_net
        self.net.eval()
        if self.anchor_net is not None:
            self.anchor_net.eval()

    def set_anchor_net(self, anchor_net: PolicyValueNet | None) -> None:
        """Dynamically update or clear the reference anchor network."""
        self.anchor_net = anchor_net
        if self.anchor_net is not None:
            self.anchor_net.eval()

    def run_games(
        self,
        start_seed: int,
        n_games: int,
        max_moves_per_game: int = 600,
    ) -> list[GameTrajectory]:
        completed_trajectories: list[GameTrajectory] = []
        next_seed = start_seed

        def create_env(seed: int, env_idx: int):
            st = be.GameState(seed=seed, players=4)
            teachers = {}
            learners = {0, 1, 2, 3}
            if self.heuristic_prob > 0.0:
                learner_seat = env_idx % 4
                rng = np.random.default_rng(seed)
                for s in range(4):
                    if s != learner_seat and rng.random() < self.heuristic_prob:
                        teachers[s] = HeuristicRoundPlayer()
                        learners.discard(s)
            return st, teachers, learners

        # Initialize slots
        active_states: list[be.GameState | None] = []
        active_teachers: list[dict[int, HeuristicRoundPlayer]] = []
        active_learners: list[set[int]] = []
        active_steps: list[list[RolloutStep]] = []
        active_seeds: list[int] = []

        for env_idx in range(min(self.env_count, n_games)):
            st, tch, lrn = create_env(next_seed, env_idx)
            active_states.append(st)
            active_teachers.append(tch)
            active_learners.append(lrn)
            active_steps.append([])
            active_seeds.append(next_seed)
            next_seed += 1

        while any(s is not None for s in active_states) and len(completed_trajectories) < n_games:
            active_indices = [i for i, s in enumerate(active_states) if s is not None and not s.game_over]
            if not active_indices:
                break

            # 1. Advance heuristic teacher turns (fast CPU operations)
            # Drain consecutive teacher moves in all active environments until either it's an NN turn or game is over
            for env_idx in active_indices:
                st = active_states[env_idx]
                while st is not None and not st.game_over:
                    pid = st.current_player_id
                    if pid not in active_teachers[env_idx]:
                        break
                    teacher_move = active_teachers[env_idx][pid].step(st)
                    st.apply_move(teacher_move)

                    if st.game_over or len(active_steps[env_idx]) >= max_moves_per_game:
                        traj = GameTrajectory(
                            seed=active_seeds[env_idx],
                            steps=active_steps[env_idx],
                            vps=np.asarray(st.player_vps(), dtype=np.float64),
                            final_ranking=list(st.final_ranking()),
                            canal_econ=st.canal_econ(),
                            final_econ=st.final_econ(),
                            learner_seats=active_learners[env_idx],
                        )
                        completed_trajectories.append(traj)

                        if next_seed - start_seed < n_games:
                            st_new, tch_new, lrn_new = create_env(next_seed, env_idx)
                            active_states[env_idx] = st_new
                            active_teachers[env_idx] = tch_new
                            active_learners[env_idx] = lrn_new
                            active_steps[env_idx] = []
                            active_seeds[env_idx] = next_seed
                            next_seed += 1
                            st = st_new
                        else:
                            active_states[env_idx] = None
                            st = None

            # 2. Pack state tokens and candidates for all remaining active environments (NN turns)
            cells_list, links_list, merch_list, seats_list, g_list = [], [], [], [], []
            candidate_features_list = []
            canonical_lists = []
            metadata = []

            for idx in active_indices:
                st = active_states[idx]
                if st is None or st.game_over:
                    continue
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

            if not metadata:
                continue

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
                if self.anchor_net is not None:
                    anchor_out = self.anchor_net(batch, padded_actions, mask)
                    anchor_logits = anchor_out["candidate_logits"]
                else:
                    anchor_logits = None

            # Step each active environment
            for b_idx, (env_idx, pid, era, snap) in enumerate(metadata):
                st = active_states[env_idx]
                canons = canonical_lists[b_idx]
                act_idx, log_prob, probs = choose_policy_action(
                    logits[b_idx], masks[b_idx], temperature=self.temperature
                )
                canon_move = canons[act_idx]
                n_canons = len(canons)

                if anchor_logits is not None:
                    anchor_sub = anchor_logits[b_idx][:n_canons]
                    anchor_p = F.softmax(anchor_sub, dim=-1).detach().cpu().numpy()
                else:
                    anchor_p = probs[:n_canons]

                active_steps[env_idx].append(RolloutStep(
                    pid=pid,
                    era=era,
                    action_index=act_idx,
                    canonical=canon_move,
                    log_prob=log_prob,
                    action_probs=probs[:n_canons],
                    snapshot=snap,
                    anchor_probs=anchor_p,
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
                        learner_seats=active_learners[env_idx],
                    )
                    completed_trajectories.append(traj)

                    # Reset environment if more games needed
                    if next_seed - start_seed < n_games:
                        st_new, tch_new, lrn_new = create_env(next_seed, env_idx)
                        active_states[env_idx] = st_new
                        active_teachers[env_idx] = tch_new
                        active_learners[env_idx] = lrn_new
                        active_steps[env_idx] = []
                        active_seeds[env_idx] = next_seed
                        next_seed += 1
                    else:
                        active_states[env_idx] = None

        return completed_trajectories
