"""Self-play: play full games with the network-guided MCTS and collect
training samples (state -> visit-distribution policy target, normalized final
VP value target).

Value target (user-approved): z_p = (vp_p - mean) / max(std, eps) over the
players of that game; each sample carries the target of its perspective player.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

import brass_engine as be

from .mcts import ISMCTS, SearchResult


@dataclass
class Sample:
    pid: int
    board: np.ndarray  # (17,49)
    links: np.ndarray  # (6,39)
    global_vec: np.ndarray  # (50,)
    own_hand: np.ndarray  # (35,)
    opp_hands: np.ndarray  # (105,)
    policy: np.ndarray  # (1316,) dense over policy slots
    value: float


@dataclass
class SelfPlayConfig:
    players: int = 4
    sims: int = 100
    temperature: float = 1.0
    max_moves: int = 600
    seed: int | None = None


def _dense_policy(visits: dict, table_size: int) -> np.ndarray:
    p = np.zeros(table_size, dtype=np.float32)
    total = sum(visits.values())
    if total <= 0:
        return p
    for s, v in visits.items():
        p[s] = v / total
    return p


def _sample_move(result: SearchResult, temperature: float):
    """Return the canonical for a slot sampled from the visit distribution."""
    if not result.visits:
        return result.best
    if temperature <= 0.0:
        slot = max(result.visits, key=result.visits.get)
        return result.canon_by_slot[slot]
    slots = list(result.visits)
    counts = np.asarray([result.visits[s] for s in slots], dtype=np.float64)
    w = np.exp(counts / max(temperature, 1e-6))
    probs = w / w.sum()
    slot = np.random.choice(slots, p=probs)
    return result.canon_by_slot[slot]


def play_game(
    mcts: ISMCTS,
    cfg: SelfPlayConfig | None = None,
) -> tuple[list, list]:
    """Play one self-play game; returns (samples, final_vps)."""
    cfg = cfg or SelfPlayConfig()
    seed = cfg.seed if cfg.seed is not None else np.random.randint(0, 2**31)
    state = be.GameState(seed=seed, players=cfg.players)

    samples: list[Sample] = []
    table_size = be.policy_table_size
    moves = 0
    while not state.game_over and moves < cfg.max_moves:
        moves += 1
        result = mcts.search(state, cfg.sims, add_root_noise=True)
        if result.best is None:
            break
        pid = state.current_player_id
        board, links, g, oh, op = state.state_to_tensor()
        policy = _dense_policy(result.visits, table_size)
        samples.append(
            Sample(pid=pid, board=board, links=links, global_vec=g,
                   own_hand=oh, opp_hands=op, policy=policy, value=0.0)
        )
        chosen = _sample_move(result, cfg.temperature)
        state.apply_move(chosen)

    vps = state.player_vps()
    z = _normalize(np.asarray(vps, dtype=np.float64))
    for s in samples:
        s.value = z[s.pid]
    return samples, vps


def _normalize(vps: np.ndarray) -> np.ndarray:
    std = vps.std()
    if std < 1e-6:
        return np.zeros_like(vps)
    return (vps - vps.mean()) / std


def play_batch(
    mcts: ISMCTS,
    n_games: int,
    cfg: SelfPlayConfig | None = None,
) -> tuple[list, np.ndarray, list]:
    """Play `n_games` self-play games; returns (all_samples, avg_vps, per_game)."""
    cfg = cfg or SelfPlayConfig()
    all_samples = []
    vps_sum = np.zeros(cfg.players, dtype=np.float64)
    per_game = []
    for _ in range(n_games):
        samples, vps = play_game(mcts, cfg)
        all_samples.extend(samples)
        vps_sum += vps
        per_game.append(vps)
    return all_samples, vps_sum / n_games, per_game
