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
    legal: np.ndarray  # (1316,) bool mask of legal policy slots


def _legal_mask_bool(state, table_size: int) -> np.ndarray:
    mask = np.zeros(table_size, dtype=bool)
    for s in state.legal_mask():
        mask[s] = True
    return mask


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
                   own_hand=oh, opp_hands=op, policy=policy, value=0.0,
                   legal=_legal_mask_bool(state, table_size))
        )
        chosen = _sample_move(result, cfg.temperature)
        try:
            state.apply_move(chosen)
        except ValueError:
            # Defensive: fall back to the search's best (executable) move, then
            # to the first legal move if needed.
            try:
                state.apply_move(result.best)
            except ValueError:
                legal = state.legal_moves()
                if not legal:
                    break
                state.apply_move(legal[0][1])

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


def generate_imitation_samples(n_games: int, players: int = 4, max_moves: int = 600):
    """Heuristic-vs-heuristic games: one-hot imitation samples (cheap, no MCTS).

    Each move records the state + the heuristic's chosen policy slot (one-hot)
    with the game's normalized VP as the value target. ~0.5s/game."""
    samples: list[Sample] = []
    table = be.policy_table_size
    for gi in range(n_games):
        state = be.GameState(seed=gi, players=players)
        local = []
        moves = 0
        while not state.game_over and moves < max_moves:
            moves += 1
            canon, _, _ = state.choose_heuristic()
            if canon is None:
                break
            pid = state.current_player_id
            slot = be.moves_to_slots(canon)[0]
            policy = np.zeros(table, dtype=np.float32)
            policy[slot] = 1.0
            board, links, g, oh, op = state.state_to_tensor()
            local.append(
                Sample(pid=pid, board=board, links=links, global_vec=g,
                       own_hand=oh, opp_hands=op, policy=policy, value=0.0,
                       legal=_legal_mask_bool(state, table))
            )
            try:
                state.apply_move(canon)
            except ValueError:
                break
        z = _normalize(np.asarray(state.player_vps(), dtype=np.float64))
        for s in local:
            s.value = z[s.pid]
        samples.extend(local)
    return samples


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
