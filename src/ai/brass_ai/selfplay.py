"""Self-play: play full games with the network-guided MCTS and collect
training samples (state -> visit-distribution policy target, normalized final
VP value target).

Value target (user-approved, 2026-08 redesign): z = (vp - mean) / max(std, eps)
over the players of that game, as the FULL 4-player vector; each sample carries
the same 4-vector (the value head predicts all players from one perspective).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

import brass_engine as be

from typing import Callable, Protocol


class SearchResultLike(Protocol):
    best: str | None
    visits: dict
    canon_by_slot: dict


class SearchLike(Protocol):
    def search(self, state, sims: int, add_root_noise: bool = False) -> SearchResultLike: ...


@dataclass
class Sample:
    pid: int
    board: np.ndarray  # (17,49)
    links: np.ndarray  # (6,39)
    global_vec: np.ndarray  # (50,)
    own_hand: np.ndarray  # (35,)
    opp_hands: np.ndarray  # (105,)
    policy: np.ndarray  # (1316,) dense over policy slots
    value: np.ndarray  # (4,) normalized final VP z-vector over all players
    legal: np.ndarray  # (1316,) bool mask of legal policy slots
    era: int = 0  # 0 = canal, 1 = rail (sample's own era at record time)
    econ: np.ndarray = None  # (2,) = (income_level, money) target for this sample


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


def _sample_move(result: SearchResultLike, temperature: float):
    """Return the canonical for a slot sampled from the visit distribution."""
    if not result.visits:
        return result.best
    if temperature <= 0.0:
        slot = max(result.visits, key=result.visits.get)
        return result.canon_by_slot[slot]
    slots = list(result.visits)
    counts = np.asarray([result.visits[s] for s in slots], dtype=np.float64)
    # Numerically stable softmax-temperature: subtract the max before exp, or
    # a concentrated visit distribution (one child with hundreds of visits)
    # overflows exp(counts/temp) -> inf/inf -> NaN probabilities.
    w = np.exp((counts - counts.max()) / max(temperature, 1e-6))
    probs = w / w.sum()
    slot = np.random.choice(slots, p=probs)
    return result.canon_by_slot[slot]


def play_game(
    mcts: SearchLike,
    cfg: SelfPlayConfig | None = None,
) -> tuple[list, list]:
    """Play one self-play game; returns (samples, final_vps)."""
    return play_game_with_roles([mcts.search] * 4, cfg)


def play_game_with_roles(
    roles,
    cfg: SelfPlayConfig | None = None,
    collect: set | None = None,
) -> tuple[list, list]:
    """Play one game where each seat is driven by its own search role.

    `roles[pid]` is a callable(state, sims, add_root_noise) -> SearchResult
    (used for matchmaking: opponent seats may run a different network).
    Samples are recorded for every move whose pid is in `collect` (default:
    all seats, matching the pure self-play path). Returns (samples, final_vps).

    Economic-supervision targets (segmented by era, per the 2026-08 design):
      * canal-era samples  -> that player's income/money at the CANAL-ERA END
        (the crucial milestone: it banks the rail-era economy)
      * rail-era samples   -> that player's FINAL income/money
    """
    cfg = cfg or SelfPlayConfig()
    seed = cfg.seed if cfg.seed is not None else np.random.randint(0, 2**31)
    state = be.GameState(seed=seed, players=cfg.players)

    samples: list[Sample] = []
    table_size = be.policy_table_size
    if collect is None:
        collect = set(range(cfg.players))
    canal_samples: list[Sample] = []
    moves = 0
    while not state.game_over and moves < cfg.max_moves:
        moves += 1
        pid = state.current_player_id
        result = roles[pid](state, cfg.sims, True)
        if result.best is None:
            break
        if pid in collect:
            board, links, g, oh, op = state.state_to_tensor()
            policy = _dense_policy(result.visits, table_size)
            s = Sample(pid=pid, board=board, links=links, global_vec=g,
                       own_hand=oh, opp_hands=op, policy=policy, value=0.0,
                       legal=_legal_mask_bool(state, table_size), era=state.era)
            samples.append(s)
            if state.era == 0:
                canal_samples.append(s)
        chosen = _sample_move(result, cfg.temperature)
        try:
            summary, ok = state.apply_move_raw(chosen)
        except ValueError:
            summary, ok = ("", False)
        if not ok:
            # Defensive: fall back to the search's best (executable) move, then
            # to the first legal move if needed.
            try:
                summary, ok = state.apply_move_raw(result.best)
            except ValueError:
                ok = False
            if not ok:
                legal = state.legal_moves()
                if not legal:
                    break
                try:
                    summary, ok = state.apply_move_raw(legal[0][1])
                except ValueError:
                    break
        tr = state.advance_turn_raw()
        if tr == "end_canal_era":
            state.finish_canal_era()
            # Stamp canal-era samples with the canal-end economy (income is
            # unchanged by era-end, so this is the canal-era-final economy).
            econ = {p: e for p, e in enumerate(state.canal_econ())}
            for s in canal_samples:
                s.econ = np.asarray(econ[s.pid], dtype=np.float32)
        elif tr == "end_game":
            state.finish_game()

    if not state.game_over:
        # A partial game has no valid final-VP target.  Treating the current
        # board as terminal previously emitted all-zero or otherwise corrupt
        # value/economy labels into the replay buffer.
        raise RuntimeError(
            f"self-play game exceeded max_moves={cfg.max_moves}; samples discarded"
        )

    vps = state.player_vps()
    z = _normalize(np.asarray(vps, dtype=np.float64))
    # Rail-era samples (and any canal samples that never got a canal-econ stamp,
    # e.g. a game that ended in the canal era) take the FINAL economy.
    final_econ = {p: e for p, e in enumerate(state.final_econ())}
    for s in samples:
        s.value = z
        if s.econ is None:
            s.econ = np.asarray(final_econ[s.pid], dtype=np.float32)
    return samples, vps


def _normalize(vps: np.ndarray) -> np.ndarray:
    std = vps.std()
    if std < 1e-6:
        return np.zeros_like(vps)
    return (vps - vps.mean()) / std


def generate_imitation_samples(n_games: int, players: int = 4, max_moves: int = 600):
    """Heuristic-vs-heuristic games: one-hot imitation samples (cheap, no MCTS).

    Each move records the state + the heuristic's chosen policy slot (one-hot)
    with the game's normalized VP as the value target and the player's FINAL
    (income, money) as the econ target. ~0.5s/game."""
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
                       legal=_legal_mask_bool(state, table), era=state.era)
            )
            try:
                state.apply_move(canon)
            except ValueError:
                break
        z = _normalize(np.asarray(state.player_vps(), dtype=np.float64))
        final_econ = {p: e for p, e in enumerate(state.final_econ())}
        for s in local:
            s.value = z
            s.econ = np.asarray(final_econ[s.pid], dtype=np.float32)
        samples.extend(local)
    return samples


def play_batch(
    mcts: SearchLike,
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
