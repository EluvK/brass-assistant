"""Self-play: play full games with the network-guided MCTS and collect
training samples (state -> visit-distribution policy target, normalized final
VP value target).

Value target (user-approved, 2026-08 redesign): z = (vp - mean) / max(std, eps)
over the players of that game, as the FULL 4-player vector; each sample carries
the same 4-vector (the value head predicts all players from one perspective).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
import multiprocessing as mp
import os

from .progress import Progress
import numpy as np

import brass_engine as be

from typing import Callable, Protocol

from .hierarchical_policy import encode_legal_candidates, encode_teacher_candidates


class SearchResultLike(Protocol):
    best: str | None
    visits: dict
    canon_by_candidate: dict


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
    candidates: np.ndarray  # (N,208) Engine structured legal-move features
    policy: np.ndarray  # (N,) visit distribution aligned to candidates
    value: np.ndarray  # (4,) normalized final VP z-vector over all players
    era: int = 0  # 0 = canal, 1 = rail (sample's own era at record time)
    econ: np.ndarray = None  # (2,) = (income_level, money) target for this sample


@dataclass
class SelfPlayConfig:
    players: int = 4
    sims: int = 100
    temperature: float = 1.0
    max_moves: int = 600
    seed: int | None = None


def _candidate_policy(canonicals: list[str], result: SearchResultLike) -> np.ndarray:
    """Map search visits to the Engine's concrete candidate ordering."""
    p = np.zeros(len(canonicals), dtype=np.float32)
    by_canonical = {canonical: i for i, canonical in enumerate(canonicals)}
    for candidate_id, visits in result.visits.items():
        canonical = result.canon_by_candidate.get(candidate_id)
        if canonical in by_canonical:
            p[by_canonical[canonical]] += visits
    if p.sum() == 0 and result.best in by_canonical:
        p[by_canonical[result.best]] = 1.0
    if p.sum() == 0:
        raise RuntimeError("search result does not map to Engine candidates")
    return p / p.sum()


def _sample_move(result: SearchResultLike, temperature: float):
    """Return the canonical for a slot sampled from the visit distribution."""
    if not result.visits:
        return result.best
    if temperature <= 0.0:
        slot = max(result.visits, key=result.visits.get)
        return result.canon_by_candidate[slot]
    slots = list(result.visits)
    counts = np.asarray([result.visits[s] for s in slots], dtype=np.float64)
    # Numerically stable softmax-temperature: subtract the max before exp, or
    # a concentrated visit distribution (one child with hundreds of visits)
    # overflows exp(counts/temp) -> inf/inf -> NaN probabilities.
    w = np.exp((counts - counts.max()) / max(temperature, 1e-6))
    probs = w / w.sum()
    slot = np.random.choice(slots, p=probs)
    return result.canon_by_candidate[slot]


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
    if collect is None:
        collect = set(range(cfg.players))
    canal_samples: list[Sample] = []
    moves = 0
    while not state.game_over and moves < cfg.max_moves:
        moves += 1
        pid = state.current_player_id
        canonical_candidates, candidate_tensor = encode_legal_candidates(state)
        result = roles[pid](state, cfg.sims, True)
        if result.best is None:
            break
        if pid in collect:
            board, links, g, oh, op = state.state_to_tensor()
            policy = _candidate_policy(canonical_candidates, result)
            s = Sample(pid=pid, board=board, links=links, global_vec=g,
                       own_hand=oh, opp_hands=op, policy=policy, value=0.0,
                       candidates=candidate_tensor.numpy(), era=state.era)
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


def _generate_imitation_game(args):
    """Generate one heuristic game in a worker process."""
    seed, players, max_moves = args

    state = be.GameState(seed=seed, players=players)
    local = []
    moves = 0
    while not state.game_over and moves < max_moves:
        moves += 1
        pid = state.current_player_id
        candidate_tensor, teacher_scores, canon, chosen_index, _score = encode_teacher_candidates(state)
        score_values = teacher_scores.numpy().astype(np.float64)
        weights = np.exp((score_values - score_values.max()) / 1.0)
        policy = (weights / weights.sum()).astype(np.float32)
        board, links, g, oh, op = state.state_to_tensor()
        local.append(
            Sample(pid=pid, board=board, links=links, global_vec=g,
                   own_hand=oh, opp_hands=op, policy=policy, value=0.0,
                   candidates=candidate_tensor.numpy(), era=state.era)
        )
        try:
            state.apply_move(canon)
        except ValueError:
            break

    if not state.game_over:
        raise RuntimeError(
            f"heuristic game seed={seed} exceeded max_moves={max_moves}; samples discarded"
        )

    vps = np.asarray(state.player_vps(), dtype=np.float64)
    z = _normalize(vps)
    final_econ = {p: e for p, e in enumerate(state.final_econ())}
    for sample in local:
        sample.value = z
        sample.econ = np.asarray(final_econ[sample.pid], dtype=np.float32)
    # Keep unnormalised scores until the parent process has decided whether
    # this game belongs in a quality-filtered imitation batch.
    return local, vps

def generate_imitation_samples(
    n_games: int,
    players: int = 4,
    max_moves: int = 600,
    workers: int | None = None,
    min_avg_vp: float | None = None,
    min_vp: float | None = None,
    max_attempts: int | None = None,
):
    """Heuristic-vs-heuristic games: one-hot imitation samples (cheap, no MCTS).

    Each move records the state + the heuristic's chosen policy slot (one-hot)
    with the game's normalized VP as the value target and the player's FINAL
    (income, money) as the econ target. Games are independent and therefore
    generated in parallel by default; pass ``workers=1`` to force serial
    execution. The automatic worker count is capped at 8 to bound memory use
    while large sample batches are in flight.

    When ``min_avg_vp`` or ``min_vp`` is set, ``n_games`` is the number of
    *accepted* games. A game is accepted only when its mean VP and every
    player's VP are strictly greater than the supplied thresholds. Generation
    stops with an error after ``max_attempts`` candidates (default: 10x the
    requested accepted games), so an overly strict filter cannot run forever.
    """
    samples: list[Sample] = []
    if n_games <= 0:
        return samples
    if workers is not None and workers < 1:
        raise ValueError("workers must be >= 1")
    if min_avg_vp is not None and not np.isfinite(min_avg_vp):
        raise ValueError("min_avg_vp must be finite")
    if min_vp is not None and not np.isfinite(min_vp):
        raise ValueError("min_vp must be finite")

    quality_filter = min_avg_vp is not None or min_vp is not None
    if max_attempts is None:
        max_attempts = n_games * 10 if quality_filter else n_games
    if max_attempts < n_games:
        raise ValueError("max_attempts must be >= n_games")

    def accepted(vps: np.ndarray) -> bool:
        return (
            (min_avg_vp is None or float(vps.mean()) > min_avg_vp)
            and (min_vp is None or float(vps.min()) > min_vp)
        )

    # A single game is faster in-process; for batches, independent games scale
    # well across processes because the Rust engine and tensor encoding are CPU
    # bound. ``spawn`` is required for Windows and avoids inheriting Rust state.
    worker_count = min(max_attempts, workers if workers is not None else min(8, os.cpu_count() or 1))
    if worker_count > 1:
        # Windows spawn workers import NumPy afresh.  One default BLAS thread
        # pool per worker can exhaust memory before a game starts, so configure
        # the environment inherited by those workers before creating the pool.
        for name in (
            "OPENBLAS_NUM_THREADS",
            "OMP_NUM_THREADS",
            "MKL_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
        ):
            os.environ[name] = "1"
    jobs = [(gi, players, max_moves) for gi in range(max_attempts)]
        
    progress = Progress(total=n_games, label="accepted imitation game")
    accepted_games = 0
    attempted_games = 0

    def consume(result) -> bool:
        nonlocal accepted_games, attempted_games
        attempted_games += 1
        local, vps = result
        if accepted(vps):
            samples.extend(local)
            accepted_games += 1
        progress.update(
            accepted_games,
            extra=(f"accepted: {accepted_games}/{n_games}, attempted: {attempted_games}, "
                   f"samples: {len(samples)}"),
        )
        return accepted_games == n_games

    if worker_count == 1:
        results = map(_generate_imitation_game, jobs)
        for result in results:
            if consume(result):
                break
    else:
        with ProcessPoolExecutor(
            max_workers=worker_count,
            mp_context=mp.get_context("spawn"),
        ) as pool:
            # Keep only one task per worker in flight.  ``Executor.map`` can
            # eagerly submit thousands of jobs, which increases the peak
            # memory used by serialized game results on large batches.
            pending = {}
            next_submit = 0
            while next_submit < worker_count:
                pending[pool.submit(_generate_imitation_game, jobs[next_submit])] = next_submit
                next_submit += 1
            completed = {}
            next_emit = 0
            while pending and accepted_games < n_games:
                done, _ = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    index = pending.pop(future)
                    completed[index] = future.result()
                    if next_submit < max_attempts:
                        pending[pool.submit(_generate_imitation_game, jobs[next_submit])] = next_submit
                        next_submit += 1
                while next_emit in completed and accepted_games < n_games:
                    result = completed.pop(next_emit)
                    next_emit += 1
                    consume(result)

    if accepted_games != n_games:
        raise RuntimeError(
            f"only accepted {accepted_games}/{n_games} imitation games after "
            f"{attempted_games} attempts; relax min_avg_vp/min_vp or increase max_attempts"
        )
                    
    progress.done()
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
