"""Self-play: play full games with the network-guided MCTS and collect
training samples (state -> visit-distribution policy target, value target,
winner target).

Value target: `(vp_p - table_mean_vp) / VP_SCALE`, the same scale the search
backs up and the network predicts (docs/ai-action-encoding.md §4.1). Each
sample carries the 4-vector plus a one-hot winner target from the engine's
official VP -> income -> cash ranking.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
import multiprocessing as mp
import os
import pickle
from pathlib import Path

from .progress import Progress
import numpy as np

from . import _engine as be

from typing import Callable, Protocol

from .hierarchical_policy import (
    encode_legal_candidates,
    encode_teacher_candidates,
    coalesce_equivalent_policy,
)


class SearchResultLike(Protocol):
    best: str | None
    visits: dict
    canon_by_candidate: dict


class SearchLike(Protocol):
    def search(self, state, sims: int, add_root_noise: bool = False) -> SearchResultLike: ...


@dataclass
class Sample:
    pid: int
    cells: np.ndarray | None = None  # (49, F_CELL)
    links: np.ndarray | None = None  # (39, F_LINK)
    merchants: np.ndarray | None = None  # (9, F_MERCHANT)
    seats: np.ndarray | None = None  # (4, F_SEAT)
    global_vec: np.ndarray | None = None  # (F_GLOBAL,)
    candidates: np.ndarray | None = None  # (N, ACTION_FEATURE_DIM)
    policy: np.ndarray | None = None  # (N,) aligned to candidates
    # (4,) terminal utility `(vp - table_mean) / VP_SCALE`, absolute seat order.
    value: np.ndarray | float = 0.0
    winner: np.ndarray | float = 0.0  # (4,) one-hot official winner
    era: int = 0  # 0 = canal, 1 = rail (sample's own era at record time)
    econ: np.ndarray = None  # (2,) = (income_level, money) target for this sample
    snapshot: bytes | None = None  # independent full GameState snapshot
    teacher_canonical: str | None = None
    # Self-play snapshot form: the visit distribution as a sparse
    # canonical -> probability map, materialized against the live candidate
    # matrix at training time. `None` for the dense and teacher forms.
    policy_by_canonical: dict | None = None
    # The move actually played at this decision point, as a canonical string.
    # The action-conditioned value head is trained on (state, played move) -> the
    # mover's final `1 - rank/n`, so the sample has to remember which sibling it
    # took. Materialization turns this into `action_index`.
    played_canonical: str | None = None
    action_index: int = -1


def _value_targets(
    vps: list[int], ranking: list[int], n_players: int
) -> tuple[np.ndarray, np.ndarray]:
    """Terminal utility and winner targets (see ai-action-encoding.md §4.1).

    Utility is the VP margin over the table mean; the winner comes from the
    engine's official VP -> income -> cash ranking, which breaks ties.
    """
    if len(ranking) != n_players or set(ranking) != set(range(n_players)):
        raise ValueError("engine returned an invalid final ranking")
    if len(vps) != n_players:
        raise ValueError("engine returned an invalid final score line")
    scores = np.asarray(vps, dtype=np.float64)
    utility = ((scores - scores.mean()) / float(be.VP_SCALE)).astype(np.float32)
    winner = np.zeros(n_players, dtype=np.float32)
    winner[ranking[0]] = 1.0
    return utility, winner


def materialize_sample(sample: Sample) -> Sample:
    """Recover dynamic full-legal inputs for a snapshot-backed replay sample.

    The whole reconstruction (snapshot restore, full-legal candidate features,
    teacher equivalence policy) runs in ONE Rust call
    (`_engine.GameState.materialize_snapshot`) so no per-float Python objects
    are ever created for the candidate matrix.

    Self-play samples (`policy_by_canonical`) take the generic path instead:
    their target is a visit distribution over many canonical moves rather than
    a single teacher action, so the policy is aligned in Python against the
    restored candidate list.
    """
    if sample.snapshot is None:
        return sample
    if sample.policy_by_canonical is not None:
        return _materialize_selfplay_sample(sample)
    (pid, era, cells, links, merchants, seats, global_vec,
     candidates, teacher_index, policy) = be.GameState.materialize_snapshot(
        sample.snapshot, sample.teacher_canonical or "")
    if pid != sample.pid or era != sample.era:
        raise ValueError("replay snapshot does not match its player/era metadata")
    return Sample(
        pid=pid, era=era, cells=cells, links=links, merchants=merchants,
        seats=seats, global_vec=global_vec,
        candidates=candidates, policy=policy, value=sample.value,
        winner=sample.winner, econ=sample.econ, snapshot=sample.snapshot,
        teacher_canonical=sample.teacher_canonical, action_index=teacher_index,
        played_canonical=sample.teacher_canonical,
    )


def _materialize_selfplay_sample(sample: Sample) -> Sample:
    """Rebuild dense tensors for a self-play (snapshot + sparse visit) sample."""
    state = be.GameState.from_snapshot(sample.snapshot)
    canonical, features = state.legal_candidates()
    features = np.asarray(features, dtype=np.float32)
    if features.ndim != 2 or len(canonical) != features.shape[0] or not canonical:
        raise ValueError("engine returned an invalid candidate matrix")
    position: dict[str, int] = {}
    for index, move in enumerate(canonical):
        position.setdefault(move, index)
    policy = np.zeros(len(canonical), dtype=np.float32)
    for move, probability in sample.policy_by_canonical.items():
        index = position.get(move)
        if index is None:
            raise ValueError(
                f"self-play policy references a move that is not legal in its "
                f"own snapshot: {move!r}"
            )
        policy[index] += float(probability)
    if policy.sum() <= 0.0:
        raise ValueError("self-play policy has no mass after candidate alignment")
    policy = coalesce_equivalent_policy(features, policy)

    action_index = -1
    if sample.played_canonical is not None:
        action_index = position.get(sample.played_canonical, -1)

    pid = state.current_player_id
    if pid != sample.pid or state.era != sample.era:
        raise ValueError("self-play snapshot does not match its player/era metadata")
    cells, links, merchants, seats, global_vec = state.state_tokens()
    return Sample(
        pid=pid, era=state.era, cells=cells, links=links, merchants=merchants,
        seats=seats, global_vec=global_vec, candidates=features,
        policy=policy, value=sample.value, winner=sample.winner, econ=sample.econ,
        snapshot=sample.snapshot, policy_by_canonical=sample.policy_by_canonical,
        played_canonical=sample.played_canonical, action_index=action_index,
    )


def materialize_samples(samples: list[Sample]) -> list[Sample]:
    return [materialize_sample(sample) for sample in samples]


def materialize_chunk(samples: list[Sample]) -> list[Sample]:
    """Worker-side bulk materialization for the replay pool."""
    return [materialize_sample(sample) for sample in samples]


def stream_materialized_batches(pool, batches: list[list[Sample]], rpc_chunk: int = 32):
    """Yield materialized batches in order, keeping one batch's tasks in flight
    so pool workers materialize batch k while batch k-1 trains on the GPU.

    ``pool=None`` materializes serially in-process. ``rpc_chunk`` bounds the
    per-task pickle payload instead of shipping one message per sample.
    """
    if pool is None:
        for raw in batches:
            yield [materialize_sample(sample) for sample in raw]
        return
    step = max(1, rpc_chunk)
    pending = None
    for raw in batches:
        futures = [
            pool.submit(materialize_chunk, raw[i:i + step])
            for i in range(0, len(raw), step)
        ]
        if pending is not None:
            yield _collect_materialized(pending)
        pending = futures
    if pending is not None:
        yield _collect_materialized(pending)


def _collect_materialized(futures) -> list[Sample]:
    out: list[Sample] = []
    for future in futures:
        out.extend(future.result())
    return out


@dataclass
class SelfPlayConfig:
    players: int = 4
    sims: int = 100
    temperature: float = 1.0
    max_moves: int = 600
    seed: int | None = None
    temperature_warmup_moves: int = 30
    temperature_decay_moves: int = 30
    temperature_final: float = 0.0
    temperature_by_move: Callable[[int], float] | None = None
    # Record the training observation from a determinization of the true state
    # (opponent hands re-sampled from the hidden pool) instead of the
    # simulator's full-information state. Search evaluates leaves under
    # per-simulation determinization, so sampling the true opponent hands here
    # would train the network on inputs it never sees at inference time.
    # The current player's own hand, the public board, the market and the
    # discard history are identical either way.
    determinize_observation: bool = True
    # Store a decision point as (determinized snapshot, sparse visit
    # distribution over canonical moves) instead of dense state tensors plus the
    # full candidate matrix. A full-legal point carries ~N*55 float32 (~66 KB
    # at N=300); the snapshot form is a few KB, which is what makes a
    # multi-iteration replay buffer affordable. Materialization happens in the
    # trainer (`materialize_sample`, parallelized by `Trainer._materialize_pool`).
    store_snapshots: bool = True
    # If positive, discard samples from collapsed/deadlock games where any player's
    # final score is below this threshold (e.g. 20.0), matching the imitation gate.
    min_vp_filter: float = 0.0

    def temperature_for_move(self, move_index: int) -> float:
        """Return the self-play sampling temperature for a zero-based move.

        A custom callback takes precedence.  Otherwise the default schedule
        keeps the initial temperature during a short exploration warmup,
        linearly interpolates to ``temperature_final``, then stays there.
        """
        if self.temperature_by_move is not None:
            temperature = self.temperature_by_move(move_index)
        else:
            start = max(float(self.temperature), 0.0)
            final = max(float(self.temperature_final), 0.0)
            warmup = max(int(self.temperature_warmup_moves), 0)
            decay = max(int(self.temperature_decay_moves), 0)
            if move_index < warmup or decay == 0:
                temperature = start if move_index < warmup else final
            else:
                progress = min((move_index - warmup) / decay, 1.0)
                temperature = start + (final - start) * progress
        temperature = float(temperature)
        if not np.isfinite(temperature):
            raise ValueError("temperature schedule returned a non-finite value")
        return max(temperature, 0.0)


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
    temperature = float(temperature)
    if not np.isfinite(temperature):
        raise ValueError("temperature must be finite")
    if temperature <= 0.0:
        slot = max(result.visits, key=result.visits.get)
        return result.canon_by_candidate[slot]
    slots = list(result.visits)
    counts = np.asarray([result.visits[s] for s in slots], dtype=np.float64)
    if np.any(counts < 0.0) or not np.isfinite(counts).all():
        raise ValueError("search visits must be finite and non-negative")
    # AlphaZero temperature sampling raises visit counts to 1/T.  In
    # particular, [60, 30] at T=1 yields [2/3, 1/3], unlike exp(visits/T).
    with np.errstate(over="ignore", invalid="ignore"):
        weights = (counts + 1e-8) ** (1.0 / temperature)
    # Very small temperatures can overflow the direct power even though the
    # normalized distribution is well-defined.  Recompute in log space while
    # preserving the same power-law probabilities.
    if not np.isfinite(weights).all() or weights.sum() <= 0.0:
        log_weights = np.log(counts + 1e-8) / temperature
        log_weights -= np.max(log_weights)
        weights = np.exp(log_weights)
    probs = weights / weights.sum()
    slot = np.random.choice(slots, p=probs)
    return result.canon_by_candidate[slot]


def play_game(
    mcts: SearchLike,
    cfg: SelfPlayConfig | None = None,
    stats: dict | None = None,
) -> tuple[list, list]:
    """Play one self-play game; returns (samples, final_vps)."""
    return play_game_with_roles([mcts.search] * 4, cfg, stats=stats)


def play_game_with_roles(
    roles,
    cfg: SelfPlayConfig | None = None,
    collect: set | None = None,
    stats: dict | None = None,
    add_root_noise: bool = True,
) -> tuple[list, list]:
    """Play one game where each seat is driven by its own search role.

    `roles[pid]` is a callable(state, sims, add_root_noise) -> SearchResult
    (used for matchmaking: opponent seats may run a different network).
    Samples are recorded for every move whose pid is in `collect` (default:
    all seats, matching the pure self-play path). Returns (samples, final_vps).

    When ``stats`` is given it is filled with run diagnostics, currently
    ``failed_applies`` (reused tree children the rules rejected) and
    ``rewritten_applies`` (reused tree children that silently executed a
    different card than enumerated).

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
    failed_applies = 0
    rewritten_applies = 0
    moves = 0
    while not state.game_over and moves < cfg.max_moves:
        moves += 1
        pid = state.current_player_id
        result = roles[pid](state, cfg.sims, add_root_noise)
        failed_applies += int(getattr(result, "failed_applies", 0) or 0)
        rewritten_applies += int(getattr(result, "rewritten_applies", 0) or 0)
        if result.best is None:
            break
        recorded: Sample | None = None
        if pid in collect and not state.is_bankrupt(pid):
            observed = state.determinize() if cfg.determinize_observation else state
            if cfg.store_snapshots:
                # Fast-path: directly aggregate canonical visit probabilities from the
                # search result without generating full-legal candidate tensors upfront.
                total_visits = sum(result.visits.values())
                if total_visits > 0:
                    sparse = {}
                    for cid, count in result.visits.items():
                        if count > 0:
                            canon = result.canon_by_candidate.get(cid)
                            if canon:
                                sparse[canon] = sparse.get(canon, 0.0) + (float(count) / total_visits)
                elif result.best:
                    sparse = {result.best: 1.0}
                else:
                    sparse = {}
                s = Sample(pid=pid, era=state.era, value=0.0, winner=0.0,
                           snapshot=bytes(observed.snapshot()),
                           policy_by_canonical=sparse)
            else:
                canonical_candidates, candidate_tensor = encode_legal_candidates(state)
                cells, links, merchants, seats, g = observed.state_tokens()
                policy = coalesce_equivalent_policy(
                    candidate_tensor.numpy(), _candidate_policy(canonical_candidates, result)
                )
                s = Sample(pid=pid, cells=cells, links=links, merchants=merchants,
                           seats=seats, global_vec=g, policy=policy, value=0.0,
                           winner=0.0, candidates=candidate_tensor.numpy(), era=state.era)
            samples.append(s)
            recorded = s
            if state.era == 0:
                canal_samples.append(s)
        chosen = _sample_move(result, cfg.temperature_for_move(moves - 1))
        if recorded is not None:
            # The sampled move is the only sibling with an outcome label, so it
            # is the supervision for the action-conditioned value head.
            recorded.played_canonical = chosen
        try:
            summary, ok = state.apply_move_raw(chosen)
        except ValueError:
            summary, ok = ("", False)
        if not ok:
            # Defensive: fall back to the search's best (executable) move, then
            # to the first legal move if needed.
            try:
                summary, ok = state.apply_move_raw(result.best)
                if ok and recorded is not None:
                    recorded.played_canonical = result.best
            except ValueError:
                ok = False
            if not ok:
                legal = state.legal_moves()
                if not legal:
                    break
                try:
                    summary, ok = state.apply_move_raw(legal[0][1])
                    if ok and recorded is not None:
                        recorded.played_canonical = legal[0][1]
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
    if cfg.min_vp_filter > 0.0 and float(np.min(vps)) < cfg.min_vp_filter:
        if stats is not None:
            stats["failed_applies"] = failed_applies
            stats["rewritten_applies"] = rewritten_applies
            stats["moves"] = moves
        return [], vps

    value, winner = _value_targets(vps, state.final_ranking(), state.player_count)
    # Rail-era samples (and any canal samples that never got a canal-econ stamp,
    # e.g. a game that ended in the canal era) take the FINAL economy.
    final_econ = {p: e for p, e in enumerate(state.final_econ())}
    for s in samples:
        s.value = value
        s.winner = winner
        if s.econ is None:
            s.econ = np.asarray(final_econ[s.pid], dtype=np.float32)
    if stats is not None:
        stats["failed_applies"] = failed_applies
        stats["rewritten_applies"] = rewritten_applies
        stats["moves"] = moves
    return samples, vps


def _generate_imitation_game(args):
    """Generate one heuristic game in a worker process."""
    seed, players, max_moves = args

    steps, canal_econ_raw, final_econ_raw, vps_raw, ranking = be.simulate_heuristic_game(
        seed=seed, players=players, max_moves=max_moves
    )
    vps = np.asarray(vps_raw, dtype=np.float64)
    value, winner = _value_targets(vps, ranking, players)
    canal_econ = {p: np.asarray(e, dtype=np.float32) for p, e in enumerate(canal_econ_raw)}
    final_econ = {p: np.asarray(e, dtype=np.float32) for p, e in enumerate(final_econ_raw)}

    local = []
    for pid, era, snapshot, canon in steps:
        econ = canal_econ[pid] if era == 0 else final_econ[pid]
        local.append(Sample(
            pid=pid, era=era, value=value, winner=winner,
            econ=econ, snapshot=bytes(snapshot), teacher_canonical=canon,
        ))
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

    Each move records the state + the heuristic's concrete candidate distribution
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
    total_table_vp = 0.0
    total_winner_vp = 0.0
    min_vp_observed = float("inf")
    max_vp_observed = float("-inf")

    def consume(result) -> bool:
        nonlocal accepted_games, attempted_games, total_table_vp, total_winner_vp, min_vp_observed, max_vp_observed
        attempted_games += 1
        local, vps = result
        if accepted(vps):
            # The in-memory API historically returned ready-to-train samples;
            # keep that contract while shard generation remains snapshot-backed.
            if local and isinstance(local[0], Sample):
                samples.extend(materialize_samples(local))
            else:
                # Keep lightweight test/dry-run producers compatible.
                samples.extend(local)
            accepted_games += 1
            total_table_vp += float(vps.sum())
            total_winner_vp += float(vps.max())
            min_vp_observed = min(min_vp_observed, float(vps.min()))
            max_vp_observed = max(max_vp_observed, float(vps.max()))

        avg_table = total_table_vp / (accepted_games * players) if accepted_games > 0 else 0.0
        avg_winner = total_winner_vp / accepted_games if accepted_games > 0 else 0.0
        min_str = f"min {min_vp_observed:.0f}" if min_vp_observed != float("inf") else "min --"
        progress.update(
            accepted_games,
            extra=(
                f"vp {avg_table:.1f} (win {avg_winner:.1f} {min_str}) | "
                f"accepted: {accepted_games}/{n_games} (att: {attempted_games})"
            ),
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


def generate_imitation_sample_shards(
    n_games: int,
    sample_dir: str | os.PathLike,
    players: int = 4,
    max_moves: int = 600,
    workers: int | None = None,
    min_avg_vp: float | None = None,
    min_vp: float | None = None,
    max_attempts: int | None = None,
) -> list[Path]:
    """Generate imitation games and spill each accepted game to disk.

    Unlike :func:`generate_imitation_samples`, this function never retains the
    complete replay set in the parent process.  A shard contains one pickled
    ``list[Sample]`` and can be loaded, trained, and released independently.
    """
    out_dir = Path(sample_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    buffered: list[Sample] = []

    def flush():
        if not buffered:
            return
        path = out_dir / f"imitation-{len(paths):06d}.pkl"
        with path.open("wb") as f:
            pickle.dump(buffered[:], f, protocol=pickle.HIGHEST_PROTOCOL)
        paths.append(path)
        buffered.clear()

    def sink(local, _vps):
        buffered.extend(local)
        # Keep at most a few dozen games' samples in memory while generating.
        if len(buffered) >= 32768:
            flush()

    _generate_imitation_with_sink(
        n_games, players, max_moves, workers, min_avg_vp, min_vp,
        max_attempts, sink,
    )
    flush()
    return paths


def _generate_imitation_with_sink(
    n_games, players, max_moves, workers, min_avg_vp, min_vp, max_attempts, sink,
):
    """Shared generator core; ``sink`` is called for each accepted game."""
    # Keep the original implementation's validation and scheduling behavior,
    # but consume accepted results immediately instead of extending a global list.
    if n_games <= 0:
        return
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
    def accepted(vps):
        return ((min_avg_vp is None or float(vps.mean()) > min_avg_vp)
                and (min_vp is None or float(vps.min()) > min_vp))
    worker_count = min(max_attempts, workers if workers is not None else min(8, os.cpu_count() or 1))
    if worker_count > 1:
        for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            os.environ[name] = "1"
    jobs = [(gi, players, max_moves) for gi in range(max_attempts)]
    accepted_games = attempted_games = 0
    total_table_vp = 0.0
    total_winner_vp = 0.0
    min_vp_observed = float("inf")
    max_vp_observed = float("-inf")
    progress = Progress(total=n_games, label="accepted imitation game")

    def consume(result):
        nonlocal accepted_games, attempted_games, total_table_vp, total_winner_vp, min_vp_observed, max_vp_observed
        attempted_games += 1
        local, vps = result
        if accepted(vps):
            sink(local, vps)
            accepted_games += 1
            total_table_vp += float(vps.sum())
            total_winner_vp += float(vps.max())
            min_vp_observed = min(min_vp_observed, float(vps.min()))
            max_vp_observed = max(max_vp_observed, float(vps.max()))

        avg_table = total_table_vp / (accepted_games * players) if accepted_games > 0 else 0.0
        avg_winner = total_winner_vp / accepted_games if accepted_games > 0 else 0.0
        min_str = f"min {min_vp_observed:.0f}" if min_vp_observed != float("inf") else "min --"
        progress.update(
            accepted_games,
            extra=(
                f"vp {avg_table:.1f} (win {avg_winner:.1f} {min_str}) | "
                f"accepted: {accepted_games}/{n_games} (att: {attempted_games})"
            ),
        )
        return accepted_games == n_games
    if worker_count == 1:
        for job in jobs:
            if consume(_generate_imitation_game(job)):
                break
    else:
        with ProcessPoolExecutor(max_workers=worker_count, mp_context=mp.get_context("spawn")) as pool:
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
        raise RuntimeError(f"only accepted {accepted_games}/{n_games} imitation games after {attempted_games} attempts; relax min_avg_vp/min_vp or increase max_attempts")
    progress.done()
    if accepted_games > 0:
        avg_table = total_table_vp / (accepted_games * players)
        avg_winner = total_winner_vp / accepted_games
        print(
            f"Imitation generation complete: {accepted_games} games | "
            f"table avg {avg_table:.1f} VP, winner avg {avg_winner:.1f} VP, "
            f"range [{min_vp_observed:.0f}..{max_vp_observed:.0f}]"
        )


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
    base_seed = cfg.seed
    for game_id in range(n_games):
        # Reusing one fixed seed for every game replays the same deal, so the
        # batch degenerates into one correlated sample. Derive a unique seed
        # per game; callers running several iterations should keep
        # ``base_seed`` disjoint across iterations (e.g. iteration * 1_000_000).
        game_cfg = cfg if base_seed is None else replace(cfg, seed=base_seed + game_id)
        samples, vps = play_game(mcts, game_cfg)
        all_samples.extend(samples)
        vps_sum += vps
        per_game.append(vps)
    return all_samples, vps_sum / n_games, per_game
