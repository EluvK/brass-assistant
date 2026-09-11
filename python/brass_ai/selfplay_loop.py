"""Long-running self-play training loop (AlphaZero-style) for Brass.

One iteration:

1. **Act** - the current network plays full games (optionally against a pool of
   historical checkpoints) and emits snapshot-backed samples: the determinized
   observation plus the sparse root-visit distribution.
2. **Remember** - samples enter a rolling replay window. Dense full-legal
   decision points are ~0.4 MB each, so the window keeps the snapshot form and
   materializes only the slice that is actually trained on.
3. **Train** - the trainer consumes a random slice of the window (recent-heavy,
   with a historical fraction so the network does not forget old opponents).
4. **Evaluate** - periodically: an arena against the current best checkpoint and
   a benchmark against the engine heuristic, both on rotated seats and fixed
   seeds so different checkpoints are compared on the same deals.
5. **Promote** - the latest network is *always* the one that keeps training; the
   best checkpoint only gates the opponent pool and the reported strength. A
   hard gate on a noisy 40-game result would stall progress.

Targets are produced by the search, not by bootstrapping a value head: MCTS
returns a root visit distribution (policy target) and the game's final ranking
supplies the value target. See `selfplay.py` for the sample contract.
"""

from __future__ import annotations

import copy
import json
import math
import time
from collections import deque
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Callable

import numpy as np
import torch

from .evaluate import benchmark_net_vs_heuristic
from .mp_selfplay import SelfPlayPool
from .net import PolicyValueNet
from .progress import Progress
from .rust_mcts import RustISMCTS, RustMCTSConfig, heuristic_search
from .selfplay import Sample, SelfPlayConfig, play_game_with_roles
from .train import TrainConfig, Trainer


def wilson_lower_bound(wins: int, games: int, z: float = 1.96) -> float:
    """Lower bound of the Wilson score interval for a win rate.

    A 40-game arena has a ~±15% standard error, so promotion decisions read this
    bound instead of the raw win rate.
    """
    if games <= 0:
        return 0.0
    p = wins / games
    z2 = z * z
    centre = p + z2 / (2 * games)
    margin = z * math.sqrt(p * (1.0 - p) / games + z2 / (4 * games * games))
    return max(0.0, (centre - margin) / (1.0 + z2 / games))


def _take(pool: list, count: int, rng: np.random.Generator) -> list:
    if count <= 0 or not pool:
        return []
    if count >= len(pool):
        return list(pool)
    indices = rng.choice(len(pool), size=count, replace=False)
    return [pool[int(i)] for i in indices]


class ReplayBuffer:
    """Rolling window of self-play samples, bounded by iterations and size."""

    def __init__(self, max_samples: int = 400_000, max_iterations: int = 20):
        if max_samples < 1 or max_iterations < 1:
            raise ValueError("replay bounds must be >= 1")
        self.max_samples = max_samples
        self.max_iterations = max_iterations
        self._samples: list[Sample] = []
        self._iteration_sizes: deque[int] = deque()

    def add(self, samples: list[Sample]) -> None:
        if not samples:
            return
        self._samples.extend(samples)
        self._iteration_sizes.append(len(samples))

    def trim(self) -> None:
        while self._iteration_sizes:
            too_many_iterations = len(self._iteration_sizes) > self.max_iterations
            too_many_samples = len(self._samples) > self.max_samples
            if not (too_many_iterations or too_many_samples):
                break
            drop = self._iteration_sizes.popleft()
            del self._samples[:drop]

    def __len__(self) -> int:
        return len(self._samples)

    def recent(self, iterations: int) -> list[Sample]:
        if iterations <= 0 or not self._iteration_sizes:
            return []
        sizes = list(self._iteration_sizes)[-iterations:]
        return self._samples[len(self._samples) - sum(sizes):]

    def draw(
        self,
        count: int,
        recent_fraction: float,
        recent_iterations: int,
        rng: np.random.Generator,
    ) -> list[Sample]:
        total = len(self._samples)
        if total == 0 or count <= 0:
            return []
        count = min(count, total)
        recent = self.recent(recent_iterations)
        older = self._samples[: total - len(recent)]
        # Prefer the newest iterations, then shift the budget to whichever side
        # actually has data (early in a run the window holds one iteration).
        n_recent = min(len(recent), int(round(count * recent_fraction)))
        n_older = count - n_recent
        if n_older > len(older):
            n_recent = min(len(recent), n_recent + (n_older - len(older)))
            n_older = count - n_recent
        return _take(recent, n_recent, rng) + _take(older, n_older, rng)


@dataclass
class LoopConfig:
    iterations: int = 50
    games_per_iter: int = 16
    sims: int = 128
    workers: int = 8
    seed: int = 0
    # Iterations are seeded as seed + iteration * seed_stride, so every
    # iteration (and every game inside it) gets a distinct deal.
    seed_stride: int = 1_000_000
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    mcts: RustMCTSConfig = field(default_factory=RustMCTSConfig)
    selfplay: SelfPlayConfig = field(default_factory=SelfPlayConfig)
    # Historical-opponent matchmaking: with this probability an opponent seat
    # plays a checkpoint from the pool instead of the current network.
    mm_prob: float = 0.25
    pool_size: int = 6
    # Heuristic teacher matchmaking: with this probability an opponent seat
    # plays the engine heuristic AI directly (zero NN cost, breaks self-play collusion).
    heuristic_prob: float = 0.0
    # Replay window and per-iteration training budget.
    max_buffer_samples: int = 400_000
    max_buffer_iterations: int = 20
    recent_fraction: float = 0.75
    recent_iterations: int = 4
    train_samples: int = 40_000
    train: TrainConfig = field(default_factory=TrainConfig)
    # Evaluation.
    eval_every: int = 5
    eval_games: int = 40
    eval_sims: int = 128
    heuristic_eval_games: int = 20
    heuristic_eval_sims: int = 128
    promote_winrate: float = 0.55


@dataclass
class IterationStats:
    iteration: int = 0
    games: int = 0
    samples: int = 0
    buffer: int = 0
    trained: int = 0
    losses: dict = field(default_factory=dict)
    selfplay_sec: float = 0.0
    train_sec: float = 0.0
    eval_sec: float = 0.0
    failed_applies: int = 0
    rewritten_applies: int = 0
    moves: int = 0
    avg_vp: float = 0.0
    min_vp: float = 0.0
    max_vp: float = 0.0
    winner_avg_vp: float = 0.0
    arena_wins: int = 0
    arena_games: int = 0
    arena_winrate: float = 0.0
    arena_lower: float = 0.0
    heuristic_winrate: float | None = None
    promoted: bool = False

    def to_json(self) -> str:
        return json.dumps(self.__dict__, sort_keys=True)


def _cpu_state_dict(net: PolicyValueNet) -> dict:
    return {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}


def arena_winrate(
    candidate: RustISMCTS,
    opponent: RustISMCTS,
    selfplay_cfg: SelfPlayConfig,
    games: int,
    sims: int,
    seed_base: int,
    players: int = 4,
) -> tuple[int, int]:
    """Play `games` candidate-vs-opponent games with rotating candidate seats.

    Temperature is forced to 0 so the arena measures the networks, not sampling
    noise; seeds are fixed per index so every checkpoint faces the same deals.
    """
    wins = 0
    prog = Progress(games, f"arena sims={sims}")
    for game in range(games):
        seat = game % players
        roles = [opponent.search] * players
        roles[seat] = candidate.search
        game_cfg = replace(
            selfplay_cfg,
            seed=seed_base + game,
            temperature=0.0,
            temperature_final=0.0,
            sims=sims,
        )
        samples, _ = play_game_with_roles(roles, game_cfg, collect={seat})
        if samples and int(np.argmax(samples[0].winner)) == seat:
            wins += 1
        prog.update(game + 1)
    prog.done()
    return wins, games


def _selfplay_opts(cfg: LoopConfig) -> dict:
    """Worker-side `SelfPlayConfig` overrides (includes the temperature plan)."""
    return {
        "store_snapshots": cfg.selfplay.store_snapshots,
        "determinize_observation": cfg.selfplay.determinize_observation,
        "temperature_warmup_moves": cfg.selfplay.temperature_warmup_moves,
        "temperature_decay_moves": cfg.selfplay.temperature_decay_moves,
        "temperature_final": cfg.selfplay.temperature_final,
        "max_moves": cfg.selfplay.max_moves,
    }


def run_selfplay(
    net: PolicyValueNet,
    trainer: Trainer,
    cfg: LoopConfig,
    on_iteration: Callable[[IterationStats, PolicyValueNet, Trainer], None] | None = None,
    should_stop: Callable[[], bool] | None = None,
    start_iteration: int = 0,
    best_state: dict | None = None,
) -> list[IterationStats]:
    """Run the self-play loop; returns per-iteration statistics.

    `on_iteration` receives the freshly updated stats plus the live net/trainer
    (for checkpointing and logging). `should_stop` is polled between iterations.
    `best_state` restores the arena reference on `--resume`.
    """
    rng = np.random.default_rng(cfg.seed)
    buffer = ReplayBuffer(cfg.max_buffer_samples, cfg.max_buffer_iterations)
    history: list[IterationStats] = []

    best_net = copy.deepcopy(net).eval()
    if best_state is not None:
        best_net.load_state_dict(best_state)
    opponent_pool: list[dict] = []
    pool: SelfPlayPool | None = None
    local_mcts: RustISMCTS | None = None
    if cfg.workers > 1:
        pool = SelfPlayPool(n_workers=cfg.workers, device="cpu")
    else:
        local_mcts = RustISMCTS(net, cfg.mcts)
    # `make_net_fn` closes over the module object, so a single adapter keeps
    # seeing the live weights as the trainer updates them in place.
    arena_mcts = RustISMCTS(net, cfg.mcts)
    best_mcts = RustISMCTS(best_net, cfg.mcts)

    try:
        for iteration in range(start_iteration, cfg.iterations):
            stats = IterationStats(iteration=iteration, games=cfg.games_per_iter)
            base_seed = cfg.seed + iteration * cfg.seed_stride

            t0 = time.time()
            if pool is not None:
                games_per_worker = max(1, -(-cfg.games_per_iter // cfg.workers))
                samples, _ = pool.generate(
                    net,
                    games_per_worker=games_per_worker,
                    sims=cfg.sims,
                    seed=base_seed,
                    mcts_cfg={
                        "c_puct": cfg.mcts.c_puct,
                        "max_depth": cfg.mcts.max_depth,
                        "dirichlet_alpha": cfg.mcts.dirichlet_alpha,
                        "dirichlet_weight": cfg.mcts.dirichlet_weight,
                        "batch_size": cfg.mcts.batch_size,
                        "candidate_k": cfg.mcts.candidate_k,
                        "prior_top_k": cfg.mcts.prior_top_k,
                        "fpu": cfg.mcts.fpu,
                        "q_init": cfg.mcts.q_init,
                    },
                    temperature=cfg.selfplay.temperature,
                    mm_pool=opponent_pool,
                    mm_prob=cfg.mm_prob if opponent_pool else 0.0,
                    heuristic_prob=cfg.heuristic_prob,
                    selfplay_opts=_selfplay_opts(cfg),
                )
                diagnostics = pool.last_diagnostics
                stats.games = int(diagnostics.get("games", 0))
            else:
                assert local_mcts is not None
                samples, diagnostics = _play_batch_local(local_mcts, cfg, base_seed)
            stats.selfplay_sec = time.time() - t0
            stats.samples = len(samples)
            stats.failed_applies = int(diagnostics.get("failed_applies", 0))
            stats.rewritten_applies = int(diagnostics.get("rewritten_applies", 0))
            stats.moves = int(diagnostics.get("moves", 0))
            game_vps = diagnostics.get("game_vps", [])
            if game_vps:
                all_vps = [float(vp) for g in game_vps for vp in g]
                stats.avg_vp = float(np.mean(all_vps))
                stats.min_vp = float(np.min(all_vps))
                stats.max_vp = float(np.max(all_vps))
                stats.winner_avg_vp = float(np.mean([max(g) for g in game_vps]))

            buffer.add(samples)
            buffer.trim()
            stats.buffer = len(buffer)

            t1 = time.time()
            draw = buffer.draw(
                cfg.train_samples, cfg.recent_fraction, cfg.recent_iterations, rng
            )
            if draw:
                losses = trainer.train_one_epoch(draw, f"train it{iteration}")
                stats.trained = len(draw)
                if losses:
                    stats.losses = {
                        key: sum(loss[key] for loss in losses) / len(losses)
                        for key in losses[0]
                    }
                trainer.step_lr()
            stats.train_sec = time.time() - t1

            t2 = time.time()
            if cfg.eval_every > 0 and (iteration + 1) % cfg.eval_every == 0:
                wins, games = arena_winrate(
                    arena_mcts, best_mcts, cfg.selfplay,
                    cfg.eval_games, cfg.eval_sims, seed_base=cfg.seed + 7_000_000,
                )
                stats.arena_wins, stats.arena_games = wins, games
                stats.arena_winrate = wins / games if games else 0.0
                stats.arena_lower = wilson_lower_bound(wins, games)
                stats.heuristic_winrate = benchmark_net_vs_heuristic(
                    net, cfg.heuristic_eval_sims, cfg.heuristic_eval_games,
                    device=cfg.device,
                )["win_rate"]
                if stats.arena_winrate >= cfg.promote_winrate and wins > games - wins:
                    best_net.load_state_dict(_cpu_state_dict(net))
                    stats.promoted = True
            stats.eval_sec = time.time() - t2

            opponent_pool.append(_cpu_state_dict(net))
            del opponent_pool[: max(0, len(opponent_pool) - cfg.pool_size)]

            history.append(stats)
            if on_iteration is not None:
                on_iteration(stats, net, trainer)
            if should_stop is not None and should_stop():
                break
    finally:
        if pool is not None:
            pool.close()
    return history


def _play_batch_local(
    mcts: RustISMCTS, cfg: LoopConfig, base_seed: int
) -> tuple[list[Sample], dict]:
    """Single-process actor path (workers == 1); also used by smoke tests."""
    samples: list[Sample] = []
    totals = {"failed_applies": 0, "rewritten_applies": 0, "moves": 0,
              "games": 0, "truncated": 0, "game_vps": []}
    for game in range(cfg.games_per_iter):
        game_cfg = replace(
            cfg.selfplay,
            seed=base_seed + game,
            sims=cfg.sims,
        )
        stats: dict = {}
        try:
            learner = game % 4
            roles = [mcts.search] * 4
            collect = {0, 1, 2, 3}
            if cfg.heuristic_prob > 0.0:
                has_mixed = False
                for seat in range(4):
                    if seat != learner and np.random.rand() < cfg.heuristic_prob:
                        roles[seat] = heuristic_search
                        has_mixed = True
                if has_mixed:
                    collect = {learner}
            game_samples, vps = play_game_with_roles(
                roles, game_cfg, collect=collect, stats=stats
            )
        except RuntimeError as exc:
            # A game that exceeds max_moves has no valid terminal target; drop
            # it and keep the batch going instead of killing a long run.
            if "samples discarded" not in str(exc):
                raise
            totals["truncated"] += 1
            continue
        samples.extend(game_samples)
        if vps:
            totals["game_vps"].append([float(x) for x in vps])
        for key in totals:
            if key in stats:
                totals[key] += int(stats[key])
        totals["games"] += 1
    return samples, totals


def write_metrics(path: str | Path, stats: IterationStats) -> None:
    """Append one iteration's statistics as a JSON line."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(stats.to_json() + "\n")
