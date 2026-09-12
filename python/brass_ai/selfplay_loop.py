"""Long-running self-play training loop (AlphaZero-style) for Brass.

One iteration:

1. **Act** - the champion network plays full games (optionally against a pool of
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
5. **Promote** - only validated champions generate new self-play games and
   enter the historical pool. The candidate continues training between gates.

Targets are produced by the search, not by bootstrapping a value head: MCTS
returns a root visit distribution (policy target) and the game's final ranking
supplies the value target. See `selfplay.py` for the sample contract.
"""

from __future__ import annotations

import copy
import json
import math
import multiprocessing as mp
import os
import time
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Callable

import numpy as np
import torch

from .evaluate import benchmark_net_vs_heuristic
from .mp_selfplay import SelfPlayPool
from .net import PolicyValueNet
from .progress import Progress
from .rust_mcts import RustISMCTS, RustMCTSConfig, heuristic_search, HeuristicSearchAdapter
from .selfplay import Sample, SelfPlayConfig, play_game_with_roles
from .train import TrainConfig, Trainer


def wilson_lower_bound(wins: int, games: int, z: float = 1.28) -> float:
    """Lower bound of the Wilson score interval for a win rate.

    z=1.28 provides a one-sided 90% confidence lower bound, suitable for
    evaluations with 12-24 games in 4-player games (baseline winrate = 0.25).
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
    train_epochs: int = 1
    train: TrainConfig = field(default_factory=TrainConfig)
    # Evaluation.
    eval_every: int = 5
    eval_games: int = 12
    eval_sims: int = 128
    heuristic_eval_games: int = 12
    heuristic_eval_sims: int = 128
    promote_winrate: float = 0.35


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
    heuristic_avg_vp: float | None = None
    heuristic_zero_games: int = 0
    champion_heuristic_avg_vp: float | None = None
    collected_avg_vp: float | None = None
    zero_vp_players: int = 0
    completed_games: int = 0
    filtered_games: int = 0
    game_vps: list = field(default_factory=list)
    promoted: bool = False

    def to_json(self) -> str:
        data = {k: v for k, v in self.__dict__.items() if not k.startswith("_")}
        return json.dumps(data, sort_keys=True)


def _cpu_state_dict(net: PolicyValueNet) -> dict:
    return {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}


def _arena_worker_fn(args):
    candidate_weights, opponent_weights, mcts_cfg_dict, selfplay_dict, sims, seed, seat, players = args
    import torch
    from .net import PolicyValueNet
    from .rust_mcts import RustISMCTS, RustMCTSConfig
    from .selfplay import SelfPlayConfig, play_game_with_roles

    torch.set_num_threads(1)
    c_net = PolicyValueNet()
    c_net.load_state_dict(candidate_weights)
    c_net.eval()
    c_mcts = RustISMCTS(c_net, RustMCTSConfig(**mcts_cfg_dict, device="cpu"))

    o_net = PolicyValueNet()
    o_net.load_state_dict(opponent_weights)
    o_net.eval()
    o_mcts = RustISMCTS(o_net, RustMCTSConfig(**mcts_cfg_dict, device="cpu"))

    roles = [o_mcts.search] * players
    roles[seat] = c_mcts.search
    game_cfg = SelfPlayConfig(
        players=players,
        sims=sims,
        seed=seed,
        temperature=0.0,
        temperature_final=0.0,
        **selfplay_dict,
    )
    diagnostics = {}
    play_game_with_roles(
        roles, game_cfg, collect=set(), stats=diagnostics, add_root_noise=False
    )
    return diagnostics["final_ranking"][0] == seat


def arena_winrate(
    candidate: RustISMCTS,
    opponent: RustISMCTS,
    selfplay_cfg: SelfPlayConfig,
    games: int,
    sims: int,
    seed_base: int,
    players: int = 4,
    workers: int = 1,
    candidate_net: PolicyValueNet | None = None,
    opponent_net: PolicyValueNet | None = None,
    mcts_cfg: RustMCTSConfig | None = None,
) -> tuple[int, int]:
    """Play `games` candidate-vs-opponent games with rotating candidate seats.

    Temperature is forced to 0 so the arena measures the networks, not sampling
    noise; seeds are fixed per index so every checkpoint faces the same deals.
    If workers > 1 and weights are provided, games run concurrently across a process pool.
    """
    if games <= 0:
        return 0, 0

    if workers > 1 and candidate_net is not None and opponent_net is not None:
        c_weights = {k: v.detach().cpu() for k, v in candidate_net.state_dict().items()}
        o_weights = {k: v.detach().cpu() for k, v in opponent_net.state_dict().items()}
        m_cfg = mcts_cfg or RustMCTSConfig()
        cfg_dict = {
            "c_puct": m_cfg.c_puct,
            "max_depth": m_cfg.max_depth,
            "dirichlet_alpha": m_cfg.dirichlet_alpha,
            "dirichlet_weight": m_cfg.dirichlet_weight,
            "batch_size": m_cfg.batch_size,
            "candidate_k": m_cfg.candidate_k,
            "prior_top_k": m_cfg.prior_top_k,
            "fpu": m_cfg.fpu,
            "fpu_reduction": m_cfg.fpu_reduction,
            "q_init": m_cfg.q_init,
        }
        selfplay_dict = {
            "store_snapshots": selfplay_cfg.store_snapshots,
            "determinize_observation": selfplay_cfg.determinize_observation,
            "max_moves": selfplay_cfg.max_moves,
            "min_vp_filter": getattr(selfplay_cfg, "min_vp_filter", 0.0),
        }
        n_workers = min(workers, games, os.cpu_count() or 1)
        jobs = [
            (c_weights, o_weights, cfg_dict, selfplay_dict, sims, seed_base + g, g % players, players)
            for g in range(games)
        ]
        wins = 0
        prog = Progress(games, f"arena sims={sims} (w={n_workers})")
        with ProcessPoolExecutor(max_workers=n_workers, mp_context=mp.get_context("spawn")) as pool:
            for done_count, is_win in enumerate(pool.map(_arena_worker_fn, jobs), 1):
                if is_win:
                    wins += 1
                prog.update(done_count)
        prog.done()
        return wins, games

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
        diagnostics = {}
        play_game_with_roles(
            roles, game_cfg, collect=set(), stats=diagnostics, add_root_noise=False
        )
        if diagnostics["final_ranking"][0] == seat:
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
        "min_vp_filter": cfg.selfplay.min_vp_filter,
        "game_log_dir": cfg.selfplay.game_log_dir,
    }


def run_selfplay(
    net: PolicyValueNet,
    trainer: Trainer,
    cfg: LoopConfig,
    on_iteration: Callable[[IterationStats, PolicyValueNet, Trainer], None] | None = None,
    should_stop: Callable[[], bool] | None = None,
    start_iteration: int = 0,
    best_state: dict | None = None,
    opponent_pool: list[dict] | None = None,
) -> list[IterationStats]:
    """Run the self-play loop; returns per-iteration statistics.

    `on_iteration` receives the freshly updated stats plus the live net/trainer
    (for checkpointing and logging). `should_stop` is polled between iterations.
    `best_state` restores the arena reference on `--resume`.
    `opponent_pool` restores the historical matchmaking pool on `--resume`.
    """
    rng = np.random.default_rng(cfg.seed)
    buffer = ReplayBuffer(cfg.max_buffer_samples, cfg.max_buffer_iterations)
    history: list[IterationStats] = []

    best_net = copy.deepcopy(net).eval()
    if best_state is not None:
        best_net.load_state_dict(best_state)
    opponent_pool = list(opponent_pool) if opponent_pool is not None else [_cpu_state_dict(best_net)]
    champion_benchmark = None
    pool: SelfPlayPool | None = None
    local_mcts: RustISMCTS | None = None
    if cfg.workers > 1:
        pool = SelfPlayPool(n_workers=cfg.workers, device="cpu")
    else:
        local_mcts = RustISMCTS(best_net, cfg.mcts)
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
                    best_net,
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
                samples, diagnostics = _play_batch_local(
                    local_mcts, cfg, base_seed, opponent_pool=opponent_pool
                )
            stats.selfplay_sec = time.time() - t0
            stats.samples = len(samples)
            stats.failed_applies = int(diagnostics.get("failed_applies", 0))
            stats.rewritten_applies = int(diagnostics.get("rewritten_applies", 0))
            stats.moves = int(diagnostics.get("moves", 0))
            game_vps = diagnostics.get("game_vps", [])
            stats.game_vps = game_vps
            stats.completed_games = len(game_vps)
            stats.filtered_games = int(diagnostics.get("filtered_games", 0))
            collected = diagnostics.get("collected_vps", [])
            stats.collected_avg_vp = float(np.mean(collected)) if collected else None
            if game_vps:
                all_vps = [float(vp) for g in game_vps for vp in g]
                stats.avg_vp = float(np.mean(all_vps))
                stats.min_vp = float(np.min(all_vps))
                stats.max_vp = float(np.max(all_vps))
                stats.winner_avg_vp = float(np.mean([max(g) for g in game_vps]))
                stats.zero_vp_players = sum(vp == 0 for vp in all_vps)

            buffer.add(samples)
            buffer.trim()
            stats.buffer = len(buffer)

            t1 = time.time()
            draw = buffer.draw(
                cfg.train_samples, cfg.recent_fraction, cfg.recent_iterations, rng
            )
            if draw:
                losses = []
                for ep in range(max(1, cfg.train_epochs)):
                    label = f"train it{iteration}" if cfg.train_epochs <= 1 else f"train it{iteration} e{ep+1}"
                    losses.extend(trainer.train_one_epoch(draw, label))
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
                    workers=min(cfg.workers, 4),
                    candidate_net=net,
                    opponent_net=best_net,
                    mcts_cfg=cfg.mcts,
                )
                stats.arena_wins, stats.arena_games = wins, games
                stats.arena_winrate = wins / games if games else 0.0
                stats.arena_lower = wilson_lower_bound(wins, games, z=1.28)
                if champion_benchmark is None:
                    champion_benchmark = benchmark_net_vs_heuristic(
                        best_net, cfg.heuristic_eval_sims, cfg.heuristic_eval_games,
                        device=cfg.device, mcts_cfg=cfg.mcts, workers=min(cfg.workers, 4),
                    )
                candidate_benchmark = benchmark_net_vs_heuristic(
                    net, cfg.heuristic_eval_sims, cfg.heuristic_eval_games,
                    device=cfg.device,
                    mcts_cfg=cfg.mcts,
                    workers=min(cfg.workers, 4),
                )
                stats.heuristic_winrate = candidate_benchmark["win_rate"]
                stats.heuristic_avg_vp = candidate_benchmark["mcts_mean"]
                stats.heuristic_zero_games = sum(vp <= 0 for vp in candidate_benchmark["mcts_vps"])
                stats.champion_heuristic_avg_vp = champion_benchmark["mcts_mean"]
                # 4-player game (1 candidate vs 3 opponents): baseline winrate is 25%.
                # Promotion requires beating the winrate threshold AND Wilson lower bound >= 0.25.
                healthy = (candidate_benchmark["games"] > 0
                           and stats.heuristic_zero_games == 0
                           and stats.heuristic_avg_vp >= stats.champion_heuristic_avg_vp)
                if healthy and stats.arena_winrate >= cfg.promote_winrate and stats.arena_lower >= 0.25:
                    best_net.load_state_dict(_cpu_state_dict(net))
                    stats.promoted = True
                    champion_benchmark = candidate_benchmark
                    opponent_pool.append(_cpu_state_dict(best_net))
            stats.eval_sec = time.time() - t2

            del opponent_pool[: max(0, len(opponent_pool) - cfg.pool_size)]
            stats._opponent_pool = list(opponent_pool)
            stats._champion_state = _cpu_state_dict(best_net)

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
    mcts: RustISMCTS,
    cfg: LoopConfig,
    base_seed: int,
    opponent_pool: list[dict] | None = None,
) -> tuple[list[Sample], dict]:
    """Single-process actor path (workers == 1); also used by smoke tests."""
    samples: list[Sample] = []
    totals = {"failed_applies": 0, "rewritten_applies": 0, "moves": 0,
              "games": 0, "truncated": 0, "game_vps": [],
              "collected_vps": [], "filtered_games": 0}
    pool_mcts: list[RustISMCTS] = []
    if opponent_pool:
        for pw in opponent_pool:
            pn = PolicyValueNet()
            pn.load_state_dict(pw)
            pn.eval()
            pool_mcts.append(RustISMCTS(pn, cfg.mcts))

    for game in range(cfg.games_per_iter):
        game_cfg = replace(
            cfg.selfplay,
            seed=base_seed + game,
            sims=cfg.sims,
        )
        stats: dict = {}
        try:
            learner = game % 4
            rng = np.random.default_rng(game_cfg.seed)
            np.random.seed(game_cfg.seed)
            roles = [mcts.search] * 4
            collect = {0, 1, 2, 3}
            for seat in range(4):
                if seat == learner:
                    continue
                r = rng.random()
                if cfg.heuristic_prob > 0.0 and r < cfg.heuristic_prob:
                    roles[seat] = HeuristicSearchAdapter()
                    collect.discard(seat)
                elif cfg.mm_prob > 0.0 and pool_mcts and r < (cfg.heuristic_prob + cfg.mm_prob):
                    opp = pool_mcts[rng.integers(len(pool_mcts))]
                    roles[seat] = opp.search
                    collect.discard(seat)
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
                if key == "collected_vps":
                    totals[key].extend(stats[key])
                else:
                    totals[key] += int(stats[key])
        totals["games"] += 1
    return samples, totals


def write_metrics(path: str | Path, stats: IterationStats) -> None:
    """Append one iteration's statistics as a JSON line."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(stats.to_json() + "\n")
