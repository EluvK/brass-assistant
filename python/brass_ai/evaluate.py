"""
Head-to-head evaluation: network-guided MCTS vs the engine heuristic.
"""

from __future__ import annotations

import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor
from typing import TYPE_CHECKING

import numpy as np

from . import _engine as be
from .progress import Progress

if TYPE_CHECKING:
    from .net import PolicyValueNet
    from .rust_mcts import RustISMCTS, RustMCTSConfig


def heuristic_policy(state) -> str | None:
    canon, _, _ = state.choose_heuristic()
    return canon


def mcts_policy(mcts: RustISMCTS, sims: int):
    def pol(state) -> str | None:
        r = mcts.search(state, sims, add_root_noise=False)
        return r.best

    return pol


def play_game_with_policies(policies, seed: int, players: int = 4, max_moves: int = 600):
    """Play a full game; `policies[pid]` is a state->canonical callable."""
    state = be.GameState(seed=seed, players=players)
    moves = 0
    while not state.game_over and moves < max_moves:
        moves += 1
        pid = state.current_player_id
        canon = policies[pid](state)
        if canon is None:
            legal = state.legal_moves()
            if not legal:
                break
            canon = legal[0][1]
        try:
            state.apply_move(canon)
        except ValueError:
            # Defensive: a canonical from the engine can occasionally be broken
            # (double-rail coal enumeration); fall back to the first legal move.
            legal = state.legal_moves()
            if not legal:
                break
            try:
                state.apply_move(legal[0][1])
            except ValueError:
                break
    if not state.game_over:
        raise RuntimeError(f"evaluation did not reach a legal terminal state after {moves} moves")
    return state.player_vps(), state.final_ranking()


def _eval_game_worker(args):
    """Worker task for running one benchmark game against heuristic AI."""
    weights, cfg_dict, sims, seed, seat, players, max_moves = args
    import torch
    from .net import PolicyValueNet
    from .rust_mcts import RustISMCTS, RustMCTSConfig

    torch.set_num_threads(1)
    net = PolicyValueNet()
    net.load_state_dict(weights)
    net.eval()
    mcts = RustISMCTS(net, RustMCTSConfig(**cfg_dict, device="cpu"))
    mcts_pol = mcts_policy(mcts, sims)
    policies = [mcts_pol if p == seat else heuristic_policy for p in range(players)]
    vps, ranking = play_game_with_policies(policies, seed=seed, players=players, max_moves=max_moves)
    return seat, vps, bool(ranking[0] == seat)


def _benchmark_net_vs_heuristic_parallel(
    net: PolicyValueNet,
    cfg: RustMCTSConfig,
    sims: int,
    games: int = 20,
    players: int = 4,
    workers: int = 4,
) -> dict:
    """Run benchmark games against heuristic across a process pool."""
    weights = {k: v.detach().cpu() for k, v in net.state_dict().items()}
    cfg_dict = {
        "c_puct": cfg.c_puct,
        "max_depth": cfg.max_depth,
        "dirichlet_alpha": cfg.dirichlet_alpha,
        "dirichlet_weight": cfg.dirichlet_weight,
        "batch_size": cfg.batch_size,
        "candidate_k": cfg.candidate_k,
        "prior_top_k": cfg.prior_top_k,
        "fpu": cfg.fpu,
        "fpu_reduction": cfg.fpu_reduction,
        "q_init": cfg.q_init,
    }
    n_workers = min(workers, games, os.cpu_count() or 1)
    jobs = [
        (weights, cfg_dict, sims, g, g % players, players, 600)
        for g in range(games)
    ]
    prog = Progress(games, f"bench sims={sims} (w={n_workers})")
    wins = 0
    mcts_vps: list[float] = []
    base_vps: list[float] = []

    with ProcessPoolExecutor(max_workers=n_workers, mp_context=mp.get_context("spawn")) as pool:
        for done_count, (seat, vps, is_win) in enumerate(pool.map(_eval_game_worker, jobs), 1):
            if is_win:
                wins += 1
            mcts_vps.append(float(vps[seat]))
            others = [v for i, v in enumerate(vps) if i != seat]
            base_vps.append(float(np.mean(others)))
            prog.update(done_count)
    prog.done()

    return {
        "win_rate": wins / games if games else 0.0,
        "wins": wins,
        "games": games,
        "mcts_vps": mcts_vps,
        "base_vps": base_vps,
        "mcts_mean": float(np.mean(mcts_vps)) if mcts_vps else 0.0,
        "mcts_median": float(np.median(mcts_vps)) if mcts_vps else 0.0,
        "base_mean": float(np.mean(base_vps)) if base_vps else 0.0,
    }


def benchmark_mcts_vs_heuristic(mcts, sims: int, games: int = 20, players: int = 4):
    """Decision-grade MCTS benchmark against the engine heuristic.

    The MCTS seat and seed both rotate over ``0..games-1``.  Returns per-game
    scores alongside aggregate statistics for training gates and analysis.
    ``games=0`` skips the benchmark and returns an empty result, so callers can
    disable evaluation without special-casing the exit path.
    """
    if games <= 0:
        return {
            "win_rate": float("nan"),
            "wins": 0,
            "games": 0,
            "mcts_vps": [],
            "base_vps": [],
            "mcts_mean": float("nan"),
            "mcts_median": float("nan"),
            "base_mean": float("nan"),
        }
    wins = 0
    mcts_vps, base_vps = [], []
    prog = Progress(games, f"bench sims={sims}")
    mcts_pol = mcts_policy(mcts, sims)
    for g in range(games):
        seat = g % players
        policies = [mcts_pol if p == seat else heuristic_policy for p in range(players)]
        vps, ranking = play_game_with_policies(policies, seed=g, players=players)
        mcts_vps.append(vps[seat])
        others = [v for i, v in enumerate(vps) if i != seat]
        base_vps.append(float(np.mean(others)))
        if ranking[0] == seat:
            wins += 1
        prog.update(g + 1)
    prog.done()
    return {
        "win_rate": wins / games,
        "wins": wins,
        "games": games,
        "mcts_vps": mcts_vps,
        "base_vps": base_vps,
        "mcts_mean": float(np.mean(mcts_vps)),
        "mcts_median": float(np.median(mcts_vps)),
        "base_mean": float(np.mean(base_vps)),
    }


def benchmark_net_vs_heuristic(
    net: PolicyValueNet,
    sims: int,
    games: int = 20,
    device: str = "cuda",
    c_puct: float | None = None,
    max_depth: int = 10,
    mcts_cfg: RustMCTSConfig | None = None,
    workers: int = 1,
):
    """Construct Rust MCTS for ``net`` and run the standard benchmark.

    If ``mcts_cfg`` is provided, it preserves all search settings (e.g. prior_top_k,
    q_init, c_puct=0.25). Workers > 1 uses a process pool for multi-game concurrency.
    """
    from .rust_mcts import RustISMCTS, RustMCTSConfig

    if mcts_cfg is not None:
        cfg = RustMCTSConfig(
            c_puct=c_puct if c_puct is not None else mcts_cfg.c_puct,
            max_depth=max_depth if max_depth != 10 else mcts_cfg.max_depth,
            dirichlet_alpha=mcts_cfg.dirichlet_alpha,
            dirichlet_weight=mcts_cfg.dirichlet_weight,
            batch_size=mcts_cfg.batch_size,
            candidate_k=mcts_cfg.candidate_k,
            prior_top_k=mcts_cfg.prior_top_k,
            fpu=mcts_cfg.fpu,
            fpu_reduction=mcts_cfg.fpu_reduction,
            q_init=mcts_cfg.q_init,
            device=device or mcts_cfg.device,
        )
    else:
        cfg = RustMCTSConfig(
            c_puct=c_puct if c_puct is not None else 0.25,
            max_depth=max_depth,
            device=device,
        )

    if workers > 1 and games > 1:
        return _benchmark_net_vs_heuristic_parallel(net, cfg, sims, games, workers=workers)

    mcts = RustISMCTS(net, cfg)
    return benchmark_mcts_vs_heuristic(mcts, sims, games)
