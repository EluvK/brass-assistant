"""
Head-to-head evaluation: network-guided MCTS vs the engine heuristic.
"""

from __future__ import annotations

import numpy as np

import python.brass_ai.brass_engine as be

from .progress import Progress


def heuristic_policy(state) -> str | None:
    canon, _, _ = state.choose_heuristic()
    return canon


def mcts_policy(mcts: ISMCTS, sims: int):
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
    return state.player_vps(), state.final_ranking()


def benchmark_mcts_vs_heuristic(mcts, sims: int, games: int = 20, players: int = 4):
    """Decision-grade MCTS benchmark against the engine heuristic.

    The MCTS seat and seed both rotate over ``0..games-1``.  Returns per-game
    scores alongside aggregate statistics for training gates and analysis.
    """
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


def benchmark_net_vs_heuristic(net, sims: int, games: int = 20, device: str = "cuda",
                               c_puct: float = 2.5, max_depth: int = 10):
    """Construct Rust MCTS for ``net`` and run the standard benchmark."""
    from .rust_mcts import RustISMCTS, RustMCTSConfig

    mcts = RustISMCTS(net, RustMCTSConfig(
        c_puct=c_puct, max_depth=max_depth, device=device,
    ))
    return benchmark_mcts_vs_heuristic(mcts, sims, games)
