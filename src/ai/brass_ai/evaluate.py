"""Head-to-head evaluation: network-guided MCTS vs the heuristic / 2-ply
baselines, with the MCTS seat rotated to cancel seat bias (mirrors the Rust
`mcts-vs-heur` harness in main.rs).
"""

from __future__ import annotations

import numpy as np

import brass_engine as be

from .progress import Progress


def heuristic_policy(state) -> str | None:
    canon, _, _ = state.choose_heuristic()
    return canon


def twoply_policy(state) -> str | None:
    canon, _, _ = state.choose_2ply()
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


def evaluate_mcts_vs_baseline(
    mcts,
    n_games: int,
    sims: int,
    players: int = 4,
    baseline: str = "heuristic",
):
    """One MCTS seat vs `baseline` seats, MCTS seat rotated per game.

    Returns (win_rate, mcts_avg_vp, baseline_avg_vp)."""
    base_pol = twoply_policy if baseline == "2ply" else heuristic_policy
    wins = 0
    mcts_vp_total = 0.0
    base_vp_total = 0.0
    prog = Progress(n_games, f"eval vs {baseline}")
    for g in range(n_games):
        seat = g % players
        policies = []
        for p in range(players):
            policies.append(mcts_policy(mcts, sims) if p == seat else base_pol)
        vps, ranking = play_game_with_policies(policies, seed=g, players=players)
        mcts_vp_total += vps[seat]
        others = [v for i, v in enumerate(vps) if i != seat]
        base_vp_total += sum(others) / len(others)
        if ranking[0] == seat:
            wins += 1
        prog.update(g + 1)
    prog.done()
    return wins / n_games, mcts_vp_total / n_games, base_vp_total / n_games


def benchmark_mcts_vs_heuristic(net, sims: int, games: int = 20, device: str = "cuda",
                                c_puct: float = 2.5, max_depth: int = 10):
    """Decision-grade benchmark (fixed seeds 0..N-1, MCTS seat rotated).

    Returns a dict with win_rate plus MCTS/heuristic VP stats. This is the
    ONLY metric trusted for accept/reject decisions (the in-loop fast eval is
    noisy, observed +/-20 VP)."""
    from .rust_mcts import RustISMCTS, RustMCTSConfig

    rm = RustISMCTS(net, RustMCTSConfig(
        c_puct=c_puct, max_depth=max_depth, device=device,
    ))

    def mcts_pol(state):
        r = rm.search(state, sims, add_root_noise=False)
        return r.best

    wins = 0
    mcts_vps, base_vps = [], []
    prog = Progress(games, f"bench sims={sims}")
    for g in range(games):
        seat = g % 4
        policies = []
        for p in range(4):
            policies.append(mcts_pol if p == seat else heuristic_policy)
        vps, ranking = play_game_with_policies(policies, seed=g, players=4)
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
