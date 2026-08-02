"""Head-to-head evaluation: network-guided MCTS vs the heuristic / 2-ply
baselines, with the MCTS seat rotated to cancel seat bias (mirrors the Rust
`mcts-vs-heur` harness in main.rs).
"""

from __future__ import annotations

import brass_engine as be

from .mcts import ISMCTS
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
    mcts: ISMCTS,
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
