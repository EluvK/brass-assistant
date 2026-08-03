"""Reliable benchmark: MCTS vs heuristic over a FIXED seed set.

The in-loop eval (train_mp) is intentionally fast (8 games) and therefore noisy
(observed +/-20 VP across seed samples). This script gives a decision-grade
reading: N games over fixed seeds 0..N-1 with the MCTS seat rotated, reporting
win rate plus mean/median MCTS and heuristic VP. Use it before/after training to
judge whether a checkpoint actually improved.

Run from repo root:
    PYTHONPATH=src/ai ./src/engine/.venv/Scripts/python.exe \\
        src/ai/experiments/benchmark.py --ckpt checkpoints/best_masked.pt --sims 600 --games 20
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import torch

import brass_engine as be
from brass_ai.net import PolicyValueNet
from brass_ai.rust_mcts import RustISMCTS, RustMCTSConfig
from brass_ai.evaluate import heuristic_policy
from brass_ai.progress import Progress


def load_net(ckpt: str, device: str) -> PolicyValueNet:
    net = PolicyValueNet()
    sd = torch.load(ckpt, map_location="cpu")
    model = sd["model"] if "model" in sd else sd
    net.load_state_dict(model)
    net.eval()
    if device != "cpu":
        net.to(device)
    return net


def play_with_policies(policies, seed: int, players: int = 4, max_moves: int = 600):
    state = be.GameState(seed=seed, players=players)
    moves = 0
    while not state.game_over and moves < max_moves:
        moves += 1
        canon = policies[state.current_player_id](state)
        if canon is None:
            legal = state.legal_moves()
            if not legal:
                break
            canon = legal[0][1]
        try:
            state.apply_move(canon)
        except ValueError:
            legal = state.legal_moves()
            if not legal:
                break
            try:
                state.apply_move(legal[0][1])
            except ValueError:
                break
    return state.player_vps(), state.final_ranking()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--sims", type=int, default=600)
    ap.add_argument("--games", type=int, default=20)
    ap.add_argument("--c_puct", type=float, default=2.5)
    ap.add_argument("--max_depth", type=int, default=10)
    ap.add_argument("--device", type=str, default="cuda")
    args = ap.parse_args()

    net = load_net(args.ckpt, args.device)
    rm = RustISMCTS(net, RustMCTSConfig(
        c_puct=args.c_puct, max_depth=args.max_depth, device=args.device,
    ))

    def mcts_pol(state):
        r = rm.search(state, args.sims, add_root_noise=False)
        return r.best

    wins = 0
    mcts_vps, base_vps = [], []
    prog = Progress(args.games, f"bench sims={args.sims}")
    t0 = time.time()
    for g in range(args.games):
        seat = g % 4
        policies = []
        for p in range(4):
            policies.append(mcts_pol if p == seat else heuristic_policy)
        vps, ranking = play_with_policies(policies, seed=g, players=4)
        mcts_vps.append(vps[seat])
        others = [v for i, v in enumerate(vps) if i != seat]
        base_vps.append(float(np.mean(others)))
        if ranking[0] == seat:
            wins += 1
        prog.update(g + 1)
    prog.done()

    m = np.array(mcts_vps)
    b = np.array(base_vps)
    print(f"\n{args.ckpt} @ sims={args.sims} over {args.games} fixed seeds "
          f"({time.time()-t0:.0f}s):")
    print(f"  win_rate : {wins/args.games:.2f} ({wins}/{args.games})")
    print(f"  MCTS     : mean={m.mean():6.1f}  median={np.median(m):6.1f}  "
          f"min={m.min():5.1f}  max={m.max():5.1f}")
    print(f"  heuristic: mean={b.mean():6.1f}  median={np.median(b):6.1f}")
    print(f"  per-seed MCTS VP: {[int(v) for v in m]}")


if __name__ == "__main__":
    main()
