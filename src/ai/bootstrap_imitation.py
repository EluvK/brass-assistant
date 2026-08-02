"""Bootstrap warm-start: imitate the heuristic to get a playable policy fast.

The heuristic plays self-play games (no MCTS needed — fast); each move records
the state, the heuristic's chosen policy slot (one-hot target) and the game's
normalized final VP (value target). The network is trained on this imitation
data, then its MCTS is evaluated vs the heuristic.

This is a standard warm-start before pure AlphaZero self-play (which needs
thousands of iterations to bootstrap from random). Run from the repo root:

    PYTHONPATH=src/ai ./src/engine/.venv/Scripts/python.exe src/ai/bootstrap_imitation.py
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np
import torch

import brass_engine as be

from brass_ai.evaluate import evaluate_mcts_vs_baseline
from brass_ai.mcts import ISMCTS, MCTSConfig
from brass_ai.net import PolicyValueNet
from brass_ai.selfplay import Sample, _normalize
from brass_ai.train import TrainConfig, train_on_samples


def gen_heuristic_samples(n_games: int, players: int = 4, max_moves: int = 600):
    """Play heuristic-vs-heuristic games, one one-hot imitation sample per move."""
    samples = []
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
                       own_hand=oh, opp_hands=op, policy=policy, value=0.0)
            )
            try:
                state.apply_move(canon)
            except ValueError:
                break
        z = _normalize(np.asarray(state.player_vps(), dtype=np.float64))
        for s in local:
            s.value = z[s.pid]
        samples.extend(local)
    return samples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=80)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--eval_games", type=int, default=2)
    ap.add_argument("--eval_sims", type=int, default=60)
    ap.add_argument("--ckpt", type=str, default="checkpoints/bootstrap.pt")
    args = ap.parse_args()

    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    t0 = time.time()
    samples = gen_heuristic_samples(args.games)
    print(f"generated {len(samples)} imitation samples from {args.games} heuristic games "
          f"({time.time()-t0:.0f}s)")

    net = PolicyValueNet()
    tc = TrainConfig(device=device, epochs=args.epochs, batch_size=args.batch, lr=args.lr)
    t1 = time.time()
    losses = train_on_samples(net, samples, tc)
    print(f"trained ({time.time()-t1:.0f}s): "
          f"policy={losses['policy']:.3f} value={losses['value']:.3f}")

    os.makedirs(os.path.dirname(args.ckpt) or ".", exist_ok=True)
    torch.save(net.state_dict(), args.ckpt)

    mcts = ISMCTS(net, MCTSConfig(c_puct=1.5, max_depth=8), device=device)
    for baseline in ("heuristic", "2ply"):
        wr, mvp, bvp = evaluate_mcts_vs_baseline(
            mcts, args.eval_games, args.eval_sims, baseline=baseline
        )
        print(f"MCTS(bootstrap net) vs {baseline}: win_rate={wr:.0%} "
              f"(mcts_vp={mvp:.1f} vs {baseline}_vp={bvp:.1f})")
    print(f"checkpoint saved: {args.ckpt}")


if __name__ == "__main__":
    main()
