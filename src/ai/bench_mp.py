"""Quick throughput check: multi-process pool vs single-process self-play.

Run from repo root:
    PYTHONPATH=src/ai ./src/engine/.venv/Scripts/python.exe src/ai/bench_mp.py --sims 60
"""

from __future__ import annotations

import argparse
import time

import torch

from brass_ai.mcts import ISMCTS, MCTSConfig
from brass_ai.mp_selfplay import SelfPlayPool
from brass_ai.net import PolicyValueNet
from brass_ai.selfplay import SelfPlayConfig, play_game


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sims", type=int, default=60)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    import multiprocessing as mp
    mp.set_start_method("spawn", force=True)

    torch.manual_seed(0)
    net = PolicyValueNet()

    # single-process baseline: 1 game
    mcts = ISMCTS(net, MCTSConfig(c_puct=1.5, max_depth=8), device="cpu")
    t0 = time.time()
    samples, _ = play_game(mcts, SelfPlayConfig(sims=args.sims, max_moves=600))
    t_single = time.time() - t0
    print(f"single-process: {len(samples)} samples in {t_single:.1f}s ({t_single/len(samples)*1e3:.0f} ms/sample)")

    # multi-process: `workers` games in parallel (1 per worker)
    with SelfPlayPool(n_workers=args.workers, device="cpu") as pool:
        t0 = time.time()
        mp_samples, counts = pool.generate(net, 1, args.sims, seed=1234)
        t_mp = time.time() - t0
    n_games = len(counts)
    print(f"mp {args.workers} workers: {len(mp_samples)} samples / {n_games} games in {t_mp:.1f}s")

    per_game_single = t_single
    per_game_mp_eff = t_mp / n_games
    print(f"effective per-game wall: single={per_game_single:.1f}s, mp={per_game_mp_eff:.1f}s "
          f"-> throughput x{per_game_single / per_game_mp_eff:.1f}")
    print(f"sample stream rate: {len(mp_samples)/t_mp:.0f} samples/s (mp)")


if __name__ == "__main__":
    main()
