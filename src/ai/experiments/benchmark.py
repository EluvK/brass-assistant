"""Reliable benchmark: MCTS vs heuristic over a FIXED seed set (decision-grade).

The in-loop eval (train_mp) is intentionally fast (8 games) and therefore noisy
(observed +/-20 VP across seed samples). This script gives a decision-grade
reading: N games over fixed seeds 0..N-1 with the MCTS seat rotated, reporting
win rate plus mean/median MCTS and heuristic VP. Use it before/after training
to judge whether a checkpoint actually improved.

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
from brass_ai.evaluate import benchmark_net_vs_heuristic


def load_net(ckpt: str, device: str) -> PolicyValueNet:
    net = PolicyValueNet()
    sd = torch.load(ckpt, map_location="cpu")
    model = sd["model"] if "model" in sd else sd
    net.load_state_dict(model)
    net.eval()
    if device != "cpu":
        net.to(device)
    return net


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
    t0 = time.time()
    r = benchmark_net_vs_heuristic(
        net, args.sims, args.games, args.device, args.c_puct, args.max_depth
    )
    m = np.array(r["mcts_vps"])
    b = np.array(r["base_vps"])
    print(f"\n{args.ckpt} @ sims={args.sims} over {args.games} fixed seeds "
          f"({time.time()-t0:.0f}s):")
    print(f"  win_rate : {r['win_rate']:.2f} ({r['wins']}/{args.games})")
    print(f"  MCTS     : mean={m.mean():6.1f}  median={np.median(m):6.1f}  "
          f"min={m.min():5.1f}  max={m.max():5.1f}")
    print(f"  heuristic: mean={b.mean():6.1f}  median={np.median(b):6.1f}")
    print(f"  per-seed MCTS VP: {[int(v) for v in m]}")


if __name__ == "__main__":
    main()
