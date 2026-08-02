"""Short training run to validate the RL pipeline end-to-end on GPU/CPU.

Self-play a few games with the network-guided MCTS, train on the samples,
then measure the win rate vs the heuristic baseline. Best model (by win rate)
is saved to --ckpt. Run from the repo root:

    PYTHONPATH=src/ai ./src/engine/.venv/Scripts/python.exe src/ai/train_short.py
"""

from __future__ import annotations

import argparse
import os
import time

import torch

from brass_ai.evaluate import evaluate_mcts_vs_baseline
from brass_ai.mcts import ISMCTS, MCTSConfig
from brass_ai.net import PolicyValueNet
from brass_ai.selfplay import SelfPlayConfig, play_batch
from brass_ai.train import TrainConfig, train_on_samples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=6)
    ap.add_argument("--games", type=int, default=2)
    ap.add_argument("--sims", type=int, default=40)
    ap.add_argument("--eval_games", type=int, default=2)
    ap.add_argument("--eval_sims", type=int, default=60)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--ckpt", type=str, default="checkpoints/best.pt")
    args = ap.parse_args()

    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}  ({torch.cuda.get_device_name(0) if device == 'cuda' else 'CPU'})")

    net = PolicyValueNet()
    mcts = ISMCTS(net, MCTSConfig(c_puct=1.5, max_depth=8), device=device)
    tc = TrainConfig(device=device, epochs=args.epochs, batch_size=args.batch, lr=args.lr)
    sp = SelfPlayConfig(players=4, sims=args.sims, temperature=1.0, max_moves=600)

    os.makedirs(os.path.dirname(args.ckpt) or ".", exist_ok=True)
    best_wr = -1.0
    print(f"{'iter':>4} {'samples':>7} {'pol_loss':>8} {'val_loss':>8} "
          f"{'vs_heur_wr':>10} {'mcts_vp/heur_vp':>18}  note")
    for it in range(args.iters):
        t0 = time.time()
        samples, _, _ = play_batch(mcts, args.games, sp)
        sp_t = time.time() - t0

        t1 = time.time()
        losses = train_on_samples(net, samples, tc)
        tr_t = time.time() - t1

        wr_h, mvp_h, hvp_h = evaluate_mcts_vs_baseline(
            mcts, args.eval_games, args.eval_sims, baseline="heuristic"
        )
        marker = ""
        if wr_h > best_wr:
            best_wr = wr_h
            torch.save(net.state_dict(), args.ckpt)
            marker = "*save"
        print(
            f"{it:>4} {len(samples):>7} {losses['policy']:>8.3f} {losses['value']:>8.3f} "
            f"{wr_h:>9.0%} {f'{mvp_h:.1f}/{hvp_h:.1f}':>18} {marker:>8}  "
            f"(sp {sp_t:.0f}s + tr {tr_t:.0f}s)"
        )

    print(f"\ndone. best win rate {best_wr:.0%}; checkpoint: {args.ckpt}")


if __name__ == "__main__":
    main()
