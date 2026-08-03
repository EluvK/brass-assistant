"""Step 1: multi-process AlphaZero training loop.

Main process: persistent Trainer (AdamW + CosineAnnealingLR, t_max = total
iterations x epochs) + low-frequency rolling-average evaluation vs the
heuristic. Workers: SelfPlayPool (spawn) plays self-play games in parallel and
streams numpy samples back.

Eval discipline: evaluate every `--eval_every` iterations with `--eval_games`
games; the displayed VP is a rolling mean over the last `--eval_window` evals
to dampen the per-game draw noise.

Run from the repo root (spawn requires the __main__ guard, present here):
    PYTHONPATH=src/ai ./src/engine/.venv/Scripts/python.exe src/ai/train_mp.py
"""

from __future__ import annotations

import argparse
import os
import time
from collections import deque

import torch

from brass_ai.evaluate import evaluate_mcts_vs_baseline
from brass_ai.mcts import ISMCTS, MCTSConfig
from brass_ai.mp_selfplay import SelfPlayPool
from brass_ai.net import PolicyValueNet
from brass_ai.train import TrainConfig, Trainer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=8)
    ap.add_argument("--start_iter", type=int, default=0,
                    help="iteration index to start at (with --resume, continues "
                         "training from a mid-run checkpoint instead of replaying)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--games_per_worker", type=int, default=2)
    ap.add_argument("--sims", type=int, default=200)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--c_puct", type=float, default=2.5)
    ap.add_argument("--temp", type=float, default=0.7)
    ap.add_argument("--dirichlet_alpha", type=float, default=0.3)
    ap.add_argument("--dirichlet_weight", type=float, default=0.15)
    ap.add_argument("--eval_every", type=int, default=3)
    ap.add_argument("--eval_games", type=int, default=6)
    ap.add_argument("--eval_sims", type=int, default=60)
    ap.add_argument("--eval_window", type=int, default=3)
    ap.add_argument("--gate", action="store_true",
                    help="revert to the best-evaluated net on VP regression")
    ap.add_argument("--ckpt", type=str, default="checkpoints/bootstrap.pt")
    ap.add_argument("--resume", action="store_true",
                    help="resume optimizer+scheduler from a full trainer-state ckpt")
    ap.add_argument("--out", type=str, default="checkpoints/train_mp.pt")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import multiprocessing as mp
    mp.set_start_method("spawn", force=True)

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device} | workers: {args.workers} | sims: {args.sims} | "
          f"iters: {args.iters}")

    net = PolicyValueNet()
    trainer = Trainer(net, TrainConfig(
        device=device, epochs=args.epochs, batch_size=args.batch, lr=args.lr,
        t_max=max(args.iters * args.epochs, 1),
    ))

    # Load checkpoint. Default: net weights only + fresh optimizer/scheduler
    # (so the LR schedule matches THIS run's t_max). With --resume: load the
    # full trainer state (optimizer + scheduler) for true continuation.
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    if os.path.exists(args.ckpt):
        sd = torch.load(args.ckpt, map_location=device)
        model = sd["model"] if "model" in sd else sd
        net.load_state_dict(model)
        if args.resume and "model" in sd:
            trainer.load_state_dict(sd)
            print(f"resumed trainer state from {args.ckpt} "
                  f"(epoch={trainer.epoch_count})")
        else:
            print(f"loaded weights from {args.ckpt} (fresh optimizer, "
                  f"t_max={args.iters * args.epochs})")
    else:
        print(f"warning: {args.ckpt} not found; starting from random weights")

    mcts = ISMCTS(net, MCTSConfig(c_puct=1.5, max_depth=8), device=device)

    mcts_cfg = {
        "c_puct": args.c_puct,
        "dirichlet_alpha": args.dirichlet_alpha,
        "dirichlet_weight": args.dirichlet_weight,
    }

    rolling = deque(maxlen=args.eval_window)
    best_vp = float("-inf")
    best_state = None
    print(f"{'iter':>4} {'samples':>7} {'pol':>6} {'val':>6} {'lr':>8} "
          f"{'sp_sec':>7} {'games':>5} | {'vs_heur':>16} {'rolling':>8}")
    with SelfPlayPool(n_workers=args.workers, device="cpu") as pool:
        for it in range(args.start_iter, args.iters):
            t0 = time.time()
            samples, counts = pool.generate(
                net, args.games_per_worker, args.sims,
                seed=args.seed + it, mcts_cfg=mcts_cfg, temperature=args.temp,
            )
            sp_t = time.time() - t0
            n_games = len(counts)

            losses = trainer.train_on_samples(samples)
            torch.save(trainer.state_dict(), args.out)

            line = (
                f"{it:>4} {len(samples):>7} {losses['policy']:>6.3f} "
                f"{losses['value']:>6.3f} {trainer.current_lr():>8.1e} "
                f"{sp_t:>7.0f} {n_games:>5}"
            )

            if it % args.eval_every == 0:
                wr, mvp, hvp = evaluate_mcts_vs_baseline(
                    mcts, args.eval_games, args.eval_sims, baseline="heuristic"
                )
                rolling.append(mvp)
                avg = sum(rolling) / len(rolling)
                line += f" | {mvp:>7.1f}/{hvp:<7.1f} {avg:>8.1f}"
                if args.gate:
                    if avg >= best_vp:
                        best_vp = avg
                        best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
                        line += "  (best)"
                    else:
                        net.load_state_dict(best_state)
                        line += "  (reverted)"
            print(line)

    if args.gate and best_state is not None:
        net.load_state_dict(best_state)
        torch.save(trainer.state_dict(), args.out)
        print(f"final weights reverted to best (rolling VP {best_vp:.1f})")

    print(f"\nsaved trainer state: {args.out}")
    if rolling:
        print(f"final rolling MCTS-vs-heuristic VP: {sum(rolling)/len(rolling):.1f}")


if __name__ == "__main__":
    main()
