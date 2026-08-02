"""Step 0 de-risk: start pure self-play from the bootstrap checkpoint.

Loads checkpoints/bootstrap.pt (a policy trained by imitating the heuristic),
then runs a few short AlphaZero-style iterations (self-play -> train) with a
persistent Trainer and low-LR fine-tuning. The concern being de-risked: the
random-start collapse observed earlier (5 iters of tiny data -> MCTS regressed
to 0 VP). Success = MCTS vs heuristic VP stays >= ~30 (the bootstrap level) or
rises, rather than collapsing.

Eval discipline: only heuristic, a few games, once per iteration.
Run from the repo root:
    PYTHONPATH=src/ai ./src/engine/.venv/Scripts/python.exe src/ai/selfplay_from_bootstrap.py
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
from brass_ai.train import TrainConfig, Trainer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--games", type=int, default=1)
    ap.add_argument("--sims", type=int, default=120)
    ap.add_argument("--eval_games", type=int, default=2)
    ap.add_argument("--eval_sims", type=int, default=60)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-4)  # low LR: fine-tune from bootstrap
    ap.add_argument("--ckpt", type=str, default="checkpoints/bootstrap.pt")
    ap.add_argument("--out", type=str, default="checkpoints/selfplay0.pt")
    args = ap.parse_args()

    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    net = PolicyValueNet()
    net.load_state_dict(torch.load(args.ckpt, map_location=device))
    net.eval()
    print(f"loaded start weights: {args.ckpt}")

    trainer = Trainer(net, TrainConfig(
        device=device, epochs=args.epochs, batch_size=args.batch, lr=args.lr,
        t_max=max(args.iters * args.epochs, 1),
    ))
    mcts = ISMCTS(net, MCTSConfig(c_puct=1.5, max_depth=8), device=device)
    sp = SelfPlayConfig(players=4, sims=args.sims, temperature=1.0, max_moves=600)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    def quick_eval(tag):
        wr, mvp, hvp = evaluate_mcts_vs_baseline(
            mcts, args.eval_games, args.eval_sims, baseline="heuristic"
        )
        print(f"  [{tag}] MCTS vs heuristic: win={wr:.0%} vp={mvp:.1f}/{hvp:.1f}")
        return mvp

    vp0 = quick_eval("pre")
    print(f"baseline (bootstrap net, before self-play): vp={vp0:.1f}\n")

    for it in range(args.iters):
        t0 = time.time()
        samples, avg_vps, _ = play_batch(mcts, args.games, sp)
        sp_t = time.time() - t0
        t1 = time.time()
        losses = trainer.train_on_samples(samples)
        tr_t = time.time() - t1
        print(
            f"iter {it}: samples={len(samples)} avg_self_vps={avg_vps} "
            f"pol={losses['policy']:.3f} val={losses['value']:.3f} lr={trainer.current_lr():.2e} "
            f"(sp {sp_t:.0f}s + tr {tr_t:.0f}s)"
        )
        vp = quick_eval(f"iter {it}")
        torch.save(trainer.state_dict(), args.out.replace(".pt", f"_it{it}.pt"))
        print()

    torch.save(trainer.state_dict(), args.out)
    print(f"saved trainer state: {args.out}")
    verdict = (
        "OK: no collapse" if vp >= vp0 * 0.9
        else "WARN: VP dropped below 90% of the bootstrap baseline"
    )
    print(f"verdict: {verdict} (pre={vp0:.1f}, post={vp:.1f})")


if __name__ == "__main__":
    main()
