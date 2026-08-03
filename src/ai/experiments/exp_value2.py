"""Experiment V2: value-head specialization on a large supervised dataset.

Hypothesis (from V1 + the degradation seen under self-play): the value head is
the weak link — self-play drove val_mse from ~0.5 up to ~0.74 while the policy
held flat, and the search's Q estimates are noisy enough to cause the
catastrophic games seen in the reliable benchmark.

This experiment freezes trunk + policy head (V1 Phase B showed unfreezing the
trunk pollutes the shared representation and hurts) and trains ONLY `value_head`
on normalized final-VP targets from a large batch of cheap heuristic-vs-heuristic
games (~0.5s/game). The target is "how well I will do against heuristics from
this state", which is exactly what the MCTS leaf evaluation needs.

Run from repo root (~15-20 min for --games 1000):
    PYTHONPATH=src/ai ./src/engine/.venv/Scripts/python.exe \\
        src/ai/experiments/exp_value2.py --ckpt checkpoints/best_masked.pt --games 1000

Then benchmark the saved checkpoint with the reliable 20-seed harness.
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import torch
import torch.nn.functional as F

from brass_ai.net import PolicyValueNet
from brass_ai.progress import Progress
from brass_ai.selfplay import generate_imitation_samples
from brass_ai.train import _to_batch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default="checkpoints/best_masked.pt")
    ap.add_argument("--games", type=int, default=1000)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--out", type=str, default="checkpoints/best_masked_value.pt")
    args = ap.parse_args()

    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    net = PolicyValueNet()
    sd = torch.load(args.ckpt, map_location=device)
    net.load_state_dict(sd["model"] if "model" in sd else sd)
    net.eval()

    def val_mse():
        with torch.no_grad():
            idx = np.random.default_rng(0).choice(len(samples), size=min(4096, len(samples)), replace=False)
            batch = _to_batch([samples[i] for i in idx])
            tensors = {k: torch.as_tensor(v, device=device) for k, v in batch.items()}
            _, _, value = net(tensors)
            return F.mse_loss(value, tensors["value"]).item()

    t0 = time.time()
    samples = generate_imitation_samples(args.games)
    print(f"generated {len(samples)} samples from {args.games} heuristic games "
          f"({time.time()-t0:.0f}s)")
    print(f"value-head MSE before: {val_mse():.4f}")

    # Freeze everything except the value head.
    for p in net.parameters():
        p.requires_grad_(False)
    net.value_head.weight.requires_grad_(True)
    net.value_head.bias.requires_grad_(True)
    opt = torch.optim.AdamW(
        [p for p in net.parameters() if p.requires_grad], lr=args.lr
    )

    n = len(samples)
    prog = Progress(args.epochs, "value-head", every_s=10.0)
    for e in range(args.epochs):
        idx = np.random.permutation(n)
        tot, cnt = 0.0, 0
        for start in range(0, n, args.batch):
            chunk = [samples[i] for i in idx[start:start + args.batch]]
            batch = _to_batch(chunk)
            tensors = {k: torch.as_tensor(v, device=device) for k, v in batch.items()}
            _, _, value = net(tensors)
            loss = F.mse_loss(value, tensors["value"])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot += loss.item()
            cnt += 1
        prog.update(e + 1, f"val_mse={tot/cnt:.4f}")

    print(f"value-head MSE after : {val_mse():.4f}")
    torch.save(net.state_dict(), args.out)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
