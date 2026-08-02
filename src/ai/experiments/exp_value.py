"""Experiment V1: is the value head the bottleneck?

Finding so far: greedy policy = ~7 VP but MCTS (same net) = ~37 VP, i.e. MCTS
is carried by the VALUE head's Q estimates. This experiment tests whether a
better value lifts MCTS:

  Phase A: freeze everything, train ONLY value_head on normalized-VP targets
           from a large batch of heuristic games (isolates 'value undertrained').
  Phase B: also unfreeze the trunk (policy_head stays frozen), keep training on
           value targets (tests whether trunk features can support a better value).

Eval after each phase (same seeds, apples-to-apples). Success = MCTS VP rises
clearly above the bootstrap level (~37). Run from repo root (~15 min):
    PYTHONPATH=src/ai ./src/engine/.venv/Scripts/python.exe src/ai/exp_value.py
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from brass_ai.evaluate import evaluate_mcts_vs_baseline
from brass_ai.mcts import ISMCTS, MCTSConfig
from brass_ai.net import PolicyValueNet
from brass_ai.progress import Progress
from brass_ai.selfplay import generate_imitation_samples
from brass_ai.train import _to_batch


def train_value(net, samples, device, lr, epochs, batch_size, freeze_trunk, label):
    for p in net.parameters():
        p.requires_grad_(False)
    net.value_head.weight.requires_grad_(True)
    net.value_head.bias.requires_grad_(True)
    if not freeze_trunk:
        for p in net.trunk.parameters():
            p.requires_grad_(True)

    opt = torch.optim.AdamW(
        [p for p in net.parameters() if p.requires_grad], lr=lr
    )
    n = len(samples)
    prog = Progress(epochs, label, every_s=10.0)
    last = 0.0
    for e in range(epochs):
        idx = np.random.permutation(n)
        tot, cnt = 0.0, 0
        for start in range(0, n, batch_size):
            chunk = [samples[i] for i in idx[start:start + batch_size]]
            batch = _to_batch(chunk)
            tensors = {k: torch.as_tensor(v, device=device) for k, v in batch.items()}
            _, value = net(tensors)
            loss = F.mse_loss(value, tensors["value"])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot += loss.item()
            cnt += 1
        last = tot / cnt
        prog.update(e + 1, f"val_mse={last:.4f}")
    return last


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default="checkpoints/bootstrap.pt")
    ap.add_argument("--games", type=int, default=250)
    ap.add_argument("--epochs_a", type=int, default=15)
    ap.add_argument("--epochs_b", type=int, default=10)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--eval_games", type=int, default=6)
    ap.add_argument("--eval_sims", type=int, default=60)
    args = ap.parse_args()

    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    net = PolicyValueNet()
    net.load_state_dict(torch.load(args.ckpt, map_location=device))
    net.eval()
    mcts = ISMCTS(net, MCTSConfig(c_puct=1.5, max_depth=8), device=device)

    def eval_step(tag):
        wr, mvp, hvp = evaluate_mcts_vs_baseline(
            mcts, args.eval_games, args.eval_sims, baseline="heuristic"
        )
        print(f"  [{tag}] MCTS vs heuristic: win={wr:.0%} vp={mvp:.1f}/{hvp:.1f}")
        return mvp

    t0 = time.time()
    samples = generate_imitation_samples(args.games)
    print(f"generated {len(samples)} samples from {args.games} heuristic games "
          f"({time.time()-t0:.0f}s)")
    vp_pre = eval_step("pre")

    print(f"\nPhase A: train value_head only ({args.epochs_a} epochs)")
    va = train_value(net, samples, device, args.lr, args.epochs_a, args.batch,
                     freeze_trunk=True, label="phaseA")
    vp_a = eval_step("phase A")
    torch.save(net.state_dict(), "checkpoints/exp_value_phaseA.pt")

    print(f"\nPhase B: also unfreeze trunk, train value ({args.epochs_b} epochs)")
    vb = train_value(net, samples, device, args.lr, args.epochs_b, args.batch,
                     freeze_trunk=False, label="phaseB")
    vp_b = eval_step("phase B")
    torch.save(net.state_dict(), "checkpoints/exp_value_phaseB.pt")

    print(f"\nsummary: pre={vp_pre:.1f} | A={vp_a:.1f} | B={vp_b:.1f} "
          f"(val_mse A={va:.4f}, B={vb:.4f})")


if __name__ == "__main__":
    main()
