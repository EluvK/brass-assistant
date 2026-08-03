"""Experiment: does masking the policy loss fix the weak policy?

Old bootstrap.pt was trained with an UNMASKED log_softmax over all 1316 slots;
~700 always-illegal double-rail slots pollute the softmax denominator (~53% of
initial probability mass sits on them), so the net wastes capacity suppressing
them instead of discriminating the real legal moves.

This trains a fresh imitation net with a MASKED policy loss (normalize only over
legal slots) and compares greedy-policy VP + MCTS VP against the old net.

Run from repo root (~12 min, has progress/ETA output):
    PYTHONPATH=src/ai ./src/engine/.venv/Scripts/python.exe src/ai/experiments/exp_masked_loss.py
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import torch

import brass_engine as be

from brass_ai import build_input
from brass_ai.evaluate import (evaluate_mcts_vs_baseline, heuristic_policy,
                               play_game_with_policies)
from brass_ai.mcts import ISMCTS, MCTSConfig
from brass_ai.net import PolicyValueNet
from brass_ai.progress import Progress
from brass_ai.selfplay import generate_imitation_samples
from brass_ai.train import TrainConfig, Trainer


def greedy_policy(net):
    dev = next(net.parameters()).device
    net.eval()

    def pol(state):
        batch = build_input.encode_state(state)
        batch = {k: v.to(dev) for k, v in batch.items()}
        with torch.no_grad():
            type_logits, goal_logits, _ = net.policy_value(batch)
        logits = net.merge_logits(type_logits, goal_logits)[0].cpu().numpy()
        mask = np.zeros(len(logits), dtype=bool)
        for s, _, _ in state.legal_moves():
            mask[s] = True
        slot = int(np.argmax(np.where(mask, logits, -1e9)))
        for s, canon, _ in state.legal_moves():
            if s == slot:
                return canon
        return None

    return pol


def eval_greedy(net, n_games=6):
    pols = [greedy_policy(net), heuristic_policy, heuristic_policy, heuristic_policy]
    mvp = hvp = wins = 0
    for g in range(n_games):
        vps, rank = play_game_with_policies(pols, seed=g)
        mvp += vps[0]
        hvp += sum(v for i, v in enumerate(vps) if i != 0) / 3
        wins += rank[0] == 0
    return wins / n_games, mvp / n_games, hvp / n_games


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=250)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--eval_games", type=int, default=4)
    ap.add_argument("--eval_sims", type=int, default=60)
    ap.add_argument("--old", type=str, default="checkpoints/bootstrap.pt")
    ap.add_argument("--out", type=str, default="checkpoints/imitation_masked.pt")
    args = ap.parse_args()

    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    old = PolicyValueNet()
    old.load_state_dict(torch.load(args.old, map_location=device))
    old.eval()
    print(f"old net: {args.old}")

    t0 = time.time()
    samples = generate_imitation_samples(args.games)
    print(f"generated {len(samples)} imitation samples from {args.games} games "
          f"({time.time()-t0:.0f}s)")

    net = PolicyValueNet()
    trainer = Trainer(net, TrainConfig(
        device=device, epochs=args.epochs, batch_size=args.batch, lr=args.lr,
    ))
    t1 = time.time()
    losses = trainer.train_on_samples(samples)
    print(f"trained ({time.time()-t1:.0f}s): policy={losses['policy']:.4f} "
          f"value={losses['value']:.4f}")
    torch.save(net.state_dict(), args.out)

    for tag, n in [("old (unmasked)", old), ("new (masked)", net)]:
        wg, mg, hg = eval_greedy(n, n_games=6)
        print(f"  [{tag}] greedy policy vs heuristic: win={wg:.0%} vp={mg:.1f}/{hg:.1f}")

    for tag, n in [("old (unmasked)", old), ("new (masked)", net)]:
        mcts = ISMCTS(n, MCTSConfig(c_puct=1.5, max_depth=8), device=device)
        wr, mvp, hvp = evaluate_mcts_vs_baseline(
            mcts, args.eval_games, args.eval_sims, baseline="heuristic"
        )
        print(f"  [{tag}] MCTS vs heuristic: win={wr:.0%} vp={mvp:.1f}/{hvp:.1f}")
    print(f"saved: {args.out}")


if __name__ == "__main__":
    main()
