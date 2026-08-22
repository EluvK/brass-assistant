"""BC baseline for the concrete-candidate network.

Trains a fresh `PolicyValueNet` from heuristic imitation data under the new
contract:
  * policy target: heuristic distribution over concrete legal candidates
  * value target: the full 4-player normalized z-vector, MSE (no tanh)

The old `best_masked.pt` is kept as an untouched comparison baseline.

Run from repo root (~15-25 min for --games 2000):
    PYTHONPATH=src/ai ./src/engine/.venv/Scripts/python.exe \\
        src/ai/experiments/bc_baseline.py --games 2000 --steps 4000 \\
        --out checkpoints/new_best.pt
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import torch

from brass_ai import build_input
from brass_ai.hierarchical_policy import encode_legal_candidates
from brass_ai.net import PolicyValueNet
from brass_ai.progress import Progress
from brass_ai.selfplay import generate_imitation_samples
from brass_ai.train import TrainConfig, Trainer
from brass_ai.evaluate import heuristic_policy, play_game_with_policies


def greedy_policy(net):
    dev = next(net.parameters()).device
    net.eval()

    def pol(state):
        batch = build_input.encode_state(state)
        batch = {k: v.to(dev) for k, v in batch.items()}
        canonicals, candidates = encode_legal_candidates(state)
        candidates = candidates.unsqueeze(0).to(dev)
        mask = torch.ones((1, len(canonicals)), dtype=torch.bool, device=dev)
        with torch.no_grad():
            out = net.policy_value(batch, candidates, mask)
        return canonicals[int(out["candidate_logits"].argmax(dim=1).item())]

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
    ap.add_argument("--games", type=int, default=2000)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default="checkpoints/new_best.pt")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    t0 = time.time()
    samples = generate_imitation_samples(args.games)
    print(f"generated {len(samples)} imitation samples from {args.games} games "
          f"({time.time()-t0:.0f}s)")

    net = PolicyValueNet()
    trainer = Trainer(net, TrainConfig(
        device=device, lr=args.lr, weight_decay=1e-4, t_max=200, min_lr=1e-5,
    ))
    t1 = time.time()
    losses = trainer.train_steps(samples, args.steps, args.batch)
    print(f"trained ({time.time()-t1:.0f}s): policy={losses['policy']:.4f} "
          f"value={losses['value']:.4f}")
    torch.save(net.state_dict(), args.out)
    print(f"saved: {args.out}")

    wg, mg, hg = eval_greedy(net, n_games=6)
    print(f"greedy policy vs heuristic: win={wg:.0%} vp={mg:.1f}/{hg:.1f}")


if __name__ == "__main__":
    main()
