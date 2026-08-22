"""Bootstrap warm-start: imitate the heuristic to get a playable policy fast.

The heuristic plays self-play games (no MCTS needed — fast); each move records
the state, a bounded scored teacher-candidate shortlist and the game's
normalized final VP (value target). The network is trained on this imitation
data, then its MCTS is evaluated vs the heuristic.

This is a standard warm-start before pure AlphaZero self-play (which needs
thousands of iterations to bootstrap from random). Run from the repo root:

    PYTHONPATH=src/ai ./src/engine/.venv/Scripts/python.exe src/ai/bootstrap_imitation.py
"""

from __future__ import annotations

import argparse
import os
import pickle
import shutil
import tempfile
import time

from brass_ai.selfplay import generate_imitation_sample_shards


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=1000)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--eval-games", type=int, default=20)
    ap.add_argument("--eval-sims", type=int, default=60)
    ap.add_argument("--ckpt", type=str, default="checkpoints/bootstrap.pt")
    ap.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1),
                    help="heuristic-game worker processes (use 1 for serial)")
    ap.add_argument("--min-avg-vp", type=float, default=None,
                    help="only keep games whose mean final VP is strictly above this value")
    ap.add_argument("--min-vp", type=float, default=None,
                    help="only keep games whose lowest final VP is strictly above this value")
    ap.add_argument("--max-attempts", type=int, default=None,
                    help="candidate-game cap when a VP quality filter is enabled (default: 10x --games)")
    args = ap.parse_args()

    # Windows spawn re-imports this script in every imitation worker.  Keep
    # CUDA Torch and MCTS imports here so workers do not load GPU DLLs.
    import torch
    from brass_ai.evaluate import benchmark_mcts_vs_heuristic
    from brass_ai.hierarchical_policy import ACTION_FEATURE_DIM, ACTION_FEATURE_SCHEMA_VERSION
    from brass_ai.net import PolicyValueNet
    from brass_ai.rust_mcts import RustISMCTS, RustMCTSConfig
    from brass_ai.train import TrainConfig, Trainer, evaluate_policy

    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    t0 = time.time()
    os.makedirs("checkpoints", exist_ok=True)
    sample_dir = tempfile.mkdtemp(prefix="bootstrap-imitation-", dir="checkpoints")
    try:
        shards = generate_imitation_sample_shards(
            args.games, sample_dir,
            workers=args.workers,
            min_avg_vp=args.min_avg_vp,
            min_vp=args.min_vp,
            max_attempts=args.max_attempts,
        )
        print(f"generated imitation shards for {args.games} heuristic games "
              f"({time.time()-t0:.0f}s)")

        net = PolicyValueNet()
        trainer = Trainer(net, TrainConfig(
            device=device, epochs=1, batch_size=args.batch, lr=args.lr,
        ))
        t1 = time.time()
        losses = []
        total_samples = 0
        for epoch in range(args.epochs):
            print(f"epoch {epoch+1}/{args.epochs} ... \n")
            for shard in shards:
                with open(shard, "rb") as f:
                    shard_samples = pickle.load(f)
                total_samples += len(shard_samples) if epoch == 0 else 0
                losses.extend(trainer.train_one_epoch(shard_samples))
                del shard_samples
            trainer.scheduler.step()
            trainer.epoch_count += 1
        mean_losses = {k: sum(x[k] for x in losses) / len(losses) for k in losses[0]}
        metrics = {}
        metric_weight = 0
        for shard in shards:
            with open(shard, "rb") as f:
                shard_samples = pickle.load(f)
            shard_metrics = evaluate_policy(net, shard_samples, device)
            weight = len(shard_samples)
            metric_weight += weight
            for key, value in shard_metrics.items():
                metrics[key] = metrics.get(key, 0.0) + value * weight
            del shard_samples
        metrics = {k: v / metric_weight for k, v in metrics.items()}
        print(f"trained {total_samples} samples ({time.time()-t1:.0f}s): "
              f"policy={mean_losses['policy']:.3f} value={mean_losses['value']:.3f} "
              f"top1={metrics['policy_top1']:.1%} "
              f"top3={metrics['policy_top3']:.1%} "
              f"top5={metrics['policy_top5']:.1%} "
              f"type_top1={metrics['action_type_top1']:.1%} "
              f"entropy={metrics['policy_entropy']:.2f} "
              f"candidates={metrics['candidate_count_mean']:.1f}"
              f"/p95={metrics['candidate_count_p95']:.0f}")

        os.makedirs(os.path.dirname(args.ckpt) or ".", exist_ok=True)
        torch.save({
            "model": net.state_dict(),
            "action_feature_dim": ACTION_FEATURE_DIM,
            "action_feature_schema_version": ACTION_FEATURE_SCHEMA_VERSION,
        }, args.ckpt)

        mcts = RustISMCTS(net, RustMCTSConfig(c_puct=2.5, max_depth=10, device=device))
        result = benchmark_mcts_vs_heuristic(
            mcts, args.eval_sims, args.eval_games
        )
        print(f"MCTS(bootstrap net) vs heuristic: win_rate={result['win_rate']:.0%} "
              f"(mcts_vp={result['mcts_mean']:.1f} "
              f"vs heuristic_vp={result['base_mean']:.1f})")
        print(f"checkpoint saved: {args.ckpt}")
    finally:
        shutil.rmtree(sample_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
