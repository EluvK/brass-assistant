"""Bootstrap warm-start: imitate the heuristic to get a playable policy fast.

The heuristic plays self-play games (no MCTS needed — fast); each move records
the state, a bounded scored teacher-candidate shortlist and the game's
normalized final VP (value target). The network is trained on this imitation
data, then its MCTS is evaluated vs the heuristic.

This is a standard warm-start before pure AlphaZero self-play (which needs
thousands of iterations to bootstrap from random). Run from the repo root:

    ./.venv/Scripts/python.exe python/bootstrap_imitation.py

For interruptible training, run one (or a few) epochs at a time.  Each
completed epoch atomically updates ``--ckpt`` with the model, optimizer and
scheduler state; restart with ``--resume`` to continue:

    ... bootstrap_imitation.py --epochs 1
    ... bootstrap_imitation.py --resume --epochs 1
"""

from __future__ import annotations

import argparse
import os
import pickle
import shutil
import tempfile
import time
from pathlib import Path

from brass_ai.selfplay import generate_imitation_sample_shards, materialize_samples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=1000)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--max-candidate-batch", type=int, default=131072,
                    help="maximum padded candidate rows per GPU batch (default: 131072)") # 24576, 32768
    ap.add_argument("--enable-policy-eval", action="store_true",
                    help="evaluate top-k policy metrics over every replay shard after training")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--eval-games", type=int, default=20)
    ap.add_argument("--eval-sims", type=int, default=60)
    ap.add_argument("--ckpt", type=str, default="checkpoints/bootstrap.pt")
    ap.add_argument("--resume", action="store_true",
                    help="resume model, optimizer and scheduler from --ckpt")
    ap.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1),
                    help="heuristic-game worker processes (use 1 for serial)")
    ap.add_argument("--materialize-workers", type=int, default=min(4, os.cpu_count() or 1),
                    help="parallel workers restoring full-legal snapshots during training")
    ap.add_argument("--min-avg-vp", type=float, default=None,
                    help="only keep games whose mean final VP is strictly above this value")
    ap.add_argument("--min-vp", type=float, default=None,
                    help="only keep games whose lowest final VP is strictly above this value")
    ap.add_argument("--max-attempts", type=int, default=None,
                    help="candidate-game cap when a VP quality filter is enabled (default: 10x --games)")
    ap.add_argument("--full-legal-candidates", action="store_true",
                    help="train on every legal candidate instead of the teacher shortlist")
    ap.add_argument("--sample-dir", type=Path,
                    help="reuse existing imitation-*.pkl shards; skips generation and never deletes this directory")
    ap.add_argument("--delete-samples-on-success", action="store_true",
                    help="delete the default --ckpt.imitation shard directory after a successful run")
    ap.add_argument("--mcts-full-legal", action="store_true",
                    help="benchmark with every legal candidate instead of the shortlist")
    args = ap.parse_args()
    if args.max_candidate_batch < 1:
        ap.error("--max-candidate-batch must be >= 1")
    if args.materialize_workers < 1:
        ap.error("--materialize-workers must be >= 1")

    # Windows spawn re-imports this script in every imitation worker.  Keep
    # CUDA Torch and MCTS imports here so workers do not load GPU DLLs.
    import torch
    from brass_ai.evaluate import benchmark_mcts_vs_heuristic
    from brass_ai.net import PolicyValueNet
    from brass_ai.rust_mcts import RustISMCTS, RustMCTSConfig
    from brass_ai.train import TrainConfig, Trainer, evaluate_policy

    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    t0 = time.time()
    os.makedirs("checkpoints", exist_ok=True)
    owns_sample_dir = args.sample_dir is None
    # A deterministic directory means an interrupted default run can resume
    # without asking the user to recover a generated tempfile path.
    sample_dir = (
        Path(f"{args.ckpt}.imitation") if owns_sample_dir else args.sample_dir
    )
    succeeded = False
    try:
        existing_shards = sorted(sample_dir.glob("imitation-*.pkl")) if sample_dir.is_dir() else []
        if existing_shards:
            shards = existing_shards
            print(f"reusing {len(shards)} imitation shards from: {sample_dir}")
        elif owns_sample_dir:
            shards = generate_imitation_sample_shards(
                args.games, sample_dir,
                workers=args.workers,
                min_avg_vp=args.min_avg_vp,
                min_vp=args.min_vp,
                max_attempts=args.max_attempts,
                full_legal_candidates=args.full_legal_candidates,
            )
            print(f"generated imitation shards for {args.games} heuristic games "
                  f"({time.time()-t0:.0f}s)")
        else:
            if not sample_dir.is_dir():
                raise ValueError(f"--sample-dir does not exist or is not a directory: {sample_dir}")
            shards = sorted(sample_dir.glob("imitation-*.pkl"))
            if not shards:
                raise ValueError(f"--sample-dir contains no imitation-*.pkl shards: {sample_dir}")
            print(f"reusing {len(shards)} imitation shards from: {sample_dir}")
        net = PolicyValueNet()
        trainer = Trainer(net, TrainConfig(
            device=device, epochs=1, batch_size=args.batch, lr=args.lr,
            max_candidate_batch=args.max_candidate_batch,
            materialize_workers=args.materialize_workers,
        ))
        ckpt_path = Path(args.ckpt)
        if args.resume:
            if not ckpt_path.is_file():
                raise ValueError(f"--resume requires an existing checkpoint: {ckpt_path}")
            checkpoint = torch.load(ckpt_path, map_location=device)
            required = {"model", "optimizer", "scheduler"}
            missing = required.difference(checkpoint) if isinstance(checkpoint, dict) else required
            if missing:
                raise ValueError(
                    f"checkpoint {ckpt_path} cannot resume training; missing "
                    f"trainer state: {', '.join(sorted(missing))}. "
                    "Start a new bootstrap run to create a resumable checkpoint."
                )
            trainer.load_state_dict(checkpoint)
            print(f"resumed trainer state from {ckpt_path} (completed epochs: {trainer.epoch_count})")

        def save_checkpoint() -> None:
            """Avoid leaving a half-written checkpoint after an interruption."""
            ckpt_path.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(
                mode="wb", prefix=f".{ckpt_path.name}.", suffix=".tmp",
                dir=ckpt_path.parent or Path("."), delete=False,
            ) as f:
                tmp_path = Path(f.name)
            try:
                torch.save(trainer.state_dict(), tmp_path)
                os.replace(tmp_path, ckpt_path)
            finally:
                if tmp_path.exists():
                    tmp_path.unlink()

        t1 = time.time()
        losses = []
        total_samples = 0
        for run_epoch in range(args.epochs):
            epoch = trainer.epoch_count + 1
            print(f"\nepoch {epoch} (this run {run_epoch+1}/{args.epochs}) ...")
            for shard_index, shard in enumerate(shards, start=1):
                with open(shard, "rb") as f:
                    shard_samples = pickle.load(f)
                total_samples += len(shard_samples) if run_epoch == 0 else 0
                progress_label = f"train e{epoch} s{shard_index}/{len(shards)}"
                # print(f"[{progress_label}] {len(shard_samples)} samples: {shard.name}")
                losses.extend(trainer.train_one_epoch(shard_samples, progress_label))
                del shard_samples
            trainer.scheduler.step()
            trainer.epoch_count += 1
            save_checkpoint()
            print(f"checkpoint updated after epoch {trainer.epoch_count}: {ckpt_path}")
        mean_losses = {k: sum(x[k] for x in losses) / len(losses) for k in losses[0]}
        print(f"trained {total_samples} samples ({time.time()-t1:.0f}s): "
              f"policy={mean_losses['policy']:.3f} value={mean_losses['value']:.3f}")
        if args.enable_policy_eval:
            metrics = {}
            metric_weight = 0
            for shard in shards:
                with open(shard, "rb") as f:
                    shard_samples = pickle.load(f)
                shard_samples = materialize_samples(shard_samples)
                shard_metrics = evaluate_policy(
                    net, shard_samples, device,
                    max_candidate_batch=args.max_candidate_batch,
                )
                weight = len(shard_samples)
                metric_weight += weight
                for key, value in shard_metrics.items():
                    metrics[key] = metrics.get(key, 0.0) + value * weight
                del shard_samples
            metrics = {k: v / metric_weight for k, v in metrics.items()}
            print(f"policy eval: top1={metrics['policy_top1']:.1%} "
                  f"top3={metrics['policy_top3']:.1%} "
                  f"top5={metrics['policy_top5']:.1%} "
                  f"type_top1={metrics['action_type_top1']:.1%} "
                  f"entropy={metrics['policy_entropy']:.2f} "
                  f"candidates={metrics['candidate_count_mean']:.1f}"
                  f"/p95={metrics['candidate_count_p95']:.0f}")

        mcts = RustISMCTS(net, RustMCTSConfig(
            c_puct=2.5, max_depth=10, device=device,
            candidate_k=0 if args.mcts_full_legal else 4,
        ))
        result = benchmark_mcts_vs_heuristic(
            mcts, args.eval_sims, args.eval_games
        )
        print(f"MCTS(bootstrap net) vs heuristic: win_rate={result['win_rate']:.0%} "
              f"(mcts_vp={result['mcts_mean']:.1f} "
              f"vs heuristic_vp={result['base_mean']:.1f})")
        print(f"checkpoint saved: {ckpt_path}")
        succeeded = True
    finally:
        if not owns_sample_dir:
            print(f"reused imitation shards preserved at: {sample_dir}")
        elif not succeeded:
            print(f"bootstrap failed; imitation shards preserved at: {sample_dir}")
        elif args.delete_samples_on_success:
            shutil.rmtree(sample_dir, ignore_errors=True)
            print(f"bootstrap succeeded; imitation shards deleted: {sample_dir}")
        else:
            print(f"imitation shards preserved for --resume at: {sample_dir}")



if __name__ == "__main__":
    main()
