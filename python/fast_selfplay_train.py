"""Pure-policy vectorized self-play training loop (Zero-MCTS).

Eliminates MCTS tree search completely during self-play data generation by using
vectorized batched environments and direct policy sampling.

Usage:
    python python/fast_selfplay_train.py \
      --init-from checkpoints/v1/bootstrap/b2000.pt \
      --ckpt-dir checkpoints/v1/runs/fast_sp01 \
      --iterations 20 \
      --games-per-iter 32 \
      --env-count 16 \
      --temperature 0.8
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

from brass_ai.fast_policy import VectorizedSelfPlay, trajectory_to_samples
from brass_ai.net import PolicyValueNet
from brass_ai.rl_league import evaluate_vs_heuristic_teachers
from brass_ai.train import TrainConfig, Trainer


def _load_model_state(path: Path, device: str) -> dict:
    payload = torch.load(path, map_location=device)
    if isinstance(payload, dict) and "model" in payload:
        return payload["model"]
    if isinstance(payload, dict):
        return payload
    raise ValueError(f"unsupported checkpoint payload in {path}")


def _atomic_save(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="wb", prefix=f".{path.name}.", suffix=".tmp",
        dir=path.parent, delete=False,
    ) as handle:
        tmp_path = Path(handle.name)
    try:
        torch.save(payload, tmp_path)
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--ckpt-dir", type=Path, default=Path("checkpoints/fast_selfplay"))
    parser.add_argument("--init-from", type=Path,
                        help="warm start model (e.g. imitation bootstrap checkpoint)")
    parser.add_argument("--resume", action="store_true",
                        help="continue from <ckpt-dir>/latest.pt")
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--games-per-iter", type=int, default=32)
    parser.add_argument("--env-count", type=int, default=16,
                        help="concurrent environments batched for GPU inference")
    parser.add_argument("--temperature", type=float, default=0.8,
                        help="initial sampling temperature for policy rollout")
    parser.add_argument("--temperature-final", type=float, default=0.2,
                        help="final sampling temperature for late-game rollout (default: 0.2)")
    parser.add_argument("--temperature-warmup-moves", type=int, default=12,
                        help="moves to hold initial temperature before decay (default: 12)")
    parser.add_argument("--temperature-decay-moves", type=int, default=24,
                        help="moves over which temperature decays to temperature-final (default: 24)")
    parser.add_argument("--heuristic-prob", type=float, default=0.25,
                        help="probability of assigning opponent seats to Rust 120-VP heuristic teachers (default: 0.25)")
    parser.add_argument("--kl-lambda", type=float, default=0.05,
                        help="weight of KL divergence loss against anchor probabilities (default: 0.05, 0.0 to disable)")
    parser.add_argument("--min-vp-filter", type=float, default=30.0,
                        help="discard trajectories where any player final VP is below this threshold (default: 30.0)")
    parser.add_argument("--no-advantage", action="store_true",
                        help="disable advantage-weighted policy importance")
    parser.add_argument("--epochs-per-iter", type=int, default=1)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-every", type=int, default=5,
                        help="evaluate vs heuristic teachers every N iterations (0 to disable)")
    parser.add_argument("--eval-games", type=int, default=20,
                        help="number of games for teacher gatekeeper evaluation")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    args = parser.parse_args()
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.ckpt_dir / "metrics.jsonl"

    net = PolicyValueNet().to(args.device)
    anchor_net = PolicyValueNet().to(args.device)
    trainer = Trainer(net, TrainConfig(
        device=args.device,
        batch_size=args.batch,
        lr=args.lr,
        kl_lambda=args.kl_lambda,
    ))

    start_iteration = 0
    best_eval_vp = -1.0

    if args.resume:
        latest_path = args.ckpt_dir / "latest.pt"
        if not latest_path.is_file():
            sys.exit(f"error: --resume requested but {latest_path} does not exist")
        checkpoint = torch.load(latest_path, map_location=args.device)
        trainer.load_state_dict(checkpoint)
        start_iteration = checkpoint.get("iteration", 0) + 1
        best_eval_vp = checkpoint.get("best_eval_vp", -1.0)
        anchor_net.load_state_dict(net.state_dict())
        best_path = args.ckpt_dir / "best.pt"
        if best_path.is_file():
            anchor_weights = _load_model_state(best_path, args.device)
            anchor_net.load_state_dict(anchor_weights, strict=False)
            print(f"resumed anchor base from {best_path}")
        print(f"resumed from {latest_path} at iteration {start_iteration}")
    elif args.init_from:
        weights = _load_model_state(args.init_from, args.device)
        net.load_state_dict(weights, strict=False)
        anchor_net.load_state_dict(net.state_dict())
        print(f"initialized weights and anchor base from {args.init_from}")
    else:
        anchor_net.load_state_dict(net.state_dict())

    anchor_net.eval()

    print(f"Starting Vectorized Fast Self-Play Training on {args.device}")
    print(f"Config: {args.iterations} iters, {args.games_per_iter} games/iter, "
          f"{args.env_count} batched envs, temp={args.temperature}, "
          f"heuristic_prob={args.heuristic_prob}, kl_lambda={args.kl_lambda}, min_vp={args.min_vp_filter}")

    runner = VectorizedSelfPlay(
        net=net,
        env_count=args.env_count,
        device=args.device,
        temperature=args.temperature,
        temperature_final=args.temperature_final,
        temperature_warmup_moves=args.temperature_warmup_moves,
        temperature_decay_moves=args.temperature_decay_moves,
        heuristic_prob=args.heuristic_prob,
        anchor_net=anchor_net if args.kl_lambda > 0.0 else None,
    )

    for iteration in range(start_iteration, args.iterations):
        iter_seed = args.seed + iteration * 10_000
        t0 = time.time()
        print(f"\n--- Iteration {iteration + 1}/{args.iterations} ---")

        # 1. Rollout pure policy games
        trajectories = runner.run_games(start_seed=iter_seed, n_games=args.games_per_iter)
        gen_time = time.time() - t0

        all_vps = [float(vp) for t in trajectories for vp in t.vps]
        mean_vp = float(np.mean(all_vps)) if all_vps else 0.0
        min_vp = float(np.min(all_vps)) if all_vps else 0.0
        max_vp = float(np.max(all_vps)) if all_vps else 0.0

        samples = []
        for t in trajectories:
            samples.extend(trajectory_to_samples(
                t,
                min_vp_filter=args.min_vp_filter,
                use_advantage=not args.no_advantage,
            ))

        t1 = time.time()
        losses = []
        for ep in range(args.epochs_per_iter):
            label = f"it{iteration+1} e{ep+1}"
            losses.extend(trainer.train_one_epoch(samples, label))
        if losses:
            trainer.scheduler.step()
            trainer.epoch_count += args.epochs_per_iter
        train_time = time.time() - t1

        mean_loss = {k: float(sum(l[k] for l in losses) / len(losses)) for k in losses[0]} if losses else {}
        print(f"[Rollout] {len(trajectories)} games in {gen_time:.1f}s ({len(trajectories)/max(gen_time, 1e-4):.1f} games/s) | "
              f"VP avg={mean_vp:.1f} min={min_vp:.1f} max={max_vp:.1f}")
        print(f"[Train] {len(samples)} samples in {train_time:.1f}s (lr={trainer.current_lr():.2e}) | "
              f"policy={mean_loss.get('policy', 0.0):.3f} val={mean_loss.get('value', 0.0):.3f} "
              f"abs_vp={mean_loss.get('abs_vp', 0.0):.3f} kl={mean_loss.get('kl', 0.0):.4f}")

        # 2. Gatekeeper Evaluation
        eval_result = None
        if args.eval_every > 0 and (iteration + 1) % args.eval_every == 0:
            print(f"[Gatekeeper] Evaluating {args.eval_games} games vs 3 Rust heuristic teachers...")
            t_eval = time.time()
            res = evaluate_vs_heuristic_teachers(
                net=net,
                n_games=args.eval_games,
                candidate_seat=0,
                device=args.device,
                temperature=0.2,
            )
            eval_result = {
                "win_rate": res.win_rate,
                "candidate_avg_vp": res.candidate_avg_vp,
                "teacher_avg_vp": res.teacher_avg_vp,
                "passed": res.passed,
            }
            print(f"[Gatekeeper] Win rate: {res.win_rate:.1%} | Avg VP: {res.candidate_avg_vp:.1f} vs Teacher: {res.teacher_avg_vp:.1f} "
                  f"({'PASSED' if res.passed else 'FAILED'}) in {time.time()-t_eval:.1f}s")

            # Model progression & Anchor update:
            # Whenever candidate score improves (or baseline established), update best.pt
            # and roll anchor forward so KL divergence doesn't freeze learning.
            # If it also passes teacher gatekeeper (win_rate >= 35% & avg_vp >= 120),
            # it earns official Champion status.
            improved = (best_eval_vp < 0.0) or (res.candidate_avg_vp > best_eval_vp)
            if improved:
                old_best = best_eval_vp
                best_eval_vp = res.candidate_avg_vp
                _atomic_save(trainer.state_dict(), args.ckpt_dir / "best.pt")

                if args.kl_lambda > 0.0:
                    anchor_net.load_state_dict(net.state_dict())
                    runner.set_anchor_net(anchor_net)
                    anchor_msg = " and updated anchor base"
                else:
                    anchor_msg = ""

                if res.passed:
                    print(f"[*] OFFICIAL CHAMPION PROMOTED! Passed teacher gatekeeper (win_rate={res.win_rate:.1%}, "
                          f"Avg VP={best_eval_vp:.1f}){anchor_msg}. Saved to {args.ckpt_dir / 'best.pt'}.")
                elif old_best < 0.0:
                    print(f"[*] Established baseline evaluation: {best_eval_vp:.1f} VP (win_rate={res.win_rate:.1%}){anchor_msg}.")
                else:
                    print(f"[*] Score improved ({best_eval_vp:.1f} > {old_best:.1f}, win_rate={res.win_rate:.1%})! "
                          f"Saved new best to {args.ckpt_dir / 'best.pt'}{anchor_msg}.")
            else:
                print(f"[Gatekeeper] No improvement ({res.candidate_avg_vp:.1f} <= best {best_eval_vp:.1f}).")

        # 3. Save latest checkpoint
        latest_payload = trainer.state_dict()
        latest_payload["iteration"] = iteration
        latest_payload["best_eval_vp"] = best_eval_vp
        _atomic_save(latest_payload, args.ckpt_dir / "latest.pt")

        # 4. Log metrics
        log_entry = {
            "iteration": iteration,
            "games": len(trajectories),
            "samples": len(samples),
            "rollout_sec": gen_time,
            "train_sec": train_time,
            "vp_mean": mean_vp,
            "vp_min": min_vp,
            "vp_max": max_vp,
            "losses": mean_loss,
            "gatekeeper": eval_result,
        }
        with open(metrics_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry) + "\n")

    print(f"\nCompleted {args.iterations} iterations. Artifacts saved in {args.ckpt_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
