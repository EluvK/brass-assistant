"""Step 1: multi-process AlphaZero training loop (benchmark-gated + matchmaking).

Main process: persistent Trainer (AdamW + CosineAnnealingLR, t_max = total
iterations) + DECISION-GRADE evaluation gate using the reliable benchmark
(fixed seeds, 20 games, win-rate + median VP). Workers: SelfPlayPool (spawn)
plays self-play games in parallel and streams numpy samples back.

Matchmaking (anti-drift): with `--mm_prob`, each self-play game has one
rotating learner seat (current net) and opponent seats drawn from a pool of
historical nets (the starting checkpoint + every gate-accepted snapshot). This
anchors training against a stable reference instead of drifting vs its own
noisy value head (the observed failure mode of pure self-play).

Gate (replaces the noisy rolling-VP gate): every `--eval_every` iterations run
`benchmark_mcts_vs_heuristic` (fixed seeds 0..N-1, seat rotated). A candidate
is accepted iff its win rate is >= the stored best win rate (ties broken by
median VP); otherwise the best weights are restored. This is the ONLY reliable
accept/reject signal (per-game VP fluctuates +/-20, so <10 games is noise).

Run from the repo root (spawn requires the __main__ guard, present here):
    PYTHONPATH=src/ai ./src/engine/.venv/Scripts/python.exe src/ai/train_mp.py
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch
import brass_engine as be

from brass_ai.dataset import load_samples, save_samples
from brass_ai.evaluate import benchmark_mcts_vs_heuristic
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
    ap.add_argument("--train_steps", type=int, default=60,
                    help="gradient steps per iteration over the replay buffer")
    ap.add_argument("--replay_size", type=int, default=6000,
                    help="samples kept in the replay buffer (data reuse across iters)")
    ap.add_argument("--worker_device", type=str, default="cpu",
                    help="device for self-play workers (Rust MCTS batched inference)")
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--c_puct", type=float, default=2.5)
    ap.add_argument("--temp", type=float, default=0.7)
    ap.add_argument("--dirichlet_alpha", type=float, default=0.3)
    ap.add_argument("--dirichlet_weight", type=float, default=0.15)
    ap.add_argument("--mm_prob", type=float, default=0.5,
                    help="probability an opponent seat uses a historical net "
                         "(matchmaking; 0 disables)")
    ap.add_argument("--pool_size", type=int, default=4,
                    help="max historical nets kept in the matchmaking pool")
    ap.add_argument("--eval_every", type=int, default=3,
                    help="run the decision-grade benchmark every N iterations")
    ap.add_argument("--bench_sims", type=int, default=400,
                    help="sims used for the decision-grade benchmark gate")
    ap.add_argument("--bench_games", type=int, default=20,
                    help="games used for the decision-grade benchmark gate")
    ap.add_argument("--gate", action="store_true",
                    help="revert to the best benchmarked net on regression")
    ap.add_argument("--ckpt", type=str, default="checkpoints/bootstrap.pt")
    ap.add_argument("--resume", action="store_true",
                    help="resume optimizer+scheduler from a full trainer-state ckpt")
    ap.add_argument("--run_dir", type=str, default="runs/train_mp",
                    help="directory containing manifest, replay shards, metrics and checkpoints")
    ap.add_argument("--out", type=str, default=None,
                    help="optional checkpoint path; defaults to <run_dir>/checkpoints/latest.pt")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    data_dir = run_dir / "replay"
    checkpoint_dir = run_dir / "checkpoints"
    run_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(exist_ok=True)
    checkpoint_dir.mkdir(exist_ok=True)
    out_path = Path(args.out) if args.out else checkpoint_dir / "latest.pt"
    manifest_path = run_dir / "manifest.json"
    manifest = {
        "format": 1,
        "engine": {
            "policy_table_size": be.policy_table_size,
            "board": [be.BOARD_PLANES, be.BOARD_CELLS],
            "links": [be.LINK_PLANES, be.LINK_CELLS],
            "global_len": be.GLOBAL_LEN,
            "hand_len": be.HAND_LEN,
        },
        "args": vars(args),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    import multiprocessing as mp
    mp.set_start_method("spawn", force=True)

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device} | workers: {args.workers} | sims: {args.sims} | "
          f"iters: {args.iters} | mm_prob: {args.mm_prob}")

    net = PolicyValueNet()
    trainer = Trainer(net, TrainConfig(
        device=device, epochs=args.epochs, batch_size=args.batch, lr=args.lr,
        t_max=max(args.iters, 1),
    ))

    load_path = out_path if args.resume and out_path.exists() else Path(args.ckpt)
    if load_path.exists():
        sd = torch.load(load_path, map_location=device)
        model = sd["model"] if "model" in sd else sd
        net.load_state_dict(model)
        if args.resume and "model" in sd:
            trainer.load_state_dict(sd)
            print(f"resumed trainer state from {load_path} "
                  f"(epoch={trainer.epoch_count})")
        else:
            print(f"loaded weights from {load_path} (fresh optimizer, "
                  f"t_max={max(args.iters, 1)})")
    else:
        print(f"warning: {load_path} not found; starting from random weights")

    mcts_cfg = {
        "c_puct": args.c_puct,
        "dirichlet_alpha": args.dirichlet_alpha,
        "dirichlet_weight": args.dirichlet_weight,
    }

    # Matchmaking pool: start with the init checkpoint; add every accepted best.
    def _as_pool_weights(state_dict):
        return {k: v.detach().cpu().clone() for k, v in state_dict.items()}

    mm_pool: list = []
    if args.mm_prob > 0.0 and os.path.exists(args.ckpt):
        mm_pool.append(_as_pool_weights(net.state_dict()))

    replay = []
    if args.resume:
        for shard in sorted(data_dir.glob("iter-*.npz")):
            replay.extend(load_samples(shard))
        if len(replay) > args.replay_size:
            replay = replay[-args.replay_size:]
        print(f"restored {len(replay)} replay samples from {data_dir}")
    best_win, best_median = -1.0, -1.0
    best_state = None
    print(f"{'iter':>4} {'samples':>7} {'pol':>6} {'val':>6} {'lr':>8} "
          f"{'sp_sec':>7} {'games':>5} | {'bench':>24} {'status':>10}")
    with SelfPlayPool(n_workers=args.workers, device=args.worker_device) as pool:
        for it in range(args.start_iter, args.iters):
            t0 = time.time()
            samples, counts = pool.generate(
                net, args.games_per_worker, args.sims,
                seed=args.seed + it, mcts_cfg=mcts_cfg, temperature=args.temp,
                mm_pool=mm_pool if args.mm_prob > 0.0 else None,
                mm_prob=args.mm_prob,
            )
            sp_t = time.time() - t0
            n_games = len(counts)

            shard_path = data_dir / f"iter-{it:05d}.npz"
            written = save_samples(shard_path, samples)

            replay.extend(samples)
            if len(replay) > args.replay_size:
                replay = replay[-args.replay_size:]
            losses = trainer.train_steps(list(replay), args.train_steps, args.batch)
            trainer.step_lr()
            torch.save(trainer.state_dict(), out_path)

            line = (
                f"{it:>4} {len(samples):>7} {losses['policy']:>6.3f} "
                f"{losses['value']:>6.3f} {trainer.current_lr():>8.1e} "
                f"{sp_t:>7.0f} {n_games:>5}"
            )

            if it % args.eval_every == 0:
                r = benchmark_mcts_vs_heuristic(
                    net, args.bench_sims, args.bench_games, device,
                )
                line += (f" | w={r['win_rate']:.2f} "
                         f"m={r['mcts_mean']:.1f}/{r['mcts_median']:.1f} "
                         f"h={r['base_mean']:.1f}")
                status = "bench"
                if args.gate:
                    if (r["win_rate"] > best_win or (
                            r["win_rate"] == best_win and r["mcts_median"] > best_median)):
                        best_win = r["win_rate"]
                        best_median = r["mcts_median"]
                        best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
                        torch.save(trainer.state_dict(), out_path)
                        if args.mm_prob > 0.0:
                            mm_pool.append(_as_pool_weights(best_state))
                            if len(mm_pool) > args.pool_size:
                                mm_pool = mm_pool[-args.pool_size:]
                        status = "(new best)"
                    else:
                        net.load_state_dict(best_state)
                        status = "(reverted)"
                line += f" {status:>10}"
            print(line)
            metric = {
                "iteration": it,
                "samples": written,
                "replay_samples": len(replay),
                "selfplay_seconds": sp_t,
                "games": n_games,
                "losses": losses,
                "lr": trainer.current_lr(),
            }
            if it % args.eval_every == 0:
                metric["benchmark"] = r
            with (run_dir / "metrics.jsonl").open("a", encoding="utf-8") as f:
                f.write(json.dumps(metric) + "\n")

    if args.gate and best_state is not None:
        net.load_state_dict(best_state)
        torch.save(trainer.state_dict(), out_path)
        print(f"final weights reverted to best (win {best_win:.2f}, "
              f"median VP {best_median:.1f})")

    print(f"\nsaved trainer state: {out_path}")
    if best_state is not None:
        print(f"final benchmarked best: win={best_win:.2f} medianVP={best_median:.1f}")


if __name__ == "__main__":
    main()
