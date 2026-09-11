"""Top-level self-play training entry point.

Warm-starts from an imitation checkpoint (the heuristic bootstrap) and then
keeps improving by playing itself. Run from the repo root:

    ./.venv/Scripts/python.exe python/selfplay_train.py --init-from checkpoints/bootstrap-0909-20000.pt

Smoke (a few minutes on CPU, proves the whole chain works end to end):

    ./.venv/Scripts/python.exe python/selfplay_train.py --iterations 2 \
        --games-per-iter 2 --workers 1 --sims 8 --train-samples 512 \
        --eval-every 1 --eval-games 2 --eval-sims 8 --heuristic-eval-games 2 \
        --ckpt-dir checkpoints/selfplay-smoke

Every iteration writes `latest.pt` (full Trainer state: model + optimizer +
scheduler + scaler + schema, loadable by `--resume` and by
`brass_ai.replay_worker --ckpt`), appends a line to `metrics.jsonl`, and on
promotion refreshes `best.pt`. Checkpoints are written to a temp file and
atomically renamed, so an interrupted run never leaves a half-written archive.
Ctrl+C stops after the current iteration and still leaves `latest.pt` valid.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path


def _load_model_state(path: Path, device: str) -> dict:
    """Accept either a Trainer checkpoint or a raw `model.state_dict()`."""
    import torch

    payload = torch.load(path, map_location=device)
    if isinstance(payload, dict) and "model" in payload:
        return payload["model"]
    if isinstance(payload, dict):
        return payload
    raise ValueError(f"unsupported checkpoint payload in {path}")


def _atomic_save(payload, path: Path) -> None:
    import torch

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
    parser.add_argument("--ckpt-dir", type=Path, default=Path("checkpoints/selfplay"))
    parser.add_argument("--init-from", type=Path,
                        help="warm start model (e.g. the imitation bootstrap checkpoint)")
    parser.add_argument("--resume", action="store_true",
                        help="continue from <ckpt-dir>/latest.pt")
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--games-per-iter", type=int, default=16)
    parser.add_argument("--sims", type=int, default=128)
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1),
                        help="self-play actor processes; 1 runs in-process")
    parser.add_argument("--max-moves", type=int, default=600)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if _cuda_available() else "cpu")
    # Matchmaking / opponent pool.
    parser.add_argument("--mm-prob", type=float, default=0.25)
    parser.add_argument("--pool-size", type=int, default=6)
    parser.add_argument("--heuristic-opponent-prob", type=float, default=0.25,
                        help="probability an opponent seat is played by the heuristic AI directly (breaks collusion, default: 0.25)")
    # Replay window and training budget.
    parser.add_argument("--buffer-samples", type=int, default=400_000)
    parser.add_argument("--buffer-iterations", type=int, default=20)
    parser.add_argument("--recent-fraction", type=float, default=0.75)
    parser.add_argument("--recent-iterations", type=int, default=4)
    parser.add_argument("--train-samples", type=int, default=40_000)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max-candidate-batch", type=int, default=65_536)
    parser.add_argument("--materialize-workers", type=int, default=min(8, os.cpu_count() or 1))
    # Search.
    parser.add_argument("--c-puct", type=float, default=0.25,
                        help="PUCT exploration constant (scaled to VP_SCALE=50, default: 0.25)")
    parser.add_argument("--prior-top-k", type=int, default=32,
                        help="search only the K highest-prior moves with stratified category preservation (0 = search every legal move, default: 32)")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="initial move sampling temperature (default: 0.0 for greedy best-visit)")
    parser.add_argument("--temperature-warmup-moves", type=int, default=0,
                        help="moves to hold initial temperature before decay")
    parser.add_argument("--temperature-decay-moves", type=int, default=0,
                        help="moves over which temperature decays to temperature-final")
    parser.add_argument("--temperature-final", type=float, default=0.0,
                        help="final sampling temperature")
    parser.add_argument("--no-fpu", action="store_true",
                        help="treat unvisited children as worth 0 instead of the parent's value")
    parser.add_argument("--no-q-init", action="store_true",
                        help="estimate unvisited children with FPU instead of the network's Q(s,a)")
    parser.add_argument("--max-depth", type=int, default=10)
    parser.add_argument("--mcts-batch", type=int, default=64)
    parser.add_argument("--candidate-k", type=int, default=0,
                        help="0 expands every legal move (default); positive values use the heuristic shortlist")
    # Evaluation.
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--eval-games", type=int, default=40)
    parser.add_argument("--eval-sims", type=int, default=128)
    parser.add_argument("--heuristic-eval-games", type=int, default=20)
    parser.add_argument("--heuristic-eval-sims", type=int, default=128)
    parser.add_argument("--promote-winrate", type=float, default=0.55)
    args = parser.parse_args()

    if args.workers < 1:
        parser.error("--workers must be >= 1")
    if args.materialize_workers < 1:
        parser.error("--materialize-workers must be >= 1")

    import torch

    from brass_ai.net import PolicyValueNet
    from brass_ai.rust_mcts import RustMCTSConfig
    from brass_ai.selfplay import SelfPlayConfig
    from brass_ai.selfplay_loop import LoopConfig, run_selfplay, write_metrics
    from brass_ai.train import TrainConfig, Trainer

    torch.manual_seed(args.seed)
    torch.set_num_threads(max(1, (os.cpu_count() or 4) // max(args.workers, 1)))

    ckpt_dir: Path = args.ckpt_dir
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    latest = ckpt_dir / "latest.pt"
    best = ckpt_dir / "best.pt"
    meta_path = ckpt_dir / "latest.json"
    metrics_path = ckpt_dir / "metrics.jsonl"

    net = PolicyValueNet()
    trainer = Trainer(net, TrainConfig(
        device=args.device, epochs=1, batch_size=args.batch, lr=args.lr,
        max_candidate_batch=args.max_candidate_batch,
        materialize_workers=args.materialize_workers,
    ))

    start_iteration = 0
    best_state = None
    try:
        if args.resume:
            if not latest.is_file():
                raise SystemExit(f"--resume requires {latest}")
            trainer.load_state_dict(torch.load(latest, map_location=args.device))
            if meta_path.is_file():
                start_iteration = int(json.loads(meta_path.read_text())["iteration"]) + 1
            if best.is_file():
                best_state = _load_model_state(best, args.device)
            print(f"resumed at iteration {start_iteration} from {latest}")
        elif args.init_from is not None:
            if not args.init_from.is_file():
                raise SystemExit(f"--init-from does not exist: {args.init_from}")
            net.load_state_dict(_load_model_state(args.init_from, args.device))
            print(f"warm start from {args.init_from}")
        else:
            print("warning: no --init-from and no --resume; starting from random weights")

        cfg = LoopConfig(
            iterations=args.iterations,
            games_per_iter=args.games_per_iter,
            sims=args.sims,
            workers=args.workers,
            seed=args.seed,
            device=args.device,
            mcts=RustMCTSConfig(
                c_puct=args.c_puct, max_depth=args.max_depth,
                batch_size=args.mcts_batch, candidate_k=args.candidate_k,
                prior_top_k=args.prior_top_k, fpu=not args.no_fpu,
                q_init=not args.no_q_init,
                device=args.device,
            ),
            selfplay=SelfPlayConfig(
                max_moves=args.max_moves,
                temperature=args.temperature,
                temperature_warmup_moves=args.temperature_warmup_moves,
                temperature_decay_moves=args.temperature_decay_moves,
                temperature_final=args.temperature_final,
            ),
            mm_prob=args.mm_prob,
            pool_size=args.pool_size,
            heuristic_prob=args.heuristic_opponent_prob,
            max_buffer_samples=args.buffer_samples,
            max_buffer_iterations=args.buffer_iterations,
            recent_fraction=args.recent_fraction,
            recent_iterations=args.recent_iterations,
            train_samples=args.train_samples,
            train=trainer.cfg,
            eval_every=args.eval_every,
            eval_games=args.eval_games,
            eval_sims=args.eval_sims,
            heuristic_eval_games=args.heuristic_eval_games,
            heuristic_eval_sims=args.heuristic_eval_sims,
            promote_winrate=args.promote_winrate,
        )

        def on_iteration(stats, live_net, live_trainer) -> None:
            write_metrics(metrics_path, stats)
            payload = live_trainer.state_dict()
            payload["meta"] = {
                "type": "selfplay",
                "run_id": ckpt_dir.name,
                "iteration": stats.iteration,
                "parent": str(args.init_from) if args.init_from else ("resumed" if args.resume else "scratch"),
                "avg_vp": round(stats.avg_vp, 2),
                "winner_avg_vp": round(stats.winner_avg_vp, 2),
                "min_vp": stats.min_vp,
                "max_vp": stats.max_vp,
                "arena_winrate": round(stats.arena_winrate, 3),
                "heuristic_winrate": round(stats.heuristic_winrate, 3) if stats.heuristic_winrate is not None else None,
                "promoted": stats.promoted,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
            _atomic_save(payload, latest)
            meta_path.write_text(json.dumps({
                "iteration": stats.iteration,
                "samples": stats.samples,
                "buffer": stats.buffer,
                "trained": stats.trained,
                "avg_vp": stats.avg_vp,
                "winner_avg_vp": stats.winner_avg_vp,
                "arena_winrate": stats.arena_winrate,
                "heuristic_winrate": stats.heuristic_winrate,
                "promoted": stats.promoted,
            }, indent=2))
            if stats.promoted:
                _atomic_save(payload, best)
            loss = stats.losses
            vp_str = (
                f"vp {stats.avg_vp:.1f} (win {stats.winner_avg_vp:.1f} min {stats.min_vp:.0f} max {stats.max_vp:.0f})"
                if stats.games > 0 and stats.avg_vp > 0 else "vp --"
            )
            print(
                f"it {stats.iteration:>4}  samples {stats.samples:>5}  "
                f"buf {stats.buffer:>6}  "
                f"pol {loss.get('policy', float('nan')):.3f} "
                f"val {loss.get('value', float('nan')):.3f} "
                f"win {loss.get('winner', float('nan')):.3f} "
                f"q {loss.get('q', float('nan')):.3f}  "
                f"{vp_str}  "
                f"sp {stats.selfplay_sec:4.0f}s tr {stats.train_sec:3.0f}s ev {stats.eval_sec:3.0f}s  "
                f"arena {stats.arena_winrate:.0%} (lo {stats.arena_lower:.0%})  "
                f"heur {('%.0f%%' % (100 * stats.heuristic_winrate)) if stats.heuristic_winrate is not None else '--'}  "
                f"reuse {stats.rewritten_applies}/{stats.failed_applies}"
                + ("  PROMOTED" if stats.promoted else "")
            )

        try:
            run_selfplay(
                net, trainer, cfg,
                on_iteration=on_iteration,
                start_iteration=start_iteration,
                best_state=best_state,
            )
        except KeyboardInterrupt:
            print("\ninterrupted: latest.pt already holds the last completed iteration")
    finally:
        trainer.close()
    return 0


def _cuda_available() -> bool:
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:
        return False


if __name__ == "__main__":
    sys.exit(main())
