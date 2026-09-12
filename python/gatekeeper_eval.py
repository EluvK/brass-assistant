"""Gatekeeper Arena Benchmark: Evaluate a checkpoint against 3 Rust Heuristic Teachers.

Usage:
    python python/gatekeeper_eval.py --ckpt checkpoints/v1/bootstrap/b10k.pt
    python python/gatekeeper_eval.py --ckpt checkpoints/v1/bootstrap/b10k.pt --games 40 --verbose
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

import brass_ai._engine as be
from brass_ai.fast_policy import FastPolicyPlayer
from brass_ai.net import PolicyValueNet
from brass_ai.rl_league import HeuristicRoundPlayer


def evaluate_checkpoint(
    ckpt_path: Path,
    games: int = 40,
    device: str | None = None,
    temperature: float = 0.2,
    greedy: bool = False,
    rotate_seats: bool = True,
    min_winrate: float = 0.35,
    min_avg_vp: float = 120.0,
    verbose: bool = False,
) -> bool:
    if not ckpt_path.is_file():
        print(f"Error: checkpoint file not found: {ckpt_path}", file=sys.stderr)
        return False

    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print("\n================== Gatekeeper Arena Benchmark ==================")
    print(f"  Checkpoint   : {ckpt_path.resolve()}")
    print(f"  Games        : {games} (Seat rotation: {rotate_seats})")
    print(f"  Device       : {dev}")
    print(f"  Temperature  : {temperature} (Greedy: {greedy})")
    print(f"  Pass Criteria: Win Rate >= {min_winrate*100:.1f}%, Avg VP >= {min_avg_vp:.1f}")
    print("================================================================")

    # Load model
    net = PolicyValueNet()
    try:
        raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        sd = raw["model"] if isinstance(raw, dict) and "model" in raw else raw
        net.load_state_dict(sd)
    except Exception as exc:
        print(f"Error loading checkpoint state dict: {exc}", file=sys.stderr)
        return False

    net.to(dev)
    player = FastPolicyPlayer(net, device=dev, temperature=temperature, greedy=greedy)

    wins = 0
    candidate_vps = []
    teacher_vps = []
    seat_records = {p: {"games": 0, "wins": 0, "vps": []} for p in range(4)}

    t0 = time.time()
    for g in range(1, games + 1):
        target_seat = (g - 1) % 4 if rotate_seats else 0
        state = be.GameState(seed=g, players=4)
        teachers = {p: HeuristicRoundPlayer() for p in range(4)}
        moves = 0

        while not state.game_over and moves < 600:
            moves += 1
            actor = state.current_player_id
            if actor == target_seat:
                canon, _, _, _ = player.step(state)
            else:
                canon = teachers[actor].step(state)
            state.apply_move(canon)

        vps = state.player_vps()
        ranking = list(state.final_ranking())
        is_winner = ranking[0] == target_seat
        if is_winner:
            wins += 1

        cand_vp = vps[target_seat]
        t_vps = [vps[i] for i in range(4) if i != target_seat]

        candidate_vps.append(cand_vp)
        teacher_vps.extend(t_vps)

        rec = seat_records[target_seat]
        rec["games"] += 1
        rec["wins"] += int(is_winner)
        rec["vps"].append(cand_vp)

        if verbose:
            win_sym = "[WIN]" if is_winner else "     "
            print(f"  Game {g:02d} {win_sym} | Seat P{target_seat} | Candidate: {cand_vp:3d} VP | "
                  f"Teachers: {t_vps} | Winner: P{ranking[0]}")
        else:
            cur_rate = wins / g
            cur_avg = np.mean(candidate_vps)
            print(f"\r  Progress: {g}/{games} games | Win Rate: {cur_rate*100:.1f}% | Cand Avg VP: {cur_avg:.1f}   ",
                  end="", flush=True)

    total_time = time.time() - t0
    if not verbose:
        print()

    win_rate = wins / games if games > 0 else 0.0
    cand_mean = float(np.mean(candidate_vps)) if candidate_vps else 0.0
    teacher_mean = float(np.mean(teacher_vps)) if teacher_vps else 0.0
    cand_max = int(np.max(candidate_vps)) if candidate_vps else 0
    cand_min = int(np.min(candidate_vps)) if candidate_vps else 0

    passed = (win_rate >= min_winrate) and (cand_mean >= min_avg_vp)

    print("----------------------------------------------------------------")
    print(f"  Total Time        : {total_time:.2f}s ({games/total_time:.1f} games/s)")
    print(f"  Overall Win Rate  : {win_rate*100:.1f}% ({wins}/{games}) [Baseline: 25.0%]")
    print(f"  Candidate Avg VP  : {cand_mean:.2f} (Range: [{cand_min} .. {cand_max}])")
    print(f"  Teacher Avg VP    : {teacher_mean:.2f}")
    print(f"  Score Differential: {cand_mean - teacher_mean:+.2f} VP")
    print("----------------------------------------------------------------")
    print("  Seat Breakdown for Candidate:")
    print("    Seat  Games  Win Rate   Avg VP")
    for p in range(4):
        r = seat_records[p]
        wr = (r["wins"] / r["games"] * 100) if r["games"] > 0 else 0.0
        avg = np.mean(r["vps"]) if r["games"] > 0 else 0.0
        print(f"      P{p}     {r['games']:2d}     {wr:5.1f}%    {avg:5.1f}")
    print("================================================================")
    if passed:
        print("  VERDICT: [PASSED] - Model meets promotion criteria!")
        print("  Action : Ready to be locked as Anchor Model for RL.")
    else:
        print("  VERDICT: [FAILED] - Model has not surpassed the 120 VP teachers.")
        print("  Action : Requires more imitation pretraining or hyperparameter tuning.")
    print("================================================================\n")

    return passed


def main() -> int:
    parser = argparse.ArgumentParser(description="Gatekeeper Arena Benchmark for Brass Birmingham AI.")
    parser.add_argument("--ckpt", type=Path, required=True, help="Path to checkpoint (.pt)")
    parser.add_argument("--games", type=int, default=40, help="Number of benchmark games (default: 40)")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda/cpu)")
    parser.add_argument("--temperature", type=float, default=0.2, help="Policy sampling temperature (default: 0.2)")
    parser.add_argument("--greedy", action="store_true", help="Select argmax action instead of sampling (default: False)")
    parser.add_argument("--no-rotate", action="store_true", help="Fix candidate to Seat 0 instead of rotating")
    parser.add_argument("--min-winrate", type=float, default=0.35, help="Pass win rate threshold (default: 0.35)")
    parser.add_argument("--min-avg-vp", type=float, default=120.0, help="Pass average VP threshold (default: 120.0)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Print per-game outcome details")
    args = parser.parse_args()

    passed = evaluate_checkpoint(
        ckpt_path=args.ckpt,
        games=args.games,
        device=args.device,
        temperature=args.temperature,
        greedy=args.greedy,
        rotate_seats=not args.no_rotate,
        min_winrate=args.min_winrate,
        min_avg_vp=args.min_avg_vp,
        verbose=args.verbose,
    )
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
