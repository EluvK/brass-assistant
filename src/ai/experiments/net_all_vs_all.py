"""Net-vs-net stats: all 4 seats play the SAME trained network (greedy by
default) over a seed range, reporting VP distributions + win rates per seat.

This is the "how strong is the net against itself" measure. Greedy mode is
fast (~0.65s/game amortized); use --sims N for MCTS (much slower).

Seat bias note: turn order is seeded, so each seat sees a mix of first/second
hands across seeds; aggregate per-seat stats cancel most ordering effects.

Run from repo root:
    PYTHONPATH=src/ai ./src/engine/.venv/Scripts/python.exe \\
        src/ai/experiments/net_all_vs_all.py --ckpt checkpoints/step5_best.pt --start 1 --end 500 --sims 0
    # with MCTS (slow, try --end 20 first):
        ... --ckpt checkpoints/step5_best.pt --start 1 --end 20 --sims 100
"""

from __future__ import annotations

import argparse
import os
import sys
import time

# Reconfigure stdio for UTF-8 (Windows GBK consoles can't encode £/中文).
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

import numpy as np
import torch

import brass_engine as be
from brass_ai.net import PolicyValueNet
from brass_ai.hierarchical_policy import encode_legal_candidates
from brass_ai.rust_mcts import RustISMCTS, RustMCTSConfig
from brass_ai.progress import Progress


def _encode_state(state, net):
    from brass_ai import build_input
    b = build_input.encode_state(state)
    dev = next(net.parameters()).device
    return {k: v.to(dev) for k, v in b.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--start", type=int, default=1)
    ap.add_argument("--end", type=int, default=500, help="inclusive")
    ap.add_argument("--sims", type=int, default=0,
                    help="MCTS sims per move; 0 = greedy (fast, no search)")
    ap.add_argument("--c_puct", type=float, default=2.5)
    ap.add_argument("--max_depth", type=int, default=10)
    ap.add_argument("--max_moves", type=int, default=600)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--out", type=str, default=None,
                    help="write per-seed CSV to this path (default: none)")
    args = ap.parse_args()

    net = PolicyValueNet()
    sd = torch.load(args.ckpt, map_location="cpu")
    net.load_state_dict(sd["model"] if "model" in sd else sd)
    net.eval()
    rm = RustISMCTS(net, RustMCTSConfig(
        c_puct=args.c_puct, max_depth=args.max_depth, device=args.device,
    ))

    def greedy_pol(state):
        batch = _encode_state(state, net)
        canonicals, candidates = encode_legal_candidates(state)
        candidates = candidates.unsqueeze(0).to(next(net.parameters()).device)
        mask = torch.ones((1, len(canonicals)), dtype=torch.bool, device=candidates.device)
        with torch.no_grad():
            out = net.policy_value(batch, candidates, mask)
        return canonicals[int(out["candidate_logits"].argmax(dim=1).item())]

    def mcts_pol(state):
        return rm.search(state, args.sims, add_root_noise=False).best

    pol = greedy_pol if args.sims <= 0 else mcts_pol
    policies = [pol] * 4

    n_games = args.end - args.start + 1
    all_vps = np.zeros((n_games, 4), dtype=np.int32)
    wins = np.zeros(4, dtype=np.int64)
    prog = Progress(n_games, f"net-vs-net {'greedy' if args.sims <= 0 else f'sims={args.sims}'}")
    t0 = time.time()
    for gi, seed in enumerate(range(args.start, args.end + 1)):
        state = be.GameState(seed=seed, players=4)
        moves = 0
        while not state.game_over and moves < args.max_moves:
            moves += 1
            pid = state.current_player_id
            canon = policies[pid](state)
            if canon is None:
                lm = state.legal_moves()
                if not lm:
                    break
                canon = lm[0][1]
            ok = state.apply_move_raw(canon)[1]
            if not ok:
                lm = state.legal_moves()
                if lm:
                    ok = state.apply_move_raw(lm[0][1])[1]
                if not ok:
                    break
            tr = state.advance_turn_raw()
            if tr == "end_canal_era":
                state.finish_canal_era()
            elif tr == "end_game":
                state.finish_game()
        if not state.game_over:
            state.finish_game()
        vps = state.player_vps()
        ranking = state.final_ranking()
        all_vps[gi] = vps
        wins[ranking[0]] += 1
        prog.update(gi + 1)
    prog.done()

    elapsed = time.time() - t0
    print(f"\n{args.ckpt} | net-vs-net ({n_games} games, "
          f"{'greedy' if args.sims <= 0 else f'sims={args.sims}'}) | {elapsed:.0f}s")
    print("per-seat VP (mean / median / min / max):")
    for p in range(4):
        v = all_vps[:, p]
        print(f"  seat{p}: mean={v.mean():6.1f} median={np.median(v):6.1f} "
              f"min={v.min():4d} max={v.max():4d}")
    tot = all_vps.sum(axis=1)
    print(f"sum-per-game: mean={tot.mean():.1f} median={np.median(tot):.0f} "
          f"min={tot.min()} max={tot.max()}")
    print(f"wins per seat: {wins.tolist()} ({wins / n_games * 100})")

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            f.write("seed,p0,p1,p2,p3,winner\n")
            for gi, seed in enumerate(range(args.start, args.end + 1)):
                r = all_vps[gi]
                w = int(np.argmax(r))
                f.write(f"{seed},{r[0]},{r[1]},{r[2]},{r[3]},{w}\n")
        print(f"csv written: {args.out}")


if __name__ == "__main__":
    main()
