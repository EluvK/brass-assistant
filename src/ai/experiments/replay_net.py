"""Fast network replay: play full games with the trained net and print a
replay.rs-style per-move log for analysis.

Two policy modes for the MCTS seat:
  * greedy (default, --sims 0): network greedy policy (one forward per move,
    no search) — MILLISECONDS per move, the fast way to eyeball what the net
    learned / where it misplays.
  * mcts (--sims N>0): Rust network-guided ISMCTS with N simulations per move
    (slower, but shows what the search actually recommends).

Other seats play the heuristic. Use --seat to fix which seat the net takes
(default rotates with seed: seat = seed % 4, matching the benchmark harness).

Output per move: round/era header, the canonical move, the engine's Chinese
result summary, the mover's money/VP/income and board snapshot (tiles/flips/
market). Era boundaries print the score split and tile inventory.

Run from repo root:
    PYTHONPATH=src/ai ./src/engine/.venv/Scripts/python.exe \\
        src/ai/experiments/replay_net.py --ckpt checkpoints/new_best.pt --seed 14 --sims 0
    # with MCTS search (slower):
        ... --ckpt checkpoints/new_best.pt --seed 0 --sims 200 --seat 2
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import torch

import brass_engine as be
from brass_ai.net import PolicyValueNet
from brass_ai.rust_mcts import RustISMCTS, RustMCTSConfig
from brass_ai.evaluate import heuristic_policy

_SAFE_PRINT = None  # set in main() to an encoding-safe writer


def out(s: str = ""):
    _SAFE_PRINT(s)


def tile_stats(state) -> dict:
    """Per-player (occupied, flipped) tile counts decoded from the board tensor."""
    board, *_ = state.state_to_tensor()
    b = np.asarray(board)
    occ = b[0] > 0.5
    flipped = b[11] > 0.5
    res = {}
    for p in range(4):
        own = b[1 + p] > 0.5
        res[p] = (int((own & occ).sum()), int((own & occ & flipped).sum()))
    return res


def board_line(state) -> str:
    t = tile_stats(state)
    occ = sum(v[0] for v in t.values())
    flip = sum(v[1] for v in t.values())
    return f"tiles {occ} (flipped {flip}) | per-P {t}"


def era_line(state, era_name: str) -> str:
    vps = state.player_vps()
    t = tile_stats(state)
    return f"--- {era_name} era end | VP {vps} | tiles(occ,flip) {t}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--seat", type=int, default=None,
                    help="seat for the net (default: seed % 4, like benchmark)")
    ap.add_argument("--sims", type=int, default=0,
                    help="MCTS sims per move; 0 = greedy (fast, no search)")
    ap.add_argument("--c_puct", type=float, default=2.5)
    ap.add_argument("--max_depth", type=int, default=10)
    ap.add_argument("--max_moves", type=int, default=600)
    ap.add_argument("--quiet", action="store_true",
                    help="skip per-move vps/board lines (only header + final)")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--multi", type=str, default=None,
                    help="comma list or range 'a-b' of seeds to replay in ONE "
                         "process (amortizes ~3s torch startup over many games)")
    args = ap.parse_args()

    global _SAFE_PRINT
    _buf = getattr(sys.stdout, "buffer", sys.stdout)

    def _write(s: str):
        _buf.write((s + "\n").encode("utf-8", errors="replace"))

    _SAFE_PRINT = _write

    net = PolicyValueNet()
    sd = torch.load(args.ckpt, map_location="cpu")
    net.load_state_dict(sd["model"] if "model" in sd else sd)
    net.eval()
    rm = RustISMCTS(net, RustMCTSConfig(
        c_puct=args.c_puct, max_depth=args.max_depth, device=args.device,
    ))

    def net_pol(state):
        if args.sims <= 0:
            batch = _encode_state(state, net)
            type_logits, goal_logits, _ = net.policy_value(batch)
            merged = net.merge_logits(type_logits, goal_logits)[0].detach().cpu().numpy()
            # legal_mask() is ~5x cheaper than legal_moves() (no describe strings)
            mask = np.zeros(len(merged), dtype=bool)
            for s in state.legal_mask():
                mask[s] = True
            slot = int(np.argmax(np.where(mask, merged, -1e9)))
            for s, canon, _ in state.legal_moves():
                if s == slot:
                    return canon
            return None
        return rm.search(state, args.sims, add_root_noise=False).best

    def replay_one(seed: int):
        seat = args.seat if args.seat is not None else seed % 4
        policies = [heuristic_policy] * 4
        policies[seat] = net_pol
        state = be.GameState(seed=seed, players=4)
        out("=" * 72)
        out(f"Replay net={args.ckpt} seed={seed} seat={seat} "
            f"mode={'greedy' if args.sims <= 0 else f'mcts sims={args.sims}'}")
        out("=" * 72)
        out(f"initial vps={state.player_vps()}")

        prev_era = state.era
        moves = 0
        while not state.game_over and moves < args.max_moves:
            moves += 1
            if state.era != prev_era:
                out()
                out(era_line(state, "Canal" if prev_era == 0 else "Rail"))
                prev_era = state.era
                out()

            pid = state.current_player_id
            out(f"[m{moves:>3} | era {'C' if state.era==0 else 'R'} r{state.round:>2} | P{pid}]")
            canon = policies[pid](state)
            if canon is None:
                lm = state.legal_moves()
                if not lm:
                    out("  NO LEGAL MOVES — stopping")
                    break
                canon = lm[0][1]
            try:
                summary = state.apply_move(canon)
            except ValueError:
                lm = state.legal_moves()
                if not lm:
                    break
                try:
                    summary = state.apply_move(lm[0][1])
                except ValueError:
                    out("  !! move failed twice — stopping")
                    break
            out(f"  -> {summary}")
            if not args.quiet:
                out(f"  vps={state.player_vps()} | {board_line(state)}")

        if not state.game_over:
            out()
            out(era_line(state, "Final"))
        vps = state.player_vps()
        ranking = state.final_ranking()
        out()
        out("=" * 72)
        out(f"FINAL vps={vps} ranking={ranking} | {board_line(state)}")
        out(f"winner = P{ranking[0]}")
        out()
        return vps[seat]

    if args.multi:
        if "-" in args.multi:
            a, b = args.multi.split("-")
            seeds = list(range(int(a), int(b) + 1))
        else:
            seeds = [int(x) for x in args.multi.split(",")]
        results = []
        for seed in seeds:
            results.append((seed, replay_one(seed)))
        out("### MULTI-SEED SUMMARY ###")
        for seed, vp in results:
            out(f"seed {seed:>3}: net-seat VP {vp}")
    else:
        replay_one(args.seed)


def _encode_state(state, net):
    from brass_ai import build_input
    b = build_input.encode_state(state)
    dev = next(net.parameters()).device
    return {k: v.to(dev) for k, v in b.items()}


if __name__ == "__main__":
    main()
