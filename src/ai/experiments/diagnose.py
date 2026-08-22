"""Diagnose candidate-policy MCTS games with era and action statistics.

The script plays one MCTS seat against the engine heuristic and records a
per-move action trace, era-end score splits and flipped-tile counts:

  1. Does the network sell enough tiles?
  2. In which era does a weak game diverge?
  3. Which action classes dominate the trace?

Run from repo root:
    PYTHONPATH=src/ai ./src/engine/.venv/Scripts/python.exe \\
        src/ai/experiments/diagnose.py --ckpt checkpoints/bootstrap.pt --sims 400
"""

from __future__ import annotations

import argparse

import numpy as np
import torch

import brass_engine as be
from brass_ai.net import PolicyValueNet
from brass_ai.rust_mcts import RustISMCTS, RustMCTSConfig
from brass_ai.evaluate import heuristic_policy

ACTION_TYPE = {
    "Build": "Build", "Network": "Net", "NetDouble": "Net2", "Develop": "Dev",
    "FreeDevelop": "Dev", "Sell": "Sell", "Loan": "Loan", "Scout": "Scout", "Pass": "Pass",
}


def action_type(canon: str) -> str:
    return next((label for prefix, label in ACTION_TYPE.items() if canon.startswith(prefix)), "?")


def tile_stats(state) -> dict:
    """Per-player (occupied, flipped) tile counts decoded from the board tensor."""
    board, *_ = state.state_to_tensor()
    b = np.asarray(board)  # (17, 49)
    occ = b[0] > 0.5
    flipped = b[11] > 0.5
    out = {}
    for p in range(4):
        own = b[1 + p] > 0.5
        out[p] = (int((own & occ).sum()), int((own & occ & flipped).sum()))
    return out


def diagnose_one(rm, seed: int, sims: int, max_moves: int = 600, trace: bool = True):
    seat = seed % 4
    policies = [heuristic_policy] * 4
    policies[seat] = lambda s: rm.search(s, sims, add_root_noise=False).best
    state = be.GameState(seed=seed, players=4)

    trace_rows = []
    moves = 0
    era = state.era
    canal_vps = None
    while not state.game_over and moves < max_moves:
        moves += 1
        pid = state.current_player_id
        canon = policies[pid](state)
        if canon is None:
            legal = state.legal_moves()
            if not legal:
                break
            canon = legal[0][1]
        try:
            state.apply_move(canon)
        except ValueError:
            legal = state.legal_moves()
            if not legal:
                break
            try:
                state.apply_move(legal[0][1])
            except ValueError:
                break
        if pid == seat and trace:
            trace_rows.append({
                "mv": moves, "act": action_type(canon), "era": state.era,
                "vp": state.player_vps()[seat],
                "money": state.current_player_money,
            })
        if state.era != era:
            # era boundary: canal -> rail. VP now includes canal-era scoring.
            if era == 0 and canal_vps is None:
                canal_vps = state.player_vps()
            era = state.era

    vps = state.player_vps()
    ranking = state.final_ranking()
    return {
        "seed": seed, "seat": seat, "vps": vps, "rank": ranking[0],
        "canal_vps": canal_vps, "tiles": tile_stats(state),
        "trace": trace_rows, "moves": moves,
    }


def summarize(rows):
    seat_rank_first = sum(1 for r in rows if r["rank"] == r["seat"])
    print(f"  MCTS seat ranks 1st in {seat_rank_first}/{len(rows)} games\n")

    # Era splits for the MCTS seat
    print("  seed  seat  final  canal  rail  rank | tiles MCTS(occ,flip)  [build sell dev loan net net2 pass scout]")
    for r in rows:
        s = r["seat"]
        cv = r["canal_vps"][s] if r["canal_vps"] else -1
        rail = r["vps"][s] - (cv if cv >= 0 else 0)
        act = [t["act"] for t in r["trace"]]
        c = {k: act.count(k) for k in ACTION_TYPE.values()}
        occ, flip = r["tiles"][s]
        print(
            f"  {r['seed']:>4}  {s:>4}  {r['vps'][s]:>5}  {cv:>4}  {rail:>4}  "
            f"{r['rank']:>4} | {occ:>2}/{flip:<2}          "
            f"[b{c['Build']} s{c['Sell']} d{c['Dev']} l{c['Loan']} "
            f"n{c['Net']} n2{c['Net2']} p{c['Pass']} sc{c['Scout']}]"
        )

    # Era-split of actions for the MCTS seat: how many actions happened in canal
    # vs rail era, and how many of the seat's sells/flips landed in each era.
    print("\n  era action split for MCTS seat (canal / rail):")
    print("  seed  final  acts  [b,s,d,l,n,n2,p]c / [b,s,d,l,n,n2,p]r")
    for r in rows:
        s = r["seat"]
        cact = [t["act"] for t in r["trace"] if t["era"] == 0]
        ract = [t["act"] for t in r["trace"] if t["era"] == 1]
        def cnt(a):
            return {k: a.count(k) for k in ACTION_TYPE.values()}
        cc, rc = cnt(cact), cnt(ract)
        def fmt(c):
            return f"b{c['Build']}s{c['Sell']}d{c['Dev']}l{c['Loan']}n{c['Net']}n2{c['Net2']}p{c['Pass']}"
        print(
            f"  {r['seed']:>4}  {r['vps'][s]:>5}  {len(r['trace']):>4}  "
            f"[{fmt(cc)}]c / [{fmt(rc)}]r"
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--sims", type=int, default=400)
    ap.add_argument("--seeds", type=str, default="0-19",
                    help="range '0-19' or comma list")
    ap.add_argument("--device", type=str, default="cuda")
    args = ap.parse_args()

    if "-" in args.seeds:
        a, b = args.seeds.split("-")
        seeds = list(range(int(a), int(b) + 1))
    else:
        seeds = [int(x) for x in args.seeds.split(",")]

    net = PolicyValueNet()
    sd = torch.load(args.ckpt, map_location="cpu")
    net.load_state_dict(sd["model"] if "model" in sd else sd)
    net.eval()
    rm = RustISMCTS(net, RustMCTSConfig(device=args.device))

    rows = []
    for seed in seeds:
        rows.append(diagnose_one(rm, seed, args.sims))
    summarize(rows)

    print("\n--- MCTS seat trace for disaster games (final < 60) ---")
    for r in rows:
        if r["vps"][r["seat"]] < 60:
            print(f"\nseed {r['seed']} (seat {r['seat']}, final {r['vps'][r['seat']]}):")
            for t in r["trace"]:
                print(f"    m{t['mv']:>3} {t['act']:<5} vp={t['vp']:>3} money={t['money']:>3}")
            print(f"    tiles: {r['tiles']}")


if __name__ == "__main__":
    main()
