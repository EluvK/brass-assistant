"""Network replay with replay.rs-grade human-readable Chinese logs.

Drives full games where one seat is the trained network (greedy or Rust
network-guided ISMCTS) and the others play the engine's native heuristic,
logging every move the same way `cargo run --bin replay` does — the shared
formatting lives in Rust `brass_engine::replay_fmt` and is exposed via the
stepwise PyO3 bindings (`apply_move_raw`/`advance_turn_raw`/`replay_*`).

The log goes to a UTF-8 file under `logs/` (Windows consoles are GBK; writing
to a file avoids the encoding trap and gives a persistent record). Use
`--out -` to print to stdout instead.

Modes for the net seat:
  * greedy (default, --sims 0): one forward per move, no search — fast.
  * mcts (--sims N>0): Rust ISMCTS with N simulations per move.

Usage (from repo root):
    PYTHONPATH=src/ai ./src/engine/.venv/Scripts/python.exe \\
        src/ai/experiments/replay_net.py --ckpt checkpoints/step5_best.pt --seed 14
    ... --ckpt checkpoints/step5_best.pt --seed 0 --sims 200 --seat 2
    ... --ckpt checkpoints/step5_best.pt --multi 0-19 --sims 0 --quiet
    # net vs net (all 4 seats the network):
    ... --ckpt checkpoints/step5_best.pt --seed 14 --all-net
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict

# Windows consoles default to GBK, which cannot encode the £/Chinese log text.
# Reconfigure stdio to UTF-8 so `--out -` and any traceback printing are safe.
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
from brass_ai.evaluate import heuristic_policy

_LOG = None  # callable(str)


def out(s: str = ""):
    _LOG(s)


class _EraStats:
    """Per-era action tallies for every player (mirrors replay_fmt::PStats)."""

    def __init__(self):
        self.build = defaultdict(int)  # pid -> industry-index -> count
        self.network = defaultdict(int)
        self.dbl_rail = defaultdict(int)
        self.develop = defaultdict(int)
        self.sell = defaultdict(int)
        self.loan = defaultdict(int)
        self.pass_ = defaultdict(int)
        self.scout = defaultdict(int)

    def record(self, pid: int, act: str, ind: int):
        if act == "build":
            self.build[pid * 10 + ind] = self.build.get(pid * 10 + ind, 0) + 1
        elif act == "network":
            self.network[pid] += 1
        elif act == "network_double":
            self.dbl_rail[pid] += 1
        elif act == "develop":
            self.develop[pid] += 1
        elif act == "sell":
            self.sell[pid] += 1
        elif act == "loan":
            self.loan[pid] += 1
        elif act == "pass":
            self.pass_[pid] += 1
        elif act == "scout":
            self.scout[pid] += 1

    def stats_line(self, state, pid: int) -> str:
        build = [self.build.get(pid * 10 + i, 0) for i in range(6)]
        return state.replay_stats_line(
            build, self.network[pid], self.dbl_rail[pid], self.develop[pid],
            self.sell[pid], self.loan[pid], self.pass_[pid], self.scout[pid],
        )


def print_era_snapshot(state, era_name: str, stats: _EraStats, n_players: int):
    out()
    out(f"--- {era_name}时代玩家汇总(动作/在场板块/翻面板块) ---")
    for pid in range(n_players):
        out(f"玩家{pid} 动作: {stats.stats_line(state, pid)}")
        out(f"玩家{pid} 在场板块: {state.replay_tiles(pid, False)}")
        out(f"玩家{pid} 翻面板块: {state.replay_tiles(pid, True)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--seat", type=int, default=None,
                    help="seat for the net (default: seed % 4, like benchmark)")
    ap.add_argument("--all-net", action="store_true",
                    help="ALL 4 seats use the network (net vs net vs net vs net)")
    ap.add_argument("--sims", type=int, default=0,
                    help="MCTS sims per move; 0 = greedy (fast, no search)")
    ap.add_argument("--c_puct", type=float, default=2.5)
    ap.add_argument("--max_depth", type=int, default=10)
    ap.add_argument("--max_moves", type=int, default=600)
    ap.add_argument("--quiet", action="store_true",
                    help="skip per-move state lines (only headers + era/final summaries)")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--out", type=str, default="auto",
                    help="log path (default: logs/replay_seed<seed>.txt; use '-' for stdout)")
    ap.add_argument("--multi", type=str, default=None,
                    help="comma list or range 'a-b' of seeds to replay in ONE "
                         "process (amortizes ~3s torch startup over many games)")
    args = ap.parse_args()

    global _LOG
    engine_logs = os.path.join(os.path.dirname(__file__), "..", "..", "engine", "logs")
    if args.out == "-":
        _LOG = lambda s: sys.stdout.write(s + "\n")
    elif args.out == "auto":
        os.makedirs(engine_logs, exist_ok=True)
        fname = os.path.join(engine_logs, f"replay_seed{args.seed}.txt")
        if args.multi:
            fname = os.path.join(engine_logs, "replay_multi.txt")
        _f = open(fname, "w", encoding="utf-8")
        _LOG = lambda s: _f.write(s + "\n")
    else:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        _f = open(args.out, "w", encoding="utf-8")
        _LOG = lambda s: _f.write(s + "\n")

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
            canonicals, candidates = encode_legal_candidates(state)
            candidates = candidates.unsqueeze(0).to(next(net.parameters()).device)
            mask = torch.ones((1, len(canonicals)), dtype=torch.bool, device=candidates.device)
            with torch.no_grad():
                out = net.policy_value(batch, candidates, mask)
            return canonicals[int(out["candidate_logits"].argmax(dim=1).item())]
        return rm.search(state, args.sims, add_root_noise=False).best

    def replay_one(seed: int):
        if args.all_net:
            seat = 0  # arbitrary; all seats are the net
            policies = [net_pol] * 4
        else:
            seat = args.seat if args.seat is not None else seed % 4
            policies = [heuristic_policy] * 4
            policies[seat] = net_pol
        state = be.GameState(seed=seed, players=4)
        n_players = 4

        out("=" * 72)
        out(f"Replay net={args.ckpt} seed={seed} "
            f"mode={'greedy' if args.sims <= 0 else f'mcts sims={args.sims}'} "
            f"| {'ALL-NET' if args.all_net else f'seat={seat}'} | 起始顺位: {state.turn_order}")
        out("=" * 72)
        for pid in range(n_players):
            out(f"开局 玩家{pid}: {state.replay_player_state(pid)} | "
                f"手牌: {state.replay_hand(pid)}")
        out(f"商家: {state.replay_merchants()}")
        out(state.replay_board())

        canal_stats = _EraStats()
        rail_stats = _EraStats()
        prev_era = state.era
        prev_round = state.round
        moves = 0
        while not state.game_over and moves < args.max_moves:
            moves += 1
            if state.era != prev_era:
                out()
                out(f"------ 时代切换: {prev_era} -> {state.era} ------")
                prev_era = state.era
            if state.round != prev_round:
                out()
                out(f"===== {'运河' if state.era == 0 else '铁路'}时代 第{state.round}轮 | "
                    f"顺位: {state.turn_order} =====")
                prev_round = state.round

            pid = state.current_player_id
            canon = policies[pid](state)
            if canon is None:
                lm = state.legal_moves()
                if not lm:
                    out(f"[!] 玩家{pid} 无合法动作，终止")
                    break
                canon = lm[0][1]

            stats = canal_stats if state.era == 0 else rail_stats
            act, ind = state.replay_action_type(canon)
            stats.record(pid, act, ind)

            detail = state.replay_move_detail(canon)
            before = state.replay_player_state(pid)
            hand_before = state.replay_hand(pid)

            out(f"玩家{pid} [{detail}]")
            out(f"    之前: {before}")
            out(f"    手牌前: {hand_before}")

            summary, ok = state.apply_move_raw(canon)
            if not ok:
                # Defensive: fall back to the first legal move.
                lm = state.legal_moves()
                if lm:
                    canon = lm[0][1]
                    summary, ok = state.apply_move_raw(canon)
            if not ok:
                out(f"    !! 动作失败: {summary}，终止")
                break

            after = state.replay_player_state(pid)
            hand_after = state.replay_hand(pid)
            out(f"    结果: {summary}")
            out(f"    之后: {after}")
            out(f"    手牌后: {hand_after}")
            out(f"    盘面: {state.replay_board()}")
            out(f"    商家: {state.replay_merchants()}")

            tr = state.advance_turn_raw()
            if tr == "end_canal_era":
                out()
                out("--- 运河时代结算明细 ---")
                for p in range(n_players):
                    out(f"玩家{p}: {state.replay_era_score(p)}")
                print_era_snapshot(state, "运河", canal_stats, n_players)
                out(state.replay_cleanup())
                state.finish_canal_era()
                out()
                out("--- 运河时代结束，进入铁路时代(连接/1级板块已清除,重新洗牌发牌) ---")
            elif tr == "end_game":
                state.finish_game()

        if not state.game_over:
            state.finish_game()

        out()
        out("=" * 72)
        out("终局结算")
        out("=" * 72)
        print_era_snapshot(state, "铁路", rail_stats, n_players)
        for pid in range(n_players):
            out(f"玩家{pid}: {state.replay_player_state(pid)}")
        ranking = state.final_ranking()
        out(f"排名: {ranking}")
        if ranking:
            out(f"胜者: 玩家{ranking[0]}")
        out()
        return state.player_vps(), ranking

    results = []
    if args.multi:
        if "-" in args.multi:
            a, b = args.multi.split("-")
            seeds = list(range(int(a), int(b) + 1))
        else:
            seeds = [int(x) for x in args.multi.split(",")]
        for seed in seeds:
            vps, ranking = replay_one(seed)
            results.append((seed, vps, ranking))
        out("### MULTI-SEED SUMMARY ###")
        for seed, vps, ranking in results:
            winner = ranking[0] if ranking else -1
            tag = f"[net-seat P{seat}]" if not args.all_net else ""
            out(f"seed {seed:>3}: vps={vps} winner=P{winner} {tag}")
    else:
        results.append((args.seed, *replay_one(args.seed)))

    if args.out != "-":
        _f.close()
        print(f"log written: {args.out if args.out != 'auto' else engine_logs + '/'}")


def _encode_state(state, net):
    from brass_ai import build_input
    b = build_input.encode_state(state)
    dev = next(net.parameters()).device
    return {k: v.to(dev) for k, v in b.items()}


if __name__ == "__main__":
    main()
