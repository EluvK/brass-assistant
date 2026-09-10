"""Go / no-go benchmark: can a head rank sibling moves by their real outcome?

The search can only improve on the policy prior if some value estimate tells it
that one child is better than another. `V(s)` is structurally bad at this: two
sibling moves produce two nearly identical states, and the measured spread of
the rank head across siblings (sd ~0.014) is an order of magnitude below its own
error (~0.14). `Q(s, a)` gets the action embedding as an input instead, so the
comparison is first-order for it.

This script measures both against a rollout reference: for the top-K children by
prior it plays the rest of the game with the engine heuristic and records where
the mover actually finishes, then reports the within-position Spearman
correlation between each predictor and the realized outcome. Higher is better;
0 means the head cannot rank siblings at all.

Run from the repo root:

    ./.venv/Scripts/python.exe python/bench_value_ranking.py --ckpt checkpoints/bootstrap-qhead.pt

Read it as a comparison, not an absolute score: one rollout per child is a single
sample, so per-position correlations are noisy and only the *paired* difference
between predictors is meaningful at this sample size. `--positions` trades
runtime for resolution (the reported +- is the standard error of the mean).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from brass_ai import _engine as be
from brass_ai.net import PolicyValueNet, load_state_dict_tolerant


def _spearman(x: np.ndarray, y: np.ndarray) -> float | None:
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return None
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    return float(np.corrcoef(rx, ry)[0, 1])


def _forward(net: PolicyValueNet, state, pid: int) -> dict:
    board, links, g, oh, op = state.state_to_tensor(pid)
    batch = {
        "board": torch.from_numpy(board).unsqueeze(0),
        "links": torch.from_numpy(links).unsqueeze(0),
        "global": torch.from_numpy(g).unsqueeze(0),
        "own_hand": torch.from_numpy(oh).unsqueeze(0),
        "opp_hands": torch.from_numpy(op).unsqueeze(0),
    }
    canonical, features = state.legal_candidates()
    actions = torch.from_numpy(np.asarray(features, dtype=np.float32)).unsqueeze(0)
    mask = torch.ones(1, actions.shape[1], dtype=torch.bool)
    with torch.no_grad():
        out = net(batch, actions, mask)
    rank = out["rank"][0].numpy()
    win = torch.softmax(out["winner_logits"][0], dim=0).numpy()
    return {
        "V(s)=1-rank/n": 1.0 - float(rank[pid]) / 4.0,
        "winner_prob": float(win[pid]),
        "prior": out["candidate_log_probs"][0].exp().numpy(),
        "Q(s,a)": out["candidate_value"][0].numpy(),
        "canonical": canonical,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--candidates", type=int, default=6,
                        help="siblings per position, taken from the prior's top K")
    parser.add_argument("--positions", type=int, default=36)
    parser.add_argument("--seeds", type=int, default=3)
    args = parser.parse_args()

    net = PolicyValueNet()
    payload = torch.load(args.ckpt, map_location="cpu")
    missing, _ = load_state_dict_tolerant(
        net, payload["model"] if "model" in payload else payload
    )
    if missing:
        print(f"note: {args.ckpt} predates {missing} - those heads are random; "
              "treat their rows below as a baseline, not a measurement")
    net.eval()

    rows = []
    per_seed = max(1, args.positions // max(args.seeds, 1))
    step = max(1, 96 // per_seed)
    for seed in range(args.seeds):
        for slot in range(per_seed):
            state = be.GameState(seed=7 + seed * 14, players=4)
            for _ in range(8 + slot * step):
                move, _, _ = state.choose_heuristic()
                state.apply_move(move)
            if state.game_over:
                continue
            pid = state.current_player_id
            world = state.determinize()
            root = _forward(net, world, pid)
            for index in np.argsort(-root["prior"])[: args.candidates]:
                child = world.clone()
                _, ok = child.apply_move_raw(root["canonical"][int(index)])
                if not ok:
                    continue
                child.advance_turn_raw()
                predicted = _forward(net, child, pid)
                guard = 0
                while not child.game_over and guard < 600:
                    move, _, _ = child.choose_heuristic()
                    child.apply_move(move)
                    guard += 1
                if not child.game_over:
                    continue
                place = child.final_ranking().index(pid) + 1
                rows.append({
                    "pos": (seed, slot),
                    "goal": 1.0 - place / 4.0,
                    "prior": float(root["prior"][int(index)]),
                    "Q(s,a)": float(root["Q(s,a)"][int(index)]),
                    "V(s)=1-rank/n": predicted["V(s)=1-rank/n"],
                    "winner_prob": predicted["winner_prob"],
                })

    keys = sorted({row["pos"] for row in rows})
    print(f"positions={len(keys)} rollouts={len(rows)}")
    for name in ("V(s)=1-rank/n", "Q(s,a)", "prior", "winner_prob"):
        correlations = []
        for key in keys:
            subset = [row for row in rows if row["pos"] == key]
            value = _spearman(
                np.array([row[name] for row in subset]),
                np.array([row["goal"] for row in subset]),
            )
            if value is not None:
                correlations.append(value)
        if not correlations:
            continue
        arr = np.array(correlations)
        se = arr.std(ddof=1) / np.sqrt(len(arr)) if len(arr) > 1 else 0.0
        print("  %-14s within-position Spearman = %+.3f +- %.3f  (%3.0f%% positive, n=%d)"
              % (name, arr.mean(), se, 100.0 * (arr > 0).mean(), len(arr)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
