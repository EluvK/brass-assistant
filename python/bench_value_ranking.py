"""Go / no-go benchmark: can a head rank sibling moves by their real outcome?

The search can only improve on the policy prior if some value estimate tells it
that one child is better than another. `V(s)` is structurally bad at this: two
sibling moves produce two nearly identical states. `Q(s, a)` receives the
action's referenced entities directly, so the comparison is first-order for it.

This script measures both against a rollout reference: for the top-K children by
prior it plays the rest of the game with the engine heuristic and records the
mover's realized terminal utility, then reports the within-position Spearman
correlation between each predictor and that outcome. Higher is better; 0 means
the head cannot rank siblings at all.

Run from the repo root:

    ./.venv/Scripts/python.exe python/bench_value_ranking.py --ckpt checkpoints/selfplay/latest.pt

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
from brass_ai.net import PolicyValueNet, state_batch
from brass_ai.rl_league import HeuristicRoundPlayer


def _spearman(x: np.ndarray, y: np.ndarray) -> float | None:
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return None
    def ranks(values):
        order = np.argsort(values, kind="stable")
        sorted_values = values[order]
        boundaries = np.r_[0, np.flatnonzero(sorted_values[1:] != sorted_values[:-1]) + 1, len(values)]
        result = np.empty(len(values), dtype=float)
        for lo, hi in zip(boundaries[:-1], boundaries[1:]):
            result[order[lo:hi]] = (lo + hi - 1) / 2
        return result

    rx, ry = ranks(x), ranks(y)
    return float(np.corrcoef(rx, ry)[0, 1])


def _forward(net: PolicyValueNet, state) -> dict:
    """One forward pass, always from the acting player's perspective."""
    cells, links, merchants, seats, g = state.state_tokens()
    batch = state_batch((
        torch.from_numpy(np.asarray(cells, dtype=np.float32)).unsqueeze(0),
        torch.from_numpy(np.asarray(links, dtype=np.float32)).unsqueeze(0),
        torch.from_numpy(np.asarray(merchants, dtype=np.float32)).unsqueeze(0),
        torch.from_numpy(np.asarray(seats, dtype=np.float32)).unsqueeze(0),
        torch.from_numpy(np.asarray(g, dtype=np.float32)).unsqueeze(0),
    ))
    canonical, features = state.legal_candidates()
    actions = torch.from_numpy(np.asarray(features, dtype=np.float32)).unsqueeze(0)
    mask = torch.ones(1, actions.shape[1], dtype=torch.bool)
    with torch.no_grad():
        out = net(batch, actions, mask)
    actor = state.current_player_id
    return {
        "actor": actor,
        "value": out["value"][0].numpy(),                       # index 0 = actor
        "winner": torch.softmax(out["winner_logits"][0], dim=0).numpy(),
        "prior": out["candidate_log_probs"][0].exp().numpy(),
        "Q(s,a)": out["candidate_value"][0].numpy(),
        "canonical": canonical,
        "features": np.asarray(features),
    }


def _seat_of(pid: int, actor: int, players: int = 4) -> int:
    """Index of absolute player `pid` in a perspective-rotated vector."""
    return (pid - actor) % players


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--candidates", type=int, default=6,
                        help="siblings per position, taken from the prior's top K")
    parser.add_argument("--positions", type=int, default=36)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()

    torch.set_num_threads(max(1, args.threads))

    net = PolicyValueNet()
    payload = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    net.load_state_dict(payload["model"] if "model" in payload else payload)
    net.eval()

    rows = []
    per_seed = max(1, args.positions // max(args.seeds, 1))
    step = max(1, 96 // per_seed)
    for seed in range(args.seeds):
        for slot in range(per_seed):
            state = be.GameState(seed=7 + seed * 14, players=4)
            prefix_player = HeuristicRoundPlayer()
            for _ in range(8 + slot * step):
                if state.game_over:
                    break
                move = prefix_player.step(state)
                state.apply_move(move)
            if state.game_over:
                continue
            pid = state.current_player_id
            world = state.determinize()
            root = _forward(net, world)
            seen = set()
            indices = []
            for index in np.argsort(-root["prior"], kind="stable"):
                identity = root["features"][index].tobytes()
                if identity not in seen:
                    seen.add(identity)
                    indices.append(index)
                if len(indices) >= args.candidates:
                    break
            for index in indices:
                child = world.clone()
                try:
                    child.apply_move(root["canonical"][int(index)])
                except ValueError:
                    continue
                if child.game_over:
                    continue
                predicted = _forward(net, child)
                guard = 0
                rollout_teachers = {p: HeuristicRoundPlayer() for p in range(4)}
                while not child.game_over and guard < 600:
                    actor = child.current_player_id
                    move = rollout_teachers[actor].step(child)
                    child.apply_move(move)
                    guard += 1
                if not child.game_over:
                    continue
                vps = np.asarray(child.player_vps(), dtype=np.float64)
                goal = (vps[pid] - vps.mean()) / float(be.VP_SCALE)
                seat = _seat_of(pid, predicted["actor"])
                rows.append({
                    "pos": (seed, slot),
                    "goal": float(goal),
                    "prior": float(root["prior"][int(index)]),
                    "Q(s,a)": float(root["Q(s,a)"][int(index)]),
                    "V(s)": float(predicted["value"][seat]),
                    "winner_prob": float(predicted["winner"][seat]),
                })

    keys = sorted({row["pos"] for row in rows})
    print(f"positions={len(keys)} rollouts={len(rows)}")
    for name in ("V(s)", "Q(s,a)", "prior", "winner_prob"):
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
