"""Replay-web strategy worker: a network checkpoint sitting at one seat.

The Rust replay session (``engine/src/ai/python_worker.rs``) spawns one worker
per ``python:`` seat and talks line-delimited JSON over stdin/stdout:

  worker -> {"type": "ready", "name": ..., "meta": {...}}            once at startup
  rust   -> {"type": "choose", "request_id": N, "snapshot": "<base64>",
             "legal": ["<canonical>", ...]}
  worker -> {"type": "choice", "request_id": N, "canonical": ...,
             "evidence": {"mode": ..., "policy": {...}, "visits": {...},
                          "root_value": ...}}
  worker -> {"type": "error", "request_id": N, "message": ...}       per-request failure

A snapshot is the exact byte format ``GameState.snapshot()`` produces
(magic + version + ``snapshot_bytes``), so the worker restores the true full
state — including RNG and all hands — via ``GameState.from_snapshot``.

The worker itself never advances the game; it only answers ``choose`` requests
and stays alive until stdin closes.

Usage (normally spawned by the Rust side, shown here for manual testing):

    python -m brass_ai.replay_worker --ckpt checkpoints/bootstrap.pt \
        --mode mcts --sims 128 --device cpu
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
from pathlib import Path

import numpy as np
import torch

from . import _engine as be
from .hierarchical_policy import encode_legal_candidates, pad_candidate_features
from .rust_mcts import RustISMCTS, RustMCTSConfig


class WorkerConfig:
    def __init__(
        self,
        ckpt: str,
        mode: str = "mcts",
        sims: int = 128,
        device: str | None = None,
        c_puct: float = 2.5,
        max_depth: int = 10,
        candidate_k: int = 0,
    ):
        self.ckpt = ckpt
        self.mode = mode
        self.sims = sims
        self.device = device
        self.c_puct = c_puct
        self.max_depth = max_depth
        self.candidate_k = candidate_k


def load_net(cfg: WorkerConfig) -> PolicyValueNet:
    """Load a checkpoint with the same schema constraints as Trainer.load_state_dict."""
    from .net import PolicyValueNet  # local import: torch startup stays lazy for --help

    ckpt = torch.load(cfg.ckpt, map_location="cpu", weights_only=False)
    if not isinstance(ckpt, dict) or "model" not in ckpt:
        raise ValueError(f"{cfg.ckpt} is not a training checkpoint (missing 'model')")
    checks = {
        "action_feature_schema_version": be.ACTION_FEATURE_SCHEMA_VERSION,
        "state_feature_schema_version": be.STATE_FEATURE_SCHEMA_VERSION,
    }
    for key, expected in checks.items():
        found = ckpt.get(key)
        if found is not None and found != expected:
            raise ValueError(f"checkpoint {key}={found} does not match engine {expected}")
    shapes = ckpt.get("state_feature_shapes") or {}
    expected_shapes = {
        "board": (be.BOARD_PLANES, be.BOARD_CELLS),
        "links": (be.LINK_PLANES, be.LINK_CELLS),
        "global": be.GLOBAL_LEN,
        "hand": be.HAND_LEN,
    }
    for key, expected in expected_shapes.items():
        found = shapes.get(key)
        if found is None:
            continue
        found = tuple(found) if isinstance(found, (list, tuple)) else found
        expected = expected if isinstance(expected, (int, float)) else tuple(expected)
        if found != expected:
            raise ValueError(f"checkpoint {key} shape {found} does not match engine {expected}")
    net = PolicyValueNet()
    net.load_state_dict(ckpt["model"])
    net.eval()
    return net


def root_forward(net, state, device: str) -> tuple[dict[str, float], float]:
    """One network forward over all legal candidates.

    Returns ``(policy, value)`` where ``policy`` maps every concrete legal
    canonical to its probability and ``value`` is the current player's value
    estimate from the rank head (search scale: higher = better).
    """
    from .rust_mcts import make_net_fn

    canonicals, features = encode_legal_candidates(state)
    padded, mask = pad_candidate_features([features])
    board, links, global_vec, own_hand, opp_hands = state.state_to_tensor()
    # make_net_fn mirrors the Rust callback contract: per-row arrays.
    logits, values = make_net_fn(net, device)(
        board, links,
        np.asarray(global_vec, dtype=np.float32).reshape(1, -1),
        np.asarray(own_hand, dtype=np.float32).reshape(1, -1),
        np.asarray(opp_hands, dtype=np.float32).reshape(1, -1),
        padded.numpy(), mask.numpy(),
    )
    probs = torch.softmax(torch.from_numpy(logits[0]), dim=-1).numpy()
    policy = {canon: float(p) for canon, p in zip(canonicals, probs)}
    return policy, float(values[0][state.current_player_id])


def handle_request(net, mcts: RustISMCTS | None, cfg: WorkerConfig, snapshot: bytes, legal: list[str]) -> dict:
    """Answer one ``choose`` request. Pure function of the inputs, for testing."""
    state = be.GameState.from_snapshot(list(snapshot))
    policy, root_value = root_forward(net, state, cfg.device)
    if cfg.mode == "mcts":
        if mcts is None:
            raise RuntimeError("mcts mode requires a constructed searcher")
        result = mcts.search(state, cfg.sims, add_root_noise=False)
        visits = {
            result.canon_by_candidate[candidate_id]: count
            for candidate_id, count in result.visits.items()
            if candidate_id in result.canon_by_candidate
        }
        best = result.best
        if best is None or best not in legal:
            # Defensive: fall back to the strongest searched move, then policy.
            best = max(visits, key=visits.get, default=None)
    else:
        visits = {}
        best = None
    if best is None or best not in legal:
        best = max(policy, key=policy.get)
    if best not in legal:
        raise ValueError(f"worker chose {best!r} which is not in the legal set")
    return {
        "canonical": best,
        "evidence": {
            "mode": cfg.mode,
            "policy": policy,
            "visits": visits,
            "root_value": root_value,
        },
    }


def emit(payload: dict) -> None:
    # ASCII-only output avoids Windows pipe encoding issues end to end.
    sys.stdout.write(json.dumps(payload, ensure_ascii=True) + "\n")
    sys.stdout.flush()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="brass replay-web network seat worker")
    parser.add_argument("--ckpt", required=True, help="training checkpoint (.pt)")
    parser.add_argument("--mode", choices=["mcts", "policy"], default="mcts")
    parser.add_argument("--sims", type=int, default=128, help="ISMCTS simulations (mcts mode)")
    parser.add_argument("--device", default=None, help="cuda/cpu; default: cuda if available")
    parser.add_argument("--c-puct", type=float, default=2.5)
    parser.add_argument("--max-depth", type=int, default=10)
    parser.add_argument("--candidate-k", type=int, default=4)
    args = parser.parse_args(argv)

    sys.stdin.reconfigure(encoding="utf-8", errors="replace")
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    cfg = WorkerConfig(
        ckpt=args.ckpt,
        mode=args.mode,
        sims=args.sims,
        device=args.device or ("cuda" if torch.cuda.is_available() else "cpu"),
        c_puct=args.c_puct,
        max_depth=args.max_depth,
        candidate_k=args.candidate_k,
    )
    if not Path(cfg.ckpt).exists():
        print(f"checkpoint not found: {cfg.ckpt}", file=sys.stderr)
        return 2
    net = load_net(cfg)
    mcts = None
    if cfg.mode == "mcts":
        mcts = RustISMCTS(net, RustMCTSConfig(
            c_puct=cfg.c_puct,
            max_depth=cfg.max_depth,
            candidate_k=cfg.candidate_k,
            device=cfg.device,
        ))
    name = f"net-{cfg.mode}(sims={cfg.sims}) {Path(cfg.ckpt).name}"
    emit({
        "type": "ready",
        "name": name,
        "meta": {
            "ckpt": cfg.ckpt,
            "mode": cfg.mode,
            "sims": cfg.sims,
            "device": cfg.device,
            "action_feature_schema_version": be.ACTION_FEATURE_SCHEMA_VERSION,
            "state_feature_schema_version": be.STATE_FEATURE_SCHEMA_VERSION,
        },
    })

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
        except json.JSONDecodeError as error:
            emit({"type": "error", "request_id": None, "message": f"bad request JSON: {error}"})
            continue
        request_id = request.get("request_id")
        try:
            if request.get("type") != "choose":
                raise ValueError(f"unsupported request type: {request.get('type')!r}")
            snapshot = base64.b64decode(request["snapshot"], validate=True)
            answer = handle_request(net, mcts, cfg, snapshot, list(request["legal"]))
            emit({"type": "choice", "request_id": request_id, **answer})
        except Exception as error:  # per-request failure: report and keep serving
            emit({"type": "error", "request_id": request_id,
                  "message": f"{type(error).__name__}: {error}"})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
