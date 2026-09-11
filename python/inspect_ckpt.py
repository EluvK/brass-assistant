"""Inspect and validate a Brass AI checkpoint without running training.

Usage:
    python python/inspect_ckpt.py checkpoints/v1/bootstrap/b2000.pt
    python python/inspect_ckpt.py checkpoints/selfplay-v4/latest.pt
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch


def inspect_checkpoint(path: Path) -> int:
    if not path.is_file():
        print(f"Error: file not found: {path}", file=sys.stderr)
        return 1

    file_size_mb = path.stat().st_size / (1024 * 1024)
    print(f"\n================ Checkpoint Inspector ================")
    print(f"Path:      {path.resolve()}")
    print(f"File Size: {file_size_mb:.2f} MB")

    try:
        data = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:
        print(f"Error loading checkpoint: {exc}", file=sys.stderr)
        return 1

    is_trainer_ckpt = isinstance(data, dict) and "model" in data
    state_dict = data["model"] if is_trainer_ckpt else data

    # Count parameters
    total_params = sum(p.numel() for p in state_dict.values())
    trainable_mb = sum(p.numel() * p.element_size() for p in state_dict.values()) / (1024 * 1024)

    print(f"Type:      {'Full Trainer State (Resumeable)' if is_trainer_ckpt else 'Raw Model Weights'}")
    print(f"Weights:   {len(state_dict)} tensors, {total_params:,} parameters (~{trainable_mb:.2f} MB)")

    if is_trainer_ckpt:
        epoch = data.get("epoch", "N/A")
        act_ver = data.get("action_schema_version", "unknown")
        tok_ver = data.get("state_token_schema_version", "unknown")
        act_dim = data.get("action_feature_dim", "unknown")
        print(f"Epoch/It:  {epoch}")
        print(f"Schema:    Action Schema v{act_ver} (dim={act_dim}), State Token Schema v{tok_ver}")
        has_opt = "optimizer" in data
        has_sched = "scheduler" in data
        has_scaler = "scaler" in data
        print(f"Contents:  optimizer={has_opt}, scheduler={has_sched}, scaler={has_scaler}")

        # Check compatibility with current engine
        try:
            from brass_ai import _engine as be
            from brass_ai.hierarchical_policy import ACTION_FEATURE_DIM, ACTION_SCHEMA_VERSION, STATE_TOKEN_SCHEMA_VERSION

            comp_act = (act_ver == ACTION_SCHEMA_VERSION and act_dim == ACTION_FEATURE_DIM)
            comp_tok = (tok_ver == STATE_TOKEN_SCHEMA_VERSION)
            if comp_act and comp_tok:
                print(f"Compatibility: [OK] Matches current engine schemas (Action v{ACTION_SCHEMA_VERSION}, State v{STATE_TOKEN_SCHEMA_VERSION})")
            else:
                print(f"Compatibility: [MISMATCH] Current engine expects Action v{ACTION_SCHEMA_VERSION} (dim={ACTION_FEATURE_DIM}), State v{STATE_TOKEN_SCHEMA_VERSION}")
        except Exception:
            pass

    # 1. Check embedded metadata inside the .pt file
    embedded_meta = data.get("meta") or data.get("metadata") if isinstance(data, dict) else None
    if embedded_meta and isinstance(embedded_meta, dict):
        print("\n--- Embedded Metadata (Inside .pt) ---")
        for k, v in embedded_meta.items():
            print(f"  {k}: {v}")

    # 2. Check accompanying latest.json in the same directory
    sidecar_json = path.parent / "latest.json"
    if sidecar_json.is_file():
        try:
            sidecar_data = json.loads(sidecar_json.read_text(encoding="utf-8"))
            print("\n--- Run Progress (from sidecar latest.json) ---")
            it = sidecar_data.get("iteration")
            avg_vp = sidecar_data.get("avg_vp")
            win_vp = sidecar_data.get("winner_avg_vp")
            buf = sidecar_data.get("buffer")
            samples = sidecar_data.get("samples")
            arena_wr = sidecar_data.get("arena_winrate")
            heur_wr = sidecar_data.get("heuristic_winrate")
            promoted = sidecar_data.get("promoted", False)

            if it is not None:
                print(f"  Iteration:        {it}")
            if avg_vp is not None:
                print(f"  Table Avg VP:     {avg_vp:.2f}")
            if win_vp is not None:
                print(f"  Winner Avg VP:    {win_vp:.2f}")
            if buf is not None:
                print(f"  Replay Buffer:    {buf:,} samples (last batch: {samples})")
            if arena_wr is not None:
                print(f"  Arena Winrate:    {arena_wr:.1%}")
            if heur_wr is not None:
                print(f"  Heuristic Winrate:{heur_wr:.1%}")
            if promoted:
                print(f"  Promoted:         Yes (Current Best)")
        except Exception as exc:
            print(f"  (failed to parse latest.json: {exc})")

    # 3. Check zoo manifest.json (in current or parent directory)
    for manifest_path in [path.parent / "manifest.json", path.parent.parent / "manifest.json"]:
        if manifest_path.is_file():
            try:
                manifest_data = json.loads(manifest_path.read_text(encoding="utf-8"))
                models = manifest_data.get("models", {})
                file_key = path.name
                if file_key in models:
                    print(f"\n--- Zoo Identity (from {manifest_path.name}) ---")
                    entry = models[file_key]
                    for ek, ev in entry.items():
                        if isinstance(ev, dict):
                            print(f"  {ek}:")
                            for sub_k, sub_v in ev.items():
                                print(f"    {sub_k}: {sub_v}")
                        else:
                            print(f"  {ek}: {ev}")
                    break
            except Exception:
                pass

    print("======================================================\n")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect a Brass AI checkpoint")
    parser.add_argument("ckpt", type=Path, help="Path to .pt checkpoint")
    args = parser.parse_args()
    return inspect_checkpoint(args.ckpt)


if __name__ == "__main__":
    sys.exit(main())
