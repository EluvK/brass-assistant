"""Benchmarks for the training-data hot paths (Rust engine via PyO3).

Run:  .venv/Scripts/python.exe python/bench_train_paths.py [n_positions]

Measures the per-decision-point cost of each stage used by the training
pipeline, on realistic midgame positions of a 4-player game:

  snapshot        : GameState.snapshot()          (imitation persistence)
  from_snapshot   : GameState.from_snapshot()    (materialization restore)
  legal_candidates: full-legal candidate features (materialization)
  state_to_tensor : state tensor encoding        (materialization)
  heuristic_cands : teacher shortlist            (imitation generation)
  materialize     : one-call Rust sample materialization
"""

from __future__ import annotations

import sys
import time

import numpy as np

sys.path.insert(0, "python")
from brass_ai import _engine as be  # noqa: E402


def midgame_positions(seed: int, n_plies: int) -> list[bytes]:
    """Advance one game `n_plies` moves and snapshot every position."""
    state = be.GameState(seed=seed, players=4)
    snaps = [bytes(state.snapshot())]
    for _ in range(n_plies):
        canon = state.heuristic_candidates()[3]
        state.apply_move(canon)
        snaps.append(bytes(state.snapshot()))
        if state.game_over:
            break
    return snaps


def bench(name: str, fn, args, iters: int) -> float:
    fn(*args)  # warmup
    t0 = time.perf_counter()
    for _ in range(iters):
        fn(*args)
    dt = (time.perf_counter() - t0) / iters
    print(f"  {name:<18} {dt * 1e3:9.3f} ms/call")
    return dt


def main() -> None:
    n_positions = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    print(f"collecting {n_positions} midgame positions (4p heuristic game)...")
    snaps = midgame_positions(2026, n_positions)
    print(f"  got {len(snaps)} snapshots")

    states = [be.GameState.from_snapshot(s) for s in snaps]
    n_cands = np.mean([len(s.legal_candidates()[0]) for s in states])
    print(f"  avg legal candidates: {n_cands:.1f}")
    n_cands_t = np.mean([s.heuristic_candidates()[7] for s in states])
    print(f"  avg teacher candidates: {n_cands_t:.1f}")

    print("\nper-call timings (avg over positions):")
    iters = max(3, 200 // len(snaps))
    bench("snapshot", lambda st: st.snapshot(), (states[0],), 200)
    bench("from_snapshot", be.GameState.from_snapshot, (snaps[0],), 200)
    bench("legal_candidates", lambda st: st.legal_candidates(), (states[0],), iters)
    bench("state_to_tensor", lambda st: st.state_to_tensor(), (states[0],), iters)
    bench("heuristic_cands", lambda st: st.heuristic_candidates(), (states[0],), iters)

    # end-to-end: materialize one training sample like selfplay.materialize_sample
    def materialize_one(snap: bytes, pid: int, era: int, teacher: str):
        (r_pid, r_era, board, links, g, oh, op, candidates, t_idx, policy) = (
            be.GameState.materialize_snapshot(snap, teacher)
        )
        assert r_pid == pid and r_era == era
        return board, links, g, oh, op, candidates, t_idx, policy

    t0 = time.perf_counter()
    for snap, st in zip(snaps, states):
        teacher = st.heuristic_candidates()[3]
        materialize_one(snap, st.current_player_id, st.era, teacher)
    dt = time.perf_counter() - t0
    print(f"\n  materialize_one (end-to-end)   {dt / len(snaps) * 1e3:9.3f} ms/sample "
          f"({len(snaps) / dt:.1f} samples/s)")

    # sanity: policy coalescing via the Rust helper
    from brass_ai.hierarchical_policy import coalesce_equivalent_policy

    canonicals, feats = states[0].legal_candidates()
    policy = np.zeros(len(canonicals), dtype=np.float32)
    policy[0] = 0.7
    if len(canonicals) > 1:
        policy[1] = 0.3
    t0 = time.perf_counter()
    for _ in range(50):
        coalesce_equivalent_policy(feats, policy)
    print(f"  coalesce_equivalent_policy     {(time.perf_counter() - t0) / 50 * 1e3:9.3f} ms/call")


if __name__ == "__main__":
    main()
