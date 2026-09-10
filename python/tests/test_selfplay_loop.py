import numpy as np
import torch

from brass_ai import _engine as be
from brass_ai.net import PolicyValueNet
from brass_ai.rust_mcts import RustISMCTS, RustMCTSConfig
from brass_ai.selfplay import Sample, SelfPlayConfig, materialize_sample, play_game
from brass_ai.selfplay_loop import (
    LoopConfig,
    ReplayBuffer,
    run_selfplay,
    wilson_lower_bound,
)
from brass_ai.train import TrainConfig, Trainer


def _make_mcts():
    torch.manual_seed(0)
    torch.set_num_threads(2)
    net = PolicyValueNet()
    return net, RustISMCTS(net, RustMCTSConfig(device="cpu", max_depth=4, batch_size=8))


def test_selfplay_samples_are_snapshot_backed_and_materialize():
    _net, mcts = _make_mcts()
    samples, _ = play_game(
        mcts, SelfPlayConfig(sims=2, seed=3, temperature=0.0, store_snapshots=True)
    )
    assert samples
    sample = samples[0]
    # Snapshot form: no dense tensors, target is a sparse canonical->visit map.
    assert sample.snapshot is not None
    assert sample.candidates is None
    assert sample.cells is None
    assert sample.policy_by_canonical

    dense = materialize_sample(sample)
    assert dense.candidates is not None
    assert dense.candidates.shape[1] == be.ACTION_FEATURE_DIM
    assert dense.cells.shape == (be.BOARD_CELLS, be.F_CELL)
    assert dense.seats.shape == (be.SEAT_COUNT, be.F_SEAT)
    assert np.isclose(dense.policy.sum(), 1.0)
    assert dense.policy.shape[0] == dense.candidates.shape[0]
    assert dense.pid == sample.pid
    # The played sibling is what supervises the action-conditioned value head.
    assert sample.played_canonical is not None
    assert 0 <= dense.action_index < dense.candidates.shape[0]
    # The stored observation is a determinization, so the materialized inputs
    # must be exactly what the snapshot encodes.
    restored = be.GameState.from_snapshot(sample.snapshot)
    np.testing.assert_allclose(dense.seats, restored.state_tokens()[3])


def test_snapshot_policy_aligns_to_the_restored_candidate_set():
    from brass_ai import _engine as be

    _net, mcts = _make_mcts()
    samples, _ = play_game(
        mcts, SelfPlayConfig(sims=2, seed=3, temperature=0.0, store_snapshots=True)
    )
    sample = samples[0]
    dense = materialize_sample(sample)
    canonical, _features = be.GameState.from_snapshot(sample.snapshot).legal_candidates()
    assert len(canonical) == dense.candidates.shape[0]
    position = {move: index for index, move in enumerate(canonical)}
    target_rows = {position[move] for move in sample.policy_by_canonical}
    # Equivalence coalescing may spread mass further, but never removes it.
    assert target_rows <= set(np.flatnonzero(dense.policy).tolist())
    assert np.isclose(dense.policy.sum(), 1.0)


def test_replay_buffer_keeps_recent_iterations_and_draws():
    buffer = ReplayBuffer(max_samples=10, max_iterations=2)
    buffer.add([Sample(pid=0, era=0)] * 6)
    buffer.add([Sample(pid=1, era=0)] * 6)
    buffer.trim()
    # The oldest iteration is dropped first: 6 + 6 -> 6 samples remain.
    assert len(buffer) == 6
    assert buffer.recent(1) == [sample for sample in buffer.recent(1)]
    rng = np.random.default_rng(0)
    drawn = buffer.draw(4, recent_fraction=0.5, recent_iterations=1, rng=rng)
    assert len(drawn) == 4
    assert all(isinstance(sample, Sample) for sample in drawn)


def test_wilson_lower_bound_is_conservative_on_small_samples():
    assert wilson_lower_bound(0, 0) == 0.0
    assert wilson_lower_bound(20, 40) < 0.5
    assert wilson_lower_bound(35, 40) > 0.5
    assert wilson_lower_bound(400, 400) > 0.99


def test_run_selfplay_smoke_trains_and_reports():
    net, _mcts = _make_mcts()
    trainer = Trainer(net, TrainConfig(
        device="cpu", epochs=1, batch_size=16, materialize_workers=1,
    ))
    seen = []
    try:
        history = run_selfplay(
            net, trainer,
            LoopConfig(
                iterations=1,
                games_per_iter=1,
                sims=2,
                workers=1,
                eval_every=0,
                train_samples=32,
                mcts=RustMCTSConfig(device="cpu", max_depth=4, batch_size=8),
                selfplay=SelfPlayConfig(max_moves=600),
            ),
            on_iteration=lambda stats, _net, _trainer: seen.append(stats),
        )
    finally:
        trainer.close()
    assert len(history) == 1
    stats = history[0]
    assert stats.samples > 0
    assert stats.trained > 0
    assert stats.buffer == stats.samples
    assert "policy" in stats.losses
    assert "q" in stats.losses
    assert np.isfinite(stats.losses["q"])
    assert seen and seen[0] is stats
    assert stats.failed_applies >= 0
    assert stats.rewritten_applies >= 0
