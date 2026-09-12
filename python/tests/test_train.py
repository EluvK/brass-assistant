import numpy as np
import pytest
import torch

from brass_ai.net import PolicyValueNet
from brass_ai import selfplay
from brass_ai.selfplay import Sample, _value_targets, generate_imitation_samples
from brass_ai.hierarchical_policy import encode_legal_candidates
from brass_ai import _engine as be
from brass_ai.train import TrainConfig, Trainer, _to_batch, compute_loss, evaluate_policy


def test_trainer_reduces_loss_and_is_persistent():
    torch.manual_seed(0)
    torch.set_num_threads(2)
    net = PolicyValueNet()
    samples = generate_imitation_samples(1, players=4, max_moves=600)
    assert len(samples) >= 10

    cfg = TrainConfig(device="cpu", epochs=3, batch_size=32, lr=1e-3)
    trainer = Trainer(net, cfg)

    b = _to_batch(samples)

    net.eval()
    with torch.no_grad():
        pl_before = compute_loss(b, net, 0.0, "cpu")[1]

    trainer.train_on_samples(samples[:])

    # Trainer must keep its optimizer across calls (state persists).
    assert trainer.optimizer.state, "optimizer should have state after a step"
    before_lr = trainer.current_lr()
    trainer.train_on_samples(samples[:])  # second call reuses the same optimizer
    assert trainer.epoch_count == 2 * cfg.epochs

    # Loss on the training data should drop after fitting.
    net.eval()
    with torch.no_grad():
        pl_after = compute_loss(b, net, 0.0, "cpu")[1]
    assert pl_after.item() < pl_before.item(), \
        "policy loss should decrease after training"
    assert before_lr > 0.0


def test_trainer_state_roundtrip():
    torch.manual_seed(1)
    net = PolicyValueNet()
    trainer = Trainer(net, TrainConfig(device="cpu", epochs=1, batch_size=16))
    sd = trainer.state_dict()
    net2 = PolicyValueNet()
    trainer2 = Trainer(net2, TrainConfig(device="cpu", epochs=1, batch_size=16))
    trainer2.load_state_dict(sd)
    assert trainer2.epoch_count == 0
    # loaded weights are identical
    for p1, p2 in zip(trainer.net.parameters(), trainer2.net.parameters()):
        assert torch.equal(p1.detach().cpu(), p2.detach().cpu())


def test_trainer_rejects_old_state_feature_schema():
    trainer = Trainer(PolicyValueNet(), TrainConfig(device="cpu", epochs=1, batch_size=2))
    checkpoint = trainer.state_dict()
    checkpoint.pop("state_token_schema_version")
    checkpoint.pop("state_token_shapes")
    with pytest.raises(ValueError, match="state-token schema"):
        trainer.load_state_dict(checkpoint)


def test_value_target_is_a_vp_margin_and_winner_uses_the_official_tiebreak():
    # Players 1 and 3 tie on VP; the engine's ranking resolves it by income
    # then cash, and the winner target must follow that order.
    value, winner = _value_targets([110, 100, 60, 100], [1, 0, 3, 2], 4)
    np.testing.assert_allclose(value, np.asarray([17.5, 7.5, -32.5, 7.5], dtype=np.float32) / 50.0)
    assert np.isclose(float(value.sum()), 0.0, atol=1e-5)
    np.testing.assert_array_equal(winner, np.asarray([0.0, 1.0, 0.0, 0.0], dtype=np.float32))


def test_policy_evaluation_materializes_snapshot_batches():
    state = be.GameState(seed=73, players=4)
    teacher, _, _ = state.choose_heuristic_round()
    sample = Sample(
        pid=state.current_player_id, era=state.era,
        value=np.zeros(4, dtype=np.float32), winner=np.zeros(4, dtype=np.float32),
        econ=np.zeros(2, dtype=np.float32), snapshot=bytes(state.snapshot()),
        teacher_canonical=teacher,
    )
    metrics = evaluate_policy(PolicyValueNet(), [sample], "cpu", batch_size=1)
    assert metrics["candidate_count_mean"] == len(encode_legal_candidates(state)[0])


def test_imitation_quality_filter_retries_until_it_has_requested_games(monkeypatch):
    seen_seeds = []

    def fake_game(args):
        seed, *_ = args
        seen_seeds.append(seed)
        # Both thresholds are strict: seed 0 fails at exactly 60 VP; seed 1
        # qualifies with mean 81 and minimum 81.
        vps = np.asarray([60, 90, 90, 90] if seed == 0 else [81, 81, 81, 81])
        return [seed], vps

    monkeypatch.setattr(selfplay, "_generate_imitation_game", fake_game)
    samples = generate_imitation_samples(
        1, workers=1, min_avg_vp=80, min_vp=60, max_attempts=2,
    )

    assert samples == [1]
    assert seen_seeds == [0, 1]


def test_imitation_quality_filter_reports_exhausted_attempts(monkeypatch):
    monkeypatch.setattr(
        selfplay,
        "_generate_imitation_game",
        lambda _args: (["rejected"], np.asarray([60, 90, 90, 90])),
    )

    with pytest.raises(RuntimeError, match="only accepted 0/1"):
        generate_imitation_samples(
            1, workers=1, min_avg_vp=80, min_vp=60, max_attempts=2,
        )


def test_abs_vp_loss_computation_and_masking():
    torch.manual_seed(42)
    net = PolicyValueNet()
    state = be.GameState(seed=10, players=4)
    teacher, _, _ = state.choose_heuristic_round()

    s_valid = Sample(
        pid=state.current_player_id,
        era=state.era,
        value=np.zeros(4, dtype=np.float32),
        abs_vp=np.array([0.2, -0.1, 0.4, 0.0], dtype=np.float32),
        winner=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        econ=np.zeros(2, dtype=np.float32),
        snapshot=bytes(state.snapshot()),
        teacher_canonical=teacher,
    )

    s_missing = Sample(
        pid=state.current_player_id,
        era=state.era,
        value=np.zeros(4, dtype=np.float32),
        abs_vp=0.0,
        winner=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        econ=np.zeros(2, dtype=np.float32),
        snapshot=bytes(state.snapshot()),
        teacher_canonical=teacher,
    )

    s_valid = selfplay.materialize_sample(s_valid)
    s_missing = selfplay.materialize_sample(s_missing)

    b_valid = _to_batch([s_valid])
    assert b_valid["abs_vp_mask"].tolist() == [True]
    losses_valid = compute_loss(b_valid, net, l2=0.0, device="cpu", abs_vp_lambda=1.0)
    abs_vp_loss_valid = losses_valid[7]
    assert abs_vp_loss_valid.item() > 0.0

    b_missing = _to_batch([s_missing])
    assert b_missing["abs_vp_mask"].tolist() == [False]
    losses_missing = compute_loss(b_missing, net, l2=0.0, device="cpu", abs_vp_lambda=1.0)
    abs_vp_loss_missing = losses_missing[7]
    assert abs_vp_loss_missing.item() == 0.0


def test_load_bin_shard(tmp_path):
    from brass_ai.selfplay import load_bin_shard
    # Create an in-memory GameState and dump a minimal valid imitation record shard
    state = be.GameState(seed=42, players=4)
    first, _, _ = state.choose_heuristic_round()
    shard_file = tmp_path / "test_shard.bin"
    records = [(
        0,
        0,
        [0.1, -0.2, 0.05, 0.05],
        [0.2, -0.1, 0.4, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [10.0, 30.0],
        bytes(state.snapshot()),
        first,
    )]
    be.dump_imitation_shard(str(shard_file), records)

    samples = load_bin_shard(shard_file)
    assert len(samples) == 1
    s0 = samples[0]
    assert s0.pid == 0
    assert s0.era == 0
    assert s0.abs_vp is not None
    assert np.allclose(s0.abs_vp, [0.2, -0.1, 0.4, 0.0])
    assert np.allclose(s0.value, [0.1, -0.2, 0.05, 0.05])
    assert np.allclose(s0.winner, [1.0, 0.0, 0.0, 0.0])
    assert np.allclose(s0.econ, [10.0, 30.0])
    assert s0.teacher_canonical == first
    recovered_state = be.GameState.from_snapshot(s0.snapshot)
    assert recovered_state.player_count == 4


def test_train_with_kl_and_sample_weights():
    torch.manual_seed(42)
    net = PolicyValueNet()
    state1 = be.GameState(seed=11, players=4)
    first1, _, _ = state1.choose_heuristic_round()
    s1 = Sample(
        pid=state1.current_player_id,
        era=state1.era,
        value=np.zeros(4, dtype=np.float32),
        abs_vp=np.zeros(4, dtype=np.float32),
        winner=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        econ=np.zeros(2, dtype=np.float32),
        snapshot=bytes(state1.snapshot()),
        teacher_canonical=first1,
        weight=2.0,
        anchor_probs=np.ones(10, dtype=np.float32) / 10.0,
    )
    state2 = be.GameState(seed=12, players=4)
    first2, _, _ = state2.choose_heuristic_round()
    s2 = Sample(
        pid=state2.current_player_id,
        era=state2.era,
        value=np.zeros(4, dtype=np.float32),
        abs_vp=np.zeros(4, dtype=np.float32),
        winner=np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32),
        econ=np.zeros(2, dtype=np.float32),
        snapshot=bytes(state2.snapshot()),
        teacher_canonical=first2,
        weight=0.5,
        anchor_probs=None,
    )
    mat_s1 = selfplay.materialize_sample(s1)
    mat_s2 = selfplay.materialize_sample(s2)
    assert mat_s1.weight == 2.0
    assert mat_s2.weight == 0.5

    # Test heterogeneous batch with partial anchor
    b = _to_batch([mat_s1, mat_s2])
    assert "weight" in b
    assert "anchor_probs" in b
    assert b["anchor_mask"].tolist() == [True, False]
    assert np.allclose(b["weight"], [2.0, 0.5])

    # Test loss computation with KL
    losses = compute_loss(b, net, l2=0.0, device="cpu", kl_lambda=0.1)
    kl_val = losses[8]
    assert kl_val.item() >= 0.0

    # Verify that weighting actually shifts policy loss
    b_heavy_s1 = dict(b, weight=torch.tensor([100.0, 0.01]))
    b_heavy_s2 = dict(b, weight=torch.tensor([0.01, 100.0]))
    loss_s1 = compute_loss(b_heavy_s1, net, l2=0.0, device="cpu")[1]
    loss_s2 = compute_loss(b_heavy_s2, net, l2=0.0, device="cpu")[1]
    assert not torch.isclose(loss_s1, loss_s2)

    # Test trainer one epoch
    trainer = Trainer(net, TrainConfig(device="cpu", batch_size=2, kl_lambda=0.1))
    epoch_losses = trainer.train_one_epoch([mat_s1, mat_s2])
    assert len(epoch_losses) > 0
    assert "kl" in epoch_losses[0]

