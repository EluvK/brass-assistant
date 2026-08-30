import numpy as np
import pytest
import torch

from brass_ai.net import PolicyValueNet
from brass_ai import selfplay
from brass_ai.selfplay import Sample, _rank_targets, generate_imitation_samples
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
    checkpoint.pop("state_feature_schema_version")
    checkpoint.pop("state_feature_shapes")
    with pytest.raises(ValueError, match="state-feature schema"):
        trainer.load_state_dict(checkpoint)


def test_rank_target_uses_official_tiebreak_order():
    # Players 0 and 1 may have tied VP, but the engine's final ranking has
    # already resolved it by income then cash.
    rank, winner = _rank_targets([1, 0, 3, 2], 4)
    np.testing.assert_array_equal(rank, np.asarray([0.5, 0.25, 1.0, 0.75], dtype=np.float32))
    np.testing.assert_array_equal(winner, np.asarray([0.0, 1.0, 0.0, 0.0], dtype=np.float32))


def test_policy_evaluation_materializes_snapshot_batches():
    state = be.GameState(seed=73, players=4)
    teacher, _, _ = state.choose_heuristic()
    sample = Sample(
        pid=state.current_player_id, era=state.era,
        rank=np.zeros(4, dtype=np.float32), winner=np.zeros(4, dtype=np.float32),
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
