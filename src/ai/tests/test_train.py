import numpy as np
import pytest
import torch

from brass_ai.net import PolicyValueNet
from brass_ai import selfplay
from brass_ai.selfplay import generate_imitation_samples
from brass_ai.train import TrainConfig, Trainer, _to_batch, compute_loss


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
        pl_before, _, _, _ = compute_loss(b, net, 0.0, "cpu")

    trainer.train_on_samples(samples[:])

    # Trainer must keep its optimizer across calls (state persists).
    assert trainer.optimizer.state, "optimizer should have state after a step"
    before_lr = trainer.current_lr()
    trainer.train_on_samples(samples[:])  # second call reuses the same optimizer
    assert trainer.epoch_count == 2 * cfg.epochs

    # Loss on the training data should drop after fitting.
    net.eval()
    with torch.no_grad():
        pl_after, _, _, _ = compute_loss(b, net, 0.0, "cpu")
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


def test_imitation_quality_filter_retries_until_it_has_requested_games(monkeypatch):
    seen_seeds = []

    def fake_game(args):
        seed, _, _ = args
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
