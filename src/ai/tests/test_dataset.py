import numpy as np

from brass_ai.dataset import load_samples, save_samples
from brass_ai.selfplay import generate_imitation_samples


def test_replay_shard_roundtrip(tmp_path):
    original = generate_imitation_samples(1, max_moves=600)
    path = tmp_path / "iter-00000.npz"
    assert save_samples(path, original) == len(original)
    restored = load_samples(path)
    assert len(restored) == len(original)
    np.testing.assert_array_equal(restored[0].policy, original[0].policy)
    np.testing.assert_array_equal(restored[-1].legal, original[-1].legal)
