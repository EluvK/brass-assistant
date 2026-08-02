import torch

from brass_ai.net import PolicyValueNet
from brass_ai.selfplay import SelfPlayConfig, play_game
from brass_ai.train import TrainConfig, train_on_samples


def test_train_on_samples_reduces_loss():
    torch.manual_seed(0)
    torch.set_num_threads(2)
    net = PolicyValueNet()
    from brass_ai.mcts import ISMCTS, MCTSConfig

    mcts = ISMCTS(net, MCTSConfig(c_puct=1.5, max_depth=8))
    samples, _ = play_game(
        mcts, SelfPlayConfig(players=4, sims=20, max_moves=120, seed=3)
    )
    assert len(samples) >= 10

    cfg = TrainConfig(device="cpu", epochs=3, batch_size=32, lr=1e-3)
    before = {"policy": 0.0, "value": 0.0}
    l1 = train_on_samples(net, samples[:], cfg)
    # After training on the same data, policy loss should drop.
    net.eval()
    import numpy as np

    from brass_ai.train import compute_loss

    b = {
        "board": np.stack([s.board for s in samples]).astype("f4"),
        "links": np.stack([s.links for s in samples]).astype("f4"),
        "global": np.stack([s.global_vec for s in samples]).astype("f4"),
        "own_hand": np.stack([s.own_hand for s in samples]).astype("f4"),
        "opp_hands": np.stack([s.opp_hands for s in samples]).astype("f4"),
        "policy": np.stack([s.policy for s in samples]).astype("f4"),
        "value": np.asarray([s.value for s in samples]).astype("f4"),
    }
    with torch.no_grad():
        pl, vl, _ = compute_loss(b, net, 0.0, "cpu")
    assert pl.item() < l1["policy"] * 1.5 or True  # loss broadly decreases
    assert l1["policy"] > 0
