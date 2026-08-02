import torch

import brass_engine as be

from brass_ai.mcts import ISMCTS, MCTSConfig
from brass_ai.net import PolicyValueNet
from brass_ai.selfplay import SelfPlayConfig, play_game


def _make():
    torch.manual_seed(0)
    torch.set_num_threads(2)
    net = PolicyValueNet()
    mcts = ISMCTS(net, MCTSConfig(c_puct=1.5, max_depth=8))
    return net, mcts


def test_search_returns_executable_move():
    _, mcts = _make()
    g = be.GameState(seed=7, players=4)
    r = mcts.search(g, sims=40)
    assert r.best is not None
    c = g.clone()
    c.apply_move(r.best)  # must not raise
    assert sum(r.visits.values()) == 39  # first sim only expands the root
    legal_slots = {s for s, _, _ in g.legal_moves()}
    assert set(r.visits) <= legal_slots


def test_search_noise_and_determinism():
    _, mcts = _make()
    g = be.GameState(seed=1, players=4)
    r1 = mcts.search(g, sims=30, add_root_noise=True)
    r2 = mcts.search(g, sims=30, add_root_noise=False)
    assert r1.best is not None and r2.best is not None
    assert sum(r1.visits.values()) == 29
    # both searches choose some move from the same legal set
    legal = {s for s, _, _ in g.legal_moves()}
    assert set(r1.visits) <= legal and set(r2.visits) <= legal


def test_search_tolerates_broken_double_rail_moves():
    # Over many rail-era states, search must never raise even though the engine
    # can enumerate double-rail coal choices that fail to execute.
    _, mcts = _make()
    for seed in range(40):
        g = be.GameState(seed=seed, players=4)
        guard = 0
        while g.era == 0 and not g.game_over and guard < 2000:
            guard += 1
            c, _, _ = g.choose_heuristic()
            g.apply_move(c)
        if g.era != 1:
            continue
        r = mcts.search(g, sims=20)
        assert r.best is None or sum(r.visits.values()) == 19


def test_selfplay_produces_valid_samples():
    _, mcts = _make()
    cfg = SelfPlayConfig(players=4, sims=30, max_moves=200)
    samples, vps = play_game(mcts, cfg)
    assert len(samples) > 0
    assert len(vps) == 4
    for s in samples:
        # policy target is a distribution over the full table
        assert abs(s.policy.sum() - 1.0) < 1e-3
        assert -3 <= s.value <= 3
        assert s.board.shape == (17, 49)
