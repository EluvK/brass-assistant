"""Python-side alignment tests for the brass_ai._engine PyO3 bindings.

Run from the repository root with:
    .venv/Scripts/python.exe -m pytest python/tests/test_engine.py
"""

import numpy as np
import pytest

from brass_ai import _engine as be


def test_constants_shape_consistency():
    assert be.BOARD_CELLS == 49
    assert be.LINK_CELLS == 39
    assert be.MERCHANT_COUNT == 9
    assert be.SEAT_COUNT == 4
    assert be.TOKEN_COUNT == 102
    assert be.STATE_TOKEN_SCHEMA_VERSION == 2
    assert be.ACTION_SCHEMA_VERSION == 1
    assert be.ACTION_FEATURE_DIM == 3 + be.ACTION_NUMBERS + 3 * be.ACTION_REF_CAP
    assert len(be.BOARD_CELL_LOCATIONS) == be.BOARD_CELLS
    assert len(be.BOARD_CELL_SLOTS) == be.BOARD_CELLS
    assert len(be.CONNECTION_ENDPOINTS) == be.LINK_CELLS * 2
    assert len(be.CONNECTION_VIA_FARMS) == be.LINK_CELLS
    assert len(be.MERCHANT_LOCATIONS) == be.MERCHANT_COUNT
    assert be.F_MERCHANT == 15
    assert be.MERCHANT_BUY == 0
    assert be.MERCHANT_BEER == 5
    assert be.MERCHANT_BONUS == 6
    assert be.MERCHANT_IN_NET == 10
    assert be.MERCHANT_DIST == 14


def test_new_state_basics():
    g = be.GameState(seed=1, players=4)
    assert g.player_count == 4
    assert g.era == 0  # canal
    assert g.round == 1
    assert not g.game_over
    assert 0 <= g.current_player_id < 4
    assert g.current_player_money > 0


def test_bad_players_rejected():
    with pytest.raises(ValueError):
        be.GameState(seed=1, players=5)
    with pytest.raises(ValueError):
        be.GameState(seed=1, players=1)


def test_legal_moves_structure():
    g = be.GameState(seed=7, players=4)
    moves = g.legal_moves()
    assert moves
    for action_id, canonical, describe in moves:
        assert action_id >= 0
        assert canonical.startswith("ResolvedMove{operation:")
        assert describe


def test_move_codec_roundtrip():
    g = be.GameState(seed=42, players=4)
    for _, canonical, _ in g.legal_moves():
        copy = g.clone()
        copy.apply_move(canonical)
        break


def test_apply_move_valid_and_invalid():
    g = be.GameState(seed=3, players=4)
    _, canonical, describe = g.legal_moves()[0]
    summary = g.apply_move(canonical)
    assert summary
    assert describe  # both describe the same move

    with pytest.raises(ValueError):
        g.apply_move("NotAMove{foo:1}")


def test_determinize_preserves_own_hand_and_count():
    g = be.GameState(seed=9, players=4)
    det = g.determinize()
    assert det.player_count == 4
    assert det.current_player_id == g.current_player_id


def test_state_tokens_shapes_and_determinism():
    g = be.GameState(seed=11, players=4)
    cells, links, merchants, seats, global_vec = g.state_tokens()
    assert cells.shape == (49, be.F_CELL)
    assert links.shape == (39, be.F_LINK)
    assert merchants.shape == (9, be.F_MERCHANT)
    assert seats.shape == (4, be.F_SEAT)
    assert global_vec.shape == (be.F_GLOBAL,)

    again = be.GameState(seed=11, players=4).state_tokens()
    for a, b in zip((cells, links, merchants, seats, global_vec), again):
        np.testing.assert_array_equal(a, b)

    # Connection 1 is rail-only; its static legality must be visible even
    # before any player builds it.
    assert links[1, 0] == 0.0  # canal-buildable
    assert links[1, 1] == 1.0  # rail-buildable
    assert links[1, 3] == 0.0  # not built yet

    # Seat 0 is always the acting player.
    assert seats[0, be.SEAT_IS_CURRENT] == 1.0
    for other in range(1, be.SEAT_COUNT):
        assert seats[other, be.SEAT_IS_CURRENT] == 0.0

    # Every group is non-negative and finite.
    for arr in (cells, links, merchants, seats, global_vec):
        assert arr.min() >= 0.0 and np.isfinite(arr).all()


def test_merchant_token_semantics():
    # Verify spatial merchant locations are within location bounds and mapped correctly
    assert len(be.MERCHANT_LOCATIONS) == be.MERCHANT_COUNT
    for loc in be.MERCHANT_LOCATIONS:
        assert 0 <= loc < be.LOCATION_COUNT
    # Shrewsbury(1), Gloucester(2), Oxford(2), Warrington(2), Nottingham(2)
    assert be.MERCHANT_LOCATIONS[0] == 20
    assert be.MERCHANT_LOCATIONS[1] == be.MERCHANT_LOCATIONS[2] == 21
    assert be.MERCHANT_LOCATIONS[3] == be.MERCHANT_LOCATIONS[4] == 22
    assert be.MERCHANT_LOCATIONS[5] == be.MERCHANT_LOCATIONS[6] == 23
    assert be.MERCHANT_LOCATIONS[7] == be.MERCHANT_LOCATIONS[8] == 24

    # 4-player game at start
    g4 = be.GameState(seed=123, players=4)
    _, _, merchants4, _, _ = g4.state_tokens()

    # 1. Check merchant bonus normalization
    # Shrewsbury (slot 0): 4 VP -> 4.0 / 10.0 = 0.4
    np.testing.assert_allclose(merchants4[0, be.MERCHANT_BONUS:be.MERCHANT_BONUS + 4], [0.4, 0.0, 0.0, 0.0])
    # Gloucester (slot 1, 2): 1 Develop -> 1.0 / 2.0 = 0.5
    for s in (1, 2):
        np.testing.assert_allclose(merchants4[s, be.MERCHANT_BONUS:be.MERCHANT_BONUS + 4], [0.0, 0.0, 0.0, 0.5])
    # Oxford (slot 3, 4): 2 Income -> 2.0 / 10.0 = 0.2
    for s in (3, 4):
        np.testing.assert_allclose(merchants4[s, be.MERCHANT_BONUS:be.MERCHANT_BONUS + 4], [0.0, 0.0, 0.2, 0.0])
    # Warrington (slot 5, 6): 5 Money -> 5.0 / 20.0 = 0.25
    for s in (5, 6):
        np.testing.assert_allclose(merchants4[s, be.MERCHANT_BONUS:be.MERCHANT_BONUS + 4], [0.0, 0.25, 0.0, 0.0])
    # Nottingham (slot 7, 8): 3 VP -> 3.0 / 10.0 = 0.3
    for s in (7, 8):
        np.testing.assert_allclose(merchants4[s, be.MERCHANT_BONUS:be.MERCHANT_BONUS + 4], [0.3, 0.0, 0.0, 0.0])

    # 2. Check initial network membership and distance
    # At start, no network exists for any player
    for s in range(be.MERCHANT_COUNT):
        assert np.all(merchants4[s, be.MERCHANT_IN_NET:be.MERCHANT_IN_NET + 4] == 0.0)
        assert merchants4[s, be.MERCHANT_DIST] == 1.0
        # Buy type is valid one-hot
        assert np.sum(merchants4[s, be.MERCHANT_BUY:be.MERCHANT_BUY + 5]) == 1.0

    # 3. 2-player game: slots 5..8 are inactive
    g2 = be.GameState(seed=123, players=2)
    _, _, merchants2, _, _ = g2.state_tokens()
    # Inactive slots must preserve dist == 1.0 (not inverted to 0.0)
    for s in range(be.MERCHANT_COUNT):
        assert merchants2[s, be.MERCHANT_DIST] == 1.0
    for s in range(5, be.MERCHANT_COUNT):
        assert np.all(merchants2[s, be.MERCHANT_BUY:be.MERCHANT_BUY + 5] == 0.0)
        assert merchants2[s, be.MERCHANT_BEER] == 0.0


def test_tensor_bounds_with_built_tiles():
    # Play a chunk of a game so the board is non-empty (includes level-5+
    # manufacturers whose level/8 normalization previously overflowed 1.0).
    g = be.GameState(seed=17, players=4)
    pending = []
    for _ in range(120):
        if g.game_over:
            break
        if pending:
            canonical = pending.pop(0)
        else:
            first, second, _ = g.choose_heuristic_round()
            canonical = first
            if second is not None:
                pending.append(second)
        g.apply_move(canonical)
    cells, links, merchants, seats, global_vec = g.state_tokens()
    for arr in (cells, links, merchants, seats, global_vec):
        assert arr.min() >= 0.0, arr.min()
        assert np.isfinite(arr).all()
    assert cells[:, be.CELL_OCCUPIED].sum() > 0  # something is occupied


def test_legal_candidates_are_complete_and_executable():
    g = be.GameState(seed=21, players=4)
    canonicals, features = g.legal_candidates()
    assert canonicals
    assert features.shape == (len(canonicals), be.ACTION_FEATURE_DIM)
    for i, canonical in enumerate(canonicals):
        assert features[i].shape == (be.ACTION_FEATURE_DIM,)
        g.clone().apply_move(canonical)


def test_snapshot_restores_state_and_full_legal_candidates():
    g = be.GameState(seed=29, players=4)
    pending = []
    for _ in range(8):
        if pending:
            canonical = pending.pop(0)
        else:
            first, second, _ = g.choose_heuristic_round()
            canonical = first
            if second is not None:
                pending.append(second)
        g.apply_move(canonical)
    restored = be.GameState.from_snapshot(g.snapshot())
    assert restored.current_player_id == g.current_player_id
    assert restored.era == g.era
    assert restored.round == g.round
    np.testing.assert_array_equal(restored.state_tokens()[0], g.state_tokens()[0])
    canonicals, features = g.legal_candidates()
    restored_canonicals, restored_features = restored.legal_candidates()
    assert restored_canonicals == canonicals
    np.testing.assert_array_equal(restored_features, features)


def test_ai_choices_return_legal_moves():
    g = be.GameState(seed=5, players=4)
    first, second, score = g.choose_heuristic_round()
    assert first
    assert g.player_count == 4


def test_play_short_game_heuristic():
    g = be.GameState(seed=1, players=4)
    guard = 0
    pending = []
    while not g.game_over:
        guard += 1
        assert guard < 50_000, "game did not terminate"
        moves = g.legal_moves()
        assert moves, f"no legal moves at round {g.round} era {g.era}"
        if guard % 2 == 0:
            if pending:
                canonical = pending.pop(0)
            else:
                first, second, _ = g.choose_heuristic_round()
                canonical = first
                if second is not None:
                    pending.append(second)
        else:
            canonical = moves[0][1]
        g.apply_move(canonical)
    assert g.game_over


def _reach_rail(seed):
    g = be.GameState(seed=seed, players=4)
    guard = 0
    pending = []
    while g.era == 0 and not g.game_over and guard < 2000:
        guard += 1
        if pending:
            canonical = pending.pop(0)
        else:
            first, second, _ = g.choose_heuristic_round()
            canonical = first
            if second is not None:
                pending.append(second)
        g.apply_move(canonical)
    return g


def test_network_double_moves_are_executable():
    seen = 0
    for seed in range(200):
        g = _reach_rail(seed)
        if g.era != 1:
            continue
        for _, canonical, _ in g.legal_moves():
            if not canonical.startswith("ResolvedMove{operation:NetDouble"):
                continue
            seen += 1
            # Replay the same line from the seed and execute the move.
            g2 = _reach_rail(seed)
            g2.apply_move(canonical)
        if seen:
            break
    assert seen > 0, "no double-rail move appeared in 200 heuristic games"
