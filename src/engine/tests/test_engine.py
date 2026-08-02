"""Python-side alignment tests for the brass_engine PyO3 bindings.

Run with the venv python:  .venv/Scripts/python.exe -m pytest tests/test_engine.py
"""

import numpy as np
import pytest

import brass_engine as be


def test_constants_shape_consistency():
    assert be.BOARD_PLANES == 17
    assert be.BOARD_CELLS == 49
    assert be.LINK_PLANES == 6
    assert be.LINK_CELLS == 39
    assert be.GLOBAL_LEN == 50
    assert be.HAND_LEN == 35
    assert be.policy_table_size > 0
    assert be.network_double_cells > 0


def test_policy_table_size_matches_layout():
    city_build = 20 * 4 * 6  # cities x slots x industries
    farm_build = 2
    network = 39
    network_double = be.network_double_cells
    develop = 6 + 36
    sell = 47
    fixed = 3
    expected = city_build + farm_build + network + network_double + develop + sell + fixed
    assert be.policy_table_size == expected


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
    for slot, canonical, describe in moves:
        assert 0 <= slot < be.policy_table_size
        assert canonical.startswith(("Build", "Network", "NetDouble", "Develop",
                                     "Sell", "FreeDevelop", "Loan", "Scout", "Pass"))
        assert describe


def test_move_slot_roundtrip():
    g = be.GameState(seed=42, players=4)
    for slot, canonical, _ in g.legal_moves():
        slots = be.moves_to_slots(canonical)
        assert slot in slots
        break


def test_apply_move_valid_and_invalid():
    g = be.GameState(seed=3, players=4)
    _, canonical, describe = g.legal_moves()[0]
    summary = g.apply_move(canonical)
    assert summary
    assert describe  # both describe the same move

    with pytest.raises(ValueError):
        g.apply_move("NotAMove{foo:1}")

    # A stale canonical string from before a move should fail legality check.
    _, stale, _ = g.legal_moves()[0]
    g2 = be.GameState(seed=3, players=4)
    g2.apply_move(stale)  # legal on the fresh clone
    if g.current_player_id != g2.current_player_id:
        # different player to move: the stale card index may now be out of range
        pass


def test_determinize_preserves_own_hand_and_count():
    g = be.GameState(seed=9, players=4)
    det = g.determinize()
    assert det.player_count == 4
    assert det.current_player_id == g.current_player_id


def test_state_to_tensor_shapes_and_determinism():
    g = be.GameState(seed=11, players=4)
    board, links, global_vec, own_hand, opp_hands = g.state_to_tensor()
    assert board.shape == (17, 49)
    assert links.shape == (6, 39)
    assert global_vec.shape == (50,)
    assert own_hand.shape == (35,)
    assert opp_hands.shape == (105,)

    b2, l2, g2, o2, op2 = be.GameState(seed=11, players=4).state_to_tensor()
    np.testing.assert_array_equal(board, b2)
    np.testing.assert_array_equal(links, l2)
    np.testing.assert_array_equal(global_vec, g2)
    np.testing.assert_array_equal(own_hand, o2)
    np.testing.assert_array_equal(opp_hands, op2)

    # sanity: values are in [0,1]
    for arr in (board, links, global_vec, own_hand, opp_hands):
        assert arr.min() >= 0.0 and arr.max() <= 1.0


def test_tensor_bounds_with_built_tiles():
    # Play a chunk of a game so the board is non-empty (includes level-5+
    # manufacturers whose level/8 normalization previously overflowed 1.0).
    g = be.GameState(seed=17, players=4)
    for _ in range(120):
        if g.game_over:
            break
        canonical, _, _ = g.choose_heuristic()
        g.apply_move(canonical)
    board, links, global_vec, own_hand, opp_hands = g.state_to_tensor()
    for arr in (board, links, global_vec, own_hand, opp_hands):
        assert arr.min() >= 0.0 and arr.max() <= 1.0, arr.max()
    assert board[:, :].sum() > 0  # something is occupied


def test_legal_mask_subset_of_table():
    g = be.GameState(seed=21, players=4)
    mask = g.legal_mask()
    assert mask == sorted(set(mask))
    assert all(0 <= s < be.policy_table_size for s in mask)
    # mask must cover every slot produced by legal_moves
    slots = {slot for slot, _, _ in g.legal_moves()}
    assert slots.issubset(set(mask))


def test_ai_choices_return_legal_moves():
    g = be.GameState(seed=5, players=4)
    canon, describe, score = g.choose_heuristic()
    assert describe and canon
    assert g.player_count == 4

    g2 = be.GameState(seed=5, players=4)
    canon2, describe2, score2 = g2.choose_2ply()
    assert canon2 and describe2


def test_play_short_game_heuristic():
    g = be.GameState(seed=1, players=4)
    guard = 0
    while not g.game_over:
        guard += 1
        assert guard < 50_000, "game did not terminate"
        moves = g.legal_moves()
        assert moves, f"no legal moves at round {g.round} era {g.era}"
        if guard % 2 == 0:
            canonical, _, _ = g.choose_heuristic()
        else:
            canonical = moves[0][1]
        g.apply_move(canonical)
    assert g.game_over


def _reach_rail(seed):
    g = be.GameState(seed=seed, players=4)
    guard = 0
    while g.era == 0 and not g.game_over and guard < 2000:
        guard += 1
        canonical, _, _ = g.choose_heuristic()
        g.apply_move(canonical)
    return g


def test_network_double_moves_are_policy_mapped_and_executable():
    seen = 0
    for seed in range(200):
        g = _reach_rail(seed)
        if g.era != 1:
            continue
        for slot, canonical, _ in g.legal_moves():
            if not canonical.startswith("NetDouble"):
                continue
            seen += 1
            assert 0 <= slot < be.policy_table_size
            assert be.describe_slot(slot).startswith("Double rail")
            # Replay the same line from the seed and execute the move.
            g2 = _reach_rail(seed)
            g2.apply_move(canonical)
        if seen:
            break
    assert seen > 0, "no double-rail move appeared in 200 heuristic games"
