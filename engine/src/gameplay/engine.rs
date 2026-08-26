//! Turn/round management, era transitions, scoring.
//!
//! Translated from gameState.js advanceTurn / endRound / endCanalEra /
//! endGame / calculateEraScore / resolveShortfall.

use crate::data::Era;
use crate::map::*;
use crate::rules::ResolvedMove;
use crate::state::GameState;

/// The boundary reached while advancing the game clock.
///
/// Era cleanup is deliberately deferred to [`handle_turn_result`] so callers
/// can inspect or report the scored Canal-era board before it is cleared.
pub enum TurnResult {
    Continue,
    EndCanalEra,
    EndGame,
}

/// Record one completed action and advance to the next action or player.
///
/// Call this only after a successful action. Merchant bonuses, including a
/// free develop, are resolved atomically by that action before the clock
/// advances. Normal game states always give the active player a card to play;
/// an empty hand is not treated as an implicit pass.
pub fn advance_turn(state: &mut GameState) -> TurnResult {
    state.actions_this_turn += 1;

    // Era may end mid-turn when all hands empty and deck exhausted.
    let all_hands_empty = state.players.iter().all(|p| p.hand.is_empty());
    if all_hands_empty && state.deck.is_empty() {
        return end_round(state);
    }

    if state.actions_this_turn >= state.actions_per_turn {
        // Draw cards for current player, move to next.
        let pid = state.current_player_id();
        state.draw_cards(pid);
        state.actions_this_turn = 0;
        state.current_index += 1;
        if state.current_index >= state.player_count() {
            state.current_index = 0;
            match end_round(state) {
                TurnResult::Continue => {}
                other => return other,
            }
        }
    }

    TurnResult::Continue
}

/// Resolve the round boundary: income, turn order, and first-round setup.
///
/// This function reports an era/game boundary but intentionally does not
/// mutate the board for the next era; see [`handle_turn_result`].
pub fn end_round(state: &mut GameState) -> TurnResult {
    let all_hands_empty = state.players.iter().all(|p| p.hand.is_empty());
    let era_ending = all_hands_empty && state.deck.is_empty();
    let is_final_round = era_ending && state.era == Era::Rail;

    // Income phase — not collected on the final round of the game.
    if !is_final_round {
        let incomes: Vec<i8> = state.players.iter().map(|p| p.income_level()).collect();
        for (pid, level) in incomes.iter().enumerate() {
            state.players[pid].money += *level as i32;
            if state.players[pid].money < 0 {
                resolve_shortfall(state, pid);
            }
        }
    }

    // Turn order: least money spent goes first; ties keep current order.
    let mut order: Vec<usize> = state.turn_order.clone();
    order.sort_by(|a, b| {
        let sa = state.money_spent_this_round[*a];
        let sb = state.money_spent_this_round[*b];
        sa.cmp(&sb)
    });
    // Stable sort keeps relative order for ties (already stable in Rust for slices)
    state.turn_order = order;
    for m in state.money_spent_this_round.iter_mut() {
        *m = 0;
    }

    state.round += 1;
    if state.is_first_round {
        state.is_first_round = false;
        state.actions_per_turn = ACTIONS_PER_TURN;
    }

    return match (era_ending, is_final_round) {
        (true, true) => TurnResult::EndGame,
        (true, false) => TurnResult::EndCanalEra,
        _ => TurnResult::Continue,
    };

    // if era_ending {
    //     return if state.era == Era::Canal {
    //         TurnResult::EndCanalEra
    //     } else {
    //         TurnResult::EndGame
    //     };
    // }
    // TurnResult::Continue
}

/// Apply the income-phase bankruptcy rule for one player.
///
/// Tiles are removed in the rule-defined priority, then any remaining deficit
/// is paid with VP. The player always leaves this function with non-negative
/// money.
fn resolve_shortfall(state: &mut GameState, pid: usize) {
    // Give up unflipped low-VP tiles first.
    let mut candidates: Vec<(usize, bool)> = Vec::new(); // (key, is_farm)
    for (k, t) in state.city_tiles.iter().enumerate() {
        if let Some(t) = t {
            if t.player == pid {
                candidates.push((k, false));
            }
        }
    }
    for (idx, t) in state.farm_tiles.iter().enumerate() {
        if let Some(t) = t {
            if t.player == pid {
                candidates.push((idx, true));
            }
        }
    }
    candidates.sort_by_key(|(k, is_farm)| {
        let tile = if *is_farm {
            &state.farm_tiles[*k]
        } else {
            &state.city_tiles[*k]
        };
        let t = tile.as_ref().unwrap();
        (if t.flipped { 1 } else { 0 }, t.def.vp)
    });

    for (key, is_farm) in candidates {
        if state.players[pid].money >= 0 {
            break;
        }
        let tile = if is_farm {
            state.farm_tiles[key].as_ref().unwrap()
        } else {
            state.city_tiles[key].as_ref().unwrap()
        };
        let refund = tile.def.cost / 2;
        if is_farm {
            state.remove_farm_by_idx(key);
        } else {
            state.remove_city_tile_by_key(key);
        }
        state.players[pid].money += refund;
    }

    if state.players[pid].money < 0 {
        let deficit = -state.players[pid].money;
        let vp_loss = (state.players[pid].vp as i32).min(deficit);
        state.players[pid].vp -= vp_loss as u16;
        state.players[pid].money = 0;
    }
}

/// Score the Canal era and construct the initial Rail-era state.
///
/// This is the destructive half of the Canal-to-Rail transition and must run
/// only after [`advance_turn`] or [`end_round`] returns `EndCanalEra`.
fn end_canal_era(state: &mut GameState) {
    // Score canal era, remove canal links, remove Level I tiles, reshuffle.
    crate::scoring::score_era(state);

    for (_id, link) in state.links.iter_mut().enumerate() {
        if let Some(l) = link {
            if l.is_canal {
                state.players[l.player].canal_links += 1;
                *link = None;
            }
        }
    }

    for t in state.city_tiles.iter_mut() {
        if let Some(tile) = t {
            if tile.def.level == 1 {
                *t = None;
            }
        }
    }
    for t in state.farm_tiles.iter_mut() {
        if let Some(tile) = t {
            if tile.def.level == 1 {
                *t = None;
            }
        }
    }

    // Bulk board clears above bypassed the per-tile cache hooks; rescan.
    state.rebuild_free_sources();
    state.rebuild_network_masks();

    state.era = Era::Rail;
    state.round = 1;
    state.is_first_round = false;
    state.actions_per_turn = ACTIONS_PER_TURN;
    state.current_index = 0;
    state.actions_this_turn = 0;

    // Reset merchant beer.
    for mt in state.merchants.iter_mut() {
        mt.has_beer = mt.buys != crate::state::BuyType::Blank;
    }

    // Return wild cards, clear hands (non-wilds go to the discard pile).
    // All canal-era cards re-enter circulation via `init_deck` below, so both
    // the anonymous discard pile and the per-player played history are reset
    // (they only describe the current era).
    for p in state.players.iter_mut() {
        for card in p.hand.drain(..) {
            match card.ctype() {
                crate::data::CardType::WildLocation => {
                    state.wild_location_pile += 1;
                    p.has_wild_location = false;
                }
                crate::data::CardType::WildIndustry => {
                    state.wild_industry_pile += 1;
                    p.has_wild_industry = false;
                }
                _ => state.discard_pile.push(card),
            }
        }
        p.played.clear();
    }
    state.discard_pile.clear();

    // Rebuild deck, deal new hands.
    // User rulebook ruling: the Rail Era does NOT repeat the per-player
    // face-down seeded discard done during initial setup, so all 64 cards enter
    // the action loop in 4p Rail (full 8 rounds).
    state.init_deck();
    state.deal_cards();
}

/// Apply the era-end / game-end transitions reported by `advance_turn` /
/// `end_round` to a state (no-op for `Continue`).
pub fn handle_turn_result(state: &mut GameState, tr: TurnResult) {
    match tr {
        TurnResult::Continue => {}
        TurnResult::EndCanalEra => end_canal_era(state),
        TurnResult::EndGame => {
            crate::scoring::score_era(state);
            state.game_over = true;
        }
    }
}

/// Convenience API for a complete action: apply it, then advance the clock.
///
/// The caller still owns the returned era/game transition and may pass it to
/// [`handle_turn_result`] immediately.
pub fn step(state: &mut GameState, mv: &ResolvedMove) -> (Result<String, String>, TurnResult) {
    let result = crate::rules::apply_move(state, mv);
    let tr = if result.is_ok() {
        advance_turn(state)
    } else {
        TurnResult::Continue
    };
    (result, tr)
}
