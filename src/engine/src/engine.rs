//! Turn/round management, era transitions, scoring.
//!
//! Translated from gameState.js advanceTurn / endRound / endCanalEra /
//! endGame / calculateEraScore / resolveShortfall.

use crate::data::Era;
use crate::map::*;
use crate::rules::Move;
use crate::state::GameState;
use rand::Rng;

pub enum TurnResult {
    Continue,
    EndCanalEra,
    EndGame,
}

pub fn advance_turn(state: &mut GameState<impl Rng>) -> TurnResult {
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
        // Skip players with no cards
        if let Some(r) = skip_empty_hand_players(state) {
            return r;
        }
    } else {
        // If current player has no cards, end their turn early.
        let pid = state.current_player_id();
        if state.players[pid].hand.is_empty() {
            state.draw_cards(pid);
            if state.players[pid].hand.is_empty() {
                state.actions_this_turn = 0;
                state.current_index += 1;
                if state.current_index >= state.player_count() {
                    state.current_index = 0;
                    match end_round(state) {
                        TurnResult::Continue => {}
                        other => return other,
                    }
                }
                if let Some(r) = skip_empty_hand_players(state) {
                    return r;
                }
            }
        }
    }

    TurnResult::Continue
}

fn skip_empty_hand_players(state: &mut GameState<impl Rng>) -> Option<TurnResult> {
    let mut safety = 0;
    while {
        let pid = state.current_player_id();
        state.players[pid].hand.is_empty() && state.deck.is_empty()
    } && safety < state.player_count()
    {
        state.current_index += 1;
        safety += 1;
        if state.current_index >= state.player_count() {
            state.current_index = 0;
            match end_round(state) {
                TurnResult::Continue => {}
                other => return Some(other),
            }
        }
    }
    None
}

pub fn end_round(state: &mut GameState<impl Rng>) -> TurnResult {
    let all_hands_empty = state.players.iter().all(|p| p.hand.is_empty());
    let era_ending = all_hands_empty && state.deck.is_empty();
    let is_final_round = era_ending && state.era == Era::Rail;

    // Income phase — not collected on the final round of the game.
    if !is_final_round {
        let incomes: Vec<i8> = state
            .players
            .iter()
            .map(|p| p.income_level())
            .collect();
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

    if era_ending {
        return if state.era == Era::Canal {
            TurnResult::EndCanalEra
        } else {
            TurnResult::EndGame
        };
    }
    TurnResult::Continue
}

fn resolve_shortfall(state: &mut GameState<impl Rng>, pid: usize) {
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
            state.farm_tiles[key] = None;
        } else {
            state.city_tiles[key] = None;
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

pub fn end_canal_era(state: &mut GameState<impl Rng>) {
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
            if !tile.def.rail_era {
                *t = None;
            }
        }
    }
    for t in state.farm_tiles.iter_mut() {
        if let Some(tile) = t {
            if !tile.def.rail_era {
                *t = None;
            }
        }
    }

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

    // Return wild cards, clear hands.
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
                _ => {}
            }
        }
    }

    // Rebuild deck, deal new hands.
    state.init_deck();
    state.deal_cards();
    state.seed_discard_piles();
}

pub fn end_game(state: &mut GameState<impl Rng>) {
    crate::scoring::score_era(state);
    state.game_over = true;
}

/// Apply a move to the current player, then advance the turn.
/// Returns the result of the action + turn advancement.
pub fn step(state: &mut GameState<impl Rng>, mv: &Move) -> (Result<String, String>, TurnResult) {
    let result = crate::rules::apply_move(state, mv);
    let tr = advance_turn(state);
    (result, tr)
}

/// Force-pass for an empty-handed current player (no legal moves).
pub fn force_pass(state: &mut GameState<impl Rng>) -> TurnResult {
    advance_turn(state)
}

pub fn is_game_over(state: &GameState<impl Rng>) -> bool {
    state.game_over
}
