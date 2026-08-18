//! Heuristic AI baseline: translated from reference/npow-brass-birmingham/js/aiPlayer.js
//!
//! Heuristic policy and candidate evaluator translated from the reference AI.
//! The public action policy uses deterministic 2-ply lookahead over the scored
//! candidate set; candidate scores remain available to MCTS and training.

use crate::data::{Era, IndustryType};
use crate::engine::{TurnResult, advance_turn};
use crate::graph::{
    connected_locations, find_beer_sources, find_coal_sources, find_iron_sources, is_in_network,
};
use crate::map::{ALL_LOCATIONS, CITY_COUNT, Loc, city_slots, connections};
use crate::rules::{
    BuildTarget, Move, can_develop, can_scout, execute_sell, get_valid_build_targets,
    get_valid_network_targets, get_valid_second_rail_links, get_valid_sell_targets, legal_moves,
    valid_build_cards,
};
use crate::state::{Card, GameState, PendingBonus, city_slot_offsets};

// Scoring weights (from aiPlayer.js).
const VP_WEIGHT: f64 = 1.0;
pub(super) const BASE_MONEY_WEIGHT: f64 = 0.12;
pub(super) const BASE_INCOME_WEIGHT: f64 = 0.25;
const FLEX_WEIGHT: f64 = 0.8;

// Hard policy constraints (temporary strategic guardrails requested by user).
const BAN_BUILD_LV1_BREWERY: bool = true;
const BAN_DEVELOP_IRON_LV2_PLUS: bool = true;
const BAN_DEVELOP_BREWERY_LV2_PLUS_AT_CANAL_EARLY: bool = true;

mod plan;
pub use plan::{EraProfile, Phase, Plan, compute_plan, era_phase, era_profile};

fn develop_guardrail_penalty(ind: IndustryType, level: u8) -> f64 {
    if level < 2 {
        return 0.0;
    }
    match ind {
        IndustryType::Brewery => 1.8 + 0.2 * (level.saturating_sub(2)) as f64,
        IndustryType::CoalMine => 1.5 + 0.2 * (level.saturating_sub(2)) as f64,
        _ => 0.0,
    }
}

pub struct Decision {
    pub mv: Move,
    pub score: f64,
}

mod lookahead;
pub use lookahead::choose_action;

/// Top-K candidates per action type, for MCTS to get a wider prior.
/// Build and Network get up to `k` candidates each; other action types keep
/// their single best (develop/sell/loan/scout/pass).
pub fn candidate_actions_k(state: &mut GameState, k: usize) -> Vec<Decision> {
    // Network masks are (re)validated inside `get_valid_build_targets` /
    // `get_valid_network_targets` / `get_second_rail_options`, which every
    // scoring path below enters before any direct `is_in_network` read — no
    // redundant ensure here (hot path: MCTS calls this once per tree node).
    let pid = state.current_player_id();

    if let Some(PendingBonus::FreeDevelop { player_id, count }) = state.pending_bonus {
        if player_id != pid {
            return Vec::new();
        }
        return score_pending_free_develop(state, pid, count)
            .into_iter()
            .collect();
    }

    let mut out = Vec::new();
    let plan = compute_plan(state, pid);

    for d in score_top_builds(state, pid, k, &plan) {
        if d.score != f64::NEG_INFINITY {
            out.push(d);
        }
    }
    out.extend(score_top_networks(state, pid, k, &plan));
    if let Some(d) = score_best_network_double(state, pid, &plan) {
        out.push(d);
    }
    if let Some(d) = score_develop_plan(state, pid, &plan) {
        out.push(d);
    }
    if let Some(d) = score_sell_plan(state, pid) {
        out.push(d);
    }
    if let Some(d) = score_loan_result(state, pid) {
        out.push(d);
    }
    if let Some(d) = score_scout_plan(state, pid) {
        out.push(d);
    }
    if let Some(d) = score_pass_result(state, pid) {
        out.push(d);
    }

    // Candidate pruning must never turn a position with executable actions
    // into a dead end (notably pending free-develop states filtered by a
    // strategic guardrail, or callers passing k=0). Keep one legal fallback
    // so MCTS always has a path to explore.
    if out.is_empty() {
        if let Some(mv) = legal_moves(state).into_iter().next() {
            out.push(Decision {
                mv,
                score: f64::NEG_INFINITY,
            });
        }
    }

    out
}

/// Fallback "pass" decision (safe even on an empty hand).
pub fn pass_decision(state: &GameState) -> Decision {
    let card_index = pick_any_card(state, state.current_player_id()).unwrap_or(0);
    Decision {
        mv: Move::Pass { card_index },
        score: -0.5,
    }
}

fn score_pending_free_develop(state: &GameState, pid: usize, count: u8) -> Option<Decision> {
    let mut types = state.players[pid].developable_types();
    if BAN_DEVELOP_IRON_LV2_PLUS {
        types.retain(|(ind, tile)| !(*ind == IndustryType::IronWorks && tile.level >= 2));
    }
    if types.is_empty() {
        return None;
    }

    let score_target = |ind: IndustryType, tile: crate::data::TileDef, unlock_offset: usize| {
        let unlocked = state.players[pid].tile_after(ind, unlock_offset);
        let mut v = if tile.rail_era { 0.35 } else { 0.12 };
        v += 0.18 * tile.level as f64;
        if let Some(ut) = unlocked {
            if ut.rail_era {
                v += 0.25;
            }
        }
        if ind == IndustryType::Brewery {
            v += 0.55;
        }
        if state.era == Era::Canal {
            v += 0.15;
        }
        if player_has_buildable_card(state, pid, ind) {
            v += 0.3;
        }
        v - develop_guardrail_penalty(ind, tile.level)
    };
    let mut scored: Vec<(IndustryType, f64)> = types
        .into_iter()
        .map(|(ind, tile)| (ind, score_target(ind, tile, 1)))
        .collect();
    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    let first = scored[0].0;
    let second = if count >= 2 {
        let best_other = scored.iter().skip(1).map(|s| (s.0, s.1)).next();
        let same_industry = state.players[pid]
            .tile_after(first, 1)
            .filter(|tile| {
                tile.can_develop
                    && !(BAN_DEVELOP_IRON_LV2_PLUS
                        && first == IndustryType::IronWorks
                        && tile.level >= 2)
            })
            .map(|tile| (first, score_target(first, tile, 2)));
        [best_other, same_industry]
            .into_iter()
            .flatten()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
    } else {
        None
    };

    let second_score = second.map(|(_, score)| score * 0.4).unwrap_or(0.0);

    Some(Decision {
        mv: Move::ResolveFreeDevelop {
            ind1: first,
            ind2: second.map(|(ind, _)| ind),
        },
        score: scored[0].1 + second_score,
    })
}

/// Position-value estimator for player `pid`, used as the MCTS leaf evaluator.
/// Combines accumulated VP + income stream + cash with the player's best
/// available move score (1-ply), which reflects actionable potential better
/// than a pure board snapshot.
pub(crate) fn evaluate_position(state: &GameState, pid: usize) -> f64 {
    let p = &state.players[pid];
    let mut value = 0.0;

    value += p.vp as f64 * VP_WEIGHT;
    value += p.money as f64 * money_weight(state);
    value += p.income_level() as f64 * income_weight(state) * 3.0;

    // Own tiles: flipped score full VP; unflipped a small fraction.
    for tile in state.city_tiles.iter().flatten() {
        if tile.player != pid {
            continue;
        }
        let vp = tile.def.vp as f64;
        value += if tile.flipped { vp } else { vp * 0.25 };
    }
    for tile in state.farm_tiles.iter().flatten() {
        if tile.player != pid {
            continue;
        }
        let vp = tile.def.vp as f64;
        value += if tile.flipped { vp } else { vp * 0.25 };
    }

    // Links: each owned link scores icons in adjacent locations at era end.
    let links = state
        .links
        .iter()
        .flatten()
        .filter(|l| l.player == pid)
        .count();
    value += links as f64 * 2.0;

    // Hand flexibility.
    let flex: f64 = p.hand.iter().map(|c| card_usefulness(state, pid, c)).sum();
    value += flex * FLEX_WEIGHT;

    value
}

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

pub(crate) fn estimate_rounds_remaining(state: &GameState) -> f64 {
    // Unplayed cards = deck + all hands. Each action spends one card, and a
    // round is actions_per_turn actions per player. The era ends only when the
    // deck AND every hand are exhausted, so using deck length alone badly
    // underestimates late-era rounds (deck empties before hands do).
    let deck_cards = state.deck.len() as f64;
    let hand_cards: f64 = state.players.iter().map(|p| p.hand.len() as f64).sum();
    let total = deck_cards + hand_cards;
    let actions_per_round = state.player_count() as f64 * state.actions_per_turn as f64;
    (total / actions_per_round).clamp(1.0, 8.0)
}

fn money_weight(state: &GameState) -> f64 {
    era_profile(state).money_w
}

pub(crate) fn income_weight(state: &GameState) -> f64 {
    era_profile(state).income_w
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn vp_equivalent(state: &GameState, vp: f64, income: f64, money: f64, flex: f64) -> f64 {
    vp * VP_WEIGHT
        + income * income_weight(state)
        + money * money_weight(state)
        + flex * FLEX_WEIGHT
}

// ---------------------------------------------------------------------------
include!("heuristic_ai/build.rs");
include!("heuristic_ai/network.rs");
include!("heuristic_ai/develop.rs");
include!("heuristic_ai/sell.rs");
include!("heuristic_ai/loan.rs");
include!("heuristic_ai/scout_pass.rs");
