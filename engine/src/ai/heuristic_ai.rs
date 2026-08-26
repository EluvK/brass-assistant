//! Heuristic AI baseline: translated from reference/npow-brass-birmingham/js/aiPlayer.js
//!
//! Heuristic policy and candidate evaluator translated from the reference AI.
//! The public action policy uses deterministic 2-ply lookahead over the scored
//! candidate set; candidate scores remain available to MCTS and training.

use crate::data::{Era, IndustryType};
use crate::engine::{TurnResult, advance_turn};
use crate::graph::{
    connected_locations, count_beer_sources, find_coal_sources, find_iron_sources, is_in_network,
};
use crate::map::{ALL_LOCATIONS, CITY_COUNT, Loc, city_slots, connections};
use crate::rules::{
    BuildTarget, Move, can_develop, can_scout, execute_sell, get_valid_build_targets,
    get_valid_network_targets, get_valid_second_rail_links, get_valid_sell_targets, legal_moves,
    valid_build_cards,
};
use crate::state::{Card, GameState, city_slot_offsets};

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

/// Estimate how valuable a card is to keep in hand. Lower values are preferred
/// when an action can consume any card. This is deliberately a transparent,
/// deterministic heuristic so replay traces can expose every component's
/// effect while the weights are tuned.
pub fn card_keep_score(state: &GameState, pid: usize, card_index: usize) -> f64 {
    let valid_targets = get_valid_build_targets(state, pid);
    card_keep_score_with_targets(state, pid, card_index, &valid_targets)
}

fn card_keep_score_with_targets(
    state: &GameState,
    pid: usize,
    card_index: usize,
    valid_targets: &[BuildTarget],
) -> f64 {
    let Some(card) = state.players[pid].hand.get(card_index) else {
        return f64::INFINITY;
    };

    let hand = &state.players[pid].hand;
    let mut score = match card {
        Card::Location(_) => 1.15,
        Card::Industry { .. } => 1.0,
        Card::WildLocation | Card::WildIndustry => 3.8,
    };

    // Repeated cards are less urgent to preserve. Industry cards are grouped
    // broadly because a duplicate production role is still less flexible in
    // the Canal era even when the printed industry pair differs.
    let duplicate_count = match card {
        Card::Location(loc) => hand
            .iter()
            .filter(|other| matches!(other, Card::Location(other_loc) if other_loc == loc))
            .count(),
        Card::Industry { .. } => hand
            .iter()
            .filter(|other| matches!(other, Card::Industry { .. }))
            .count(),
        Card::WildLocation | Card::WildIndustry => 0,
    };
    score -= 0.48 * duplicate_count.saturating_sub(1) as f64;

    match card {
        Card::Location(loc) if loc.is_city() => {
            let target_count = valid_targets.iter().filter(|t| t.loc == *loc).count();
            let city_is_full = city_slots(*loc).iter().enumerate().all(|(slot, _)| {
                let key = state
                    .city_slot_key(*loc, slot)
                    .expect("city slot from city_slots must have a key");
                state.city_tiles[key].is_some()
            });
            if city_is_full {
                // A location card still bypasses network access for an own
                // resource overbuild. Preserve it in Rail when no matching
                // industry card in hand can serve as the alternative.
                let resource_upgrade = state.era == Era::Rail
                    && city_slots(*loc).iter().enumerate().any(|(slot, _)| {
                        let key = state
                            .city_slot_key(*loc, slot)
                            .expect("city slot from city_slots must have a key");
                        let Some(tile) = state.city_tiles[key].as_ref() else {
                            return false;
                        };
                        matches!(
                            tile.ind,
                            IndustryType::CoalMine
                                | IndustryType::IronWorks
                                | IndustryType::Brewery
                        ) && tile.player == pid
                            && state.players[pid]
                                .next_tile(tile.ind)
                                .is_some_and(|next| next.level > tile.def.level)
                            && !hand.iter().any(|other| {
                                other.is_industry(tile.ind) || matches!(other, Card::WildIndustry)
                            })
                    });
                score -= if resource_upgrade { 0.45 } else { 1.05 };
            } else {
                score += (target_count.min(3) as f64) * 0.28;
            }
        }
        Card::Location(_) => {
            // should not happen: all valid build targets are cities, and the only non-city
            panic!("non-city location card in hand: {card:?}");
        }
        Card::Industry { industries, n } => {
            let mut best_role_targets = 0usize;
            for ind in industries.iter().take(*n as usize) {
                best_role_targets =
                    best_role_targets.max(valid_targets.iter().filter(|t| t.ind == *ind).count());
            }
            if best_role_targets == 0 {
                score -= 0.65;
            } else {
                score += (best_role_targets.min(3) as f64) * 0.22;
            }
            if state.era == Era::Canal && duplicate_count > 1 {
                score -= 0.22;
            }
        }
        Card::WildLocation | Card::WildIndustry => {
            // Wild cards are intentionally expensive to discard. Their score
            // is still allowed to fall when the hand contains duplicates.
            score += 0.35 * duplicate_count.saturating_sub(1) as f64;
        }
    }

    score
}

/// Card indices ordered from least valuable to most valuable.
pub fn ranked_card_choices(state: &GameState, pid: usize) -> Vec<(usize, f64)> {
    // Build-target enumeration performs connectivity, resource and era checks;
    // compute it once per ranking instead of once per card.
    let valid_targets = get_valid_build_targets(state, pid);
    ranked_card_choices_with_targets(state, pid, &valid_targets)
}

fn ranked_card_choices_with_targets(
    state: &GameState,
    pid: usize,
    valid_targets: &[BuildTarget],
) -> CardChoices {
    let mut ranked: Vec<(usize, f64)> = (0..state.players[pid].hand.len())
        .map(|index| {
            (
                index,
                card_keep_score_with_targets(state, pid, index, &valid_targets),
            )
        })
        .collect();
    ranked.sort_by(|a, b| a.1.total_cmp(&b.1).then(a.0.cmp(&b.0)));
    ranked
}

type CardChoices = Vec<(usize, f64)>;

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

    // Card utility is state-wide and independent of the concrete action type.
    // Compute it once for this candidate batch and share it with all consumers.
    let build_targets = get_valid_build_targets(state, pid);
    let card_choices = ranked_card_choices_with_targets(state, pid, &build_targets);
    let mut out = Vec::new();
    let plan = compute_plan(state, pid);

    for d in score_top_builds(state, pid, k, &plan, &build_targets) {
        if d.score != f64::NEG_INFINITY {
            out.push(d);
        }
    }
    out.extend(score_top_networks(state, pid, k, &plan, &card_choices));
    if let Some(d) = score_best_network_double(state, pid, &plan, &card_choices) {
        out.push(d);
    }
    if let Some(d) = score_develop_plan(state, pid, &plan, &card_choices) {
        out.push(d);
    }
    if let Some(d) = score_sell_plan(state, pid, &card_choices) {
        out.push(d);
    }
    if let Some(d) = score_loan_result(state, pid, &card_choices) {
        out.push(d);
    }
    if let Some(d) = score_scout_plan(state, pid, &card_choices) {
        out.push(d);
    }
    if let Some(d) = score_pass_result(state, pid, &card_choices) {
        out.push(d);
    }

    // Candidate pruning must never turn a position with executable actions
    // into a dead end (for example callers passing k=0). Keep one legal fallback
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
    let card_index = ranked_card_choices(state, state.current_player_id())
        .first()
        .map(|(card_index, _)| *card_index)
        .unwrap_or(0);
    Decision {
        mv: Move::Pass { card_index },
        score: -0.5,
    }
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
    let flex: f64 = ranked_card_choices(state, pid)
        .into_iter()
        .map(|(_, keep_score)| keep_score)
        .sum();
    value += flex * FLEX_WEIGHT;

    value
}

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

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
