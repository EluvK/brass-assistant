//! Card keep-score: how valuable each hand card is to hold.
//!
//! This is the "card-selection head" of the heuristic: operations and card
//! choice are separate policy dimensions. Lower keep-scores are preferred
//! when an action can consume any card. Deliberately transparent and
//! deterministic so replay traces can expose every component's effect
//! while the weights are tuned (see `CardWeights`).

use super::config::HeuristicConfig;
use crate::data::{Era, IndustryType};
use crate::map::city_slots;
use crate::rules::{BuildTarget, ResolvedMove, valid_build_cards};
use crate::state::{Card, GameState};

/// Card choices ordered from least valuable to most valuable:
/// `(hand_index, keep_score)`.
pub type CardChoices = Vec<(usize, f64)>;

/// Config-parameterised keep-score for one card.
pub(crate) fn card_keep_score_with(
    state: &GameState,
    pid: usize,
    card_index: usize,
    valid_targets: &[BuildTarget],
    cfg: &HeuristicConfig,
) -> f64 {
    let w = &cfg.cards;
    let Some(card) = state.players[pid].hand.get(card_index) else {
        return f64::INFINITY;
    };

    let hand = &state.players[pid].hand;
    let mut score = match card {
        Card::Location(_) => w.location_base,
        Card::Industry { .. } => w.industry_base,
        Card::WildLocation | Card::WildIndustry => w.wild_base,
    };

    // Repeated cards are less urgent to preserve. Industry cards are grouped
    // broadly because a duplicate production role is still less flexible in
    // the canal era even when the printed industry pair differs.
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
    score -= w.duplicate_penalty * duplicate_count.saturating_sub(1) as f64;

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
                score -= if resource_upgrade {
                    w.city_full_resource_upgrade_penalty
                } else {
                    w.city_full_useless_penalty
                };
            } else {
                score += w.city_target_bonus * (target_count.min(w.city_target_cap)) as f64;
            }
        }
        // Non-city location cards cannot occur: the deck only prints city
        // location cards. Keep a conservative value instead of panicking if
        // data ever changes.
        Card::Location(_) => {
            debug_assert!(false, "non-city location card in hand: {card:?}");
            score -= w.city_full_useless_penalty;
        }
        Card::Industry { industries, n } => {
            let mut best_role_targets = 0usize;
            for ind in industries.iter().take(*n as usize) {
                best_role_targets =
                    best_role_targets.max(valid_targets.iter().filter(|t| t.ind == *ind).count());
            }
            if best_role_targets == 0 {
                score -= w.industry_no_target_penalty;
            } else {
                score +=
                    w.industry_target_bonus * (best_role_targets.min(w.industry_target_cap)) as f64;
            }
            if state.era == Era::Canal && duplicate_count > 1 {
                score -= w.canal_industry_duplicate_penalty;
            }
        }
        Card::WildLocation | Card::WildIndustry => {
            // Wild cards are intentionally expensive to discard. Their score
            // is still allowed to fall when the hand contains duplicates.
            score += w.wild_duplicate_bonus * duplicate_count.saturating_sub(1) as f64;
        }
    }

    score
}

/// Card keep-score against all current build targets (public helper for
/// replay tools; hot paths should precompute the target list once).
pub fn card_keep_score(state: &GameState, pid: usize, card_index: usize) -> f64 {
    let cfg = HeuristicConfig::default();
    let valid_targets = crate::rules::get_valid_build_targets(state, pid);
    card_keep_score_with(state, pid, card_index, &valid_targets, &cfg)
}

/// All cards ranked from least to most valuable to keep.
pub fn ranked_card_choices(state: &GameState, pid: usize) -> CardChoices {
    // Build-target enumeration performs connectivity, resource and era
    // checks; compute it once per ranking instead of once per card.
    let cfg = HeuristicConfig::default();
    let valid_targets = crate::rules::get_valid_build_targets(state, pid);
    ranked_card_choices_with(state, pid, &valid_targets, &cfg)
}

/// Target-list-sharing variant used by the candidate batch, which has
/// already enumerated build targets.
pub(crate) fn ranked_card_choices_with(
    state: &GameState,
    pid: usize,
    valid_targets: &[BuildTarget],
    cfg: &HeuristicConfig,
) -> CardChoices {
    let mut ranked: Vec<(usize, f64)> = (0..state.players[pid].hand.len())
        .map(|index| {
            (
                index,
                card_keep_score_with(state, pid, index, valid_targets, cfg),
            )
        })
        .collect();
    ranked.sort_by(|a, b| a.1.total_cmp(&b.1).then(a.0.cmp(&b.0)));
    ranked
}

/// Card choices available to the card-selection head for a structural move.
/// Build is restricted by its target; all other ordinary operations may use
/// every card in hand.
pub fn card_choices_for_move(state: &GameState, mv: &ResolvedMove) -> CardChoices {
    let pid = state.current_player_id();
    match mv {
        ResolvedMove::Build { loc, ind, .. } => {
            valid_build_cards(state, &state.players[pid], pid, *loc, *ind)
                .into_iter()
                .map(|i| (i, card_keep_score(state, pid, i)))
                .collect()
        }
        _ => ranked_card_choices(state, pid),
    }
}

/// Keep-value score for the card references carried by a move. Lower values
/// mean the card is a better discard. Scout consumes three cards, so its
/// score is the sum of the three individual scores.
pub(crate) fn move_card_score(state: &GameState, mv: &ResolvedMove) -> f64 {
    let pid = state.current_player_id();
    let choices = ranked_card_choices(state, pid);
    let score = |index: usize| {
        choices
            .iter()
            .find(|(i, _)| *i == index)
            .map(|(_, s)| *s)
            .unwrap_or(f64::INFINITY)
    };
    match mv {
        ResolvedMove::Scout { card_indices } => card_indices.iter().map(|i| score(*i)).sum(),
        ResolvedMove::Build { card_index, .. }
        | ResolvedMove::Network { card_index, .. }
        | ResolvedMove::NetworkDouble { card_index, .. }
        | ResolvedMove::Develop { card_index, .. }
        | ResolvedMove::Sell { card_index, .. }
        | ResolvedMove::Loan { card_index }
        | ResolvedMove::Pass { card_index } => score(*card_index),
    }
}
