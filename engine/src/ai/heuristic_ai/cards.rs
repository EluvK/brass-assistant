//! Heuristic hand-card keep scoring and card-choice ranking.

use super::board::{city_supports_industry, empty_industry_slot_count};
use super::config::{CardWeights, HeuristicConfig};
use crate::data::{Era, IndustryType};
use crate::map::city_slots;
use crate::rules::{BuildTarget, ResolvedMove, valid_build_cards};
use crate::state::{Card, GameState};

pub type CardChoices = Vec<(usize, f64)>;

fn duplicate_ordinal(hand: &[Card], card_index: usize, ind: IndustryType) -> usize {
    hand[..=card_index]
        .iter()
        .filter(|card| card.is_industry(ind))
        .count()
}

fn industry_score(
    state: &GameState,
    pid: usize,
    card_index: usize,
    ind: IndustryType,
    w: &CardWeights,
) -> f64 {
    let ordinal = duplicate_ordinal(&state.players[pid].hand, card_index, ind);
    let mut score = match state.era {
        Era::Canal => match ind {
            IndustryType::IronWorks => {
                w.canal_iron_base - ordinal.saturating_sub(2) as f64 * w.canal_iron_third_penalty
            }
            IndustryType::CottonMill | IndustryType::Manufacturer | IndustryType::Pottery => {
                w.canal_sellable_base
                    - ordinal.saturating_sub(1) as f64 * w.canal_sellable_duplicate_penalty
            }
            IndustryType::CoalMine | IndustryType::Brewery => {
                w.canal_resource_base
                    - ordinal.saturating_sub(1) as f64 * w.canal_resource_duplicate_penalty
            }
        },
        Era::Rail => match ind {
            IndustryType::CoalMine => {
                w.rail_coal_base - ordinal.saturating_sub(2) as f64 * w.rail_coal_third_penalty
            }
            IndustryType::Brewery => {
                w.rail_brewery_base
                    - ordinal.saturating_sub(1) as f64 * w.rail_industry_duplicate_penalty
            }
            IndustryType::IronWorks => {
                w.rail_iron_base
                    - ordinal.saturating_sub(1) as f64 * w.rail_industry_duplicate_penalty
            }
            IndustryType::CottonMill | IndustryType::Manufacturer | IndustryType::Pottery => {
                w.rail_sellable_base
                    - ordinal.saturating_sub(1) as f64 * w.rail_industry_duplicate_penalty
            }
        },
    };
    if matches!(ind, IndustryType::IronWorks | IndustryType::Brewery)
        && ordinal > empty_industry_slot_count(state, ind)
    {
        score -= w.unconsumable_industry_penalty;
    }
    score
}

fn location_score(
    state: &GameState,
    pid: usize,
    card_index: usize,
    loc: crate::map::Loc,
    w: &CardWeights,
) -> f64 {
    let hand = &state.players[pid].hand;
    let ordinal = hand[..=card_index]
        .iter()
        .filter(|card| matches!(card, Card::Location(other) if *other == loc))
        .count();
    let mut score = match state.era {
        Era::Canal => {
            w.canal_location_base
                + if city_supports_industry(loc, IndustryType::IronWorks)
                    || city_supports_industry(loc, IndustryType::Brewery)
                {
                    w.canal_resource_city_bonus
                } else {
                    0.0
                }
        }
        Era::Rail => w.rail_location_base,
    };
    let occupied = city_slots(loc)
        .iter()
        .enumerate()
        .filter(|(slot_index, _)| state.tile_at(loc, *slot_index).is_some())
        .count();
    score -= occupied as f64 * w.occupied_city_slot_penalty;
    if ordinal >= 2 {
        score -= w.location_second_duplicate_penalty;
    }
    if ordinal >= 3 {
        score -= (ordinal - 2) as f64 * w.location_third_duplicate_penalty;
    }
    score
}

pub(crate) fn card_keep_score_with(
    state: &GameState,
    pid: usize,
    card_index: usize,
    _valid_targets: &[BuildTarget],
    cfg: &HeuristicConfig,
) -> f64 {
    let Some(card) = state.players[pid].hand.get(card_index) else {
        return f64::INFINITY;
    };
    match card {
        Card::Location(loc) => location_score(state, pid, card_index, *loc, &cfg.cards),
        Card::Industry { industries, n } => industries[..*n as usize]
            .iter()
            .map(|ind| industry_score(state, pid, card_index, *ind, &cfg.cards))
            .fold(f64::NEG_INFINITY, f64::max),
        Card::WildLocation | Card::WildIndustry => 3.0,
    }
    .clamp(0.0, 3.0)
}

pub fn card_keep_score(state: &GameState, pid: usize, card_index: usize) -> f64 {
    card_keep_score_with(state, pid, card_index, &[], &HeuristicConfig::default())
}

pub fn ranked_card_choices(state: &GameState, pid: usize) -> CardChoices {
    ranked_card_choices_with(state, pid, &[], &HeuristicConfig::default())
}

pub(crate) fn ranked_card_choices_with(
    state: &GameState,
    pid: usize,
    valid_targets: &[BuildTarget],
    cfg: &HeuristicConfig,
) -> CardChoices {
    let mut ranked: Vec<(usize, f64)> = (0..state.players[pid].hand.len())
        .map(|index| (index, card_keep_score_with(state, pid, index, valid_targets, cfg)))
        .collect();
    ranked.sort_by(|a, b| a.1.total_cmp(&b.1).then(a.0.cmp(&b.0)));
    ranked
}

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ai::heuristic_ai::board::industry_slots;
    use crate::data::industry_tiles;
    use crate::map::Loc;
    use crate::state::BoardTile;
    use rand_chacha::ChaCha12Rng;
    use rand_chacha::rand_core::SeedableRng;

    fn state(era: Era, hand: Vec<Card>) -> GameState {
        let mut state = GameState::new(ChaCha12Rng::seed_from_u64(13), 4);
        state.era = era;
        state.players[0].hand = hand;
        state
    }

    fn industry(ind: IndustryType) -> Card {
        Card::Industry {
            industries: [ind; 2],
            n: 1,
        }
    }

    fn tile(ind: IndustryType) -> BoardTile {
        let def = industry_tiles(ind)[0];
        BoardTile {
            player: 1,
            ind,
            def,
            flipped: false,
            resource_cubes: def.resource_cubes,
        }
    }

    #[test]
    fn industry_thresholds_and_capacity_are_scored_in_the_ai_layer() {
        let canal = state(
            Era::Canal,
            vec![
                industry(IndustryType::IronWorks),
                industry(IndustryType::IronWorks),
                industry(IndustryType::IronWorks),
            ],
        );
        assert_eq!(card_keep_score(&canal, 0, 0), 2.0);
        assert_eq!(card_keep_score(&canal, 0, 1), 2.0);
        assert_eq!(card_keep_score(&canal, 0, 2), 1.5);

        let mut rail = state(Era::Rail, vec![industry(IndustryType::IronWorks)]);
        assert_eq!(card_keep_score(&rail, 0, 0), 1.5);
        for slot in industry_slots(IndustryType::IronWorks) {
            rail.place_tile(slot.loc, slot.slot_index, tile(IndustryType::IronWorks));
        }
        assert_eq!(card_keep_score(&rail, 0, 0), 0.0);
    }

    #[test]
    fn location_duplicates_and_scout_ranking_use_the_new_model() {
        let canal = state(
            Era::Canal,
            vec![
                Card::Location(Loc::Derby),
                Card::Location(Loc::Derby),
                Card::WildLocation,
                industry(IndustryType::CoalMine),
                industry(IndustryType::CoalMine),
                industry(IndustryType::IronWorks),
                industry(IndustryType::IronWorks),
                industry(IndustryType::IronWorks),
            ],
        );
        assert_eq!(card_keep_score(&canal, 0, 0), 2.0);
        assert_eq!(card_keep_score(&canal, 0, 1), 1.5);
        assert_eq!(card_keep_score(&canal, 0, 2), 3.0);
        assert_eq!(
            ranked_card_choices(&canal, 0)
                .iter()
                .take(3)
                .map(|(index, _)| *index)
                .collect::<Vec<_>>(),
            vec![4, 1, 3]
        );
    }
}
