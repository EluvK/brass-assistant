//! Heuristic hand-card keep scoring and card-choice ranking.

use super::board::{city_supports_industry, empty_industry_slot_count};
use super::context::define_era_round_factor;
use crate::data::{Era, IndustryType};
use crate::map::city_slots;
use crate::rules::{ResolvedMove, valid_build_cards};
use crate::state::{Card, GameState};

pub type CardChoices = Vec<(usize, f64)>;

/// Hand facts the first-round opening policy reads, split by the card kind
/// that grants each industry's support.
///
/// A `Card::Location` only supports an industry while its city still has an
/// empty slot that allows that industry.  Each location card is counted at
/// most once per industry even when the city has several empty slots for it;
/// a dual industry card is counted once per industry it names.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct HandIndustrySupport {
    /// `Card::Industry` cards (including dual cards) naming each industry.
    pub industry: [usize; 6],
    /// `Card::Location` cards whose city has an empty slot for each industry.
    pub location: [usize; 6],
    /// Whether the player holds a wild industry card.
    pub wild_industry: bool,
    /// Whether the player holds a wild location card.
    pub wild_location: bool,
}

impl HandIndustrySupport {
    /// Total cards that can unlock a build of `ind`.  Wild industry always
    /// counts; wild location only counts while an empty buildable slot for
    /// `ind` still exists somewhere on the board (a wild location cannot be
    /// played on a brewery farm).
    pub fn support_total(&self, state: &GameState, ind: IndustryType) -> usize {
        let idx = ind as usize;
        let mut total = self.industry[idx] + self.location[idx];
        if self.wild_industry {
            total += 1;
        }
        if self.wild_location && empty_industry_slot_count(state, ind) > 0 {
            total += 1;
        }
        total
    }
}

/// Read-only analysis of one player's hand for the opening templates.
pub(crate) fn analyze_hand(state: &GameState, pid: usize) -> HandIndustrySupport {
    let mut out = HandIndustrySupport::default();
    for card in &state.players[pid].hand {
        match card {
            Card::Industry { industries, n } => {
                for ind in industries[..*n as usize].iter().copied() {
                    out.industry[ind as usize] += 1;
                }
            }
            Card::Location(loc) => {
                let mut touched = [false; 6];
                for (slot_index, allowed) in city_slots(*loc).iter().enumerate() {
                    if state.tile_at(*loc, slot_index).is_some() {
                        continue;
                    }
                    for ind in allowed.iter().copied() {
                        touched[ind as usize] = true;
                    }
                }
                for (idx, supported) in touched.into_iter().enumerate() {
                    if supported {
                        out.location[idx] += 1;
                    }
                }
            }
            Card::WildIndustry => out.wild_industry = true,
            Card::WildLocation => out.wild_location = true,
        }
    }
    out
}

// Keep-scores are an independent policy head.  These values intentionally
// live next to the model instead of in the general action configuration:
// card utility does not depend on the operation being scored.
const KEEP_SCORE_MIN: f64 = 0.0;
const KEEP_SCORE_MAX: f64 = 3.0;
const WILD_KEEP_SCORE: f64 = 3.0;

const CANAL_IRON_BASE: f64 = 2.0;
const CANAL_SELLABLE_BASE: f64 = 1.8;
const CANAL_RESOURCE_BASE: f64 = 1.5;
const RAIL_COAL_BASE: f64 = 2.5;
const RAIL_BREWERY_BASE: f64 = 2.0;
const RAIL_IRON_BASE: f64 = 1.5;
const RAIL_SELLABLE_BASE: f64 = 1.8;

const CANAL_IRON_THIRD_PENALTY: f64 = 0.5;
const CANAL_SELLABLE_DUPLICATE_PENALTY: f64 = 0.3;
const CANAL_RESOURCE_DUPLICATE_PENALTY: f64 = 0.5;
const RAIL_COAL_THIRD_PENALTY: f64 = 1.0;
const RAIL_INDUSTRY_DUPLICATE_PENALTY: f64 = 0.3;

const CANAL_LOCATION_BASE: f64 = 1.5;
const CANAL_RESOURCE_CITY_BONUS: f64 = 0.5;
const RAIL_LOCATION_BASE: f64 = 2.0;
const OCCUPIED_CITY_SLOT_PENALTY: f64 = 0.5;
const LOCATION_SECOND_DUPLICATE_PENALTY: f64 = 0.5;
const LOCATION_THIRD_DUPLICATE_PENALTY: f64 = 1.0;
const UNCONSUMABLE_INDUSTRY_PENALTY: f64 = 1.5;

define_era_round_factor!(
    PlainLocationKeepScore,
    canal: (CANAL_LOCATION_BASE, CANAL_LOCATION_BASE),
    rail: (RAIL_LOCATION_BASE, RAIL_LOCATION_BASE),
);

/// Plain location-card baseline shared with Scout's hand-refresh model.
pub(super) fn plain_location_keep_score(state: &GameState) -> f64 {
    PlainLocationKeepScore::factor(state)
}

fn duplicate_ordinal(hand: &[Card], card_index: usize, ind: IndustryType) -> usize {
    hand[..=card_index]
        .iter()
        .filter(|card| card.is_industry(ind))
        .count()
}

fn industry_score(state: &GameState, pid: usize, card_index: usize, ind: IndustryType) -> f64 {
    let ordinal = duplicate_ordinal(&state.players[pid].hand, card_index, ind);
    let mut score = match state.era {
        Era::Canal => match ind {
            IndustryType::IronWorks => {
                CANAL_IRON_BASE - ordinal.saturating_sub(2) as f64 * CANAL_IRON_THIRD_PENALTY
            }
            IndustryType::CottonMill | IndustryType::Manufacturer | IndustryType::Pottery => {
                CANAL_SELLABLE_BASE
                    - ordinal.saturating_sub(1) as f64 * CANAL_SELLABLE_DUPLICATE_PENALTY
            }
            IndustryType::CoalMine | IndustryType::Brewery => {
                CANAL_RESOURCE_BASE
                    - ordinal.saturating_sub(1) as f64 * CANAL_RESOURCE_DUPLICATE_PENALTY
            }
        },
        Era::Rail => match ind {
            IndustryType::CoalMine => {
                RAIL_COAL_BASE - ordinal.saturating_sub(2) as f64 * RAIL_COAL_THIRD_PENALTY
            }
            IndustryType::Brewery => {
                RAIL_BREWERY_BASE
                    - ordinal.saturating_sub(1) as f64 * RAIL_INDUSTRY_DUPLICATE_PENALTY
            }
            IndustryType::IronWorks => {
                RAIL_IRON_BASE - ordinal.saturating_sub(1) as f64 * RAIL_INDUSTRY_DUPLICATE_PENALTY
            }
            IndustryType::CottonMill | IndustryType::Manufacturer | IndustryType::Pottery => {
                RAIL_SELLABLE_BASE
                    - ordinal.saturating_sub(1) as f64 * RAIL_INDUSTRY_DUPLICATE_PENALTY
            }
        },
    };
    if matches!(ind, IndustryType::IronWorks | IndustryType::Brewery)
        && ordinal > empty_industry_slot_count(state, ind)
    {
        score -= UNCONSUMABLE_INDUSTRY_PENALTY;
    }
    score
}

fn location_score(state: &GameState, pid: usize, card_index: usize, loc: crate::map::Loc) -> f64 {
    let hand = &state.players[pid].hand;
    let ordinal = hand[..=card_index]
        .iter()
        .filter(|card| matches!(card, Card::Location(other) if *other == loc))
        .count();
    let mut score = plain_location_keep_score(state);
    if state.is_canal_era()
        && (city_supports_industry(loc, IndustryType::IronWorks)
            || city_supports_industry(loc, IndustryType::Brewery))
    {
        score += CANAL_RESOURCE_CITY_BONUS;
    }
    let occupied = city_slots(loc)
        .iter()
        .enumerate()
        .filter(|(slot_index, _)| state.tile_at(loc, *slot_index).is_some())
        .count();
    score -= occupied as f64 * OCCUPIED_CITY_SLOT_PENALTY;
    if ordinal >= 2 {
        score -= LOCATION_SECOND_DUPLICATE_PENALTY;
    }
    if ordinal >= 3 {
        score -= (ordinal - 2) as f64 * LOCATION_THIRD_DUPLICATE_PENALTY;
    }
    score
}

fn card_keep_score_at(state: &GameState, pid: usize, card_index: usize) -> f64 {
    let Some(card) = state.players[pid].hand.get(card_index) else {
        return f64::INFINITY;
    };
    match card {
        Card::Location(loc) => location_score(state, pid, card_index, *loc),
        Card::Industry { industries, n } => industries
            .iter()
            .take(*n as usize)
            .map(|ind| industry_score(state, pid, card_index, *ind))
            .fold(f64::NEG_INFINITY, f64::max),
        Card::WildLocation | Card::WildIndustry => WILD_KEEP_SCORE,
    }
    .clamp(KEEP_SCORE_MIN, KEEP_SCORE_MAX)
}

pub fn card_keep_score(state: &GameState, pid: usize, card_index: usize) -> f64 {
    card_keep_score_at(state, pid, card_index)
}

pub fn ranked_card_choices(state: &GameState, pid: usize) -> CardChoices {
    let mut ranked: Vec<(usize, f64)> = (0..state.players[pid].hand.len())
        .map(|index| (index, card_keep_score_at(state, pid, index)))
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
    fn plain_location_baseline_tracks_the_era_factor() {
        let mut game = state(Era::Canal, Vec::new());
        assert_eq!(plain_location_keep_score(&game), 1.5);
        game.era = Era::Rail;
        assert_eq!(plain_location_keep_score(&game), 2.0);
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

    #[test]
    fn hand_analysis_counts_each_location_card_once_per_industry() {
        let mut game = state(Era::Canal, Vec::new());
        // Tamworth has two coal slots: the card must still count as one
        // coal support, and as no iron support.
        game.players[0].hand = vec![
            Card::Location(Loc::Tamworth),
            Card::Industry {
                industries: [IndustryType::CottonMill, IndustryType::Manufacturer],
                n: 2,
            },
            Card::WildIndustry,
            Card::WildLocation,
        ];
        let analyzed = analyze_hand(&game, 0);
        assert_eq!(analyzed.location[IndustryType::CoalMine as usize], 1);
        assert_eq!(analyzed.location[IndustryType::IronWorks as usize], 0);
        assert_eq!(analyzed.industry[IndustryType::CottonMill as usize], 1);
        assert_eq!(analyzed.industry[IndustryType::Manufacturer as usize], 1);
        assert_eq!(analyzed.industry[IndustryType::IronWorks as usize], 0);
        assert!(analyzed.wild_industry && analyzed.wild_location);

        let coal_total = analyzed.support_total(&game, IndustryType::CoalMine);
        assert_eq!(coal_total, 3);
        let iron_total = analyzed.support_total(&game, IndustryType::IronWorks);
        // No concrete iron cards, but both wilds can unlock an iron build
        // while empty iron slots remain on the board.
        assert_eq!(iron_total, 2);
    }
}
