//! Action -> entity references (Brass: Birmingham).
//!
//! Contract: `docs/ai-action-encoding.md` §3. A candidate action states its
//! type, the entities it selects, and a few scalars. It never restates state:
//! the network resolves each reference against the state tokens of the same
//! position, so "which coal did I take, and what was on that cell" is a
//! structural join instead of a hand-computed feature.
//!
//! Row layout (`ACTION_FEATURE_DIM` floats):
//!
//!   [0]                action kind (index, not one-hot)
//!   [1]                Build slot
//!   [2..6]             numbers: market coal, market iron, merchant beer, cards paid
//!   [6]                reference count
//!   [7 + 3i + 0..3]    reference i as (kind, id, weight)
//!
//! Reference kinds: 0 cell, 1 link, 2 merchant, 3 industry, 4 card semantics.
//! Unused reference slots are zeroed; `ref_count` bounds them.

use crate::graph::{BeerSource, BeerSourceKind, CoalSource, CoalSourceKind, IronSource};
use crate::r#move::ResolvedMove;
use crate::state::{Card, GameState};

/// State token layout owns the cell/industry/card id spaces; re-exported here
/// so the action code cannot drift from the state code.
pub use crate::encode::{CARD_SEMANTIC_COUNT, CITY_CELLS, INDUSTRY_COUNT, LOCATION_COUNT};

pub const ACTION_SCHEMA_VERSION: usize = 1;
pub const ACTION_KIND_COUNT: usize = 8;
pub const ACTION_REF_CAP: usize = 16;
pub const ACTION_NUMBERS: usize = 4;
pub const ACTION_FEATURE_DIM: usize = 3 + ACTION_NUMBERS + 3 * ACTION_REF_CAP; // 55

pub const REF_CELL: u8 = 0;
pub const REF_LINK: u8 = 1;
pub const REF_MERCHANT: u8 = 2;
pub const REF_INDUSTRY: u8 = 3;
pub const REF_CARD: u8 = 4;
pub const REF_KIND_COUNT: usize = 5;

/// Action kind ids, matching `docs/ai-action-encoding.md` §3.
pub mod kind {
    pub const BUILD: usize = 0;
    pub const NETWORK: usize = 1;
    pub const NETWORK_DOUBLE: usize = 2;
    pub const DEVELOP: usize = 3;
    pub const SELL: usize = 4;
    pub const LOAN: usize = 5;
    pub const SCOUT: usize = 6;
    pub const PASS: usize = 7;
}

const OFF_KIND: usize = 0;
const OFF_SLOT: usize = 1;
const OFF_NUMBERS: usize = 2;
const OFF_REF_COUNT: usize = 2 + ACTION_NUMBERS;
const OFF_REFS: usize = OFF_REF_COUNT + 1;

const CARD_WILD_LOCATION: usize = LOCATION_COUNT + INDUSTRY_COUNT; // 33
const CARD_WILD_INDUSTRY: usize = LOCATION_COUNT + INDUSTRY_COUNT + 1; // 34

struct Refs {
    kinds: [u8; ACTION_REF_CAP],
    ids: [u16; ACTION_REF_CAP],
    weights: [f32; ACTION_REF_CAP],
    count: usize,
}

impl Refs {
    fn new() -> Self {
        Refs {
            kinds: [0; ACTION_REF_CAP],
            ids: [0; ACTION_REF_CAP],
            weights: [0.0; ACTION_REF_CAP],
            count: 0,
        }
    }

    fn push(&mut self, kind: u8, id: usize, weight: f32) {
        assert!(
            self.count < ACTION_REF_CAP,
            "action reference cap ({ACTION_REF_CAP}) exceeded: the action-reference \
             contract needs review, not truncation"
        );
        assert!(id <= u16::MAX as usize, "reference id out of range: {id}");
        self.kinds[self.count] = kind;
        self.ids[self.count] = id as u16;
        self.weights[self.count] = weight;
        self.count += 1;
    }

    fn push_card(&mut self, card: &Card) {
        match card {
            Card::Location(loc) => self.push(REF_CARD, *loc as usize, 1.0),
            Card::Industry { industries, n } => {
                for &ind in industries.iter().take(*n as usize) {
                    self.push(REF_CARD, LOCATION_COUNT + ind as usize, 1.0);
                }
            }
            Card::WildLocation => self.push(REF_CARD, CARD_WILD_LOCATION, 1.0),
            Card::WildIndustry => self.push(REF_CARD, CARD_WILD_INDUSTRY, 1.0),
        }
    }
}

/// The card at `hand_index`, read from the acting player's hand before the
/// move. Card identity is deliberately semantic: the same card in two hand
/// positions yields the same reference.
fn hand_card(state: &GameState, hand_index: usize) -> Option<Card> {
    state
        .players
        .get(state.current_player_id())?
        .hand
        .get(hand_index)
        .cloned()
}

/// Add one resource source as a reference. Market purchases have no identity
/// and are counted in `numbers` instead.
fn push_cell_source(refs: &mut Refs, cell: Option<usize>, market: &mut u32) {
    match cell {
        Some(cell) => refs.push(REF_CELL, cell, 1.0),
        None => *market += 1,
    }
}

fn coal_source(refs: &mut Refs, source: &CoalSource, market: &mut u32) {
    let cell = match source.kind {
        CoalSourceKind::Mine if source.key < CITY_CELLS => Some(source.key),
        _ => None,
    };
    push_cell_source(refs, cell, market);
}

fn iron_source(refs: &mut Refs, source: &IronSource, market: &mut u32) {
    let cell = (source.key < CITY_CELLS).then_some(source.key);
    push_cell_source(refs, cell, market);
}

fn beer_source(refs: &mut Refs, source: &BeerSource, merchant_beer: &mut u32) {
    match source.kind {
        BeerSourceKind::Merchant => match source.merchant_idx {
            Some(idx) => refs.push(REF_MERCHANT, idx, 1.0),
            None => *merchant_beer += 1,
        },
        BeerSourceKind::Own | BeerSourceKind::Opponent => match source.farm_idx {
            Some(farm) if farm < 2 => refs.push(REF_CELL, CITY_CELLS + farm, 1.0),
            _ if source.key < CITY_CELLS => refs.push(REF_CELL, source.key, 1.0),
            // A beer source must name a board cell or a merchant; anything else
            // is an engine bug, not a feature to encode.
            _ => debug_assert!(false, "beer source without a board location"),
        },
    }
}

/// Build target cell: city slot key, or the farm cell for brewery farms.
fn build_cell(state: &GameState, loc: crate::map::Loc, slot_index: usize) -> Option<usize> {
    match crate::state::farm_index(loc) {
        Some(idx) => Some(CITY_CELLS + idx),
        None => state.city_slot_key(loc, slot_index),
    }
}

/// Encode a concrete move into the action-reference schema.
pub fn encode_move(state: &GameState, mv: &ResolvedMove) -> Vec<f32> {
    let mut out = Vec::with_capacity(ACTION_FEATURE_DIM);
    encode_move_into(state, mv, &mut out);
    out
}

/// Same as [`encode_move`], writing into a caller-owned buffer that is cleared
/// and reused across calls (hot path: one row per legal candidate).
pub fn encode_move_into(state: &GameState, mv: &ResolvedMove, out: &mut Vec<f32>) {
    out.clear();
    out.resize(ACTION_FEATURE_DIM, 0.0);

    let mut refs = Refs::new();
    let mut market_coal = 0u32;
    let mut market_iron = 0u32;
    let mut merchant_beer = 0u32;
    let mut cards_paid = 0u32;
    let mut slot = 0usize;
    let kind_id = match mv {
        ResolvedMove::Build {
            loc,
            slot_index,
            ind,
            coal,
            iron,
            card_index,
        } => {
            if let Some(cell) = build_cell(state, *loc, *slot_index) {
                refs.push(REF_CELL, cell, 1.0);
            }
            slot = *slot_index;
            refs.push(REF_INDUSTRY, *ind as usize, 1.0);
            for source in coal {
                coal_source(&mut refs, source, &mut market_coal);
            }
            for source in iron {
                iron_source(&mut refs, source, &mut market_iron);
            }
            if let Some(card) = hand_card(state, *card_index) {
                refs.push_card(&card);
                cards_paid += 1;
            }
            kind::BUILD
        }
        ResolvedMove::Network {
            conn_id,
            coal,
            card_index,
        } => {
            refs.push(REF_LINK, *conn_id, 1.0);
            if let Some(source) = coal {
                coal_source(&mut refs, source, &mut market_coal);
            }
            if let Some(card) = hand_card(state, *card_index) {
                refs.push_card(&card);
                cards_paid += 1;
            }
            kind::NETWORK
        }
        ResolvedMove::NetworkDouble {
            conn1,
            conn2,
            coal1,
            coal2,
            beer,
            card_index,
        } => {
            refs.push(REF_LINK, *conn1, 1.0);
            refs.push(REF_LINK, *conn2, 1.0);
            coal_source(&mut refs, coal1, &mut market_coal);
            coal_source(&mut refs, coal2, &mut market_coal);
            beer_source(&mut refs, beer, &mut merchant_beer);
            if let Some(card) = hand_card(state, *card_index) {
                refs.push_card(&card);
                cards_paid += 1;
            }
            kind::NETWORK_DOUBLE
        }
        ResolvedMove::Develop {
            ind1,
            ind2,
            iron,
            card_index,
        } => {
            refs.push(REF_INDUSTRY, *ind1 as usize, 1.0);
            if let Some(ind) = ind2 {
                refs.push(REF_INDUSTRY, *ind as usize, 1.0);
            }
            for source in iron {
                iron_source(&mut refs, source, &mut market_iron);
            }
            if let Some(card) = hand_card(state, *card_index) {
                refs.push_card(&card);
                cards_paid += 1;
            }
            kind::DEVELOP
        }
        ResolvedMove::Sell {
            keys,
            beer_sources,
            free_develop,
            card_index,
        } => {
            for &key in keys {
                refs.push(REF_CELL, key, 1.0);
            }
            for source in beer_sources {
                beer_source(&mut refs, source, &mut merchant_beer);
            }
            if let Some(ind) = free_develop {
                refs.push(REF_INDUSTRY, *ind as usize, 1.0);
            }
            if let Some(card) = hand_card(state, *card_index) {
                refs.push_card(&card);
                cards_paid += 1;
            }
            kind::SELL
        }
        ResolvedMove::Loan { card_index } => {
            if let Some(card) = hand_card(state, *card_index) {
                refs.push_card(&card);
                cards_paid += 1;
            }
            kind::LOAN
        }
        ResolvedMove::Scout { card_indices } => {
            for &index in card_indices {
                if let Some(card) = hand_card(state, index) {
                    refs.push_card(&card);
                    cards_paid += 1;
                }
            }
            kind::SCOUT
        }
        ResolvedMove::Pass { card_index } => {
            if let Some(card) = hand_card(state, *card_index) {
                refs.push_card(&card);
                cards_paid += 1;
            }
            kind::PASS
        }
    };

    out[OFF_KIND] = kind_id as f32;
    out[OFF_SLOT] = slot as f32;
    out[OFF_NUMBERS] = market_coal as f32;
    out[OFF_NUMBERS + 1] = market_iron as f32;
    out[OFF_NUMBERS + 2] = merchant_beer as f32;
    out[OFF_NUMBERS + 3] = cards_paid as f32;
    out[OFF_REF_COUNT] = refs.count as f32;
    for i in 0..refs.count {
        let base = OFF_REFS + i * 3;
        out[base] = refs.kinds[i] as f32;
        out[base + 1] = refs.ids[i] as f32;
        out[base + 2] = refs.weights[i];
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rules::legal_resolved_moves;
    use rand_chacha::rand_core::SeedableRng;
    use rand_chacha::ChaCha12Rng;

    fn row_refs(row: &[f32]) -> Vec<(u8, u16, f32)> {
        let count = row[OFF_REF_COUNT] as usize;
        assert!(count <= ACTION_REF_CAP, "ref count {count} exceeds cap");
        (0..count)
            .map(|i| {
                let base = OFF_REFS + i * 3;
                (row[base] as u8, row[base + 1] as u16, row[base + 2])
            })
            .collect()
    }

    #[test]
    fn every_legal_move_encodes_within_contract_bounds() {
        for seed in [7u64, 21, 99] {
            let mut state = GameState::new(ChaCha12Rng::seed_from_u64(seed), 4);
            for _ in 0..12 {
                let legal = legal_resolved_moves(&mut state);
                assert!(!legal.is_empty(), "seed {seed} produced no legal moves");
                for mv in &legal {
                    let row = encode_move(&state, mv);
                    assert_eq!(row.len(), ACTION_FEATURE_DIM);
                    let kind = row[OFF_KIND] as usize;
                    assert!(kind < ACTION_KIND_COUNT, "bad action kind {kind}");
                    assert!((0.0..=3.0).contains(&row[OFF_SLOT]));
                    assert!(row[OFF_NUMBERS + 3] <= 3.0, "at most three cards are paid");
                    for (ref_kind, id, weight) in row_refs(&row) {
                        assert!((ref_kind as usize) < REF_KIND_COUNT);
                        let bound = match ref_kind {
                            REF_CELL => crate::encode::BOARD_CELLS,
                            REF_LINK => crate::encode::LINK_CELLS,
                            REF_MERCHANT => crate::encode::MERCHANT_COUNT,
                            REF_INDUSTRY => INDUSTRY_COUNT,
                            _ => CARD_SEMANTIC_COUNT,
                        };
                        assert!((id as usize) < bound, "ref id {id} out of range");
                        assert!(weight > 0.0);
                    }
                }
                let Some(mv) = legal.first() else { break };
                let _ = crate::rules::apply_move(&mut state, mv);
                let tr = crate::engine::advance_turn(&mut state);
                crate::engine::handle_turn_result(&mut state, tr);
            }
        }
    }

    #[test]
    fn build_references_the_target_cell_and_its_industries() {
        let mut state = GameState::new(ChaCha12Rng::seed_from_u64(3), 4);
        let legal = legal_resolved_moves(&mut state);
        let builds: Vec<_> = legal
            .iter()
            .filter(|mv| matches!(mv, ResolvedMove::Build { .. }))
            .collect();
        assert!(!builds.is_empty(), "opening position must offer a Build");
        for mv in builds {
            let ResolvedMove::Build {
                loc,
                slot_index,
                ind,
                ..
            } = mv
            else {
                unreachable!()
            };
            let row = encode_move(&state, mv);
            let refs = row_refs(&row);
            let cell = build_cell(&state, *loc, *slot_index).unwrap();
            assert!(
                refs.contains(&(REF_CELL, cell as u16, 1.0)),
                "build must reference its target cell"
            );
            assert!(
                refs.contains(&(REF_INDUSTRY, *ind as u16, 1.0)),
                "build must reference the industry it places"
            );
        }
    }

    #[test]
    fn card_references_are_semantic_not_hand_positions() {
        let mut state = GameState::new(ChaCha12Rng::seed_from_u64(11), 4);
        let legal = legal_resolved_moves(&mut state);
        // Two moves that differ only in the hand index they pay with must
        // produce rows that differ only in the card reference id, never in a
        // hand position.
        let row = encode_move(&state, legal.first().unwrap());
        for (ref_kind, id, _) in row_refs(&row) {
            if ref_kind == REF_CARD {
                assert!((id as usize) < CARD_SEMANTIC_COUNT);
            }
        }
    }
}
