//! Structured neural features for one concrete legal move.
//!
//! This representation is deliberately independent from the legacy policy
//! table. It describes the complete executable move selected by the rules
//! engine, including card and resource choices.

use crate::graph::{BeerSourceKind, CoalSourceKind};
use crate::r#move::Move;

pub const ACTION_FEATURE_DIM: usize = 208;
pub const ACTION_FEATURE_SCHEMA_VERSION: usize = 1;

const ACTION: usize = 0; // 7 action-type one-hot entries
const CARD: usize = 7; // 8 card-index entries
const LOCATION: usize = 15; // 27 location entries
const CITY_SLOT: usize = 42; // 4 city-slot entries
const INDUSTRY_1: usize = 46; // 6 industry entries
const INDUSTRY_2: usize = 52; // 6 industry entries
const CONNECTION_1: usize = 58; // 39 connection entries
const CONNECTION_2: usize = 97; // 39 connection entries
const SELL_KEY: usize = 136; // 47 city-tile entries
const MERCHANT: usize = 183; // 9 merchant-tile entries
const SUMMARY: usize = 192; // 16 aggregate resource/shape features

fn one_hot(out: &mut [f32], offset: usize, width: usize, index: usize) {
    if index < width {
        out[offset + index] = 1.0;
    }
}

fn card(out: &mut [f32], index: usize) {
    one_hot(out, CARD, 8, index);
}

fn industry(out: &mut [f32], offset: usize, ind: crate::data::IndustryType) {
    one_hot(out, offset, 6, ind as usize);
}

/// Encode a concrete move into the stable v1 action-feature schema.
pub fn encode_move(mv: &Move) -> Vec<f32> {
    let mut out = vec![0.0; ACTION_FEATURE_DIM];
    one_hot(&mut out, ACTION, 7, mv.action().index());

    match mv {
        Move::Build {
            loc,
            slot_index,
            ind,
            coal,
            iron,
            card_index,
        } => {
            card(&mut out, *card_index);
            one_hot(&mut out, LOCATION, 27, *loc as usize);
            one_hot(&mut out, CITY_SLOT, 4, *slot_index);
            industry(&mut out, INDUSTRY_1, *ind);
            out[SUMMARY] = coal.len() as f32 / 2.0;
            out[SUMMARY + 1] = iron.len() as f32 / 2.0;
            for source in coal {
                match source.kind {
                    CoalSourceKind::Mine => out[SUMMARY + 4] += 0.5,
                    CoalSourceKind::Market => out[SUMMARY + 3] += 0.5,
                }
            }
            for source in iron {
                if source.key == usize::MAX {
                    out[SUMMARY + 5] += 0.5;
                }
            }
        }
        Move::Network {
            conn_id,
            coal,
            card_index,
        } => {
            card(&mut out, *card_index);
            one_hot(&mut out, CONNECTION_1, 39, *conn_id);
            out[SUMMARY + 9] = 1.0;
            if let Some(source) = coal {
                out[SUMMARY] = 0.5;
                match source.kind {
                    CoalSourceKind::Mine => out[SUMMARY + 4] = 0.5,
                    CoalSourceKind::Market => out[SUMMARY + 3] = 0.5,
                }
            }
        }
        Move::NetworkDouble {
            conn1,
            conn2,
            coal1,
            coal2,
            beer,
            card_index,
        } => {
            card(&mut out, *card_index);
            one_hot(&mut out, CONNECTION_1, 39, *conn1);
            one_hot(&mut out, CONNECTION_2, 39, *conn2);
            out[SUMMARY] = 1.0;
            out[SUMMARY + 2] = 1.0;
            out[SUMMARY + 10] = 1.0;
            for source in [coal1, coal2] {
                match source.kind {
                    CoalSourceKind::Mine => out[SUMMARY + 4] += 0.5,
                    CoalSourceKind::Market => out[SUMMARY + 3] += 0.5,
                }
            }
            match beer.kind {
                BeerSourceKind::Own => out[SUMMARY + 6] = 1.0,
                BeerSourceKind::Opponent => out[SUMMARY + 7] = 1.0,
                BeerSourceKind::Merchant => out[SUMMARY + 8] = 1.0,
            }
        }
        Move::Develop {
            ind1,
            ind2,
            iron,
            card_index,
        } => {
            card(&mut out, *card_index);
            industry(&mut out, INDUSTRY_1, *ind1);
            if let Some(ind) = ind2 {
                industry(&mut out, INDUSTRY_2, *ind);
            }
            out[SUMMARY + 1] = iron.len() as f32 / 2.0;
            for source in iron {
                if source.key == usize::MAX {
                    out[SUMMARY + 5] += 0.5;
                }
            }
        }
        Move::ResolveFreeDevelop { ind1, ind2 } => {
            industry(&mut out, INDUSTRY_1, *ind1);
            if let Some(ind) = ind2 {
                industry(&mut out, INDUSTRY_2, *ind);
            }
            out[SUMMARY + 11] = 1.0;
        }
        Move::Sell {
            keys,
            merchant_indices,
            use_merchant_beer,
            card_index,
        } => {
            card(&mut out, *card_index);
            out[SUMMARY + 12] = keys.len() as f32 / 4.0;
            for &key in keys {
                one_hot(&mut out, SELL_KEY, 47, key);
            }
            for &merchant in merchant_indices {
                one_hot(&mut out, MERCHANT, 9, merchant);
            }
            for &uses_merchant in use_merchant_beer {
                if uses_merchant {
                    out[SUMMARY + 8] += 0.25;
                } else {
                    out[SUMMARY + 6] += 0.25;
                }
                out[SUMMARY + 2] += 0.25;
            }
        }
        Move::Loan { card_index } | Move::Pass { card_index } => card(&mut out, *card_index),
        Move::Scout { card_indices } => {
            for &index in card_indices {
                card(&mut out, index);
            }
            out[SUMMARY + 13] = 1.0;
        }
    }
    out
}
