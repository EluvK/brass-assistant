//! Structured neural features for one concrete legal move (schema v4).
//!
//! v4 principle: the action encoding states the CHOICE (which board cells are
//! drained, which connection, which merchant), while tile identity — industry,
//! owner, flip state — lives in the state tensor at the same board-cell index
//! and is joined by the scoring head.
//!
//! Layout (301 dims, all values quarter-step):
//!   ACTION     0(7)    action-type one-hot
//!   CARD       7(35)   card-semantic COUNTS (single card = 1.0; Scout adds up
//!                      to 3 -> {A,A,B} and {A,B,B} stay distinct)
//!   LOCATION  42(27)   Build target location one-hot
//!   CITY_SLOT 69(4)    Build target slot
//!   INDUSTRY_1 73(6)   main industry
//!   INDUSTRY_2 79(6)   second Develop industry
//!   CONNECTION_1  85(39)   single link / NetDouble first link one-hot
//!   CONNECTION_2 124(39)   NetDouble second link one-hot
//!   SELL_KEY   163(47)  sold city-slot keys
//!   MERCHANT   210(9)   Sell target merchants
//!   DRAIN      219(49)  cubes taken per BOARD CELL (47 city slots + 2 farms,
//!                       same index as the board tensor). Mine/ironworks/
//!                       brewery beer share one block: a cell hosts a single
//!                       industry, so kind and owner come from the state.
//!                       Market purchases have no identity and never enter.
//!   MERCHANT_BEER 268(9) merchant beer cubes taken
//!   CONSEQUENCE 277(12) analytic post-state facts (see below)
//!   SUMMARY    289(12)  [0]coal/2 [1]iron/2 [2]beer [3]market_coal [4]market_iron
//!                       [5]merchant_beer [6]sell_tiles/4 [7]single [8]double
//!                       [9]free_develop [10]scout [11]reserved

use crate::graph::{BeerSource, BeerSourceKind, CoalSource, CoalSourceKind, IronSource};
use crate::map::{MERCHANT_LOC_MASK, city_slots, connections};
use crate::r#move::ResolvedMove;
use crate::state::{Card, GameState};

pub const ACTION_FEATURE_DIM: usize = 301;
pub const ACTION_FEATURE_SCHEMA_VERSION: usize = 4;

const ACTION: usize = 0;
const CARD: usize = 7;
const LOCATION: usize = 42;
const CITY_SLOT: usize = 69;
const INDUSTRY_1: usize = 73;
const INDUSTRY_2: usize = 79;
const CONNECTION_1: usize = 85;
const CONNECTION_2: usize = 124;
const SELL_KEY: usize = 163;
const MERCHANT: usize = 210;
const DRAIN: usize = 219;
const MERCHANT_BEER: usize = 268;
const CONSEQUENCE: usize = 277;
const SUMMARY: usize = 289;

const CITY_CELLS: usize = 47; // state::total_city_slots(); cells 47/48 are the farms

// CONSEQUENCE offsets
const CONS_LINKS_BUILT: usize = 0; // /2
const CONS_NEW_REACH: usize = 1; // /4
const CONS_NEW_MERCHANT_REACH: usize = 2;
const CONS_FLIPS_OWN: usize = 3; // /4
const CONS_FLIPS_OPP: usize = 4; // /4
const CONS_OVERBUILD: usize = 5;
const CONS_UPGRADE: usize = 6;
const CONS_CITY_COMPLETION: usize = 7;
const CONS_SELL_TILES: usize = 8; // /4
const CONS_FREE_DEVELOP: usize = 9;

fn one_hot(out: &mut [f32], offset: usize, width: usize, index: usize) {
    if index < width {
        out[offset + index] = 1.0;
    }
}

/// Additive card-semantic counts: a single card lights its semantic indices at
/// 1.0 (identical to the old one-hot), Scout's three discards accumulate so
/// the discard multiset stays distinguishable.
fn add_card(out: &mut [f32], state: &GameState, card_index: usize) {
    let pid = state.current_player_id();
    let Some(card) = state
        .players
        .get(pid)
        .and_then(|player| player.hand.get(card_index))
    else {
        return;
    };
    match card {
        Card::Location(loc) => out[CARD + *loc as usize] += 1.0,
        Card::Industry { industries, n } => {
            for ind in industries.iter().take(*n as usize) {
                out[CARD + 27 + *ind as usize] += 1.0;
            }
        }
        Card::WildLocation => out[CARD + 33] += 1.0,
        Card::WildIndustry => out[CARD + 34] += 1.0,
    }
}

fn industry(out: &mut [f32], offset: usize, ind: crate::data::IndustryType) {
    one_hot(out, offset, 6, ind as usize);
}

/// Where one resource cube is taken from, in board-cell coordinates.
fn drain_cube(drain: &mut [u8; 49], merchant_beer: &mut [u8; 9], source: &BeerSource) {
    match source.kind {
        BeerSourceKind::Merchant => {
            if let Some(idx) = source.merchant_idx {
                if idx < 9 {
                    merchant_beer[idx] += 1;
                }
            }
        }
        BeerSourceKind::Own | BeerSourceKind::Opponent => {
            if let Some(farm) = source.farm_idx {
                if farm < 2 {
                    drain[CITY_CELLS + farm] += 1;
                }
            } else if source.key < CITY_CELLS {
                drain[source.key] += 1;
            }
        }
    }
}

fn coal_cube(drain: &mut [u8; 49], source: &CoalSource) -> bool {
    // Returns true when the cube came from the market (no identity).
    match source.kind {
        CoalSourceKind::Mine => {
            if source.key < CITY_CELLS {
                drain[source.key] += 1;
                false
            } else {
                true
            }
        }
        CoalSourceKind::Market => true,
    }
}

fn iron_cube(drain: &mut [u8; 49], source: &IronSource) -> bool {
    if source.key < CITY_CELLS {
        drain[source.key] += 1;
        false
    } else {
        true
    }
}

/// Flip consequence of the accumulated drains: a tile flips when its last cube
/// was taken. Returns (own_flips, opponent_flips).
fn count_flips(state: &GameState, drain: &[u8; 49], pid: usize) -> (u32, u32) {
    let mut own = 0;
    let mut opp = 0;
    for (cell, &taken) in drain.iter().enumerate() {
        if taken == 0 {
            continue;
        }
        let tile = if cell < CITY_CELLS {
            state.city_tiles[cell].as_ref()
        } else {
            state.farm_tiles[cell - CITY_CELLS].as_ref()
        };
        let Some(tile) = tile else { continue };
        if tile.resource_cubes as usize == taken as usize {
            if tile.player == pid {
                own += 1;
            } else {
                opp += 1;
            }
        }
    }
    (own, opp)
}

/// Locations of the mover's own network newly touched by the built links
/// (endpoints + via farm). Returns (new count, any new merchant reached).
fn new_network_reach(state: &GameState, pid: usize, conns: &[usize]) -> (u32, bool) {
    // The mask accumulates across the links of one move, so a location shared
    // by two links of a NetworkDouble is counted once.
    let mut mask = state.network_mask(pid);
    let mut fresh = 0u32;
    let mut merchant = false;
    for &conn_id in conns {
        let conn = &connections()[conn_id];
        let mut locs = vec![conn.a as usize, conn.b as usize];
        if let Some(farm) = conn.via_farm {
            locs.push(farm as usize);
        }
        for loc in locs {
            let bit = 1u32 << loc;
            if mask & bit == 0 {
                fresh += 1;
                if MERCHANT_LOC_MASK & bit != 0 {
                    merchant = true;
                }
                mask |= bit;
            }
        }
    }
    (fresh, merchant)
}

/// Encode a concrete move into the stable v4 action-feature schema.
///
/// `card_index` is an execution reference only. Card semantics are read from
/// the pre-move current player's hand so the network never learns hand order.
pub fn encode_move(state: &GameState, mv: &ResolvedMove) -> Vec<f32> {
    let mut out = vec![0.0; ACTION_FEATURE_DIM];
    one_hot(&mut out, ACTION, 7, mv.action().index());

    // Drains accumulate across every branch, then land in DRAIN / MERCHANT_BEER
    // and drive the flip-consequence features.
    let mut drain = [0u8; 49];
    let mut merchant_beer = [0u8; 9];
    let pid = state.current_player_id();

    match mv {
        ResolvedMove::Build {
            loc,
            slot_index,
            ind,
            coal,
            iron,
            card_index,
        } => {
            add_card(&mut out, state, *card_index);
            one_hot(&mut out, LOCATION, 27, *loc as usize);
            one_hot(&mut out, CITY_SLOT, 4, *slot_index);
            industry(&mut out, INDUSTRY_1, *ind);
            let mut market_coal = 0u32;
            let mut market_iron = 0u32;
            for source in coal {
                if coal_cube(&mut drain, source) {
                    market_coal += 1;
                }
            }
            for source in iron {
                if iron_cube(&mut drain, source) {
                    market_iron += 1;
                }
            }
            out[SUMMARY] = coal.len() as f32 / 2.0;
            out[SUMMARY + 1] = iron.len() as f32 / 2.0;
            out[SUMMARY + 3] = market_coal as f32;
            out[SUMMARY + 4] = market_iron as f32;

            // Build consequences. Farms are single-slot build locations with
            // their own board cells (47/48); city_slot_key is city-only.
            let farm_cell = crate::state::farm_index(*loc).map(|idx| CITY_CELLS + idx);
            let key_opt = match farm_cell {
                Some(cell) => Some(cell),
                None => state.city_slot_key(*loc, *slot_index),
            };
            if let Some(key) = key_opt {
                let existing = if farm_cell.is_some() {
                    state.farm_tiles[key - CITY_CELLS].as_ref()
                } else {
                    state.city_tiles[key].as_ref()
                };
                out[CONSEQUENCE + CONS_OVERBUILD] = existing.is_some() as u8 as f32;
                out[CONSEQUENCE + CONS_UPGRADE] =
                    existing.is_some_and(|t| t.player == pid) as u8 as f32;
                let slots = if farm_cell.is_some() {
                    1usize
                } else {
                    city_slots(*loc).len()
                };
                let occupied = if farm_cell.is_some() {
                    usize::from(existing.is_some())
                } else {
                    (0..slots)
                        .filter(|&slot| {
                            state
                                .city_slot_key(*loc, slot)
                                .and_then(|k| state.city_tiles[k].as_ref())
                                .is_some()
                        })
                        .count()
                };
                // Occupied count includes this build's target slot pre-state;
                // after the build it becomes occupied + (was empty ? 1 : 0).
                // Completion means THIS build filled the last empty slot, so
                // overbuilding in an already-full city does not count.
                let after = occupied + usize::from(existing.is_none());
                out[CONSEQUENCE + CONS_CITY_COMPLETION] =
                    (existing.is_none() && after >= slots) as u8 as f32;
            }
        }
        ResolvedMove::Network {
            conn_id,
            coal,
            card_index,
        } => {
            add_card(&mut out, state, *card_index);
            one_hot(&mut out, CONNECTION_1, 39, *conn_id);
            out[SUMMARY + 7] = 1.0;
            let mut market_coal = 0u32;
            if let Some(source) = coal {
                out[SUMMARY] = 0.5;
                if coal_cube(&mut drain, source) {
                    market_coal += 1;
                }
            }
            out[SUMMARY + 3] = market_coal as f32;
            let (fresh, merchant) = new_network_reach(state, pid, &[*conn_id]);
            out[CONSEQUENCE + CONS_LINKS_BUILT] = 0.5;
            out[CONSEQUENCE + CONS_NEW_REACH] = fresh as f32 / 4.0;
            out[CONSEQUENCE + CONS_NEW_MERCHANT_REACH] = merchant as u8 as f32;
        }
        ResolvedMove::NetworkDouble {
            conn1,
            conn2,
            coal1,
            coal2,
            beer,
            card_index,
        } => {
            add_card(&mut out, state, *card_index);
            one_hot(&mut out, CONNECTION_1, 39, *conn1);
            one_hot(&mut out, CONNECTION_2, 39, *conn2);
            out[SUMMARY] = 1.0;
            out[SUMMARY + 2] = 1.0;
            out[SUMMARY + 8] = 1.0;
            let mut market_coal = 0u32;
            for source in [coal1, coal2] {
                if coal_cube(&mut drain, source) {
                    market_coal += 1;
                }
            }
            out[SUMMARY + 3] = market_coal as f32;
            drain_cube(&mut drain, &mut merchant_beer, beer);
            let (fresh, merchant) = new_network_reach(state, pid, &[*conn1, *conn2]);
            out[CONSEQUENCE + CONS_LINKS_BUILT] = 1.0;
            out[CONSEQUENCE + CONS_NEW_REACH] = fresh as f32 / 4.0;
            out[CONSEQUENCE + CONS_NEW_MERCHANT_REACH] = merchant as u8 as f32;
        }
        ResolvedMove::Develop {
            ind1,
            ind2,
            iron,
            card_index,
        } => {
            add_card(&mut out, state, *card_index);
            industry(&mut out, INDUSTRY_1, *ind1);
            if let Some(ind) = ind2 {
                industry(&mut out, INDUSTRY_2, *ind);
            }
            let mut market_iron = 0u32;
            for source in iron {
                if iron_cube(&mut drain, source) {
                    market_iron += 1;
                }
            }
            out[SUMMARY + 1] = iron.len() as f32 / 2.0;
            out[SUMMARY + 4] = market_iron as f32;
        }
        ResolvedMove::Sell {
            keys,
            merchant_indices,
            use_merchant_beer: _,
            beer_sources,
            free_develop,
            card_index,
        } => {
            add_card(&mut out, state, *card_index);
            out[SUMMARY + 6] = keys.len() as f32 / 4.0;
            out[CONSEQUENCE + CONS_SELL_TILES] = keys.len() as f32 / 4.0;
            for &key in keys {
                one_hot(&mut out, SELL_KEY, 47, key);
            }
            for &merchant in merchant_indices {
                one_hot(&mut out, MERCHANT, 9, merchant);
            }
            // The engine's deterministic beer payment plan is explicit input:
            // brewery beer drains cells, merchant beer drains merchants. The
            // aligned `use_merchant_beer` flags are subsumed by the plan.
            for plan in beer_sources {
                for source in plan {
                    drain_cube(&mut drain, &mut merchant_beer, source);
                }
            }
            if let Some(ind) = free_develop {
                industry(&mut out, INDUSTRY_1, *ind);
                out[SUMMARY + 9] = 1.0;
                out[CONSEQUENCE + CONS_FREE_DEVELOP] = 1.0;
            }
        }
        ResolvedMove::Loan { card_index } | ResolvedMove::Pass { card_index } => {
            add_card(&mut out, state, *card_index)
        }
        ResolvedMove::Scout { card_indices } => {
            for &index in card_indices {
                add_card(&mut out, state, index);
            }
            out[SUMMARY + 10] = 1.0;
        }
    }

    for (cell, &taken) in drain.iter().enumerate() {
        if taken > 0 {
            out[DRAIN + cell] = taken as f32 / 4.0;
        }
    }
    let merchant_beer_total: u32 = merchant_beer.iter().map(|&c| c as u32).sum();
    for (idx, &taken) in merchant_beer.iter().enumerate() {
        if taken > 0 {
            out[MERCHANT_BEER + idx] = taken as f32 / 4.0;
        }
    }
    out[SUMMARY + 5] = merchant_beer_total as f32;
    let (flips_own, flips_opp) = count_flips(state, &drain, pid);
    out[CONSEQUENCE + CONS_FLIPS_OWN] = flips_own as f32 / 4.0;
    out[CONSEQUENCE + CONS_FLIPS_OPP] = flips_opp as f32 / 4.0;
    out
}
