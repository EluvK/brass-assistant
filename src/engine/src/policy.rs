//! Fixed canonical action table (the Policy-head output space) and the
//! move -> slot mapping.
//!
//! Policy slots collapse moves that differ only in resource-source or card
//! choice (they are strategically identical), matching the MCTS tree
//! matching logic (`mcts_ai::move_key`). The network emits one logit per
//! slot; `legal_mask` zeroes the illegal ones before softmax.
//!
//! Table layout (offsets are stable as long as the map constants hold):
//!   Build (city)       20 cities x 4 slots x 6 industries  = 480
//!   Build (farm)       2 farm breweries                     =   2
//!   Network (single)   39 connections
//!   Network (double)   all unordered rail-link pairs (703; superset)
//!   Develop            6 singles + 36 doubles (ind1, ind2 incl. same)
//!   Sell               47 city slot keys (one per tile)
//!   Loan / Scout / Pass 1 each

use crate::data::IndustryType;
use crate::map::{connections, Loc, ALL_LOCATIONS, CITY_COUNT, MAX_SLOTS_PER_CITY};
use crate::rules::{legal_slot_moves, Move};
use crate::state::GameState;
use rand::Rng;
use std::sync::OnceLock;

pub const INDUSTRY_COUNT: usize = 6;
pub const CITY_BUILD_CELLS: usize = CITY_COUNT * MAX_SLOTS_PER_CITY * INDUSTRY_COUNT; // 480
pub const FARM_BUILD_CELLS: usize = 2;
pub const BUILD_CELLS: usize = CITY_BUILD_CELLS + FARM_BUILD_CELLS; // 482
pub const NETWORK_CELLS: usize = 39; // connections().len()
pub const DEVELOP_SINGLE_CELLS: usize = INDUSTRY_COUNT;
pub const DEVELOP_DOUBLE_CELLS: usize = INDUSTRY_COUNT * INDUSTRY_COUNT;
pub const DEVELOP_CELLS: usize = DEVELOP_SINGLE_CELLS + DEVELOP_DOUBLE_CELLS; // 42
pub const SELL_CELLS: usize = 47; // == crate::state::total_city_slots()
pub const NETWORK_OFFSET: usize = BUILD_CELLS; // 482

/// Action-type bands used by the branched policy head. The policy table is laid
/// out in type-contiguous bands, so a slot's action type is pure band arithmetic
/// over the table offsets (see the layout comment at the top of this file):
///   0 Build (city + farm) | 1 Network (single + double) | 2 Develop |
///   3 Sell | 4 Loan | 5 Scout | 6 Pass
pub const ACTION_TYPE_BUILD: usize = 0;
pub const ACTION_TYPE_NETWORK: usize = 1;
pub const ACTION_TYPE_DEVELOP: usize = 2;
pub const ACTION_TYPE_SELL: usize = 3;
pub const ACTION_TYPE_LOAN: usize = 4;
pub const ACTION_TYPE_SCOUT: usize = 5;
pub const ACTION_TYPE_PASS: usize = 6;
pub const ACTION_TYPE_COUNT: usize = 7;

/// Action type of a policy slot, via band arithmetic over the table layout.
pub fn slot_type(slot: usize) -> usize {
    if slot < BUILD_CELLS {
        ACTION_TYPE_BUILD
    } else if slot < develop_offset() {
        // Network single band + NetworkDouble band are both "network" actions.
        ACTION_TYPE_NETWORK
    } else if slot < sell_offset() {
        ACTION_TYPE_DEVELOP
    } else if slot < loan_offset() {
        ACTION_TYPE_SELL
    } else {
        match slot - loan_offset() {
            0 => ACTION_TYPE_LOAN,
            1 => ACTION_TYPE_SCOUT,
            _ => ACTION_TYPE_PASS,
        }
    }
}

/// Per-slot action type for the whole policy table, as a dense array aligned
/// with the goal head's output (index = policy slot).
pub fn slot_types() -> Vec<usize> {
    (0..policy_table_size()).map(slot_type).collect()
}

/// Runtime guard that the hard-coded cell counts match the actual map data.
fn assert_layout() {
    debug_assert_eq!(SELL_CELLS, crate::state::total_city_slots());
    debug_assert_eq!(NETWORK_CELLS, connections().len());
    debug_assert_eq!(CITY_BUILD_CELLS, CITY_COUNT * MAX_SLOTS_PER_CITY * INDUSTRY_COUNT);
}

/// NetworkDouble occupies the band right after the single networks; its size
/// depends on the static pair table, so later offsets are runtime functions.
pub fn network_double_offset() -> usize {
    NETWORK_OFFSET + NETWORK_CELLS
}

pub fn develop_offset() -> usize {
    network_double_offset() + network_double_cells()
}

pub fn sell_offset() -> usize {
    develop_offset() + DEVELOP_CELLS
}

pub fn loan_offset() -> usize {
    sell_offset() + SELL_CELLS
}

pub fn pass_offset() -> usize {
    loan_offset() + 2
}

/// Static list of unordered rail-link pairs. Double-rail legality in the
/// engine does NOT require the two links to share a node (both only need to be
/// in the player's network after the first is placed — matching the npow
/// reference), so the policy table must cover every unordered rail pair as a
/// superset. The legality mask decides what is actually reachable.
fn double_rail_pairs() -> &'static Vec<(usize, usize)> {
    static PAIRS: OnceLock<Vec<(usize, usize)>> = OnceLock::new();
    PAIRS.get_or_init(|| {
        let conns = connections();
        let mut pairs = Vec::new();
        for i in 0..conns.len() {
            if !conns[i].rail {
                continue;
            }
            for j in (i + 1)..conns.len() {
                if conns[j].rail {
                    pairs.push((conns[i].id, conns[j].id));
                }
            }
        }
        pairs
    })
}

pub fn network_double_cells() -> usize {
    double_rail_pairs().len()
}

fn double_rail_index(conn1: usize, conn2: usize) -> Option<usize> {
    let (lo, hi) = if conn1 < conn2 {
        (conn1, conn2)
    } else {
        (conn2, conn1)
    };
    double_rail_pairs().iter().position(|(a, b)| *a == lo && *b == hi)
}

/// Total number of policy slots (the network's policy-head output width).
pub fn policy_table_size() -> usize {
    assert_layout();
    pass_offset() + 1
}

/// Map a move to the policy slot(s) it occupies. Build/Network/Develop/Loan/
/// Scout/Pass produce exactly one slot; a multi-tile Sell produces one per
/// tile. Returns an empty vec for moves that have no canonical slot.
pub fn move_slots(mv: &Move) -> Vec<usize> {
    match mv {
        Move::Build {
            loc,
            slot_index,
            ind,
            ..
        } => {
            if loc.is_city() {
                let city = *loc as usize;
                if city < CITY_COUNT && *slot_index < MAX_SLOTS_PER_CITY {
                    vec![city * (MAX_SLOTS_PER_CITY * INDUSTRY_COUNT)
                        + *slot_index * INDUSTRY_COUNT
                        + *ind as usize]
                } else {
                    vec![]
                }
            } else if matches!(*loc, Loc::BreweryNorth | Loc::BrewerySouth) {
                let farm_idx = *loc as usize - Loc::BreweryNorth as usize;
                vec![CITY_BUILD_CELLS + farm_idx]
            } else {
                vec![]
            }
        }
        Move::Network { conn_id, .. } => vec![NETWORK_OFFSET + *conn_id],
        Move::NetworkDouble {
            conn1, conn2, ..
        } => double_rail_index(*conn1, *conn2)
            .map(|i| network_double_offset() + i)
            .into_iter()
            .collect(),
        Move::Develop { ind1, ind2, .. } | Move::ResolveFreeDevelop { ind1, ind2 } => {
            let base = develop_offset();
            match ind2 {
                Some(i2) => vec![base + DEVELOP_SINGLE_CELLS + (*ind1 as usize) * INDUSTRY_COUNT + *i2 as usize],
                None => vec![base + *ind1 as usize],
            }
        }
        Move::Sell { keys, .. } => keys
            .iter()
            .map(|k| sell_offset() + *k)
            .collect(),
        Move::Loan { .. } => vec![loan_offset()],
        Move::Scout { .. } => vec![loan_offset() + 1],
        Move::Pass { .. } => vec![pass_offset()],
    }
}

/// The set of policy slots reachable by at least one legal move (the mask).
/// Uses the slot-level generator (one representative per slot) — the same
/// coverage as `legal_moves` de-duplicated, but far cheaper.
pub fn legal_mask(state: &mut GameState<impl Rng + Clone>) -> Vec<usize> {
    let mut seen = std::collections::HashSet::new();
    for sm in legal_slot_moves(state) {
        seen.insert(sm.slot);
    }
    let mut out: Vec<usize> = seen.into_iter().collect();
    out.sort_unstable();
    out
}

/// Human-readable hint for a policy slot (debugging / UI), best-effort.
pub fn describe_slot(slot: usize) -> String {
    if slot < CITY_BUILD_CELLS {
        let city = slot / (MAX_SLOTS_PER_CITY * INDUSTRY_COUNT);
        let rem = slot % (MAX_SLOTS_PER_CITY * INDUSTRY_COUNT);
        let slot_idx = rem / INDUSTRY_COUNT;
        let ind = rem % INDUSTRY_COUNT;
        return format!(
            "Build {} @ {} slot {}",
            IndustryType::ALL[ind].name(),
            ALL_LOCATIONS[city].name(),
            slot_idx
        );
    }
    if slot < BUILD_CELLS {
        let farm = slot - CITY_BUILD_CELLS;
        return format!("Build Brewery @ farm {}", if farm == 0 { "N" } else { "S" });
    }
    if slot < NETWORK_OFFSET + NETWORK_CELLS {
        return format!("Network link {}", slot - NETWORK_OFFSET);
    }
    if slot < develop_offset() {
        let i = slot - (NETWORK_OFFSET + NETWORK_CELLS);
        let pairs = double_rail_pairs();
        if let Some((a, b)) = pairs.get(i) {
            return format!("Double rail {}-{}", a, b);
        }
    }
    if slot < sell_offset() {
        let d = slot - develop_offset();
        if d < DEVELOP_SINGLE_CELLS {
            return format!("Develop {}", IndustryType::ALL[d].name());
        }
        let d2 = d - DEVELOP_SINGLE_CELLS;
        let i1 = d2 / INDUSTRY_COUNT;
        let i2 = d2 % INDUSTRY_COUNT;
        return format!(
            "Develop {} + {}",
            IndustryType::ALL[i1].name(),
            IndustryType::ALL[i2].name()
        );
    }
    if slot < loan_offset() {
        return format!("Sell tile @ slot {}", slot - sell_offset());
    }
    match slot - loan_offset() {
        0 => "Loan".into(),
        1 => "Scout (wild cards)".into(),
        _ => "Pass".into(),
    }
}
