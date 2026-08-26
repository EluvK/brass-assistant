//! State -> Tensor feature encoding (Brass: Birmingham).
//!
//! Layout (all normalized to ~[0,1], mostly one-hot / fractional):
//!
//!   board   (17, 49)  plane-major. 49 cells = 47 city slots + 2 brewery farms.
//!     0 occupied | 1-4 owner one-hot | 5-10 industry one-hot |
//!     11 flipped | 12 cubes/5 | 13 level/8 | 14 vp/20 | 15 income/7 | 16 is_farm
//!
//!   links   (7, 39)  plane-major over the 39 connections.
//!     0 can_build_canal | 1 can_build_rail | 2 built | 3-6 owner one-hot
//!
//!   global  (114,)
//!     temporal (4) | per-player public state (4 * 16) |
//!     coal-market one-hot (15) | iron-market one-hot (11) |
//!     current execution queue (4 * 4)
//!
//!   own_hand  (35,)  bag-of-cards for the current player:
//!     0-26 location counts/8 | 27-32 industry counts/8 |
//!     33 wild-location/8 | 34 wild-industry/8
//!
//!   opp_hands (3*35,) same bag-of-cards for the other players (determinized).

use crate::data::Era;
use crate::map::{ALL_LOCATIONS, CITY_COUNT, connections};
use crate::state::{Card, GameState};

pub const LOC_COUNT: usize = ALL_LOCATIONS.len(); // 27

pub const BOARD_PLANES: usize = 17;
pub const BOARD_CELLS: usize = 49; // 47 city slots + 2 farms
pub const LINK_PLANES: usize = 7;
pub const LINK_CELLS: usize = 39;
pub const GLOBAL_LEN: usize = 114;
pub const STATE_FEATURE_SCHEMA_VERSION: usize = 3;
pub const HAND_LEN: usize = 35; // 27 locations + 6 industries + 2 wilds
pub const MAX_PLAYERS: usize = 4;
pub const MAX_OPPONENTS: usize = 3;

pub struct EncodedTensor {
    pub board: Vec<f32>,
    pub links: Vec<f32>,
    pub global: Vec<f32>,
    pub own_hand: Vec<f32>,
    pub opp_hands: Vec<f32>,
}

impl EncodedTensor {
    pub fn total_len(&self) -> usize {
        self.board.len()
            + self.links.len()
            + self.global.len()
            + self.own_hand.len()
            + self.opp_hands.len()
    }
}

fn hand_bag(state: &GameState, pid: usize) -> Vec<f32> {
    let mut bag = vec![0.0f32; HAND_LEN];
    let hand = &state.players[pid].hand;
    let n = hand.len().max(1) as f32;
    for card in hand {
        match card {
            Card::Location(loc) => {
                bag[*loc as usize] += 1.0;
            }
            Card::Industry { industries, n: nn } => {
                for &ind in &industries[..*nn as usize] {
                    bag[LOC_COUNT + ind as usize] += 1.0;
                }
            }
            Card::WildLocation => bag[LOC_COUNT + 6] += 1.0,
            Card::WildIndustry => bag[LOC_COUNT + 7] += 1.0,
        }
    }
    for v in bag.iter_mut() {
        *v /= n;
    }
    bag
}

/// Public board-cell -> location mapping, owned by the engine so Python graph
/// models never duplicate the static map definition. City cells precede the
/// two brewery-farm cells, matching `board_vec`.
pub fn board_cell_locations() -> Vec<usize> {
    let mut out = Vec::with_capacity(BOARD_CELLS);
    for &loc in ALL_LOCATIONS[..CITY_COUNT].iter() {
        out.extend(std::iter::repeat_n(
            loc as usize,
            crate::map::city_slots(loc).len(),
        ));
    }
    out.push(crate::map::Loc::BreweryNorth as usize);
    out.push(crate::map::Loc::BrewerySouth as usize);
    debug_assert_eq!(out.len(), BOARD_CELLS);
    out
}

/// Flat `(a, b)` endpoint pairs for the 39 map connections. A via-farm edge
/// retains its rule-level endpoints; `connection_via_farms` exposes its farm.
pub fn connection_endpoints() -> Vec<usize> {
    connections()
        .iter()
        .flat_map(|c| [c.a as usize, c.b as usize])
        .collect()
}

/// Per-connection via-farm location, or `LOC_COUNT` when none exists.
pub fn connection_via_farms() -> Vec<usize> {
    connections()
        .iter()
        .map(|c| c.via_farm.map(|loc| loc as usize).unwrap_or(LOC_COUNT))
        .collect()
}

fn global_vec(state: &GameState, _pid: usize) -> Vec<f32> {
    let mut g = vec![0.0f32; GLOBAL_LEN];

    g[0] = if state.era == Era::Rail { 1.0 } else { 0.0 };
    g[1] = (state.round as f32 / 8.0).min(1.0);
    g[2] = (state.rounds_remaining() as f32 / 10.0).min(1.0);
    g[3] = if state.actions_per_turn > 0 {
        state.actions_this_turn as f32 / state.actions_per_turn as f32
    } else {
        0.0
    };

    for p in 0..MAX_PLAYERS {
        let base = 4 + p * 17;
        let Some(player) = state.players.get(p) else {
            continue;
        };
        g[base] = 1.0;
        g[base + 1] = (player.money as f32 / 200.0).clamp(0.0, 1.0);
        g[base + 2] = (player.hand.len() as f32 / 8.0).min(1.0);
        g[base + 3] = (player.income_space as f32 / 99.0).clamp(0.0, 1.0);
        g[base + 4] = ((player.income_level() as f32 + 10.0) / 40.0).clamp(0.0, 1.0);
        g[base + 5] = (player.vp as f32 / 200.0).min(1.0);
        g[base + 6] = (player.canal_links as f32 / 14.0).min(1.0);
        g[base + 7] = (player.rail_links as f32 / 14.0).min(1.0);
        g[base + 8] = player.has_wild_location as u8 as f32;
        g[base + 9] = player.has_wild_industry as u8 as f32;
        for ind in 0..6 {
            let total = crate::state::player_industry_stack(crate::data::IndustryType::ALL[ind])
                .len()
                .max(1);
            g[base + 10 + ind] = (player.industry_next[ind] as f32 / total as f32).min(1.0);
        }
        g[base + 16] = (state.money_spent_this_round[p] as f32 / 100.0).clamp(0.0, 1.0);
    }

    let market_base = 4 + MAX_PLAYERS * 17;
    g[market_base + state.coal_market.min(14)] = 1.0;
    g[market_base + 15 + state.iron_market.min(10)] = 1.0;

    let queue_base = market_base + 26;
    for queue_pos in 0..state.player_count() {
        let seat = state.turn_order[(state.current_index + queue_pos) % state.player_count()];
        g[queue_base + queue_pos * MAX_PLAYERS + seat] = 1.0;
    }

    g
}

fn board_vec(state: &GameState) -> Vec<f32> {
    let mut b = vec![0.0f32; BOARD_PLANES * BOARD_CELLS];
    let city_slots_count = crate::state::total_city_slots(); // 47
    for cell in 0..BOARD_CELLS {
        let (tile, is_farm) = if cell < city_slots_count {
            (state.city_tiles[cell].as_ref(), false)
        } else {
            (state.farm_tiles[cell - city_slots_count].as_ref(), true)
        };
        if let Some(t) = tile {
            b[0 * BOARD_CELLS + cell] = 1.0;
            if t.player < MAX_PLAYERS {
                b[(1 + t.player) * BOARD_CELLS + cell] = 1.0;
            }
            b[(5 + t.ind as usize) * BOARD_CELLS + cell] = 1.0;
            if t.flipped {
                b[11 * BOARD_CELLS + cell] = 1.0;
            }
            b[12 * BOARD_CELLS + cell] = t.resource_cubes as f32 / 5.0;
            b[13 * BOARD_CELLS + cell] = (t.def.level as f32 / 8.0).clamp(0.0, 1.0); // max level = 8 (Manufacturer)
            b[14 * BOARD_CELLS + cell] = t.def.vp as f32 / 20.0;
            b[15 * BOARD_CELLS + cell] = t.def.income as f32 / 7.0;
        }
        if is_farm {
            b[16 * BOARD_CELLS + cell] = 1.0;
        }
    }
    b
}

fn links_vec(state: &GameState) -> Vec<f32> {
    let mut l = vec![0.0f32; LINK_PLANES * LINK_CELLS];
    for connection in connections() {
        l[connection.id] = connection.canal as u8 as f32;
        l[LINK_CELLS + connection.id] = connection.rail as u8 as f32;
    }
    for (id, link) in state.links.iter().enumerate().take(LINK_CELLS) {
        if let Some(link) = link {
            l[2 * LINK_CELLS + id] = 1.0;
            if link.player < MAX_PLAYERS {
                l[(3 + link.player) * LINK_CELLS + id] = 1.0;
            }
        }
    }
    l
}

/// Encode a full game state from an explicit perspective player `pid` (the
/// caller decides whether it equals the current player). Opponent hands are
/// read from whatever is in the state (caller decides determinized vs belief);
/// this encoder is agnostic to their origin.
pub fn state_to_tensor(state: &GameState, pid: usize) -> EncodedTensor {
    let board = board_vec(state);
    let links = links_vec(state);
    let global = global_vec(state, pid);
    let own_hand = hand_bag(state, pid);

    let mut opp_hands = Vec::with_capacity(MAX_OPPONENTS * HAND_LEN);
    for p in 0..MAX_PLAYERS {
        if p == pid || p >= state.player_count() {
            continue;
        }
        opp_hands.extend(hand_bag(state, p));
    }
    // Pad to a fixed 3 opponents (unused players => zero bags).
    let opp_count = state.player_count().saturating_sub(1).min(MAX_OPPONENTS);
    for _ in opp_count..MAX_OPPONENTS {
        opp_hands.extend(std::iter::repeat(0.0f32).take(HAND_LEN));
    }

    EncodedTensor {
        board,
        links,
        global,
        own_hand,
        opp_hands,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand_chacha::ChaCha12Rng;

    #[test]
    fn v2_encodes_static_link_capabilities_and_public_turn_economy() {
        let mut state = GameState::new(ChaCha12Rng::seed_from_u64(7), 4);
        state.spend_money(0, 80);
        let encoded = state_to_tensor(&state, 0);

        assert_eq!(encoded.links.len(), LINK_PLANES * LINK_CELLS);
        // Belper-Leek is rail-only and must be distinguishable while unbuilt.
        assert_eq!(encoded.links[1], 0.0);
        assert_eq!(encoded.links[LINK_CELLS + 1], 1.0);
        assert_eq!(encoded.links[2 * LINK_CELLS + 1], 0.0);

        let p0 = 4;
        assert_eq!(encoded.global[p0 + 2], 1.0, "opening hand has eight cards");
        assert_eq!(encoded.global[p0 + 16], 0.8);
    }
}
