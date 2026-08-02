//! State -> Tensor feature encoding (Brass: Birmingham).
//!
//! Layout (all normalized to ~[0,1], mostly one-hot / fractional):
//!
//!   board   (17, 49)  plane-major. 49 cells = 47 city slots + 2 brewery farms.
//!     0 occupied | 1-4 owner one-hot | 5-10 industry one-hot |
//!     11 flipped | 12 cubes/5 | 13 level/8 | 14 vp/20 | 15 income/7 | 16 is_farm
//!
//!   links   (6, 39)  plane-major over the 39 connections.
//!     0 built | 1 is_canal | 2-5 owner one-hot
//!
//!   global  (50,)
//!     0 era | 1 round/8 | 2 rounds_left/8 | 3-6 cur-player one-hot |
//!     7 actions_this_turn/actions_per_turn |
//!     8-39 per-player (8 each, players 0..4):
//!         money/100, (income+10)/40, vp/200, canal_links/14, rail_links/14,
//!         has_wild_location, has_wild_industry, develops_sum/8 |
//!     40 coal_market/14 | 41 iron_market/10 | 42 coal_price/8 | 43 iron_price/6 |
//!     44-47 turn_order/4 | 48 pending_bonus_active | 49 pending_bonus_player/4
//!
//!   own_hand  (35,)  bag-of-cards for the current player:
//!     0-26 location counts/8 | 27-32 industry counts/8 |
//!     33 wild-location/8 | 34 wild-industry/8
//!
//!   opp_hands (3*35,) same bag-of-cards for the other players (determinized).

use crate::data::Era;
use crate::heuristic_ai::estimate_rounds_remaining;
use crate::map::ALL_LOCATIONS;
use crate::state::{Card, GameState, PendingBonus};
use rand::Rng;

const LOC_COUNT: usize = ALL_LOCATIONS.len(); // 27

pub const BOARD_PLANES: usize = 17;
pub const BOARD_CELLS: usize = 49; // 47 city slots + 2 farms
pub const LINK_PLANES: usize = 6;
pub const LINK_CELLS: usize = 39;
pub const GLOBAL_LEN: usize = 50;
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
        self.board.len() + self.links.len() + self.global.len()
            + self.own_hand.len() + self.opp_hands.len()
    }
}

fn hand_bag(state: &GameState<impl Rng>, pid: usize) -> Vec<f32> {
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

fn global_vec(state: &GameState<impl Rng>) -> Vec<f32> {
    let mut g = vec![0.0f32; GLOBAL_LEN];
    let pid = state.current_player_id();

    g[0] = if state.era == Era::Rail { 1.0 } else { 0.0 };
    g[1] = (state.round as f32 / 8.0).min(1.0);
    g[2] = (estimate_rounds_remaining(state) as f32 / 8.0).min(1.0);
    g[3 + pid.min(MAX_PLAYERS - 1)] = 1.0;
    g[7] = if state.actions_per_turn > 0 {
        state.actions_this_turn as f32 / state.actions_per_turn as f32
    } else {
        0.0
    };

    for p in 0..MAX_PLAYERS {
        let base = 8 + p * 8;
        let player = match state.players.get(p) {
            Some(x) => x,
            None => continue,
        };
        g[base] = (player.money as f32 / 100.0).clamp(0.0, 1.0);
        g[base + 1] = ((player.income_level() as f32 + 10.0) / 40.0).clamp(0.0, 1.0);
        g[base + 2] = (player.vp as f32 / 200.0).min(1.0);
        g[base + 3] = (player.canal_links as f32 / 14.0).min(1.0);
        g[base + 4] = (player.rail_links as f32 / 14.0).min(1.0);
        g[base + 5] = player.has_wild_location as u8 as f32;
        g[base + 6] = player.has_wild_industry as u8 as f32;
        g[base + 7] =
            ((player.develops_in_canal + player.develops_in_rail) as f32 / 8.0).min(1.0);
    }

    g[40] = (state.coal_market as f32 / 14.0).min(1.0);
    g[41] = (state.iron_market as f32 / 10.0).min(1.0);
    g[42] = (state.coal_price() as f32 / 8.0).min(1.0);
    g[43] = (state.iron_price() as f32 / 6.0).min(1.0);

    for (i, &t) in state.turn_order.iter().enumerate().take(MAX_PLAYERS) {
        g[44 + i] = t as f32 / MAX_PLAYERS as f32;
    }

    if let Some(PendingBonus::FreeDevelop { player_id, .. }) = state.pending_bonus {
        g[48] = 1.0;
        g[49] = (player_id as f32 / MAX_PLAYERS as f32).min(1.0);
    }
    g
}

fn board_vec(state: &GameState<impl Rng>) -> Vec<f32> {
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

fn links_vec(state: &GameState<impl Rng>) -> Vec<f32> {
    let mut l = vec![0.0f32; LINK_PLANES * LINK_CELLS];
    for (id, link) in state.links.iter().enumerate().take(LINK_CELLS) {
        if let Some(link) = link {
            l[0 * LINK_CELLS + id] = 1.0;
            if link.is_canal {
                l[1 * LINK_CELLS + id] = 1.0;
            }
            if link.player < MAX_PLAYERS {
                l[(2 + link.player) * LINK_CELLS + id] = 1.0;
            }
        }
    }
    l
}

/// Encode a full game state from the current player's perspective. Opponent
/// hands are read from whatever is in the state (caller decides determinized
/// vs belief); this encoder is agnostic to their origin.
pub fn state_to_tensor(state: &GameState<impl Rng>) -> EncodedTensor {
    let pid = state.current_player_id();
    let board = board_vec(state);
    let links = links_vec(state);
    let global = global_vec(state);
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
