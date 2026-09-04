//! The scoring currency and board-value estimators.
//!
//! Network and Build actions use local VP-equivalent policies because their
//! inputs and time factors are self-contained. Board VP estimation lives here
//! (read-only mirror of
//! `gameplay::scoring::score_era`), so scorers and the MCTS leaf evaluator
//! share one implementation.

use crate::map::{city_slots, connections};
use crate::state::GameState;

// ---------------------------------------------------------------------------
// Board VP estimation (read-only mirror of gameplay::scoring::score_era)
// ---------------------------------------------------------------------------

/// Estimated end-of-era VP currently on the board for one player.
#[derive(Debug, Clone, Copy, Default)]
pub struct PlayerVpEstimate {
    /// VP of already flipped industry tiles (scored in full at era end).
    pub flipped_industry: f64,
    /// Expected VP of unflipped tiles (they may still flip).
    pub unflipped_expected: f64,
    /// VP the player's links score right now: flipped industry link icons
    /// around each link, +2 per touched merchant, including via-farm icons.
    pub link_current: f64,
}

/// Read-only estimate of the VP `pid` has banked on the board. Mirrors the
/// era-end settlement in `gameplay::scoring::score_era` without mutating
/// state.
pub fn estimate_player_vp(
    state: &GameState,
    pid: usize,
    unflipped_vp_share: f64,
) -> PlayerVpEstimate {
    let mut est = PlayerVpEstimate::default();

    for tile in state.city_tiles.iter().flatten() {
        if tile.player != pid {
            continue;
        }
        let vp = tile.def.vp as f64;
        if tile.flipped {
            est.flipped_industry += vp;
        } else {
            est.unflipped_expected += vp * unflipped_vp_share;
        }
    }
    for tile in state.farm_tiles.iter().flatten() {
        if tile.player != pid {
            continue;
        }
        let vp = tile.def.vp as f64;
        if tile.flipped {
            est.flipped_industry += vp;
        } else {
            est.unflipped_expected += vp * unflipped_vp_share;
        }
    }

    for (id, link) in state.links.iter().enumerate() {
        let Some(l) = link else { continue };
        if l.player != pid {
            continue;
        }
        est.link_current += link_icons_at(state, connections()[id].a);
        est.link_current += link_icons_at(state, connections()[id].b);
        if let Some(via) = connections()[id].via_farm {
            est.link_current += link_icons_at(state, via);
        }
    }

    est
}

/// Link icons one location contributes to an adjacent owned link: merchants
/// score a flat 2, flipped industries their printed link VP. Same rule as
/// the era-end settlement (`scoring::score_era`), read-only.
pub fn link_icons_at(state: &GameState, loc: crate::map::Loc) -> f64 {
    if loc.is_merchant() {
        return 2.0;
    }
    if loc.is_city() {
        let mut v = 0.0;
        for slot in 0..city_slots(loc).len() {
            if let Some(k) = state.city_slot_key(loc, slot)
                && let Some(t) = &state.city_tiles[k]
                && t.flipped
            {
                v += t.def.link_vp as f64;
            }
        }
        return v;
    }
    if let Some(t) = state.farm_tile(loc)
        && t.flipped
    {
        return t.def.link_vp as f64;
    }
    0.0
}

/// Potential link icons an *unoccupied* location would contribute once
/// developed: every empty city slot is worth 1 future VP, or 2 when it can
/// host a brewery; an unbuilt brewery farm contributes 2.
pub fn future_link_node_potential(state: &GameState, loc: crate::map::Loc) -> usize {
    use crate::data::IndustryType;
    if loc.is_farm() {
        return usize::from(state.farm_tile(loc).is_none()) * 2;
    }
    if !loc.is_city() {
        return 0;
    }
    city_slots(loc)
        .iter()
        .enumerate()
        .filter(|(slot_idx, _)| {
            state
                .city_slot_key(loc, *slot_idx)
                .is_some_and(|k| state.city_tiles[k].is_none())
        })
        .map(|(_, slot_types)| {
            if slot_types.contains(&IndustryType::Brewery) {
                2
            } else {
                1
            }
        })
        .sum()
}

/// `(current_link_vp, future_link_vp)` a new link over `cities` would
/// provide, including its via-farm leg.
pub fn link_current_and_potential_vps(
    state: &GameState,
    conn_id: usize,
    cities: &[crate::map::Loc; 2],
) -> (f64, f64) {
    let mut current = 0.0;
    let mut future = 0.0;
    for loc in cities {
        current += link_icons_at(state, *loc);
        future += future_link_node_potential(state, *loc) as f64;
    }
    if let Some(farm) = connections()[conn_id].via_farm {
        current += link_icons_at(state, farm);
        future += future_link_node_potential(state, farm) as f64;
    }
    (current, future)
}

// ---------------------------------------------------------------------------
// Market model
// ---------------------------------------------------------------------------

/// How much cash selling `cubes` into the market earns right now, and
/// whether that sale empties the tile (flipping it for income). Mirrors
/// `GameState::auto_sell_to_market`: cubes fill market spaces from the
/// cheapest up, stopping when the market is full.
#[derive(Debug, Clone, Copy)]
pub struct MarketSale {
    pub cash: f64,
    pub sold: u8,
    pub total: u8,
    pub flips: bool,
}

pub fn simulate_market_sale(state: &GameState, is_coal: bool, cubes: u8) -> MarketSale {
    let (prices, market) = if is_coal {
        (crate::map::COAL_MARKET_PRICES.as_slice(), state.coal_market)
    } else {
        (crate::map::IRON_MARKET_PRICES.as_slice(), state.iron_market)
    };
    let mut cash = 0.0;
    let mut m = market;
    let mut sold = 0u8;
    while sold < cubes && m < prices.len() {
        let idx = prices.len() - m - 1;
        cash += prices[idx] as f64;
        m += 1;
        sold += 1;
    }
    MarketSale {
        cash,
        sold,
        total: cubes,
        flips: sold == cubes && cubes > 0,
    }
}

/// Unsold market headroom as a fraction of capacity: 1 = empty market
/// (starving), 0 = full market (cannot absorb more cubes).
pub fn market_scarcity(state: &GameState, is_coal: bool) -> f64 {
    let (capacity, market) = if is_coal {
        (
            crate::map::COAL_MARKET_PRICES.len() as f64,
            state.coal_market as f64,
        )
    } else {
        (
            crate::map::IRON_MARKET_PRICES.len() as f64,
            state.iron_market as f64,
        )
    };
    ((capacity - market) / capacity).clamp(0.0, 1.0)
}

/// Demand "heat" of a buy price inside a window `(price - base) / span`,
/// clamped to 0..1. Prices at/above `base + span` are maximally hot.
pub fn price_heat(price: u8, base: f64, span: f64) -> f64 {
    ((price as f64 - base) / span).clamp(0.0, 1.0)
}
