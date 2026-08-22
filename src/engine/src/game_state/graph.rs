//! Network connectivity & resource-flow search.
//!
//! Translated from gameState.js isInNetwork / getConnectedLocations /
//! findCoalSource / findIronSource / findBeerSources.
//!
//! Connectivity is NOT searched here: `GameState` maintains a lazily-rebuilt
//! connected-component cache (bitmasks per location) plus cached lists of free
//! coal/iron cubes, so the hot paths below are O(free sources + market slots).

use crate::data::IndustryType;
use crate::map::{ALL_LOCATIONS, Loc};
use crate::state::{BeerCubeEntry, GameState};
use std::collections::HashSet;

/// Locations in player `pid`'s own network (own tile, or reachable through the
/// player's own links). O(1) mask iteration via the `GameState` network-mask
/// cache.
pub fn network_locations(state: &GameState, pid: usize) -> Vec<Loc> {
    let mask = state.network_mask(pid);
    (0..27u8)
        .filter(|i| mask & (1 << i) != 0)
        .map(|i| ALL_LOCATIONS[i as usize])
        .collect()
}

/// Locations reachable from `start` by following ANY built link (regardless
/// of owner). Includes `start` itself and brewery farms passed through.
/// Backed by the `GameState` connected-component cache (no per-call BFS).
pub fn connected_locations(state: &GameState, start: Loc) -> Vec<Loc> {
    let mask = state.connected_mask(start);
    (0..27u8)
        .filter(|i| mask & (1 << i) != 0)
        .map(|i| crate::map::ALL_LOCATIONS[i as usize])
        .collect()
}

/// Is `loc` in player `pid`'s own network (own tile there, or own link touching)?
/// O(1) via the cached per-player network mask.
pub fn is_in_network(state: &GameState, pid: usize, loc: Loc) -> bool {
    state.network_mask(pid) & (1u32 << (loc as u8)) != 0
}

/// Does the player have any tile or link on the board? O(1) via the cached
/// per-player network mask (any presence implies a non-empty network).
pub fn player_has_presence(state: &GameState, pid: usize) -> bool {
    state.network_mask(pid) != 0
}

/// Given a connection (a,b,via) and the node you're at, which nodes can you
/// move to next through this connection?
#[allow(dead_code)]
fn neighbors_of(a: Loc, b: Loc, via: Option<Loc>, here: Loc) -> [Option<Loc>; 2] {
    if here == a {
        return [Some(b), via];
    }
    if here == b {
        return [Some(a), via];
    }
    if via == Some(here) {
        return [Some(a), Some(b)];
    }
    [None, None]
}

// ---------------------------------------------------------------------------
// Resource sources
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum CoalSourceKind {
    Mine,
    Market,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CoalSource {
    pub kind: CoalSourceKind,
    /// City slot key for a mine; usize::MAX for market.
    pub key: usize,
    pub price: u8,
    pub free: bool,
}

/// Find coal sources for `loc`: unflipped coal mines reachable via any built
/// link (free, one entry per cube), then market cubes at their own slot price
/// (only if a merchant is reachable), then General Supply at the empty price.
/// Free entries are ordered by slot key for determinism.
pub fn find_coal_sources(state: &GameState, loc: Loc) -> Vec<CoalSource> {
    let mask = state.connected_mask(loc);
    let mut sources: Vec<CoalSource> = Vec::new();
    for m in state
        .free_coal_mines
        .iter()
        .filter(|m| mask & (1u32 << (m.loc as u8)) != 0)
    {
        for _ in 0..m.cubes {
            sources.push(CoalSource {
                kind: CoalSourceKind::Mine,
                key: m.key,
                price: 0,
                free: true,
            });
        }
    }
    sources.sort_by_key(|s| s.key);

    // Market coal requires a connection to a merchant location. Each cube is
    // listed at its own market-slot price, cheapest first, so a multi-cube
    // purchase pays the ascending slot prices (the first entry matches
    // `state.coal_price()`). After the real cubes come GENERAL_SUPPLY_CAP
    // General-Supply entries at the empty price.
    if mask & crate::map::MERCHANT_LOC_MASK != 0 {
        let prices = crate::map::COAL_MARKET_PRICES;
        for i in 0..state.coal_market {
            sources.push(CoalSource {
                kind: CoalSourceKind::Market,
                key: usize::MAX,
                price: prices[prices.len() - state.coal_market + i],
                free: false,
            });
        }
        for _ in 0..crate::map::GENERAL_SUPPLY_CAP {
            sources.push(CoalSource {
                kind: CoalSourceKind::Market,
                key: usize::MAX,
                price: crate::map::COAL_EMPTY_PRICE,
                free: false,
            });
        }
    }

    sources
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct IronSource {
    pub key: usize, // city slot key; usize::MAX for market
    pub price: u8,
    pub free: bool,
}
/// Iron needs no connection: any unflipped iron works on the board (free, one
/// entry per cube), then market cubes at their own slot price, then General
/// Supply at the empty price. Free entries are ordered by slot key.
pub fn find_iron_sources(state: &GameState) -> Vec<IronSource> {
    let mut sources: Vec<IronSource> = Vec::new();
    for w in &state.free_iron_works {
        for _ in 0..w.cubes {
            sources.push(IronSource {
                key: w.key,
                price: 0,
                free: true,
            });
        }
    }
    sources.sort_by_key(|s| s.key);

    // Each market cube at its own slot price, cheapest first (a multi-cube
    // purchase pays ascending slot prices); then GENERAL_SUPPLY_CAP
    // General-Supply entries at the empty price. No connection is ever required
    // for iron.
    let prices = crate::map::IRON_MARKET_PRICES;
    for i in 0..state.iron_market {
        sources.push(IronSource {
            key: usize::MAX,
            price: prices[prices.len() - state.iron_market + i],
            free: false,
        });
    }
    for _ in 0..crate::map::GENERAL_SUPPLY_CAP {
        sources.push(IronSource {
            key: usize::MAX,
            price: crate::map::IRON_EMPTY_PRICE,
            free: false,
        });
    }
    sources
}

/// Cash needed to draw `needed` coal cubes at `loc`: free reachable mines
/// first, then cheapest market slots (per-cube ascending), then General Supply
/// at the empty price. `None` if a paid shortfall remains with no merchant
/// reachable (the market and General Supply are unavailable).
pub fn coal_purchase_cost(state: &GameState, loc: Loc, needed: usize) -> Option<i32> {
    let mask = state.connected_mask(loc);
    let free: usize = state
        .free_coal_mines
        .iter()
        .filter(|m| mask & (1u32 << (m.loc as u8)) != 0)
        .map(|m| m.cubes as usize)
        .sum();
    let paid = needed.saturating_sub(free);
    if paid == 0 {
        return Some(0);
    }
    if mask & crate::map::MERCHANT_LOC_MASK == 0 {
        return None;
    }
    let prices = crate::map::COAL_MARKET_PRICES;
    let mut cost = 0i32;
    let mut remaining = paid;
    for i in 0..state.coal_market {
        if remaining == 0 {
            break;
        }
        cost += prices[prices.len() - state.coal_market + i] as i32;
        remaining -= 1;
    }
    if remaining > 0 {
        cost += remaining as i32 * crate::map::COAL_EMPTY_PRICE as i32;
    }
    Some(cost)
}

/// Cash needed to draw `needed` iron cubes: free works first, then cheapest
/// market slots, then General Supply at the empty price. Always affordable
/// (General Supply is unlimited; iron needs no connection).
pub fn iron_purchase_cost(state: &GameState, needed: usize) -> i32 {
    let free: usize = state.free_iron_works.iter().map(|w| w.cubes as usize).sum();
    let paid = needed.saturating_sub(free);
    if paid == 0 {
        return 0;
    }
    let prices = crate::map::IRON_MARKET_PRICES;
    let mut cost = 0i32;
    let mut remaining = paid;
    for i in 0..state.iron_market {
        if remaining == 0 {
            break;
        }
        cost += prices[prices.len() - state.iron_market + i] as i32;
        remaining -= 1;
    }
    if remaining > 0 {
        cost += remaining as i32 * crate::map::IRON_EMPTY_PRICE as i32;
    }
    cost
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BeerSourceKind {
    Own,
    Opponent,
    Merchant,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BeerSource {
    pub kind: BeerSourceKind,
    pub key: usize, // city slot key; usize::MAX for farm/merchant
    pub farm_idx: Option<usize>,
    pub merchant_idx: Option<usize>,
}

/// Find beer for a sale at `loc`. Own breweries anywhere; opponent breweries
/// connected to `loc`; merchant beer only from `allowed_merchants` indices.
/// Backed by the `free_beer_cubes` cache (unflipped breweries with cubes);
/// connectivity for opponent breweries is applied via the component-mask cache.
/// Output order is preserved exactly: own city breweries by key, then own farms
/// by index, then connected opponent breweries in ascending location order.
pub fn find_beer_sources(
    state: &GameState,
    loc: Loc,
    player_id: usize,
    allowed_merchants: &[usize],
) -> Vec<BeerSource> {
    let mask = state.connected_mask(loc);
    let mut sources = Vec::new();

    // Own breweries anywhere, in deterministic order (city slots by key, then
    // farms by index) — identical to the pre-cache full-board scan.
    let mut own_city: Vec<&BeerCubeEntry> = state
        .free_beer_cubes
        .iter()
        .filter(|b| b.owner == player_id && b.farm_idx.is_none())
        .collect();
    own_city.sort_by_key(|b| b.key);
    let mut own_farm: Vec<&BeerCubeEntry> = state
        .free_beer_cubes
        .iter()
        .filter(|b| b.owner == player_id && b.farm_idx.is_some())
        .collect();
    own_farm.sort_by_key(|b| b.farm_idx.unwrap());
    for b in own_city.into_iter().chain(own_farm) {
        for _ in 0..b.cubes {
            sources.push(BeerSource {
                kind: BeerSourceKind::Own,
                key: b.key,
                farm_idx: b.farm_idx,
                merchant_idx: None,
            });
        }
    }

    // Opponent breweries connected to `loc`, in ascending location order (then
    // slot/farm order within a location) — identical to the pre-cache
    // connected-locations scan.
    let mut opp: Vec<&BeerCubeEntry> = state
        .free_beer_cubes
        .iter()
        .filter(|b| b.owner != player_id && mask & (1u32 << (b.loc as u8)) != 0)
        .collect();
    opp.sort_by_key(|b| (b.loc as u8, b.farm_idx.unwrap_or(usize::MAX), b.key));
    for b in opp {
        for _ in 0..b.cubes {
            sources.push(BeerSource {
                kind: BeerSourceKind::Opponent,
                key: b.key,
                farm_idx: b.farm_idx,
                merchant_idx: None,
            });
        }
    }

    if !allowed_merchants.is_empty() {
        for &mi in allowed_merchants {
            if let Some(mt) = state.merchants.get(mi) {
                if mt.has_beer && mask & (1u32 << (mt.loc as u8)) != 0 {
                    sources.push(BeerSource {
                        kind: BeerSourceKind::Merchant,
                        key: usize::MAX,
                        farm_idx: None,
                        merchant_idx: Some(mi),
                    });
                }
            }
        }
    }

    sources
}

/// Count beer cubes available for a sale without constructing/sorting source
/// descriptors. Used by heuristic scoring when only availability is needed.
pub fn count_beer_sources(
    state: &GameState,
    loc: Loc,
    player_id: usize,
    allowed_merchants: &[usize],
) -> usize {
    let mask = state.connected_mask(loc);
    let mut count = 0usize;
    for b in &state.free_beer_cubes {
        if b.owner == player_id || mask & (1u32 << (b.loc as u8)) != 0 {
            count += b.cubes as usize;
        }
    }
    if !allowed_merchants.is_empty() {
        count += allowed_merchants
            .iter()
            .filter(|&&mi| {
                state
                    .merchants
                    .get(mi)
                    .is_some_and(|mt| mt.has_beer && mask & (1u32 << (mt.loc as u8)) != 0)
            })
            .count();
    }
    count
}

/// Cheapest coal source reachable from either endpoint of a connection.
pub fn cheapest_coal_for_connection(
    state: &GameState,
    conn: &crate::map::Connection,
) -> Option<CoalSource> {
    let mut best: Option<CoalSource> = None;
    let mut seen: HashSet<(u8, usize)> = HashSet::new();
    for endpoint in [conn.a, conn.b] {
        for src in find_coal_sources(state, endpoint) {
            let dedupe = match src.kind {
                CoalSourceKind::Mine => (1u8, src.key),
                CoalSourceKind::Market => (0u8, 0usize),
            };
            if !seen.insert(dedupe) {
                continue;
            }
            let sv = if src.free { 0 } else { src.price as i16 };
            let better = match &best {
                None => true,
                Some(b) => {
                    let bv = if b.free { 0 } else { b.price as i16 };
                    sv < bv
                }
            };
            if better {
                best = Some(src);
            }
        }
    }
    best
}

/// Is the resource (coal or iron) fully depleted everywhere (board + market)?
/// Only meaningful for coal mines / iron works (overbuilding condition).
pub fn is_resource_depleted(state: &GameState, ind: IndustryType) -> bool {
    match ind {
        IndustryType::CoalMine => {
            if state.coal_market > 0 {
                return false;
            }
            !state.free_coal_mines.iter().any(|m| m.cubes > 0)
        }
        IndustryType::IronWorks => {
            if state.iron_market > 0 {
                return false;
            }
            !state.free_iron_works.iter().any(|w| w.cubes > 0)
        }
        _ => false,
    }
}
