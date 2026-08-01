//! Network connectivity & resource-flow search.
//!
//! Translated from gameState.js isInNetwork / getConnectedLocations /
//! findCoalSource / findIronSource / findBeerSources.

use crate::data::IndustryType;
use crate::map::{city_slots, connections, Loc};
use crate::state::GameState;
use rand::Rng;
use std::collections::{HashSet, VecDeque};

/// Locations reachable from `start` by following ANY built link (regardless
/// of owner). Includes `start` itself and brewery farms passed through.
pub fn connected_locations(state: &GameState<impl Rng>, start: Loc) -> Vec<Loc> {
    let mut visited = Vec::new();
    let mut queue = VecDeque::new();
    visited.push(start);
    queue.push_back(start);

    while let Some(loc) = queue.pop_front() {
        for c in connections() {
            let Some(_link) = &state.links[c.id] else { continue };
            for nb in neighbors_of(c.a, c.b, c.via_farm, loc) {
                if let Some(n) = nb {
                    if !visited.contains(&n) {
                        visited.push(n);
                        queue.push_back(n);
                    }
                }
            }
        }
    }
    visited
}

/// Is `loc` in player `pid`'s own network (own tile there, or own link touching)?
pub fn is_in_network(state: &GameState<impl Rng>, pid: usize, loc: Loc) -> bool {
    // Own industry at the location
    if loc.is_city() {
        for slot in 0..city_slots(loc).len() {
            if let Some(k) = state.city_slot_key(loc, slot) {
                if let Some(t) = &state.city_tiles[k] {
                    if t.player == pid {
                        return true;
                    }
                }
            }
        }
    }
    if let Some(t) = state.farm_tile(loc) {
        if t.player == pid {
            return true;
        }
    }
    // Own link touching the location
    for c in connections() {
        if let Some(link) = &state.links[c.id] {
            if link.player != pid {
                continue;
            }
            if c.a == loc || c.b == loc || c.via_farm == Some(loc) {
                return true;
            }
        }
    }
    false
}

/// Does the player have any tile or link on the board?
pub fn player_has_presence(state: &GameState<impl Rng>, pid: usize) -> bool {
    if state
        .city_tiles
        .iter()
        .flatten()
        .any(|t| t.player == pid)
    {
        return true;
    }
    if state.farm_tiles.iter().flatten().any(|t| t.player == pid) {
        return true;
    }
    state
        .links
        .iter()
        .flatten()
        .any(|l| l.player == pid)
}

/// Given a connection (a,b,via) and the node you're at, which nodes can you
/// move to next through this connection?
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CoalSourceKind {
    Mine,
    Market,
}

#[derive(Debug, Clone, Copy)]
pub struct CoalSource {
    pub kind: CoalSourceKind,
    /// City slot key for a mine; usize::MAX for market.
    pub key: usize,
    pub price: u8,
    pub free: bool,
}

/// Find coal sources for `loc`, nearest mines first, then market (only if a
/// merchant is reachable). One entry per cube.
pub fn find_coal_sources(state: &GameState<impl Rng>, loc: Loc) -> Vec<CoalSource> {
    let mut sources = Vec::new();
    let mut visited: Vec<(Loc, u8)> = Vec::new();
    let mut queue: VecDeque<(Loc, u8)> = VecDeque::new();
    visited.push((loc, 0));
    queue.push_back((loc, 0));

    while let Some((cur, dist)) = queue.pop_front() {
        if cur.is_city() {
            for slot in 0..city_slots(cur).len() {
                if let Some(k) = state.city_slot_key(cur, slot) {
                    if let Some(tile) = &state.city_tiles[k] {
                        if tile.ind == IndustryType::CoalMine
                            && !tile.flipped
                            && tile.resource_cubes > 0
                        {
                            for _ in 0..tile.resource_cubes {
                                sources.push(CoalSource {
                                    kind: CoalSourceKind::Mine,
                                    key: k,
                                    price: 0,
                                    free: true,
                                });
                            }
                        }
                    }
                }
            }
        }

        for c in connections() {
            let Some(_link) = &state.links[c.id] else { continue };
            for nb in neighbors_of(c.a, c.b, c.via_farm, cur) {
                if let Some(n) = nb {
                    if !visited.iter().any(|(v, _)| *v == n) {
                        visited.push((n, dist + 1));
                        queue.push_back((n, dist + 1));
                    }
                }
            }
        }
    }

    sources.sort_by_key(|s| if s.free { 0 } else { 1 });

    // Market coal requires a connection to a merchant location.
    if visited.iter().any(|(v, _)| v.is_merchant()) {
        for _ in 0..state.coal_market {
            sources.push(CoalSource {
                kind: CoalSourceKind::Market,
                key: usize::MAX,
                price: state.coal_price(),
                free: false,
            });
        }
        for _ in 0..6 {
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

#[derive(Debug, Clone, Copy)]
pub struct IronSource {
    pub key: usize, // city slot key; usize::MAX for market
    pub price: u8,
    pub free: bool,
}

/// Iron needs no connection: any unflipped iron works on the board, then market.
pub fn find_iron_sources(state: &GameState<impl Rng>) -> Vec<IronSource> {
    let mut sources = Vec::new();
    for (k, tile) in state.city_tiles.iter().enumerate() {
        if let Some(t) = tile {
            if t.ind == IndustryType::IronWorks && !t.flipped && t.resource_cubes > 0 {
                for _ in 0..t.resource_cubes {
                    sources.push(IronSource {
                        key: k,
                        price: 0,
                        free: true,
                    });
                }
            }
        }
    }
    for _ in 0..state.iron_market {
        sources.push(IronSource {
            key: usize::MAX,
            price: state.iron_price(),
            free: false,
        });
    }
    for _ in 0..6 {
        sources.push(IronSource {
            key: usize::MAX,
            price: crate::map::IRON_EMPTY_PRICE,
            free: false,
        });
    }
    sources
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BeerSourceKind {
    Own,
    Opponent,
    Merchant,
}

#[derive(Debug, Clone, Copy)]
pub struct BeerSource {
    pub kind: BeerSourceKind,
    pub key: usize, // city slot key; usize::MAX for farm/merchant
    pub farm_idx: Option<usize>,
    pub merchant_idx: Option<usize>,
}

/// Find beer for a sale at `loc`. Own breweries anywhere; opponent breweries
/// connected to `loc`; merchant beer only from `allowed_merchants` indices.
pub fn find_beer_sources(
    state: &GameState<impl Rng>,
    loc: Loc,
    player_id: usize,
    allowed_merchants: &[usize],
) -> Vec<BeerSource> {
    let mut sources = Vec::new();

    for (k, tile) in state.city_tiles.iter().enumerate() {
        if let Some(t) = tile {
            if t.ind == IndustryType::Brewery
                && t.player == player_id
                && !t.flipped
                && t.resource_cubes > 0
            {
                for _ in 0..t.resource_cubes {
                    sources.push(BeerSource {
                        kind: BeerSourceKind::Own,
                        key: k,
                        farm_idx: None,
                        merchant_idx: None,
                    });
                }
            }
        }
    }
    for (idx, tile) in state.farm_tiles.iter().enumerate() {
        if let Some(t) = tile {
            if t.player == player_id && !t.flipped && t.resource_cubes > 0 {
                for _ in 0..t.resource_cubes {
                    sources.push(BeerSource {
                        kind: BeerSourceKind::Own,
                        key: usize::MAX,
                        farm_idx: Some(idx),
                        merchant_idx: None,
                    });
                }
            }
        }
    }

    let connected = connected_locations(state, loc);
    for reachable in connected {
        if reachable.is_city() {
            for slot in 0..city_slots(reachable).len() {
                if let Some(k) = state.city_slot_key(reachable, slot) {
                    if let Some(t) = &state.city_tiles[k] {
                        if t.ind == IndustryType::Brewery
                            && t.player != player_id
                            && !t.flipped
                            && t.resource_cubes > 0
                        {
                            for _ in 0..t.resource_cubes {
                                sources.push(BeerSource {
                                    kind: BeerSourceKind::Opponent,
                                    key: k,
                                    farm_idx: None,
                                    merchant_idx: None,
                                });
                            }
                        }
                    }
                }
            }
        }
        if let Some(idx) = farm_index(reachable) {
            if let Some(t) = &state.farm_tiles[idx] {
                if t.player != player_id && !t.flipped && t.resource_cubes > 0 {
                    for _ in 0..t.resource_cubes {
                        sources.push(BeerSource {
                            kind: BeerSourceKind::Opponent,
                            key: usize::MAX,
                            farm_idx: Some(idx),
                            merchant_idx: None,
                        });
                    }
                }
            }
        }
    }

    if !allowed_merchants.is_empty() {
        let connected = connected_locations(state, loc);
        for &mi in allowed_merchants {
            if let Some(mt) = state.merchants.get(mi) {
                if mt.has_beer && connected.contains(&mt.loc) {
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

fn farm_index(loc: Loc) -> Option<usize> {
    match loc {
        Loc::BreweryNorth => Some(0),
        Loc::BrewerySouth => Some(1),
        _ => None,
    }
}

/// Cheapest coal source reachable from either endpoint of a connection.
pub fn cheapest_coal_for_connection(
    state: &GameState<impl Rng>,
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
pub fn is_resource_depleted(state: &GameState<impl Rng>, ind: IndustryType) -> bool {
    match ind {
        IndustryType::CoalMine => {
            if state.coal_market > 0 {
                return false;
            }
            !state.city_tiles.iter().flatten().any(|t| {
                t.ind == IndustryType::CoalMine && !t.flipped && t.resource_cubes > 0
            })
        }
        IndustryType::IronWorks => {
            if state.iron_market > 0 {
                return false;
            }
            !state.city_tiles.iter().flatten().any(|t| {
                t.ind == IndustryType::IronWorks && !t.flipped && t.resource_cubes > 0
            })
        }
        _ => false,
    }
}
