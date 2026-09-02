//! Shared board queries used by multiple scorers.
//!
//! Each predicate here used to be re-implemented inline in several scoring
//! functions; they now live once. All functions are read-only.

use crate::data::IndustryType;
use crate::graph::{connected_locations, count_beer_sources, find_coal_sources, find_iron_sources};
use crate::map::{Loc, connections};
use crate::rules::BuildTarget;
use crate::state::{Card, GameState};

/// A static physical board slot used by the heuristic's long-term card model.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct IndustrySlot {
    pub loc: Loc,
    pub slot_index: usize,
}

#[rustfmt::skip]
const IRON_SLOTS: [IndustrySlot; 9] = [
    IndustrySlot { loc: Loc::Derby, slot_index: 2 },
    IndustrySlot { loc: Loc::StokeOnTrent, slot_index: 1 },
    IndustrySlot { loc: Loc::Walsall, slot_index: 0 },
    IndustrySlot { loc: Loc::Coalbrookdale, slot_index: 0 },
    IndustrySlot { loc: Loc::Coalbrookdale, slot_index: 1 },
    IndustrySlot { loc: Loc::Dudley, slot_index: 1 },
    IndustrySlot { loc: Loc::Birmingham, slot_index: 2 },
    IndustrySlot { loc: Loc::Coventry, slot_index: 2 },
    IndustrySlot { loc: Loc::Redditch, slot_index: 1 },
];

#[rustfmt::skip]
const BREWERY_SLOTS: [IndustrySlot; 11] = [
    IndustrySlot { loc: Loc::Derby, slot_index: 0 },
    IndustrySlot { loc: Loc::Stone, slot_index: 0 },
    IndustrySlot { loc: Loc::Uttoxeter, slot_index: 0 },
    IndustrySlot { loc: Loc::Uttoxeter, slot_index: 1 },
    IndustrySlot { loc: Loc::Stafford, slot_index: 0 },
    IndustrySlot { loc: Loc::BurtonOnTrent, slot_index: 1 },
    IndustrySlot { loc: Loc::Walsall, slot_index: 1 },
    IndustrySlot { loc: Loc::Coalbrookdale, slot_index: 0 },
    IndustrySlot { loc: Loc::Nuneaton, slot_index: 0 },
    IndustrySlot { loc: Loc::BreweryNorth, slot_index: 0 },
    IndustrySlot { loc: Loc::BrewerySouth, slot_index: 0 },
];

/// Static slots relevant to Iron/Brewery hand capacity.
pub fn industry_slots(ind: IndustryType) -> &'static [IndustrySlot] {
    match ind {
        IndustryType::IronWorks => &IRON_SLOTS,
        IndustryType::Brewery => &BREWERY_SLOTS,
        _ => &[],
    }
}

pub fn city_supports_industry(loc: Loc, ind: IndustryType) -> bool {
    loc.is_city() && industry_slots(ind).iter().any(|slot| slot.loc == loc)
}

/// Empty physical slots, independent of cash, network, cards, and era.
pub fn empty_industry_slots(state: &GameState, ind: IndustryType) -> Vec<IndustrySlot> {
    industry_slots(ind)
        .iter()
        .copied()
        .filter(|slot| {
            if slot.loc.is_city() {
                state.tile_at(slot.loc, slot.slot_index).is_none()
            } else {
                state.farm_tile(slot.loc).is_none()
            }
        })
        .collect()
}

pub fn empty_industry_slot_count(state: &GameState, ind: IndustryType) -> usize {
    empty_industry_slots(state, ind).len()
}

/// Does a merchant that accepts `ind` sit in `pid`'s network reach from `loc`?
pub fn merchant_reachable(state: &GameState, loc: Loc, ind: IndustryType) -> bool {
    let connected = connected_locations(state, loc);
    state
        .merchants
        .iter()
        .any(|mt| connected.contains(&mt.loc) && mt.accepts(ind))
}

/// Can `pid` reach a merchant barrel (any accepted industry) from `loc`?
pub fn beer_barrels_reachable(state: &GameState, loc: Loc) -> bool {
    let connected = connected_locations(state, loc);
    state
        .merchants
        .iter()
        .any(|mt| mt.has_beer && connected.contains(&mt.loc))
}

/// Is there enough beer available to fuel a sale of `need` barrels: board
/// beer sources reachable from `loc`, or merchant barrels in reach?
pub fn beer_available(state: &GameState, loc: Loc, pid: usize, need: usize) -> bool {
    count_beer_sources(state, loc, pid, &[]) >= need || beer_barrels_reachable(state, loc)
}

/// Number of beer barrels the player currently holds (unflipped breweries).
pub fn owned_beer_barrels(state: &GameState, pid: usize) -> usize {
    state
        .city_tiles
        .iter()
        .chain(state.farm_tiles.iter())
        .flatten()
        .filter(|t| t.player == pid && t.ind == IndustryType::Brewery && !t.flipped)
        .map(|t| t.resource_cubes as usize)
        .sum()
}

/// Total beer barrels needed to sell ALL of the player's unflipped sellable
/// tiles. Extra barrels beyond this (plus a rail-network buffer) are wasted.
pub fn sellable_beer_demand(state: &GameState, pid: usize) -> usize {
    state
        .city_tiles
        .iter()
        .flatten()
        .filter(|t| t.player == pid && !t.flipped && t.ind.is_sellable())
        .map(|t| t.def.beers_to_sell.unwrap_or(0) as usize)
        .sum()
}

/// Unbuilt links adjacent to `city_id` (expansion potential).
pub fn unbuilt_neighbor_connections(state: &GameState, city_id: Loc) -> usize {
    connections()
        .iter()
        .filter(|c| state.links[c.id].is_none() && (c.a == city_id || c.b == city_id))
        .count()
}

/// Does `pid` own a link touching `city_id`?
pub fn player_owns_link_touching(state: &GameState, pid: usize, city_id: Loc) -> bool {
    connections().iter().any(|c| {
        if let Some(l) = &state.links[c.id] {
            l.player == pid && (c.a == city_id || c.b == city_id)
        } else {
            false
        }
    })
}

/// Fraction of the coal/iron needed by a build that can come from free board
/// sources (own or opponent mines/works) rather than the paid market.
/// Consuming opponent resources is "free riding": it costs nothing extra and
/// flips their tile. A high ratio means the build is cheap to run.
pub fn resource_source_ratio(state: &GameState, cand: &BuildTarget) -> f64 {
    let needed = cand.cost_coal as f64 + cand.cost_iron as f64;
    if needed <= 0.0 {
        return 1.0;
    }
    let free_coal = if cand.cost_coal > 0 {
        find_coal_sources(state, cand.loc)
            .iter()
            .filter(|s| s.free)
            .count() as f64
    } else {
        f64::MAX
    };
    let free_iron = if cand.cost_iron > 0 {
        find_iron_sources(state).iter().filter(|s| s.free).count() as f64
    } else {
        f64::MAX
    };
    let mut free_available = 0.0;
    free_available += free_coal.min(cand.cost_coal as f64);
    free_available += free_iron.min(cand.cost_iron as f64);
    (free_available / needed).clamp(0.0, 1.0)
}

/// VP forfeited by replacing one of our own tiles. Overbuilding an
/// opponent's depleted resource tile does not discard any of our scoring
/// potential.
pub fn own_overbuild_vp_loss(
    state: &GameState,
    pid: usize,
    cand: &BuildTarget,
    weight: f64,
) -> f64 {
    let existing = if cand.loc.is_city() {
        state.tile_at(cand.loc, cand.slot_index)
    } else {
        state.farm_tile(cand.loc)
    };
    existing
        .filter(|tile| tile.player == pid)
        .map(|tile| tile.def.vp as f64 * weight)
        .unwrap_or(0.0)
}

/// Hand-access value of connecting `cand_cities`: location cards newly in
/// network and industry cards generally become playable. Returns
/// `(location_cards, industry_cards)` counts; the weights live in
/// `NetworkWeights`.
pub fn hand_access_gain(state: &GameState, pid: usize, cand_cities: &[Loc; 2]) -> (usize, usize) {
    let player = &state.players[pid];
    let mut location_cards = 0usize;
    let mut industry_cards = 0usize;
    for card in &player.hand {
        match card {
            Card::Location(l) => {
                if !crate::graph::is_in_network(state, pid, *l) && cand_cities.contains(l) {
                    location_cards += 1;
                }
            }
            Card::Industry { .. } => industry_cards += 1,
            _ => {}
        }
    }
    (location_cards, industry_cards)
}

/// Does at least one of the link endpoints touch a merchant location?
pub fn touches_merchant(cand_cities: &[Loc; 2]) -> bool {
    cand_cities.iter().any(|c| c.is_merchant())
}

#[cfg(test)]
mod tests {
    use rand_chacha::{ChaCha12Rng, rand_core::SeedableRng};

    use super::*;
    use crate::data::industry_tiles;
    use crate::state::BoardTile;

    #[test]
    fn owned_beer_includes_unflipped_farm_brewery() {
        let mut state = GameState::new(ChaCha12Rng::seed_from_u64(7), 2);
        let pid = state.current_player_id();
        state.farm_tiles[0] = Some(BoardTile {
            player: pid,
            ind: IndustryType::Brewery,
            def: industry_tiles(IndustryType::Brewery)[0],
            flipped: false,
            resource_cubes: 2,
        });

        assert_eq!(owned_beer_barrels(&state, pid), 2);
    }

    #[test]
    fn own_overbuild_deducts_replaced_tile_vp_only() {
        let mut state = GameState::new(ChaCha12Rng::seed_from_u64(9), 2);
        let pid = state.current_player_id();
        let loc = Loc::Birmingham;
        let cand = BuildTarget {
            loc,
            slot_index: 0,
            ind: IndustryType::CottonMill,
            cost_money: 0,
            cost_total: 0,
            cost_coal: 0,
            cost_iron: 0,
        };

        assert_eq!(own_overbuild_vp_loss(&state, pid, &cand, 1.0), 0.0);

        let def = industry_tiles(IndustryType::CottonMill)[0];
        state.place_tile(
            loc,
            0,
            BoardTile {
                player: pid,
                ind: IndustryType::CottonMill,
                def,
                flipped: false,
                resource_cubes: 0,
            },
        );
        assert_eq!(
            own_overbuild_vp_loss(&state, pid, &cand, 1.0),
            def.vp as f64
        );

        let key = state.city_slot_key(loc, 0).unwrap();
        state.city_tiles[key].as_mut().unwrap().player = (pid + 1) % state.players.len();
        assert_eq!(own_overbuild_vp_loss(&state, pid, &cand, 1.0), 0.0);
    }

    #[test]
    fn industry_slot_queries_cover_static_and_current_empty_capacity() {
        assert_eq!(industry_slots(IndustryType::IronWorks).len(), 9);
        assert_eq!(industry_slots(IndustryType::Brewery).len(), 11);
        assert!(city_supports_industry(
            Loc::Birmingham,
            IndustryType::IronWorks
        ));
        assert!(city_supports_industry(
            Loc::Uttoxeter,
            IndustryType::Brewery
        ));

        let mut state = GameState::new(ChaCha12Rng::seed_from_u64(3), 4);
        assert_eq!(
            empty_industry_slot_count(&state, IndustryType::IronWorks),
            9
        );
        assert_eq!(empty_industry_slot_count(&state, IndustryType::Brewery), 11);
        let def = industry_tiles(IndustryType::Brewery)[0];
        state.place_tile(
            Loc::BreweryNorth,
            0,
            BoardTile {
                player: 0,
                ind: IndustryType::Brewery,
                def,
                flipped: false,
                resource_cubes: def.resource_cubes,
            },
        );
        assert_eq!(empty_industry_slot_count(&state, IndustryType::Brewery), 10);
    }
}
