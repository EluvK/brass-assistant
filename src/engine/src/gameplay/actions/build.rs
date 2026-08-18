//! BUILD action legality, costs, and execution.

use super::{
    discard_card, require_card_index, source_purchase_cost, valid_build_cards,
    validate_coal_choice, validate_iron_choice,
};
use crate::data::{Era, IndustryType};
use crate::graph::{
    CoalSource, IronSource, connected_locations, find_coal_sources, find_iron_sources,
    is_resource_depleted,
};
use crate::map::*;
use crate::state::{BoardTile, GameState};
// ---------------------------------------------------------------------------
// BUILD
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct BuildTarget {
    pub loc: Loc,
    pub slot_index: usize,
    pub ind: IndustryType,
    pub cost_money: i32,
    pub cost_total: i32,
    pub cost_coal: u8,
    pub cost_iron: u8,
}

pub fn get_valid_build_targets(state: &GameState, pid: usize) -> Vec<BuildTarget> {
    state.ensure_network_masks();
    let mut targets = Vec::new();

    // Iron sources are location-independent: compute once for the whole call.
    let iron_sources = find_iron_sources(state);

    for loc in ALL_LOCATIONS.iter().take(CITY_COUNT) {
        // Coal sources depend only on `loc`: compute once per location.
        let coal_sources = find_coal_sources(state, *loc);
        let slots = city_slots(*loc);
        for (slot_idx, allowed) in slots.iter().enumerate() {
            let key = state.city_slot_key(*loc, slot_idx).unwrap();
            let existing = state.city_tiles[key].as_ref();

            for &ind in allowed.iter() {
                if let Some(t) = check_build_target(
                    state,
                    pid,
                    *loc,
                    slot_idx,
                    ind,
                    existing,
                    &coal_sources,
                    &iron_sources,
                ) {
                    targets.push(t);
                }
            }
        }
    }

    // Farm breweries: standalone brewery spaces, industry/wild cards only.
    for farm in [Loc::BreweryNorth, Loc::BrewerySouth] {
        let coal_sources = find_coal_sources(state, farm);
        let existing = state.farm_tile(farm);
        if let Some(t) = check_build_target(
            state,
            pid,
            farm,
            0,
            IndustryType::Brewery,
            existing,
            &coal_sources,
            &iron_sources,
        ) {
            targets.push(t);
        }
    }

    targets
}

#[allow(clippy::too_many_arguments)]
fn check_build_target(
    state: &GameState,
    pid: usize,
    loc: Loc,
    slot_index: usize,
    ind: IndustryType,
    existing: Option<&BoardTile>,
    coal_sources: &[CoalSource],
    iron_sources: &[IronSource],
) -> Option<BuildTarget> {
    let player = &state.players[pid];
    let next_tile = player.next_tile(ind)?;

    // Era restrictions
    if state.era == Era::Canal && !next_tile.canal_era {
        return None;
    }
    if state.era == Era::Rail && !next_tile.rail_era {
        return None;
    }

    // Canal era: only one tile per location per player (exclude current slot).
    if state.era == Era::Canal && loc.is_city() {
        let slots = city_slots(loc);
        for (i, _) in slots.iter().enumerate() {
            if i == slot_index {
                continue;
            }
            let k = state.city_slot_key(loc, i).unwrap();
            if let Some(t) = &state.city_tiles[k] {
                if t.player == pid {
                    return None;
                }
            }
        }
    }

    // Overbuilding: requires a higher-level tile of the SAME industry.
    if let Some(ex) = existing {
        if ex.ind != ind {
            return None;
        }
        if next_tile.level <= ex.def.level {
            return None;
        }
        if ex.player != pid {
            // Opponent tiles: only Coal/Iron and only when depleted everywhere.
            if ex.ind != IndustryType::CoalMine && ex.ind != IndustryType::IronWorks {
                return None;
            }
            if !is_resource_depleted(state, ex.ind) {
                return None;
            }
        }
    }

    // Cost (uses caller-provided sources to avoid redundant BFS/full scans)
    let cost = calculate_build_cost_with_sources(state, pid, ind, coal_sources, iron_sources)?;
    if cost.cost_total > player.money {
        return None;
    }

    // Card availability
    let has_card = valid_build_cards(state, player, pid, loc, ind).len() > 0;
    if !has_card {
        return None;
    }

    Some(BuildTarget {
        loc,
        slot_index,
        ind,
        cost_money: cost.money,
        cost_total: cost.cost_total,
        cost_coal: cost.coal,
        cost_iron: cost.iron,
    })
}

#[derive(Debug, Clone)]
pub struct BuildCost {
    pub money: i32,
    pub coal: u8,
    pub coal_cost: i32,
    pub iron: u8,
    pub iron_cost: i32,
    pub cost_total: i32,
}

pub fn calculate_build_cost(
    state: &GameState,
    pid: usize,
    ind: IndustryType,
    loc: Loc,
) -> Option<BuildCost> {
    let iron_sources = find_iron_sources(state);
    let coal_sources = find_coal_sources(state, loc);
    calculate_build_cost_with_sources(state, pid, ind, &coal_sources, &iron_sources)
}

/// Cost computation with caller-provided resource sources (avoids recomputing
/// the same BFS/full-table scans for every candidate build target).
fn calculate_build_cost_with_sources(
    state: &GameState,
    pid: usize,
    ind: IndustryType,
    coal_sources: &[CoalSource],
    iron_sources: &[IronSource],
) -> Option<BuildCost> {
    let player = &state.players[pid];
    let tile = player.next_tile(ind)?;

    let money = tile.cost;
    let coal_needed = tile.cost_coal;
    let iron_needed = tile.cost_iron;
    let coal_cost = source_purchase_cost(coal_sources, coal_needed as usize)?;
    let iron_cost = source_purchase_cost(iron_sources, iron_needed as usize)?;

    let total = money + coal_cost + iron_cost;
    if total > player.money {
        return None;
    }

    Some(BuildCost {
        money,
        coal: coal_needed,
        coal_cost,
        iron: iron_needed,
        iron_cost,
        cost_total: total,
    })
}

pub fn execute_build(
    state: &mut GameState,
    pid: usize,
    loc: Loc,
    slot_index: usize,
    ind: IndustryType,
    coal: &[CoalSource],
    iron: &[IronSource],
    card_index: usize,
) -> Result<String, String> {
    require_card_index(state, pid, card_index)?;

    let existing = if loc.is_city() {
        let allowed = city_slots(loc).get(slot_index).ok_or("Invalid city slot")?;
        if !allowed.contains(&ind) {
            return Err("Industry is not allowed in this city slot".into());
        }
        let key = state
            .city_slot_key(loc, slot_index)
            .ok_or("Invalid city slot")?;
        state.city_tiles[key].as_ref()
    } else {
        if !matches!(loc, Loc::BreweryNorth | Loc::BrewerySouth)
            || slot_index != 0
            || ind != IndustryType::Brewery
        {
            return Err("Invalid brewery farm slot".into());
        }
        state.farm_tile(loc)
    };

    // Determine how much coal/iron this tile needs and validate the chosen set.
    let tile_def = state.players[pid]
        .next_tile(ind)
        .ok_or("No tile available")?;

    // Era restrictions (defense in depth: `get_valid_build_targets` already
    // enforces these, but raw Move::Build from Python / move_codec must also be
    // rejected). E.g. Brewery IV and Pottery V are canal_era=false.
    if state.era == Era::Canal && !tile_def.canal_era {
        return Err(format!(
            "{} Lv{} cannot be built in the canal era",
            ind.name(),
            tile_def.level
        ));
    }
    if state.era == Era::Rail && !tile_def.rail_era {
        return Err(format!(
            "{} Lv{} cannot be built in the rail era",
            ind.name(),
            tile_def.level
        ));
    }

    // A raw move must obey the same occupancy and overbuild rules as generated
    // moves. `place_tile` intentionally overwrites, so this check must happen
    // before consuming money, resources, or a tile from the player mat.
    if let Some(ex) = existing {
        if ex.ind != ind || tile_def.level <= ex.def.level {
            return Err("Target slot cannot be overbuilt by this tile".into());
        }
        if ex.player != pid
            && (ex.ind != IndustryType::CoalMine && ex.ind != IndustryType::IronWorks
                || !is_resource_depleted(state, ex.ind))
        {
            return Err("Cannot overbuild this opponent tile".into());
        }
    }

    // Canal era: only one tile per location per player (defense in depth;
    // `get_valid_build_targets` already enforces it).
    if state.era == Era::Canal && loc.is_city() {
        let slots = city_slots(loc);
        for (i, _) in slots.iter().enumerate() {
            if i == slot_index {
                continue;
            }
            let Some(k) = state.city_slot_key(loc, i) else {
                continue;
            };
            if let Some(t) = &state.city_tiles[k] {
                if t.player == pid {
                    return Err("Canal era: only one build per city per player".into());
                }
            }
        }
    }

    // Card validity (defense in depth): the discarded card must actually allow
    // this build. In particular an industry/wild-industry card requires the
    // location to be in the player's network (unless they have no presence),
    // while a location card targets its city regardless of network.
    let valid_cards = valid_build_cards(state, &state.players[pid], pid, loc, ind);
    if !valid_cards.contains(&card_index) {
        return Err("Discarded card does not allow this build".into());
    }

    let coal_needed = tile_def.cost_coal as usize;
    let iron_needed = tile_def.cost_iron as usize;
    validate_coal_choice(state, loc, coal_needed, coal)?;
    validate_iron_choice(state, iron_needed, iron)?;

    // Cost = tile cost + market prices of the chosen market sources.
    let total = tile_def.cost
        + source_purchase_cost(coal, coal_needed)
            .expect("validated coal choice must contain every required source")
        + source_purchase_cost(iron, iron_needed)
            .expect("validated iron choice must contain every required source");
    if total > state.players[pid].money {
        return Err("Cannot afford this build".into());
    }

    let tile_def = {
        let p = &mut state.players[pid];
        p.consume_tile(ind).ok_or("No tile available")?
    };

    state.spend_money(pid, total);

    // Consume exactly the chosen coal sources.
    for s in coal {
        state.consume_coal_source(s);
    }

    // Consume exactly the chosen iron sources.
    for s in iron {
        state.consume_iron_source(s);
    }

    // Breweries produce 1 beer (canal) / 2 (rail); mines/works use tile cubes.
    let cubes = tile_def.resource_cubes;
    let cubes = if ind == IndustryType::Brewery {
        if state.era == Era::Rail { 2 } else { 1 }
    } else {
        cubes
    };

    let loc_for_market_sale = loc;
    state.place_tile(
        loc,
        slot_index,
        BoardTile {
            player: pid,
            ind,
            def: tile_def,
            flipped: false,
            resource_cubes: cubes,
        },
    );

    let mut money_gained = 0;
    let key = if loc.is_city() {
        state.city_slot_key(loc, slot_index)
    } else {
        None
    };
    if ind == IndustryType::IronWorks {
        if let Some(k) = key {
            money_gained = state.auto_sell_to_market(k);
        }
    } else if ind == IndustryType::CoalMine {
        let connected = connected_locations(state, loc_for_market_sale);
        let merchant_connected = connected.iter().any(|l| l.is_merchant());
        if merchant_connected {
            if let Some(k) = key {
                money_gained = state.auto_sell_to_market(k);
            }
        }
    }
    if money_gained > 0 {
        state.gain_money(pid, money_gained);
    }

    discard_card(state, pid, card_index);

    Ok(format!(
        "Built {} Lv{} in {}",
        ind.name(),
        tile_def.level,
        loc.name()
    ))
}
