//! Legal action generation & execution.
//!
//! Translated from gameLogic.js (getValidBuildTargets / checkBuildTarget /
//! getValidNetworkTargets / executeNetworkDouble / getValidSellTargets /
//! executeSell / executeDevelop / executeLoan / canScout / executeScout).

use crate::data::{Action, Era, IndustryType};
use crate::graph::{
    cheapest_coal_for_connection, find_coal_sources, find_iron_sources, CoalSource, IronSource,
};
use crate::graph::{
    connected_locations, find_beer_sources, is_in_network, is_resource_depleted,
    player_has_presence, BeerSource, CoalSourceKind,
};
use crate::map::*;
use crate::state::{city_slot_offsets, BoardTile, Card, GameState, Player};
use rand::Rng;

// ---------------------------------------------------------------------------
// Move representation
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub enum Move {
    Build {
        loc: Loc,
        slot_index: usize,
        ind: IndustryType,
        card_index: usize,
    },
    Network {
        conn_id: usize,
        card_index: usize,
    },
    NetworkDouble {
        conn1: usize,
        conn2: usize,
        card_index: usize,
    },
    Develop {
        ind1: IndustryType,
        ind2: Option<IndustryType>,
        card_index: usize,
    },
    Sell {
        keys: Vec<usize>,
        use_merchant_beer: Vec<bool>,
        card_index: usize,
    },
    Loan {
        card_index: usize,
    },
    Scout {
        card_indices: [usize; 3],
    },
    Pass {
        card_index: usize,
    },
}

impl Move {
    pub fn action(&self) -> Action {
        match self {
            Move::Build { .. } => Action::Build,
            Move::Network { .. } => Action::Network,
            Move::NetworkDouble { .. } => Action::Network,
            Move::Develop { .. } => Action::Develop,
            Move::Sell { .. } => Action::Sell,
            Move::Loan { .. } => Action::Loan,
            Move::Scout { .. } => Action::Scout,
            Move::Pass { .. } => Action::Pass,
        }
    }

    pub fn describe(&self, _state: &GameState<impl Rng>) -> String {
        match self {
            Move::Build { loc, ind, .. } => format!("Build {} at {}", ind.name(), loc.name()),
            Move::Network { conn_id, .. } => {
                let c = &connections()[*conn_id];
                format!("Network {} - {}", c.a.name(), c.b.name())
            }
            Move::NetworkDouble { conn1, conn2, .. } => {
                let c1 = &connections()[*conn1];
                let c2 = &connections()[*conn2];
                format!(
                    "Network x2: {} - {} and {} - {}",
                    c1.a.name(),
                    c1.b.name(),
                    c2.a.name(),
                    c2.b.name()
                )
            }
            Move::Develop { ind1, ind2, .. } => {
                if let Some(i2) = ind2 {
                    format!("Develop {} + {}", ind1.name(), i2.name())
                } else {
                    format!("Develop {}", ind1.name())
                }
            }
            Move::Sell { keys, .. } => {
                format!("Sell {} tile(s)", keys.len())
            }
            Move::Loan { .. } => "Take loan".to_string(),
            Move::Scout { .. } => "Scout (wild cards)".to_string(),
            Move::Pass { .. } => "Pass".to_string(),
        }
    }
}

// ---------------------------------------------------------------------------
// Card helpers
// ---------------------------------------------------------------------------

pub fn discard_card(state: &mut GameState<impl Rng>, player_id: usize, card_index: usize) {
    let p = &mut state.players[player_id];
    if card_index >= p.hand.len() {
        return;
    }
    let removed = p.hand[card_index].clone();
    match removed.ctype() {
        crate::data::CardType::WildLocation => {
            state.wild_location_pile += 1;
            p.has_wild_location = false;
        }
        crate::data::CardType::WildIndustry => {
            state.wild_industry_pile += 1;
            p.has_wild_industry = false;
        }
        _ => {
            // Non-wild discarded cards join the face-down discard pile.
            state.discard_pile.push(removed);
        }
    }
    p.hand.remove(card_index);
}

/// Indices of cards in `player`'s hand valid for a BUILD at `loc`/`ind`.
/// `loc` being a farm restricts to industry/wild-industry cards.
pub fn valid_build_cards(
    state: &GameState<impl Rng>,
    player: &Player,
    pid: usize,
    loc: Loc,
    ind: IndustryType,
) -> Vec<usize> {
    let is_farm = loc.is_farm();
    let in_network = is_in_network(state, pid, loc);
    let industry_ok = in_network || !player_has_presence(state, pid);
    let mut out = Vec::new();
    for (i, card) in player.hand.iter().enumerate() {
        match card {
            Card::Location(cl) => {
                if !is_farm && *cl == loc {
                    out.push(i);
                }
            }
            Card::Industry { .. } => {
                if card.is_industry(ind) && industry_ok {
                    out.push(i);
                }
            }
            Card::WildLocation => {
                if !is_farm {
                    out.push(i);
                }
            }
            Card::WildIndustry => {
                if industry_ok {
                    out.push(i);
                }
            }
        }
    }
    out
}

/// Any card may be discarded for network/develop/sell/loan/pass.
pub fn any_card_indices(player: &Player) -> Vec<usize> {
    (0..player.hand.len()).collect()
}

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

/// Is there a vacant single-icon slot anywhere for this industry type?
fn vacant_single_icon_exists(state: &GameState<impl Rng>, ind: IndustryType) -> bool {
    for loc in ALL_LOCATIONS.iter().take(CITY_COUNT) {
        let slots = city_slots(*loc);
        for (slot_idx, allowed) in slots.iter().enumerate() {
            let key = match state.city_slot_key(*loc, slot_idx) {
                Some(k) => k,
                None => continue,
            };
            if state.city_tiles[key].is_none() && allowed.len() == 1 && allowed[0] == ind {
                return true;
            }
        }
    }
    false
}

pub fn get_valid_build_targets(state: &GameState<impl Rng>, pid: usize) -> Vec<BuildTarget> {
    // Precompute once per call: which industries still have a vacant
    // single-icon slot somewhere (used to deprioritize multi-icon slots).
    let mut single_icon_available = [false; 6];
    for (i, ind) in IndustryType::ALL.iter().enumerate() {
        single_icon_available[i] = vacant_single_icon_exists(state, *ind);
    }

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
                // Slot preference: skip multi-icon vacant slots when a vacant
                // single-icon slot exists for this industry.
                if existing.is_none() && allowed.len() > 1 && single_icon_available[ind as usize]
                {
                    continue;
                }
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
    state: &GameState<impl Rng>,
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
    state: &GameState<impl Rng>,
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
    state: &GameState<impl Rng>,
    pid: usize,
    ind: IndustryType,
    coal_sources: &[CoalSource],
    iron_sources: &[IronSource],
) -> Option<BuildCost> {
    let player = &state.players[pid];
    let tile = player.next_tile(ind)?;

    let money = tile.cost;
    let mut coal_cost = 0i32;
    let mut iron_cost = 0i32;

    let coal_needed = tile.cost_coal;
    if coal_needed > 0 {
        let mut remaining = coal_needed;
        for src in coal_sources {
            if remaining == 0 {
                break;
            }
            if src.free {
                remaining -= 1;
            } else {
                coal_cost += src.price as i32;
                remaining -= 1;
            }
        }
        if remaining > 0 {
            return None; // not enough coal
        }
    }

    let iron_needed = tile.cost_iron;
    if iron_needed > 0 {
        let mut remaining = iron_needed;
        for src in iron_sources {
            if remaining == 0 {
                break;
            }
            if src.free {
                remaining -= 1;
            } else {
                iron_cost += src.price as i32;
                remaining -= 1;
            }
        }
        if remaining > 0 {
            return None; // not enough iron
        }
    }

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
    state: &mut GameState<impl Rng>,
    pid: usize,
    loc: Loc,
    slot_index: usize,
    ind: IndustryType,
    card_index: usize,
) -> Result<String, String> {
    let cost = calculate_build_cost(state, pid, ind, loc).ok_or("Cannot afford this build")?;

    let tile_def = {
        let p = &mut state.players[pid];
        p.consume_tile(ind).ok_or("No tile available")?
    };

    state.spend_money(pid, cost.cost_total);

    // Consume coal
    if cost.coal > 0 {
        let sources = find_coal_sources(state, loc);
        let mut remaining = cost.coal;
        for src in sources {
            if remaining == 0 {
                break;
            }
            match src.kind {
                CoalSourceKind::Mine => {
                    state.consume_from_city(src.key);
                    remaining -= 1;
                }
                CoalSourceKind::Market => {
                    state.take_market_coal();
                    remaining -= 1;
                }
            }
        }
    }

    // Consume iron
    if cost.iron > 0 {
        let sources = find_iron_sources(state);
        let mut remaining = cost.iron;
        for src in sources {
            if remaining == 0 {
                break;
            }
            if src.free {
                state.consume_from_city(src.key);
                remaining -= 1;
            } else {
                state.take_market_iron();
                remaining -= 1;
            }
        }
    }

    // Breweries produce 1 beer (canal) / 2 (rail); mines/works use tile cubes.
    let cubes = tile_def.resource_cubes;
    let cubes = if ind == IndustryType::Brewery {
        if state.era == Era::Rail { 2 } else { 1 }
    } else {
        cubes
    };

    let loc_for_market_sale = loc;
    state.place_tile(loc, slot_index, BoardTile {
        player: pid,
        ind,
        def: tile_def,
        flipped: false,
        resource_cubes: cubes,
    });

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

// ---------------------------------------------------------------------------
// NETWORK
// ---------------------------------------------------------------------------

pub fn get_valid_network_targets(state: &GameState<impl Rng>, pid: usize) -> Vec<usize> {
    let player = &state.players[pid];
    let mut out = Vec::new();
    let has_no_presence = !player_has_presence(state, pid);

    for conn in connections() {
        if state.links[conn.id].is_some() {
            continue;
        }
        match state.era {
            Era::Canal => {
                if !conn.canal {
                    continue;
                }
                if player.canal_links == 0 {
                    continue;
                }
                if !has_no_presence {
                    let e1 = is_in_network(state, pid, conn.a);
                    let e2 = is_in_network(state, pid, conn.b);
                    if !e1 && !e2 {
                        continue;
                    }
                }
                if CANAL_LINK_COST > player.money {
                    continue;
                }
            }
            Era::Rail => {
                if !conn.rail {
                    continue;
                }
                if player.rail_links == 0 {
                    continue;
                }
                if !has_no_presence {
                    let e1 = is_in_network(state, pid, conn.a);
                    let e2 = is_in_network(state, pid, conn.b);
                    if !e1 && !e2 {
                        continue;
                    }
                }
                let coal = cheapest_coal_for_connection(state, conn);
                if coal.is_none() {
                    continue;
                }
                let cost = RAIL_LINK_COST + if coal.unwrap().free { 0 } else { coal.unwrap().price as i32 };
                if cost > player.money {
                    continue;
                }
            }
        }
        out.push(conn.id);
    }
    out
}

pub fn execute_network(
    state: &mut GameState<impl Rng>,
    pid: usize,
    conn_id: usize,
    card_index: usize,
) -> Result<String, String> {
    let conn = connections()
        .iter()
        .find(|c| c.id == conn_id)
        .ok_or("Invalid connection")?;
    if state.links[conn_id].is_some() {
        return Err("Connection already built".into());
    }

    match state.era {
        Era::Canal => {
            state.spend_money(pid, CANAL_LINK_COST);
            let p = &mut state.players[pid];
            p.canal_links -= 1;
        }
        Era::Rail => {
            let coal = cheapest_coal_for_connection(state, conn).ok_or("No coal available")?;
            let mut total = RAIL_LINK_COST;
            match coal.kind {
                CoalSourceKind::Mine => {
                    state.consume_from_city(coal.key);
                }
                CoalSourceKind::Market => {
                    total += coal.price as i32;
                    state.take_market_coal();
                }
            }
            state.spend_money(pid, total);
            let p = &mut state.players[pid];
            p.rail_links -= 1;
        }
    }

    state.links[conn_id] = Some(crate::state::Link {
        player: pid,
        is_canal: state.era == Era::Canal,
    });

    discard_card(state, pid, card_index);
    Ok(format!("Built link {} - {}", conn.a.name(), conn.b.name()))
}

/// Candidates for the second rail link after `first_conn` has been placed.
pub fn get_valid_second_rail_links(
    state: &mut GameState<impl Rng>,
    pid: usize,
    first_conn: usize,
) -> Vec<usize> {
    if state.era != Era::Rail {
        return Vec::new();
    }
    let rail_remaining = state.players[pid].rail_links;
    if rail_remaining < 2 {
        return Vec::new();
    }
    let first = match connections().iter().find(|c| c.id == first_conn) {
        Some(c) => c,
        None => return Vec::new(),
    };
    let has_no_presence = !player_has_presence(state, pid);
    if !has_no_presence && !is_in_network(state, pid, first.a) && !is_in_network(state, pid, first.b) {
        return Vec::new();
    }

    let coal1 = match cheapest_coal_for_connection(state, first) {
        Some(c) => c,
        None => return Vec::new(),
    };
    let mut budget_left = state.players[pid].money - RAIL_DOUBLE_LINK_COST;
    if !coal1.free {
        budget_left -= coal1.price as i32;
    }
    if budget_left < 0 {
        return Vec::new();
    }

    // Place the first link and dry-run its coal consumption so second-link
    // generation matches the real execution order.
    state.links[first_conn] = Some(crate::state::Link { player: pid, is_canal: false });
    let mut consumed_coal1 = None;
    let mut flipped_coal1 = None;
    let mut income_space_before = None;
    let mut coal1_owner = None;
    if coal1.kind == CoalSourceKind::Mine {
        if let Some(tile) = state.city_tiles[coal1.key].as_ref() {
            consumed_coal1 = Some(tile.resource_cubes);
            flipped_coal1 = Some(tile.flipped);
            coal1_owner = Some(tile.player);
            income_space_before = Some(state.players[tile.player].income_space);
        }
        state.consume_from_city(coal1.key);
    }

    let mut out = Vec::new();
    for conn in connections() {
        if conn.id == first_conn || !conn.rail || state.links[conn.id].is_some() {
            continue;
        }

        if !is_in_network(state, pid, conn.a) && !is_in_network(state, pid, conn.b) {
            continue;
        }
        let coal2 = match cheapest_coal_for_connection(state, conn) {
            Some(c) => c,
            None => continue,
        };
        let total_cost = if coal2.free { 0 } else { coal2.price as i32 };
        if total_cost > budget_left {
            continue;
        }
        if find_beer_for_link(state, pid, conn).is_none() {
            continue;
        }
        out.push(conn.id);
    }

    if let Some(prev_cubes) = consumed_coal1 {
        if let Some(tile) = state.city_tiles[coal1.key].as_mut() {
            tile.resource_cubes = prev_cubes;
            if let Some(prev_flipped) = flipped_coal1 {
                tile.flipped = prev_flipped;
            }
        }
    }
    if let (Some(owner), Some(income_space)) = (coal1_owner, income_space_before) {
        state.players[owner].income_space = income_space;
    }
    state.links[first_conn] = None;
    out
}

/// Own breweries anywhere; opponent breweries connected to the second link.
fn find_beer_for_link(
    state: &GameState<impl Rng>,
    pid: usize,
    conn: &Connection,
) -> Option<BeerSource> {
    let sources = find_beer_sources(state, conn.a, pid, &[]);
    // find_beer_sources already includes own breweries anywhere + connected opponents
    if sources.is_empty() {
        // Also check the other endpoint for opponent breweries
        return find_beer_sources(state, conn.b, pid, &[])
            .into_iter()
            .next();
    }
    sources.into_iter().next()
}

pub fn execute_network_double(
    state: &mut GameState<impl Rng>,
    pid: usize,
    conn1: usize,
    conn2: usize,
    card_index: usize,
) -> Result<String, String> {
    if state.era != Era::Rail {
        return Err("Double links are Rail Era only".into());
    }
    if state.players[pid].rail_links < 2 {
        return Err("Not enough rail links".into());
    }
    if state.links[conn1].is_some() || state.links[conn2].is_some() {
        return Err("Connection already built".into());
    }
    let c1 = connections().iter().find(|c| c.id == conn1).unwrap();
    let c2 = connections().iter().find(|c| c.id == conn2).unwrap();

    let has_no_presence = !player_has_presence(state, pid);

    // Link 1 adjacency
    if !has_no_presence
        && !is_in_network(state, pid, c1.a)
        && !is_in_network(state, pid, c1.b)
    {
        return Err("First link is not adjacent to your network".into());
    }
    state.links[conn1] = Some(crate::state::Link { player: pid, is_canal: false });

    let coal1 = cheapest_coal_for_connection(state, c1).ok_or("No coal for first link")?;
    let mut total = RAIL_DOUBLE_LINK_COST + if coal1.free { 0 } else { coal1.price as i32 };
    // dry-run coal1 consumption
    if coal1.kind == CoalSourceKind::Mine {
        state.consume_from_city(coal1.key);
    }

    // Link 2 adjacency (with link1 on board)
    if !is_in_network(state, pid, c2.a) && !is_in_network(state, pid, c2.b) {
        // rollback coal1
        if coal1.kind == CoalSourceKind::Mine {
            if let Some(t) = state.city_tiles[coal1.key].as_mut() {
                if t.flipped {
                    // undo flip: extremely unlikely but restore
                }
            }
        }
        return Err("Second link is not adjacent to your network".into());
    }
    state.links[conn2] = Some(crate::state::Link { player: pid, is_canal: false });

    let coal2 = match cheapest_coal_for_connection(state, c2) {
        Some(c) => c,
        None => {
            state.links[conn1] = None;
            state.links[conn2] = None;
            return Err("No coal for second link".into());
        }
    };
    total += if coal2.free { 0 } else { coal2.price as i32 };
    if coal2.kind == CoalSourceKind::Mine {
        state.consume_from_city(coal2.key);
    }

    let beer = match find_beer_for_link(state, pid, c2) {
        Some(b) => b,
        None => {
            state.links[conn1] = None;
            state.links[conn2] = None;
            return Err("No beer available for the second link".into());
        }
    };

    if total > state.players[pid].money {
        state.links[conn1] = None;
        state.links[conn2] = None;
        return Err("Cannot afford both links".into());
    }

    // Commit
    state.consume_beer_source(&beer);
    state.spend_money(pid, total);
    state.players[pid].rail_links -= 2;
    discard_card(state, pid, card_index);

    Ok(format!(
        "Built 2 rail links: {} - {} and {} - {}",
        c1.a.name(),
        c1.b.name(),
        c2.a.name(),
        c2.b.name()
    ))
}

// ---------------------------------------------------------------------------
// DEVELOP
// ---------------------------------------------------------------------------

pub fn can_develop(state: &GameState<impl Rng>, pid: usize) -> bool {
    let player = &state.players[pid];
    let iron = find_iron_sources(state);
    if iron.is_empty() {
        return false;
    }
    let first = iron[0];
    if !first.free && first.price as i32 > player.money {
        return false;
    }
    !player.developable_types().is_empty()
}

pub fn execute_develop(
    state: &mut GameState<impl Rng>,
    pid: usize,
    ind1: IndustryType,
    ind2: Option<IndustryType>,
    card_index: usize,
) -> Result<String, String> {
    let tiles_to_develop = if ind2.is_some() { 2 } else { 1 };
    let iron = find_iron_sources(state);
    if iron.len() < tiles_to_develop {
        return Err("Not enough iron available".into());
    }

    // Pre-check affordability of market iron
    let mut money_needed = 0;
    for i in 0..tiles_to_develop {
        if !iron[i].free {
            money_needed += iron[i].price as i32;
        }
    }
    if money_needed > state.players[pid].money {
        return Err("Cannot afford market iron".into());
    }

    // Consume iron
    for i in 0..tiles_to_develop {
        let src = iron[i];
        if src.free {
            state.consume_from_city(src.key);
        } else {
            let price = state.iron_price();
            state.spend_money(pid, price as i32);
            state.take_market_iron();
        }
    }

    // Remove tiles from mat
    let _t1 = state.players[pid].consume_tile(ind1);
    if let Some(i2) = ind2 {
        let _t2 = state.players[pid].consume_tile(i2);
    }

    discard_card(state, pid, card_index);
    Ok(format!(
        "Developed {} {}",
        ind1.name(),
        ind2.map(|i| format!("+ {}", i.name())).unwrap_or_default()
    ))
}

// ---------------------------------------------------------------------------
// SELL
// ---------------------------------------------------------------------------

pub struct SellTarget {
    pub key: usize,
    pub merchant_indices: Vec<usize>,
    pub beer_needed: u8,
    pub merchant_beer_available: bool,
}

pub fn get_valid_sell_targets(state: &GameState<impl Rng>, pid: usize) -> Vec<SellTarget> {
    let mut out = Vec::new();
    for (k, tile) in state.city_tiles.iter().enumerate() {
        let Some(t) = tile else { continue };
        if t.player != pid || t.flipped || !t.ind.is_sellable() {
            continue;
        }
        // Loc from key
        let loc = loc_from_key(state, k);
        if loc.is_none() {
            continue;
        }
        let loc = loc.unwrap();
        let merchant_indices = sell_merchants_for(state, pid, loc, t.ind);
        if merchant_indices.is_empty() {
            continue;
        }
        let beer_needed = t.def.beers_to_sell.unwrap_or(0);
        let merchant_beer_available = if beer_needed > 0 {
            let total_beer = find_beer_sources(state, loc, pid, &merchant_indices);
            if total_beer.len() < beer_needed as usize {
                continue;
            }

            let brewery_beer = find_beer_sources(state, loc, pid, &[]);
            merchant_indices.iter().any(|&i| {
                state.merchants[i].has_beer
                    && brewery_beer.len() + 1 >= beer_needed as usize
            })
        } else {
            merchant_indices.iter().any(|&i| state.merchants[i].has_beer)
        };
        out.push(SellTarget {
            key: k,
            merchant_indices,
            beer_needed,
            merchant_beer_available,
        });
    }
    out
}

fn sell_merchants_for(
    state: &GameState<impl Rng>,
    _pid: usize,
    loc: Loc,
    ind: IndustryType,
) -> Vec<usize> {
    let connected = connected_locations(state, loc);
    let mut out = Vec::new();
    for (i, mt) in state.merchants.iter().enumerate() {
        if !mt.accepts(ind) {
            continue;
        }
        if connected.contains(&mt.loc) {
            out.push(i);
        }
    }
    out
}

fn loc_from_key(_state: &GameState<impl Rng>, key: usize) -> Option<Loc> {
    let offsets = city_slot_offsets();
    for (i, loc) in ALL_LOCATIONS[..CITY_COUNT].iter().enumerate() {
        let off = offsets[i];
        let n = city_slots(*loc).len();
        if key >= off && key < off + n {
            return Some(*loc);
        }
    }
    None
}

pub fn execute_sell(
    state: &mut GameState<impl Rng>,
    pid: usize,
    keys: &[usize],
    use_merchant_beer: &[bool],
    card_index: usize,
) -> Result<String, String> {
    let mut sold = 0usize;
    let mut notes = Vec::new();

    for (entry_idx, &key) in keys.iter().enumerate() {
        // Read tile info without holding a long-lived mutable borrow.
        let (t_player, t_flipped, ind, beer_needed, loc) = {
            let Some(tile) = state.city_tiles[key].as_ref() else {
                continue;
            };
            let loc = match loc_from_key(state, key) {
                Some(l) => l,
                None => continue,
            };
            (tile.player, tile.flipped, tile.ind, tile.def.beers_to_sell.unwrap_or(0), loc)
        };
        if t_player != pid || t_flipped {
            continue;
        }

        let merchant_indices = sell_merchants_for(state, pid, loc, ind);
        if merchant_indices.is_empty() {
            continue;
        }

        let mut beer_remaining = beer_needed;
        let use_merchant = use_merchant_beer
            .get(entry_idx)
            .copied()
            .unwrap_or(false);

        // Merchant beer first, if requested: pick best-bonus merchant with a barrel
        if beer_remaining > 0 && use_merchant {
            let with_beer: Vec<usize> = merchant_indices
                .iter()
                .copied()
                .filter(|&i| state.merchants[i].has_beer)
                .collect();
            if let Some(&mi) = with_beer.first() {
                let mt_loc = state.merchants[mi].loc;
                {
                    let mt = &mut state.merchants[mi];
                    mt.has_beer = false;
                }
                beer_remaining -= 1;
                // Grant merchant bonus
                let bonus = merchant_bonus_for(state, mt_loc);
                let note = apply_merchant_bonus(state, pid, bonus);
                notes.push(note);
            }
        }

        // Remaining beer from breweries (own anywhere, opponents connected)
        if beer_remaining > 0 {
            let beer = find_beer_sources(state, loc, pid, &[]);
            if beer.len() < beer_remaining as usize {
                continue; // can't pay beer; skip tile
            }
            for b in beer.into_iter().take(beer_remaining as usize) {
                state.consume_beer_source(&b);
            }
        }

        // Flip the tile (advances income)
        let (player, income) = {
            let Some(t) = state.city_tiles[key].as_mut() else {
                continue;
            };
            if t.flipped {
                continue;
            }
            t.flipped = true;
            (t.player, t.def.income)
        };
        state.advance_income_spaces(player, income);
        sold += 1;
    }

    if sold == 0 {
        return Err("Nothing could be sold".into());
    }

    discard_card(state, pid, card_index);
    Ok(format!("Sold {} tile(s){}", sold, if notes.is_empty() { String::new() } else { format!(" [{}]", notes.join("; ")) }))
}

fn merchant_bonus_for(_state: &GameState<impl Rng>, loc: Loc) -> MerchantBonus {
    for def in merchant_defs() {
        if def.loc == loc {
            return def.bonus;
        }
    }
    MerchantBonus::Vp(0)
}

fn apply_merchant_bonus(
    state: &mut GameState<impl Rng>,
    pid: usize,
    bonus: MerchantBonus,
) -> String {
    match bonus {
        MerchantBonus::Vp(v) => {
            state.players[pid].vp += v as u16;
            format!("+{} VP (merchant)", v)
        }
        MerchantBonus::Money(m) => {
            state.gain_money(pid, m);
            format!("+£{} (merchant)", m)
        }
        MerchantBonus::Income(spaces) => {
            state.advance_income_spaces(pid, spaces);
            format!("+{} income spaces (merchant)", spaces)
        }
        MerchantBonus::Develop(_n) => {
            // Auto-resolve: develop up to n canal-only/lowest tiles for free.
            let removed = apply_free_develop(state, pid, 1);
            if removed.is_empty() {
                "free develop earned".to_string()
            } else {
                format!("free develop: {}", removed.join(", "))
            }
        }
    }
}

pub fn apply_free_develop(state: &mut GameState<impl Rng>, pid: usize, count: usize) -> Vec<String> {
    let mut removed = Vec::new();
    for _ in 0..count {
        let types = state.players[pid].developable_types();
        if types.is_empty() {
            break;
        }
        // Prefer clearing canal-only tiles, then lowest level.
        let mut sorted = types;
        sorted.sort_by_key(|(_, t)| {
            (if t.rail_era { 0 } else { 1 }, t.level)
        });
        let (ind, t) = sorted[0];
        state.players[pid].consume_tile(ind);
        removed.push(format!("{} Lv{}", ind.name(), t.level));
    }
    removed
}

// ---------------------------------------------------------------------------
// LOAN / SCOUT / PASS
// ---------------------------------------------------------------------------

pub fn execute_loan(
    state: &mut GameState<impl Rng>,
    pid: usize,
    card_index: usize,
) -> Result<String, String> {
    if !state.can_take_loan(pid) {
        return Err("A loan cannot take your income below -10".into());
    }
    state.gain_money(pid, LOAN_AMOUNT);
    state.apply_loan_income_drop(pid);
    discard_card(state, pid, card_index);
    Ok(format!("Took £{} loan (income -{} levels)", LOAN_AMOUNT, LOAN_INCOME_PENALTY))
}

pub fn can_scout(state: &GameState<impl Rng>, pid: usize) -> bool {
    let p = &state.players[pid];
    if p.hand.len() < 3 {
        return false;
    }
    if p.has_wild_location || p.has_wild_industry {
        return false;
    }
    state.wild_location_pile > 0 && state.wild_industry_pile > 0
}

pub fn execute_scout(
    state: &mut GameState<impl Rng>,
    pid: usize,
    card_indices: [usize; 3],
) -> Result<String, String> {
    let p = &mut state.players[pid];
    if p.hand.len() < 3 {
        return Err("Must have 3 cards to scout".into());
    }
    if p.has_wild_location || p.has_wild_industry {
        return Err("Already have wild cards".into());
    }
    if state.wild_location_pile == 0 || state.wild_industry_pile == 0 {
        return Err("No wild cards available".into());
    }

    let mut sorted = card_indices;
    sorted.sort_by(|a, b| b.cmp(a));
    for idx in sorted {
        if idx < p.hand.len() {
            let removed = p.hand[idx].clone();
            match removed.ctype() {
                crate::data::CardType::WildLocation | crate::data::CardType::WildIndustry => {}
                _ => state.discard_pile.push(removed),
            }
            p.hand.remove(idx);
        }
    }
    p.hand.push(Card::WildLocation);
    p.hand.push(Card::WildIndustry);
    p.has_wild_location = true;
    p.has_wild_industry = true;
    state.wild_location_pile -= 1;
    state.wild_industry_pile -= 1;
    Ok("Scouted: gained Wild Location + Wild Industry".into())
}

pub fn execute_pass(
    state: &mut GameState<impl Rng>,
    pid: usize,
    card_index: usize,
) -> Result<String, String> {
    discard_card(state, pid, card_index);
    Ok("Passed".into())
}

// ---------------------------------------------------------------------------
// Legal move enumeration
// ---------------------------------------------------------------------------

pub fn legal_moves(state: &mut GameState<impl Rng>) -> Vec<Move> {
    let pid = state.current_player_id();
    let player_hand_len = state.players[pid].hand.len();
    if player_hand_len == 0 {
        return vec![];
    }

    let mut moves = Vec::new();

    // BUILD
    for t in get_valid_build_targets(state, pid) {
        for ci in valid_build_cards(state, &state.players[pid], pid, t.loc, t.ind) {
            moves.push(Move::Build {
                loc: t.loc,
                slot_index: t.slot_index,
                ind: t.ind,
                card_index: ci,
            });
        }
    }

    // NETWORK (single + double)
    if state.era == Era::Canal && state.players[pid].canal_links > 0
        || state.era == Era::Rail && state.players[pid].rail_links > 0
    {
        for conn in get_valid_network_targets(state, pid) {
            for ci in any_card_indices(&state.players[pid]) {
                moves.push(Move::Network {
                    conn_id: conn,
                    card_index: ci,
                });
            }
        }
    }
    if state.era == Era::Rail {
        // For each valid first link, find valid second links (double rail).
        let single = get_valid_network_targets(state, pid);
        for conn1 in single {
            for conn2 in get_valid_second_rail_links(state, pid, conn1) {
                for ci in any_card_indices(&state.players[pid]) {
                    moves.push(Move::NetworkDouble {
                        conn1,
                        conn2,
                        card_index: ci,
                    });
                }
            }
        }
    }

    // DEVELOP
    if can_develop(state, pid) {
        let types: Vec<IndustryType> = state.players[pid]
            .developable_types()
            .into_iter()
            .map(|(ind, _)| ind)
            .collect();
        for &ind1 in &types {
            // Single develop
            for ci in any_card_indices(&state.players[pid]) {
                moves.push(Move::Develop {
                    ind1,
                    ind2: None,
                    card_index: ci,
                });
            }
            // Double develop (two distinct types, or same type if count>1)
            let rem1 = state.players[pid].remaining_count(ind1);
            let mut second_types: Vec<IndustryType> = types.clone();
            if rem1 < 2 {
                second_types.retain(|&x| x != ind1);
            }
            for &ind2 in &second_types {
                for ci in any_card_indices(&state.players[pid]) {
                    moves.push(Move::Develop {
                        ind1,
                        ind2: Some(ind2),
                        card_index: ci,
                    });
                }
            }
        }
    }

    // SELL
    let sell_targets = get_valid_sell_targets(state, pid);
    if !sell_targets.is_empty() {
        // Single-tile sells (one tile per action) — matches real rules.
        for t in &sell_targets {
            let use_merchant = t.merchant_beer_available;
            for ci in any_card_indices(&state.players[pid]) {
                moves.push(Move::Sell {
                    keys: vec![t.key],
                    use_merchant_beer: vec![use_merchant],
                    card_index: ci,
                });
            }
        }
    }

    // LOAN
    if state.can_take_loan(pid) {
        for ci in any_card_indices(&state.players[pid]) {
            moves.push(Move::Loan { card_index: ci });
        }
    }

    // SCOUT
    if can_scout(state, pid) {
        // All 3-card combinations
        let n = state.players[pid].hand.len();
        for i in 0..n {
            for j in (i + 1)..n {
                for k in (j + 1)..n {
                    moves.push(Move::Scout {
                        card_indices: [i, j, k],
                    });
                }
            }
        }
    }

    // PASS
    for ci in any_card_indices(&state.players[pid]) {
        moves.push(Move::Pass { card_index: ci });
    }

    moves
}

pub fn apply_move(state: &mut GameState<impl Rng>, mv: &Move) -> Result<String, String> {
    let pid = state.current_player_id();
    match mv {
        Move::Build { loc, slot_index, ind, card_index } => {
            execute_build(state, pid, *loc, *slot_index, *ind, *card_index)
        }
        Move::Network { conn_id, card_index } => execute_network(state, pid, *conn_id, *card_index),
        Move::NetworkDouble { conn1, conn2, card_index } => {
            execute_network_double(state, pid, *conn1, *conn2, *card_index)
        }
        Move::Develop { ind1, ind2, card_index } => {
            execute_develop(state, pid, *ind1, *ind2, *card_index)
        }
        Move::Sell { keys, use_merchant_beer, card_index } => {
            execute_sell(state, pid, keys, use_merchant_beer, *card_index)
        }
        Move::Loan { card_index } => execute_loan(state, pid, *card_index),
        Move::Scout { card_indices } => execute_scout(state, pid, *card_indices),
        Move::Pass { card_index } => execute_pass(state, pid, *card_index),
    }
}
