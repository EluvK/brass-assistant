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
    player_has_presence, BeerSource, BeerSourceKind, CoalSourceKind,
};
use crate::map::*;
use crate::state::{city_slot_offsets, BoardTile, Card, GameState, Player};
use rand::Rng;
use std::collections::HashMap;

fn log_loc_label(loc: Loc) -> String {
    format!("{}({})", loc.zh_name(), loc.name())
}

fn log_industry_label(ind: IndustryType) -> &'static str {
    match ind {
        IndustryType::CottonMill => "棉纺厂(Cotton Mill)",
        IndustryType::CoalMine => "煤矿(Coal Mine)",
        IndustryType::IronWorks => "铁厂(Iron Works)",
        IndustryType::Manufacturer => "制造厂(Manufacturer)",
        IndustryType::Pottery => "陶器厂(Pottery)",
        IndustryType::Brewery => "啤酒厂(Brewery)",
    }
}

// ---------------------------------------------------------------------------
// Move representation
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub enum Move {
    Build {
        loc: Loc,
        slot_index: usize,
        ind: IndustryType,
        coal: Vec<CoalSource>,
        iron: Vec<IronSource>,
        card_index: usize,
    },
    Network {
        conn_id: usize,
        coal: Option<CoalSource>,
        card_index: usize,
    },
    NetworkDouble {
        conn1: usize,
        conn2: usize,
        coal1: CoalSource,
        coal2: CoalSource,
        beer: BeerSource,
        card_index: usize,
    },
    Develop {
        ind1: IndustryType,
        ind2: Option<IndustryType>,
        iron: Vec<IronSource>,
        card_index: usize,
    },
    Sell {
        keys: Vec<usize>,
        merchant_indices: Vec<usize>,
        use_merchant_beer: Vec<bool>,
        card_index: usize,
    },
    ResolveFreeDevelop {
        ind1: IndustryType,
        ind2: Option<IndustryType>,
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
            Move::ResolveFreeDevelop { .. } => Action::Develop,
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
            Move::ResolveFreeDevelop { ind1, ind2 } => {
                if let Some(i2) = ind2 {
                    format!("Resolve free develop {} + {}", ind1.name(), i2.name())
                } else {
                    format!("Resolve free develop {}", ind1.name())
                }
            }
            Move::Loan { .. } => "Take loan".to_string(),
            Move::Scout { .. } => "Scout (wild cards)".to_string(),
            Move::Pass { .. } => "Pass".to_string(),
        }
    }
}

// ---------------------------------------------------------------------------
// Resource source selection & validation
// ---------------------------------------------------------------------------

/// All legal selections of `needed` source cubes from `available`, respecting
/// the free-first rule: free sources must be used before the market, and the
/// player may choose WHICH free sources to drain (that choice decides whose
/// building flips).
fn source_options<T: Copy + PartialEq>(
    available: &[T],
    needed: usize,
    is_free: impl Fn(&T) -> bool,
    key: impl Fn(&T) -> usize,
) -> Vec<Vec<T>> {
    if needed == 0 {
        return vec![vec![]];
    }
    let free_count = available.iter().filter(|s| is_free(s)).count();
    let market_needed = needed.saturating_sub(free_count);
    let free_needed = needed - market_needed;

    // Group free sources by key (one entry per cube).
    let mut free_groups: Vec<(usize, T, usize)> = Vec::new();
    for s in available.iter().filter(|s| is_free(s)) {
        let k = key(s);
        if let Some((_, _, c)) = free_groups.iter_mut().find(|(kk, _, _)| *kk == k) {
            *c += 1;
        } else {
            free_groups.push((k, *s, 1));
        }
    }

    // Enumerate multisets of free source keys totaling `free_needed` cubes.
    let mut free_combos: Vec<Vec<T>> = Vec::new();
    let mut cur: Vec<T> = Vec::new();
    fn rec<T: Copy + PartialEq>(
        groups: &[(usize, T, usize)],
        idx: usize,
        remaining: usize,
        cur: &mut Vec<T>,
        out: &mut Vec<Vec<T>>,
    ) {
        if remaining == 0 {
            out.push(cur.clone());
            return;
        }
        if idx >= groups.len() {
            return;
        }
        let (_, rep, count) = groups[idx];
        let take_max = remaining.min(count);
        for take in 0..=take_max {
            for _ in 0..take {
                cur.push(rep);
            }
            rec(groups, idx + 1, remaining - take, cur, out);
            for _ in 0..take {
                cur.pop();
            }
        }
    }
    rec(&free_groups, 0, free_needed, &mut cur, &mut free_combos);

    // Market sources all share one identity; fill the shortfall with them.
    let market_rep = available.iter().find(|s| !is_free(s)).copied();
    let mut out = Vec::new();
    for combo in free_combos {
        let mut sel = combo;
        if let Some(m) = market_rep {
            for _ in 0..market_needed {
                sel.push(m);
            }
        }
        out.push(sel);
    }
    out
}

/// All legal coal source selections for `needed` cubes at `loc`.
pub fn coal_source_options(
    state: &GameState<impl Rng>,
    loc: Loc,
    needed: usize,
) -> Vec<Vec<CoalSource>> {
    source_options(&find_coal_sources(state, loc), needed, |s| s.free, |s| s.key)
}

/// All legal iron source selections for `needed` cubes.
pub fn iron_source_options(state: &GameState<impl Rng>, needed: usize) -> Vec<Vec<IronSource>> {
    source_options(&find_iron_sources(state), needed, |s| s.free, |s| s.key)
}

/// Coal sources reachable from either endpoint of a connection (union, one
/// entry per cube), used for single/double rail legality.
fn coal_sources_for_connection(state: &GameState<impl Rng>, conn: &Connection) -> Vec<CoalSource> {
    let mut out = find_coal_sources(state, conn.a);
    for s in find_coal_sources(state, conn.b) {
        if !out.contains(&s) {
            out.push(s);
        }
    }
    out
}

/// All legal coal source selections for a single rail link on `conn`.
pub fn coal_options_for_connection(
    state: &GameState<impl Rng>,
    conn: &Connection,
    needed: usize,
) -> Vec<Vec<CoalSource>> {
    source_options(
        &coal_sources_for_connection(state, conn),
        needed,
        |s| s.free,
        |s| s.key,
    )
}

/// Validate a chosen source set: correct count, each currently available, and
/// the free-first rule (market may only cover the shortfall).
fn validate_source_choice<T: Copy + PartialEq>(
    available: &[T],
    needed: usize,
    chosen: &[T],
    is_free: impl Fn(&T) -> bool,
    key: impl Fn(&T) -> usize,
) -> Result<(), String> {
    if chosen.len() != needed {
        return Err(format!("Need {needed} source(s), got {}", chosen.len()));
    }
    let free_avail = available.iter().filter(|s| is_free(s)).count();
    let market_needed = needed.saturating_sub(free_avail);
    let market_chosen = chosen.iter().filter(|s| !is_free(s)).count();
    if market_chosen != market_needed {
        return Err("Free sources must be used before the market".into());
    }
    // Chosen must be a sub-multiset of currently available sources (by identity).
    let mut avail_counts: HashMap<(bool, usize), usize> = HashMap::new();
    for s in available {
        *avail_counts.entry((is_free(s), key(s))).or_insert(0) += 1;
    }
    for s in chosen {
        let e = (is_free(s), key(s));
        let c = avail_counts.get_mut(&e).ok_or("Chosen source is not available")?;
        if *c == 0 {
            return Err("Chosen source is not available".into());
        }
        *c -= 1;
    }
    Ok(())
}

fn validate_coal_choice(
    state: &GameState<impl Rng>,
    loc: Loc,
    needed: usize,
    chosen: &[CoalSource],
) -> Result<(), String> {
    validate_source_choice(
        &find_coal_sources(state, loc),
        needed,
        chosen,
        |s| s.free,
        |s| s.key,
    )
}

fn validate_connection_coal_choice(
    state: &GameState<impl Rng>,
    conn: &Connection,
    chosen: &[CoalSource],
) -> Result<(), String> {
    validate_source_choice(
        &coal_sources_for_connection(state, conn),
        1,
        chosen,
        |s| s.free,
        |s| s.key,
    )
}

fn validate_iron_choice(
    state: &GameState<impl Rng>,
    needed: usize,
    chosen: &[IronSource],
) -> Result<(), String> {
    validate_source_choice(&find_iron_sources(state), needed, chosen, |s| s.free, |s| s.key)
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
    coal: &[CoalSource],
    iron: &[IronSource],
    card_index: usize,
) -> Result<String, String> {
    // Determine how much coal/iron this tile needs and validate the chosen set.
    let tile_def = state.players[pid].next_tile(ind).ok_or("No tile available")?;
    let coal_needed = tile_def.cost_coal as usize;
    let iron_needed = tile_def.cost_iron as usize;
    validate_coal_choice(state, loc, coal_needed, coal)?;
    validate_iron_choice(state, iron_needed, iron)?;

    // Cost = tile cost + market prices of the chosen market sources.
    let mut total = tile_def.cost;
    for s in coal {
        if !s.free {
            total += s.price as i32;
        }
    }
    for s in iron {
        if !s.free {
            total += s.price as i32;
        }
    }
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
        match s.kind {
            CoalSourceKind::Mine => {
                state.consume_from_city(s.key);
            }
            CoalSourceKind::Market => {
                state.take_market_coal();
            }
        }
    }

    // Consume exactly the chosen iron sources.
    for s in iron {
        if s.free {
            state.consume_from_city(s.key);
        } else {
            state.take_market_iron();
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
    coal: Option<CoalSource>,
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
            if coal.is_some() {
                return Err("Canal links do not consume coal".into());
            }
            state.spend_money(pid, CANAL_LINK_COST);
            let p = &mut state.players[pid];
            p.canal_links -= 1;
        }
        Era::Rail => {
            let coal = coal.ok_or("A coal source is required for a rail link")?;
            validate_connection_coal_choice(state, conn, &[coal])?;
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

/// A legal second-rail link together with the beer sources the player may
/// choose to power it (own breweries anywhere + connected opponent breweries).
pub struct SecondRailOption {
    pub conn: usize,
    pub beers: Vec<BeerSource>,
    /// Coal source selections for the second link, computed with the FIRST
    /// link in place (matching execution order). Empty outer = illegal.
    pub coal2_opts: Vec<Vec<CoalSource>>,
}

/// All beer sources legal for powering a double-rail second link: own breweries
/// anywhere, plus opponent breweries connected to either endpoint. Merchant beer
/// is never usable for rail building.
pub fn beer_sources_for_link(
    state: &GameState<impl Rng>,
    pid: usize,
    conn: &Connection,
) -> Vec<BeerSource> {
    let mut out = find_beer_sources(state, conn.a, pid, &[]);
    for s in find_beer_sources(state, conn.b, pid, &[]) {
        if !out.contains(&s) {
            out.push(s);
        }
    }
    out
}

/// Candidates for the second rail link after `first_conn` has been placed,
/// each with the full set of legal beer sources (matching `execute_network_double`).
/// Uses the cheapest first-link coal as a representative dry-run (only the
/// set of reachable second links matters here, not the coal pairing).
pub fn get_valid_second_rail_links(
    state: &mut GameState<impl Rng>,
    pid: usize,
    first_conn: usize,
) -> Vec<usize> {
    let first = match connections().iter().find(|c| c.id == first_conn) {
        Some(c) => c,
        None => return Vec::new(),
    };
    let Some(coal1) = cheapest_coal_for_connection(state, first) else {
        return Vec::new();
    };
    get_second_rail_options(state, pid, first_conn, coal1)
        .into_iter()
        .map(|o| o.conn)
        .collect()
}

pub fn get_second_rail_options(
    state: &mut GameState<impl Rng>,
    pid: usize,
    first_conn: usize,
    coal1: CoalSource,
) -> Vec<SecondRailOption> {
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

    // `coal1` is the ACTUAL first-link coal choice (the caller's canonical).
    // Enumerate coal2 against its consumption, NOT some internally-cheapest
    // choice — otherwise the emitted (coal1, coal2) pair may be unexecutable.
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
    let coal1_undo = consume_coal_for_double_rail(state, coal1);

    let mut out = Vec::new();
    for conn in connections() {
        if conn.id == first_conn || !conn.rail || state.links[conn.id].is_some() {
            continue;
        }

        if !is_in_network(state, pid, conn.a) && !is_in_network(state, pid, conn.b) {
            continue;
        }
        // Enumerate the coal choices for the second link with the first link
        // still in place (this is the state the player will execute from).
        // Keep only affordable selections; empty means no legal coal source.
        let coal2_opts: Vec<Vec<CoalSource>> =
            coal_options_for_connection(state, conn, 1)
                .into_iter()
                .filter(|sel| !sel.is_empty())
                .filter(|sel| {
                    sel.iter().all(|s| {
                        let cost = if s.free { 0 } else { s.price as i32 };
                        cost <= budget_left
                    })
                })
                .collect();
        if coal2_opts.is_empty() {
            continue;
        }
        // Place the second link so beer connectivity matches execution time.
        state.links[conn.id] = Some(crate::state::Link { player: pid, is_canal: false });
        let beers = beer_sources_for_link(state, pid, conn);
        state.links[conn.id] = None;
        if beers.is_empty() {
            continue;
        }
        out.push(SecondRailOption {
            conn: conn.id,
            beers,
            coal2_opts,
        });
    }

    rollback_double_rail_coal(state, coal1_undo);
    state.links[first_conn] = None;
    out
}

#[derive(Clone, Copy)]
enum ResourceUndo {
    Market { prev_market: usize },
    City {
        key: usize,
        prev_cubes: u8,
        prev_flipped: bool,
        owner: usize,
        prev_income_space: u8,
    },
}

fn consume_coal_for_double_rail(
    state: &mut GameState<impl Rng>,
    coal: CoalSource,
) -> ResourceUndo {
    match coal.kind {
        CoalSourceKind::Mine => {
            let tile = state.city_tiles[coal.key]
                .as_ref()
                .expect("coal mine source must exist when consumed");
            let undo = ResourceUndo::City {
                key: coal.key,
                prev_cubes: tile.resource_cubes,
                prev_flipped: tile.flipped,
                owner: tile.player,
                prev_income_space: state.players[tile.player].income_space,
            };
            state.consume_from_city(coal.key);
            undo
        }
        CoalSourceKind::Market => {
            let undo = ResourceUndo::Market {
                prev_market: state.coal_market,
            };
            state.take_market_coal();
            undo
        }
    }
}

fn rollback_double_rail_coal(state: &mut GameState<impl Rng>, undo: ResourceUndo) {
    match undo {
        ResourceUndo::Market { prev_market } => state.coal_market = prev_market,
        ResourceUndo::City {
            key,
            prev_cubes,
            prev_flipped,
            owner,
            prev_income_space,
        } => {
            if let Some(tile) = state.city_tiles[key].as_mut() {
                tile.resource_cubes = prev_cubes;
                tile.flipped = prev_flipped;
            }
            state.players[owner].income_space = prev_income_space;
        }
    }
}

/// Human-readable description of one coal consumption (source, owner, flip, income).
fn describe_coal_consumption(state: &GameState<impl Rng>, undo: &ResourceUndo) -> String {
    match undo {
        ResourceUndo::Market { prev_market } => format!(
            "市场煤 {}→{}",
            prev_market,
            state.coal_market
        ),
        ResourceUndo::City {
            key,
            prev_cubes,
            prev_flipped,
            owner,
            prev_income_space,
        } => {
            let loc = loc_from_key(state, *key).unwrap_or(Loc::Birmingham);
            let now = state.city_tiles[*key].as_ref();
            let cubes_now = now.map_or(0, |t| t.resource_cubes);
            let flipped = now.map_or(false, |t| t.flipped);
            // Only the consumption that empties the tile (prev==1) is the one
            // that flips it; earlier partial consumptions must not claim the flip.
            let flip = if !*prev_flipped && flipped && *prev_cubes == 1 {
                let gained = state.players[*owner].income_space.saturating_sub(*prev_income_space);
                format!(" 翻面!P{}收入+{}", owner, gained)
            } else {
                String::new()
            };
            format!(
                "煤@{} P{} {cubes_now}/{}桶{flip}",
                log_loc_label(loc),
                owner,
                prev_cubes
            )
        }
    }
}

/// Describe a beer source before consumption: label + (owner, cubes, income) if it has an owner.
fn describe_beer_before(state: &GameState<impl Rng>, src: &BeerSource) -> (String, Option<(usize, u8, u8)>) {
    match src.kind {
        BeerSourceKind::Merchant => {
            let loc = state.merchants[src.merchant_idx.unwrap_or(0)].loc;
            (format!("商家酒@{}", log_loc_label(loc)), None)
        }
        _ => {
            if let Some(fi) = src.farm_idx {
                let loc = if fi == 0 { Loc::BreweryNorth } else { Loc::BrewerySouth };
                let info = state.farm_tiles[fi]
                    .as_ref()
                    .map(|t| (t.player, t.resource_cubes, t.def.income));
                (
                    format!("酒@{}", log_loc_label(loc)),
                    info.map(|(o, c, i)| (o, c, i)),
                )
            } else {
                let loc = loc_from_key(state, src.key).unwrap_or(Loc::Birmingham);
                let info = state.city_tiles[src.key]
                    .as_ref()
                    .map(|t| (t.player, t.resource_cubes, t.def.income));
                (
                    format!("酒@{}", log_loc_label(loc)),
                    info.map(|(o, c, i)| (o, c, i)),
                )
            }
        }
    }
}

/// Whether a beer source's tile is flipped (fully depleted) now.
fn beer_source_flipped(state: &GameState<impl Rng>, src: &BeerSource) -> bool {
    match src.kind {
        BeerSourceKind::Merchant => false,
        _ => {
            if let Some(fi) = src.farm_idx {
                state.farm_tiles[fi].as_ref().map_or(false, |t| t.flipped)
            } else {
                state.city_tiles[src.key].as_ref().map_or(false, |t| t.flipped)
            }
        }
    }
}

pub fn execute_network_double(
    state: &mut GameState<impl Rng>,
    pid: usize,
    conn1: usize,
    conn2: usize,
    coal1: CoalSource,
    coal2: CoalSource,
    beer: BeerSource,
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

    // The player must explicitly choose a legal coal source for the first link.
    let legal_coal1 = coal_options_for_connection(state, c1, 1);
    if !legal_coal1.iter().any(|opt| opt.as_slice() == [coal1]) {
        state.links[conn1] = None;
        return Err("Chosen coal source is not legal for the first link".into());
    }
    let mut total = RAIL_DOUBLE_LINK_COST + if coal1.free { 0 } else { coal1.price as i32 };
    let coal1_undo = consume_coal_for_double_rail(state, coal1);

    // Link 2 adjacency (with link1 on board)
    if !is_in_network(state, pid, c2.a) && !is_in_network(state, pid, c2.b) {
        rollback_double_rail_coal(state, coal1_undo);
        state.links[conn1] = None;
        return Err("Second link is not adjacent to your network".into());
    }
    state.links[conn2] = Some(crate::state::Link { player: pid, is_canal: false });

    // The player must explicitly choose a legal coal source for the second link.
    let legal_coal2 = coal_options_for_connection(state, c2, 1);
    if !legal_coal2.iter().any(|opt| opt.as_slice() == [coal2]) {
        rollback_double_rail_coal(state, coal1_undo);
        state.links[conn1] = None;
        state.links[conn2] = None;
        return Err("Chosen coal source is not legal for the second link".into());
    }
    total += if coal2.free { 0 } else { coal2.price as i32 };
    let coal2_undo = consume_coal_for_double_rail(state, coal2);

    // The player must explicitly choose a legal beer source for the second link.
    let legal_beers = beer_sources_for_link(state, pid, c2);
    if !legal_beers.contains(&beer) {
        rollback_double_rail_coal(state, coal2_undo);
        rollback_double_rail_coal(state, coal1_undo);
        state.links[conn1] = None;
        state.links[conn2] = None;
        return Err("Chosen beer source is not legal for the second link".into());
    }

    if total > state.players[pid].money {
        rollback_double_rail_coal(state, coal2_undo);
        rollback_double_rail_coal(state, coal1_undo);
        state.links[conn1] = None;
        state.links[conn2] = None;
        return Err("Cannot afford both links".into());
    }

    // Capture pre-consumption state for the log.
    let coal1_desc = describe_coal_consumption(state, &coal1_undo);
    let coal2_desc = describe_coal_consumption(state, &coal2_undo);
    let (beer_label, beer_before) = describe_beer_before(state, &beer);

    // Commit
    state.consume_beer_source(&beer);
    state.spend_money(pid, total);
    state.players[pid].rail_links -= 2;
    discard_card(state, pid, card_index);

    let beer_note = match beer_before {
        Some((owner, cubes, income)) => {
            let flipped = beer_source_flipped(state, &beer);
            let flip_note = if flipped {
                let gained = state.players[owner].income_space;
                format!(" 翻面!P{owner}收入+{income}(→{gained})")
            } else {
                format!(" {cubes}→{}桶(未翻)", cubes.saturating_sub(1))
            };
            format!("{beer_label} P{owner}{flip_note}")
        }
        None => format!("{beer_label} (商家)"),
    };

    Ok(format!(
        "Built 2 rail links: {} - {} and {} - {} | 煤1:{coal1_desc} | 煤2:{coal2_desc} | 酒:{beer_note}",
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
    affordable_develop_iron_count(state, pid) >= 1 && !state.players[pid].developable_types().is_empty()
}

fn affordable_develop_iron_count(state: &GameState<impl Rng>, pid: usize) -> usize {
    let mut money = state.players[pid].money;
    let mut count = 0usize;

    for src in find_iron_sources(state) {
        if src.free {
            count += 1;
        } else if money >= src.price as i32 {
            money -= src.price as i32;
            count += 1;
        } else {
            break;
        }
    }

    count
}

pub fn execute_develop(
    state: &mut GameState<impl Rng>,
    pid: usize,
    ind1: IndustryType,
    ind2: Option<IndustryType>,
    iron: &[IronSource],
    card_index: usize,
) -> Result<String, String> {
    let tiles_to_develop = if ind2.is_some() { 2 } else { 1 };
    validate_iron_choice(state, tiles_to_develop, iron)?;

    // Pre-check affordability of market iron
    let mut money_needed = 0;
    for src in iron {
        if !src.free {
            money_needed += src.price as i32;
        }
    }
    if money_needed > state.players[pid].money {
        return Err("Cannot afford market iron".into());
    }

    // Consume exactly the chosen iron sources.
    for src in iron {
        if src.free {
            state.consume_from_city(src.key);
        } else {
            let price = state.iron_price();
            state.spend_money(pid, price as i32);
            state.take_market_iron();
        }
    }

    // Remove tiles from mat
    let t1 = state.players[pid]
        .consume_tile(ind1)
        .ok_or("No tile available to develop for first industry")?;
    let mut removed = vec![format!("{} Lv{}", log_industry_label(ind1), t1.level)];
    if let Some(i2) = ind2 {
        let t2 = state.players[pid]
            .consume_tile(i2)
            .ok_or("No tile available to develop for second industry")?;
        removed.push(format!("{} Lv{}", log_industry_label(i2), t2.level));
    }

    discard_card(state, pid, card_index);
    Ok(format!("Developed {}", removed.join(" + ")))
}

// ---------------------------------------------------------------------------
// SELL
// ---------------------------------------------------------------------------

#[derive(Clone)]
pub struct SellTarget {
    pub key: usize,
    pub beer_needed: u8,
    pub routes: Vec<SellRoute>,
}

#[derive(Debug, Clone, Copy)]
pub struct SellRoute {
    pub merchant_index: usize,
    pub use_merchant_beer: bool,
}

fn sell_routes_for_target(
    state: &GameState<impl Rng>,
    pid: usize,
    loc: Loc,
    merchant_indices: &[usize],
    beer_needed: u8,
) -> Vec<SellRoute> {
    let mut routes = Vec::new();
    if merchant_indices.is_empty() {
        return routes;
    }

    let brewery_beer = find_beer_sources(state, loc, pid, &[]).len();

    if beer_needed == 0 {
        for &merchant_index in merchant_indices {
            routes.push(SellRoute {
                merchant_index,
                use_merchant_beer: false,
            });
        }
        return routes;
    }

    if brewery_beer >= beer_needed as usize {
        for &merchant_index in merchant_indices {
            routes.push(SellRoute {
                merchant_index,
                use_merchant_beer: false,
            });
        }
    }

    for &merchant_index in merchant_indices {
        if state.merchants[merchant_index].has_beer && brewery_beer + 1 >= beer_needed as usize {
            routes.push(SellRoute {
                merchant_index,
                use_merchant_beer: true,
            });
        }
    }

    routes
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
        let beer_needed = t.def.beers_to_sell.unwrap_or(0);
        let routes = sell_routes_for_target(state, pid, loc, &merchant_indices, beer_needed);
        if routes.is_empty() {
            continue;
        }
        out.push(SellTarget {
            key: k,
            beer_needed,
            routes,
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

fn beer_source_label(state: &GameState<impl Rng>, src: &BeerSource) -> String {
    match src.kind {
        crate::graph::BeerSourceKind::Merchant => src
            .merchant_idx
            .and_then(|mi| state.merchants.get(mi))
            .map(|mt| format!("商家酒@{}", log_loc_label(mt.loc)))
            .unwrap_or_else(|| "商家酒@?".to_string()),
        crate::graph::BeerSourceKind::Own | crate::graph::BeerSourceKind::Opponent => {
            if let Some(fi) = src.farm_idx {
                let loc = if fi == 0 {
                    Loc::BreweryNorth
                } else {
                    Loc::BrewerySouth
                };
                format!("酒@{}", log_loc_label(loc))
            } else {
                let loc = loc_from_key(state, src.key)
                    .map(log_loc_label)
                    .unwrap_or_else(|| "?".to_string());
                format!("酒@{loc}")
            }
        }
    }
}

pub fn execute_sell(
    state: &mut GameState<impl Rng>,
    pid: usize,
    keys: &[usize],
    merchant_indices: &[usize],
    use_merchant_beer: &[bool],
    card_index: usize,
) -> Result<String, String> {
    if merchant_indices.len() != keys.len() || use_merchant_beer.len() != keys.len() {
        return Err("Sell move shape mismatch".into());
    }

    let mut sold = 0usize;
    let mut notes = Vec::new();
    let mut beer_notes = Vec::new();

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

        let valid_merchants = sell_merchants_for(state, pid, loc, ind);
        if valid_merchants.is_empty() {
            continue;
        }

        let mut beer_remaining = beer_needed;
        let chosen_merchant = merchant_indices[entry_idx];
        let use_merchant = use_merchant_beer
            .get(entry_idx)
            .copied()
            .unwrap_or(false);

        if !valid_merchants.contains(&chosen_merchant) {
            return Err("Chosen merchant is not a valid buyer".into());
        }

        // Merchant beer first, if requested: use the explicitly chosen merchant.
        if beer_remaining > 0 && use_merchant {
            if !state.merchants[chosen_merchant].has_beer {
                return Err("Chosen merchant has no beer".into());
            }
            let brewery_beer = find_beer_sources(state, loc, pid, &[]);
            if brewery_beer.len() + 1 < beer_remaining as usize {
                return Err("Not enough beer available for chosen merchant route".into());
            }
            let mt_loc = state.merchants[chosen_merchant].loc;
            state.merchants[chosen_merchant].has_beer = false;
            beer_remaining -= 1;
            beer_notes.push(format!(
                "{} 使用 商家酒@{}",
                log_industry_label(ind),
                log_loc_label(mt_loc)
            ));
            let bonus = merchant_bonus_for(state, mt_loc);
            let note = apply_merchant_bonus(state, pid, bonus);
            notes.push(note);
        } else if beer_remaining > 0 {
            let brewery_beer = find_beer_sources(state, loc, pid, &[]);
            if brewery_beer.len() < beer_remaining as usize {
                return Err("Not enough brewery beer for chosen merchant route".into());
            }
        }

        // Remaining beer from breweries (own anywhere, opponents connected)
        if beer_remaining > 0 {
            let beer = find_beer_sources(state, loc, pid, &[]);
            if beer.len() < beer_remaining as usize {
                continue; // can't pay beer; skip tile
            }
            let consumed: Vec<BeerSource> = beer.into_iter().take(beer_remaining as usize).collect();
            let labels: Vec<String> = consumed
                .iter()
                .map(|b| beer_source_label(state, b))
                .collect();
            for b in consumed {
                state.consume_beer_source(&b);
            }
            if !labels.is_empty() {
                beer_notes.push(format!(
                    "{} 使用 {}",
                    log_industry_label(ind),
                    labels.join(", ")
                ));
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
    let mut extra = Vec::new();
    if !notes.is_empty() {
        extra.push(notes.join("; "));
    }
    if !beer_notes.is_empty() {
        extra.push(format!("beer: {}", beer_notes.join("; ")));
    }
    Ok(format!(
        "Sold {} tile(s){}",
        sold,
        if extra.is_empty() {
            String::new()
        } else {
            format!(" [{}]", extra.join(" | "))
        }
    ))
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
            state.pending_bonus = Some(crate::state::PendingBonus::FreeDevelop {
                player_id: pid,
                count: 1,
            });
            "free develop pending".to_string()
        }
    }
}

pub fn execute_resolve_free_develop(
    state: &mut GameState<impl Rng>,
    pid: usize,
    ind1: IndustryType,
    ind2: Option<IndustryType>,
) -> Result<String, String> {
    let Some(crate::state::PendingBonus::FreeDevelop { player_id, count }) = state.pending_bonus else {
        return Err("No pending free develop bonus".into());
    };
    if player_id != pid {
        return Err("Pending bonus belongs to another player".into());
    }

    let available: Vec<IndustryType> = state.players[pid]
        .developable_types()
        .into_iter()
        .map(|(ind, _)| ind)
        .collect();
    if !available.contains(&ind1) {
        return Err("First free develop choice is not developable".into());
    }

    let remaining_first = state.players[pid].remaining_count(ind1);
    if count >= 2 {
        let Some(ind2) = ind2 else {
            return Err("Free develop bonus requires two choices".into());
        };
        if !available.contains(&ind2) {
            return Err("Second free develop choice is not developable".into());
        }
        if ind1 == ind2 && remaining_first < 2 {
            return Err("Not enough tiles remaining to free develop the same industry twice".into());
        }
    } else if ind2.is_some() {
        return Err("This free develop bonus only allows one choice".into());
    }

    let first = state.players[pid]
        .consume_tile(ind1)
        .ok_or("No tile available to develop for first industry")?;
    let mut removed = vec![format!("{} Lv{}", log_industry_label(ind1), first.level)];
    if let Some(i2) = ind2 {
        let t2 = state.players[pid]
            .consume_tile(i2)
            .ok_or("No tile available to develop for second industry")?;
        removed.push(format!("{} Lv{}", log_industry_label(i2), t2.level));
    }
    // Track develop-action economy (the free-develop bonus path does NOT go
    // through here, so it never counts against the action budget).
    match state.era {
        crate::data::Era::Canal => state.players[pid].develops_in_canal += 1,
        crate::data::Era::Rail => state.players[pid].develops_in_rail += 1,
    }

    state.pending_bonus = None;
    Ok(format!("Resolved free develop: {}", removed.join(", ")))
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

pub fn legal_moves(state: &mut GameState<impl Rng + Clone>) -> Vec<Move> {
    let pid = state.current_player_id();

    if let Some(crate::state::PendingBonus::FreeDevelop { player_id, count }) = state.pending_bonus {
        if player_id != pid {
            return Vec::new();
        }
        let types: Vec<IndustryType> = state.players[pid]
            .developable_types()
            .into_iter()
            .map(|(ind, _)| ind)
            .collect();
        let mut moves = Vec::new();
        for &ind1 in &types {
            if count >= 2 {
                let rem1 = state.players[pid].remaining_count(ind1);
                let mut second_types = types.clone();
                if rem1 < 2 {
                    second_types.retain(|&x| x != ind1);
                }
                for &ind2 in &second_types {
                    moves.push(Move::ResolveFreeDevelop {
                        ind1,
                        ind2: Some(ind2),
                    });
                }
            } else {
                moves.push(Move::ResolveFreeDevelop { ind1, ind2: None });
            }
        }
        return moves;
    }

    let player_hand_len = state.players[pid].hand.len();
    if player_hand_len == 0 {
        return vec![];
    }

    let mut moves = Vec::new();

    // BUILD
    for t in get_valid_build_targets(state, pid) {
        let coal_needed = t.cost_coal as usize;
        let iron_needed = t.cost_iron as usize;
        let coal_opts = coal_source_options(state, t.loc, coal_needed);
        let iron_opts = iron_source_options(state, iron_needed);
        for coal in &coal_opts {
            for iron in &iron_opts {
                for ci in valid_build_cards(state, &state.players[pid], pid, t.loc, t.ind) {
                    moves.push(Move::Build {
                        loc: t.loc,
                        slot_index: t.slot_index,
                        ind: t.ind,
                        coal: coal.clone(),
                        iron: iron.clone(),
                        card_index: ci,
                    });
                }
            }
        }
    }

    // NETWORK (single + double)
    if state.era == Era::Canal && state.players[pid].canal_links > 0
        || state.era == Era::Rail && state.players[pid].rail_links > 0
    {
        for conn in get_valid_network_targets(state, pid) {
            for ci in any_card_indices(&state.players[pid]) {
                if state.era == Era::Canal {
                    moves.push(Move::Network {
                        conn_id: conn,
                        coal: None,
                        card_index: ci,
                    });
                } else {
                    let c = &connections()[conn];
                    let coal_opts = coal_options_for_connection(state, c, 1);
                    for coal in &coal_opts {
                        moves.push(Move::Network {
                            conn_id: conn,
                            coal: coal.first().copied(),
                            card_index: ci,
                        });
                    }
                }
            }
        }
    }
    if state.era == Era::Rail {
        // For each valid first link, find valid second links (double rail) and
        // enumerate every legal coal/beer source choice for the second link.
        let single = get_valid_network_targets(state, pid);
        for conn1 in single {
            let c1 = &connections()[conn1];
            let coal1_opts = coal_options_for_connection(state, c1, 1);
            for coal1 in &coal1_opts {
                if coal1.is_empty() {
                    continue;
                }
                for opt in get_second_rail_options(state, pid, conn1, coal1[0]) {
                    for coal2 in &opt.coal2_opts {
                        for beer in &opt.beers {
                            for ci in any_card_indices(&state.players[pid]) {
                                moves.push(Move::NetworkDouble {
                                    conn1,
                                    conn2: opt.conn,
                                    coal1: coal1[0],
                                    coal2: coal2[0],
                                    beer: *beer,
                                    card_index: ci,
                                });
                            }
                        }
                    }
                }
            }
        }
    }

    // DEVELOP
    if can_develop(state, pid) {
        let affordable_iron = affordable_develop_iron_count(state, pid);
        let types: Vec<IndustryType> = state.players[pid]
            .developable_types()
            .into_iter()
            .map(|(ind, _)| ind)
            .collect();
        for &ind1 in &types {
            // Single develop
            if affordable_iron >= 1 {
                let iron_opts = iron_source_options(state, 1);
                for iron in &iron_opts {
                    for ci in any_card_indices(&state.players[pid]) {
                        moves.push(Move::Develop {
                            ind1,
                            ind2: None,
                            iron: iron.clone(),
                            card_index: ci,
                        });
                    }
                }
            }
            // Double develop (two distinct types, or same type if count>1)
            if affordable_iron >= 2 {
                let rem1 = state.players[pid].remaining_count(ind1);
                let mut second_types: Vec<IndustryType> = types.clone();
                if rem1 < 2 {
                    second_types.retain(|&x| x != ind1);
                }
                let iron_opts = iron_source_options(state, 2);
                for &ind2 in &second_types {
                    for iron in &iron_opts {
                        for ci in any_card_indices(&state.players[pid]) {
                            moves.push(Move::Develop {
                                ind1,
                                ind2: Some(ind2),
                                iron: iron.clone(),
                                card_index: ci,
                            });
                        }
                    }
                }
            }
        }
    }

    // SELL (single-tile + multi-tile plans; a Sell action may sell several tiles)
    let sell_targets = get_valid_sell_targets(state, pid);
    if !sell_targets.is_empty() {
        // Single-tile sells (one tile per action).
        for t in &sell_targets {
            for route in &t.routes {
                for ci in any_card_indices(&state.players[pid]) {
                    moves.push(Move::Sell {
                        keys: vec![t.key],
                        merchant_indices: vec![route.merchant_index],
                        use_merchant_beer: vec![route.use_merchant_beer],
                        card_index: ci,
                    });
                }
            }
        }
        // Multi-tile sells: bounded subsets (2..=3 tiles) with compatible routes.
        let multi_plans = build_multi_sell_plans(state, pid, &sell_targets);
        for ci in any_card_indices(&state.players[pid]) {
            for plan in &multi_plans {
                moves.push(Move::Sell {
                    keys: plan.0.clone(),
                    merchant_indices: plan.1.clone(),
                    use_merchant_beer: plan.2.clone(),
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

// ---------------------------------------------------------------------------
// Slot-level legal move generation (fast path for MCTS / policy head)
// ---------------------------------------------------------------------------

/// A policy slot together with ONE executable representative move occupying it.
/// Resource-source and card choices are folded at generation time (they are
/// strategically identical per slot), so the search tree can key children
/// directly on slots instead of paying the raw-move combinatorial cost.
pub struct SlotMove {
    pub slot: usize,
    pub mv: Move,
}

fn push_slot(out: &mut Vec<SlotMove>, mv: Move) {
    for slot in crate::policy::move_slots(&mv) {
        out.push(SlotMove { slot, mv: mv.clone() });
    }
}

/// Legal actions at SLOT granularity: one executable representative per occupied
/// policy slot, mirroring the slot coverage of `legal_moves`. Sell emits one
/// entry per route (a slot may appear several times; callers group by slot).
pub fn legal_slot_moves(state: &mut GameState<impl Rng + Clone>) -> Vec<SlotMove> {
    let pid = state.current_player_id();
    let mut out: Vec<SlotMove> = Vec::new();

    if let Some(crate::state::PendingBonus::FreeDevelop { player_id, count }) = state.pending_bonus {
        if player_id != pid {
            return Vec::new();
        }
        let types: Vec<IndustryType> = state.players[pid]
            .developable_types()
            .into_iter()
            .map(|(ind, _)| ind)
            .collect();
        for &ind1 in &types {
            if count >= 2 {
                let rem1 = state.players[pid].remaining_count(ind1);
                let mut second_types = types.clone();
                if rem1 < 2 {
                    second_types.retain(|&x| x != ind1);
                }
                for &ind2 in &second_types {
                    push_slot(&mut out, Move::ResolveFreeDevelop { ind1, ind2: Some(ind2) });
                }
            } else {
                push_slot(&mut out, Move::ResolveFreeDevelop { ind1, ind2: None });
            }
        }
        return out;
    }

    let player_hand_len = state.players[pid].hand.len();
    if player_hand_len == 0 {
        return Vec::new();
    }

    // BUILD: one representative (first coal / iron / card) per legal target.
    for t in get_valid_build_targets(state, pid) {
        let Some(coal) = coal_source_options(state, t.loc, t.cost_coal as usize).into_iter().next() else {
            continue;
        };
        let Some(iron) = iron_source_options(state, t.cost_iron as usize).into_iter().next() else {
            continue;
        };
        let Some(&card_index) = valid_build_cards(state, &state.players[pid], pid, t.loc, t.ind).first()
        else {
            continue;
        };
        push_slot(
            &mut out,
            Move::Build { loc: t.loc, slot_index: t.slot_index, ind: t.ind, coal, iron, card_index },
        );
    }

    // NETWORK (single + double): one representative per connection / pair.
    if state.era == Era::Canal && state.players[pid].canal_links > 0
        || state.era == Era::Rail && state.players[pid].rail_links > 0
    {
        let Some(&card_index) = any_card_indices(&state.players[pid]).first() else {
            return out;
        };
        for conn in get_valid_network_targets(state, pid) {
            let mv = if state.era == Era::Canal {
                Move::Network { conn_id: conn, coal: None, card_index }
            } else {
                let c = &connections()[conn];
                let Some(coal) = coal_options_for_connection(state, c, 1)
                    .into_iter().next().and_then(|o| o.into_iter().next()) else {
                    continue;
                };
                Move::Network { conn_id: conn, coal: Some(coal), card_index }
            };
            push_slot(&mut out, mv);
        }
    }
    if state.era == Era::Rail {
        let single = get_valid_network_targets(state, pid);
        let Some(&card_index) = any_card_indices(&state.players[pid]).first() else {
            return out;
        };
        for conn1 in single {
            let c1 = &connections()[conn1];
            let Some(coal1) = coal_options_for_connection(state, c1, 1)
                .into_iter().next().and_then(|o| o.into_iter().next()) else {
                continue;
            };
            for opt in get_second_rail_options(state, pid, conn1, coal1) {
                let Some(&coal2) = opt.coal2_opts.first().and_then(|o| o.first()) else {
                    continue;
                };
                let Some(&beer) = opt.beers.first() else {
                    continue;
                };
                push_slot(
                    &mut out,
                    Move::NetworkDouble { conn1, conn2: opt.conn, coal1, coal2, beer, card_index },
                );
            }
        }
    }

    // DEVELOP (single + double): one representative iron selection per slot.
    if can_develop(state, pid) {
        let affordable_iron = affordable_develop_iron_count(state, pid);
        let types: Vec<IndustryType> = state.players[pid]
            .developable_types()
            .into_iter()
            .map(|(ind, _)| ind)
            .collect();
        for &ind1 in &types {
            if affordable_iron >= 1 {
                let Some(iron) = iron_source_options(state, 1).into_iter().next() else {
                    continue;
                };
                let Some(&card_index) = any_card_indices(&state.players[pid]).first() else {
                    continue;
                };
                push_slot(&mut out, Move::Develop { ind1, ind2: None, iron, card_index });
            }
            if affordable_iron >= 2 {
                let rem1 = state.players[pid].remaining_count(ind1);
                let mut second_types: Vec<IndustryType> = types.clone();
                if rem1 < 2 {
                    second_types.retain(|&x| x != ind1);
                }
                let Some(iron) = iron_source_options(state, 2).into_iter().next() else {
                    continue;
                };
                for &ind2 in &second_types {
                    let Some(&card_index) = any_card_indices(&state.players[pid]).first() else {
                        continue;
                    };
                    push_slot(
                        &mut out,
                        Move::Develop { ind1, ind2: Some(ind2), iron: iron.clone(), card_index },
                    );
                }
            }
        }
    }

    // SELL: single-tile per route + multi-tile plans (each plan occupies the
    // slots of every tile it sells).
    let sell_targets = get_valid_sell_targets(state, pid);
    if !sell_targets.is_empty() {
        let Some(&card_index) = any_card_indices(&state.players[pid]).first() else {
            return out;
        };
        for t in &sell_targets {
            for route in &t.routes {
                push_slot(
                    &mut out,
                    Move::Sell {
                        keys: vec![t.key],
                        merchant_indices: vec![route.merchant_index],
                        use_merchant_beer: vec![route.use_merchant_beer],
                        card_index,
                    },
                );
            }
        }
        for plan in build_multi_sell_plans(state, pid, &sell_targets) {
            push_slot(
                &mut out,
                Move::Sell {
                    keys: plan.0,
                    merchant_indices: plan.1,
                    use_merchant_beer: plan.2,
                    card_index,
                },
            );
        }
    }

    // LOAN / SCOUT / PASS: one representative each.
    if state.can_take_loan(pid) {
        if let Some(&card_index) = any_card_indices(&state.players[pid]).first() {
            push_slot(&mut out, Move::Loan { card_index });
        }
    }
    if can_scout(state, pid) && state.players[pid].hand.len() >= 3 {
        push_slot(&mut out, Move::Scout { card_indices: [0, 1, 2] });
    }
    if let Some(&card_index) = any_card_indices(&state.players[pid]).first() {
        push_slot(&mut out, Move::Pass { card_index });
    }

    out
}

/// Generate all `k`-subsets of `0..n` (for the bounded multi-sell enumeration).
fn gen_combos(out: &mut Vec<Vec<usize>>, n: usize, k: usize, start: usize, cur: &mut Vec<usize>) {
    if cur.len() == k {
        out.push(cur.clone());
        return;
    }
    for i in start..n {
        cur.push(i);
        gen_combos(out, n, k, i + 1, cur);
        cur.pop();
    }
}

/// Bounded set of multi-tile sell plans: one action selling 2..=3 tiles, each
/// to a compatible merchant route with distinct merchant-beer sources. Each
/// plan is dry-run validated (must flip every planned tile). Returns
/// (keys, merchant_indices, use_merchant_beer) triples.
fn build_multi_sell_plans<R: Rng + Clone>(
    state: &GameState<R>,
    pid: usize,
    targets: &[SellTarget],
) -> Vec<(Vec<usize>, Vec<usize>, Vec<bool>)> {
    if targets.len() < 2 {
        return Vec::new();
    }
    let n = targets.len();
    let mut combos = Vec::new();
    let mut cur = Vec::new();
    for k in 2..=3 {
        gen_combos(&mut combos, n, k, 0, &mut cur);
    }
    let max_plans = 24;
    let mut out: Vec<(Vec<usize>, Vec<usize>, Vec<bool>)> = Vec::new();
    let mut seen = std::collections::HashSet::new();
    for combo in combos {
        if out.len() >= max_plans {
            break;
        }
        let mut keys = Vec::with_capacity(combo.len());
        let mut merchants = Vec::with_capacity(combo.len());
        let mut use_beer = Vec::with_capacity(combo.len());
        let mut committed_merchant_beer = Vec::new();
        let mut ok = true;
        for &ti in &combo {
            let t = &targets[ti];
            let mut assigned = false;
            for route in &t.routes {
                if route.use_merchant_beer && committed_merchant_beer.contains(&route.merchant_index) {
                    continue;
                }
                keys.push(t.key);
                merchants.push(route.merchant_index);
                use_beer.push(route.use_merchant_beer);
                if route.use_merchant_beer {
                    committed_merchant_beer.push(route.merchant_index);
                }
                assigned = true;
                break;
            }
            if !assigned {
                ok = false;
                break;
            }
        }
        if !ok {
            continue;
        }
        // Dry-run: the plan must actually sell every planned tile.
        let mut sim = state.clone();
        if execute_sell(&mut sim, pid, &keys, &merchants, &use_beer, 0).is_err() {
            continue;
        }
        if !keys
            .iter()
            .all(|&key| sim.city_tiles[key].as_ref().map(|tile| tile.flipped).unwrap_or(false))
        {
            continue;
        }
        if seen.insert(keys.clone()) {
            out.push((keys, merchants, use_beer));
        }
    }
    out
}


pub fn apply_move(state: &mut GameState<impl Rng>, mv: &Move) -> Result<String, String> {
    let pid = state.current_player_id();
    match mv {
        Move::Build { loc, slot_index, ind, coal, iron, card_index } => {
            execute_build(state, pid, *loc, *slot_index, *ind, coal, iron, *card_index)
        }
        Move::Network { conn_id, coal, card_index } => {
            execute_network(state, pid, *conn_id, *coal, *card_index)
        }
        Move::NetworkDouble { conn1, conn2, coal1, coal2, beer, card_index } => {
            execute_network_double(state, pid, *conn1, *conn2, *coal1, *coal2, *beer, *card_index)
        }
        Move::Develop { ind1, ind2, iron, card_index } => {
            execute_develop(state, pid, *ind1, *ind2, iron, *card_index)
        }
        Move::Sell {
            keys,
            merchant_indices,
            use_merchant_beer,
            card_index,
        } => {
            execute_sell(state, pid, keys, merchant_indices, use_merchant_beer, *card_index)
        }
        Move::ResolveFreeDevelop { ind1, ind2 } => {
            execute_resolve_free_develop(state, pid, *ind1, *ind2)
        }
        Move::Loan { card_index } => execute_loan(state, pid, *card_index),
        Move::Scout { card_indices } => execute_scout(state, pid, *card_indices),
        Move::Pass { card_index } => execute_pass(state, pid, *card_index),
    }
}
