//! NETWORK action legality and execution, including double rail transactions.

use super::{
    coal_options_for_connection, discard_card, log_loc_label, require_card_index,
    source_purchase_cost, validate_connection_coal_choice,
};
use crate::data::Era;
use crate::graph::{
    BeerSource, BeerSourceKind, CoalSource, CoalSourceKind, cheapest_coal_for_connection,
    find_beer_sources, is_in_network, player_has_presence,
};
use crate::map::*;
use crate::state::GameState;
// ---------------------------------------------------------------------------
// NETWORK
// ---------------------------------------------------------------------------

pub fn get_valid_network_targets(state: &GameState, pid: usize) -> Vec<usize> {
    state.ensure_network_masks();
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
                let cost = RAIL_LINK_COST
                    + if coal.unwrap().free {
                        0
                    } else {
                        coal.unwrap().price as i32
                    };
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
    state: &mut GameState,
    pid: usize,
    conn_id: usize,
    coal: Option<CoalSource>,
    card_index: usize,
) -> Result<String, String> {
    require_card_index(state, pid, card_index)?;
    let conn = connections()
        .iter()
        .find(|c| c.id == conn_id)
        .ok_or("Invalid connection")?;
    if state.links[conn_id].is_some() {
        return Err("Connection already built".into());
    }

    let has_no_presence = !player_has_presence(state, pid);
    if !has_no_presence && !is_in_network(state, pid, conn.a) && !is_in_network(state, pid, conn.b)
    {
        return Err("Link is not adjacent to your network".into());
    }

    match state.era {
        Era::Canal => {
            if !conn.canal {
                return Err("Connection is not available in the canal era".into());
            }
            if state.players[pid].canal_links == 0 {
                return Err("No canal links remaining".into());
            }
            if coal.is_some() {
                return Err("Canal links do not consume coal".into());
            }
            if CANAL_LINK_COST > state.players[pid].money {
                return Err("Cannot afford this link".into());
            }
            state.spend_money(pid, CANAL_LINK_COST);
            let p = &mut state.players[pid];
            p.canal_links -= 1;
        }
        Era::Rail => {
            if !conn.rail {
                return Err("Connection is not available in the rail era".into());
            }
            if state.players[pid].rail_links == 0 {
                return Err("No rail links remaining".into());
            }
            let coal = coal.ok_or("A coal source is required for a rail link")?;
            validate_connection_coal_choice(state, conn, &[coal])?;
            let total = RAIL_LINK_COST
                + source_purchase_cost(&[coal], 1).expect("validated coal source must be present");
            if total > state.players[pid].money {
                return Err("Cannot afford this link".into());
            }
            state.consume_coal_source(&coal);
            state.spend_money(pid, total);
            let p = &mut state.players[pid];
            p.rail_links -= 1;
        }
    }

    state.set_link(conn_id, pid);

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
pub fn beer_sources_for_link(state: &GameState, pid: usize, conn: &Connection) -> Vec<BeerSource> {
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
    state: &mut GameState,
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
    state: &mut GameState,
    pid: usize,
    first_conn: usize,
    coal1: CoalSource,
) -> Vec<SecondRailOption> {
    state.ensure_network_masks();
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
    if !has_no_presence
        && !is_in_network(state, pid, first.a)
        && !is_in_network(state, pid, first.b)
    {
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
    // generation matches the real execution order. The transaction rolls the
    // mutation back on drop, so no early return / panic can leak it.
    let mut tx = RailTx::new(state);
    tx.place_link(first_conn, pid);
    tx.consume_coal(coal1);

    let mut out = Vec::new();
    for conn in connections() {
        if conn.id == first_conn || !conn.rail || tx.state_ref().links[conn.id].is_some() {
            continue;
        }

        if !is_in_network(tx.state_ref(), pid, conn.a)
            && !is_in_network(tx.state_ref(), pid, conn.b)
        {
            continue;
        }
        // Enumerate the coal choices for the second link with the first link
        // still in place (this is the state the player will execute from).
        // Keep only affordable selections; empty means no legal coal source.
        let coal2_opts: Vec<Vec<CoalSource>> = coal_options_for_connection(tx.state_ref(), conn, 1)
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
        tx.state().set_link(conn.id, pid);
        let beers = beer_sources_for_link(tx.state_ref(), pid, conn);
        tx.state().remove_link(conn.id);
        if beers.is_empty() {
            continue;
        }
        out.push(SecondRailOption {
            conn: conn.id,
            beers,
            coal2_opts,
        });
    }

    out
}

#[derive(Clone, Copy)]
enum ResourceUndo {
    Market {
        prev_market: usize,
    },
    City {
        key: usize,
        prev_cubes: u8,
        prev_flipped: bool,
        owner: usize,
        prev_income_space: u8,
    },
}

fn consume_coal_for_double_rail(state: &mut GameState, coal: CoalSource) -> ResourceUndo {
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
            state.consume_coal_source(&coal);
            undo
        }
        CoalSourceKind::Market => {
            let undo = ResourceUndo::Market {
                prev_market: state.coal_market,
            };
            state.consume_coal_source(&coal);
            undo
        }
    }
}

fn rollback_double_rail_coal(state: &mut GameState, undo: ResourceUndo) {
    match undo {
        ResourceUndo::Market { prev_market } => state.coal_market = prev_market,
        ResourceUndo::City {
            key,
            prev_cubes,
            prev_flipped,
            owner,
            prev_income_space,
        } => state.restore_consumed_city_tile(
            key,
            prev_cubes,
            prev_flipped,
            owner,
            prev_income_space,
        ),
    }
}

/// Transactional rail placement + coal consumption.
///
/// Placements and consumptions are recorded and rolled back (links removed,
/// coal sources restored) unless the transaction is committed. `Drop`
/// guarantees cleanup on every exit path, so legal-move dry-runs and
/// validation-error returns can never leak mutations into the board.
struct RailTx<'a> {
    state: Option<&'a mut GameState>,
    links: Vec<usize>,
    coals: Vec<ResourceUndo>,
    committed: bool,
}

impl<'a> RailTx<'a> {
    fn new(state: &'a mut GameState) -> Self {
        RailTx {
            state: Some(state),
            links: Vec::new(),
            coals: Vec::new(),
            committed: false,
        }
    }

    fn state(&mut self) -> &mut GameState {
        self.state.as_mut().expect("RailTx state already taken")
    }

    fn state_ref(&self) -> &GameState {
        self.state.as_ref().expect("RailTx state already taken")
    }

    fn place_link(&mut self, id: usize, pid: usize) {
        self.state().set_link(id, pid);
        self.links.push(id);
    }

    fn consume_coal(&mut self, coal: CoalSource) {
        let undo = consume_coal_for_double_rail(self.state(), coal);
        self.coals.push(undo);
    }

    fn describe_coal(&self, idx: usize) -> String {
        describe_coal_consumption(self.state_ref(), &self.coals[idx])
    }

    /// Keep the placements and consumptions (mark committed) and hand the
    /// state back for further mutation.
    fn commit(mut self) -> &'a mut GameState {
        self.committed = true;
        self.state.take().expect("RailTx state already taken")
    }
}

impl Drop for RailTx<'_> {
    fn drop(&mut self) {
        if self.committed {
            return;
        }
        if let Some(state) = self.state.as_mut() {
            for undo in self.coals.iter().rev() {
                rollback_double_rail_coal(state, *undo);
            }
            for &link in self.links.iter().rev() {
                state.remove_link(link);
            }
        }
    }
}

/// Human-readable description of one coal consumption (source, owner, flip, income).
fn describe_coal_consumption(state: &GameState, undo: &ResourceUndo) -> String {
    match undo {
        ResourceUndo::Market { prev_market } => {
            format!("市场煤 {}→{}", prev_market, state.coal_market)
        }
        ResourceUndo::City {
            key,
            prev_cubes,
            prev_flipped,
            owner,
            prev_income_space,
        } => {
            let loc = crate::state::loc_from_key(*key)
                .map(|(l, _)| l)
                .unwrap_or(Loc::Birmingham);
            let now = state.city_tiles[*key].as_ref();
            let cubes_now = now.map_or(0, |t| t.resource_cubes);
            let flipped = now.map_or(false, |t| t.flipped);
            // Only the consumption that empties the tile (prev==1) is the one
            // that flips it; earlier partial consumptions must not claim the flip.
            let flip = if !*prev_flipped && flipped && *prev_cubes == 1 {
                let gained = state.players[*owner]
                    .income_space
                    .saturating_sub(*prev_income_space);
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
fn describe_beer_before(state: &GameState, src: &BeerSource) -> (String, Option<(usize, u8, u8)>) {
    match src.kind {
        BeerSourceKind::Merchant => {
            let loc = state.merchants[src.merchant_idx.unwrap_or(0)].loc;
            (format!("商家酒@{}", log_loc_label(loc)), None)
        }
        _ => {
            if let Some(fi) = src.farm_idx {
                let loc = if fi == 0 {
                    Loc::BreweryNorth
                } else {
                    Loc::BrewerySouth
                };
                let info = state.farm_tiles[fi]
                    .as_ref()
                    .map(|t| (t.player, t.resource_cubes, t.def.income));
                (
                    format!("酒@{}", log_loc_label(loc)),
                    info.map(|(o, c, i)| (o, c, i)),
                )
            } else {
                let loc = crate::state::loc_from_key(src.key)
                    .map(|(l, _)| l)
                    .unwrap_or(Loc::Birmingham);
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
fn beer_source_flipped(state: &GameState, src: &BeerSource) -> bool {
    match src.kind {
        BeerSourceKind::Merchant => false,
        _ => {
            if let Some(fi) = src.farm_idx {
                state.farm_tiles[fi].as_ref().map_or(false, |t| t.flipped)
            } else {
                state.city_tiles[src.key]
                    .as_ref()
                    .map_or(false, |t| t.flipped)
            }
        }
    }
}

pub fn execute_network_double(
    state: &mut GameState,
    pid: usize,
    conn1: usize,
    conn2: usize,
    coal1: CoalSource,
    coal2: CoalSource,
    beer: BeerSource,
    card_index: usize,
) -> Result<String, String> {
    require_card_index(state, pid, card_index)?;
    if state.era != Era::Rail {
        return Err("Double links are Rail Era only".into());
    }
    if state.players[pid].rail_links < 2 {
        return Err("Not enough rail links".into());
    }
    if conn1 == conn2 {
        return Err("Double links must use distinct connections".into());
    }
    let c1 = connections()
        .iter()
        .find(|c| c.id == conn1)
        .ok_or("Invalid first connection")?;
    let c2 = connections()
        .iter()
        .find(|c| c.id == conn2)
        .ok_or("Invalid second connection")?;
    if !c1.rail || !c2.rail {
        return Err("Both connections must be rail-enabled".into());
    }
    if state.links[conn1].is_some() || state.links[conn2].is_some() {
        return Err("Connection already built".into());
    }

    let has_no_presence = !player_has_presence(state, pid);

    // All rail mutations (link placements + coal consumption) run inside a
    // transaction: any validation error rolls the board back on return.
    let mut tx = RailTx::new(state);

    // Link 1 adjacency
    if !has_no_presence
        && !is_in_network(tx.state_ref(), pid, c1.a)
        && !is_in_network(tx.state_ref(), pid, c1.b)
    {
        return Err("First link is not adjacent to your network".into());
    }
    tx.place_link(conn1, pid);

    // The player must explicitly choose a legal coal source for the first link.
    let legal_coal1 = coal_options_for_connection(tx.state_ref(), c1, 1);
    if !legal_coal1.iter().any(|opt| opt.as_slice() == [coal1]) {
        return Err("Chosen coal source is not legal for the first link".into());
    }
    let mut total = RAIL_DOUBLE_LINK_COST
        + source_purchase_cost(&[coal1], 1).expect("chosen coal source must be present");
    tx.consume_coal(coal1);

    // Link 2 adjacency (with link1 on board)
    if !is_in_network(tx.state_ref(), pid, c2.a) && !is_in_network(tx.state_ref(), pid, c2.b) {
        return Err("Second link is not adjacent to your network".into());
    }
    tx.place_link(conn2, pid);

    // The player must explicitly choose a legal coal source for the second link.
    let legal_coal2 = coal_options_for_connection(tx.state_ref(), c2, 1);
    if !legal_coal2.iter().any(|opt| opt.as_slice() == [coal2]) {
        return Err("Chosen coal source is not legal for the second link".into());
    }
    total += source_purchase_cost(&[coal2], 1).expect("chosen coal source must be present");
    tx.consume_coal(coal2);

    // The player must explicitly choose a legal beer source for the second link.
    let legal_beers = beer_sources_for_link(tx.state_ref(), pid, c2);
    if !legal_beers.contains(&beer) {
        return Err("Chosen beer source is not legal for the second link".into());
    }

    if total > tx.state_ref().players[pid].money {
        return Err("Cannot afford both links".into());
    }

    // Capture pre-consumption state for the log.
    let coal1_desc = tx.describe_coal(0);
    let coal2_desc = tx.describe_coal(1);
    let (beer_label, beer_before) = describe_beer_before(tx.state_ref(), &beer);

    // Commit: keep the placed links and consumed coal.
    let state = tx.commit();

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
