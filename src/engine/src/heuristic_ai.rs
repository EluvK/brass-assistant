//! Heuristic AI baseline: translated from reference/npow-brass-birmingham/js/aiPlayer.js
//!
//! Stateless greedy evaluator. Reads GameState/legal moves (never mutates the
//! state beyond what action generation needs) and returns the Move with the
//! highest VP-equivalent score. One-ply, no search.

use crate::data::{Era, IndustryType};
use crate::engine::{advance_turn, TurnResult};
use crate::graph::{
    connected_locations, find_beer_sources, find_coal_sources, find_iron_sources, is_in_network,
};
use crate::map::{city_slots, connections, Loc, ALL_LOCATIONS, CITY_COUNT};
use crate::rules::{
    can_develop, can_scout, execute_sell, get_valid_build_targets, get_valid_network_targets,
    get_valid_second_rail_links, get_valid_sell_targets, valid_build_cards, BuildTarget, Move,
};
use crate::state::{city_slot_offsets, Card, GameState, PendingBonus};
use rand::Rng;

// Scoring weights (from aiPlayer.js).
const VP_WEIGHT: f64 = 1.0;
const BASE_MONEY_WEIGHT: f64 = 0.12;
const BASE_INCOME_WEIGHT: f64 = 0.35;
const FLEX_WEIGHT: f64 = 0.8;

// Hard policy constraints (temporary strategic guardrails requested by user).
const BAN_BUILD_LV1_BREWERY: bool = true;
const BAN_DEVELOP_IRON_LV2_PLUS: bool = true;

fn develop_guardrail_penalty(ind: IndustryType, level: u8) -> f64 {
    if level < 2 {
        return 0.0;
    }
    match ind {
        IndustryType::Brewery => 1.8 + 0.2 * (level.saturating_sub(2)) as f64,
        IndustryType::CoalMine => 1.5 + 0.2 * (level.saturating_sub(2)) as f64,
        _ => 0.0,
    }
}

pub struct Decision {
    pub mv: Move,
    pub score: f64,
}

pub fn choose_action<R: Rng + Clone>(state: &mut GameState<R>) -> Decision {
    let mut candidates = candidate_actions(state);
    candidates.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());

    for cand in candidates {
        let mut sim = state.clone();
        if crate::rules::apply_move(&mut sim, &cand.mv).is_ok() {
            return cand;
        }
    }

    pass_decision(state)
}

/// Best move per action type (the 1-ply candidate set). Used both by
/// `choose_action` and by the 2-ply lookahead in search_ai.
pub fn candidate_actions<R: Rng + Clone>(state: &mut GameState<R>) -> Vec<Decision> {
    candidate_actions_k(state, 1)
}

/// Top-K candidates per action type, for MCTS to get a wider prior.
/// Build and Network get up to `k` candidates each; other action types keep
/// their single best (develop/sell/loan/scout/pass).
pub fn candidate_actions_k<R: Rng + Clone>(state: &mut GameState<R>, k: usize) -> Vec<Decision> {
    let pid = state.current_player_id();

    if let Some(PendingBonus::FreeDevelop { player_id, count }) = state.pending_bonus {
        if player_id != pid {
            return Vec::new();
        }
        return score_pending_free_develop(state, pid, count)
            .into_iter()
            .collect();
    }

    let mut out = Vec::new();

    for d in score_top_builds(state, pid, k) {
        if d.score != f64::NEG_INFINITY {
            out.push(d);
        }
    }
    out.extend(score_top_networks(state, pid, k));
    if let Some(d) = score_best_network_double(state, pid) {
        out.push(d);
    }
    if let Some(d) = score_develop_plan(state, pid) {
        out.push(d);
    }
    if let Some(d) = score_sell_plan(state, pid) {
        out.push(d);
    }
    if let Some(d) = score_loan_result(state, pid) {
        out.push(d);
    }
    if let Some(d) = score_scout_plan(state, pid) {
        out.push(d);
    }
    if let Some(d) = score_pass_result(state, pid) {
        out.push(d);
    }

    out
}

/// Fallback "pass" decision (safe even on an empty hand).
pub fn pass_decision(state: &GameState<impl Rng>) -> Decision {
    let card_index = pick_any_card(state, state.current_player_id()).unwrap_or(0);
    Decision {
        mv: Move::Pass { card_index },
        score: -0.5,
    }
}

fn score_pending_free_develop(state: &GameState<impl Rng>, pid: usize, count: u8) -> Option<Decision> {
    let mut types = state.players[pid].developable_types();
    if BAN_DEVELOP_IRON_LV2_PLUS {
        types.retain(|(ind, tile)| !(*ind == IndustryType::IronWorks && tile.level >= 2));
    }
    if types.is_empty() {
        return None;
    }

    let mut scored: Vec<(IndustryType, f64)> = types
        .into_iter()
        .map(|(ind, tile)| {
            let unlocked = state.players[pid].tile_after(ind, 1);
            let mut v = if tile.rail_era { 0.35 } else { 0.12 };
            v += 0.18 * tile.level as f64;
            if let Some(ut) = unlocked {
                if ut.rail_era {
                    v += 0.25;
                }
            }
            if ind == IndustryType::Brewery {
                v += 0.55;
            }
            if state.era == Era::Canal {
                v += 0.15;
            }
            if player_has_buildable_card(state, pid, ind) {
                v += 0.3;
            }
            v -= develop_guardrail_penalty(ind, tile.level);
            (ind, v)
        })
        .collect();
    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    let first = scored[0].0;
    let second = if count >= 2 {
        let rem1 = state.players[pid].remaining_count(first);
        scored[1..]
            .iter()
            .find(|s| s.0 != first || rem1 > 1)
            .map(|s| s.0)
    } else {
        None
    };

    let second_score = second
        .and_then(|ind| scored.iter().find(|s| s.0 == ind).map(|s| s.1 * 0.4))
        .unwrap_or(0.0);

    Some(Decision {
        mv: Move::ResolveFreeDevelop {
            ind1: first,
            ind2: second,
        },
        score: scored[0].1 + second_score,
    })
}

/// Position-value estimator for player `pid`, used as the MCTS leaf evaluator.
/// Combines accumulated VP + income stream + cash with the player's best
/// available move score (1-ply), which reflects actionable potential better
/// than a pure board snapshot.
pub(crate) fn evaluate_position(state: &GameState<impl Rng>, pid: usize) -> f64 {
    let p = &state.players[pid];
    let mut value = 0.0;

    value += p.vp as f64 * VP_WEIGHT;
    value += p.money as f64 * money_weight(state);
    value += p.income_level() as f64 * income_weight(state) * 3.0;

    // Own tiles: flipped score full VP; unflipped a small fraction.
    for tile in state.city_tiles.iter().flatten() {
        if tile.player != pid {
            continue;
        }
        let vp = tile.def.vp as f64;
        value += if tile.flipped { vp } else { vp * 0.25 };
    }
    for tile in state.farm_tiles.iter().flatten() {
        if tile.player != pid {
            continue;
        }
        let vp = tile.def.vp as f64;
        value += if tile.flipped { vp } else { vp * 0.25 };
    }

    // Links: each owned link scores icons in adjacent locations at era end.
    let links = state
        .links
        .iter()
        .flatten()
        .filter(|l| l.player == pid)
        .count();
    value += links as f64 * 2.0;

    // Hand flexibility.
    let flex: f64 = p.hand.iter().map(|c| card_usefulness(state, pid, c)).sum();
    value += flex * FLEX_WEIGHT;

    value
}

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

pub(crate) fn estimate_rounds_remaining(state: &GameState<impl Rng>) -> f64 {
    // Unplayed cards = deck + all hands. Each action spends one card, and a
    // round is actions_per_turn actions per player. The era ends only when the
    // deck AND every hand are exhausted, so using deck length alone badly
    // underestimates late-era rounds (deck empties before hands do).
    let deck_cards = state.deck.len() as f64;
    let hand_cards: f64 = state.players.iter().map(|p| p.hand.len() as f64).sum();
    let total = deck_cards + hand_cards;
    let actions_per_round = state.player_count() as f64 * state.actions_per_turn as f64;
    (total / actions_per_round).clamp(1.0, 8.0)
}

pub(crate) fn is_late_rail_half(state: &GameState<impl Rng>) -> bool {
    state.era == Era::Rail && estimate_rounds_remaining(state) <= 4.0
}

fn money_weight(state: &GameState<impl Rng>) -> f64 {
    if is_late_rail_half(state) {
        0.2
    } else if state.era == Era::Canal {
        BASE_MONEY_WEIGHT * 0.55
    } else {
        BASE_MONEY_WEIGHT * 0.8
    }
}

pub(crate) fn income_weight(state: &GameState<impl Rng>) -> f64 {
    if is_late_rail_half(state) {
        return 0.0;
    }
    let rounds = estimate_rounds_remaining(state);
    if state.era == Era::Canal {
        // Canal economy is income-driven: recovery runway is long and powers
        // the whole rail transition.
        BASE_INCOME_WEIGHT * (2.2 + 0.6 * (rounds / 8.0).clamp(0.0, 1.0))
    } else {
        // Rail first half: still meaningful, but less than canal.
        BASE_INCOME_WEIGHT * (1.2 + 0.5 * (rounds / 8.0).clamp(0.0, 1.0))
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn vp_equivalent(
    state: &GameState<impl Rng>,
    vp: f64,
    income: f64,
    money: f64,
    flex: f64,
) -> f64 {
    vp * VP_WEIGHT + income * income_weight(state) + money * money_weight(state) + flex * FLEX_WEIGHT
}

// ---------------------------------------------------------------------------
// BUILD
// ---------------------------------------------------------------------------

/// How much cash a newly built coal/iron works would earn by selling its cubes
/// into the market, and whether that sale empties the tile (flipping it for
/// income). Mirrors `GameState::auto_sell_to_market`: cubes fill market spaces
/// from the cheapest up, and stop when the market is full.
struct MarketSale {
    cash: f64,
    sold: u8,
    total: u8,
    flips: bool,
}

fn simulate_market_sale(state: &GameState<impl Rng>, is_coal: bool, cubes: u8) -> MarketSale {
    let (prices, market) = if is_coal {
        (crate::map::COAL_MARKET_PRICES.as_slice(), state.coal_market)
    } else {
        (crate::map::IRON_MARKET_PRICES.as_slice(), state.iron_market)
    };
    let mut cash = 0.0;
    let mut m = market;
    let mut sold = 0u8;
    while sold < cubes && m < prices.len() {
        // Filling the cheapest open space first (same as auto_sell_to_market).
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

/// How much sell beer remains for the tile being evaluated in `loc`'s beer
/// pool. The new tile only flips if there is beer LEFT OVER after the player's
/// already-unflipped sellable tiles in that pool take their share — the new
/// tile is at the back of the queue. 1.0 = beer to spare; ~0 = every existing
/// tile already claims the beer, so the marginal tile can't sell this era.
fn sell_saturation(state: &GameState<impl Rng>, pid: usize, loc: Loc) -> f64 {
    let connected = connected_locations(state, loc);
    let supply = find_beer_sources(state, loc, pid, &[]).len() as f64
        + state
            .merchants
            .iter()
            .filter(|mt| mt.has_beer && connected.contains(&mt.loc))
            .count() as f64;
    let mut existing_demand = 0.0;
    for (i, l) in ALL_LOCATIONS[..CITY_COUNT].iter().enumerate() {
        for slot in 0..city_slots(*l).len() {
            let key = city_slot_offsets()[i] + slot;
            if let Some(t) = state.city_tiles[key].as_ref() {
                if t.player == pid && !t.flipped && t.ind.is_sellable() {
                    if *l == loc || connected.contains(l) {
                        existing_demand += t.def.beers_to_sell.unwrap_or(1) as f64;
                    }
                }
            }
        }
    }
    let remaining = (supply - existing_demand).max(0.0);
    remaining.clamp(0.0, 1.0)
}

fn estimate_flip_probability(
    state: &GameState<impl Rng>,
    pid: usize,
    ind: IndustryType,
    city_id: Loc,
) -> f64 {    let base = if matches!(ind, IndustryType::CoalMine | IndustryType::IronWorks) {
        // Resources flip when consumed (building/networking) or when a build
        // immediately sells its cubes into the market (iron always; coal if
        // connected to a merchant). Flip odds follow MARKET HUNGER + ERA DEMAND:
        //   * sparse market => opponents must consume this resource => flips fast;
        //   * rail era is coal-hungry (every build/link needs coal) => near-guaranteed;
        //   * a build that sells out on placement flips immediately.
        let is_coal = ind == IndustryType::CoalMine;
        let cubes = state
            .players
            .get(pid)
            .and_then(|p| p.next_tile(ind))
            .map(|t| t.resource_cubes)
            .unwrap_or(1);
        let capacity = if is_coal { 14.0 } else { 10.0 };
        let market = if is_coal {
            state.coal_market as f64
        } else {
            state.iron_market as f64
        };
        let scarcity = ((capacity - market) / capacity).clamp(0.0, 1.0);
        let connected = connected_locations(state, city_id);
        let can_sell = ind == IndustryType::IronWorks
            || connected.iter().any(|l| l.is_merchant());
        // Coal flips by selling to a merchant market OR being consumed by the
        // table. A coal mine with NO merchant connection (an "island mine")
        // can't sell on build and — especially in the canal era — nobody will
        // build a link out to it, so it sits unflipped and vanishes at era end.
        // Humans only build connected coal (build their own link, piggyback a
        // neighbor's, or wild-city teleport in). Island coal is a bad move.
        if is_coal && !can_sell {
            if state.era == Era::Canal {
                // Canal coal that cannot sell on placement is still weak, but
                // less so when market coal is very expensive (6/7/8): demand
                // pressure means table consumption is likely soon.
                let heat = ((state.coal_price() as f64 - 5.0) / 3.0).clamp(0.0, 1.0);
                (0.12 + 0.18 * heat).min(0.4)
            } else {
                // Rail era: even without merchant auto-sell, coal demand is
                // table-wide and urgent; isolated mines are consumed quickly
                // once the rail graph densifies.
                let heat = ((state.coal_price() as f64 - 4.0) / 4.0).clamp(0.0, 1.0);
                (0.6 + 0.25 * heat).min(0.9)
            }
        } else {
            let sale = simulate_market_sale(state, is_coal, cubes);
            if can_sell && sale.flips {
                0.9
            } else {
                // Relies on consumption by the table. Coal demand is enormous
                // in the rail era (all builds and links consume it); iron
                // demand is steadier. Market scarcity adds urgency on top.
                let era_demand = if is_coal {
                    if state.era == Era::Rail { 0.85 } else { 0.55 }
                } else {
                    if state.era == Era::Rail { 0.5 } else { 0.4 }
                };
                (era_demand + 0.35 * scarcity).min(0.9)
            }
        }
    } else if ind == IndustryType::Brewery {
        // Beer barrels are consumed by sells AND rail links; a brewery flips
        // only if there is actual demand for its beer. In the canal era beer
        // has no use beyond fueling your own sells, so a brewery with no
        // sellable tiles to feed (or surplus barrels beyond demand) is waste:
        // it will sit unflipped. In the rail era double-rails also eat beer,
        // so demand gets a small floor.
        let next_cubes = state
            .players
            .get(pid)
            .and_then(|p| p.next_tile(ind))
            .map(|t| t.resource_cubes as usize)
            .unwrap_or(1);
        let demand = sellable_beer_demand(state, pid) as f64
            + if state.era == Era::Rail { 1.0 } else { 0.0 };
        let barrels = (owned_beer_barrels(state, pid) + next_cubes) as f64;
        if demand <= 0.5 && state.era == Era::Canal {
            0.25
        } else if barrels > demand {
            0.45
        } else {
            0.7
        }
    } else {
        // Sellable tiles ONLY flip when sold: that requires a reachable
        // merchant that accepts this industry AND beer to fuel the sale. A
        // tile with neither path is nearly worthless unflipped — keep the
        // base honestly low so expensive builds don't get credit for a
        // "double-score" they can never realize.
        let mut b = 0.12f64;
        let connected = connected_locations(state, city_id);
        let has_reachable_merchant = state.merchants.iter().any(|mt| {
            connected.contains(&mt.loc) && mt.accepts(ind)
        });
        if has_reachable_merchant {
            // A merchant is only worth real flip credit if there is beer to
            // fuel the sale; without beer it's a dead end this era.
            let beer_ok = find_beer_sources(state, city_id, pid, &[]).len() > 0
                || beer_barrels_reachable(state, city_id);
            if beer_ok {
                b += 0.6;
            } else {
                b += 0.1;
            }
        }
        // Adjacent unbuilt links mean a merchant can still be connected later.
        let open_links = count_new_unbuilt_neighbor_connections(state, city_id);
        if open_links > 0 {
            b += 0.1;
        }
        if estimate_rounds_remaining(state) < 2.0 {
            b -= 0.2;
        }
        // Beer-capacity gate: if the player already has more unflipped
        // sellable tiles in this beer pool than beer can fuel, this marginal
        // tile likely never sells. Scale the flip odds down accordingly.
        b *= sell_saturation(state, pid, city_id);
        b
    };
    base.clamp(0.05, 1.0)
}

/// True if the player holds a card that can build `ind` (industry or wild).
fn player_has_buildable_card(state: &GameState<impl Rng>, pid: usize, ind: IndustryType) -> bool {
    let player = &state.players[pid];
    player.hand.iter().any(|c| {
        matches!(c, Card::Industry { .. } if c.is_industry(ind))
            || c.ctype() == crate::data::CardType::WildIndustry
    })
}

fn player_owns_link_touching(state: &GameState<impl Rng>, pid: usize, city_id: Loc) -> bool {
    connections().iter().any(|c| {
        if let Some(l) = &state.links[c.id] {
            l.player == pid && (c.a == city_id || c.b == city_id)
        } else {
            false
        }
    })
}

fn count_new_unbuilt_neighbor_connections(state: &GameState<impl Rng>, city_id: Loc) -> usize {
    connections()
        .iter()
        .filter(|c| state.links[c.id].is_none() && (c.a == city_id || c.b == city_id))
        .count()
}

fn own_brewery_stats(state: &GameState<impl Rng>, pid: usize) -> (usize, usize) {
    let mut barrels = 0usize;
    let mut flipped = 0usize;
    for tile in state.city_tiles.iter().flatten() {
        if tile.player == pid && tile.ind == IndustryType::Brewery {
            barrels += tile.resource_cubes as usize;
            if tile.flipped {
                flipped += 1;
            }
        }
    }
    for tile in state.farm_tiles.iter().flatten() {
        if tile.player == pid {
            barrels += tile.resource_cubes as usize;
            if tile.flipped {
                flipped += 1;
            }
        }
    }
    (barrels, flipped)
}

/// Can `pid` reach a merchant barrel (any accepted industry) from `loc`?
fn beer_barrels_reachable(state: &GameState<impl Rng>, loc: Loc) -> bool {    let connected = connected_locations(state, loc);
    state
        .merchants
        .iter()
        .any(|mt| mt.has_beer && connected.contains(&mt.loc))
}

/// Number of beer barrels the player currently holds (unflipped breweries).
fn owned_beer_barrels(state: &GameState<impl Rng>, pid: usize) -> usize {
    state
        .city_tiles
        .iter()
        .flatten()
        .filter(|t| t.player == pid && t.ind == IndustryType::Brewery && !t.flipped)
        .map(|t| t.resource_cubes as usize)
        .sum()
}

/// Total beer barrels needed to sell ALL of the player's unflipped sellable
/// tiles. Extra barrels beyond this (plus a rail-network buffer) are wasted.
fn sellable_beer_demand(state: &GameState<impl Rng>, pid: usize) -> usize {
    state
        .city_tiles
        .iter()
        .flatten()
        .filter(|t| t.player == pid && !t.flipped && t.ind.is_sellable())
        .map(|t| t.def.beers_to_sell.unwrap_or(0) as usize)
        .sum()
}

/// Fraction of the coal/iron needed by a build that can come from free board
/// sources (own or opponent mines/works) rather than the paid market. Consuming
/// opponent resources is "free riding": it costs nothing extra and flips their
/// tile, keeping the shared resource pool cheap for everyone. A high ratio
/// means the build is cheap to run; a low one forces paying market price.
fn resource_source_ratio(state: &GameState<impl Rng>, cand: &BuildTarget) -> f64 {
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
        find_iron_sources(state)
            .iter()
            .filter(|s| s.free)
            .count() as f64
    } else {
        f64::MAX
    };
    let mut free_available = 0.0;
    free_available += free_coal.min(cand.cost_coal as f64);
    free_available += free_iron.min(cand.cost_iron as f64);
    (free_available / needed).clamp(0.0, 1.0)
}

/// Can `pid` sell a tile of this industry at `loc` right now (reachable
/// merchant accepting it + beer available to fuel the sale)?
fn immediate_sellable(state: &GameState<impl Rng>, pid: usize, ind: IndustryType, loc: Loc) -> bool {
    if !ind.is_sellable() {
        return false;
    }
    let connected = connected_locations(state, loc);
    let merchant_ok = state
        .merchants
        .iter()
        .any(|mt| connected.contains(&mt.loc) && mt.accepts(ind));
    if !merchant_ok {
        return false;
    }
    find_beer_sources(state, loc, pid, &[]).len() > 0 || beer_barrels_reachable(state, loc)
}

fn score_build_candidate(
    state: &GameState<impl Rng>,
    pid: usize,
    cand: &BuildTarget,
) -> f64 {
    let tile = state.players[pid].next_tile(cand.ind);
    let Some(tile) = tile else { return f64::NEG_INFINITY };

    if BAN_BUILD_LV1_BREWERY && cand.ind == IndustryType::Brewery && tile.level == 1 {
        return f64::NEG_INFINITY;
    }

    let player = &state.players[pid];
    let cash = player.money as f64;
    let cost = cand.cost_total as f64;

    // Cash is the hard constraint that drives the early economy. If we cannot
    // afford the build, heavily discount it (still show a path via loan, but
    // not a first choice). If it consumes nearly all cash, require that it pay
    // off in income soon.
    if cost > cash {
        let unaffordable = -(cost - cash) * 0.3;
        return unaffordable;
    }

    let flip_prob = estimate_flip_probability(state, pid, cand.ind, cand.loc);
    let owns_adjacent_link = player_owns_link_touching(state, pid, cand.loc);
    let link_self_value = if owns_adjacent_link {
        tile.link_vp as f64 * flip_prob * 0.5
    } else {
        0.0
    };
    let is_resource = matches!(cand.ind, IndustryType::CoalMine | IndustryType::IronWorks);
    let resource_self_sufficiency = if is_resource {
        0.15 * tile.resource_cubes as f64
    } else {
        0.0
    };
    // Building a coal/iron works can immediately sell its production to the
    // market for cash (iron always; coal if connected to a merchant). The
    // value of such a build is dominated by MARKET HUNGER:
    //   * a sparse market (few cubes, high prices) rewards the fill richly —
    //     big cash-back, immediate flip for income, and service to the table;
    //   * a full market means cubes can't fit: they sit unflipped, wastefully
    //     waiting on opponents' consumption. Leftover cubes are a penalty.
    // "If everything sells, it's a great move; the more that stays on the
    // tile, the worse (unless spent on your very next action)."
    let market_adjust = if is_resource {
        let connected = connected_locations(state, cand.loc);
        let market_ok = cand.ind == IndustryType::IronWorks
            || connected.iter().any(|l| l.is_merchant());
        let is_coal = cand.ind == IndustryType::CoalMine;
        // Coal demand is far higher than iron (every rail build/link eats it);
        // iron demand is steadier and the market fills faster. Discount iron
        // scarcity so we don't over-produce iron that nobody will consume.
        let scarcity = if is_coal {
            (14 - state.coal_market) as f64 / 14.0
        } else {
            0.6 * (10 - state.iron_market) as f64 / 10.0
        };
        if market_ok {
            let sale = simulate_market_sale(state, is_coal, tile.resource_cubes);
            let sell_value = sale.cash * money_weight(state);
            let cash_back_bonus = if sale.cash > 0.0 {
                sale.cash * 0.11 + if sale.flips { 0.6 } else { 0.0 }
            } else {
                0.0
            };
            // Coal market spike prior: when buy price is 6/7/8, placing a
            // connected coal mine that can auto-sell is typically a top-tier
            // tempo play (cash now + flip income + future board demand).
            let coal_spike_bonus = if is_coal && sale.sold > 0 {
                let buy_price = state.coal_price() as f64;
                let heat = ((buy_price - 5.0) / 3.0).clamp(0.0, 1.0);
                let sold_factor = sale.sold as f64;
                let era_mult = if state.era == Era::Canal { 1.25 } else { 1.0 };
                heat * sold_factor * 1.9 * era_mult
            } else {
                0.0
            };
            let scarcity_value = scarcity * (1.0 + sale.sold as f64) * 0.6;
            let leftover_penalty = if is_coal && state.era == Era::Rail {
                // In the rail era, coal is consumed by nearly every build and
                // rail link — a full market right now doesn't mean the cubes
                // won't be eaten soon. Don't punish unsold coal hard.
                0.0
            } else {
                (sale.total - sale.sold) as f64 * 0.5
            };
            sell_value + cash_back_bonus + coal_spike_bonus + scarcity_value - leftover_penalty
        } else {
            // Not merchant-connected: cubes can't be sold today. For an
            // ISLAND COAL MINE this is a strongly negative move in the canal
            // era (tiles vanish at era end, nobody helps consume it) and merely
            // speculative in the rail era (links will be built out). Iron needs
            // no connection to be consumed, so it keeps a market-value floor.
            if is_coal {
                if state.era == Era::Canal {
                    -0.5
                } else {
                    // Rail coal without merchant reach is still a strong play
                    // under scarcity: it backfills the table's fuel demand,
                    // gets consumed fast, and often flips for income soon.
                    scarcity * (1.2 + 0.25 * tile.resource_cubes as f64)
                }
            } else {
                scarcity * 1.2
            }
        }
    } else {
        0.0
    };
    let network_expansion =
        0.1 * count_new_unbuilt_neighbor_connections(state, cand.loc) as f64;

    // Emergency coal supply prior (rail): if coal market is near empty, any
    // legal coal mine is strategically premium because the whole table must pay
    // expensive market coal otherwise. This is independent of immediate
    // merchant auto-sell.
    let rail_coal_shortage_bonus = if cand.ind == IndustryType::CoalMine && state.era == Era::Rail {
        let shortage = (1.0 - (state.coal_market as f64 / 14.0)).clamp(0.0, 1.0);
        let level_factor = 1.0 + 0.2 * (tile.level.saturating_sub(1)) as f64;
        let cubes_factor = 0.7 + 0.15 * tile.resource_cubes as f64;
        shortage * level_factor * cubes_factor * 3.0
    } else {
        0.0
    };

    // Cost efficiency: the build must be worth its price tag. Cheap builds
    // (coal £5, iron £7, brewery £5) get a relative edge early.
    let cost_efficiency = if cost > 0.0 {
        let eff = (tile.income as f64 + tile.vp as f64) / cost;
        eff.min(2.0)
    } else {
        0.0
    };

    // Beer economy: sellable tiles are only worth their VP if we can actually
    // sell them (merchant reachable + beer available). Breweries are worth a
    // premium when we already have unflipped sellable tiles waiting on beer.
    let sellable = cand.ind.is_sellable();
    let mut beer_bonus = 0.0;
    if sellable {
        // A reachable merchant that accepts this industry raises flip odds.
        let connected = connected_locations(state, cand.loc);
        let has_merchant = state.merchants.iter().any(|mt| {
            connected.contains(&mt.loc) && mt.accepts(cand.ind)
        });
        if has_merchant {
            beer_bonus += 0.6;
        }
        // Beer scarcity: if we hold no beer source and can't reach one with a
        // barrel, the sellable tile is near-worthless unflipped. (Don't punish
        // too hard: a brewery can still be added later.)
        let beer_ok = has_merchant
            && (find_beer_sources(state, cand.loc, pid, &[]).len()
                >= tile.beers_to_sell.unwrap_or(0) as usize
                || beer_barrels_reachable(state, cand.loc));
        if beer_ok {
            // We have the beer AND the merchant: this is a guaranteed sellable.
            beer_bonus += 0.8;
        } else {
            beer_bonus -= 0.3;
        }
    } else if cand.ind == IndustryType::Brewery {
        // Breweries feed sells AND rail links. The winning line develops
        // level-1 away and builds level-2/3/4 (they survive to rail). Match
        // output to need: barrels should cover our unflipped sellable tiles'
        // beer demand plus a small rail-network buffer. Building beyond that is
        // wasted, and level-1 breweries vanish at era end so they're weak.
        let barrels = owned_beer_barrels(state, pid) as f64
            + tile.resource_cubes as f64; // this brewery's contribution
        let demand = sellable_beer_demand(state, pid) as f64
            + if state.era == Era::Rail { 1.0 } else { 0.5 };
        let surplus = (barrels - demand).max(0.0);
        let sat = -1.0 * surplus;
        let sell_support = if demand > 0.0 { 0.4 } else { 0.1 };
        let rail_beer_value = if state.era == Era::Rail {
            0.4 * tile.resource_cubes as f64
        } else {
            0.0
        };
        let level_bonus = if tile.level >= 2 { 0.3 } else { 0.0 };
        beer_bonus += sell_support + rail_beer_value + level_bonus + sat;

        // Canal-era level-1 brewery is often a weak tempo move: it disappears
        // at era end and does not support rail links. Bias toward develop-first
        // (brewery 2+) unless there is clear immediate sell demand.
        if state.era == Era::Canal && tile.level == 1 {
            let rounds_left = estimate_rounds_remaining(state);
            let demand_now = sellable_beer_demand(state, pid) as f64;
            let no_real_demand = if demand_now < 1.0 { 1.0 } else { 0.0 };
            let long_horizon = if rounds_left > 3.0 { 1.0 } else { 0.0 };
            beer_bonus -= 1.0 + 0.7 * no_real_demand + 0.4 * long_horizon;
        }
    }

    // Rail-era tiles score their flipped VP at BOTH era ends (canal survives
    // into rail). In the rail era the double-count is fully earned (2x); in the
    // canal era it's potential — only realized if the tile survives — so weight
    // it lightly (1.2x) to avoid pushing expensive rail tiles into the cash-
    // strapped canal economy.
    let double_vp = if tile.rail_era {
        if state.era == Era::Rail { 2.0 } else { 1.1 }
    } else {
        1.0
    };

    // Level-1 tiles vanish at canal-era end: their build should be slightly
    // discounted (they're stepping stones toward level-2+ that double-score).
    // Level-2+ tiles get a modest canal-era weight boost for the cross-era win.
    let level_adjust = if tile.level == 1 {
        -0.4
    } else if tile.level >= 2 && state.era == Era::Canal {
        0.5
    } else {
        0.0
    };

    // "Free-riding" efficiency: if the build's coal/iron can come from board
    // mines/works (own or opponents) instead of the paid market, the action is
    // cheaper and faster. Reward builds on an established resource pool.
    let resource_ratio = resource_source_ratio(state, cand);
    let interaction_bonus = (resource_ratio - 0.5).max(0.0) * 0.8;

    let mut score = vp_equivalent(
        state,
        tile.vp as f64 * flip_prob * double_vp + link_self_value,
        tile.income as f64 * flip_prob,
        -(cand.cost_total as f64),
        0.0,
    ) + resource_self_sufficiency
        + network_expansion
        + rail_coal_shortage_bonus
        + beer_bonus
        + cost_efficiency
        + market_adjust
        + level_adjust
        + interaction_bonus;

    if state.era == Era::Canal {
        if tile.level >= 2 {
            // The canal cross-era boost (rail tiles double-score at both era
            // ends) is only real if the tile will actually flip. A sellable
            // tile with no merchant+beer path can't be sold, so its double-
            // score promise is hollow — scale the boost down (or off) when
            // flip odds are low. `flip_prob` already encodes merchant/beer
            // reachability for sellable tiles, market hunger for coal/iron,
            // and consumption pressure for breweries.
            let boost = if flip_prob >= 0.6 {
                2.0
            } else if flip_prob >= 0.35 {
                1.3
            } else {
                1.0
            };
            score *= boost;
        } else {
            // Canal level-1 board presence is often disposable tempo only; keep
            // it legal but clearly behind develop->level-2 lines.
            score -= 1.8;
        }
    }

    // Doomed-build guardrail: a canal-only (level-1) tile built so late in the
    // canal era that it cannot flip before era end is removed at the
    // transition without ever scoring — a pure cash sink (e.g. a £21 pottery
    // built in round 7-8 that nobody can sell in time). Heavily penalize
    // unless it can be sold immediately this turn. This also propagates into
    // the loan scorer's `best_affordable_build_score`, so it stops loaning
    // just to fund such a build.
    if state.era == Era::Canal && !tile.rail_era {
        let rounds_left = estimate_rounds_remaining(state);
        if rounds_left <= 2.0 && !immediate_sellable(state, pid, cand.ind, cand.loc) {
            let urgency = (2.0 - rounds_left).max(0.0);
            score -= 5.0 + urgency * 4.0;
        }
    }

    score
}

fn pick_build_card(state: &GameState<impl Rng>, pid: usize, cand: &BuildTarget) -> Option<usize> {
    let player = &state.players[pid];
    let indices = valid_build_cards(state, player, pid, cand.loc, cand.ind);
    // Prefer a non-wild matching card over a wild one.
    indices.into_iter().find(|&i| {
        !matches!(
            player.hand[i].ctype(),
            crate::data::CardType::WildLocation | crate::data::CardType::WildIndustry
        )
    })
}

/// Top-K build candidates by 1-ply score. Used by MCTS to get a wider prior.
pub(crate) fn score_top_builds(state: &mut GameState<impl Rng>, pid: usize, k: usize) -> Vec<Decision> {
    let targets = get_valid_build_targets(state, pid);
    let mut scored: Vec<(BuildTarget, f64)> = targets
        .into_iter()
        .map(|t| {
            let s = score_build_candidate(state, pid, &t);
            (t, s)
        })
        .collect();
    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    scored.truncate(k);
    let mut out = Vec::new();
    for (cand, score) in scored {
        if let Some(card_index) = pick_build_card(state, pid, &cand) {
            let coal_needed = cand.cost_coal as usize;
            let iron_needed = cand.cost_iron as usize;
            let coal = crate::rules::coal_source_options(state, cand.loc, coal_needed)
                .into_iter()
                .next()
                .unwrap_or_default();
            let iron = crate::rules::iron_source_options(state, iron_needed)
                .into_iter()
                .next()
                .unwrap_or_default();
            out.push(Decision {
                mv: Move::Build {
                    loc: cand.loc,
                    slot_index: cand.slot_index,
                    ind: cand.ind,
                    coal,
                    iron,
                    card_index,
                },
                score,
            });
        }
    }
    out
}

// ---------------------------------------------------------------------------
// NETWORK
// ---------------------------------------------------------------------------

fn count_hand_cards_newly_in_network(
    state: &GameState<impl Rng>,
    pid: usize,
    cand_cities: &[Loc; 2],
) -> f64 {
    let player = &state.players[pid];
    let mut count = 0.0;
    for card in &player.hand {
        if let Card::Location(l) = card {
            if !is_in_network(state, pid, *l) && cand_cities.contains(l) {
                count += 1.0;
            }
        } else if let Card::Industry { .. } = card {
            count += 0.25;
        }
    }
    count
}

fn connects_to_new_merchant(_state: &GameState<impl Rng>, cand_cities: &[Loc; 2]) -> bool {
    cand_cities.iter().any(|c| c.is_merchant())
}

fn coal_effective_price(src: &crate::graph::CoalSource) -> i32 {
    if src.free { 0 } else { src.price as i32 }
}

fn cheapest_connection_coal_source(
    state: &GameState<impl Rng>,
    conn_id: usize,
) -> Option<crate::graph::CoalSource> {
    let conn = &connections()[conn_id];
    crate::rules::coal_options_for_connection(state, conn, 1)
        .into_iter()
        .flatten()
        .min_by_key(coal_effective_price)
}

fn estimated_connection_coal_cost(state: &GameState<impl Rng>, conn_id: usize) -> i32 {
    cheapest_connection_coal_source(state, conn_id)
        .map(|s| coal_effective_price(&s))
        .unwrap_or(crate::map::COAL_EMPTY_PRICE as i32)
}

/// Rail-era link VP potential: sum the top flipped VP a player could still
/// land in the two cities this link connects (vacant slots + remaining tiles),
/// weighted by whether they can actually place there. Dense industrial hubs
/// are worth racing for in the rail era.
fn link_vp_potential(state: &GameState<impl Rng>, pid: usize, cities: &[Loc; 2]) -> f64 {
    let mut total = 0.0f64;
    for loc in cities {
        if !loc.is_city() {
            continue;
        }
        let slots = city_slots(*loc);
        for (slot_idx, slot_types) in slots.iter().enumerate() {
            // Skip occupied slots (someone already built here).
            if let Some(k) = state.city_slot_key(*loc, slot_idx) {
                if state.city_tiles[k].is_some() {
                    continue;
                }
            }
            // Best remaining tile we could build in this slot.
            let mut best = 0.0f64;
            for t in *slot_types {
                if let Some(tile) = state.players[pid].next_tile(*t) {
                    let dvp = if tile.rail_era { 1.4 } else { 1.0 };
                    let v = tile.vp as f64 * dvp;
                    if v > best {
                        best = v;
                    }
                }
            }
            total += best;
        }
    }
    total
}

fn immediate_link_icons_at(state: &GameState<impl Rng>, loc: Loc) -> f64 {
    if loc.is_merchant() {
        return 2.0;
    }
    if loc.is_city() {
        let mut v = 0.0;
        for slot in 0..city_slots(loc).len() {
            if let Some(k) = state.city_slot_key(loc, slot) {
                if let Some(t) = &state.city_tiles[k] {
                    if t.flipped {
                        v += t.def.link_vp as f64;
                    }
                }
            }
        }
        return v;
    }
    if let Some(t) = state.farm_tile(loc) {
        if t.flipped {
            return t.def.link_vp as f64;
        }
    }
    0.0
}

fn future_link_node_potential(state: &GameState<impl Rng>, pid: usize, loc: Loc) -> f64 {
    if !loc.is_city() {
        return 0.0;
    }
    let mut total = 0.0;
    for (slot_idx, slot_types) in city_slots(loc).iter().enumerate() {
        if let Some(k) = state.city_slot_key(loc, slot_idx) {
            if state.city_tiles[k].is_some() {
                continue;
            }
        }
        let mut best = 0.0;
        for t in *slot_types {
            if let Some(tile) = state.players[pid].next_tile(*t) {
                // Approximate future node value for link scoring around this hub.
                let mut v = tile.link_vp as f64 + tile.vp as f64 * 0.2;
                if tile.level >= 2 && state.era == Era::Canal {
                    v += 0.6;
                }
                if player_has_buildable_card(state, pid, *t) {
                    v += 0.25;
                }
                if v > best {
                    best = v;
                }
            }
        }
        total += best;
    }
    total
}

fn potential_link_vps(
    state: &GameState<impl Rng>,
    pid: usize,
    conn_id: usize,
    cities: &[Loc; 2],
) -> f64 {
    let mut immediate = 0.0;
    let mut future = 0.0;
    for loc in cities {
        immediate += immediate_link_icons_at(state, *loc);
        future += future_link_node_potential(state, pid, *loc);
    }
    if let Some(via) = connections()[conn_id].via_farm {
        immediate += immediate_link_icons_at(state, via);
    }
    immediate + 0.7 * future
}

fn score_network_candidate(
    state: &GameState<impl Rng>,
    pid: usize,
    conn_id: usize,
    cost: i32,
    cities: &[Loc; 2],
    is_canal: bool,
) -> f64 {
    let access_gain = count_hand_cards_newly_in_network(state, pid, cities);
    let merchant_gain = if connects_to_new_merchant(state, cities) {
        1.5
    } else {
        0.0
    };
    // Rail-era "road multiplier": links into dense industrial hubs are the
    // highest-value plays once the rail era starts. Weight by the VP potential
    // of the cities they open up, scaled down to VP-equivalent units.
    let vp_potential = link_vp_potential(state, pid, cities);
    let potential_link_vp = potential_link_vps(state, pid, conn_id, cities);
    let hub_bonus = if state.era == Era::Rail {
        vp_potential * 0.08 + potential_link_vp * 0.28
    } else {
        vp_potential * 0.03 + potential_link_vp * 0.20
    };
    let player = &state.players[pid];
    let links_built = if is_canal {
        14 - player.canal_links
    } else {
        14 - player.rail_links
    };
    // Links are only as valuable as the tiles they connect. A player building
    // far more links than they have productive (flippable) tiles is over-
    // networking: money locked into potential that never pays off, which is
    // how broken boards end up with 8-14 links and 0 flips. Penalize the
    // surplus links proportionally so network "tempo" can't be spammed from a
    // position with nothing to connect.
    let productive = state
        .city_tiles
        .iter()
        .flatten()
        .filter(|t| t.player == pid)
        .count() as f64;
    let over_networking_penalty = if productive <= 0.0 {
        (links_built as f64 - 1.0).max(0.0) * 1.2
    } else {
        (links_built as f64 - 2.0 * productive - 1.0).max(0.0) * 0.6
    };
    let exploration_bonus = (1.6 - links_built as f64 * 0.3).max(0.0);

    vp_equivalent(
        state,
        access_gain + merchant_gain,
        0.0,
        -(cost as f64),
        0.0,
    ) + exploration_bonus
        - over_networking_penalty
        + hub_bonus
}

fn pick_any_card(state: &GameState<impl Rng>, pid: usize) -> Option<usize> {
    let hand = &state.players[pid].hand;
    (0..hand.len()).find(|_| true)
}

/// Top-K single network candidates by 1-ply score.
pub(crate) fn score_top_networks(state: &mut GameState<impl Rng>, pid: usize, k: usize) -> Vec<Decision> {
    let player = &state.players[pid];
    if state.era == Era::Canal && player.canal_links == 0 {
        return Vec::new();
    }
    if state.era == Era::Rail && player.rail_links == 0 {
        return Vec::new();
    }
    let targets = get_valid_network_targets(state, pid);
    let mut scored: Vec<(usize, f64)> = Vec::new();
    for conn_id in targets {
        let conn = &connections()[conn_id];
        let mut cost = if state.era == Era::Canal {
            crate::map::CANAL_LINK_COST
        } else {
            crate::map::RAIL_LINK_COST
        };
        if state.era == Era::Rail {
            // Rail links consume 1 coal; charging only base link cost massively
            // overvalues network spam and undervalues coal supply actions.
            cost += estimated_connection_coal_cost(state, conn_id);
        }
        let s = score_network_candidate(
            state,
            pid,
            conn_id,
            cost,
            &[conn.a, conn.b],
            state.era == Era::Canal,
        );
        scored.push((conn_id, s));
    }
    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    scored.truncate(k);
    scored
        .into_iter()
        .filter_map(|(conn_id, score)| {
            let card_index = pick_any_card(state, pid)?;
            let coal = if state.era == Era::Rail {
                cheapest_connection_coal_source(state, conn_id)
            } else {
                None
            };
            Some(Decision {
                mv: Move::Network {
                    conn_id,
                    coal,
                    card_index,
                },
                score,
            })
        })
        .collect()
}

fn score_best_network_double(state: &mut GameState<impl Rng>, pid: usize) -> Option<Decision> {
    if state.era != Era::Rail {
        return None;
    }
    let player = &state.players[pid];
    if player.rail_links < 2 {
        return None;
    }
    let singles = get_valid_network_targets(state, pid);
    let mut best: Option<(usize, usize, f64)> = None;
    for conn1 in singles {
        for conn2 in get_valid_second_rail_links(state, pid, conn1) {
            // Value both links and charge realistic resource cost:
            // total = £15 base + 2 coal (explicitly estimated from legal sources).
            let c1 = &connections()[conn1];
            let c2 = &connections()[conn2];
            let cost1 = crate::map::RAIL_LINK_COST + estimated_connection_coal_cost(state, conn1);
            let cost2 = crate::map::RAIL_LINK_COST + estimated_connection_coal_cost(state, conn2);
            let s1 = score_network_candidate(
                state,
                pid,
                conn1,
                cost1,
                &[c1.a, c1.b],
                false,
            );
            let s2 = score_network_candidate(
                state,
                pid,
                conn2,
                cost2,
                &[c2.a, c2.b],
                false,
            );
            let extra_base = (crate::map::RAIL_DOUBLE_LINK_COST
                - 2 * crate::map::RAIL_LINK_COST) as f64;
            let s = s1 + s2 - extra_base * money_weight(state);
            let better = match &best {
                Some((_, _, bs)) => s > *bs,
                None => true,
            };
            if better {
                best = Some((conn1, conn2, s));
            }
        }
    }
    let (conn1, conn2, score) = best?;
    // Prefer consuming our own beer (advances our own income when it flips) over
    // an opponent's beer (which would advance theirs).
    let opt = crate::rules::get_second_rail_options(state, pid, conn1)
        .into_iter()
        .find(|o| o.conn == conn2)?;
    let beer = opt
        .beers
        .iter()
        .copied()
        .find(|b| b.kind == crate::graph::BeerSourceKind::Own)
        .or_else(|| opt.beers.first().copied())?;
    let coal1 = crate::rules::coal_options_for_connection(state, &connections()[conn1], 1)
        .into_iter()
        .next()
        .and_then(|o| o.into_iter().next())?;
    let coal2 = opt
        .coal2_opts
        .first()
        .and_then(|o| o.first())
        .copied()?;
    let card_index = pick_any_card(state, pid)?;
    Some(Decision {
        mv: Move::NetworkDouble {
            conn1,
            conn2,
            coal1,
            coal2,
            beer,
            card_index,
        },
        score,
    })
}

// ---------------------------------------------------------------------------
// DEVELOP
// ---------------------------------------------------------------------------

fn score_develop_plan(state: &GameState<impl Rng>, pid: usize) -> Option<Decision> {
    if !can_develop(state, pid) {
        return None;
    }
    let mut types = state.players[pid].developable_types();
    if BAN_DEVELOP_IRON_LV2_PLUS {
        types.retain(|(ind, tile)| !(*ind == IndustryType::IronWorks && tile.level >= 2));
    }
    if types.is_empty() {
        return None;
    }
    let iron = find_iron_sources(state);
    if iron.is_empty() {
        return None;
    }
    let player = &state.players[pid];
    if !iron[0].free && iron[0].price as i32 > player.money {
        return None;
    }

    let mut scored: Vec<(IndustryType, f64, u8)> = types
        .into_iter()
        .map(|(ind, tile)| {
            // Develop removes a level-1 tile and reveals the NEXT (level-2+)
            // tile. Its payoff is the unlocked rail-era tile that double-scores
            // flipped VP, but develop must stay well below an immediate build.
            let unlocked = state.players[pid].tile_after(ind, 1);
            let mut v = if tile.rail_era { 0.35 } else { 0.12 };
            v += 0.18 * tile.level as f64;
            if let Some(ut) = unlocked {
                // Small premium for unlocking a double-scoring rail-era tile.
                if ut.rail_era {
                    v += 0.25;
                }
            }
            // Breweries are the economic engine (develop 1 -> build 2/3/4).
            if ind == IndustryType::Brewery {
                v += 0.55;
            }
            // Canal-era develop wins the cross-era bonus; nudge it.
            if state.era == Era::Canal {
                v += 0.15;
            }
            // Only meaningful if we actually hold a card to build it next.
            if player_has_buildable_card(state, pid, ind) {
                v += 0.3;
            }
            v -= develop_guardrail_penalty(ind, tile.level);
            (ind, v, tile.level)
        })
        .collect();
    scored.sort_by(|a, b| (b.1).partial_cmp(&a.1).unwrap());

    let first = scored[0];
    let rem_first = state.players[pid].remaining_count(first.0);
    let can_do_second_same = rem_first > 1;
    let two_source_cost: i32 = iron
        .iter()
        .take(2)
        .map(|s| if s.free { 0 } else { s.price as i32 })
        .sum();
    let can_afford_second = iron.len() >= 2 && two_source_cost <= player.money;

    let mut second: Option<IndustryType> = None;
    if can_afford_second {
        if let Some(s) = scored[1..]
            .iter()
            .find(|s| s.0 != first.0 || can_do_second_same)
        {
            second = Some(s.0);
        }
    }

    let iron_cost: i32 = iron
        .iter()
        .take(if second.is_some() { 2 } else { 1 })
        .map(|s| if s.free { 0 } else { s.price as i32 })
        .sum();
    // Iron is scarce and the develop action burns a full turn; charge a real
    // opportunity cost so develop isn't a free high-score default.
    let iron_scarcity = if iron[0].free { 0.0 } else { 0.6 };
    let second_value = if let Some(_) = second { scored[1].1 * 0.4 } else { 0.0 };

    let mut score = vp_equivalent(
        state,
        first.1 + second_value,
        0.0,
        -(iron_cost as f64),
        0.0,
    ) - iron_scarcity;

    if state.era == Era::Canal {
        score *= 2.0;
        // In canal, spending a Develop action for only one tile is usually
        // tempo-inefficient; prefer two-tile develops when legal/affordable.
        if second.is_none() {
            score -= 2.0;
        } else {
            score += 0.5;
        }
    }

    // Action-economy guardrail: develop is only worth the action budget it
    // consumes. Humans develop at most ~4 times in the canal era (8 tiles via
    // double-develops) and ~0-1 times in the rail era — beyond that, develop
    // crowds out builds/links/sells that actually score (even a bare link
    // beats a 5th develop). Penalize past the era's limit, growing steeply.
    let already = if state.era == Era::Canal {
        state.players[pid].develops_in_canal
    } else {
        state.players[pid].develops_in_rail
    };
    let this_would_be = already + 1;
    let limit = if state.era == Era::Canal { 4.0 } else { 1.0 };
    let over = (this_would_be as f64 - limit).max(0.0);
    let develop_count_penalty = over * over * 2.0 + over;
    score -= develop_count_penalty;

    let iron_needed = if second.is_some() { 2 } else { 1 };
    let iron_choice = crate::rules::iron_source_options(state, iron_needed)
        .into_iter()
        .next()?;
    let card_index = pick_any_card(state, pid)?;
    Some(Decision {
        mv: Move::Develop {
            ind1: first.0,
            ind2: second,
            iron: iron_choice,
            card_index,
        },
        score,
    })
}

// ---------------------------------------------------------------------------
// SELL
// ---------------------------------------------------------------------------

fn best_merchant_beer_bonus(
    state: &GameState<impl Rng>,
    merchant_indices: &[usize],
) -> f64 {
    let mut best = 0.0f64;
    for &i in merchant_indices {
        let mt = &state.merchants[i];
        if !mt.has_beer {
            continue;
        }
        let value = merchant_bonus_value(state, mt.loc);
        if value > best {
            best = value;
        }
    }
    best
}

fn merchant_bonus_value(state: &GameState<impl Rng>, loc: Loc) -> f64 {
    for def in crate::map::merchant_defs() {
        if def.loc == loc {
            return match def.bonus {
                crate::map::MerchantBonus::Vp(v) => v as f64 * VP_WEIGHT,
                crate::map::MerchantBonus::Money(m) => m as f64 * money_weight(state),
                crate::map::MerchantBonus::Income(spaces) => {
                    spaces as f64 * income_weight(state)
                }
                crate::map::MerchantBonus::Develop(_) => 0.5,
            };
        }
    }
    0.0
}

fn sell_route_value(state: &GameState<impl Rng>, route: &crate::rules::SellRoute) -> f64 {
    if route.use_merchant_beer {
        merchant_bonus_value(state, state.merchants[route.merchant_index].loc)
    } else {
        0.0
    }
}

fn sell_plan_executes_all<R: Rng + Clone>(
    state: &GameState<R>,
    pid: usize,
    keys: &[usize],
    merchant_indices: &[usize],
    use_merchant: &[bool],
    card_index: usize,
) -> bool {
    let mut sim = state.clone();
    if execute_sell(
        &mut sim,
        pid,
        keys,
        merchant_indices,
        use_merchant,
        card_index,
    )
    .is_err()
    {
        return false;
    }

    keys.iter().all(|&key| {
        sim.city_tiles[key]
            .as_ref()
            .map(|tile| tile.player == pid && tile.flipped)
            .unwrap_or(false)
    })
}

fn score_sell_plan<R: Rng + Clone>(state: &GameState<R>, pid: usize) -> Option<Decision> {
    let targets = get_valid_sell_targets(state, pid);
    if targets.is_empty() {
        return None;
    }

    let card_index = pick_any_card(state, pid)?;

    // Rank targets, then sell all (matches aiPlayer.js sellPlan behavior).
    let mut ranked: Vec<(usize, f64, f64)> = targets
        .iter()
        .map(|t| {
            let merchant_choices: Vec<usize> = t.routes.iter().map(|r| r.merchant_index).collect();
            let bonus = best_merchant_beer_bonus(state, &merchant_choices);
            let tile = &state.city_tiles[t.key];
            let tile_value = tile.as_ref().map_or(0.0, |tile| {
                tile.def.vp as f64 + tile.def.income as f64 * 0.3
            });
            (t.key, tile_value, bonus)
        })
        .collect();
    ranked.sort_by(|a, b| {
        (b.2)
            .partial_cmp(&a.2)
            .unwrap()
            .then((b.1).partial_cmp(&a.1).unwrap())
    });

    let mut keys = Vec::new();
    let mut merchant_indices = Vec::new();
    let mut use_merchant = Vec::new();
    for (key, _, _) in &ranked {
        let Some(target) = targets.iter().find(|t| t.key == *key) else {
            continue;
        };
        let mut routes = target.routes.clone();
        routes.sort_by(|a, b| sell_route_value(state, b).partial_cmp(&sell_route_value(state, a)).unwrap());

        for route in routes {
            let mut try_keys = keys.clone();
            let mut try_merchants = merchant_indices.clone();
            let mut try_use = use_merchant.clone();
            try_keys.push(*key);
            try_merchants.push(route.merchant_index);
            try_use.push(route.use_merchant_beer);
            if sell_plan_executes_all(state, pid, &try_keys, &try_merchants, &try_use, card_index) {
                keys = try_keys;
                merchant_indices = try_merchants;
                use_merchant = try_use;
                break;
            }
        }
    }

    if keys.is_empty() {
        return None;
    }

    let total_income: f64 = keys
        .iter()
        .map(|key| {
            state.city_tiles[*key]
                .as_ref()
                .map_or(0.0, |tile| tile.def.income as f64)
        })
        .sum();
    let total_bonus: f64 = keys
        .iter()
        .zip(merchant_indices.iter().copied())
        .zip(use_merchant.iter().copied())
        .map(|((_key, merchant_index), use_merchant)| {
            if use_merchant {
                merchant_bonus_value(state, state.merchants[merchant_index].loc)
            } else {
                0.0
            }
        })
        .sum();
    let total_vp: f64 = keys
        .iter()
        .map(|key| {
            state.city_tiles[*key]
                .as_ref()
                .map_or(0.0, |tile| tile.def.vp as f64 + tile.def.income as f64 * 0.3)
        })
        .sum();

    // End-of-era urgency: in the final round of the canal era, canal-only
    // (level-1) sellables vanish if unflipped — selling them now is essential
    // or the VP + income is permanently lost. Boost the sell action sharply.
    let rounds_left = estimate_rounds_remaining(state);
    let canal_end = state.era == Era::Canal && rounds_left <= 2.0;
    let rail_end = state.era == Era::Rail && rounds_left <= 1.0;
    let urgent = canal_end || rail_end;
    let urgency_bonus = if urgent { 3.0 } else { 0.0 };

    // Canal late-game tactical prior: deplete beer and flip breweries before
    // level-1 cleanup, or the barrels evaporate with no value.
    let mut canal_beer_drain_bonus = 0.0;
    if state.era == Era::Canal && rounds_left <= 2.5 {
        let (before_barrels, before_flipped) = own_brewery_stats(state, pid);
        let mut sim = state.clone();
        if execute_sell(
            &mut sim,
            pid,
            &keys,
            &merchant_indices,
            &use_merchant,
            card_index,
        )
        .is_ok()
        {
            let (after_barrels, after_flipped) = own_brewery_stats(&sim, pid);
            let consumed = before_barrels.saturating_sub(after_barrels) as f64;
            let flipped_gain = after_flipped.saturating_sub(before_flipped) as f64;
            canal_beer_drain_bonus = consumed * 0.9 + flipped_gain * 2.2;
            if before_barrels > 0 && after_barrels == 0 {
                canal_beer_drain_bonus += 3.8;
            }
        }
    }

    // Flipping a sellable advances income: that's a recurring cash stream,
    // not just a one-time VP. Add the income value explicitly.
    let income_stream = total_income * income_weight(state) * 0.5;

    let score = vp_equivalent(state, total_vp, total_income, 0.0, 0.0)
        + total_bonus
        + urgency_bonus
        + income_stream
        + canal_beer_drain_bonus;

    Some(Decision {
        mv: Move::Sell {
            keys,
            merchant_indices,
            use_merchant_beer: use_merchant,
            card_index,
        },
        score,
    })
}

// ---------------------------------------------------------------------------
// LOAN
// ---------------------------------------------------------------------------

fn score_loan_result<R: Rng + Clone>(state: &GameState<R>, pid: usize) -> Option<Decision> {
    if !state.can_take_loan(pid) {
        return None;
    }
    let player = &state.players[pid];
    let income_level = player.income_level();

    // A loan is the engine of the early economy: it buys the income-recovery
    // build that keeps the turn engine running. It is worth taking when cash is
    // so low that without it we'd idle-pass, provided income isn't already in
    // a death spiral.
    let rounds_left = estimate_rounds_remaining(state);
    let income_cost = crate::map::LOAN_INCOME_PENALTY as f64 * income_weight(state);
    let _ = rounds_left;

    // The post-loan budget opens up builds that buy back income fast.
    let cash = player.money as f64;
    let after = best_affordable_build_score(state, pid, cash + crate::map::LOAN_AMOUNT as f64);
    let now = best_affordable_build_score(state, pid, cash);
    let gain = (after - now).max(0.0);

    // Same-turn combo value: Loan is strongest when it immediately unlocks a
    // productive second action (Loan -> Build / Network / Develop / Sell).
    let mut combo_gain = 0.0;
    if cash < 24.0 && rounds_left > 1.5 {
        if let Some(sim_gain) = best_same_turn_after_loan(state, pid) {
            combo_gain = sim_gain.max(0.0) * 0.7;
        }
    }

    // Idle protection: if the current hand can't do anything positive, borrow.
    let idle_bonus = if cash < 6.0 { 2.0 } else { 0.0 };

    // Income floor: never borrow into deep debt.
    let post_loan_income = income_level - crate::map::LOAN_INCOME_PENALTY;
    let floor_penalty = if post_loan_income <= -8 {
        7.0
    } else if post_loan_income <= -6 {
        4.2
    } else if post_loan_income <= -4 {
        2.6
    } else if post_loan_income <= -2 {
        1.4
    } else if post_loan_income < 0 {
        0.6
    } else {
        0.0
    };

    // Do not over-borrow with plenty of cash, and avoid debt spiral in late
    // era where income recovery window is too short.
    let rich_penalty = if cash >= 55.0 {
        5.0
    } else if cash >= 42.0 {
        2.4
    } else if cash >= 30.0 {
        1.0
    } else {
        0.0
    };
    let late_era_penalty = if rounds_left <= 1.0 {
        3.0
    } else if rounds_left <= 1.5 {
        1.0
    } else {
        0.0
    };
    let unlock_bonus = if now <= 0.0 && after > 0.8 {
        3.2
    } else {
        0.0
    };

    // Slight tactical nudge: early canal loans are often the cleanest way to
    // unlock the engine (loan -> build/develop) before tempo stalls.
    let canal_early_loan_bonus = if state.era == Era::Canal && rounds_left >= 5.5 {
        if cash < 18.0 {
            0.9
        } else {
            0.4
        }
    } else {
        0.0
    };

    let score = gain + combo_gain + idle_bonus + unlock_bonus + canal_early_loan_bonus
        - income_cost - floor_penalty - rich_penalty - late_era_penalty;
    let card_index = pick_any_card(state, pid)?;
    Some(Decision {
        mv: Move::Loan { card_index },
        score,
    })
}

fn best_same_turn_after_loan<R: Rng + Clone>(state: &GameState<R>, pid: usize) -> Option<f64> {
    let card_index = pick_any_card(state, pid)?;
    let mut sim = state.clone();
    crate::rules::apply_move(&mut sim, &Move::Loan { card_index }).ok()?;
    let tr = advance_turn(&mut sim);
    match tr {
        TurnResult::Continue => {}
        TurnResult::EndCanalEra | TurnResult::EndGame => return Some(0.0),
    }
    if sim.current_player_id() != pid {
        return Some(0.0);
    }
    let cands = candidate_actions_k(&mut sim, 3);
    cands
        .into_iter()
        .filter(|d| !matches!(d.mv, Move::Loan { .. }))
        .map(|d| d.score)
        .max_by(|a, b| a.partial_cmp(b).unwrap())
}

/// Best build score the player could afford within a cash budget.
fn best_affordable_build_score(
    state: &GameState<impl Rng>,
    pid: usize,
    budget: f64,
) -> f64 {
    let mut best = f64::NEG_INFINITY;
    for t in get_valid_build_targets(state, pid) {
        if (t.cost_total as f64) > budget {
            continue;
        }
        let s = score_build_candidate(state, pid, &t);
        if s > best {
            best = s;
        }
    }
    if best == f64::NEG_INFINITY {
        0.0
    } else {
        best
    }
}

// ---------------------------------------------------------------------------
// SCOUT
// ---------------------------------------------------------------------------

fn card_usefulness(state: &GameState<impl Rng>, pid: usize, card: &Card) -> f64 {
    match card {
        Card::WildLocation | Card::WildIndustry => 5.0,
        Card::Location(loc) => {
            if loc.is_farm() {
                return 0.0;
            }
            let remaining = &state.players[pid];
            for slot_types in city_slots(*loc) {
                for t in *slot_types {
                    if remaining.remaining_count(*t) > 0 {
                        return 2.0;
                    }
                }
            }
            0.0
        }
        Card::Industry { .. } => {
            let remaining = &state.players[pid];
            let has = if let Card::Industry { industries, n } = card {
                industries[..*n as usize]
                    .iter()
                    .any(|t| remaining.remaining_count(*t) > 0)
            } else {
                false
            };
            if has {
                1.5
            } else {
                0.0
            }
        }
    }
}

fn score_scout_plan(state: &GameState<impl Rng>, pid: usize) -> Option<Decision> {
    if !can_scout(state, pid) {
        return None;
    }
    let player = &state.players[pid];
    let mut usefulness: Vec<(usize, f64)> = player
        .hand
        .iter()
        .enumerate()
        .map(|(i, c)| (i, card_usefulness(state, pid, c)))
        .collect();
    usefulness.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

    let discard = &usefulness[..3];
    let dead_count = discard.iter().filter(|(_, s)| *s <= 0.0).count();
    let score = vp_equivalent(
        state,
        0.0,
        0.0,
        0.0,
        dead_count as f64 * 1.2 - (3.0 - dead_count as f64) * 0.6,
    );

    let card_indices: [usize; 3] = [discard[0].0, discard[1].0, discard[2].0];
    Some(Decision {
        mv: Move::Scout { card_indices },
        score,
    })
}

// ---------------------------------------------------------------------------
// PASS
// ---------------------------------------------------------------------------

fn score_pass_result(state: &GameState<impl Rng>, pid: usize) -> Option<Decision> {
    let card_index = pick_any_card(state, pid).unwrap_or(0);
    Some(Decision {
        mv: Move::Pass { card_index },
        score: -0.5,
    })
}

// --- temporary debug helpers ------------------------------------------------
pub fn debug_flip(
    state: &GameState<impl rand::Rng>,
    pid: usize,
    ind: IndustryType,
    loc: Loc,
) -> f64 {
    estimate_flip_probability(state, pid, ind, loc)
}

pub fn debug_market_adjust(
    state: &GameState<impl rand::Rng>,
    ind: IndustryType,
    loc: Loc,
    cubes: u8,
) -> f64 {
    let is_resource = matches!(ind, IndustryType::CoalMine | IndustryType::IronWorks);
    if !is_resource {
        return 0.0;
    }
    let connected = connected_locations(state, loc);
    let market_ok = ind == IndustryType::IronWorks || connected.iter().any(|l| l.is_merchant());
    let is_coal = ind == IndustryType::CoalMine;
    let scarcity = if is_coal {
        (14 - state.coal_market) as f64 / 14.0
    } else {
        0.6 * (10 - state.iron_market) as f64 / 10.0
    };
    if market_ok {
        let sale = simulate_market_sale(state, is_coal, cubes);
        let sell_value = sale.cash * money_weight(state);
        let scarcity_value = scarcity * (1.0 + sale.sold as f64) * 0.6;
        let leftover_penalty = (sale.total - sale.sold) as f64 * 0.5;
        sell_value + scarcity_value - leftover_penalty
    } else if is_coal {
        if state.era == Era::Canal {
            -0.5
        } else {
            scarcity * 0.6
        }
    } else {
        scarcity * 1.2
    }
}
