//! The single flip-probability model.
//!
//! Two independent models used to coexist (`estimate_flip_probability` for
//! concrete builds and `plan_flip_probability` for the abstract production
//! plan); both are now thin wrappers over the per-industry estimators here.
//!
//! Flip odds follow three regimes:
//! - **Resources** (coal/iron) flip when consumed by the table or when a
//!   build immediately sells its cubes into the market. Odds follow MARKET
//!   HUNGER (sparse market => fast consumption) and ERA DEMAND (rail era is
//!   coal-hungry). A coal mine with no merchant reach ("island mine")
//!   cannot sell on placement and — especially in the canal era — flips
//!   rarely.
//! - **Breweries** flip only when their beer is actually demanded by sells
//!   (and rail links in the rail era).
//! - **Sellables** (cotton/goods/pottery) only flip when sold: that needs a
//!   reachable accepting merchant AND beer.

use super::board::{
    beer_available, merchant_reachable, owned_beer_barrels, sellable_beer_demand,
    unbuilt_neighbor_connections,
};
use super::context::EvalContext;
use super::value::{market_scarcity, price_heat, simulate_market_sale};
use crate::data::IndustryType;
use crate::map::Loc;
use crate::state::GameState;

/// Probability that the next tile of `ind` built at `loc` eventually flips.
///
/// `loc` may be `None` for the abstract production plan, which judges
/// industry-wide feasibility (any accepting merchant on the board) instead
/// of site connectivity.
pub fn flip_probability(
    state: &GameState,
    ctx: &EvalContext,
    ind: IndustryType,
    loc: Option<Loc>,
) -> f64 {
    let cfg = &ctx.cfg.flip;
    flip_probability_with(state, ctx.pid, ctx.is_canal(), cfg, ind, loc)
}

/// Context-free implementation shared by concrete Build scoring and the
/// context-aware plan scorer. Keeping the policy parameters explicit avoids
/// constructing an evaluation context for every build candidate.
fn flip_probability_with(
    state: &GameState,
    pid: usize,
    is_canal: bool,
    cfg: &super::config::FlipWeights,
    ind: IndustryType,
    loc: Option<Loc>,
) -> f64 {
    let base = if matches!(ind, IndustryType::CoalMine | IndustryType::IronWorks) {
        let cubes = state
            .players
            .get(pid)
            .and_then(|p| p.next_tile(ind))
            .map(|t| t.resource_cubes)
            .unwrap_or(1);
        resource_flip(state, is_canal, cfg, ind, cubes, loc)
    } else if ind == IndustryType::Brewery {
        let next_cubes = state
            .players
            .get(pid)
            .and_then(|p| p.next_tile(ind))
            .map(|t| t.resource_cubes as usize)
            .unwrap_or(1);
        brewery_flip(state, pid, is_canal, cfg, next_cubes)
    } else {
        let hand_len = state.players[pid].hand.len();
        sellable_flip(state, pid, cfg, ind, loc, hand_len)
    };
    base.clamp(cfg.floor, cfg.cap.max(cfg.floor))
}

/// Resource flip model. See module docs for the regimes.
fn resource_flip(
    state: &GameState,
    is_canal: bool,
    cfg: &super::config::FlipWeights,
    ind: IndustryType,
    cubes: u8,
    loc: Option<Loc>,
) -> f64 {
    let is_coal = ind == IndustryType::CoalMine;
    let scarcity = market_scarcity(state, is_coal);
    let can_sell = match loc {
        Some(loc) => ind == IndustryType::IronWorks || merchant_reachable(state, loc, ind),
        // Plan-level view: an iron works always sells; coal needs *some*
        // accepting merchant on the board.
        None => ind == IndustryType::IronWorks || state.merchants.iter().any(|mt| mt.accepts(ind)),
    };

    // An island coal mine can't sell on build and — especially in the canal
    // era — nobody will build a link out to it, so it sits unflipped and
    // vanishes at era end. Demand pressure (expensive market coal) still
    // gives it a chance.
    if is_coal && !can_sell {
        let heat_price = state.coal_price();
        return if is_canal {
            let heat = price_heat(
                heat_price,
                cfg.island_coal_canal_price_base,
                cfg.island_coal_canal_price_span,
            );
            (cfg.island_coal_canal_base + cfg.island_coal_canal_heat_bonus * heat)
                .min(cfg.island_coal_canal_cap)
        } else {
            let heat = price_heat(
                heat_price,
                cfg.island_coal_rail_price_base,
                cfg.island_coal_rail_price_span,
            );
            (cfg.island_coal_rail_base + cfg.island_coal_rail_heat_bonus * heat)
                .min(cfg.island_coal_rail_cap)
        };
    }

    let sale = simulate_market_sale(state, is_coal, cubes);
    if can_sell && sale.flips {
        // Sells out on placement: flips immediately.
        return cfg.sellout;
    }

    // Relies on consumption by the table. Coal demand is enormous in the
    // rail era (all builds and links consume it); iron demand is steadier.
    let era_demand = match (is_coal, !is_canal) {
        (true, true) => cfg.coal_demand_rail,
        (true, false) => cfg.coal_demand_canal,
        (false, true) => cfg.iron_demand_rail,
        (false, false) => cfg.iron_demand_canal,
    };
    (era_demand + cfg.scarcity_bonus * scarcity).min(cfg.cap)
}

/// Brewery flip model: beer demand from our unflipped sellables (plus a
/// rail-network buffer in the rail era) versus supply including this
/// brewery's own barrels.
fn brewery_flip(
    state: &GameState,
    pid: usize,
    is_canal: bool,
    cfg: &super::config::FlipWeights,
    next_cubes: usize,
) -> f64 {
    let demand = sellable_beer_demand(state, pid) as f64
        + if !is_canal {
            cfg.brewery_rail_demand_buffer
        } else {
            0.0
        };
    let barrels = (owned_beer_barrels(state, pid) + next_cubes) as f64;
    if demand <= 0.5 && is_canal {
        cfg.brewery_canal_no_demand
    } else if barrels > demand {
        cfg.brewery_surplus
    } else {
        cfg.brewery_satisfied
    }
}

/// Sellable flip model: a sellable tile only flips when sold, which needs a
/// reachable accepting merchant and beer to fuel the sale.
fn sellable_flip(
    state: &GameState,
    pid: usize,
    cfg: &super::config::FlipWeights,
    ind: IndustryType,
    loc: Option<Loc>,
    hand_len: usize,
) -> f64 {
    let Some(loc) = loc else {
        // Plan-level view: judge board-wide feasibility only.
        if !state.merchants.iter().any(|mt| mt.accepts(ind)) {
            return cfg.plan_no_merchant;
        }
        let beer_ok = owned_beer_barrels(state, pid) > 0
            || state
                .merchants
                .iter()
                .any(|mt| mt.has_beer && mt.accepts(ind));
        return if beer_ok {
            cfg.plan_ready
        } else {
            cfg.plan_no_beer
        };
    };

    let mut b = cfg.sellable_base;
    if merchant_reachable(state, loc, ind) {
        // A merchant is only worth real flip credit if there is beer to
        // fuel the sale; without beer it's a dead end this era.
        if beer_available(state, loc, pid, 1) {
            b += cfg.sellable_merchant_with_beer;
        } else {
            b += cfg.sellable_merchant_only;
        }
    } else {
        b += cfg.sellable_no_merchant;
    }
    // Adjacent unbuilt links mean a merchant can still be connected later.
    if unbuilt_neighbor_connections(state, loc) > 0 {
        b += cfg.sellable_open_link;
    }
    // Hand poverty: a sellable can only flip if we can still reach a sell
    // action this era.
    b -= match hand_len {
        0 => cfg.hand_empty_penalty,
        1 => cfg.hand_one_card_penalty,
        2..=3 => cfg.hand_few_cards_penalty,
        _ => 0.0,
    };
    b
}

/// Probability that a build at `loc` flips, for the concrete-build path.
pub fn build_flip_probability(state: &GameState, pid: usize, ind: IndustryType, loc: Loc) -> f64 {
    // Build scoring supplies only the state and player. The default flip
    // policy is shared with the plan-level model, but no evaluation context is
    // constructed on this hot path.
    let cfg = super::config::HeuristicConfig::default();
    flip_probability_with(state, pid, state.is_canal_era(), &cfg.flip, ind, Some(loc))
}

/// Probability that the plan industry flips at all (plan-level view).
pub fn plan_flip_probability(state: &GameState, ctx: &EvalContext, ind: IndustryType) -> f64 {
    flip_probability(state, ctx, ind, None)
}
