//! Sell action scoring.

use super::context::EvalContext;
use super::value::ScoreParts;
use super::{CardChoices, Decision};
use crate::gameplay::actions::SellTarget;
use crate::map::{Loc, merchant_bonus_at};
use crate::rules::ResolvedMove;
use crate::rules::{SellRoute, execute_sell, get_valid_sell_targets, plan_sell_beer_sources};
use crate::state::GameState;

/// Value of the merchant bonus gained by selling with a merchant's own
/// beer, as score components (VP / money / income / free develop).
fn merchant_bonus_parts(_state: &GameState, ctx: &EvalContext, loc: Loc) -> ScoreParts {
    let w = &ctx.cfg.sell;
    match merchant_bonus_at(loc) {
        crate::map::MerchantBonus::Vp(v) => ScoreParts {
            vp: v as f64 * ctx.cfg.value.vp,
            ..Default::default()
        },
        crate::map::MerchantBonus::Money(m) => ScoreParts {
            money: m as f64,
            ..Default::default()
        },
        crate::map::MerchantBonus::Income(spaces) => ScoreParts {
            income: spaces as f64,
            ..Default::default()
        },
        crate::map::MerchantBonus::Develop(_) => ScoreParts {
            strategic: w.develop_bonus_value,
            ..Default::default()
        },
    }
}

fn sell_route_value(state: &GameState, ctx: &EvalContext, route: &SellRoute) -> f64 {
    if route.use_merchant_beer {
        merchant_bonus_parts(state, ctx, state.merchants[route.merchant_index].loc).total(ctx)
    } else {
        0.0
    }
}

/// Verify the greedy plan actually executes and flips every included tile.
fn sell_plan_executes_all(
    state: &GameState,
    pid: usize,
    keys: &[usize],
    merchant_indices: &[usize],
    use_merchant: &[bool],
    card_index: usize,
) -> bool {
    let mut sim = state.clone();
    let Ok(beer_sources) = plan_sell_beer_sources(state, pid, keys, merchant_indices, use_merchant)
    else {
        return false;
    };
    if execute_sell(
        &mut sim,
        pid,
        keys,
        merchant_indices,
        use_merchant,
        &beer_sources,
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

/// Printed VP plus the income advance of flipping one tile — the ranking
/// key for which tiles to sell.
fn tile_sell_value(state: &GameState, ctx: &EvalContext, key: usize) -> f64 {
    let w = &ctx.cfg.sell;
    state.city_tiles[key].as_ref().map_or(0.0, |tile| {
        tile.def.vp as f64 + tile.def.income as f64 * w.tile_income_share
    })
}

pub(super) fn score_sell_plans(
    state: &GameState,
    ctx: &EvalContext,
    card_choices: &CardChoices,
) -> Vec<Decision> {
    let pid = ctx.pid;
    let targets = get_valid_sell_targets(state, pid);
    if targets.is_empty() {
        return Vec::new();
    }

    let Some(card_index) = card_choices.first().map(|(index, _)| *index) else {
        return Vec::new();
    };

    // Rank targets by the merchant beer bonus they can grab, then by tile
    // value; sell all (matches aiPlayer.js sellPlan behavior).
    let mut ranked: Vec<(usize, f64, f64)> = targets
        .iter()
        .map(|t| {
            let merchant_choices: Vec<usize> = t.routes.iter().map(|r| r.merchant_index).collect();
            let bonus = merchant_choices
                .iter()
                .filter(|&&i| state.merchants[i].has_beer)
                .map(|&i| merchant_bonus_parts(state, ctx, state.merchants[i].loc).total(ctx))
                .fold(0.0f64, f64::max);
            (t.key, tile_sell_value(state, ctx, t.key), bonus)
        })
        .collect();
    ranked.sort_by(|a, b| b.2.total_cmp(&a.2).then(b.1.total_cmp(&a.1)));

    // v4: emit up to SOURCE_VARIANTS plans; variant n>0 gives the FIRST ranked
    // tile the runner-up merchant route so search can choose between them.
    let mut out = Vec::new();
    for first_route_offset in 0..super::SOURCE_VARIANTS {
        if let Some(decision) = sell_plan_for(
            state,
            ctx,
            card_choices,
            card_index,
            &targets,
            &ranked,
            first_route_offset,
        ) {
            out.push(decision);
        }
    }
    out
}

/// One Sell plan: greedy route pick over ranked targets; the first ranked tile
/// starts scanning its routes at `first_route_offset`.
fn sell_plan_for(
    state: &GameState,
    ctx: &EvalContext,
    card_choices: &CardChoices,
    card_index: usize,
    targets: &[SellTarget],
    ranked: &[(usize, f64, f64)],
    first_route_offset: usize,
) -> Option<Decision> {
    let pid = ctx.pid;
    let w = &ctx.cfg.sell;

    let mut keys = Vec::new();
    let mut merchant_indices = Vec::new();
    let mut use_merchant = Vec::new();
    for (index, (key, _, _)) in ranked.iter().enumerate() {
        let Some(target) = targets.iter().find(|t| t.key == *key) else {
            continue;
        };
        let mut routes = target.routes.clone();
        routes.sort_by(|a, b| {
            sell_route_value(state, ctx, b).total_cmp(&sell_route_value(state, ctx, a))
        });
        let offset = if index == 0 { first_route_offset } else { 0 };

        for route in routes.iter().skip(offset) {
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

    // Sum the realised effects of the accepted plan.
    let mut parts = ScoreParts::default();
    for (key, &merchant, &uses_beer) in keys
        .iter()
        .zip(merchant_indices.iter())
        .zip(use_merchant.iter())
        .map(|((k, m), u)| (k, m, u))
    {
        if let Some(tile) = state.city_tiles[*key].as_ref() {
            parts.vp += tile.def.vp as f64;
            parts.income += tile.def.income as f64;
        }
        if uses_beer {
            parts.add(&merchant_bonus_parts(
                state,
                ctx,
                state.merchants[merchant].loc,
            ));
        }
    }

    // Flipping a sellable advances income: a recurring cash stream, not
    // just one-time VP — credit the income again at a reduced share.
    parts.income *= 1.0 + w.income_stream_share;

    // Early sells are mostly about income (VP grows later); the realised VP
    // counts more as the era runs out.
    let vp_scale = w.vp_scale_floor + w.vp_scale_span * (1.0 - ctx.era_frac);
    parts.vp *= vp_scale;

    // Era-end urgency: in the final canal rounds, canal-only (level-1)
    // sellables vanish if unflipped — selling now is essential or the VP +
    // income is permanently lost. In Rail-Late selling IS the whole point.
    if ctx.is_era_endgame() {
        parts.strategic += w.urgency_bonus;
    } else if ctx.phase == super::plan::Phase::RailLate {
        parts.strategic += w.rail_late_baseline_bonus;
    }

    // Canonical Sell actions use ascending city-tile keys. The plan above is
    // ranked by merchant bonus/value, so its construction order can differ
    // from legal_resolved_moves() even for the same concrete action. Sort
    // the aligned route vectors before returning the teacher move.
    let mut routes: Vec<(usize, usize, bool)> = keys
        .into_iter()
        .zip(merchant_indices)
        .zip(use_merchant)
        .map(|((key, merchant), use_merchant)| (key, merchant, use_merchant))
        .collect();
    routes.sort_by_key(|(key, _, _)| *key);
    let (keys, merchant_indices, use_merchant) = routes.into_iter().fold(
        (Vec::<usize>::new(), Vec::<usize>::new(), Vec::<bool>::new()),
        |(mut keys, mut merchants, mut beer), (key, merchant, use_merchant)| {
            keys.push(key);
            merchants.push(merchant);
            beer.push(use_merchant);
            (keys, merchants, beer)
        },
    );
    let beer_sources =
        plan_sell_beer_sources(state, pid, &keys, &merchant_indices, &use_merchant).ok()?;
    let free_develop =
        merchant_indices
            .iter()
            .zip(use_merchant.iter())
            .find_map(|(&merchant, &uses_beer)| {
                (uses_beer
                    && matches!(
                        merchant_bonus_at(state.merchants[merchant].loc),
                        crate::map::MerchantBonus::Develop(_)
                    ))
                .then(|| {
                    state.players[pid]
                        .developable_types()
                        .into_iter()
                        .max_by_key(|(_, tile)| tile.level)
                        .map(|(ind, _)| ind)
                })
                .flatten()
            });

    Some(Decision {
        mv: ResolvedMove::Sell {
            keys,
            merchant_indices,
            use_merchant_beer: use_merchant,
            beer_sources,
            free_develop,
            card_index,
        },
        score: parts.total(ctx),
        card_score: card_choices
            .first()
            .map(|(_, s)| *s)
            .unwrap_or(f64::INFINITY),
    })
}
