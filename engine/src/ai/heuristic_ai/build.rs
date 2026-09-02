//! Build action scoring.
//!
//! `score_build_candidate` composes one [`ScoreParts`] from named factor
//! functions; every factor maps 1:1 to a config group in
//! `HeuristicConfig::build`, so tuning a coefficient means touching one
//! named parameter instead of a magic literal.

use super::Decision;
use super::board::{
    beer_available, beer_barrels_reachable, merchant_reachable, own_overbuild_vp_loss,
    owned_beer_barrels, player_owns_link_touching, resource_source_ratio, sellable_beer_demand,
    unbuilt_neighbor_connections,
};
use super::context::EvalContext;
use super::plan::{Phase, Plan};
use super::probability::build_flip_probability;
use super::value::{ScoreParts, market_scarcity, price_heat, simulate_market_sale};
use crate::data::IndustryType;
use crate::graph::count_beer_sources;
use crate::rules::{BuildTarget, ResolvedMove, valid_build_cards};
use crate::state::GameState;

/// Market value of the cubes a new coal/iron works produces: immediate sale
/// cash (money), sale tempo/spike/scarcity value (strategic), and the
/// penalty for cubes that stay stuck on the tile (risk).
///
/// "If everything sells, it's a great move; the more that stays on the
/// tile, the worse (unless spent on your very next action)."
fn market_value(state: &GameState, ctx: &EvalContext, cand: &BuildTarget, cubes: u8) -> ScoreParts {
    let w = &ctx.cfg.build;
    let is_coal = cand.ind == IndustryType::CoalMine;
    // Coal demand is far higher than iron (every rail build/link eats it);
    // iron demand is steadier and the market fills faster, so iron scarcity
    // is discounted to avoid over-producing iron nobody will consume.
    let scarcity =
        market_scarcity(state, is_coal) * if is_coal { 1.0 } else { w.iron_scarcity_share };

    // Iron works sell anywhere; coal needs a merchant in reach.
    let market_ok = !is_coal || merchant_reachable(state, cand.loc, cand.ind);
    if !market_ok {
        // Not merchant-connected: cubes can't be sold today. For an ISLAND
        // COAL MINE this is a strongly negative move in the canal era (tiles
        // vanish at era end, nobody helps consume it) and merely speculative
        // in the rail era (links will be built out). Iron needs no
        // connection to be consumed, so it keeps a market-value floor.
        let strategic = if is_coal {
            if ctx.is_canal() {
                w.island_coal_canal_penalty
            } else {
                scarcity * (w.island_coal_rail_base + w.island_coal_rail_per_cube * cubes as f64)
            }
        } else {
            scarcity * w.island_iron_value
        };
        return ScoreParts {
            strategic,
            ..Default::default()
        };
    }

    let sale = simulate_market_sale(state, is_coal, cubes);
    // Immediate cash from the auto-sale is real money.
    let money = sale.cash;
    let cash_back_bonus = if sale.cash > 0.0 {
        sale.cash * w.market_cash_back_share
            + if sale.flips {
                w.market_sellout_bonus
            } else {
                0.0
            }
    } else {
        0.0
    };
    // Coal market spike prior: when the buy price is in the demand window
    // (6/7/8), placing a connected coal mine that can auto-sell is typically
    // a top-tier tempo play (cash now + flip income + future board demand).
    let coal_spike_bonus = if is_coal && sale.sold > 0 {
        price_heat(
            state.coal_price(),
            w.coal_spike_price_base,
            w.coal_spike_price_span,
        ) * sale.sold as f64
            * w.coal_spike_per_sold
            * if ctx.is_canal() {
                w.coal_spike_canal_mult
            } else {
                1.0
            }
    } else {
        0.0
    };
    let scarcity_value = scarcity * (1.0 + sale.sold as f64) * w.scarcity_value_per_unit;
    // In the rail era coal is consumed by nearly every build and rail link —
    // a full market right now doesn't mean the cubes won't be eaten soon,
    // so unsold rail coal is not punished.
    let leftover_penalty = if is_coal && ctx.is_rail() {
        0.0
    } else {
        (sale.total - sale.sold) as f64 * w.leftover_per_cube
    };

    ScoreParts {
        money,
        strategic: cash_back_bonus + coal_spike_bonus + scarcity_value,
        risk: -leftover_penalty,
        ..Default::default()
    }
}

/// Beer economy of the build: sellables need merchant + beer to flip, and
/// breweries are worth a premium while they still have sellables to feed.
fn beer_economy(
    state: &GameState,
    ctx: &EvalContext,
    cand: &BuildTarget,
    beers_to_sell: Option<u8>,
) -> ScoreParts {
    let w = &ctx.cfg.build;
    let mut strategic = 0.0;
    if cand.ind.is_sellable() {
        if merchant_reachable(state, cand.loc, cand.ind) {
            strategic += w.merchant_reachable_bonus;
        }
        if merchant_reachable(state, cand.loc, cand.ind)
            && beer_available(
                state,
                cand.loc,
                ctx.pid,
                beers_to_sell.unwrap_or(0) as usize,
            )
        {
            // We have the beer AND the merchant: a guaranteed sellable.
            strategic += w.beer_available_bonus;
        } else {
            // Don't punish too hard: a brewery can still be added later.
            strategic += w.beer_missing_penalty;
        }
    } else if cand.ind == IndustryType::Brewery {
        // Breweries feed sells AND rail links. The winning line develops
        // level-1 away and builds level-2/3/4 (they survive to rail). Match
        // output to need: barrels should cover our unflipped sellables'
        // beer demand plus a small rail-network buffer.
        let tile_cubes = state.players[ctx.pid]
            .next_tile(cand.ind)
            .map(|t| t.resource_cubes as f64)
            .unwrap_or(0.0);
        let barrels = owned_beer_barrels(state, ctx.pid) as f64 + tile_cubes;
        let demand = sellable_beer_demand(state, ctx.pid) as f64;
        let surplus = (barrels - demand).max(0.0);
        let sell_support = if demand > 0.0 {
            w.brewery_sell_support_with_demand
        } else {
            w.brewery_sell_support_base
        };
        strategic += sell_support
            + if ctx.is_rail() {
                w.rail_brewery_value
            } else {
                0.0
            }
            - w.brewery_surplus_penalty_per_barrel * surplus;
    }
    ScoreParts {
        strategic,
        ..Default::default()
    }
}

/// Rail-era emergency coal prior: when the coal market is near empty, any
/// legal coal mine is strategically premium because the whole table must
/// pay expensive market coal otherwise. Independent of immediate auto-sell.
fn rail_coal_shortage(
    state: &GameState,
    ctx: &EvalContext,
    cand: &BuildTarget,
    level: u8,
    cubes: u8,
) -> f64 {
    let w = &ctx.cfg.build;
    if cand.ind != IndustryType::CoalMine || !ctx.is_rail() {
        return 0.0;
    }
    let shortage = market_scarcity(state, true);
    let level_factor = 1.0 + w.rail_coal_shortage_per_level * (level.saturating_sub(1)) as f64;
    let cubes_factor =
        w.rail_coal_shortage_cubes_base + w.rail_coal_shortage_per_cube * cubes as f64;
    shortage * level_factor * cubes_factor * w.rail_coal_shortage
}

/// Cost-efficiency factor: the build must be worth its price tag. Cheap
/// builds (coal £5, iron £7, brewery £5) get a relative edge early.
fn cost_efficiency(ctx: &EvalContext, income: u8, vp: u8, cost: f64) -> f64 {
    let w = &ctx.cfg.build;
    if cost > 0.0 {
        ((income as f64 + vp as f64) / cost).min(w.cost_efficiency_cap)
    } else {
        0.0
    }
}

/// Score one build candidate in VP-equivalent components.
pub(super) fn score_build_candidate(
    state: &GameState,
    ctx: &EvalContext,
    cand: &BuildTarget,
    plan: &Plan,
) -> ScoreParts {
    let Some(tile) = state.players[ctx.pid].next_tile(cand.ind) else {
        return ScoreParts {
            risk: f64::NEG_INFINITY,
            ..Default::default()
        };
    };
    if ctx.cfg.guardrails.ban_build_lv1_brewery
        && cand.ind == IndustryType::Brewery
        && tile.level == 1
    {
        return ScoreParts {
            risk: f64::NEG_INFINITY,
            ..Default::default()
        };
    }

    // Cash is the hard constraint that drives the early economy. A build we
    // cannot afford is heavily discounted (a loan remains a path, but is
    // not a first choice).
    let cash = state.players[ctx.pid].money as f64;
    let cost = cand.cost_total as f64;
    if cost > cash {
        return ScoreParts {
            risk: -(cost - cash) * ctx.cfg.build.unaffordable_per_pound,
            ..Default::default()
        };
    }

    let flip_prob = build_flip_probability(state, ctx, cand.ind, cand.loc);

    // Link icons this tile would add to a link we already own next to it.
    let link_self_value = if player_owns_link_touching(state, ctx.pid, cand.loc) {
        tile.link_vp as f64 * flip_prob * ctx.cfg.build.link_self_value_share
    } else {
        0.0
    };

    let is_resource = matches!(cand.ind, IndustryType::CoalMine | IndustryType::IronWorks);
    let resource_self_sufficiency = if is_resource {
        ctx.cfg.build.self_sufficiency_per_cube * tile.resource_cubes as f64
    } else {
        0.0
    };

    let mut parts = ScoreParts {
        // Expected VP: printed VP realised on flip, plus owned-link icons.
        vp: tile.vp as f64 * flip_prob + link_self_value,
        // Expected income advance on flip.
        income: tile.income as f64 * flip_prob,
        // The build itself costs money.
        money: -(cand.cost_total as f64),
        strategic: resource_self_sufficiency
            + ctx.cfg.build.expansion_per_link
                * unbuilt_neighbor_connections(state, cand.loc) as f64
            + rail_coal_shortage(state, ctx, cand, tile.level, tile.resource_cubes)
            + cost_efficiency(ctx, tile.income, tile.vp, cand.cost_total as f64),
        ..Default::default()
    };

    if is_resource {
        parts.add(&market_value(state, ctx, cand, tile.resource_cubes));
    }
    parts.add(&beer_economy(state, ctx, cand, tile.beers_to_sell));

    // "Free-riding" efficiency: sourcing coal/iron from board mines/works is
    // cheaper and faster, and consuming opponents' cubes keeps the shared
    // pool cheap for everyone.
    let ratio = resource_source_ratio(state, cand);
    parts.strategic +=
        (ratio - ctx.cfg.build.free_riding_threshold).max(0.0) * ctx.cfg.build.free_riding_bonus;

    // Rebuilding over one of our own tiles forfeits that tile's end-game VP.
    parts.risk -= own_overbuild_vp_loss(state, ctx.pid, cand, ctx.cfg.value.own_overbuild_vp_loss);

    // Plan ("流派") soft bonus: building the plan industry aligns with the
    // production plan. Only from Canal-Late onward — in Canal-Early the
    // priority is the coal/iron economy engine, not committing to the
    // sellable line yet.
    if plan.count > 0 && plan.industry == cand.ind && ctx.phase != Phase::CanalEarly {
        parts.strategic += ctx.cfg.build.plan_bonus;
    }

    // Rail-Late beer-gated finish: the whole late game hinges on flipping
    // sellables — "有酒才建产业". Reward the sellable build when beer is
    // genuinely available; `flip_prob` already keeps it low when not.
    if ctx.phase == Phase::RailLate
        && cand.ind.is_sellable()
        && beer_available_for_sellable(state, ctx, cand)
    {
        parts.strategic += ctx.cfg.build.rail_late_beer_bonus;
    }

    parts
}

/// Beer for a Rail-Late sellable finish: own board sources at the location,
/// or reachable merchant barrels.
fn beer_available_for_sellable(state: &GameState, ctx: &EvalContext, cand: &BuildTarget) -> bool {
    count_beer_sources(state, cand.loc, ctx.pid, &[]) > 0 || beer_barrels_reachable(state, cand.loc)
}

/// Pick the cheapest-to-discard legal card for a build target. Consumes the
/// precomputed hand keep-scores instead of re-deriving them (the target
/// enumeration inside `card_keep_score` is expensive enough that calling it
/// inside a sort comparator used to dominate scoring time).
fn pick_build_card(
    state: &GameState,
    pid: usize,
    cand: &BuildTarget,
    keep_scores: &[(usize, f64)],
) -> Option<(usize, f64)> {
    let keep = |i: usize| {
        keep_scores
            .iter()
            .find(|(idx, _)| *idx == i)
            .map(|(_, s)| *s)
            .unwrap_or(f64::INFINITY)
    };
    let player = &state.players[pid];
    let indices = valid_build_cards(state, player, pid, cand.loc, cand.ind);
    // Prefer a non-wild matching card over a wild one.
    let non_wild: Vec<_> = indices
        .iter()
        .copied()
        .filter(|index| {
            !matches!(
                player.hand[*index],
                crate::state::Card::WildLocation | crate::state::Card::WildIndustry
            )
        })
        .collect();
    let candidates = if non_wild.is_empty() {
        &indices
    } else {
        &non_wild
    };
    let index = candidates
        .iter()
        .copied()
        .into_iter()
        .min_by(|a, b| keep(*a).total_cmp(&keep(*b)).then(a.cmp(b)))?;
    Some((index, keep(index)))
}

#[cfg(test)]
mod build_card_tests {
    use super::*;
    use crate::rules::get_valid_build_targets;
    use crate::state::Card;
    use rand_chacha::ChaCha12Rng;
    use rand_chacha::rand_core::SeedableRng;

    #[test]
    fn build_keeps_a_wild_when_a_matching_industry_card_exists() {
        let mut state = GameState::new(ChaCha12Rng::seed_from_u64(22), 4);
        let pid = state.current_player_id();
        let target = get_valid_build_targets(&state, pid)
            .into_iter()
            .next()
            .expect("opening state has a build target");
        state.players[pid].hand = vec![
            Card::Industry {
                industries: [target.ind; 2],
                n: 1,
            },
            Card::WildIndustry,
        ];
        // A deliberately lower Wild keep-score verifies that Build's explicit
        // non-Wild preference, rather than incidental score ordering, wins.
        let chosen = pick_build_card(&state, pid, &target, &[(0, 2.0), (1, 0.0)]);
        assert_eq!(chosen, Some((0, 2.0)));
    }
}

/// Top-K build candidates by 1-ply score. Used by MCTS to get a wider prior.
pub(crate) fn score_top_builds(
    state: &mut GameState,
    ctx: &EvalContext,
    k: usize,
    plan: &Plan,
    targets: &[BuildTarget],
    keep_scores: &[(usize, f64)],
) -> Vec<Decision> {
    let pid = ctx.pid;
    let mut scored: Vec<(BuildTarget, ScoreParts)> = targets
        .iter()
        .cloned()
        .map(|t| {
            let s = score_build_candidate(state, ctx, &t, plan);
            (t, s)
        })
        .collect();
    scored.sort_by(|a, b| b.1.total(ctx).total_cmp(&a.1.total(ctx)));
    scored.truncate(k);
    let mut out = Vec::new();
    for (cand, parts) in scored {
        let score = parts.total(ctx);
        if score == f64::NEG_INFINITY {
            continue;
        }
        if let Some((card_index, card_score)) = pick_build_card(state, pid, &cand, keep_scores) {
            let coal_needed = cand.cost_coal as usize;
            let iron_needed = cand.cost_iron as usize;
            // v4: emit up to SOURCE_VARIANTS variants of this geometry that
            // differ in free-source identity; search resolves whose buildings
            // flip, the generator only guarantees the alternatives exist.
            let coal_opts = super::distinct_source_options(
                crate::rules::coal_source_options(state, cand.loc, coal_needed),
                |s: &crate::graph::CoalSource| (s.kind, s.key),
                super::SOURCE_VARIANTS,
            );
            let iron_opts = super::distinct_source_options(
                crate::rules::iron_source_options(state, iron_needed),
                |s: &crate::graph::IronSource| (s.key, s.free),
                super::SOURCE_VARIANTS,
            );
            let empty: Vec<crate::graph::CoalSource> = Vec::new();
            let coal_opts: Vec<Vec<crate::graph::CoalSource>> = if coal_opts.is_empty() {
                vec![empty]
            } else {
                coal_opts
            };
            let empty_iron: Vec<crate::graph::IronSource> = Vec::new();
            let iron_opts: Vec<Vec<crate::graph::IronSource>> = if iron_opts.is_empty() {
                vec![empty_iron]
            } else {
                iron_opts
            };
            let mut variants = 0usize;
            'variants: for coal in &coal_opts {
                for iron in &iron_opts {
                    out.push(Decision {
                        mv: ResolvedMove::Build {
                            loc: cand.loc,
                            slot_index: cand.slot_index,
                            ind: cand.ind,
                            coal: coal.clone(),
                            iron: iron.clone(),
                            card_index,
                        },
                        score,
                        card_score,
                    });
                    variants += 1;
                    if variants >= super::SOURCE_VARIANTS {
                        break 'variants;
                    }
                }
            }
        }
    }
    out
}
