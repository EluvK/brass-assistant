//! Network (link) action scoring: single links and double-rail combos.

use super::board::{hand_access_gain, touches_merchant};
use super::context::EvalContext;
use super::plan::{Phase, Plan};
use super::value::{ScoreParts, link_current_and_potential_vps};
use super::{CardChoices, Decision};
use crate::graph::BeerSourceKind;
use crate::map::{CANAL_LINK_COST, connections};
use crate::rules::{
    ResolvedMove, get_second_rail_options, get_valid_network_targets, get_valid_second_rail_links,
};
use crate::state::GameState;

fn coal_effective_price(src: &crate::graph::CoalSource) -> i32 {
    if src.free {
        0
    } else {
        src.price as i32
    }
}

fn cheapest_connection_coal_source(
    state: &GameState,
    conn_id: usize,
) -> Option<crate::graph::CoalSource> {
    let conn = &connections()[conn_id];
    crate::rules::coal_options_for_connection(state, conn, 1)
        .into_iter()
        .flatten()
        .min_by_key(coal_effective_price)
}

fn estimated_connection_coal_cost(state: &GameState, conn_id: usize) -> i32 {
    cheapest_connection_coal_source(state, conn_id)
        .map(|s| coal_effective_price(&s))
        .unwrap_or(crate::map::COAL_EMPTY_PRICE as i32)
}

/// Score one link in VP-equivalent components.
///
/// - `vp`: link icons available now plus future icons, discounted by how
///   much era remains and scaled by the phase's network weight.
/// - `flex`: hand cards that become playable through the new connection.
/// - `money`: link cost (canal/rail base + estimated coal in rail era).
/// - `strategic`: merchant access, exploration, plan alignment, beer locks.
fn score_network_candidate(
    state: &GameState,
    ctx: &EvalContext,
    conn_id: usize,
    cost: i32,
    cities: &[crate::map::Loc; 2],
    plan: &Plan,
) -> ScoreParts {
    let w = &ctx.cfg.network;

    // Hand cards that become playable once these cities join the network.
    let (loc_cards, ind_cards) = hand_access_gain(state, ctx.pid, cities);
    let flex =
        loc_cards as f64 * w.access_per_location_card + ind_cards as f64 * w.access_per_industry_card;

    let (current_link_vp, future_link_vp) = link_current_and_potential_vps(state, conn_id, cities);
    // Network-era weighting: Rail-Early builds the net everything scores
    // through, so its links are worth more.
    let vp = (current_link_vp + ctx.future_discount() * future_link_vp) * ctx.profile.network_w;

    let links_left = if ctx.is_canal() {
        14 - state.players[ctx.pid].canal_links
    } else {
        14 - state.players[ctx.pid].rail_links
    };
    // Exploration prior: the first links into empty regions are premium.
    let exploration = (w.exploration_base - w.exploration_per_link * links_left as f64).max(0.0);

    // Plan ("流派") bonus: a link touching a city with a vacant slot for the
    // plan industry opens production capacity. Only from Canal-Late onward
    // (Canal-Early builds the economy engine first). A tiebreaker.
    let mut plan_bonus = 0.0;
    if plan.count > 0
        && ctx.phase != Phase::CanalEarly
        && state.players[ctx.pid].remaining_count(plan.industry) > 0
    {
        'outer: for loc in cities {
            if !loc.is_city() {
                continue;
            }
            for (slot_idx, allowed) in crate::map::city_slots(*loc).iter().enumerate() {
                if !allowed.contains(&plan.industry) {
                    continue;
                }
                if let Some(k) = state.city_slot_key(*loc, slot_idx)
                    && state.city_tiles[k].is_none() {
                        plan_bonus = w.plan_bonus;
                        break 'outer;
                    }
            }
        }
    }

    // Rail-Early beer-farm lock: links touching a brewery farm lock down
    // beer supply (double-rails and late-game sells depend on it).
    let beer_lock = if ctx.phase == Phase::RailEarly {
        let touches_farm = cities.iter().any(|l| l.is_farm())
            || connections()[conn_id]
                .via_farm
                .is_some_and(|f| f.is_farm());
        if touches_farm {
            w.beer_lock_bonus
        } else {
            0.0
        }
    } else {
        0.0
    };

    let merchant_gain = if touches_merchant(cities) {
        w.merchant_bonus
    } else {
        0.0
    };

    ScoreParts {
        vp,
        flex,
        money: -(cost as f64),
        strategic: merchant_gain + exploration + plan_bonus + beer_lock,
        ..Default::default()
    }
}

/// Top-K single network candidates by 1-ply score.
pub(crate) fn score_top_networks(
    state: &mut GameState,
    ctx: &EvalContext,
    k: usize,
    plan: &Plan,
    card_choices: &CardChoices,
) -> Vec<Decision> {
    let player = &state.players[ctx.pid];
    if ctx.is_canal() && player.canal_links == 0 {
        return Vec::new();
    }
    if ctx.is_rail() && player.rail_links == 0 {
        return Vec::new();
    }
    let targets = get_valid_network_targets(state, ctx.pid);
    let mut scored: Vec<(usize, ScoreParts)> = Vec::new();
    for conn_id in targets {
        let conn = &connections()[conn_id];
        let mut cost = if ctx.is_canal() {
            CANAL_LINK_COST
        } else {
            crate::map::RAIL_LINK_COST
        };
        if ctx.is_rail() {
            // Rail links consume 1 coal; charging only the base link cost
            // would massively overvalue network spam and undervalue coal
            // supply actions.
            cost += estimated_connection_coal_cost(state, conn_id);
        }
        let parts = score_network_candidate(
            state,
            ctx,
            conn_id,
            cost,
            &[conn.a, conn.b],
            plan,
        );
        scored.push((conn_id, parts));
    }
    scored.sort_by(|a, b| {
        b.1.total(ctx)
            .total_cmp(&a.1.total(ctx))
            .then(a.0.cmp(&b.0))
    });
    scored.truncate(k);
    let Some(card_index) = card_choices.first().map(|(index, _)| *index) else {
        return Vec::new();
    };
    let mut out = Vec::new();
    for (conn_id, parts) in scored {
        // v4: default (cheapest) coal plus up to SOURCE_VARIANTS-1 distinct
        // free-source identities per connection.
        let coal_variants: Vec<Option<crate::graph::CoalSource>> = if ctx.is_rail() {
            let mut variants = vec![cheapest_connection_coal_source(state, conn_id)];
            for option in super::distinct_source_options(
                crate::rules::coal_options_for_connection(state, &connections()[conn_id], 1),
                |s: &crate::graph::CoalSource| (s.kind, s.key),
                super::SOURCE_VARIANTS,
            ) {
                if let Some(source) = option.into_iter().next() {
                    if !variants.contains(&Some(source)) {
                        variants.push(Some(source));
                    }
                }
                if variants.len() >= super::SOURCE_VARIANTS {
                    break;
                }
            }
            variants
        } else {
            vec![None]
        };
        for coal in coal_variants {
            out.push(Decision {
                mv: ResolvedMove::Network {
                    conn_id,
                    coal,
                    card_index,
                },
                score: parts.total(ctx),
                card_score: card_choices
                    .first()
                    .map(|(_, s)| *s)
                    .unwrap_or(f64::INFINITY),
            });
        }
    }
    out
}

pub(crate) fn score_top_network_doubles(
    state: &mut GameState,
    ctx: &EvalContext,
    k: usize,
    plan: &Plan,
    card_choices: &CardChoices,
) -> Vec<Decision> {
    if k == 0 || !ctx.is_rail() {
        return Vec::new();
    }
    if state.players[ctx.pid].rail_links < 2 {
        return Vec::new();
    }
    let w = &ctx.cfg.network;
    let singles = get_valid_network_targets(state, ctx.pid);
    let mut scored: Vec<(usize, usize, f64)> = Vec::new();
    for conn1 in singles {
        for conn2 in get_valid_second_rail_links(state, ctx.pid, conn1) {
            // Value both links and charge realistic resource cost:
            // total = £15 base + 2 coal (explicitly estimated from legal
            // sources). The double-rail surcharge (£15 vs 2×£5) is charged
            // as money through the same conversion as everything else.
            let c1 = &connections()[conn1];
            let c2 = &connections()[conn2];
            let cost1 = crate::map::RAIL_LINK_COST + estimated_connection_coal_cost(state, conn1);
            let cost2 = crate::map::RAIL_LINK_COST + estimated_connection_coal_cost(state, conn2);
            let s1 = score_network_candidate(state, ctx, conn1, cost1, &[c1.a, c1.b], plan);
            let s2 = score_network_candidate(state, ctx, conn2, cost2, &[c2.a, c2.b], plan);
            let surcharge = ctx.money_value(
                (crate::map::RAIL_DOUBLE_LINK_COST - 2 * crate::map::RAIL_LINK_COST) as f64
                    * w.double_surcharge_weight,
            );
            let mut total = s1.total(ctx) + s2.total(ctx) - surcharge;
            // Double-rail synergy: one action builds two links (tempo win),
            // strongest in Rail-Early when the net is being laid out.
            total += if ctx.phase == Phase::RailEarly {
                w.double_tempo_rail_early
            } else {
                w.double_tempo_other
            };
            // Beer-lock synergy: grabbing a brewery farm link locks beer
            // while saving tempo.
            let touches_farm = [&connections()[conn1], &connections()[conn2]]
                .iter()
                .any(|c| {
                    c.a.is_farm() || c.b.is_farm() || c.via_farm.is_some_and(|f| f.is_farm())
                });
            if touches_farm {
                total += w.double_farm_lock_bonus;
            }
            scored.push((conn1, conn2, total));
        }
    }
    scored.sort_by(|a, b| {
        b.2.total_cmp(&a.2)
            .then(a.0.cmp(&b.0))
            .then(a.1.cmp(&b.1))
    });
    let Some(card_index) = card_choices.first().map(|(index, _)| *index) else {
        return Vec::new();
    };
    let mut out = Vec::with_capacity(k);
    let mut geometries = 0usize;
    for (conn1, conn2, score) in scored {
        if geometries >= k {
            break;
        }
        // v4: up to SOURCE_VARIANTS variants per geometry, differing in coal1
        // identity; the best coal1 also varies the beer source (Own first).
        // The coal2 options must be enumerated against the SAME coal1 we
        // actually use, otherwise the emitted move may fail to execute.
        let coal1_opts = super::distinct_source_options(
            crate::rules::coal_options_for_connection(state, &connections()[conn1], 1),
            |s: &crate::graph::CoalSource| (s.kind, s.key),
            super::SOURCE_VARIANTS,
        );
        for (vi, mut option) in coal1_opts.into_iter().enumerate() {
            let Some(coal1) = option.pop() else { continue };
            let Some(opt) = get_second_rail_options(state, ctx.pid, conn1, coal1)
                .into_iter()
                .find(|o| o.conn == conn2)
            else {
                continue;
            };
            // Prefer consuming our own beer (advances our own income when it
            // flips) over an opponent's.
            let mut beers: Vec<crate::graph::BeerSource> = opt.beers.iter().copied().collect();
            beers.sort_by_key(|b| b.kind != BeerSourceKind::Own);
            let beer_cap = if vi == 0 { super::SOURCE_VARIANTS } else { 1 };
            beers.truncate(beer_cap);
            for beer in beers {
                let Some(coal2) = opt.coal2_opts.first().and_then(|o| o.first()).copied() else {
                    continue;
                };
                out.push(Decision {
                    mv: ResolvedMove::NetworkDouble {
                        conn1,
                        conn2,
                        coal1,
                        coal2,
                        beer,
                        card_index,
                    },
                    score,
                    card_score: card_choices
                        .first()
                        .map(|(_, s)| *s)
                        .unwrap_or(f64::INFINITY),
                });
            }
        }
        geometries += 1;
    }
    out
}
