//! Network (link) action scoring: single links and double-rail combos.

use super::board::{hand_access_gain, touches_merchant};
use super::context::{define_era_round_factor, define_round_factor};
use super::plan::Plan;
use super::value::link_current_and_potential_vps;
use super::{CardChoices, Decision};
use crate::map::{CANAL_LINK_COST, connections};
use crate::rules::{ResolvedMove, enumerate_double_rail_candidates, get_valid_network_targets};
use crate::state::GameState;

// Scores in this module are already in VP equivalents. Keep the policy close
// to the action it ranks: unlike general value conversion, none of these
// coefficients are shared by another scorer.
const LOCATION_CARD_ACCESS_SCORE: f64 = 0.48;
const INDUSTRY_CARD_ACCESS_SCORE: f64 = 0.08;
const MERCHANT_CONNECTION_BONUS: f64 = 1.5;
const EXPLORATION_BASE: f64 = 1.6;
const EXPLORATION_PER_REMAINING_LINK: f64 = 0.3;
const PLAN_CAPACITY_BONUS: f64 = 0.5;
const DOUBLE_FARM_LOCK_BONUS: f64 = 0.8;
const CASH_VALUE_BASE: f64 = 0.12;
const EARLY_PHASE_CASH_MULTIPLIER: f64 = 0.8;
const RAIL_LATE_CASH_MULTIPLIER: f64 = 5.0 / 3.0;

// A future link icon has more time to be realised early in an era. Link
// scoring is deliberately weak in Canal, then eases from the Rail expansion
// rush into the Rail finish. Cash spent on rails becomes more urgent late.
define_round_factor!(FutureLinkVpDiscount, 0.8, 0.3);
define_era_round_factor!(
    LinkVpWeight,
    canal: (0.1, 0.1),
    rail: (1.0, 0.85),
);
define_era_round_factor!(
    LinkCostWeight,
    canal: (
        CASH_VALUE_BASE * EARLY_PHASE_CASH_MULTIPLIER,
        CASH_VALUE_BASE * EARLY_PHASE_CASH_MULTIPLIER
    ),
    rail: (
        CASH_VALUE_BASE * EARLY_PHASE_CASH_MULTIPLIER,
        CASH_VALUE_BASE * RAIL_LATE_CASH_MULTIPLIER
    ),
);
define_era_round_factor!(
    BeerFarmLockBonus,
    canal: (0.0, 0.0),
    rail: (1.2, 0.0),
);
define_era_round_factor!(
    DoubleRailTempoBonus,
    canal: (0.0, 0.0),
    rail: (1.2, 0.6),
);

fn coal_effective_price(src: &crate::graph::CoalSource) -> i32 {
    if src.free { 0 } else { src.price as i32 }
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

fn connection_touches_farm(conn_id: usize) -> bool {
    let connection = &connections()[conn_id];
    connection.a.is_farm()
        || connection.b.is_farm()
        || connection.via_farm.is_some_and(|farm| farm.is_farm())
}

/// Score one link directly in VP equivalents.
///
/// The score combines immediate/future link icons, newly playable cards,
/// actual resource cost, and positional bonuses. It intentionally has no
/// shared evaluation context: every weight needed to compare network actions
/// is defined above and its time-dependent part reads from `state`.
fn score_network_candidate(
    state: &GameState,
    pid: usize,
    conn_id: usize,
    cost: i32,
    plan: &Plan,
) -> f64 {
    let is_canal = state.is_canal_era();
    let connection = &connections()[conn_id];
    let cities = [connection.a, connection.b];

    // Hand cards that become playable once these cities join the network.
    let (loc_cards, ind_cards) = hand_access_gain(state, pid, &cities);
    let hand_access = loc_cards as f64 * LOCATION_CARD_ACCESS_SCORE
        + ind_cards as f64 * INDUSTRY_CARD_ACCESS_SCORE;

    let (current_link_vp, future_link_vp) = link_current_and_potential_vps(state, conn_id, &cities);
    let link_vp = (current_link_vp + FutureLinkVpDiscount::factor(state) * future_link_vp)
        * LinkVpWeight::factor(state);

    let links_left = if is_canal {
        14 - state.players[pid].canal_links
    } else {
        14 - state.players[pid].rail_links
    };
    // Exploration prior: the first links into empty regions are premium.
    let exploration =
        (EXPLORATION_BASE - EXPLORATION_PER_REMAINING_LINK * links_left as f64).max(0.0);

    // Plan ("流派") bonus: a link touching a city with a vacant slot for the
    // plan industry opens production capacity. Only from Canal-Late onward
    // (Canal-Early builds the economy engine first). A tiebreaker.
    let mut plan_bonus = 0.0;
    if plan.count > 0
        && (!is_canal || state.round > 4)
        && state.players[pid].remaining_count(plan.industry) > 0
    {
        'outer: for loc in &cities {
            if !loc.is_city() {
                continue;
            }
            for (slot_idx, allowed) in crate::map::city_slots(*loc).iter().enumerate() {
                if !allowed.contains(&plan.industry) {
                    continue;
                }
                if let Some(k) = state.city_slot_key(*loc, slot_idx)
                    && state.city_tiles[k].is_none()
                {
                    plan_bonus = PLAN_CAPACITY_BONUS;
                    break 'outer;
                }
            }
        }
    }

    // Rail beer-farm lock: links touching a brewery farm lock down beer
    // supply (double-rails and late-game sells depend on it).
    let beer_lock = if connection_touches_farm(conn_id) {
        BeerFarmLockBonus::factor(state)
    } else {
        0.0
    };

    let merchant_gain = if touches_merchant(&cities) {
        MERCHANT_CONNECTION_BONUS
    } else {
        0.0
    };

    link_vp + hand_access - cost as f64 * LinkCostWeight::factor(state)
        + merchant_gain
        + exploration
        + plan_bonus
        + beer_lock
}

/// Top-K single network candidates by 1-ply score.
pub(crate) fn score_top_networks(
    state: &mut GameState,
    k: usize,
    plan: &Plan,
    card_choices: &CardChoices,
) -> Vec<Decision> {
    if k == 0 || card_choices.is_empty() {
        return Vec::new();
    }

    let pid = state.current_player_id();
    let is_canal = state.is_canal_era();
    let player = &state.players[pid];
    if is_canal && player.canal_links == 0 {
        return Vec::new();
    }
    if !is_canal && player.rail_links == 0 {
        return Vec::new();
    }
    let targets = get_valid_network_targets(state, pid);
    let mut scored = Vec::with_capacity(targets.len());
    for conn_id in targets {
        let mut cost = if is_canal {
            CANAL_LINK_COST
        } else {
            crate::map::RAIL_LINK_COST
        };
        if !is_canal {
            // Rail links consume 1 coal; charging only the base link cost
            // would massively overvalue network spam and undervalue coal
            // supply actions.
            cost += estimated_connection_coal_cost(state, conn_id);
        }
        let score = score_network_candidate(state, pid, conn_id, cost, plan);
        scored.push((conn_id, score));
    }
    scored.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
    scored.truncate(k);
    let card_index = card_choices[0].0;
    let mut out = Vec::new();
    for (conn_id, score) in scored {
        // v4: default (cheapest) coal plus up to SOURCE_VARIANTS-1 distinct
        // free-source identities per connection.
        let coal_variants: Vec<Option<crate::graph::CoalSource>> = if !is_canal {
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
                score,
                card_score: card_choices[0].1,
            });
        }
    }
    out
}

pub(crate) fn score_top_network_doubles(
    state: &mut GameState,
    k: usize,
    plan: &Plan,
    card_choices: &CardChoices,
) -> Vec<Decision> {
    if k == 0 || card_choices.is_empty() || state.is_canal_era() {
        return Vec::new();
    }
    let pid = state.current_player_id();
    if state.players[pid].rail_links < 2 {
        return Vec::new();
    }
    let mut scored = Vec::new();
    for candidate in enumerate_double_rail_candidates(state, pid) {
        // Value both links and charge realistic resource cost:
        // total = £15 base + 2 coal (explicitly estimated from legal
        // sources). The double-rail surcharge (£15 vs 2×£5) uses the same
        // era-sensitive cash weight as each individual link.
        let cost1 = crate::map::RAIL_LINK_COST + coal_effective_price(&candidate.coal1);
        let cost2 = crate::map::RAIL_LINK_COST + coal_effective_price(&candidate.coal2);
        let s1 = score_network_candidate(state, pid, candidate.conn1, cost1, plan);
        let s2 = score_network_candidate(state, pid, candidate.conn2, cost2, plan);
        let surcharge = (crate::map::RAIL_DOUBLE_LINK_COST - 2 * crate::map::RAIL_LINK_COST) as f64
            * LinkCostWeight::factor(state);
        let mut total = s1 + s2 - surcharge;
        // Double-rail synergy: one action builds two links (tempo win),
        // strongest early in Rail while the net is being laid out.
        total += DoubleRailTempoBonus::factor(state);
        // Beer-lock synergy: grabbing a brewery farm link locks beer
        // while saving tempo.
        if connection_touches_farm(candidate.conn1) || connection_touches_farm(candidate.conn2) {
            total += DOUBLE_FARM_LOCK_BONUS;
        }
        scored.push((candidate, total));
    }
    scored.sort_by(|a, b| b.1.total_cmp(&a.1));
    let card_index = card_choices[0].0;
    let mut out = Vec::with_capacity(k);
    for (candidate, score) in scored.into_iter().take(k) {
        // v4: up to SOURCE_VARIANTS variants per geometry, differing in coal1
        // identity; the best coal1 also varies the beer source (Own first).
        // The coal2 options must be enumerated against the SAME coal1 we
        // actually use, otherwise the emitted move may fail to execute.
        out.push(Decision {
            mv: candidate.to_move(card_index),
            score,
            card_score: card_choices[0].1,
        });
    }
    out
}
