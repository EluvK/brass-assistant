//! Heuristic AI baseline: translated from reference/npow-brass-birmingham/js/aiPlayer.js
//!
//! Stateless greedy evaluator. Reads GameState/legal moves (never mutates the
//! state beyond what action generation needs) and returns the Move with the
//! highest VP-equivalent score. One-ply, no search.

use crate::data::{Era, IndustryType};
use crate::graph::{connected_locations, find_iron_sources, is_in_network};
use crate::map::{city_slots, connections, Loc};
use crate::rules::{
    can_develop, can_scout, get_valid_build_targets, get_valid_network_targets,
    get_valid_second_rail_links, get_valid_sell_targets, valid_build_cards, BuildTarget, Move,
};
use crate::state::{Card, GameState};
use rand::Rng;

// Scoring weights (from aiPlayer.js).
const VP_WEIGHT: f64 = 1.0;
const MONEY_WEIGHT: f64 = 0.12;
const BASE_INCOME_WEIGHT: f64 = 0.35;
const FLEX_WEIGHT: f64 = 0.8;

pub struct Decision {
    pub mv: Move,
    pub score: f64,
}

pub fn choose_action(state: &mut GameState<impl Rng>) -> Decision {
    candidate_actions(state)
        .into_iter()
        .max_by(|a, b| a.score.partial_cmp(&b.score).unwrap())
        .unwrap_or_else(|| pass_decision(state))
}

/// Best move per action type (the 1-ply candidate set). Used both by
/// `choose_action` and by the 2-ply lookahead in search_ai.
pub fn candidate_actions(state: &mut GameState<impl Rng>) -> Vec<Decision> {
    candidate_actions_k(state, 1)
}

/// Top-K candidates per action type, for MCTS to get a wider prior.
/// Build and Network get up to `k` candidates each; other action types keep
/// their single best (develop/sell/loan/scout/pass).
pub fn candidate_actions_k(state: &mut GameState<impl Rng>, k: usize) -> Vec<Decision> {
    let pid = state.current_player_id();
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

/// Position-value estimator for player `pid`, used as the MCTS leaf evaluator.
/// Combines accumulated VP + income stream + cash with the player's best
/// available move score (1-ply), which reflects actionable potential better
/// than a pure board snapshot.
pub(crate) fn evaluate_position(state: &GameState<impl Rng>, pid: usize) -> f64 {
    let p = &state.players[pid];
    let mut value = 0.0;

    value += p.vp as f64 * VP_WEIGHT;
    value += p.money as f64 * MONEY_WEIGHT;
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
    let proxy = state.deck.len() as f64 / state.player_count() as f64;
    proxy.clamp(1.0, 10.0)
}

pub(crate) fn income_weight(state: &GameState<impl Rng>) -> f64 {
    BASE_INCOME_WEIGHT * (estimate_rounds_remaining(state) / 5.0)
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn vp_equivalent(
    state: &GameState<impl Rng>,
    vp: f64,
    income: f64,
    money: f64,
    flex: f64,
) -> f64 {
    vp * VP_WEIGHT + income * income_weight(state) + money * MONEY_WEIGHT + flex * FLEX_WEIGHT
}

// ---------------------------------------------------------------------------
// BUILD
// ---------------------------------------------------------------------------

fn estimate_flip_probability(
    state: &GameState<impl Rng>,
    _pid: usize,
    ind: IndustryType,
    city_id: Loc,
) -> f64 {
    let base = if matches!(ind, IndustryType::CoalMine | IndustryType::IronWorks) {
        0.7
    } else if ind == IndustryType::Brewery {
        0.5
    } else {
        let mut b = 0.6f64;
        let connected = connected_locations(state, city_id);
        let has_reachable_merchant = state.merchants.iter().any(|mt| {
            connected.contains(&mt.loc) && mt.accepts(ind)
        });
        if has_reachable_merchant {
            b += 0.2;
        }
        if estimate_rounds_remaining(state) < 2.0 {
            b -= 0.3;
        }
        b
    };
    base.clamp(0.1, 1.0)
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

fn score_build_candidate(
    state: &GameState<impl Rng>,
    pid: usize,
    cand: &BuildTarget,
) -> f64 {
    let tile = state.players[pid].next_tile(cand.ind);
    let Some(tile) = tile else { return f64::NEG_INFINITY };

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
    let network_expansion =
        0.1 * count_new_unbuilt_neighbor_connections(state, cand.loc) as f64;

    vp_equivalent(
        state,
        tile.vp as f64 * flip_prob + link_self_value,
        tile.income as f64 * flip_prob,
        -(cand.cost_total as f64),
        0.0,
    ) + resource_self_sufficiency
        + network_expansion
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
            out.push(Decision {
                mv: Move::Build {
                    loc: cand.loc,
                    slot_index: cand.slot_index,
                    ind: cand.ind,
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

fn score_network_candidate(
    state: &GameState<impl Rng>,
    pid: usize,
    _conn_id: usize,
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
    let has_industry = state
        .city_tiles
        .iter()
        .flatten()
        .any(|t| t.player == pid);
    let player = &state.players[pid];
    let links_built = if is_canal {
        14 - player.canal_links
    } else {
        14 - player.rail_links
    };
    let over_networking_penalty = if !has_industry && links_built >= 1 {
        1.0
    } else {
        0.0
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
        let cost = if state.era == Era::Canal {
            crate::map::CANAL_LINK_COST
        } else {
            crate::map::RAIL_LINK_COST
        };
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
            Some(Decision {
                mv: Move::Network {
                    conn_id,
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
            // Value double-rail similarly; cost flat 15 + coal (approx).
            let c2 = &connections()[conn2];
            let s = score_network_candidate(
                state,
                pid,
                conn2,
                crate::map::RAIL_DOUBLE_LINK_COST,
                &[c2.a, c2.b],
                false,
            );
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
    let card_index = pick_any_card(state, pid)?;
    Some(Decision {
        mv: Move::NetworkDouble {
            conn1,
            conn2,
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
    let types = state.players[pid].developable_types();
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
            let urgency = if !tile.rail_era { 2.0 } else { 0.5 };
            let unlock_value = 0.3 * tile.level as f64;
            (ind, urgency + unlock_value, tile.level)
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

    let score = vp_equivalent(
        state,
        first.1 + if let Some(_) = second { scored[1].1 } else { 0.0 },
        0.0,
        -(iron_cost as f64),
        0.0,
    );

    let card_index = pick_any_card(state, pid)?;
    Some(Decision {
        mv: Move::Develop {
            ind1: first.0,
            ind2: second,
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
                crate::map::MerchantBonus::Money(m) => m as f64 * MONEY_WEIGHT,
                crate::map::MerchantBonus::Income(spaces) => {
                    spaces as f64 * income_weight(state)
                }
                crate::map::MerchantBonus::Develop(_) => 0.5,
            };
        }
    }
    0.0
}

fn score_sell_plan(state: &GameState<impl Rng>, pid: usize) -> Option<Decision> {
    let targets = get_valid_sell_targets(state, pid);
    if targets.is_empty() {
        return None;
    }

    // Rank targets, then sell all (matches aiPlayer.js sellPlan behavior).
    let mut ranked: Vec<(usize, f64, f64)> = targets
        .iter()
        .map(|t| {
            let bonus = best_merchant_beer_bonus(state, &t.merchant_indices);
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

    let keys: Vec<usize> = ranked.iter().map(|r| r.0).collect();
    // align use_merchant to keys
    let mut use_map = std::collections::HashMap::new();
    for (i, t) in targets.iter().enumerate() {
        let _ = i;
        use_map.insert(t.key, t.merchant_beer_available);
    }
    let use_merchant: Vec<bool> = keys.iter().map(|k| *use_map.get(k).unwrap_or(&false)).collect();

    let total_vp: f64 = ranked.iter().map(|r| r.1).sum();
    let total_income: f64 = targets
        .iter()
        .map(|t| {
            state.city_tiles[t.key]
                .as_ref()
                .map_or(0.0, |tile| tile.def.income as f64)
        })
        .sum();
    let total_bonus: f64 = ranked.iter().map(|r| r.2).sum();

    let score = vp_equivalent(state, total_vp, total_income, 0.0, 0.0) + total_bonus;

    let card_index = pick_any_card(state, pid)?;
    Some(Decision {
        mv: Move::Sell {
            keys,
            use_merchant_beer: use_merchant,
            card_index,
        },
        score,
    })
}

// ---------------------------------------------------------------------------
// LOAN
// ---------------------------------------------------------------------------

fn cheapest_affordable_build_or_network_cost(state: &GameState<impl Rng>, pid: usize) -> f64 {
    let mut cheapest = f64::INFINITY;
    for t in get_valid_build_targets(state, pid) {
        if (t.cost_total as f64) < cheapest {
            cheapest = t.cost_total as f64;
        }
    }
    let player = &state.players[pid];
    if state.era == Era::Canal && player.canal_links > 0
        || state.era == Era::Rail && player.rail_links > 0
    {
        for _conn in get_valid_network_targets(state, pid) {
            let cost = if state.era == Era::Canal {
                crate::map::CANAL_LINK_COST
            } else {
                crate::map::RAIL_LINK_COST
            };
            if (cost as f64) < cheapest {
                cheapest = cost as f64;
            }
        }
    }
    cheapest
}

fn score_loan_result(state: &GameState<impl Rng>, pid: usize) -> Option<Decision> {
    if !state.can_take_loan(pid) {
        return None;
    }
    let player = &state.players[pid];
    let income_level = player.income_level();
    let cheapest = cheapest_affordable_build_or_network_cost(state, pid);
    let cash_crunch_bonus = if (player.money as f64) < cheapest {
        4.0
    } else {
        0.0
    };
    let already_flush_penalty = if (player.money as f64) > cheapest * 2.5 {
        2.5
    } else {
        0.0
    };
    let income_floor_penalty = if income_level <= (crate::map::MIN_INCOME + 5) {
        2.5
    } else {
        0.0
    };

    let score = vp_equivalent(
        state,
        0.0,
        -(crate::map::LOAN_INCOME_PENALTY as f64),
        crate::map::LOAN_AMOUNT as f64 * 0.25,
        0.0,
    ) + cash_crunch_bonus
        - income_floor_penalty
        - already_flush_penalty;

    let card_index = pick_any_card(state, pid)?;
    Some(Decision {
        mv: Move::Loan { card_index },
        score,
    })
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
