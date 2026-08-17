//! Legal move enumeration for raw execution and policy slots.

use crate::data::{Era, IndustryType};
use crate::gameplay::actions::{
    SellTarget, affordable_develop_iron_count, any_card_indices, can_develop, can_scout,
    coal_options_for_connection, coal_source_options, execute_sell, get_second_rail_options,
    get_valid_build_targets, get_valid_network_targets, get_valid_sell_targets,
    iron_source_options, valid_build_cards,
};
use crate::map::connections;
use crate::r#move::Move;
use crate::state::GameState;
// ---------------------------------------------------------------------------
// Legal move enumeration
// ---------------------------------------------------------------------------

/// Expansion mode for legal-move enumeration.
///
/// `All` emits every legal source/card combination (the raw move space);
/// `OnePerSlot` emits a single representative move per policy slot (the
/// MCTS / policy-head fast path). Both modes share ONE generator below, so
/// their coverage can never drift apart.
#[derive(Clone, Copy)]
enum MoveExpansion {
    All,
    OnePerSlot,
}

impl MoveExpansion {
    /// Yield the full slice (`All`) or just its first element (`OnePerSlot`).
    fn each<'a, T>(self, items: &'a [T]) -> impl Iterator<Item = &'a T> {
        items.iter().take(match self {
            MoveExpansion::All => items.len(),
            MoveExpansion::OnePerSlot => 1,
        })
    }
}

pub fn legal_moves(state: &mut GameState) -> Vec<Move> {
    generate_moves(state, MoveExpansion::All)
}

/// One executable representative for every raw action choice class. Policy
/// encoding intentionally lives in `bridge::policy`; gameplay only decides
/// which moves are legal.
pub(crate) fn legal_policy_representatives(state: &mut GameState) -> Vec<Move> {
    generate_moves(state, MoveExpansion::OnePerSlot)
}

/// The single legal-move generator shared by `legal_moves` (raw) and
/// `legal_slot_moves` (one representative per policy slot). Every legality
/// decision (build targets, network targets, double-rail options, develop
/// sets, sell plans, scout combos) lives exactly here; the expansion mode only
/// decides whether source/card choices are enumerated fully or truncated to
/// their first representative.
fn generate_moves(state: &mut GameState, exp: MoveExpansion) -> Vec<Move> {
    state.ensure_network_masks();
    let pid = state.current_player_id();

    if let Some(crate::state::PendingBonus::FreeDevelop { player_id, count }) = state.pending_bonus
    {
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
    let cards = any_card_indices(&state.players[pid]);

    // BUILD
    for t in get_valid_build_targets(state, pid) {
        let coal_needed = t.cost_coal as usize;
        let iron_needed = t.cost_iron as usize;
        let coal_opts = coal_source_options(state, t.loc, coal_needed);
        let iron_opts = iron_source_options(state, iron_needed);
        for coal in exp.each(&coal_opts) {
            for iron in exp.each(&iron_opts) {
                for ci in exp.each(&valid_build_cards(
                    state,
                    &state.players[pid],
                    pid,
                    t.loc,
                    t.ind,
                )) {
                    moves.push(Move::Build {
                        loc: t.loc,
                        slot_index: t.slot_index,
                        ind: t.ind,
                        coal: coal.clone(),
                        iron: iron.clone(),
                        card_index: *ci,
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
            for ci in exp.each(&cards) {
                if state.era == Era::Canal {
                    moves.push(Move::Network {
                        conn_id: conn,
                        coal: None,
                        card_index: *ci,
                    });
                } else {
                    let c = &connections()[conn];
                    let coal_opts = coal_options_for_connection(state, c, 1);
                    for coal in exp.each(&coal_opts) {
                        moves.push(Move::Network {
                            conn_id: conn,
                            coal: coal.first().copied(),
                            card_index: *ci,
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
            for coal1 in exp.each(&coal1_opts) {
                let Some(&coal1) = coal1.first() else {
                    continue;
                };
                for opt in get_second_rail_options(state, pid, conn1, coal1) {
                    for coal2 in exp.each(&opt.coal2_opts) {
                        for beer in exp.each(&opt.beers) {
                            for ci in exp.each(&cards) {
                                moves.push(Move::NetworkDouble {
                                    conn1,
                                    conn2: opt.conn,
                                    coal1,
                                    coal2: coal2[0],
                                    beer: *beer,
                                    card_index: *ci,
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
                for iron in exp.each(&iron_opts) {
                    for ci in exp.each(&cards) {
                        moves.push(Move::Develop {
                            ind1,
                            ind2: None,
                            iron: iron.clone(),
                            card_index: *ci,
                        });
                    }
                }
            }
            // Double develop: the second choice is evaluated after removing
            // the first tile, which matters when both choices use one industry.
            if affordable_iron >= 2 {
                let mut player_after_first = state.players[pid].clone();
                player_after_first
                    .consume_tile(ind1)
                    .expect("developable first tile must be present");
                let second_types: Vec<IndustryType> = player_after_first
                    .developable_types()
                    .into_iter()
                    .map(|(ind, _)| ind)
                    .collect();
                let iron_opts = iron_source_options(state, 2);
                for &ind2 in &second_types {
                    for iron in exp.each(&iron_opts) {
                        for ci in exp.each(&cards) {
                            moves.push(Move::Develop {
                                ind1,
                                ind2: Some(ind2),
                                iron: iron.clone(),
                                card_index: *ci,
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
        // Single-tile sells (one tile per action). Each route is a distinct
        // merchant/beer option, so routes are never collapsed.
        for t in &sell_targets {
            for route in &t.routes {
                for ci in exp.each(&cards) {
                    moves.push(Move::Sell {
                        keys: vec![t.key],
                        merchant_indices: vec![route.merchant_index],
                        use_merchant_beer: vec![route.use_merchant_beer],
                        card_index: *ci,
                    });
                }
            }
        }
        // Multi-tile sells: bounded subsets (2..=3 tiles) with compatible routes.
        let multi_plans = build_multi_sell_plans(state, pid, &sell_targets);
        for ci in exp.each(&cards) {
            for plan in &multi_plans {
                moves.push(Move::Sell {
                    keys: plan.0.clone(),
                    merchant_indices: plan.1.clone(),
                    use_merchant_beer: plan.2.clone(),
                    card_index: *ci,
                });
            }
        }
    }

    // LOAN
    if state.can_take_loan(pid) {
        for ci in exp.each(&cards) {
            moves.push(Move::Loan { card_index: *ci });
        }
    }

    // SCOUT
    if can_scout(state, pid) {
        // All 3-card combinations
        let n = state.players[pid].hand.len();
        let mut combos = Vec::new();
        for i in 0..n {
            for j in (i + 1)..n {
                for k in (j + 1)..n {
                    combos.push([i, j, k]);
                }
            }
        }
        for &indices in exp.each(&combos) {
            moves.push(Move::Scout {
                card_indices: indices,
            });
        }
    }

    // PASS
    for ci in exp.each(&cards) {
        moves.push(Move::Pass { card_index: *ci });
    }

    moves
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
fn build_multi_sell_plans(
    state: &GameState,
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
                if route.use_merchant_beer
                    && committed_merchant_beer.contains(&route.merchant_index)
                {
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
        if !keys.iter().all(|&key| {
            sim.city_tiles[key]
                .as_ref()
                .map(|tile| tile.flipped)
                .unwrap_or(false)
        }) {
            continue;
        }
        if seen.insert(keys.clone()) {
            out.push((keys, merchants, use_beer));
        }
    }
    out
}
