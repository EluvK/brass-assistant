//! Legal concrete move enumeration for execution and candidate scoring.

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

/// Enumerate every complete executable move, including resource and card
/// choices. Candidate scoring intentionally keeps these distinctions.
pub fn legal_moves(state: &mut GameState) -> Vec<Move> {
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
        for coal in &coal_opts {
            for iron in &iron_opts {
                for ci in &valid_build_cards(state, &state.players[pid], pid, t.loc, t.ind) {
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
            for ci in &cards {
                if state.era == Era::Canal {
                    moves.push(Move::Network {
                        conn_id: conn,
                        coal: None,
                        card_index: *ci,
                    });
                } else {
                    let c = &connections()[conn];
                    let coal_opts = coal_options_for_connection(state, c, 1);
                    for coal in &coal_opts {
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
            for coal1 in &coal1_opts {
                let Some(&coal1) = coal1.first() else {
                    continue;
                };
                for opt in get_second_rail_options(state, pid, conn1, coal1) {
                    for coal2 in &opt.coal2_opts {
                        for beer in &opt.beers {
                            for ci in &cards {
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
                for iron in &iron_opts {
                    for ci in &cards {
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
                    for iron in &iron_opts {
                        for ci in &cards {
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

    // SELL
    let sell_targets = get_valid_sell_targets(state, pid);
    if !sell_targets.is_empty() {
        // A Sell action may include any non-empty subset of sellable tiles.
        // A plan retains every per-tile merchant/beer route because they can
        // produce different merchant bonuses and consume different beer.
        let sell_plans = build_sell_plans(state, pid, &sell_targets);
        for ci in &cards {
            for plan in &sell_plans {
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
        for ci in &cards {
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
        for &indices in &combos {
            moves.push(Move::Scout {
                card_indices: indices,
            });
        }
    }

    // PASS
    for ci in &cards {
        moves.push(Move::Pass { card_index: *ci });
    }

    moves
}

/// All Sell plans: every non-empty subset of targets combined with every
/// per-target merchant route. Plans are dry-run validated because beer and
/// merchant-beer availability are shared across the tiles in one action.
///
/// Keys are emitted in target order, which is ascending city-slot key order,
/// so an unordered set of sold tiles has one canonical representation.
fn build_sell_plans(
    state: &GameState,
    pid: usize,
    targets: &[SellTarget],
) -> Vec<(Vec<usize>, Vec<usize>, Vec<bool>)> {
    let mut out: Vec<(Vec<usize>, Vec<usize>, Vec<bool>)> = Vec::new();
    let mut keys = Vec::new();
    let mut merchants = Vec::new();
    let mut use_beer = Vec::new();
    enumerate_sell_plans(
        state,
        pid,
        targets,
        0,
        &mut keys,
        &mut merchants,
        &mut use_beer,
        &mut out,
    );
    out
}

#[allow(clippy::too_many_arguments)]
fn enumerate_sell_plans(
    state: &GameState,
    pid: usize,
    targets: &[SellTarget],
    target_index: usize,
    keys: &mut Vec<usize>,
    merchants: &mut Vec<usize>,
    use_beer: &mut Vec<bool>,
    out: &mut Vec<(Vec<usize>, Vec<usize>, Vec<bool>)>,
) {
    if target_index == targets.len() {
        if keys.is_empty() {
            return;
        }
        let mut sim = state.clone();
        if execute_sell(&mut sim, pid, keys, merchants, use_beer, 0).is_ok()
            && keys.iter().all(|&key| {
                sim.city_tiles[key]
                    .as_ref()
                    .map(|tile| tile.flipped)
                    .unwrap_or(false)
            })
        {
            out.push((keys.clone(), merchants.clone(), use_beer.clone()));
        }
        return;
    }

    // This target is not part of the action.
    enumerate_sell_plans(
        state,
        pid,
        targets,
        target_index + 1,
        keys,
        merchants,
        use_beer,
        out,
    );

    // This target is sold through each of its distinct merchant routes.
    let target = &targets[target_index];
    for route in &target.routes {
        keys.push(target.key);
        merchants.push(route.merchant_index);
        use_beer.push(route.use_merchant_beer);
        enumerate_sell_plans(
            state,
            pid,
            targets,
            target_index + 1,
            keys,
            merchants,
            use_beer,
            out,
        );
        keys.pop();
        merchants.pop();
        use_beer.pop();
    }
}
