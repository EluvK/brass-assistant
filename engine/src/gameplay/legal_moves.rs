//! Legal concrete move enumeration for execution and candidate scoring.

use crate::data::{Era, IndustryType};
use crate::gameplay::actions::{
    SellTarget, affordable_develop_iron_count, any_card_indices, can_develop, can_scout,
    coal_options_for_connection, coal_source_options, get_second_rail_options,
    get_valid_build_targets, get_valid_network_targets, get_valid_sell_targets,
    iron_source_options, valid_build_cards,
};
use crate::map::connections;
use crate::r#move::{CardCandidate, Move, ResolvedMove};
use crate::state::GameState;
// ---------------------------------------------------------------------------
// Legal move enumeration
// ---------------------------------------------------------------------------

/// Enumerate every complete executable move, including resource and card
/// choices. Candidate scoring intentionally keeps these distinctions.
pub fn legal_resolved_moves(state: &mut GameState) -> Vec<ResolvedMove> {
    state.ensure_network_masks();
    let pid = state.current_player_id();

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
                    moves.push(ResolvedMove::Build {
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
                    moves.push(ResolvedMove::Network {
                        conn_id: conn,
                        coal: None,
                        card_index: *ci,
                    });
                } else {
                    let c = &connections()[conn];
                    let coal_opts = coal_options_for_connection(state, c, 1);
                    for coal in &coal_opts {
                        moves.push(ResolvedMove::Network {
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
                                moves.push(ResolvedMove::NetworkDouble {
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
                        moves.push(ResolvedMove::Develop {
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
                            moves.push(ResolvedMove::Develop {
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
                for free_develop in sell_free_develop_options(state, pid, &plan.1, &plan.2) {
                    moves.push(ResolvedMove::Sell {
                        keys: plan.0.clone(),
                        merchant_indices: plan.1.clone(),
                        use_merchant_beer: plan.2.clone(),
                        beer_sources: plan.3.clone(),
                        free_develop,
                        card_index: *ci,
                    });
                }
            }
        }
    }

    // LOAN
    if state.can_take_loan(pid) {
        for ci in &cards {
            moves.push(ResolvedMove::Loan { card_index: *ci });
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
            moves.push(ResolvedMove::Scout {
                card_indices: indices,
            });
        }
    }

    // PASS
    for ci in &cards {
        moves.push(ResolvedMove::Pass { card_index: *ci });
    }

    moves
}

/// Enumerate structural actions once, with their legal card choices attached.
/// Resource/source choices remain part of the operation; only card selection
/// is factored out of the action space.
pub fn legal_moves(state: &mut GameState) -> Vec<Move> {
    let pid = state.current_player_id();
    let resolved = legal_resolved_moves(state);
    let mut positions: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
    let mut out: Vec<Move> = Vec::new();
    for mv in resolved {
        let (key, mut structural, card_indices) = structural_from_resolved(&mv);
        let candidates = match &mut structural {
            Move::Build {
                card_candidates, ..
            }
            | Move::Network {
                card_candidates, ..
            }
            | Move::NetworkDouble {
                card_candidates, ..
            }
            | Move::Develop {
                card_candidates, ..
            }
            | Move::Sell {
                card_candidates, ..
            }
            | Move::Loan { card_candidates }
            | Move::Scout { card_candidates }
            | Move::Pass { card_candidates } => card_candidates,
        };
        for card_index in card_indices {
            candidates.push(CardCandidate {
                hand_index: card_index,
                keep_score: card_keep_score(state, pid, card_index),
            });
        }
        if let Some(&position) = positions.get(&key) {
            let dst = match &mut out[position] {
                Move::Build {
                    card_candidates, ..
                }
                | Move::Network {
                    card_candidates, ..
                }
                | Move::NetworkDouble {
                    card_candidates, ..
                }
                | Move::Develop {
                    card_candidates, ..
                }
                | Move::Sell {
                    card_candidates, ..
                }
                | Move::Loan { card_candidates }
                | Move::Scout { card_candidates }
                | Move::Pass { card_candidates } => card_candidates,
            };
            dst.extend(candidates.clone());
        } else {
            positions.insert(key, out.len());
            out.push(structural);
        }
    }
    for mv in &mut out {
        let cards = match mv {
            Move::Build {
                card_candidates, ..
            }
            | Move::Network {
                card_candidates, ..
            }
            | Move::NetworkDouble {
                card_candidates, ..
            }
            | Move::Develop {
                card_candidates, ..
            }
            | Move::Sell {
                card_candidates, ..
            }
            | Move::Loan { card_candidates }
            | Move::Scout { card_candidates }
            | Move::Pass { card_candidates } => card_candidates,
        };
        cards.sort_by_key(|c| c.hand_index);
        cards.dedup_by_key(|c| c.hand_index);
    }
    out
}

fn card_keep_score(state: &GameState, pid: usize, index: usize) -> f64 {
    match state.players[pid].hand.get(index) {
        Some(crate::state::Card::Location(_)) => 1.15,
        Some(crate::state::Card::Industry { .. }) => 1.0,
        Some(crate::state::Card::WildLocation | crate::state::Card::WildIndustry) => 3.8,
        None => f64::INFINITY,
    }
}

fn structural_from_resolved(mv: &ResolvedMove) -> (String, Move, Vec<usize>) {
    match mv {
        ResolvedMove::Build {
            loc,
            slot_index,
            ind,
            coal,
            iron,
            card_index,
        } => (
            format!("build:{loc:?}:{slot_index}:{ind:?}:{coal:?}:{iron:?}"),
            Move::Build {
                loc: *loc,
                slot_index: *slot_index,
                ind: *ind,
                coal: coal.clone(),
                iron: iron.clone(),
                card_candidates: vec![],
            },
            vec![*card_index],
        ),
        ResolvedMove::Network {
            conn_id,
            coal,
            card_index,
        } => (
            format!("network:{conn_id}:{coal:?}"),
            Move::Network {
                conn_id: *conn_id,
                coal: *coal,
                card_candidates: vec![],
            },
            vec![*card_index],
        ),
        ResolvedMove::NetworkDouble {
            conn1,
            conn2,
            coal1,
            coal2,
            beer,
            card_index,
        } => (
            format!("network2:{conn1}:{conn2}:{coal1:?}:{coal2:?}:{beer:?}"),
            Move::NetworkDouble {
                conn1: *conn1,
                conn2: *conn2,
                coal1: *coal1,
                coal2: *coal2,
                beer: *beer,
                card_candidates: vec![],
            },
            vec![*card_index],
        ),
        ResolvedMove::Develop {
            ind1,
            ind2,
            iron,
            card_index,
        } => (
            format!("develop:{ind1:?}:{ind2:?}:{iron:?}"),
            Move::Develop {
                ind1: *ind1,
                ind2: *ind2,
                iron: iron.clone(),
                card_candidates: vec![],
            },
            vec![*card_index],
        ),
        ResolvedMove::Sell {
            keys,
            merchant_indices,
            use_merchant_beer,
            beer_sources,
            free_develop,
            card_index,
        } => (
            format!(
                "sell:{keys:?}:{merchant_indices:?}:{use_merchant_beer:?}:{beer_sources:?}:{free_develop:?}"
            ),
            Move::Sell {
                keys: keys.clone(),
                merchant_indices: merchant_indices.clone(),
                use_merchant_beer: use_merchant_beer.clone(),
                beer_sources: beer_sources.clone(),
                free_develop: *free_develop,
                card_candidates: vec![],
            },
            vec![*card_index],
        ),
        ResolvedMove::Loan { card_index } => (
            "loan".into(),
            Move::Loan {
                card_candidates: vec![],
            },
            vec![*card_index],
        ),
        ResolvedMove::Scout { card_indices } => (
            "scout".into(),
            Move::Scout {
                card_candidates: vec![],
            },
            card_indices.to_vec(),
        ),
        ResolvedMove::Pass { card_index } => (
            "pass".into(),
            Move::Pass {
                card_candidates: vec![],
            },
            vec![*card_index],
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::{legal_moves, legal_resolved_moves};
    use crate::rules::apply_move;
    use crate::state::GameState;
    use rand_chacha::ChaCha12Rng;
    use rand_chacha::rand_core::SeedableRng;

    #[test]
    fn structural_moves_factor_card_choices_and_resolve() {
        let mut state = GameState::new(ChaCha12Rng::seed_from_u64(7), 4);
        let structural = legal_moves(&mut state);
        let resolved = legal_resolved_moves(&mut state);
        assert!(!structural.is_empty());
        assert!(
            structural.len() < resolved.len(),
            "cards should not multiply structural moves"
        );
        for mv in structural {
            let n = match mv.card_requirement() {
                crate::r#move::CardRequirement::One => 1,
                crate::r#move::CardRequirement::ScoutThree => 3,
            };
            let cards: Vec<_> = mv
                .card_candidates()
                .iter()
                .take(n)
                .map(|c| c.hand_index)
                .collect();
            let resolved = mv.resolve(&cards).expect("candidate cards must resolve");
            let mut sim = state.clone();
            assert!(apply_move(&mut sim, &resolved).is_ok());
        }
    }
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
) -> Vec<(
    Vec<usize>,
    Vec<usize>,
    Vec<bool>,
    Vec<Vec<crate::graph::BeerSource>>,
)> {
    let mut out = Vec::new();
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
    out: &mut Vec<(
        Vec<usize>,
        Vec<usize>,
        Vec<bool>,
        Vec<Vec<crate::graph::BeerSource>>,
    )>,
) {
    if target_index == targets.len() {
        if keys.is_empty() {
            return;
        }
        let Ok(beer_sources) =
            crate::rules::plan_sell_beer_sources(state, pid, keys, merchants, use_beer)
        else {
            return;
        };
        out.push((
            keys.clone(),
            merchants.clone(),
            use_beer.clone(),
            beer_sources,
        ));
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

fn sell_free_develop_options(
    state: &GameState,
    pid: usize,
    merchant_indices: &[usize],
    use_merchant_beer: &[bool],
) -> Vec<Option<IndustryType>> {
    let awards = merchant_indices
        .iter()
        .zip(use_merchant_beer)
        .filter(|(merchant, uses_beer)| {
            **uses_beer
                && state.merchants.get(**merchant).is_some_and(|mt| {
                    matches!(
                        crate::map::merchant_bonus_at(mt.loc),
                        crate::map::MerchantBonus::Develop(_)
                    )
                })
        })
        .count();
    match awards {
        0 => vec![None],
        1 => state.players[pid]
            .developable_types()
            .into_iter()
            .map(|(ind, _)| Some(ind))
            .collect(),
        _ => Vec::new(),
    }
}
