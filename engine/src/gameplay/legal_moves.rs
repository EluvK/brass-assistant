//! Legal concrete move enumeration for execution and candidate scoring.

use crate::data::{Era, IndustryType};
use crate::gameplay::actions::{
    affordable_develop_iron_count, any_card_indices, can_develop, can_scout,
    coal_options_for_connection, coal_source_options, get_second_rail_options,
    get_valid_build_targets, get_valid_network_targets, iron_source_options, valid_build_cards,
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
        let double_types = if affordable_iron >= 2 {
            state.players[pid].double_developable_types()
        } else {
            Vec::new()
        };
        // Single develop
        let iron_opts = iron_source_options(state, 1);
        for &ind1 in &types {
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
        // Double develop. `double_developable_types` already accounts for
        // the tile uncovered by the first development.
        if affordable_iron >= 2 {
            let iron_opts = iron_source_options(state, 2);
            for &(ind1, ind2) in &double_types {
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

    // SELL
    let sell_plans = build_sell_plans(state, pid);
    if !sell_plans.is_empty() {
        for ci in &cards {
            for plan in &sell_plans {
                for free_develop in sell_free_develop_options(state, pid, &plan.0, &plan.1) {
                    moves.push(ResolvedMove::Sell {
                        keys: plan.0.clone(),
                        beer_sources: plan.1.clone(),
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
            beer_sources,
            free_develop,
            card_index,
        } => (
            format!("sell:{keys:?}:{beer_sources:?}:{free_develop:?}"),
            Move::Sell {
                keys: keys.clone(),
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

/// Backtracking source-combination enumeration.  The validator is the single
/// source of truth for buyer assignment and shared-resource legality.
fn build_sell_plans(
    state: &GameState,
    pid: usize,
) -> Vec<(Vec<usize>, Vec<crate::graph::BeerSource>)> {
    use crate::graph::{BeerSource, BeerSourceKind, find_beer_sources};
    use std::collections::HashSet;
    let targets: Vec<usize> = state
        .city_tiles
        .iter()
        .enumerate()
        .filter_map(|(k, t)| {
            let t = t.as_ref()?;
            (t.player == pid && !t.flipped && t.ind.is_sellable()).then_some(k)
        })
        .collect();
    let mut out = Vec::new();
    let mut seen = HashSet::new();
    for mask in 1usize..(1usize << targets.len()) {
        let keys: Vec<_> = targets
            .iter()
            .enumerate()
            .filter_map(|(i, &k)| (mask & (1 << i) != 0).then_some(k))
            .collect();
        let needed: usize = keys
            .iter()
            .map(|&k| {
                state.city_tiles[k]
                    .as_ref()
                    .unwrap()
                    .def
                    .beers_to_sell
                    .unwrap_or(0) as usize
            })
            .sum();
        let mut pool = Vec::<BeerSource>::new();
        for &key in &keys {
            let tile = state.city_tiles[key].as_ref().unwrap();
            let loc = crate::state::loc_from_key(key).unwrap().0;
            pool.extend(find_beer_sources(state, loc, pid, &[]));
            for (i, merchant) in state.merchants.iter().enumerate() {
                if merchant.has_beer
                    && merchant.accepts(tile.ind)
                    && crate::graph::connected_locations(state, loc).contains(&merchant.loc)
                {
                    pool.push(BeerSource {
                        kind: BeerSourceKind::Merchant,
                        key: usize::MAX,
                        farm_idx: None,
                        merchant_idx: Some(i),
                    });
                }
            }
        }
        pool.sort_by_key(crate::gameplay::actions::beer_source_sort_key);
        // Each physical source appears at most as often as it exists in the
        // state, even if it is reachable from several selected tiles.
        let mut capped = Vec::new();
        for src in pool {
            let available = match src.kind {
                BeerSourceKind::Merchant => 1,
                _ => find_beer_sources(
                    state,
                    crate::state::loc_from_key(keys[0]).unwrap().0,
                    pid,
                    &[],
                )
                .iter()
                .filter(|x| **x == src)
                .count(),
            };
            if capped.iter().filter(|x| **x == src).count() < available {
                capped.push(src);
            }
        }
        fn choose(
            state: &GameState,
            pid: usize,
            keys: &[usize],
            pool: &[BeerSource],
            start: usize,
            left: usize,
            picked: &mut Vec<BeerSource>,
            out: &mut Vec<(Vec<usize>, Vec<BeerSource>)>,
            seen: &mut HashSet<crate::gameplay::actions::SellIdentity>,
        ) {
            if left == 0 {
                let id = crate::gameplay::actions::sell_identity(keys, picked, None);
                if sell_plan_is_valid(state, pid, &id.keys, &id.beer_sources)
                    && seen.insert(id.clone())
                {
                    out.push((id.keys, id.beer_sources));
                }
                return;
            }
            for i in start..pool.len() {
                picked.push(pool[i]);
                choose(state, pid, keys, pool, i + 1, left - 1, picked, out, seen);
                picked.pop();
            }
        }
        if needed == 0 {
            let id = crate::gameplay::actions::sell_identity(&keys, &[], None);
            if sell_plan_is_valid(state, pid, &id.keys, &id.beer_sources) && seen.insert(id.clone())
            {
                out.push((id.keys, id.beer_sources));
            }
        } else if needed <= capped.len() {
            choose(
                state,
                pid,
                &keys,
                &capped,
                0,
                needed,
                &mut Vec::new(),
                &mut out,
                &mut seen,
            );
        }
    }
    out
}

fn sell_plan_is_valid(
    state: &GameState,
    pid: usize,
    keys: &[usize],
    sources: &[crate::graph::BeerSource],
) -> bool {
    crate::gameplay::actions::validate_sell_plan(state, pid, keys, sources, None).is_ok()
        || state.players[pid]
            .developable_types()
            .into_iter()
            .any(|(ind, _)| {
                crate::gameplay::actions::validate_sell_plan(state, pid, keys, sources, Some(ind))
                    .is_ok()
            })
}

fn sell_free_develop_options(
    state: &GameState,
    pid: usize,
    keys: &[usize],
    sources: &[crate::graph::BeerSource],
) -> Vec<Option<IndustryType>> {
    let eligible = sources.iter().any(|s| {
        s.kind == crate::graph::BeerSourceKind::Merchant
            && s.merchant_idx.is_some_and(|i| {
                matches!(
                    crate::map::merchant_bonus_at(state.merchants[i].loc),
                    crate::map::MerchantBonus::Develop(_)
                )
            })
    });
    if !eligible {
        return vec![None];
    }
    state.players[pid]
        .developable_types()
        .into_iter()
        .filter_map(|(ind, _)| {
            crate::gameplay::actions::validate_sell_plan(state, pid, keys, sources, Some(ind))
                .ok()
                .map(|_| Some(ind))
        })
        .collect()
}
