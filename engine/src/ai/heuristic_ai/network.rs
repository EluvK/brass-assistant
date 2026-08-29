// NETWORK
// ---------------------------------------------------------------------------

fn count_hand_cards_newly_in_network(state: &GameState, pid: usize, cand_cities: &[Loc; 2]) -> f64 {
    let player = &state.players[pid];
    let mut count = 0.0;
    for card in &player.hand {
        if let Card::Location(l) = card {
            if !is_in_network(state, pid, *l) && cand_cities.contains(l) {
                count += 0.6;
            }
        } else if let Card::Industry { .. } = card {
            count += 0.1;
        }
    }
    count
}

fn connects_to_new_merchant(_state: &GameState, cand_cities: &[Loc; 2]) -> bool {
    cand_cities.iter().any(|c| c.is_merchant())
}

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

fn immediate_link_icons_at(state: &GameState, loc: Loc) -> f64 {
    if loc.is_merchant() {
        return 2.0;
    }
    if loc.is_city() {
        let mut v = 0.0;
        for slot in 0..city_slots(loc).len() {
            if let Some(k) = state.city_slot_key(loc, slot) {
                if let Some(t) = &state.city_tiles[k] {
                    if t.flipped {
                        v += t.def.link_vp as f64;
                    }
                }
            }
        }
        return v;
    }
    if let Some(t) = state.farm_tile(loc) {
        if t.flipped {
            return t.def.link_vp as f64;
        }
    }
    0.0
}

/// Return `(current_link_vp, future_link_vp)` for a connection.
fn link_current_and_potential_vps(
    state: &GameState,
    conn_id: usize,
    cities: &[Loc; 2],
) -> (usize, usize) {
    /// Potential link VP around an unoccupied location. Every empty city slot is
    /// worth one future VP, or two when it can host a brewery. An unbuilt brewery
    /// farm likewise contributes two future VP.
    fn future_link_node_potential(state: &GameState, loc: Loc) -> usize {
        if loc.is_farm() {
            return usize::from(state.farm_tile(loc).is_none()) * 2;
        }
        if !loc.is_city() {
            return 0;
        }

        city_slots(loc)
            .iter()
            .enumerate()
            .filter(|(slot_idx, _)| {
                state
                    .city_slot_key(loc, *slot_idx)
                    .is_some_and(|k| state.city_tiles[k].is_none())
            })
            .map(|(_, slot_types)| {
                if slot_types.contains(&IndustryType::Brewery) {
                    2
                } else {
                    1
                }
            })
            .sum()
    }

    let mut current_link_vp = 0;
    let mut future_link_vp = 0;
    for loc in cities {
        current_link_vp += immediate_link_icons_at(state, *loc) as usize;
        future_link_vp += future_link_node_potential(state, *loc);
    }
    if let Some(farm) = connections()[conn_id].via_farm {
        current_link_vp += immediate_link_icons_at(state, farm) as usize;
        future_link_vp += future_link_node_potential(state, farm);
    }
    (current_link_vp, future_link_vp)
}

fn score_network_candidate(
    state: &GameState,
    pid: usize,
    conn_id: usize,
    cost: i32,
    cities: &[Loc; 2],
    is_canal: bool,
    plan: &Plan,
) -> f64 {
    let access_gain = count_hand_cards_newly_in_network(state, pid, cities);
    let merchant_gain = if connects_to_new_merchant(state, cities) {
        1.5
    } else {
        0.0
    };

    let (current_link_vp, future_link_vp) = link_current_and_potential_vps(state, conn_id, cities);
    let potential_link_vp = match state.round {
        0..=3 => current_link_vp as f64 + future_link_vp as f64 * 0.8,
        4..=6 => current_link_vp as f64 + future_link_vp as f64 * 0.5,
        _ => current_link_vp as f64 + future_link_vp as f64 * 0.3,
    };

    let player = &state.players[pid];
    let links_built = if is_canal {
        14 - player.canal_links
    } else {
        14 - player.rail_links
    };

    let exploration_bonus = (1.6 - links_built as f64 * 0.3).max(0.0);

    // Plan ("流派") network bonus: a link that touches a city where the plan
    // industry still has a vacant slot (and we still hold tiles for it) opens
    // up production capacity. Only from Canal-Late onward (Canal-Early builds
    // the economy engine first). Small — a tiebreaker, not a driver.
    let mut plan_bonus = 0.0;
    if plan.count > 0
        && era_phase(state) != Phase::CanalEarly
        && state.players[pid].remaining_count(plan.industry) > 0
    {
        for loc in cities {
            if !loc.is_city() {
                continue;
            }
            let slots = city_slots(*loc);
            for (slot_idx, allowed) in slots.iter().enumerate() {
                if !allowed.contains(&plan.industry) {
                    continue;
                }
                if let Some(k) = state.city_slot_key(*loc, slot_idx) {
                    if state.city_tiles[k].is_none() {
                        plan_bonus = 0.5;
                    }
                }
            }
            if plan_bonus > 0.0 {
                break;
            }
        }
    }

    // Rail-Early beer-farm lock: links touching a brewery farm lock down beer
    // supply (double-rails and late-game sells depend on it). Strategic even if
    // the immediate link score is low — the two farm links are prime real
    // estate, so Rail-Early rewards grabbing them.
    let mut beer_lock_bonus = 0.0;
    if era_phase(state) == Phase::RailEarly {
        let touches_farm = cities.iter().any(|l| l.is_farm())
            || connections()[conn_id]
                .via_farm
                .map_or(false, |f| f.is_farm());
        if touches_farm {
            beer_lock_bonus = 1.2;
        }
    }

    // Network-era weighting: scale the strategic (non-cash) value of the link
    // by the phase's network_w. Rail-Early builds the net everything scores
    // through, so its links are worth more.
    let profile = era_profile(state);
    let net_scale = profile.network_w;
    let mut score = vp_equivalent(state, access_gain + merchant_gain, 0.0, -(cost as f64), 0.0)
        + exploration_bonus
        + plan_bonus
        + beer_lock_bonus;
    score += potential_link_vp * net_scale;
    score
}

/// Top-K single network candidates by 1-ply score.
pub(crate) fn score_top_networks(
    state: &mut GameState,
    pid: usize,
    k: usize,
    plan: &Plan,
    card_choices: &CardChoices,
) -> Vec<Decision> {
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
        let mut cost = if state.era == Era::Canal {
            crate::map::CANAL_LINK_COST
        } else {
            crate::map::RAIL_LINK_COST
        };
        if state.era == Era::Rail {
            // Rail links consume 1 coal; charging only base link cost massively
            // overvalues network spam and undervalues coal supply actions.
            cost += estimated_connection_coal_cost(state, conn_id);
        }
        let s = score_network_candidate(
            state,
            pid,
            conn_id,
            cost,
            &[conn.a, conn.b],
            state.era == Era::Canal,
            plan,
        );
        scored.push((conn_id, s));
    }
    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    scored.truncate(k);
    let Some(card_index) = card_choices.first().map(|(index, _)| *index) else {
        return Vec::new();
    };
    scored
        .into_iter()
        .map(|(conn_id, score)| {
            let coal = if state.era == Era::Rail {
                cheapest_connection_coal_source(state, conn_id)
            } else {
                None
            };
            Decision {
                mv: ResolvedMove::Network {
                    conn_id,
                    coal,
                    card_index,
                },
                score,
                card_score: card_choices
                    .first()
                    .map(|(_, s)| *s)
                    .unwrap_or(f64::INFINITY),
            }
        })
        .collect()
}

pub(crate) fn score_top_network_doubles(
    state: &mut GameState,
    pid: usize,
    k: usize,
    plan: &Plan,
    card_choices: &CardChoices,
) -> Vec<Decision> {
    if k == 0 {
        return Vec::new();
    }
    if state.era != Era::Rail {
        return Vec::new();
    }
    let player = &state.players[pid];
    if player.rail_links < 2 {
        return Vec::new();
    }
    let singles = get_valid_network_targets(state, pid);
    let mut scored: Vec<(usize, usize, f64)> = Vec::new();
    for conn1 in singles {
        for conn2 in get_valid_second_rail_links(state, pid, conn1) {
            // Value both links and charge realistic resource cost:
            // total = £15 base + 2 coal (explicitly estimated from legal sources).
            let c1 = &connections()[conn1];
            let c2 = &connections()[conn2];
            let cost1 = crate::map::RAIL_LINK_COST + estimated_connection_coal_cost(state, conn1);
            let cost2 = crate::map::RAIL_LINK_COST + estimated_connection_coal_cost(state, conn2);
            let s1 = score_network_candidate(state, pid, conn1, cost1, &[c1.a, c1.b], false, plan);
            let s2 = score_network_candidate(state, pid, conn2, cost2, &[c2.a, c2.b], false, plan);
            let extra_base =
                (crate::map::RAIL_DOUBLE_LINK_COST - 2 * crate::map::RAIL_LINK_COST) as f64;
            let s = s1 + s2 - extra_base * money_weight(state);
            // Double-rail synergy: one action builds two links (a tempo win vs
            // two separate actions), and in Rail-Early the net is being laid
            // out fast. Add a modest action-economy bonus so the combo is
            // preferred when the two links are comparable.
            let mut s = s + if era_phase(state) == Phase::RailEarly {
                1.2
            } else {
                0.6
            };
            // Beer-lock synergy: a double-rail that grabs a brewery farm link
            // locks beer while saving tempo — especially valuable.
            let touches_farm = [&connections()[conn1], &connections()[conn2]]
                .iter()
                .any(|c| {
                    c.a.is_farm() || c.b.is_farm() || c.via_farm.map_or(false, |f| f.is_farm())
                });
            if touches_farm {
                s += 0.8;
            }
            scored.push((conn1, conn2, s));
        }
    }
    scored.sort_by(|a, b| b.2.total_cmp(&a.2).then(a.0.cmp(&b.0)).then(a.1.cmp(&b.1)));
    let Some(card_index) = card_choices.first().map(|(index, _)| *index) else {
        return Vec::new();
    };
    let mut out = Vec::with_capacity(k);
    for (conn1, conn2, score) in scored {
        if out.len() >= k {
            break;
        }
        // Prefer consuming our own beer (advances our own income when it flips) over
        // an opponent's beer (which would advance theirs). The coal2 options must be
        // enumerated against the SAME coal1 we actually use (not an internally
        // cheapest one), otherwise the emitted move may fail to execute.
        let coal1 = crate::rules::coal_options_for_connection(state, &connections()[conn1], 1)
            .into_iter()
            .next()
            .and_then(|o| o.into_iter().next());
        let Some(coal1) = coal1 else { continue };
        let Some(opt) = crate::rules::get_second_rail_options(state, pid, conn1, coal1)
            .into_iter()
            .find(|o| o.conn == conn2)
        else {
            continue;
        };
        let Some(beer) = opt
            .beers
            .iter()
            .copied()
            .find(|b| b.kind == crate::graph::BeerSourceKind::Own)
            .or_else(|| opt.beers.first().copied())
        else {
            continue;
        };
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
    out
}
