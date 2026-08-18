// NETWORK
// ---------------------------------------------------------------------------

fn count_hand_cards_newly_in_network(state: &GameState, pid: usize, cand_cities: &[Loc; 2]) -> f64 {
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

/// Rail-era link VP potential: sum the top flipped VP a player could still
/// land in the two cities this link connects (vacant slots + remaining tiles),
/// weighted by whether they can actually place there. Dense industrial hubs
/// are worth racing for in the rail era.
fn link_vp_potential(state: &GameState, pid: usize, cities: &[Loc; 2]) -> f64 {
    let mut total = 0.0f64;
    for loc in cities {
        if !loc.is_city() {
            continue;
        }
        let slots = city_slots(*loc);
        for (slot_idx, slot_types) in slots.iter().enumerate() {
            // Skip occupied slots (someone already built here).
            if let Some(k) = state.city_slot_key(*loc, slot_idx) {
                if state.city_tiles[k].is_some() {
                    continue;
                }
            }
            // Best remaining tile we could build in this slot.
            let mut best = 0.0f64;
            for t in *slot_types {
                if let Some(tile) = state.players[pid].next_tile(*t) {
                    // Only level-2+ tiles survive the canal→rail transition and
                    // double-score their flipped VP.
                    let dvp = if tile.level >= 2 { 1.4 } else { 1.0 };
                    let v = tile.vp as f64 * dvp;
                    if v > best {
                        best = v;
                    }
                }
            }
            total += best;
        }
    }
    total
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

fn future_link_node_potential(state: &GameState, pid: usize, loc: Loc) -> f64 {
    if !loc.is_city() {
        return 0.0;
    }
    let mut total = 0.0;
    for (slot_idx, slot_types) in city_slots(loc).iter().enumerate() {
        if let Some(k) = state.city_slot_key(loc, slot_idx) {
            if state.city_tiles[k].is_some() {
                continue;
            }
        }
        let mut best = 0.0;
        for t in *slot_types {
            if let Some(tile) = state.players[pid].next_tile(*t) {
                // Approximate future node value for link scoring around this hub.
                let mut v = tile.link_vp as f64 + tile.vp as f64 * 0.2;
                if tile.level >= 2 && state.era == Era::Canal {
                    v += 0.6;
                }
                if player_has_buildable_card(state, pid, *t) {
                    v += 0.25;
                }
                if v > best {
                    best = v;
                }
            }
        }
        total += best;
    }
    total
}

fn potential_link_vps(state: &GameState, pid: usize, conn_id: usize, cities: &[Loc; 2]) -> f64 {
    let mut immediate = 0.0;
    let mut future = 0.0;
    for loc in cities {
        immediate += immediate_link_icons_at(state, *loc);
        future += future_link_node_potential(state, pid, *loc);
    }
    if let Some(via) = connections()[conn_id].via_farm {
        immediate += immediate_link_icons_at(state, via);
    }
    immediate + 0.7 * future
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
    // Rail-era "road multiplier": links into dense industrial hubs are the
    // highest-value plays once the rail era starts. Weight by the VP potential
    // of the cities they open up, scaled down to VP-equivalent units.
    let vp_potential = link_vp_potential(state, pid, cities);
    let potential_link_vp = potential_link_vps(state, pid, conn_id, cities);
    let hub_bonus = if state.era == Era::Rail {
        vp_potential * 0.08 + potential_link_vp * 0.28
    } else {
        vp_potential * 0.03 + potential_link_vp * 0.20
    };
    let player = &state.players[pid];
    let links_built = if is_canal {
        14 - player.canal_links
    } else {
        14 - player.rail_links
    };
    // Links are only as valuable as the tiles they connect. A player building
    // far more links than they have productive (flippable) tiles is over-
    // networking: money locked into potential that never pays off, which is
    // how broken boards end up with 8-14 links and 0 flips. Penalize the
    // surplus links proportionally so network "tempo" can't be spammed from a
    // position with nothing to connect.
    let productive = state
        .city_tiles
        .iter()
        .flatten()
        .filter(|t| t.player == pid)
        .count() as f64;
    let over_networking_penalty = if productive <= 0.0 {
        (links_built as f64 - 1.0).max(0.0) * 1.2
    } else {
        (links_built as f64 - 2.0 * productive - 1.0).max(0.0) * 0.6
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

    // Canal-Late "critical path only": extra actions must NOT be spent on a
    // dead-end link that neither unlocks a build for the plan industry nor
    // connects a reachable merchant (i.e. it doesn't help sell the goods we
    // built). A link that fails every check is a pure cash sink this late.
    let mut critical_penalty = 0.0;
    if era_phase(state) == Phase::CanalLate {
        let mut connects_useful = false;
        for loc in cities {
            if loc.is_merchant() || loc.is_farm() {
                connects_useful = true;
                break;
            }
            // Vacant slot for the plan industry (or any sellable we still hold).
            let slots = city_slots(*loc);
            for (slot_idx, allowed) in slots.iter().enumerate() {
                if let Some(k) = state.city_slot_key(*loc, slot_idx) {
                    if state.city_tiles[k].is_some() {
                        continue;
                    }
                    for ind in *allowed {
                        if state.players[pid].remaining_count(*ind) > 0 {
                            connects_useful = true;
                            break;
                        }
                    }
                }
                if connects_useful {
                    break;
                }
            }
            if connects_useful {
                break;
            }
        }
        if !connects_useful {
            critical_penalty = 2.5;
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
        - over_networking_penalty
        + plan_bonus
        - critical_penalty
        + beer_lock_bonus;
    score += hub_bonus * net_scale;
    score
}

fn pick_any_card(state: &GameState, pid: usize) -> Option<usize> {
    let hand = &state.players[pid].hand;
    (0..hand.len()).find(|_| true)
}

/// Top-K single network candidates by 1-ply score.
pub(crate) fn score_top_networks(
    state: &mut GameState,
    pid: usize,
    k: usize,
    plan: &Plan,
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
    scored
        .into_iter()
        .filter_map(|(conn_id, score)| {
            let card_index = pick_any_card(state, pid)?;
            let coal = if state.era == Era::Rail {
                cheapest_connection_coal_source(state, conn_id)
            } else {
                None
            };
            Some(Decision {
                mv: Move::Network {
                    conn_id,
                    coal,
                    card_index,
                },
                score,
            })
        })
        .collect()
}

fn score_best_network_double(state: &mut GameState, pid: usize, plan: &Plan) -> Option<Decision> {
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
    // Prefer consuming our own beer (advances our own income when it flips) over
    // an opponent's beer (which would advance theirs). The coal2 options must be
    // enumerated against the SAME coal1 we actually use (not an internally
    // cheapest one), otherwise the emitted move may fail to execute.
    let coal1 = crate::rules::coal_options_for_connection(state, &connections()[conn1], 1)
        .into_iter()
        .next()
        .and_then(|o| o.into_iter().next())?;
    let opt = crate::rules::get_second_rail_options(state, pid, conn1, coal1)
        .into_iter()
        .find(|o| o.conn == conn2)?;
    let beer = opt
        .beers
        .iter()
        .copied()
        .find(|b| b.kind == crate::graph::BeerSourceKind::Own)
        .or_else(|| opt.beers.first().copied())?;
    let coal2 = opt.coal2_opts.first().and_then(|o| o.first()).copied()?;
    let card_index = pick_any_card(state, pid)?;
    Some(Decision {
        mv: Move::NetworkDouble {
            conn1,
            conn2,
            coal1,
            coal2,
            beer,
            card_index,
        },
        score,
    })
}
