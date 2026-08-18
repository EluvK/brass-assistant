// DEVELOP
// ---------------------------------------------------------------------------

fn score_develop_plan(state: &GameState, pid: usize, plan: &Plan) -> Option<Decision> {
    if !can_develop(state, pid) {
        return None;
    }
    let mut types = state.players[pid].developable_types();
    if BAN_DEVELOP_IRON_LV2_PLUS {
        types.retain(|(ind, tile)| !(*ind == IndustryType::IronWorks && tile.level >= 2));
    }
    if BAN_DEVELOP_BREWERY_LV2_PLUS_AT_CANAL_EARLY && state.era == Era::Canal {
        types.retain(|(ind, tile)| !(*ind == IndustryType::Brewery && tile.level >= 2));
    }
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

    let score_target = |ind: IndustryType, tile: crate::data::TileDef, unlock_offset: usize| {
        // Develop removes a level-1 tile and reveals the NEXT (level-2+)
        // tile. Its payoff is the unlocked rail-era tile that double-scores
        // flipped VP, but develop must stay well below an immediate build.
        let unlocked = state.players[pid].tile_after(ind, unlock_offset);
        let mut v = if tile.rail_era { 0.35 } else { 0.12 };
        v += 0.18 * tile.level as f64;
        if let Some(ut) = unlocked {
            // Small premium for unlocking a double-scoring rail-era tile.
            if ut.rail_era {
                v += 0.25;
            }
        }
        // Breweries are the economic engine (develop 1 -> build 2/3/4).
        // In Canal-Early, grabbing the beer spot early is almost a must —
        // but it must stay rational: developing costs iron, so only develop
        // the brewery when iron is cheap (£1-2). At £3+ the iron is better
        // spent building/supplying iron first instead.
        if ind == IndustryType::Brewery && tile.level == 1 {
            v += 0.55;
            if era_phase(state) == Phase::CanalEarly {
                let iron_price = state.iron_price();
                if iron_price < 2 {
                    v += 3.0
                } else if iron_price <= 2 {
                    v += 2.0;
                } else if iron_price <= 3 {
                    v += 0.5;
                } else if iron_price > 3 {
                    v -= 1.5;
                }
            }
        }
        // Canal-era develop wins the cross-era bonus; nudge it.
        if state.era == Era::Canal {
            v += 0.15;
        }
        // Plan ("流派") soft bonus: developing toward the target industry
        // aligns with the production plan (unlock higher-level tiles of the
        // goods we intend to sell).
        if plan.industry == ind {
            v += 0.3;
        }
        // Only meaningful if we actually hold a card to build it next.
        if player_has_buildable_card(state, pid, ind) {
            v += 0.3;
        }
        v - develop_guardrail_penalty(ind, tile.level)
    };
    let mut scored: Vec<(IndustryType, f64)> = types
        .into_iter()
        .map(|(ind, tile)| (ind, score_target(ind, tile, 1)))
        .collect();
    scored.sort_by(|a, b| (b.1).partial_cmp(&a.1).unwrap());

    let first = scored[0];
    let two_source_cost: i32 = iron
        .iter()
        .take(2)
        .map(|s| if s.free { 0 } else { s.price as i32 })
        .sum();
    let can_afford_second = iron.len() >= 2 && two_source_cost <= player.money;

    let second = if can_afford_second {
        let best_other = scored.iter().skip(1).map(|s| (s.0, s.1)).next();
        let same_industry = state.players[pid]
            .tile_after(first.0, 1)
            .filter(|tile| {
                tile.can_develop
                    && !(BAN_DEVELOP_IRON_LV2_PLUS
                        && first.0 == IndustryType::IronWorks
                        && tile.level >= 2)
            })
            .map(|tile| (first.0, score_target(first.0, tile, 2)));
        [best_other, same_industry]
            .into_iter()
            .flatten()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
    } else {
        None
    };

    let iron_cost: i32 = iron
        .iter()
        .take(if second.is_some() { 2 } else { 1 })
        .map(|s| if s.free { 0 } else { s.price as i32 })
        .sum();
    // Iron is scarce and the develop action burns a full turn; charge a real
    // opportunity cost so develop isn't a free high-score default.
    let iron_scarcity = if iron[0].free { 0.0 } else { 0.6 };
    let second_value = second.map(|(_, score)| score * 0.4).unwrap_or(0.0);

    let mut score =
        vp_equivalent(state, first.1 + second_value, 0.0, -(iron_cost as f64), 0.0) - iron_scarcity;

    if state.era == Era::Canal {
        score *= 2.0;
        // In canal, spending a Develop action for only one tile is usually
        // tempo-inefficient; prefer two-tile develops when legal/affordable.
        if second.is_none() {
            score -= 2.0;
        } else {
            score += 0.5;
        }
    }

    // Action-economy guardrail: develop is only worth the action budget it
    // consumes. Humans develop at most ~4 times in the canal era (8 tiles via
    // double-develops) and ~0-1 times in the rail era — beyond that, develop
    // crowds out builds/links/sells that actually score (even a bare link
    // beats a 5th develop). Penalize past the era's limit, growing steeply.
    let already = if state.era == Era::Canal {
        state.players[pid].develops_in_canal
    } else {
        state.players[pid].develops_in_rail
    };
    let this_would_be = already + 1;
    let limit = if state.era == Era::Canal { 4.0 } else { 1.0 };
    let over = (this_would_be as f64 - limit).max(0.0);
    let develop_count_penalty = over * over * 2.0 + over;
    score -= develop_count_penalty;

    let iron_needed = if second.is_some() { 2 } else { 1 };
    let iron_choice = crate::rules::iron_source_options(state, iron_needed)
        .into_iter()
        .next()?;
    let card_index = pick_any_card(state, pid)?;
    Some(Decision {
        mv: Move::Develop {
            ind1: first.0,
            ind2: second.map(|(ind, _)| ind),
            iron: iron_choice,
            card_index,
        },
        score,
    })
}
