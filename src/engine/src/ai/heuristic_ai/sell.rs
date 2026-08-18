// SELL
// ---------------------------------------------------------------------------

fn best_merchant_beer_bonus(state: &GameState, merchant_indices: &[usize]) -> f64 {
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

fn merchant_bonus_value(state: &GameState, loc: Loc) -> f64 {
    match crate::map::merchant_bonus_at(loc) {
        crate::map::MerchantBonus::Vp(v) => v as f64 * VP_WEIGHT,
        crate::map::MerchantBonus::Money(m) => m as f64 * money_weight(state),
        crate::map::MerchantBonus::Income(spaces) => spaces as f64 * income_weight(state),
        crate::map::MerchantBonus::Develop(_) => 0.5,
    }
}

fn sell_route_value(state: &GameState, route: &crate::rules::SellRoute) -> f64 {
    if route.use_merchant_beer {
        merchant_bonus_value(state, state.merchants[route.merchant_index].loc)
    } else {
        0.0
    }
}

fn sell_plan_executes_all(
    state: &GameState,
    pid: usize,
    keys: &[usize],
    merchant_indices: &[usize],
    use_merchant: &[bool],
    card_index: usize,
) -> bool {
    let mut sim = state.clone();
    if execute_sell(
        &mut sim,
        pid,
        keys,
        merchant_indices,
        use_merchant,
        card_index,
    )
    .is_err()
    {
        return false;
    }

    keys.iter().all(|&key| {
        sim.city_tiles[key]
            .as_ref()
            .map(|tile| tile.player == pid && tile.flipped)
            .unwrap_or(false)
    })
}

fn score_sell_plan(state: &GameState, pid: usize) -> Option<Decision> {
    let targets = get_valid_sell_targets(state, pid);
    if targets.is_empty() {
        return None;
    }

    let card_index = pick_any_card(state, pid)?;

    // Rank targets, then sell all (matches aiPlayer.js sellPlan behavior).
    let mut ranked: Vec<(usize, f64, f64)> = targets
        .iter()
        .map(|t| {
            let merchant_choices: Vec<usize> = t.routes.iter().map(|r| r.merchant_index).collect();
            let bonus = best_merchant_beer_bonus(state, &merchant_choices);
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

    let mut keys = Vec::new();
    let mut merchant_indices = Vec::new();
    let mut use_merchant = Vec::new();
    for (key, _, _) in &ranked {
        let Some(target) = targets.iter().find(|t| t.key == *key) else {
            continue;
        };
        let mut routes = target.routes.clone();
        routes.sort_by(|a, b| {
            sell_route_value(state, b)
                .partial_cmp(&sell_route_value(state, a))
                .unwrap()
        });

        for route in routes {
            let mut try_keys = keys.clone();
            let mut try_merchants = merchant_indices.clone();
            let mut try_use = use_merchant.clone();
            try_keys.push(*key);
            try_merchants.push(route.merchant_index);
            try_use.push(route.use_merchant_beer);
            if sell_plan_executes_all(state, pid, &try_keys, &try_merchants, &try_use, card_index) {
                keys = try_keys;
                merchant_indices = try_merchants;
                use_merchant = try_use;
                break;
            }
        }
    }

    if keys.is_empty() {
        return None;
    }

    let total_income: f64 = keys
        .iter()
        .map(|key| {
            state.city_tiles[*key]
                .as_ref()
                .map_or(0.0, |tile| tile.def.income as f64)
        })
        .sum();
    let total_bonus: f64 = keys
        .iter()
        .zip(merchant_indices.iter().copied())
        .zip(use_merchant.iter().copied())
        .map(|((_key, merchant_index), use_merchant)| {
            if use_merchant {
                merchant_bonus_value(state, state.merchants[merchant_index].loc)
            } else {
                0.0
            }
        })
        .sum();
    let total_vp: f64 = keys
        .iter()
        .map(|key| {
            state.city_tiles[*key].as_ref().map_or(0.0, |tile| {
                tile.def.vp as f64 + tile.def.income as f64 * 0.3
            })
        })
        .sum();

    // End-of-era urgency: in the final round of the canal era, canal-only
    // (level-1) sellables vanish if unflipped — selling them now is essential
    // or the VP + income is permanently lost. Boost the sell action sharply.
    // In Rail-Late the finish is selling your built goods for VP — selling is
    // the whole point of the late game, so weight it across the entire phase.
    let rounds_left = estimate_rounds_remaining(state);
    let canal_end = state.era == Era::Canal && rounds_left <= 2.0;
    let rail_end = state.era == Era::Rail && rounds_left <= 1.0;
    let urgent = canal_end || rail_end;
    let mut urgency_bonus = if urgent { 3.0 } else { 0.0 };
    if era_phase(state) == Phase::RailLate && !urgent {
        urgency_bonus += 1.2;
    }

    // Canal late-game tactical prior: deplete beer and flip breweries before
    // level-1 cleanup, or the barrels evaporate with no value.
    let mut canal_beer_drain_bonus = 0.0;
    if state.era == Era::Canal && rounds_left <= 2.5 {
        let (before_barrels, before_flipped) = own_brewery_stats(state, pid);
        let mut sim = state.clone();
        if execute_sell(
            &mut sim,
            pid,
            &keys,
            &merchant_indices,
            &use_merchant,
            card_index,
        )
        .is_ok()
        {
            let (after_barrels, after_flipped) = own_brewery_stats(&sim, pid);
            let consumed = before_barrels.saturating_sub(after_barrels) as f64;
            let flipped_gain = after_flipped.saturating_sub(before_flipped) as f64;
            canal_beer_drain_bonus = consumed * 0.9 + flipped_gain * 2.2;
            if before_barrels > 0 && after_barrels == 0 {
                canal_beer_drain_bonus += 3.8;
            }
        }
    }

    // Flipping a sellable advances income: that's a recurring cash stream,
    // not just a one-time VP. Add the income value explicitly.
    let income_stream = total_income * income_weight(state) * 0.5;

    let score = vp_equivalent(state, total_vp, total_income, 0.0, 0.0)
        + total_bonus
        + urgency_bonus
        + income_stream
        + canal_beer_drain_bonus;

    Some(Decision {
        mv: Move::Sell {
            keys,
            merchant_indices,
            use_merchant_beer: use_merchant,
            card_index,
        },
        score,
    })
}
