//! SELL action route discovery, execution, and merchant bonuses.

use super::{discard_card, log_industry_label, log_loc_label, require_card_index};
use crate::data::IndustryType;
use crate::graph::{BeerSource, connected_locations, find_beer_sources};
use crate::map::{Loc, MerchantBonus};
use crate::state::GameState;
// ---------------------------------------------------------------------------
// SELL
// ---------------------------------------------------------------------------

#[derive(Clone)]
pub struct SellTarget {
    pub key: usize,
    pub beer_needed: u8,
    pub routes: Vec<SellRoute>,
}

#[derive(Debug, Clone, Copy)]
pub struct SellRoute {
    pub merchant_index: usize,
    pub use_merchant_beer: bool,
}

#[derive(Debug, Clone)]
pub struct SellPlan {
    pub keys: Vec<usize>,
    pub merchant_indices: Vec<usize>,
    pub use_merchant_beer: Vec<bool>,
    /// Per-tile brewery beer payment. Merchant beer is represented by the
    /// aligned `use_merchant_beer` flag and is deliberately not duplicated.
    pub beer_sources: Vec<Vec<BeerSource>>,
}

pub fn plan_sell_beer_sources(
    state: &GameState,
    pid: usize,
    keys: &[usize],
    merchant_indices: &[usize],
    use_merchant_beer: &[bool],
) -> Result<Vec<Vec<BeerSource>>, String> {
    if keys.len() != merchant_indices.len() || keys.len() != use_merchant_beer.len() {
        return Err("Sell move shape mismatch".into());
    }
    let mut sim = state.clone();
    let mut plans = Vec::with_capacity(keys.len());
    for ((&key, &merchant), &use_merchant) in
        keys.iter().zip(merchant_indices).zip(use_merchant_beer)
    {
        let tile = sim
            .city_tiles
            .get(key)
            .and_then(Option::as_ref)
            .ok_or("Invalid sell tile")?;
        let loc = crate::state::loc_from_key(key)
            .map(|(loc, _)| loc)
            .ok_or("Invalid sell tile")?;
        let mut needed = tile.def.beers_to_sell.unwrap_or(0) as usize;
        if use_merchant && needed > 0 {
            if !sim.merchants.get(merchant).is_some_and(|mt| mt.has_beer) {
                return Err("Chosen merchant has no beer".into());
            }
            sim.merchants[merchant].has_beer = false;
            needed -= 1;
        }
        let sources = find_beer_sources(&sim, loc, pid, &[]);
        if sources.len() < needed {
            return Err("Not enough brewery beer for chosen merchant route".into());
        }
        let payment: Vec<BeerSource> = sources.into_iter().take(needed).collect();
        for source in &payment {
            sim.consume_beer_source(source);
        }
        plans.push(payment);
    }
    Ok(plans)
}

fn sell_routes_for_target(
    state: &GameState,
    pid: usize,
    loc: Loc,
    merchant_indices: &[usize],
    beer_needed: u8,
) -> Vec<SellRoute> {
    let mut routes = Vec::new();
    if merchant_indices.is_empty() {
        return routes;
    }

    let brewery_beer = find_beer_sources(state, loc, pid, &[]).len();

    if beer_needed == 0 {
        for &merchant_index in merchant_indices {
            routes.push(SellRoute {
                merchant_index,
                use_merchant_beer: false,
            });
        }
        return routes;
    }

    if brewery_beer >= beer_needed as usize {
        for &merchant_index in merchant_indices {
            routes.push(SellRoute {
                merchant_index,
                use_merchant_beer: false,
            });
        }
    }

    for &merchant_index in merchant_indices {
        if state.merchants[merchant_index].has_beer && brewery_beer + 1 >= beer_needed as usize {
            routes.push(SellRoute {
                merchant_index,
                use_merchant_beer: true,
            });
        }
    }

    routes
}

pub fn get_valid_sell_targets(state: &GameState, pid: usize) -> Vec<SellTarget> {
    let mut out = Vec::new();
    for (k, tile) in state.city_tiles.iter().enumerate() {
        let Some(t) = tile else { continue };
        if t.player != pid || t.flipped || !t.ind.is_sellable() {
            continue;
        }
        // Loc from key
        let Some(loc) = crate::state::loc_from_key(k).map(|(l, _)| l) else {
            continue;
        };
        let merchant_indices = sell_merchants_for(state, pid, loc, t.ind);
        let beer_needed = t.def.beers_to_sell.unwrap_or(0);
        let routes = sell_routes_for_target(state, pid, loc, &merchant_indices, beer_needed);
        if routes.is_empty() {
            continue;
        }
        out.push(SellTarget {
            key: k,
            beer_needed,
            routes,
        });
    }
    out
}

fn sell_merchants_for(state: &GameState, _pid: usize, loc: Loc, ind: IndustryType) -> Vec<usize> {
    let connected = connected_locations(state, loc);
    let mut out = Vec::new();
    for (i, mt) in state.merchants.iter().enumerate() {
        if !mt.accepts(ind) {
            continue;
        }
        if connected.contains(&mt.loc) {
            out.push(i);
        }
    }
    out
}

fn beer_source_label(state: &GameState, src: &BeerSource) -> String {
    match src.kind {
        crate::graph::BeerSourceKind::Merchant => src
            .merchant_idx
            .and_then(|mi| state.merchants.get(mi))
            .map(|mt| format!("商家酒@{}", log_loc_label(mt.loc)))
            .unwrap_or_else(|| "商家酒@?".to_string()),
        crate::graph::BeerSourceKind::Own | crate::graph::BeerSourceKind::Opponent => {
            if let Some(fi) = src.farm_idx {
                let loc = if fi == 0 {
                    Loc::BreweryNorth
                } else {
                    Loc::BrewerySouth
                };
                format!("酒@{}", log_loc_label(loc))
            } else {
                let loc = crate::state::loc_from_key(src.key)
                    .map(|(l, _)| l)
                    .map(log_loc_label)
                    .unwrap_or_else(|| "?".to_string());
                format!("酒@{loc}")
            }
        }
    }
}

pub fn execute_sell(
    state: &mut GameState,
    pid: usize,
    keys: &[usize],
    merchant_indices: &[usize],
    use_merchant_beer: &[bool],
    beer_sources: &[Vec<BeerSource>],
    card_index: usize,
) -> Result<String, String> {
    // Compatibility helper for internal simulations. Concrete executable
    // moves always carry the selected industry; callers that only ask whether
    // a plan works receive the first available legal choice.
    let free_develop = merchant_indices
        .iter()
        .zip(use_merchant_beer)
        .any(|(&merchant, &uses_beer)| {
            uses_beer
                && state.merchants.get(merchant).is_some_and(|mt| {
                    matches!(
                        crate::map::merchant_bonus_at(mt.loc),
                        MerchantBonus::Develop(_)
                    )
                })
        })
        .then(|| {
            state.players[pid]
                .developable_types()
                .first()
                .map(|(ind, _)| *ind)
        })
        .flatten();
    execute_sell_with_free_develop(
        state,
        pid,
        keys,
        merchant_indices,
        use_merchant_beer,
        beer_sources,
        free_develop,
        card_index,
    )
}

pub fn execute_sell_with_free_develop(
    state: &mut GameState,
    pid: usize,
    keys: &[usize],
    merchant_indices: &[usize],
    use_merchant_beer: &[bool],
    beer_sources: &[Vec<BeerSource>],
    free_develop: Option<IndustryType>,
    card_index: usize,
) -> Result<String, String> {
    let snapshot = state.clone();
    let result = execute_sell_inner(
        state,
        pid,
        keys,
        merchant_indices,
        use_merchant_beer,
        beer_sources,
        free_develop,
        card_index,
    );
    if result.is_err() {
        *state = snapshot;
    }
    result
}

fn execute_sell_inner(
    state: &mut GameState,
    pid: usize,
    keys: &[usize],
    merchant_indices: &[usize],
    use_merchant_beer: &[bool],
    beer_sources: &[Vec<BeerSource>],
    free_develop: Option<IndustryType>,
    card_index: usize,
) -> Result<String, String> {
    require_card_index(state, pid, card_index)?;
    if merchant_indices.len() != keys.len()
        || use_merchant_beer.len() != keys.len()
        || beer_sources.len() != keys.len()
    {
        return Err("Sell move shape mismatch".into());
    }
    if keys.is_empty() {
        return Err("Sell move must include at least one tile".into());
    }
    let free_develop_count = merchant_indices
        .iter()
        .zip(use_merchant_beer)
        .filter(|&(&merchant, &uses_beer)| {
            uses_beer
                && state.merchants.get(merchant).is_some_and(|mt| {
                    matches!(
                        crate::map::merchant_bonus_at(mt.loc),
                        MerchantBonus::Develop(_)
                    )
                })
        })
        .count();
    if free_develop_count > 1 {
        return Err("A sell action cannot award more than one free develop".into());
    }
    match (free_develop_count, free_develop) {
        (0, None) => {}
        (0, Some(_)) => return Err("Sell does not award a free develop".into()),
        (1, None) => return Err("Sell must choose its free develop".into()),
        (1, Some(ind)) => {
            if !state.players[pid]
                .developable_types()
                .iter()
                .any(|(x, _)| *x == ind)
            {
                return Err("Free develop choice is not developable".into());
            }
        }
        _ => unreachable!(),
    }
    let mut seen = std::collections::HashSet::new();
    for (entry_idx, &key) in keys.iter().enumerate() {
        if !seen.insert(key) {
            return Err("Sell move contains the same tile more than once".into());
        }
        let tile = state
            .city_tiles
            .get(key)
            .and_then(Option::as_ref)
            .ok_or("Invalid sell tile")?;
        if tile.player != pid || tile.flipped || !tile.ind.is_sellable() {
            return Err("Tile cannot be sold".into());
        }
        let loc = crate::state::loc_from_key(key)
            .map(|(loc, _)| loc)
            .ok_or("Invalid sell tile")?;
        let merchant = *merchant_indices
            .get(entry_idx)
            .ok_or("Sell move shape mismatch")?;
        if !sell_merchants_for(state, pid, loc, tile.ind).contains(&merchant) {
            return Err("Chosen merchant is not a valid buyer".into());
        }
    }

    let mut sold = 0usize;
    let mut notes = Vec::new();
    let mut beer_notes = Vec::new();

    for (entry_idx, &key) in keys.iter().enumerate() {
        // Read tile info without holding a long-lived mutable borrow.
        let (t_player, t_flipped, ind, beer_needed, loc) = {
            let Some(tile) = state.city_tiles[key].as_ref() else {
                continue;
            };
            let loc = match crate::state::loc_from_key(key) {
                Some((l, _)) => l,
                None => continue,
            };
            (
                tile.player,
                tile.flipped,
                tile.ind,
                tile.def.beers_to_sell.unwrap_or(0),
                loc,
            )
        };
        if t_player != pid || t_flipped {
            continue;
        }

        let valid_merchants = sell_merchants_for(state, pid, loc, ind);
        if valid_merchants.is_empty() {
            continue;
        }

        let mut beer_remaining = beer_needed;
        let chosen_merchant = merchant_indices[entry_idx];
        let use_merchant = use_merchant_beer.get(entry_idx).copied().unwrap_or(false);

        if !valid_merchants.contains(&chosen_merchant) {
            return Err("Chosen merchant is not a valid buyer".into());
        }

        // Merchant beer first, if requested: use the explicitly chosen merchant.
        if beer_remaining > 0 && use_merchant {
            if !state.merchants[chosen_merchant].has_beer {
                return Err("Chosen merchant has no beer".into());
            }
            if beer_sources[entry_idx].len() + 1 < beer_remaining as usize {
                return Err("Not enough beer available for chosen merchant route".into());
            }
            let mt_loc = state.merchants[chosen_merchant].loc;
            state.merchants[chosen_merchant].has_beer = false;
            beer_remaining -= 1;
            beer_notes.push(format!(
                "{} 使用 商家酒@{}",
                log_industry_label(ind),
                log_loc_label(mt_loc)
            ));
            let bonus = crate::map::merchant_bonus_at(mt_loc);
            let note = apply_merchant_bonus(state, pid, bonus);
            notes.push(note);
        }

        if beer_sources[entry_idx].len() != beer_remaining as usize {
            return Err("Sell beer payment has wrong number of sources".into());
        }

        // Remaining beer from breweries (own anywhere, opponents connected)
        if beer_remaining > 0 {
            let available = find_beer_sources(state, loc, pid, &[]);
            let mut remaining_available = available;
            for source in &beer_sources[entry_idx] {
                let Some(position) = remaining_available
                    .iter()
                    .position(|candidate| candidate == source)
                else {
                    return Err("Chosen brewery beer source is unavailable or unreachable".into());
                };
                remaining_available.remove(position);
            }
            let consumed = &beer_sources[entry_idx];
            let labels: Vec<String> = consumed
                .iter()
                .map(|b| beer_source_label(state, b))
                .collect();
            for source in consumed {
                state.consume_beer_source(source);
            }
            if !labels.is_empty() {
                beer_notes.push(format!(
                    "{} 使用 {}",
                    log_industry_label(ind),
                    labels.join(", ")
                ));
            }
        }

        // Flip the tile (advances income)
        let (player, income) = {
            let Some(t) = state.city_tiles[key].as_mut() else {
                continue;
            };
            if t.flipped {
                continue;
            }
            t.flipped = true;
            (t.player, t.def.income)
        };
        state.advance_income_spaces(player, income);
        sold += 1;
    }

    if sold == 0 {
        return Err("Nothing could be sold".into());
    }

    if let Some(ind) = free_develop {
        let tile = state.players[pid]
            .consume_tile(ind)
            .ok_or("No tile available to free develop")?;
        match state.era {
            crate::data::Era::Canal => state.players[pid].develops_in_canal += 1,
            crate::data::Era::Rail => state.players[pid].develops_in_rail += 1,
        }
        notes.push(format!(
            "free develop {} Lv{}",
            log_industry_label(ind),
            tile.level
        ));
    }

    discard_card(state, pid, card_index);
    let mut extra = Vec::new();
    if !notes.is_empty() {
        extra.push(notes.join("; "));
    }
    if !beer_notes.is_empty() {
        extra.push(format!("beer: {}", beer_notes.join("; ")));
    }
    Ok(format!(
        "Sold {} tile(s){}",
        sold,
        if extra.is_empty() {
            String::new()
        } else {
            format!(" [{}]", extra.join(" | "))
        }
    ))
}

fn apply_merchant_bonus(state: &mut GameState, pid: usize, bonus: MerchantBonus) -> String {
    match bonus {
        MerchantBonus::Vp(v) => {
            state.players[pid].vp += v as u16;
            format!("+{} VP (merchant)", v)
        }
        MerchantBonus::Money(m) => {
            state.gain_money(pid, m);
            format!("+£{} (merchant)", m)
        }
        MerchantBonus::Income(spaces) => {
            state.advance_income_spaces(pid, spaces);
            format!("+{} income spaces (merchant)", spaces)
        }
        MerchantBonus::Develop(_) => "free develop".to_string(),
    }
}
