//! SELL action identity, validation and execution.

use super::{discard_card, log_industry_label, require_card_index};
use crate::data::IndustryType;
use crate::graph::{BeerSource, BeerSourceKind, connected_locations, find_beer_sources};
use crate::map::{Loc, MerchantBonus};
use crate::state::GameState;

/// Structural sell identity; buyer choice and source-to-tile allocation are
/// deliberately execution details, not part of an action.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct SellIdentity {
    pub keys: Vec<usize>,
    pub beer_sources: Vec<BeerSource>,
    pub free_develop: Option<IndustryType>,
}

pub fn beer_source_sort_key(s: &BeerSource) -> (u8, u8, usize, usize) {
    let kind = match s.kind {
        BeerSourceKind::Own => 0,
        BeerSourceKind::Opponent => 1,
        BeerSourceKind::Merchant => 2,
    };
    match s.kind {
        BeerSourceKind::Merchant => (kind, 2, s.merchant_idx.unwrap_or(usize::MAX), 0),
        _ if s.farm_idx.is_some() => (kind, 1, s.farm_idx.unwrap(), s.key),
        _ => (kind, 0, s.key, s.farm_idx.unwrap_or(usize::MAX)),
    }
}

pub fn sell_identity(
    keys: &[usize],
    sources: &[BeerSource],
    free_develop: Option<IndustryType>,
) -> SellIdentity {
    let mut keys = keys.to_vec();
    keys.sort_unstable();
    let mut beer_sources = sources.to_vec();
    beer_sources.sort_by_key(beer_source_sort_key);
    SellIdentity {
        keys,
        beer_sources,
        free_develop,
    }
}

#[derive(Debug, Clone)]
pub struct SellAssignment {
    pub key: usize,
    pub buyer: usize,
    pub beer_sources: Vec<BeerSource>,
}
#[derive(Debug, Clone)]
pub struct SellExecutionPlan {
    pub assignments: Vec<SellAssignment>,
    pub free_develop: Option<IndustryType>,
}

fn buyers(state: &GameState, loc: Loc, ind: IndustryType) -> Vec<usize> {
    let connected = connected_locations(state, loc);
    state
        .merchants
        .iter()
        .enumerate()
        .filter_map(|(i, m)| (m.accepts(ind) && connected.contains(&m.loc)).then_some(i))
        .collect()
}

fn source_available(state: &GameState, pid: usize, loc: Loc, source: &BeerSource) -> bool {
    match source.kind {
        BeerSourceKind::Merchant => source
            .merchant_idx
            .is_some_and(|i| state.merchants.get(i).is_some_and(|m| m.has_beer)),
        _ => find_beer_sources(state, loc, pid, &[]).contains(source),
    }
}

/// Validate the canonical representation and reconstruct one deterministic
/// assignment. The chosen source list must be exact: no unused barrels.
pub fn validate_sell_plan(
    state: &GameState,
    pid: usize,
    keys: &[usize],
    sources: &[BeerSource],
    free_develop: Option<IndustryType>,
) -> Result<SellExecutionPlan, String> {
    let identity = sell_identity(keys, sources, free_develop);
    if keys.is_empty() || identity.keys.len() != keys.len() {
        return Err("Sell must contain unique non-empty keys".into());
    }
    if identity.keys != keys || identity.beer_sources != sources {
        return Err("Sell keys and beer sources must be canonical sorted".into());
    }
    for &key in keys {
        let tile = state
            .city_tiles
            .get(key)
            .and_then(Option::as_ref)
            .ok_or("Invalid sell tile")?;
        let loc = crate::state::loc_from_key(key)
            .map(|x| x.0)
            .ok_or("Invalid sell tile")?;
        if tile.player != pid || tile.flipped || !tile.ind.is_sellable() {
            return Err("Tile cannot be sold".into());
        }
        if buyers(state, loc, tile.ind).is_empty() {
            return Err("Sell tile has no valid buyer".into());
        }
    }
    fn rec(
        state: &GameState,
        pid: usize,
        keys: &[usize],
        sources: &[BeerSource],
        used: &mut [bool],
        at: usize,
        out: &mut Vec<SellAssignment>,
    ) -> bool {
        if at == keys.len() {
            return used.iter().all(|x| *x);
        }
        let key = keys[at];
        let tile = state.city_tiles[key].as_ref().unwrap();
        let loc = crate::state::loc_from_key(key).unwrap().0;
        let need = tile.def.beers_to_sell.unwrap_or(0) as usize;
        fn choose(
            state: &GameState,
            pid: usize,
            keys: &[usize],
            sources: &[BeerSource],
            used: &mut [bool],
            at: usize,
            key: usize,
            loc: Loc,
            ind: IndustryType,
            start: usize,
            left: usize,
            picked: &mut Vec<BeerSource>,
            out: &mut Vec<SellAssignment>,
        ) -> bool {
            if left == 0 {
                let merchant: Vec<_> = picked
                    .iter()
                    .filter_map(|s| {
                        (s.kind == BeerSourceKind::Merchant).then(|| s.merchant_idx.unwrap())
                    })
                    .collect();
                if merchant.len() > 1 {
                    return false;
                }
                let valid = buyers(state, loc, ind);
                let buyer = merchant
                    .first()
                    .copied()
                    .filter(|m| valid.contains(m))
                    .or_else(|| valid.first().copied());
                let Some(buyer) = buyer else { return false };
                out.push(SellAssignment {
                    key,
                    buyer,
                    beer_sources: picked.clone(),
                });
                if rec(state, pid, keys, sources, used, at + 1, out) {
                    return true;
                }
                out.pop();
                return false;
            }
            for i in start..sources.len() {
                if used[i] || !source_available(state, pid, loc, &sources[i]) {
                    continue;
                }
                used[i] = true;
                picked.push(sources[i]);
                if choose(
                    state,
                    pid,
                    keys,
                    sources,
                    used,
                    at,
                    key,
                    loc,
                    ind,
                    i + 1,
                    left - 1,
                    picked,
                    out,
                ) {
                    return true;
                }
                picked.pop();
                used[i] = false;
            }
            false
        }
        choose(
            state,
            pid,
            keys,
            sources,
            used,
            at,
            key,
            loc,
            tile.ind,
            0,
            need,
            &mut Vec::new(),
            out,
        )
    }
    let mut assignments = Vec::new();
    let mut used = vec![false; sources.len()];
    if !rec(state, pid, keys, sources, &mut used, 0, &mut assignments) {
        return Err("Beer sources cannot satisfy this sell plan".into());
    }
    let dev_awards = assignments
        .iter()
        .filter(|a| {
            a.beer_sources.iter().any(|s| {
                s.kind == BeerSourceKind::Merchant
                    && s.merchant_idx.is_some_and(|i| {
                        matches!(
                            crate::map::merchant_bonus_at(state.merchants[i].loc),
                            MerchantBonus::Develop(_)
                        )
                    })
            })
        })
        .count();
    match (dev_awards, free_develop) {
        (0, None) => {}
        (0, Some(_)) => return Err("Sell does not award a free develop".into()),
        (1, Some(ind))
            if state.players[pid]
                .developable_types()
                .iter()
                .any(|(x, _)| *x == ind) => {}
        (1, Some(_)) => return Err("Free develop choice is not developable".into()),
        (1, None) => return Err("Sell must choose its free develop".into()),
        _ => return Err("A sell action cannot award more than one free develop".into()),
    }
    Ok(SellExecutionPlan {
        assignments,
        free_develop,
    })
}

pub fn execute_sell_with_free_develop(
    state: &mut GameState,
    pid: usize,
    keys: &[usize],
    sources: &[BeerSource],
    free_develop: Option<IndustryType>,
    card_index: usize,
) -> Result<String, String> {
    let snapshot = state.clone();
    let result = (|| {
        require_card_index(state, pid, card_index)?;
        let plan = validate_sell_plan(state, pid, keys, sources, free_develop)?;
        let mut notes = Vec::new();
        for a in &plan.assignments {
            for source in &a.beer_sources {
                if source.kind == BeerSourceKind::Merchant {
                    let i = source.merchant_idx.unwrap();
                    let loc = state.merchants[i].loc;
                    state.merchants[i].has_beer = false;
                    notes.push(apply_merchant_bonus(
                        state,
                        pid,
                        crate::map::merchant_bonus_at(loc),
                    ));
                } else {
                    state.consume_beer_source(source);
                }
            }
        }
        for a in &plan.assignments {
            let income = {
                let tile = state.city_tiles[a.key].as_mut().unwrap();
                tile.flipped = true;
                tile.def.income
            };
            state.advance_income_spaces(pid, income);
        }
        if let Some(ind) = plan.free_develop {
            let tile = state.players[pid]
                .consume_tile(ind)
                .ok_or("No tile available to free develop")?;
            match state.era {
                crate::data::Era::Canal => state.players[pid].develops_in_canal += 1,
                crate::data::Era::Rail => state.players[pid].develops_in_rail += 1,
            };
            notes.push(format!(
                "free develop {} Lv{}",
                log_industry_label(ind),
                tile.level
            ));
        }
        discard_card(state, pid, card_index);
        Ok(format!(
            "Sold {} tile(s){}",
            plan.assignments.len(),
            if notes.is_empty() {
                String::new()
            } else {
                format!(" [{}]", notes.join("; "))
            }
        ))
    })();
    if result.is_err() {
        *state = snapshot;
    }
    result
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
        MerchantBonus::Income(s) => {
            state.advance_income_spaces(pid, s);
            format!("+{} income spaces (merchant)", s)
        }
        MerchantBonus::Develop(_) => "free develop".into(),
    }
}
