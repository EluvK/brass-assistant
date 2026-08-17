//! DEVELOP action legality and execution.

use super::{discard_card, log_industry_label, require_card_index, validate_iron_choice};
use crate::data::IndustryType;
use crate::graph::{IronSource, find_iron_sources};
use crate::state::GameState;
// ---------------------------------------------------------------------------
// DEVELOP
// ---------------------------------------------------------------------------

pub fn can_develop(state: &GameState, pid: usize) -> bool {
    affordable_develop_iron_count(state, pid) >= 1
        && !state.players[pid].developable_types().is_empty()
}

pub(crate) fn affordable_develop_iron_count(state: &GameState, pid: usize) -> usize {
    let mut money = state.players[pid].money;
    let mut count = 0usize;

    for src in find_iron_sources(state) {
        if src.free {
            count += 1;
        } else if money >= src.price as i32 {
            money -= src.price as i32;
            count += 1;
        } else {
            break;
        }
    }

    count
}

pub fn execute_develop(
    state: &mut GameState,
    pid: usize,
    ind1: IndustryType,
    ind2: Option<IndustryType>,
    iron: &[IronSource],
    card_index: usize,
) -> Result<String, String> {
    require_card_index(state, pid, card_index)?;
    let tiles_to_develop = if ind2.is_some() { 2 } else { 1 };
    let available: Vec<IndustryType> = state.players[pid]
        .developable_types()
        .into_iter()
        .map(|(ind, _)| ind)
        .collect();
    if !available.contains(&ind1) {
        return Err("First industry is not developable".into());
    }
    if let Some(ind2) = ind2 {
        if !available.contains(&ind2) {
            return Err("Second industry is not developable".into());
        }
        if ind1 == ind2 && state.players[pid].remaining_count(ind1) < 2 {
            return Err("Not enough tiles remaining to develop this industry twice".into());
        }
    }
    validate_iron_choice(state, tiles_to_develop, iron)?;

    // Pre-check affordability of market iron
    let mut money_needed = 0;
    for src in iron {
        if !src.free {
            money_needed += src.price as i32;
        }
    }
    if money_needed > state.players[pid].money {
        return Err("Cannot afford market iron".into());
    }

    // Consume exactly the chosen iron sources.
    for src in iron {
        if src.free {
            state.consume_from_city(src.key);
        } else {
            state.spend_money(pid, src.price as i32);
            state.take_market_iron();
        }
    }

    // Remove tiles from mat
    let t1 = state.players[pid]
        .consume_tile(ind1)
        .ok_or("No tile available to develop for first industry")?;
    let mut removed = vec![format!("{} Lv{}", log_industry_label(ind1), t1.level)];
    if let Some(i2) = ind2 {
        let t2 = state.players[pid]
            .consume_tile(i2)
            .ok_or("No tile available to develop for second industry")?;
        removed.push(format!("{} Lv{}", log_industry_label(i2), t2.level));
    }

    discard_card(state, pid, card_index);
    Ok(format!("Developed {}", removed.join(" + ")))
}
