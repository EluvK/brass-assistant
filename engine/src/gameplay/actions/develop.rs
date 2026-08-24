//! DEVELOP action legality and execution.

use super::{
    affordable_source_count, discard_card, log_industry_label, require_card_index,
    source_purchase_cost, validate_iron_choice,
};
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
    affordable_source_count(&find_iron_sources(state), state.players[pid].money, 2)
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

    // Validate in development order on a player-mat copy. In particular, a
    // same-industry double develop must validate the tile uncovered by the
    // first removal, rather than checking the original top tile twice.
    let mut simulated_player = state.players[pid].clone();
    let first_tile = simulated_player
        .next_tile(ind1)
        .filter(|tile| tile.can_develop)
        .ok_or("First industry is not developable")?;
    simulated_player
        .consume_tile(ind1)
        .expect("validated first develop tile must be present");

    let second_tile = if let Some(ind2) = ind2 {
        let tile = simulated_player
            .next_tile(ind2)
            .filter(|tile| tile.can_develop)
            .ok_or("Second industry is not developable")?;
        simulated_player
            .consume_tile(ind2)
            .expect("validated second develop tile must be present");
        Some(tile)
    } else {
        None
    };
    validate_iron_choice(state, tiles_to_develop, iron)?;

    let money_needed = source_purchase_cost(iron, tiles_to_develop)
        .expect("validated iron choice must contain every required source");
    if money_needed > state.players[pid].money {
        return Err("Cannot afford market iron".into());
    }

    // Consume exactly the chosen iron sources.
    for src in iron {
        if !src.free {
            state.spend_money(pid, src.price as i32);
        }
        state.consume_iron_source(src);
    }

    // Remove tiles from mat
    state.players[pid]
        .consume_tile(ind1)
        .expect("validated first develop tile must be present");
    let mut removed = vec![format!(
        "{} Lv{}",
        log_industry_label(ind1),
        first_tile.level
    )];
    if let (Some(i2), Some(t2)) = (ind2, second_tile) {
        state.players[pid]
            .consume_tile(i2)
            .expect("validated second develop tile must be present");
        removed.push(format!("{} Lv{}", log_industry_label(i2), t2.level));
    }

    discard_card(state, pid, card_index);
    Ok(format!("Developed {}", removed.join(" + ")))
}
