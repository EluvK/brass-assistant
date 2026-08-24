//! Stable public facade for action rules.
//!
//! Action-specific legality and execution live in [`crate::gameplay::actions`];
//! concrete legal-move enumeration lives in [`crate::gameplay::legal_moves`].

use crate::state::GameState;

pub use crate::gameplay::actions::{
    BuildCost, BuildTarget, SecondRailOption, SellRoute, SellTarget, any_card_indices,
    beer_sources_for_link, calculate_build_cost, can_develop, can_scout,
    coal_options_for_connection, coal_source_options, discard_card, execute_build, execute_develop,
    execute_loan, execute_network, execute_network_double, execute_pass, execute_scout,
    execute_sell, execute_sell_with_free_develop, get_second_rail_options, get_valid_build_targets,
    get_valid_network_targets, get_valid_second_rail_links, get_valid_sell_targets,
    iron_source_options, plan_sell_beer_sources, valid_build_cards,
};
pub use crate::gameplay::legal_moves::legal_moves;
pub use crate::r#move::Move;
pub fn apply_move(state: &mut GameState, mv: &Move) -> Result<String, String> {
    // The public canonical-move boundary accepts decoded, potentially stale
    // input. Keep it atomic even if a future executor adds a late validation.
    let snapshot = state.clone();
    let pid = state.current_player_id();
    let res = match mv {
        Move::Build {
            loc,
            slot_index,
            ind,
            coal,
            iron,
            card_index,
        } => execute_build(state, pid, *loc, *slot_index, *ind, coal, iron, *card_index),
        Move::Network {
            conn_id,
            coal,
            card_index,
        } => execute_network(state, pid, *conn_id, *coal, *card_index),
        Move::NetworkDouble {
            conn1,
            conn2,
            coal1,
            coal2,
            beer,
            card_index,
        } => execute_network_double(
            state,
            pid,
            *conn1,
            *conn2,
            *coal1,
            *coal2,
            *beer,
            *card_index,
        ),
        Move::Develop {
            ind1,
            ind2,
            iron,
            card_index,
        } => execute_develop(state, pid, *ind1, *ind2, iron, *card_index),
        Move::Sell {
            keys,
            merchant_indices,
            use_merchant_beer,
            beer_sources,
            free_develop,
            card_index,
        } => execute_sell_with_free_develop(
            state,
            pid,
            keys,
            merchant_indices,
            use_merchant_beer,
            beer_sources,
            *free_develop,
            *card_index,
        ),
        Move::Loan { card_index } => execute_loan(state, pid, *card_index),
        Move::Scout { card_indices } => execute_scout(state, pid, *card_indices),
        Move::Pass { card_index } => execute_pass(state, pid, *card_index),
    };
    match res {
        Ok(summary) => {
            #[cfg(debug_assertions)]
            state.assert_caches_consistent();
            Ok(summary)
        }
        Err(err) => {
            *state = snapshot;
            Err(err)
        }
    }
}
