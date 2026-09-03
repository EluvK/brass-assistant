//! Stable public facade for action rules.
//!
//! Action-specific legality and execution live in [`crate::gameplay::actions`];
//! concrete legal-move enumeration lives in [`crate::gameplay::legal_resolved_moves`].

use crate::state::GameState;

pub use crate::gameplay::actions::{
    BuildCost, BuildTarget, DoubleRailCandidate, SellExecutionPlan, SellIdentity, any_card_indices,
    beer_sources_for_link, calculate_build_cost, can_develop, can_scout,
    coal_options_for_connection, coal_source_options, discard_card, execute_build, execute_develop,
    execute_loan, execute_network, execute_network_double, execute_pass, execute_scout,
    enumerate_double_rail_candidates, execute_sell_with_free_develop, get_valid_build_targets,
    get_valid_network_targets, iron_source_options, sell_identity,
    valid_build_cards, validate_sell_plan,
};
pub use crate::gameplay::legal_moves::{legal_moves, legal_resolved_moves};
pub use crate::r#move::{Move, ResolvedMove};
pub fn apply_move(state: &mut GameState, mv: &ResolvedMove) -> Result<String, String> {
    // The public canonical-move boundary accepts decoded, potentially stale
    // input. Keep it atomic even if a future executor adds a late validation.
    let snapshot = state.clone();
    let pid = state.current_player_id();
    let res = match mv {
        ResolvedMove::Build {
            loc,
            slot_index,
            ind,
            coal,
            iron,
            card_index,
        } => execute_build(state, pid, *loc, *slot_index, *ind, coal, iron, *card_index),
        ResolvedMove::Network {
            conn_id,
            coal,
            card_index,
        } => execute_network(state, pid, *conn_id, *coal, *card_index),
        ResolvedMove::NetworkDouble {
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
        ResolvedMove::Develop {
            ind1,
            ind2,
            iron,
            card_index,
        } => execute_develop(state, pid, *ind1, *ind2, iron, *card_index),
        ResolvedMove::Sell {
            keys,
            beer_sources,
            free_develop,
            card_index,
        } => execute_sell_with_free_develop(
            state,
            pid,
            keys,
            beer_sources,
            *free_develop,
            *card_index,
        ),
        ResolvedMove::Loan { card_index } => execute_loan(state, pid, *card_index),
        ResolvedMove::Scout { card_indices } => execute_scout(state, pid, *card_indices),
        ResolvedMove::Pass { card_index } => execute_pass(state, pid, *card_index),
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
