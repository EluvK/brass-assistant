//! 2-ply deterministic lookahead.
//!
//! For each candidate first action (the 1-ply candidate set), simulate it, then
//! score the best same-player second action (the natural "same-turn combo",
//! e.g. Build -> Sell, Network -> Build). The value of a first action is
//! `score(first) + ALPHA * score(best_second_after_first)`.
//!
//! Anchoring on the 1-ply score (ALPHA < 1) keeps decisions consistent with
//! the calibrated heuristic while adding a bonus for actions that unlock a
//! strong follow-up — i.e. the intra-turn combos a greedy 1-ply cannot see.

use crate::engine::{advance_turn, end_canal_era, end_game, TurnResult};
use crate::heuristic_ai::{self, Decision};
use crate::rules::apply_move;
use crate::state::GameState;

/// Discount applied to the look-ahead (best second action) score.
const ALPHA: f64 = 0.6;

/// Choose the current player's action by 2-ply lookahead over their two
/// same-turn actions.
pub fn choose_action_2ply<R: rand::Rng + Clone>(state: &mut GameState<R>) -> Decision {
    let pid = state.current_player_id();
    let first_candidates = heuristic_ai::candidate_actions(state);

    let mut best: Option<(crate::rules::Move, f64)> = None;
    for c1 in first_candidates {
        // Simulate the first action on a clone.
        let mut s1 = state.clone();
        if apply_move(&mut s1, &c1.mv).is_err() {
            continue;
        }
        let tr = advance_turn(&mut s1);
        handle_turn_result(&mut s1, tr);

        // If the same player still has a second action this turn, add a
        // discounted bonus for the best same-player continuation.
        let value = if s1.current_player_id() == pid {
            let second_candidates = heuristic_ai::candidate_actions(&mut s1);
            let mut best_second = f64::NEG_INFINITY;
            for c2 in second_candidates {
                if c2.score > best_second {
                    best_second = c2.score;
                }
            }
            c1.score + ALPHA * best_second.max(0.0)
        } else {
            // First action ended the turn: no combo bonus.
            c1.score
        };

        let better = match &best {
            Some((_, bv)) => value > *bv,
            None => true,
        };
        if better {
            best = Some((c1.mv.clone(), value));
        }
    }

    best.map(|(mv, score)| Decision { mv, score })
        .unwrap_or_else(|| heuristic_ai::pass_decision(state))
}

/// Apply era-end / game-end transitions to a simulated state.
fn handle_turn_result<R: rand::Rng>(state: &mut GameState<R>, tr: TurnResult) {
    match tr {
        TurnResult::Continue => {}
        TurnResult::EndCanalEra => end_canal_era(state),
        TurnResult::EndGame => end_game(state),
    }
}
