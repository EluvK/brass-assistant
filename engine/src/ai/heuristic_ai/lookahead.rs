//! Deterministic same-turn lookahead for the heuristic policy.

use crate::data::Era;
use crate::engine::{advance_turn, handle_turn_result};
use crate::rules::apply_move;
use crate::state::GameState;

use super::{Decision, candidate_actions_k, era_profile, pass_decision};

const FIRST_ACTION_K: usize = 3;
const SECOND_ACTION_K: usize = 2;
const LOW_MONEY_THRESHOLD: i32 = 15;
const END_TURN_PENALTY_SCALE: f64 = 6.5;

/// Choose the current player's action with deterministic 2-ply lookahead.
pub fn choose_action(state: &mut GameState) -> Decision {
    let pid = state.current_player_id();
    let first_candidates = candidate_actions_k(state, FIRST_ACTION_K);
    let income_before = state.players[pid].income_level();

    let mut best: Option<(crate::rules::ResolvedMove, f64)> = None;
    for c1 in first_candidates {
        let mut s1 = state.clone();
        if apply_move(&mut s1, &c1.mv).is_err() {
            continue;
        }
        let tr = advance_turn(&mut s1);
        handle_turn_result(&mut s1, tr);

        let value = if s1.current_player_id() == pid {
            let second_candidates = candidate_actions_k(&mut s1, SECOND_ACTION_K);
            let best_second = second_candidates
                .iter()
                .map(|c| c.score)
                .fold(f64::NEG_INFINITY, f64::max);
            if let Some(c2) = second_candidates
                .into_iter()
                .max_by(|a, b| a.score.partial_cmp(&b.score).unwrap())
            {
                if apply_move(&mut s1, &c2.mv).is_ok() {
                    let tr2 = advance_turn(&mut s1);
                    handle_turn_result(&mut s1, tr2);
                }
            }
            c1.score + era_profile(state).alpha * best_second.max(0.0)
        } else {
            c1.score
        };

        let value = value + end_of_turn_penalty(&s1, pid, income_before);
        if best.as_ref().is_none_or(|(_, score)| value > *score) {
            best = Some((c1.mv, value));
        }
    }

    best.map(|(mv, score)| {
        let card_score = super::move_card_score(state, &mv);
        Decision {
            mv,
            score,
            card_score,
        }
    })
    .unwrap_or_else(|| pass_decision(state))
}

fn end_of_turn_penalty(state: &GameState, pid: usize, income_before: i8) -> f64 {
    let p = &state.players[pid];
    if p.money >= LOW_MONEY_THRESHOLD {
        return 0.0;
    }
    let income_delta = p.income_level() as f64 - income_before as f64;
    if income_delta >= 2.5 {
        return 0.0;
    }
    let scarcity =
        ((LOW_MONEY_THRESHOLD - p.money) as f64 / LOW_MONEY_THRESHOLD as f64).clamp(0.0, 1.0);
    let income_term = if p.income_level() < 0 { 1.4 } else { 0.9 };
    let rounds = state.rounds_remaining();
    let runway = (rounds / 8.0).clamp(0.0, 1.0);
    let era_term = if state.era == Era::Rail { 1.0 } else { 0.8 };
    let runway_term = 0.6 + 0.4 * (1.0 - runway);
    -END_TURN_PENALTY_SCALE * scarcity * income_term * era_term * runway_term
}
