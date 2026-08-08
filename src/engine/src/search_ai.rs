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
//!
//! Two safeguards against the late-game "starve-then-pass" spiral observed in
//! low-VP seeds (e.g. seed=239: players loaned into deep negative income and
//! then idled passing every turn):
//!
//! * The combo bonus is dampened in the late rail era, so short-term
//!   Loan -> Network (and similar) combos stop overriding the heuristic's own
//!   debt-spiral guardrails.
//! * The end-of-turn position is assessed for liquidity: if the turn ends with
//!   almost no cash, no income improvement, and little runway left, the line is
//!   penalized (this suppresses lines that leave the player unable to act next
//!   turn).

use crate::engine::{advance_turn, end_canal_era, end_game, TurnResult};
use crate::heuristic_ai::{self, Decision};
use crate::rules::apply_move;
use crate::state::GameState;
use crate::data::Era;

/// Discount applied to the look-ahead (best second action) score.
/// (Values now live in `EraProfile::alpha`, defined in heuristic_ai.)
/// Wider first-action candidate pool: helps uncover tactical combos such as
/// Network -> Build and Loan -> Build that can be missed by a single
/// representative candidate per action type.
const FIRST_ACTION_K: usize = 3;
/// Slightly wider second-action set for better same-turn continuation quality.
const SECOND_ACTION_K: usize = 2;

/// Cash at or below which the end-of-turn liquidity check starts to bite.
const LOW_MONEY_THRESHOLD: i32 = 15;
/// Scales the end-of-turn liquidity penalty (worst case ~ -10).
const END_TURN_PENALTY_SCALE: f64 = 6.5;

/// Choose the current player's action by 2-ply lookahead over their two
/// same-turn actions.
pub fn choose_action_2ply<R: rand::Rng + Clone>(state: &mut GameState<R>) -> Decision {
    let pid = state.current_player_id();
    let first_candidates = heuristic_ai::candidate_actions_k(state, FIRST_ACTION_K);
    let income_before = state.players[pid].income_level();

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
        // discounted bonus for the best same-player continuation and apply it
        // so `s1` reflects the real end-of-turn position.
        let value = if s1.current_player_id() == pid {
            let second_candidates = heuristic_ai::candidate_actions_k(&mut s1, SECOND_ACTION_K);
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
            c1.score + combo_alpha(state) * best_second.max(0.0)
        } else {
            // First action ended the turn: no combo bonus.
            c1.score
        };

        let value = value + end_of_turn_penalty(&s1, pid, income_before);

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

/// Era-aware combo discount: full weight early, dampened in the late rail.
fn combo_alpha<R: rand::Rng>(state: &GameState<R>) -> f64 {
    heuristic_ai::era_profile(state).alpha
}

/// Penalize lines that strand the player without cash at the end of their
/// turn, unless the turn bought a clear income jump (a good flip) which makes
/// low cash acceptable.
fn end_of_turn_penalty<R: rand::Rng>(
    state: &GameState<R>,
    pid: usize,
    income_before: i8,
) -> f64 {
    let p = &state.players[pid];
    if p.money >= LOW_MONEY_THRESHOLD {
        return 0.0;
    }
    let income_delta = p.income_level() as f64 - income_before as f64;
    if income_delta >= 2.5 {
        // All-in for a strong income flip is fine even at zero cash.
        return 0.0;
    }

    let scarcity =
        ((LOW_MONEY_THRESHOLD - p.money) as f64 / LOW_MONEY_THRESHOLD as f64).clamp(0.0, 1.0);
    // Negative income means there's no passive recovery from the hole.
    let income_term = if p.income_level() < 0 { 1.4 } else { 0.9 };
    let rounds = heuristic_ai::estimate_rounds_remaining(state);
    let runway = (rounds / 8.0).clamp(0.0, 1.0);
    let era_term = if state.era == Era::Rail { 1.0 } else { 0.8 };
    let runway_term = 0.6 + 0.4 * (1.0 - runway);
    -END_TURN_PENALTY_SCALE * scarcity * income_term * era_term * runway_term
}

/// Apply era-end / game-end transitions to a simulated state.
fn handle_turn_result<R: rand::Rng>(state: &mut GameState<R>, tr: TurnResult) {
    match tr {
        TurnResult::Continue => {}
        TurnResult::EndCanalEra => end_canal_era(state),
        TurnResult::EndGame => end_game(state),
    }
}
