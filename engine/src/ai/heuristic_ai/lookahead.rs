//! Deterministic same-turn lookahead for the heuristic policy.
//!
//! Expands the top first actions, simulates each, and (when the turn
//! continues with us) blends in the best second action. A turn-end cash
//! safety penalty keeps the policy from stranding itself broke.

use super::config::HeuristicConfig;
use super::context::ERA_ROUNDS;
use super::{Decision, candidate_actions_k, pass_decision};
use crate::engine::{advance_turn, handle_turn_result};
use crate::rules::{ResolvedMove, apply_move};
use crate::state::GameState;

/// Choose the current player's action with deterministic 2-ply lookahead.
pub fn choose_action(state: &mut GameState) -> Decision {
    let cfg = HeuristicConfig::default();
    let pid = state.current_player_id();
    let ctx = super::context::EvalContext::new(state, pid, &cfg);
    let first_candidates = candidate_actions_k(state, cfg.lookahead.first_action_k);
    let income_before = state.players[pid].income_level();

    let mut best: Option<(ResolvedMove, f64)> = None;
    for c1 in first_candidates {
        let mut s1 = state.clone();
        if apply_move(&mut s1, &c1.mv).is_err() {
            continue;
        }
        let tr = advance_turn(&mut s1);
        handle_turn_result(&mut s1, tr);

        let value = if s1.current_player_id() == pid {
            let second_candidates = candidate_actions_k(&mut s1, cfg.lookahead.second_action_k);
            let best_second = second_candidates
                .iter()
                .map(|c| c.score)
                .fold(f64::NEG_INFINITY, f64::max);
            if let Some(c2) = second_candidates
                .into_iter()
                .max_by(|a, b| a.score.total_cmp(&b.score))
                && apply_move(&mut s1, &c2.mv).is_ok()
            {
                let tr2 = advance_turn(&mut s1);
                handle_turn_result(&mut s1, tr2);
            }
            // 2-ply blend: a productive second action adds to the turn's
            // value, discounted by the phase's alpha.
            c1.score + ctx.profile.alpha * best_second.max(0.0)
        } else {
            c1.score
        };

        let value = value + end_of_turn_penalty(&s1, pid, income_before, &cfg);
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

/// Penalise ending the turn broke: the deeper below the cash line and the
/// more rounds still to survive, the worse the position (an income gain
/// this turn can buy back the safety, and is exempted).
fn end_of_turn_penalty(
    state: &GameState,
    pid: usize,
    income_before: i8,
    cfg: &HeuristicConfig,
) -> f64 {
    let lw = &cfg.lookahead;
    let p = &state.players[pid];
    if p.money >= lw.low_money_threshold {
        return 0.0;
    }
    let income_delta = p.income_level() as f64 - income_before as f64;
    if income_delta >= lw.end_turn_income_exempt {
        return 0.0;
    }
    let scarcity =
        ((lw.low_money_threshold - p.money) as f64 / lw.low_money_threshold as f64).clamp(0.0, 1.0);
    let income_term = if p.income_level() < 0 {
        lw.end_turn_negative_income_weight
    } else {
        lw.end_turn_income_weight
    };
    let runway = (state.rounds_remaining() / ERA_ROUNDS).clamp(0.0, 1.0);
    let era_term = if state.era == crate::data::Era::Rail {
        lw.end_turn_rail_era_term
    } else {
        lw.end_turn_canal_era_term
    };
    let runway_term = lw.end_turn_runway_base + lw.end_turn_runway_span * (1.0 - runway);
    -lw.end_turn_penalty_scale * scarcity * income_term * era_term * runway_term
}
