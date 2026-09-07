//! Deterministic tactical lookahead for the heuristic policy.
//!
//! Expands the top first actions, simulates each, and (when the turn
//! continues with us) blends in the best second action.  A last-position
//! prerequisite plus a post-two-action low-spend check additionally values
//! two actions in the next round.

use super::config::HeuristicConfig;
use super::{Decision, RoundDecision, candidate_actions_k, pass_decision};
use crate::engine::{advance_turn, handle_turn_result};
use crate::rules::{ResolvedMove, apply_move};
use crate::state::GameState;

/// Choose the current player's action with deterministic 2-ply lookahead and
/// an optional cross-round four-link extension.  The returned round plan is
/// meaningful at the start of a player's action block (`actions_this_turn ==
/// 0`); per-action callers that are already inside the block receive a
/// one-move plan because `continues_same_turn` only recognises the second
/// action of the same block.
pub fn choose_action(state: &mut GameState) -> RoundDecision {
    let cfg = HeuristicConfig::default();

    // The first round has a dedicated, progressively extensible opening
    // template.  Keep candidate generation and template selection together
    // so the normal lookahead path is not involved at all.
    if state.is_first_round {
        return round_from_first(choose_opening(state).unwrap_or_else(|| pass_decision(state)));
    }

    let pid = state.current_player_id();
    if is_last_in_order(state, pid) {
        // Being last is the structural prerequisite for a four-link.  The
        // spend-lead check remains in the second-action evaluator, after the
        // first two actions have been simulated.
        return choose_four_action(state, &cfg);
    }

    choose_lookahead(state, &cfg)
}

/// Evaluate a last-in-order turn, including a possible four-link extension.
fn choose_four_action(state: &mut GameState, cfg: &HeuristicConfig) -> RoundDecision {
    let pid = state.current_player_id();

    let ctx = super::context::EvalContext::new(state, pid, cfg);
    let first_candidates = candidate_actions_k(state, cfg.lookahead.first_action_k);
    let mut best: Option<(ResolvedMove, Option<Decision>, f64)> = None;

    for c1 in first_candidates {
        let mut after_first = state.clone();
        if apply_move(&mut after_first, &c1.mv).is_err() {
            continue;
        }
        let tr = advance_turn(&mut after_first);
        handle_turn_result(&mut after_first, tr);

        let mut best_second_decision = None;
        let value = if continues_same_turn(&after_first, pid) {
            let mut best_second = f64::NEG_INFINITY;
            // Expanding every second action through the cross-round
            // continuation would re-score the next round for each of them.
            // The second action of the round is selected on its standalone
            // score; only the top few candidates are worth paying for the
            // four-link continuation, which is what this bounded sweep does.
            let mut second_candidates =
                candidate_actions_k(&mut after_first, cfg.lookahead.second_action_k);
            second_candidates.sort_by(|a, b| b.score.total_cmp(&a.score));
            let mut evaluated = 0usize;
            for c2 in second_candidates {
                if evaluated >= cfg.lookahead.four_link_second_keep {
                    break;
                }
                let mut after_second = after_first.clone();
                if apply_move(&mut after_second, &c2.mv).is_err() {
                    continue;
                }
                evaluated += 1;

                let spend_still_low = spend_lead(&after_second, pid);
                let tr = advance_turn(&mut after_second);
                handle_turn_result(&mut after_second, tr);
                let mut score = ctx.profile.alpha * c2.score;
                if spend_still_low {
                    score += four_link_value(
                        &mut after_second,
                        pid,
                        cfg.lookahead.four_action_k,
                        cfg.lookahead.four_action_alpha,
                    );
                }
                if score > best_second {
                    best_second = score;
                    best_second_decision = Some(c2);
                }
            }
            c1.score + best_second.max(0.0)
        } else {
            c1.score
        };

        if best.as_ref().is_none_or(|(_, _, score)| value > *score) {
            best = Some((c1.mv, best_second_decision, value));
        }
    }

    best.map(|(mv, second, score)| RoundDecision {
        card_score: super::move_card_score(state, &mv),
        mv,
        score,
        second,
    })
    .unwrap_or_else(|| round_from_first(pass_decision(state)))
}

/// Evaluate the ordinary two-action turn.
fn choose_lookahead(state: &mut GameState, cfg: &HeuristicConfig) -> RoundDecision {
    let pid = state.current_player_id();
    let ctx = super::context::EvalContext::new(state, pid, &cfg);
    let first_candidates = candidate_actions_k(state, cfg.lookahead.first_action_k);

    let mut best: Option<(ResolvedMove, Option<Decision>, f64)> = None;
    for c1 in first_candidates {
        let mut s1 = state.clone();
        if apply_move(&mut s1, &c1.mv).is_err() {
            continue;
        }
        let tr = advance_turn(&mut s1);
        handle_turn_result(&mut s1, tr);

        let mut best_second_decision = None;
        let value = if continues_same_turn(&s1, pid) {
            // The second action is selected by the one-action candidate
            // score; applying it here only validates executability.  The
            // post-move state is no longer needed once the turn-end penalty
            // was removed.
            let mut second_candidates = candidate_actions_k(&mut s1, cfg.lookahead.second_action_k);
            second_candidates.sort_by(|a, b| b.score.total_cmp(&a.score));
            let mut best_second = 0.0;
            for c2 in second_candidates {
                if apply_move(&mut s1, &c2.mv).is_err() {
                    continue;
                }
                let scaled = ctx.profile.alpha * c2.score;
                best_second_decision = Some(c2);
                best_second = scaled;
                break;
            }
            // 2-ply blend: a productive second action adds to the turn's
            // value, discounted by the phase's alpha.
            c1.score + best_second.max(0.0)
        } else {
            c1.score
        };

        if best.as_ref().is_none_or(|(_, _, score)| value > *score) {
            best = Some((c1.mv, best_second_decision, value));
        }
    }

    best.map(|(mv, second, score)| {
        let card_score = super::move_card_score(state, &mv);
        RoundDecision {
            mv,
            score,
            card_score,
            second,
        }
    })
    .unwrap_or_else(|| round_from_first(pass_decision(state)))
}

fn round_from_first(first: Decision) -> RoundDecision {
    RoundDecision {
        mv: first.mv,
        score: first.score,
        card_score: first.card_score,
        second: None,
    }
}

/// Structural prerequisite for a four-link: the active player is last in the
/// current action order.  Whether the sequence is actually retained is
/// checked after simulating the first two actions with [`spend_lead`].
fn is_last_in_order(state: &GameState, pid: usize) -> bool {
    // Any 2+ player game can produce the sequence; the active player only
    // needs to be last in order at this stage.
    if state.player_count() < 2 || state.current_index + 1 != state.player_count() {
        return false;
    }
    state.current_player_id() == pid
}

/// True when the clock is still inside the same player action block after a
/// simulated first action: `pid` is active again and has already used one of
/// the round's actions without reaching the block's action limit.  A round or
/// era boundary resets `actions_this_turn`, so this rejects cross-round
/// continuations that only look like a same-turn follow-up.
fn continues_same_turn(state: &GameState, pid: usize) -> bool {
    state.current_player_id() == pid
        && state.actions_this_turn > 0
        && state.actions_this_turn < state.actions_per_turn
}

fn spend_lead(state: &GameState, pid: usize) -> bool {
    let mine = state.money_spent_this_round[pid];
    state.turn_order[..state.current_index]
        .iter()
        .all(|other| mine < state.money_spent_this_round[*other])
}

fn four_link_value(state: &mut GameState, pid: usize, k: usize, discount: f64) -> f64 {
    // A genuine four-link must return to us immediately after the round
    // boundary.  Passing opponents here would evaluate a hypothetical order
    // that the real game does not grant.
    if state.current_player_id() != pid {
        return 0.0;
    }
    let candidates = candidate_actions_k(state, k.max(1));
    let Some(c3) = candidates
        .into_iter()
        .max_by(|a, b| a.score.total_cmp(&b.score))
    else {
        return 0.0;
    };
    let mut value = discount * c3.score.max(0.0);
    if apply_move(state, &c3.mv).is_err() {
        return value;
    }
    let tr = advance_turn(state);
    handle_turn_result(state, tr);
    if state.current_player_id() != pid {
        return value;
    }
    let c4 = candidate_actions_k(state, k.max(1))
        .into_iter()
        .max_by(|a, b| a.score.total_cmp(&b.score));
    let Some(c4) = c4 else {
        return value;
    };
    if apply_move(state, &c4.mv).is_ok() {
        value += discount * 0.5 * c4.score.max(0.0);
    }
    value
}

/// Hard-coded opening templates. Bonuses only affect selection; the returned
/// decision keeps the evaluator's score so callers never mistake a template
/// preference for a board evaluation.
fn choose_opening(state: &mut GameState) -> Option<Decision> {
    let cfg = HeuristicConfig::default();
    let candidates = candidate_actions_k(state, cfg.lookahead.first_action_k);
    let mut best: Option<(Decision, f64)> = None;
    let coal_price = state.coal_price() as f64;
    let iron_price = state.iron_price() as f64;
    for c in candidates {
        let bonus = match &c.mv {
            ResolvedMove::Build { ind, .. } => match ind {
                crate::data::IndustryType::CoalMine => 1.5 + coal_price * 0.08,
                crate::data::IndustryType::IronWorks => 1.3 + iron_price * 0.08,
                crate::data::IndustryType::Brewery => 0.8,
                _ => 0.2,
            },
            ResolvedMove::Network { .. } | ResolvedMove::NetworkDouble { .. } => 0.9,
            ResolvedMove::Develop { .. } => 0.4,
            ResolvedMove::Loan { .. } => -0.8,
            ResolvedMove::Scout { .. } => -0.4,
            ResolvedMove::Sell { .. } => -0.6,
            ResolvedMove::Pass { .. } => -2.0,
        };
        let value = c.score + bonus;
        if best.as_ref().is_none_or(|(_, score)| value > *score) {
            best = Some((
                Decision {
                    mv: c.mv.clone(),
                    score: c.score,
                    card_score: c.card_score,
                },
                value,
            ));
        }
    }
    best.map(|(d, _)| d)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand_chacha::ChaCha12Rng;
    use rand_chacha::rand_core::SeedableRng;

    #[test]
    fn four_link_checks_position_then_spend_separately() {
        let mut state = GameState::new(ChaCha12Rng::seed_from_u64(41), 4);
        state.current_index = 3;
        let pid = state.current_player_id();
        state.money_spent_this_round = vec![10; 4];
        assert!(is_last_in_order(&state, pid));
        assert!(!spend_lead(&state, pid));
        state.money_spent_this_round[pid] = 0;
        assert!(spend_lead(&state, pid));

        let state3 = GameState::new(ChaCha12Rng::seed_from_u64(42), 3);
        assert!(!is_last_in_order(&state3, state3.current_player_id()));
    }

    #[test]
    fn opening_policy_returns_a_legal_candidate() {
        let mut state = GameState::new(ChaCha12Rng::seed_from_u64(43), 4);
        let candidates = candidate_actions_k(&mut state, 3);
        let decision = choose_opening(&mut state).expect("opening candidate");
        assert!(candidates.iter().any(
            |c| super::super::operation_key(&c.mv) == super::super::operation_key(&decision.mv)
        ));
    }

    #[test]
    fn continuation_only_recognises_an_unfinished_action_block() {
        let mut state = GameState::new(ChaCha12Rng::seed_from_u64(44), 4);
        state.actions_per_turn = 2;
        state.actions_this_turn = 1;
        let pid = state.current_player_id();
        assert!(continues_same_turn(&state, pid));

        // A round/era boundary resets the action counter; even if the same
        // player is active again this is the next round, not a follow-up.
        state.actions_this_turn = 0;
        assert!(!continues_same_turn(&state, pid));

        // The action block is over once the per-turn limit is reached.
        state.actions_this_turn = state.actions_per_turn;
        assert!(!continues_same_turn(&state, pid));
    }
}
