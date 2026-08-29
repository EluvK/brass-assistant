//! Loan action scoring.

use super::build::score_build_candidate;
use super::context::EvalContext;
use super::plan::Plan;
use super::value::ScoreParts;
use super::{CardChoices, Decision, candidate_actions_k};
use crate::engine::{TurnResult, advance_turn};
use crate::map::{LOAN_AMOUNT, LOAN_INCOME_PENALTY};
use crate::rules::{ResolvedMove, apply_move, get_valid_build_targets, get_valid_sell_targets};
use crate::state::GameState;

/// Score the loan option for the current player.
///
/// A loan is the engine of the early economy: it buys the income-recovery
/// build that keeps the turn engine running. Worth taking when cash is so
/// low we would otherwise idle-pass, provided income isn't in a death
/// spiral.
pub(super) fn score_loan_result(
    state: &GameState,
    ctx: &EvalContext,
    plan: &Plan,
    card_choices: &CardChoices,
) -> Option<Decision> {
    let pid = ctx.pid;
    let w = &ctx.cfg.loan;
    if !state.can_take_loan(pid) {
        return None;
    }
    let player = &state.players[pid];
    let post_loan_income = player.income_level() - LOAN_INCOME_PENALTY;

    // The post-loan budget opens up builds that buy back income fast.
    let cash = player.money as f64;
    let after = best_affordable_build_score(state, ctx, cash + LOAN_AMOUNT as f64, plan);
    let now = best_affordable_build_score(state, ctx, cash, plan);
    let gain = (after - now).max(0.0);

    let mut parts = ScoreParts {
        // The loan costs 3 income levels; the £30 itself is priced through
        // `gain` (what extra budget unlocks), not counted again here.
        income: -(LOAN_INCOME_PENALTY as f64),
        strategic: gain,
        ..Default::default()
    };

    // Same-turn combo value: Loan is strongest when it immediately unlocks
    // a productive second action (Loan -> Build / Network / Develop / Sell).
    if cash < w.combo_cash_threshold && ctx.rounds_remaining > w.combo_min_rounds_left
        && let Some(sim_gain) = best_same_turn_after_loan(state, ctx, card_choices) {
            parts.strategic += sim_gain.max(0.0) * w.combo_scale;
        }

    // Idle protection: if the current hand can't do anything positive, borrow.
    if cash < w.idle_cash_threshold {
        parts.strategic += w.idle_bonus;
    }

    // Unlock bonus: the loan is what makes an affordable build appear.
    if now <= 0.0 && after > w.unlock_min_after_score {
        parts.strategic += w.unlock_bonus;
    }

    // Startup-loan peak: Canal-Early rounds 1-2 with low cash — the loan
    // funds the economy engine (build/develop) before tempo stalls.
    if ctx.phase == super::plan::Phase::CanalEarly && state.round <= w.startup_max_round {
        parts.strategic += if cash < w.startup_low_cash_threshold {
            w.startup_low_cash_bonus
        } else {
            w.startup_bonus
        };
    }

    // Canal-Late end-of-era loan: borrow the rail-era startup capital when
    // cash is short. The -3 income is acceptable because the very next
    // action (selling built goods) recovers it — only if a sellable exists
    // that we can realistically flip soon.
    if ctx.phase == super::plan::Phase::CanalLate && state.round >= w.canal_late_min_round {
        let can_flip_soon = !get_valid_sell_targets(state, pid).is_empty()
            || state
                .city_tiles
                .iter()
                .flatten()
                .any(|t| t.player == pid && !t.flipped && t.ind.is_sellable());
        if can_flip_soon && cash < w.canal_late_cash_threshold {
            parts.strategic += if cash < w.idle_cash_threshold {
                w.canal_late_low_cash_bonus
            } else {
                w.canal_late_bonus
            };
        }
    }

    // Income floor: never borrow into a death spiral.
    parts.risk -= if post_loan_income <= w.floor_deep_debt_income {
        w.floor_deep_debt_penalty
    } else if post_loan_income <= w.floor_debt_income {
        w.floor_debt_penalty
    } else if post_loan_income <= w.floor_breakeven_income {
        w.floor_breakeven_penalty
    } else {
        0.0
    };

    // Do not over-borrow with plenty of cash.
    parts.risk -= if cash >= w.rich_heavy_cash {
        w.rich_heavy_penalty
    } else if cash >= w.rich_moderate_cash {
        w.rich_moderate_penalty
    } else if cash >= w.rich_light_cash {
        w.rich_light_penalty
    } else {
        0.0
    };

    let card_index = card_choices.first().map(|(index, _)| *index)?;
    Some(Decision {
        mv: ResolvedMove::Loan { card_index },
        score: parts.total(ctx),
        card_score: card_choices.first().map(|(_, s)| *s).unwrap_or(f64::INFINITY),
    })
}

fn best_same_turn_after_loan(
    state: &GameState,
    ctx: &EvalContext,
    card_choices: &CardChoices,
) -> Option<f64> {
    let card_index = card_choices.first().map(|(index, _)| *index)?;
    let mut sim = state.clone();
    apply_move(&mut sim, &ResolvedMove::Loan { card_index }).ok()?;
    let tr = advance_turn(&mut sim);
    match tr {
        TurnResult::Continue => {}
        TurnResult::EndCanalEra | TurnResult::EndGame => return Some(0.0),
    }
    if sim.current_player_id() != ctx.pid {
        return Some(0.0);
    }
    let cands = candidate_actions_k(&mut sim, 3);
    cands
        .into_iter()
        .filter(|d| !matches!(d.mv, ResolvedMove::Loan { .. }))
        .map(|d| d.score)
        .max_by(|a, b| a.total_cmp(b))
}

/// Best build score the player could afford within a cash budget.
fn best_affordable_build_score(state: &GameState, ctx: &EvalContext, budget: f64, plan: &Plan) -> f64 {
    let mut best = f64::NEG_INFINITY;
    for t in get_valid_build_targets(state, ctx.pid) {
        if (t.cost_total as f64) > budget {
            continue;
        }
        let s = score_build_candidate(state, ctx, &t, plan).total(ctx);
        if s > best {
            best = s;
        }
    }
    if best == f64::NEG_INFINITY {
        0.0
    } else {
        best
    }
}
