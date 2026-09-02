//! Loan action scoring.

use super::context::{EvalContext, era_round_factor, round_factor};
use super::{CardChoices, Decision};
use crate::map::LOAN_AMOUNT;
use crate::rules::ResolvedMove;
use crate::state::GameState;

/// Intermediate values of the loan model, kept explicit for tuning diagnostics.
#[derive(Debug, Clone, Copy, PartialEq)]
struct LoanEvaluation {
    a: f64,
    b: f64,
    economic_value: f64,
    normalized_value: f64,
    score: f64,
}

/// Evaluate the economic value of receiving one £30 loan at the current point
/// in an era. The engine's income-floor rule remains a legality check outside
/// this pure model.
fn evaluate_loan(ctx: &EvalContext, cash: f64, income_level: f64) -> LoanEvaluation {
    let w = &ctx.cfg.loan;
    let progress = round_factor!(ctx.round_progress, 0.0_f64, 1.0_f64);
    let a = era_round_factor!(
        ctx.is_canal(),
        progress,
        w.canal_a_start,
        w.canal_a_end,
        w.rail_a_start,
        w.rail_a_end,
    );
    let b = era_round_factor!(
        ctx.is_canal(),
        progress,
        w.canal_b_start,
        w.canal_b_end,
        w.rail_b_start,
        w.rail_b_end,
    );
    debug_assert!(b > 0.0, "loan income divisor b must be positive");

    let economic_value =
        ((a - cash) / LOAN_AMOUNT as f64 - income_level / b) * ctx.rounds_left_in_era;
    let normalized_value = economic_value.clamp(0.0, 1.0);
    let score = w.action_cost + w.economic_score_scale * normalized_value;
    LoanEvaluation {
        a,
        b,
        economic_value,
        normalized_value,
        score,
    }
}

/// Score the loan option for the current player.
///
/// A loan occupies one action, so its base cost is offset only by the
/// normalised economic benefit of £30: lower cash/income and more numbered
/// rounds remaining make the option stronger.
pub(super) fn score_loan_result(
    state: &GameState,
    ctx: &EvalContext,
    _plan: &super::plan::Plan,
    card_choices: &CardChoices,
) -> Option<Decision> {
    let pid = ctx.pid;
    if !state.can_take_loan(pid) {
        return None;
    }

    let player = &state.players[pid];
    let evaluation = evaluate_loan(ctx, player.money as f64, player.income_level() as f64);
    let card_index = card_choices.first().map(|(index, _)| *index)?;
    Some(Decision {
        mv: ResolvedMove::Loan { card_index },
        score: evaluation.score,
        card_score: card_choices
            .first()
            .map(|(_, score)| *score)
            .unwrap_or(f64::INFINITY),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ai::heuristic_ai::HeuristicConfig;
    use crate::data::Era;
    use crate::income::income_highest_space_of_level;
    use rand_chacha::ChaCha12Rng;
    use rand_chacha::rand_core::SeedableRng;

    fn evaluation(era: Era, round: u32, cash: f64, income: f64) -> LoanEvaluation {
        let mut state = GameState::new(ChaCha12Rng::seed_from_u64(7), 4);
        state.era = era;
        state.round = round;
        let cfg = HeuristicConfig::default();
        let ctx = EvalContext::new(&state, state.current_player_id(), &cfg);
        evaluate_loan(&ctx, cash, income)
    }

    #[test]
    fn loan_parameter_debug_matrix() {
        // round, cash, income
        let cases = [
            ("Canal opening cash 17", Era::Canal, 1, 17.0, 0.0),
            ("Canal opening cash 15", Era::Canal, 1, 15.0, 0.0),
            ("Canal opening cash 5", Era::Canal, 3, 5.0, 2.0),
            ("Canal middle", Era::Canal, 5, 15.0, 4.0),
            ("Canal final round", Era::Canal, 8, 10.0, 6.0),
            ("Rail opening", Era::Rail, 1, 35.0, 0.0),
            ("Rail middle", Era::Rail, 5, 30.0, 4.0),
            ("Rail final round", Era::Rail, 8, 25.0, 6.0),
            ("Rail low cash", Era::Rail, 8, 5.0, 0.0),
            ("High income", Era::Canal, 5, 15.0, 12.0),
            ("Negative income", Era::Canal, 5, 15.0, -4.0),
            ("Clamp low", Era::Rail, 8, 100.0, 20.0),
            ("Clamp high", Era::Canal, 1, -100.0, -10.0),
        ];

        println!("label | era/round | cash | income | left | a | b | economic | norm | score");
        println!("|-|-|-|-|-|-|-|-|-|-|");
        for (label, era, round, cash, income) in cases {
            let result = evaluation(era, round, cash, income);
            println!(
                "{label} | {era:?}/{round} | {cash:.1} | {income:.1} | {:.0} | {:.2} | {:.2} | {:.4} | {:.4} | {:.4}",
                9 - round,
                result.a,
                result.b,
                result.economic_value,
                result.normalized_value,
                result.score,
            );
            assert!((-4.0..=6.0).contains(&result.score));
        }

        let canal_low_cash = evaluation(Era::Canal, 1, 5.0, 0.0);
        let canal_rich = evaluation(Era::Canal, 1, 17.0, 0.0);
        let rail_rich = evaluation(Era::Rail, 1, 35.0, 0.0);
        assert!(canal_low_cash.score >= canal_rich.score);
        assert!(canal_low_cash.score > rail_rich.score);
        assert!(
            evaluation(Era::Canal, 5, 15.0, 0.0).score
                >= evaluation(Era::Canal, 5, 15.0, 4.0).score
        );
        assert!(
            evaluation(Era::Canal, 1, 15.0, 0.0).score
                >= evaluation(Era::Canal, 8, 15.0, 0.0).score
        );
        assert_eq!(evaluation(Era::Rail, 8, 100.0, 20.0).score, -4.0);
        assert_eq!(evaluation(Era::Canal, 1, -100.0, -10.0).score, 6.0);

        let mut state = GameState::new(ChaCha12Rng::seed_from_u64(8), 4);
        let pid = state.current_player_id();
        state.players[pid].income_space = income_highest_space_of_level(-8);
        let cfg = HeuristicConfig::default();
        let ctx = EvalContext::new(&state, pid, &cfg);
        let no_cards = CardChoices::new();
        assert!(!state.can_take_loan(pid));
        assert!(
            score_loan_result(
                &state,
                &ctx,
                &super::super::plan::compute_plan(&state, pid),
                &no_cards,
            )
            .is_none()
        );
    }
}
