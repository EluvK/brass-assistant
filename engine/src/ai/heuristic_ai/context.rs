//! Shared evaluation context: build one per candidate batch, pass it down.
//!
//! `EvalContext` collapses everything "what point of the game are we at"
//! into a single object — strategy phase, per-phase weights, remaining
//! rounds — and exposes the currency conversions and round-dependent
//! convenience predicates every scorer needs. Round/era logic that used to
//! be re-implemented per scorer now lives here (or in
//! [`super::config::EraWeights`]) exactly once.

use super::config::HeuristicConfig;
use super::plan::{Phase, era_phase};
use super::value::ScoreParts;
use crate::state::GameState;

/// Rounds in one era (game rule; used only to normalise "how much era is
/// left" into 0..1).
pub(crate) const ERA_ROUNDS: f64 = 8.0;

/// Interpolate a value from `start` to `end` over a normalised round progress.
/// The progress is clamped so callers can safely use it with diagnostic states.
macro_rules! round_factor {
    ($progress:expr, $start:expr, $end:expr) => {{
        let progress = ($progress).clamp(0.0, 1.0);
        ($start) + (($end) - ($start)) * progress
    }};
}

/// Choose a Canal or Rail round interpolation without duplicating phase checks
/// in every action scorer.
macro_rules! era_round_factor {
    ($is_canal:expr, $progress:expr,
     $canal_start:expr, $canal_end:expr,
     $rail_start:expr, $rail_end:expr $(,)?) => {{
        let progress = ($progress).clamp(0.0, 1.0);
        if $is_canal {
            ($canal_start) + (($canal_end) - ($canal_start)) * progress
        } else {
            ($rail_start) + (($rail_end) - ($rail_start)) * progress
        }
    }};
}

pub(crate) use era_round_factor;
pub(crate) use round_factor;

// ---------------------------------------------------------------------------
// Declarative state factors (new API; existing macros above remain intact).
// ---------------------------------------------------------------------------

/// A round-dependent linear factor. The factor is clamped to the numbered
/// round range used by the current era (round 1..=8).
#[allow(dead_code)]
#[derive(Debug, Clone, Copy)]
pub(crate) struct RoundFactor {
    start: f64,
    end: f64,
}

#[allow(dead_code)]
impl RoundFactor {
    pub(crate) const fn new(start: f64, end: f64) -> Self {
        Self { start, end }
    }

    /// Evaluate this factor from the state's current numbered round.
    pub(crate) fn factor(&self, state: &GameState) -> f64 {
        let progress =
            ((state.round.clamp(1, ERA_ROUNDS as usize) as f64) - 1.0) / (ERA_ROUNDS - 1.0);
        self.start + (self.end - self.start) * progress
    }
}

/// An era-dependent linear factor. The selected pair is determined by the
/// state's current era; interpolation itself follows the numbered round.
#[allow(dead_code)]
#[derive(Debug, Clone, Copy)]
pub(crate) struct EraRoundFactor {
    canal_start: f64,
    canal_end: f64,
    rail_start: f64,
    rail_end: f64,
}

#[allow(dead_code)]
impl EraRoundFactor {
    pub(crate) const fn new(
        canal_start: f64,
        canal_end: f64,
        rail_start: f64,
        rail_end: f64,
    ) -> Self {
        Self {
            canal_start,
            canal_end,
            rail_start,
            rail_end,
        }
    }

    /// Evaluate this factor using Canal/Rail parameters selected from state.
    pub(crate) fn factor(&self, state: &GameState) -> f64 {
        let progress =
            ((state.round.clamp(1, ERA_ROUNDS as usize) as f64) - 1.0) / (ERA_ROUNDS - 1.0);
        let (start, end) = if matches!(state.era, crate::data::Era::Canal) {
            (self.canal_start, self.canal_end)
        } else {
            (self.rail_start, self.rail_end)
        };
        start + (end - start) * progress
    }
}

/// Declare a factor type whose associated function evaluates from state:
/// `MyFactor::factor(state)`.
#[allow(unused_macros)]
macro_rules! define_round_factor {
    ($type:ident, $start:expr, $end:expr $(,)?) => {
        struct $type;
        impl $type {
            fn factor(state: &$crate::state::GameState) -> f64 {
                $crate::ai::heuristic_ai::context::RoundFactor::new($start, $end).factor(state)
            }
        }
    };
}

/// Declare a Canal/Rail factor with the same associated-function call shape:
/// `MyFactor::factor(state)`.
#[allow(unused_macros)]
macro_rules! define_era_round_factor {
    (
        $type:ident,
        canal: ($canal_start:expr, $canal_end:expr),
        rail: ($rail_start:expr, $rail_end:expr) $(,)?
    ) => {
        struct $type;
        impl $type {
            fn factor(state: &$crate::state::GameState) -> f64 {
                $crate::ai::heuristic_ai::context::EraRoundFactor::new(
                    $canal_start,
                    $canal_end,
                    $rail_start,
                    $rail_end,
                )
                .factor(state)
            }
        }
    };
}

#[allow(unused_imports)]
pub(crate) use define_era_round_factor;
#[allow(unused_imports)]
pub(crate) use define_round_factor;

/// One-shot evaluation context for scoring the current player's options.
/// ready for deprecation. do not use in new code.
pub struct EvalContext<'a> {
    pub cfg: &'a HeuristicConfig,
    pub pid: usize,
    pub phase: Phase,
    /// Per-phase weights derived from [`HeuristicConfig::era`].
    pub profile: EraProfile,
    /// Estimated rounds (incl. the current one) until the era ends.
    pub rounds_remaining: f64,
    /// Fraction of the era still ahead, clamped to 0..1.
    pub era_frac: f64,
    /// Era-local round progress: round 1 is 0.0 and round 8 is 1.0.
    /// Unlike `era_frac`, this follows the rules' numbered rounds rather than
    /// the remaining card budget.
    pub round_progress: f64,
    /// Numbered rounds including the current round until this era ends (8..1).
    /// This is intentionally distinct from the fractional `rounds_remaining`.
    pub rounds_left_in_era: f64,
}

/// Per-phase evaluation weights resolved from the config.
/// ready for deprecation. do not use in new code.
#[derive(Debug, Clone, Copy)]
pub struct EraProfile {
    pub phase: Phase,
    pub income_w: f64,
    pub money_w: f64,
    pub network_w: f64,
    pub alpha: f64,
}

impl<'a> EvalContext<'a> {
    pub fn new(state: &GameState, pid: usize, cfg: &'a HeuristicConfig) -> Self {
        let phase = era_phase(state);
        let rounds_remaining = state.rounds_remaining();
        let era_frac = (rounds_remaining / ERA_ROUNDS).clamp(0.0, 1.0);
        let era_round = state.round.clamp(1, ERA_ROUNDS as usize) as f64;
        let round_progress = (era_round - 1.0) / (ERA_ROUNDS - 1.0);
        let rounds_left_in_era = ERA_ROUNDS - era_round + 1.0;
        let params = cfg.era.params(phase);
        let profile = EraProfile {
            phase,
            income_w: cfg.value.income_base * (params.income_add + params.income_frac * era_frac),
            money_w: cfg.value.money_base * params.money_mult,
            network_w: params.network_w,
            alpha: params.alpha,
        };
        Self {
            cfg,
            pid,
            phase,
            profile,
            rounds_remaining,
            era_frac,
            round_progress,
            rounds_left_in_era,
        }
    }

    // -- currency conversions -------------------------------------------

    /// Convert cash (£) into VP equivalents.
    pub fn money_value(&self, pounds: f64) -> f64 {
        pounds * self.profile.money_w
    }

    /// Convert income-level changes into VP equivalents.
    pub fn income_value(&self, levels: f64) -> f64 {
        levels * self.profile.income_w
    }

    /// Convert summed hand keep-score (flexibility) into VP equivalents.
    pub fn flex_value(&self, keep_scores: f64) -> f64 {
        keep_scores * self.cfg.value.flex
    }

    /// Composite conversion used by every scorer through
    /// [`ScoreParts::total`].
    pub(crate) fn total_of(&self, parts: &ScoreParts) -> f64 {
        parts.vp
            + self.money_value(parts.money)
            + self.income_value(parts.income)
            + self.flex_value(parts.flex)
            + parts.strategic
            + parts.risk
    }

    // -- round / era predicates ------------------------------------------

    /// Discount for value realised later than the current action (future
    /// link VP, early sells, hand access used on later turns). Continuous
    /// in the remaining era fraction.
    pub fn future_discount(&self) -> f64 {
        self.cfg.discount.floor + self.cfg.discount.span * self.era_frac
    }

    /// Era-endgame window: the last rounds of the era where unflipped
    /// canal-only tiles vanish and selling becomes urgent.
    pub fn is_era_endgame(&self) -> bool {
        self.rounds_remaining <= self.cfg.era.params(self.phase).endgame_rounds
    }

    pub fn is_canal(&self) -> bool {
        matches!(self.phase, Phase::CanalEarly | Phase::CanalLate)
    }

    pub fn is_rail(&self) -> bool {
        matches!(self.phase, Phase::RailEarly | Phase::RailLate)
    }
}

#[cfg(test)]
mod tests {
    use super::super::config::HeuristicConfig;
    use super::*;
    use crate::state::GameState;
    use rand_chacha::rand_core::SeedableRng;

    fn ctx_for<'a>(state: &'a GameState, cfg: &'a HeuristicConfig) -> EvalContext<'a> {
        EvalContext::new(state, state.current_player_id(), cfg)
    }

    #[test]
    fn declarative_factors_query_game_state() {
        define_round_factor!(TestRoundFactor, 10.0, 20.0);
        define_era_round_factor!(
            TestEraFactor,
            canal: (0.0, 8.0),
            rail: (100.0, 108.0),
        );
        let mut state = GameState::new(rand_chacha::ChaCha12Rng::seed_from_u64(7), 2);
        state.round = 5;
        assert!((TestRoundFactor::factor(&state) - 15.714285714).abs() < 1e-8);
        assert!((TestEraFactor::factor(&state) - 4.571428571).abs() < 1e-8);
        state.era = crate::data::Era::Rail;
        assert!((TestEraFactor::factor(&state) - 104.571428571).abs() < 1e-8);
    }

    #[test]
    fn currency_conversions_use_the_phase_profile() {
        let state = GameState::new(rand_chacha::ChaCha12Rng::seed_from_u64(7), 2);
        let cfg = HeuristicConfig::default();
        let ctx = ctx_for(&state, &cfg);
        assert!((ctx.money_value(10.0) - 10.0 * ctx.profile.money_w).abs() < 1e-9);
        assert!((ctx.income_value(2.0) - 2.0 * ctx.profile.income_w).abs() < 1e-9);
        assert!((ctx.flex_value(3.0) - 3.0 * cfg.value.flex).abs() < 1e-9);
    }

    #[test]
    fn future_discount_is_continuous_and_bounded() {
        let state = GameState::new(rand_chacha::ChaCha12Rng::seed_from_u64(7), 2);
        let cfg = HeuristicConfig::default();
        let ctx = ctx_for(&state, &cfg);
        let d = ctx.future_discount();
        assert!(d >= cfg.discount.floor && d <= cfg.discount.floor + cfg.discount.span);
        // Early era: with a full era ahead, the future is worth near the top.
        assert!(ctx.era_frac > 0.9);
        assert!(d > cfg.discount.floor + cfg.discount.span * 0.9);
    }

    #[test]
    fn numbered_round_tools_have_explicit_clamped_boundaries() {
        let cfg = HeuristicConfig::default();
        let mut state = GameState::new(rand_chacha::ChaCha12Rng::seed_from_u64(7), 4);

        state.round = 1;
        let first = ctx_for(&state, &cfg);
        assert_eq!(first.round_progress, 0.0);
        assert_eq!(first.rounds_left_in_era, 8.0);

        state.round = 8;
        let last = ctx_for(&state, &cfg);
        assert_eq!(last.round_progress, 1.0);
        assert_eq!(last.rounds_left_in_era, 1.0);

        state.round = 99;
        let clamped = ctx_for(&state, &cfg);
        assert_eq!(clamped.round_progress, 1.0);
        assert_eq!(clamped.rounds_left_in_era, 1.0);
    }

    #[test]
    fn round_factor_macros_interpolate_era_endpoints() {
        assert_eq!(round_factor!(0.5_f64, 20.0, 40.0), 30.0);
        assert_eq!(
            era_round_factor!(true, 1.0_f64, 20.0, 40.0, 40.0, 30.0),
            40.0
        );
        assert_eq!(
            era_round_factor!(false, 1.0_f64, 20.0, 40.0, 40.0, 30.0),
            30.0
        );
    }
}
