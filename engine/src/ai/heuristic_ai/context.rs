//! Shared evaluation context: build one per candidate batch, pass it down.
//!
//! `EvalContext` collapses everything "what point of the game are we at"
//! into a single object — strategy phase and per-phase currency weights — and
//! exposes the conversions and convenience predicates shared by scorers.
//! Declarative round/era factors used by individual scorers live here too.

use super::plan::Phase;
use crate::state::GameState;

/// Rounds in one era (game rule; used only to normalise "how much era is
/// left" into 0..1).
pub(crate) const ERA_ROUNDS: f64 = 8.0;

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

/// Per-phase evaluation weights resolved from the config.
/// ready for deprecation. do not use in new code.
#[derive(Debug, Clone, Copy)]
pub struct EraProfile {
    pub phase: Phase,
    pub income_w: f64,
    pub money_w: f64,
    pub alpha: f64,
}

#[cfg(test)]
mod tests {
    use crate::state::GameState;
    use rand_chacha::rand_core::SeedableRng;

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
}
