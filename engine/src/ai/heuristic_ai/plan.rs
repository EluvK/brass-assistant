//! Era-phase strategy profile and quantified production-plan selection.

use crate::data::Era;
use crate::state::GameState;

/// Four strategy phases: each era is split into an early and late half.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Phase {
    CanalEarly,
    CanalLate,
    RailEarly,
    RailLate,
}

pub fn era_phase(state: &GameState) -> Phase {
    match state.era {
        Era::Canal if state.round <= 4 => Phase::CanalEarly,
        Era::Canal => Phase::CanalLate,
        Era::Rail if state.round <= 4 => Phase::RailEarly,
        Era::Rail => Phase::RailLate,
    }
}
