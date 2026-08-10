//! Shared full-game driver.
//!
//! Every binary that plays a game (`main.rs`, replay, stat_game, sweep_scores,
//! sweep_canal, bench_mcts, sweep_mcts, debug_mcts) used to hand-write the same
//! loop — guard check, `apply_move`, `advance_turn`, era/game-end transitions,
//! `end_game` on exit. This module owns that loop once; binaries supply only
//! the move policy and the (optional) per-move / era hooks they need.
//!
//! Hooks run at fixed points:
//!
//! ```text
//! loop {
//!     [game_over / guard check]
//!     move = policy(state)                  <- caller's closure
//!     before_move(state, &move)             <- capture "before" state / stats
//!     result = apply_move(state, move)
//!     after_move(state, &move, &result)     <- print "after" / report
//!     if result.err -> return IllegalMove
//!     tr = advance_turn(state)
//!     on_era(state, era) -> AfterEra        <- BEFORE cleanup, may stop
//!     [handle_turn_result runs the transition unless stopped-before]
//!     after_era(state, era)                 <- AFTER cleanup (if it ran)
//! }
//! ```

use crate::data::Era;
use crate::engine::{advance_turn, handle_turn_result, TurnResult};
use crate::rules::{apply_move, Move};
use crate::state::GameState;

/// What ended the loop.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LoopOutcome {
    /// Reached `state.game_over` naturally.
    GameOver,
    /// Hit `max_moves` without finishing.
    GuardExceeded,
    /// A move failed to apply (no turn was advanced).
    IllegalMove,
    /// An `on_era` hook asked to stop (e.g. canal-only mode).
    StoppedByEraEnd,
}

/// What the driver should do after an era/game-end transition fires.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AfterEra {
    /// Run the transition, then keep playing.
    Continue,
    /// Do NOT run the transition (board keeps the pre-cleanup state) and stop.
    /// Used by canal-only replays that print the cleanup detail by hand.
    StopBeforeCleanup,
    /// Run the transition, then stop.
    StopAfterCleanup,
}

/// Optional per-move / per-era hooks. `on_era` returns `AfterEra`; the others
/// are fire-and-forget (or stats/printing).
pub struct GameHooks<'a> {
    /// Called before `apply_move` (state is still pre-move).
    pub before_move: Option<&'a mut dyn FnMut(&mut GameState, &Move)>,
    /// Called after `apply_move` with its result (before `advance_turn`).
    pub after_move: Option<&'a mut dyn FnMut(&mut GameState, &Move, &Result<String, String>)>,
    /// Called on `EndCanalEra` / `EndGame`, BEFORE the era cleanup runs.
    pub on_era: Option<&'a mut dyn FnMut(&mut GameState, Era) -> AfterEra>,
    /// Called AFTER the era cleanup ran (only when the transition was run).
    pub after_era: Option<&'a mut dyn FnMut(&mut GameState, Era)>,
}

impl Default for GameHooks<'_> {
    fn default() -> Self {
        GameHooks {
            before_move: None,
            after_move: None,
            on_era: None,
            after_era: None,
        }
    }
}

/// Play a game (or the next `max_moves` turns) with the given move policy.
/// `choose` returning `None` means the current player has no move to make (e.g.
/// an empty hand): the turn is advanced without applying any action.
pub fn play(
    state: &mut GameState,
    max_moves: usize,
    mut hooks: GameHooks<'_>,
    mut choose: impl FnMut(&mut GameState) -> Option<Move>,
) -> LoopOutcome {
    let mut moves = 0;
    loop {
        if state.game_over {
            return LoopOutcome::GameOver;
        }
        if moves >= max_moves {
            return LoopOutcome::GuardExceeded;
        }
        moves += 1;

        if let Some(mv) = choose(state) {
            if let Some(h) = hooks.before_move.as_mut() {
                h(state, &mv);
            }
            let result = apply_move(state, &mv);
            if let Some(h) = hooks.after_move.as_mut() {
                h(state, &mv, &result);
            }
            if result.is_err() {
                return LoopOutcome::IllegalMove;
            }
        }

        let tr = advance_turn(state);
        let era = match tr {
            TurnResult::Continue => None,
            TurnResult::EndCanalEra => Some(Era::Canal),
            TurnResult::EndGame => Some(Era::Rail),
        };
        let Some(era) = era else {
            continue;
        };
        let action = hooks
            .on_era
            .as_mut()
            .map(|h| h(state, era))
            .unwrap_or(AfterEra::Continue);
        let run_cleanup = matches!(
            action,
            AfterEra::Continue | AfterEra::StopAfterCleanup
        );
        if run_cleanup {
            handle_turn_result(state, tr);
            if let Some(h) = hooks.after_era.as_mut() {
                h(state, era);
            }
        }
        match action {
            AfterEra::Continue => {}
            AfterEra::StopBeforeCleanup | AfterEra::StopAfterCleanup => {
                return LoopOutcome::StoppedByEraEnd;
            }
        }
    }
}

/// Force the end-of-game scoring if the loop ended before the game was over
/// (e.g. guard exceeded). Mirrors the `if !state.game_over { end_game }`
/// epilogue the old hand-written loops repeated.
pub fn finish_game(state: &mut GameState) {
    if !state.game_over {
        handle_turn_result(state, TurnResult::EndGame);
    }
}
