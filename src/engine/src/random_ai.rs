//! Random policy baselines: picks uniformly (or with mild preference) among
//! legal moves. Used to sanity-check the engine and as a weak baseline.

use crate::rules::{legal_moves, Move};
use crate::state::GameState;
use rand::Rng;

pub fn choose_random_move(state: &mut GameState<impl Rng>) -> Option<Move> {
    let moves = legal_moves(state);
    if moves.is_empty() {
        return None;
    }
    let idx = state.rng.gen_range(0..moves.len());
    Some(moves[idx].clone())
}

/// Slightly-smart random: prefers non-Pass actions so games actually produce
/// buildings/VPs. Useful as a sanity baseline (still essentially random).
pub fn choose_random_action_first(state: &mut GameState<impl Rng>) -> Option<Move> {
    let moves = legal_moves(state);
    if moves.is_empty() {
        return None;
    }
    // Split into non-pass and pass.
    let (non_pass, pass): (Vec<_>, Vec<_>) = moves
        .iter()
        .partition(|m| !matches!(m, Move::Pass { .. }));
    let pool = if !non_pass.is_empty() {
        non_pass
    } else {
        pass
    };
    let idx = state.rng.gen_range(0..pool.len());
    Some(pool[idx].clone())
}

