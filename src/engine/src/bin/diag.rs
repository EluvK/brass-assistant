//! Diagnostic: how many degenerate full-tie games occur, and confirm Build
//! remains legal mid-game (guards against a real action-gen bug).

use brass_engine::engine::{advance_turn, end_canal_era, end_game, TurnResult};
use brass_engine::random_ai::choose_random_move;
use brass_engine::rules;
use brass_engine::state::GameState;
use rand::SeedableRng;

fn main() {
    let games = 1000;
    let mut full_tie = 0u64;
    let mut all_zero_vp = 0u64;
    let mut max_build_targets = 0usize;
    let mut games_with_build_targets = 0u64;

    for g in 0..games {
        let rng = rand::rngs::StdRng::seed_from_u64(g as u64);
        let mut state = GameState::new(rng, 4);
        let mut guard = 0;
        loop {
            guard += 1;
            if guard > 200_000 || state.game_over {
                break;
            }
            // Sample build legality periodically
            if guard % 10 == 0 {
                let pid = state.current_player_id();
                let n = rules::get_valid_build_targets(&state, pid).len();
                if n > max_build_targets {
                    max_build_targets = n;
                }
                if n > 0 {
                    games_with_build_targets += 1;
                }
            }
            let m = choose_random_move(&mut state);
            if let Some(mv) = m {
                let _ = rules::apply_move(&mut state, &mv);
            }
            match advance_turn(&mut state) {
                TurnResult::Continue => {}
                TurnResult::EndCanalEra => end_canal_era(&mut state),
                TurnResult::EndGame => end_game(&mut state),
            }
        }
        if !state.game_over {
            end_game(&mut state);
        }

        // Full tie: all four identical (vp, income, money)
        let (v0, i0, m0) = {
            let p = &state.players[0];
            (p.vp, p.income_level(), p.money)
        };
        let is_full_tie = state.players[1..]
            .iter()
            .all(|p| p.vp == v0 && p.income_level() == i0 && p.money == m0);
        if is_full_tie {
            full_tie += 1;
        }
        if state.players.iter().all(|p| p.vp == 0) {
            all_zero_vp += 1;
        }
    }

    println!("Games: {games}");
    println!("Full ties (vp+income+money equal): {full_tie} ({:.1}%)", full_tie as f64 * 100.0 / games as f64);
    println!("All players at 0 VP: {all_zero_vp} ({:.1}%)", all_zero_vp as f64 * 100.0 / games as f64);
    println!("Max build targets seen in a sample: {max_build_targets}");
    println!("Samples where build was legal: {games_with_build_targets}");
}
