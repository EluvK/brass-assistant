//! Benchmark a single MCTS decision (sims at various budgets).
//!
//! Usage: cargo run --release --bin bench_mcts -- <seed> <players> [sims...]
//! e.g. bench_mcts 7 4 5000 10000

use brass_engine::game_loop;
use brass_engine::heuristic_ai;
use brass_engine::mcts_ai::{self, MctsConfig};
use brass_engine::search_ai;
use brass_engine::state::GameState;
use rand::SeedableRng;
use std::time::Instant;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let seed: u64 = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(7);
    let players: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(4);
    let sims_list: Vec<usize> = args[3..]
        .iter()
        .map(|s| s.parse().unwrap_or(5000))
        .collect();
    let sims_list = if sims_list.is_empty() {
        vec![5000, 10000]
    } else {
        sims_list
    };

    // Build a mid-game state: play N heuristic-AI moves so MCTS sees a
    // non-trivial position (a full board isn't needed, but a few plies help).
    let rng = rand::rngs::StdRng::seed_from_u64(seed);
    let mut state = GameState::new(rng, players);
    let mut actions = 0;
    game_loop::play(&mut state, 60, game_loop::GameHooks::default(), |state| {
        actions += 1;
        Some(heuristic_ai::choose_action(state).mv)
    });

    println!(
        "MCTS decision bench | seed={seed} players={players} | state after {actions} actions, era={:?}, current=player{}",
        state.era,
        state.current_player_id()
    );

    // Reference: what does 1-ply heuristic pick here?
    let h = heuristic_ai::choose_action(&mut state);
    println!(
        "  1-ply heuristic picks: {}  score={:.2}",
        h.mv.describe(&state),
        h.score
    );
    let two = search_ai::choose_action_2ply(&mut state);
    println!(
        "  2-ply picks:            {}  score={:.2}",
        two.mv.describe(&state),
        two.score
    );
    for d in heuristic_ai::candidate_actions(&mut state) {
        println!(
            "     candidate: {} score={:.2}",
            d.mv.describe(&state),
            d.score
        );
    }

    for &sims in &sims_list {
        let cfg = MctsConfig {
            simulations: sims,
            ..Default::default()
        };
        let t = Instant::now();
        let d = mcts_ai::choose_action_mcts(&mut state, &cfg);
        let elapsed = t.elapsed();
        println!(
            "  sims={sims:<6} {:.2?} ({:.0} sims/s) -> {}  score={:.2}",
            elapsed,
            sims as f64 / elapsed.as_secs_f64(),
            d.mv.describe(&state),
            d.score
        );
    }
}

// Keep the earlier reference helpers referenced to avoid unused warnings.
#[allow(dead_code)]
fn _keep(_s: &str) {}
