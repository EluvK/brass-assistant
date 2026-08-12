//! Diagnostic: inspect the MCTS root tree after a decision.
//!
//! Usage: cargo run --release --bin debug_mcts -- <seed> <players> <sims> [ply]
//!   ply: how many heuristic moves to play before the inspected decision.

use brass_engine::game_loop;
use brass_engine::heuristic_ai;
use brass_engine::mcts_ai::{self, MctsConfig};
use brass_engine::state::GameState;
use rand::SeedableRng;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let seed: u64 = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(7);
    let players: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(4);
    let sims: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(2000);
    let ply: usize = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(60);

    let rng = rand::rngs::StdRng::seed_from_u64(seed);
    let mut state = GameState::new(rng, players);
    game_loop::play(&mut state, ply, game_loop::GameHooks::default(), |state| {
        Some(heuristic_ai::choose_action(state).mv)
    });

    println!(
        "state after {ply} actions, era={:?}, current=player{}, money={}, income={}, vp={}, hand={}",
        state.era,
        state.current_player_id(),
        state.players[state.current_player_id()].money,
        state.players[state.current_player_id()].income_level(),
        state.players[state.current_player_id()].vp,
        state.players[state.current_player_id()].hand.len()
    );

    let leaf_eval = if std::env::var("BRASS_MCTS_2PLY").is_ok() {
        brass_engine::mcts_ai::LeafEval::TwoPly
    } else {
        brass_engine::mcts_ai::LeafEval::OnePly
    };
    let cfg = MctsConfig {
        simulations: sims,
        max_depth: 6,
        c_puct: 1.0,
        prior_temp: 2.0,
        k_candidates: 3,
        leaf_eval,
    };
    let t = std::time::Instant::now();
    let d = mcts_ai::choose_action_mcts(&mut state, &cfg);
    let elapsed = t.elapsed();

    println!(
        "MCTS chose: {}  score={:.2}  ({:.2?})",
        d.mv.describe(&state),
        d.score,
        elapsed
    );
    println!(
        "1-ply would: {}",
        heuristic_ai::choose_action(&mut state).mv.describe(&state)
    );
    println!(
        "2-ply would: {}",
        brass_engine::search_ai::choose_action_2ply(&mut state)
            .mv
            .describe(&state)
    );
    // Candidates considered at the root by the 1-ply prior:
    println!("root candidate set (k=3):");
    for c in brass_engine::heuristic_ai::candidate_actions_k(&mut state, 3) {
        println!("  {}  score={:.2}", c.mv.describe(&state), c.score);
    }
}
