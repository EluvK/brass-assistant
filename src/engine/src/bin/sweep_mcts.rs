//! Parameter sweep for MCTS decision quality.
//!
//! Usage: cargo run --release --bin sweep_mcts -- <seed> <players> <sims>
//! Tests (depth, c_puct) combos and reports which action each picks vs the
//! 2-ply reference.

use brass_engine::game_loop;
use brass_engine::heuristic_ai;
use brass_engine::mcts_ai::{self, MctsConfig, LeafEval};
use brass_engine::search_ai;
use brass_engine::state::GameState;
use rand::SeedableRng;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let seed: u64 = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(7);
    let players: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(4);
    let sims: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(2000);

    let rng = rand::rngs::StdRng::seed_from_u64(seed);
    let mut state = GameState::new(rng, players);
    game_loop::play(&mut state, 60, game_loop::GameHooks::default(), |state| {
        Some(heuristic_ai::choose_action(state).mv)
    });
    let ref2 = search_ai::choose_action_2ply(&mut state).mv.describe(&state);

    for leaf in [LeafEval::OnePly, LeafEval::TwoPly] {
        for &(depth, c_puct) in &[(4, 10.0), (6, 10.0), (6, 30.0), (8, 30.0)] {
            let cfg = MctsConfig {
                simulations: sims,
                max_depth: depth,
                c_puct,
                prior_temp: 2.0,
                k_candidates: 3,
                leaf_eval: leaf,
            };
            let t = std::time::Instant::now();
            let d = mcts_ai::choose_action_mcts(&mut state, &cfg);
            let el = t.elapsed();
            let pick = d.mv.describe(&state);
            let mark = if pick == ref2 { "OK" } else { "xx" };
            println!(
                "[{leaf:?}] depth={depth} cp={c_puct:<4} {el:.2?} -> {pick}  {mark}  (ref: {ref2})"
            );
        }
    }
}
