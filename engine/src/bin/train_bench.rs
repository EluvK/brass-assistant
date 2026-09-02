//! Training-data hot-path benchmark (developer tool).
//!
//! Usage:
//!   cargo run --release -p brass-engine --bin train_bench -- [positions] [seed]
//!
//! Plays one heuristic game, snapshots every position, then measures the
//! per-position cost of each stage used by the Python training pipeline:
//!   legal move enumeration, candidate feature encoding, state_to_tensor,
//!   snapshot serialize/restore, determinize, state clone, teacher scoring.

use std::time::Instant;

use rand_chacha::ChaCha12Rng;
use rand_chacha::rand_core::SeedableRng;

use _engine::bridge::action_features;
use _engine::encode;
use _engine::heuristic_ai;
use _engine::mcts_ai::MctsConfig;
use _engine::rules::{apply_move, legal_resolved_moves};
use _engine::state::GameState;

fn bench<F: FnMut()>(name: &str, iters: usize, mut f: F) -> f64 {
    f(); // warmup
    let start = Instant::now();
    for _ in 0..iters {
        f();
    }
    let per = start.elapsed().as_secs_f64() / iters as f64;
    println!("  {name:<36} {:>10.3} us/call", per * 1e6);
    per
}

fn main() {
    let positions = std::env::args()
        .nth(1)
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(40);
    let seed = std::env::args()
        .nth(2)
        .and_then(|s| s.parse::<u64>().ok())
        .unwrap_or(2026);

    // Collect midgame positions from one heuristic game.
    let mut state = GameState::new(ChaCha12Rng::seed_from_u64(seed), 4);
    let mut snapshots: Vec<Vec<u8>> = Vec::new();
    snapshots.push(state.snapshot_bytes().unwrap());
    while snapshots.len() < positions && !state.game_over {
        let mv = heuristic_ai::choose_action(&mut state).mv;
        apply_move(&mut state, &mv).unwrap();
        let tr = _engine::engine::advance_turn(&mut state);
        _engine::engine::handle_turn_result(&mut state, tr);
        snapshots.push(state.snapshot_bytes().unwrap());
    }
    let n = snapshots.len();
    let total_candidates: usize = snapshots
        .iter()
        .map(|s| {
            let mut st = GameState::from_snapshot_bytes(s).unwrap();
            legal_resolved_moves(&mut st).len()
        })
        .sum();
    println!(
        "train_bench | {n} positions | avg legal candidates {:.1}",
        total_candidates as f64 / n as f64
    );
    println!("  --- single-position component detail ---");
    {
        let mut work = GameState::from_snapshot_bytes(&snapshots[n / 2]).unwrap();
        let iters = 200;
        bench("legal_resolved_moves", iters, || {
            let _ = legal_resolved_moves(&mut work);
        });
        let legal = legal_resolved_moves(&mut work);
        bench("encode_move x all candidates", iters, || {
            for mv in &legal {
                let _ = action_features::encode_move(&work, mv);
            }
        });
        bench("state_to_tensor", iters, || {
            let _ = encode::state_to_tensor(&work, work.current_player_id());
        });
        bench("snapshot_bytes", iters, || {
            let _ = work.snapshot_bytes();
        });
        bench("from_snapshot_bytes", iters, || {
            let _ = GameState::from_snapshot_bytes(&snapshots[n / 2]);
        });
        bench("determinize", iters, || {
            let mut rng = ChaCha12Rng::seed_from_u64(1);
            let cfg = MctsConfig::default();
            let _ = _engine::mcts_ai::determinize_for_test(&work, &mut rng, &cfg);
        });
        bench("GameState::clone", iters, || {
            let _ = work.clone();
        });
        bench("candidate_actions_k(4)", iters, || {
            let _ = heuristic_ai::candidate_actions_k(&mut work, 4);
        });
        bench("choose_action (2-ply lookahead)", iters, || {
            let _ = heuristic_ai::choose_action(&mut work);
        });
    }

    println!("  --- averaged over all {n} positions ---");
    let mut legal_t = 0.0;
    let mut encode_t = 0.0;
    let mut tensor_t = 0.0;
    let start = Instant::now();
    for snap in &snapshots {
        let mut work = GameState::from_snapshot_bytes(snap).unwrap();
        let s0 = Instant::now();
        let moves = legal_resolved_moves(&mut work);
        legal_t += s0.elapsed().as_secs_f64();
        let s1 = Instant::now();
        for mv in &moves {
            let _ = action_features::encode_move(&work, mv);
        }
        encode_t += s1.elapsed().as_secs_f64();
        let s2 = Instant::now();
        let _ = encode::state_to_tensor(&work, work.current_player_id());
        tensor_t += s2.elapsed().as_secs_f64();
    }
    println!(
        "  legal_resolved_moves            {:>10.3} us/pos",
        legal_t / n as f64 * 1e6
    );
    println!(
        "  encode_move x all candidates    {:>10.3} us/pos",
        encode_t / n as f64 * 1e6
    );
    println!(
        "  state_to_tensor                 {:>10.3} us/pos",
        tensor_t / n as f64 * 1e6
    );
    println!("  total bench time {:.2?}", start.elapsed());
}
