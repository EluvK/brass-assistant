//! Replay one heuristic game while printing the scored candidate list before
//! every decision.
//!
//! Usage:
//!   cargo run --release --bin heuristic_trace -- [seed] [players] [candidate-k] [max-moves]

use brass_engine::game_loop::{self, GameHooks, LoopOutcome};
use brass_engine::heuristic_ai;
use brass_engine::replay_fmt;
use brass_engine::rules::Move;
use brass_engine::scoring;
use brass_engine::state::GameState;
use rand::SeedableRng;
use std::env;

fn main() {
    let args: Vec<String> = env::args().collect();
    let seed = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(7u64);
    let players = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(4usize);
    let candidate_k = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(30usize);
    let max_moves = args
        .get(4)
        .and_then(|s| s.parse().ok())
        .unwrap_or(200_000usize);

    let rng = rand::rngs::StdRng::seed_from_u64(seed);
    let mut state = GameState::new(rng, players);
    let mut move_no = 0usize;

    println!("Heuristic trace: seed={seed} players={players} candidate_k={candidate_k}");
    println!("起始顺位: {:?}", state.turn_order);

    let mut before_move = |state: &mut GameState, mv: &Move| {
        move_no += 1;
        let pid = state.current_player_id();
        println!(
            "\n[move {move_no}] 玩家{pid} 执行: {}",
            replay_fmt::move_detail(state, mv)
        );
    };
    let mut after_move = |state: &mut GameState, _mv: &Move, result: &Result<String, String>| {
        let status = result.as_deref().unwrap_or_else(|err| err.as_str());
        println!("  结果: {status}");
        println!(
            "  玩家: {}",
            replay_fmt::player_state(state, state.current_player_id())
        );
    };
    let hooks = GameHooks {
        before_move: Some(&mut before_move),
        after_move: Some(&mut after_move),
        ..Default::default()
    };

    let outcome = game_loop::play(&mut state, max_moves, hooks, |state| {
        let pid = state.current_player_id();
        println!(
            "\n--- {:?} era, round {}, player {pid} ---",
            state.era, state.round
        );
        println!("hand: {}", replay_fmt::hand_display(state, pid));

        let mut candidates = heuristic_ai::candidate_actions_k(state, candidate_k);
        candidates.sort_by(|a, b| b.score.total_cmp(&a.score));
        println!("candidates ({}):", candidates.len());
        for (rank, decision) in candidates.iter().enumerate() {
            println!(
                "  {:>2}. {} score={:.2}",
                rank + 1,
                decision.mv.describe(state),
                decision.score
            );
        }

        let decision = heuristic_ai::choose_action(state);
        println!(
            "selected: {} score={:.2}",
            decision.mv.describe(state),
            decision.score
        );
        Some(decision.mv)
    });

    if outcome == LoopOutcome::GameOver {
        println!("\n终局排名: {:?}", scoring::final_ranking(&state));
    } else {
        println!("\ntrace stopped: {outcome:?}");
        println!("当前排名（未结算）: {:?}", scoring::final_ranking(&state));
    }
    for pid in 0..players {
        println!("玩家{pid}: {}", replay_fmt::player_state(&state, pid));
    }
}
