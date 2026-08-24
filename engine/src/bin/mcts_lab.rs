//! MCTS experiment workbench.
//!
//! Usage:
//!   cargo run --release --bin mcts_lab -- bench [seed] [players] [sims...]
//!   cargo run --release --bin mcts_lab -- inspect [seed] [players] [sims] [ply]
//!   cargo run --release --bin mcts_lab -- sweep [seed] [players] [sims]

use _engine::game_loop;
use _engine::heuristic_ai;
use _engine::mcts_ai::{self, LeafEval, MctsConfig};
use _engine::state::GameState;
use rand_chacha::rand_core::SeedableRng;

fn midgame(seed: u64, players: usize, ply: usize) -> GameState {
    let rng = rand_chacha::ChaCha12Rng::seed_from_u64(seed);
    let mut state = GameState::new(rng, players);
    game_loop::play(&mut state, ply, game_loop::GameHooks::default(), |state| {
        Some(heuristic_ai::choose_action(state).mv)
    });
    state
}

fn bench(args: &[String]) {
    let seed = args.first().and_then(|s| s.parse().ok()).unwrap_or(7);
    let players = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(4);
    let sims_list: Vec<usize> = args[2..].iter().filter_map(|s| s.parse().ok()).collect();
    let sims_list = if sims_list.is_empty() {
        vec![5_000, 10_000]
    } else {
        sims_list
    };
    let mut state = midgame(seed, players, 60);
    println!(
        "MCTS bench | seed={seed} players={players} era={:?} current=P{}",
        state.era,
        state.current_player_id()
    );
    let heuristic = heuristic_ai::choose_action(&mut state);
    println!(
        "  heuristic: {} score={:.2}",
        heuristic.mv.describe(&state),
        heuristic.score
    );
    for sims in sims_list {
        let cfg = MctsConfig {
            simulations: sims,
            ..Default::default()
        };
        let start = std::time::Instant::now();
        let decision = mcts_ai::choose_action_mcts(&mut state, &cfg);
        let elapsed = start.elapsed();
        println!(
            "  sims={sims:<6} {elapsed:.2?} ({:.0} sims/s) -> {} score={:.2}",
            sims as f64 / elapsed.as_secs_f64(),
            decision.mv.describe(&state),
            decision.score
        );
    }
}

fn inspect(args: &[String]) {
    let seed = args.first().and_then(|s| s.parse().ok()).unwrap_or(7);
    let players = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(4);
    let sims = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(2_000);
    let ply = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(60);
    let mut state = midgame(seed, players, ply);
    let leaf_eval = if std::env::var("BRASS_MCTS_TWO_PLY").is_ok() {
        LeafEval::RootTwoPly
    } else {
        LeafEval::OnePly
    };
    let cfg = MctsConfig {
        simulations: sims,
        leaf_eval,
        ..Default::default()
    };
    println!(
        "MCTS inspect | seed={seed} players={players} after={ply} era={:?} current=P{}",
        state.era,
        state.current_player_id()
    );
    let start = std::time::Instant::now();
    let decision = mcts_ai::choose_action_mcts(&mut state, &cfg);
    println!(
        "  MCTS: {} score={:.2} ({:.2?})",
        decision.mv.describe(&state),
        decision.score,
        start.elapsed()
    );
    println!(
        "  heuristic: {}",
        heuristic_ai::choose_action(&mut state).mv.describe(&state)
    );
    println!("  root prior candidates (k={}):", cfg.k_candidates);
    for candidate in heuristic_ai::candidate_actions_k(&mut state, cfg.k_candidates) {
        println!(
            "    {} score={:.2}",
            candidate.mv.describe(&state),
            candidate.score
        );
    }
}

fn sweep(args: &[String]) {
    let seed = args.first().and_then(|s| s.parse().ok()).unwrap_or(7);
    let players = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(4);
    let sims = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(2_000);
    let mut state = midgame(seed, players, 60);
    let reference = heuristic_ai::choose_action(&mut state).mv.describe(&state);
    println!("MCTS parameter sweep | seed={seed} players={players} reference={reference}");
    for leaf_eval in [LeafEval::OnePly, LeafEval::RootTwoPly] {
        for (max_depth, c_puct) in [(4, 10.0), (6, 10.0), (6, 30.0), (8, 30.0)] {
            let cfg = MctsConfig {
                simulations: sims,
                max_depth,
                c_puct,
                leaf_eval,
                ..Default::default()
            };
            let start = std::time::Instant::now();
            let decision = mcts_ai::choose_action_mcts(&mut state, &cfg);
            let picked = decision.mv.describe(&state);
            let marker = if picked == reference { "OK" } else { "--" };
            println!(
                "  {leaf_eval:?} depth={max_depth} cp={c_puct:<3} {elapsed:.2?} -> {picked} {marker}",
                elapsed = start.elapsed()
            );
        }
    }
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("bench") => bench(&args[1..]),
        Some("inspect") => inspect(&args[1..]),
        Some("sweep") => sweep(&args[1..]),
        _ => eprintln!("用法: mcts_lab <bench|inspect|sweep> [参数]"),
    }
}
