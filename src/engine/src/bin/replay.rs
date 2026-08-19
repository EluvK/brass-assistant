//! Replay a single AI game with a detailed, human-readable log.
//!
//! All the human-readable Chinese formatting lives in the library
//! (`brass_engine::replay_fmt`); this binary only drives the game loop and
//! prints the formatted lines — the SAME formatting the Python network-replay
//! driver (`src/ai/experiments/replay_net.py`) uses, so both produce
//! byte-identical log structure.
//!
//! Usage: cargo run --release --bin replay -- <seed> <players> [policy] [sims] [canal-only] [full|summary] [trace] [candidate-k] [max-moves]
//!   canal-only: "1"/"true" stops after the canal era (no rail era played).
//!   trace: "trace" prints scored heuristic candidates before each heuristic decision.

use brass_engine::data::Era;
use brass_engine::game_loop::{self, AfterEra, GameHooks, LoopOutcome};
use brass_engine::heuristic_ai;
use brass_engine::random_ai;
use brass_engine::replay_fmt;
use brass_engine::rules::Move;
use brass_engine::scoring;
use brass_engine::state::GameState;
use rand::SeedableRng;
use std::cell::{Cell, RefCell};
use std::env;

fn print_era_snapshot(state: &GameState, era_name: &str, action_stats: &[replay_fmt::PStats]) {
    println!("\n--- {era_name}时代玩家汇总(动作/在场板块/翻面板块) ---");
    for (pid, stats) in action_stats.iter().enumerate() {
        println!("玩家{pid} 动作: {}", replay_fmt::fmt_stats_line(stats));
        println!(
            "玩家{pid} 动作序列: {}",
            replay_fmt::fmt_action_sequence(stats, era_name == "运河")
        );
        println!(
            "玩家{pid} 在场板块: {}",
            replay_fmt::player_tiles_detail(state, pid, false)
        );
        println!(
            "玩家{pid} 翻面板块: {}",
            replay_fmt::player_tiles_detail(state, pid, true)
        );
    }
}

fn choose_heuristic_action(
    state: &mut GameState,
    trace_enabled: bool,
    candidate_k: usize,
    move_no: usize,
) -> Move {
    if trace_enabled {
        let pid = state.current_player_id();
        println!(
            "\n--- [玩家{pid}]|{:?}时代|第{}轮|决策追踪:第{}步 |  ---",
            state.era,
            state.round,
            move_no + 1,
        );
        // println!("手牌: {}", replay_fmt::hand_display(state, pid));

        let mut candidates = heuristic_ai::candidate_actions_k(state, candidate_k);
        candidates.sort_by(|a, b| b.score.total_cmp(&a.score));
        println!("候选动作({}):", candidates.len());
        for (rank, decision) in candidates.iter().enumerate() {
            println!(
                "  {:>2}. {} score={:.2}",
                rank + 1,
                decision.mv.describe(state),
                decision.score
            );
        }
    }

    let decision = heuristic_ai::choose_action(state);
    if trace_enabled {
        println!(
            "选择: {} score={:.2}",
            decision.mv.describe(state),
            decision.score
        );
        let pid = state.current_player_id();
        let selected: Vec<usize> = match &decision.mv {
            brass_engine::r#move::Move::Build { card_index, .. }
            | brass_engine::r#move::Move::Network { card_index, .. }
            | brass_engine::r#move::Move::NetworkDouble { card_index, .. }
            | brass_engine::r#move::Move::Develop { card_index, .. }
            | brass_engine::r#move::Move::Sell { card_index, .. }
            | brass_engine::r#move::Move::Loan { card_index }
            | brass_engine::r#move::Move::Pass { card_index } => vec![*card_index],
            brass_engine::r#move::Move::Scout { card_indices } => card_indices.to_vec(),
            brass_engine::r#move::Move::ResolveFreeDevelop { .. } => Vec::new(),
        };
        let ranked = heuristic_ai::ranked_card_choices(state, pid);
        println!("卡片保留价值(越低越适合消耗):");
        for (index, score) in ranked {
            let marker = if selected.contains(&index) { "*" } else { " " };
            let label = state.players[pid]
                .hand
                .get(index)
                .map(brass_engine::replay_fmt::card_label)
                .unwrap_or_else(|| "?".to_string());
            println!("  {marker} 手牌[{index}] {label} keep_score={score:.3}");
        }
    }
    decision.mv
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let seed: u64 = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(7);
    let players: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(4);
    // policy: "heuristic" (default) | "mcts" | "random"
    //       | "mcts-vs-heur" | "mcts-vs-random"
    let policy = args
        .get(3)
        .cloned()
        .unwrap_or_else(|| "heuristic".to_string());
    let sims: usize = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(300);
    let canal_only = args
        .get(5)
        .map(|s| matches!(s.as_str(), "1" | "true" | "canal"))
        .unwrap_or(false);
    // `summary` retains the era-end diagnostics without printing every move.
    // It supersedes the former `stat_game` binary.
    let verbose = args.get(6).map(|s| s.as_str()).unwrap_or("full") == "full";
    let trace_enabled = args.get(7).map(|s| s.as_str()).unwrap_or("trace") == "trace";
    let candidate_k: usize = args.get(8).and_then(|s| s.parse().ok()).unwrap_or(30);
    let max_moves: usize = args.get(9).and_then(|s| s.parse().ok()).unwrap_or(200_000);

    let rng = rand::rngs::StdRng::seed_from_u64(seed);
    let mut state = GameState::new(rng, players);
    // RNG for the random baseline (state.rng is setup-only).
    let mut rand_rng = rand::rngs::StdRng::seed_from_u64(seed ^ 0x5245_504C_4159);

    let mut prev_round = state.round;
    let mut prev_era = state.era;
    // RefCell so both the per-move recording hook and the era-end snapshot
    // hook can share the same accumulated stats.
    let canal_stats: RefCell<Vec<replay_fmt::PStats>> =
        RefCell::new(vec![replay_fmt::PStats::default(); players]);
    let rail_stats: RefCell<Vec<replay_fmt::PStats>> =
        RefCell::new(vec![replay_fmt::PStats::default(); players]);

    println!(
        "Replay: {players}玩家 {policy}AI, seed={seed} | 模式={}{}",
        if verbose { "full" } else { "summary" },
        if trace_enabled {
            format!(" | trace(候选数={candidate_k})")
        } else {
            String::new()
        }
    );
    if verbose {
        println!("起始顺位: {:?}", state.turn_order);
        for pid in 0..players {
            println!(
                "开局 玩家{pid}: {} | 手牌: {}",
                replay_fmt::player_state(&state, pid),
                replay_fmt::hand_display(&state, pid)
            );
        }
        println!("商家: {}", replay_fmt::merchant_state(&state));
        println!("{}", replay_fmt::board_state(&state));
    }

    let move_no = Cell::new(0usize);
    let mut on_before = |state: &mut GameState, mv: &Move| {
        move_no.set(move_no.get() + 1);
        let pid = state.current_player_id();
        let stat_target = if state.era == Era::Canal {
            &mut canal_stats.borrow_mut()[pid]
        } else {
            &mut rail_stats.borrow_mut()[pid]
        };
        stat_target.record(mv, state.era);
        if verbose {
            let detail = replay_fmt::move_detail(state, mv);
            let before = replay_fmt::player_state(state, pid);
            let hand_before = replay_fmt::hand_display(state, pid);
            println!("玩家{pid} [{detail}]");
            println!("    之前: {before}");
            println!("    手牌前: {hand_before}");
        }
    };
    let mut on_after = |state: &mut GameState, _mv: &Move, res: &Result<String, String>| {
        let pid = state.current_player_id();
        if verbose {
            let after = replay_fmt::player_state(state, pid);
            let hand_after = replay_fmt::hand_display(state, pid);
            let status = match res {
                Ok(msg) => msg.clone(),
                Err(e) => format!("!!失败: {e}"),
            };
            println!("    结果: {status}");
            println!("    之后: {after}");
            println!("    手牌后: {hand_after}");
            println!("    盘面: {}", replay_fmt::board_state(state));
            println!("    商家: {}", replay_fmt::merchant_state(state));
        }
    };
    let mut on_era = |state: &mut GameState, era: Era| -> AfterEra {
        if era == Era::Canal {
            println!("\n--- 运河时代结算明细 ---");
            for pid in 0..players {
                println!("玩家{pid}: {}", replay_fmt::era_score_detail(state, pid));
            }
            print_era_snapshot(state, "运河", &canal_stats.borrow());
            if canal_only {
                // Canal-only mode: print the canal cleanup detail and final
                // standings, then stop WITHOUT running the cleanup.
                println!("{}", replay_fmt::canal_cleanup_detail(state));
                let ranking = scoring::final_ranking(state);
                println!("运河末排名: {:?}", ranking);
                if let Some(&w) = ranking.first() {
                    println!("运河末第一: 玩家{w}");
                }
                return AfterEra::StopBeforeCleanup;
            }
            if verbose {
                println!("{}", replay_fmt::canal_cleanup_detail(state));
            }
        }
        AfterEra::Continue
    };
    let mut on_era_after = |_state: &mut GameState, era: Era| {
        if era == Era::Canal && verbose {
            println!("\n--- 运河时代结束，进入铁路时代(连接/1级板块已清除,重新洗牌发牌) ---");
        }
    };
    let hooks = GameHooks {
        before_move: Some(&mut on_before),
        after_move: Some(&mut on_after),
        on_era: Some(&mut on_era),
        after_era: Some(&mut on_era_after),
    };
    let outcome = game_loop::play(&mut state, max_moves, hooks, |state| {
        // Era / round header change
        if verbose && state.era != prev_era {
            println!(
                "\n------ 时代切换: {} -> {} ------",
                replay_fmt::era_label(prev_era),
                replay_fmt::era_label(state.era)
            );
            prev_era = state.era;
        }
        if verbose && state.round != prev_round {
            println!(
                "\n===== {}时代 第{}轮 | 顺位: {:?} =====",
                replay_fmt::era_label(state.era),
                state.round,
                state.turn_order
            );
            prev_round = state.round;
        }

        let pid = state.current_player_id();
        Some(match policy.as_str() {
            "random" => random_ai::choose_random_move(state, &mut rand_rng)
                .unwrap_or_else(|| heuristic_ai::pass_decision(state).mv),
            "mcts-vs-random" => {
                let mcts_seat = (seed as usize) % players;
                if pid == mcts_seat {
                    use brass_engine::mcts_ai::{self, MctsConfig};
                    let cfg = MctsConfig {
                        simulations: sims,
                        ..Default::default()
                    };
                    mcts_ai::choose_action_mcts(state, &cfg).mv
                } else {
                    random_ai::choose_random_move(state, &mut rand_rng)
                        .unwrap_or_else(|| heuristic_ai::pass_decision(state).mv)
                }
            }
            "mcts-vs-heur" => {
                let mcts_seat = (seed as usize) % players;
                if pid == mcts_seat {
                    use brass_engine::mcts_ai::{self, MctsConfig};
                    let cfg = MctsConfig {
                        simulations: sims,
                        ..Default::default()
                    };
                    mcts_ai::choose_action_mcts(state, &cfg).mv
                } else {
                    choose_heuristic_action(state, trace_enabled, candidate_k, move_no.get())
                }
            }
            "heuristic" => {
                choose_heuristic_action(state, trace_enabled, candidate_k, move_no.get())
            }
            "mcts" => {
                use brass_engine::mcts_ai::{self, MctsConfig};
                let cfg = MctsConfig {
                    simulations: sims,
                    ..Default::default()
                };
                mcts_ai::choose_action_mcts(state, &cfg).mv
            }
            _ => choose_heuristic_action(state, trace_enabled, candidate_k, move_no.get()),
        })
    });
    if outcome == LoopOutcome::StoppedByEraEnd {
        return;
    }
    if outcome == LoopOutcome::GuardExceeded {
        println!("[!] guard exceeded, aborting");
    }
    if outcome == LoopOutcome::IllegalMove {
        println!("[!] 非法动作，终止回放以避免污染轮次/手牌状态");
    }
    game_loop::finish_game(&mut state);

    println!("\n======================================================================");
    println!("终局结算");
    println!("======================================================================");
    print_era_snapshot(&state, "铁路", &rail_stats.borrow());
    for pid in 0..players {
        println!("玩家{pid}: {}", replay_fmt::player_state(&state, pid));
    }
    let ranking = scoring::final_ranking(&state);
    println!("排名: {:?}", ranking);
    if let Some(&w) = ranking.first() {
        println!("胜者: 玩家{w}");
    }
}
