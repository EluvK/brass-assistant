//! Replay a single AI game with a detailed, human-readable log.
//!
//! All the human-readable Chinese formatting lives in the library
//! (`brass_engine::replay_fmt`); this binary only drives the game loop and
//! prints the formatted lines — the SAME formatting the Python network-replay
//! driver (`src/ai/experiments/replay_net.py`) uses, so both produce
//! byte-identical log structure.
//!
//! Usage: cargo run --release --bin replay -- <seed> <players> [policy] [sims] [canal-only] [full|summary]
//!   canal-only: "1"/"true" stops after the canal era (no rail era played).

use brass_engine::data::Era;
use brass_engine::game_loop::{self, AfterEra, GameHooks, LoopOutcome};
use brass_engine::heuristic_ai;
use brass_engine::random_ai;
use brass_engine::replay_fmt;
use brass_engine::rules::Move;
use brass_engine::scoring;
use brass_engine::search_ai;
use brass_engine::state::GameState;
use rand::SeedableRng;
use std::cell::RefCell;
use std::env;

fn print_era_snapshot(state: &GameState, era_name: &str, action_stats: &[replay_fmt::PStats]) {
    println!("\n--- {era_name}时代玩家汇总（动作/在场板块/翻面板块） ---");
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

fn main() {
    let args: Vec<String> = env::args().collect();
    let seed: u64 = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(7);
    let players: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(4);
    // policy: "heuristic" (default) | "2ply" | "mcts" | "random"
    //       | "mcts-vs-heur" | "mcts-vs-random" | "mcts-vs-2ply"
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
    let verbose = !matches!(args.get(6).map(String::as_str), Some("summary"));

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
        "Replay: {players}玩家 {policy}AI, seed={seed} | 模式={}",
        if verbose { "full" } else { "summary" }
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

    let mut on_before = |state: &mut GameState, mv: &Move| {
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
            println!("\n--- 运河时代结束，进入铁路时代（连接/1级板块已清除，重新洗牌发牌） ---");
        }
    };
    let hooks = GameHooks {
        before_move: Some(&mut on_before),
        after_move: Some(&mut on_after),
        on_era: Some(&mut on_era),
        after_era: Some(&mut on_era_after),
    };
    let outcome = game_loop::play(&mut state, 200_000, hooks, |state| {
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
                    heuristic_ai::choose_action(state).mv
                }
            }
            "mcts-vs-2ply" => {
                let mcts_seat = (seed as usize) % players;
                if pid == mcts_seat {
                    use brass_engine::mcts_ai::{self, MctsConfig};
                    let cfg = MctsConfig {
                        simulations: sims,
                        ..Default::default()
                    };
                    mcts_ai::choose_action_mcts(state, &cfg).mv
                } else {
                    search_ai::choose_action_2ply(state).mv
                }
            }
            "2ply" => search_ai::choose_action_2ply(state).mv,
            "mcts" => {
                use brass_engine::mcts_ai::{self, MctsConfig};
                let cfg = MctsConfig {
                    simulations: sims,
                    ..Default::default()
                };
                mcts_ai::choose_action_mcts(state, &cfg).mv
            }
            _ => heuristic_ai::choose_action(state).mv,
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
