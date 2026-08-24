//! Batch score sweep across a seed range, output as CSV with per-player final
//! VP plus economic columns (final income, final money, canal-era income).
//!
//! Usage:
//!   cargo run --release --bin sweep_scores -- <start_seed> <end_seed> <policy> [sims] [full|canal] > out.csv
//!
//!   policy: heuristic | mcts
//!   mcts uses sims (default 200); heuristic ignores sims.

use _engine::data::Era;
use _engine::game_loop::{self, AfterEra, GameHooks, LoopOutcome};
use _engine::mcts_ai::{self, MctsConfig};
use _engine::state::GameState;
use rand_chacha::rand_core::SeedableRng;
use rayon::prelude::*;
use std::time::Instant;

#[derive(Default)]
struct SweepResult {
    seed: u64,
    vp: [i64; 4],
    final_income: [i8; 4],
    final_money: [i32; 4],
    canal_income: [i8; 4],
    flipped: u64,
    links: u64,
    actions: [u64; 6], // build, network, develop, sell, loan, pass
    elapsed_us: u64,
    illegal: bool,
    stuck: bool,
}

fn play_one(seed: u64, players: usize, policy: &str, sims: usize, canal_only: bool) -> SweepResult {
    let started = Instant::now();
    let rng = rand_chacha::ChaCha12Rng::seed_from_u64(seed);
    let mut state = GameState::new(rng, players);
    let res = std::cell::RefCell::new(SweepResult {
        seed,
        ..Default::default()
    });

    let mut on_move = |_state: &mut GameState, mv: &_engine::rules::Move| {
        let mut r = res.borrow_mut();
        match mv {
            _engine::rules::Move::Build { .. } => r.actions[0] += 1,
            _engine::rules::Move::Network { .. } | _engine::rules::Move::NetworkDouble { .. } => {
                r.actions[1] += 1
            }
            _engine::rules::Move::Develop { .. } => r.actions[2] += 1,
            _engine::rules::Move::Sell { .. } => r.actions[3] += 1,
            _engine::rules::Move::Loan { .. } => r.actions[4] += 1,
            _engine::rules::Move::Pass { .. } => r.actions[5] += 1,
            _engine::rules::Move::Scout { .. } => {}
        }
    };
    let mut on_era = |state: &mut GameState, era: Era| -> AfterEra {
        if era == Era::Canal {
            let mut r = res.borrow_mut();
            let mut inc = [0i8; 4];
            for (i, p) in state.players.iter().enumerate() {
                if i < inc.len() {
                    inc[i] = p.income_level();
                }
            }
            r.canal_income = inc;
            r.links = state.links.iter().flatten().count() as u64;
            r.flipped = state
                .city_tiles
                .iter()
                .flatten()
                .filter(|tile| tile.flipped)
                .count() as u64;
            if canal_only {
                return AfterEra::StopAfterCleanup;
            }
        }
        AfterEra::Continue
    };
    let mut after_era = |state: &mut GameState, era: Era| {
        if canal_only && era == Era::Canal {
            let mut r = res.borrow_mut();
            for (i, p) in state.players.iter().enumerate() {
                if i < r.vp.len() {
                    r.vp[i] = p.vp as i64;
                    r.final_income[i] = p.income_level();
                    r.final_money[i] = p.money;
                }
            }
        }
    };
    let hooks = GameHooks {
        before_move: Some(&mut on_move),
        on_era: Some(&mut on_era),
        after_era: Some(&mut after_era),
        ..Default::default()
    };
    let outcome = game_loop::play(&mut state, 200_000, hooks, |state| {
        Some(match policy {
            "heuristic" => _engine::heuristic_ai::choose_action(state).mv,
            "mcts" => {
                let cfg = MctsConfig {
                    simulations: sims,
                    ..Default::default()
                };
                mcts_ai::choose_action_mcts(state, &cfg).mv
            }
            _ => _engine::heuristic_ai::choose_action(state).mv,
        })
    });
    if !canal_only {
        game_loop::finish_game(&mut state);
    }
    let mut res = res.into_inner();
    res.illegal = outcome == LoopOutcome::IllegalMove;
    res.stuck = !matches!(
        outcome,
        LoopOutcome::GameOver | LoopOutcome::StoppedByEraEnd
    );

    if !canal_only {
        for (i, p) in state.players.iter().enumerate() {
            if i < res.vp.len() {
                res.vp[i] = p.vp as i64;
            }
            if i < res.final_income.len() {
                res.final_income[i] = p.income_level();
            }
            if i < res.final_money.len() {
                res.final_money[i] = p.money;
            }
        }
    }
    res.elapsed_us = started.elapsed().as_micros().min(u64::MAX as u128) as u64;
    res
}

fn mean_variance(values: impl Iterator<Item = f64>) -> (f64, f64) {
    let mut count = 0.0;
    let mut mean = 0.0;
    let mut m2 = 0.0;
    for value in values {
        count += 1.0;
        let delta = value - mean;
        mean += delta / count;
        m2 += delta * (value - mean);
    }
    if count == 0.0 {
        (0.0, 0.0)
    } else {
        (mean, m2 / count)
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let start: u64 = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(0);
    let end: u64 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(500);
    let policy = args
        .get(3)
        .cloned()
        .unwrap_or_else(|| "heuristic".to_string());
    let sims: usize = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(200);
    let canal_only = matches!(args.get(5).map(String::as_str), Some("canal"));
    let players: usize = 4;

    let results: Vec<SweepResult> = (start..end)
        .into_par_iter()
        .map(|seed| play_one(seed, players, &policy, sims, canal_only))
        .collect();

    if canal_only {
        println!(
            "seed,p0,p1,p2,p3,winner,build,network,develop,sell,loan,pass,flipped,links,elapsed_us"
        );
    } else {
        println!(
            "seed,p0,p1,p2,p3,avg,income0,income1,income2,income3,money0,money1,money2,money3,canal_income0,canal_income1,canal_income2,canal_income3,elapsed_us"
        );
    }
    let mut illegal = 0;
    let mut stuck = 0;
    let (game_mean, game_variance) = mean_variance(
        results
            .iter()
            .map(|r| r.vp.iter().sum::<i64>() as f64 / players as f64),
    );
    let (player_mean, player_variance) = mean_variance(
        results
            .iter()
            .flat_map(|r| r.vp.iter().map(|&vp| vp as f64)),
    );
    let (winner_mean, winner_variance) = mean_variance(
        results
            .iter()
            .map(|r| r.vp.iter().max().copied().unwrap_or(0) as f64),
    );
    let (time_mean_us, time_variance_us) =
        mean_variance(results.iter().map(|r| r.elapsed_us as f64));
    let unique_winners = results
        .iter()
        .filter(|r| {
            let max = r.vp.iter().max().copied().unwrap_or(0);
            r.vp.iter().filter(|&&vp| vp == max).count() == 1
        })
        .count();
    for r in &results {
        if r.illegal {
            illegal += 1;
        }
        if r.stuck {
            stuck += 1;
        }
        let avg = r.vp.iter().sum::<i64>() as f64 / players as f64;
        if canal_only {
            println!(
                "{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}",
                r.seed,
                r.vp[0],
                r.vp[1],
                r.vp[2],
                r.vp[3],
                r.vp.iter().max().copied().unwrap_or(0),
                r.actions[0],
                r.actions[1],
                r.actions[2],
                r.actions[3],
                r.actions[4],
                r.actions[5],
                r.flipped,
                r.links,
                r.elapsed_us
            );
        } else {
            println!(
                "{},{},{},{},{},{:.2},{},{},{},{},{},{},{},{},{},{},{},{},{}",
                r.seed,
                r.vp[0],
                r.vp[1],
                r.vp[2],
                r.vp[3],
                avg,
                r.final_income[0],
                r.final_income[1],
                r.final_income[2],
                r.final_income[3],
                r.final_money[0],
                r.final_money[1],
                r.final_money[2],
                r.final_money[3],
                r.canal_income[0],
                r.canal_income[1],
                r.canal_income[2],
                r.canal_income[3],
                r.elapsed_us
            );
        }
    }
    eprintln!(
        "[sweep_scores]:\n\nscope={} games={} policy={} illegal={} stuck={}\nunique_winner_rate={:.3} winner_mean={:.3} winner_variance={:.3}\nplayer_mean={:.3} player_variance={:.3}\ngame_mean={:.3} game_variance={:.3}\ntime_mean_us={:.1} time_variance_us={:.1}",
        if canal_only { "canal" } else { "full" },
        results.len(),
        policy,
        illegal,
        stuck,
        unique_winners as f64 / results.len().max(1) as f64,
        winner_mean,
        winner_variance,
        player_mean,
        player_variance,
        game_mean,
        game_variance,
        time_mean_us,
        time_variance_us
    );
}
