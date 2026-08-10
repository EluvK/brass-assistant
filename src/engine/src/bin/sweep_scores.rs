//! Batch score sweep across a seed range, output as CSV with per-player final
//! VP plus economic columns (final income, final money, canal-era income).
//!
//! Usage:
//!   cargo run --release --bin sweep_scores -- <start_seed> <end_seed> <policy> [sims] > out.csv
//!
//!   policy: heuristic | 2ply | mcts
//!   mcts uses sims (default 200). heuristic/2ply ignore sims.

use brass_engine::data::Era;
use brass_engine::game_loop::{self, AfterEra, GameHooks, LoopOutcome};
use brass_engine::heuristic_ai;
use brass_engine::mcts_ai::{self, MctsConfig};
use brass_engine::search_ai;
use brass_engine::state::GameState;
use rand::SeedableRng;
use rayon::prelude::*;

#[derive(Default)]
struct SweepResult {
    seed: u64,
    vp: [i64; 4],
    final_income: [i8; 4],
    final_money: [i32; 4],
    canal_income: [i8; 4],
    illegal: bool,
    stuck: bool,
}

fn play_one(seed: u64, players: usize, policy: &str, sims: usize) -> SweepResult {
    let rng = rand::rngs::StdRng::seed_from_u64(seed);
    let mut state = GameState::new(rng, players);
    let mut res = SweepResult {
        seed,
        ..Default::default()
    };

    let mut canal_income: Option<[i8; 4]> = None;
    let mut on_era = |state: &mut GameState, era: Era| -> AfterEra {
        if era == Era::Canal {
            let mut inc = [0i8; 4];
            for (i, p) in state.players.iter().enumerate() {
                if i < inc.len() {
                    inc[i] = p.income_level();
                }
            }
            canal_income = Some(inc);
        }
        AfterEra::Continue
    };
    let hooks = GameHooks {
        on_era: Some(&mut on_era),
        ..Default::default()
    };
    let outcome = game_loop::play(&mut state, 200_000, hooks, |state| {
        Some(match policy {
            "2ply" => search_ai::choose_action_2ply(state).mv,
            "mcts" => {
                let cfg = MctsConfig {
                    simulations: sims,
                    ..Default::default()
                };
                mcts_ai::choose_action_mcts(state, &cfg).mv
            }
            _ => heuristic_ai::choose_action(state).mv,
        })
    });
    game_loop::finish_game(&mut state);
    res.illegal = outcome == LoopOutcome::IllegalMove;
    res.stuck = !matches!(outcome, LoopOutcome::GameOver);

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
    if let Some(c) = canal_income {
        res.canal_income = c;
    }
    res
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let start: u64 = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(0);
    let end: u64 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(500);
    let policy = args.get(3).cloned().unwrap_or_else(|| "2ply".to_string());
    let sims: usize = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(200);
    let players: usize = 4;

    let results: Vec<SweepResult> = (start..end)
        .into_par_iter()
        .map(|seed| play_one(seed, players, &policy, sims))
        .collect();

    println!(
        "seed,p0,p1,p2,p3,avg,income0,income1,income2,income3,money0,money1,money2,money3,canal_income0,canal_income1,canal_income2,canal_income3"
    );
    let mut illegal = 0;
    let mut stuck = 0;
    let mut winner_sum = 0i64;
    let mut player_sum = 0i64;
    for r in &results {
        if r.illegal {
            illegal += 1;
        }
        if r.stuck {
            stuck += 1;
        }
        let avg = r.vp.iter().sum::<i64>() as f64 / players as f64;
        winner_sum += r.vp.iter().max().copied().unwrap_or(0);
        player_sum += r.vp.iter().sum::<i64>();
        println!(
            "{},{},{},{},{},{:.2},{},{},{},{},{},{},{},{},{},{},{},{}",
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
            r.canal_income[3]
        );
    }
    eprintln!(
        "[sweep_scores] games={} policy={} illegal={} stuck={} winner_mean={:.1} player_mean={:.1}",
        results.len(),
        policy,
        illegal,
        stuck,
        winner_sum as f64 / results.len().max(1) as f64,
        player_sum as f64 / (results.len().max(1) as f64 * players as f64)
    );
}
