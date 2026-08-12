//! Batch canal-era-only score sweep. Plays games up to the canal-era end and
//! records the canal-era result (VP scored in the era, income, money, flipped
//! tiles, links, action counts). Use to tune the canal-era strategy in isolation.
//!
//! Usage:
//!   cargo run --release --bin sweep_canal -- <start_seed> <end_seed> <policy> [sims]
//!     policy: heuristic | 2ply | mcts
//! Output: CSV on stdout, summary on stderr.

use brass_engine::data::Era;
use brass_engine::game_loop::{self, AfterEra, GameHooks, LoopOutcome};
use brass_engine::heuristic_ai;
use brass_engine::mcts_ai::{self, MctsConfig};
use brass_engine::rules::Move;
use brass_engine::search_ai;
use brass_engine::state::GameState;
use rand::SeedableRng;
use rayon::prelude::*;

#[derive(Default)]
struct CanalResult {
    seed: u64,
    vp: [i64; 4],
    income: [i8; 4],
    money: [i32; 4],
    flipped: u64,
    links: u64,
    actions: [u64; 6], // build, network, develop, sell, loan, pass
    illegal: bool,
    stuck: bool,
}

fn play_canal(seed: u64, players: usize, policy: &str, sims: usize) -> CanalResult {
    let rng = rand::rngs::StdRng::seed_from_u64(seed);
    let mut state = GameState::new(rng, players);
    // RefCell so the per-move / era hooks can all mutate disjoint parts of the
    // result without aliasing a single `&mut CanalResult`.
    let res = std::cell::RefCell::new(CanalResult {
        seed,
        ..Default::default()
    });

    let mut on_move = |_state: &mut GameState, mv: &Move| {
        let mut r = res.borrow_mut();
        match mv {
            Move::Build { .. } => r.actions[0] += 1,
            Move::Network { .. } | Move::NetworkDouble { .. } => r.actions[1] += 1,
            Move::Develop { .. } | Move::ResolveFreeDevelop { .. } => r.actions[2] += 1,
            Move::Sell { .. } => r.actions[3] += 1,
            Move::Loan { .. } => r.actions[4] += 1,
            Move::Pass { .. } => r.actions[5] += 1,
            Move::Scout { .. } => {}
        }
    };
    let mut on_era = |state: &mut GameState, _era: Era| -> AfterEra {
        // Snapshot links + flipped tiles BEFORE cleanup (end_canal_era removes
        // canal links and level-1 tiles).
        let mut r = res.borrow_mut();
        r.links = state.links.iter().flatten().count() as u64;
        r.flipped = state
            .city_tiles
            .iter()
            .flatten()
            .filter(|t| t.flipped)
            .count() as u64;
        AfterEra::StopAfterCleanup
    };
    let mut on_era_after = |state: &mut GameState, _era: Era| {
        let mut r = res.borrow_mut();
        for (i, p) in state.players.iter().enumerate() {
            if i < r.vp.len() {
                r.vp[i] = p.vp as i64;
            }
            if i < r.income.len() {
                r.income[i] = p.income_level();
            }
            if i < r.money.len() {
                r.money[i] = p.money;
            }
        }
    };
    let hooks = GameHooks {
        before_move: Some(&mut on_move),
        on_era: Some(&mut on_era),
        after_era: Some(&mut on_era_after),
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

    let mut res = res.into_inner();
    res.illegal = outcome == LoopOutcome::IllegalMove;
    res.stuck = outcome == LoopOutcome::GuardExceeded;
    res
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let start: u64 = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(0);
    let end: u64 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(500);
    let policy = args.get(3).cloned().unwrap_or_else(|| "2ply".to_string());
    let sims: usize = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(200);
    let players: usize = 4;

    let results: Vec<CanalResult> = (start..end)
        .into_par_iter()
        .map(|seed| play_canal(seed, players, &policy, sims))
        .collect();

    println!("seed,p0,p1,p2,p3,avg_winner,build,network,develop,sell,loan,pass,flipped,links");
    let mut winner_sum = 0i64;
    let mut player_sum = 0i64;
    let mut illegal = 0;
    let mut stuck = 0;
    let mut agg = [0u64; 6];
    let mut flipped_sum = 0u64;
    let mut links_sum = 0u64;
    for r in &results {
        if r.illegal {
            illegal += 1;
        }
        if r.stuck {
            stuck += 1;
        }
        winner_sum += r.vp.iter().max().copied().unwrap_or(0);
        player_sum += r.vp.iter().sum::<i64>();
        for i in 0..6 {
            agg[i] += r.actions[i];
        }
        flipped_sum += r.flipped;
        links_sum += r.links;
        println!(
            "{},{},{},{},{},{},{},{},{},{},{},{},{},{}",
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
            r.links
        );
    }
    let n = results.len().max(1) as f64;
    eprintln!(
        "[sweep_canal] games={} policy={} illegal={} stuck={} canal_winner_mean={:.1} canal_player_mean={:.1} \
         build={:.1} network={:.1} develop={:.1} sell={:.1} loan={:.1} pass={:.1} flipped/game={:.1} links/game={:.1}",
        results.len(),
        policy,
        illegal,
        stuck,
        winner_sum as f64 / n,
        player_sum as f64 / (n * players as f64),
        agg[0] as f64 / n,
        agg[1] as f64 / n,
        agg[2] as f64 / n,
        agg[3] as f64 / n,
        agg[4] as f64 / n,
        agg[5] as f64 / n,
        flipped_sum as f64 / n,
        links_sum as f64 / n
    );
}
