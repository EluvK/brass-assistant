//! Batch score sweep across a seed range, output as CSV with per-player final
//! VP plus economic columns (final income, final money, canal-era income).
//!
//! Usage:
//!   cargo run --release --bin sweep_scores -- <start_seed> <end_seed> [policy] [full|canal] > out.csv
//!
//!   policy: heuristic (default; the only swept policy)
//!   range:  full (default) or canal

use _engine::data::Era;
use _engine::game_loop::{self, AfterEra, GameHooks, LoopOutcome};
use _engine::state::GameState;
use rand_chacha::rand_core::SeedableRng;
use rayon::prelude::*;
use std::collections::BTreeMap;
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

fn play_one(seed: u64, players: usize, canal_only: bool) -> SweepResult {
    let started = Instant::now();
    let rng = rand_chacha::ChaCha12Rng::seed_from_u64(seed);
    let mut state = GameState::new(rng, players);
    let res = std::cell::RefCell::new(SweepResult {
        seed,
        ..Default::default()
    });

    let mut on_move = |_state: &mut GameState, mv: &_engine::rules::ResolvedMove| {
        let mut r = res.borrow_mut();
        match mv {
            _engine::rules::ResolvedMove::Build { .. } => r.actions[0] += 1,
            _engine::rules::ResolvedMove::Network { .. }
            | _engine::rules::ResolvedMove::NetworkDouble { .. } => r.actions[1] += 1,
            _engine::rules::ResolvedMove::Develop { .. } => r.actions[2] += 1,
            _engine::rules::ResolvedMove::Sell { .. } => r.actions[3] += 1,
            _engine::rules::ResolvedMove::Loan { .. } => r.actions[4] += 1,
            _engine::rules::ResolvedMove::Pass { .. } => r.actions[5] += 1,
            _engine::rules::ResolvedMove::Scout { .. } => {}
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
    let outcome = game_loop::play_rounds(&mut state, 200_000, hooks, |state| {
        _engine::heuristic_ai::choose_action(state).into_moves()
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

fn mean(values: impl Iterator<Item = f64>) -> f64 {
    let mut count = 0.0;
    let mut sum = 0.0;
    for value in values {
        count += 1.0;
        sum += value;
    }
    if count == 0.0 { 0.0 } else { sum / count }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let start: u64 = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(0);
    let end: u64 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(500);
    let policy = args
        .get(3)
        .cloned()
        .unwrap_or_else(|| "heuristic".to_string());
    if policy != "heuristic" {
        eprintln!("unknown policy {policy:?}; only heuristic is supported");
        std::process::exit(2);
    }
    let canal_only = matches!(args.get(4).map(String::as_str), Some("canal"));
    let players: usize = 4;

    let sweep_start = Instant::now();
    let results: Vec<SweepResult> = (start..end)
        .into_par_iter()
        .map(|seed| play_one(seed, players, canal_only))
        .collect();
    let total_elapsed = sweep_start.elapsed();

    if canal_only {
        println!(
            "seed,p0,p1,p2,p3,winner,build,network,develop,sell,loan,pass,flipped,links,elapsed_us"
        );
    } else {
        println!(
            "seed,p0,p1,p2,p3,avg,income0,income1,income2,income3,money0,money1,money2,money3,canal_income0,canal_income1,canal_income2,canal_income3,elapsed_us"
        );
    }

    if results.is_empty() {
        eprintln!("[sweep_scores]: 0 games evaluated");
        return;
    }

    let n = results.len() as f64;
    let mut illegal = 0;
    let mut stuck = 0;

    let mut winner_scores = Vec::with_capacity(results.len());
    let mut min_scores = Vec::with_capacity(results.len());
    let mut game_avgs = Vec::with_capacity(results.len());

    let mut seat_vp_sums = [0i64; 4];
    let mut seat_wins = [0usize; 4];
    let mut seat_final_income_sums = [0i64; 4];
    let mut seat_final_money_sums = [0i64; 4];
    let mut seat_canal_income_sums = [0i64; 4];
    let mut action_sums = [0u64; 6];
    let mut flipped_sum = 0u64;
    let mut links_sum = 0u64;

    for r in &results {
        if r.illegal {
            illegal += 1;
        }
        if r.stuck {
            stuck += 1;
        }

        let w = r.vp.iter().copied().max().unwrap_or(0);
        let m = r.vp.iter().copied().min().unwrap_or(0);
        let avg = r.vp.iter().sum::<i64>() as f64 / players as f64;

        winner_scores.push(w);
        min_scores.push(m);
        game_avgs.push(avg);

        for p in 0..players.min(4) {
            seat_vp_sums[p] += r.vp[p];
            if r.vp[p] == w {
                seat_wins[p] += 1;
            }
            seat_final_income_sums[p] += r.final_income[p] as i64;
            seat_final_money_sums[p] += r.final_money[p] as i64;
            seat_canal_income_sums[p] += r.canal_income[p] as i64;
        }

        for (i, &act) in r.actions.iter().enumerate() {
            action_sums[i] += act;
        }
        flipped_sum += r.flipped;
        links_sum += r.links;

        if canal_only {
            println!(
                "{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}",
                r.seed,
                r.vp[0],
                r.vp[1],
                r.vp[2],
                r.vp[3],
                w,
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

    let winner_mean = mean(winner_scores.iter().map(|&s| s as f64));
    let winner_min = winner_scores.iter().copied().min().unwrap_or(0);
    let winner_max = winner_scores.iter().copied().max().unwrap_or(0);

    let game_mean = mean(game_avgs.iter().copied());
    let game_min = game_avgs.iter().copied().fold(f64::INFINITY, f64::min);
    let game_max = game_avgs.iter().copied().fold(f64::NEG_INFINITY, f64::max);

    let min_mean = mean(min_scores.iter().map(|&s| s as f64));
    let min_min = min_scores.iter().copied().min().unwrap_or(0);
    let min_max = min_scores.iter().copied().max().unwrap_or(0);

    let time_mean = mean(results.iter().map(|r| r.elapsed_us as f64));

    eprintln!(
        "\n[sweep_scores]:\nscope={} games={} policy={} illegal={} stuck={}\nwinner_mean={:.3} (min={}, max={})\ngame_mean={:.3} (min={:.1}, max={:.1})\nlowest_mean={:.3} (min={}, max={})\ntime_mean_us={:.1} (total={:.2?}, {:.1} games/s)",
        if canal_only { "canal" } else { "full" },
        results.len(),
        policy,
        illegal,
        stuck,
        winner_mean,
        winner_min,
        winner_max,
        game_mean,
        game_min,
        game_max,
        min_mean,
        min_min,
        min_max,
        time_mean,
        total_elapsed,
        if total_elapsed.as_secs_f64() > 0.0 {
            results.len() as f64 / total_elapsed.as_secs_f64()
        } else {
            0.0
        },
    );

    if canal_only {
        eprintln!("\nSeat Breakdown:");
        eprintln!(
            "  {:>4}   {:>8}   {:>8}   {:>12}",
            "Seat", "Avg VP", "Win Rate", "Canal Income"
        );
        eprintln!("  {:-<4}   {:-<8}   {:-<8}   {:-<12}", "", "", "", "");
        for p in 0..players.min(4) {
            eprintln!(
                "    P{}   {:>8.2}   {:>7.1}%   {:>12.2}",
                p,
                seat_vp_sums[p] as f64 / n,
                (seat_wins[p] as f64 / n) * 100.0,
                seat_canal_income_sums[p] as f64 / n,
            );
        }

        let total_actions: u64 = action_sums.iter().sum();
        eprintln!("\nCanal Actions Breakdown (total={}):", total_actions);
        let act_names = ["Build", "Network", "Develop", "Sell", "Loan", "Pass"];
        for (i, name) in act_names.iter().enumerate() {
            let cnt = action_sums[i];
            let pct = if total_actions > 0 {
                (cnt as f64 / total_actions as f64) * 100.0
            } else {
                0.0
            };
            eprintln!("  {:>8}: {:>6} ({:>5.1}%)", name, cnt, pct);
        }
        eprintln!(
            "  Avg Flipped Tiles: {:.2}, Avg Built Links: {:.2}",
            flipped_sum as f64 / n,
            links_sum as f64 / n
        );
    } else {
        eprintln!("\nSeat Breakdown:");
        eprintln!(
            "  {:>4}   {:>8}   {:>8}   {:>12}   {:>11}   {:>12}",
            "Seat", "Avg VP", "Win Rate", "Final Income", "Final Money", "Canal Income"
        );
        eprintln!(
            "  {:-<4}   {:-<8}   {:-<8}   {:-<12}   {:-<11}   {:-<12}",
            "", "", "", "", "", ""
        );
        for p in 0..players.min(4) {
            eprintln!(
                "    P{}   {:>8.2}   {:>7.1}%   {:>12.2}   {:>11.2}   {:>12.2}",
                p,
                seat_vp_sums[p] as f64 / n,
                (seat_wins[p] as f64 / n) * 100.0,
                seat_final_income_sums[p] as f64 / n,
                seat_final_money_sums[p] as f64 / n,
                seat_canal_income_sums[p] as f64 / n,
            );
        }
    }

    let min_tier = min_min.div_euclid(20) * 20;
    let max_tier = min_max.div_euclid(20) * 20;

    let mut tier_counts = BTreeMap::<i64, usize>::new();
    for &score in &min_scores {
        *tier_counts.entry(score.div_euclid(20) * 20).or_insert(0) += 1;
    }

    let max_count = tier_counts.values().copied().max().unwrap_or(1);
    let max_bar_width = 40;

    eprintln!("\nLowest Score Distribution (20 VP tiers):");
    eprintln!(
        "  {:>11}   {:>6}   {:>7}   {}",
        "Tier Range", "Count", "Percent", "Histogram"
    );
    eprintln!("  {:-<11}   {:-<6}   {:-<7}   {:-<40}", "", "", "", "");
    let mut tier = min_tier;
    while tier <= max_tier {
        let count = tier_counts.get(&tier).copied().unwrap_or(0);
        let pct = (count as f64 / n) * 100.0;
        let bar_len = if max_count > 0 {
            ((count as f64 / max_count as f64) * max_bar_width as f64).round() as usize
        } else {
            0
        };
        let bar = "#".repeat(bar_len);
        eprintln!(
            "  [{:>3}, {:>3}]:   {:>6}   {:>6.1}%   {}",
            tier,
            tier + 19,
            count,
            pct,
            bar
        );
        tier += 20;
    }
}
