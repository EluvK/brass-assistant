//! High-throughput native Rust generator for imitation learning shards.
//!
//! Replaces the slower Python-orchestrated multiprocessing generation by running
//! pure Rust heuristic games concurrently with Rayon, applying quality filters,
//! computing relative and absolute VP targets, and streaming compact binary shards.
//!
//! Usage:
//!   cargo run --release --bin gen_imitation -- --games 10000 --out-dir data/imitation_shards --min-vp 30

use _engine::gameplay::game_loop::{self, GameHooks, LoopOutcome};
use _engine::heuristic_ai;
use _engine::move_codec;
use _engine::scoring::final_ranking;
use _engine::state::GameState;
use rand_chacha::ChaCha12Rng;
use rand_chacha::rand_core::SeedableRng;
use rayon::prelude::*;
use std::fs::{self, File};
use std::io::BufWriter;
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc::channel;
use std::time::Instant;

pub const SNAPSHOT_MAGIC: &[u8; 4] = b"BASS";
pub const SNAPSHOT_VERSION: u8 = 3;

pub use _engine::imitation::ImitationRecord;

struct StepData {
    pid: usize,
    era: usize,
    snapshot: Vec<u8>,
    canon: String,
}

struct GameResult {
    steps: Vec<StepData>,
    canal_econ: Vec<(i32, i32)>,
    final_econ: Vec<(i32, i32)>,
    vps: Vec<i32>,
    winner_idx: usize,
}

fn simulate_game(seed: u64, players: usize, max_moves: usize) -> Option<GameResult> {
    let rng = ChaCha12Rng::seed_from_u64(seed);
    let mut state = GameState::new(rng, players);
    let mut steps = Vec::new();
    let mut canal_econ = Vec::new();

    let mut before_move = |s: &mut GameState, mv: &_engine::rules::ResolvedMove| {
        let pid = s.current_player_id();
        let era = match s.era {
            _engine::data::Era::Canal => 0usize,
            _engine::data::Era::Rail => 1usize,
        };
        let mut snapshot = Vec::new();
        snapshot.extend_from_slice(SNAPSHOT_MAGIC);
        snapshot.push(SNAPSHOT_VERSION);
        if let Ok(b) = s.snapshot_bytes() {
            snapshot.extend_from_slice(&b);
        }
        let canon = move_codec::encode(mv);
        steps.push(StepData {
            pid,
            era,
            snapshot,
            canon,
        });
    };

    let mut after_era = |s: &mut GameState, era: _engine::data::Era| {
        if era == _engine::data::Era::Canal {
            canal_econ = s
                .players
                .iter()
                .map(|p| (p.income_level() as i32, p.money))
                .collect();
        }
    };

    let hooks = GameHooks {
        before_move: Some(&mut before_move),
        after_era: Some(&mut after_era),
        ..Default::default()
    };

    let outcome = game_loop::play_rounds(&mut state, max_moves, hooks, |s| {
        heuristic_ai::choose_action(s).into_moves()
    });

    if outcome != LoopOutcome::GameOver {
        return None;
    }

    game_loop::finish_game(&mut state);

    let final_econ = state
        .players
        .iter()
        .map(|p| (p.income_level() as i32, p.money))
        .collect();
    let vps: Vec<i32> = state.players.iter().map(|p| p.vp as i32).collect();
    let ranking = final_ranking(&state);
    let winner_idx = ranking.first().copied().unwrap_or(0);

    Some(GameResult {
        steps,
        canal_econ,
        final_econ,
        vps,
        winner_idx,
    })
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mut target_games = 1000usize;
    let mut out_dir = PathBuf::from("data/imitation_shards");
    let mut min_avg_vp = 0.0f32;
    let mut min_vp = 0.0f32;
    let mut shard_size = 32768usize;
    let mut start_seed = 1000000u64;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--games" | "-n" => {
                target_games = args.get(i + 1).and_then(|s| s.parse().ok()).unwrap_or(target_games);
                i += 1;
            }
            "--out-dir" | "-o" => {
                if let Some(d) = args.get(i + 1) {
                    out_dir = PathBuf::from(d);
                }
                i += 1;
            }
            "--min-avg-vp" => {
                min_avg_vp = args.get(i + 1).and_then(|s| s.parse().ok()).unwrap_or(min_avg_vp);
                i += 1;
            }
            "--min-vp" => {
                min_vp = args.get(i + 1).and_then(|s| s.parse().ok()).unwrap_or(min_vp);
                i += 1;
            }
            "--shard-size" => {
                shard_size = args.get(i + 1).and_then(|s| s.parse().ok()).unwrap_or(shard_size);
                i += 1;
            }
            "--seed" => {
                start_seed = args.get(i + 1).and_then(|s| s.parse().ok()).unwrap_or(start_seed);
                i += 1;
            }
            _ => {}
        }
        i += 1;
    }

    fs::create_dir_all(&out_dir).expect("failed to create output directory");

    println!("============================================================");
    println!("  Brass Birmingham - Native Rust Imitation Generator");
    println!("  Target accepted games : {target_games}");
    println!("  Output directory      : {}", out_dir.display());
    println!("  Quality filter        : min_vp >= {min_vp}, min_avg_vp >= {min_avg_vp}");
    println!("  Shard size (steps)    : {shard_size}");
    println!("============================================================");

    let started = Instant::now();
    let accepted_games = AtomicUsize::new(0);
    let attempted_games = AtomicUsize::new(0);

    let (tx, rx) = channel::<Vec<ImitationRecord>>();

    // Writer thread to decouple disk I/O from simulation cores
    let writer_out_dir = out_dir.clone();
    let writer_handle = std::thread::spawn(move || {
        let mut buffer: Vec<ImitationRecord> = Vec::with_capacity(shard_size * 2);
        let mut shard_idx = 0usize;
        let mut total_records = 0usize;

        while let Ok(records) = rx.recv() {
            buffer.extend(records);
            while buffer.len() >= shard_size {
                let to_save: Vec<ImitationRecord> = buffer.drain(..shard_size).collect();
                let path = writer_out_dir.join(format!("imitation-{shard_idx:06}.bin"));
                let file = File::create(&path).expect("failed to create shard file");
                let mut writer = BufWriter::new(file);
                bincode::serialize_into(&mut writer, &to_save).expect("failed to serialize shard");
                println!("  [Shard Saved] {} ({} steps)", path.display(), to_save.len());
                shard_idx += 1;
                total_records += to_save.len();
            }
        }

        if !buffer.is_empty() {
            let path = writer_out_dir.join(format!("imitation-{shard_idx:06}.bin"));
            let file = File::create(&path).expect("failed to create shard file");
            let mut writer = BufWriter::new(file);
            bincode::serialize_into(&mut writer, &buffer).expect("failed to serialize shard");
            println!("  [Final Shard Saved] {} ({} steps)", path.display(), buffer.len());
            total_records += buffer.len();
        }

        total_records
    });

    let batch_size = 50usize;
    let mut current_seed = start_seed;

    while accepted_games.load(Ordering::Relaxed) < target_games {
        let seeds: Vec<u64> = (current_seed..current_seed + batch_size as u64).collect();
        current_seed += batch_size as u64;

        let batch_results: Vec<Vec<ImitationRecord>> = seeds
            .into_par_iter()
            .filter_map(|seed| {
                attempted_games.fetch_add(1, Ordering::Relaxed);
                let res = simulate_game(seed, 4, 600)?;

                let avg_vp = res.vps.iter().sum::<i32>() as f32 / 4.0;
                let lowest_vp = *res.vps.iter().min().unwrap_or(&0) as f32;

                if min_vp > 0.0 && lowest_vp < min_vp {
                    return None;
                }
                if min_avg_vp > 0.0 && avg_vp < min_avg_vp {
                    return None;
                }

                if accepted_games.fetch_add(1, Ordering::Relaxed) >= target_games {
                    return None;
                }

                let mut val = [0.0f32; 4];
                let mut abs = [0.0f32; 4];
                let mut win = [0.0f32; 4];
                for p in 0..4 {
                    val[p] = (res.vps[p] as f32 - avg_vp) / _engine::VP_SCALE;
                    abs[p] = (res.vps[p] as f32 - 100.0) / _engine::VP_SCALE;
                }
                win[res.winner_idx] = 1.0;

                let mut records = Vec::with_capacity(res.steps.len());
                for step in res.steps {
                    let econ_src = if step.era == 0 {
                        res.canal_econ.get(step.pid).copied().unwrap_or((0, 0))
                    } else {
                        res.final_econ.get(step.pid).copied().unwrap_or((0, 0))
                    };
                    records.push(ImitationRecord {
                        pid: step.pid,
                        era: step.era,
                        value: val,
                        abs_vp: abs,
                        winner: win,
                        econ: [econ_src.0 as f32, econ_src.1 as f32],
                        snapshot: step.snapshot,
                        teacher_canonical: step.canon,
                    });
                }

                Some(records)
            })
            .collect();

        for records in batch_results {
            tx.send(records).expect("receiver hung up");
        }

        let acc = accepted_games.load(Ordering::Relaxed).min(target_games);
        let att = attempted_games.load(Ordering::Relaxed);
        let elapsed = started.elapsed().as_secs_f64();
        let rate = att as f64 / elapsed.max(0.001);
        print!("\r  Progress: accepted {acc}/{target_games} | attempted {att} | {rate:.1} games/sec    ");
        use std::io::Write;
        std::io::stdout().flush().unwrap();
    }

    println!();
    drop(tx); // Close channel to signal writer thread to finish

    let total_steps = writer_handle.join().expect("writer thread panicked");
    let total_time = started.elapsed().as_secs_f64();
    println!("------------------------------------------------------------");
    println!("  Completed in {:.2} seconds", total_time);
    println!("  Total Steps Produced : {total_steps}");
    println!("  Throughput           : {:.1} games/second", target_games as f64 / total_time);
    println!("============================================================");
}
