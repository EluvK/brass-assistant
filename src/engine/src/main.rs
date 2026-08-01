use brass_engine::engine::{advance_turn, end_canal_era, end_game, TurnResult};
use brass_engine::heuristic_ai;
use brass_engine::random_ai::choose_random_move;
use brass_engine::scoring;
use brass_engine::search_ai;
use brass_engine::state::GameState;
use rand::SeedableRng;
use rayon::prelude::*;
use std::env;

#[derive(Default, Clone, Copy)]
struct GameStats {
    wins: [u64; 4],
    vp: [i64; 4],
    built: u64,
    flipped: u64,
    links: u64,
    canal_events: u64,
    stuck: bool,
}

fn play_one_game(players: usize, policy: &str, seed: u64) -> GameStats {
    let rng = rand::rngs::StdRng::seed_from_u64(seed);
    let mut state = GameState::new(rng, players);
    let mut canal_events = 0u64;
    let mut stuck = false;

    let mut guard = 0;
    loop {
        guard += 1;
        if guard > 200_000 {
            stuck = true;
            break;
        }
        if state.game_over {
            break;
        }

        let m = match policy {
            "random" => choose_random_move(&mut state),
            "2ply" => Some(search_ai::choose_action_2ply(&mut state).mv),
            _ => Some(heuristic_ai::choose_action(&mut state).mv),
        };
        if let Some(mv) = m {
            let _ = brass_engine::rules::apply_move(&mut state, &mv);
        }

        match advance_turn(&mut state) {
            TurnResult::Continue => {}
            TurnResult::EndCanalEra => {
                end_canal_era(&mut state);
                canal_events += 1;
            }
            TurnResult::EndGame => {
                end_game(&mut state);
            }
        }
    }

    if !state.game_over {
        end_game(&mut state);
    }

    let built = state.city_tiles.iter().flatten().count() as u64;
    let flipped = state
        .city_tiles
        .iter()
        .flatten()
        .filter(|t| t.flipped)
        .count() as u64;
    let links = state.links.iter().flatten().count() as u64;

    let mut wins = [0u64; 4];
    if let Some(&w) = scoring::final_ranking(&state).first() {
        if w < 4 {
            wins[w] = 1;
        }
    }
    let mut vp = [0i64; 4];
    for (i, p) in state.players.iter().enumerate() {
        if i < vp.len() {
            vp[i] = p.vp as i64;
        }
    }

    GameStats {
        wins,
        vp,
        built,
        flipped,
        links,
        canal_events,
        stuck,
    }
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let games: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(100);
    let players: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(4);
    // policy: "random" | "heuristic" | "2ply" (default heuristic)
    let policy = args.get(3).cloned().unwrap_or_else(|| "heuristic".to_string());
    // threads: optional 4th arg to override rayon default pool size
    let threads: Option<usize> = args.get(4).and_then(|s| s.parse().ok());

    let pool = match threads {
        Some(t) => rayon::ThreadPoolBuilder::new().num_threads(t).build().unwrap(),
        None => rayon::ThreadPoolBuilder::new().build().unwrap(),
    };

    let start = std::time::Instant::now();
    let total: GameStats = pool.install(|| {
        (0..games)
            .into_par_iter()
            .map(|g| play_one_game(players, &policy, g as u64))
            .reduce(GameStats::default, |a, b| GameStats {
                wins: [
                    a.wins[0] + b.wins[0],
                    a.wins[1] + b.wins[1],
                    a.wins[2] + b.wins[2],
                    a.wins[3] + b.wins[3],
                ],
                vp: [
                    a.vp[0] + b.vp[0],
                    a.vp[1] + b.vp[1],
                    a.vp[2] + b.vp[2],
                    a.vp[3] + b.vp[3],
                ],
                built: a.built + b.built,
                flipped: a.flipped + b.flipped,
                links: a.links + b.links,
                canal_events: a.canal_events + b.canal_events,
                stuck: a.stuck || b.stuck,
            })
    });
    let elapsed = start.elapsed();

    println!("Games: {games}, Players: {players}, policy: {policy}");
    println!(
        "Elapsed: {:.2?} ({:.0} games/s)",
        elapsed,
        games as f64 / elapsed.as_secs_f64()
    );
    if total.stuck {
        println!("[!] at least one game hit the guard");
    }
    println!("Canal-era transitions: {}", total.canal_events);
    println!(
        "Avg final VP per player: {:?}",
        total
            .vp
            .iter()
            .map(|v| *v as f64 / games as f64)
            .collect::<Vec<_>>()
    );
    println!(
        "Avg per game: built {:.1}, flipped {:.1}, links {:.1}",
        total.built as f64 / games as f64,
        total.flipped as f64 / games as f64,
        total.links as f64 / games as f64
    );
    for (p, w) in total.wins.iter().enumerate() {
        if *w > 0 || games <= 4 {
            println!("Player {p}: {w} wins ({:.1}%)", *w as f64 * 100.0 / games as f64);
        }
    }
}

