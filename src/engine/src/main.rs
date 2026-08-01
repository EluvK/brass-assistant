use brass_engine::engine::{advance_turn, end_canal_era, end_game, TurnResult};
use brass_engine::heuristic_ai;
use brass_engine::random_ai::choose_random_move;
use brass_engine::scoring;
use brass_engine::state::GameState;
use rand::SeedableRng;
use std::env;

fn main() {
    let args: Vec<String> = env::args().collect();
    let games: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(100);
    let players: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(4);
    // policy: "random" | "heuristic" (default heuristic)
    let policy = args.get(3).cloned().unwrap_or_else(|| "heuristic".to_string());

    let mut wins = vec![0u64; players];
    let mut canal_events = 0u64;
    let mut vp_acc = vec![0i64; players];
    let mut built_acc = 0u64;
    let mut flipped_acc = 0u64;
    let mut links_acc = 0u64;

    let start = std::time::Instant::now();
    for g in 0..games {
        let rng = rand::rngs::StdRng::seed_from_u64(g as u64);
        let mut state = GameState::new(rng, players);

        let mut guard = 0;
        loop {
            guard += 1;
            if guard > 200_000 {
                println!("game {g}: stuck (guard)");
                break;
            }
            if state.game_over {
                break;
            }

            let m = if policy == "random" {
                choose_random_move(&mut state)
            } else {
                Some(heuristic_ai::choose_action(&mut state).mv)
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

        built_acc += state.city_tiles.iter().flatten().count() as u64;
        flipped_acc += state
            .city_tiles
            .iter()
            .flatten()
            .filter(|t| t.flipped)
            .count() as u64;
        links_acc += state.links.iter().flatten().count() as u64;

        let ranking = scoring::final_ranking(&state);
        if let Some(&winner) = ranking.first() {
            wins[winner] += 1;
        }
        for (i, p) in state.players.iter().enumerate() {
            vp_acc[i] += p.vp as i64;
        }
    }

    let elapsed = start.elapsed();

    println!("Games: {games}, Players: {players}, policy: {policy}");
    println!(
        "Elapsed: {:.2?} ({:.0} games/s)",
        elapsed,
        games as f64 / elapsed.as_secs_f64()
    );
    println!("Canal-era transitions: {canal_events}");
    println!(
        "Avg final VP per player: {:?}",
        vp_acc.iter().map(|v| *v as f64 / games as f64).collect::<Vec<_>>()
    );
    println!(
        "Avg per game: built {:.1}, flipped {:.1}, links {:.1}",
        built_acc as f64 / games as f64,
        flipped_acc as f64 / games as f64,
        links_acc as f64 / games as f64
    );
    for (p, w) in wins.iter().enumerate() {
        println!("Player {p}: {w} wins ({:.1}%)", *w as f64 * 100.0 / games as f64);
    }
}

