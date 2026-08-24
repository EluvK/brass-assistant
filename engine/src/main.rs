use brass_engine::game_loop::{self, AfterEra, GameHooks, LoopOutcome};
use brass_engine::heuristic_ai;
use brass_engine::mcts_ai::{self, MctsConfig};
use brass_engine::random_ai::choose_random_move;
use brass_engine::rules::Move;
use brass_engine::scoring;
use brass_engine::state::GameState;
use rand_chacha::rand_core::SeedableRng;
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
    illegal_move: bool,
    /// Action counts across the whole game (all players summed).
    builds: u64,
    networks: u64,
    develops: u64,
    sells: u64,
    loans: u64,
    passes: u64,
    /// Final income level per player.
    final_income: [i64; 4],
    /// Mixed mode: MCTS-seat outcome.
    mcts_games: u64,
    mcts_wins: u64,
    mcts_vp: i64,
    other_vp: i64,
}

fn play_one_game(players: usize, policy: &str, seed: u64, sims: usize) -> GameStats {
    let rng = rand_chacha::ChaCha12Rng::seed_from_u64(seed);
    let mut state = GameState::new(rng, players);
    // RNG for the random baseline (state.rng is setup-only).
    let mut rand_rng = rand_chacha::ChaCha12Rng::from_rng(&mut rand::rng());
    let mut canal_events = 0u64;
    let mut stuck = false;
    let mut illegal_move = false;
    let mut builds = 0u64;
    let mut networks = 0u64;
    let mut develops = 0u64;
    let mut sells = 0u64;
    let mut loans = 0u64;
    let mut passes = 0u64;

    // "mcts-vs-heur": exactly one seat plays MCTS, the rest heuristic. The
    // MCTS seat rotates per game to cancel seat bias.
    let mcts_vs_heur = policy == "mcts-vs-heur";
    let mcts_vs_random = policy == "mcts-vs-random";
    let mcts_mixed = mcts_vs_heur || mcts_vs_random;
    let policy = if mcts_mixed { "mixed" } else { policy };
    let mcts_seat = if mcts_mixed {
        (seed as usize) % players
    } else {
        usize::MAX
    };

    let mut on_move = |_state: &mut GameState, mv: &Move| match mv {
        Move::Build { .. } => builds += 1,
        Move::Network { .. } | Move::NetworkDouble { .. } => networks += 1,
        Move::Develop { .. } | Move::ResolveFreeDevelop { .. } => develops += 1,
        Move::Sell { .. } => sells += 1,
        Move::Loan { .. } => loans += 1,
        Move::Pass { .. } => passes += 1,
        Move::Scout { .. } => {}
    };
    let mut on_after = |state: &mut GameState, mv: &Move, result: &Result<String, String>| {
        if let Err(err) = result {
            eprintln!(
                "[illegal] seed={} policy={} pid={} era={:?} round={} hand={} move={} err={}",
                seed,
                policy,
                state.current_player_id(),
                state.era,
                state.round,
                state.players[state.current_player_id()].hand.len(),
                mv.describe(state),
                err
            );
        }
    };
    let mut on_era = |_state: &mut GameState, era: brass_engine::data::Era| -> AfterEra {
        if era == brass_engine::data::Era::Canal {
            canal_events += 1;
        }
        AfterEra::Continue
    };
    let hooks = GameHooks {
        before_move: Some(&mut on_move),
        after_move: Some(&mut on_after),
        on_era: Some(&mut on_era),
        ..Default::default()
    };
    let outcome = game_loop::play(&mut state, 200_000, hooks, |state| {
        let pid = state.current_player_id();
        if mcts_mixed {
            if pid == mcts_seat {
                let cfg = MctsConfig {
                    simulations: sims,
                    ..Default::default()
                };
                Some(mcts_ai::choose_action_mcts(state, &cfg).mv)
            } else if mcts_vs_heur {
                Some(heuristic_ai::choose_action(state).mv)
            } else {
                choose_random_move(state, &mut rand_rng)
            }
        } else {
            match policy {
                "random" => choose_random_move(state, &mut rand_rng),
                "heuristic" => Some(heuristic_ai::choose_action(state).mv),
                "mcts" => {
                    let cfg = MctsConfig {
                        simulations: sims,
                        ..Default::default()
                    };
                    Some(mcts_ai::choose_action_mcts(state, &cfg).mv)
                }
                _ => Some(heuristic_ai::choose_action(state).mv),
            }
        }
    });
    if outcome == LoopOutcome::IllegalMove {
        illegal_move = true;
    }
    if outcome == LoopOutcome::GuardExceeded {
        stuck = true;
    }
    game_loop::finish_game(&mut state);

    let built = state.city_tiles.iter().flatten().count() as u64;
    let flipped = state
        .city_tiles
        .iter()
        .flatten()
        .filter(|t| t.flipped)
        .count() as u64;
    let links = state.links.iter().flatten().count() as u64;

    let mut wins = [0u64; 4];
    let winner = scoring::final_ranking(&state).first().copied();
    if let Some(w) = winner {
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

    let mut mcts_games = 0;
    let mut mcts_wins = 0;
    let mut mcts_vp = 0;
    let mut other_vp = 0;
    if mcts_mixed {
        mcts_games = 1;
        if winner == Some(mcts_seat) {
            mcts_wins = 1;
        }
        mcts_vp = vp[mcts_seat];
        other_vp = (0..players)
            .filter(|&i| i != mcts_seat)
            .map(|i| vp[i])
            .sum::<i64>()
            / (players.saturating_sub(1) as i64);
    }

    let mut final_income = [0i64; 4];
    for (i, p) in state.players.iter().enumerate() {
        if i < final_income.len() {
            final_income[i] = p.income_level() as i64;
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
        illegal_move,
        builds,
        networks,
        develops,
        sells,
        loans,
        passes,
        final_income,
        mcts_games,
        mcts_wins,
        mcts_vp,
        other_vp,
    }
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let games: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(100);
    let players: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(4);
    // policy: "random" | "heuristic" | "mcts" | "mcts-vs-heur"
    let policy = args
        .get(3)
        .cloned()
        .unwrap_or_else(|| "heuristic".to_string());
    let mcts_vs_heur = policy == "mcts-vs-heur";
    // threads: optional 4th arg to override rayon default pool size
    let threads: Option<usize> = args.get(4).and_then(|s| s.parse().ok());
    // mcts sims: optional 5th arg
    let sims: usize = args.get(5).and_then(|s| s.parse().ok()).unwrap_or(200);

    let pool = match threads {
        Some(t) => rayon::ThreadPoolBuilder::new()
            .num_threads(t)
            .build()
            .unwrap(),
        None => rayon::ThreadPoolBuilder::new().build().unwrap(),
    };

    let start = std::time::Instant::now();
    let total: GameStats = pool.install(|| {
        (0..games)
            .into_par_iter()
            .map(|g| play_one_game(players, &policy, g as u64, sims))
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
                illegal_move: a.illegal_move || b.illegal_move,
                builds: a.builds + b.builds,
                networks: a.networks + b.networks,
                develops: a.develops + b.develops,
                sells: a.sells + b.sells,
                loans: a.loans + b.loans,
                passes: a.passes + b.passes,
                final_income: [
                    a.final_income[0] + b.final_income[0],
                    a.final_income[1] + b.final_income[1],
                    a.final_income[2] + b.final_income[2],
                    a.final_income[3] + b.final_income[3],
                ],
                mcts_games: a.mcts_games + b.mcts_games,
                mcts_wins: a.mcts_wins + b.mcts_wins,
                mcts_vp: a.mcts_vp + b.mcts_vp,
                other_vp: a.other_vp + b.other_vp,
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
    if total.illegal_move {
        println!("[!] at least one game aborted due to an illegal move returned by a policy");
    }
    println!("Canal-era transitions: {}", total.canal_events);

    if mcts_vs_heur {
        let g = total.mcts_games.max(1);
        let opp = "heuristic";
        println!(
            "  [MCTS vs {opp}, seat rotated] MCTS seat: {:.1}% wins ({}/{}), avg VP {:.1} vs {opp} avg {:.1}",
            total.mcts_wins as f64 * 100.0 / g as f64,
            total.mcts_wins,
            g,
            total.mcts_vp as f64 / g as f64,
            total.other_vp as f64 / g as f64
        );
    }
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
    let g = games as f64;
    println!(
        "Avg actions/game: build {:.1}, network {:.1}, develop {:.1}, sell {:.1}, loan {:.1}, pass {:.1}",
        total.builds as f64 / g,
        total.networks as f64 / g,
        total.develops as f64 / g,
        total.sells as f64 / g,
        total.loans as f64 / g,
        total.passes as f64 / g
    );
    println!(
        "Avg final income per player: {:?}",
        total
            .final_income
            .iter()
            .map(|v| *v as f64 / g)
            .collect::<Vec<_>>()
    );
    for (p, w) in total.wins.iter().enumerate() {
        if *w > 0 || games <= 4 {
            println!(
                "Player {p}: {w} wins ({:.1}%)",
                *w as f64 * 100.0 / games as f64
            );
        }
    }
}
