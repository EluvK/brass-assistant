//! Temporary: measure build composition and VP sources across games.
use brass_engine::engine::{advance_turn, end_canal_era, end_game, TurnResult};
use brass_engine::heuristic_ai;
use brass_engine::rules::{apply_move, Move};
use brass_engine::state::GameState;
use rand::SeedableRng;

fn main() {
    let games: u64 = std::env::args().nth(1).unwrap_or("200".into()).parse().unwrap();
    let mut builds = [0u64; 6];
    let mut flips = [0u64; 6];
    let mut vp_hist = [0u64; 4];
    let mut sell_actions = 0u64;
    let mut total_vp = 0f64;
    for g in 0..games {
        let rng = rand::rngs::StdRng::seed_from_u64(g);
        let mut state = GameState::new(rng, 4);
        let mut guard = 0;
        while !state.game_over && guard < 200_000 {
            guard += 1;
            let pid = state.current_player_id();
            let mv = heuristic_ai::choose_action(&mut state).mv;
            if let Move::Build { ind, .. } = &mv {
                builds[*ind as usize] += 1;
            }
            if let Move::Sell { .. } = &mv {
                sell_actions += 1;
            }
            if apply_move(&mut state, &mv).is_err() {
                break;
            }
            match advance_turn(&mut state) {
                TurnResult::Continue => {}
                TurnResult::EndCanalEra => end_canal_era(&mut state),
                TurnResult::EndGame => end_game(&mut state),
            }
        }
        for tile in state.city_tiles.iter().flatten() {
            if tile.flipped {
                flips[tile.ind as usize] += 1;
            }
        }
        for p in state.players.iter() {
            vp_hist[(p.vp as usize / 50).min(3)] += 1;
            total_vp += p.vp as f64;
        }
    }
    let names = ["Cotton", "Coal", "Iron", "Manufacturer", "Pottery", "Brewery"];
    println!("=== builds per game (all players) ===");
    for (i, n) in names.iter().enumerate() {
        println!("  {}: {:.1}", n, builds[i] as f64 / games as f64);
    }
    println!("=== flips per game (all players) ===");
    for (i, n) in names.iter().enumerate() {
        println!("  {}: {:.1}", n, flips[i] as f64 / games as f64);
    }
    println!("=== sell actions/game: {:.1}", sell_actions as f64 / games as f64);
    println!("=== avg final VP: {:.1}", total_vp / games as f64);
    println!(
        "=== VP distribution per player: 0-49={} 50-99={} 100-149={} 150+={}",
        vp_hist[0], vp_hist[1], vp_hist[2], vp_hist[3]
    );
}
