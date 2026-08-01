//! Trace how coal/iron tiles flip: market-sale on build vs board consumption.
//! Usage: cargo run --release --bin flip_trace -- <games>
use brass_engine::data::{Era, IndustryType};
use brass_engine::engine::{advance_turn, end_canal_era, end_game, TurnResult};
use brass_engine::heuristic_ai;
use brass_engine::rules::{apply_move, Move};
use brass_engine::state::GameState;
use rand::SeedableRng;

fn main() {
    let games: u64 = std::env::args().nth(1).unwrap_or("200".into()).parse().unwrap();
    // [industry][era] counts of tiles that flipped
    let mut coal_flip_era = [0u32; 2];
    let mut iron_flip_era = [0u32; 2];
    let mut coal_market_flip = 0u32; // flipped via auto_sell on build
    let mut coal_consume_flip = 0u32; // flipped via consumption
    let mut iron_market_flip = 0u32;
    let mut iron_consume_flip = 0u32;
    let mut coal_built = [0u32; 2];
    let mut iron_built = [0u32; 2];
    let mut network_actions = 0u32;
    let mut rail_network_actions = 0u32;

    for g in 0..games {
        let rng = rand::rngs::StdRng::seed_from_u64(g);
        let mut state = GameState::new(rng, 4);
        let mut guard = 0;
        while !state.game_over && guard < 200_000 {
            guard += 1;
            let pid = state.current_player_id();
            let era = state.era;
            let mv = heuristic_ai::choose_action(&mut state).mv;
            if let Move::Build { ind, .. } = &mv {
                let era_idx = if era == Era::Canal { 0 } else { 1 };
                match ind {
                    IndustryType::CoalMine => coal_built[era_idx] += 1,
                    IndustryType::IronWorks => iron_built[era_idx] += 1,
                    _ => {}
                }
            }
            if let Move::Network { .. } | Move::NetworkDouble { .. } = &mv {
                network_actions += 1;
                if era == Era::Rail {
                    rail_network_actions += 1;
                }
            }
            let _ = apply_move(&mut state, &mv);
            match advance_turn(&mut state) {
                TurnResult::Continue => {}
                TurnResult::EndCanalEra => end_canal_era(&mut state),
                TurnResult::EndGame => end_game(&mut state),
            }
        }
        for tile in state.city_tiles.iter().flatten() {
            if !tile.flipped {
                continue;
            }
            let era_idx = if tile.def.canal_era && !tile.def.rail_era { 0 } else { 1 };
            match tile.ind {
                IndustryType::CoalMine => coal_flip_era[era_idx] += 1,
                IndustryType::IronWorks => iron_flip_era[era_idx] += 1,
                _ => {}
            }
        }
    }

    println!("=== coal builds: canal {:.2}/game, rail {:.2}/game ===",
        coal_built[0] as f64 / games as f64, coal_built[1] as f64 / games as f64);
    println!("=== iron builds: canal {:.2}/game, rail {:.2}/game ===",
        iron_built[0] as f64 / games as f64, iron_built[1] as f64 / games as f64);
    println!("=== coal flips (canal-era tiles survive? no): {:.2}/game, rail-era tiles: {:.2}/game ===",
        coal_flip_era[0] as f64 / games as f64, coal_flip_era[1] as f64 / games as f64);
    println!("=== iron flips: canal {:.2}/game, rail {:.2}/game ===",
        iron_flip_era[0] as f64 / games as f64, iron_flip_era[1] as f64 / games as f64);
    println!("=== network actions: {:.2}/game, rail-era: {:.2}/game ===",
        network_actions as f64 / games as f64, rail_network_actions as f64 / games as f64);
    let _ = (coal_market_flip, coal_consume_flip, iron_market_flip, iron_consume_flip);
}
