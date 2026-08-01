//! Track market state at each resource build. Usage:
//! cargo run --release --bin market_build -- <games>
use brass_engine::engine::{advance_turn, end_canal_era, end_game, TurnResult};
use brass_engine::heuristic_ai;
use brass_engine::rules::{apply_move, Move};
use brass_engine::state::GameState;
use rand::SeedableRng;

fn main() {
    let games: u64 = std::env::args().nth(1).unwrap_or("200".into()).parse().unwrap();
    let mut coal_builds = Vec::new();
    let mut iron_builds = Vec::new();
    let mut coal_flip = Vec::new();
    let mut iron_flip = Vec::new();
    for g in 0..games {
        let rng = rand::rngs::StdRng::seed_from_u64(g);
        let mut state = GameState::new(rng, 4);
        let mut guard = 0;
        while !state.game_over && guard < 200_000 {
            guard += 1;
            let pid = state.current_player_id();
            let era = state.era;
            let cm = state.coal_market;
            let im = state.iron_market;
            let mv = heuristic_ai::choose_action(&mut state).mv;
            if let Move::Build { ind, .. } = &mv {
                let ind = *ind;
                if ind == brass_engine::data::IndustryType::CoalMine {
                    coal_builds.push((era as u8, cm, im));
                } else if ind == brass_engine::data::IndustryType::IronWorks {
                    iron_builds.push((era as u8, cm, im));
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
            if tile.ind == brass_engine::data::IndustryType::CoalMine && tile.flipped {
                coal_flip.push(());
            }
            if tile.ind == brass_engine::data::IndustryType::IronWorks && tile.flipped {
                iron_flip.push(());
            }
        }
        if g == 0 {
            println!(
                "seed0 end: coal_market={}/14 iron_market={}/10",
                state.coal_market, state.iron_market
            );
        }
    }
    let avg = |v: &[(u8, usize, usize)], idx: usize| -> f64 {
        v.iter()
            .map(|x| match idx {
                0 => x.0 as f64,
                1 => x.1 as f64,
                _ => x.2 as f64,
            })
            .sum::<f64>()
            / v.len().max(1) as f64
    };
    println!("=== coal builds: {} (per game {:.1}) ===", coal_builds.len(), coal_builds.len() as f64 / games as f64);
    println!("  avg coal market at build: {:.1}/14, iron: {:.1}/10", avg(&coal_builds, 1), avg(&coal_builds, 2));
    let canal_coal = coal_builds.iter().filter(|x| x.0 == 0).count();
    let rail_coal = coal_builds.iter().filter(|x| x.0 == 1).count();
    println!("  canal-era coal builds: {canal_coal} (per game {:.2}), rail-era: {rail_coal} (per game {:.2})", canal_coal as f64 / games as f64, rail_coal as f64 / games as f64);
    println!("=== iron builds: {} (per game {:.1}) ===", iron_builds.len(), iron_builds.len() as f64 / games as f64);
    println!("  avg coal market at build: {:.1}/14, iron: {:.1}/10", avg(&iron_builds, 1), avg(&iron_builds, 2));
    let canal_iron = iron_builds.iter().filter(|x| x.0 == 0).count();
    let rail_iron = iron_builds.iter().filter(|x| x.0 == 1).count();
    println!("  canal-era iron builds: {canal_iron} (per game {:.2}), rail-era: {rail_iron} (per game {:.2})", canal_iron as f64 / games as f64, rail_iron as f64 / games as f64);
    println!("=== flips: coal {:.1}/game, iron {:.1}/game ===", coal_flip.len() as f64 / games as f64, iron_flip.len() as f64 / games as f64);
}
