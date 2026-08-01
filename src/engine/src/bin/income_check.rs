//! Measure canal-era-end income level distribution across games.
//! Usage: cargo run --release --bin income_check -- <games> <players>
use brass_engine::engine::{advance_turn, end_canal_era, end_game, TurnResult};
use brass_engine::heuristic_ai;
use brass_engine::rules::apply_move;
use brass_engine::state::GameState;
use rand::SeedableRng;

fn main() {
    let games: u64 = std::env::args().nth(1).unwrap_or("200".into()).parse().unwrap();
    let players: usize = std::env::args().nth(2).unwrap_or("4".into()).parse().unwrap();
    let mut canal_income: Vec<Vec<i32>> = vec![Vec::new(); players];
    let mut canal_loans = [0u64; 5];
    let mut canal_develop = [0u64; 5];
    let mut rail_loans = [0u64; 5];
    let mut rail_develop = [0u64; 5];

    for g in 0..games {
        let rng = rand::rngs::StdRng::seed_from_u64(g);
        let mut state = GameState::new(rng, players);
        let mut guard = 0;
        while !state.game_over && guard < 200_000 {
            guard += 1;
            let pid = state.current_player_id();
            let era_loans = if state.era == brass_engine::data::Era::Canal { &mut canal_loans } else { &mut rail_loans };
            let era_dev = if state.era == brass_engine::data::Era::Canal { &mut canal_develop } else { &mut rail_develop };
            let mv = heuristic_ai::choose_action(&mut state).mv;
            match &mv {
                brass_engine::rules::Move::Loan { .. } => era_loans[pid.min(4)] += 1,
                brass_engine::rules::Move::Develop { .. } => era_dev[pid.min(4)] += 1,
                _ => {}
            }
            let _ = apply_move(&mut state, &mv);
            match advance_turn(&mut state) {
                TurnResult::Continue => {}
                TurnResult::EndCanalEra => {
                    for p in 0..players {
                        canal_income[p].push(state.players[p].income_level() as i32);
                    }
                    end_canal_era(&mut state);
                }
                TurnResult::EndGame => end_game(&mut state),
            }
        }
        if !state.game_over { end_game(&mut state); }
    }

    println!("=== Canal-era-end income levels (each player, {} games) ===", games);
    for p in 0..players {
        let v = &canal_income[p];
        let avg = v.iter().sum::<i32>() as f64 / v.len().max(1) as f64;
        let neg = v.iter().filter(|&&x| x < 0).count();
        let low = v.iter().filter(|&&x| x < 6).count();
        let high = v.iter().filter(|&&x| x >= 6).count();
        println!("  P{p}: avg={avg:.1} min={} max={} | <0: {neg}  <6: {low}  >=6: {high} ({})", v.iter().min().unwrap_or(&0), v.iter().max().unwrap_or(&0), v.len());
    }
    println!("=== Loans/player (canal) ===");
    for p in 0..players.min(5) {
        println!("  P{p}: {} (avg {:.1})", canal_loans[p], canal_loans[p] as f64 / games as f64);
    }
    println!("=== Develops/player (canal) ===");
    for p in 0..players.min(5) {
        println!("  P{p}: {} (avg {:.1})", canal_develop[p], canal_develop[p] as f64 / games as f64);
    }
    println!("=== Loans/player (rail) ===");
    for p in 0..players.min(5) {
        println!("  P{p}: {} (avg {:.1})", rail_loans[p], rail_loans[p] as f64 / games as f64);
    }
    println!("=== Develops/player (rail) ===");
    for p in 0..players.min(5) {
        println!("  P{p}: {} (avg {:.1})", rail_develop[p], rail_develop[p] as f64 / games as f64);
    }
}
