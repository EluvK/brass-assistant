//! Debug: print score composition for resource builds in a live game.
//! Usage: cargo run --release --bin dbg_build -- <seed> <rounds>
use brass_engine::engine::{advance_turn, end_canal_era, end_game, TurnResult};
use brass_engine::heuristic_ai;
use brass_engine::rules::{apply_move, get_valid_build_targets, Move};
use brass_engine::state::GameState;
use rand::SeedableRng;

fn main() {
    let seed: u64 = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(7);
    let rounds: u32 = std::env::args().nth(2).and_then(|s| s.parse().ok()).unwrap_or(4);
    let rng = rand::rngs::StdRng::seed_from_u64(seed);
    let mut state = GameState::new(rng, 4);
    let mut guard = 0;
    while !state.game_over && guard < 200_000 && state.round <= rounds {
        guard += 1;
        let pid = state.current_player_id();
        // Print resource builds' scores when it's P2's turn, every action.
        if pid == 2 {
            let targets = get_valid_build_targets(&state, pid);
            for t in targets.iter().filter(|t| {
                matches!(t.ind, brass_engine::data::IndustryType::CoalMine | brass_engine::data::IndustryType::IronWorks)
            }) {
                let tile = state.players[pid].next_tile(t.ind).unwrap();
                let flip = heuristic_ai::debug_flip(&state, pid, t.ind, t.loc);
                let ma = heuristic_ai::debug_market_adjust(&state, t.ind, t.loc, tile.resource_cubes);
                println!(
                    "R{} era={:?} P{} {} Lv{} @{} 煤市{}铁市{} cubes={} flip={:.2} market_adjust={:.2}",
                    state.round,
                    state.era,
                    pid,
                    t.ind.name(),
                    tile.level,
                    t.loc.name(),
                    state.coal_market,
                    state.iron_market,
                    tile.resource_cubes,
                    flip,
                    ma
                );
            }
        }
        let mv = heuristic_ai::choose_action(&mut state).mv;
        if let Move::Build { ind, .. } = &mv {
            println!("  >> P{} chose build {}", pid, ind.name());
        }
        let _ = apply_move(&mut state, &mv);
        match advance_turn(&mut state) {
            TurnResult::Continue => {}
            TurnResult::EndCanalEra => end_canal_era(&mut state),
            TurnResult::EndGame => end_game(&mut state),
        }
    }
}
