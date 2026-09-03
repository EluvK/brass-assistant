use _engine::data::IndustryType;
use _engine::game_loop::{self, GameHooks};
use _engine::state::GameState;
use rand_chacha::rand_core::SeedableRng;
use rayon::prelude::*;
use std::cell::RefCell;

#[derive(Default)]
struct R {
    seed: u64,
    built_north: bool,
    built_south: bool,
    pre_cleanup: bool,
    final_present: bool,
    owner: i32,
    level: i32,
    flipped: bool,
    illegal: bool,
    stuck: bool,
}

fn one(seed: u64) -> R {
    let mut state = GameState::new(rand_chacha::ChaCha12Rng::seed_from_u64(seed), 4);
    let out = RefCell::new(R {
        seed,
        owner: -1,
        ..Default::default()
    });
    let mut before = |_: &mut GameState, mv: &_engine::rules::ResolvedMove| {
        if let _engine::rules::ResolvedMove::Build { loc, ind, .. } = mv {
            if *loc == _engine::map::Loc::BreweryNorth && *ind == IndustryType::Brewery {
                out.borrow_mut().built_north = true;
            }
            if *loc == _engine::map::Loc::BrewerySouth && *ind == IndustryType::Brewery {
                out.borrow_mut().built_south = true;
            }
        }
    };
    let mut on_era = |s: &mut GameState, era: _engine::data::Era| -> _engine::game_loop::AfterEra {
        if era == _engine::data::Era::Canal {
            out.borrow_mut().pre_cleanup = s.farm_tile(_engine::map::Loc::BreweryNorth).is_some();
        }
        _engine::game_loop::AfterEra::Continue
    };
    let hooks = GameHooks {
        before_move: Some(&mut before),
        on_era: Some(&mut on_era),
        ..Default::default()
    };
    let outcome = game_loop::play(&mut state, 200_000, hooks, |s| {
        Some(_engine::heuristic_ai::choose_action(s).mv)
    });
    if !state.game_over {
        game_loop::finish_game(&mut state);
    }
    let mut r = out.into_inner();
    if let Some(t) = state.farm_tile(_engine::map::Loc::BreweryNorth) {
        r.final_present = true;
        r.owner = t.player as i32;
        r.level = t.def.level as i32;
        r.flipped = t.flipped;
    }
    r.illegal = matches!(outcome, _engine::game_loop::LoopOutcome::IllegalMove);
    r.stuck = !matches!(
        outcome,
        _engine::game_loop::LoopOutcome::GameOver
            | _engine::game_loop::LoopOutcome::StoppedByEraEnd
    );
    r
}

fn main() {
    let start: u64 = std::env::args()
        .nth(1)
        .and_then(|x| x.parse().ok())
        .unwrap_or(0);
    let end: u64 = std::env::args()
        .nth(2)
        .and_then(|x| x.parse().ok())
        .unwrap_or(500);
    let start_ts = std::time::Instant::now();
    let rs: Vec<R> = (start..end).into_par_iter().map(one).collect();
    println!(
        "seed,built_north,built_south,pre_cleanup,final_present,owner,level,flipped,illegal,stuck"
    );
    for r in &rs {
        println!(
            "{},{},{},{},{},{},{},{},{},{}",
            r.seed,
            r.built_north,
            r.built_south,
            r.pre_cleanup,
            r.final_present,
            r.owner,
            r.level,
            r.flipped,
            r.illegal,
            r.stuck
        );
    }
    eprintln!(
        "games={} built_north={} built_south={} pre_cleanup={} final_present={} illegal={} stuck={}",
        rs.len(),
        rs.iter().filter(|r| r.built_north).count(),
        rs.iter().filter(|r| r.built_south).count(),
        rs.iter().filter(|r| r.pre_cleanup).count(),
        rs.iter().filter(|r| r.final_present).count(),
        rs.iter().filter(|r| r.illegal).count(),
        rs.iter().filter(|r| r.stuck).count()
    );
    let duration = start_ts.elapsed();
    eprintln!("elapsed_time={:?}", duration);
}
