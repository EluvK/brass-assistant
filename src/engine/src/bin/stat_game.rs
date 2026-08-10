//! Per-game diagnostic: per-player action counts split by era, plus the
//! exact flipped tiles at each era end. Usage: cargo run --release --bin stat_game -- <seed>
use brass_engine::data::Era;
use brass_engine::game_loop::{self, AfterEra, GameHooks, LoopOutcome};
use brass_engine::heuristic_ai;
use brass_engine::mcts_ai::{self, MctsConfig};
use brass_engine::random_ai;
use brass_engine::rules::Move;
use brass_engine::search_ai;
use brass_engine::state::GameState;
use rand::SeedableRng;

fn flipped_desc(state: &GameState) -> Vec<String> {
    let mut out = Vec::new();
    for (key, tile) in state.city_tiles.iter().enumerate() {
        if let Some(t) = tile {
            if t.flipped {
                let (loc, slot) = match brass_engine::state::loc_from_key(key) {
                    Some((l, s)) => (Some(l), s),
                    None => (None, 0),
                };
                out.push(format!(
                    "P{} {}Lv{}@{}槽{}",
                    t.player,
                    t.ind.name(),
                    t.def.level,
                    loc.map(|l| l.name()).unwrap_or("?"),
                    slot
                ));
            }
        }
    }
    for (fi, t) in state.farm_tiles.iter().enumerate() {
        if let Some(t) = t {
            if t.flipped {
                out.push(format!(
                    "P{} {}Lv{}@农场{}",
                    t.player, t.ind.name(), t.def.level, fi
                ));
            }
        }
    }
    out.sort();
    out
}

fn main() {
    let seed: u64 = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(7);
    let players: usize = std::env::args().nth(2).and_then(|s| s.parse().ok()).unwrap_or(4);
    let policy = std::env::args()
        .nth(3)
        .unwrap_or_else(|| "heuristic".to_string());
    let sims: usize = std::env::args().nth(4).and_then(|s| s.parse().ok()).unwrap_or(300);
    let rng = rand::rngs::StdRng::seed_from_u64(seed);
    let mut state = GameState::new(rng, players);
    // RNG for the random baseline (state.rng is setup-only).
    let mut rand_rng = rand::rngs::StdRng::from_entropy();

    #[derive(Default, Clone)]
    struct PStats {
        build: [u32; 6],
        network: u32,
        dbl_rail: u32,
        develop: u32,
        sell: u32,
        loan: u32,
        pass: u32,
        scout: u32,
    }
    let mut canal: Vec<PStats> = vec![PStats::default(); players];
    let mut rail: Vec<PStats> = vec![PStats::default(); players];

    let mut last_err: Option<String> = None;
    let mut on_move = |state: &mut GameState, mv: &Move| {
        let pid = state.current_player_id();
        let target = if state.era == Era::Canal {
            &mut canal[pid]
        } else {
            &mut rail[pid]
        };
        match mv {
            Move::Build { ind, .. } => target.build[*ind as usize] += 1,
            Move::Network { .. } => target.network += 1,
            Move::NetworkDouble { .. } => target.dbl_rail += 1,
            Move::Develop { .. } | Move::ResolveFreeDevelop { .. } => target.develop += 1,
            Move::Sell { .. } => target.sell += 1,
            Move::Loan { .. } => target.loan += 1,
            Move::Pass { .. } => target.pass += 1,
            Move::Scout { .. } => target.scout += 1,
        }
    };
    let mut on_after = |_state: &mut GameState, _mv: &Move, result: &Result<String, String>| {
        if let Err(e) = result {
            last_err = Some(e.clone());
        }
    };
    let mut on_era = |state: &mut GameState, era: Era| -> AfterEra {
        if era == Era::Canal {
            println!("\n===== 运河时代结束 =====");
            for pid in 0..players {
                let p = &state.players[pid];
                println!(
                    "  P{}: 收入{} 钱£{} VP{} 运河链接{}",
                    pid,
                    p.income_level(),
                    p.money,
                    p.vp,
                    p.canal_links
                );
            }
            println!("  翻面建筑: {}", flipped_desc(state).join(", "));
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
        Some(match policy.as_str() {
            "random" => random_ai::choose_random_move(state, &mut rand_rng)
                .unwrap_or_else(|| heuristic_ai::pass_decision(state).mv),
            "2ply" => search_ai::choose_action_2ply(state).mv,
            "mcts" => {
                let cfg = MctsConfig {
                    simulations: sims,
                    ..Default::default()
                };
                mcts_ai::choose_action_mcts(state, &cfg).mv
            }
            _ => heuristic_ai::choose_action(state).mv,
        })
    });
    if outcome == LoopOutcome::IllegalMove {
        println!("[!] 非法动作: {}", last_err.unwrap_or_default());
    }

    println!("\n===== 铁路时代结束 / 终局 =====");
    for pid in 0..players {
        let p = &state.players[pid];
        println!(
            "  P{}: 收入{} 钱£{} VP{} 铁路链接{}",
            pid,
            p.income_level(),
            p.money,
            p.vp,
            p.rail_links
        );
    }
    println!("  翻面建筑: {}", flipped_desc(&state).join(", "));

    println!("\n===== 各玩家动作汇总（运河时代 / 铁路时代）=====");
    // Must match IndustryType discriminant order:
    // Cotton, Coal, Iron, Manufacturer, Pottery, Brewery.
    let ind_names = ["棉", "煤", "铁", "制造", "陶", "酒"];
    for pid in 0..players {
        let c = &canal[pid];
        let r = &rail[pid];
        let fmt = |s: &PStats, era: &str| {
            let builds: Vec<String> = (0..6)
                .map(|i| format!("{}x{}", ind_names[i], s.build[i]))
                .filter(|x| !x.ends_with('0'))
                .collect();
            format!(
                "{era}: 建[{}] 网{} 双铁{} 研发{} 卖{} 贷{} 过{} 斥{}",
                builds.join(" "),
                s.network,
                s.dbl_rail,
                s.develop,
                s.sell,
                s.loan,
                s.pass,
                s.scout
            )
        };
        println!("P{} {}", pid, fmt(c, "运河"));
        println!("   {}", fmt(r, "铁路"));
    }

    // Market check at end
    println!(
        "\n终局资源: 煤市{}/14 铁市{}/10 | 牌堆{}",
        state.coal_market, state.iron_market, state.deck.len()
    );
}
