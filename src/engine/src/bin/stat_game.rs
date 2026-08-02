//! Per-game diagnostic: per-player action counts split by era, plus the
//! exact flipped tiles at each era end. Usage: cargo run --release --bin stat_game -- <seed>
use brass_engine::data::Era;
use brass_engine::engine::{advance_turn, end_canal_era, end_game, TurnResult};
use brass_engine::heuristic_ai;
use brass_engine::map::{city_slots, Loc};
use brass_engine::rules::Move;
use brass_engine::state::{city_slot_offsets, GameState};
use rand::SeedableRng;

fn key_loc(key: usize) -> (Option<Loc>, usize) {
    let offsets = city_slot_offsets();
    for (i, loc) in brass_engine::map::ALL_LOCATIONS
        .iter()
        .take(brass_engine::map::CITY_COUNT)
        .enumerate()
    {
        let off = offsets[i];
        let n = city_slots(*loc).len();
        if key >= off && key < off + n {
            return (Some(*loc), key - off);
        }
    }
    (None, 0)
}

fn flipped_desc(state: &GameState<impl rand::Rng>) -> Vec<String> {
    let mut out = Vec::new();
    for (key, tile) in state.city_tiles.iter().enumerate() {
        if let Some(t) = tile {
            if t.flipped {
                let (loc, slot) = key_loc(key);
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
    let rng = rand::rngs::StdRng::seed_from_u64(seed);
    let mut state = GameState::new(rng, players);

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

    let mut guard = 0;
    while !state.game_over && guard < 200_000 {
        guard += 1;
        let pid = state.current_player_id();
        let target = if state.era == Era::Canal { &mut canal[pid] } else { &mut rail[pid] };
        let mv = heuristic_ai::choose_action(&mut state).mv;
        match &mv {
            Move::Build { ind, .. } => target.build[*ind as usize] += 1,
            Move::Network { .. } => target.network += 1,
            Move::NetworkDouble { .. } => target.dbl_rail += 1,
            Move::Develop { .. } | Move::ResolveFreeDevelop { .. } => target.develop += 1,
            Move::Sell { .. } => target.sell += 1,
            Move::Loan { .. } => target.loan += 1,
            Move::Pass { .. } => target.pass += 1,
            Move::Scout { .. } => target.scout += 1,
        }
        if let Err(err) = brass_engine::rules::apply_move(&mut state, &mv) {
            println!("[!] 非法动作: {err}");
            break;
        }
        match advance_turn(&mut state) {
            TurnResult::Continue => {}
            TurnResult::EndCanalEra => {
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
                println!("  翻面建筑: {}", flipped_desc(&state).join(", "));
                end_canal_era(&mut state);
            }
            TurnResult::EndGame => end_game(&mut state),
        }
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
    let ind_names = [
        "煤", "铁", "棉", "制造", "陶", "酒",
    ];
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
