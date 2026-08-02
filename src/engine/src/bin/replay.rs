//! Replay a single heuristic-AI game with a detailed, human-readable log.
//!
//! Usage: cargo run --release --bin replay -- <seed> <players>

use brass_engine::data::{Era, IndustryType};
use brass_engine::engine::{advance_turn, end_canal_era, end_game, TurnResult};
use brass_engine::heuristic_ai;
use brass_engine::map::{city_slots, connections, ALL_LOCATIONS, Loc, CITY_COUNT};
use brass_engine::rules::Move;
use brass_engine::scoring;
use brass_engine::search_ai;
use brass_engine::state::{city_slot_offsets, GameState};
use rand::SeedableRng;
use std::env;

fn hand_display(state: &GameState<impl rand::Rng>, pid: usize) -> String {
    let hand = &state.players[pid].hand;
    let mut parts = Vec::new();
    for (i, c) in hand.iter().enumerate() {
        let tag = match c.ctype() {
            brass_engine::data::CardType::Location => "loc",
            brass_engine::data::CardType::Industry => "ind",
            brass_engine::data::CardType::WildLocation => "wildL",
            brass_engine::data::CardType::WildIndustry => "wildI",
        };
        parts.push(format!("{i}:{tag}({})", c.name()));
    }
    parts.join(" ")
}

fn move_detail(state: &GameState<impl rand::Rng>, mv: &Move) -> String {
    let pid = state.current_player_id();
    let card = state.players[pid].hand.get(mv_card_index(mv));
    let card_name = card.map(|c| c.name()).unwrap_or_else(|| "?".to_string());
    match mv {
        Move::Build { loc, slot_index, ind, .. } => {
            let cost = crate_cost(state, pid, *ind, *loc);
            format!(
                "建厂 {} Lv{} @ {}(槽{})，用牌[{}]，花费£{}",
                ind.name(),
                state.players[pid].next_tile(*ind).map(|t| t.level).unwrap_or(0),
                loc.name(),
                slot_index,
                card_name,
                cost
            )
        }
        Move::Network { conn_id, .. } => {
            let c = &connections()[*conn_id];
            format!(
                "建网 {}↔{}（{}时代），用牌[{}]",
                c.a.name(),
                c.b.name(),
                if state.era == Era::Canal { "运河" } else { "铁路" },
                card_name
            )
        }
        Move::NetworkDouble { conn1, conn2, .. } => {
            let c1 = &connections()[*conn1];
            let c2 = &connections()[*conn2];
            format!(
                "双铁路 {}↔{} + {}↔{}，用牌[{}]",
                c1.a.name(),
                c1.b.name(),
                c2.a.name(),
                c2.b.name(),
                card_name
            )
        }
        Move::Develop { ind1, ind2, .. } => {
            let s = format!("研发 {}", ind1.name());
            let s = if let Some(i2) = ind2 {
                format!("{s} + {}", i2.name())
            } else {
                s
            };
            format!("{s}，用牌[{}]", card_name)
        }
        Move::Sell { keys, use_merchant_beer, .. } => {
            let mut parts = Vec::new();
            for (i, k) in keys.iter().enumerate() {
                let (loc, slot) = loc_from_key(state, *k);
                let ind = state.city_tiles[*k].as_ref().map(|t| t.ind);
                let beer = use_merchant_beer.get(i).copied().unwrap_or(false);
                let loc_n = loc.map(|l| l.name()).unwrap_or("?");
                let ind_n = ind.map(|i| i.name()).unwrap_or("?");
                let _ = slot;
                parts.push(format!(
                    "{}({})@{}",
                    ind_n,
                    if beer { "喝商家酒" } else { "自酿酒" },
                    loc_n
                ));
            }
            format!("卖货 [{}]，用牌[{}]", parts.join(" + "), card_name)
        }
        Move::Loan { .. } => format!("贷款 £30（收入-3级），用牌[{}]", card_name),
        Move::Scout { card_indices } => {
            let names: Vec<String> = card_indices
                .iter()
                .map(|&i| {
                    state.players[pid]
                        .hand
                        .get(i)
                        .map(|c| c.name().to_string())
                        .unwrap_or_else(|| "?".to_string())
                })
                .collect();
            format!("斥候：弃[{}]换2张万用牌", names.join(", "))
        }
        Move::Pass { .. } => format!("过牌（弃[{}]）", card_name),
    }
}

fn mv_card_index(mv: &Move) -> usize {
    match mv {
        Move::Build { card_index, .. } => *card_index,
        Move::Network { card_index, .. } => *card_index,
        Move::NetworkDouble { card_index, .. } => *card_index,
        Move::Develop { card_index, .. } => *card_index,
        Move::Sell { card_index, .. } => *card_index,
        Move::Loan { card_index } => *card_index,
        Move::Pass { card_index } => *card_index,
        Move::Scout { .. } => 0,
    }
}

fn crate_cost(
    state: &GameState<impl rand::Rng>,
    pid: usize,
    ind: IndustryType,
    loc: Loc,
) -> String {
    match brass_engine::rules::calculate_build_cost(state, pid, ind, loc) {
        Some(c) => format!("{}(+煤{}铁{})", c.cost_total, c.coal_cost, c.iron_cost),
        None => "?".to_string(),
    }
}

fn loc_from_key(_state: &GameState<impl rand::Rng>, key: usize) -> (Option<Loc>, usize) {
    let offsets = city_slot_offsets();
    for (i, loc) in ALL_LOCATIONS[..CITY_COUNT].iter().enumerate() {
        let off = offsets[i];
        let n = city_slots(*loc).len();
        if key >= off && key < off + n {
            return (Some(*loc), key - off);
        }
    }
    (None, 0)
}

fn merchant_state(state: &GameState<impl rand::Rng>) -> String {
    state
        .merchants
        .iter()
        .map(|mt| {
            let buys = match mt.buys {
                brass_engine::state::BuyType::Blank => "空".to_string(),
                brass_engine::state::BuyType::Any => "任意".to_string(),
                brass_engine::state::BuyType::Industry(t) => t.name().to_string(),
            };
            format!("{}[{}]{}", mt.loc.name(), buys, if mt.has_beer { "·桶" } else { "" })
        })
        .collect::<Vec<_>>()
        .join("  ")
}

fn player_state(state: &GameState<impl rand::Rng>, pid: usize) -> String {
    let p = &state.players[pid];
    format!(
        "钱£{:<3} 收入{:>2} 手牌{} VP{} 运河{} 铁路{}",
        p.money,
        p.income_level(),
        p.hand.len(),
        p.vp,
        p.canal_links,
        p.rail_links
    )
}

fn board_state(state: &GameState<impl rand::Rng>) -> String {
    let built = state.city_tiles.iter().flatten().count()
        + state.farm_tiles.iter().flatten().count();
    let flipped = state.city_tiles.iter().flatten().filter(|t| t.flipped).count()
        + state.farm_tiles.iter().flatten().filter(|t| t.flipped).count();
    let links = state.links.iter().flatten().count();
    format!(
        "煤市{}/{} 铁市{}/{} | 板块{}(翻{}) 连接{} 牌堆{}",
        state.coal_market,
        14,
        state.iron_market,
        10,
        built,
        flipped,
        links,
        state.deck.len()
    )
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let seed: u64 = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(7);
    let players: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(4);
    // policy: "heuristic" (default) | "2ply"
    let policy = args
        .get(3)
        .cloned()
        .unwrap_or_else(|| "heuristic".to_string());

    let rng = rand::rngs::StdRng::seed_from_u64(seed);
    let mut state = GameState::new(rng, players);

    let mut prev_round = state.round;
    let mut prev_era = state.era;

    println!(
        "======================================================================"
    );
    println!(
        "Replay: {players}玩家 {policy}AI, seed={seed} | 起始顺位: {:?}",
        state.turn_order
    );
    println!(
        "======================================================================"
    );
    for pid in 0..players {
        println!(
            "开局 玩家{pid}: {} | 手牌: {}",
            player_state(&state, pid),
            hand_display(&state, pid)
        );
    }
    println!("商家: {}", merchant_state(&state));
    println!("{}", board_state(&state));

    let mut guard = 0;
    loop {
        guard += 1;
        if guard > 200_000 {
            println!("[!] guard exceeded, aborting");
            break;
        }
        if state.game_over {
            break;
        }

        // Era / round header change
        if state.era != prev_era {
            println!(
                "\n------ 时代切换: {} -> {} ------",
                prev_era, state.era
            );
            prev_era = state.era;
        }
        if state.round != prev_round {
            println!(
                "\n===== {}时代 第{}轮 | 顺位: {:?} =====",
                state.era, state.round, state.turn_order
            );
            prev_round = state.round;
        }

        let pid = state.current_player_id();
        let mv = match policy.as_str() {
            "2ply" => search_ai::choose_action_2ply(&mut state).mv,
            "mcts" => {
                use brass_engine::mcts_ai::{self, MctsConfig};
                let cfg = MctsConfig {
                    simulations: 300,
                    ..Default::default()
                };
                mcts_ai::choose_action_mcts(&mut state, &cfg).mv
            }
            _ => heuristic_ai::choose_action(&mut state).mv,
        };
        let detail = move_detail(&state, &mv);
        let before = player_state(&state, pid);
        let res = brass_engine::rules::apply_move(&mut state, &mv);
        let after = player_state(&state, pid);

        let status = match &res {
            Ok(msg) => msg.clone(),
            Err(e) => format!("!!失败: {e}"),
        };
        println!("玩家{pid} [{detail}]");
        println!("    之前: {before}");
        println!("    结果: {status}");
        println!("    之后: {after}");

        if res.is_ok() {
            match advance_turn(&mut state) {
                TurnResult::Continue => {}
                TurnResult::EndCanalEra => {
                    end_canal_era(&mut state);
                    println!(
                        "\n--- 运河时代结束，进入铁路时代（连接/1级板块已清除，重新洗牌发牌） ---"
                    );
                }
                TurnResult::EndGame => {
                    end_game(&mut state);
                }
            }
        } else {
            println!("[!] 非法动作，终止回放以避免污染轮次/手牌状态");
            break;
        }
    }

    if !state.game_over {
        end_game(&mut state);
    }

    println!("\n======================================================================");
    println!("终局结算");
    println!("======================================================================");
    for pid in 0..players {
        println!("玩家{pid}: {}", player_state(&state, pid));
    }
    let ranking = scoring::final_ranking(&state);
    println!("排名: {:?}", ranking);
    if let Some(&w) = ranking.first() {
        println!("胜者: 玩家{w}");
    }
}
