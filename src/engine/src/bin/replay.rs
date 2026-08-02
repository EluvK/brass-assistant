//! Replay a single heuristic-AI game with a detailed, human-readable log.
//!
//! Usage: cargo run --release --bin replay -- <seed> <players>

use brass_engine::data::{Era, IndustryType};
use brass_engine::engine::{advance_turn, end_canal_era, end_game, TurnResult};
use brass_engine::heuristic_ai;
use brass_engine::map::{city_slots, connections, ALL_LOCATIONS, Loc, CITY_COUNT};
use brass_engine::random_ai;
use brass_engine::rules::Move;
use brass_engine::scoring;
use brass_engine::search_ai;
use brass_engine::state::{city_slot_offsets, GameState};
use rand::SeedableRng;
use std::env;

fn era_label(era: Era) -> &'static str {
    match era {
        Era::Canal => "运河",
        Era::Rail => "铁路",
    }
}

fn loc_label(loc: Loc) -> String {
    format!("{}({})", loc.zh_name(), loc.name())
}

fn industry_label(ind: IndustryType) -> &'static str {
    match ind {
        IndustryType::CottonMill => "棉纺厂(Cotton Mill)",
        IndustryType::CoalMine => "煤矿(Coal Mine)",
        IndustryType::IronWorks => "铁厂(Iron Works)",
        IndustryType::Manufacturer => "制造厂(Manufacturer)",
        IndustryType::Pottery => "陶器厂(Pottery)",
        IndustryType::Brewery => "啤酒厂(Brewery)",
    }
}

fn card_label(card: &brass_engine::state::Card) -> String {
    match card {
        brass_engine::state::Card::Location(loc) => loc_label(*loc),
        brass_engine::state::Card::Industry { industries, n } => {
            if *n == 2 {
                format!(
                    "{} / {}",
                    industry_label(industries[0]),
                    industry_label(industries[1])
                )
            } else {
                industry_label(industries[0]).to_string()
            }
        }
        brass_engine::state::Card::WildLocation => "万能地点牌(Wild Location)".to_string(),
        brass_engine::state::Card::WildIndustry => "万能产业牌(Wild Industry)".to_string(),
    }
}

fn hand_display(state: &GameState<impl rand::Rng>, pid: usize) -> String {
    let hand = &state.players[pid].hand;
    let mut parts = Vec::new();
    for (i, c) in hand.iter().enumerate() {
        let tag = match c.ctype() {
            brass_engine::data::CardType::Location => "地",
            brass_engine::data::CardType::Industry => "产",
            brass_engine::data::CardType::WildLocation => "万地",
            brass_engine::data::CardType::WildIndustry => "万产",
        };
        parts.push(format!("{i}:{tag}({})", card_label(c)));
    }
    parts.join(" ")
}

fn move_detail(state: &GameState<impl rand::Rng>, mv: &Move) -> String {
    let pid = state.current_player_id();
    let card = state.players[pid].hand.get(mv_card_index(mv));
    let card_name = card.map(card_label).unwrap_or_else(|| "?".to_string());
    match mv {
        Move::Build { loc, slot_index, ind, .. } => {
            let cost = crate_cost(state, pid, *ind, *loc);
            format!(
                "建厂 {} Lv{} @ {}(槽{})，用牌[{}]，花费£{}",
                industry_label(*ind),
                state.players[pid].next_tile(*ind).map(|t| t.level).unwrap_or(0),
                loc_label(*loc),
                slot_index + 1,
                card_name,
                cost
            )
        }
        Move::Network { conn_id, .. } => {
            let c = &connections()[*conn_id];
            format!(
                "建网 {}↔{}（{}时代），用牌[{}]",
                loc_label(c.a),
                loc_label(c.b),
                era_label(state.era),
                card_name
            )
        }
        Move::NetworkDouble { conn1, conn2, .. } => {
            let c1 = &connections()[*conn1];
            let c2 = &connections()[*conn2];
            format!(
                "双铁路 {}↔{} + {}↔{}，用牌[{}]",
                loc_label(c1.a),
                loc_label(c1.b),
                loc_label(c2.a),
                loc_label(c2.b),
                card_name
            )
        }
        Move::Develop { ind1, ind2, .. } => {
            let lv1 = state.players[pid]
                .next_tile(*ind1)
                .map(|t| t.level)
                .unwrap_or(0);
            let s = format!("研发 {} Lv{}", industry_label(*ind1), lv1);
            let s = if let Some(i2) = ind2 {
                let lv2 = state.players[pid]
                    .next_tile(*i2)
                    .map(|t| t.level)
                    .unwrap_or(0);
                format!("{s} + {} Lv{}", industry_label(*i2), lv2)
            } else {
                s
            };
            format!("{s}，用牌[{}]", card_name)
        }
        Move::ResolveFreeDevelop { ind1, ind2 } => {
            let lv1 = state.players[pid]
                .next_tile(*ind1)
                .map(|t| t.level)
                .unwrap_or(0);
            let s = format!("结算免费研发 {} Lv{}", industry_label(*ind1), lv1);
            if let Some(i2) = ind2 {
                let lv2 = state.players[pid]
                    .next_tile(*i2)
                    .map(|t| t.level)
                    .unwrap_or(0);
                format!("{s} + {} Lv{}", industry_label(*i2), lv2)
            } else {
                s
            }
        }
        Move::Sell { keys, merchant_indices, use_merchant_beer, .. } => {
            let mut parts = Vec::new();
            for (i, k) in keys.iter().enumerate() {
                let (loc, slot) = loc_from_key(state, *k);
                let ind = state.city_tiles[*k].as_ref().map(|t| t.ind);
                let merchant = merchant_indices
                    .get(i)
                    .and_then(|&mi| state.merchants.get(mi))
                    .map(|mt| loc_label(mt.loc))
                    .unwrap_or_else(|| "?".to_string());
                let beer = use_merchant_beer.get(i).copied().unwrap_or(false);
                let loc_n = loc.map(loc_label).unwrap_or_else(|| "?".to_string());
                let ind_n = ind
                    .map(|i| industry_label(i).to_string())
                    .unwrap_or_else(|| "?".to_string());
                parts.push(format!(
                    "{}({})@{}(槽{})=>{}",
                    ind_n,
                    if beer { "喝商家酒" } else { "自酿酒" },
                    loc_n,
                    slot + 1,
                    merchant
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
                        .map(card_label)
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
        Move::ResolveFreeDevelop { .. } => 0,
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
                brass_engine::state::BuyType::Industry(t) => industry_label(t).to_string(),
            };
            format!("{}[{}]{}", loc_label(mt.loc), buys, if mt.has_beer { "·桶" } else { "" })
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

fn tile_label(loc: Loc, slot_index: usize, tile: &brass_engine::state::BoardTile) -> String {
    format!(
        "{} Lv{} @ {}(槽{})",
        industry_label(tile.ind),
        tile.def.level,
        loc_label(loc),
        slot_index + 1
    )
}

fn era_score_detail(state: &GameState<impl rand::Rng>, pid: usize) -> String {
    let mut flipped_tiles = Vec::new();
    let mut industry_vp = 0u16;
    for (i, loc) in ALL_LOCATIONS[..CITY_COUNT].iter().enumerate() {
        for slot_index in 0..city_slots(*loc).len() {
            let key = city_slot_offsets()[i] + slot_index;
            if let Some(tile) = state.city_tiles[key].as_ref() {
                if tile.player == pid && tile.flipped {
                    industry_vp += tile.def.vp as u16;
                    flipped_tiles.push(format!("{} => {}VP", tile_label(*loc, slot_index, tile), tile.def.vp));
                }
            }
        }
    }
    for farm_loc in [Loc::BreweryNorth, Loc::BrewerySouth] {
        if let Some(tile) = state.farm_tile(farm_loc) {
            if tile.player == pid && tile.flipped {
                industry_vp += tile.def.vp as u16;
                flipped_tiles.push(format!("{} Lv{} @ {} => {}VP", industry_label(tile.ind), tile.def.level, loc_label(farm_loc), tile.def.vp));
            }
        }
    }

    let mut link_parts = Vec::new();
    let mut link_vp = 0u16;
    for (id, link) in state.links.iter().enumerate() {
        let Some(link) = link else { continue };
        if link.player != pid {
            continue;
        }
        let conn = &connections()[id];
        let mut value = 0u16;
        let mut sources = Vec::new();
        for end in [conn.a, conn.b] {
            if end.is_city() {
                for slot in 0..city_slots(end).len() {
                    if let Some(tile) = state.tile_at(end, slot) {
                        if tile.flipped {
                            value += tile.def.link_vp as u16;
                            sources.push(format!("{}:{}", loc_label(end), tile.def.link_vp));
                        }
                    }
                }
            }
            if end.is_merchant() {
                value += 2;
                sources.push(format!("{}:2", loc_label(end)));
            }
            if end.is_farm() {
                if let Some(tile) = state.farm_tile(end) {
                    if tile.flipped {
                        value += tile.def.link_vp as u16;
                        sources.push(format!("{}:{}", loc_label(end), tile.def.link_vp));
                    }
                }
            }
        }
        if let Some(via) = conn.via_farm {
            if let Some(tile) = state.farm_tile(via) {
                if tile.flipped {
                    value += tile.def.link_vp as u16;
                    sources.push(format!("途经{}:{}", loc_label(via), tile.def.link_vp));
                }
            }
        }
        link_vp += value;
        link_parts.push(format!("{}↔{} => {}VP [{}]", loc_label(conn.a), loc_label(conn.b), value, sources.join(", ")));
    }

    format!(
        "连接{}VP{} | 翻面板块{}VP{}",
        link_vp,
        if link_parts.is_empty() {
            String::new()
        } else {
            format!(" => {}", link_parts.join(" ; "))
        },
        industry_vp,
        if flipped_tiles.is_empty() {
            String::new()
        } else {
            format!(" => {}", flipped_tiles.join(" ; "))
        }
    )
}

fn canal_cleanup_detail(state: &GameState<impl rand::Rng>) -> String {
    let mut removed_links = Vec::new();
    for (id, link) in state.links.iter().enumerate() {
        let Some(link) = link else { continue };
        if link.is_canal {
            let conn = &connections()[id];
            removed_links.push(format!("玩家{}:{}↔{}", link.player, loc_label(conn.a), loc_label(conn.b)));
        }
    }

    let mut removed_tiles = Vec::new();
    for loc in ALL_LOCATIONS[..CITY_COUNT].iter() {
        for slot_index in 0..city_slots(*loc).len() {
            if let Some(tile) = state.tile_at(*loc, slot_index) {
                if !tile.def.rail_era {
                    removed_tiles.push(format!("玩家{}:{}", tile.player, tile_label(*loc, slot_index, tile)));
                }
            }
        }
    }
    for farm_loc in [Loc::BreweryNorth, Loc::BrewerySouth] {
        if let Some(tile) = state.farm_tile(farm_loc) {
            if !tile.def.rail_era {
                removed_tiles.push(format!(
                    "玩家{}:{} Lv{} @ {}",
                    tile.player,
                    industry_label(tile.ind),
                    tile.def.level,
                    loc_label(farm_loc)
                ));
            }
        }
    }

    format!(
        "清除连接: {}\n清除板块: {}",
        if removed_links.is_empty() {
            "无".to_string()
        } else {
            removed_links.join(" ; ")
        },
        if removed_tiles.is_empty() {
            "无".to_string()
        } else {
            removed_tiles.join(" ; ")
        }
    )
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let seed: u64 = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(7);
    let players: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(4);
    // policy: "heuristic" (default) | "2ply" | "mcts" | "random"
    //       | "mcts-vs-heur" | "mcts-vs-random" | "mcts-vs-2ply"
    let policy = args
        .get(3)
        .cloned()
        .unwrap_or_else(|| "heuristic".to_string());
    let sims: usize = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(300);

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
                era_label(prev_era),
                era_label(state.era)
            );
            prev_era = state.era;
        }
        if state.round != prev_round {
            println!(
                "\n===== {}时代 第{}轮 | 顺位: {:?} =====",
                era_label(state.era),
                state.round,
                state.turn_order
            );
            prev_round = state.round;
        }

        let pid = state.current_player_id();
        let mv = match policy.as_str() {
            "random" => random_ai::choose_random_move(&mut state)
                .unwrap_or_else(|| heuristic_ai::pass_decision(&state).mv),
            "mcts-vs-random" => {
                let mcts_seat = (seed as usize) % players;
                if pid == mcts_seat {
                    use brass_engine::mcts_ai::{self, MctsConfig};
                    let cfg = MctsConfig {
                        simulations: sims,
                        ..Default::default()
                    };
                    mcts_ai::choose_action_mcts(&mut state, &cfg).mv
                } else {
                    random_ai::choose_random_move(&mut state)
                        .unwrap_or_else(|| heuristic_ai::pass_decision(&state).mv)
                }
            }
            "mcts-vs-heur" => {
                let mcts_seat = (seed as usize) % players;
                if pid == mcts_seat {
                    use brass_engine::mcts_ai::{self, MctsConfig};
                    let cfg = MctsConfig {
                        simulations: sims,
                        ..Default::default()
                    };
                    mcts_ai::choose_action_mcts(&mut state, &cfg).mv
                } else {
                    heuristic_ai::choose_action(&mut state).mv
                }
            }
            "mcts-vs-2ply" => {
                let mcts_seat = (seed as usize) % players;
                if pid == mcts_seat {
                    use brass_engine::mcts_ai::{self, MctsConfig};
                    let cfg = MctsConfig {
                        simulations: sims,
                        ..Default::default()
                    };
                    mcts_ai::choose_action_mcts(&mut state, &cfg).mv
                } else {
                    search_ai::choose_action_2ply(&mut state).mv
                }
            }
            "2ply" => search_ai::choose_action_2ply(&mut state).mv,
            "mcts" => {
                use brass_engine::mcts_ai::{self, MctsConfig};
                let cfg = MctsConfig {
                    simulations: sims,
                    ..Default::default()
                };
                mcts_ai::choose_action_mcts(&mut state, &cfg).mv
            }
            _ => heuristic_ai::choose_action(&mut state).mv,
        };
        let detail = move_detail(&state, &mv);
        let before = player_state(&state, pid);
        let hand_before = hand_display(&state, pid);
        let res = brass_engine::rules::apply_move(&mut state, &mv);
        let after = player_state(&state, pid);
        let hand_after = hand_display(&state, pid);

        let status = match &res {
            Ok(msg) => msg.clone(),
            Err(e) => format!("!!失败: {e}"),
        };
        println!("玩家{pid} [{detail}]");
        println!("    之前: {before}");
        println!("    手牌前: {hand_before}");
        println!("    结果: {status}");
        println!("    之后: {after}");
        println!("    手牌后: {hand_after}");
        println!("    盘面: {}", board_state(&state));
        println!("    商家: {}", merchant_state(&state));

        if res.is_ok() {
            match advance_turn(&mut state) {
                TurnResult::Continue => {}
                TurnResult::EndCanalEra => {
                    println!("\n--- 运河时代结算明细 ---");
                    for pid in 0..players {
                        println!("玩家{pid}: {}", era_score_detail(&state, pid));
                    }
                    println!("{}", canal_cleanup_detail(&state));
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
