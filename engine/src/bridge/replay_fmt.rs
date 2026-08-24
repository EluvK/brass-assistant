//! Human-readable Chinese replay formatting (replay.rs's log layer, lifted into
//! the library so BOTH the `replay` binary and the Python driver can produce
//! identical output).
//!
//! All functions are **pure read-only**: they take `&GameState` and never mutate
//! it, so they are safe to call from the Python replay driver between the
//! stepwise move/advance calls. They are also NOT on any training/search hot
//! path (never called by `nn_mcts` / `rules` / `mcts_ai`), so exposing them
//! costs nothing for training or inference performance.

use crate::data::{CardType, Era, IndustryType};
use crate::map::{ALL_LOCATIONS, CITY_COUNT, Loc, city_slots, connections};
use crate::rules::{Move, calculate_build_cost};
use crate::state::{Card, GameState, city_slot_offsets};

// ---------------------------------------------------------------------------
// Labels
// ---------------------------------------------------------------------------

pub fn era_label(era: Era) -> &'static str {
    match era {
        Era::Canal => "运河",
        Era::Rail => "铁路",
    }
}

pub fn loc_label(loc: Loc) -> String {
    format!("{}({})", loc.zh_name(), loc.name())
}

pub fn industry_label(ind: IndustryType) -> &'static str {
    match ind {
        IndustryType::CottonMill => "棉纺厂",
        IndustryType::CoalMine => "煤矿",
        IndustryType::IronWorks => "铁厂",
        IndustryType::Manufacturer => "制造厂",
        IndustryType::Pottery => "陶器厂",
        IndustryType::Brewery => "啤酒厂",
    }
}

pub fn card_label(card: &Card) -> String {
    match card {
        Card::Location(loc) => loc_label(*loc),
        Card::Industry { industries, n } => {
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
        Card::WildLocation => "万能地点牌(Wild Location)".to_string(),
        Card::WildIndustry => "万能产业牌(Wild Industry)".to_string(),
    }
}

// ---------------------------------------------------------------------------
// State displays
// ---------------------------------------------------------------------------

pub fn hand_display(state: &GameState, pid: usize) -> String {
    let hand = &state.players[pid].hand;
    let mut parts = Vec::new();
    for (i, c) in hand.iter().enumerate() {
        let tag = match c.ctype() {
            CardType::Location => "地",
            CardType::Industry => "产",
            CardType::WildLocation => "万地",
            CardType::WildIndustry => "万产",
        };
        parts.push(format!("{i}:{tag}({})", card_label(c)));
    }
    parts.join(" ")
}

pub fn player_state(state: &GameState, pid: usize) -> String {
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

pub fn board_state(state: &GameState) -> String {
    let built =
        state.city_tiles.iter().flatten().count() + state.farm_tiles.iter().flatten().count();
    let flipped = state
        .city_tiles
        .iter()
        .flatten()
        .filter(|t| t.flipped)
        .count()
        + state
            .farm_tiles
            .iter()
            .flatten()
            .filter(|t| t.flipped)
            .count();
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

pub fn merchant_state(state: &GameState) -> String {
    state
        .merchants
        .iter()
        .map(|mt| {
            let buys = match mt.buys {
                crate::state::BuyType::Blank => "空".to_string(),
                crate::state::BuyType::Any => "任意".to_string(),
                crate::state::BuyType::Industry(t) => industry_label(t).to_string(),
            };
            format!(
                "{}[{}]{}",
                loc_label(mt.loc),
                buys,
                if mt.has_beer { "·桶" } else { "" }
            )
        })
        .collect::<Vec<_>>()
        .join("  ")
}

// ---------------------------------------------------------------------------
// Move detail
// ---------------------------------------------------------------------------

pub fn mv_card_index(mv: &Move) -> usize {
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

pub fn move_detail(state: &GameState, mv: &Move) -> String {
    let pid = state.current_player_id();
    let card = state.players[pid].hand.get(mv_card_index(mv));
    let card_name = card.map(card_label).unwrap_or_else(|| "?".to_string());
    match mv {
        Move::Build {
            loc,
            slot_index,
            ind,
            ..
        } => {
            let cost = crate_cost(state, pid, *ind, *loc);
            format!(
                "建厂 {} Lv{} @ {}(槽{})，用牌[{}]，花费£{}",
                industry_label(*ind),
                state.players[pid]
                    .next_tile(*ind)
                    .map(|t| t.level)
                    .unwrap_or(0),
                loc_label(*loc),
                slot_index + 1,
                card_name,
                cost
            )
        }
        Move::Network { conn_id, .. } => {
            let c = &connections()[*conn_id];
            format!(
                "建网 {}↔{}({}时代)，用牌[{}]",
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
                    // A double develop removes ind1 before resolving ind2.
                    // Read the following tile when both choices are the same
                    // industry, rather than reporting the first tile twice.
                    .tile_after(*i2, usize::from(*ind1 == *i2))
                    .map(|t| t.level)
                    .unwrap_or(0);
                format!("{s} + {} Lv{}", industry_label(*i2), lv2)
            } else {
                s
            };
            format!("{s}，用牌[{}]", card_name)
        }
        Move::Sell {
            keys,
            merchant_indices,
            use_merchant_beer,
            ..
        } => {
            let mut parts = Vec::new();
            for (i, k) in keys.iter().enumerate() {
                let (loc, slot) = match crate::state::loc_from_key(*k) {
                    Some((l, s)) => (Some(l), s),
                    None => (None, 0),
                };
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
        Move::Loan { .. } => format!("贷款 £30(收入-3级),用牌[{}]", card_name),
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
        Move::Pass { .. } => format!("过牌(弃[{}])", card_name),
    }
}

fn crate_cost(state: &GameState, pid: usize, ind: IndustryType, loc: Loc) -> String {
    match calculate_build_cost(state, pid, ind, loc) {
        Some(c) => format!("{}(+煤{}铁{})", c.cost_total, c.coal_cost, c.iron_cost),
        None => "?".to_string(),
    }
}

// ---------------------------------------------------------------------------
// Era scoring detail
// ---------------------------------------------------------------------------

pub fn tile_label(loc: Loc, slot_index: usize, tile: &crate::state::BoardTile) -> String {
    format!(
        "{} Lv{} @ {}(槽{})",
        industry_label(tile.ind),
        tile.def.level,
        loc_label(loc),
        slot_index + 1
    )
}

pub fn era_score_detail(state: &GameState, pid: usize) -> String {
    let mut flipped_tiles = Vec::new();
    let mut industry_vp = 0u16;
    for (i, loc) in ALL_LOCATIONS[..CITY_COUNT].iter().enumerate() {
        for slot_index in 0..city_slots(*loc).len() {
            let key = city_slot_offsets()[i] + slot_index;
            if let Some(tile) = state.city_tiles[key].as_ref() {
                if tile.player == pid && tile.flipped {
                    industry_vp += tile.def.vp as u16;
                    flipped_tiles.push(format!(
                        "{} => {}VP",
                        tile_label(*loc, slot_index, tile),
                        tile.def.vp
                    ));
                }
            }
        }
    }
    for farm_loc in [Loc::BreweryNorth, Loc::BrewerySouth] {
        if let Some(tile) = state.farm_tile(farm_loc) {
            if tile.player == pid && tile.flipped {
                industry_vp += tile.def.vp as u16;
                flipped_tiles.push(format!(
                    "{} Lv{} @ {} => {}VP",
                    industry_label(tile.ind),
                    tile.def.level,
                    loc_label(farm_loc),
                    tile.def.vp
                ));
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
        link_parts.push(format!(
            "{}↔{} => {}VP [{}]",
            loc_label(conn.a),
            loc_label(conn.b),
            value,
            sources.join(", ")
        ));
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

pub fn canal_cleanup_detail(state: &GameState) -> String {
    let mut removed_links = Vec::new();
    for (id, link) in state.links.iter().enumerate() {
        let Some(link) = link else { continue };
        if link.is_canal {
            let conn = &connections()[id];
            removed_links.push(format!(
                "玩家{}:{}↔{}",
                link.player,
                loc_label(conn.a),
                loc_label(conn.b)
            ));
        }
    }

    let mut removed_tiles = Vec::new();
    for loc in ALL_LOCATIONS[..CITY_COUNT].iter() {
        for slot_index in 0..city_slots(*loc).len() {
            if let Some(tile) = state.tile_at(*loc, slot_index) {
                if tile.def.level == 1 {
                    removed_tiles.push(format!(
                        "玩家{}:{}",
                        tile.player,
                        tile_label(*loc, slot_index, tile)
                    ));
                }
            }
        }
    }
    for farm_loc in [Loc::BreweryNorth, Loc::BrewerySouth] {
        if let Some(tile) = state.farm_tile(farm_loc) {
            if tile.def.level == 1 {
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

// ---------------------------------------------------------------------------
// Per-player action stats + tile inventory
// ---------------------------------------------------------------------------

#[derive(Default, Clone)]
pub struct PStats {
    pub build: [u32; 6],
    pub network: u32,
    pub dbl_rail: u32,
    pub develop: u32,
    pub sell: u32,
    pub loan: u32,
    pub pass: u32,
    pub scout: u32,
    /// Ordered, compact action labels for era-end replay summaries.
    pub actions: Vec<String>,
}

impl PStats {
    pub fn record(&mut self, mv: &Move, era: Era) {
        self.actions.push(action_abbrev(mv, era));
        match mv {
            Move::Build { ind, .. } => self.build[*ind as usize] += 1,
            Move::Network { .. } => self.network += 1,
            Move::NetworkDouble { .. } => self.dbl_rail += 1,
            Move::Develop { .. } => self.develop += 1,
            Move::Sell { .. } => self.sell += 1,
            Move::Loan { .. } => self.loan += 1,
            Move::Pass { .. } => self.pass += 1,
            Move::Scout { .. } => self.scout += 1,
        }
    }
}

fn action_abbrev(mv: &Move, era: Era) -> String {
    const IND_ABBREVS: [&str; 6] = ["棉", "煤", "铁", "箱", "陶", "酒"];
    let ind = |ind: IndustryType| IND_ABBREVS[ind as usize];

    match mv {
        Move::Build { ind: industry, .. } => ind(*industry).to_string(),
        Move::Network { .. } => match era {
            Era::Canal => "路".to_string(),
            Era::Rail => "路".to_string(),
        },
        Move::NetworkDouble { .. } => "双修".to_string(),
        Move::Develop { ind1, ind2, .. } => {
            let mut label = format!("研{}", ind(*ind1));
            if let Some(ind2) = ind2 {
                label.push_str(ind(*ind2));
            }
            label
        }
        Move::Sell { .. } => "卖".to_string(),
        Move::Loan { .. } => "贷".to_string(),
        Move::Pass { .. } => "过".to_string(),
        Move::Scout { .. } => "斥".to_string(),
    }
}

pub fn fmt_action_sequence(stats: &PStats, first: bool) -> String {
    if stats.actions.is_empty() {
        "无".to_string()
    } else {
        let mut actions = stats.actions.clone();
        if first {
            actions.insert(0, "".to_string());
        }
        let chunks: Vec<String> = actions
            .chunks(2)
            .map(|chunk| chunk.join("+"))
            .filter(|s| !s.trim().is_empty())
            .collect();
        chunks.join(" ")
    }
}

pub fn fmt_stats_line(stats: &PStats) -> String {
    let ind_names = ["棉", "煤", "铁", "箱", "陶", "酒"];
    let builds: Vec<String> = (0..6)
        .map(|i| format!("{}x{}", ind_names[i], stats.build[i]))
        .filter(|x| !x.ends_with('0'))
        .collect();
    format!(
        "建[{}] 网{} 双铁{} 研发{} 卖{} 贷{} 过{} 斥{}",
        builds.join(" "),
        stats.network,
        stats.dbl_rail,
        stats.develop,
        stats.sell,
        stats.loan,
        stats.pass,
        stats.scout
    )
}

pub fn player_tiles_detail(state: &GameState, pid: usize, flipped_only: bool) -> String {
    let mut entries: Vec<String> = Vec::new();
    for (i, loc) in ALL_LOCATIONS[..CITY_COUNT].iter().enumerate() {
        for slot_index in 0..city_slots(*loc).len() {
            let key = city_slot_offsets()[i] + slot_index;
            if let Some(tile) = state.city_tiles[key].as_ref() {
                if tile.player == pid && (!flipped_only || tile.flipped) {
                    entries.push(tile_label(*loc, slot_index, tile));
                }
            }
        }
    }
    for farm_loc in [Loc::BreweryNorth, Loc::BrewerySouth] {
        if let Some(tile) = state.farm_tile(farm_loc) {
            if tile.player == pid && (!flipped_only || tile.flipped) {
                entries.push(format!(
                    "{} Lv{} @ {}",
                    industry_label(tile.ind),
                    tile.def.level,
                    loc_label(farm_loc)
                ));
            }
        }
    }
    if entries.is_empty() {
        "无".to_string()
    } else {
        entries.join(" ; ")
    }
}

#[cfg(test)]
mod tests {
    use super::move_detail;
    use crate::data::IndustryType;
    use crate::rules::Move;
    use crate::state::GameState;
    use rand_chacha::ChaCha12Rng;
    use rand_chacha::rand_core::SeedableRng;

    #[test]
    fn same_industry_double_develop_reports_consecutive_tile_levels() {
        let state = GameState::new(ChaCha12Rng::seed_from_u64(1), 2);
        let develop = Move::Develop {
            ind1: IndustryType::CoalMine,
            ind2: Some(IndustryType::CoalMine),
            iron: Vec::new(),
            card_index: 0,
        };
        assert!(
            move_detail(&state, &develop).contains("煤矿 Lv1 + 煤矿 Lv2"),
            "double develop must report both consecutively removed tiles"
        );
    }
}
