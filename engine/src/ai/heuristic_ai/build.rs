// BUILD
// ---------------------------------------------------------------------------

/// How much cash a newly built coal/iron works would earn by selling its cubes
/// into the market, and whether that sale empties the tile (flipping it for
/// income). Mirrors `GameState::auto_sell_to_market`: cubes fill market spaces
/// from the cheapest up, and stop when the market is full.
struct MarketSale {
    cash: f64,
    sold: u8,
    total: u8,
    flips: bool,
}

fn simulate_market_sale(state: &GameState, is_coal: bool, cubes: u8) -> MarketSale {
    let (prices, market) = if is_coal {
        (crate::map::COAL_MARKET_PRICES.as_slice(), state.coal_market)
    } else {
        (crate::map::IRON_MARKET_PRICES.as_slice(), state.iron_market)
    };
    let mut cash = 0.0;
    let mut m = market;
    let mut sold = 0u8;
    while sold < cubes && m < prices.len() {
        // Filling the cheapest open space first (same as auto_sell_to_market).
        let idx = prices.len() - m - 1;
        cash += prices[idx] as f64;
        m += 1;
        sold += 1;
    }
    MarketSale {
        cash,
        sold,
        total: cubes,
        flips: sold == cubes && cubes > 0,
    }
}

fn estimate_flip_probability(
    state: &GameState,
    pid: usize,
    ind: IndustryType,
    city_id: Loc,
) -> f64 {
    let base = if matches!(ind, IndustryType::CoalMine | IndustryType::IronWorks) {
        // Resources flip when consumed (building/networking) or when a build
        // immediately sells its cubes into the market (iron always; coal if
        // connected to a merchant). Flip odds follow MARKET HUNGER + ERA DEMAND:
        //   * sparse market => opponents must consume this resource => flips fast;
        //   * rail era is coal-hungry (every build/link needs coal) => near-guaranteed;
        //   * a build that sells out on placement flips immediately.
        let is_coal = ind == IndustryType::CoalMine;
        let cubes = state
            .players
            .get(pid)
            .and_then(|p| p.next_tile(ind))
            .map(|t| t.resource_cubes)
            .unwrap_or(1);
        let capacity = if is_coal { 14.0 } else { 10.0 };
        let market = if is_coal {
            state.coal_market as f64
        } else {
            state.iron_market as f64
        };
        let scarcity = ((capacity - market) / capacity).clamp(0.0, 1.0);
        let connected = connected_locations(state, city_id);
        let can_sell = ind == IndustryType::IronWorks || connected.iter().any(|l| l.is_merchant());
        // Coal flips by selling to a merchant market OR being consumed by the
        // table. A coal mine with NO merchant connection (an "island mine")
        // can't sell on build and — especially in the canal era — nobody will
        // build a link out to it, so it sits unflipped and vanishes at era end.
        // Humans only build connected coal (build their own link, piggyback a
        // neighbor's, or wild-city teleport in). Island coal is a bad move.
        if is_coal && !can_sell {
            if state.era == Era::Canal {
                // Canal coal that cannot sell on placement is still weak, but
                // less so when market coal is very expensive (6/7/8): demand
                // pressure means table consumption is likely soon.
                let heat = ((state.coal_price() as f64 - 5.0) / 3.0).clamp(0.0, 1.0);
                (0.12 + 0.18 * heat).min(0.4)
            } else {
                // Rail era: even without merchant auto-sell, coal demand is
                // table-wide and urgent; isolated mines are consumed quickly
                // once the rail graph densifies.
                let heat = ((state.coal_price() as f64 - 4.0) / 4.0).clamp(0.0, 1.0);
                (0.6 + 0.25 * heat).min(0.9)
            }
        } else {
            let sale = simulate_market_sale(state, is_coal, cubes);
            if can_sell && sale.flips {
                0.9
            } else {
                // Relies on consumption by the table. Coal demand is enormous
                // in the rail era (all builds and links consume it); iron
                // demand is steadier. Market scarcity adds urgency on top.
                let era_demand = if is_coal {
                    if state.era == Era::Rail { 0.85 } else { 0.55 }
                } else {
                    if state.era == Era::Rail { 0.5 } else { 0.4 }
                };
                (era_demand + 0.35 * scarcity).min(0.9)
            }
        }
    } else if ind == IndustryType::Brewery {
        // Beer barrels are consumed by sells AND rail links; a brewery flips
        // only if there is actual demand for its beer. In the canal era beer
        // has no use beyond fueling your own sells, so a brewery with no
        // sellable tiles to feed (or surplus barrels beyond demand) is waste:
        // it will sit unflipped. In the rail era double-rails also eat beer,
        // so demand gets a small floor.
        let next_cubes = state
            .players
            .get(pid)
            .and_then(|p| p.next_tile(ind))
            .map(|t| t.resource_cubes as usize)
            .unwrap_or(1);
        let demand = sellable_beer_demand(state, pid) as f64
            + if state.era == Era::Rail { 1.0 } else { 0.0 };
        let barrels = (owned_beer_barrels(state, pid) + next_cubes) as f64;
        if demand <= 0.5 && state.era == Era::Canal {
            0.25
        } else if barrels > demand {
            0.45
        } else {
            0.7
        }
    } else {
        // Sellable tiles ONLY flip when sold: that requires a reachable
        // merchant that accepts this industry AND beer to fuel the sale. A
        // tile with neither path is nearly worthless unflipped — keep the
        // base honestly low so expensive builds don't get credit for a
        // "double-score" they can never realize.
        let mut b = 0.12f64;
        let connected = connected_locations(state, city_id);
        let has_reachable_merchant = state
            .merchants
            .iter()
            .any(|mt| connected.contains(&mt.loc) && mt.accepts(ind));
        if has_reachable_merchant {
            // A merchant is only worth real flip credit if there is beer to
            // fuel the sale; without beer it's a dead end this era.
            let beer_ok = count_beer_sources(state, city_id, pid, &[]) > 0
                || beer_barrels_reachable(state, city_id);
            if beer_ok {
                b += 0.6;
            } else {
                b += 0.1;
            }
        }
        // Adjacent unbuilt links mean a merchant can still be connected later.
        let open_links = count_new_unbuilt_neighbor_connections(state, city_id);
        if open_links > 0 {
            b += 0.1;
        }
        match state.players[pid].hand.len() {
            0 => b -= 10.0,
            1 => b -= 5.0,
            2..=3 => b -= 2.0, 
            _ => {}
        }
        b
    };
    base.clamp(0.05, 1.0)
}

/// True if the player holds a card that can build `ind` (industry or wild).
fn player_has_buildable_card(state: &GameState, pid: usize, ind: IndustryType) -> bool {
    let player = &state.players[pid];
    player.hand.iter().any(|c| {
        matches!(c, Card::Industry { .. } if c.is_industry(ind))
            || c.ctype() == crate::data::CardType::WildIndustry
    })
}

fn player_owns_link_touching(state: &GameState, pid: usize, city_id: Loc) -> bool {
    connections().iter().any(|c| {
        if let Some(l) = &state.links[c.id] {
            l.player == pid && (c.a == city_id || c.b == city_id)
        } else {
            false
        }
    })
}

fn count_new_unbuilt_neighbor_connections(state: &GameState, city_id: Loc) -> usize {
    connections()
        .iter()
        .filter(|c| state.links[c.id].is_none() && (c.a == city_id || c.b == city_id))
        .count()
}

fn own_brewery_stats(state: &GameState, pid: usize) -> (usize, usize) {
    let mut barrels = 0usize;
    let mut flipped = 0usize;
    for tile in state.city_tiles.iter().flatten() {
        if tile.player == pid && tile.ind == IndustryType::Brewery {
            barrels += tile.resource_cubes as usize;
            if tile.flipped {
                flipped += 1;
            }
        }
    }
    for tile in state.farm_tiles.iter().flatten() {
        if tile.player == pid {
            barrels += tile.resource_cubes as usize;
            if tile.flipped {
                flipped += 1;
            }
        }
    }
    (barrels, flipped)
}

/// Can `pid` reach a merchant barrel (any accepted industry) from `loc`?
fn beer_barrels_reachable(state: &GameState, loc: Loc) -> bool {
    let connected = connected_locations(state, loc);
    state
        .merchants
        .iter()
        .any(|mt| mt.has_beer && connected.contains(&mt.loc))
}

/// Number of beer barrels the player currently holds (unflipped breweries).
pub(super) fn owned_beer_barrels(state: &GameState, pid: usize) -> usize {
    state
        .city_tiles
        .iter()
        .chain(state.farm_tiles.iter())
        .flatten()
        .filter(|t| t.player == pid && t.ind == IndustryType::Brewery && !t.flipped)
        .map(|t| t.resource_cubes as usize)
        .sum()
}

/// Total beer barrels needed to sell ALL of the player's unflipped sellable
/// tiles. Extra barrels beyond this (plus a rail-network buffer) are wasted.
fn sellable_beer_demand(state: &GameState, pid: usize) -> usize {
    state
        .city_tiles
        .iter()
        .flatten()
        .filter(|t| t.player == pid && !t.flipped && t.ind.is_sellable())
        .map(|t| t.def.beers_to_sell.unwrap_or(0) as usize)
        .sum()
}

/// Fraction of the coal/iron needed by a build that can come from free board
/// sources (own or opponent mines/works) rather than the paid market. Consuming
/// opponent resources is "free riding": it costs nothing extra and flips their
/// tile, keeping the shared resource pool cheap for everyone. A high ratio
/// means the build is cheap to run; a low one forces paying market price.
fn resource_source_ratio(state: &GameState, cand: &BuildTarget) -> f64 {
    let needed = cand.cost_coal as f64 + cand.cost_iron as f64;
    if needed <= 0.0 {
        return 1.0;
    }
    let free_coal = if cand.cost_coal > 0 {
        find_coal_sources(state, cand.loc)
            .iter()
            .filter(|s| s.free)
            .count() as f64
    } else {
        f64::MAX
    };
    let free_iron = if cand.cost_iron > 0 {
        find_iron_sources(state).iter().filter(|s| s.free).count() as f64
    } else {
        f64::MAX
    };
    let mut free_available = 0.0;
    free_available += free_coal.min(cand.cost_coal as f64);
    free_available += free_iron.min(cand.cost_iron as f64);
    (free_available / needed).clamp(0.0, 1.0)
}

fn score_build_candidate(state: &GameState, pid: usize, cand: &BuildTarget, plan: &Plan) -> f64 {
    let tile = state.players[pid].next_tile(cand.ind);
    let Some(tile) = tile else {
        return f64::NEG_INFINITY;
    };

    if BAN_BUILD_LV1_BREWERY && cand.ind == IndustryType::Brewery && tile.level == 1 {
        return f64::NEG_INFINITY;
    }

    let player = &state.players[pid];
    let cash = player.money as f64;
    let cost = cand.cost_total as f64;

    // Cash is the hard constraint that drives the early economy. If we cannot
    // afford the build, heavily discount it (still show a path via loan, but
    // not a first choice). If it consumes nearly all cash, require that it pay
    // off in income soon.
    if cost > cash {
        let unaffordable = -(cost - cash) * 0.3;
        return unaffordable;
    }

    let flip_prob = estimate_flip_probability(state, pid, cand.ind, cand.loc);
    let owns_adjacent_link = player_owns_link_touching(state, pid, cand.loc);
    let link_self_value = if owns_adjacent_link {
        tile.link_vp as f64 * flip_prob * 0.5
    } else {
        0.0
    };
    let is_resource = matches!(cand.ind, IndustryType::CoalMine | IndustryType::IronWorks);
    let resource_self_sufficiency = if is_resource {
        0.15 * tile.resource_cubes as f64
    } else {
        0.0
    };
    // Building a coal/iron works can immediately sell its production to the
    // market for cash (iron always; coal if connected to a merchant). The
    // value of such a build is dominated by MARKET HUNGER:
    //   * a sparse market (few cubes, high prices) rewards the fill richly —
    //     big cash-back, immediate flip for income, and service to the table;
    //   * a full market means cubes can't fit: they sit unflipped, wastefully
    //     waiting on opponents' consumption. Leftover cubes are a penalty.
    // "If everything sells, it's a great move; the more that stays on the
    // tile, the worse (unless spent on your very next action)."
    let market_adjust = if is_resource {
        let connected = connected_locations(state, cand.loc);
        let market_ok =
            cand.ind == IndustryType::IronWorks || connected.iter().any(|l| l.is_merchant());
        let is_coal = cand.ind == IndustryType::CoalMine;
        // Coal demand is far higher than iron (every rail build/link eats it);
        // iron demand is steadier and the market fills faster. Discount iron
        // scarcity so we don't over-produce iron that nobody will consume.
        let scarcity = if is_coal {
            (14 - state.coal_market) as f64 / 14.0
        } else {
            0.6 * (10 - state.iron_market) as f64 / 10.0
        };
        if market_ok {
            let sale = simulate_market_sale(state, is_coal, tile.resource_cubes);
            let sell_value = sale.cash * money_weight(state);
            let cash_back_bonus = if sale.cash > 0.0 {
                sale.cash * 0.4 + if sale.flips { 1.5 } else { 0.0 }
            } else {
                0.0
            };
            // Coal market spike prior: when buy price is 6/7/8, placing a
            // connected coal mine that can auto-sell is typically a top-tier
            // tempo play (cash now + flip income + future board demand).
            let coal_spike_bonus = if is_coal && sale.sold > 0 {
                let buy_price = state.coal_price() as f64;
                let heat = ((buy_price - 5.0) / 3.0).clamp(0.0, 1.0);
                let sold_factor = sale.sold as f64;
                let era_mult = if state.era == Era::Canal { 1.25 } else { 1.0 };
                heat * sold_factor * 1.9 * era_mult
            } else {
                0.0
            };
            let scarcity_value = scarcity * (1.0 + sale.sold as f64) * 0.6;
            let leftover_penalty = if is_coal && state.era == Era::Rail {
                // In the rail era, coal is consumed by nearly every build and
                // rail link — a full market right now doesn't mean the cubes
                // won't be eaten soon. Don't punish unsold coal hard.
                0.0
            } else {
                (sale.total - sale.sold) as f64 * 0.5
            };
            sell_value + cash_back_bonus + coal_spike_bonus + scarcity_value - leftover_penalty
        } else {
            // Not merchant-connected: cubes can't be sold today. For an
            // ISLAND COAL MINE this is a strongly negative move in the canal
            // era (tiles vanish at era end, nobody helps consume it) and merely
            // speculative in the rail era (links will be built out). Iron needs
            // no connection to be consumed, so it keeps a market-value floor.
            if is_coal {
                if state.era == Era::Canal {
                    -0.5
                } else {
                    // Rail coal without merchant reach is still a strong play
                    // under scarcity: it backfills the table's fuel demand,
                    // gets consumed fast, and often flips for income soon.
                    scarcity * (1.2 + 0.25 * tile.resource_cubes as f64)
                }
            } else {
                scarcity * 1.2
            }
        }
    } else {
        0.0
    };
    let network_expansion = 0.1 * count_new_unbuilt_neighbor_connections(state, cand.loc) as f64;

    // Emergency coal supply prior (rail): if coal market is near empty, any
    // legal coal mine is strategically premium because the whole table must pay
    // expensive market coal otherwise. This is independent of immediate
    // merchant auto-sell.
    let rail_coal_shortage_bonus = if cand.ind == IndustryType::CoalMine && state.era == Era::Rail {
        let shortage = (1.0 - (state.coal_market as f64 / 14.0)).clamp(0.0, 1.0);
        let level_factor = 1.0 + 0.2 * (tile.level.saturating_sub(1)) as f64;
        let cubes_factor = 0.7 + 0.15 * tile.resource_cubes as f64;
        shortage * level_factor * cubes_factor * 3.0
    } else {
        0.0
    };

    // Cost efficiency: the build must be worth its price tag. Cheap builds
    // (coal £5, iron £7, brewery £5) get a relative edge early.
    let cost_efficiency = if cost > 0.0 {
        let eff = (tile.income as f64 + tile.vp as f64) / cost;
        eff.min(2.0)
    } else {
        0.0
    };

    // Beer economy: sellable tiles are only worth their VP if we can actually
    // sell them (merchant reachable + beer available). Breweries are worth a
    // premium when we already have unflipped sellable tiles waiting on beer.
    let sellable = cand.ind.is_sellable();
    let mut beer_bonus = 0.0;
    if sellable {
        // A reachable merchant that accepts this industry raises flip odds.
        let connected = connected_locations(state, cand.loc);
        let has_merchant = state
            .merchants
            .iter()
            .any(|mt| connected.contains(&mt.loc) && mt.accepts(cand.ind));
        if has_merchant {
            beer_bonus += 0.6;
        }
        // Beer scarcity: if we hold no beer source and can't reach one with a
        // barrel, the sellable tile is near-worthless unflipped. (Don't punish
        // too hard: a brewery can still be added later.)
        let beer_ok = has_merchant
            && (count_beer_sources(state, cand.loc, pid, &[])
                >= tile.beers_to_sell.unwrap_or(0) as usize
                || beer_barrels_reachable(state, cand.loc));
        if beer_ok {
            // We have the beer AND the merchant: this is a guaranteed sellable.
            beer_bonus += 0.8;
        } else {
            beer_bonus -= 0.3;
        }
    } else if cand.ind == IndustryType::Brewery {
        // Breweries feed sells AND rail links. The winning line develops
        // level-1 away and builds level-2/3/4 (they survive to rail). Match
        // output to need: barrels should cover our unflipped sellable tiles'
        // beer demand plus a small rail-network buffer. Building beyond that is
        // wasted, and level-1 breweries vanish at era end so they're weak.
        let barrels = owned_beer_barrels(state, pid) as f64 + tile.resource_cubes as f64; // this brewery's contribution
        let demand = sellable_beer_demand(state, pid) as f64;
        let surplus = (barrels - demand).max(0.0);
        let sat = -0.6 * surplus;
        let sell_support = if demand > 0.0 { 0.8 } else { 0.4 };
        let rail_beer_value = if state.era == Era::Rail { 2.0 } else { 0.0 };
        beer_bonus += sell_support + rail_beer_value + sat;
    }

    // Level-2+ tiles score their flipped VP at BOTH era ends
    let double_vp = if tile.level >= 2 && state.era == Era::Rail {
        2.0
    } else {
        1.0
    };

    // "Free-riding" efficiency: if the build's coal/iron can come from board
    // mines/works (own or opponents) instead of the paid market, the action is
    // cheaper and faster. Reward builds on an established resource pool.
    let resource_ratio = resource_source_ratio(state, cand);
    let interaction_bonus = (resource_ratio - 0.5).max(0.0) * 0.8;

    let mut score = vp_equivalent(
        state,
        tile.vp as f64 * flip_prob * double_vp + link_self_value,
        tile.income as f64 * flip_prob,
        -(cand.cost_total as f64),
        0.0,
    ) + resource_self_sufficiency
        + network_expansion
        + rail_coal_shortage_bonus
        + beer_bonus
        + cost_efficiency
        + market_adjust
        + interaction_bonus;

    // Plan ("流派") soft bonus: building the target industry aligns with the
    // player's production plan. Only applies from Canal-Late onward — in
    // Canal-Early the priority is the coal/iron economy engine, not committing
    // to the sellable line yet. Additive + small so it nudges, not overrides.
    if plan.count > 0 && plan.industry == cand.ind && era_phase(state) != Phase::CanalEarly {
        score += 0.5;
    }

    // Rail-Late beer-gated finish: the whole late game hinges on flipping
    // sellables. A sellable tile is only worth building now if we have beer to
    // sell it (own barrels to spare OR reachable merchant beer) — that's the
    // "有酒才建产业" rule. When beer is genuinely available, reward the build
    // (it's the finishing move); when not, `flip_prob` already keeps it low.
    if era_phase(state) == Phase::RailLate && cand.ind.is_sellable() {
        let beer_ok = {
            let connected = connected_locations(state, cand.loc);
            count_beer_sources(state, cand.loc, pid, &[]) > 0
                || state
                    .merchants
                    .iter()
                    .any(|mt| mt.has_beer && connected.contains(&mt.loc))
        };
        if beer_ok {
            score += 1.2;
        }
    }

    score
}

fn pick_build_card(state: &GameState, pid: usize, cand: &BuildTarget) -> Option<(usize, f64)> {
    let player = &state.players[pid];
    let indices = valid_build_cards(state, player, pid, cand.loc, cand.ind);
    // Prefer a non-wild matching card over a wild one.
    let index = indices.into_iter().min_by(|a, b| {
        card_keep_score(state, pid, *a)
            .total_cmp(&card_keep_score(state, pid, *b))
            .then(a.cmp(b))
    })?;
    Some((index, card_keep_score(state, pid, index)))
}

/// Top-K build candidates by 1-ply score. Used by MCTS to get a wider prior.
pub(crate) fn score_top_builds(
    state: &mut GameState,
    pid: usize,
    k: usize,
    plan: &Plan,
    targets: &[BuildTarget],
) -> Vec<Decision> {
    let mut scored: Vec<(BuildTarget, f64)> = targets
        .iter()
        .cloned()
        .map(|t| {
            let s = score_build_candidate(state, pid, &t, plan);
            (t, s)
        })
        .collect();
    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    scored.truncate(k);
    let mut out = Vec::new();
    for (cand, score) in scored {
        if let Some((card_index, card_score)) = pick_build_card(state, pid, &cand) {
            let coal_needed = cand.cost_coal as usize;
            let iron_needed = cand.cost_iron as usize;
            let coal = crate::rules::coal_source_options(state, cand.loc, coal_needed)
                .into_iter()
                .next()
                .unwrap_or_default();
            let iron = crate::rules::iron_source_options(state, iron_needed)
                .into_iter()
                .next()
                .unwrap_or_default();
            out.push(Decision {
                mv: ResolvedMove::Build {
                    loc: cand.loc,
                    slot_index: cand.slot_index,
                    ind: cand.ind,
                    coal,
                    iron,
                    card_index,
                },
                score,
                card_score,
            });
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use rand_chacha::{ChaCha12Rng, rand_core::SeedableRng};

    use super::*;
    use crate::data::industry_tiles;
    use crate::state::BoardTile;

    #[test]
    fn owned_beer_includes_unflipped_farm_brewery() {
        let mut state = GameState::new(ChaCha12Rng::seed_from_u64(7), 2);
        let pid = state.current_player_id();
        state.farm_tiles[0] = Some(BoardTile {
            player: pid,
            ind: IndustryType::Brewery,
            def: industry_tiles(IndustryType::Brewery)[0],
            flipped: false,
            resource_cubes: 2,
        });

        assert_eq!(owned_beer_barrels(&state, pid), 2);
    }
}
