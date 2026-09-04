//! Build action scoring and candidate generation.
//!
//! Build scores are already expressed in VP equivalents. The policy is kept
//! local to this module so the action scorer can be read and tuned without
//! chasing a shared evaluation context or an unrelated configuration structure.

use super::board::{
    beer_available, beer_barrels_reachable, merchant_reachable, own_overbuild_vp_loss,
    owned_beer_barrels, player_owns_link_touching, resource_source_ratio, sellable_beer_demand,
    unbuilt_neighbor_connections,
};
use super::context::define_era_round_factor;
use super::plan::Plan;
use super::probability::build_flip_probability;
use super::value::{market_scarcity, price_heat, simulate_market_sale};
use super::{Decision, SOURCE_VARIANTS};
use crate::data::IndustryType;
use crate::graph::count_beer_sources;
use crate::rules::{BuildTarget, ResolvedMove, valid_build_cards};
use crate::state::GameState;

// ---------------------------------------------------------------------------
// Local policy
// ---------------------------------------------------------------------------

const BAN_BUILD_LEVEL1_BREWERY: bool = true;

const UNAFFORDABLE_PER_POUND: f64 = 0.3;
const LINK_SELF_VALUE_SHARE: f64 = 0.5;
const SELF_SUFFICIENCY_PER_CUBE: f64 = 0.15;
const IRON_SCARCITY_SHARE: f64 = 0.6;
const MARKET_CASH_BACK_SHARE: f64 = 0.4;
const MARKET_SELLOUT_BONUS: f64 = 1.5;
const COAL_SPIKE_PRICE_BASE: f64 = 5.0;
const COAL_SPIKE_PRICE_SPAN: f64 = 3.0;
const COAL_SPIKE_PER_SOLD: f64 = 1.9;
const COAL_SPIKE_CANAL_MULTIPLIER: f64 = 1.25;
const SCARCITY_VALUE_PER_UNIT: f64 = 0.6;
const LEFTOVER_PER_CUBE: f64 = 0.5;
const ISLAND_COAL_CANAL_PENALTY: f64 = -0.5;
const ISLAND_COAL_RAIL_BASE: f64 = 1.2;
const ISLAND_COAL_RAIL_PER_CUBE: f64 = 0.25;
const ISLAND_IRON_VALUE: f64 = 1.2;
const EXPANSION_PER_LINK: f64 = 0.1;
const RAIL_COAL_SHORTAGE: f64 = 3.0;
const RAIL_COAL_SHORTAGE_PER_LEVEL: f64 = 0.2;
const RAIL_COAL_SHORTAGE_CUBES_BASE: f64 = 0.7;
const RAIL_COAL_SHORTAGE_PER_CUBE: f64 = 0.15;
const COST_EFFICIENCY_CAP: f64 = 2.0;
const MERCHANT_REACHABLE_BONUS: f64 = 0.6;
const BEER_AVAILABLE_BONUS: f64 = 0.8;
const BEER_MISSING_PENALTY: f64 = -0.3;
const BREWERY_SURPLUS_PENALTY_PER_BARREL: f64 = 0.6;
const BREWERY_SELL_SUPPORT_WITH_DEMAND: f64 = 0.8;
const BREWERY_SELL_SUPPORT_BASE: f64 = 0.4;
const RAIL_BREWERY_VALUE: f64 = 2.0;
const FREE_RIDING_THRESHOLD: f64 = 0.5;
const FREE_RIDING_BONUS: f64 = 0.8;
const PLAN_BONUS: f64 = 0.5;
const RAIL_LATE_BEER_BONUS: f64 = 1.2;

// Shared conversion constants for converting cash/income to VP equivalents.
const MONEY_BASE: f64 = 0.12;
const INCOME_BASE: f64 = 0.25;

// Money becomes more important as an era closes. Income is valuable while
// there is time for it to compound, and is worthless in Rail's last round.
define_era_round_factor!(
    BuildMoneyWeight,
    canal: (MONEY_BASE * 0.55, MONEY_BASE * 0.55),
    rail: (MONEY_BASE * 0.8, MONEY_BASE * (5.0 / 3.0)),
);
define_era_round_factor!(
    BuildIncomeWeight,
    canal: (
        INCOME_BASE * (1.8 + 0.6),
        INCOME_BASE * (1.8 + 0.6 / 8.0)
    ),
    rail: (INCOME_BASE * (1.2 + 0.5), 0.0),
);

fn market_value(state: &GameState, cand: &BuildTarget, cubes: u8) -> f64 {
    let is_coal = cand.ind == IndustryType::CoalMine;
    let scarcity =
        market_scarcity(state, is_coal) * if is_coal { 1.0 } else { IRON_SCARCITY_SHARE };

    // Iron works sell anywhere; coal needs a merchant in reach.
    let market_ok = !is_coal || merchant_reachable(state, cand.loc, cand.ind);
    if !market_ok {
        return if is_coal {
            if state.is_canal_era() {
                ISLAND_COAL_CANAL_PENALTY
            } else {
                scarcity * (ISLAND_COAL_RAIL_BASE + ISLAND_COAL_RAIL_PER_CUBE * cubes as f64)
            }
        } else {
            scarcity * ISLAND_IRON_VALUE
        };
    }

    let sale = simulate_market_sale(state, is_coal, cubes);
    let money = sale.cash * BuildMoneyWeight::factor(state);
    let cash_back = if sale.cash > 0.0 {
        sale.cash * MARKET_CASH_BACK_SHARE
            + if sale.flips {
                MARKET_SELLOUT_BONUS
            } else {
                0.0
            }
    } else {
        0.0
    };
    let coal_spike = if is_coal && sale.sold > 0 {
        price_heat(
            state.coal_price(),
            COAL_SPIKE_PRICE_BASE,
            COAL_SPIKE_PRICE_SPAN,
        ) * sale.sold as f64
            * COAL_SPIKE_PER_SOLD
            * if state.is_canal_era() {
                COAL_SPIKE_CANAL_MULTIPLIER
            } else {
                1.0
            }
    } else {
        0.0
    };
    let scarcity_value = scarcity * (1.0 + sale.sold as f64) * SCARCITY_VALUE_PER_UNIT;
    // Rail coal is normally consumed quickly, so unsold cubes are not a risk
    // there. In Canal they can remain stranded until the era ends.
    let leftover = if is_coal && !state.is_canal_era() {
        0.0
    } else {
        (sale.total - sale.sold) as f64 * LEFTOVER_PER_CUBE
    };

    money + cash_back + coal_spike + scarcity_value - leftover
}

fn beer_economy(
    state: &GameState,
    pid: usize,
    cand: &BuildTarget,
    beers_to_sell: Option<u8>,
) -> f64 {
    if cand.ind.is_sellable() {
        let merchant = merchant_reachable(state, cand.loc, cand.ind);
        let beer =
            merchant && beer_available(state, cand.loc, pid, beers_to_sell.unwrap_or(0) as usize);
        return (if merchant {
            MERCHANT_REACHABLE_BONUS
        } else {
            0.0
        }) + if beer {
            BEER_AVAILABLE_BONUS
        } else {
            BEER_MISSING_PENALTY
        };
    }

    if cand.ind != IndustryType::Brewery {
        return 0.0;
    }

    let tile_cubes = state.players[pid]
        .next_tile(cand.ind)
        .map(|tile| tile.resource_cubes as f64)
        .unwrap_or(0.0);
    let barrels = owned_beer_barrels(state, pid) as f64 + tile_cubes;
    let demand = sellable_beer_demand(state, pid) as f64;
    let surplus = (barrels - demand).max(0.0);
    let support = if demand > 0.0 {
        BREWERY_SELL_SUPPORT_WITH_DEMAND
    } else {
        BREWERY_SELL_SUPPORT_BASE
    };
    support
        + if state.is_canal_era() {
            0.0
        } else {
            RAIL_BREWERY_VALUE
        }
        - BREWERY_SURPLUS_PENALTY_PER_BARREL * surplus
}

fn rail_coal_shortage(state: &GameState, cand: &BuildTarget, level: u8, cubes: u8) -> f64 {
    if cand.ind != IndustryType::CoalMine || state.is_canal_era() {
        return 0.0;
    }
    let shortage = market_scarcity(state, true);
    let level_factor = 1.0 + RAIL_COAL_SHORTAGE_PER_LEVEL * level.saturating_sub(1) as f64;
    let cubes_factor = RAIL_COAL_SHORTAGE_CUBES_BASE + RAIL_COAL_SHORTAGE_PER_CUBE * cubes as f64;
    shortage * level_factor * cubes_factor * RAIL_COAL_SHORTAGE
}

fn cost_efficiency(income: u8, vp: u8, cost: f64) -> f64 {
    if cost > 0.0 {
        ((income as f64 + vp as f64) / cost).min(COST_EFFICIENCY_CAP)
    } else {
        0.0
    }
}

/// Score one legal build candidate directly in VP equivalents.
pub(super) fn score_build_candidate(
    state: &GameState,
    pid: usize,
    cand: &BuildTarget,
    plan: &Plan,
) -> f64 {
    let Some(tile) = state.players[pid].next_tile(cand.ind) else {
        return f64::NEG_INFINITY;
    };
    if BAN_BUILD_LEVEL1_BREWERY && cand.ind == IndustryType::Brewery && tile.level == 1 {
        return f64::NEG_INFINITY;
    }

    let cost = cand.cost_total as f64;
    let cash = state.players[pid].money as f64;
    if cost > cash {
        return -(cost - cash) * UNAFFORDABLE_PER_POUND;
    }

    let flip_prob = build_flip_probability(state, pid, cand.ind, cand.loc);
    let link_value = if player_owns_link_touching(state, pid, cand.loc) {
        tile.link_vp as f64 * flip_prob * LINK_SELF_VALUE_SHARE
    } else {
        0.0
    };
    let is_resource = matches!(cand.ind, IndustryType::CoalMine | IndustryType::IronWorks);
    let self_supply = if is_resource {
        SELF_SUFFICIENCY_PER_CUBE * tile.resource_cubes as f64
    } else {
        0.0
    };

    let mut score = tile.vp as f64 * flip_prob
        + link_value
        + tile.income as f64 * flip_prob * BuildIncomeWeight::factor(state)
        - cost * BuildMoneyWeight::factor(state)
        + self_supply
        + EXPANSION_PER_LINK * unbuilt_neighbor_connections(state, cand.loc) as f64
        + rail_coal_shortage(state, cand, tile.level, tile.resource_cubes)
        + cost_efficiency(tile.income, tile.vp, cost)
        + beer_economy(state, pid, cand, tile.beers_to_sell);

    if is_resource {
        score += market_value(state, cand, tile.resource_cubes);
    }

    // Free-riding keeps the common resource pool cheap and avoids spending an
    // action on a resource tile that opponents can supply for us.
    let ratio = resource_source_ratio(state, cand);
    score += (ratio - FREE_RIDING_THRESHOLD).max(0.0) * FREE_RIDING_BONUS;

    score -= own_overbuild_vp_loss(state, pid, cand, 1.0);

    // Canal-Early is reserved for establishing the coal/iron engine; from
    // Canal-Late onward, follow the production plan.
    if plan.count > 0 && plan.industry == cand.ind && (!state.is_canal_era() || state.round > 4) {
        score += PLAN_BONUS;
    }

    // In Rail-Late, a sellable with beer available is a preferred finisher.
    if !state.is_canal_era()
        && state.round > 4
        && cand.ind.is_sellable()
        && beer_available_for_sellable(state, pid, cand)
    {
        score += RAIL_LATE_BEER_BONUS;
    }

    score
}

fn beer_available_for_sellable(state: &GameState, pid: usize, cand: &BuildTarget) -> bool {
    count_beer_sources(state, cand.loc, pid, &[]) > 0 || beer_barrels_reachable(state, cand.loc)
}

/// Pick the cheapest-to-discard legal card for a build target. Consumes the
/// precomputed hand keep-scores instead of re-deriving them in a comparator.
fn pick_build_card(
    state: &GameState,
    pid: usize,
    cand: &BuildTarget,
    keep_scores: &[(usize, f64)],
) -> Option<(usize, f64)> {
    let keep = |i: usize| {
        keep_scores
            .iter()
            .find(|(idx, _)| *idx == i)
            .map(|(_, score)| *score)
            .unwrap_or(f64::INFINITY)
    };
    let player = &state.players[pid];
    let indices = valid_build_cards(state, player, pid, cand.loc, cand.ind);
    let non_wild: Vec<_> = indices
        .iter()
        .copied()
        .filter(|index| {
            !matches!(
                player.hand[*index],
                crate::state::Card::WildLocation | crate::state::Card::WildIndustry
            )
        })
        .collect();
    let candidates = if non_wild.is_empty() {
        &indices
    } else {
        &non_wild
    };
    let index = candidates
        .iter()
        .copied()
        .min_by(|a, b| keep(*a).total_cmp(&keep(*b)).then(a.cmp(b)))?;
    Some((index, keep(index)))
}

#[cfg(test)]
mod build_card_tests {
    use super::*;
    use crate::rules::get_valid_build_targets;
    use crate::state::Card;
    use rand_chacha::ChaCha12Rng;
    use rand_chacha::rand_core::SeedableRng;

    #[test]
    fn build_keeps_a_wild_when_a_matching_industry_card_exists() {
        let mut state = GameState::new(ChaCha12Rng::seed_from_u64(22), 4);
        let pid = state.current_player_id();
        let target = get_valid_build_targets(&state, pid)
            .into_iter()
            .next()
            .expect("opening state has a build target");
        state.players[pid].hand = vec![
            Card::Industry {
                industries: [target.ind; 2],
                n: 1,
            },
            Card::WildIndustry,
        ];
        let chosen = pick_build_card(&state, pid, &target, &[(0, 2.0), (1, 0.0)]);
        assert_eq!(chosen, Some((0, 2.0)));
    }
}

/// Top-K build candidates by one-ply score.
pub(crate) fn score_top_builds(
    state: &mut GameState,
    k: usize,
    plan: &Plan,
    targets: &[BuildTarget],
    keep_scores: &[(usize, f64)],
) -> Vec<Decision> {
    if k == 0 || keep_scores.is_empty() {
        return Vec::new();
    }
    let pid = state.current_player_id();
    let mut scored: Vec<(BuildTarget, f64)> = targets
        .iter()
        .cloned()
        .map(|target| {
            let score = score_build_candidate(state, pid, &target, plan);
            (target, score)
        })
        .collect();
    scored.sort_by(|a, b| b.1.total_cmp(&a.1));
    scored.truncate(k);

    let mut out = Vec::new();
    for (cand, score) in scored {
        if score == f64::NEG_INFINITY {
            continue;
        }
        let Some((card_index, card_score)) = pick_build_card(state, pid, &cand, keep_scores) else {
            continue;
        };
        let coal_opts = super::distinct_source_options(
            crate::rules::coal_source_options(state, cand.loc, cand.cost_coal as usize),
            |source: &crate::graph::CoalSource| (source.kind, source.key),
            SOURCE_VARIANTS,
        );
        let iron_opts = super::distinct_source_options(
            crate::rules::iron_source_options(state, cand.cost_iron as usize),
            |source: &crate::graph::IronSource| (source.key, source.free),
            SOURCE_VARIANTS,
        );
        let coal_opts = if coal_opts.is_empty() {
            vec![Vec::new()]
        } else {
            coal_opts
        };
        let iron_opts = if iron_opts.is_empty() {
            vec![Vec::new()]
        } else {
            iron_opts
        };

        let mut variants = 0;
        'variants: for coal in &coal_opts {
            for iron in &iron_opts {
                out.push(Decision {
                    mv: ResolvedMove::Build {
                        loc: cand.loc,
                        slot_index: cand.slot_index,
                        ind: cand.ind,
                        coal: coal.clone(),
                        iron: iron.clone(),
                        card_index,
                    },
                    score,
                    card_score,
                });
                variants += 1;
                if variants >= SOURCE_VARIANTS {
                    break 'variants;
                }
            }
        }
    }
    out
}
