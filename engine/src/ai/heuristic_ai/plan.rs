//! Era-phase strategy profile and quantified production-plan selection.

use crate::data::{Era, IndustryType};
use crate::map::{ALL_LOCATIONS, CITY_COUNT, city_slots};
use crate::state::{Card, GameState};

use super::owned_beer_barrels;

/// Four strategy phases: each era is split into an early and late half.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Phase {
    CanalEarly,
    CanalLate,
    RailEarly,
    RailLate,
}

/// Per-phase evaluation weights used throughout the heuristic.
pub struct EraProfile {
    pub phase: Phase,
    pub income_w: f64,
    pub money_w: f64,
    pub network_w: f64,
    pub alpha: f64,
}

pub fn era_phase(state: &GameState) -> Phase {
    match state.era {
        Era::Canal if state.round <= 4 => Phase::CanalEarly,
        Era::Canal => Phase::CanalLate,
        Era::Rail if state.round <= 4 => Phase::RailEarly,
        Era::Rail => Phase::RailLate,
    }
}

pub fn era_profile(state: &GameState) -> EraProfile {
    let phase = era_phase(state);
    let rounds = state.rounds_remaining();
    let frac = (rounds / 8.0).clamp(0.0, 1.0);
    match phase {
        Phase::CanalEarly | Phase::CanalLate => EraProfile {
            phase,
            income_w: super::BASE_INCOME_WEIGHT * (1.8 + 0.6 * frac),
            money_w: super::BASE_MONEY_WEIGHT * 0.55,
            network_w: 0.1,
            alpha: 0.6,
        },
        Phase::RailEarly => EraProfile {
            phase,
            income_w: super::BASE_INCOME_WEIGHT * (1.2 + 0.5 * frac),
            money_w: super::BASE_MONEY_WEIGHT * 0.8,
            network_w: 1.0,
            alpha: 0.6,
        },
        Phase::RailLate => EraProfile {
            phase,
            income_w: 0.0,
            money_w: 0.2,
            network_w: 0.85,
            alpha: 0.35,
        },
    }
}

#[cfg(test)]
mod tests {
    #[allow(unused_imports)]
    use super::*;
    use rand_chacha::rand_core::SeedableRng;
    #[test]
    fn show_era_profile() {
        let mut state = crate::state::GameState::new(rand_chacha::ChaCha12Rng::seed_from_u64(7), 4);
        for era in [Era::Canal, Era::Rail] {
            state.era = era;
            for round in 1..=8 {
                state.round = round;
                let profile = era_profile(&state);
                println!(
                    "era={:?} round={} phase={:?} income_w={:.3} money_w={:.3} network_w={:.3} alpha={:.3}",
                    era,
                    round,
                    profile.phase,
                    profile.income_w,
                    profile.money_w,
                    profile.network_w,
                    profile.alpha
                );
            }
        }
    }
}

/// The player's target sellable industry, achievable tile count, and beer need.
#[derive(Clone, Copy, Debug)]
pub struct Plan {
    pub industry: IndustryType,
    pub count: usize,
    pub beer_needed: usize,
}

fn vacant_board_slots(state: &GameState) -> [usize; 6] {
    let mut counts = [0usize; 6];
    for loc in ALL_LOCATIONS.iter().take(CITY_COUNT) {
        for (slot_idx, allowed) in city_slots(*loc).iter().enumerate() {
            let Some(key) = state.city_slot_key(*loc, slot_idx) else {
                continue;
            };
            if state.city_tiles[key].is_some() {
                continue;
            }
            for ind in *allowed {
                counts[*ind as usize] += 1;
            }
        }
    }
    counts
}

fn hand_support(state: &GameState, pid: usize, ind: IndustryType) -> usize {
    let mut support = 0usize;
    for card in &state.players[pid].hand {
        match card {
            Card::Location(loc) if !loc.is_farm() => {
                for (slot_idx, allowed) in city_slots(*loc).iter().enumerate() {
                    if !allowed.contains(&ind) {
                        continue;
                    }
                    if state
                        .city_slot_key(*loc, slot_idx)
                        .is_some_and(|key| state.city_tiles[key].is_none())
                    {
                        support += 1;
                        break;
                    }
                }
            }
            Card::Industry { .. } if card.is_industry(ind) => support += 1,
            Card::WildIndustry => support += 1,
            _ => {}
        }
    }
    support.min(3)
}

fn plan_flip_probability(state: &GameState, pid: usize, ind: IndustryType) -> f64 {
    if !ind.is_sellable() {
        return 0.0;
    }
    if !state.merchants.iter().any(|mt| mt.accepts(ind)) {
        return 0.15;
    }
    let beer_ok = owned_beer_barrels(state, pid) > 0
        || state
            .merchants
            .iter()
            .any(|mt| mt.has_beer && mt.accepts(ind));
    if beer_ok { 0.7 } else { 0.3 }
}

/// Compute the production plan from board capacity, remaining tiles, cards,
/// sell potential, and beer availability.
pub fn compute_plan(state: &GameState, pid: usize) -> Plan {
    let slots = vacant_board_slots(state);
    let mut best_industry = IndustryType::CottonMill;
    let mut best_score = f64::NEG_INFINITY;
    let mut best_count = 0usize;

    for ind in IndustryType::ALL {
        if !ind.is_sellable() {
            continue;
        }
        let remaining = state.players[pid].remaining_count(ind);
        let avail = slots[ind as usize];
        if remaining == 0 || avail == 0 {
            continue;
        }
        let plan_count = remaining.min(avail);
        let (vp_sum, beers) = (0..plan_count)
            .filter_map(|off| state.players[pid].tile_after(ind, off))
            .fold((0.0, 0usize), |(vp, beer), tile| {
                (
                    vp + tile.vp as f64,
                    beer + tile.beers_to_sell.unwrap_or(0) as usize,
                )
            });
        let avg_vp = vp_sum / plan_count as f64;
        let own_beer = owned_beer_barrels(state, pid);
        let beer_factor = if own_beer >= beers {
            1.0
        } else {
            (0.4 + 0.6 * (own_beer as f64 / beers.max(1) as f64)).min(1.0)
        };
        let hand_factor = 0.5 + 0.25 * hand_support(state, pid, ind) as f64;
        let score = plan_count as f64
            * avg_vp
            * plan_flip_probability(state, pid, ind)
            * beer_factor
            * hand_factor;
        if score > best_score {
            best_score = score;
            best_industry = ind;
            best_count = plan_count;
        }
    }

    if best_score == f64::NEG_INFINITY {
        return Plan {
            industry: IndustryType::CottonMill,
            count: 0,
            beer_needed: 0,
        };
    }
    let beer_needed = (0..best_count)
        .filter_map(|off| state.players[pid].tile_after(best_industry, off))
        .map(|tile| tile.beers_to_sell.unwrap_or(0) as usize)
        .sum();
    Plan {
        industry: best_industry,
        count: best_count,
        beer_needed,
    }
}
