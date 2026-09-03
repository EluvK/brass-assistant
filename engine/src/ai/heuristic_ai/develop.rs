//! Develop action scoring based on the tiles actually removed from a mat.

use super::{CardChoices, Decision, SOURCE_VARIANTS, distinct_source_options};
use crate::data::{IndustryType, TileDef, industry_tiles};
use crate::heuristic_ai::context::define_era_round_factor;
use crate::rules::{ResolvedMove, can_develop, iron_source_options};
use crate::state::GameState;

/// Static, market-independent facts about one removable industry tile.
///
/// This is deliberately visible only to the heuristic parent module: it is a
/// scoring diagnostic, not a game-rule data type.
#[derive(Debug, Clone, Copy)]
pub(super) struct DevelopTileProfile {
    pub level: u8,
    pub vp_value: f64,
    pub income_value: f64,
    pub cost_penalty: f64,
}

/// Complete static/profile diagnostic for one industry level at one point in
/// the game. Callers can render this directly without reconstructing weights.
#[derive(Debug, Clone, Copy)]
#[cfg(test)]
pub(super) struct DevelopTileDiagnostic {
    pub industry: IndustryType,
    pub level: u8,
    pub profile: DevelopTileProfile,
    pub profile_raw: f64,
    pub removal_score: f64,
    pub vp_weight: f64,
    pub income_weight: f64,
    pub cost_weight: f64,
}

/// Read a tile's printed facts into the static development profile. Resource
/// requirements use the configured generic prices instead of board/market data.
pub(super) fn static_develop_profile(tile: TileDef) -> DevelopTileProfile {
    DevelopTileProfile {
        level: tile.level,
        vp_value: tile.vp as f64,
        income_value: tile.income as f64,
        cost_penalty: tile.cost as f64
            + tile.cost_coal as f64 * 3.0
            + tile.cost_iron as f64 * 3.0
            + tile.beers_to_sell.unwrap_or(0) as f64 * 3.0,
    }
}

const BAN_DEVELOP_BREWERY_LEVEL2_PLUS_IN_CANAL: bool = true;

define_era_round_factor!(
    DevelopVpFactor,
    canal: (0.9, 1.0),
    rail: (1.0, 1.1),
);

define_era_round_factor!(
    DevelopIncomeFactor,
    canal: (0.8, 0.2),
    rail: (0.2, 0.0),
);

define_era_round_factor!(
    DevelopCostFactor,
    canal: (1.6, 1.0),
    rail: (1.0, 0.5),
);

/// Stage-weighted value of removing a tile: costly tiles are better develop
/// fodder, while printed VP and income make a tile worth retaining to build.
fn profile_raw_value(profile: DevelopTileProfile, state: &GameState) -> f64 {
    let double_vp = if matches!(state.era, crate::data::Era::Canal) && profile.level != 1 {
        2.0
    } else {
        1.0
    };
    (8.0 - profile.vp_value * DevelopVpFactor::factor(state) * double_vp
        + profile.income_value * DevelopIncomeFactor::factor(state))
        / profile.cost_penalty
        * DevelopCostFactor::factor(state)
}

/// The highest printed level of an industry is its top level tile. Developing
/// past it has no strategic upside, so top level tiles are never candidates or
/// normalization anchors.
fn is_top_level_industry_tile(ind: IndustryType, tile: TileDef) -> bool {
    industry_tiles(ind)
        .iter()
        .map(|candidate| candidate.level)
        .max()
        == Some(tile.level)
}

/// Pottery levels 2 and 4 are deliberately attractive development targets.
/// Their fixed policy value must not also define the normalized-profile range.
fn has_fixed_develop_value(ind: IndustryType, tile: TileDef) -> bool {
    ind == IndustryType::Pottery && matches!(tile.level, 2 | 4)
}

/// The min/max range is static for a given evaluation phase: it covers every
/// non-top-level, non-fixed developable Cotton, Manufacturer and Pottery tile,
/// never merely the current player's mat. This keeps the score comparable
/// between game states without letting policy-fixed tiles distort the scale.
fn profile_raw_range(state: &GameState) -> (f64, f64) {
    [
        IndustryType::CottonMill,
        IndustryType::Manufacturer,
        IndustryType::Pottery,
    ]
    .into_iter()
    .flat_map(|ind| {
        industry_tiles(ind).iter().copied().filter(move |tile| {
            tile.can_develop
                && !is_top_level_industry_tile(ind, *tile)
                && !has_fixed_develop_value(ind, *tile)
        })
    })
    .map(|tile| profile_raw_value(static_develop_profile(tile), state))
    .fold((f64::INFINITY, f64::NEG_INFINITY), |(min, max), value| {
        (min.min(value), max.max(value))
    })
}

fn normalized_profile_value(tile: TileDef, state: &GameState, profile_range: (f64, f64)) -> f64 {
    let raw = profile_raw_value(static_develop_profile(tile), state);
    let (min, max) = profile_range;
    if (max - min).abs() <= f64::EPSILON {
        0.0
    } else {
        ((raw - min) / (max - min) * 1.6 - 0.8).clamp(-0.8, 0.8)
    }
}

/// Return every currently allowed development target's static vector and score
/// at the context's current era/round weights. Tests use this instead of
/// maintaining a partial hand-written table.
#[cfg(test)]
pub(super) fn develop_tile_diagnostics(state: &GameState) -> Vec<DevelopTileDiagnostic> {
    let profile_range = profile_raw_range(state);
    IndustryType::ALL
        .into_iter()
        .flat_map(|industry| {
            industry_tiles(industry)
                .iter()
                .copied()
                .filter(move |tile| allowed_develop_target(state.is_canal_era(), industry, *tile))
                .map(move |tile| {
                    let profile = static_develop_profile(tile);
                    let weights = (
                        DevelopVpFactor::factor(state),
                        DevelopIncomeFactor::factor(state),
                        DevelopCostFactor::factor(state),
                    );
                    DevelopTileDiagnostic {
                        industry,
                        level: tile.level,
                        profile,
                        profile_raw: profile_raw_value(profile, state),
                        removal_score: develop_target_value_with_profile_range(
                            state,
                            industry,
                            tile,
                            profile_range,
                        ),
                        vp_weight: weights.0,
                        income_weight: weights.1,
                        cost_weight: weights.2,
                    }
                })
        })
        .collect()
}

/// Default value for removing a specific tile. Only the three sellable
/// industries derive their value from the static profile; resource/brewery
/// defaults are explicit, configurable policy preferences.
fn develop_target_value_with_profile_range(
    state: &GameState,
    ind: IndustryType,
    tile: TileDef,
    profile_range: (f64, f64),
) -> f64 {
    if is_top_level_industry_tile(ind, tile) {
        return -1.0;
    }
    if has_fixed_develop_value(ind, tile) {
        return 0.8;
    }
    match ind {
        IndustryType::Brewery if tile.level == 1 => 0.75,
        IndustryType::Brewery => -0.5,
        IndustryType::CoalMine if tile.level == 1 => 0.8,
        IndustryType::CoalMine => -0.5,
        IndustryType::IronWorks if tile.level == 1 => 0.8,
        IndustryType::IronWorks => -0.3,
        IndustryType::CottonMill | IndustryType::Manufacturer | IndustryType::Pottery => {
            normalized_profile_value(tile, state, profile_range)
        }
    }
}

/// Convenience entry point for isolated callers such as unit tests. Production
/// planning uses the precomputed-range variant above.
#[cfg(test)]
fn develop_target_value(state: &GameState, ind: IndustryType, tile: TileDef) -> f64 {
    develop_target_value_with_profile_range(state, ind, tile, profile_raw_range(state))
}

fn allowed_develop_target(is_canal: bool, ind: IndustryType, tile: TileDef) -> bool {
    tile.can_develop
        && !is_top_level_industry_tile(ind, tile)
        // && !(ban_develop_iron_lv2_plus
        //     && ind == IndustryType::IronWorks
        //     && tile.level >= 2)
        && !(BAN_DEVELOP_BREWERY_LEVEL2_PLUS_IN_CANAL
            && is_canal
            && ind == IndustryType::Brewery
            && tile.level >= 2)
}

#[derive(Debug, Clone, Copy)]
struct DevelopTarget {
    ind: IndustryType,
    value: f64,
}

fn iron_baseline(iron: &[crate::graph::IronSource]) -> f64 {
    let average_paid_price = iron
        .iter()
        .map(|source| {
            if source.free {
                0.0
            } else {
                source.price as f64
            }
        })
        .sum::<f64>()
        / iron.len() as f64;
    (3.0 + (3.0 - average_paid_price) * 0.5).clamp(2.0, 4.0)
}

/// Quantize heuristic scores before they are compared/sorted.
fn quantize_score(score: f64) -> f64 {
    (score * 100.0).round() / 100.0
}

pub(super) fn score_develop_plans(state: &GameState, card_choices: &CardChoices) -> Vec<Decision> {
    let pid = state.current_player_id();
    let already_develop_cnt = match state.era {
        crate::data::Era::Canal => state.players[pid].develops_in_canal,
        crate::data::Era::Rail => {
            state.players[pid].develops_in_rail + state.players[pid].develops_in_canal
        }
    };
    if !can_develop(state, pid) || card_choices.is_empty() {
        return Vec::new();
    }
    // Static for this context; reuse it for every single- and double-develop
    // target rather than rescanning all profile tiles per candidate.
    let profile_range = profile_raw_range(state);

    let first_targets: Vec<DevelopTarget> = state.players[pid]
        .developable_types()
        .into_iter()
        .filter(|(ind, tile)| allowed_develop_target(state.is_canal_era(), *ind, *tile))
        .map(|(ind, tile)| DevelopTarget {
            ind,
            value: develop_target_value_with_profile_range(state, ind, tile, profile_range),
        })
        .collect();
    if first_targets.is_empty() {
        return Vec::new();
    }

    let mut decisions = Vec::new();
    // Score all single-develop plans independently from double plans.
    for &first in &first_targets {
        score_target_plan(
            state,
            card_choices,
            first,
            None,
            already_develop_cnt,
            &mut decisions,
        );
    }

    // Build and score each legal pair once. The player helper handles the
    // game-rule check (including same-industry uncovering); the additional
    // filters here only apply heuristic restrictions and canonical ordering.
    for (ind1, ind2) in state.players[pid].double_developable_types() {
        let Some(&first) = first_targets.iter().find(|target| target.ind == ind1) else {
            continue;
        };
        let tile = if ind1 == ind2 {
            state.players[pid].tile_after(ind2, 1)
        } else {
            state.players[pid].next_tile(ind2)
        };
        let Some(tile) = tile else { continue };
        if !allowed_develop_target(state.is_canal_era(), ind2, tile) {
            continue;
        }
        // For different industries, keep one canonical ordering. Same-
        // industry pairs remain ordered because the second tile is exposed
        // only after removing the first.
        if ind1 != ind2 && (ind2 as usize) < (ind1 as usize) {
            continue;
        }
        score_target_plan(
            state,
            card_choices,
            first,
            Some(DevelopTarget {
                ind: ind2,
                value: develop_target_value_with_profile_range(state, ind2, tile, profile_range),
            }),
            already_develop_cnt,
            &mut decisions,
        );
    }
    decisions.sort_by(|a, b| b.score.total_cmp(&a.score));
    decisions
}

fn score_target_plan(
    state: &GameState,
    card_choices: &CardChoices,
    first: DevelopTarget,
    second: Option<DevelopTarget>,
    already_develop_cnt: u8,
    decisions: &mut Vec<Decision>,
) {
    let iron_needed = usize::from(second.is_some()) + 1;
    let target_value = first.value + second.map(|target| target.value).unwrap_or(0.0);
    for iron in distinct_source_options(
        iron_source_options(state, iron_needed),
        |source| source.key,
        SOURCE_VARIANTS,
    ) {
        let actual_cost: i32 = iron
            .iter()
            .map(|source| if source.free { 0 } else { source.price as i32 })
            .sum();
        if actual_cost > state.players[state.current_player_id()].money {
            continue;
        }
        let second_industry_bonus = match second {
            Some(ind) if ind.ind == first.ind => 0.2, // same as first, more valuable
            Some(_) => 0.1,
            None => -0.5,
        };
        let score = iron_baseline(&iron) + target_value + second_industry_bonus
            - already_develop_cnt as f64 * 0.2;
        decisions.push(Decision {
            mv: ResolvedMove::Develop {
                ind1: first.ind,
                ind2: second.map(|target| target.ind),
                iron,
                card_index: card_choices[0].0,
            },
            score: quantize_score(score.clamp(0.0, 6.0)),
            card_score: card_choices[0].1,
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::Era;
    use rand_chacha::ChaCha12Rng;
    use rand_chacha::rand_core::SeedableRng;

    fn ctx_for(era: Era, round: usize) -> GameState {
        let mut state = GameState::new(ChaCha12Rng::seed_from_u64(17), 2);
        state.era = era;
        state.round = round;
        state
    }

    #[test]
    fn diagnostic_profile_ranking_has_required_default_extremes() {
        for (era, round) in [
            (Era::Canal, 1),
            // (Era::Canal, 8),
            // (Era::Rail, 1),
            // (Era::Rail, 8),
        ] {
            let state = ctx_for(era, round);
            let diagnostics = develop_tile_diagnostics(&state);
            assert_eq!(
                diagnostics.len(),
                IndustryType::ALL
                    .into_iter()
                    .map(|ind| {
                        industry_tiles(ind)
                            .iter()
                            .filter(|tile| {
                                allowed_develop_target(state.is_canal_era(), ind, **tile)
                            })
                            .count()
                    })
                    .sum::<usize>()
            );
            println!("{era:?} round {round}",);
            for row in diagnostics {
                println!(
                    "  {:?} Lv{}: vp={:.1}, income={:.1}, cost={:.1}, raw={:.3}, score={:.3}; vp_w={:.3}, income_w={:.3}, cost_w={:.3}",
                    row.industry,
                    row.level,
                    row.profile.vp_value,
                    row.profile.income_value,
                    row.profile.cost_penalty,
                    row.profile_raw,
                    row.removal_score,
                    row.vp_weight,
                    row.income_weight,
                    row.cost_weight,
                );
                if matches!(
                    row.industry,
                    IndustryType::CottonMill | IndustryType::Manufacturer | IndustryType::Pottery
                ) {
                    assert!((-1.0..=1.0).contains(&row.removal_score));
                }
            }
        }

        let state = ctx_for(Era::Rail, 5);
        let brewery_one = develop_target_value(
            &state,
            IndustryType::Brewery,
            industry_tiles(IndustryType::Brewery)[0],
        );
        let coal_two = develop_target_value(
            &state,
            IndustryType::CoalMine,
            industry_tiles(IndustryType::CoalMine)[1],
        );
        assert_eq!(brewery_one, 0.75);
        assert_eq!(coal_two, -0.5);

        let pottery_two = develop_target_value(
            &state,
            IndustryType::Pottery,
            industry_tiles(IndustryType::Pottery)[1],
        );
        let pottery_four = develop_target_value(
            &state,
            IndustryType::Pottery,
            industry_tiles(IndustryType::Pottery)[3],
        );
        assert_eq!(pottery_two, 0.8);
        assert_eq!(pottery_four, 0.8);

        let manufacturer_three = develop_target_value(
            &state,
            IndustryType::Manufacturer,
            industry_tiles(IndustryType::Manufacturer)[2],
        );
        let manufacturer_four = develop_target_value(
            &state,
            IndustryType::Manufacturer,
            industry_tiles(IndustryType::Manufacturer)[3],
        );
        assert!(manufacturer_three > 0.0 && manufacturer_four > 0.0);
    }

    #[test]
    fn cheap_iron_never_lowers_the_develop_baseline() {
        let cheap = crate::graph::IronSource {
            key: 0,
            free: false,
            price: 1,
        };
        let expensive = crate::graph::IronSource {
            key: 0,
            free: false,
            price: 5,
        };
        assert!(iron_baseline(&[cheap]) >= iron_baseline(&[expensive]));
        assert_eq!(iron_baseline(&[cheap]), 4.0);
        assert_eq!(iron_baseline(&[expensive]), 2.0);
    }

    #[test]
    fn canal_round_one_brewery_pair_matches_coal_iron_pair() {
        let state = ctx_for(Era::Canal, 1);
        let decisions = score_develop_plans(&state, &vec![(0, 0.0)]);

        let brewery_pair = decisions
            .iter()
            .filter_map(|decision| match decision.mv {
                ResolvedMove::Develop {
                    ind1: IndustryType::Brewery,
                    ind2: Some(IndustryType::Brewery),
                    ..
                } => Some(decision.score),
                _ => None,
            })
            .max_by(f64::total_cmp)
            .expect("Canal round one should allow developing two level-one breweries");
        let coal_iron_pair = decisions
            .iter()
            .filter_map(|decision| match decision.mv {
                ResolvedMove::Develop {
                    ind1: IndustryType::CoalMine,
                    ind2: Some(IndustryType::IronWorks),
                    ..
                } => Some(decision.score),
                _ => None,
            })
            .max_by(f64::total_cmp)
            .expect("Canal round one should allow developing one coal mine and one iron works");

        println!("brewery={brewery_pair}, coal_iron={coal_iron_pair}");
        assert_eq!(brewery_pair, coal_iron_pair);
    }

    #[test]
    fn profile_values_are_bounded_for_every_phase_and_removable_tile() {
        for (era, round) in [
            (Era::Canal, 1),
            (Era::Canal, 8),
            (Era::Rail, 1),
            (Era::Rail, 8),
        ] {
            let state = ctx_for(era, round);
            for ind in [
                IndustryType::CottonMill,
                IndustryType::Manufacturer,
                IndustryType::Pottery,
            ] {
                for tile in industry_tiles(ind)
                    .iter()
                    .copied()
                    .filter(|tile| tile.can_develop)
                {
                    assert!((-1.0..=1.0).contains(&develop_target_value(&state, ind, tile)));
                }
            }
        }
    }

    #[test]
    fn hard_develop_bans_remain_filters_not_score_penalties() {
        let mut state = ctx_for(Era::Canal, 1);
        let pid = state.current_player_id();
        state.players[pid].develop_tile(IndustryType::IronWorks, state.era);
        state.players[pid].develop_tile(IndustryType::Brewery, state.era);
        state.players[pid].develop_tile(IndustryType::Brewery, state.era);
        // let iron_two = state.players[pid]
        // .next_tile(IndustryType::IronWorks)
        // .unwrap();
        let brewery_two = state.players[pid].next_tile(IndustryType::Brewery).unwrap();
        // assert!(!allowed_develop_target(
        //     state.is_canal_era(),
        //     IndustryType::IronWorks,
        //     iron_two
        // ));
        assert!(!allowed_develop_target(
            state.is_canal_era(),
            IndustryType::Brewery,
            brewery_two
        ));
    }

    #[test]
    fn top_level_industry_tiles_are_not_develop_candidates_or_profile_values() {
        let state = ctx_for(Era::Rail, 5);
        for ind in IndustryType::ALL {
            let top_level = industry_tiles(ind)
                .iter()
                .copied()
                .max_by_key(|tile| tile.level)
                .expect("every industry must have at least one tile");
            assert!(is_top_level_industry_tile(ind, top_level));
            assert!(!allowed_develop_target(
                state.is_canal_era(),
                ind,
                top_level
            ));
            assert_eq!(develop_target_value(&state, ind, top_level), -1.0);
        }
    }

    #[test]
    fn candidates_are_sorted_by_removal_value_and_scores_are_clamped() {
        let state = ctx_for(Era::Rail, 5);
        let decisions = score_develop_plans(&state, &vec![(0, 0.0)]);
        assert!(!decisions.is_empty());
        assert!(
            decisions
                .iter()
                .all(|decision| (0.0..=6.0).contains(&decision.score))
        );
        assert!(decisions.iter().any(|decision| {
            matches!(
                decision.mv,
                ResolvedMove::Develop {
                    ind1: IndustryType::Brewery,
                    ind2: Some(IndustryType::Brewery),
                    ..
                }
            )
        }));
        assert!(decisions.iter().all(|decision| {
            !matches!(
                decision.mv,
                ResolvedMove::Develop {
                    ind1,
                    ind2: Some(ind2),
                    ..
                } if ind1 != ind2 && (ind1 as usize) > (ind2 as usize)
            )
        }));
    }
}
