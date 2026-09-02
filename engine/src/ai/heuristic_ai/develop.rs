//! Develop action scoring based on the tiles actually removed from a mat.

use super::context::{EvalContext, era_round_factor, round_factor};
use super::value::ScoreParts;
use super::{CardChoices, Decision, SOURCE_VARIANTS, distinct_source_options};
use crate::data::{IndustryType, TileDef, industry_tiles};
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
    pub era_frac: f64,
    pub round_progress: f64,
    pub vp_weight: f64,
    pub income_weight: f64,
    pub cost_weight: f64,
}

/// Read a tile's printed facts into the static development profile. Resource
/// requirements use the configured generic prices instead of board/market data.
pub(super) fn static_develop_profile(
    tile: TileDef,
    w: &super::config::DevelopWeights,
) -> DevelopTileProfile {
    DevelopTileProfile {
        level: tile.level,
        vp_value: tile.vp as f64,
        income_value: tile.income as f64,
        cost_penalty: tile.cost as f64
            + tile.cost_coal as f64 * w.static_coal_price
            + tile.cost_iron as f64 * w.static_iron_price
            + tile.beers_to_sell.unwrap_or(0) as f64 * w.static_iron_price, // whatever,
    }
}

#[derive(Debug, Clone, Copy)]
struct ProfileWeights {
    vp: f64,
    income: f64,
    cost: f64,
}

/// Independently tunable static-profile weights at the current numbered round.
/// These intentionally do not reuse global ScoreParts money/income conversion.
fn profile_weights(ctx: &EvalContext) -> ProfileWeights {
    let w = &ctx.cfg.develop.profile;
    let progress = round_factor!(ctx.round_progress, 0.0_f64, 1.0_f64);
    ProfileWeights {
        vp: era_round_factor!(
            ctx.is_canal(),
            progress,
            w.canal_vp_start,
            w.canal_vp_end,
            w.rail_vp_start,
            w.rail_vp_end,
        ),
        income: era_round_factor!(
            ctx.is_canal(),
            progress,
            w.canal_income_start,
            w.canal_income_end,
            w.rail_income_start,
            w.rail_income_end,
        ),
        cost: era_round_factor!(
            ctx.is_canal(),
            progress,
            w.canal_cost_start,
            w.canal_cost_end,
            w.rail_cost_start,
            w.rail_cost_end,
        ),
    }
}

/// Stage-weighted value of removing a tile: costly tiles are better develop
/// fodder, while printed VP and income make a tile worth retaining to build.
fn profile_raw_value(profile: DevelopTileProfile, ctx: &EvalContext) -> f64 {
    let weights = profile_weights(ctx);
    let double_vp = if ctx.is_canal() && profile.level != 1 {
        2.0
    } else {
        1.0
    };
    (8.0 - profile.vp_value * weights.vp * double_vp + profile.income_value * weights.income)
        / profile.cost_penalty
        * weights.cost
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
fn profile_raw_range(ctx: &EvalContext) -> (f64, f64) {
    [
        IndustryType::CottonMill,
        IndustryType::Manufacturer,
        IndustryType::Pottery,
    ]
    .into_iter()
    .flat_map(|ind| {
        industry_tiles(ind)
            .iter()
            .copied()
            .filter(move |tile| {
                tile.can_develop
                    && !is_top_level_industry_tile(ind, *tile)
                    && !has_fixed_develop_value(ind, *tile)
            })
    })
    .map(|tile| profile_raw_value(static_develop_profile(tile, &ctx.cfg.develop), ctx))
    .fold((f64::INFINITY, f64::NEG_INFINITY), |(min, max), value| {
        (min.min(value), max.max(value))
    })
}

fn normalized_profile_value(
    tile: TileDef,
    ctx: &EvalContext,
    profile_range: (f64, f64),
) -> f64 {
    let raw = profile_raw_value(static_develop_profile(tile, &ctx.cfg.develop), ctx);
    let (min, max) = profile_range;
    if (max - min).abs() <= f64::EPSILON {
        0.0
    } else {
        ((raw - min) / (max - min) * 2.0 - 1.0).clamp(-1.0, 1.0)
    }
}

/// Return every currently allowed development target's static vector and score
/// at the context's current era/round weights. Tests use this instead of
/// maintaining a partial hand-written table.
#[cfg(test)]
pub(super) fn develop_tile_diagnostics(ctx: &EvalContext) -> Vec<DevelopTileDiagnostic> {
    let profile_range = profile_raw_range(ctx);
    IndustryType::ALL
        .into_iter()
        .flat_map(|industry| {
            industry_tiles(industry)
                .iter()
                .copied()
                .filter(move |tile| allowed_develop_target(ctx, industry, *tile))
                .map(move |tile| {
                let profile = static_develop_profile(tile, &ctx.cfg.develop);
                let weights = profile_weights(ctx);
                DevelopTileDiagnostic {
                    industry,
                    level: tile.level,
                    profile,
                    profile_raw: profile_raw_value(profile, ctx),
                    removal_score: develop_target_value_with_profile_range(
                        ctx,
                        industry,
                        tile,
                        profile_range,
                    ),
                    era_frac: ctx.era_frac,
                    round_progress: ctx.round_progress,
                    vp_weight: weights.vp,
                    income_weight: weights.income,
                    cost_weight: weights.cost,
                }
                }
            )
        })
        .collect()
}

/// Default value for removing a specific tile. Only the three sellable
/// industries derive their value from the static profile; resource/brewery
/// defaults are explicit, configurable policy preferences.
fn develop_target_value_with_profile_range(
    ctx: &EvalContext,
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
        IndustryType::Brewery if tile.level == 1 => 0.8,
        IndustryType::Brewery => -0.5,
        IndustryType::CoalMine if tile.level == 1 => 0.2,
        IndustryType::CoalMine => -0.5,
        IndustryType::IronWorks if tile.level == 1 => 0.2,
        IndustryType::IronWorks => -0.3,
        IndustryType::CottonMill | IndustryType::Manufacturer | IndustryType::Pottery => {
            normalized_profile_value(tile, ctx, profile_range)
        }
    }
}

/// Convenience entry point for isolated callers such as unit tests. Production
/// planning uses the precomputed-range variant above.
#[cfg(test)]
fn develop_target_value(ctx: &EvalContext, ind: IndustryType, tile: TileDef) -> f64 {
    develop_target_value_with_profile_range(ctx, ind, tile, profile_raw_range(ctx))
}

fn allowed_develop_target(ctx: &EvalContext, ind: IndustryType, tile: TileDef) -> bool {
    tile.can_develop
        && !is_top_level_industry_tile(ind, tile)
        && !(ctx.cfg.guardrails.ban_develop_iron_lv2_plus
            && ind == IndustryType::IronWorks
            && tile.level >= 2)
        && !(ctx.cfg.guardrails.ban_develop_brewery_lv2_canal_early
            && ctx.is_canal()
            && ind == IndustryType::Brewery
            && tile.level >= 2)
}

#[derive(Debug, Clone, Copy)]
struct DevelopTarget {
    ind: IndustryType,
    value: f64,
}

fn iron_baseline(ctx: &EvalContext, iron: &[crate::graph::IronSource]) -> f64 {
    let w = &ctx.cfg.develop;
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
    (w.base_score_center
        + (w.iron_price_reference - average_paid_price) * w.iron_price_score_per_pound)
        .clamp(w.base_score_min, w.base_score_max)
}

fn score_develop_parts(
    ctx: &EvalContext,
    target_value: f64,
    iron: &[crate::graph::IronSource],
) -> ScoreParts {
    let actual_iron_cost: f64 = iron
        .iter()
        .map(|source| {
            if source.free {
                0.0
            } else {
                source.price as f64
            }
        })
        .sum();
    ScoreParts {
        // This is the value of the removed development object plus the action
        // baseline. Real cash spent on iron remains separately in `money`.
        strategic: iron_baseline(ctx, iron) + target_value,
        money: -actual_iron_cost,
        ..Default::default()
    }
}

pub(super) fn score_develop_plans(
    state: &GameState,
    ctx: &EvalContext,
    card_choices: &CardChoices,
) -> Vec<Decision> {
    let pid = ctx.pid;
    if !can_develop(state, pid) || card_choices.is_empty() {
        return Vec::new();
    }
    // Static for this context; reuse it for every single- and double-develop
    // target rather than rescanning all profile tiles per candidate.
    let profile_range = profile_raw_range(ctx);

    let first_targets: Vec<DevelopTarget> = state.players[pid]
        .developable_types()
        .into_iter()
        .filter(|(ind, tile)| allowed_develop_target(ctx, *ind, *tile))
        .map(|(ind, tile)| DevelopTarget {
            ind,
            value: develop_target_value_with_profile_range(ctx, ind, tile, profile_range),
        })
        .collect();
    if first_targets.is_empty() {
        return Vec::new();
    }

    let mut decisions = Vec::new();
    for first in first_targets {
        score_target_plan(state, ctx, card_choices, first, None, &mut decisions);
        let mut player_after_first = state.players[pid].clone();
        player_after_first
            .consume_tile(first.ind)
            .expect("first develop target must remain on the player mat");
        for (ind, tile) in player_after_first
            .developable_types()
            .into_iter()
            .filter(|(ind, tile)| allowed_develop_target(ctx, *ind, *tile))
        {
            // For two different industries, A -> B and B -> A remove the
            // same pair of tiles and spend the same iron. Keep one canonical
            // ordering; same-industry pairs remain because their second tile
            // is the one uncovered after removing the first.
            if ind != first.ind && (ind as usize) < (first.ind as usize) {
                continue;
            }
            score_target_plan(
                state,
                ctx,
                card_choices,
                first,
                Some(DevelopTarget {
                    ind,
                    value: develop_target_value_with_profile_range(ctx, ind, tile, profile_range),
                }),
                &mut decisions,
            );
        }
    }
    decisions.sort_by(|a, b| b.score.total_cmp(&a.score));
    decisions
}

fn score_target_plan(
    state: &GameState,
    ctx: &EvalContext,
    card_choices: &CardChoices,
    first: DevelopTarget,
    second: Option<DevelopTarget>,
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
        if actual_cost > state.players[ctx.pid].money {
            continue;
        }
        let parts = score_develop_parts(ctx, target_value, &iron);
        decisions.push(Decision {
            mv: ResolvedMove::Develop {
                ind1: first.ind,
                ind2: second.map(|target| target.ind),
                iron,
                card_index: card_choices[0].0,
            },
            score: parts.total(ctx).clamp(0.0, 6.0),
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

    fn ctx_for(era: Era, round: u32) -> (GameState, super::super::config::HeuristicConfig) {
        let mut state = GameState::new(ChaCha12Rng::seed_from_u64(17), 2);
        state.era = era;
        state.round = round;
        (state, super::super::config::HeuristicConfig::default())
    }

    #[test]
    fn static_profiles_use_fixed_generic_resource_prices() {
        let cfg = super::super::config::HeuristicConfig::default();
        let profile =
            static_develop_profile(industry_tiles(IndustryType::CottonMill)[2], &cfg.develop);
        assert_eq!(profile.vp_value, 9.0);
        assert_eq!(profile.income_value, 3.0);
        assert_eq!(profile.cost_penalty, 25.0); // £16 + coal £3 + iron £3 + beers £3
    }

    #[test]
    fn diagnostic_profile_ranking_has_required_default_extremes() {
        for (era, round) in [
            (Era::Canal, 1),
            // (Era::Canal, 8),
            // (Era::Rail, 1),
            // (Era::Rail, 8),
        ] {
            let (state, cfg) = ctx_for(era, round);
            let ctx = EvalContext::new(&state, state.current_player_id(), &cfg);
            let diagnostics = develop_tile_diagnostics(&ctx);
            assert_eq!(
                diagnostics.len(),
                IndustryType::ALL
                    .into_iter()
                    .map(|ind| {
                        industry_tiles(ind)
                            .iter()
                            .filter(|tile| allowed_develop_target(&ctx, ind, **tile))
                            .count()
                    })
                    .sum()
            );
            println!(
                "{era:?} round {round}: era_frac={:.3}, round_progress={:.3}",
                ctx.era_frac, ctx.round_progress,
            );
            for row in diagnostics {
                println!(
                    "  {:?} Lv{}: vp={:.1}, income={:.1}, cost={:.1}, raw={:.3}, score={:.3}; era_frac={:.3}, round_progress={:.3}, vp_w={:.3}, income_w={:.3}, cost_w={:.3}",
                    row.industry,
                    row.level,
                    row.profile.vp_value,
                    row.profile.income_value,
                    row.profile.cost_penalty,
                    row.profile_raw,
                    row.removal_score,
                    row.era_frac,
                    row.round_progress,
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

        let (state, cfg) = ctx_for(Era::Rail, 5);
        let ctx = EvalContext::new(&state, state.current_player_id(), &cfg);
        let brewery_one = develop_target_value(
            &ctx,
            IndustryType::Brewery,
            industry_tiles(IndustryType::Brewery)[0],
        );
        let coal_two = develop_target_value(
            &ctx,
            IndustryType::CoalMine,
            industry_tiles(IndustryType::CoalMine)[1],
        );
        assert_eq!(brewery_one, 0.8);
        assert_eq!(coal_two, -0.5);

        let pottery_two = develop_target_value(
            &ctx,
            IndustryType::Pottery,
            industry_tiles(IndustryType::Pottery)[1],
        );
        let pottery_four = develop_target_value(
            &ctx,
            IndustryType::Pottery,
            industry_tiles(IndustryType::Pottery)[3],
        );
        assert_eq!(pottery_two, 0.8);
        assert_eq!(pottery_four, 0.8);

        let manufacturer_three = develop_target_value(
            &ctx,
            IndustryType::Manufacturer,
            industry_tiles(IndustryType::Manufacturer)[2],
        );
        let manufacturer_four = develop_target_value(
            &ctx,
            IndustryType::Manufacturer,
            industry_tiles(IndustryType::Manufacturer)[3],
        );
        assert!(manufacturer_three > 0.0 && manufacturer_four > 0.0);
    }

    #[test]
    fn cheap_iron_never_lowers_the_develop_baseline() {
        let (state, cfg) = ctx_for(Era::Canal, 1);
        let ctx = EvalContext::new(&state, state.current_player_id(), &cfg);
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
        assert!(iron_baseline(&ctx, &[cheap]) >= iron_baseline(&ctx, &[expensive]));
        assert_eq!(iron_baseline(&ctx, &[cheap]), 4.0);
        assert_eq!(iron_baseline(&ctx, &[expensive]), 2.0);
    }

    #[test]
    fn profile_values_are_bounded_for_every_phase_and_removable_tile() {
        for (era, round) in [
            (Era::Canal, 1),
            (Era::Canal, 8),
            (Era::Rail, 1),
            (Era::Rail, 8),
        ] {
            let (state, cfg) = ctx_for(era, round);
            let ctx = EvalContext::new(&state, state.current_player_id(), &cfg);
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
                    assert!((-1.0..=1.0).contains(&develop_target_value(&ctx, ind, tile)));
                }
            }
        }
    }

    #[test]
    fn hard_develop_bans_remain_filters_not_score_penalties() {
        let (mut state, cfg) = ctx_for(Era::Canal, 1);
        let pid = state.current_player_id();
        state.players[pid].consume_tile(IndustryType::IronWorks);
        state.players[pid].consume_tile(IndustryType::Brewery);
        state.players[pid].consume_tile(IndustryType::Brewery);
        let ctx = EvalContext::new(&state, pid, &cfg);
        let iron_two = state.players[pid]
            .next_tile(IndustryType::IronWorks)
            .unwrap();
        let brewery_two = state.players[pid].next_tile(IndustryType::Brewery).unwrap();
        assert!(!allowed_develop_target(
            &ctx,
            IndustryType::IronWorks,
            iron_two
        ));
        assert!(!allowed_develop_target(
            &ctx,
            IndustryType::Brewery,
            brewery_two
        ));
    }

    #[test]
    fn top_level_industry_tiles_are_not_develop_candidates_or_profile_values() {
        let (state, cfg) = ctx_for(Era::Rail, 5);
        let ctx = EvalContext::new(&state, state.current_player_id(), &cfg);
        for ind in IndustryType::ALL {
            let top_level = industry_tiles(ind)
                .iter()
                .copied()
                .max_by_key(|tile| tile.level)
                .expect("every industry must have at least one tile");
            assert!(is_top_level_industry_tile(ind, top_level));
            assert!(!allowed_develop_target(&ctx, ind, top_level));
            assert_eq!(develop_target_value(&ctx, ind, top_level), -1.0);
        }
    }

    #[test]
    fn candidates_are_sorted_by_removal_value_and_scores_are_clamped() {
        let (state, cfg) = ctx_for(Era::Rail, 5);
        let pid = state.current_player_id();
        let ctx = EvalContext::new(&state, pid, &cfg);
        let decisions = score_develop_plans(&state, &ctx, &vec![(0, 0.0)]);
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
