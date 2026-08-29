//! Develop action scoring.

use super::board::player_has_buildable_card;
use super::context::EvalContext;
use super::plan::Plan;
use super::value::ScoreParts;
use super::{CardChoices, Decision};
use crate::data::{IndustryType, TileDef};
use crate::graph::find_iron_sources;
use crate::rules::{ResolvedMove, can_develop};
use crate::state::GameState;

/// Abstract value of developing one target tile away.
///
/// Develop removes a level-N tile and reveals the NEXT tile; its payoff is
/// the unlocked (often rail-era, double-scoring) tile that build can use
/// later. The raw value lives in the `vp` slot of [`ScoreParts`]; its scale
/// is intentionally well below an immediate build.
fn develop_target_value(
    state: &GameState,
    ctx: &EvalContext,
    ind: IndustryType,
    tile: TileDef,
    unlock_offset: usize,
    plan: &Plan,
) -> f64 {
    let w = &ctx.cfg.develop;
    let unlocked = state.players[ctx.pid].tile_after(ind, unlock_offset);
    let mut v = if tile.rail_era {
        w.rail_era_tile
    } else {
        w.canal_era_tile
    };
    v += w.per_level * tile.level as f64;
    // Premium for unlocking a double-scoring rail-era tile.
    if unlocked.is_some_and(|ut| ut.rail_era) {
        v += w.rail_unlock_bonus;
    }
    // Breweries are the economic engine (develop 1 -> build 2/3/4). In
    // Canal-Early grabbing the beer spot early is almost a must — but
    // developing costs iron, so only do it while iron is cheap.
    if ind == IndustryType::Brewery && tile.level == 1 {
        v += w.brewery_lv1_bonus;
        if ctx.phase == super::plan::Phase::CanalEarly {
            v += match state.iron_price() {
                p if p < 2 => w.iron_price_very_cheap_bonus,
                p if p <= 2 => w.iron_price_cheap_bonus,
                p if p <= 3 => w.iron_price_marginal_bonus,
                _ => w.iron_price_expensive_penalty,
            };
        }
    }
    // Canal-era develop wins the cross-era bonus; nudge it.
    if ctx.is_canal() {
        v += w.canal_bonus;
    }
    // Plan ("流派") alignment: developing toward the plan industry unlocks
    // higher-level tiles of the goods we intend to sell.
    if plan.industry == ind {
        v += w.plan_bonus;
    }
    // Only meaningful if we actually hold a card to build it next.
    if player_has_buildable_card(state, ctx.pid, ind) {
        v += w.buildable_card_bonus;
    }
    v - develop_guardrail_penalty(ctx, ind, tile.level)
}

/// Policy guardrail penalties (see `Guardrails`): developing breweries or
/// coal mines at level 2+ fights the intended engine, so make the scorer
/// treat it as clearly inferior.
fn develop_guardrail_penalty(ctx: &EvalContext, ind: IndustryType, level: u8) -> f64 {
    let g = &ctx.cfg.guardrails;
    if level < 2 {
        return 0.0;
    }
    let (base, per_level) = match ind {
        IndustryType::Brewery => (
            g.develop_brewery_penalty_base,
            g.develop_brewery_penalty_per_level,
        ),
        IndustryType::CoalMine => (
            g.develop_coal_penalty_base,
            g.develop_coal_penalty_per_level,
        ),
        _ => return 0.0,
    };
    base + per_level * (level.saturating_sub(2)) as f64
}

pub(super) fn score_develop_plans(
    state: &GameState,
    ctx: &EvalContext,
    plan: &Plan,
    card_choices: &CardChoices,
) -> Vec<Decision> {
    let w = &ctx.cfg.develop;
    let pid = ctx.pid;
    if !can_develop(state, pid) {
        return Vec::new();
    }
    let mut types = state.players[pid].developable_types();
    if ctx.cfg.guardrails.ban_develop_iron_lv2_plus {
        types.retain(|(ind, tile)| !(*ind == IndustryType::IronWorks && tile.level >= 2));
    }
    if ctx.cfg.guardrails.ban_develop_brewery_lv2_canal_early && ctx.is_canal() {
        types.retain(|(ind, tile)| !(*ind == IndustryType::Brewery && tile.level >= 2));
    }
    if types.is_empty() {
        return Vec::new();
    }
    let iron = find_iron_sources(state);
    if iron.is_empty() {
        return Vec::new();
    }
    let player = &state.players[pid];
    if !iron[0].free && iron[0].price as i32 > player.money {
        return Vec::new();
    }

    let mut scored: Vec<(IndustryType, f64)> = types
        .into_iter()
        .map(|(ind, tile)| (ind, develop_target_value(state, ctx, ind, tile, 1, plan)))
        .collect();
    scored.sort_by(|a, b| b.1.total_cmp(&a.1));

    // v4: emit up to SOURCE_VARIANTS plans, each led by a different target
    // industry, so search can pick between removal alternatives.
    let mut out = Vec::new();
    for primary_idx in 0..super::SOURCE_VARIANTS.min(scored.len()) {
        if let Some(decision) =
            develop_plan_for(state, ctx, plan, card_choices, w, pid, &iron, &scored, primary_idx)
        {
            out.push(decision);
        }
    }
    out
}

/// One Develop plan led by `scored[primary_idx]` as the first removed industry.
#[allow(clippy::too_many_arguments)]
fn develop_plan_for(
    state: &GameState,
    ctx: &EvalContext,
    plan: &Plan,
    card_choices: &CardChoices,
    w: &super::config::DevelopWeights,
    pid: usize,
    iron: &[crate::graph::IronSource],
    scored: &[(IndustryType, f64)],
    primary_idx: usize,
) -> Option<Decision> {
    let first = scored[primary_idx];
    let two_source_cost: i32 = iron
        .iter()
        .take(2)
        .map(|s| if s.free { 0 } else { s.price as i32 })
        .sum();
    let player = &state.players[pid];
    let can_afford_second = iron.len() >= 2 && two_source_cost <= player.money;

    let second = if can_afford_second {
        let best_other = scored
            .iter()
            .enumerate()
            .filter(|(idx, _)| *idx != primary_idx)
            .map(|(_, s)| (s.0, s.1))
            .next();
        let same_industry = state
            .players[pid]
            .tile_after(first.0, 1)
            .filter(|tile| {
                tile.can_develop
                    && !(ctx.cfg.guardrails.ban_develop_iron_lv2_plus
                        && first.0 == IndustryType::IronWorks
                        && tile.level >= 2)
            })
            .map(|tile| (first.0, develop_target_value(state, ctx, first.0, tile, 2, plan)));
        [best_other, same_industry]
            .into_iter()
            .flatten()
            .max_by(|a, b| a.1.total_cmp(&b.1))
    } else {
        None
    };

    let iron_cost: i32 = iron
        .iter()
        .take(if second.is_some() { 2 } else { 1 })
        .map(|s| if s.free { 0 } else { s.price as i32 })
        .sum();
    // Iron is scarce and develop burns a full turn: charge a real
    // opportunity cost so develop isn't a free high-score default.
    let iron_scarcity = if iron[0].free { 0.0 } else { w.iron_scarcity_cost };
    let second_value = second.map(|(_, score)| score * w.second_target_scale).unwrap_or(0.0);

    let mut parts = ScoreParts {
        vp: first.1 + second_value,
        money: -(iron_cost as f64),
        ..Default::default()
    };

    // Canal-era shaping: develop targets are worth more with a whole era
    // ahead (scale), but spending the action on a single tile is usually
    // tempo-inefficient — prefer double develops when legal/affordable.
    if ctx.is_canal() {
        parts.vp *= w.canal_scale;
        if second.is_none() {
            parts.strategic -= w.canal_single_target_penalty;
        } else {
            parts.strategic += w.canal_double_target_bonus;
        }
    }

    // Action-economy guardrail: develop is only worth the action budget it
    // consumes. Humans develop at most ~4 times in the canal era (8 tiles
    // via double-develops) and ~0-1 times in the rail era — beyond that,
    // develop crowds out builds/links/sells that actually score.
    let already = if ctx.is_canal() {
        state.players[pid].develops_in_canal
    } else {
        state.players[pid].develops_in_rail
    };
    let this_would_be = already + 1;
    let limit = if ctx.is_canal() {
        w.canal_count_limit
    } else {
        w.rail_count_limit
    };
    let over = (this_would_be as f64 - limit).max(0.0);
    parts.risk -= over * over * w.over_limit_steepness + over;

    parts.risk -= iron_scarcity;

    let iron_needed = if second.is_some() { 2 } else { 1 };
    let iron_choice = crate::rules::iron_source_options(state, iron_needed)
        .into_iter()
        .next()?;
    let card_index = card_choices.first().map(|(index, _)| *index)?;
    Some(Decision {
        mv: ResolvedMove::Develop {
            ind1: first.0,
            ind2: second.map(|(ind, _)| ind),
            iron: iron_choice,
            card_index,
        },
        score: parts.total(ctx),
        card_score: card_choices.first().map(|(_, s)| *s).unwrap_or(f64::INFINITY),
    })
}
