//! Sell scoring consumes the validated canonical sell candidates.
use super::context::EvalContext;
use super::{CardChoices, Decision};
use crate::rules::{ResolvedMove, legal_resolved_moves};
use crate::state::GameState;

pub(super) fn score_sell_plans(
    state: &GameState,
    ctx: &EvalContext,
    cards: &CardChoices,
) -> Vec<Decision> {
    let Some(card_index) = cards.first().map(|x| x.0) else {
        return Vec::new();
    };
    let mut work = state.clone();
    let mut out: Vec<_> = legal_resolved_moves(&mut work)
        .into_iter()
        .filter_map(|mv| {
            let ResolvedMove::Sell {
                keys,
                beer_sources,
                free_develop,
                ..
            } = &mv
            else {
                return None;
            };
            let mut score = 0.0;
            for &key in keys {
                if let Some(tile) = state.city_tiles[key].as_ref() {
                    score += tile.def.vp as f64
                        + tile.def.income as f64 * ctx.cfg.sell.tile_income_share;
                }
            }
            for source in beer_sources {
                if source.kind == crate::graph::BeerSourceKind::Merchant {
                    if let Some(i) = source.merchant_idx {
                        score += match crate::map::merchant_bonus_at(state.merchants[i].loc) {
                            crate::map::MerchantBonus::Vp(v) => v as f64 * ctx.cfg.value.vp,
                            crate::map::MerchantBonus::Money(v) => v as f64,
                            crate::map::MerchantBonus::Income(v) => v as f64,
                            crate::map::MerchantBonus::Develop(_) => {
                                ctx.cfg.sell.develop_bonus_value
                            }
                        };
                    }
                }
            }
            if free_develop.is_some() {
                score += ctx.cfg.sell.develop_bonus_value;
            }
            Some(Decision {
                mv: ResolvedMove::Sell {
                    keys: keys.clone(),
                    beer_sources: beer_sources.clone(),
                    free_develop: *free_develop,
                    card_index,
                },
                score,
                card_score: cards.first().map(|x| x.1).unwrap_or(f64::INFINITY),
            })
        })
        .collect();
    out.sort_by(|a, b| b.score.total_cmp(&a.score));
    out.truncate(super::SOURCE_VARIANTS);
    out
}
