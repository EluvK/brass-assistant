//! Sell scoring consumes the validated canonical sell candidates.
use super::{CardChoices, Decision};
use crate::heuristic_ai::context::define_era_round_factor;
use crate::rules::{ResolvedMove, legal_resolved_moves};
use crate::state::GameState;

/// Fraction of a flipped industry's printed income credited when ranking
/// immediate sell targets.
const TILE_INCOME_SHARE: f64 = 0.3;
/// Heuristic value of the free-develop reward from a merchant.
const DEVELOP_BONUS_VALUE: f64 = 0.5;
/// Value of one victory point when ranking sell targets.
const VP_VALUE: f64 = 1.0;

define_era_round_factor!(
    SellVpFactor,
    canal: (0.0, 1.0),
    rail: (0.0, 1.0),
);

define_era_round_factor!(
    SellIncomeFactor,
    canal: (1.0, 0.6),
    rail: (0.2, 0.2),
);

pub(super) fn score_sell_plans(state: &GameState, cards: &CardChoices) -> Vec<Decision> {
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
            let mut score_vp = 0.0;
            let mut score_income = 0.0;
            let mut other_bonus = 0.0;
            for &key in keys {
                if let Some(tile) = state.city_tiles[key].as_ref() {
                    score_vp += tile.def.vp as f64;
                    score_income += tile.def.income as f64 * TILE_INCOME_SHARE;
                }
            }
            for source in beer_sources {
                if source.kind == crate::graph::BeerSourceKind::Merchant {
                    if let Some(i) = source.merchant_idx {
                        match crate::map::merchant_bonus_at(state.merchants[i].loc) {
                            crate::map::MerchantBonus::Vp(v) => score_vp += v as f64 * VP_VALUE,
                            crate::map::MerchantBonus::Money(v) => other_bonus += v as f64,
                            crate::map::MerchantBonus::Income(v) => score_income += v as f64,
                            crate::map::MerchantBonus::Develop(_) => {
                                other_bonus += DEVELOP_BONUS_VALUE
                            }
                        };
                    }
                }
            }
            let score = score_vp * SellVpFactor::factor(state)
                + score_income * SellIncomeFactor::factor(state)
                + other_bonus;
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
