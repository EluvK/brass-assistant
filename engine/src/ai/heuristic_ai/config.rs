//! Central tuning parameters for the heuristic AI.
//!
//! Tunable weights, thresholds, and policy switches shared by multiple
//! scorers live here, grouped by concern. Action-local policies stay next to
//! their scorer so each action can be tuned independently.
//! `Default` mirrors the historical constants so the refactor starts from the
//! previous behaviour; tuning means overriding individual groups (same
//! pattern as [`crate::ai::mcts_ai::MctsConfig`]).
//!
//! All scores are expressed in one currency: **VP equivalents**. Weights that
//! convert raw quantities (cash, income levels, hand flexibility) into that
//! currency are in [`ValueWeights`]; everything else scores one action type
//! or sub-model.

use super::plan::Phase;

/// Conversion of raw quantities into the scoring currency (VP equivalents).
#[derive(Debug, Clone, Copy)]
pub struct ValueWeights {
    /// Score weight of one victory point.
    pub vp: f64,
    /// Base value of £1 before per-phase scaling (`PhaseParams::money_mult`).
    pub money_base: f64,
    /// Base value of one income level before per-phase scaling.
    pub income_base: f64,
    /// Weight of one point of summed hand keep-score (position flexibility).
    pub flex: f64,
    /// VP forfeited when a build replaces one of our own tiles (its end-game
    /// VP leaves the board with the old tile).
    pub own_overbuild_vp_loss: f64,
    /// Share of printed VP credited to an unflipped tile in the MCTS leaf
    /// evaluator (it may still flip later).
    pub unflipped_vp_share: f64,
    /// Extra weight on income levels in the MCTS leaf evaluator: income is
    /// a recurring stream, so a leaf snapshot values it above one turn's
    /// worth (`income_value`).
    pub leaf_income_scale: f64,
}

/// Per-phase evaluation profile parameters. `EraProfile` values are derived
/// from these instead of hard-coded formulas.
#[derive(Debug, Clone, Copy)]
pub struct PhaseParams {
    /// `income_w = income_base * (income_add + income_frac * era_frac)`
    /// where `era_frac` is the fraction of the era still ahead of us.
    pub income_add: f64,
    pub income_frac: f64,
    /// `money_w = money_base * money_mult`.
    pub money_mult: f64,
    /// 2-ply lookahead blending factor for a second own action.
    pub alpha: f64,
    /// "Era endgame" cash-rescue rounds: era-end urgency triggers when
    /// `rounds_remaining <= endgame_rounds`.
    pub endgame_rounds: f64,
}

impl PhaseParams {
    /// Profile that scales `income_base`/`money_base` by fixed factors.
    const fn scaled(income_add: f64, income_frac: f64, money_mult: f64) -> Self {
        Self {
            income_add,
            income_frac,
            money_mult,
            alpha: 0.0,
            endgame_rounds: 0.0,
        }
    }
}

/// The four phase profiles.
#[derive(Debug, Clone, Copy)]
pub struct EraWeights {
    pub canal_early: PhaseParams,
    pub canal_late: PhaseParams,
    pub rail_early: PhaseParams,
    pub rail_late: PhaseParams,
}

/// Flip-probability model (`probability.rs`). Probabilities are clamped to
/// `floor..=cap` at the end.
#[derive(Debug, Clone, Copy)]
pub struct FlipWeights {
    /// Resource flip floor/cap.
    pub floor: f64,
    pub cap: f64,
    /// Immediate flip credit when a build sells out on placement.
    pub sellout: f64,
    /// Base era demand for coal / iron outside the canal era.
    pub coal_demand_canal: f64,
    pub coal_demand_rail: f64,
    pub iron_demand_canal: f64,
    pub iron_demand_rail: f64,
    /// Extra flip odds per unit of market scarcity.
    pub scarcity_bonus: f64,
    /// Canal island-coal mine: base + heat * heat_bonus, capped. Heat window
    /// `(price - heat_price_base) / heat_price_span`.
    pub island_coal_canal_base: f64,
    pub island_coal_canal_heat_bonus: f64,
    pub island_coal_canal_cap: f64,
    pub island_coal_canal_price_base: f64,
    pub island_coal_canal_price_span: f64,
    /// Rail island-coal mine: base + heat * heat_bonus, capped.
    pub island_coal_rail_base: f64,
    pub island_coal_rail_heat_bonus: f64,
    pub island_coal_rail_cap: f64,
    pub island_coal_rail_price_base: f64,
    pub island_coal_rail_price_span: f64,
    /// Brewery (beer supply) flip model: canal era with no beer demand,
    /// surplus barrels beyond demand, and demand >= supply.
    pub brewery_canal_no_demand: f64,
    pub brewery_surplus: f64,
    pub brewery_satisfied: f64,
    /// Extra beer demand attributed to rail links in the rail era.
    pub brewery_rail_demand_buffer: f64,
    /// Sellable-tile (cotton/goods/pottery) flip model: base, plus bonuses
    /// for a reachable merchant with / without beer, or penalty without one.
    pub sellable_base: f64,
    pub sellable_merchant_with_beer: f64,
    pub sellable_merchant_only: f64,
    pub sellable_no_merchant: f64,
    /// Bonus per adjacent unbuilt link (merchant may connect later).
    pub sellable_open_link: f64,
    /// Hand-poverty penalties: a sellable can only flip if we can pay for
    /// and reach a sell action.
    pub hand_empty_penalty: f64,
    pub hand_one_card_penalty: f64,
    pub hand_few_cards_penalty: f64,
    /// Plan-level flip estimate when no concrete site is known: no
    /// accepting merchant exists at all / merchant exists but beer is
    /// missing / merchant and beer available.
    pub plan_no_merchant: f64,
    pub plan_no_beer: f64,
    pub plan_ready: f64,
}

/// Scout (and pass) scoring.
#[derive(Debug, Clone, Copy)]
pub struct ScoutWeights {
    /// Keep-score thresholds classifying retained cards as low/high value.
    pub low_keep: f64,
    pub high_keep: f64,
    /// How many high-value cards the retained hand should ideally hold.
    pub desired_high_value: usize,
    /// Maximum hand-refresh score (all conditions perfect). Scaled so a
    /// perfect refresh competes with a solid build instead of being
    /// invisible next to one — Scout used to be un-selectable because its
    /// whole score range sat an order of magnitude below build's.
    pub max_refresh: f64,
    /// Per-dead-card reward and per-alive-card cost of the discard trio.
    pub dead_discard_value: f64,
    pub alive_discard_penalty: f64,
    /// Pass fallback score when no candidates were produced at all.
    pub pass_fallback_score: f64,
}

/// Card keep-score parameters (`cards.rs`), all expressed on the 0..3 scale.
#[derive(Debug, Clone, Copy)]
pub struct CardWeights {
    pub canal_iron_base: f64,
    pub canal_sellable_base: f64,
    pub canal_resource_base: f64,
    pub rail_coal_base: f64,
    pub rail_brewery_base: f64,
    pub rail_iron_base: f64,
    pub rail_sellable_base: f64,
    pub canal_iron_third_penalty: f64,
    pub canal_sellable_duplicate_penalty: f64,
    pub canal_resource_duplicate_penalty: f64,
    pub rail_coal_third_penalty: f64,
    pub rail_industry_duplicate_penalty: f64,
    pub canal_location_base: f64,
    pub canal_resource_city_bonus: f64,
    pub rail_location_base: f64,
    pub occupied_city_slot_penalty: f64,
    pub location_second_duplicate_penalty: f64,
    pub location_third_duplicate_penalty: f64,
    pub unconsumable_industry_penalty: f64,
}

/// 2-ply lookahead and turn-end safety parameters.
#[derive(Debug, Clone, Copy)]
pub struct LookaheadParams {
    pub first_action_k: usize,
    pub second_action_k: usize,
    /// Cash line below which the turn-end penalty engages.
    pub low_money_threshold: i32,
    /// Overall turn-end penalty scale.
    pub end_turn_penalty_scale: f64,
    /// Income gained this turn that exempts the position from the penalty.
    pub end_turn_income_exempt: f64,
    /// Income-term weight by whether income is negative.
    pub end_turn_negative_income_weight: f64,
    pub end_turn_income_weight: f64,
    /// Rail-era term (canal scales down by the canal factor).
    pub end_turn_rail_era_term: f64,
    pub end_turn_canal_era_term: f64,
    /// Runway term `base + span * (1 - runway)` penalises low cash early
    /// in the era (many rounds left to survive).
    pub end_turn_runway_base: f64,
    pub end_turn_runway_span: f64,
}

/// Hard policy constraints (temporary strategic guardrails).
#[derive(Debug, Clone, Copy)]
pub struct Guardrails {
    /// Forbid building the level-1 brewery (vanishes at era end; develop
    /// it instead).
    pub ban_build_lv1_brewery: bool,
}

/// Full heuristic parameter set.
#[derive(Debug, Clone, Copy)]
pub struct HeuristicConfig {
    pub value: ValueWeights,
    pub era: EraWeights,
    pub flip: FlipWeights,
    pub scout: ScoutWeights,
    pub cards: CardWeights,
    pub lookahead: LookaheadParams,
    pub guardrails: Guardrails,
}

impl Default for HeuristicConfig {
    fn default() -> Self {
        Self {
            value: ValueWeights {
                vp: 1.0,
                money_base: 0.12,
                income_base: 0.25,
                flex: 0.8,
                own_overbuild_vp_loss: 1.0,
                unflipped_vp_share: 0.25,
                leaf_income_scale: 3.0,
            },
            era: EraWeights {
                canal_early: PhaseParams {
                    alpha: 0.6,
                    endgame_rounds: 2.0,
                    ..PhaseParams::scaled(1.8, 0.6, 0.55)
                },
                canal_late: PhaseParams {
                    alpha: 0.6,
                    endgame_rounds: 2.0,
                    ..PhaseParams::scaled(1.8, 0.6, 0.55)
                },
                rail_early: PhaseParams {
                    alpha: 0.6,
                    endgame_rounds: 1.0,
                    ..PhaseParams::scaled(1.2, 0.5, 0.8)
                },
                rail_late: PhaseParams {
                    // Rail-late income is worthless (the era ends before the
                    // extra income compounds), cash is king.
                    income_add: 0.0,
                    income_frac: 0.0,
                    money_mult: 5.0 / 3.0,
                    alpha: 0.35,
                    endgame_rounds: 1.0,
                },
            },
            flip: FlipWeights {
                floor: 0.05,
                cap: 0.9,
                sellout: 0.9,
                coal_demand_canal: 0.55,
                coal_demand_rail: 0.85,
                iron_demand_canal: 0.4,
                iron_demand_rail: 0.5,
                scarcity_bonus: 0.35,
                island_coal_canal_base: 0.12,
                island_coal_canal_heat_bonus: 0.18,
                island_coal_canal_cap: 0.4,
                island_coal_canal_price_base: 5.0,
                island_coal_canal_price_span: 3.0,
                island_coal_rail_base: 0.6,
                island_coal_rail_heat_bonus: 0.25,
                island_coal_rail_cap: 0.9,
                island_coal_rail_price_base: 4.0,
                island_coal_rail_price_span: 4.0,
                brewery_canal_no_demand: 0.25,
                brewery_surplus: 0.45,
                brewery_satisfied: 0.7,
                brewery_rail_demand_buffer: 1.0,
                sellable_base: 0.12,
                sellable_merchant_with_beer: 0.6,
                sellable_merchant_only: 0.1,
                sellable_no_merchant: -0.6,
                sellable_open_link: 0.1,
                hand_empty_penalty: 10.0,
                hand_one_card_penalty: 5.0,
                hand_few_cards_penalty: 2.0,
                plan_no_merchant: 0.15,
                plan_no_beer: 0.3,
                plan_ready: 0.7,
            },
            scout: ScoutWeights {
                low_keep: 1.0,
                high_keep: 1.8,
                desired_high_value: 2,
                max_refresh: 5.0,
                dead_discard_value: 0.96,
                alive_discard_penalty: 0.48,
                pass_fallback_score: -0.5,
            },
            cards: CardWeights {
                canal_iron_base: 2.0,
                canal_sellable_base: 1.8,
                canal_resource_base: 1.5,
                rail_coal_base: 2.5,
                rail_brewery_base: 2.0,
                rail_iron_base: 1.5,
                rail_sellable_base: 1.8,
                canal_iron_third_penalty: 0.5,
                canal_sellable_duplicate_penalty: 0.3,
                canal_resource_duplicate_penalty: 0.5,
                rail_coal_third_penalty: 1.0,
                rail_industry_duplicate_penalty: 0.3,
                canal_location_base: 1.5,
                canal_resource_city_bonus: 0.5,
                rail_location_base: 2.0,
                occupied_city_slot_penalty: 0.5,
                location_second_duplicate_penalty: 0.5,
                location_third_duplicate_penalty: 1.0,
                unconsumable_industry_penalty: 1.5,
            },
            lookahead: LookaheadParams {
                first_action_k: 3,
                second_action_k: 2,
                low_money_threshold: 15,
                end_turn_penalty_scale: 6.5,
                end_turn_income_exempt: 2.5,
                end_turn_negative_income_weight: 1.4,
                end_turn_income_weight: 0.9,
                end_turn_rail_era_term: 1.0,
                end_turn_canal_era_term: 0.8,
                end_turn_runway_base: 0.6,
                end_turn_runway_span: 0.4,
            },
            guardrails: Guardrails {
                ban_build_lv1_brewery: true,
            },
        }
    }
}

impl EraWeights {
    /// Profile lookup by strategy phase.
    pub fn params(&self, phase: Phase) -> PhaseParams {
        match phase {
            Phase::CanalEarly => self.canal_early,
            Phase::CanalLate => self.canal_late,
            Phase::RailEarly => self.rail_early,
            Phase::RailLate => self.rail_late,
        }
    }
}
