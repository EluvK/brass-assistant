//! Central tuning parameters for the heuristic AI.
//!
//! Tunable weights, thresholds, and policy switches shared by multiple
//! scorers live here, grouped by concern. Action-local policies stay next to
//! their scorer so each action can be tuned independently.
//! `Default` mirrors the historical constants so the refactor starts from the
//! previous behaviour; tuning means overriding individual groups by topic.
//!
//! All scores are expressed in one currency: **VP equivalents**. Weights that
//! convert raw quantities (cash, income levels, hand flexibility) into that
//! currency are in [`ValueWeights`]; everything else scores one action type
//! or sub-model.

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
    /// Share of printed VP credited to an unflipped tile when estimating
    /// board value (it may still flip later).
    pub unflipped_vp_share: f64,
    /// Extra weight on income levels when converting them into VP: income is
    /// a recurring stream, so a position snapshot values it above one turn's
    /// worth (`income_value`).
    pub leaf_income_scale: f64,
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

/// 2-ply lookahead parameters.
#[derive(Debug, Clone, Copy)]
pub struct LookaheadParams {
    pub first_action_k: usize,
    pub second_action_k: usize,
    /// Up to four own actions when the player can chain across a round
    /// boundary (last in order while spending less than every predecessor).
    pub four_action_k: usize,
    /// Only this many top standalone second actions are expanded through the
    /// expensive cross-round continuation. Lower-ranked second actions are
    /// ignored for the round plan; this bounds the four-link fan-out without
    /// changing the normal two-action path.
    pub four_link_second_keep: usize,
    /// Discount applied to the two actions obtained in the following round.
    pub four_action_alpha: f64,
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
    pub flip: FlipWeights,
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
            lookahead: LookaheadParams {
                first_action_k: 3,
                second_action_k: 2,
                four_action_k: 2,
                four_link_second_keep: 2,
                four_action_alpha: 0.35,
            },
            guardrails: Guardrails {
                ban_build_lv1_brewery: true,
            },
        }
    }
}
