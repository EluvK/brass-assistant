//! Central tuning parameters for the heuristic AI.
//!
//! Every weight, threshold, and policy switch used by the scoring pipeline
//! lives here, grouped by concern. `Default` mirrors the historical constants
//! so the refactor starts from the previous behaviour; tuning means
//! overriding individual groups (same pattern as [`crate::ai::mcts_ai::MctsConfig`]).
//!
//! All scores are expressed in one currency: **VP equivalents** (see
//! [`super::value::ScoreParts`]). Weights that convert raw quantities (cash,
//! income levels, hand flexibility) into that currency are in
//! [`ValueWeights`]; everything else scores one action type or sub-model.

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
/// from these instead of hard-coded formulas, and the future-income discount
/// for link/plan scoring is continuous in remaining rounds.
#[derive(Debug, Clone, Copy)]
pub struct PhaseParams {
    /// `income_w = income_base * (income_add + income_frac * era_frac)`
    /// where `era_frac` is the fraction of the era still ahead of us.
    pub income_add: f64,
    pub income_frac: f64,
    /// `money_w = money_base * money_mult`.
    pub money_mult: f64,
    /// Scale for strategic (non-cash) link value this phase.
    pub network_w: f64,
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
            network_w: 0.0,
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

/// Continuous discount applied to value realised *later* than the current
/// action (future link VP, early sells, hand access gained for later turns).
/// `discount = floor + span * era_frac`: with a full era ahead the future is
/// worth almost as much as now; with none left it degrades to the floor.
#[derive(Debug, Clone, Copy)]
pub struct DiscountWeights {
    pub floor: f64,
    pub span: f64,
}

/// Build action scoring.
#[derive(Debug, Clone, Copy)]
pub struct BuildWeights {
    /// Score lost per £1 a build costs beyond current cash (soft constraint;
    /// loan remains a path but is not a first choice).
    pub unaffordable_per_pound: f64,
    /// Share of a built tile's link VP credited when we already own an
    /// adjacent link (the icon is only realised once the tile flips).
    pub link_self_value_share: f64,
    /// Strategic value per resource cube produced by a coal/iron works we
    /// keep for ourselves (self-sufficiency).
    pub self_sufficiency_per_cube: f64,
    /// Iron market scarcity is discounted relative to coal: iron demand is
    /// steadier and the market fills faster, so over-producing iron is risky.
    pub iron_scarcity_share: f64,
    /// Share of immediate market-sale cash credited a second time as a
    /// tempo bonus (cash now beats cash later).
    pub market_cash_back_share: f64,
    /// Flat bonus when a resource build sells out completely on placement
    /// (immediate flip for income).
    pub market_sellout_bonus: f64,
    /// Coal market "heat" price window: heat = (buy_price - base) / span,
    /// clamped to 0..1. Buy prices inside the window mark a demand spike.
    pub coal_spike_price_base: f64,
    pub coal_spike_price_span: f64,
    /// Bonus per sold cube during a coal price spike.
    pub coal_spike_per_sold: f64,
    /// Spike multiplier in the canal era (demand pressure is higher).
    pub coal_spike_canal_mult: f64,
    /// Strategic value of feeding a hungry market, per unit scarcity and
    /// per cube actually sold.
    pub scarcity_value_per_unit: f64,
    /// Penalty per cube that cannot be sold on placement and stays on the
    /// tile (rail-era coal is exempt — the table consumes it soon).
    pub leftover_per_cube: f64,
    /// Penalty for a canal-era island coal mine (no merchant reach: cannot
    /// sell on placement, unlikely to ever flip).
    pub island_coal_canal_penalty: f64,
    /// Base + per-cube strategic value of a rail-era island coal mine under
    /// scarcity (the table consumes it even without merchant reach).
    pub island_coal_rail_base: f64,
    pub island_coal_rail_per_cube: f64,
    /// Market-value floor of an unconnected iron works under scarcity.
    pub island_iron_value: f64,
    /// Strategic value per unbuilt link adjacent to the build (expansion
    /// potential).
    pub expansion_per_link: f64,
    /// Rail-era emergency coal prior: base bonus under full shortage,
    /// scaled by `1 + per_level * (level-1)` and
    /// `cubes_base + per_cube * cubes`.
    pub rail_coal_shortage: f64,
    pub rail_coal_shortage_per_level: f64,
    pub rail_coal_shortage_cubes_base: f64,
    pub rail_coal_shortage_per_cube: f64,
    /// Cap on the cost-efficiency factor `(income + vp) / cost`.
    pub cost_efficiency_cap: f64,
    /// Sellable-tile placement bonuses: a reachable accepting merchant,
    /// then beer actually available for the sell.
    pub merchant_reachable_bonus: f64,
    pub beer_available_bonus: f64,
    pub beer_missing_penalty: f64,
    /// Brewery oversupply: penalty per surplus barrel beyond the beer
    /// demand of our unflipped sellables.
    pub brewery_surplus_penalty_per_barrel: f64,
    /// Brewery sale support when we do / don't yet have sellables to feed.
    pub brewery_sell_support_with_demand: f64,
    pub brewery_sell_support_base: f64,
    /// Brewery strategic value in the rail era (double-rails consume beer).
    pub rail_brewery_value: f64,
    /// Free-riding bonus for sourcing build inputs from board mines/works:
    /// `(ratio - threshold).max(0) * bonus`, where ratio is the free-source
    /// share of required coal/iron.
    pub free_riding_threshold: f64,
    pub free_riding_bonus: f64,
    /// Plan ("流派") alignment bonus when building the plan industry, from
    /// Canal-Late onward.
    pub plan_bonus: f64,
    /// Rail-Late "beer-gated finish" bonus for a sellable with beer truly
    /// available (the finishing move of the late game).
    pub rail_late_beer_bonus: f64,
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

/// Network (link) action scoring.
#[derive(Debug, Clone, Copy)]
pub struct NetworkWeights {
    /// Hand-access value: location cards newly in network / industry cards
    /// generally, as flexibility (not raw VP).
    pub access_per_location_card: f64,
    pub access_per_industry_card: f64,
    /// Bonus when the link reaches a merchant location.
    pub merchant_bonus: f64,
    /// Exploration prior: `base - per_link * links_left_in_era`, floored at 0.
    pub exploration_base: f64,
    pub exploration_per_link: f64,
    /// Plan-alignment bonus when a link opens a vacant slot for the plan
    /// industry (Canal-Late onward).
    pub plan_bonus: f64,
    /// Rail-Early brewery-farm lock bonus.
    pub beer_lock_bonus: f64,
    /// Double-rail action-economy bonus (two links for one action), by phase
    /// tempo, plus an extra farm-lock synergy bonus.
    pub double_tempo_rail_early: f64,
    pub double_tempo_other: f64,
    pub double_farm_lock_bonus: f64,
    /// Penalty per £ of the double-rail surcharge (base £15 vs 2×£5),
    /// converted through `money_value`.
    pub double_surcharge_weight: f64,
}

/// Develop action scoring.
#[derive(Debug, Clone, Copy)]
pub struct DevelopWeights {
    /// Base target value by the printed era of the tile being developed.
    pub rail_era_tile: f64,
    pub canal_era_tile: f64,
    /// Value per level of the tile being developed away.
    pub per_level: f64,
    /// Bonus for unlocking a double-scoring rail-era tile.
    pub rail_unlock_bonus: f64,
    /// Brewery-engine premium for developing the level-1 brewery away.
    pub brewery_lv1_bonus: f64,
    /// Canal-Early iron-price ladder for the brewery develop (developing
    /// burns iron; only do it while iron is cheap).
    pub iron_price_very_cheap_bonus: f64,
    pub iron_price_cheap_bonus: f64,
    pub iron_price_marginal_bonus: f64,
    pub iron_price_expensive_penalty: f64,
    /// Canal-era nudge (cross-era bonus window).
    pub canal_bonus: f64,
    /// Plan alignment and "we hold a card that can build it" bonuses.
    pub plan_bonus: f64,
    pub buildable_card_bonus: f64,
    /// Opportunity cost charged when the iron spent is not free.
    pub iron_scarcity_cost: f64,
    /// Scale applied to the second develop target (diminishing returns).
    pub second_target_scale: f64,
    /// Canal-era develop tempo shaping: overall scale, penalty for spending
    /// the action on a single tile, small bonus for a double develop.
    pub canal_scale: f64,
    pub canal_single_target_penalty: f64,
    pub canal_double_target_bonus: f64,
    /// Develop-count guardrail: per-era limits and a steep quadratic penalty
    /// `over * over * steepness + over` beyond the limit.
    pub canal_count_limit: f64,
    pub rail_count_limit: f64,
    pub over_limit_steepness: f64,
}

/// Sell action scoring.
#[derive(Debug, Clone, Copy)]
pub struct SellWeights {
    /// Value of a merchant's free-develop bonus (already realised during
    /// the same action).
    pub develop_bonus_value: f64,
    /// Income share added to a tile's printed VP when ranking sell targets
    /// (flipping advances the income track too).
    pub tile_income_share: f64,
    /// Era-end urgency bonus when sellables vanish unflipped (canal) or the
    /// finish is selling (rail), plus the rail-late baseline bonus.
    pub urgency_bonus: f64,
    pub rail_late_baseline_bonus: f64,
    /// Share of the income gain credited as a recurring stream.
    pub income_stream_share: f64,
    /// Scale of the realised VP by how much era remains (sells are about
    /// income early and VP late): `floor + span * (1 - era_frac)`.
    pub vp_scale_floor: f64,
    pub vp_scale_span: f64,
}

/// Loan action scoring.
#[derive(Debug, Clone, Copy)]
pub struct LoanWeights {
    /// Base VP-equivalent opportunity cost of spending one action on a loan.
    pub action_cost: f64,
    /// Multiplier applied after economic value is normalised to 0..1.
    pub economic_score_scale: f64,
    /// Cash-need threshold `a`, linearly interpolated by numbered round.
    pub canal_a_start: f64,
    pub canal_a_end: f64,
    pub rail_a_start: f64,
    pub rail_a_end: f64,
    /// Income penalty divisor `b`, also linearly interpolated by numbered
    /// round so tuning can change its strength within each era.
    pub canal_b_start: f64,
    pub canal_b_end: f64,
    pub rail_b_start: f64,
    pub rail_b_end: f64,
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

/// Hard policy constraints (temporary strategic guardrails) and their
/// penalty coefficients.
#[derive(Debug, Clone, Copy)]
pub struct Guardrails {
    /// Forbid building the level-1 brewery (vanishes at era end; develop
    /// it instead).
    pub ban_build_lv1_brewery: bool,
    /// Forbid developing iron works level 2+ (iron is for builds, not
    /// develop fodder).
    pub ban_develop_iron_lv2_plus: bool,
    /// Forbid developing brewery level 2+ during the canal era.
    pub ban_develop_brewery_lv2_canal_early: bool,
    /// Develop-guardrail penalties for brewing/coal at level 2+:
    /// `base + per_level * (level - 2)`.
    pub develop_brewery_penalty_base: f64,
    pub develop_brewery_penalty_per_level: f64,
    pub develop_coal_penalty_base: f64,
    pub develop_coal_penalty_per_level: f64,
}

/// Full heuristic parameter set.
#[derive(Debug, Clone, Copy)]
pub struct HeuristicConfig {
    pub value: ValueWeights,
    pub era: EraWeights,
    pub discount: DiscountWeights,
    pub flip: FlipWeights,
    pub build: BuildWeights,
    pub network: NetworkWeights,
    pub develop: DevelopWeights,
    pub sell: SellWeights,
    pub loan: LoanWeights,
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
                    network_w: 0.1,
                    alpha: 0.6,
                    endgame_rounds: 2.0,
                    ..PhaseParams::scaled(1.8, 0.6, 0.55)
                },
                canal_late: PhaseParams {
                    network_w: 0.1,
                    alpha: 0.6,
                    endgame_rounds: 2.0,
                    ..PhaseParams::scaled(1.8, 0.6, 0.55)
                },
                rail_early: PhaseParams {
                    network_w: 1.0,
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
                    network_w: 0.85,
                    alpha: 0.35,
                    endgame_rounds: 1.0,
                },
            },
            discount: DiscountWeights {
                floor: 0.3,
                span: 0.5,
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
            build: BuildWeights {
                unaffordable_per_pound: 0.3,
                link_self_value_share: 0.5,
                self_sufficiency_per_cube: 0.15,
                iron_scarcity_share: 0.6,
                market_cash_back_share: 0.4,
                market_sellout_bonus: 1.5,
                coal_spike_price_base: 5.0,
                coal_spike_price_span: 3.0,
                coal_spike_per_sold: 1.9,
                coal_spike_canal_mult: 1.25,
                scarcity_value_per_unit: 0.6,
                leftover_per_cube: 0.5,
                island_coal_canal_penalty: -0.5,
                island_coal_rail_base: 1.2,
                island_coal_rail_per_cube: 0.25,
                island_iron_value: 1.2,
                expansion_per_link: 0.1,
                rail_coal_shortage: 3.0,
                rail_coal_shortage_per_level: 0.2,
                rail_coal_shortage_cubes_base: 0.7,
                rail_coal_shortage_per_cube: 0.15,
                cost_efficiency_cap: 2.0,
                merchant_reachable_bonus: 0.6,
                beer_available_bonus: 0.8,
                beer_missing_penalty: -0.3,
                brewery_surplus_penalty_per_barrel: 0.6,
                brewery_sell_support_with_demand: 0.8,
                brewery_sell_support_base: 0.4,
                rail_brewery_value: 2.0,
                free_riding_threshold: 0.5,
                free_riding_bonus: 0.8,
                plan_bonus: 0.5,
                rail_late_beer_bonus: 1.2,
            },
            network: NetworkWeights {
                access_per_location_card: 0.6,
                access_per_industry_card: 0.1,
                merchant_bonus: 1.5,
                exploration_base: 1.6,
                exploration_per_link: 0.3,
                plan_bonus: 0.5,
                beer_lock_bonus: 1.2,
                double_tempo_rail_early: 1.2,
                double_tempo_other: 0.6,
                double_farm_lock_bonus: 0.8,
                double_surcharge_weight: 1.0,
            },
            develop: DevelopWeights {
                rail_era_tile: 0.35,
                canal_era_tile: 0.12,
                per_level: 0.18,
                rail_unlock_bonus: 0.25,
                brewery_lv1_bonus: 0.55,
                iron_price_very_cheap_bonus: 3.0,
                iron_price_cheap_bonus: 2.0,
                iron_price_marginal_bonus: 0.5,
                iron_price_expensive_penalty: -1.5,
                canal_bonus: 0.15,
                plan_bonus: 0.3,
                buildable_card_bonus: 0.3,
                iron_scarcity_cost: 0.6,
                second_target_scale: 0.4,
                canal_scale: 2.0,
                canal_single_target_penalty: 2.0,
                canal_double_target_bonus: 0.5,
                canal_count_limit: 4.0,
                rail_count_limit: 1.0,
                over_limit_steepness: 2.0,
            },
            sell: SellWeights {
                develop_bonus_value: 0.5,
                tile_income_share: 0.3,
                urgency_bonus: 3.0,
                rail_late_baseline_bonus: 1.2,
                income_stream_share: 0.5,
                vp_scale_floor: 0.1,
                vp_scale_span: 0.5,
            },
            loan: LoanWeights {
                action_cost: -4.0,
                economic_score_scale: 10.0,
                canal_a_start: 20.0,
                canal_a_end: 40.0,
                rail_a_start: 36.0,
                rail_a_end: 30.0,
                canal_b_start: 10.0,
                canal_b_end: 10.0,
                rail_b_start: 10.0,
                rail_b_end: 10.0,
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
                ban_develop_iron_lv2_plus: true,
                ban_develop_brewery_lv2_canal_early: true,
                develop_brewery_penalty_base: 1.8,
                develop_brewery_penalty_per_level: 0.2,
                develop_coal_penalty_base: 1.5,
                develop_coal_penalty_per_level: 0.2,
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
