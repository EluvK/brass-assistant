//! Deterministic tactical lookahead for the heuristic policy.
//!
//! Expands the top first actions, simulates each, and (when the turn
//! continues with us) blends in the best second action.  A last-position
//! prerequisite plus a post-two-action low-spend check additionally values
//! two actions in the next round.

use super::board::{connection_touches_city_for_industry, connection_touches_merchant};
use super::cards;
use super::config::HeuristicConfig;
use super::{Decision, RoundDecision, candidate_actions_k, pass_decision};
use crate::data::IndustryType;
use crate::engine::{advance_turn, handle_turn_result};
use crate::map::{CANAL_LINK_COST, Loc, connections};
use crate::rules::{ResolvedMove, apply_move, iron_source_options};
use crate::state::{Card, GameState};

/// Choose the current player's action with deterministic 2-ply lookahead and
/// an optional cross-round four-link extension.  The returned round plan is
/// meaningful at the start of a player's action block (`actions_this_turn ==
/// 0`); per-action callers that are already inside the block receive a
/// one-move plan because `continues_same_turn` only recognises the second
/// action of the same block.
pub fn choose_action(state: &mut GameState) -> RoundDecision {
    let cfg = HeuristicConfig::default();

    // The first round has a dedicated, progressively extensible opening
    // template.  Keep candidate generation and template selection together
    // so the normal lookahead path is not involved at all.
    if state.is_first_round {
        return round_from_first(choose_opening(state).unwrap_or_else(|| pass_decision(state)));
    }

    let pid = state.current_player_id();
    if is_last_in_order(state, pid) {
        // Being last is the structural prerequisite for a four-link.  The
        // spend-lead check remains in the second-action evaluator, after the
        // first two actions have been simulated.
        return choose_four_action(state, &cfg);
    }

    choose_lookahead(state, &cfg)
}

/// Evaluate a last-in-order turn, including a possible four-link extension.
fn choose_four_action(state: &mut GameState, cfg: &HeuristicConfig) -> RoundDecision {
    let pid = state.current_player_id();

    let mut first_candidates = candidate_actions_k(state, cfg.lookahead.first_action_k);
    first_candidates.sort_by(|a, b| b.score.total_cmp(&a.score));

    // A conservative upper bound for second action + four-link continuation.
    // Even if second action scores 60 VP (0.6x = 36) and four-link continuation adds 50 VP,
    // total gain is comfortably within 90.0.
    const MAX_FOUR_LINK_GAIN: f64 = 90.0;

    let mut best: Option<(ResolvedMove, Option<Decision>, f64, f64)> = None;

    for c1 in first_candidates {
        if let Some((_, _, best_value, _)) = best {
            if c1.score + MAX_FOUR_LINK_GAIN <= best_value {
                break;
            }
        }

        let mut after_first = state.clone();
        if apply_move(&mut after_first, &c1.mv).is_err() {
            continue;
        }
        let tr = advance_turn(&mut after_first);
        handle_turn_result(&mut after_first, tr);

        let mut best_second_decision = None;
        let value = if continues_same_turn(&after_first, pid) {
            let mut best_second = f64::NEG_INFINITY;
            // Expanding every second action through the cross-round
            // continuation would re-score the next round for each of them.
            // The second action of the round is selected on its standalone
            // score; only the top few candidates are worth paying for the
            // four-link continuation, which is what this bounded sweep does.
            let mut second_candidates =
                candidate_actions_k(&mut after_first, cfg.lookahead.second_action_k);
            second_candidates.sort_by(|a, b| b.score.total_cmp(&a.score));
            let mut evaluated = 0usize;
            for c2 in second_candidates {
                if evaluated >= cfg.lookahead.four_link_second_keep {
                    break;
                }
                let mut after_second = after_first.clone();
                if apply_move(&mut after_second, &c2.mv).is_err() {
                    continue;
                }
                evaluated += 1;

                let spend_still_low = spend_lead(&after_second, pid);
                let tr = advance_turn(&mut after_second);
                handle_turn_result(&mut after_second, tr);
                let mut score = 0.6 * c2.score;
                if spend_still_low {
                    score += four_link_value(
                        &mut after_second,
                        pid,
                        cfg.lookahead.four_action_k,
                        cfg.lookahead.four_action_alpha,
                    );
                }
                if score > best_second {
                    best_second = score;
                    best_second_decision = Some(c2);
                }
            }
            c1.score + best_second.max(0.0)
        } else {
            c1.score
        };

        if best.as_ref().is_none_or(|(_, _, score, _)| value > *score) {
            best = Some((c1.mv, best_second_decision, value, c1.score));
        }
    }

    best.map(|(mv, second, _value, first_score)| RoundDecision {
        card_score: super::move_card_score(state, &mv),
        mv,
        score: first_score,
        second,
    })
    .unwrap_or_else(|| round_from_first(pass_decision(state)))
}

/// Evaluate the ordinary two-action turn.
fn choose_lookahead(state: &mut GameState, cfg: &HeuristicConfig) -> RoundDecision {
    let pid = state.current_player_id();
    let mut first_candidates = candidate_actions_k(state, cfg.lookahead.first_action_k);
    first_candidates.sort_by(|a, b| b.score.total_cmp(&a.score));

    // Bound Pruning: In Brass heuristic scoring, single action scores rarely exceed 60 VP
    // (even double rail or large multi-sell), yielding a 0.6x second-action gain of <= 36 VP.
    // 50.0 is an extremely safe upper bound.
    const MAX_SECOND_ACTION_GAIN: f64 = 50.0;

    let mut best: Option<(ResolvedMove, Option<Decision>, f64, f64)> = None;
    for c1 in first_candidates {
        if let Some((_, _, best_value, _)) = best {
            if c1.score + MAX_SECOND_ACTION_GAIN <= best_value {
                break;
            }
        }

        let mut s1 = state.clone();
        if apply_move(&mut s1, &c1.mv).is_err() {
            continue;
        }
        let tr = advance_turn(&mut s1);
        handle_turn_result(&mut s1, tr);

        let mut best_second_decision = None;
        let value = if continues_same_turn(&s1, pid) {
            // The second action is selected by the one-action candidate
            // score; applying it here only validates executability.  The
            // post-move state is no longer needed once the turn-end penalty
            // was removed.
            let mut second_candidates = candidate_actions_k(&mut s1, cfg.lookahead.second_action_k);
            second_candidates.sort_by(|a, b| b.score.total_cmp(&a.score));
            let mut best_second = 0.0;
            for c2 in second_candidates {
                if apply_move(&mut s1, &c2.mv).is_err() {
                    continue;
                }
                let scaled = 0.6 * c2.score;
                best_second_decision = Some(c2);
                best_second = scaled;
                break;
            }
            // 2-ply blend: a productive second action adds to the turn's
            // value, discounted by the phase's alpha.
            c1.score + best_second.max(0.0)
        } else {
            c1.score
        };

        if best.as_ref().is_none_or(|(_, _, score, _)| value > *score) {
            best = Some((c1.mv, best_second_decision, value, c1.score));
        }
    }

    best.map(|(mv, second, _value, first_score)| {
        let card_score = super::move_card_score(state, &mv);
        RoundDecision {
            mv,
            score: first_score,
            card_score,
            second,
        }
    })
    .unwrap_or_else(|| round_from_first(pass_decision(state)))
}

fn round_from_first(first: Decision) -> RoundDecision {
    RoundDecision {
        mv: first.mv,
        score: first.score,
        card_score: first.card_score,
        second: None,
    }
}

/// Structural prerequisite for a four-link: the active player is last in the
/// current action order.  Whether the sequence is actually retained is
/// checked after simulating the first two actions with [`spend_lead`].
fn is_last_in_order(state: &GameState, pid: usize) -> bool {
    // Any 2+ player game can produce the sequence; the active player only
    // needs to be last in order at this stage.
    if state.player_count() < 2 || state.current_index + 1 != state.player_count() {
        return false;
    }
    state.current_player_id() == pid
}

/// True when the clock is still inside the same player action block after a
/// simulated first action: `pid` is active again and has already used one of
/// the round's actions without reaching the block's action limit.  A round or
/// era boundary resets `actions_this_turn`, so this rejects cross-round
/// continuations that only look like a same-turn follow-up.
fn continues_same_turn(state: &GameState, pid: usize) -> bool {
    state.current_player_id() == pid
        && state.actions_this_turn > 0
        && state.actions_this_turn < state.actions_per_turn
}

fn spend_lead(state: &GameState, pid: usize) -> bool {
    let mine = state.money_spent_this_round[pid];
    state.turn_order[..state.current_index]
        .iter()
        .all(|other| mine < state.money_spent_this_round[*other])
}

fn four_link_value(state: &mut GameState, pid: usize, k: usize, discount: f64) -> f64 {
    // A genuine four-link must return to us immediately after the round
    // boundary.  Passing opponents here would evaluate a hypothetical order
    // that the real game does not grant.
    if state.current_player_id() != pid {
        return 0.0;
    }
    let candidates = candidate_actions_k(state, k.max(1));
    let Some(c3) = candidates
        .into_iter()
        .max_by(|a, b| a.score.total_cmp(&b.score))
    else {
        return 0.0;
    };
    let mut value = discount * c3.score.max(0.0);
    if apply_move(state, &c3.mv).is_err() {
        return value;
    }
    let tr = advance_turn(state);
    handle_turn_result(state, tr);
    if state.current_player_id() != pid {
        return value;
    }
    let c4 = candidate_actions_k(state, k.max(1))
        .into_iter()
        .max_by(|a, b| a.score.total_cmp(&b.score));
    let Some(c4) = c4 else {
        return value;
    };
    if apply_move(state, &c4.mv).is_ok() {
        value += discount * 0.5 * c4.score.max(0.0);
    }
    value
}

// ---------------------------------------------------------------------------
// First-round opening policy
// ---------------------------------------------------------------------------

// Tunable situation thresholds for the hard-coded opening rules.  All counts
// come from `cards::analyze_hand`, which counts each card once per industry
// it can unlock (location cards need an empty buildable slot).
const LOAN_IRON_SUPPORT_MIN: usize = 2;
const BREWERY_DEVELOP_SUPPORT_MIN: usize = 3;
/// Both fixed double-develop openings (Brewery×2 and CoalMine+IronWorks)
/// require market iron to stay this cheap so two cubes are affordable.
const DEVELOP_MAX_IRON_PRICE: u8 = 3;
const RESOURCE_FLEX_COAL_SUPPORT_MIN: usize = 1;
const RESOURCE_FLEX_IRON_SUPPORT_MIN: usize = 1;
/// Minimum count of iron industry cards or Birmingham cluster city cards
/// (Coventry, Birmingham, Dudley, Walsall, Tamworth) required to commit
/// to the fixed resource network opening; otherwise fall back to a loan.
const RESOURCE_NETWORK_CARD_SUPPORT_MIN: usize = 2;
/// Fixed openings carry no calibrated VP score: they are not chosen against
/// scored candidates, so callers must not compare this with evaluator scores.
const FIXED_OPENING_SCORE: f64 = 0.0;

/// Check if a hand card is an iron industry card or a location card in the
/// Birmingham cluster (Coventry, Birmingham, Dudley, Walsall, Tamworth).
/// Wild industry and wild location cards also qualify.
fn is_birmingham_cluster_or_iron_card(card: &Card) -> bool {
    match card {
        Card::Industry { .. } => card.is_industry(IndustryType::IronWorks),
        Card::Location(loc) => matches!(
            loc,
            Loc::Coventry | Loc::Birmingham | Loc::Dudley | Loc::Walsall | Loc::Tamworth
        ),
        Card::WildIndustry | Card::WildLocation => true,
    }
}

/// Count of iron industry cards and Birmingham cluster location cards held in hand.
fn birmingham_cluster_or_iron_card_count(state: &GameState, pid: usize) -> usize {
    state.players[pid]
        .hand
        .iter()
        .filter(|card| is_birmingham_cluster_or_iron_card(card))
        .count()
}

/// Candidate pool for the fixed "resource link" opening (rule 3's road
/// alternative).  Every entry is a Canal-era connection that touches a
/// coal/iron city and does not directly touch a merchant.  Links already
/// taken are skipped, and one of the still-open routes is picked by a stable
/// per-position hash so different seeds/players/hands do not always open the
/// same first route.
#[rustfmt::skip]
const RESOURCE_OPENING_ROUTES: [usize; 4] = [
    7,  // Birmingham-Tamworth
    8,  // Birmingham-Walsall
    14, // Burton-Walsall
    21, // Coalbrookdale-Wolverhampton
    // 17, // Cannock-Walsall
    // 11, // Burton-Derby
    // 38, // Walsall-Wolverhampton
    // 0,  // Belper-Derby
    // 34, // Stoke-Stone
    // 30, // Leek-Stoke
    // 3,  // Birmingham-Dudley
    // 2,  // Birmingham-Coventry
    // 13, // Burton-Tamworth
    // 12, // Burton-Stone
    // 9,  // Birmingham-Worcester
    // 15, // Cannock-Stafford
    // 16, // Cannock-BreweryNorth
    // 18, // Cannock-Wolverhampton
    // 19, // Coalbrookdale-Kidderminster
    // 25, // Dudley-Kidderminster
    // 26, // Dudley-Wolverhampton
    // 29, // Kidderminster-Worcester
    // 31, // Nuneaton-Tamworth
    // 33, // Stafford-Stone
];

/// A recognised first-round situation.  Each rule prescribes one exact fixed
/// first action; `candidate_actions_k` is only consulted when no rule fires.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum OpeningRule {
    /// No predecessor spent £0 this round and the hand supports ≥2 iron
    /// builds: take the fixed loan (no round spend) to claim the round-2
    /// first action for the iron route.
    LoanForInitiative,
    /// Plenty of brewery support while market iron is still cheap: fixed
    /// double-develop both level-1 breweries to set up the beer + industry
    /// line.
    BreweryDevelop,
    /// A predecessor already spent £0 (loan/pass), so the initiative loan is
    /// unavailable; with coal+iron support and market iron ≤3 the opening
    /// pivots to the fixed coal+iron double-develop.
    ResourceDevelop,
    /// Same predecessor/hand situation as [`OpeningRule::ResourceDevelop`],
    /// but when market iron is above the develop cap and the hand holds
    /// ≥2 iron industry or Birmingham cluster cards, the fixed fallback
    /// lays a resource link instead (no direct merchant route).
    ResourceNetwork,
    /// A predecessor already spent £0 and market iron is above the develop
    /// cap, but the hand lacks sufficient Birmingham cluster / iron support
    /// (≥2 cards) for a safe network opening: take a second loan to secure
    /// round-2 position and cash.
    SecondLoan,
}

impl OpeningRule {
    /// The deterministic fixed first action for this rule.  The classification
    /// preconditions (full 8-card first-round hand, starting cash, price cap
    /// for develops, and at most three earlier players) guarantee the move is
    /// always executable, so no fallible result is needed here.
    fn fixed_move(self, state: &GameState) -> ResolvedMove {
        match self {
            OpeningRule::LoanForInitiative | OpeningRule::SecondLoan => ResolvedMove::Loan {
                card_index: weakest_discard_index(state),
            },
            OpeningRule::BreweryDevelop => {
                fixed_double_develop(state, IndustryType::Brewery, IndustryType::Brewery)
            }
            OpeningRule::ResourceDevelop => {
                fixed_double_develop(state, IndustryType::CoalMine, IndustryType::IronWorks)
            }
            OpeningRule::ResourceNetwork => fixed_resource_network(state),
        }
    }
}

/// The least valuable discardable hand card (`ranked_card_choices` is sorted
/// ascending by keep value).  Fixed actions discard that card so valuable
/// industry/location cards stay in hand for the following round.  The first
/// round always starts with a full hand, so a card index always exists.
fn weakest_discard_index(state: &GameState) -> usize {
    super::ranked_card_choices(state, state.current_player_id())
        .first()
        .map(|(index, _)| *index)
        .expect("first-round hands always have cards to discard")
}

/// First source selection for `needed` market/free iron cubes.  The rule price
/// cap (`DEVELOP_MAX_IRON_PRICE`) plus the starting £17 makes every selection
/// affordable; source enumeration guarantees the free-first rule is obeyed.
fn iron_sources_for(state: &GameState, needed: usize) -> Vec<crate::graph::IronSource> {
    iron_source_options(state, needed)
        .into_iter()
        .next()
        .expect("iron source options always exist for up to two cubes")
}

/// Fixed double-develop action with the default iron source selection and the
/// least valuable discardable card.
fn fixed_double_develop(
    state: &GameState,
    first: IndustryType,
    second: IndustryType,
) -> ResolvedMove {
    ResolvedMove::Develop {
        ind1: first,
        ind2: Some(second),
        iron: iron_sources_for(state, 2),
        card_index: weakest_discard_index(state),
    }
}

/// FNV-1a mix used by the stable route hash (no `Hash` derive needed for the
/// hand, and the result is reproducible across runs/platforms).
fn route_seed_mix(hash: u64, value: u64) -> u64 {
    (hash ^ value).wrapping_mul(0x1000_0000_01b3)
}

/// Reproducible per-position hash used to vary the fixed resource-route
/// opening.  Mixes the turn position, market state, predecessor spends, and
/// the semantic contents of the current hand; it never consumes the engine's
/// private RNG stream, so decision calls on clones stay independent.
fn route_seed(state: &GameState) -> u64 {
    let pid = state.current_player_id();
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    hash = route_seed_mix(hash, state.current_index as u64);
    hash = route_seed_mix(hash, pid as u64);
    hash = route_seed_mix(hash, state.coal_market as u64);
    hash = route_seed_mix(hash, state.iron_market as u64);
    for &other in &state.turn_order[..state.current_index] {
        hash = route_seed_mix(hash, state.money_spent_this_round[other] as u64);
    }
    for (index, card) in state.players[pid].hand.iter().enumerate() {
        hash = route_seed_mix(hash, index as u64);
        match card {
            Card::Location(loc) => {
                hash = route_seed_mix(hash, *loc as u64);
            }
            Card::Industry { industries, n } => {
                hash = route_seed_mix(hash, 32 + industries[0] as u64);
                hash = route_seed_mix(hash, *n as u64);
            }
            Card::WildLocation => hash = route_seed_mix(hash, 64),
            Card::WildIndustry => hash = route_seed_mix(hash, 65),
        }
    }
    hash
}

/// Fixed resource link: this is a first-round, no-presence first action, so
/// no network-adjacency filter is needed — only links already built by an
/// earlier player are excluded.  One of the still-open routes is selected by
/// the stable per-position hash instead of always taking the table's first.
fn fixed_resource_network(state: &GameState) -> ResolvedMove {
    let pid = state.current_player_id();
    let player = &state.players[pid];
    debug_assert!(state.is_first_round && state.actions_this_turn == 0);
    debug_assert!(!crate::graph::player_has_presence(state, pid));
    debug_assert!(player.canal_links > 0 && player.money >= CANAL_LINK_COST);

    let card_index = weakest_discard_index(state);
    let available: Vec<usize> = RESOURCE_OPENING_ROUTES
        .iter()
        .copied()
        .filter(|conn_id| {
            // The table already satisfies these predicates; re-checking them
            // here keeps this function robust if the table is edited later.
            state.links[*conn_id].is_none()
                && !connection_touches_merchant(*conn_id)
                && connections()[*conn_id].canal
                && (connection_touches_city_for_industry(*conn_id, IndustryType::CoalMine)
                    || connection_touches_city_for_industry(*conn_id, IndustryType::IronWorks))
        })
        .collect();
    assert!(
        !available.is_empty(),
        "at most three earlier players can occupy routes before the fixed opening"
    );
    let conn_id = available[route_seed(state) as usize % available.len()];
    ResolvedMove::Network {
        conn_id,
        coal: None,
        card_index,
    }
}

/// Classify the current first-round opening position.
///
/// Signals: own hand support (industry + location + wild cards), market iron
/// tightness (cheap enough to develop), and the recorded round spend of the
/// players that already acted.  Loans do not add to `money_spent_this_round`
/// in the engine, so a predecessor who loaned is observable as a £0 spend.
/// The rules are intentionally ordered — a hand that fits several templates
/// follows the first one listed here.  Classification keeps the preconditions
/// that make the exact fixed action executable in the first round; the fixed
/// action itself does not re-check them.
fn opening_rule(state: &GameState) -> Option<OpeningRule> {
    if !state.is_first_round {
        return None;
    }
    let pid = state.current_player_id();
    let hand = cards::analyze_hand(state, pid);
    let iron_support = hand.support_total(state, IndustryType::IronWorks);
    let brewery_support = hand.support_total(state, IndustryType::Brewery);
    let coal_support = hand.support_total(state, IndustryType::CoalMine);
    let predecessor_zero_spend = state.turn_order[..state.current_index]
        .iter()
        .any(|&other| state.money_spent_this_round[other] == 0);

    if iron_support >= LOAN_IRON_SUPPORT_MIN && !predecessor_zero_spend {
        return Some(OpeningRule::LoanForInitiative);
    }
    // develop something.
    if brewery_support >= BREWERY_DEVELOP_SUPPORT_MIN
        && state.iron_price() <= DEVELOP_MAX_IRON_PRICE
    {
        return Some(OpeningRule::BreweryDevelop);
    }
    if predecessor_zero_spend
        && coal_support >= RESOURCE_FLEX_COAL_SUPPORT_MIN
        && iron_support >= RESOURCE_FLEX_IRON_SUPPORT_MIN
    {
        if state.iron_price() <= DEVELOP_MAX_IRON_PRICE {
            return Some(OpeningRule::ResourceDevelop);
        }
        if birmingham_cluster_or_iron_card_count(state, pid) >= RESOURCE_NETWORK_CARD_SUPPORT_MIN {
            return Some(OpeningRule::ResourceNetwork);
        }
        return Some(OpeningRule::SecondLoan);
    }
    None
}

/// Hard-coded opening template.  When a rule matches, its exact fixed move is
/// returned directly — no candidate search or scoring.  Only positions that
/// match no rule fall back to `candidate_actions_k`.
fn choose_opening(state: &mut GameState) -> Option<Decision> {
    let cfg = HeuristicConfig::default();
    let Some(rule) = opening_rule(state) else {
        return choose_default_opening(state, &cfg);
    };

    let mv = rule.fixed_move(state);
    let card_score = super::move_card_score(state, &mv);
    Some(Decision {
        mv,
        score: FIXED_OPENING_SCORE,
        card_score,
    })
}

/// Fallback for positions no fixed rule matches: return the best-scoring
/// candidate from `candidate_actions_k` (no extra opening bonuses).
fn choose_default_opening(state: &mut GameState, cfg: &HeuristicConfig) -> Option<Decision> {
    let candidates = candidate_actions_k(state, cfg.lookahead.first_action_k);
    candidates.into_iter().max_by(|a, b| {
        a.score
            .total_cmp(&b.score)
            .then_with(|| a.card_score.total_cmp(&b.card_score))
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::map::Loc;
    use crate::state::Card;
    use rand_chacha::ChaCha12Rng;
    use rand_chacha::rand_core::SeedableRng;

    fn industry(ind: IndustryType) -> Card {
        Card::Industry {
            industries: [ind; 2],
            n: 1,
        }
    }

    /// A first-round state with a fixed identity turn order and the current
    /// player at `index`.  All players start with £0 spent, as after setup.
    fn first_round_state(seed: u64, players: usize, index: usize) -> GameState {
        let mut state = GameState::new(ChaCha12Rng::seed_from_u64(seed), players);
        assert!(state.is_first_round);
        state.turn_order = (0..players).collect();
        state.current_index = index;
        state.actions_this_turn = 0;
        state.money_spent_this_round = vec![0; players];
        state
    }

    /// A hand with exactly `support` iron plays: two iron/iron-city cards are
    /// the scenario-1 trigger, so build it by duplicating the industry card.
    fn iron_hand(extra: Vec<Card>) -> Vec<Card> {
        let mut hand = vec![industry(IndustryType::IronWorks); 2];
        hand.extend(extra);
        hand
    }

    /// Non-resource filler cards (Worcester only hosts cotton), so opening
    /// analyses that count coal/iron/brewery are not polluted by the filler.
    fn filler_cards(count: usize) -> Vec<Card> {
        std::iter::repeat_with(|| Card::Location(Loc::Worcester))
            .take(count)
            .collect()
    }

    #[test]
    fn four_link_checks_position_then_spend_separately() {
        let mut state = GameState::new(ChaCha12Rng::seed_from_u64(41), 4);
        state.current_index = 3;
        let pid = state.current_player_id();
        state.money_spent_this_round = vec![10; 4];
        assert!(is_last_in_order(&state, pid));
        assert!(!spend_lead(&state, pid));
        state.money_spent_this_round[pid] = 0;
        assert!(spend_lead(&state, pid));

        let state3 = GameState::new(ChaCha12Rng::seed_from_u64(42), 3);
        assert!(!is_last_in_order(&state3, state3.current_player_id()));
    }

    #[test]
    fn opening_policy_returns_an_executable_move() {
        let mut state = GameState::new(ChaCha12Rng::seed_from_u64(43), 4);
        // Fixed openings are not required to sit inside any candidate pool;
        // the contract is that the returned move executes in the current state.
        let decision = choose_opening(&mut state).expect("opening candidate");
        let mut probe = state.clone();
        assert!(
            apply_move(&mut probe, &decision.mv).is_ok(),
            "opening decision must be executable"
        );
    }

    #[test]
    fn continuation_only_recognises_an_unfinished_action_block() {
        let mut state = GameState::new(ChaCha12Rng::seed_from_u64(44), 4);
        state.actions_per_turn = 2;
        state.actions_this_turn = 1;
        let pid = state.current_player_id();
        assert!(continues_same_turn(&state, pid));

        // A round/era boundary resets the action counter; even if the same
        // player is active again this is the next round, not a follow-up.
        state.actions_this_turn = 0;
        assert!(!continues_same_turn(&state, pid));

        // The action block is over once the per-turn limit is reached.
        state.actions_this_turn = state.actions_per_turn;
        assert!(!continues_same_turn(&state, pid));
    }

    #[test]
    fn initiative_loan_is_taken_with_strong_iron_hand_and_spending_predecessors() {
        let mut state = first_round_state(51, 4, 1);
        let pid = state.current_player_id();
        let mut hand = iron_hand(vec![]);
        hand.extend(filler_cards(6));
        state.players[pid].hand = hand;
        // Player 0 already acted and spent (built/developed), so no
        // predecessor is sitting on the £0 initiative slot.
        state.money_spent_this_round[0] = 30;

        assert_eq!(opening_rule(&state), Some(OpeningRule::LoanForInitiative));
        let decision = choose_opening(&mut state).expect("opening candidate");
        match decision.mv {
            ResolvedMove::Loan { card_index } => {
                assert!(
                    matches!(
                        state.players[pid].hand[card_index],
                        Card::Location(Loc::Worcester)
                    ),
                    "loan should discard the filler card, not an iron card"
                );
            }
            other => panic!("iron opening should loan, got {other:?}"),
        }
    }

    #[test]
    fn initiative_loan_is_withheld_when_a_predecessor_has_zero_spend() {
        let mut state = first_round_state(52, 4, 1);
        let pid = state.current_player_id();
        let mut hand = iron_hand(vec![]);
        hand.extend(filler_cards(6));
        state.players[pid].hand = hand;
        // A predecessor loaned/passed: our loan would tie at £0 and lose the
        // stable-sort tie-break, so the rule must not fire.
        state.money_spent_this_round[0] = 0;

        assert_ne!(opening_rule(&state), Some(OpeningRule::LoanForInitiative));
    }

    #[test]
    fn brewery_develop_rule_fires_for_beer_hand_with_cheap_iron() {
        let mut state = first_round_state(53, 4, 0);
        let pid = state.current_player_id();
        let mut hand = vec![industry(IndustryType::Brewery); 3];
        hand.extend(filler_cards(5));
        state.players[pid].hand = hand;
        state.iron_market = 8; // initial price 2 <= 4

        assert_eq!(opening_rule(&state), Some(OpeningRule::BreweryDevelop));
        let decision = choose_opening(&mut state).expect("opening candidate");
        match decision.mv {
            ResolvedMove::Develop {
                ind1: IndustryType::Brewery,
                ind2: Some(IndustryType::Brewery),
                ..
            } => {}
            // Two level-1 breweries are removed in one fixed double-develop.
            other => panic!("brewery rule returned {other:?}"),
        }
    }

    #[test]
    fn resource_develop_is_fixed_after_predecessor_loan_with_coal_and_iron_hand() {
        let mut state = first_round_state(54, 4, 1);
        let pid = state.current_player_id();
        let mut hand = vec![
            industry(IndustryType::CoalMine),
            industry(IndustryType::IronWorks),
        ];
        hand.extend(filler_cards(6));
        state.players[pid].hand = hand;
        state.money_spent_this_round[0] = 0;

        assert_eq!(opening_rule(&state), Some(OpeningRule::ResourceDevelop));
        let decision = choose_opening(&mut state).expect("opening candidate");
        match decision.mv {
            ResolvedMove::Develop {
                ind1: IndustryType::CoalMine,
                ind2: Some(IndustryType::IronWorks),
                ..
            } => {}
            other => panic!("resource rule should fixed-develop coal+iron, got {other:?}"),
        }
    }

    #[test]
    fn second_loan_is_the_fixed_fallback_when_iron_is_too_expensive_and_support_lacking() {
        let mut state = first_round_state(55, 4, 1);
        let pid = state.current_player_id();
        let mut hand = vec![
            industry(IndustryType::CoalMine),
            industry(IndustryType::IronWorks),
        ];
        hand.extend(filler_cards(6));
        state.players[pid].hand = hand;
        state.iron_market = 0; // empty market: iron price 6 > 3, no develop
        state.money_spent_this_round[0] = 0;

        assert_eq!(opening_rule(&state), Some(OpeningRule::SecondLoan));
        let decision = choose_opening(&mut state).expect("opening candidate");
        match decision.mv {
            ResolvedMove::Loan { .. } => {}
            other => panic!("insufficient support should fall back to second loan, got {other:?}"),
        }
    }

    #[test]
    fn resource_network_is_the_fixed_fallback_when_iron_is_too_expensive() {
        let mut state = first_round_state(55, 4, 1);
        let pid = state.current_player_id();
        let mut hand = vec![
            industry(IndustryType::CoalMine),
            industry(IndustryType::IronWorks),
            Card::Location(Loc::Birmingham),
        ];
        hand.extend(filler_cards(5));
        state.players[pid].hand = hand;
        state.iron_market = 0; // empty market: iron price 6 > 3, no develop
        state.money_spent_this_round[0] = 0;

        assert_eq!(opening_rule(&state), Some(OpeningRule::ResourceNetwork));
        let decision = choose_opening(&mut state).expect("opening candidate");
        match decision.mv {
            ResolvedMove::Network { conn_id, .. } => {
                assert!(
                    RESOURCE_OPENING_ROUTES.contains(&conn_id),
                    "network fallback must use a hard-coded resource route"
                );
            }
            other => panic!("resource rule should lay a fixed resource link, got {other:?}"),
        }
    }

    #[test]
    fn resource_network_skips_taken_routes_and_varies_between_positions() {
        // Route 17 is taken by an earlier player: the fixed fallback must not
        // build on top of it.
        let mut taken = first_round_state(60, 4, 1);
        let pid = taken.current_player_id();
        let mut hand = vec![
            industry(IndustryType::CoalMine),
            industry(IndustryType::IronWorks),
        ];
        hand.extend(filler_cards(6));
        taken.players[pid].hand = hand;
        taken.iron_market = 0;
        taken.money_spent_this_round[0] = 0;
        taken.links[17] = Some(crate::state::Link {
            player: 0,
            is_canal: true,
        });
        match OpeningRule::ResourceNetwork.fixed_move(&taken) {
            ResolvedMove::Network { conn_id, .. } => assert_ne!(conn_id, 17),
            other => panic!("resource network returned {other:?}"),
        }

        // Different seats/hands hash to different routes, so seed sweeps are
        // not stuck on one opening route.
        let mut seen = Vec::new();
        for (index, extra) in [
            (1usize, vec![industry(IndustryType::CoalMine)]),
            (
                2usize,
                vec![
                    industry(IndustryType::IronWorks),
                    industry(IndustryType::CoalMine),
                ],
            ),
            (3usize, vec![Card::Location(Loc::Coalbrookdale)]),
        ] {
            let mut state = first_round_state(70 + index as u64, 4, index);
            let seat = state.current_player_id();
            let mut hand = vec![
                industry(IndustryType::CoalMine),
                industry(IndustryType::IronWorks),
            ];
            hand.extend(extra);
            state.players[seat].hand = hand;
            state.iron_market = 0;
            for &other in &state.turn_order[..state.current_index] {
                state.money_spent_this_round[other] = 0;
            }
            match OpeningRule::ResourceNetwork.fixed_move(&state) {
                ResolvedMove::Network { conn_id, .. } if !seen.contains(&conn_id) => {
                    seen.push(conn_id);
                }
                _ => {}
            }
        }
        assert!(
            seen.len() > 1,
            "resource opening should vary across seats/hands, got {seen:?}"
        );
    }

    #[test]
    fn brewery_double_develop_requires_iron_price_three_or_below() {
        let mut state = first_round_state(56, 4, 0);
        let pid = state.current_player_id();
        let mut hand = vec![industry(IndustryType::Brewery); 3];
        hand.extend(filler_cards(5));
        state.players[pid].hand = hand;
        state.iron_market = 4; // market price 4 -> double develop must not fire

        assert_ne!(opening_rule(&state), Some(OpeningRule::BreweryDevelop));
    }
}
