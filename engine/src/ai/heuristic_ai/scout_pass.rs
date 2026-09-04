//! Scout and pass scoring.

use super::{CardChoices, Decision, cards::plain_location_keep_score};
use crate::rules::{ResolvedMove, can_scout};
use crate::state::GameState;

// Scout is a self-contained policy.  Its weights are deliberately local: the
// operation has no shared currency conversion or phase profile to resolve.
const SCOUT_DISCARD_COUNT: usize = 3;
const LOW_KEEP_SCORE: f64 = 1.0;
const HIGH_KEEP_SCORE: f64 = 1.8;
const DESIRED_HIGH_VALUE_COUNT: usize = 2;
const MAX_REFRESH_SCORE: f64 = 5.0;
const DEAD_DISCARD_VALUE: f64 = 0.96;
const ALIVE_DISCARD_PENALTY: f64 = 0.48;
const REFRESH_BASE_FACTOR: f64 = 0.35;
const REFRESH_HIGH_VALUE_FACTOR: f64 = 0.65;

/// Fallback used only when candidate generation has no executable action.
pub(crate) const PASS_FALLBACK_SCORE: f64 = -0.5;

/// Estimate how urgently the cards retained after scouting need replacing.
///
/// The three least valuable cards are discarded, so a hand with only three
/// dead cards and several strong cards should not receive a broad
/// hand-quality bonus. A hand still dominated by weak cards, with too few
/// valuable cards to preserve, benefits substantially more from the two
/// wild cards Scout gains.
fn scout_hand_refresh_score(state: &GameState, card_choices: &CardChoices) -> f64 {
    let Some(retained) = card_choices.get(SCOUT_DISCARD_COUNT..) else {
        return 0.0;
    };
    if retained.is_empty() {
        return 0.0;
    }

    let retained_count = retained.len() as f64;
    let low_value_ratio = retained
        .iter()
        .filter(|(_, score)| *score <= LOW_KEEP_SCORE)
        .count() as f64
        / retained_count;
    let high_value_count = retained
        .iter()
        .filter(|(_, score)| *score >= HIGH_KEEP_SCORE)
        .count();
    let high_value_shortfall = (DESIRED_HIGH_VALUE_COUNT.saturating_sub(high_value_count) as f64
        / DESIRED_HIGH_VALUE_COUNT as f64)
        .clamp(0.0, 1.0);
    // Average quality shortfall measured against this era's plain location
    // card baseline.
    let anchor = plain_location_keep_score(state);
    let average_keep_score = retained.iter().map(|(_, score)| score).sum::<f64>() / retained_count;
    let average_quality_shortfall = ((anchor - average_keep_score) / anchor).clamp(0.0, 1.0);

    MAX_REFRESH_SCORE
        * low_value_ratio
        * (REFRESH_BASE_FACTOR + REFRESH_HIGH_VALUE_FACTOR * high_value_shortfall)
        * average_quality_shortfall
}

pub(super) fn score_scout_plan(state: &GameState, card_choices: &CardChoices) -> Option<Decision> {
    let pid = state.current_player_id();
    if !can_scout(state, pid) {
        return None;
    }
    if card_choices.len() < SCOUT_DISCARD_COUNT {
        return None;
    }
    let discard = &card_choices[..SCOUT_DISCARD_COUNT];
    let dead_count = discard.iter().filter(|(_, s)| *s <= 0.0).count();
    // Scout has no immediate VP, income, or money effect. Its value is the
    // dead cards safely discarded plus the quality of the refreshed hand.
    // `MAX_REFRESH_SCORE` is scaled so a perfect refresh competes with a solid
    // build instead of being invisible next to one.
    let discard_score = dead_count as f64 * DEAD_DISCARD_VALUE
        - (SCOUT_DISCARD_COUNT as f64 - dead_count as f64) * ALIVE_DISCARD_PENALTY;
    let hand_refresh_score = scout_hand_refresh_score(state, card_choices);
    let score = discard_score + hand_refresh_score;

    // legal_resolved_moves emits Scout combinations in ascending index
    // order. Keep the teacher canonical representation identical regardless
    // of the utility ranking order used to choose discarded cards.
    let mut card_indices: [usize; SCOUT_DISCARD_COUNT] = [discard[0].0, discard[1].0, discard[2].0];
    card_indices.sort_unstable();
    Some(Decision {
        mv: ResolvedMove::Scout { card_indices },
        score,
        card_score: discard.iter().map(|(_, s)| *s).sum(),
    })
}

/// Pass does nothing: no cash, income, VP, or flexibility change, so its
/// score is naturally zero — below any positive-value action, above any
/// money-losing one. In the unified currency there is no need for a magic
/// negative constant.
pub(super) fn score_pass_result(card_choices: &CardChoices) -> Option<Decision> {
    let card_index = card_choices.first().map(|(index, _)| *index).unwrap_or(0);
    Some(Decision {
        mv: ResolvedMove::Pass { card_index },
        score: 0.0,
        card_score: card_choices
            .first()
            .map(|(_, s)| *s)
            .unwrap_or(f64::INFINITY),
    })
}

#[cfg(test)]
mod scout_pass_tests {
    use super::scout_hand_refresh_score;
    use crate::state::GameState;
    use rand_chacha::ChaCha12Rng;
    use rand_chacha::rand_core::SeedableRng;

    #[test]
    fn scout_hand_refresh_rewards_a_weak_retained_hand_without_high_value_cards() {
        let state = GameState::new(ChaCha12Rng::seed_from_u64(7), 2);
        let weak_hand = vec![
            (0, -0.4),
            (1, -0.3),
            (2, -0.2),
            (3, 0.0),
            (4, 0.2),
            (5, 0.4),
            (6, 0.5),
            (7, 0.7),
        ];
        let strong_remainder = vec![
            (0, -0.4),
            (1, -0.3),
            (2, -0.2),
            (3, 2.1),
            (4, 2.4),
            (5, 2.8),
            (6, 3.1),
            (7, 3.8),
        ];

        assert!(
            scout_hand_refresh_score(&state, &weak_hand)
                > scout_hand_refresh_score(&state, &strong_remainder)
        );
    }
}
