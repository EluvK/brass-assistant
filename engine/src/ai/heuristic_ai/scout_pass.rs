// SCOUT
// ---------------------------------------------------------------------------

const LOW_KEEP_SCORE: f64 = 1.0;
const HIGH_KEEP_SCORE: f64 = 1.8;
const DESIRED_HIGH_VALUE_CARDS: usize = 2;
const MAX_HAND_REFRESH_SCORE: f64 = 1.92;

/// Estimate how urgently the cards retained after scouting need replacing.
///
/// The three least valuable cards are discarded, so a hand with only three
/// dead cards and several strong cards should not receive a broad hand-quality
/// bonus. A hand still dominated by weak cards, with too few valuable cards to
/// preserve, benefits substantially more from the two wild cards Scout gains.
fn scout_hand_refresh_score(card_choices: &CardChoices) -> f64 {
    let retained = &card_choices[3..];
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
    let high_value_shortfall = (DESIRED_HIGH_VALUE_CARDS.saturating_sub(high_value_count) as f64
        / DESIRED_HIGH_VALUE_CARDS as f64)
        .clamp(0.0, 1.0);
    let average_keep_score = retained.iter().map(|(_, score)| score).sum::<f64>() / retained_count;
    let average_quality_shortfall = ((1.15 - average_keep_score) / 1.15).clamp(0.0, 1.0);

    MAX_HAND_REFRESH_SCORE
        * low_value_ratio
        * (0.35 + 0.65 * high_value_shortfall)
        * average_quality_shortfall
}

fn score_scout_plan(
    state: &GameState,
    pid: usize,
    card_choices: &CardChoices,
) -> Option<Decision> {
    if !can_scout(state, pid) {
        return None;
    }
    if card_choices.len() < 3 {
        return None;
    }
    let discard = &card_choices[..3];
    let dead_count = discard.iter().filter(|(_, s)| *s <= 0.0).count();
    // Scout has no immediate VP, income, or money effect. Score it directly
    // from the cards safely discarded and the quality of the hand it refreshes.
    let discard_score = dead_count as f64 * 0.96 - (3.0 - dead_count as f64) * 0.48;
    let hand_refresh_score = scout_hand_refresh_score(card_choices);
    let score = discard_score + hand_refresh_score;

    // legal_moves emits Scout combinations in ascending index order. Keep
    // the teacher canonical representation identical regardless of the
    // utility ranking order used to choose discarded cards.
    let mut card_indices: [usize; 3] = [discard[0].0, discard[1].0, discard[2].0];
    card_indices.sort_unstable();
    Some(Decision {
        mv: Move::Scout { card_indices },
        score,
        card_score: discard.iter().map(|(_, s)| *s).sum(),
    })
}

#[cfg(test)]
mod scout_pass_tests {
    use super::scout_hand_refresh_score;

    #[test]
    fn scout_hand_refresh_rewards_a_weak_retained_hand_without_high_value_cards() {
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
            scout_hand_refresh_score(&weak_hand) > scout_hand_refresh_score(&strong_remainder)
        );
    }
}

// ---------------------------------------------------------------------------
// PASS
// ---------------------------------------------------------------------------

fn score_pass_result(
    _state: &GameState,
    _pid: usize,
    card_choices: &CardChoices,
) -> Option<Decision> {
    let card_index = card_choices.first().map(|(index, _)| *index).unwrap_or(0);
    Some(Decision {
        mv: Move::Pass { card_index },
        score: -5.0,
        card_score: card_choices.first().map(|(_, s)| *s).unwrap_or(f64::INFINITY),
    })
}
