// SCOUT
// ---------------------------------------------------------------------------

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
    let score = vp_equivalent(
        state,
        0.0,
        0.0,
        0.0,
        dead_count as f64 * 1.2 - (3.0 - dead_count as f64) * 0.6,
    );

    let card_indices: [usize; 3] = [discard[0].0, discard[1].0, discard[2].0];
    Some(Decision {
        mv: Move::Scout { card_indices },
        score,
    })
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
    })
}
