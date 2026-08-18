// SCOUT
// ---------------------------------------------------------------------------

fn card_usefulness(state: &GameState, pid: usize, card: &Card) -> f64 {
    match card {
        Card::WildLocation | Card::WildIndustry => 5.0,
        Card::Location(loc) => {
            if loc.is_farm() {
                return 0.0;
            }
            let remaining = &state.players[pid];
            for slot_types in city_slots(*loc) {
                for t in *slot_types {
                    if remaining.remaining_count(*t) > 0 {
                        return 2.0;
                    }
                }
            }
            0.0
        }
        Card::Industry { .. } => {
            let remaining = &state.players[pid];
            let has = if let Card::Industry { industries, n } = card {
                industries[..*n as usize]
                    .iter()
                    .any(|t| remaining.remaining_count(*t) > 0)
            } else {
                false
            };
            if has { 1.5 } else { 0.0 }
        }
    }
}

fn score_scout_plan(state: &GameState, pid: usize) -> Option<Decision> {
    if !can_scout(state, pid) {
        return None;
    }
    let player = &state.players[pid];
    let mut usefulness: Vec<(usize, f64)> = player
        .hand
        .iter()
        .enumerate()
        .map(|(i, c)| (i, card_usefulness(state, pid, c)))
        .collect();
    usefulness.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

    let discard = &usefulness[..3];
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

fn score_pass_result(state: &GameState, pid: usize) -> Option<Decision> {
    let card_index = pick_any_card(state, pid).unwrap_or(0);
    Some(Decision {
        mv: Move::Pass { card_index },
        score: -5.0,
    })
}
