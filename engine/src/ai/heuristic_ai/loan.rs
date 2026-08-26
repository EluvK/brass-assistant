// LOAN
// ---------------------------------------------------------------------------

fn score_loan_result(
    state: &GameState,
    pid: usize,
    card_choices: &CardChoices,
) -> Option<Decision> {
    if !state.can_take_loan(pid) {
        return None;
    }
    let player = &state.players[pid];
    let income_level = player.income_level();
    let plan = compute_plan(state, pid);

    // A loan is the engine of the early economy: it buys the income-recovery
    // build that keeps the turn engine running. It is worth taking when cash is
    // so low that without it we'd idle-pass, provided income isn't already in
    // a death spiral.
    let rounds_left = state.rounds_remaining();
    let income_cost = crate::map::LOAN_INCOME_PENALTY as f64 * income_weight(state);
    let _ = rounds_left;

    // The post-loan budget opens up builds that buy back income fast.
    let cash = player.money as f64;
    let after =
        best_affordable_build_score(state, pid, cash + crate::map::LOAN_AMOUNT as f64, &plan);
    let now = best_affordable_build_score(state, pid, cash, &plan);
    let gain = (after - now).max(0.0);

    // Same-turn combo value: Loan is strongest when it immediately unlocks a
    // productive second action (Loan -> Build / Network / Develop / Sell).
    let mut combo_gain = 0.0;
    if cash < 24.0 && rounds_left > 1.5 {
        if let Some(sim_gain) = best_same_turn_after_loan(state, pid, card_choices) {
            combo_gain = sim_gain.max(0.0) * 0.7;
        }
    }

    // Idle protection: if the current hand can't do anything positive, borrow.
    let idle_bonus = if cash < 18.0 { 2.0 } else { 0.0 };

    // Income floor: never borrow into deep debt.
    let post_loan_income = income_level - crate::map::LOAN_INCOME_PENALTY;
    let floor_penalty = if post_loan_income <= -7 {
        7.0
    } else if post_loan_income <= -4 {
        2.0
    } else if post_loan_income <= 0 {
        0.3
    } else {
        0.0
    };

    // Do not over-borrow with plenty of cash, and avoid debt spiral in late
    // era where income recovery window is too short.
    let rich_penalty = if cash >= 55.0 {
        5.0
    } else if cash >= 42.0 {
        2.4
    } else if cash >= 30.0 {
        1.0
    } else {
        0.0
    };
    let late_era_penalty = 0.0;
    // let late_era_penalty = if rounds_left <= 1.0 {
    //     3.0
    // } else if rounds_left <= 1.5 {
    //     1.0
    // } else {
    //     0.0
    // };
    let unlock_bonus = if now <= 0.0 && after > 0.8 { 3.2 } else { 0.0 };

    // Startup-loan peak: Canal-Early round 1-2 with low cash — the loan funds
    // the economy engine (build/develop) before tempo stalls.
    let startup_loan_bonus = if era_phase(state) == Phase::CanalEarly && state.round <= 2 {
        if cash < 18.0 { 6.0 } else { 0.5 }
    } else {
        0.0
    };

    // Canal-Late end-of-era loan: round 6-8, cash below £30 — borrow the rail-era
    // startup capital. The -3 income is acceptable because the very next action
    // (selling our built goods / flipping a sellable) recovers income fast.
    // Only if there is a sellable tile we can realistically flip soon.
    let canal_late_loan_bonus = if era_phase(state) == Phase::CanalLate && state.round >= 6 {
        let can_flip_soon = get_valid_sell_targets(state, pid).len() > 0
            || state
                .city_tiles
                .iter()
                .flatten()
                .any(|t| t.player == pid && !t.flipped && t.ind.is_sellable());
        if can_flip_soon && cash < 30.0 {
            if cash < 18.0 { 2.8 } else { 1.8 }
        } else {
            0.0
        }
    } else {
        0.0
    };

    let score =
        gain + combo_gain + idle_bonus + unlock_bonus + startup_loan_bonus + canal_late_loan_bonus
            - income_cost
            - floor_penalty
            - rich_penalty
            - late_era_penalty;
    let card_index = card_choices.first().map(|(index, _)| *index)?;
    Some(Decision {
        mv: Move::Loan { card_index },
        score,
    })
}

fn best_same_turn_after_loan(
    state: &GameState,
    pid: usize,
    card_choices: &CardChoices,
) -> Option<f64> {
    let card_index = card_choices.first().map(|(index, _)| *index)?;
    let mut sim = state.clone();
    crate::rules::apply_move(&mut sim, &Move::Loan { card_index }).ok()?;
    let tr = advance_turn(&mut sim);
    match tr {
        TurnResult::Continue => {}
        TurnResult::EndCanalEra | TurnResult::EndGame => return Some(0.0),
    }
    if sim.current_player_id() != pid {
        return Some(0.0);
    }
    let cands = candidate_actions_k(&mut sim, 3);
    cands
        .into_iter()
        .filter(|d| !matches!(d.mv, Move::Loan { .. }))
        .map(|d| d.score)
        .max_by(|a, b| a.partial_cmp(b).unwrap())
}

/// Best build score the player could afford within a cash budget.
fn best_affordable_build_score(state: &GameState, pid: usize, budget: f64, plan: &Plan) -> f64 {
    let mut best = f64::NEG_INFINITY;
    for t in get_valid_build_targets(state, pid) {
        if (t.cost_total as f64) > budget {
            continue;
        }
        let s = score_build_candidate(state, pid, &t, plan);
        if s > best {
            best = s;
        }
    }
    if best == f64::NEG_INFINITY { 0.0 } else { best }
}
