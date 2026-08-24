//! LOAN, SCOUT, and PASS action execution.

use super::{discard_card, require_card_index};
use crate::map::{LOAN_AMOUNT, LOAN_INCOME_PENALTY};
use crate::state::{Card, GameState};
// ---------------------------------------------------------------------------
// LOAN / SCOUT / PASS
// ---------------------------------------------------------------------------

pub fn execute_loan(
    state: &mut GameState,
    pid: usize,
    card_index: usize,
) -> Result<String, String> {
    require_card_index(state, pid, card_index)?;
    if !state.can_take_loan(pid) {
        return Err("A loan cannot take your income below -10".into());
    }
    state.gain_money(pid, LOAN_AMOUNT);
    state.apply_loan_income_drop(pid);
    discard_card(state, pid, card_index);
    Ok(format!(
        "Took £{} loan (income -{} levels)",
        LOAN_AMOUNT, LOAN_INCOME_PENALTY
    ))
}

pub fn can_scout(state: &GameState, pid: usize) -> bool {
    let p = &state.players[pid];
    if p.hand.len() < 3 {
        return false;
    }
    if p.has_wild_location || p.has_wild_industry {
        return false;
    }
    state.wild_location_pile > 0 && state.wild_industry_pile > 0
}

pub fn execute_scout(
    state: &mut GameState,
    pid: usize,
    card_indices: [usize; 3],
) -> Result<String, String> {
    let mut unique = card_indices.to_vec();
    unique.sort_unstable();
    unique.dedup();
    if unique.len() != 3
        || unique
            .iter()
            .any(|&idx| idx >= state.players[pid].hand.len())
    {
        return Err("Scout requires three distinct valid card indices".into());
    }
    let p = &mut state.players[pid];
    if p.hand.len() < 3 {
        return Err("Must have 3 cards to scout".into());
    }
    if p.has_wild_location || p.has_wild_industry {
        return Err("Already have wild cards".into());
    }
    if state.wild_location_pile == 0 || state.wild_industry_pile == 0 {
        return Err("No wild cards available".into());
    }

    let mut sorted = card_indices;
    sorted.sort_by(|a, b| b.cmp(a));
    for idx in sorted {
        if idx < p.hand.len() {
            let removed = p.hand[idx].clone();
            match removed.ctype() {
                // Scout requires the player to hold no wilds, so the three
                // cards are always non-wild; the arm is defensive only.
                crate::data::CardType::WildLocation | crate::data::CardType::WildIndustry => {}
                _ => {
                    state.discard_pile.push(removed.clone());
                    p.played.push(removed);
                }
            }
            p.hand.remove(idx);
        }
    }
    p.hand.push(Card::WildLocation);
    p.hand.push(Card::WildIndustry);
    p.has_wild_location = true;
    p.has_wild_industry = true;
    state.wild_location_pile -= 1;
    state.wild_industry_pile -= 1;
    Ok("Scouted: gained Wild Location + Wild Industry".into())
}

pub fn execute_pass(
    state: &mut GameState,
    pid: usize,
    card_index: usize,
) -> Result<String, String> {
    require_card_index(state, pid, card_index)?;
    discard_card(state, pid, card_index);
    Ok("Passed".into())
}
