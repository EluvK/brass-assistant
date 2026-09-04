//! Hidden-information determinization shared by search implementations.
//!
//! Brass hands are private information.  A determinized state keeps the
//! current player's known hand intact and samples every opponent's non-wild
//! cards from the remaining era card pool.  The helper is shared by the
//! network-guided ISMCTS implementation and the Python diagnostics API.

use crate::state::{Card, GameState, deck_composition};
use rand::seq::SliceRandom;
use rand_chacha::ChaCha12Rng;

fn is_wild(card: &Card) -> bool {
    matches!(card, Card::WildLocation | Card::WildIndustry)
}

/// Sample opponent hands from the hidden pool, leaving our own hand intact.
///
/// The pool is the full era deck minus our known non-wild hand cards and every
/// card already out of circulation this era (the face-down discard pile).  A
/// sampled state therefore preserves the card multiset invariant while still
/// representing one possible hidden-information world.
pub fn determinize(state: &GameState, rng: &mut ChaCha12Rng) -> GameState {
    let mut det = state.clone();
    let me = det.current_player_id();

    let mut pool = deck_composition(det.player_count());
    let known: Vec<Card> = det.players[me]
        .hand
        .iter()
        .filter(|card| !is_wild(card))
        .cloned()
        .collect();
    for card in &known {
        if let Some(idx) = pool.iter().position(|candidate| candidate == card) {
            pool.swap_remove(idx);
        }
    }

    for card in &det.discard_pile {
        if let Some(idx) = pool.iter().position(|candidate| candidate == card) {
            pool.swap_remove(idx);
        }
    }
    pool.shuffle(rng);

    for i in 0..det.player_count() {
        if i == me {
            continue;
        }
        let wilds: Vec<Card> = det.players[i]
            .hand
            .iter()
            .filter(|card| is_wild(card))
            .cloned()
            .collect();
        let need = det.players[i].hand.len() - wilds.len();
        let mut new_hand = wilds;
        for _ in 0..need {
            match pool.pop() {
                Some(card) => new_hand.push(card),
                None => break,
            }
        }
        det.players[i].hand = new_hand;
    }

    debug_assert_eq!(
        det.deck.len(),
        pool.len(),
        "determinize pool must equal the deck"
    );
    det.deck = pool;
    det
}
