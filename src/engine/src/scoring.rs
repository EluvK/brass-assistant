//! Era / end-game scoring. Translated from gameState.js calculateEraScore.

use crate::data::Era;
use crate::map::*;
use crate::state::GameState;

pub struct ScoreBreakdown {
    pub player_id: usize,
    pub link_vp: u16,
    pub industry_vp: u16,
    pub total_vp: u16,
}

pub fn score_era(state: &mut GameState) -> Vec<ScoreBreakdown> {
    let n = state.player_count();
    let mut scores: Vec<ScoreBreakdown> = (0..n)
        .map(|i| ScoreBreakdown {
            player_id: i,
            link_vp: 0,
            industry_vp: 0,
            total_vp: state.players[i].vp,
        })
        .collect();

    // Score links: only flipped industries contribute link icons; merchants score 2 each.
    for (id, link) in state.links.iter().enumerate() {
        let Some(l) = link else { continue };
        let conn = &connections()[id];
        let mut value = 0u16;

        for end in [conn.a, conn.b] {
            if end.is_city() {
                let slots = city_slots(end);
                for slot in 0..slots.len() {
                    if let Some(k) = state.city_slot_key(end, slot) {
                        if let Some(t) = &state.city_tiles[k] {
                            if t.flipped {
                                value += t.def.link_vp as u16;
                            }
                        }
                    }
                }
            }
            if end.is_merchant() {
                value += 2;
            }
            if end.is_farm() {
                if let Some(t) = state.farm_tile(end) {
                    if t.flipped {
                        value += t.def.link_vp as u16;
                    }
                }
            }
        }
        // Also score farms the link passes through (e.g. via southern).
        if let Some(via) = conn.via_farm {
            if let Some(t) = state.farm_tile(via) {
                if t.flipped {
                    value += t.def.link_vp as u16;
                }
            }
        }

        scores[l.player].link_vp += value;
    }

    // Score flipped industries.
    for tile in state.city_tiles.iter().flatten() {
        if tile.flipped {
            scores[tile.player].industry_vp += tile.def.vp as u16;
        }
    }
    for tile in state.farm_tiles.iter().flatten() {
        if tile.flipped {
            scores[tile.player].industry_vp += tile.def.vp as u16;
        }
    }

    // Apply.
    for s in scores.iter_mut() {
        let p = &mut state.players[s.player_id];
        p.vp += s.link_vp + s.industry_vp;
        s.total_vp = p.vp;
    }

    scores
}

/// Final result: ranks players by VP (ties broken by income level, then cash).
pub fn final_ranking(state: &GameState) -> Vec<usize> {
    let mut order: Vec<usize> = (0..state.player_count()).collect();
    order.sort_by(|a, b| {
        let pa = &state.players[*a];
        let pb = &state.players[*b];
        pb.vp
            .cmp(&pa.vp)
            .then(pb.income_level().cmp(&pa.income_level()))
            .then(pb.money.cmp(&pa.money))
    });
    order
}

pub fn _unused(_e: Era) {}
