//! Shared resource and card rules used by action implementations.

use crate::data::IndustryType;
use crate::graph::{
    CoalSource, IronSource, find_coal_sources, find_iron_sources, is_in_network,
    player_has_presence,
};
use crate::map::{Connection, Loc};
use crate::state::{Card, GameState, Player};
use std::collections::HashMap;

/// A coal or iron source that may be free on the board or paid from supply.
pub(crate) trait PricedResourceSource {
    fn is_free(self) -> bool;
    fn price(self) -> i32;
}

impl PricedResourceSource for CoalSource {
    fn is_free(self) -> bool {
        self.free
    }

    fn price(self) -> i32 {
        self.price as i32
    }
}

impl PricedResourceSource for IronSource {
    fn is_free(self) -> bool {
        self.free
    }

    fn price(self) -> i32 {
        self.price as i32
    }
}

/// Cost of taking `needed` sources in their rules-defined order (free sources
/// first, then cheapest supply slots). `None` means too few sources exist.
pub(crate) fn source_purchase_cost<T: PricedResourceSource + Copy>(
    sources: &[T],
    needed: usize,
) -> Option<i32> {
    if sources.len() < needed {
        return None;
    }
    Some(
        sources
            .iter()
            .take(needed)
            .map(|source| if source.is_free() { 0 } else { source.price() })
            .sum(),
    )
}

/// Count affordable sources, stopping at the action's maximum requirement.
pub(crate) fn affordable_source_count<T: PricedResourceSource + Copy>(
    sources: &[T],
    mut money: i32,
    maximum: usize,
) -> usize {
    let mut count = 0;
    for source in sources.iter().take(maximum) {
        if source.is_free() {
            count += 1;
        } else if money >= source.price() {
            money -= source.price();
            count += 1;
        } else {
            break;
        }
    }
    count
}

pub(crate) fn log_loc_label(loc: Loc) -> String {
    format!("{}({})", loc.zh_name(), loc.name())
}

pub(crate) fn log_industry_label(ind: IndustryType) -> &'static str {
    match ind {
        IndustryType::CottonMill => "棉纺厂",
        IndustryType::CoalMine => "煤矿",
        IndustryType::IronWorks => "铁厂",
        IndustryType::Manufacturer => "制造厂",
        IndustryType::Pottery => "陶器厂",
        IndustryType::Brewery => "啤酒厂",
    }
}

// ---------------------------------------------------------------------------
// Resource source selection & validation
// ---------------------------------------------------------------------------

/// All legal selections of `needed` source cubes from `available`, respecting
/// the free-first rule: free sources must be used before the market, and the
/// player may choose WHICH free sources to drain (that choice decides whose
/// building flips).
fn source_options<T: Copy + PartialEq>(
    available: &[T],
    needed: usize,
    is_free: impl Fn(&T) -> bool,
    key: impl Fn(&T) -> usize,
) -> Vec<Vec<T>> {
    if needed == 0 {
        return vec![vec![]];
    }
    let free_count = available.iter().filter(|s| is_free(s)).count();
    let market_needed = needed.saturating_sub(free_count);
    let free_needed = needed - market_needed;

    // Group free sources by key (one entry per cube).
    let mut free_groups: Vec<(usize, T, usize)> = Vec::new();
    for s in available.iter().filter(|s| is_free(s)) {
        let k = key(s);
        if let Some((_, _, c)) = free_groups.iter_mut().find(|(kk, _, _)| *kk == k) {
            *c += 1;
        } else {
            free_groups.push((k, *s, 1));
        }
    }

    // Enumerate multisets of free source keys totaling `free_needed` cubes.
    let mut free_combos: Vec<Vec<T>> = Vec::new();
    let mut cur: Vec<T> = Vec::new();
    fn rec<T: Copy + PartialEq>(
        groups: &[(usize, T, usize)],
        idx: usize,
        remaining: usize,
        cur: &mut Vec<T>,
        out: &mut Vec<Vec<T>>,
    ) {
        if remaining == 0 {
            out.push(cur.clone());
            return;
        }
        if idx >= groups.len() {
            return;
        }
        let (_, rep, count) = groups[idx];
        let take_max = remaining.min(count);
        for take in 0..=take_max {
            for _ in 0..take {
                cur.push(rep);
            }
            rec(groups, idx + 1, remaining - take, cur, out);
            for _ in 0..take {
                cur.pop();
            }
        }
    }
    rec(&free_groups, 0, free_needed, &mut cur, &mut free_combos);

    // Market sources are listed cheapest-first (per-slot prices, then General
    // Supply). Fill the shortfall with the cheapest `market_needed` distinct
    // entries so a multi-cube market purchase pays the ascending slot prices,
    // not `market_needed` copies of the cheapest cube.
    let market_sources: Vec<T> = available.iter().filter(|s| !is_free(s)).copied().collect();
    let mut out = Vec::new();
    for combo in free_combos {
        let mut sel = combo;
        let take = market_needed.min(market_sources.len());
        sel.extend_from_slice(&market_sources[..take]);
        out.push(sel);
    }
    out
}

/// All legal coal source selections for `needed` cubes at `loc`.
pub fn coal_source_options(state: &GameState, loc: Loc, needed: usize) -> Vec<Vec<CoalSource>> {
    source_options(
        &find_coal_sources(state, loc),
        needed,
        |s| s.free,
        |s| s.key,
    )
}

/// All legal iron source selections for `needed` cubes.
pub fn iron_source_options(state: &GameState, needed: usize) -> Vec<Vec<IronSource>> {
    source_options(&find_iron_sources(state), needed, |s| s.free, |s| s.key)
}

/// Coal sources reachable from either endpoint of a connection (union, one
/// entry per cube), used for single/double rail legality.
fn coal_sources_for_connection(state: &GameState, conn: &Connection) -> Vec<CoalSource> {
    let mut out = find_coal_sources(state, conn.a);
    for s in find_coal_sources(state, conn.b) {
        if !out.contains(&s) {
            out.push(s);
        }
    }
    out
}

/// All legal coal source selections for a single rail link on `conn`.
pub fn coal_options_for_connection(
    state: &GameState,
    conn: &Connection,
    needed: usize,
) -> Vec<Vec<CoalSource>> {
    source_options(
        &coal_sources_for_connection(state, conn),
        needed,
        |s| s.free,
        |s| s.key,
    )
}

/// Validate a chosen source set: correct count, each currently available, and
/// the free-first rule (market may only cover the shortfall).
fn validate_source_choice<T: Copy + Eq + std::hash::Hash>(
    available: &[T],
    needed: usize,
    chosen: &[T],
    is_free: impl Fn(&T) -> bool,
) -> Result<(), String> {
    if chosen.len() != needed {
        return Err(format!("Need {needed} source(s), got {}", chosen.len()));
    }
    let free_avail = available.iter().filter(|s| is_free(s)).count();
    let market_needed = needed.saturating_sub(free_avail);
    let market_chosen = chosen.iter().filter(|s| !is_free(s)).count();
    if market_chosen != market_needed {
        return Err("Free sources must be used before the market".into());
    }
    // Chosen must be a sub-multiset of currently available sources. A source's
    // full value is its identity: market slots share `key == usize::MAX`, but
    // their prices (and coal source kind) must not be forgeable by raw input.
    let mut avail_counts: HashMap<T, usize> = HashMap::new();
    for s in available {
        *avail_counts.entry(*s).or_insert(0) += 1;
    }
    for s in chosen {
        let c = avail_counts
            .get_mut(s)
            .ok_or("Chosen source is not available")?;
        if *c == 0 {
            return Err("Chosen source is not available".into());
        }
        *c -= 1;
    }
    Ok(())
}

pub(crate) fn validate_coal_choice(
    state: &GameState,
    loc: Loc,
    needed: usize,
    chosen: &[CoalSource],
) -> Result<(), String> {
    validate_source_choice(&find_coal_sources(state, loc), needed, chosen, |s| s.free)
}

pub(crate) fn validate_connection_coal_choice(
    state: &GameState,
    conn: &Connection,
    chosen: &[CoalSource],
) -> Result<(), String> {
    validate_source_choice(&coal_sources_for_connection(state, conn), 1, chosen, |s| {
        s.free
    })
}

pub(crate) fn validate_iron_choice(
    state: &GameState,
    needed: usize,
    chosen: &[IronSource],
) -> Result<(), String> {
    validate_source_choice(&find_iron_sources(state), needed, chosen, |s| s.free)
}

#[cfg(test)]
mod tests {
    use super::validate_source_choice;
    use crate::graph::{CoalSource, CoalSourceKind, IronSource};

    #[test]
    fn rejects_forged_market_resource_fields() {
        let market_coal = CoalSource {
            kind: CoalSourceKind::Market,
            key: usize::MAX,
            price: 5,
            free: false,
        };
        let forged_price = CoalSource {
            price: 0,
            ..market_coal
        };
        let forged_kind = CoalSource {
            kind: CoalSourceKind::Mine,
            ..market_coal
        };
        for forged in [forged_price, forged_kind] {
            assert_eq!(
                validate_source_choice(&[market_coal], 1, &[forged], |s| s.free),
                Err("Chosen source is not available".into())
            );
        }

        let market_iron = IronSource {
            key: usize::MAX,
            price: 4,
            free: false,
        };
        let forged_iron = IronSource {
            price: 0,
            ..market_iron
        };
        assert_eq!(
            validate_source_choice(&[market_iron], 1, &[forged_iron], |s| s.free),
            Err("Chosen source is not available".into())
        );
    }
}

// ---------------------------------------------------------------------------
// Card helpers
// ---------------------------------------------------------------------------

pub fn discard_card(state: &mut GameState, player_id: usize, card_index: usize) {
    let p = &mut state.players[player_id];
    if card_index >= p.hand.len() {
        return;
    }
    let removed = p.hand[card_index].clone();
    match removed.ctype() {
        crate::data::CardType::WildLocation => {
            state.wild_location_pile += 1;
            p.has_wild_location = false;
        }
        crate::data::CardType::WildIndustry => {
            state.wild_industry_pile += 1;
            p.has_wild_industry = false;
        }
        _ => {
            // Non-wild discarded cards join the face-down discard pile. The
            // per-player played history mirrors this anonymous pile (wilds
            // return to the supply instead, so they never enter either).
            state.discard_pile.push(removed.clone());
            p.played.push(removed);
        }
    }
    p.hand.remove(card_index);
}

pub(crate) fn require_card_index(
    state: &GameState,
    pid: usize,
    card_index: usize,
) -> Result<(), String> {
    if card_index >= state.players[pid].hand.len() {
        return Err("Invalid card index".into());
    }
    Ok(())
}

/// Indices of cards in `player`'s hand valid for a BUILD at `loc`/`ind`.
/// `loc` being a farm restricts to industry/wild-industry cards.
pub fn valid_build_cards(
    state: &GameState,
    player: &Player,
    pid: usize,
    loc: Loc,
    ind: IndustryType,
) -> Vec<usize> {
    let is_farm = loc.is_farm();
    let in_network = is_in_network(state, pid, loc);
    let industry_ok = in_network || !player_has_presence(state, pid);
    let mut out = Vec::new();
    for (i, card) in player.hand.iter().enumerate() {
        match card {
            Card::Location(cl) => {
                if !is_farm && *cl == loc {
                    out.push(i);
                }
            }
            Card::Industry { .. } => {
                if card.is_industry(ind) && industry_ok {
                    out.push(i);
                }
            }
            Card::WildLocation => {
                if !is_farm {
                    out.push(i);
                }
            }
            Card::WildIndustry => {
                if industry_ok {
                    out.push(i);
                }
            }
        }
    }
    out
}

/// Any card may be discarded for network/develop/sell/loan/pass.
pub fn any_card_indices(player: &Player) -> Vec<usize> {
    (0..player.hand.len()).collect()
}
