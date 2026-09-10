//! State -> token tensors (Brass: Birmingham).
//!
//! Contract: `docs/ai-action-encoding.md`. One state becomes five feature
//! groups; Python projects each group to `d_model` and concatenates them into
//! a 102-token sequence.
//!
//!   cells      (49, F_CELL)      47 city slots + 2 brewery farms
//!   links      (39, F_LINK)      the 39 map connections
//!   merchants  (9,  F_MERCHANT)  randomized buyer type + beer
//!   seats      (4,  F_SEAT)      public state + hand information
//!   global     (F_GLOBAL,)       era, round, markets, turn queue
//!
//! Every observation is encoded from the acting player's perspective: seat 0 is
//! always "me", and owner / network membership planes are relative to that
//! rotation. The network never has to recover its own seat identity.

use crate::data::{Era, IndustryType};
use crate::map::{ALL_LOCATIONS, CITY_COUNT, city_slots, connections};
use crate::state::{BuyType, Card, GameState, loc_from_key};

pub const LOCATION_COUNT: usize = ALL_LOCATIONS.len(); // 27
pub const BOARD_CELLS: usize = 49; // 47 city slots + 2 farms
pub const CITY_CELLS: usize = 47;
pub const LINK_CELLS: usize = 39;
pub const MERCHANT_COUNT: usize = 9;
pub const SEAT_COUNT: usize = 4;
pub const INDUSTRY_COUNT: usize = 6;
pub const CARD_SEMANTIC_COUNT: usize = 35; // 27 locations + 6 industries + 2 wilds
pub const TOKEN_COUNT: usize = BOARD_CELLS + LINK_CELLS + MERCHANT_COUNT + SEAT_COUNT + 1; // 102

pub const STATE_TOKEN_SCHEMA_VERSION: usize = 1;

// ---------------------------------------------------------------------------
// cell features
// ---------------------------------------------------------------------------

/// Owner plane, relative to the acting player (index 0 = me). All zero when the
/// slot is empty.
pub const CELL_OWNER: usize = 0; // 4
pub const CELL_OCCUPIED: usize = 4;
pub const CELL_INDUSTRY: usize = 5; // 6
pub const CELL_FLIPPED: usize = 11;
pub const CELL_CUBES: usize = 12; // /5
pub const CELL_LEVEL: usize = 13; // /8
pub const CELL_VP: usize = 14; // /20
pub const CELL_INCOME: usize = 15; // /7
pub const CELL_IS_FARM: usize = 16;
pub const CELL_SLOT: usize = 17; // /4
pub const CELL_ALLOWED: usize = 18; // 6, static map capability
pub const CELL_IN_NET: usize = 24; // 4, relative
pub const CELL_DIST: usize = 28; // /6, links away from my network
pub const F_CELL: usize = 29;

// ---------------------------------------------------------------------------
// link features
// ---------------------------------------------------------------------------

pub const LINK_CANAL: usize = 0; // static: canal-buildable
pub const LINK_RAIL: usize = 1; // static: rail-buildable
pub const LINK_VIA_FARM: usize = 2;
pub const LINK_BUILT: usize = 3;
pub const LINK_OWNER: usize = 4; // 4, relative
pub const LINK_IS_CANAL: usize = 8;
pub const LINK_IN_NET: usize = 9; // 4, relative
pub const LINK_TOUCHES_NET: usize = 13; // 1: an endpoint is in my network
pub const F_LINK: usize = 14;

// ---------------------------------------------------------------------------
// merchant features
// ---------------------------------------------------------------------------

pub const MERCHANT_BUY: usize = 0; // 5: Blank/Any/Cotton/Manufacturer/Pottery
pub const MERCHANT_BEER: usize = 5;
pub const F_MERCHANT: usize = 6;

// ---------------------------------------------------------------------------
// seat features
// ---------------------------------------------------------------------------

pub const SEAT_MONEY: usize = 0; // /200
pub const SEAT_INCOME_SPACE: usize = 1; // /99
pub const SEAT_INCOME_LEVEL: usize = 2; // (level + 10) / 40
pub const SEAT_VP: usize = 3; // /200
pub const SEAT_CANAL_LINKS: usize = 4; // /14
pub const SEAT_RAIL_LINKS: usize = 5; // /14
pub const SEAT_HAND_SIZE: usize = 6; // /8
pub const SEAT_WILD_LOCATION: usize = 7;
pub const SEAT_WILD_INDUSTRY: usize = 8;
pub const SEAT_REMAINING: usize = 9; // 6, remaining tiles / stack size
pub const SEAT_SPENT: usize = 15; // /100
pub const SEAT_IS_CURRENT: usize = 16;
/// 1.0 when this seat's `hand_sampled` is a determinization draw rather than a
/// real hand.
pub const SEAT_HAND_SAMPLED_FLAG: usize = 17;
pub const SEAT_HAND_REAL: usize = 18; // 35
pub const SEAT_HAND_SAMPLED: usize = 53; // 35
pub const SEAT_HAND_PUBLIC: usize = 88; // 35
pub const F_SEAT: usize = 123;

// ---------------------------------------------------------------------------
// global features
// ---------------------------------------------------------------------------

pub const GLOBAL_ERA: usize = 0;
pub const GLOBAL_ROUND: usize = 1; // /8
pub const GLOBAL_ROUNDS_REMAINING: usize = 2; // /10
pub const GLOBAL_ACTIONS: usize = 3; // actions_this_turn / actions_per_turn
pub const GLOBAL_COAL_MARKET: usize = 4; // 15 one-hot
pub const GLOBAL_IRON_MARKET: usize = 19; // 11 one-hot
pub const GLOBAL_DECK: usize = 30; // /64
pub const GLOBAL_DISCARD: usize = 31; // /64
pub const GLOBAL_WILD_LOCATION_PILE: usize = 32;
pub const GLOBAL_WILD_INDUSTRY_PILE: usize = 33;
pub const GLOBAL_QUEUE: usize = 34; // 4x4: queue position -> relative seat
pub const F_GLOBAL: usize = 50;

// normalization constants
const CUBES_SCALE: f32 = 6.0;
const LEVEL_SCALE: f32 = 8.0;
const TILE_VP_SCALE: f32 = 20.0;
const INCOME_SCALE: f32 = 7.0;
const DIST_SCALE: f32 = 6.0;
const SLOT_SCALE: f32 = 4.0;
const MONEY_SCALE: f32 = 200.0;
const INCOME_SPACE_SCALE: f32 = 99.0;
const HAND_SCALE: f32 = 8.0;
const PLAYED_SCALE: f32 = 16.0;
const SPENT_SCALE: f32 = 100.0;
const LINK_SCALE: f32 = 14.0;
const DECK_SCALE: f32 = 64.0;

pub struct EncodedState {
    pub cells: Vec<f32>,
    pub links: Vec<f32>,
    pub merchants: Vec<f32>,
    pub seats: Vec<f32>,
    pub global: Vec<f32>,
}

/// Absolute seat id -> perspective-relative seat index (0 = `base`).
fn rel_seat(pid: usize, base: usize, n: usize) -> usize {
    (pid + n - base) % n
}

/// Card semantics id: 0-26 location, 27-32 industry icon, 33/34 wild.
fn card_semantics(card: &Card, out: &mut Vec<usize>) {
    match card {
        Card::Location(loc) => out.push(*loc as usize),
        Card::Industry { industries, n } => {
            for &ind in industries.iter().take(*n as usize) {
                out.push(LOCATION_COUNT + ind as usize);
            }
        }
        Card::WildLocation => out.push(LOCATION_COUNT + INDUSTRY_COUNT),
        Card::WildIndustry => out.push(LOCATION_COUNT + INDUSTRY_COUNT + 1),
    }
}

fn card_bag(cards: &[Card], scale: f32) -> Vec<f32> {
    let mut bag = vec![0.0f32; CARD_SEMANTIC_COUNT];
    let mut ids = Vec::new();
    for card in cards {
        ids.clear();
        card_semantics(card, &mut ids);
        for id in &ids {
            bag[*id] += 1.0;
        }
    }
    for v in bag.iter_mut() {
        *v /= scale;
    }
    bag
}

/// Public board-cell -> location mapping. City cells precede the two
/// brewery-farm cells, matching the cell feature order.
pub fn board_cell_locations() -> Vec<usize> {
    let mut out = Vec::with_capacity(BOARD_CELLS);
    for &loc in ALL_LOCATIONS[..CITY_COUNT].iter() {
        out.extend(std::iter::repeat_n(
            loc as usize,
            crate::map::city_slots(loc).len(),
        ));
    }
    out.push(crate::map::Loc::BreweryNorth as usize);
    out.push(crate::map::Loc::BrewerySouth as usize);
    debug_assert_eq!(out.len(), BOARD_CELLS);
    out
}

/// Per-cell slot index (`0` for the two farm cells).
pub fn board_cell_slots() -> Vec<usize> {
    let mut out = Vec::with_capacity(BOARD_CELLS);
    for &loc in ALL_LOCATIONS[..CITY_COUNT].iter() {
        for slot in 0..crate::map::city_slots(loc).len() {
            out.push(slot);
        }
    }
    out.push(0);
    out.push(0);
    debug_assert_eq!(out.len(), BOARD_CELLS);
    out
}

/// Flat `(a, b)` endpoint pairs for the 39 map connections. A via-farm edge
/// keeps its rule-level endpoints; `connection_via_farms` exposes its farm.
pub fn connection_endpoints() -> Vec<usize> {
    connections()
        .iter()
        .flat_map(|c| [c.a as usize, c.b as usize])
        .collect()
}

/// Per-connection via-farm location, or `LOCATION_COUNT` when none exists.
pub fn connection_via_farms() -> Vec<usize> {
    connections()
        .iter()
        .map(|c| c.via_farm.map(|loc| loc as usize).unwrap_or(LOCATION_COUNT))
        .collect()
}

/// Location adjacency used for the distance-to-my-network feature. A via-farm
/// connection links both endpoints to the farm as well.
fn location_adjacency() -> [Vec<usize>; LOCATION_COUNT] {
    let mut adj: [Vec<usize>; LOCATION_COUNT] = std::array::from_fn(|_| Vec::new());
    for c in connections() {
        let (a, b) = (c.a as usize, c.b as usize);
        adj[a].push(b);
        adj[b].push(a);
        if let Some(farm) = c.via_farm {
            let f = farm as usize;
            adj[a].push(f);
            adj[f].push(a);
            adj[b].push(f);
            adj[f].push(b);
        }
    }
    adj
}

/// Links away from the acting player's network, capped at `DIST_SCALE`.
fn distance_from_network(net_mask: u32) -> [f32; LOCATION_COUNT] {
    let adj = location_adjacency();
    let mut dist = [usize::MAX; LOCATION_COUNT];
    let mut queue = std::collections::VecDeque::new();
    for loc in 0..LOCATION_COUNT {
        if net_mask & (1u32 << loc) != 0 {
            dist[loc] = 0;
            queue.push_back(loc);
        }
    }
    while let Some(loc) = queue.pop_front() {
        for &next in &adj[loc] {
            if dist[next] == usize::MAX {
                dist[next] = dist[loc] + 1;
                queue.push_back(next);
            }
        }
    }
    let mut out = [1.0f32; LOCATION_COUNT];
    for loc in 0..LOCATION_COUNT {
        if dist[loc] != usize::MAX {
            out[loc] = (dist[loc] as f32 / DIST_SCALE).min(1.0);
        }
    }
    out
}

fn cells_vec(state: &GameState, pid: usize, net_masks: &[u32; SEAT_COUNT]) -> Vec<f32> {
    let n = state.player_count();
    let mut c = vec![0.0f32; BOARD_CELLS * F_CELL];
    for cell in 0..BOARD_CELLS {
        let (tile, is_farm) = if cell < CITY_CELLS {
            (state.city_tiles[cell].as_ref(), false)
        } else {
            (state.farm_tiles[cell - CITY_CELLS].as_ref(), true)
        };
        let base = cell * F_CELL;
        if let Some(t) = tile {
            c[base + CELL_OCCUPIED] = 1.0;
            c[base + CELL_OWNER + rel_seat(t.player, pid, n)] = 1.0;
            c[base + CELL_INDUSTRY + t.ind as usize] = 1.0;
            c[base + CELL_FLIPPED] = t.flipped as u8 as f32;
            c[base + CELL_CUBES] = t.resource_cubes as f32 / CUBES_SCALE;
            c[base + CELL_LEVEL] =
                (t.def.level as f32 / LEVEL_SCALE).clamp(0.0, 1.0); // max level 8
            c[base + CELL_VP] = t.def.vp as f32 / TILE_VP_SCALE;
            c[base + CELL_INCOME] = t.def.income as f32 / INCOME_SCALE;
        }
        if is_farm {
            c[base + CELL_IS_FARM] = 1.0;
            c[base + CELL_ALLOWED + IndustryType::Brewery as usize] = 1.0;
        } else if let Some((loc, slot_index)) = loc_from_key(cell) {
            // Static slot capability, so empty slots still say what fits here.
            if let Some(allowed) = city_slots(loc).get(slot_index) {
                for &ind in allowed.iter() {
                    c[base + CELL_ALLOWED + ind as usize] = 1.0;
                }
            }
            c[base + CELL_SLOT] = slot_index as f32 / SLOT_SCALE;
        }
    }
    // Network membership and distance are location-level facts; every cell of a
    // location carries them.
    let locations = board_cell_locations();
    let distances = distance_from_network(net_masks[pid]);
    for cell in 0..BOARD_CELLS {
        let loc = locations[cell];
        let base = cell * F_CELL;
        for p in 0..n {
            if net_masks[p] & (1u32 << loc) != 0 {
                c[base + CELL_IN_NET + rel_seat(p, pid, n)] = 1.0;
            }
        }
        c[base + CELL_DIST] = distances[loc];
    }
    c
}

fn links_vec(state: &GameState, pid: usize, net_masks: &[u32; SEAT_COUNT]) -> Vec<f32> {
    let n = state.player_count();
    let mut l = vec![0.0f32; LINK_CELLS * F_LINK];
    for conn in connections() {
        let base = conn.id * F_LINK;
        l[base + LINK_CANAL] = conn.canal as u8 as f32;
        l[base + LINK_RAIL] = conn.rail as u8 as f32;
        l[base + LINK_VIA_FARM] = conn.via_farm.is_some() as u8 as f32;
        let (a, b) = (conn.a as usize, conn.b as usize);
        for p in 0..n {
            let mask = net_masks[p];
            if mask & (1u32 << a) != 0 || mask & (1u32 << b) != 0 {
                l[base + LINK_IN_NET + rel_seat(p, pid, n)] = 1.0;
            }
        }
        let mine = net_masks[pid];
        l[base + LINK_TOUCHES_NET] =
            (mine & (1u32 << a) != 0 || mine & (1u32 << b) != 0) as u8 as f32;
    }
    for (id, link) in state.links.iter().enumerate().take(LINK_CELLS) {
        if let Some(link) = link {
            let base = id * F_LINK;
            l[base + LINK_BUILT] = 1.0;
            l[base + LINK_OWNER + rel_seat(link.player, pid, n)] = 1.0;
            l[base + LINK_IS_CANAL] = link.is_canal as u8 as f32;
        }
    }
    l
}

fn merchants_vec(state: &GameState) -> Vec<f32> {
    let mut m = vec![0.0f32; MERCHANT_COUNT * F_MERCHANT];
    for (i, merchant) in state.merchants.iter().enumerate().take(MERCHANT_COUNT) {
        let base = i * F_MERCHANT;
        let code = match merchant.buys {
            BuyType::Blank => 0usize,
            BuyType::Any => 1,
            BuyType::Industry(IndustryType::CottonMill) => 2,
            BuyType::Industry(IndustryType::Manufacturer) => 3,
            BuyType::Industry(IndustryType::Pottery) => 4,
            BuyType::Industry(_) => {
                debug_assert!(false, "merchant cannot buy raw materials");
                1
            }
        };
        m[base + MERCHANT_BUY + code] = 1.0;
        m[base + MERCHANT_BEER] = merchant.has_beer as u8 as f32;
    }
    m
}

fn seats_vec(state: &GameState, pid: usize) -> Vec<f32> {
    let n = state.player_count();
    let mut s = vec![0.0f32; SEAT_COUNT * F_SEAT];
    for rel in 0..n {
        let p = (pid + rel) % n;
        let player = &state.players[p];
        let base = rel * F_SEAT;
        s[base + SEAT_MONEY] = (player.money as f32 / MONEY_SCALE).clamp(0.0, 1.0);
        s[base + SEAT_INCOME_SPACE] =
            (player.income_space as f32 / INCOME_SPACE_SCALE).clamp(0.0, 1.0);
        s[base + SEAT_INCOME_LEVEL] =
            ((player.income_level() as f32 + 10.0) / 40.0).clamp(0.0, 1.0);
        s[base + SEAT_VP] = (player.vp as f32 / MONEY_SCALE).min(1.0);
        s[base + SEAT_CANAL_LINKS] = (player.canal_links as f32 / LINK_SCALE).min(1.0);
        s[base + SEAT_RAIL_LINKS] = (player.rail_links as f32 / LINK_SCALE).min(1.0);
        s[base + SEAT_HAND_SIZE] = (player.hand.len() as f32 / HAND_SCALE).min(1.0);
        s[base + SEAT_WILD_LOCATION] = player.has_wild_location as u8 as f32;
        s[base + SEAT_WILD_INDUSTRY] = player.has_wild_industry as u8 as f32;
        for ind in 0..INDUSTRY_COUNT {
            let total =
                crate::state::player_industry_stack(IndustryType::ALL[ind]).len().max(1);
            s[base + SEAT_REMAINING + ind] =
                (player.remaining_count(IndustryType::ALL[ind]) as f32 / total as f32).min(1.0);
        }
        s[base + SEAT_SPENT] =
            (state.money_spent_this_round[p] as f32 / SPENT_SCALE).clamp(0.0, 1.0);
        s[base + SEAT_IS_CURRENT] = (p == state.current_player_id()) as u8 as f32;

        // Own hand is real; opponent hands are whatever the caller put in the
        // state (a determinization draw), and are flagged as such.
        let (bag, offset) = if rel == 0 {
            (card_bag(&player.hand, HAND_SCALE), SEAT_HAND_REAL)
        } else {
            s[base + SEAT_HAND_SAMPLED_FLAG] = 1.0;
            (card_bag(&player.hand, HAND_SCALE), SEAT_HAND_SAMPLED)
        };
        s[base + offset..base + offset + CARD_SEMANTIC_COUNT].copy_from_slice(&bag);

        let public = card_bag(&player.played, PLAYED_SCALE);
        s[base + SEAT_HAND_PUBLIC..base + SEAT_HAND_PUBLIC + CARD_SEMANTIC_COUNT]
            .copy_from_slice(&public);
    }
    s
}

fn global_vec(state: &GameState, pid: usize) -> Vec<f32> {
    let n = state.player_count();
    let mut g = vec![0.0f32; F_GLOBAL];
    g[GLOBAL_ERA] = (state.era == Era::Rail) as u8 as f32;
    g[GLOBAL_ROUND] = (state.round as f32 / 8.0).min(1.0);
    g[GLOBAL_ROUNDS_REMAINING] = (state.rounds_remaining() as f32 / 10.0).min(1.0);
    g[GLOBAL_ACTIONS] = if state.actions_per_turn > 0 {
        state.actions_this_turn as f32 / state.actions_per_turn as f32
    } else {
        0.0
    };
    g[GLOBAL_COAL_MARKET + state.coal_market.min(14)] = 1.0;
    g[GLOBAL_IRON_MARKET + state.iron_market.min(10)] = 1.0;
    g[GLOBAL_DECK] = (state.deck.len() as f32 / DECK_SCALE).min(1.0);
    g[GLOBAL_DISCARD] = (state.discard_pile.len() as f32 / DECK_SCALE).min(1.0);
    g[GLOBAL_WILD_LOCATION_PILE] = state.wild_location_pile as f32;
    g[GLOBAL_WILD_INDUSTRY_PILE] = state.wild_industry_pile as f32;
    for q in 0..n {
        let seat = state.turn_order[(state.current_index + q) % n];
        g[GLOBAL_QUEUE + q * SEAT_COUNT + rel_seat(seat, pid, n)] = 1.0;
    }
    g
}

/// Encode a full game state from the acting player's perspective.
///
/// `perspective` defaults to the current player. Opponent hands are read from
/// whatever is in `state` (the caller decides determinized vs real); this
/// encoder only tags them as sampled.
pub fn state_tokens(state: &GameState, perspective: usize) -> EncodedState {
    let pid = perspective;
    let mut net_masks = [0u32; SEAT_COUNT];
    for p in 0..state.player_count().min(SEAT_COUNT) {
        net_masks[p] = state.network_mask(p);
    }
    EncodedState {
        cells: cells_vec(state, pid, &net_masks),
        links: links_vec(state, pid, &net_masks),
        merchants: merchants_vec(state),
        seats: seats_vec(state, pid),
        global: global_vec(state, pid),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand_chacha::ChaCha12Rng;

    #[test]
    fn token_groups_have_contract_shapes() {
        let state = GameState::new(ChaCha12Rng::seed_from_u64(7), 4);
        let t = state_tokens(&state, 0);
        assert_eq!(t.cells.len(), BOARD_CELLS * F_CELL);
        assert_eq!(t.links.len(), LINK_CELLS * F_LINK);
        assert_eq!(t.merchants.len(), MERCHANT_COUNT * F_MERCHANT);
        assert_eq!(t.seats.len(), SEAT_COUNT * F_SEAT);
        assert_eq!(t.global.len(), F_GLOBAL);
        assert_eq!(TOKEN_COUNT, 102);
    }

    #[test]
    fn perspective_rotation_puts_me_first() {
        let mut state = GameState::new(ChaCha12Rng::seed_from_u64(21), 4);
        state.current_index = 2;
        let pid = state.current_player_id();
        assert_eq!(pid, 2);
        let t = state_tokens(&state, pid);
        // Seat 0 of the encoding is the acting player.
        assert_eq!(t.seats[SEAT_IS_CURRENT], 1.0);
        let other = 3 * F_SEAT;
        assert_eq!(t.seats[other + SEAT_IS_CURRENT], 0.0);
        // Own hand is real, every opponent hand is flagged as sampled.
        assert_eq!(t.seats[SEAT_HAND_SAMPLED_FLAG], 0.0);
        assert_eq!(t.seats[other + SEAT_HAND_SAMPLED_FLAG], 1.0);
    }

    #[test]
    fn static_slot_capability_and_farm_cells_are_encoded() {
        let state = GameState::new(ChaCha12Rng::seed_from_u64(99), 4);
        let t = state_tokens(&state, 0);
        let coal_plane = CELL_ALLOWED + IndustryType::CoalMine as usize;
        assert!(t.cells.iter().skip(coal_plane).step_by(F_CELL).any(|&v| v == 1.0));
        for farm in 0..2 {
            let base = (CITY_CELLS + farm) * F_CELL;
            assert_eq!(t.cells[base + CELL_IS_FARM], 1.0);
            assert_eq!(t.cells[base + CELL_ALLOWED + IndustryType::Brewery as usize], 1.0);
        }
    }

    #[test]
    fn empty_slots_still_report_distance_to_my_network() {
        let state = GameState::new(ChaCha12Rng::seed_from_u64(5), 4);
        let t = state_tokens(&state, 0);
        // Nothing is built at game start, so my network is empty and every
        // location is at the cap.
        for cell in 0..BOARD_CELLS {
            assert_eq!(t.cells[cell * F_CELL + CELL_DIST], 1.0);
            for p in 0..SEAT_COUNT {
                assert_eq!(t.cells[cell * F_CELL + CELL_IN_NET + p], 0.0);
            }
        }
    }
}
