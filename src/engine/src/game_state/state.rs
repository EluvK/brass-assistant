use crate::data::{CardType, Era, IndustryType, TileDef, industry_tiles};
use crate::map::*;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use std::sync::OnceLock;

// ---------------------------------------------------------------------------
// Cards
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq)]
pub enum Card {
    Location(Loc),
    /// An industry card may permit up to 2 industry types (dual card).
    Industry {
        industries: [IndustryType; 2],
        n: u8,
    },
    WildLocation,
    WildIndustry,
}

impl Card {
    pub fn ctype(&self) -> CardType {
        match self {
            Card::Location(_) => CardType::Location,
            Card::Industry { .. } => CardType::Industry,
            Card::WildLocation => CardType::WildLocation,
            Card::WildIndustry => CardType::WildIndustry,
        }
    }

    pub fn location(&self) -> Option<Loc> {
        match self {
            Card::Location(loc) => Some(*loc),
            _ => None,
        }
    }

    pub fn is_industry(&self, ind: IndustryType) -> bool {
        match self {
            Card::Industry { industries, n } => industries[..*n as usize].contains(&ind),
            _ => false,
        }
    }

    pub fn name(&self) -> String {
        match self {
            Card::Location(loc) => loc.name().to_string(),
            Card::Industry { industries, n } => {
                if *n == 2 {
                    "Cotton Mill / Manufacturer".to_string()
                } else {
                    industries[0].name().to_string()
                }
            }
            Card::WildLocation => "Wild Location".to_string(),
            Card::WildIndustry => "Wild Industry".to_string(),
        }
    }
}

// ---------------------------------------------------------------------------
// Board tiles & links
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct BoardTile {
    pub player: usize,
    pub ind: IndustryType,
    pub def: TileDef,
    pub flipped: bool,
    pub resource_cubes: u8,
}

#[derive(Debug, Clone, Copy)]
pub struct Link {
    pub player: usize,
    pub is_canal: bool,
}

// ---------------------------------------------------------------------------
// Free-resource caches (maintained by `GameState`; single source of truth for
// "which unflipped coal/iron cubes are on the board"). Connectivity for coal
// is applied at query time via the component-mask cache.
// ---------------------------------------------------------------------------

/// A free (unflipped, cubes > 0) coal mine on the board.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct CoalMineEntry {
    pub key: usize,
    pub loc: Loc,
    pub owner: usize,
    pub cubes: u8,
}

/// A free (unflipped, cubes > 0) iron works on the board (location-independent).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct IronWorksEntry {
    pub key: usize,
    pub owner: usize,
    pub cubes: u8,
}

/// A free (unflipped, beer cubes > 0) brewery on the board — city slot or farm.
/// Kept in sync with `city_tiles` / `farm_tiles`; own breweries need no
/// connectivity filter at query time, opponent breweries are filtered by the
/// component-mask cache (`connected_mask`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct BeerCubeEntry {
    /// City slot key; `usize::MAX` for a farm.
    pub key: usize,
    pub farm_idx: Option<usize>,
    pub loc: Loc,
    pub owner: usize,
    pub cubes: u8,
}

// ---------------------------------------------------------------------------
// Merchant tiles
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BuyType {
    Blank,
    Any,
    Industry(IndustryType),
}

#[derive(Debug, Clone)]
pub struct MerchantTile {
    pub loc: Loc,
    pub buys: BuyType,
    pub has_beer: bool,
}

impl MerchantTile {
    pub fn accepts(&self, ind: IndustryType) -> bool {
        match self.buys {
            BuyType::Blank => false,
            BuyType::Any => true,
            BuyType::Industry(t) => t == ind,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PendingBonus {
    FreeDevelop { player_id: usize, count: u8 },
}

// ---------------------------------------------------------------------------
// Players
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct Player {
    pub money: i32,
    pub income_space: u8,
    pub vp: u16,
    pub hand: Vec<Card>,
    /// Index of next unused tile per industry (into the expanded stack).
    pub industry_next: [u8; 6],
    pub canal_links: u8,
    pub rail_links: u8,
    /// Develop actions taken this era (for the action-economy guardrail:
    /// humans develop ~4x in canal, ~0-1x in rail).
    pub develops_in_canal: u8,
    pub develops_in_rail: u8,
    pub has_wild_location: bool,
    pub has_wild_industry: bool,
    /// Cards this player has used (played/discarded) this era, in order.
    /// Mirrors the anonymous `discard_pile` (which also holds the seeded
    /// face-down burns): only NON-WILD cards are recorded — wilds return to
    /// the supply on use (`rules::discard_card` / `execute_scout`), so they
    /// enter neither this list nor the discard pile. Current wild holding is
    /// public info tracked separately by `has_wild_location` /
    /// `has_wild_industry`. Reserved for future NN features / belief modelling;
    /// cleared at the Canal-to-Rail transition in `engine::handle_turn_result`.
    pub played: Vec<Card>,
}

/// Expanded per-player industry stacks (identical for all players), built once.
fn player_industry_stacks() -> &'static [Vec<TileDef>] {
    static STACKS: OnceLock<Vec<Vec<TileDef>>> = OnceLock::new();
    STACKS.get_or_init(|| {
        let mut stacks = Vec::with_capacity(6);
        for ind in IndustryType::ALL {
            let mut v = Vec::new();
            for def in industry_tiles(ind) {
                for _ in 0..def.count {
                    v.push(*def);
                }
            }
            stacks.push(v);
        }
        stacks
    })
}

/// Expanded per-player industry stack (index into `Player::industry_next`).
pub fn player_industry_stack(ind: IndustryType) -> &'static [TileDef] {
    &player_industry_stacks()[ind as usize]
}

impl Player {
    pub fn new() -> Self {
        Player {
            money: INITIAL_MONEY,
            income_space: INITIAL_INCOME_SPACE,
            vp: 0,
            hand: Vec::with_capacity(HAND_SIZE),
            industry_next: [0; 6],
            canal_links: LINKS_PER_PLAYER,
            rail_links: LINKS_PER_PLAYER,
            develops_in_canal: 0,
            develops_in_rail: 0,
            has_wild_location: false,
            has_wild_industry: false,
            played: Vec::new(),
        }
    }

    /// Next buildable tile of an industry, if any.
    pub fn next_tile(&self, ind: IndustryType) -> Option<TileDef> {
        let stack = player_industry_stack(ind);
        let idx = self.industry_next[ind as usize] as usize;
        stack.get(idx).copied()
    }

    /// The tile `offset` positions after the current next one (0 = next).
    pub fn tile_after(&self, ind: IndustryType, offset: usize) -> Option<TileDef> {
        let stack = player_industry_stack(ind);
        let idx = self.industry_next[ind as usize] as usize + offset;
        stack.get(idx).copied()
    }

    pub fn income_level(&self) -> i8 {
        crate::income::income_level_from_space(self.income_space)
    }

    /// Returns how many unused tiles remain of a type.
    pub fn remaining_count(&self, ind: IndustryType) -> usize {
        player_industry_stack(ind).len() - self.industry_next[ind as usize] as usize
    }

    pub fn developable_types(&self) -> Vec<(IndustryType, TileDef)> {
        let mut out = Vec::new();
        for ind in IndustryType::ALL {
            if let Some(t) = self.next_tile(ind) {
                if t.can_develop {
                    out.push((ind, t));
                }
            }
        }
        out
    }

    /// Mark the next tile of `ind` as used (build or develop).
    pub fn consume_tile(&mut self, ind: IndustryType) -> Option<TileDef> {
        let stack = player_industry_stack(ind);
        let idx = self.industry_next[ind as usize] as usize;
        let t = stack.get(idx).copied()?;
        self.industry_next[ind as usize] += 1;
        Some(t)
    }
}

// ---------------------------------------------------------------------------
// GameState
// ---------------------------------------------------------------------------

pub struct GameState {
    /// Only used during setup (deal/shuffle); engine logic is deterministic.
    rng: StdRng,
    pub era: Era,
    pub round: u32,
    pub turn_order: Vec<usize>,
    pub current_index: usize,
    pub actions_this_turn: usize,
    pub actions_per_turn: usize,
    pub is_first_round: bool,
    pub game_over: bool,

    pub players: Vec<Player>,
    pub money_spent_this_round: Vec<i32>,

    /// City slots flattened: index = city_offset + slot_index.
    pub city_tiles: Vec<Option<BoardTile>>,
    pub farm_tiles: [Option<BoardTile>; 2],
    /// Links indexed by connection id.
    pub links: Vec<Option<Link>>,

    /// Unflipped coal mines with cubes, kept in sync with `city_tiles`
    /// (`place_tile` / `remove_tile` / `consume_*` / `auto_sell_to_market` /
    /// bulk rebuilds). Connectivity is applied at query time.
    pub(crate) free_coal_mines: Vec<CoalMineEntry>,
    /// Unflipped iron works with cubes (location-independent).
    pub(crate) free_iron_works: Vec<IronWorksEntry>,
    /// Unflipped breweries with beer cubes (city + farm). Connectivity for
    /// opponent breweries is applied at query time via `connected_mask`; own
    /// breweries are reachable anywhere.
    pub(crate) free_beer_cubes: Vec<BeerCubeEntry>,
    /// Connected-component cache over built links:
    /// `component_masks[loc]` = bitmask of every location in `loc`'s component.
    /// Lazily rebuilt whenever the set of built links changes (fingerprint).
    /// `RwLock` (not `RefCell`) so `GameState` stays `Send + Sync` for PyO3;
    /// a single game is single-threaded, so the lock is uncontended.
    pub(crate) component_cache: std::sync::RwLock<(u64, [u32; 27])>,

    /// Cached per-player "own network" location masks (bitmask over the 27
    /// locations): bit `l` set iff `graph::is_in_network(pid, loc)` holds (own
    /// tile at `loc`, or own link touching `loc` incl. via-farm). Maintained
    /// incrementally at every board-mutation site and lazily self-healed
    /// against any direct `links`/`city_tiles`/`farm_tiles` write via the
    /// fingerprint in the first tuple element (per-player: own links in bits
    /// 0..39, own tile locations in bits 39..66).
    pub(crate) network_cache: std::sync::RwLock<([u128; 4], [u32; 4])>,

    pub coal_market: usize,
    pub iron_market: usize,
    pub merchants: Vec<MerchantTile>,

    pub deck: Vec<Card>,
    /// Face-down discard pile (non-wild cards only; wilds return to piles).
    /// Every card here is OUT of circulation this era (seeded face-down burns
    /// plus non-wild discards). `mcts_ai::determinize` subtracts it from the
    /// hidden pool so a played card can never reappear in an opponent hand or
    /// the deck. Reset to empty at the era transition in
    /// `engine::handle_turn_result` (all canal cards are reshuffled into the rail
    /// deck). Per-player attribution of non-seeded plays lives in
    /// `Player::played`.
    pub discard_pile: Vec<Card>,
    pub wild_location_pile: u8,
    pub wild_industry_pile: u8,
    pub pending_bonus: Option<PendingBonus>,
}

impl Clone for GameState {
    fn clone(&self) -> Self {
        // `RwLock` is not Clone; copy the cached (fingerprint, masks) pair.
        let component_cache =
            std::sync::RwLock::new(*self.component_cache.read().expect("component cache"));
        let network_cache =
            std::sync::RwLock::new(*self.network_cache.read().expect("network cache"));
        GameState {
            rng: self.rng.clone(),
            era: self.era,
            round: self.round,
            turn_order: self.turn_order.clone(),
            current_index: self.current_index,
            actions_this_turn: self.actions_this_turn,
            actions_per_turn: self.actions_per_turn,
            is_first_round: self.is_first_round,
            game_over: self.game_over,
            players: self.players.clone(),
            money_spent_this_round: self.money_spent_this_round.clone(),
            city_tiles: self.city_tiles.clone(),
            farm_tiles: self.farm_tiles.clone(),
            links: self.links.clone(),
            free_coal_mines: self.free_coal_mines.clone(),
            free_iron_works: self.free_iron_works.clone(),
            free_beer_cubes: self.free_beer_cubes.clone(),
            component_cache,
            network_cache,
            coal_market: self.coal_market,
            iron_market: self.iron_market,
            merchants: self.merchants.clone(),
            deck: self.deck.clone(),
            discard_pile: self.discard_pile.clone(),
            wild_location_pile: self.wild_location_pile,
            wild_industry_pile: self.wild_industry_pile,
            pending_bonus: self.pending_bonus,
        }
    }
}

// City slot offsets: slot_offsets[loc as usize] = start index in city_tiles.
pub fn city_slot_offsets() -> [usize; CITY_COUNT] {
    static OFFSETS: OnceLock<[usize; CITY_COUNT]> = OnceLock::new();
    *OFFSETS.get_or_init(|| {
        let mut offsets = [0usize; CITY_COUNT];
        let mut acc = 0;
        for (i, loc) in ALL_LOCATIONS[..CITY_COUNT].iter().enumerate() {
            offsets[i] = acc;
            acc += city_slots(*loc).len();
        }
        offsets
    })
}

pub fn total_city_slots() -> usize {
    static TOTAL: OnceLock<usize> = OnceLock::new();
    *TOTAL.get_or_init(|| {
        ALL_LOCATIONS[..CITY_COUNT]
            .iter()
            .map(|l| city_slots(*l).len())
            .sum()
    })
}

/// Reverse map a flat city-slot key back to its (city, slot_index), O(1).
pub fn loc_from_key(key: usize) -> Option<(Loc, usize)> {
    loc_from_key_table().get(key).copied()
}

fn loc_from_key_table() -> &'static [(Loc, usize)] {
    static TABLE: OnceLock<Vec<(Loc, usize)>> = OnceLock::new();
    TABLE.get_or_init(|| {
        let mut table = Vec::with_capacity(total_city_slots());
        for loc in ALL_LOCATIONS[..CITY_COUNT].iter() {
            for slot in 0..city_slots(*loc).len() {
                table.push((*loc, slot));
            }
        }
        table
    })
}

// ---------------------------------------------------------------------------
// Card pool reconstruction (for MCTS determinization)
// ---------------------------------------------------------------------------

/// Full multiset of cards dealt this era for `player_count` (before shuffling).
/// Deterministic in `player_count` only (no era dependence); the composition is
/// cached once per player count and cloned on every call — `init_deck` and
/// each MCTS determinization (`mcts_ai::determinize`) share the cache instead
/// of rebuilding the ~64-card Vec from the static tables every time.
pub fn deck_composition(player_count: usize) -> Vec<Card> {
    static CACHE: OnceLock<[Vec<Card>; 5]> = OnceLock::new();
    let comps = CACHE.get_or_init(|| {
        let mut arr: [Vec<Card>; 5] = Default::default();
        for n in 2..=4 {
            arr[n] = build_deck_composition(n);
        }
        arr
    });
    comps[player_count].clone()
}

fn build_deck_composition(player_count: usize) -> Vec<Card> {
    let mut cards = Vec::new();
    for (loc, count) in location_cards(player_count) {
        for _ in 0..*count {
            cards.push(Card::Location(*loc));
        }
    }
    for (ind, count) in industry_cards(player_count) {
        for _ in 0..*count {
            cards.push(Card::Industry {
                industries: [*ind; 2],
                n: 1,
            });
        }
    }
    for _ in 0..dual_cotton_manufacturer_cards(player_count) {
        cards.push(Card::Industry {
            industries: [IndustryType::CottonMill, IndustryType::Manufacturer],
            n: 2,
        });
    }
    cards
}

impl GameState {
    pub fn new(rng: StdRng, num_players: usize) -> Self {
        assert!(num_players >= 2 && num_players <= 4, "2-4 players");
        let mut state = GameState {
            rng,
            era: Era::Canal,
            round: 1,
            turn_order: (0..num_players).collect(),
            current_index: 0,
            actions_this_turn: 0,
            actions_per_turn: FIRST_ROUND_ACTIONS,
            is_first_round: true,
            game_over: false,
            players: (0..num_players).map(|_| Player::new()).collect(),
            money_spent_this_round: vec![0; num_players],
            city_tiles: vec![None; total_city_slots()],
            farm_tiles: [None, None],
            links: vec![None; connections().len()],
            free_coal_mines: Vec::new(),
            free_iron_works: Vec::new(),
            free_beer_cubes: Vec::new(),
            component_cache: std::sync::RwLock::new((u64::MAX, [0u32; 27])),
            network_cache: std::sync::RwLock::new(([0u128; 4], [0u32; 4])),
            coal_market: COAL_MARKET_INITIAL,
            iron_market: IRON_MARKET_INITIAL,
            merchants: Vec::new(),
            deck: Vec::new(),
            discard_pile: Vec::new(),
            wild_location_pile: WILD_LOCATION_PILE,
            wild_industry_pile: WILD_INDUSTRY_PILE,
            pending_bonus: None,
        };

        state.turn_order.shuffle(&mut state.rng);
        state.init_merchants();
        state.init_deck();
        state.deal_cards();
        state.seed_discard_piles();
        state
    }

    // --- setup -------------------------------------------------------------

    pub fn init_merchants(&mut self) {
        // Take the active merchant-tile set for THIS player count, shuffle it,
        // and deal one to each active merchant slot.
        let mut mix: Vec<BuyType> = Vec::new();
        for e in merchant_tile_mix(self.player_count()) {
            mix.push(match e {
                MerchantMixEntry::Blank => BuyType::Blank,
                MerchantMixEntry::Any => BuyType::Any,
                MerchantMixEntry::Buys(t) => BuyType::Industry(*t),
            });
        }
        mix.shuffle(&mut self.rng);

        for def in merchant_defs() {
            if def.min_players > self.player_count() {
                continue;
            }
            for _ in 0..def.slots {
                let buys = mix.pop().unwrap_or(BuyType::Blank);
                self.merchants.push(MerchantTile {
                    loc: def.loc,
                    buys,
                    has_beer: buys != BuyType::Blank,
                });
            }
        }
    }

    pub fn init_deck(&mut self) {
        self.deck.clear();
        self.deck.extend(deck_composition(self.player_count()));
        self.deck.shuffle(&mut self.rng);
    }

    pub fn deal_cards(&mut self) {
        for p in self.players.iter_mut() {
            while p.hand.len() < HAND_SIZE && !self.deck.is_empty() {
                p.hand.push(self.deck.pop().unwrap());
            }
        }
    }

    /// Each player burns one card face-down from the deck as their discard.
    pub fn seed_discard_piles(&mut self) {
        for _ in 0..self.player_count() {
            if let Some(c) = self.deck.pop() {
                self.discard_pile.push(c);
            }
        }
    }

    pub fn draw_cards(&mut self, player_id: usize) {
        let p = &mut self.players[player_id];
        while p.hand.len() < HAND_SIZE && !self.deck.is_empty() {
            p.hand.push(self.deck.pop().unwrap());
        }
    }

    // --- accessors ---------------------------------------------------------

    pub fn player_count(&self) -> usize {
        self.players.len()
    }

    pub fn current_player_id(&self) -> usize {
        self.turn_order[self.current_index]
    }

    pub fn current_player(&self) -> &Player {
        &self.players[self.current_player_id()]
    }

    pub fn current_player_mut(&mut self) -> &mut Player {
        let id = self.current_player_id();
        &mut self.players[id]
    }

    /// Price of the cheapest currently-available market coal cube (the empty
    /// price when the market is dry). This is a scalar "market tightness"
    /// signal used by NN features (`encode.rs`) and heuristics — NOT a
    /// multi-cube cost estimator. Actual per-cube slot prices live in
    /// `graph::find_coal_sources`; total purchase costs come from
    /// `rules::calculate_build_cost` / `coal_source_options`.
    pub fn coal_price(&self) -> u8 {
        if self.coal_market == 0 {
            return COAL_EMPTY_PRICE;
        }
        COAL_MARKET_PRICES[COAL_MARKET_PRICES.len() - self.coal_market]
    }

    /// Price of the cheapest currently-available market iron cube. Same scalar
    /// role as `coal_price()`; see its docs. Per-slot prices live in
    /// `graph::find_iron_sources`.
    pub fn iron_price(&self) -> u8 {
        if self.iron_market == 0 {
            return IRON_EMPTY_PRICE;
        }
        IRON_MARKET_PRICES[IRON_MARKET_PRICES.len() - self.iron_market]
    }

    pub fn can_take_loan(&self, player_id: usize) -> bool {
        let level = self.players[player_id].income_level();
        crate::income::can_take_loan_at(level)
    }

    // --- board manipulation ------------------------------------------------

    /// Flat slot index for (city, slot_index); None if out of bounds.
    pub fn city_slot_key(&self, loc: Loc, slot_index: usize) -> Option<usize> {
        let off = city_slot_offsets()[loc as usize];
        let n = city_slots(loc).len();
        if slot_index < n {
            Some(off + slot_index)
        } else {
            None
        }
    }

    pub fn tile_at(&self, loc: Loc, slot_index: usize) -> Option<&BoardTile> {
        if !loc.is_city() {
            return None;
        }
        self.city_slot_key(loc, slot_index)
            .and_then(|k| self.city_tiles[k].as_ref())
    }

    pub fn farm_tile(&self, loc: Loc) -> Option<&BoardTile> {
        match loc {
            Loc::BreweryNorth => self.farm_tiles[0].as_ref(),
            Loc::BrewerySouth => self.farm_tiles[1].as_ref(),
            _ => None,
        }
    }

    /// Place a tile at a location (city slot or farm), overwriting any existing.
    pub fn place_tile(&mut self, loc: Loc, slot_index: usize, tile: BoardTile) {
        let player = tile.player;
        if loc.is_city() {
            if let Some(k) = self.city_slot_key(loc, slot_index) {
                self.city_tiles[k] = Some(tile);
                self.sync_free_source(k);
                self.mark_network(player, loc);
            }
        } else if let Some(idx) = farm_index(loc) {
            self.farm_tiles[idx] = Some(tile);
            self.sync_farm_beer(idx);
            self.mark_network(player, loc);
        }
    }

    /// Remove a city tile by its flat slot key (keeps the free-source cache in
    /// sync). Used by shortfall repayment.
    pub fn remove_city_tile_by_key(&mut self, key: usize) {
        if key < self.city_tiles.len() {
            let owner = self.city_tiles[key].as_ref().map(|t| t.player);
            self.city_tiles[key] = None;
            self.drop_free_source(key);
            if let Some(owner) = owner {
                if let Some((loc, _)) = loc_from_key(key) {
                    self.clear_network_location(owner, loc);
                }
            }
        }
    }

    /// Remove a farm tile by index.
    pub fn remove_farm_by_idx(&mut self, idx: usize) {
        if idx < self.farm_tiles.len() {
            let owner = self.farm_tiles[idx].as_ref().map(|t| t.player);
            self.farm_tiles[idx] = None;
            self.drop_farm_beer(idx);
            if let Some(owner) = owner {
                let loc = if idx == 0 {
                    Loc::BreweryNorth
                } else {
                    Loc::BrewerySouth
                };
                self.clear_network_location(owner, loc);
            }
        }
    }

    pub fn remove_tile(&mut self, loc: Loc, slot_index: usize) {
        if loc.is_city() {
            if let Some(k) = self.city_slot_key(loc, slot_index) {
                let owner = self.city_tiles[k].as_ref().map(|t| t.player);
                self.city_tiles[k] = None;
                self.drop_free_source(k);
                if let Some(owner) = owner {
                    self.clear_network_location(owner, loc);
                }
            }
        } else if let Some(idx) = farm_index(loc) {
            let owner = self.farm_tiles[idx].as_ref().map(|t| t.player);
            self.farm_tiles[idx] = None;
            self.drop_farm_beer(idx);
            if let Some(owner) = owner {
                self.clear_network_location(owner, loc);
            }
        }
    }

    // --- free-resource & connectivity caches -------------------------------

    /// Rescan `city_tiles` and rebuild the free coal/iron/beer caches from
    /// scratch. Called after bulk board changes (era end) or direct tile writes
    /// (tests).
    pub fn rebuild_free_sources(&mut self) {
        self.free_coal_mines.clear();
        self.free_iron_works.clear();
        self.free_beer_cubes.clear();
        for k in 0..self.city_tiles.len() {
            self.sync_free_source(k);
        }
        for idx in 0..self.farm_tiles.len() {
            self.sync_farm_beer(idx);
        }
    }

    /// Drop any cached free-source entry for a city slot key.
    fn drop_free_source(&mut self, key: usize) {
        self.free_coal_mines.retain(|e| e.key != key);
        self.free_iron_works.retain(|e| e.key != key);
        self.free_beer_cubes.retain(|e| e.key != key);
    }

    /// Drop any cached beer entry for a farm index.
    fn drop_farm_beer(&mut self, idx: usize) {
        self.free_beer_cubes.retain(|e| e.farm_idx != Some(idx));
    }

    /// Re-derive the cached free-source entry for one city slot key from the
    /// tile's current state (an unflipped coal/iron/brewery tile with cubes).
    fn sync_free_source(&mut self, key: usize) {
        self.drop_free_source(key);
        let Some(tile) = &self.city_tiles[key] else {
            return;
        };
        if tile.flipped || tile.resource_cubes == 0 {
            return;
        }
        let Some((loc, _)) = loc_from_key(key) else {
            return;
        };
        match tile.ind {
            IndustryType::CoalMine => self.free_coal_mines.push(CoalMineEntry {
                key,
                loc,
                owner: tile.player,
                cubes: tile.resource_cubes,
            }),
            IndustryType::IronWorks => self.free_iron_works.push(IronWorksEntry {
                key,
                owner: tile.player,
                cubes: tile.resource_cubes,
            }),
            IndustryType::Brewery => self.free_beer_cubes.push(BeerCubeEntry {
                key,
                farm_idx: None,
                loc,
                owner: tile.player,
                cubes: tile.resource_cubes,
            }),
            _ => {}
        }
    }

    /// Re-derive the cached beer entry for a farm index from the tile's current
    /// state (farms are always breweries).
    fn sync_farm_beer(&mut self, idx: usize) {
        self.drop_farm_beer(idx);
        let Some(tile) = &self.farm_tiles[idx] else {
            return;
        };
        if tile.flipped || tile.resource_cubes == 0 {
            return;
        }
        self.free_beer_cubes.push(BeerCubeEntry {
            key: usize::MAX,
            farm_idx: Some(idx),
            loc: if idx == 0 {
                Loc::BreweryNorth
            } else {
                Loc::BrewerySouth
            },
            owner: tile.player,
            cubes: tile.resource_cubes,
        });
    }

    /// Bitmask of every location connected to `loc` via any built link
    /// (includes `loc` and brewery farms passed through). Cached; rebuilt only
    /// when the set of built links changes. Self-healing against any direct
    /// `links` write (tests / double-rail dry-runs).
    pub(crate) fn connected_mask(&self, loc: Loc) -> u32 {
        let fp = self.link_fingerprint();
        let mut cache = self.component_cache.write().unwrap();
        if cache.0 != fp {
            *cache = (fp, self.compute_component_masks());
        }
        cache.1[loc as usize]
    }

    /// Presence fingerprint of the built-link set (one bit per connection).
    fn link_fingerprint(&self) -> u64 {
        let mut fp = 0u64;
        for (i, link) in self.links.iter().enumerate() {
            if link.is_some() {
                fp |= 1u64 << i;
            }
        }
        fp
    }

    /// Compute `component_masks` (one bitmask per location) via a full
    /// component sweep. Runs only when the link fingerprint changes.
    fn compute_component_masks(&self) -> [u32; 27] {
        let adj = crate::map::adjacency();
        let mut masks = [0u32; 27];
        for start in 0..27usize {
            let mut mask = 1u32 << (start as u8);
            let mut queue = [0usize; 27];
            let (mut head, mut tail) = (0usize, 1usize);
            queue[0] = start;
            while head < tail {
                let cur = queue[head];
                head += 1;
                for &(nb, conn_id) in &adj[cur] {
                    if self.links[conn_id].is_none() {
                        continue;
                    }
                    let bit = 1u32 << (nb as u8);
                    if mask & bit != 0 {
                        continue;
                    }
                    mask |= bit;
                    queue[tail] = nb as usize;
                    tail += 1;
                }
            }
            masks[start] = mask;
        }
        masks
    }

    // --- per-player network mask cache --------------------------------------

    /// Bitmask of locations in player `pid`'s own network: own tile at the
    /// location, or an own link touching it (incl. via-farm). Backed by the
    /// cached `network_cache` masks; `ensure_network_masks` keeps them fresh.
    pub(crate) fn network_mask(&self, pid: usize) -> u32 {
        self.network_cache.read().expect("network cache").1[pid]
    }

    /// Lazily validate/rebuild the cached network masks against the current
    /// board. Called once per legal-move / heuristic batch (NOT per location
    /// query): the fingerprint scan is O(links+tiles), and a recompute only
    /// happens when a direct `links`/`city_tiles`/`farm_tiles` write (tests,
    /// dry-runs) bypassed the incremental maintenance.
    pub(crate) fn ensure_network_masks(&self) {
        let fp = self.network_fingerprint();
        let mut cache = self.network_cache.write().expect("network cache");
        if cache.0 != fp {
            for pid in 0..cache.1.len() {
                cache.1[pid] = self.compute_network_mask(pid);
            }
            cache.0 = fp;
        }
    }

    /// Recompute all network masks from scratch and store the fresh fingerprint.
    /// Used after bulk board clears (era end) that bypass per-tile hooks.
    pub fn rebuild_network_masks(&mut self) {
        let fp = self.network_fingerprint();
        let mut cache = self.network_cache.write().expect("network cache");
        for pid in 0..cache.1.len() {
            cache.1[pid] = self.compute_network_mask(pid);
        }
        cache.0 = fp;
    }

    /// Per-player fingerprint of the board data the network masks derive from:
    /// own links in bits 0..39, own tile locations in bits 39..66. Any change
    /// to an own link or to the tile occupying any location flips it.
    fn network_fingerprint(&self) -> [u128; 4] {
        let mut fp = [0u128; 4];
        for (i, link) in self.links.iter().enumerate() {
            if let Some(l) = link {
                fp[l.player] |= 1u128 << i;
            }
        }
        for (k, tile) in self.city_tiles.iter().enumerate() {
            if let Some(t) = tile {
                if let Some((loc, _)) = loc_from_key(k) {
                    fp[t.player] |= 1u128 << (39 + loc as u8);
                }
            }
        }
        for (idx, tile) in self.farm_tiles.iter().enumerate() {
            if let Some(t) = tile {
                let loc = if idx == 0 {
                    Loc::BreweryNorth
                } else {
                    Loc::BrewerySouth
                };
                fp[t.player] |= 1u128 << (39 + loc as u8);
            }
        }
        fp
    }

    /// Recompute one player's network mask from scratch, mirroring
    /// `graph::is_in_network` exactly: every location with an own tile, plus
    /// every location touched by an own link (endpoint or via-farm).
    fn compute_network_mask(&self, pid: usize) -> u32 {
        let mut mask = 0u32;
        for (k, tile) in self.city_tiles.iter().enumerate() {
            if let Some(t) = tile {
                if t.player == pid {
                    if let Some((loc, _)) = loc_from_key(k) {
                        mask |= 1u32 << (loc as u8);
                    }
                }
            }
        }
        for (idx, tile) in self.farm_tiles.iter().enumerate() {
            if let Some(t) = tile {
                if t.player == pid {
                    let loc = if idx == 0 {
                        Loc::BreweryNorth
                    } else {
                        Loc::BrewerySouth
                    };
                    mask |= 1u32 << (loc as u8);
                }
            }
        }
        for (i, link) in self.links.iter().enumerate() {
            if let Some(l) = link {
                if l.player == pid {
                    let c = &connections()[i];
                    mask |= 1u32 << (c.a as u8);
                    mask |= 1u32 << (c.b as u8);
                    if let Some(f) = c.via_farm {
                        mask |= 1u32 << (f as u8);
                    }
                }
            }
        }
        mask
    }

    /// Set the "own network" bit for a location of a player (place tile / link).
    fn mark_network(&mut self, pid: usize, loc: Loc) {
        self.network_cache.write().expect("network cache").1[pid] |= 1u32 << (loc as u8);
    }

    /// Recompute one location's bit for a player after a tile/link removal:
    /// keep it only while an own tile or an own touching link remains there.
    fn clear_network_location(&mut self, pid: usize, loc: Loc) {
        let mut keep = false;
        if loc.is_city() {
            for slot in 0..crate::map::city_slots(loc).len() {
                if let Some(k) = self.city_slot_key(loc, slot) {
                    if let Some(t) = &self.city_tiles[k] {
                        if t.player == pid {
                            keep = true;
                            break;
                        }
                    }
                }
            }
        }
        if !keep {
            if let Some(t) = self.farm_tile(loc) {
                keep = t.player == pid;
            }
        }
        if !keep {
            for &conn_id in &crate::map::loc_connections()[loc as usize] {
                if let Some(l) = &self.links[conn_id] {
                    if l.player == pid {
                        keep = true;
                        break;
                    }
                }
            }
        }
        if !keep {
            self.network_cache.write().expect("network cache").1[pid] &= !(1u32 << (loc as u8));
        }
    }

    /// Place a link owned by `pid`, keeping the network mask in sync (endpoints
    /// + via-farm all enter the player's network). Centralizes every link
    /// write so RailTx dry-runs and era-end cleanup stay consistent.
    pub fn set_link(&mut self, conn_id: usize, pid: usize) {
        let c = &connections()[conn_id];
        self.links[conn_id] = Some(Link {
            player: pid,
            is_canal: self.era == Era::Canal,
        });
        self.mark_network(pid, c.a);
        self.mark_network(pid, c.b);
        if let Some(f) = c.via_farm {
            self.mark_network(pid, f);
        }
    }

    /// Remove a link, clearing the affected network-mask bits for its former
    /// owner (unless another own link/tile still covers them).
    pub fn remove_link(&mut self, conn_id: usize) {
        let owner = self.links[conn_id].map(|l| l.player);
        self.links[conn_id] = None;
        if let Some(owner) = owner {
            let c = &connections()[conn_id];
            self.clear_network_location(owner, c.a);
            self.clear_network_location(owner, c.b);
            if let Some(f) = c.via_farm {
                self.clear_network_location(owner, f);
            }
        }
    }

    /// Verify the free-source caches match a full rescan of `city_tiles` and
    /// `farm_tiles`. Cheap (47 slots + 2 farms); used by tests and, in debug
    /// builds, by `apply_move` to catch any mutation site that bypassed the
    /// cache.
    pub fn assert_caches_consistent(&self) {
        let mut coal = Vec::new();
        let mut iron = Vec::new();
        let mut beer = Vec::new();
        for (k, tile) in self.city_tiles.iter().enumerate() {
            let Some(t) = tile else { continue };
            if t.flipped || t.resource_cubes == 0 {
                continue;
            }
            let Some((loc, _)) = loc_from_key(k) else {
                continue;
            };
            match t.ind {
                IndustryType::CoalMine => coal.push(CoalMineEntry {
                    key: k,
                    loc,
                    owner: t.player,
                    cubes: t.resource_cubes,
                }),
                IndustryType::IronWorks => iron.push(IronWorksEntry {
                    key: k,
                    owner: t.player,
                    cubes: t.resource_cubes,
                }),
                IndustryType::Brewery => beer.push(BeerCubeEntry {
                    key: k,
                    farm_idx: None,
                    loc,
                    owner: t.player,
                    cubes: t.resource_cubes,
                }),
                _ => {}
            }
        }
        for (idx, tile) in self.farm_tiles.iter().enumerate() {
            let Some(t) = tile else { continue };
            if t.flipped || t.resource_cubes == 0 {
                continue;
            }
            beer.push(BeerCubeEntry {
                key: usize::MAX,
                farm_idx: Some(idx),
                loc: if idx == 0 {
                    Loc::BreweryNorth
                } else {
                    Loc::BrewerySouth
                },
                owner: t.player,
                cubes: t.resource_cubes,
            });
        }
        // Compare order-insensitively: incremental syncs can reorder the live
        // cache (e.g. overbuilding a lower-key mine re-pushes it last).
        coal.sort_by_key(|e| e.key);
        iron.sort_by_key(|e| e.key);
        beer.sort_by_key(|e| (e.farm_idx.unwrap_or(usize::MAX), e.key));
        let mut live_coal = self.free_coal_mines.clone();
        let mut live_iron = self.free_iron_works.clone();
        let mut live_beer = self.free_beer_cubes.clone();
        live_coal.sort_by_key(|e| e.key);
        live_iron.sort_by_key(|e| e.key);
        live_beer.sort_by_key(|e| (e.farm_idx.unwrap_or(usize::MAX), e.key));
        assert_eq!(live_coal, coal, "free coal cache drifted");
        assert_eq!(live_iron, iron, "free iron cache drifted");
        assert_eq!(live_beer, beer, "free beer cache drifted");

        // Network masks: each cached mask must match a from-scratch recompute
        // of the current board. (The fingerprint is only a laziness marker for
        // `ensure_network_masks`; incremental maintenance keeps masks correct
        // without touching it, so it is not compared here.)
        let cache = self.network_cache.read().expect("network cache");
        for pid in 0..cache.1.len() {
            assert_eq!(
                cache.1[pid],
                self.compute_network_mask(pid),
                "network mask drifted for P{pid}"
            );
        }
    }

    // --- money & income ----------------------------------------------------

    pub fn spend_money(&mut self, player_id: usize, amount: i32) {
        self.players[player_id].money -= amount;
        self.money_spent_this_round[player_id] += amount;
    }

    pub fn gain_money(&mut self, player_id: usize, amount: i32) {
        self.players[player_id].money += amount;
    }

    pub fn advance_income_spaces(&mut self, player_id: usize, spaces: u8) {
        let p = &mut self.players[player_id];
        p.income_space = (p.income_space as u16 + spaces as u16).min(MAX_INCOME_SPACE as u16) as u8;
    }

    pub fn apply_loan_income_drop(&mut self, player_id: usize) {
        let level = self.players[player_id].income_level();
        let new_level = (level - LOAN_INCOME_PENALTY).max(MIN_INCOME);
        self.players[player_id].income_space =
            crate::income::income_highest_space_of_level(new_level);
    }

    /// Consume a beer barrel from a BeerSource (own/opponent brewery or merchant).
    pub fn consume_beer_source(&mut self, src: &crate::graph::BeerSource) {
        use crate::graph::BeerSourceKind;
        match src.kind {
            BeerSourceKind::Own | BeerSourceKind::Opponent => {
                if let Some(f) = src.farm_idx {
                    self.consume_from_farm(f);
                } else if src.key != usize::MAX {
                    self.consume_from_city(src.key);
                }
            }
            BeerSourceKind::Merchant => {
                if let Some(mi) = src.merchant_idx {
                    if let Some(mt) = self.merchants.get_mut(mi) {
                        mt.has_beer = false;
                    }
                }
            }
        }
    }

    /// Flip a tile, granting its income spaces to its owner.
    pub fn flip_tile(&mut self, tile: &mut BoardTile) {
        if tile.flipped {
            return;
        }
        let (player, income) = (tile.player, tile.def.income);
        tile.flipped = true;
        self.advance_income_spaces(player, income);
    }

    /// Flip by flat city key (used when iterating).
    pub fn flip_by_key(&mut self, key: usize) {
        let (player, income, already) = {
            let Some(tile) = self.city_tiles[key].as_mut() else {
                return;
            };
            if tile.flipped {
                return;
            }
            tile.flipped = true;
            (tile.player, tile.def.income, true)
        };
        let _ = already;
        self.advance_income_spaces(player, income);
        self.sync_free_source(key);
    }

    pub fn flip_farm(&mut self, idx: usize) {
        let (player, income) = {
            let Some(tile) = self.farm_tiles[idx].as_mut() else {
                return;
            };
            if tile.flipped {
                return;
            }
            tile.flipped = true;
            (tile.player, tile.def.income)
        };
        self.advance_income_spaces(player, income);
        self.sync_farm_beer(idx);
    }

    // --- resources ---------------------------------------------------------

    pub fn take_market_coal(&mut self) {
        if self.coal_market > 0 {
            self.coal_market -= 1;
        }
    }

    pub fn take_market_iron(&mut self) {
        if self.iron_market > 0 {
            self.iron_market -= 1;
        }
    }

    /// Consume one explicitly selected coal source. Pricing belongs to the
    /// action; this method only applies the physical resource-state change.
    pub fn consume_coal_source(&mut self, src: &crate::graph::CoalSource) {
        match src.kind {
            crate::graph::CoalSourceKind::Mine => {
                self.consume_from_city(src.key);
            }
            crate::graph::CoalSourceKind::Market => self.take_market_coal(),
        }
    }

    /// Consume one explicitly selected iron source. Pricing belongs to the
    /// action; this method only applies the physical resource-state change.
    pub fn consume_iron_source(&mut self, src: &crate::graph::IronSource) {
        if src.free {
            self.consume_from_city(src.key);
        } else {
            self.take_market_iron();
        }
    }

    /// Consume one cube from a city tile or farm (key prefixes handled by caller).
    pub fn consume_from_city(&mut self, key: usize) -> bool {
        let (player, income) = {
            let tile = match self.city_tiles[key].as_mut() {
                Some(t) => t,
                None => return false,
            };
            if tile.resource_cubes == 0 {
                return false;
            }
            tile.resource_cubes -= 1;
            if tile.resource_cubes == 0 {
                tile.flipped = true;
                (tile.player, tile.def.income)
            } else {
                (0, 0)
            }
        };
        self.sync_free_source(key);
        // Only when a flip happened do we advance income.
        let (player, income) = (player, income);
        if self.city_tiles[key].as_ref().map_or(false, |t| t.flipped) {
            self.advance_income_spaces(player, income);
        }
        true
    }

    /// Restore a city tile after a dry-run consumption rollback (double rail),
    /// re-syncing the free-source cache.
    pub fn restore_consumed_city_tile(
        &mut self,
        key: usize,
        prev_cubes: u8,
        prev_flipped: bool,
        owner: usize,
        prev_income_space: u8,
    ) {
        if let Some(tile) = self.city_tiles[key].as_mut() {
            tile.resource_cubes = prev_cubes;
            tile.flipped = prev_flipped;
        }
        self.players[owner].income_space = prev_income_space;
        self.sync_free_source(key);
    }

    pub fn consume_from_farm(&mut self, idx: usize) -> bool {
        let (player, income) = {
            let tile = match self.farm_tiles[idx].as_mut() {
                Some(t) => t,
                None => return false,
            };
            if tile.resource_cubes == 0 {
                return false;
            }
            tile.resource_cubes -= 1;
            if tile.resource_cubes == 0 {
                tile.flipped = true;
                (tile.player, tile.def.income)
            } else {
                (0, 0)
            }
        };
        if self.farm_tiles[idx].as_ref().map_or(false, |t| t.flipped) {
            self.advance_income_spaces(player, income);
        }
        self.sync_farm_beer(idx);
        true
    }

    /// Move a newly built coal/iron tile's cubes into its market at the most
    /// expensive empty spaces; builder collects the price of each space filled.
    /// Returns money gained (caller credits the player). Flips tile if emptied.
    pub fn auto_sell_to_market(&mut self, key: usize) -> i32 {
        let mut money = 0i32;
        let mut should_flip = false;
        let (player, income) = {
            let Some(tile) = self.city_tiles[key].as_mut() else {
                return 0;
            };
            let is_coal = tile.ind == IndustryType::CoalMine;
            let prices: &[u8] = if is_coal {
                &crate::map::COAL_MARKET_PRICES
            } else {
                &crate::map::IRON_MARKET_PRICES
            };
            while tile.resource_cubes > 0 {
                let market = if is_coal {
                    self.coal_market
                } else {
                    self.iron_market
                };
                if market >= prices.len() {
                    break;
                }
                let idx = prices.len() - market - 1;
                money += prices[idx] as i32;
                if is_coal {
                    self.coal_market += 1;
                } else {
                    self.iron_market += 1;
                }
                tile.resource_cubes -= 1;
            }
            if tile.resource_cubes == 0 {
                tile.flipped = true;
                should_flip = true;
                (tile.player, tile.def.income)
            } else {
                (0, 0)
            }
        };
        self.sync_free_source(key);
        if should_flip {
            self.advance_income_spaces(player, income);
        }
        money
    }
}

fn farm_index(loc: Loc) -> Option<usize> {
    match loc {
        Loc::BreweryNorth => Some(0),
        Loc::BrewerySouth => Some(1),
        _ => None,
    }
}
