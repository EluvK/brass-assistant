use crate::data::{industry_tiles, CardType, Era, IndustryType, TileDef};
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
    Industry { industries: [IndustryType; 2], n: u8 },
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
            Card::Industry { industries, n } => {
                industries[..*n as usize].contains(&ind)
            }
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
    /// Connected-component cache over built links:
    /// `component_masks[loc]` = bitmask of every location in `loc`'s component.
    /// Lazily rebuilt whenever the set of built links changes (fingerprint).
    /// `RwLock` (not `RefCell`) so `GameState` stays `Send + Sync` for PyO3;
    /// a single game is single-threaded, so the lock is uncontended.
    pub(crate) component_cache: std::sync::RwLock<(u64, [u32; 27])>,

    pub coal_market: usize,
    pub iron_market: usize,
    pub merchants: Vec<MerchantTile>,

    pub deck: Vec<Card>,
    /// Face-down discard pile (non-wild cards only; wilds return to piles).
    /// Used to build the hidden-card pool for MCTS determinization.
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
            component_cache,
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
pub fn deck_composition(player_count: usize) -> Vec<Card> {
    let mut cards = Vec::new();
    for (loc, count) in location_cards(player_count) {
        for _ in 0..*count {
            cards.push(Card::Location(*loc));
        }
    }
    for (ind, count) in industry_cards(player_count) {
        for _ in 0..*count {
            cards.push(Card::Industry { industries: [*ind; 2], n: 1 });
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
            component_cache: std::sync::RwLock::new((u64::MAX, [0u32; 27])),
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

    pub     fn init_merchants(&mut self) {
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
        if loc.is_city() {
            if let Some(k) = self.city_slot_key(loc, slot_index) {
                self.city_tiles[k] = Some(tile);
                self.sync_free_source(k);
            }
        } else if let Some(idx) = farm_index(loc) {
            self.farm_tiles[idx] = Some(tile);
        }
    }

    /// Remove a city tile by its flat slot key (keeps the free-source cache in
    /// sync). Used by shortfall repayment.
    pub fn remove_city_tile_by_key(&mut self, key: usize) {
        if key < self.city_tiles.len() {
            self.city_tiles[key] = None;
            self.drop_free_source(key);
        }
    }

    /// Remove a farm tile by index.
    pub fn remove_farm_by_idx(&mut self, idx: usize) {
        if idx < self.farm_tiles.len() {
            self.farm_tiles[idx] = None;
        }
    }

    pub fn remove_tile(&mut self, loc: Loc, slot_index: usize) {
        if loc.is_city() {
            if let Some(k) = self.city_slot_key(loc, slot_index) {
                self.city_tiles[k] = None;
                self.drop_free_source(k);
            }
        } else if let Some(idx) = farm_index(loc) {
            self.farm_tiles[idx] = None;
        }
    }

    // --- free-resource & connectivity caches -------------------------------

    /// Rescan `city_tiles` and rebuild the free coal/iron caches from scratch.
    /// Called after bulk board changes (era end) or direct tile writes (tests).
    pub fn rebuild_free_sources(&mut self) {
        self.free_coal_mines.clear();
        self.free_iron_works.clear();
        for k in 0..self.city_tiles.len() {
            self.sync_free_source(k);
        }
    }

    /// Drop any cached free-source entry for a city slot key.
    fn drop_free_source(&mut self, key: usize) {
        self.free_coal_mines.retain(|e| e.key != key);
        self.free_iron_works.retain(|e| e.key != key);
    }

    /// Re-derive the cached free-source entry for one city slot key from the
    /// tile's current state (an unflipped coal/iron tile with cubes).
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
            _ => {}
        }
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

    /// Verify the free-source caches match a full rescan of `city_tiles`.
    /// Cheap (47 slots); used by tests and, in debug builds, by `apply_move`
    /// to catch any mutation site that bypassed the cache.
    pub fn assert_caches_consistent(&self) {
        let mut coal = Vec::new();
        let mut iron = Vec::new();
        for (k, tile) in self.city_tiles.iter().enumerate() {
            let Some(t) = tile else { continue };
            if t.flipped || t.resource_cubes == 0 {
                continue;
            }
            let Some((loc, _)) = loc_from_key(k) else { continue };
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
                _ => {}
            }
        }
        // Compare order-insensitively: incremental syncs can reorder the live
        // cache (e.g. overbuilding a lower-key mine re-pushes it last).
        coal.sort_by_key(|e| e.key);
        iron.sort_by_key(|e| e.key);
        let mut live_coal = self.free_coal_mines.clone();
        let mut live_iron = self.free_iron_works.clone();
        live_coal.sort_by_key(|e| e.key);
        live_iron.sort_by_key(|e| e.key);
        assert_eq!(live_coal, coal, "free coal cache drifted");
        assert_eq!(live_iron, iron, "free iron cache drifted");
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
                let market = if is_coal { self.coal_market } else { self.iron_market };
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
