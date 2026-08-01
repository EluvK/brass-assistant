use crate::data::{industry_tiles, CardType, Era, IndustryType, TileDef};
use crate::map::*;
use rand::seq::SliceRandom;
use rand::Rng;

// ---------------------------------------------------------------------------
// Cards
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
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
    pub has_wild_location: bool,
    pub has_wild_industry: bool,
}

// Expanded per-player industry stacks (identical for all players).
pub fn player_industry_stack(ind: IndustryType) -> Vec<TileDef> {
    let mut v = Vec::new();
    for def in industry_tiles(ind) {
        for _ in 0..def.count {
            v.push(*def);
        }
    }
    v
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

#[derive(Clone)]
pub struct GameState<R: Rng> {
    pub rng: R,
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

    pub coal_market: usize,
    pub iron_market: usize,
    pub merchants: Vec<MerchantTile>,

    pub deck: Vec<Card>,
    pub wild_location_pile: u8,
    pub wild_industry_pile: u8,
}

// City slot offsets: slot_offsets[loc as usize] = start index in city_tiles.
pub fn city_slot_offsets() -> [usize; CITY_COUNT] {
    let mut offsets = [0usize; CITY_COUNT];
    let mut acc = 0;
    for (i, loc) in ALL_LOCATIONS[..CITY_COUNT].iter().enumerate() {
        offsets[i] = acc;
        acc += city_slots(*loc).len();
    }
    offsets
}

pub fn total_city_slots() -> usize {
    ALL_LOCATIONS[..CITY_COUNT]
        .iter()
        .map(|l| city_slots(*l).len())
        .sum()
}

impl<R: Rng> GameState<R> {
    pub fn new(rng: R, num_players: usize) -> Self {
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
            coal_market: COAL_MARKET_INITIAL,
            iron_market: IRON_MARKET_INITIAL,
            merchants: Vec::new(),
            deck: Vec::new(),
            wild_location_pile: WILD_LOCATION_PILE,
            wild_industry_pile: WILD_INDUSTRY_PILE,
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
        for (loc, count) in location_cards(self.player_count()) {
            for _ in 0..*count {
                self.deck.push(Card::Location(*loc));
            }
        }
        for (ind, count) in industry_cards(self.player_count()) {
            for _ in 0..*count {
                self.deck
                    .push(Card::Industry { industries: [*ind; 2], n: 1 });
            }
        }
        for _ in 0..dual_cotton_manufacturer_cards(self.player_count()) {
            self.deck.push(Card::Industry {
                industries: [IndustryType::CottonMill, IndustryType::Manufacturer],
                n: 2,
            });
        }
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
            if !self.deck.is_empty() {
                self.deck.pop();
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

    pub fn coal_price(&self) -> u8 {
        if self.coal_market == 0 {
            return COAL_EMPTY_PRICE;
        }
        COAL_MARKET_PRICES[COAL_MARKET_PRICES.len() - self.coal_market]
    }

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
            }
        } else if let Some(idx) = farm_index(loc) {
            self.farm_tiles[idx] = Some(tile);
        }
    }

    pub fn remove_tile(&mut self, loc: Loc, slot_index: usize) {
        if loc.is_city() {
            if let Some(k) = self.city_slot_key(loc, slot_index) {
                self.city_tiles[k] = None;
            }
        } else if let Some(idx) = farm_index(loc) {
            self.farm_tiles[idx] = None;
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
        // Only when a flip happened do we advance income.
        let (player, income) = (player, income);
        if self.city_tiles[key].as_ref().map_or(false, |t| t.flipped) {
            self.advance_income_spaces(player, income);
        }
        true
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
