use crate::data::IndustryType;
use serde::{Deserialize, Serialize};
use std::fmt;
use std::sync::OnceLock;

// ---------------------------------------------------------------------------
// Locations: cities (20) + merchants (5) + brewery farms (2)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Loc {
    // Cities
    Belper,
    Derby,
    Leek,
    StokeOnTrent,
    Stone,
    Uttoxeter,
    Stafford,
    BurtonOnTrent,
    Cannock,
    Tamworth,
    Walsall,
    Wolverhampton,
    Coalbrookdale,
    Dudley,
    Kidderminster,
    Worcester,
    Birmingham,
    Coventry,
    Nuneaton,
    Redditch,
    // Merchants
    Shrewsbury,
    Gloucester,
    Oxford,
    Warrington,
    Nottingham,
    // Brewery farms
    BreweryNorth,
    BrewerySouth,
}

pub const CITY_COUNT: usize = 20;
pub const ALL_LOCATIONS: [Loc; 27] = [
    Loc::Belper,
    Loc::Derby,
    Loc::Leek,
    Loc::StokeOnTrent,
    Loc::Stone,
    Loc::Uttoxeter,
    Loc::Stafford,
    Loc::BurtonOnTrent,
    Loc::Cannock,
    Loc::Tamworth,
    Loc::Walsall,
    Loc::Wolverhampton,
    Loc::Coalbrookdale,
    Loc::Dudley,
    Loc::Kidderminster,
    Loc::Worcester,
    Loc::Birmingham,
    Loc::Coventry,
    Loc::Nuneaton,
    Loc::Redditch,
    Loc::Shrewsbury,
    Loc::Gloucester,
    Loc::Oxford,
    Loc::Warrington,
    Loc::Nottingham,
    Loc::BreweryNorth,
    Loc::BrewerySouth,
];

impl Loc {
    pub fn name(&self) -> &'static str {
        use Loc::*;
        match self {
            Belper => "Belper",
            Derby => "Derby",
            Leek => "Leek",
            StokeOnTrent => "Stoke-on-Trent",
            Stone => "Stone",
            Uttoxeter => "Uttoxeter",
            Stafford => "Stafford",
            BurtonOnTrent => "Burton-on-Trent",
            Cannock => "Cannock",
            Tamworth => "Tamworth",
            Walsall => "Walsall",
            Wolverhampton => "Wolverhampton",
            Coalbrookdale => "Coalbrookdale",
            Dudley => "Dudley",
            Kidderminster => "Kidderminster",
            Worcester => "Worcester",
            Birmingham => "Birmingham",
            Coventry => "Coventry",
            Nuneaton => "Nuneaton",
            Redditch => "Redditch",
            Shrewsbury => "Shrewsbury",
            Gloucester => "Gloucester",
            Oxford => "Oxford",
            Warrington => "Warrington",
            Nottingham => "Nottingham",
            BreweryNorth => "Brewery (N)",
            BrewerySouth => "Brewery (S)",
        }
    }

    pub fn zh_name(&self) -> &'static str {
        use Loc::*;
        match self {
            Belper => "贝尔珀",
            Derby => "德比",
            Leek => "利克",
            StokeOnTrent => "斯托克",
            Stone => "斯通",
            Uttoxeter => "阿托克西特",
            Stafford => "斯塔福德",
            BurtonOnTrent => "伯顿",
            Cannock => "坎诺克",
            Tamworth => "塔姆沃思",
            Walsall => "沃尔索尔",
            Wolverhampton => "伍尔弗汉普顿",
            Coalbrookdale => "科尔布鲁克代尔",
            Dudley => "达德利",
            Kidderminster => "基德明斯特",
            Worcester => "伍斯特",
            Birmingham => "伯明翰",
            Coventry => "考文垂",
            Nuneaton => "纳尼顿",
            Redditch => "雷迪奇",
            Shrewsbury => "什鲁斯伯里",
            Gloucester => "格洛斯特",
            Oxford => "牛津",
            Warrington => "沃灵顿",
            Nottingham => "诺丁汉",
            BreweryNorth => "北部啤酒农场",
            BrewerySouth => "南部啤酒农场",
        }
    }

    pub fn is_city(&self) -> bool {
        (*self as usize) < CITY_COUNT
    }

    pub fn is_merchant(&self) -> bool {
        let v = *self as usize;
        v >= CITY_COUNT && v < CITY_COUNT + 5
    }

    pub fn is_farm(&self) -> bool {
        let v = *self as usize;
        v >= CITY_COUNT + 5
    }
}

impl fmt::Display for Loc {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.name())
    }
}

// ---------------------------------------------------------------------------
// City slots: each slot permits a set of industry types.
// ---------------------------------------------------------------------------

pub const MAX_SLOTS_PER_CITY: usize = 4;

pub fn city_slots(loc: Loc) -> &'static [&'static [IndustryType]] {
    use IndustryType::*;
    use Loc::*;
    match loc {
        Belper => &[&[CottonMill, Manufacturer], &[CoalMine], &[Pottery]],
        Derby => &[
            &[CottonMill, Brewery],
            &[CottonMill, Manufacturer],
            &[IronWorks],
        ],
        Leek => &[&[CottonMill, Manufacturer], &[CottonMill, CoalMine]],
        StokeOnTrent => &[
            &[CottonMill, Manufacturer],
            &[Pottery, IronWorks],
            &[Manufacturer],
        ],
        Stone => &[&[CottonMill, Brewery], &[Manufacturer, CoalMine]],
        Uttoxeter => &[&[Manufacturer, Brewery], &[CottonMill, Brewery]],
        Stafford => &[&[Manufacturer, Brewery], &[Pottery]],
        BurtonOnTrent => &[&[Manufacturer, CoalMine], &[Brewery]],
        Cannock => &[&[Manufacturer, CoalMine], &[CoalMine]],
        Tamworth => &[&[CottonMill, CoalMine], &[CottonMill, CoalMine]],
        Walsall => &[&[IronWorks, Manufacturer], &[Manufacturer, Brewery]],
        Wolverhampton => &[&[Manufacturer], &[Manufacturer, CoalMine]],
        Coalbrookdale => &[&[IronWorks, Brewery], &[IronWorks], &[CoalMine]],
        Dudley => &[&[CoalMine], &[IronWorks]],
        Kidderminster => &[&[CottonMill, CoalMine], &[CottonMill]],
        Worcester => &[&[CottonMill], &[CottonMill]],
        Birmingham => &[
            &[CottonMill, Manufacturer],
            &[Manufacturer],
            &[IronWorks],
            &[Manufacturer],
        ],
        Coventry => &[
            &[Pottery],
            &[Manufacturer, CoalMine],
            &[IronWorks, Manufacturer],
        ],
        Nuneaton => &[&[Manufacturer, Brewery], &[CottonMill, CoalMine]],
        Redditch => &[&[Manufacturer, CoalMine], &[IronWorks]],
        _ => &[],
    }
}

// ---------------------------------------------------------------------------
// Merchants
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MerchantBonus {
    Vp(u8),
    Money(i32),
    Income(u8),
    Develop(u8),
}

pub struct MerchantDef {
    pub loc: Loc,
    pub slots: usize,
    pub min_players: usize,
    pub bonus: MerchantBonus,
}

pub fn merchant_defs() -> &'static [MerchantDef] {
    &[
        MerchantDef {
            loc: Loc::Shrewsbury,
            slots: 1,
            min_players: 2,
            bonus: MerchantBonus::Vp(4),
        },
        MerchantDef {
            loc: Loc::Gloucester,
            slots: 2,
            min_players: 2,
            bonus: MerchantBonus::Develop(1),
        },
        MerchantDef {
            loc: Loc::Oxford,
            slots: 2,
            min_players: 2,
            bonus: MerchantBonus::Income(2),
        },
        MerchantDef {
            loc: Loc::Warrington,
            slots: 2,
            min_players: 3,
            bonus: MerchantBonus::Money(5),
        },
        MerchantDef {
            loc: Loc::Nottingham,
            slots: 2,
            min_players: 4,
            bonus: MerchantBonus::Vp(3),
        },
    ]
}

/// Bonus granted by the merchant located at `loc` (no-op `Vp(0)` if none).
pub fn merchant_bonus_at(loc: Loc) -> MerchantBonus {
    merchant_defs()
        .iter()
        .find(|def| def.loc == loc)
        .map(|def| def.bonus)
        .unwrap_or(MerchantBonus::Vp(0))
}

// Merchant tile mix: the ACTIVE set of tiles for each player count.
// Each player count has its own independent set (NOT a cumulative "2p plus
// 3p plus 4p" pool — that would yield two 'Any' tiles at 4 players, which
// the physical rulebook does not).
//
// User-verified against the physical rulebook:
//   2p: Blank, Blank, Any, Cotton, Manufacturer        (5 tiles)
//   3p: + Blank, Pottery                               (7 tiles)
//   4p: + Cotton, Manufacturer                         (9 tiles)
// So the 4p set is: Blank x3, Any x1, Cotton x2, Manufacturer x2, Pottery x1.
pub fn merchant_tile_mix(player_count: usize) -> &'static [MerchantMixEntry] {
    match player_count {
        2 => &[
            MerchantMixEntry::Blank,
            MerchantMixEntry::Blank,
            MerchantMixEntry::Any,
            MerchantMixEntry::Buys(IndustryType::CottonMill),
            MerchantMixEntry::Buys(IndustryType::Manufacturer),
        ],
        3 => &[
            MerchantMixEntry::Blank,
            MerchantMixEntry::Blank,
            MerchantMixEntry::Blank,
            MerchantMixEntry::Any,
            MerchantMixEntry::Buys(IndustryType::CottonMill),
            MerchantMixEntry::Buys(IndustryType::Manufacturer),
            MerchantMixEntry::Buys(IndustryType::Pottery),
        ],
        _ => &[
            MerchantMixEntry::Blank,
            MerchantMixEntry::Blank,
            MerchantMixEntry::Blank,
            MerchantMixEntry::Any,
            MerchantMixEntry::Buys(IndustryType::CottonMill),
            MerchantMixEntry::Buys(IndustryType::CottonMill),
            MerchantMixEntry::Buys(IndustryType::Manufacturer),
            MerchantMixEntry::Buys(IndustryType::Manufacturer),
            MerchantMixEntry::Buys(IndustryType::Pottery),
        ],
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MerchantMixEntry {
    Blank,
    Any,
    Buys(IndustryType),
}

// ---------------------------------------------------------------------------
// Connections
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy)]
pub struct Connection {
    pub id: usize,
    pub a: Loc,
    pub b: Loc,
    pub canal: bool,
    pub rail: bool,
    pub via_farm: Option<Loc>,
}

/// (neighbor, conn_id) adjacency per location, built once from the static map.
/// Shared by the connectivity cache (`GameState::connected_mask`) and any
/// graph traversal.
pub fn adjacency() -> &'static [Vec<(Loc, usize)>] {
    static ADJ: OnceLock<Vec<Vec<(Loc, usize)>>> = OnceLock::new();
    ADJ.get_or_init(|| {
        let mut adj = vec![Vec::new(); ALL_LOCATIONS.len()];
        for c in connections() {
            let (a, b) = (c.a as usize, c.b as usize);
            adj[a].push((c.b, c.id));
            adj[b].push((c.a, c.id));
            if let Some(v) = c.via_farm {
                let v = v as usize;
                adj[a].push((c.via_farm.unwrap(), c.id));
                adj[b].push((c.via_farm.unwrap(), c.id));
                adj[v].push((c.a, c.id));
                adj[v].push((c.b, c.id));
            }
        }
        adj
    })
}

/// Connection ids touching each location (endpoint or via-farm). Used by the
/// per-player network-mask maintenance (`GameState::clear_network_location`).
pub(crate) fn loc_connections() -> &'static [Vec<usize>] {
    static LC: OnceLock<Vec<Vec<usize>>> = OnceLock::new();
    LC.get_or_init(|| {
        let mut v = vec![Vec::new(); ALL_LOCATIONS.len()];
        for c in connections() {
            v[c.a as usize].push(c.id);
            v[c.b as usize].push(c.id);
            if let Some(f) = c.via_farm {
                v[f as usize].push(c.id);
            }
        }
        v
    })
}

// Translated verbatim from CONNECTIONS in gameData.js.
// via_farm only for kidderminster-worcester (via southern brewery).
pub fn connections() -> &'static [Connection] {
    use Loc::*;
    macro_rules! conn {
        ($id:expr, $a:expr, $b:expr, $canal:expr, $rail:expr) => {
            Connection {
                id: $id,
                a: $a,
                b: $b,
                canal: $canal,
                rail: $rail,
                via_farm: None,
            }
        };
    }
    macro_rules! conn_via_farm {
        ($id:expr, $a:expr, $b:expr, $canal:expr, $rail:expr, $farm:expr) => {
            Connection {
                id: $id,
                a: $a,
                b: $b,
                canal: $canal,
                rail: $rail,
                via_farm: Some($farm),
            }
        };
    }
    &[
        conn!(0, Belper, Derby, true, true),
        conn!(1, Belper, Leek, false, true),
        conn!(2, Birmingham, Coventry, true, true),
        conn!(3, Birmingham, Dudley, true, true),
        conn!(4, Birmingham, Nuneaton, false, true),
        conn!(5, Birmingham, Oxford, true, true),
        conn!(6, Birmingham, Redditch, false, true),
        conn!(7, Birmingham, Tamworth, true, true),
        conn!(8, Birmingham, Walsall, true, true),
        conn!(9, Birmingham, Worcester, true, true),
        conn!(10, BurtonOnTrent, Cannock, false, true),
        conn!(11, BurtonOnTrent, Derby, true, true),
        conn!(12, BurtonOnTrent, Stone, true, true),
        conn!(13, BurtonOnTrent, Tamworth, true, true),
        conn!(14, BurtonOnTrent, Walsall, true, false),
        conn!(15, Cannock, Stafford, true, true),
        conn!(16, Cannock, BreweryNorth, true, true),
        conn!(17, Cannock, Walsall, true, true),
        conn!(18, Cannock, Wolverhampton, true, true),
        conn!(19, Coalbrookdale, Kidderminster, true, true),
        conn!(20, Coalbrookdale, Shrewsbury, true, true),
        conn!(21, Coalbrookdale, Wolverhampton, true, true),
        conn!(22, Coventry, Nuneaton, false, true),
        conn!(23, Derby, Nottingham, true, true),
        conn!(24, Derby, Uttoxeter, false, true),
        conn!(25, Dudley, Kidderminster, true, true),
        conn!(26, Dudley, Wolverhampton, true, true),
        conn!(27, Gloucester, Redditch, true, true),
        conn!(28, Gloucester, Worcester, true, true),
        conn_via_farm!(29, Kidderminster, Worcester, true, true, BrewerySouth),
        conn!(30, Leek, StokeOnTrent, true, true),
        conn!(31, Nuneaton, Tamworth, true, true),
        conn!(32, Redditch, Oxford, true, true),
        conn!(33, Stafford, Stone, true, true),
        conn!(34, StokeOnTrent, Stone, true, true),
        conn!(35, StokeOnTrent, Warrington, true, true),
        conn!(36, Stone, Uttoxeter, false, true),
        conn!(37, Tamworth, Walsall, false, true),
        conn!(38, Walsall, Wolverhampton, true, true),
    ]
}

pub fn connection_via_farm() -> usize {
    // kidderminster-worcester carries the southern brewery farm
    29
}

// ---------------------------------------------------------------------------
// Card deck composition (by player count)
// ---------------------------------------------------------------------------

pub const HAND_SIZE: usize = 8;

pub fn location_cards(player_count: usize) -> &'static [(Loc, u8)] {
    use Loc::*;
    match player_count {
        2 => &[
            (Stafford, 2),
            (BurtonOnTrent, 2),
            (Cannock, 2),
            (Tamworth, 1),
            (Walsall, 1),
            (Coalbrookdale, 3),
            (Dudley, 2),
            (Kidderminster, 2),
            (Wolverhampton, 2),
            (Worcester, 2),
            (Birmingham, 3),
            (Coventry, 3),
            (Nuneaton, 1),
            (Redditch, 1),
        ],
        3 => &[
            (Leek, 2),
            (StokeOnTrent, 3),
            (Stone, 2),
            (Uttoxeter, 1),
            (Stafford, 2),
            (BurtonOnTrent, 2),
            (Cannock, 2),
            (Tamworth, 1),
            (Walsall, 1),
            (Coalbrookdale, 3),
            (Dudley, 2),
            (Kidderminster, 2),
            (Wolverhampton, 2),
            (Worcester, 2),
            (Birmingham, 3),
            (Coventry, 3),
            (Nuneaton, 1),
            (Redditch, 1),
        ],
        _ => &[
            (Belper, 2),
            (Derby, 3),
            (Leek, 2),
            (StokeOnTrent, 3),
            (Stone, 2),
            (Uttoxeter, 2),
            (Stafford, 2),
            (BurtonOnTrent, 2),
            (Cannock, 2),
            (Tamworth, 1),
            (Walsall, 1),
            (Coalbrookdale, 3),
            (Dudley, 2),
            (Kidderminster, 2),
            (Wolverhampton, 2),
            (Worcester, 2),
            (Birmingham, 3),
            (Coventry, 3),
            (Nuneaton, 1),
            (Redditch, 1),
        ],
    }
}

pub fn industry_cards(player_count: usize) -> &'static [(IndustryType, u8)] {
    use IndustryType::*;
    let _common: &[(IndustryType, u8)] = &[(IronWorks, 4), (Pottery, 2), (Brewery, 5)];
    match player_count {
        2 => &[(IronWorks, 4), (CoalMine, 2), (Pottery, 2), (Brewery, 5)],
        3 => &[(IronWorks, 4), (CoalMine, 2), (Pottery, 2), (Brewery, 5)],
        _ => &[(IronWorks, 4), (CoalMine, 3), (Pottery, 3), (Brewery, 5)],
    }
}

pub fn dual_cotton_manufacturer_cards(player_count: usize) -> u8 {
    match player_count {
        2 => 0,
        3 => 6,
        _ => 8,
    }
}

// ---------------------------------------------------------------------------
// Market / income / money constants (from gameData.js)
// ---------------------------------------------------------------------------

pub const COAL_MARKET_PRICES: [u8; 14] = [1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7];
pub const COAL_MARKET_INITIAL: usize = 13;
pub const COAL_EMPTY_PRICE: u8 = 8;

pub const IRON_MARKET_PRICES: [u8; 10] = [1, 1, 2, 2, 3, 3, 4, 4, 5, 5];
pub const IRON_MARKET_INITIAL: usize = 8;
pub const IRON_EMPTY_PRICE: u8 = 6;

/// How many "General Supply" market entries `find_coal_sources` /
/// `find_iron_sources` append after the real market cubes. General Supply is
/// effectively UNLIMITED at the empty price (8/6) once the market runs out, so
/// the pool only needs enough entries to cover the maximum any single action
/// can draw from one market: a single action needs at most 2 cubes (build /
/// develop / double-rail), a player's full round (two actions) at most 4.
pub const GENERAL_SUPPLY_CAP: usize = 4;

/// Bitmask over the 27 locations of all merchant locations. Used with the
/// connected-component cache to test whether a merchant is reachable.
pub const MERCHANT_LOC_MASK: u32 = (1 << (Loc::Shrewsbury as u8))
    | (1 << (Loc::Gloucester as u8))
    | (1 << (Loc::Oxford as u8))
    | (1 << (Loc::Warrington as u8))
    | (1 << (Loc::Nottingham as u8));

pub const INITIAL_MONEY: i32 = 17;
pub const INITIAL_INCOME_SPACE: u8 = 10;
pub const LOAN_AMOUNT: i32 = 30;
pub const LOAN_INCOME_PENALTY: i8 = 3;
pub const MAX_INCOME: i8 = 30;
pub const MIN_INCOME: i8 = -10;
pub const MAX_INCOME_SPACE: u8 = 99;

pub const CANAL_LINK_COST: i32 = 3;
pub const RAIL_LINK_COST: i32 = 5;
pub const RAIL_DOUBLE_LINK_COST: i32 = 15;
pub const COAL_PER_RAIL_LINK: u8 = 1;
pub const BEER_FOR_DOUBLE_RAIL: u8 = 1;

pub const ACTIONS_PER_TURN: usize = 2;
pub const FIRST_ROUND_ACTIONS: usize = 1;

pub const LINKS_PER_PLAYER: u8 = 14;

pub const WILD_LOCATION_PILE: u8 = 4;
pub const WILD_INDUSTRY_PILE: u8 = 4;
