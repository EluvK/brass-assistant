use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum IndustryType {
    CottonMill,
    CoalMine,
    IronWorks,
    Manufacturer,
    Pottery,
    Brewery,
}

impl IndustryType {
    pub const ALL: [IndustryType; 6] = [
        IndustryType::CottonMill,
        IndustryType::CoalMine,
        IndustryType::IronWorks,
        IndustryType::Manufacturer,
        IndustryType::Pottery,
        IndustryType::Brewery,
    ];

    pub fn name(&self) -> &'static str {
        match self {
            IndustryType::CottonMill => "Cotton Mill",
            IndustryType::CoalMine => "Coal Mine",
            IndustryType::IronWorks => "Iron Works",
            IndustryType::Manufacturer => "Manufacturer",
            IndustryType::Pottery => "Pottery",
            IndustryType::Brewery => "Brewery",
        }
    }

    pub fn is_resource(&self) -> bool {
        matches!(
            self,
            IndustryType::CoalMine | IndustryType::IronWorks | IndustryType::Brewery
        )
    }

    pub fn is_sellable(&self) -> bool {
        matches!(
            self,
            IndustryType::CottonMill | IndustryType::Manufacturer | IndustryType::Pottery
        )
    }
}

impl fmt::Display for IndustryType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.name())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Era {
    Canal,
    Rail,
}

impl fmt::Display for Era {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Era::Canal => write!(f, "Canal"),
            Era::Rail => write!(f, "Rail"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Action {
    Build,
    Network,
    Develop,
    Sell,
    Loan,
    Scout,
    Pass,
}

impl Action {
    pub fn index(&self) -> usize {
        match self {
            Action::Build => 0,
            Action::Network => 1,
            Action::Develop => 2,
            Action::Sell => 3,
            Action::Loan => 4,
            Action::Scout => 5,
            Action::Pass => 6,
        }
    }

    pub const ALL: [Action; 7] = [
        Action::Build,
        Action::Network,
        Action::Develop,
        Action::Sell,
        Action::Loan,
        Action::Scout,
        Action::Pass,
    ];
}

impl fmt::Display for Action {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}",
            match self {
                Action::Build => "Build",
                Action::Network => "Network",
                Action::Develop => "Develop",
                Action::Sell => "Sell",
                Action::Loan => "Loan",
                Action::Scout => "Scout",
                Action::Pass => "Pass",
            }
        )
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CardType {
    Location,
    Industry,
    WildLocation,
    WildIndustry,
}

// ---------------------------------------------------------------------------
// Tile definitions (from INDUSTRY_DATA in gameData.js)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy)]
pub struct TileDef {
    pub level: u8,
    pub count: u8,
    pub canal_era: bool,
    pub rail_era: bool,
    pub cost: i32,
    pub cost_coal: u8,
    pub cost_iron: u8,
    pub beers_to_sell: Option<u8>,
    pub vp: u8,
    pub income: u8,
    pub link_vp: u8,
    pub can_develop: bool,
    pub resource_cubes: u8,
}

macro_rules! tile {
    ($level:expr, $count:expr, $canal:expr, $rail:expr, $cost:expr, $coal:expr,
     $iron:expr, $beer:expr, $vp:expr, $income:expr, $link:expr, $dev:expr, $cubes:expr) => {
        TileDef {
            level: $level,
            count: $count,
            canal_era: $canal,
            rail_era: $rail,
            cost: $cost,
            cost_coal: $coal,
            cost_iron: $iron,
            beers_to_sell: $beer,
            vp: $vp,
            income: $income,
            link_vp: $link,
            can_develop: $dev,
            resource_cubes: $cubes,
        }
    };
}

// Translated verbatim from INDUSTRY_DATA in gameData.js
pub fn industry_tiles(ind: IndustryType) -> &'static [TileDef] {
    use IndustryType::*;
    match ind {
        Brewery => &[
            tile!(1, 2, true, false, 5, 0, 1, None, 4, 4, 2, true, 1),
            tile!(2, 2, true, true, 7, 0, 1, None, 5, 5, 2, true, 1),
            tile!(3, 2, true, true, 9, 0, 1, None, 7, 5, 2, true, 1),
            tile!(4, 1, false, true, 9, 0, 1, None, 10, 5, 2, true, 2),
        ],
        CoalMine => &[
            tile!(1, 1, true, false, 5, 0, 0, None, 1, 4, 2, true, 2),
            tile!(2, 2, true, true, 7, 0, 0, None, 2, 7, 1, true, 3),
            tile!(3, 2, true, true, 8, 0, 1, None, 3, 6, 1, true, 4),
            tile!(4, 2, true, true, 10, 0, 1, None, 4, 5, 1, true, 5),
        ],
        CottonMill => &[
            tile!(1, 3, true, false, 12, 0, 0, Some(1), 5, 5, 1, true, 0),
            tile!(2, 2, true, true, 14, 1, 0, Some(1), 5, 4, 2, true, 0),
            tile!(3, 3, true, true, 16, 1, 1, Some(1), 9, 3, 1, true, 0),
            tile!(4, 3, true, true, 18, 1, 1, Some(1), 12, 2, 1, true, 0),
        ],
        IronWorks => &[
            tile!(1, 1, true, false, 5, 1, 0, None, 3, 3, 1, true, 4),
            tile!(2, 1, true, true, 7, 1, 0, None, 5, 3, 1, true, 4),
            tile!(3, 1, true, true, 9, 1, 0, None, 7, 2, 1, true, 5),
            tile!(4, 1, true, true, 12, 1, 0, None, 9, 1, 1, true, 6),
        ],
        Manufacturer => &[
            tile!(1, 1, true, false, 8, 1, 0, Some(1), 3, 5, 2, true, 0),
            tile!(2, 2, true, true, 10, 0, 1, Some(1), 5, 1, 1, true, 0),
            tile!(3, 1, true, true, 12, 2, 0, Some(0), 4, 4, 0, true, 0),
            tile!(4, 1, true, true, 8, 0, 1, Some(1), 3, 6, 1, true, 0),
            tile!(5, 2, true, true, 16, 1, 0, Some(2), 8, 2, 2, true, 0),
            tile!(6, 1, true, true, 20, 0, 0, Some(1), 7, 6, 1, true, 0),
            tile!(7, 1, true, true, 16, 1, 1, Some(0), 9, 4, 0, true, 0),
            tile!(8, 2, true, true, 20, 0, 2, Some(1), 11, 1, 1, true, 0),
        ],
        Pottery => &[
            tile!(1, 1, true, true, 17, 0, 1, Some(1), 10, 5, 1, false, 0),
            tile!(2, 1, true, true, 0, 1, 0, Some(1), 1, 1, 1, true, 0),
            tile!(3, 1, true, true, 22, 2, 0, Some(2), 11, 5, 1, false, 0),
            tile!(4, 1, true, true, 0, 1, 0, Some(1), 1, 1, 1, true, 0),
            tile!(5, 1, false, true, 24, 2, 0, Some(2), 20, 5, 1, true, 0),
        ],
    }
}
