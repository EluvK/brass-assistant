//! Canonical action representation shared by gameplay, AI, and bridges.

use crate::data::{Action, IndustryType};
use crate::graph::{BeerSource, CoalSource, IronSource};
use crate::map::{Loc, connections};
use crate::state::GameState;

#[derive(Debug, Clone)]
pub enum Move {
    Build {
        loc: Loc,
        slot_index: usize,
        ind: IndustryType,
        coal: Vec<CoalSource>,
        iron: Vec<IronSource>,
        card_index: usize,
    },
    Network {
        conn_id: usize,
        coal: Option<CoalSource>,
        card_index: usize,
    },
    NetworkDouble {
        conn1: usize,
        conn2: usize,
        coal1: CoalSource,
        coal2: CoalSource,
        beer: BeerSource,
        card_index: usize,
    },
    Develop {
        ind1: IndustryType,
        ind2: Option<IndustryType>,
        iron: Vec<IronSource>,
        card_index: usize,
    },
    Sell {
        keys: Vec<usize>,
        merchant_indices: Vec<usize>,
        use_merchant_beer: Vec<bool>,
        beer_sources: Vec<Vec<BeerSource>>,
        /// Gloucester's merchant bonus is resolved atomically with this sell.
        /// `None` means the selected sale does not award a free develop.
        free_develop: Option<IndustryType>,
        card_index: usize,
    },
    Loan {
        card_index: usize,
    },
    Scout {
        card_indices: [usize; 3],
    },
    Pass {
        card_index: usize,
    },
}

impl Move {
    pub fn action(&self) -> Action {
        match self {
            Move::Build { .. } => Action::Build,
            Move::Network { .. } | Move::NetworkDouble { .. } => Action::Network,
            Move::Develop { .. } => Action::Develop,
            Move::Sell { .. } => Action::Sell,
            Move::Loan { .. } => Action::Loan,
            Move::Scout { .. } => Action::Scout,
            Move::Pass { .. } => Action::Pass,
        }
    }

    pub fn describe(&self, _state: &GameState) -> String {
        match self {
            Move::Build { loc, ind, .. } => format!("Build {} at {}", ind.name(), loc.name()),
            Move::Network { conn_id, .. } => {
                let c = &connections()[*conn_id];
                format!("Network {} - {}", c.a.name(), c.b.name())
            }
            Move::NetworkDouble { conn1, conn2, .. } => {
                let c1 = &connections()[*conn1];
                let c2 = &connections()[*conn2];
                format!(
                    "Network x2: {} - {} and {} - {}",
                    c1.a.name(),
                    c1.b.name(),
                    c2.a.name(),
                    c2.b.name()
                )
            }
            Move::Develop { ind1, ind2, .. } => match ind2 {
                Some(i2) => format!("Develop {} + {}", ind1.name(), i2.name()),
                None => format!("Develop {}", ind1.name()),
            },
            Move::Sell {
                keys, free_develop, ..
            } => match free_develop {
                Some(ind) => format!("Sell {} tile(s) + free develop {}", keys.len(), ind.name()),
                None => format!("Sell {} tile(s)", keys.len()),
            },
            Move::Loan { .. } => "Take loan".to_string(),
            Move::Scout { .. } => "Scout (wild cards)".to_string(),
            Move::Pass { .. } => "Pass".to_string(),
        }
    }
}
