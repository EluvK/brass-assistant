//! Canonical action representation shared by gameplay, AI, and bridges.

use crate::data::{Action, IndustryType};
use crate::graph::{BeerSource, CoalSource, IronSource};
use crate::map::{Loc, connections};
use crate::state::GameState;

/// A card that may be used to complete a structural move.
#[derive(Debug, Clone, PartialEq)]
pub struct CardCandidate {
    pub hand_index: usize,
}

/// The card-selection shape required to turn a [`Move`] into a
/// [`ResolvedMove`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CardRequirement {
    One,
    ScoutThree,
}

/// A structural legal operation.  It deliberately contains no chosen card.
#[derive(Debug, Clone)]
pub enum Move {
    Build {
        loc: Loc,
        slot_index: usize,
        ind: IndustryType,
        coal: Vec<CoalSource>,
        iron: Vec<IronSource>,
        card_candidates: Vec<CardCandidate>,
    },
    Network {
        conn_id: usize,
        coal: Option<CoalSource>,
        card_candidates: Vec<CardCandidate>,
    },
    NetworkDouble {
        conn1: usize,
        conn2: usize,
        coal1: CoalSource,
        coal2: CoalSource,
        beer: BeerSource,
        card_candidates: Vec<CardCandidate>,
    },
    Develop {
        ind1: IndustryType,
        ind2: Option<IndustryType>,
        iron: Vec<IronSource>,
        card_candidates: Vec<CardCandidate>,
    },
    Sell {
        keys: Vec<usize>,
        merchant_indices: Vec<usize>,
        use_merchant_beer: Vec<bool>,
        beer_sources: Vec<Vec<BeerSource>>,
        free_develop: Option<IndustryType>,
        card_candidates: Vec<CardCandidate>,
    },
    Loan {
        card_candidates: Vec<CardCandidate>,
    },
    Scout {
        card_candidates: Vec<CardCandidate>,
    },
    Pass {
        card_candidates: Vec<CardCandidate>,
    },
}

/// A fully specified operation that rules may execute.
#[derive(Debug, Clone, PartialEq)]
pub enum ResolvedMove {
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

    pub fn card_requirement(&self) -> CardRequirement {
        match self {
            Move::Scout { .. } => CardRequirement::ScoutThree,
            _ => CardRequirement::One,
        }
    }

    pub fn card_candidates(&self) -> &[CardCandidate] {
        match self {
            Move::Build {
                card_candidates, ..
            }
            | Move::Network {
                card_candidates, ..
            }
            | Move::NetworkDouble {
                card_candidates, ..
            }
            | Move::Develop {
                card_candidates, ..
            }
            | Move::Sell {
                card_candidates, ..
            }
            | Move::Loan { card_candidates }
            | Move::Scout { card_candidates }
            | Move::Pass { card_candidates } => card_candidates,
        }
    }

    /// Validate a card-selection sequence and produce an executable move.
    pub fn resolve(&self, selected_cards: &[usize]) -> Result<ResolvedMove, String> {
        let expected = match self.card_requirement() {
            CardRequirement::One => 1,
            CardRequirement::ScoutThree => 3,
        };
        if selected_cards.len() != expected {
            return Err(format!("move requires {expected} selected card(s)"));
        }
        if selected_cards
            .iter()
            .any(|i| !self.card_candidates().iter().any(|c| c.hand_index == *i))
        {
            return Err("selected card is not a candidate for this move".into());
        }
        if self.card_requirement() == CardRequirement::ScoutThree {
            let mut unique = selected_cards.to_vec();
            unique.sort_unstable();
            unique.dedup();
            if unique.len() != 3 {
                return Err("Scout requires three distinct cards".into());
            }
        }
        Ok(match self {
            Move::Build {
                loc,
                slot_index,
                ind,
                coal,
                iron,
                ..
            } => ResolvedMove::Build {
                loc: *loc,
                slot_index: *slot_index,
                ind: *ind,
                coal: coal.clone(),
                iron: iron.clone(),
                card_index: selected_cards[0],
            },
            Move::Network { conn_id, coal, .. } => ResolvedMove::Network {
                conn_id: *conn_id,
                coal: *coal,
                card_index: selected_cards[0],
            },
            Move::NetworkDouble {
                conn1,
                conn2,
                coal1,
                coal2,
                beer,
                ..
            } => ResolvedMove::NetworkDouble {
                conn1: *conn1,
                conn2: *conn2,
                coal1: *coal1,
                coal2: *coal2,
                beer: *beer,
                card_index: selected_cards[0],
            },
            Move::Develop {
                ind1, ind2, iron, ..
            } => ResolvedMove::Develop {
                ind1: *ind1,
                ind2: *ind2,
                iron: iron.clone(),
                card_index: selected_cards[0],
            },
            Move::Sell {
                keys,
                merchant_indices,
                use_merchant_beer,
                beer_sources,
                free_develop,
                ..
            } => ResolvedMove::Sell {
                keys: keys.clone(),
                merchant_indices: merchant_indices.clone(),
                use_merchant_beer: use_merchant_beer.clone(),
                beer_sources: beer_sources.clone(),
                free_develop: *free_develop,
                card_index: selected_cards[0],
            },
            Move::Loan { .. } => ResolvedMove::Loan {
                card_index: selected_cards[0],
            },
            Move::Scout { .. } => ResolvedMove::Scout {
                card_indices: selected_cards
                    .try_into()
                    .expect("validated Scout card count"),
            },
            Move::Pass { .. } => ResolvedMove::Pass {
                card_index: selected_cards[0],
            },
        })
    }

    pub fn describe(&self, state: &GameState) -> String {
        if matches!(self, Move::Scout { .. }) {
            return "Scout (wild cards)".to_string();
        }
        self.resolve(
            &self
                .card_candidates()
                .first()
                .map(|c| c.hand_index)
                .into_iter()
                .collect::<Vec<_>>(),
        )
        .map(|mv| mv.describe(state))
        .unwrap_or_else(|_| format!("{:?}", self.action()))
    }
}

impl ResolvedMove {
    pub fn action(&self) -> Action {
        match self {
            ResolvedMove::Build { .. } => Action::Build,
            ResolvedMove::Network { .. } | ResolvedMove::NetworkDouble { .. } => Action::Network,
            ResolvedMove::Develop { .. } => Action::Develop,
            ResolvedMove::Sell { .. } => Action::Sell,
            ResolvedMove::Loan { .. } => Action::Loan,
            ResolvedMove::Scout { .. } => Action::Scout,
            ResolvedMove::Pass { .. } => Action::Pass,
        }
    }

    pub fn describe(&self, _state: &GameState) -> String {
        match self {
            ResolvedMove::Build { loc, ind, .. } => {
                format!("Build {} at {}", ind.name(), loc.name())
            }
            ResolvedMove::Network { conn_id, .. } => {
                let c = &connections()[*conn_id];
                format!("Network {} - {}", c.a.name(), c.b.name())
            }
            ResolvedMove::NetworkDouble { conn1, conn2, .. } => {
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
            ResolvedMove::Develop { ind1, ind2, .. } => match ind2 {
                Some(i2) => format!("Develop {} + {}", ind1.name(), i2.name()),
                None => format!("Develop {}", ind1.name()),
            },
            ResolvedMove::Sell {
                keys, free_develop, ..
            } => match free_develop {
                Some(ind) => format!("Sell {} tile(s) + free develop {}", keys.len(), ind.name()),
                None => format!("Sell {} tile(s)", keys.len()),
            },
            ResolvedMove::Loan { .. } => "Take loan".to_string(),
            ResolvedMove::Scout { .. } => "Scout (wild cards)".to_string(),
            ResolvedMove::Pass { .. } => "Pass".to_string(),
        }
    }
}
