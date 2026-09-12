//! Heuristic AI baseline for Brass Birmingham.
//!
//! Architecture (one concern per module):
//!
//! - [`config`]: every tunable weight/threshold/switch, grouped by topic.
//! - [`context`]: shared strategy-phase data and declarative round/era
//!   factors.  Action-local scorers read the state directly when no profile
//!   is needed.
//! - [`value`]: shared market/board-value helpers. Build and network scores
//!   are emitted directly in VP equivalents.
//! - [`board`] / [`probability`]: shared board queries (merchant reach,
//!   beer availability, ...) and the single flip-probability model.
//! - One scoring module per action type (`build`, `network`, `develop`,
//!   `sell`, `loan`, `scout_pass`), each decomposed into named factor
//!   functions; `cards` is the independent card-selection head.
//! - [`plan`]: the "流派" production-plan selection; [`lookahead`]: the
//!   deterministic 2-ply action policy on top of the scored candidates.
//!
//! Public entry points are consumed by replay, the binaries, the PyO3 bridge
//! and network-guided search; their signatures are a stability contract.

use crate::data::IndustryType;
use crate::graph::{BeerSource, CoalSource, IronSource};
use crate::map::Loc;
use crate::rules::ResolvedMove;
use crate::state::GameState;

mod board;
mod build;
mod cards;
mod config;
mod context;
mod develop;
mod loan;
mod lookahead;
mod network;
mod plan;
mod probability;
mod scout_pass;
mod sell;
mod value;

pub use config::HeuristicConfig;
pub use context::EraProfile;
pub use lookahead::choose_action;
pub use plan::{Phase, era_phase};

// Card-selection head: public helpers for replay tooling and tests.
pub use cards::{card_choices_for_move, card_keep_score, ranked_card_choices};

use build::score_top_builds;
use cards::CardChoices;
pub(crate) use cards::move_card_score;
use develop::score_develop_plans;
use loan::score_loan_result;
use network::{score_top_network_doubles, score_top_networks};
use scout_pass::{score_pass_result, score_scout_plan};
use sell::score_sell_plans;

/// Candidate source-variant width: for each Build/Network/NetDouble
/// geometry (and Sell/Develop plan) the generator emits up to this many
/// variants that differ in free-source identity, so search — not the
/// generator — decides whose buildings flip. See docs/ai-action-encoding.md §6.
pub(crate) const SOURCE_VARIANTS: usize = 2;

pub struct Decision {
    pub mv: ResolvedMove,
    /// Score of the operation itself (independent of the card used).
    pub score: f64,
    /// Keep-value score of the card(s) consumed by this decision.
    pub card_score: f64,
}

/// Actions selected for one round.  The compatibility fields (`mv`, `score`,
/// and `card_score`) mirror the first action so existing one-action callers
/// can keep executing the plan incrementally.
pub struct RoundDecision {
    pub mv: ResolvedMove,
    pub score: f64,
    pub card_score: f64,
    /// The second action selected for the same round, when the turn permits
    /// two actions.  A last-position four-link is used only for evaluation;
    /// actions from the next round are intentionally not returned here.
    /// `choose_action` only fills this when the caller is at the start of a
    /// two-action block; callers that advance action-by-action inside the
    /// block receive a one-move plan and should re-plan on each action.
    pub second: Option<Decision>,
}

impl RoundDecision {
    /// Consume the plan as executable moves in round order.
    pub fn into_moves(self) -> Vec<ResolvedMove> {
        let mut moves = vec![self.mv];
        if let Some(second) = self.second {
            moves.push(second.mv);
        }
        moves
    }
}

/// First `m` source options that differ in free-source identity, judged by
/// `key` over each option's sources (market-only re-pricings collapse). The
/// first entry is the engine-default option; `T` is the option, `S` the source.
pub(crate) fn distinct_source_options<I, S, K>(
    options: I,
    key: impl Fn(&S) -> K,
    m: usize,
) -> Vec<Vec<S>>
where
    I: IntoIterator,
    I::Item: IntoIterator<Item = S>,
    S: Clone,
    K: Eq + std::hash::Hash,
{
    let mut seen = std::collections::HashSet::new();
    let mut out: Vec<Vec<S>> = Vec::new();
    for option in options {
        let option: Vec<S> = option.into_iter().collect();
        let signature: Vec<K> = option.iter().map(&key).collect();
        if seen.insert(signature) {
            out.push(option);
            if out.len() >= m {
                break;
            }
        }
    }
    out
}

/// Stable identity for the operation layer. Card references are intentionally
/// excluded; card selection is a separate policy dimension.
#[derive(Debug, Clone, Copy, Hash)]
pub enum OperationKey<'a> {
    Build {
        loc: Loc,
        slot_index: usize,
        ind: IndustryType,
        coal: &'a [CoalSource],
        iron: &'a [IronSource],
    },
    Network {
        conn_id: usize,
        coal: Option<CoalSource>,
    },
    NetworkDouble {
        conn1: usize,
        conn2: usize,
        coal1: CoalSource,
        coal2: CoalSource,
        beer: BeerSource,
    },
    Develop {
        ind1: IndustryType,
        ind2: Option<IndustryType>,
        iron: &'a [IronSource],
    },
    Sell {
        keys: &'a [usize],
        beer_sources: &'a [BeerSource],
        free_develop: Option<IndustryType>,
    },
    Loan,
    Scout,
    Pass,
}

impl<'a, 'b> PartialEq<OperationKey<'b>> for OperationKey<'a> {
    fn eq(&self, other: &OperationKey<'b>) -> bool {
        match (self, other) {
            (
                OperationKey::Build { loc: l1, slot_index: s1, ind: i1, coal: c1, iron: ir1 },
                OperationKey::Build { loc: l2, slot_index: s2, ind: i2, coal: c2, iron: ir2 },
            ) => l1 == l2 && s1 == s2 && i1 == i2 && c1 == c2 && ir1 == ir2,
            (
                OperationKey::Network { conn_id: c1, coal: co1 },
                OperationKey::Network { conn_id: c2, coal: co2 },
            ) => c1 == c2 && co1 == co2,
            (
                OperationKey::NetworkDouble { conn1: a1, conn2: b1, coal1: c1, coal2: d1, beer: e1 },
                OperationKey::NetworkDouble { conn1: a2, conn2: b2, coal1: c2, coal2: d2, beer: e2 },
            ) => a1 == a2 && b1 == b2 && c1 == c2 && d1 == d2 && e1 == e2,
            (
                OperationKey::Develop { ind1: a1, ind2: b1, iron: c1 },
                OperationKey::Develop { ind1: a2, ind2: b2, iron: c2 },
            ) => a1 == a2 && b1 == b2 && c1 == c2,
            (
                OperationKey::Sell { keys: k1, beer_sources: b1, free_develop: f1 },
                OperationKey::Sell { keys: k2, beer_sources: b2, free_develop: f2 },
            ) => k1 == k2 && b1 == b2 && f1 == f2,
            (OperationKey::Loan, OperationKey::Loan) => true,
            (OperationKey::Scout, OperationKey::Scout) => true,
            (OperationKey::Pass, OperationKey::Pass) => true,
            _ => false,
        }
    }
}
impl<'a> Eq for OperationKey<'a> {}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum OwnedOperationKey {
    Build {
        loc: Loc,
        slot_index: usize,
        ind: IndustryType,
        coal: Vec<CoalSource>,
        iron: Vec<IronSource>,
    },
    Network {
        conn_id: usize,
        coal: Option<CoalSource>,
    },
    NetworkDouble {
        conn1: usize,
        conn2: usize,
        coal1: CoalSource,
        coal2: CoalSource,
        beer: BeerSource,
    },
    Develop {
        ind1: IndustryType,
        ind2: Option<IndustryType>,
        iron: Vec<IronSource>,
    },
    Sell {
        keys: Vec<usize>,
        beer_sources: Vec<BeerSource>,
        free_develop: Option<IndustryType>,
    },
    Loan,
    Scout,
    Pass,
}

impl OwnedOperationKey {
    pub fn as_borrowed(&self) -> OperationKey<'_> {
        match self {
            OwnedOperationKey::Build {
                loc,
                slot_index,
                ind,
                coal,
                iron,
            } => OperationKey::Build {
                loc: *loc,
                slot_index: *slot_index,
                ind: *ind,
                coal: coal.as_slice(),
                iron: iron.as_slice(),
            },
            OwnedOperationKey::Network { conn_id, coal } => OperationKey::Network {
                conn_id: *conn_id,
                coal: *coal,
            },
            OwnedOperationKey::NetworkDouble {
                conn1,
                conn2,
                coal1,
                coal2,
                beer,
            } => OperationKey::NetworkDouble {
                conn1: *conn1,
                conn2: *conn2,
                coal1: *coal1,
                coal2: *coal2,
                beer: *beer,
            },
            OwnedOperationKey::Develop { ind1, ind2, iron } => OperationKey::Develop {
                ind1: *ind1,
                ind2: *ind2,
                iron: iron.as_slice(),
            },
            OwnedOperationKey::Sell {
                keys,
                beer_sources,
                free_develop,
            } => OperationKey::Sell {
                keys: keys.as_slice(),
                beer_sources: beer_sources.as_slice(),
                free_develop: *free_develop,
            },
            OwnedOperationKey::Loan => OperationKey::Loan,
            OwnedOperationKey::Scout => OperationKey::Scout,
            OwnedOperationKey::Pass => OperationKey::Pass,
        }
    }
}

impl<'a> PartialEq<OwnedOperationKey> for OperationKey<'a> {
    fn eq(&self, other: &OwnedOperationKey) -> bool {
        *self == other.as_borrowed()
    }
}

impl<'a> PartialEq<OperationKey<'a>> for OwnedOperationKey {
    fn eq(&self, other: &OperationKey<'a>) -> bool {
        self.as_borrowed() == *other
    }
}

impl<'a> OperationKey<'a> {
    pub fn to_owned(&self) -> OwnedOperationKey {
        match *self {
            OperationKey::Build {
                loc,
                slot_index,
                ind,
                coal,
                iron,
            } => OwnedOperationKey::Build {
                loc,
                slot_index,
                ind,
                coal: coal.to_vec(),
                iron: iron.to_vec(),
            },
            OperationKey::Network { conn_id, coal } => OwnedOperationKey::Network { conn_id, coal },
            OperationKey::NetworkDouble {
                conn1,
                conn2,
                coal1,
                coal2,
                beer,
            } => OwnedOperationKey::NetworkDouble {
                conn1,
                conn2,
                coal1,
                coal2,
                beer,
            },
            OperationKey::Develop { ind1, ind2, iron } => OwnedOperationKey::Develop {
                ind1,
                ind2,
                iron: iron.to_vec(),
            },
            OperationKey::Sell {
                keys,
                beer_sources,
                free_develop,
            } => OwnedOperationKey::Sell {
                keys: keys.to_vec(),
                beer_sources: beer_sources.to_vec(),
                free_develop,
            },
            OperationKey::Loan => OwnedOperationKey::Loan,
            OperationKey::Scout => OwnedOperationKey::Scout,
            OperationKey::Pass => OwnedOperationKey::Pass,
        }
    }
}

pub fn operation_key<'a>(mv: &'a ResolvedMove) -> OperationKey<'a> {
    match mv {
        ResolvedMove::Build {
            loc,
            slot_index,
            ind,
            coal,
            iron,
            ..
        } => OperationKey::Build {
            loc: *loc,
            slot_index: *slot_index,
            ind: *ind,
            coal: coal.as_slice(),
            iron: iron.as_slice(),
        },
        ResolvedMove::Network { conn_id, coal, .. } => OperationKey::Network {
            conn_id: *conn_id,
            coal: *coal,
        },
        ResolvedMove::NetworkDouble {
            conn1,
            conn2,
            coal1,
            coal2,
            beer,
            ..
        } => OperationKey::NetworkDouble {
            conn1: *conn1,
            conn2: *conn2,
            coal1: *coal1,
            coal2: *coal2,
            beer: *beer,
        },
        ResolvedMove::Develop {
            ind1,
            ind2,
            iron,
            ..
        } => OperationKey::Develop {
            ind1: *ind1,
            ind2: *ind2,
            iron: iron.as_slice(),
        },
        ResolvedMove::Sell {
            keys,
            beer_sources,
            free_develop,
            ..
        } => OperationKey::Sell {
            keys: keys.as_slice(),
            beer_sources: beer_sources.as_slice(),
            free_develop: *free_develop,
        },
        ResolvedMove::Loan { .. } => OperationKey::Loan,
        ResolvedMove::Scout { .. } => OperationKey::Scout,
        ResolvedMove::Pass { .. } => OperationKey::Pass,
    }
}

pub fn hash_operation_key(mv: &ResolvedMove) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    operation_key(mv).hash(&mut hasher);
    hasher.finish()
}

/// Top-K candidates per action type, for consumers that want a wider prior.
/// Build, single Network, and double-Rail Network get up to `k` candidates
/// each; other action types keep their single best (develop/sell/loan/scout/pass).
pub fn candidate_actions_k(state: &mut GameState, k: usize) -> Vec<Decision> {
    // Network masks are (re)validated inside `get_valid_build_targets` /
    // `get_valid_network_targets` / `get_second_rail_options`, which every
    // scoring path below enters before any direct `is_in_network` read — no
    // redundant ensure here (candidate generation is on the search hot path).
    let pid = state.current_player_id();

    // Card utility is state-wide and independent of the concrete action type.
    // Compute it once for this candidate batch and share it with all consumers.
    let build_targets = crate::rules::get_valid_build_targets(state, pid);
    let card_choices = ranked_card_choices(state, pid);
    let mut out = Vec::new();

    for d in score_top_builds(state, k, &build_targets, &card_choices) {
        if d.score != f64::NEG_INFINITY {
            out.push(d);
        }
    }
    out.extend(score_top_networks(state, k, &card_choices));
    out.extend(score_top_network_doubles(state, k, &card_choices));
    // These scorers emit up to SOURCE_VARIANTS alternative plans per type.
    // Keep the iterator-based shape here so each branch obeys the same Top-K
    // contract and can grow to emit alternatives without changing this dispatcher.
    out.extend(
        score_develop_plans(state, &card_choices)
            .into_iter()
            .take(k),
    );

    // Enforce the operation/card separation at the candidate boundary. The
    // first candidate wins ties; its ResolvedMove retains one executable card index,
    // while `card_choices_for_move` exposes the independent card dimension.
    let mut unique = std::collections::HashSet::new();
    out.retain(|d| unique.insert(hash_operation_key(&d.mv)));

    out.extend(score_sell_plans(state, &card_choices).into_iter().take(k));
    out.extend(score_loan_result(state, &card_choices).into_iter().take(k));
    out.extend(score_scout_plan(state, &card_choices).into_iter().take(k));
    out.extend(score_pass_result(&card_choices).into_iter().take(k));

    // Candidate pruning must never turn a position with executable actions
    // into a dead end (for example callers passing k=0). Keep one legal
    // fallback so callers always have a path to explore.
    if out.is_empty()
        && let Some(mv) = crate::rules::legal_resolved_moves(state).into_iter().next()
    {
        let card_score = move_card_score(state, &mv);
        out.push(Decision {
            mv,
            score: f64::NEG_INFINITY,
            card_score,
        });
    }

    out
}

/// Fallback "pass" decision
pub fn pass_decision(state: &GameState) -> Decision {
    let card_index = ranked_card_choices(state, state.current_player_id())
        .first()
        .map(|(card_index, _)| *card_index)
        .unwrap_or(0);
    Decision {
        mv: ResolvedMove::Pass { card_index },
        score: scout_pass::PASS_FALLBACK_SCORE,
        card_score: card_keep_score(state, state.current_player_id(), card_index),
    }
}
