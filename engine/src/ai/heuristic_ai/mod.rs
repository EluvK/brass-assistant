//! Heuristic AI baseline for Brass Birmingham.
//!
//! Architecture (one concern per module):
//!
//! - [`config`]: every tunable weight/threshold/switch, grouped by topic.
//! - [`context`]: `EvalContext`, built once per candidate batch — strategy
//!   phase, per-phase currency weights, and shared convenience predicates.
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
pub use plan::{Phase, Plan, compute_plan, era_phase};

// Card-selection head: public helpers for replay tooling and tests.
pub use cards::{card_choices_for_move, card_keep_score, ranked_card_choices};

use build::score_top_builds;
pub(crate) use cards::move_card_score;
use cards::{CardChoices, ranked_card_choices_with};
use context::EvalContext;
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
pub(crate) fn operation_key(mv: &ResolvedMove) -> String {
    match mv {
        ResolvedMove::Build {
            loc,
            slot_index,
            ind,
            coal,
            iron,
            ..
        } => format!("build:{loc:?}:{slot_index}:{ind:?}:{coal:?}:{iron:?}"),
        ResolvedMove::Network { conn_id, coal, .. } => format!("network:{conn_id}:{coal:?}"),
        ResolvedMove::NetworkDouble {
            conn1,
            conn2,
            coal1,
            coal2,
            beer,
            ..
        } => format!("network2:{conn1}:{conn2}:{coal1:?}:{coal2:?}:{beer:?}"),
        ResolvedMove::Develop {
            ind1, ind2, iron, ..
        } => format!("develop:{ind1:?}:{ind2:?}:{iron:?}"),
        ResolvedMove::Sell {
            keys,
            beer_sources,
            free_develop,
            ..
        } => format!("sell:{keys:?}:{beer_sources:?}:{free_develop:?}"),
        ResolvedMove::Loan { .. } => "loan".into(),
        // Scout is one operation; the three discarded cards belong to the
        // separate card-selection head and must not multiply operation nodes.
        ResolvedMove::Scout { .. } => "scout".into(),
        ResolvedMove::Pass { .. } => "pass".into(),
    }
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
    let cfg = HeuristicConfig::default();
    let ctx = EvalContext::new(state, pid, &cfg);

    // Card utility is state-wide and independent of the concrete action type.
    // Compute it once for this candidate batch and share it with all consumers.
    let build_targets = crate::rules::get_valid_build_targets(state, pid);
    let card_choices = ranked_card_choices_with(state, pid, &build_targets, &cfg);
    let mut out = Vec::new();
    let plan = compute_plan(state, pid);

    for d in score_top_builds(state, k, &plan, &build_targets, &card_choices) {
        if d.score != f64::NEG_INFINITY {
            out.push(d);
        }
    }
    out.extend(score_top_networks(state, k, &plan, &card_choices));
    out.extend(score_top_network_doubles(state, k, &plan, &card_choices));
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
    out.retain(|d| unique.insert(operation_key(&d.mv)));

    out.extend(score_sell_plans(state, &card_choices).into_iter().take(k));
    out.extend(
        score_loan_result(state, &plan, &card_choices)
            .into_iter()
            .take(k),
    );
    out.extend(
        score_scout_plan(state, &ctx, &card_choices)
            .into_iter()
            .take(k),
    );
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
    let cfg = HeuristicConfig::default();
    let card_index = ranked_card_choices(state, state.current_player_id())
        .first()
        .map(|(card_index, _)| *card_index)
        .unwrap_or(0);
    Decision {
        mv: ResolvedMove::Pass { card_index },
        score: cfg.scout.pass_fallback_score,
        card_score: card_keep_score(state, state.current_player_id(), card_index),
    }
}
