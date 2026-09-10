//! Network-guided ISMCTS (AlphaZero-style search) with batched PyO3 inference.
//!
//! The tree lives in Rust as an arena. Its children are concrete legal moves,
//! each assigned a node-local candidate id for policy outputs. Per simulation
//! the root is determinized by sampling opponent hands via the shared
//! `ai::determinize` helper.
//!
//! The callback returns one logit per padded concrete legal-action candidate
//! plus a four-player value vector. Rust masks by construction and applies
//! softmax only to the legal candidates.
//!
//! Network inference is BATCHED: simulations park at expand/leaf points, their
//! states are encoded in Rust and handed to a Python callback in waves; results
//! are applied before the next wave. This removes the per-sim Python/PyO3
//! orchestration that capped the previous Python MCTS at ~4 ms/sim.

use crate::bridge::action_features;
use crate::engine::{advance_turn, handle_turn_result};
use crate::heuristic_ai;
use crate::move_codec;
use crate::rules::ResolvedMove;
use crate::state::{Card, GameState};
use numpy::PyArray1;
use numpy::PyArray2;
use numpy::PyArrayMethods;
use pyo3::Py;
use pyo3::prelude::*;
use pyo3::types::PyAny;
use rand::RngExt;
use rand_chacha::ChaCha12Rng;
use rand_chacha::rand_core::SeedableRng;

pub const MAX_PLAYERS: usize = 4;

// ---------------------------------------------------------------------------
// Config & result
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy)]
pub struct NnMctsConfig {
    pub c_puct: f64,
    pub max_depth: usize,
    pub dirichlet_alpha: f64,
    pub dirichlet_weight: f64,
    pub batch_size: usize,
    /// Per-type shortlist width, shared with imitation teacher candidates.
    /// Zero means expand every legal concrete move.
    pub candidate_k: usize,
    /// Prior top-K pruning: score every legal candidate, then keep only the K
    /// highest-prior children in the tree. Zero keeps all of them.
    ///
    /// This is what makes PUCT well conditioned at this game's branching
    /// factor. The exploration term is `c_puct * P * sqrt(N) / (1 + n)`; with
    /// ~350 legal moves P ~ 1/350 and the term is ~0.08 at N = 500, far below
    /// any evaluated child's Q (~0.4), so unvisited children can never be
    /// selected and extra simulations only refine the handful the prior
    /// happened to rank first. At K = 32 the same term is ~0.9 and visits are
    /// allocated by value again.
    pub prior_top_k: usize,
    /// First-play urgency for unvisited children (see `select_child`).
    pub fpu: bool,
    pub fpu_reduction: f64,
}

impl Default for NnMctsConfig {
    fn default() -> Self {
        NnMctsConfig {
            c_puct: 2.5,
            max_depth: 10,
            dirichlet_alpha: 0.3,
            dirichlet_weight: 0.15,
            batch_size: 64,
            candidate_k: 0,
            prior_top_k: 0,
            fpu: true,
            fpu_reduction: 0.0,
        }
    }
}

/// Root children as (node-local candidate id, canonical, visits), best-first.
#[derive(Debug)]
pub struct NnSearchResult {
    pub best_canonical: Option<String>,
    pub children: Vec<(usize, String, u32)>,
    pub legal_candidate_ids: Vec<usize>,
    /// Simulations that aborted because a tree child could not be executed in
    /// the current determinization (see the note in `descend`). Diagnostic
    /// only: a growing count means the tree is reusing stale card selections.
    pub failed_applies: u32,
    /// Simulations that executed a tree child whose stored hand index now names
    /// a *different* card than the one enumerated (rules that only bound-check
    /// the index accept it silently). This is the common form of the tree-reuse
    /// problem and the one that biases value estimates; see `descend`.
    pub rewritten_applies: u32,
}

// ---------------------------------------------------------------------------
// Tree
// ---------------------------------------------------------------------------

struct Child {
    candidate_id: usize,
    mv: ResolvedMove,
    node: usize,
    /// The card this move discards, captured when the node was expanded.
    /// `ResolvedMove` stores a hand *index*, and a tree node outlives the
    /// determinization that created it, so the index can resolve to a
    /// different card in a later simulation. `None` for Scout (multi-card).
    declared_card: Option<Card>,
}

struct Node {
    player: usize,
    visits: u32,
    value_sum: [f64; MAX_PLAYERS],
    children: Vec<Child>,
    prior: Vec<f64>,
    legal_candidate_ids: Vec<usize>,
    expanded: bool,
}

impl Node {
    fn new(player: usize) -> Self {
        Node {
            player,
            visits: 0,
            value_sum: [0.0; MAX_PLAYERS],
            children: Vec::new(),
            prior: Vec::new(),
            legal_candidate_ids: Vec::new(),
            expanded: false,
        }
    }

    fn ready_to_select(&self) -> bool {
        self.expanded && !self.children.is_empty() && self.prior.len() == self.children.len()
    }

    fn q(&self, player: usize) -> f64 {
        if self.visits == 0 {
            0.0
        } else {
            self.value_sum[player] / self.visits as f64
        }
    }
}

// ---------------------------------------------------------------------------
// Batched network requests
// ---------------------------------------------------------------------------

enum RequestKind {
    /// Need policy priors over concrete legal candidates + the 4-player value.
    Expand {
        node_idx: usize,
        candidate_features: Vec<Vec<f32>>,
    },
    /// Need the 4-player value only (depth cap / no legal moves).
    Leaf,
}

struct Request {
    kind: RequestKind,
    state: GameState,
}

enum ParkedOutcome {
    Terminal {
        path: Vec<usize>,
        value: Vec<f64>,
    },
    Net {
        path: Vec<usize>,
        request_idx: usize,
    },
}

struct BatchResult {
    value: Vec<f64>,
    priors: Option<Vec<f64>>,
}

// ---------------------------------------------------------------------------
// Search
// ---------------------------------------------------------------------------

/// Run `sims` network-guided ISMCTS simulations from `state`.
pub fn search_net(
    state: &GameState,
    cfg: &NnMctsConfig,
    sims: usize,
    add_root_noise: bool,
    net_fn: &Py<PyAny>,
    py: Python<'_>,
) -> PyResult<NnSearchResult> {
    let root_pid = state.current_player_id();
    let n_players = state.player_count();

    let mut arena: Vec<Node> = vec![Node::new(root_pid)];

    // Independent simulation RNG (state.rng is setup-only and private).
    let mut sim_rng = ChaCha12Rng::from_rng(&mut rand::rng());
    let mut sims_left = sims;

    let mut requests: Vec<Request> = Vec::new();
    let mut request_by_node: std::collections::HashMap<usize, usize> =
        std::collections::HashMap::new();
    let mut parked: Vec<ParkedOutcome> = Vec::new();
    // Tree-reuse diagnostics: see the note in `descend` for what each counts.
    let mut failed_applies: u32 = 0;
    let mut rewritten_applies: u32 = 0;

    // Prime the root: expand it and fetch priors so the very first simulation
    // can descend past it. Without this, all sims of the first (and only, for
    // small sims budgets) wave would park at the root and no child would ever
    // be visited.
    {
        let mut work = crate::ai::determinize::determinize(state, &mut sim_rng);
        let mut root_reqs = Vec::new();
        let mut root_by_node = std::collections::HashMap::new();
        match descend(
            &mut work,
            &mut arena,
            cfg,
            &mut root_reqs,
            &mut root_by_node,
            false, // priming does not count as a simulation visit
            &mut failed_applies,
            &mut rewritten_applies,
        ) {
            ParkedOutcome::Net { .. } => {
                let results = flush_net(py, net_fn, &root_reqs)?;
                for (req, res) in root_reqs.iter().zip(results.iter()) {
                    if let RequestKind::Expand { node_idx, .. } = req.kind {
                        if let Some(p) = &res.priors {
                            apply_priors(&mut arena, node_idx, p.clone(), cfg.prior_top_k);
                        }
                    }
                }
                if add_root_noise {
                    apply_dirichlet_noise(
                        &mut arena[0],
                        cfg.dirichlet_alpha,
                        cfg.dirichlet_weight,
                        &mut sim_rng,
                    );
                }
            }
            ParkedOutcome::Terminal { .. } => {
                // (terminal game at the root: nothing to prime)
            }
        }
    }

    loop {
        // Phase 1: park simulations until the batch fills (or sims run out).
        // The wave is bounded by SIMS parked (each needs a value), not by the
        // (deduplicated) request count — otherwise a frontier collapse would
        // keep every sim on one request and the wave would never flush.
        while sims_left > 0 && parked.len() < cfg.batch_size {
            sims_left -= 1;
            let _ = sim_rng.random::<u64>(); // vary determinization per simulation
            let mut work = crate::ai::determinize::determinize(state, &mut sim_rng);
            match descend(
                &mut work,
                &mut arena,
                cfg,
                &mut requests,
                &mut request_by_node,
                true,
                &mut failed_applies,
                &mut rewritten_applies,
            ) {
                ParkedOutcome::Terminal { path, value } => {
                    add_value(&mut arena, &path, &value);
                }
                park @ ParkedOutcome::Net { .. } => parked.push(park),
            }
        }

        // Phase 2: flush the batch and apply priors.
        if !requests.is_empty() {
            let results = flush_net(py, net_fn, &requests)?;
            for (req, res) in requests.iter().zip(results.iter()) {
                if let RequestKind::Expand { node_idx, .. } = req.kind {
                    if let Some(p) = &res.priors {
                        apply_priors(&mut arena, node_idx, p.clone(), cfg.prior_top_k);
                    }
                }
            }

            // Phase 3: back-propagate the values of every parked simulation.
            for p in parked.drain(..) {
                if let ParkedOutcome::Net { path, request_idx } = p {
                    let value = results[request_idx].value.clone();
                    add_value(&mut arena, &path, &value);
                }
            }
            requests.clear();
            request_by_node.clear();
        }

        if sims_left == 0 {
            break;
        }
    }

    // Best child = most visits; fall back to any child if none was visited.
    let root = &arena[0];
    let mut children: Vec<(usize, String, u32)> = root
        .children
        .iter()
        .map(|c| {
            (
                c.candidate_id,
                move_codec::encode(&c.mv),
                arena[c.node].visits,
            )
        })
        .collect();
    children.sort_by(|a, b| b.2.cmp(&a.2));
    let best_candidate_id = children
        .first()
        .filter(|(_, _, v)| *v > 0)
        .map(|(s, _, _)| *s);
    let best_canonical = best_candidate_id.and_then(|candidate_id| {
        root.children
            .iter()
            .find(|c| c.candidate_id == candidate_id)
            .map(|c| move_codec::encode(&c.mv))
    });

    Ok(NnSearchResult {
        best_canonical,
        children,
        legal_candidate_ids: root.legal_candidate_ids.clone(),
        failed_applies,
        rewritten_applies,
    })
}

/// A simulation walks down the arena, applying moves to `work`, until it parks
/// at a terminal state, a leaf (depth cap), or an unexpanded node.
///
/// **Approximation (intentional):** node visits are incremented DURING descent
/// so PUCT exploration responds to progress within a wave, but `value_sum` is
/// only added at the batch flush. During a wave, `q = value_sum/visits` is
/// therefore transiently DEPRESSED for heavily-traversed nodes (their value_sum
/// lags behind their visit count). This acts as a mild "virtual loss" that
/// prevents the whole wave from collapsing onto the max-prior child; values are
/// exact once the wave flushes, so the final visit statistics are correct.
fn descend(
    work: &mut GameState,
    arena: &mut Vec<Node>,
    cfg: &NnMctsConfig,
    requests: &mut Vec<Request>,
    request_by_node: &mut std::collections::HashMap<usize, usize>,
    count_visits: bool,
    failed_applies: &mut u32,
    rewritten_applies: &mut u32,
) -> ParkedOutcome {
    let mut path = vec![0usize];
    let mut node_idx = 0usize;
    let mut depth = 0usize;

    loop {
        if count_visits {
            arena[node_idx].visits += 1;
        }
        if work.game_over {
            return ParkedOutcome::Terminal {
                path,
                value: terminal_value(work, work.player_count()),
            };
        }
        if depth >= cfg.max_depth {
            return park(work, requests, request_by_node, path, |_| RequestKind::Leaf);
        }

        let node = &arena[node_idx];
        if !node.expanded {
            // Keep inference candidates aligned with the imitation teacher.
            // The teacher uses candidate_actions_k(..., 4); expanding every
            // legal concrete move here would expose the net to many actions
            // that never appeared in the bootstrap training targets.
            let moves: Vec<ResolvedMove> = if cfg.candidate_k == 0 {
                crate::rules::legal_resolved_moves(work)
            } else {
                heuristic_ai::candidate_actions_k(work, cfg.candidate_k)
                    .into_iter()
                    .map(|decision| decision.mv)
                    .collect()
            };
            let mut children: Vec<Child> = Vec::new();
            let mut legal_candidate_ids: Vec<usize> = Vec::new();
            let mut candidate_features: Vec<Vec<f32>> = Vec::new();
            let mut row = Vec::new();
            let hand = &work.players[work.current_player_id()].hand;
            for (action_id, mv) in moves.into_iter().enumerate() {
                legal_candidate_ids.push(action_id);
                let child_idx = arena.len();
                arena.push(Node::new(0));
                children.push(Child {
                    candidate_id: action_id,
                    declared_card: declared_card(hand, &mv),
                    mv,
                    node: child_idx,
                });
                action_features::encode_move_into(work, &children.last().unwrap().mv, &mut row);
                candidate_features.push(row.clone());
            }
            arena[node_idx].children = children;
            arena[node_idx].legal_candidate_ids = legal_candidate_ids.clone();
            arena[node_idx].expanded = true;

            if legal_candidate_ids.is_empty() {
                // No legal moves: treat as a leaf (network value).
                return park(work, requests, request_by_node, path, |_| RequestKind::Leaf);
            }
            return park(work, requests, request_by_node, path, move |node_idx| {
                RequestKind::Expand {
                    node_idx,
                    candidate_features: candidate_features.clone(),
                }
            });
        }

        if !node.ready_to_select() {
            // Expanded but priors are still pending (another sim parked here in
            // this wave): share its request.
            let mut row = Vec::new();
            let candidate_features = arena[node_idx]
                .children
                .iter()
                .map(|child| {
                    action_features::encode_move_into(work, &child.mv, &mut row);
                    row.clone()
                })
                .collect();
            return park(work, requests, request_by_node, path, move |node_idx| {
                RequestKind::Expand {
                    node_idx,
                    candidate_features,
                }
            });
        }

        // Select the child maximizing the mover's OWN Q + PUCT.
        let pid = arena[node_idx].player;
        let child_slot = select_child(arena, node_idx, pid, cfg);
        let child = &arena[node_idx].children[child_slot];
        let mv = child.mv.clone();
        let child_node = arena[node_idx].children[child_slot].node;

        // The tree outlives one determinization. A node's children were
        // enumerated against the mover's hand *at expansion time*, but
        // `determinize` re-samples opponent hands every simulation, and
        // `ResolvedMove` stores a hand index rather than a card identity.
        // Detect the two ways that can go wrong:
        //   * `failed_applies`  - the rules rejected the stale index outright
        //     (`execute_build` re-validates the card).
        //   * `rewritten_applies` - the index still bounds-checks but now names
        //     a *different* card, so the simulation silently discards a card
        //     the enumerated move did not choose (network/develop/sell/loan/
        //     pass only call `require_card_index`). This is the common case and
        //     the one that biases value estimates.
        if let Some(declared) = &child.declared_card {
            if declared_card(&work.players[pid].hand, &mv).as_ref() != Some(declared) {
                *rewritten_applies += 1;
            }
        }

        if crate::rules::apply_move(work, &mv).is_err() {
            // A child is executable only in the determinization that created
            // it: `ResolvedMove` stores a hand index (`card_index`), and a
            // later simulation re-samples the mover's hand, so the stored
            // index can name a different (or no longer legal) card. Rules that
            // re-validate the card (`execute_build`) reject it; rules that only
            // bound-check (`require_card_index`) silently consume a wrong card.
            // Until nodes store re-resolvable semantic moves, count these and
            // end the line as a leaf so the search stays robust.
            *failed_applies += 1;
            return park(work, requests, request_by_node, path, |_| RequestKind::Leaf);
        }
        let tr = advance_turn(work);
        handle_turn_result(work, tr);
        arena[child_node].player = work.current_player_id();
        node_idx = child_node;
        path.push(node_idx);
        depth += 1;
    }
}

/// Park a request for `node_idx`, de-duplicating by node (multiple sims may
/// share one evaluation). The `kind` closure is invoked only when a new request
/// is actually created.
///
/// **Approximation (intentional):** sims that share a request also share the
/// first-parker's `work` state for encoding, so the node is evaluated in ONE
/// determinized world even though each sim drew its own. Strictly this biases
/// the value toward the first world; it is the standard batched-ISMCTS trade
/// for avoiding one net call per sim.
fn park(
    work: &mut GameState,
    requests: &mut Vec<Request>,
    request_by_node: &mut std::collections::HashMap<usize, usize>,
    path: Vec<usize>,
    kind: impl FnOnce(usize) -> RequestKind,
) -> ParkedOutcome {
    let node_idx = *path.last().expect("path is never empty");
    let idx = if let Some(&i) = request_by_node.get(&node_idx) {
        i
    } else {
        let i = requests.len();
        requests.push(Request {
            kind: kind(node_idx),
            state: work.clone(),
        });
        request_by_node.insert(node_idx, i);
        i
    };
    ParkedOutcome::Net {
        path,
        request_idx: idx,
    }
}

/// The card a single-card move would discard, read from `hand` at the move's
/// stored index. `None` for Scout, which selects three cards at once.
fn declared_card(hand: &[Card], mv: &ResolvedMove) -> Option<Card> {
    let index = match mv {
        ResolvedMove::Build { card_index, .. }
        | ResolvedMove::Network { card_index, .. }
        | ResolvedMove::NetworkDouble { card_index, .. }
        | ResolvedMove::Develop { card_index, .. }
        | ResolvedMove::Sell { card_index, .. }
        | ResolvedMove::Loan { card_index }
        | ResolvedMove::Pass { card_index } => *card_index,
        ResolvedMove::Scout { .. } => return None,
    };
    hand.get(index).cloned()
}

fn select_child(arena: &[Node], node_idx: usize, pid: usize, cfg: &NnMctsConfig) -> usize {
    let node = &arena[node_idx];
    let parent_visits = node.visits.max(1) as f64;
    // First-play urgency. Terminal utility is zero-mean across seats, so 0 is
    // already a neutral guess for an unvisited child; FPU instead starts it at
    // the parent's value, a strictly stronger prior whenever the parent sits
    // away from the table average.
    let fpu = if cfg.fpu { node.q(pid) - cfg.fpu_reduction } else { 0.0 };
    let mut best: Option<(usize, f64)> = None;
    for (i, child) in node.children.iter().enumerate() {
        let cn = &arena[child.node];
        // n = 0 for unvisited children (the standard PUCT denominator), so a
        // child that has never been expanded gets the full exploration bonus.
        let nv = cn.visits as f64;
        let q = if cn.visits == 0 { fpu } else { cn.q(pid) };
        let explore = cfg.c_puct * node.prior[i] * parent_visits.sqrt() / (1.0 + nv);
        let uct = q + explore;
        if best.map_or(true, |(_, b)| uct > b) {
            best = Some((i, uct));
        }
    }
    best.map(|(i, _)| i).unwrap_or(0)
}

/// Install a node's priors, optionally keeping only the `top_k` highest-prior
/// children (`reprioritize`). Pruned children keep their original
/// `candidate_id`, so visit counts still map back to the same canonical move.
fn apply_priors(arena: &mut Vec<Node>, node_idx: usize, priors: Vec<f64>, top_k: usize) {
    if top_k == 0 || priors.len() <= top_k {
        arena[node_idx].prior = priors;
        return;
    }
    let children = std::mem::take(&mut arena[node_idx].children);
    let mut ranked: Vec<(usize, Child)> = children.into_iter().enumerate().collect();
    ranked.sort_by(|(ia, _), (ib, _)| {
        priors[*ib]
            .partial_cmp(&priors[*ia])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    ranked.truncate(top_k);
    // Restore the engine's candidate order so ties and the returned child list
    // stay deterministic regardless of the sort's instability.
    ranked.sort_by_key(|(index, _)| *index);
    let kept: Vec<f64> = ranked.iter().map(|(index, _)| priors[*index]).collect();
    let total: f64 = kept.iter().sum();
    arena[node_idx].prior = if total > 0.0 {
        kept.iter().map(|p| p / total).collect()
    } else {
        vec![1.0 / kept.len() as f64; kept.len()]
    };
    arena[node_idx].children = ranked.into_iter().map(|(_, child)| child).collect();
}

fn add_value(arena: &mut Vec<Node>, path: &[usize], value: &[f64]) {
    for &n in path {
        for (p, v) in value.iter().enumerate() {
            arena[n].value_sum[p] += v;
        }
    }
}

/// Per-seat terminal utility: `(vp - table_mean_vp) / VP_SCALE`.
///
/// Zero-mean across seats and measured in VP, so sibling moves differ by real
/// score margin and "an unvisited child is worth 0" is a neutral assumption
/// rather than a claim that it is the worst child.
fn terminal_value(state: &GameState, n_players: usize) -> Vec<f64> {
    let n = n_players.max(1).min(state.players.len());
    let scores: Vec<f64> = (0..n).map(|p| state.players[p].vp as f64).collect();
    let mean = scores.iter().sum::<f64>() / n as f64;
    let scale = crate::bridge::VP_SCALE as f64;
    scores.iter().map(|s| (s - mean) / scale).collect()
}

/// Clamp a raw network value onto the terminal utility scale. An untrained or
/// early value head is an unconstrained linear layer and can emit values far
/// outside the observed range; PUCT compares `value_sum / visits` against
/// exploration terms, so an out-of-range Q destabilizes child selection.
/// `VALUE_LIMIT` is well above the reachable spread (±3 VP-margin units at
/// VP_SCALE = 50), so clamping only ever catches blow-ups. NaN maps to neutral.
fn search_value(raw: f32) -> f64 {
    let v = raw as f64;
    if v.is_nan() { 0.0 } else { v.clamp(-VALUE_LIMIT, VALUE_LIMIT) }
}

const VALUE_LIMIT: f64 = 4.0;

fn apply_dirichlet_noise(
    node: &mut Node,
    alpha: f64,
    weight: f64,
    rng: &mut rand_chacha::ChaCha12Rng,
) {
    use rand_distr::Distribution;
    let n = node.prior.len();
    if n == 0 {
        return;
    }
    let d = match rand_distr::multi::Dirichlet::new(&vec![alpha; n]) {
        Ok(d) => d,
        Err(_) => return,
    };
    let sample: Vec<f64> = d.sample(rng);
    for (i, s) in sample.iter().enumerate() {
        node.prior[i] = weight * s + (1.0 - weight) * node.prior[i];
    }
}

// ---------------------------------------------------------------------------
// Batched network inference (Python callback)
// ---------------------------------------------------------------------------

/// Encode every pending request (ONE row each, from the state's current player
/// perspective) and ask the Python `net_fn` for (candidate_logits (B,max_N),
/// values (B,4)); split results back per request.
///
/// Arrays cross into Python as flat buffers reshaped into numpy matrices —
/// one copy each, never per-element Python boxing.
fn flush_net(
    py: Python<'_>,
    net_fn: &Py<PyAny>,
    requests: &[Request],
) -> PyResult<Vec<BatchResult>> {
    let n_rows = requests.len();
    let cell_len = crate::encode::BOARD_CELLS * crate::encode::F_CELL;
    let link_len = crate::encode::LINK_CELLS * crate::encode::F_LINK;
    let merchant_len = crate::encode::MERCHANT_COUNT * crate::encode::F_MERCHANT;
    let seat_len = crate::encode::SEAT_COUNT * crate::encode::F_SEAT;
    let mut cells: Vec<f32> = Vec::with_capacity(n_rows * cell_len);
    let mut links: Vec<f32> = Vec::with_capacity(n_rows * link_len);
    let mut merchants: Vec<f32> = Vec::with_capacity(n_rows * merchant_len);
    let mut seats: Vec<f32> = Vec::with_capacity(n_rows * seat_len);
    let mut globals: Vec<f32> = Vec::with_capacity(n_rows * crate::encode::F_GLOBAL);
    for req in requests {
        let pid = req.state.current_player_id();
        let t = crate::encode::state_tokens(&req.state, pid);
        cells.extend_from_slice(&t.cells);
        links.extend_from_slice(&t.links);
        merchants.extend_from_slice(&t.merchants);
        seats.extend_from_slice(&t.seats);
        globals.extend_from_slice(&t.global);
    }

    let max_candidates = requests
        .iter()
        .filter_map(|req| match &req.kind {
            RequestKind::Expand {
                candidate_features, ..
            } => Some(candidate_features.len()),
            RequestKind::Leaf => None,
        })
        .max()
        .unwrap_or(1);
    let dim = action_features::ACTION_FEATURE_DIM;
    let mut candidate_rows = vec![0.0f32; n_rows * max_candidates * dim];
    let mut candidate_masks = vec![0.0f32; n_rows * max_candidates];
    for (row, req) in requests.iter().enumerate() {
        match &req.kind {
            RequestKind::Expand {
                candidate_features, ..
            } => {
                for (i, features) in candidate_features.iter().enumerate() {
                    let begin = row * max_candidates * dim + i * dim;
                    candidate_rows[begin..begin + dim].copy_from_slice(features);
                    candidate_masks[row * max_candidates + i] = 1.0;
                }
            }
            // Value-only leaves still receive one inert candidate because the
            // Python policy API requires every padded row to have one entry.
            RequestKind::Leaf => candidate_masks[row * max_candidates] = 1.0,
        }
    }

    let cells_arr = PyArray1::from_vec(py, cells).reshape((n_rows, cell_len))?;
    let links_arr = PyArray1::from_vec(py, links).reshape((n_rows, link_len))?;
    let merchants_arr = PyArray1::from_vec(py, merchants).reshape((n_rows, merchant_len))?;
    let seats_arr = PyArray1::from_vec(py, seats).reshape((n_rows, seat_len))?;
    let global_arr =
        PyArray1::from_vec(py, globals).reshape((n_rows, crate::encode::F_GLOBAL))?;
    let candidates_arr =
        PyArray1::from_vec(py, candidate_rows).reshape((n_rows, max_candidates * dim))?;
    let candidate_mask_arr =
        PyArray1::from_vec(py, candidate_masks).reshape((n_rows, max_candidates))?;

    let out = net_fn.call1(
        py,
        (
            cells_arr,
            links_arr,
            merchants_arr,
            seats_arr,
            global_arr,
            candidates_arr,
            candidate_mask_arr,
        ),
    )?;
    let (candidate_arr, values_arr): (Bound<PyArray2<f32>>, Bound<PyArray2<f32>>) =
        out.bind(py).extract()?;
    let candidate_ro = candidate_arr.readonly();
    let candidate_logits = candidate_ro
        .as_slice()
        .map_err(|_| pyo3::exceptions::PyValueError::new_err("candidate array not contiguous"))?;
    let values_ro = values_arr.readonly();
    let values = values_ro
        .as_slice()
        .map_err(|_| pyo3::exceptions::PyValueError::new_err("values array not contiguous"))?;

    let mut results = Vec::with_capacity(requests.len());
    for (ri, req) in requests.iter().enumerate() {
        let r0 = ri * 4;
        // Value = terminal utility per seat (VP margin over the table mean,
        // higher = better), the same scale as `terminal_value` backups.
        //
        // The value head is an unconstrained linear layer, so `search_value`
        // clamps blow-ups before they reach PUCT's value/exploration
        // comparison.
        let value: Vec<f64> = (0..MAX_PLAYERS).map(|p| search_value(values[r0 + p])).collect();
        let priors = match &req.kind {
            RequestKind::Expand {
                candidate_features, ..
            } => {
                let start = ri * max_candidates;
                Some(softmax(
                    &candidate_logits[start..start + candidate_features.len()],
                ))
            }
            RequestKind::Leaf => None,
        };
        results.push(BatchResult { value, priors });
    }
    Ok(results)
}

/// Softmax over a pre-merged list of legal-slot logits (already masked by
/// construction: only legal slots are in the vector).
fn softmax(logits: &[f32]) -> Vec<f64> {
    let mut max = f32::NEG_INFINITY;
    for &l in logits {
        max = max.max(l);
    }
    let mut exps = Vec::with_capacity(logits.len());
    let mut sum = 0.0f32;
    for &l in logits {
        let e = (l - max).exp();
        exps.push(e);
        sum += e;
    }
    let denom = if sum <= 0.0 { 1.0 } else { sum };
    exps.iter().map(|e| (*e / denom) as f64).collect()
}

#[cfg(test)]
mod tests {
    use super::{search_value, terminal_value};
    use crate::state::GameState;
    use rand_chacha::rand_core::SeedableRng;
    use rand_chacha::ChaCha12Rng;

    #[test]
    fn terminal_value_is_a_zero_mean_vp_margin() {
        let mut state = GameState::new(ChaCha12Rng::seed_from_u64(3), 4);
        for (p, vp) in [110u16, 100, 60, 100].iter().enumerate() {
            state.players[p].vp = *vp;
        }
        let value = terminal_value(&state, 4);
        let scale = crate::bridge::VP_SCALE as f64;
        // Table mean is 92.5; margins are +17.5 / +7.5 / -32.5 / +7.5.
        let expected = [17.5 / scale, 7.5 / scale, -32.5 / scale, 7.5 / scale];
        for (got, want) in value.iter().zip(expected.iter()) {
            assert!((got - want).abs() < 1e-9, "got {got}, want {want}");
        }
        assert!(value.iter().sum::<f64>().abs() < 1e-9);
    }

    #[test]
    fn search_value_is_clamped_to_the_terminal_scale() {
        assert_eq!(search_value(-3.0), -3.0);
        assert_eq!(search_value(0.0), 0.0);
        assert_eq!(search_value(-0.75), -0.75);
        assert_eq!(search_value(7.0), 4.0);
        assert_eq!(search_value(f32::NAN), 0.0);
        assert_eq!(search_value(f32::INFINITY), 4.0);
        assert_eq!(search_value(f32::NEG_INFINITY), -4.0);
    }
}
