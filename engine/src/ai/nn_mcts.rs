//! Network-guided ISMCTS (AlphaZero-style search) with batched PyO3 inference.
//!
//! The tree lives in Rust as an arena. Its children are concrete legal moves,
//! each assigned a node-local candidate id for policy outputs. Per simulation
//! the root is determinized by sampling opponent hands via
//! `mcts_ai::determinize`.
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
use crate::state::GameState;
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
}

impl Default for NnMctsConfig {
    fn default() -> Self {
        NnMctsConfig {
            c_puct: 2.5,
            max_depth: 10,
            dirichlet_alpha: 0.3,
            dirichlet_weight: 0.15,
            batch_size: 64,
            candidate_k: 4,
        }
    }
}

/// Root children as (node-local candidate id, canonical, visits), best-first.
#[derive(Debug)]
pub struct NnSearchResult {
    pub best_canonical: Option<String>,
    pub children: Vec<(usize, String, u32)>,
    pub legal_candidate_ids: Vec<usize>,
}

// ---------------------------------------------------------------------------
// Tree
// ---------------------------------------------------------------------------

struct Child {
    candidate_id: usize,
    mv: ResolvedMove,
    node: usize,
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
        vps: Vec<i32>,
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

    // Prime the root: expand it and fetch priors so the very first simulation
    // can descend past it. Without this, all sims of the first (and only, for
    // small sims budgets) wave would park at the root and no child would ever
    // be visited.
    {
        let mut work = crate::mcts_ai::determinize(state, &mut sim_rng);
        let mut root_reqs = Vec::new();
        let mut root_by_node = std::collections::HashMap::new();
        match descend(
            &mut work,
            &mut arena,
            cfg,
            &mut root_reqs,
            &mut root_by_node,
            false, // priming does not count as a simulation visit
        ) {
            ParkedOutcome::Net { .. } => {
                let results = flush_net(py, net_fn, &root_reqs)?;
                for (req, res) in root_reqs.iter().zip(results.iter()) {
                    if let RequestKind::Expand { node_idx, .. } = req.kind {
                        if let Some(p) = &res.priors {
                            arena[node_idx].prior = p.clone();
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
            let mut work = crate::mcts_ai::determinize(state, &mut sim_rng);
            match descend(
                &mut work,
                &mut arena,
                cfg,
                &mut requests,
                &mut request_by_node,
                true,
            ) {
                ParkedOutcome::Terminal { path, vps } => {
                    let value = terminal_value(vps, n_players);
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
                        arena[node_idx].prior = p.clone();
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
                vps: work.players.iter().map(|p| p.vp as i32).collect(),
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
            for (action_id, mv) in moves.into_iter().enumerate() {
                legal_candidate_ids.push(action_id);
                let child_idx = arena.len();
                arena.push(Node::new(0));
                children.push(Child {
                    candidate_id: action_id,
                    mv,
                    node: child_idx,
                });
                candidate_features.push(action_features::encode_move(
                    work,
                    &children.last().unwrap().mv,
                ));
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
            let candidate_features = arena[node_idx]
                .children
                .iter()
                .map(|child| action_features::encode_move(work, &child.mv))
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
        let child_slot = select_child(arena, node_idx, pid, cfg.c_puct);
        let mv = arena[node_idx].children[child_slot].mv.clone();
        let child_node = arena[node_idx].children[child_slot].node;

        if crate::rules::apply_move(work, &mv).is_err() {
            // Should not happen (slot-level moves are executable by construction);
            // treat the line as a draw-ish leaf to keep the search robust.
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

fn select_child(arena: &[Node], node_idx: usize, pid: usize, c_puct: f64) -> usize {
    let node = &arena[node_idx];
    let parent_visits = node.visits.max(1) as f64;
    let mut best: Option<(usize, f64)> = None;
    for (i, child) in node.children.iter().enumerate() {
        let cn = &arena[child.node];
        let nv = cn.visits.max(1) as f64;
        let explore = c_puct * node.prior[i] * parent_visits.sqrt() / (1.0 + nv);
        let uct = cn.q(pid) + explore;
        if best.map_or(true, |(_, b)| uct > b) {
            best = Some((i, uct));
        }
    }
    best.map(|(i, _)| i).unwrap_or(0)
}

fn add_value(arena: &mut Vec<Node>, path: &[usize], value: &[f64]) {
    for &n in path {
        for (p, v) in value.iter().enumerate() {
            arena[n].value_sum[p] += v;
        }
    }
}

/// Per-seat terminal search value: `1 - normalized final rank` (ties share the
/// average rank), matching the Python rank-head target scale (rank/n, MSE) so
/// network leaf values and terminal backups stay in one unit system.
fn terminal_value(vps: Vec<i32>, n_players: usize) -> Vec<f64> {
    let n = vps.len().max(1);
    let mut order: Vec<usize> = (0..n).collect();
    // Rank 1 belongs to the most VP (the winner), so `1 - rank/n` is highest
    // for the best seat and stays aligned with the winner head.
    order.sort_by_key(|&i| std::cmp::Reverse(vps[i]));
    let mut rank = vec![0.0f64; n];
    let mut i = 0;
    while i < n {
        let mut j = i;
        while j + 1 < n && vps[order[j + 1]] == vps[order[i]] {
            j += 1;
        }
        let avg = (i + j) as f64 / 2.0 + 1.0; // average 1-based rank of the tie group
        for &idx in &order[i..=j] {
            rank[idx] = avg;
        }
        i = j + 1;
    }
    (0..n_players)
        .map(|p| 1.0 - rank[p] / n as f64)
        .collect()
}

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
fn flush_net(
    py: Python<'_>,
    net_fn: &Py<PyAny>,
    requests: &[Request],
) -> PyResult<Vec<BatchResult>> {
    let n_rows = requests.len();
    let mut boards: Vec<Vec<f32>> = Vec::with_capacity(n_rows);
    let mut links: Vec<Vec<f32>> = Vec::with_capacity(n_rows);
    let mut globals: Vec<Vec<f32>> = Vec::with_capacity(n_rows);
    let mut own_hands: Vec<Vec<f32>> = Vec::with_capacity(n_rows);
    let mut opp_hands: Vec<Vec<f32>> = Vec::with_capacity(n_rows);
    for req in requests {
        let pid = req.state.current_player_id();
        let t = crate::encode::state_to_tensor(&req.state, pid);
        boards.push(t.board);
        links.push(t.links);
        globals.push(t.global);
        own_hands.push(t.own_hand);
        opp_hands.push(t.opp_hands);
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
    let mut candidate_rows =
        vec![vec![0.0; max_candidates * action_features::ACTION_FEATURE_DIM]; n_rows];
    let mut candidate_masks = vec![vec![0.0; max_candidates]; n_rows];
    for (row, req) in requests.iter().enumerate() {
        match &req.kind {
            RequestKind::Expand {
                candidate_features, ..
            } => {
                for (i, features) in candidate_features.iter().enumerate() {
                    let begin = i * action_features::ACTION_FEATURE_DIM;
                    candidate_rows[row][begin..begin + action_features::ACTION_FEATURE_DIM]
                        .copy_from_slice(features);
                    candidate_masks[row][i] = 1.0;
                }
            }
            // Value-only leaves still receive one inert candidate because the
            // Python policy API requires every padded row to have one entry.
            RequestKind::Leaf => candidate_masks[row][0] = 1.0,
        }
    }

    let board_arr = PyArray2::from_vec2(py, &boards)?;
    let links_arr = PyArray2::from_vec2(py, &links)?;
    let global_arr = PyArray2::from_vec2(py, &globals)?;
    let own_arr = PyArray2::from_vec2(py, &own_hands)?;
    let opp_arr = PyArray2::from_vec2(py, &opp_hands)?;
    let candidates_arr = PyArray2::from_vec2(py, &candidate_rows)?;
    let candidate_mask_arr = PyArray2::from_vec2(py, &candidate_masks)?;

    let out = net_fn.call1(
        py,
        (
            board_arr,
            links_arr,
            global_arr,
            own_arr,
            opp_arr,
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
        // Value = 1 - normalized rank per seat (higher = better), the same
        // scale as `terminal_value` backups.
        let value: Vec<f64> = (0..MAX_PLAYERS).map(|p| values[r0 + p] as f64).collect();
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
