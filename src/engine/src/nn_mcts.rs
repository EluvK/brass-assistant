//! Network-guided ISMCTS (the AlphaZero search) running entirely in Rust.
//!
//! The tree lives in Rust as an arena, keyed by **policy slot** (resource/card
//! choices already folded by `rules::legal_slot_moves`). Per simulation the
//! root is determinized (opponent hands sampled, reusing `mcts_ai::determinize`).
//!
//! Priors come from the network's **branched policy head**: one row per request
//! (the moving player's perspective) returning `type[7], goal[1316]`, and the
//! prior logit for slot s is `type[slot_type(s)] + goal[s]`, then masked
//! softmax over the legal slots. Leaf values come from the value head's
//! 4-player vector directly (single perspective, no 4x encode).
//!
//! Network inference is BATCHED: simulations park at expand/leaf points, their
//! states are encoded in Rust and handed to a Python callback in waves; results
//! are applied before the next wave. This removes the per-sim Python/PyO3
//! orchestration that capped the previous Python MCTS at ~4 ms/sim.

use crate::engine::{advance_turn, end_canal_era, end_game, TurnResult};
use crate::move_codec;
use crate::rules::{legal_slot_moves, Move};
use crate::state::GameState;
use numpy::PyArray2;
use numpy::PyArrayMethods;
use pyo3::prelude::*;
use pyo3::types::PyAny;
use pyo3::Py;
use rand::rngs::StdRng;
use rand::Rng;

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
}

impl Default for NnMctsConfig {
    fn default() -> Self {
        NnMctsConfig {
            c_puct: 2.5,
            max_depth: 10,
            dirichlet_alpha: 0.3,
            dirichlet_weight: 0.15,
            batch_size: 64,
        }
    }
}

/// Root children as (slot, canonical, visits), best-first.
#[derive(Debug)]
pub struct NnSearchResult {
    pub best_canonical: Option<String>,
    pub children: Vec<(usize, String, u32)>,
    pub legal_slots: Vec<usize>,
}

// ---------------------------------------------------------------------------
// Tree
// ---------------------------------------------------------------------------

struct Child {
    slot: usize,
    mv: Move,
    node: usize,
}

struct Node {
    player: usize,
    visits: u32,
    value_sum: [f64; MAX_PLAYERS],
    children: Vec<Child>,
    prior: Vec<f64>,
    legal_slots: Vec<usize>,
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
            legal_slots: Vec::new(),
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
    /// Need policy priors (masked over `legal_slots`) + the 4-player value.
    Expand {
        node_idx: usize,
        legal_slots: Vec<usize>,
    },
    /// Need the 4-player value only (depth cap / no legal moves).
    Leaf,
}

struct Request {
    kind: RequestKind,
    state: GameState<StdRng>,
}

enum ParkedOutcome {
    Terminal { path: Vec<usize>, vps: Vec<i32> },
    Net { path: Vec<usize>, request_idx: usize },
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
    state: &GameState<StdRng>,
    cfg: &NnMctsConfig,
    sims: usize,
    add_root_noise: bool,
    net_fn: &Py<PyAny>,
    py: Python<'_>,
) -> PyResult<NnSearchResult> {
    let root_pid = state.current_player_id();
    let n_players = state.player_count();

    let mut arena: Vec<Node> = vec![Node::new(root_pid)];

    // Independent simulation RNG derived from (but not consuming) state.rng.
    let mut sim_rng = state.rng.clone();
    let mut sims_left = sims;

    let mut requests: Vec<Request> = Vec::new();
    let mut request_by_node: std::collections::HashMap<usize, usize> = std::collections::HashMap::new();
    let mut parked: Vec<ParkedOutcome> = Vec::new();

    // Prime the root: expand it and fetch priors so the very first simulation
    // can descend past it. Without this, all sims of the first (and only, for
    // small sims budgets) wave would park at the root and no child would ever
    // be visited.
    {
        let mut work = crate::mcts_ai::determinize(state, &mut sim_rng);
        work.rng = sim_rng.clone();
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
            let _ = sim_rng.gen::<u64>(); // vary determinization per simulation
            let mut work = crate::mcts_ai::determinize(state, &mut sim_rng);
            work.rng = sim_rng.clone();

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
        .map(|c| (c.slot, move_codec::encode(&c.mv), arena[c.node].visits))
        .collect();
    children.sort_by(|a, b| b.2.cmp(&a.2));
    let best_slot = children.first().filter(|(_, _, v)| *v > 0).map(|(s, _, _)| *s);
    let best_canonical = best_slot.and_then(|slot| {
        root.children
            .iter()
            .find(|c| c.slot == slot)
            .map(|c| move_codec::encode(&c.mv))
    });

    Ok(NnSearchResult {
        best_canonical,
        children,
        legal_slots: root.legal_slots.clone(),
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
    work: &mut GameState<StdRng>,
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
            // Expand: generate slot-level legal moves, create one child per slot.
            let slots = legal_slot_moves(work);
            let mut children: Vec<Child> = Vec::new();
            let mut legal_slots: Vec<usize> = Vec::new();
            for sm in slots {
                if legal_slots.contains(&sm.slot) {
                    continue; // fold duplicate-slot entries (e.g. sell routes)
                }
                legal_slots.push(sm.slot);
                let child_idx = arena.len();
                arena.push(Node::new(0));
                children.push(Child {
                    slot: sm.slot,
                    mv: sm.mv,
                    node: child_idx,
                });
            }
            arena[node_idx].children = children;
            arena[node_idx].legal_slots = legal_slots.clone();
            arena[node_idx].expanded = true;

            if legal_slots.is_empty() {
                // No legal moves: treat as a leaf (network value).
                return park(work, requests, request_by_node, path, |_| {
                    RequestKind::Leaf
                });
            }
            return park(work, requests, request_by_node, path, move |node_idx| {
                RequestKind::Expand {
                    node_idx,
                    legal_slots: legal_slots.clone(),
                }
            });
        }

        if !node.ready_to_select() {
            // Expanded but priors are still pending (another sim parked here in
            // this wave): share its request.
            let legal_slots = arena[node_idx].legal_slots.clone();
            return park(work, requests, request_by_node, path, move |node_idx| {
                RequestKind::Expand {
                    node_idx,
                    legal_slots,
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
        match tr {
            TurnResult::Continue => {}
            TurnResult::EndCanalEra => end_canal_era(work),
            TurnResult::EndGame => end_game(work),
        }
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
    work: &mut GameState<StdRng>,
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
    ParkedOutcome::Net { path, request_idx: idx }
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

fn terminal_value(vps: Vec<i32>, n_players: usize) -> Vec<f64> {
    let vals: Vec<f64> = vps.iter().map(|&v| v as f64).collect();
    let mean = vals.iter().sum::<f64>() / vals.len() as f64;
    let var = vals.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / vals.len() as f64;
    let std = var.sqrt();
    if std < 1e-6 {
        return vec![0.0; n_players];
    }
    vals.iter().map(|v| (v - mean) / std).collect()
}

fn apply_dirichlet_noise(node: &mut Node, alpha: f64, weight: f64, rng: &mut impl Rng) {
    use rand_distr::Distribution;
    let n = node.prior.len();
    if n == 0 {
        return;
    }
    let d = match rand_distr::Dirichlet::new_with_size(alpha, n) {
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
/// perspective) and ask the Python `net_fn` for (type (B,7), goal (B,P),
/// value (B,4)); split results back per request.
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

    let board_arr = PyArray2::from_vec2(py, &boards)?;
    let links_arr = PyArray2::from_vec2(py, &links)?;
    let global_arr = PyArray2::from_vec2(py, &globals)?;
    let own_arr = PyArray2::from_vec2(py, &own_hands)?;
    let opp_arr = PyArray2::from_vec2(py, &opp_hands)?;

    let out = net_fn.call1(py, (board_arr, links_arr, global_arr, own_arr, opp_arr))?;
    let (type_arr, goal_arr, values_arr): (
        Bound<PyArray2<f32>>,
        Bound<PyArray2<f32>>,
        Bound<PyArray2<f32>>,
    ) = out.bind(py).extract()?;
    let type_ro = type_arr.readonly();
    let type_vals = type_ro
        .as_slice()
        .map_err(|_| pyo3::exceptions::PyValueError::new_err("type array not contiguous"))?;
    let goal_ro = goal_arr.readonly();
    let goal_vals = goal_ro
        .as_slice()
        .map_err(|_| pyo3::exceptions::PyValueError::new_err("goal array not contiguous"))?;
    let values_ro = values_arr.readonly();
    let values = values_ro
        .as_slice()
        .map_err(|_| pyo3::exceptions::PyValueError::new_err("values array not contiguous"))?;

    let n_type = crate::policy::ACTION_TYPE_COUNT; // 7
    let psize = crate::policy::policy_table_size();
    let mut results = Vec::with_capacity(requests.len());
    for (ri, req) in requests.iter().enumerate() {
        let r0 = ri * 4;
        // Value head predicts the 4-player z vector directly from one perspective.
        let value: Vec<f64> = (0..MAX_PLAYERS).map(|p| values[r0 + p] as f64).collect();
        let priors = match &req.kind {
            RequestKind::Expand { legal_slots, .. } => {
                // Merged branched prior: logit(s) = type[t(s)] + goal(s).
                let tr0 = ri * n_type;
                let gr0 = ri * psize;
                let mut merged: Vec<f32> = Vec::with_capacity(legal_slots.len());
                for &s in legal_slots {
                    let t = crate::policy::slot_type(s);
                    merged.push(type_vals[tr0 + t] + goal_vals[gr0 + s]);
                }
                Some(softmax(&merged))
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
