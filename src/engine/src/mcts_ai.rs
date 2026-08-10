//! ISMCTS with determinization + 1-ply heuristic prior & leaf evaluation.
//!
//! Design (per project decision):
//! - Each simulation determinizes ONCE at the root: opponent hands are sampled
//!   from the hidden pool (`deck_composition` minus our known hand), then a
//!   shallow 3-5 ply search runs in that sampled world.
//! - Node selection uses PUCT with the 1-ply heuristic's candidate scores as
//!   priors (softmax over `candidate_actions` scores).
//! - Leaf evaluation uses the 1-ply position estimator (`evaluate_position`)
//!   for EACH player; nodes store the full MaxN value vector. The player to
//!   move at a node selects the child maximizing their OWN value (Brass is
//!   non-zero-sum: opponents maximize their own VP, not a coalition vs us).

use crate::data::IndustryType;
use crate::engine::{advance_turn, end_canal_era, end_game, TurnResult};
use crate::heuristic_ai::{self, Decision};
use crate::map::Loc;
use crate::rules::{apply_move, Move};
use crate::state::{deck_composition, Card, GameState};
use rand::seq::SliceRandom;
use rand::Rng;

/// Leaf-evaluation strategy (pluggable, per project decision "1-ply first,
/// add a switch later").
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LeafEval {
    /// Position snapshot for each player (`evaluate_position`). Fast but flat:
    /// shallow leaf values often can't tell good actions apart.
    OnePly,
    /// Run a 2-ply lookahead at the leaf and use the best resulting move score
    /// per player. Captures same-turn combos (Build->Sell, Network->Build);
    /// slower but far more discriminating.
    TwoPly,
}

pub struct MctsConfig {
    /// Number of simulations (determinizations + shallow searches) per decision.
    pub simulations: usize,
    /// Max plies searched below the root before leaf evaluation.
    pub max_depth: usize,
    /// PUCT exploration constant (scaled to the leaf-value magnitude).
    pub c_puct: f64,
    /// Softmax temperature for converting 1-ply scores into priors.
    pub prior_temp: f64,
    /// Candidates per action type (build/network get up to K; others 1).
    pub k_candidates: usize,
    /// Leaf evaluation strategy.
    pub leaf_eval: LeafEval,
}

impl Default for MctsConfig {
    fn default() -> Self {
        MctsConfig {
            simulations: 5000,
            max_depth: 6,
            c_puct: 1.0,
            prior_temp: 2.0,
            k_candidates: 3,
            leaf_eval: LeafEval::OnePly,
        }
    }
}

pub const MAX_PLAYERS: usize = 4;

// ---------------------------------------------------------------------------
// Move identity (ignores card choice, which is irrelevant for tree matching)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum MoveKey {
    Build {
        loc: Loc,
        slot: usize,
        ind: IndustryType,
        coal: Vec<(crate::graph::CoalSourceKind, usize)>,
        iron: Vec<(usize, bool)>,
    },
    Network {
        conn: usize,
        coal: Option<crate::graph::CoalSource>,
    },
    NetworkDouble {
        c1: usize,
        c2: usize,
        coal1: crate::graph::CoalSource,
        coal2: crate::graph::CoalSource,
        beer: crate::graph::BeerSource,
    },
    Develop {
        i1: IndustryType,
        i2: Option<IndustryType>,
        iron: Vec<(usize, bool)>,
    },
    ResolveFreeDevelop {
        i1: IndustryType,
        i2: Option<IndustryType>,
    },
    Sell {
        keys: Vec<usize>,
        merchant_indices: Vec<usize>,
        use_merchant_beer: Vec<bool>,
    },
    Loan,
    Scout {
        cards: [usize; 3],
    },
    Pass,
}

fn move_key(mv: &Move) -> MoveKey {
    match mv {
        Move::Build { loc, slot_index, ind, coal, iron, .. } => {
            let mut coal_ids: Vec<(crate::graph::CoalSourceKind, usize)> =
                coal.iter().map(|c| (c.kind, c.key)).collect();
            coal_ids.sort_unstable();
            let mut iron_ids: Vec<(usize, bool)> =
                iron.iter().map(|i| (i.key, i.free)).collect();
            iron_ids.sort_unstable();
            MoveKey::Build {
                loc: *loc,
                slot: *slot_index,
                ind: *ind,
                coal: coal_ids,
                iron: iron_ids,
            }
        }
        Move::Network { conn_id, coal, .. } => MoveKey::Network {
            conn: *conn_id,
            coal: *coal,
        },
        Move::NetworkDouble { conn1, conn2, coal1, coal2, beer, .. } => MoveKey::NetworkDouble {
            c1: *conn1,
            c2: *conn2,
            coal1: *coal1,
            coal2: *coal2,
            beer: *beer,
        },
        Move::Develop { ind1, ind2, iron, .. } => {
            let mut iron_ids: Vec<(usize, bool)> =
                iron.iter().map(|i| (i.key, i.free)).collect();
            iron_ids.sort_unstable();
            MoveKey::Develop {
                i1: *ind1,
                i2: *ind2,
                iron: iron_ids,
            }
        }
        Move::ResolveFreeDevelop { ind1, ind2 } => MoveKey::ResolveFreeDevelop {
            i1: *ind1,
            i2: *ind2,
        },
        Move::Sell {
            keys,
            merchant_indices,
            use_merchant_beer,
            ..
        } => {
            let mut k = keys.clone();
            let mut pairs: Vec<(usize, usize, bool)> = k
                .drain(..)
                .zip(merchant_indices.iter().copied())
                .zip(use_merchant_beer.iter().copied())
                .map(|((key, merchant_index), use_merchant)| (key, merchant_index, use_merchant))
                .collect();
            pairs.sort_unstable_by_key(|(key, _, _)| *key);
            let keys = pairs.iter().map(|(key, _, _)| *key).collect();
            let merchant_indices = pairs.iter().map(|(_, merchant_index, _)| *merchant_index).collect();
            let use_merchant_beer = pairs.iter().map(|(_, _, use_merchant)| *use_merchant).collect();
            MoveKey::Sell {
                keys,
                merchant_indices,
                use_merchant_beer,
            }
        }
        Move::Loan { .. } => MoveKey::Loan,
        Move::Scout { card_indices } => {
            let mut c = *card_indices;
            c.sort_unstable();
            MoveKey::Scout { cards: c }
        }
        Move::Pass { .. } => MoveKey::Pass,
    }
}

fn is_wild(c: &Card) -> bool {
    matches!(c, Card::WildLocation | Card::WildIndustry)
}

// ---------------------------------------------------------------------------
// Determinization
// ---------------------------------------------------------------------------

/// Sample opponent hands from the hidden pool, leaving our own hand intact.
/// The pool is the full era deck minus our known (non-wild) hand cards.
pub(crate) fn determinize<R: Rng + Clone>(state: &GameState<R>, rng: &mut R) -> GameState<R> {    let mut det = state.clone();
    let me = det.current_player_id();

    let mut pool = deck_composition(det.player_count());
    // Remove our known non-wild hand cards from the pool.
    let known: Vec<Card> = det.players[me]
        .hand
        .iter()
        .filter(|c| !is_wild(c))
        .cloned()
        .collect();
    for card in &known {
        if let Some(idx) = pool.iter().position(|c| c == card) {
            pool.swap_remove(idx);
        }
    }
    pool.shuffle(rng);

    // Rebuild each opponent hand: keep their wilds, re-sample non-wilds.
    for i in 0..det.player_count() {
        if i == me {
            continue;
        }
        let wilds: Vec<Card> = det.players[i]
            .hand
            .iter()
            .filter(|c| is_wild(c))
            .cloned()
            .collect();
        let need = det.players[i].hand.len() - wilds.len();
        let mut new_hand = wilds;
        for _ in 0..need {
            match pool.pop() {
                Some(c) => new_hand.push(c),
                None => break,
            }
        }
        det.players[i].hand = new_hand;
    }
    det.deck = pool;
    det
}

/// Test-only wrapper around `determinize`.
#[doc(hidden)]
pub fn determinize_for_test<R: Rng + Clone>(state: &GameState<R>, rng: &mut R, _cfg: &MctsConfig) -> GameState<R> {
    determinize(state, rng)
}

// ---------------------------------------------------------------------------
// Tree (MaxN: value_sum is a vector over players)
// ---------------------------------------------------------------------------

struct Node {
    /// Player to move at this node.
    player: usize,
    /// Index of the parent node (0 for the root).
    parent: usize,
    visits: u32,
    /// Accumulated leaf value per player (MaxN).
    value_sum: [f64; MAX_PLAYERS],
    children: Vec<Child>,
}

struct Child {
    mv: Move,
    node: usize,
}

// ---------------------------------------------------------------------------
// Search
// ---------------------------------------------------------------------------

/// Choose the current player's action by ISMCTS (determinized MCTS, MaxN).
pub fn choose_action_mcts<R: Rng + Clone>(state: &mut GameState<R>, cfg: &MctsConfig) -> Decision {
    let root_pid = state.current_player_id();
    let n_players = state.player_count();

    let mut arena: Vec<Node> = Vec::with_capacity(cfg.simulations / 2 + 16);
    arena.push(Node {
        player: root_pid,
        parent: 0,
        visits: 0,
        value_sum: [0.0; MAX_PLAYERS],
        children: Vec::new(),
    });

    // Independent simulation RNG derived from (but not consuming) state.rng.
    let mut sim_rng = state.rng.clone();
    let mut total_depth = 0u64;
    let mut apply_fail = 0u64;
    let mut empty_cand = 0u64;
    for _ in 0..cfg.simulations {
        let _ = sim_rng.r#gen::<u64>(); // vary determinization per simulation
        let mut work = determinize(state, &mut sim_rng);
        work.rng = sim_rng.clone(); // future draws during the sim use this rng

        let mut node_idx = 0usize;
        let mut depth = 0usize;

        // Descend until depth limit or no legal moves.
        while depth < cfg.max_depth {
            let player_to_move = arena[node_idx].player;
            let candidates = heuristic_ai::candidate_actions_k(&mut work, cfg.k_candidates);
            if candidates.is_empty() {
                empty_cand += 1;
                break;
            }
            // Softmax priors over candidate scores.
            let scores: Vec<f64> = candidates.iter().map(|d| d.score).collect();
            let max_s = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let exps: Vec<f64> = scores
                .iter()
                .map(|s| ((s - max_s) / cfg.prior_temp).exp())
                .collect();
            let sum: f64 = exps.iter().sum();

            // Select child by PUCT: the player to move maximizes their OWN value.
            let parent_visits = arena[node_idx].visits as f64;
            let (best_cand, _best_uct, _prior) = candidates
                .iter()
                .zip(exps.iter())
                .enumerate()
                .map(|(ci, (cand, e))| {
                    let key = move_key(&cand.mv);
                    let child_node = arena[node_idx]
                        .children
                        .iter()
                        .find(|ch| move_key(&ch.mv) == key)
                        .map(|ch| ch.node);
                    let (visits, q) = match child_node {
                        Some(cn) => {
                            let n = &arena[cn];
                            let nv = n.visits.max(1) as f64;
                            (nv, n.value_sum[player_to_move] / nv)
                        }
                        None => (0.0, 0.0),
                    };
                    let prior = e / sum;
                    let explore = cfg.c_puct * prior
                        * ((parent_visits + 1.0).ln() / (1.0 + visits)).sqrt();
                    let uct = q + explore;
                    (ci, uct, prior)
                })
                .max_by(|a, b| {
                    a.1.partial_cmp(&b.1)
                        .unwrap()
                        .then(a.2.partial_cmp(&b.2).unwrap())
                })
                .unwrap();

            let cand = &candidates[best_cand];
            let key = move_key(&cand.mv);

            // Ensure the child node exists in the arena.
            let child_node = if let Some(ch) = arena[node_idx]
                .children
                .iter()
                .find(|ch| move_key(&ch.mv) == key)
            {
                ch.node
            } else {
                let idx = arena.len();
                arena.push(Node {
                    player: 0, // filled after applying the move
                    parent: node_idx,
                    visits: 0,
                    value_sum: [0.0; MAX_PLAYERS],
                    children: Vec::new(),
                });
                arena[node_idx].children.push(Child {
                    mv: cand.mv.clone(),
                    node: idx,
                });
                idx
            };

            // Apply the chosen move and advance the turn.
            if apply_move(&mut work, &cand.mv).is_err() {
                apply_fail += 1;
                break;
            }
            let tr = advance_turn(&mut work);
            handle_turn_result(&mut work, tr);
            let next_player = work.current_player_id();

            arena[child_node].player = next_player;
            node_idx = child_node;
            depth += 1;
        }
        total_depth += depth as u64;

        // Leaf evaluation for EVERY player (MaxN vector).
        // - OnePly: position snapshot per player.
        // - TwoPly: root player uses 2-ply lookahead score (captures combos);
        //   opponents use the cheaper 1-ply snapshot.
        let mut leaf: [f64; MAX_PLAYERS] = [0.0; MAX_PLAYERS];
        for (p, v) in leaf.iter_mut().enumerate() {
            if p >= n_players {
                break;
            }
            if cfg.leaf_eval == LeafEval::TwoPly && p == root_pid {
                let d = crate::search_ai::choose_action_2ply(&mut work);
                *v = d.score;
            } else {
                *v = heuristic_ai::evaluate_position(&work, p);
            }
        }
        // Normalize to a small band so PUCT exploration can compete with q.
        // OnePly position values are ~10-70; 2-ply scores are ~-5..10. Use a
        // divisor per strategy so first-visit values don't permanently dominate.
        let div = if cfg.leaf_eval == LeafEval::TwoPly {
            12.0
        } else {
            60.0
        };
        for v in leaf.iter_mut() {
            *v = (*v / div).clamp(-1.0, 1.0);
        }
        let mut cursor = node_idx;
        loop {
            arena[cursor].visits += 1;
            for (p, v) in leaf.iter().enumerate() {
                if p < n_players {
                    arena[cursor].value_sum[p] += v;
                }
            }
            if cursor == 0 {
                break;
            }
            cursor = arena[cursor].parent;
        }
    }

    // Pick the root child with the most visits.
    let root_node = &arena[0];
    if root_node.children.is_empty() {
        return heuristic_ai::pass_decision(state);
    }
    let best = root_node
        .children
        .iter()
        .map(|ch| {
            let n = &arena[ch.node];
            (ch, n.visits, n.value_sum[root_pid] / n.visits.max(1) as f64)
        })
        .max_by(|a, b| a.1.cmp(&b.1).then(a.2.partial_cmp(&b.2).unwrap()))
        .map(|(ch, _, _)| ch)
        .unwrap();
    let score = arena[best.node].value_sum[root_pid] / arena[best.node].visits.max(1) as f64;
    let mut ordered_root: Vec<(&Child, u32, f64)> = root_node
        .children
        .iter()
        .map(|ch| {
            let n = &arena[ch.node];
            (ch, n.visits, n.value_sum[root_pid] / n.visits.max(1) as f64)
        })
        .collect();
    ordered_root.sort_by(|a, b| b.1.cmp(&a.1).then(b.2.partial_cmp(&a.2).unwrap()));

    for (ch, _visits, child_score) in ordered_root {
        let mut sim = state.clone();
        if apply_move(&mut sim, &ch.mv).is_ok() {
            return Decision {
                mv: ch.mv.clone(),
                score: child_score,
            };
        }
    }

    if std::env::var("BRASS_MCTS_DEBUG").is_ok() {
        eprintln!(
            "[mcts] sims={} avg_depth={:.2} apply_fail={} empty_cand={} nodes={}",
            cfg.simulations,
            total_depth as f64 / cfg.simulations as f64,
            apply_fail,
            empty_cand,
            arena.len()
        );
        for ch in &arena[0].children {
            let n = &arena[ch.node];
            eprintln!(
                "  root child: {} visits={} q={:.2}",
                ch.mv.describe(state),
                n.visits,
                n.value_sum[root_pid] / n.visits.max(1) as f64
            );
        }
    }
    Decision {
        mv: best.mv.clone(),
        score,
    }
}

/// Apply era-end / game-end transitions to a simulated state.
fn handle_turn_result<R: rand::Rng>(state: &mut GameState<R>, tr: TurnResult) {
    match tr {
        TurnResult::Continue => {}
        TurnResult::EndCanalEra => end_canal_era(state),
        TurnResult::EndGame => end_game(state),
    }
}
