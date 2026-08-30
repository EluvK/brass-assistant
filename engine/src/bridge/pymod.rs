//! PyO3 bindings: expose the Rust engine to Python (module `brass_ai._engine`).
//!
//! Raw API surface (`from brass_ai import _engine`):
//!   GameState(seed, players)                # 2-4 players
//!     .current_player_id / .player_count    # int properties
//!     .era                                   # 0 = canal, 1 = rail
//!     .round / .game_over / .current_player_money
//!     .legal_moves() -> legacy fully resolved executable moves
//!     .legal_operations() -> structural operations with card candidates
//!     .legal_candidates() -> (canonicals: list[str], features: f32 (N,301))
//!     .heuristic_candidates() -> (features f32 (K,301), scores, card_scores,
//!                                 canonical, teacher_index, score, card_score,
//!                                 count)  # numpy arrays at the boundary
//!     .materialize_snapshot(snapshot, teacher_canonical) -> full training
//!         sample arrays: state tensors + uint8 quarter-step candidates +
//!         one-hot equivalence-class policy (see the method docs)
//!     .apply_move(canonical:str) -> str     # full step; ValueError if illegal
//!     .determinize() -> GameState            # opponent-hand sampling
//!     .state_to_tensor() -> (board, links, global, own_hand, opp_hands)
//!     .choose_heuristic() -> (canonical, describe, score)
//!   module: action/state feature schemas and state-graph topology constants
//!
//! Bulk float payloads cross the boundary as numpy arrays, never as Python
//! lists: boxing one f32 per PyFloat dominated the materialization profile
//! (~100x the Rust compute for a full-legal candidate matrix).

use numpy::{PyArray1, PyArray2, PyArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use rand::SeedableRng;
use rand_chacha::ChaCha12Rng;
// use rand_core::SeedableRng;

use crate::encode;
use crate::heuristic_ai;
use crate::mcts_ai;
use crate::move_codec;
use crate::rules::{apply_move, legal_moves, legal_resolved_moves};
use crate::state::GameState;

#[pyclass(name = "GameState")]
pub struct PyGame {
    state: GameState,
}

/// Wire format shared with the Python replay worker: `GameState.snapshot()`
/// bytes are magic + version + `snapshot_bytes`, and `GameState.from_snapshot`
/// is the only sanctioned restore path.
pub(crate) const SNAPSHOT_MAGIC: &[u8; 4] = b"BASS";
pub(crate) const SNAPSHOT_VERSION: u8 = 3;

#[pymethods]
impl PyGame {
    #[new]
    fn new(seed: u64, players: usize) -> PyResult<Self> {
        if !(2..=4).contains(&players) {
            return Err(PyValueError::new_err(format!(
                "players must be 2..=4, got {players}"
            )));
        }
        let rng = ChaCha12Rng::seed_from_u64(seed);
        Ok(PyGame {
            state: GameState::new(rng, players),
        })
    }

    #[getter]
    fn current_player_id(&self) -> usize {
        self.state.current_player_id()
    }

    #[getter]
    fn player_count(&self) -> usize {
        self.state.player_count()
    }

    #[getter]
    fn era(&self) -> u8 {
        match self.state.era {
            crate::data::Era::Canal => 0,
            crate::data::Era::Rail => 1,
        }
    }

    #[getter]
    fn round(&self) -> u32 {
        self.state.round
    }

    #[getter]
    fn turn_order(&self) -> Vec<usize> {
        self.state.turn_order.clone()
    }

    #[getter]
    fn game_over(&self) -> bool {
        self.state.game_over
    }

    #[getter]
    fn current_player_money(&self) -> i32 {
        self.state.current_player().money
    }

    /// Independent deep clone of the game state (for MCTS child nodes).
    fn clone(&self) -> Self {
        PyGame {
            state: self.state.clone(),
        }
    }

    /// Opaque, versioned snapshot of the complete current Rust GameState.
    /// Restore is direct deserialization; no action history is replayed.
    fn snapshot<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(SNAPSHOT_MAGIC);
        bytes.push(SNAPSHOT_VERSION);
        bytes.extend_from_slice(&self.state.snapshot_bytes().map_err(PyValueError::new_err)?);
        Ok(PyBytes::new(py, &bytes))
    }

    #[staticmethod]
    fn from_snapshot(snapshot: Vec<u8>) -> PyResult<Self> {
        restore_snapshot(&snapshot)
    }

    /// One-call materialization of a snapshot-backed training sample.
    ///
    /// Restores the snapshot, then computes everything
    /// `selfplay.materialize_sample` needs in Rust:
    ///   (pid, era, board, links, global, own_hand, opp_hands,
    ///    candidates u8 (N,301), teacher_index, policy f32 (N,))
    ///
    /// * `candidates` rows are the v4 action features packed as lossless
    ///   uint8 quarter-steps (x4), matching `hierarchical_policy.
    ///   compress_candidate_features`; a non-quarter-step value raises.
    /// * `policy` is the teacher one-hot spread uniformly over the complete
    ///   v4-observable equivalence class (rows identical to the teacher row),
    ///   mirroring `hierarchical_policy.teacher_equivalence_policy`.
    /// * `teacher_index` is the position of `teacher_canonical`; ValueError
    ///   when it is not a legal concrete move of the restored state.
    #[allow(clippy::type_complexity)]
    #[staticmethod]
    fn materialize_snapshot<'py>(
        py: Python<'py>,
        snapshot: Vec<u8>,
        teacher_canonical: &str,
    ) -> PyResult<(
        usize,
        u8,
        Bound<'py, PyArray2<f32>>,
        Bound<'py, PyArray2<f32>>,
        Bound<'py, PyArray1<f32>>,
        Bound<'py, PyArray1<f32>>,
        Bound<'py, PyArray1<f32>>,
        Bound<'py, PyArray2<u8>>,
        usize,
        Bound<'py, PyArray1<f32>>,
    )> {
        let dim = crate::bridge::action_features::ACTION_FEATURE_DIM;
        let PyGame { state } = restore_snapshot(&snapshot)?;
        let pid = state.current_player_id();
        let era = match state.era {
            crate::data::Era::Canal => 0u8,
            crate::data::Era::Rail => 1u8,
        };
        let mut work = state;
        let legal = legal_resolved_moves(&mut work);
        if legal.is_empty() {
            return Err(PyValueError::new_err("state has no legal candidates"));
        }
        let n = legal.len();
        // Decode the teacher canonical once and match moves structurally
        // (string-encoding every candidate just for the lookup would dominate
        // the whole materialization cost).
        let teacher_mv = move_codec::decode(teacher_canonical)
            .map_err(|e| PyValueError::new_err(format!("decode: {e}")))?;
        let teacher_index = legal
            .iter()
            .position(|mv| *mv == teacher_mv)
            .ok_or_else(|| {
                PyValueError::new_err(
                    "replay teacher action is not legal in its restored GameState",
                )
            })?;

        // Encode every candidate row and quantize to uint8 quarter-steps.
        let mut packed = vec![0u8; n * dim];
        let mut row = Vec::with_capacity(dim);
        for (i, mv) in legal.iter().enumerate() {
            crate::bridge::action_features::encode_move_into(&work, mv, &mut row);
            let dst = &mut packed[i * dim..(i + 1) * dim];
            for (o, &v) in dst.iter_mut().zip(row.iter()) {
                let scaled = v * 4.0;
                let r = scaled.round();
                if !(0.0..=255.0).contains(&r) || (scaled - r).abs() > 1e-3 {
                    return Err(PyValueError::new_err(format!(
                        "action features contain values outside the quarter-step schema: {v}"
                    )));
                }
                *o = r as u8;
            }
        }

        // Teacher mass spread uniformly over the identical-feature class.
        let teacher_row = &packed[teacher_index * dim..(teacher_index + 1) * dim];
        let mut class_size = 0usize;
        let mut policy = vec![0.0f32; n];
        for (i, member) in policy.iter_mut().enumerate() {
            if &packed[i * dim..(i + 1) * dim] == teacher_row {
                *member = 1.0;
                class_size += 1;
            }
        }
        for member in policy.iter_mut() {
            if *member > 0.0 {
                *member /= class_size as f32;
            }
        }

        let t = encode::state_to_tensor(&work, pid);
        let board = PyArray1::from_vec(py, t.board).reshape((encode::BOARD_PLANES, encode::BOARD_CELLS))?;
        let links = PyArray1::from_vec(py, t.links).reshape((encode::LINK_PLANES, encode::LINK_CELLS))?;
        Ok((
            pid,
            era,
            board,
            links,
            PyArray1::from_vec(py, t.global),
            PyArray1::from_vec(py, t.own_hand),
            PyArray1::from_vec(py, t.opp_hands),
            PyArray1::from_vec(py, packed).reshape((n, dim))?,
            teacher_index,
            PyArray1::from_vec(py, policy),
        ))
    }

    /// List of concrete legal moves as (node-local action_id, canonical,
    /// describe) triples.
    fn legal_moves(&self) -> Vec<(usize, String, String)> {
        let mut state = self.state.clone();
        legal_resolved_moves(&mut state)
            .into_iter()
            .enumerate()
            .map(|(action_id, mv)| {
                let describe = mv.describe(&state);
                (action_id, move_codec::encode(&mv), describe)
            })
            .collect()
    }

    /// Structural legal operations.  Each row contains its per-state id,
    /// description, required card count (1 or 3 for Scout), and candidates as
    /// `(hand_index, keep_score)`.  Resolve the selected cards through
    /// `resolve_legal_operation` before applying the result.
    fn legal_operations(&self) -> Vec<(usize, String, usize, Vec<(usize, f64)>)> {
        let mut state = self.state.clone();
        legal_moves(&mut state)
            .into_iter()
            .enumerate()
            .map(|(id, mv)| {
                (
                    id,
                    mv.describe(&state),
                    match mv.card_requirement() {
                        crate::r#move::CardRequirement::One => 1,
                        crate::r#move::CardRequirement::ScoutThree => 3,
                    },
                    mv.card_candidates()
                        .iter()
                        .map(|c| (c.hand_index, c.keep_score))
                        .collect(),
                )
            })
            .collect()
    }

    /// Resolve a structural operation from the current state into the nested
    /// executable representation consumed by `apply_move`.
    fn resolve_legal_operation(
        &self,
        operation_id: usize,
        selected_cards: Vec<usize>,
    ) -> PyResult<String> {
        let mut state = self.state.clone();
        let mv = legal_moves(&mut state)
            .into_iter()
            .nth(operation_id)
            .ok_or_else(|| PyValueError::new_err("unknown structural operation id"))?;
        let resolved = mv.resolve(&selected_cards).map_err(PyValueError::new_err)?;
        Ok(move_codec::encode(&resolved))
    }

    /// Concrete executable legal moves with structured action features
    /// (schema version exposed at module level).
    /// Every returned row is a complete action the engine can apply.
    /// Returns `(canonicals, features)` with `features` an f32 ndarray of
    /// shape `(N, ACTION_FEATURE_DIM)`, row i aligned to `canonicals[i]`.
    fn legal_candidates<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<(Vec<String>, Bound<'py, PyArray2<f32>>)> {
        let mut state = self.state.clone();
        let legal = legal_resolved_moves(&mut state);
        let dim = crate::bridge::action_features::ACTION_FEATURE_DIM;
        let mut canonicals = Vec::with_capacity(legal.len());
        let mut flat = vec![0.0f32; legal.len() * dim];
        let mut row = Vec::with_capacity(dim);
        for (i, mv) in legal.iter().enumerate() {
            canonicals.push(move_codec::encode(mv));
            crate::bridge::action_features::encode_move_into(&state, mv, &mut row);
            flat[i * dim..(i + 1) * dim].copy_from_slice(&row);
        }
        let features = PyArray1::from_vec(py, flat).reshape((legal.len(), dim))?;
        Ok((canonicals, features))
    }

    /// Return a bounded, scored teacher shortlist and the 2-ply selected move.
    /// Training does not retain every legal concrete action: that would make a
    /// large replay buffer grow with the full combinatorial action space. The
    /// MCTS inference path still evaluates every legal candidate.
    ///
    /// Returns `(features (K,301), scores (K,), card_scores (K,), canonical,
    /// teacher_index, score, card_score, count)`; the first three are numpy
    /// arrays (row i aligned with candidate i, `count == features.shape[0]`).
    #[allow(clippy::type_complexity)]
    fn heuristic_candidates<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<(
        Bound<'py, PyArray2<f32>>,
        Bound<'py, PyArray1<f32>>,
        Bound<'py, PyArray1<f32>>,
        String,
        usize,
        f64,
        f64,
        usize,
    )> {
        let dim = crate::bridge::action_features::ACTION_FEATURE_DIM;
        // `choose_action` and `candidate_actions_k` only mutate interior
        // caches, so one scratch clone serves both calls.
        let mut work = self.state.clone();
        let decision = heuristic_ai::choose_action(&mut work);
        let teacher_canonical = move_codec::encode(&decision.mv);
        let scored = heuristic_ai::candidate_actions_k(&mut work, 4);
        let mut canonical_candidates: Vec<String> = Vec::new();
        let mut seen = std::collections::HashSet::new();
        // When the deduped shortlist misses the teacher move, append it with
        // the 2-ply decision's own scores (distinct from the shortlist's).
        let mut teacher_appended = true;
        for scored_move in &scored {
            let canonical = move_codec::encode(&scored_move.mv);
            if canonical == teacher_canonical {
                teacher_appended = false;
            }
            if seen.insert(canonical.clone()) {
                canonical_candidates.push(canonical);
            }
        }
        if teacher_appended {
            canonical_candidates.push(teacher_canonical.clone());
        }
        let teacher_index = canonical_candidates
            .iter()
            .position(|canonical| canonical == &teacher_canonical)
            .unwrap_or(0);
        let count = canonical_candidates.len();

        // One encode pass over the deduped candidate order, matching the
        // scores of the first scored move behind each canonical.
        let mut flat = vec![0.0f32; count * dim];
        let mut scores = vec![0.0f32; count];
        let mut card_scores = vec![0.0f32; count];
        let mut row = Vec::with_capacity(dim);
        {
            let mut fill = |slot: usize,
                            mv: &crate::rules::ResolvedMove,
                            score: f64,
                            card_score: f64,
                            state: &GameState| {
                crate::bridge::action_features::encode_move_into(state, mv, &mut row);
                flat[slot * dim..(slot + 1) * dim].copy_from_slice(&row);
                scores[slot] = score as f32;
                card_scores[slot] = card_score as f32;
            };
            let mut slot = 0usize;
            for scored_move in &scored {
                let canonical = move_codec::encode(&scored_move.mv);
                if seen.remove(&canonical) {
                    fill(
                        slot,
                        &scored_move.mv,
                        scored_move.score,
                        scored_move.card_score,
                        &work,
                    );
                    slot += 1;
                }
            }
            if teacher_appended {
                fill(
                    count - 1,
                    &decision.mv,
                    decision.score,
                    decision.card_score,
                    &work,
                );
            }
        }

        let features = PyArray1::from_vec(py, flat).reshape((count, dim))?;
        Ok((
            features,
            PyArray1::from_vec(py, scores),
            PyArray1::from_vec(py, card_scores),
            teacher_canonical,
            teacher_index,
            decision.score,
            decision.card_score,
            count,
        ))
    }

    /// Network-guided ISMCTS search with the tree in Rust (`nn_mcts`).
    ///
    /// `net_fn(board, links, global, own_hand, opp_hands, candidates, mask)`
    /// receives batched state arrays plus flattened padded candidate features
    /// and must return `(candidate_logits (rows,max_candidates), values
    /// (rows,4))`. Returns (best_canonical, root children as (slot, canonical,
    /// visits) best-first, legal candidate ids).
    #[pyo3(signature = (net_fn, sims, c_puct, max_depth, dirichlet_alpha, dirichlet_weight, add_root_noise, batch_size=64, candidate_k=4))]
    fn search_net(
        &self,
        py: Python<'_>,
        net_fn: pyo3::Py<pyo3::types::PyAny>,
        sims: usize,
        c_puct: f64,
        max_depth: usize,
        dirichlet_alpha: f64,
        dirichlet_weight: f64,
        add_root_noise: bool,
        batch_size: usize,
        candidate_k: usize,
    ) -> PyResult<(Option<String>, Vec<(usize, String, u32)>, Vec<usize>)> {
        let cfg = crate::nn_mcts::NnMctsConfig {
            c_puct,
            max_depth,
            dirichlet_alpha,
            dirichlet_weight,
            batch_size: batch_size.max(1),
            candidate_k,
        };
        let res = crate::nn_mcts::search_net(&self.state, &cfg, sims, add_root_noise, &net_fn, py)?;
        Ok((res.best_canonical, res.children, res.legal_candidate_ids))
    }

    /// Apply a canonical move string; returns a human-readable summary.
    /// The full game step is performed: move + turn advance + era/game end
    /// transitions (mirrors `engine::step` plus the era handlers used by
    /// `main.rs`). Merchant free-develop bonuses resolve within the Sell move.
    fn apply_move(&mut self, canonical: &str) -> PyResult<String> {
        let mv = move_codec::decode(canonical)
            .map_err(|e| PyValueError::new_err(format!("decode: {e}")))?;
        let summary = apply_move(&mut self.state, &mv).map_err(|e| PyValueError::new_err(e))?;
        let tr = crate::engine::advance_turn(&mut self.state);
        crate::engine::handle_turn_result(&mut self.state, tr);
        Ok(summary)
    }

    // ------------------------------------------------------- replay (stepwise)
    // The replay driver needs fine-grained control that `apply_move` (action +
    // turn advance + era end in one call) does not give: it must print the
    // canal-era score detail BEFORE the era-end cleanup erases the tiles/links.
    // These stepwise methods make the game loop explicit, mirroring `bin/replay.rs`.

    /// Apply a move WITHOUT advancing the turn or running era-end scoring.
    /// Returns (summary, ok). Caller must then call `advance_turn_raw()`.
    fn apply_move_raw(&mut self, canonical: &str) -> PyResult<(String, bool)> {
        let mv = move_codec::decode(canonical)
            .map_err(|e| PyValueError::new_err(format!("decode: {e}")))?;
        match apply_move(&mut self.state, &mv) {
            Ok(summary) => Ok((summary, true)),
            Err(e) => Ok((format!("!!失败: {e}"), false)),
        }
    }

    /// Advance the turn after a raw move. Returns
    /// "continue" | "end_canal_era" | "end_game" WITHOUT running the era
    /// transition (the caller decides when to call finish_canal_era/finish_game).
    fn advance_turn_raw(&mut self) -> PyResult<String> {
        use crate::engine::TurnResult;
        let result = match crate::engine::advance_turn(&mut self.state) {
            TurnResult::Continue => "continue".to_string(),
            TurnResult::EndCanalEra => "end_canal_era".to_string(),
            TurnResult::EndGame => "end_game".to_string(),
        };
        Ok(result)
    }

    /// Run the canal-era score + cleanup + reshuffle (the driver calls this
    /// AFTER printing the era score detail / cleanup log).
    fn finish_canal_era(&mut self) {
        crate::engine::handle_turn_result(&mut self.state, crate::engine::TurnResult::EndCanalEra);
    }

    /// Run the end-of-game scoring (the driver calls this after the final move).
    fn finish_game(&mut self) {
        crate::engine::handle_turn_result(&mut self.state, crate::engine::TurnResult::EndGame);
    }

    // ------------------------------------------------------- replay (logging)

    /// Opening metadata + per-player state/hand + merchants + board.
    fn replay_header(&self) -> String {
        use crate::replay_fmt;
        let n = self.state.player_count();
        let mut out = format!(
            "Replay: {}players seed-less | 起始顺位: {:?}\n",
            n, self.state.turn_order
        );
        for pid in 0..n {
            out.push_str(&format!(
                "开局 玩家{pid}: {} | 手牌: {}\n",
                replay_fmt::player_state(&self.state, pid),
                replay_fmt::hand_display(&self.state, pid)
            ));
        }
        out.push_str(&format!(
            "商家: {}\n",
            replay_fmt::merchant_state(&self.state)
        ));
        out.push_str(&format!("{}\n", replay_fmt::board_state(&self.state)));
        out
    }

    /// Human-readable detail of a move (call BEFORE applying it).
    fn replay_move_detail(&self, canonical: &str) -> PyResult<String> {
        let mv = move_codec::decode(canonical)
            .map_err(|e| PyValueError::new_err(format!("decode: {e}")))?;
        Ok(crate::replay_fmt::move_detail(&self.state, &mv))
    }

    /// Action-class + industry-index for tallying per-player action stats.
    fn replay_action_type(&self, canonical: &str) -> PyResult<(String, usize)> {
        let mv = move_codec::decode(canonical)
            .map_err(|e| PyValueError::new_err(format!("decode: {e}")))?;
        Ok(match &mv {
            crate::rules::ResolvedMove::Build { ind, .. } => ("build".to_string(), *ind as usize),
            crate::rules::ResolvedMove::Network { .. } => ("network".to_string(), 0),
            crate::rules::ResolvedMove::NetworkDouble { .. } => ("network_double".to_string(), 0),
            crate::rules::ResolvedMove::Develop { .. } => ("develop".to_string(), 0),
            crate::rules::ResolvedMove::Sell { .. } => ("sell".to_string(), 0),
            crate::rules::ResolvedMove::Loan { .. } => ("loan".to_string(), 0),
            crate::rules::ResolvedMove::Pass { .. } => ("pass".to_string(), 0),
            crate::rules::ResolvedMove::Scout { .. } => ("scout".to_string(), 0),
        })
    }

    /// Era score detail for one player (links VP + flipped-tile VP sources).
    fn replay_era_score(&self, pid: usize) -> String {
        crate::replay_fmt::era_score_detail(&self.state, pid)
    }

    /// Per-player tiles on board (or only flipped ones).
    #[pyo3(signature = (pid, flipped_only=false))]
    fn replay_tiles(&self, pid: usize, flipped_only: bool) -> String {
        crate::replay_fmt::player_tiles_detail(&self.state, pid, flipped_only)
    }

    /// Canal-era cleanup detail (links/tiles that get removed).
    fn replay_cleanup(&self) -> String {
        crate::replay_fmt::canal_cleanup_detail(&self.state)
    }

    /// Player state line (money/income/hand/VP/links) for the log.
    fn replay_player_state(&self, pid: usize) -> String {
        crate::replay_fmt::player_state(&self.state, pid)
    }

    /// Board market/tile/link summary for the log.
    fn replay_board(&self) -> String {
        crate::replay_fmt::board_state(&self.state)
    }

    /// Merchant tile summary for the log.
    fn replay_merchants(&self) -> String {
        crate::replay_fmt::merchant_state(&self.state)
    }

    /// Full hand of a player for the log.
    fn replay_hand(&self, pid: usize) -> String {
        crate::replay_fmt::hand_display(&self.state, pid)
    }

    /// Compose the per-era action-statistics line (Python tallies the counts
    /// via `replay_action_type`; this formats them the same way as replay.rs).
    #[allow(clippy::too_many_arguments)]
    fn replay_stats_line(
        &self,
        build: Vec<u32>,
        network: u32,
        dbl_rail: u32,
        develop: u32,
        sell: u32,
        loan: u32,
        pass_: u32,
        scout: u32,
    ) -> String {
        let mut stats = crate::replay_fmt::PStats {
            build: [0; 6],
            network,
            dbl_rail,
            develop,
            sell,
            loan,
            pass: pass_,
            scout,
            actions: Vec::new(),
        };
        for (i, v) in build.iter().take(6).enumerate() {
            stats.build[i] = *v;
        }
        crate::replay_fmt::fmt_stats_line(&stats)
    }

    /// Final per-player (income_level, money) — economic-supervision targets.
    fn final_econ(&self) -> Vec<(i32, i32)> {
        self.state
            .players
            .iter()
            .map(|p| (p.income_level() as i32, p.money))
            .collect()
    }

    /// Per-player (income_level, money) at the moment the CANAL era ends
    /// (i.e. right after `finish_canal_era`; income is unchanged by era-end,
    /// so this equals the canal-era-final economy the driver reads there).
    fn canal_econ(&self) -> Vec<(i32, i32)> {
        self.state
            .players
            .iter()
            .map(|p| (p.income_level() as i32, p.money))
            .collect()
    }

    /// Determinize: sample opponent hands from the hidden card pool. The
    /// resulting state's RNG is seeded from the sampling stream (same pattern
    /// as `mcts_ai`), so subsequent draws vary per determinization.
    fn determinize(&self) -> Self {
        let mut rng = ChaCha12Rng::from_rng(&mut rand::rng());
        let state = mcts_ai::determinize(&self.state, &mut rng);
        PyGame { state }
    }

    /// Encode the state into numpy arrays:
    /// board (24,49), links (7,39), global (168,), own_hand (35,), opp_hands (105,).
    /// `perspective` defaults to the current player; pass another player id to
    /// encode from that player's viewpoint (used for MaxN value evaluation).
    #[pyo3(signature = (perspective=None))]
    fn state_to_tensor<'py>(
        &self,
        py: Python<'py>,
        perspective: Option<usize>,
    ) -> PyResult<(
        Bound<'py, PyArray2<f32>>,
        Bound<'py, PyArray2<f32>>,
        Bound<'py, PyArray1<f32>>,
        Bound<'py, PyArray1<f32>>,
        Bound<'py, PyArray1<f32>>,
    )> {
        let pid = perspective.unwrap_or_else(|| self.state.current_player_id());
        if pid >= self.state.player_count() {
            return Err(PyValueError::new_err(format!(
                "perspective {pid} out of range (players={})",
                self.state.player_count()
            )));
        }
        let t = encode::state_to_tensor(&self.state, pid);

        let board = reshape2(py, &t.board, encode::BOARD_PLANES, encode::BOARD_CELLS)?;
        let links = reshape2(py, &t.links, encode::LINK_PLANES, encode::LINK_CELLS)?;
        let global = PyArray1::from_vec(py, t.global);
        let own_hand = PyArray1::from_vec(py, t.own_hand);
        let opp_hands = PyArray1::from_vec(py, t.opp_hands);

        Ok((board, links, global, own_hand, opp_hands))
    }

    /// Final VP per player (index order), for value targets & evaluation.
    fn player_vps(&self) -> Vec<i32> {
        self.state.players.iter().map(|p| p.vp as i32).collect()
    }

    /// Player ids ranked by final standing (best first, tiebreaks applied).
    fn final_ranking(&self) -> Vec<usize> {
        crate::scoring::final_ranking(&self.state)
    }

    /// Choose an action with the engine's default heuristic policy.
    ///
    /// The current implementation includes 2-ply lookahead; callers should
    /// use this single entry point rather than distinguishing the internals.
    fn choose_heuristic(&self) -> (String, String, f64) {
        let mut state = self.state.clone();
        let d = heuristic_ai::choose_action(&mut state);
        (
            move_codec::encode(&d.mv),
            d.mv.describe(&self.state),
            d.score,
        )
    }
}

fn restore_snapshot(snapshot: &[u8]) -> PyResult<PyGame> {
    if snapshot.len() < 5 || &snapshot[..4] != SNAPSHOT_MAGIC {
        return Err(PyValueError::new_err("invalid GameState snapshot magic"));
    }
    if snapshot[4] == SNAPSHOT_VERSION {
        let state =
            GameState::from_snapshot_bytes(&snapshot[5..]).map_err(PyValueError::new_err)?;
        return Ok(PyGame { state });
    }
    if snapshot[4] != SNAPSHOT_VERSION {
        return Err(PyValueError::new_err(format!(
            "unsupported GameState snapshot version: {}",
            snapshot[4]
        )));
    }
    let state = GameState::from_snapshot_bytes(&snapshot[5..]).map_err(PyValueError::new_err)?;
    Ok(PyGame { state })
}

fn reshape2<'py>(
    py: Python<'py>,
    data: &[f32],
    rows: usize,
    cols: usize,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    debug_assert_eq!(data.len(), rows * cols);
    PyArray1::from_vec(py, data.to_vec()).reshape((rows, cols))
}

/// Rust port of `hierarchical_policy.coalesce_equivalent_policy`: spread each
/// concrete-policy mass uniformly across rows with identical features. The
/// hot self-play path coalesces a full-legal candidate matrix every move; the
/// numpy twin allocates a boolean class mask per non-zero class.
#[pyfunction]
fn coalesce_equivalent_policy<'py>(
    py: Python<'py>,
    features: &Bound<'py, PyArray2<f32>>,
    policy: &Bound<'py, PyArray1<f32>>,
) -> PyResult<Bound<'py, PyArray1<f32>>> {
    let dim = crate::bridge::action_features::ACTION_FEATURE_DIM;
    let feats = features.readonly();
    let array = feats.as_slice().map_err(|_| {
        PyValueError::new_err("candidate features must be contiguous")
    })?;
    let target = policy.readonly();
    let target = target
        .as_slice()
        .map_err(|_| PyValueError::new_err("policy target must be contiguous"))?;
    let n = array.len() / dim;
    if array.len() != n * dim || target.len() != n {
        return Err(PyValueError::new_err(
            "invalid candidate features for policy target",
        ));
    }
    if target.iter().any(|&v| !v.is_finite() || v < 0.0) {
        return Err(PyValueError::new_err("invalid policy target"));
    }
    let total: f32 = target.iter().sum();
    if total <= 0.0 {
        return Err(PyValueError::new_err(
            "policy target must contain positive mass",
        ));
    }
    let mut out = vec![0.0f32; n];
    let mut pending: Vec<usize> = (0..n).filter(|&i| target[i] > 0.0).collect();
    let mut members: Vec<usize> = Vec::new();
    while let Some(&index) = pending.first() {
        let class_row = &array[index * dim..(index + 1) * dim];
        members.clear();
        let mut mass = 0.0f32;
        for i in 0..n {
            if &array[i * dim..(i + 1) * dim] == class_row {
                mass += target[i];
                members.push(i);
            }
        }
        let share = mass / members.len() as f32 / total;
        for &i in &members {
            out[i] = share;
        }
        pending.retain(|i| !members.contains(i));
    }
    Ok(PyArray1::from_vec(py, out))
}

#[pymodule(name = "_engine")]
fn _engine(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyGame>()?;
    m.add_function(wrap_pyfunction!(coalesce_equivalent_policy, m)?)?;
    m.add(
        "ACTION_FEATURE_DIM",
        crate::bridge::action_features::ACTION_FEATURE_DIM,
    )?;
    m.add(
        "ACTION_FEATURE_SCHEMA_VERSION",
        crate::bridge::action_features::ACTION_FEATURE_SCHEMA_VERSION,
    )?;
    m.add("BOARD_PLANES", encode::BOARD_PLANES)?;
    m.add("BOARD_CELLS", encode::BOARD_CELLS)?;
    m.add("LINK_PLANES", encode::LINK_PLANES)?;
    m.add("LINK_CELLS", encode::LINK_CELLS)?;
    m.add("GLOBAL_LEN", encode::GLOBAL_LEN)?;
    m.add("HAND_LEN", encode::HAND_LEN)?;
    m.add("MAX_PLAYERS", encode::MAX_PLAYERS)?;
    m.add(
        "STATE_FEATURE_SCHEMA_VERSION",
        encode::STATE_FEATURE_SCHEMA_VERSION,
    )?;
    m.add("LOCATION_COUNT", encode::LOC_COUNT)?;
    m.add("BOARD_CELL_LOCATIONS", encode::board_cell_locations())?;
    m.add("CONNECTION_ENDPOINTS", encode::connection_endpoints())?;
    m.add("CONNECTION_VIA_FARMS", encode::connection_via_farms())?;
    Ok(())
}
