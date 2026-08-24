//! PyO3 bindings: expose the Rust engine to Python (package `brass_engine`).
//!
//! Raw API surface (`import brass_engine`):
//!   GameState(seed, players)                # 2-4 players
//!     .current_player_id / .player_count    # int properties
//!     .era                                   # 0 = canal, 1 = rail
//!     .round / .game_over / .current_player_money / .has_pending_bonus
//!     .legal_moves() -> list[(action_id:int, canonical:str, describe:str)]
//!     .legal_candidates() -> list[(canonical:str, features: float32[235])]
//!     .apply_move(canonical:str) -> str     # full step; ValueError if illegal
//!     .determinize() -> GameState            # opponent-hand sampling
//!     .legal_candidates() -> list[(canonical:str, features: float32[235])]
//!     .state_to_tensor() -> (board, links, global, own_hand, opp_hands)
//!     .choose_heuristic() -> (canonical, describe, score)
//!   module: ACTION_FEATURE_DIM, ACTION_FEATURE_SCHEMA_VERSION

use numpy::{PyArray1, PyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use rand_chacha::ChaCha12Rng;
use rand::SeedableRng;
// use rand_core::SeedableRng;

use crate::encode;
use crate::heuristic_ai;
use crate::mcts_ai;
use crate::move_codec;
use crate::rules::{apply_move, legal_moves};
use crate::state::GameState;

#[pyclass(name = "GameState")]
pub struct PyGame {
    state: GameState,
}

const SNAPSHOT_MAGIC: &[u8; 4] = b"BASS";
const SNAPSHOT_VERSION: u8 = 2;

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

    #[getter]
    fn has_pending_bonus(&self) -> bool {
        self.state.pending_bonus.is_some()
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

    /// List of concrete legal moves as (node-local action_id, canonical,
    /// describe) triples.
    fn legal_moves(&self) -> Vec<(usize, String, String)> {
        let mut state = self.state.clone();
        legal_moves(&mut state)
            .into_iter()
            .enumerate()
            .map(|(action_id, mv)| {
                let describe = mv.describe(&state);
                (action_id, move_codec::encode(&mv), describe)
            })
            .collect()
    }

    /// Concrete executable legal moves with structured v2 action features.
    /// Every returned row is a complete action the engine can apply.
    fn legal_candidates(&self) -> Vec<(String, Vec<f32>)> {
        let mut state = self.state.clone();
        legal_moves(&mut state)
            .into_iter()
            .map(|mv| {
                let canonical = move_codec::encode(&mv);
                let features = crate::bridge::action_features::encode_move(&state, &mv);
                (canonical, features)
            })
            .collect()
    }

    /// Return a bounded, scored teacher shortlist and the 2-ply selected move.
    /// Training does not retain every legal concrete action: that would make a
    /// large replay buffer grow with the full combinatorial action space. The
    /// MCTS inference path still evaluates every legal candidate.
    fn heuristic_candidates(&self) -> (Vec<f32>, Vec<f32>, String, usize, f64, usize) {
        let mut heuristic_state = self.state.clone();
        let decision = heuristic_ai::choose_action(&mut heuristic_state);
        let teacher_canonical = move_codec::encode(&decision.mv);
        let mut scoring_state = self.state.clone();
        let scored = heuristic_ai::candidate_actions_k(&mut scoring_state, 4);
        let mut features = Vec::new();
        let mut scores = Vec::new();
        let mut canonical_candidates = Vec::new();
        let mut seen = std::collections::HashSet::new();
        for scored_move in scored {
            let canonical = move_codec::encode(&scored_move.mv);
            if seen.insert(canonical.clone()) {
                features.extend(crate::bridge::action_features::encode_move(
                    &scoring_state,
                    &scored_move.mv,
                ));
                scores.push(scored_move.score as f32);
                canonical_candidates.push(canonical);
            }
        }
        let teacher_index = canonical_candidates
            .iter()
            .position(|canonical| canonical == &teacher_canonical)
            .unwrap_or_else(|| {
                features.extend(crate::bridge::action_features::encode_move(
                    &heuristic_state,
                    &decision.mv,
                ));
                scores.push(decision.score as f32);
                canonical_candidates.push(teacher_canonical.clone());
                canonical_candidates.len() - 1
            });
        (
            features,
            scores,
            teacher_canonical,
            teacher_index,
            decision.score,
            canonical_candidates.len(),
        )
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
    /// `main.rs`). A free-develop bonus leaves the same player to move again
    /// (`has_pending_bonus`).
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
            crate::rules::Move::Build { ind, .. } => ("build".to_string(), *ind as usize),
            crate::rules::Move::Network { .. } => ("network".to_string(), 0),
            crate::rules::Move::NetworkDouble { .. } => ("network_double".to_string(), 0),
            crate::rules::Move::Develop { .. } | crate::rules::Move::ResolveFreeDevelop { .. } => {
                ("develop".to_string(), 0)
            }
            crate::rules::Move::Sell { .. } => ("sell".to_string(), 0),
            crate::rules::Move::Loan { .. } => ("loan".to_string(), 0),
            crate::rules::Move::Pass { .. } => ("pass".to_string(), 0),
            crate::rules::Move::Scout { .. } => ("scout".to_string(), 0),
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
    /// board (17,49), links (6,39), global (50,), own_hand (35,), opp_hands (105,).
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
    let vv: Vec<Vec<f32>> = data.chunks_exact(cols).map(|c| c.to_vec()).collect();
    Ok(PyArray2::from_vec2(py, &vv)?)
}

#[pymodule]
fn brass_engine(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyGame>()?;
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
    Ok(())
}
