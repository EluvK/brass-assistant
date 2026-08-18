//! PyO3 bindings: expose the Rust engine to Python (package `brass_engine`).
//!
//! Raw API surface (`import brass_engine`):
//!   GameState(seed, players)                # 2-4 players
//!     .current_player_id / .player_count    # int properties
//!     .era                                   # 0 = canal, 1 = rail
//!     .round / .game_over / .current_player_money / .has_pending_bonus
//!     .legal_moves() -> list[(policy_slot:int, canonical:str, describe:str)]
//!     .apply_move(canonical:str) -> str     # full step; ValueError if illegal
//!     .determinize() -> GameState            # opponent-hand sampling
//!     .legal_mask() -> list[int]             # legal policy slots
//!     .state_to_tensor() -> (board, links, global, own_hand, opp_hands)
//!     .choose_heuristic() -> (canonical, describe, score)
//!   module: policy_table_size (int), describe_slot(slot), moves_to_slots(canonical)

use numpy::{PyArray1, PyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rand::SeedableRng;
use rand::rngs::StdRng;

use crate::encode;
use crate::heuristic_ai;
use crate::mcts_ai;
use crate::move_codec;
use crate::policy;
use crate::rules::{apply_move, legal_moves};
use crate::state::GameState;

#[pyclass(name = "GameState")]
pub struct PyGame {
    state: GameState,
}

#[pymethods]
impl PyGame {
    #[new]
    fn new(seed: u64, players: usize) -> PyResult<Self> {
        if !(2..=4).contains(&players) {
            return Err(PyValueError::new_err(format!(
                "players must be 2..=4, got {players}"
            )));
        }
        let rng = StdRng::seed_from_u64(seed);
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

    /// List of legal moves as (policy_slot, canonical, describe) triples.
    fn legal_moves(&self) -> Vec<(usize, String, String)> {
        let mut state = self.state.clone();
        legal_moves(&mut state)
            .into_iter()
            .map(|mv| {
                let slots = policy::move_slots(&mv);
                let slot = slots.first().copied().unwrap_or(usize::MAX);
                let describe = mv.describe(&state);
                (slot, move_codec::encode(&mv), describe)
            })
            .collect()
    }

    /// Lean variant of `legal_moves` for the MCTS hot path: returns only
    /// (policy_slot, canonical) pairs, skipping the human-readable `describe`
    /// string that the search never reads. Uses the slot-level generator, so
    /// it is also several times faster than `legal_moves`.
    fn legal_moves_slots(&self) -> Vec<(usize, String)> {
        let mut state = self.state.clone();
        crate::rules::legal_slot_moves(&mut state)
            .into_iter()
            .map(|sm| (sm.slot, move_codec::encode(&sm.mv)))
            .collect()
    }

    /// Network-guided ISMCTS search with the tree in Rust (`nn_mcts`).
    ///
    /// `net_fn(board, links, global, own_hand, opp_hands)` receives 2-D numpy
    /// arrays with one row per request (each encoded from that request state's
    /// current player) and must return `(type_logits (rows,7), goal_logits
    /// (rows,P), values (rows,4))`. Returns (best_canonical, root children as
    /// (slot, canonical, visits) best-first, legal_slots).
    #[pyo3(signature = (net_fn, sims, c_puct, max_depth, dirichlet_alpha, dirichlet_weight, add_root_noise, batch_size=64))]
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
    ) -> PyResult<(Option<String>, Vec<(usize, String, u32)>, Vec<usize>)> {
        let cfg = crate::nn_mcts::NnMctsConfig {
            c_puct,
            max_depth,
            dirichlet_alpha,
            dirichlet_weight,
            batch_size: batch_size.max(1),
        };
        let res = crate::nn_mcts::search_net(&self.state, &cfg, sims, add_root_noise, &net_fn, py)?;
        Ok((res.best_canonical, res.children, res.legal_slots))
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
        Ok(match crate::engine::advance_turn(&mut self.state) {
            TurnResult::Continue => "continue".to_string(),
            TurnResult::EndCanalEra => "end_canal_era".to_string(),
            TurnResult::EndGame => "end_game".to_string(),
        })
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
        let mut rng = StdRng::from_entropy();
        let state = mcts_ai::determinize(&self.state, &mut rng);
        PyGame { state }
    }

    /// Set of policy slots reachable by at least one legal move.
    fn legal_mask(&self) -> Vec<usize> {
        let mut state = self.state.clone();
        policy::legal_mask(&mut state)
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
    m.add("policy_table_size", policy::policy_table_size())?;
    m.add("network_double_cells", policy::network_double_cells())?;
    m.add("BOARD_PLANES", encode::BOARD_PLANES)?;
    m.add("BOARD_CELLS", encode::BOARD_CELLS)?;
    m.add("LINK_PLANES", encode::LINK_PLANES)?;
    m.add("LINK_CELLS", encode::LINK_CELLS)?;
    m.add("GLOBAL_LEN", encode::GLOBAL_LEN)?;
    m.add("HAND_LEN", encode::HAND_LEN)?;
    m.add("MAX_PLAYERS", encode::MAX_PLAYERS)?;
    m.add("slot_types", policy::slot_types())?;
    m.add_function(wrap_pyfunction!(py_describe_slot, m)?)?;
    m.add_function(wrap_pyfunction!(py_moves_to_slots, m)?)?;
    Ok(())
}

#[pyfunction(name = "describe_slot")]
fn py_describe_slot(slot: usize) -> String {
    policy::describe_slot(slot)
}

/// Map a canonical move string to its policy slot(s).
#[pyfunction(name = "moves_to_slots")]
fn py_moves_to_slots(canonical: &str) -> PyResult<Vec<usize>> {
    let mv =
        move_codec::decode(canonical).map_err(|e| PyValueError::new_err(format!("decode: {e}")))?;
    Ok(policy::move_slots(&mv))
}
