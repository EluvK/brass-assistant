//! Brass game engine organized by architectural responsibility.
//!
//! The flat module paths below are re-exported for compatibility with existing
//! Rust callers and the Python binding. New code should prefer the layer paths
//! (`model`, `game_state`, `gameplay`, `ai`, and `bridge`).

pub mod ai;
pub mod bridge;
pub mod game_state;
pub mod gameplay;
pub mod model;

pub use ai::{heuristic_ai, mcts_ai, nn_mcts, random_ai, replay};
pub use bridge::{encode, move_codec, pymod, replay_fmt};
pub use game_state::{graph, income, state};
pub use gameplay::{engine, game_loop, rules, scoring};
pub use model::{data, map, r#move};
