//! Brass game engine organized by architectural responsibility.
//!
//! The flat module paths below are re-exported for compatibility with existing
//! Rust callers and the Python binding. New code should prefer the layer paths
//! (`model`, `game_state`, `gameplay`, `ai`, and `bridge`).

#[path = "bridge/action_features.rs"]
pub mod action_features;
pub mod ai;
#[cfg(feature = "python")]
pub mod bridge;
#[path = "bridge/encode.rs"]
pub mod encode;
pub mod game_state;
pub mod gameplay;
pub mod model;
#[path = "bridge/move_codec.rs"]
pub mod move_codec;
// Human-readable diagnostics are useful to the pure-Rust replay binary too.
// The source remains beside the related bridge code until that directory is
// split physically; it is not part of the Python feature boundary.
#[path = "bridge/replay_fmt.rs"]
pub mod replay_fmt;

pub use ai::replay;
pub use ai::{heuristic_ai, mcts_ai, random_ai};
#[cfg(feature = "python")]
pub use ai::{nn_mcts, python_worker};
#[cfg(feature = "python")]
pub use bridge::pymod;
pub use game_state::{graph, income, state};
pub use gameplay::{engine, game_loop, rules, scoring};
pub use model::{data, map, r#move};
