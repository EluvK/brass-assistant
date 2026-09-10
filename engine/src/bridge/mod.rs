//! Serialization, tensor encoding, and Python bindings.

pub mod pymod;

/// Scale of the terminal value target: `(vp - table_mean_vp) / VP_SCALE`.
/// The Rust backup and the Python loss must use the same number
/// (docs/ai-action-encoding.md §4.1).
pub const VP_SCALE: f32 = 50.0;

// Compatibility paths for callers that historically imported these helpers
// through `bridge`; the implementations are now available in pure-Rust
// builds as top-level modules.
pub use crate::{action_features, encode, move_codec};
