//! Serialization, tensor encoding, and Python bindings.

pub mod pymod;

pub use crate::VP_SCALE;

// Compatibility paths for callers that historically imported these helpers
// through `bridge`; the implementations are now available in pure-Rust
// builds as top-level modules.
pub use crate::{action_features, encode, move_codec};
