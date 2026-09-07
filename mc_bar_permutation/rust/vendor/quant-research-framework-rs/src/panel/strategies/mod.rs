//! Panel-level strategy primitives (Rust mirror).

#![cfg(feature = "panel")]

pub mod long_short;
pub use long_short::{LongShortBasket, momentum_alpha};
