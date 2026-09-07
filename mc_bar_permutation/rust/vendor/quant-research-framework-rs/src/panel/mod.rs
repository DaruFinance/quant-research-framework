//! Multi-asset panel plugin (Phase 2). Behind `#[cfg(feature = "panel")]`.
//!
//! Mirror of Python's ``backtester.panel`` package. Lands the
//! data loader; subsequent Phase 2 items (#4 cross-asset regime,
//! #5(iter) basket orchestrator, #6-#8 sizing/neutralisation, #44-#45
//! objective+constraints) plug into the same module.

#![cfg(feature = "panel")]

pub mod constraints;
pub mod loader;
pub mod neutralize;
pub mod orchestrator;
pub mod regime;
pub mod sizing;
pub mod strategies;
pub use constraints::apply_constraints;
pub use loader::{PanelData, PanelError, load_panel};
pub use neutralize::{Mode as NeutralizeMode, estimate_betas, estimate_vols};
pub use orchestrator::bars_for_asset;
pub use regime::{
    PanelRegimeDetector, PerAssetRegime, MarketRegime,
    LABEL_RANGING, LABEL_UPTREND, LABEL_DOWNTREND, LABEL_NAMES,
};
pub use sizing::{equal_weights, erc_weights, risk_contributions};
pub use strategies::{LongShortBasket, momentum_alpha};
