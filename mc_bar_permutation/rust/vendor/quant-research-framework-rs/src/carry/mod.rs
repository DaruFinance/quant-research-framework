//! Carry / basis / funding plugin (Rust mirror).
//!
//! Mirror of Python's ``backtester.carry`` package.  All loaders read
//! CSV (the parity scripts always shake out CSV first; parquet adds a
//! heavyweight cross-language dep we don't need here).
//!
//! Public surface tracks the Python side 1:1:
//! - `funding`  : load_funding, next_funding_time, rate_at, FUNDING_INTERVAL_S (#38).
//! - `basis`    : load_basis, basis_at (#39).
//! - `triggers` : FundingFlipTrigger, BasisBlowoutTrigger, TriggerEvent (#39s).
//! - `oi`       : load_oi, oi_at (#40).
//! - `onchain`  : load_onchain, value_at (#41).
//! - `scheduler`: EventDrivenScheduler, ScheduledRebalance (#42).
//! - `models`   : PersistentFundingSign, FundingMomentum, FundingOICointegration (#43).

#![cfg(feature = "carry")]

pub mod funding;
pub mod basis;
pub mod triggers;
pub mod oi;
pub mod onchain;
pub mod scheduler;
pub mod models;

pub use funding::{load_funding, next_funding_time, rate_at, FundingEvent, FundingFrame, FUNDING_INTERVAL_S};
pub use basis::{basis_at, load_basis, BasisFrame, BasisRecord};
pub use triggers::{BasisBlowoutTrigger, FundingFlipTrigger, TriggerEvent};
pub use oi::{load_oi, oi_at, OIFrame, OIRecord};
pub use onchain::{load_onchain, value_at, OnChainFrame, OnChainSnapshot};
pub use scheduler::{EventDrivenScheduler, RebalanceKind, ScheduledRebalance};
pub use models::{
    FundingMomentumModel, FundingOICointegrationModel,
    PersistentFundingSignModel, SignalEmission,
};
