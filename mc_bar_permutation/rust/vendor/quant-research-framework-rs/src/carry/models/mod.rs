//! Funding-signal model library (Rust mirror).

#![cfg(feature = "carry")]

pub mod base;
pub mod persistent_sign;
pub mod momentum;
pub mod oi_cointegration;

pub use base::SignalEmission;
pub use momentum::FundingMomentumModel;
pub use oi_cointegration::FundingOICointegrationModel;
pub use persistent_sign::PersistentFundingSignModel;
