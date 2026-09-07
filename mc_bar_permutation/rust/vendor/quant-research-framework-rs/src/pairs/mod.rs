//! Pairs / stat-arb plugin.
//!
//! Mirror of Python's ``backtester.pairs`` package.  All primitives
//! consume only data at indices ``<= t`` for any decision at logical
//! time ``t``.
//!
//! - `spread`    : log_ratio, ols_resid, kalman_beta_spread (#10).
//! - `eligibility`: half_life_ou, is_eligible_pair (#13).
//! - `screener`  : engle_granger, distance_ssd, screen_pairs (#9).
//! - `cadence`   : re-estimation engine (#11, HIGH-RISK).
//! - `stops`     : z_multiple, half_life_multiple, breakdown (#12).
//!
//! `pca_resid` and `ml_resid` from the Python side are **not** ported.
//! Both depend on sklearn-equivalent stacks (eigendecomposition of a
//! rolling cov, generic predictor protocol) that don't fit cleanly
//! cross-language.  Re-add when a need lands; until then the parity
//! coverage is for the closed-form spread methods only.

#![cfg(feature = "pairs")]

pub mod spread;
pub mod eligibility;
pub mod screener;
pub mod cadence;
pub mod stops;

pub use spread::{
    log_ratio, ols_resid, kalman_beta_spread, SpreadResult,
};
pub use eligibility::{
    half_life_ou, is_eligible_pair, EligibilityCriteria,
};
pub use screener::{
    engle_granger, distance_ssd, screen_pairs, ScreenedPair, ScreenMethod,
};
pub use cadence::{Cadence, CadenceEngine, CadenceMode};
pub use stops::{
    z_multiple_stop, half_life_multiple_stop, breakdown_trigger_stop,
    StopDecision, StopReason,
};
