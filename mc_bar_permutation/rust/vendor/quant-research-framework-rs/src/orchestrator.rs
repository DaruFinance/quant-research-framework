//! Walk-forward orchestrator dispatch (Rust mirror).
//!
//! The Python framework's dispatch uses a 5-bool ``RouteKey`` dict; the
//! Rust port mirrors the same concept via an `OrchestratorMode` enum
//! and a parallel `RouteKey` struct. Phase 1 lands the public surface
//! and asserts the existing two single-asset routes resolve correctly;
//! the actual control flow inside `walk_forward` /
//! `walk_forward_regime` stays untouched so parity is preserved
//! byte-for-byte.
//!
//! Phase 2 will unify the two function signatures behind this dispatch
//! when panel routes (`multi_asset = true`) need to share the same
//! entry point.
//!
//! Cargo features `panel`, `pairs`, `carry` (introduced in Phase 2)
//! will register new variants of `OrchestratorMode`; Phase 1's
//! `OrchestratorMode::SingleNoRegime` and `::SingleWithRegime` are the
//! only entries today.

/// Composite key for routing. Mirrors Python's `backtester.orchestrator.RouteKey`.
/// All fields default to false so call sites construct partial keys by
/// initialising only the flag they care about and letting the rest
/// stay False via `..RouteKey::default()`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct RouteKey {
    pub regime: bool,
    pub multi_asset: bool,
    pub multi_leg: bool,
    pub record_costs: bool,
    pub hold_period_set: bool,
}

/// Concrete enum form for the routes that are wired today. Each
/// variant maps to one of the two existing `walk_forward*` functions.
/// Adding `MultiAsset { regime: bool, ... }` is the Phase 2 extension
/// point.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OrchestratorMode {
    /// Single asset, no regime segmentation. Routes to
    /// `walk_forward` in `lib.rs`.
    SingleNoRegime,
    /// Single asset, regime segmentation enabled. Routes to
    /// `walk_forward_regime` in `lib.rs`.
    SingleWithRegime,
}

/// Resolve a `RouteKey` to its concrete `OrchestratorMode`. Phase 1
/// only knows the two single-asset modes; panel / pair / cohort
/// routes added in later phases extend the match arms.
pub fn dispatch(key: RouteKey) -> Result<OrchestratorMode, RouteError> {
    if key.multi_asset {
        return Err(RouteError::NotYetSupported(
            "multi_asset routes ship with the panel plugin"
                .to_string(),
        ));
    }
    if key.regime {
        Ok(OrchestratorMode::SingleWithRegime)
    } else {
        Ok(OrchestratorMode::SingleNoRegime)
    }
}

#[derive(Debug, PartialEq, Eq, Clone)]
pub enum RouteError {
    NotYetSupported(String),
}

impl std::fmt::Display for RouteError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RouteError::NotYetSupported(msg) => write!(f, "route not supported: {}", msg),
        }
    }
}

impl std::error::Error for RouteError {}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_route_is_single_no_regime() {
        let key = RouteKey::default();
        assert_eq!(dispatch(key).unwrap(), OrchestratorMode::SingleNoRegime);
    }

    #[test]
    fn regime_flag_selects_single_with_regime() {
        let key = RouteKey { regime: true, ..Default::default() };
        assert_eq!(dispatch(key).unwrap(), OrchestratorMode::SingleWithRegime);
    }

    #[test]
    fn multi_asset_flag_errors_until_the_panel_plugin_lands() {
        let key = RouteKey { multi_asset: true, ..Default::default() };
        let err = dispatch(key).unwrap_err();
        assert!(format!("{}", err).contains("panel plugin"));
    }

    #[test]
    fn route_key_is_hashable_and_eq() {
        use std::collections::HashSet;
        let mut s = HashSet::new();
        s.insert(RouteKey { regime: true, ..Default::default() });
        s.insert(RouteKey { regime: true, ..Default::default() });
        s.insert(RouteKey { regime: false, ..Default::default() });
        assert_eq!(s.len(), 2);
    }
}
