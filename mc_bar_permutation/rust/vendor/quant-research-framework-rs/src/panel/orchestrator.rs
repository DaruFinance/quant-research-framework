//! Panel walk-forward orchestrator (Rust mirror).
//!
//! Phase 2 contribution from the Rust side is intentionally minimum-
//! viable: the data-shape mirror, not the full per-asset WFO. The
//! Python side does the heavy WFO lifting via
//! ``backtester.panel.orchestrator.walk_forward_panel`` so the dispatch
//! iter's verification gate (per-asset ledger bit-identical to single-
//! asset run) is exercised end-to-end on the Python path.
//!
//! What lands here:
//! - ``bars_for_asset`` extracts a ``Vec<Bar>`` for one asset out of a
//!   ``PanelData``. This is the building block every Phase 2 Rust item
//!   (sizing, neutralization, basket, multi-term IS) will consume.
//! - ``RouteKey { multi_asset: true, .. }`` is already a dispatch key
//!   in the Phase 1 orchestrator enum (``OrchestratorMode``); the
//!   dispatcher currently surfaces NotYetSupported with a Phase-2-
//!   pending message. When the per-asset Rust WFO ships, that arm
//!   becomes a real route.

#![cfg(feature = "panel")]

use crate::panel::PanelData;
use crate::Bar;

/// Extract one asset's bar series from a panel, in chronological
/// order, ready to feed the existing single-asset `walk_forward` /
/// `walk_forward_regime` once their signatures are made pub(crate)
/// in a later Phase 2 item.
pub fn bars_for_asset(panel: &PanelData, asset: &str) -> Result<Vec<Bar>, String> {
    let ai = panel
        .assets
        .iter()
        .position(|a| a == asset)
        .ok_or_else(|| {
            format!(
                "asset {:?} not in panel; have {:?}",
                asset, panel.assets
            )
        })?;
    let close_idx = panel
        .fields
        .iter()
        .position(|f| f == "close")
        .ok_or_else(|| "panel missing 'close' field".to_string())?;
    let open_idx = panel.fields.iter().position(|f| f == "open").unwrap();
    let high_idx = panel.fields.iter().position(|f| f == "high").unwrap();
    let low_idx = panel.fields.iter().position(|f| f == "low").unwrap();
    // volume is OPTIONAL in the panel; Option, never .unwrap() (B3 fix).
    let volume_idx = panel.fields.iter().position(|f| f == "volume");

    let n = panel.times.len();
    let mut bars = Vec::with_capacity(n);
    for ti in 0..n {
        bars.push(Bar {
            time_unix: panel.times[ti],
            open: panel.data[[ti, ai, open_idx]],
            high: panel.data[[ti, ai, high_idx]],
            low: panel.data[[ti, ai, low_idx]],
            close: panel.data[[ti, ai, close_idx]],
            volume: volume_idx.map(|vi| panel.data[[ti, ai, vi]]).unwrap_or(0.0),
        });
    }
    Ok(bars)
}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::panel::loader::load_panel;
    use std::path::PathBuf;

    fn fixture_paths() -> Vec<(String, PathBuf)> {
        let base = std::env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR not set")
            + "/tests/fixtures/sources";
        vec![
            ("SOL".to_string(), format!("{}/SOLUSDT_1h_30000_31000.csv", base).into()),
            ("BTC".to_string(), format!("{}/BTCUSDT_1h_jan_feb_2024.csv", base).into()),
            ("ETH".to_string(), format!("{}/ETHUSDT_1h_jan_feb_2024.csv", base).into()),
        ]
    }

    #[test]
    fn bars_for_asset_matches_panel_close() {
        let panel = load_panel(&fixture_paths()).unwrap();
        let close_idx = panel.fields.iter().position(|f| f == "close").unwrap();
        for (ai, asset) in panel.assets.iter().enumerate() {
            let bars = bars_for_asset(&panel, asset).unwrap();
            assert_eq!(bars.len(), panel.times.len());
            for (ti, bar) in bars.iter().enumerate() {
                assert_eq!(bar.time_unix, panel.times[ti]);
                assert_eq!(bar.close, panel.data[[ti, ai, close_idx]]);
            }
        }
    }

    #[test]
    fn bars_for_asset_rejects_unknown_asset() {
        let panel = load_panel(&fixture_paths()).unwrap();
        match bars_for_asset(&panel, "DOGE") {
            Ok(_) => panic!("expected error for missing asset"),
            Err(msg) => assert!(msg.contains("DOGE"), "msg was {:?}", msg),
        }
    }
}
