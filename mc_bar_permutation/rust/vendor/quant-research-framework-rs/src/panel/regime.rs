//! Cross-asset regime detection (Rust mirror).
//!
//! Mirrors the Python ``backtester.panel.regime`` module. Two
//! detector flavours ship:
//!
//! - ``per_asset_regime``: each asset's labels come from its own
//!   EMA-200 + N-bar consistency check. Trivially leak-free across
//!   assets.
//! - ``market_regime(market_asset)``: every asset inherits the
//!   named market asset's label. Leak-free because the market
//!   asset reads only its own past.
//!
//! The ``PanelRegimeDetector`` trait is the plug point; future
//! variants (volatility-quantile, HMM, ML) implement it.

#![cfg(feature = "panel")]

use std::collections::HashMap;

use crate::panel::PanelData;

/// Categorical labels mirrored from the single-asset detector. We
/// keep them as `u8` indices into a label table so the Rust API can
/// hand the result to a `Vec<u8>` regime array without string
/// allocation in the hot loop.
pub const LABEL_RANGING:   u8 = 0;
pub const LABEL_UPTREND:   u8 = 1;
pub const LABEL_DOWNTREND: u8 = 2;
pub const LABEL_NAMES: &[&str] = &["Ranging", "Uptrend", "Downtrend"];

/// Implementors describe one panel regime detector. ``detect`` returns
/// a map asset -> Vec<u8> of label indices, one per panel timestamp.
pub trait PanelRegimeDetector {
    fn detect(&self, panel: &PanelData) -> HashMap<String, Vec<u8>>;
}

fn ema(close: &[f64], span: usize) -> Vec<f64> {
    let alpha = 2.0 / (span as f64 + 1.0);
    let mut out = Vec::with_capacity(close.len());
    let mut prev = close[0];
    out.push(prev);
    for &c in &close[1..] {
        // pandas' ewm constant-run guard; see `compute_ema` in src/lib.rs.
        prev = if prev == c { c } else { alpha * c + (1.0 - alpha) * prev };
        out.push(prev);
    }
    out
}

fn ema_regime(close: &[f64], length: usize) -> Vec<u8> {
    let n = close.len();
    let ema200 = ema(close, 200);
    let mut labels = vec![LABEL_RANGING; n];
    if n < length {
        return labels;
    }
    let above: Vec<u8> = close.iter().zip(ema200.iter())
        .map(|(c, e)| if c > e { 1u8 } else { 0u8 })
        .collect();
    let below: Vec<u8> = close.iter().zip(ema200.iter())
        .map(|(c, e)| if c < e { 1u8 } else { 0u8 })
        .collect();
    for i in length - 1..n {
        let a: u32 = above[i + 1 - length..=i].iter().map(|&v| v as u32).sum();
        let b: u32 = below[i + 1 - length..=i].iter().map(|&v| v as u32).sum();
        if a as usize >= length {
            labels[i] = LABEL_UPTREND;
        } else if b as usize >= length {
            labels[i] = LABEL_DOWNTREND;
        }
    }
    labels
}

/// Per-asset detector. Each asset's labels come from its own close
/// series only, no cross-asset coupling.
pub struct PerAssetRegime {
    pub length: usize,
}
impl PerAssetRegime {
    pub fn new() -> Self { Self { length: 8 } }
}
impl Default for PerAssetRegime {
    fn default() -> Self { Self::new() }
}
impl PanelRegimeDetector for PerAssetRegime {
    fn detect(&self, panel: &PanelData) -> HashMap<String, Vec<u8>> {
        let close_idx = panel.fields.iter().position(|f| f == "close")
            .expect("panel must have 'close' field");
        let mut out = HashMap::new();
        for (ai, asset) in panel.assets.iter().enumerate() {
            let close: Vec<f64> = (0..panel.times.len())
                .map(|ti| panel.data[[ti, ai, close_idx]])
                .collect();
            out.insert(asset.clone(), ema_regime(&close, self.length));
        }
        out
    }
}

/// Market regime: broadcast `market_asset`'s labels to all assets.
pub struct MarketRegime {
    pub market_asset: String,
    pub length: usize,
}
impl MarketRegime {
    pub fn new(market_asset: impl Into<String>) -> Self {
        Self { market_asset: market_asset.into(), length: 8 }
    }
}
impl PanelRegimeDetector for MarketRegime {
    fn detect(&self, panel: &PanelData) -> HashMap<String, Vec<u8>> {
        let mi = panel.assets.iter().position(|a| a == &self.market_asset)
            .unwrap_or_else(|| panic!(
                "MarketRegime: market_asset {:?} not in panel assets {:?}",
                self.market_asset, panel.assets
            ));
        let close_idx = panel.fields.iter().position(|f| f == "close")
            .expect("panel must have 'close' field");
        let market_close: Vec<f64> = (0..panel.times.len())
            .map(|ti| panel.data[[ti, mi, close_idx]])
            .collect();
        let market_labels = ema_regime(&market_close, self.length);
        let mut out = HashMap::new();
        for asset in &panel.assets {
            out.insert(asset.clone(), market_labels.clone());
        }
        out
    }
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
    fn per_asset_emits_label_per_bar_per_asset() {
        let panel = load_panel(&fixture_paths()).unwrap();
        let labels = PerAssetRegime::new().detect(&panel);
        for a in &panel.assets {
            assert_eq!(labels[a].len(), panel.times.len());
        }
    }

    #[test]
    fn market_broadcasts_btc_labels_to_all_assets() {
        let panel = load_panel(&fixture_paths()).unwrap();
        let labels = MarketRegime::new("BTC").detect(&panel);
        let btc = labels.get("BTC").unwrap();
        for a in &panel.assets {
            if a == "BTC" { continue; }
            assert_eq!(labels[a], *btc, "asset {} did not inherit BTC labels", a);
        }
    }

    #[test]
    #[should_panic(expected = "DOGE")]
    fn market_panics_on_missing_market_asset() {
        let panel = load_panel(&fixture_paths()).unwrap();
        MarketRegime::new("DOGE").detect(&panel);
    }

    /// HIGH-RISK 50-T cross-asset pollute battery (Rust mirror of the
    /// Python test). Polluting any one asset's tail must not change
    /// any other asset's labels at indices < cut.
    #[test]
    fn per_asset_no_cross_asset_leak_50t() {
        let panel = load_panel(&fixture_paths()).unwrap();
        let detector = PerAssetRegime::new();
        let clean = detector.detect(&panel);
        let n = panel.times.len();
        // 50 cut points uniformly across the panel beyond the EMA-200
        // warmup. Use simple stride; no RNG needed for a deterministic
        // sweep.
        for k in 0..50 {
            let cut = 220 + (n - 220) * k / 50;
            for victim in &panel.assets {
                let mut polluted = panel.clone();
                let vi = panel.assets.iter().position(|a| a == victim).unwrap();
                let close_idx = panel.fields.iter().position(|f| f == "close").unwrap();
                for ti in cut..n {
                    for fi in 0..polluted.fields.len() {
                        polluted.data[[ti, vi, fi]] = f64::NAN;
                    }
                    let _ = close_idx;
                }
                let poll_labels = detector.detect(&polluted);
                for witness in &panel.assets {
                    if witness == victim { continue; }
                    let cl = &clean[witness];
                    let pl = &poll_labels[witness];
                    for i in 0..cut {
                        assert_eq!(cl[i], pl[i],
                            "cross-asset leak: polluting {} at >= {} changed {} at {}",
                            victim, cut, witness, i);
                    }
                }
            }
        }
    }
}
