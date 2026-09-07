//! Long-short basket primitive (Rust mirror).
//!
//! Mirrors ``backtester.panel.strategies.long_short.LongShortBasket``.

#![cfg(feature = "panel")]

use std::collections::HashMap;

use crate::panel::neutralize::{
    estimate_betas, estimate_vols, neutralize_beta, neutralize_dollar,
    neutralize_sigma,
};
use crate::panel::PanelData;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NeutralizeMode {
    Dollar,
    Beta,
    Sigma,
}

/// Standard N-bar momentum factory. Returns a closure that computes
/// the alpha at index `t` using only data at indices `<= t`.
pub fn momentum_alpha(lookback: usize) -> impl Fn(&PanelData, usize) -> Vec<f64> {
    move |panel: &PanelData, t_idx: usize| -> Vec<f64> {
        let close_idx = panel
            .fields
            .iter()
            .position(|f| f == "close")
            .expect("panel must have 'close'");
        let n_assets = panel.assets.len();
        if t_idx < lookback {
            return vec![f64::NAN; n_assets];
        }
        let mut out = vec![0.0f64; n_assets];
        for ai in 0..n_assets {
            let c_t = panel.data[[t_idx, ai, close_idx]];
            let c_back = panel.data[[t_idx - lookback, ai, close_idx]];
            out[ai] = (c_t / c_back) - 1.0;
        }
        out
    }
}

#[derive(Debug)]
pub struct LongShortBasket<AlphaFn: Fn(&PanelData, usize) -> Vec<f64>> {
    pub alpha_fn: AlphaFn,
    pub neutralize_mode: NeutralizeMode,
    pub n_long: usize,
    pub n_short: usize,
    pub market_asset: Option<String>,
    pub returns_lookback: usize,
}

impl<F: Fn(&PanelData, usize) -> Vec<f64>> LongShortBasket<F> {
    pub fn new(
        alpha_fn: F,
        neutralize_mode: NeutralizeMode,
        n_long: usize,
        n_short: usize,
    ) -> Self {
        Self {
            alpha_fn,
            neutralize_mode,
            n_long,
            n_short,
            market_asset: None,
            returns_lookback: 60,
        }
    }

    pub fn with_market_asset(mut self, asset: impl Into<String>) -> Self {
        self.market_asset = Some(asset.into());
        self
    }

    pub fn with_returns_lookback(mut self, n: usize) -> Self {
        self.returns_lookback = n;
        self
    }

    pub fn positions(&self, panel: &PanelData, t_idx: usize) -> Result<HashMap<String, f64>, String> {
        let n_assets = panel.assets.len();
        if self.n_long + self.n_short > n_assets {
            return Err(format!(
                "n_long+n_short={} exceeds n_assets={}",
                self.n_long + self.n_short,
                n_assets
            ));
        }
        let alpha = (self.alpha_fn)(panel, t_idx);
        if alpha.len() != n_assets {
            return Err(format!(
                "alpha returned len {} != n_assets {}",
                alpha.len(),
                n_assets
            ));
        }
        // All-NaN -> all-zero (pre-warmup).
        if alpha.iter().all(|x| x.is_nan()) {
            return Ok(panel.assets.iter().map(|a| (a.clone(), 0.0)).collect());
        }
        // Argsort ascending. NaN -> +inf so they sink past valid longs.
        let mut indexed: Vec<(usize, f64)> = alpha
            .iter()
            .enumerate()
            .map(|(i, &x)| (i, if x.is_nan() { f64::INFINITY } else { x }))
            .collect();
        indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        let long_ids: Vec<usize> =
            indexed.iter().rev().take(self.n_long).map(|(i, _)| *i).collect();
        let short_ids: Vec<usize> = indexed
            .iter()
            .take(self.n_short)
            .map(|(i, _)| *i)
            .filter(|i| !long_ids.contains(i))
            .collect();

        let mut raw = vec![0.0f64; n_assets];
        for i in &long_ids {
            raw[*i] = 1.0;
        }
        for i in &short_ids {
            raw[*i] = -1.0;
        }

        let weights = match self.neutralize_mode {
            NeutralizeMode::Dollar => neutralize_dollar(&raw)?,
            NeutralizeMode::Beta => {
                let market = self.market_asset.as_ref().ok_or_else(|| {
                    "neutralize_mode=Beta requires market_asset".to_string()
                })?;
                let mi = panel
                    .assets
                    .iter()
                    .position(|a| a == market)
                    .ok_or_else(|| {
                        format!("market_asset {:?} not in panel", market)
                    })?;
                let window = self.returns_window(panel, t_idx)?;
                let betas = estimate_betas(&window, mi)?;
                neutralize_beta(&raw, &betas, Some(mi))?
            }
            NeutralizeMode::Sigma => {
                // Sigma requires non-zero weights; zero out unselected legs.
                let window = self.returns_window(panel, t_idx)?;
                let vols = estimate_vols(&window)?;
                let mut selected = Vec::new();
                let mut sel_vols = Vec::new();
                for (i, &r) in raw.iter().enumerate() {
                    if r != 0.0 {
                        selected.push((i, r));
                        sel_vols.push(vols[i]);
                    }
                }
                if selected.is_empty() {
                    raw.clone()
                } else {
                    let raw_sel: Vec<f64> = selected.iter().map(|(_, r)| *r).collect();
                    let w_sel = neutralize_sigma(&raw_sel, &sel_vols)?;
                    let mut full = vec![0.0f64; n_assets];
                    for (k, (idx, _)) in selected.iter().enumerate() {
                        full[*idx] = w_sel[k];
                    }
                    full
                }
            }
        };
        Ok(panel
            .assets
            .iter()
            .zip(weights.iter())
            .map(|(a, w)| (a.clone(), *w))
            .collect())
    }

    fn returns_window(&self, panel: &PanelData, t_idx: usize) -> Result<Vec<Vec<f64>>, String> {
        if t_idx < self.returns_lookback + 1 {
            return Err(format!(
                "t_idx={} insufficient for returns_lookback={}",
                t_idx, self.returns_lookback
            ));
        }
        let close_idx = panel
            .fields
            .iter()
            .position(|f| f == "close")
            .expect("panel must have 'close'");
        let n_assets = panel.assets.len();
        let start = t_idx - self.returns_lookback - 1;
        let mut prev: Vec<f64> = (0..n_assets)
            .map(|ai| panel.data[[start, ai, close_idx]])
            .collect();
        let mut window = Vec::with_capacity(self.returns_lookback);
        for ti in (start + 1)..t_idx {
            let mut row = Vec::with_capacity(n_assets);
            for ai in 0..n_assets {
                let c = panel.data[[ti, ai, close_idx]];
                row.push((c / prev[ai]).ln());
                prev[ai] = c;
            }
            window.push(row);
        }
        Ok(window)
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
    fn exactly_one_long_one_short_per_rebalance() {
        let panel = load_panel(&fixture_paths()).unwrap();
        let basket = LongShortBasket::new(momentum_alpha(20), NeutralizeMode::Dollar, 1, 1);
        for t in [100usize, 250, 500, 750, 999] {
            let w = basket.positions(&panel, t).unwrap();
            let n_long = w.values().filter(|v| **v > 0.0).count();
            let n_short = w.values().filter(|v| **v < 0.0).count();
            assert_eq!(n_long, 1, "t={}: expected 1 long, got {}", t, n_long);
            assert_eq!(n_short, 1, "t={}: expected 1 short, got {}", t, n_short);
        }
    }

    #[test]
    fn dollar_neutral_balances() {
        let panel = load_panel(&fixture_paths()).unwrap();
        let basket = LongShortBasket::new(momentum_alpha(20), NeutralizeMode::Dollar, 1, 1);
        let w = basket.positions(&panel, 500).unwrap();
        let longs: f64 = w.values().filter(|v| **v > 0.0).sum();
        let shorts: f64 = -w.values().filter(|v| **v < 0.0).sum::<f64>();
        assert!((longs - shorts).abs() < 1e-12);
        assert!((longs - 0.5).abs() < 1e-12);
    }

    #[test]
    fn pre_warmup_returns_all_zero() {
        let panel = load_panel(&fixture_paths()).unwrap();
        let basket = LongShortBasket::new(momentum_alpha(20), NeutralizeMode::Dollar, 1, 1);
        let w = basket.positions(&panel, 10).unwrap();
        for v in w.values() {
            assert_eq!(*v, 0.0);
        }
    }

    #[test]
    fn beta_neutral_zeros_market_beta() {
        let panel = load_panel(&fixture_paths()).unwrap();
        let basket = LongShortBasket::new(momentum_alpha(20), NeutralizeMode::Beta, 2, 1)
            .with_market_asset("BTC")
            .with_returns_lookback(60);
        let w = basket.positions(&panel, 500).unwrap();
        // Re-derive betas the way the basket does and check w · β ≈ 0.
        let close_idx = panel.fields.iter().position(|f| f == "close").unwrap();
        let mut prev: Vec<f64> = (0..panel.assets.len())
            .map(|ai| panel.data[[500 - 61, ai, close_idx]])
            .collect();
        let mut window = Vec::new();
        for ti in (500 - 60)..500 {
            let mut row = Vec::new();
            for ai in 0..panel.assets.len() {
                let c = panel.data[[ti, ai, close_idx]];
                row.push((c / prev[ai]).ln());
                prev[ai] = c;
            }
            window.push(row);
        }
        let bi = panel.assets.iter().position(|a| a == "BTC").unwrap();
        let betas = estimate_betas(&window, bi).unwrap();
        let mut port_beta = 0.0;
        for (asset, w_val) in &w {
            let i = panel.assets.iter().position(|a| a == asset).unwrap();
            port_beta += w_val * betas[i];
        }
        assert!(port_beta.abs() < 1e-10);
    }
}
