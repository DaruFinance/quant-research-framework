//! Spread re-estimation cadence engine (HIGH-RISK Rust mirror).
//!
//! Drives β refits over a panel.  Three modes mirror the Python side:
//! `Bars` (every-N), `Trigger` (user-supplied predicate), `OnBreakdown`
//! (built-in z-score breakout cap).  Each refit uses only data at
//! row indices `<= refit_bar`.

#![cfg(feature = "pairs")]

use crate::panel::PanelData;
use crate::pairs::spread::SpreadResult;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CadenceMode {
    Bars,
    Trigger,
    OnBreakdown,
}

pub type TriggerFn = std::sync::Arc<dyn Fn(&[f64], usize) -> bool + Send + Sync>;

#[derive(Clone)]
pub struct Cadence {
    pub mode: CadenceMode,
    pub every: usize,
    /// `(spread_so_far, t_idx) -> bool`.  Required for `Trigger` mode;
    /// ignored otherwise.
    pub trigger_fn: Option<TriggerFn>,
}

impl Default for Cadence {
    fn default() -> Self {
        Self {
            mode: CadenceMode::Bars,
            every: 100,
            trigger_fn: None,
        }
    }
}

impl std::fmt::Debug for Cadence {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Cadence")
            .field("mode", &self.mode)
            .field("every", &self.every)
            .field("trigger_fn", &self.trigger_fn.is_some())
            .finish()
    }
}

/// Spread function signature: `(panel, asset_a, asset_b, t_idx) -> SpreadResult`.
pub type SpreadFn = std::sync::Arc<
    dyn Fn(&PanelData, &str, &str, usize) -> Result<SpreadResult, String> + Send + Sync,
>;

#[derive(Clone)]
pub struct CadenceEngine {
    pub spread_fn: SpreadFn,
    pub cadence: Cadence,
}

impl CadenceEngine {
    pub fn new(spread_fn: SpreadFn, cadence: Cadence) -> Self {
        Self { spread_fn, cadence }
    }

    /// Iterate refit bars from `t_start` to `t_end` (inclusive on both
    /// ends).  Returns `(refit_idx, SpreadResult)` pairs in fire order.
    /// First refit is always at `t_start`.
    pub fn run(
        &self,
        panel: &PanelData,
        asset_a: &str,
        asset_b: &str,
        t_start: usize,
        t_end: usize,
    ) -> Result<Vec<(usize, SpreadResult)>, String> {
        let mut results = vec![(
            t_start,
            (self.spread_fn)(panel, asset_a, asset_b, t_start)?,
        )];
        match self.cadence.mode {
            CadenceMode::Bars => {
                let mut t = t_start + self.cadence.every;
                while t <= t_end {
                    results.push((t, (self.spread_fn)(panel, asset_a, asset_b, t)?));
                    t += self.cadence.every;
                }
            }
            CadenceMode::Trigger => {
                let trig = self
                    .cadence
                    .trigger_fn
                    .clone()
                    .ok_or_else(|| "trigger mode requires trigger_fn".to_string())?;
                for t in (t_start + 1)..=t_end {
                    let latest = &results.last().unwrap().1;
                    let prefix: Vec<f64> = latest
                        .spread
                        .iter()
                        .take(t + 1)
                        .copied()
                        .collect();
                    if trig(&prefix, t) {
                        results
                            .push((t, (self.spread_fn)(panel, asset_a, asset_b, t)?));
                    }
                }
            }
            CadenceMode::OnBreakdown => {
                let min_gap = 50usize;
                let mut last_refit = t_start;
                for t in (t_start + 1)..=t_end {
                    if t - last_refit < min_gap {
                        continue;
                    }
                    let latest = &results.last().unwrap().1;
                    let lo = t.saturating_sub(60);
                    let seg: Vec<f64> = latest.spread[lo..=t]
                        .iter()
                        .filter(|v| !v.is_nan())
                        .copied()
                        .collect();
                    if seg.len() < 10 {
                        continue;
                    }
                    let body = &seg[..seg.len() - 1];
                    let mean: f64 = body.iter().sum::<f64>() / body.len() as f64;
                    // numpy std() defaults to ddof=0.
                    let var: f64 = body
                        .iter()
                        .map(|v| (v - mean).powi(2))
                        .sum::<f64>()
                        / body.len() as f64;
                    let std = var.sqrt() + 1e-12;
                    let z = (seg[seg.len() - 1] - mean) / std;
                    if z.abs() > 3.0 {
                        results.push((t, (self.spread_fn)(panel, asset_a, asset_b, t)?));
                        last_refit = t;
                    }
                }
            }
        }
        Ok(results)
    }
}
