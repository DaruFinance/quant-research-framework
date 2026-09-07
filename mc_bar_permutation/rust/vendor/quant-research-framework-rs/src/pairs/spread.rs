//! Spread-definition primitives (Phase 3 T2, Rust mirror).
//!
//! Closed-form ports of `log_ratio`, `ols_resid`, `kalman_beta_spread`.
//! Each consumes panel cells at row indices `<= t_idx` only.
//!
//! Parity (vs. Python `backtester.pairs.spread`):
//! - `log_ratio`        : bit-equal under f64.
//! - `ols_resid`        : closed-form OLS via the same normal equations
//!   numpy.polyfit(deg=1) implements; agrees to
//!   ~1e-12 in practice on financial-scale data.
//! - `kalman_beta_spread`: deterministic float math; agrees to ~1e-12.

#![cfg(feature = "pairs")]

use crate::panel::PanelData;

#[derive(Debug, Clone)]
pub struct SpreadResult {
    pub spread: Vec<f64>,
    /// Static beta for log_ratio (1.0) / ols_resid (window-final fit);
    /// for the dynamic-beta methods this is the per-bar trajectory.
    pub beta_scalar: Option<f64>,
    pub beta_traj: Option<Vec<f64>>,
    pub method: &'static str,
    pub asset_a: String,
    pub asset_b: String,
}

fn close_series(panel: &PanelData, asset: &str) -> Result<Vec<f64>, String> {
    let close_idx = panel
        .fields
        .iter()
        .position(|f| f == "close")
        .ok_or_else(|| "panel missing 'close' field".to_string())?;
    let asset_idx = panel
        .assets
        .iter()
        .position(|a| a == asset)
        .ok_or_else(|| format!("panel missing asset '{}'", asset))?;
    Ok((0..panel.times.len())
        .map(|t| panel.data[[t, asset_idx, close_idx]])
        .collect())
}

/// `spread[i] = ln(close_a[i]) - ln(close_b[i])`.
pub fn log_ratio(
    panel: &PanelData,
    asset_a: &str,
    asset_b: &str,
    t_idx: usize,
) -> Result<SpreadResult, String> {
    let a = close_series(panel, asset_a)?;
    let b = close_series(panel, asset_b)?;
    let n = t_idx + 1;
    if n > a.len() || n > b.len() {
        return Err(format!(
            "log_ratio: t_idx={} out of range (n_times={})",
            t_idx,
            a.len()
        ));
    }
    let spread: Vec<f64> = (0..n).map(|i| a[i].ln() - b[i].ln()).collect();
    Ok(SpreadResult {
        spread,
        beta_scalar: Some(1.0),
        beta_traj: None,
        method: "log_ratio",
        asset_a: asset_a.to_string(),
        asset_b: asset_b.to_string(),
    })
}

/// Closed-form OLS slope+intercept on `(x, y)`, mean-centered for
/// numerical stability.  Mathematically equivalent to numpy.polyfit
/// but agrees with it to ~1e-12 on ill-conditioned inputs (e.g.
/// log-prices that are slow-varying).  The naive normal-equation
/// form (Σx*Σy / Σx² shape) loses ~5–7 digits when cond(X'X) is
/// large; the centered form keeps cond(X) near 1.
fn ols_slope_intercept(x: &[f64], y: &[f64]) -> Result<(f64, f64), String> {
    if x.len() != y.len() || x.len() < 2 {
        return Err(format!(
            "ols: need len(x) == len(y) >= 2 (got {}, {})",
            x.len(),
            y.len()
        ));
    }
    let n = x.len() as f64;
    let mean_x: f64 = x.iter().sum::<f64>() / n;
    let mean_y: f64 = y.iter().sum::<f64>() / n;
    let mut num = 0.0f64;
    let mut den = 0.0f64;
    for (xi, yi) in x.iter().zip(y.iter()) {
        let dx = xi - mean_x;
        num += dx * (yi - mean_y);
        den += dx * dx;
    }
    if den == 0.0 {
        return Err("ols: zero variance in x".to_string());
    }
    let slope = num / den;
    let intercept = mean_y - slope * mean_x;
    Ok((slope, intercept))
}

/// Rolling OLS-residual spread.  For each bar `i in [lookback-1, n)`,
/// regress `ln(close_a) ~ alpha + beta * ln(close_b)` on the
/// `lookback`-bar window ending at `i` and emit the residual.
pub fn ols_resid(
    panel: &PanelData,
    asset_a: &str,
    asset_b: &str,
    t_idx: usize,
    lookback: usize,
) -> Result<SpreadResult, String> {
    let a = close_series(panel, asset_a)?;
    let b = close_series(panel, asset_b)?;
    let n = t_idx + 1;
    if n > a.len() {
        return Err(format!("ols_resid: t_idx={} out of range", t_idx));
    }
    let mut spread = vec![f64::NAN; n];
    if n < lookback {
        return Ok(SpreadResult {
            spread,
            beta_scalar: Some(f64::NAN),
            beta_traj: None,
            method: "ols_resid",
            asset_a: asset_a.to_string(),
            asset_b: asset_b.to_string(),
        });
    }
    let log_a: Vec<f64> = a[..n].iter().map(|v| v.ln()).collect();
    let log_b: Vec<f64> = b[..n].iter().map(|v| v.ln()).collect();
    for i in (lookback - 1)..n {
        let s = i + 1 - lookback;
        let e = i + 1;
        let (beta, alpha) = ols_slope_intercept(&log_b[s..e], &log_a[s..e])?;
        spread[i] = log_a[i] - alpha - beta * log_b[i];
    }
    let s_final = n - lookback;
    let (beta_final, _) = ols_slope_intercept(&log_b[s_final..n], &log_a[s_final..n])?;
    Ok(SpreadResult {
        spread,
        beta_scalar: Some(beta_final),
        beta_traj: None,
        method: "ols_resid",
        asset_a: asset_a.to_string(),
        asset_b: asset_b.to_string(),
    })
}

/// Dynamic-β Kalman filter on `log_a = alpha + beta * log_b + noise`,
/// 2-D random-walk state.  `delta` scales the state covariance,
/// `observation_var` is the measurement noise.
pub fn kalman_beta_spread(
    panel: &PanelData,
    asset_a: &str,
    asset_b: &str,
    t_idx: usize,
    delta: f64,
    observation_var: f64,
) -> Result<SpreadResult, String> {
    let a = close_series(panel, asset_a)?;
    let b = close_series(panel, asset_b)?;
    let n = t_idx + 1;
    if n > a.len() {
        return Err(format!("kalman_beta_spread: t_idx={} out of range", t_idx));
    }
    let log_a: Vec<f64> = a[..n].iter().map(|v| v.ln()).collect();
    let log_b: Vec<f64> = b[..n].iter().map(|v| v.ln()).collect();

    // State: [alpha, beta]. Covariance P (2x2). Process noise Q = delta*I.
    let mut state = [0.0f64, 0.0];
    let mut p = [[1.0f64, 0.0], [0.0, 1.0]];
    let q = [[delta, 0.0], [0.0, delta]];
    let mut spread = vec![f64::NAN; n];
    let mut beta_traj = vec![f64::NAN; n];

    for i in 0..n {
        let h = [1.0, log_b[i]];
        // Predict: P = P + Q
        for r in 0..2 {
            for c in 0..2 {
                p[r][c] += q[r][c];
            }
        }
        // Innovation
        let y_hat = h[0] * state[0] + h[1] * state[1];
        let v = log_a[i] - y_hat;
        // S = H P H^T + obs_var (scalar)
        let ph = [
            p[0][0] * h[0] + p[0][1] * h[1],
            p[1][0] * h[0] + p[1][1] * h[1],
        ];
        let s = h[0] * ph[0] + h[1] * ph[1] + observation_var;
        // K = P H^T / S (2x1)
        let k = [ph[0] / s, ph[1] / s];
        // state += K * v
        state[0] += k[0] * v;
        state[1] += k[1] * v;
        // P = P - K H P  (note: K H is 2x2 outer product; (K H) P is 2x2)
        // outer = K * H : [[k0*h0, k0*h1],[k1*h0, k1*h1]]
        let outer = [
            [k[0] * h[0], k[0] * h[1]],
            [k[1] * h[0], k[1] * h[1]],
        ];
        // outer @ P
        let mut op = [[0.0f64; 2]; 2];
        for r in 0..2 {
            for c in 0..2 {
                op[r][c] = outer[r][0] * p[0][c] + outer[r][1] * p[1][c];
            }
        }
        for r in 0..2 {
            for c in 0..2 {
                p[r][c] -= op[r][c];
            }
        }
        spread[i] = log_a[i] - state[0] - state[1] * log_b[i];
        beta_traj[i] = state[1];
    }
    Ok(SpreadResult {
        spread,
        beta_scalar: None,
        beta_traj: Some(beta_traj),
        method: "kalman_beta",
        asset_a: asset_a.to_string(),
        asset_b: asset_b.to_string(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fake_panel(closes: Vec<Vec<f64>>) -> PanelData {
        // closes is shape (n_assets, n_times); transpose into (t, asset, field).
        let n_assets = closes.len();
        let n_times = closes[0].len();
        let assets = (0..n_assets).map(|i| format!("A{}", i)).collect();
        let fields = vec!["close".to_string()];
        let mut data =
            ndarray::Array3::<f64>::zeros((n_times, n_assets, 1));
        for (ai, series) in closes.iter().enumerate() {
            for (ti, v) in series.iter().enumerate() {
                data[[ti, ai, 0]] = *v;
            }
        }
        PanelData {
            times: (0..n_times as i64).collect(),
            assets,
            fields,
            data,
            interval_seconds: 3600,
        }
    }

    #[test]
    fn log_ratio_matches_manual() {
        let panel = fake_panel(vec![vec![1.0, 2.0, 4.0], vec![1.0, 1.0, 2.0]]);
        let r = log_ratio(&panel, "A0", "A1", 2).unwrap();
        assert!((r.spread[0] - 0.0_f64).abs() < 1e-12);
        assert!((r.spread[1] - (2.0_f64.ln() - 1.0_f64.ln())).abs() < 1e-12);
        assert!((r.spread[2] - (4.0_f64.ln() - 2.0_f64.ln())).abs() < 1e-12);
        assert_eq!(r.beta_scalar, Some(1.0));
    }

    #[test]
    fn ols_slope_intercept_matches_polyfit() {
        // y = 2x + 3 exactly
        let x: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let y: Vec<f64> = x.iter().map(|v| 2.0 * v + 3.0).collect();
        let (slope, intercept) = ols_slope_intercept(&x, &y).unwrap();
        assert!((slope - 2.0).abs() < 1e-12);
        assert!((intercept - 3.0).abs() < 1e-12);
    }

    #[test]
    fn ols_resid_warmup_is_nan_then_zero_for_perfect_fit() {
        // Perfect linear cointegration: log_a = 2 * log_b => residuals 0.
        let n = 100;
        let b: Vec<f64> = (1..=n).map(|i| i as f64).collect();
        let a: Vec<f64> = b.iter().map(|v| v.powi(2)).collect();
        let panel = fake_panel(vec![a, b]);
        let r = ols_resid(&panel, "A0", "A1", n - 1, 60).unwrap();
        // First 59 residuals are NaN (warmup).
        for i in 0..59 {
            assert!(r.spread[i].is_nan());
        }
        // Subsequent residuals are exactly 0 to numeric noise.
        for i in 59..n {
            assert!(r.spread[i].abs() < 1e-9, "i={} spread={}", i, r.spread[i]);
        }
        assert!((r.beta_scalar.unwrap() - 2.0).abs() < 1e-9);
    }

    #[test]
    fn kalman_beta_converges_on_constant_relationship() {
        // log_a = 1 + 0.5 * log_b, no noise.  After warmup the
        // estimated beta should approach 0.5.
        let n = 200;
        let b: Vec<f64> = (1..=n).map(|i| (i as f64) * 1.001).collect();
        let log_a: Vec<f64> = b.iter().map(|v| 1.0 + 0.5 * v.ln()).collect();
        let a: Vec<f64> = log_a.iter().map(|v| v.exp()).collect();
        let panel = fake_panel(vec![a, b]);
        let r = kalman_beta_spread(&panel, "A0", "A1", n - 1, 1e-4, 1e-3).unwrap();
        let traj = r.beta_traj.unwrap();
        assert!(
            (traj[n - 1] - 0.5).abs() < 0.05,
            "kalman beta did not converge: final={}",
            traj[n - 1]
        );
    }
}
