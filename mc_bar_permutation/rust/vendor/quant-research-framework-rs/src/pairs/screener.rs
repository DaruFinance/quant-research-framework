//! Pair / spread screener (Phase 3 T2, HIGH-RISK Rust mirror).
//!
//! Mirrors `backtester.pairs.screener` with one deliberate deviation:
//! the Rust `engle_granger` returns the **ADF tau test statistic**
//! (not a p-value).  The Python production code returns the
//! statsmodels MacKinnon p-value, which has cross-impl table
//! differences too noisy for a 1e-9 parity claim; the test statistic
//! is closed-form OLS and bit-equal across languages when the Python
//! caller passes `maxlag=0, autolag=None`.  The parity script
//! `tools/parity_pairs.py` does exactly that.
//!
//! `screen_pairs` sorts by ascending statistic (more-negative tau is
//! stronger evidence of cointegration; lower distance is closer
//! pairs), so the rank-order across the language boundary is the same
//! even though the absolute statistic differs.

#![cfg(feature = "pairs")]

use crate::panel::PanelData;

#[derive(Debug, Clone)]
pub struct ScreenedPair {
    pub asset_a: String,
    pub asset_b: String,
    pub method: ScreenMethod,
    pub statistic: f64,
    /// `beta` for engle_granger; empty for distance_ssd.
    pub beta: Option<f64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScreenMethod {
    EngleGranger,
    DistanceSsd,
}

impl ScreenMethod {
    pub fn name(self) -> &'static str {
        match self {
            ScreenMethod::EngleGranger => "engle_granger",
            ScreenMethod::DistanceSsd => "distance_ssd",
        }
    }
}

/// Engle-Granger cointegration step on `(close_a, close_b)`.
/// Returns `(adf_tau, ols_beta)` where `adf_tau` is the t-statistic
/// of the lag-1 coefficient in the no-constant, no-augmented-lag
/// Dickey-Fuller regression on the OLS residuals.
///
/// Matches Python `adfuller(resid, regression='n', maxlag=0, autolag=None)`
/// to f64 numerical noise (~1e-12 in practice).
pub fn engle_granger(close_a: &[f64], close_b: &[f64]) -> Result<(f64, f64), String> {
    if close_a.len() != close_b.len() {
        return Err(format!(
            "engle_granger: len mismatch ({} vs {})",
            close_a.len(),
            close_b.len()
        ));
    }
    if close_a.len() < 4 {
        return Err("engle_granger: need >= 4 observations".to_string());
    }
    let log_a: Vec<f64> = close_a.iter().map(|v| v.ln()).collect();
    let log_b: Vec<f64> = close_b.iter().map(|v| v.ln()).collect();
    // OLS log_a ~ alpha + beta * log_b in mean-centered form for
    // numerical stability.  The naive normal-equation form loses
    // 5-7 f64 digits on ill-conditioned log-price inputs (slow-
    // varying second column => cond(X'X) huge).  Centering keeps
    // cond(X) near 1 and matches numpy.polyfit to ~1e-12.
    let n = log_b.len() as f64;
    let mean_x: f64 = log_b.iter().sum::<f64>() / n;
    let mean_y: f64 = log_a.iter().sum::<f64>() / n;
    let mut num = 0.0f64;
    let mut den = 0.0f64;
    for (xi, yi) in log_b.iter().zip(log_a.iter()) {
        let dx = xi - mean_x;
        num += dx * (yi - mean_y);
        den += dx * dx;
    }
    if den == 0.0 {
        return Err("engle_granger: zero variance in log_b".to_string());
    }
    let beta = num / den;
    let alpha = mean_y - beta * mean_x;
    let resid: Vec<f64> = log_a
        .iter()
        .zip(log_b.iter())
        .map(|(y, x)| y - alpha - beta * x)
        .collect();

    // Dickey-Fuller (no constant, lag 0): regress Δy_t on y_{t-1}.
    let n_resid = resid.len();
    let dy: Vec<f64> = (1..n_resid).map(|i| resid[i] - resid[i - 1]).collect();
    let y_lag: Vec<f64> = resid[..n_resid - 1].to_vec();
    let sxx_lag: f64 = y_lag.iter().map(|v| v * v).sum();
    if sxx_lag == 0.0 {
        return Err("engle_granger: zero variance in lagged residual".to_string());
    }
    let sxy_lag: f64 = y_lag.iter().zip(dy.iter()).map(|(x, y)| x * y).sum();
    let rho = sxy_lag / sxx_lag;
    let resid_df: Vec<f64> = dy
        .iter()
        .zip(y_lag.iter())
        .map(|(d, y)| d - rho * y)
        .collect();
    let n_obs = dy.len();
    // OLS variance with k=1 regressor => dof = n - 1.
    let rss: f64 = resid_df.iter().map(|v| v * v).sum();
    let sigma2 = rss / (n_obs as f64 - 1.0);
    let se = (sigma2 / sxx_lag).sqrt();
    let tau = rho / se;
    Ok((tau, beta))
}

/// Sum of squared deviations between z-scored log-prices.
pub fn distance_ssd(close_a: &[f64], close_b: &[f64]) -> Result<f64, String> {
    if close_a.len() != close_b.len() || close_a.is_empty() {
        return Err("distance_ssd: empty or mismatched series".to_string());
    }
    let log_a: Vec<f64> = close_a.iter().map(|v| v.ln()).collect();
    let log_b: Vec<f64> = close_b.iter().map(|v| v.ln()).collect();
    let n = log_a.len() as f64;
    let mean_a: f64 = log_a.iter().sum::<f64>() / n;
    let mean_b: f64 = log_b.iter().sum::<f64>() / n;
    // numpy std() defaults to ddof=0.
    let var_a: f64 = log_a.iter().map(|v| (v - mean_a).powi(2)).sum::<f64>() / n;
    let var_b: f64 = log_b.iter().map(|v| (v - mean_b).powi(2)).sum::<f64>() / n;
    let std_a = var_a.sqrt();
    let std_b = var_b.sqrt();
    let za: Vec<f64> = log_a.iter().map(|v| (v - mean_a) / std_a).collect();
    let zb: Vec<f64> = log_b.iter().map(|v| (v - mean_b) / std_b).collect();
    Ok(za.iter().zip(zb.iter()).map(|(a, b)| (a - b).powi(2)).sum())
}

/// Score every ordered pair `(a, b)` with `a < b` over the
/// `lookback` bars ending at `t_idx`.  Returns a Vec sorted by
/// ascending statistic.
pub fn screen_pairs(
    panel: &PanelData,
    t_idx: usize,
    method: ScreenMethod,
    lookback: usize,
    top_n: Option<usize>,
) -> Result<Vec<ScreenedPair>, String> {
    let n_assets = panel.assets.len();
    if t_idx + 1 < lookback {
        return Ok(Vec::new());
    }
    let start = t_idx + 1 - lookback;
    let end = t_idx + 1;
    let close_idx = panel
        .fields
        .iter()
        .position(|f| f == "close")
        .ok_or_else(|| "panel missing 'close' field".to_string())?;
    let mut close_window: Vec<Vec<f64>> = Vec::with_capacity(n_assets);
    for ai in 0..n_assets {
        close_window.push(
            (start..end).map(|t| panel.data[[t, ai, close_idx]]).collect(),
        );
    }

    let mut results = Vec::new();
    for i in 0..n_assets {
        for j in (i + 1)..n_assets {
            let a = &panel.assets[i];
            let b = &panel.assets[j];
            match method {
                ScreenMethod::EngleGranger => {
                    match engle_granger(&close_window[i], &close_window[j]) {
                        Ok((tau, beta)) => results.push(ScreenedPair {
                            asset_a: a.clone(),
                            asset_b: b.clone(),
                            method,
                            statistic: tau,
                            beta: Some(beta),
                        }),
                        Err(_) => results.push(ScreenedPair {
                            asset_a: a.clone(),
                            asset_b: b.clone(),
                            method,
                            statistic: f64::INFINITY,
                            beta: None,
                        }),
                    }
                }
                ScreenMethod::DistanceSsd => {
                    match distance_ssd(&close_window[i], &close_window[j]) {
                        Ok(d) => results.push(ScreenedPair {
                            asset_a: a.clone(),
                            asset_b: b.clone(),
                            method,
                            statistic: d,
                            beta: None,
                        }),
                        Err(_) => results.push(ScreenedPair {
                            asset_a: a.clone(),
                            asset_b: b.clone(),
                            method,
                            statistic: f64::INFINITY,
                            beta: None,
                        }),
                    }
                }
            }
        }
    }
    // Stable ascending sort. NaN treated as max.
    results.sort_by(|a, b| {
        a.statistic
            .partial_cmp(&b.statistic)
            .unwrap_or(std::cmp::Ordering::Greater)
    });
    if let Some(k) = top_n {
        results.truncate(k);
    }
    Ok(results)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn engle_granger_strongly_negative_on_cointegrated_series() {
        // Two cointegrated series: log_a = 0.5 + 0.8 * log_b + AR(1)
        // residual with rho=-0.6 (strong mean reversion).  The AR(1)
        // residual makes the OLS-residual time series fast-decaying,
        // so the DF tau is very negative.  Use a deterministic
        // alternating shock so the test is reproducible.
        let n = 300;
        let mut log_b = vec![0.0f64; n];
        for i in 1..n {
            log_b[i] = log_b[i - 1] + 0.01;
        }
        // Build AR(1) shock series with deterministic alternating sign
        //, fast mean-reverting around zero.
        let mut e = vec![0.0f64; n];
        for i in 1..n {
            let shock = if i % 2 == 0 { 0.005 } else { -0.005 };
            e[i] = -0.6 * e[i - 1] + shock;
        }
        let log_a: Vec<f64> = log_b
            .iter()
            .zip(e.iter())
            .map(|(lb, ee)| 0.5 + 0.8 * lb + ee)
            .collect();
        let a: Vec<f64> = log_a.iter().map(|v| v.exp()).collect();
        let b: Vec<f64> = log_b.iter().map(|v| v.exp()).collect();
        let (tau, beta) = engle_granger(&a, &b).unwrap();
        assert!((beta - 0.8).abs() < 0.01, "beta={}", beta);
        // Strong rejection of unit root => tau very negative.
        assert!(tau < -3.0, "tau={}", tau);
    }

    #[test]
    fn distance_ssd_zero_for_identical_series() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let d = distance_ssd(&a, &a).unwrap();
        assert!(d < 1e-12);
    }
}
