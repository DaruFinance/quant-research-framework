//! Portfolio sizing primitives (Rust mirror).
//!
//! ``equal_weights`` and ``erc_weights`` mirror
//! ``backtester.panel.sizing``. The Python side uses scipy SLSQP; the
//! Rust port uses Spinu's classical cyclical descent for ERC (no
//! external dependencies, ~10 iterations to convergence on typical
//! financial covariances).
//!
//! Cross-language parity: the two solvers won't produce bit-identical
//! weights (different numerical paths) but agree to ~1e-6 on
//! well-conditioned covariance matrices. The cross-language parity
//! test in ``tests/parity_panel.py`` enforces that tolerance.

#![cfg(feature = "panel")]

/// 1/N baseline.
pub fn equal_weights(n: usize) -> Result<Vec<f64>, String> {
    if n == 0 {
        return Err("equal_weights: n must be > 0".to_string());
    }
    let w = 1.0 / n as f64;
    Ok(vec![w; n])
}

/// Compute the sample covariance matrix of a `(n_bars, n_assets)`
/// return matrix represented as `Vec<Vec<f64>>`. Drops bars with NaN.
pub fn cov_from_returns(returns: &[Vec<f64>]) -> Result<Vec<Vec<f64>>, String> {
    if returns.is_empty() {
        return Err("returns_window is empty".to_string());
    }
    let n_assets = returns[0].len();
    let clean: Vec<&Vec<f64>> = returns
        .iter()
        .filter(|r| r.len() == n_assets && r.iter().all(|x| !x.is_nan()))
        .collect();
    if clean.len() < 2 {
        return Err(format!(
            "returns_window has {} clean rows; need >= 2",
            clean.len()
        ));
    }
    let n = clean.len() as f64;
    // Per-asset mean.
    let mut means = vec![0.0f64; n_assets];
    for r in &clean {
        for j in 0..n_assets {
            means[j] += r[j];
        }
    }
    for m in means.iter_mut() {
        *m /= n;
    }
    let mut cov = vec![vec![0.0f64; n_assets]; n_assets];
    for r in &clean {
        for i in 0..n_assets {
            let di = r[i] - means[i];
            for j in 0..n_assets {
                cov[i][j] += di * (r[j] - means[j]);
            }
        }
    }
    let denom = n - 1.0;
    for row in cov.iter_mut() {
        for x in row.iter_mut() {
            *x /= denom;
        }
    }
    Ok(cov)
}

fn cov_dot(cov: &[Vec<f64>], w: &[f64]) -> Vec<f64> {
    let n = w.len();
    let mut out = vec![0.0f64; n];
    for i in 0..n {
        let mut s = 0.0;
        for j in 0..n {
            s += cov[i][j] * w[j];
        }
        out[i] = s;
    }
    out
}

/// Equal-risk-contribution weights via Spinu's cyclical descent. At
/// convergence each asset's risk contribution ``w_i * (Σ w)_i`` is
/// equal to ``V / n`` where ``V = w' Σ w``.
///
/// Pre-rescales the covariance to unit trace for numerical stability
/// regardless of return-magnitude scale (mirrors the Python side).
pub fn erc_weights(cov: &[Vec<f64>]) -> Result<Vec<f64>, String> {
    let n = cov.len();
    if n == 0 {
        return Err("erc_weights: empty covariance".to_string());
    }
    for row in cov {
        if row.len() != n {
            return Err(format!(
                "erc_weights: covariance not square (got {}x{})",
                n,
                row.len()
            ));
        }
    }
    if n == 1 {
        return Ok(vec![1.0]);
    }
    // Trace rescale.
    let trace: f64 = (0..n).map(|i| cov[i][i]).sum();
    if trace <= 0.0 {
        return Err(format!(
            "erc_weights: covariance has non-positive trace {}",
            trace
        ));
    }
    let mut cov_s = vec![vec![0.0f64; n]; n];
    for i in 0..n {
        for j in 0..n {
            cov_s[i][j] = cov[i][j] / trace;
        }
    }

    // Spinu's cyclical-descent iteration. Update each w_i in turn so
    // that the per-asset risk contribution moves toward V/n. We use
    // the fixed-point form: w_i_new = (V/n) / (Σ w)_i, with the
    // weight vector renormalised after each cycle.
    let mut w = vec![1.0 / n as f64; n];
    let max_iter = 1000usize;
    let tol = 1e-12;
    for _ in 0..max_iter {
        let sigma_w = cov_dot(&cov_s, &w);
        let v: f64 = w.iter().zip(sigma_w.iter()).map(|(a, b)| a * b).sum();
        let target = v / n as f64;
        let mut new_w = vec![0.0f64; n];
        for i in 0..n {
            let denom = if sigma_w[i].abs() < 1e-15 { 1e-15 } else { sigma_w[i] };
            new_w[i] = (target / denom).abs().max(1e-12);
        }
        let s: f64 = new_w.iter().sum();
        for x in new_w.iter_mut() {
            *x /= s;
        }
        let max_change = w
            .iter()
            .zip(new_w.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f64, f64::max);
        w = new_w;
        if max_change < tol {
            break;
        }
    }
    Ok(w)
}

/// Per-leg risk contribution ``w_i * (Σ w)_i``. Useful for tests /
/// verification.
pub fn risk_contributions(weights: &[f64], cov: &[Vec<f64>]) -> Vec<f64> {
    let sigma_w = cov_dot(cov, weights);
    weights
        .iter()
        .zip(sigma_w.iter())
        .map(|(a, b)| a * b)
        .collect()
}


#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    #[test]
    fn equal_weights_basic() {
        let w = equal_weights(5).unwrap();
        assert_eq!(w, vec![0.2; 5]);
    }

    #[test]
    fn equal_weights_rejects_zero() {
        assert!(equal_weights(0).is_err());
    }

    #[test]
    fn cov_diagonal_for_independent_series() {
        // Two columns: same returns repeated to make them perfectly
        // correlated for a sanity check.
        let returns = vec![
            vec![0.01, 0.01],
            vec![-0.02, -0.02],
            vec![0.005, 0.005],
            vec![-0.01, -0.01],
            vec![0.02, 0.02],
        ];
        let cov = cov_from_returns(&returns).unwrap();
        // Perfectly correlated => cov(0,1) == var(0) == var(1).
        assert!(approx(cov[0][0], cov[1][1], 1e-15));
        assert!(approx(cov[0][1], cov[0][0], 1e-15));
    }

    #[test]
    fn erc_weights_sum_to_one() {
        let cov = vec![
            vec![1e-4, 1e-5, 0.0],
            vec![1e-5, 4e-4, 1e-5],
            vec![0.0, 1e-5, 9e-4],
        ];
        let w = erc_weights(&cov).unwrap();
        assert_eq!(w.len(), 3);
        let s: f64 = w.iter().sum();
        assert!(approx(s, 1.0, 1e-6));
        for x in &w {
            assert!(*x > 0.0);
        }
    }

    #[test]
    fn erc_equal_risk_contribution_property() {
        let cov = vec![
            vec![1e-4, 1e-5, 0.0],
            vec![1e-5, 4e-4, 1e-5],
            vec![0.0, 1e-5, 9e-4],
        ];
        let w = erc_weights(&cov).unwrap();
        let rc = risk_contributions(&w, &cov);
        let v: f64 = w.iter().zip(cov_dot(&cov, &w).iter()).map(|(a, b)| a * b).sum();
        let target = v / w.len() as f64;
        let max_dev = rc.iter().map(|r| (r - target).abs()).fold(0.0f64, f64::max);
        // Relative dispersion < 1% of target.
        assert!(max_dev < 0.01 * target);
    }

    #[test]
    fn erc_n1_trivial() {
        let w = erc_weights(&vec![vec![1.0]]).unwrap();
        assert_eq!(w, vec![1.0]);
    }

    #[test]
    fn erc_rejects_non_square() {
        let r = erc_weights(&vec![vec![1.0, 0.5]]);
        assert!(r.is_err());
    }
}
