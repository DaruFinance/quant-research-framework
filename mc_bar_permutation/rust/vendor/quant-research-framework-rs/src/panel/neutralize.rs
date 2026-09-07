//! Position-construction neutralizations (Rust mirror).
//!
//! Three modes (``Dollar``, ``Beta``, ``Sigma``) layered on top of a
//! raw weight vector. Pure-function with the same semantics as the
//! Python ``backtester.panel.neutralize`` module.

#![cfg(feature = "panel")]

/// Neutralization mode selector.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    Dollar,
    Beta,
    Sigma,
}

/// OLS slope per asset against ``market_idx``'s column in the same
/// 2-D returns matrix. ``returns`` is `(n_bars, n_assets)` flattened
/// row-major in a `Vec<Vec<f64>>` for ergonomic Rust use; alternative
/// shape is in the cross-language parity test.
///
/// Caller MUST pass a window that ends strictly before the rebalance
/// bar to keep the betas leak-free.
pub fn estimate_betas(
    returns: &[Vec<f64>],
    market_idx: usize,
) -> Result<Vec<f64>, String> {
    if returns.is_empty() {
        return Err("returns_window is empty".to_string());
    }
    let n_assets = returns[0].len();
    if market_idx >= n_assets {
        return Err(format!(
            "market_idx={} out of range [0, {})",
            market_idx, n_assets
        ));
    }
    // Filter rows with NaN.
    let clean: Vec<&Vec<f64>> = returns
        .iter()
        .filter(|row| row.len() == n_assets && row.iter().all(|x| !x.is_nan()))
        .collect();
    if clean.len() < 2 {
        return Err(format!(
            "returns_window has {} clean rows; need >= 2",
            clean.len()
        ));
    }
    let n = clean.len() as f64;
    let market_mean: f64 = clean.iter().map(|r| r[market_idx]).sum::<f64>() / n;
    let market_var: f64 = clean
        .iter()
        .map(|r| (r[market_idx] - market_mean).powi(2))
        .sum::<f64>()
        / (n - 1.0);
    if market_var <= 0.0 {
        return Err(format!(
            "market column {} has zero variance",
            market_idx
        ));
    }
    let mut betas = vec![0.0f64; n_assets];
    for i in 0..n_assets {
        let mean_i: f64 = clean.iter().map(|r| r[i]).sum::<f64>() / n;
        let cov_im: f64 = clean
            .iter()
            .map(|r| (r[i] - mean_i) * (r[market_idx] - market_mean))
            .sum::<f64>()
            / (n - 1.0);
        betas[i] = cov_im / market_var;
    }
    Ok(betas)
}

/// Per-asset stdev (ddof=1) of returns. Same NaN-row filtering as
/// ``estimate_betas``.
pub fn estimate_vols(returns: &[Vec<f64>]) -> Result<Vec<f64>, String> {
    if returns.is_empty() {
        return Err("returns_window is empty".to_string());
    }
    let n_assets = returns[0].len();
    let clean: Vec<&Vec<f64>> = returns
        .iter()
        .filter(|row| row.len() == n_assets && row.iter().all(|x| !x.is_nan()))
        .collect();
    if clean.len() < 2 {
        return Err(format!(
            "returns_window has {} clean rows; need >= 2",
            clean.len()
        ));
    }
    let n = clean.len() as f64;
    let mut out = vec![0.0f64; n_assets];
    for i in 0..n_assets {
        let m: f64 = clean.iter().map(|r| r[i]).sum::<f64>() / n;
        let var: f64 = clean
            .iter()
            .map(|r| (r[i] - m).powi(2))
            .sum::<f64>()
            / (n - 1.0);
        out[i] = var.sqrt();
    }
    Ok(out)
}

pub fn neutralize_dollar(raw_weights: &[f64]) -> Result<Vec<f64>, String> {
    let mut out = raw_weights.to_vec();
    let long_sum: f64 = out.iter().filter(|x| **x > 0.0).sum();
    let short_sum: f64 = -out.iter().filter(|x| **x < 0.0).sum::<f64>();
    if long_sum == 0.0 || short_sum == 0.0 {
        return Err(format!(
            "dollar-neutral requires both long and short raw weights; \
             long_sum={} short_sum={}",
            long_sum, short_sum
        ));
    }
    for x in out.iter_mut() {
        if *x > 0.0 {
            *x *= 0.5 / long_sum;
        } else if *x < 0.0 {
            *x *= 0.5 / short_sum;
        }
    }
    Ok(out)
}

pub fn neutralize_beta(
    raw_weights: &[f64],
    betas: &[f64],
    market_idx: Option<usize>,
) -> Result<Vec<f64>, String> {
    let mut w = raw_weights.to_vec();
    if w.len() != betas.len() {
        return Err(format!(
            "raw_weights and betas shape mismatch: {} vs {}",
            w.len(),
            betas.len()
        ));
    }
    let mi = market_idx.unwrap_or_else(|| {
        // argmax |beta|
        let mut best = 0usize;
        let mut best_abs = betas[0].abs();
        for (i, &b) in betas.iter().enumerate() {
            if b.abs() > best_abs {
                best_abs = b.abs();
                best = i;
            }
        }
        best
    });
    if mi >= w.len() {
        return Err(format!("market_idx {} out of range", mi));
    }
    if betas[mi] == 0.0 {
        return Err(format!("market_idx={} has beta=0", mi));
    }
    let mut net_no_market = 0.0;
    for (i, (&wi, &bi)) in w.iter().zip(betas.iter()).enumerate() {
        if i != mi {
            net_no_market += wi * bi;
        }
    }
    w[mi] = -net_no_market / betas[mi];
    Ok(w)
}

pub fn neutralize_sigma(
    raw_weights: &[f64],
    vols: &[f64],
) -> Result<Vec<f64>, String> {
    if raw_weights.len() != vols.len() {
        return Err(format!(
            "raw_weights and vols shape mismatch: {} vs {}",
            raw_weights.len(),
            vols.len()
        ));
    }
    if vols.iter().any(|v| *v <= 0.0) {
        return Err(format!("sigma-neutral requires positive vols; got {:?}", vols));
    }
    if raw_weights.contains(&0.0) {
        return Err("sigma-neutral requires every raw weight non-zero".to_string());
    }
    let signs: Vec<f64> = raw_weights.iter().map(|x| x.signum()).collect();
    let mut abs_target: Vec<f64> = vols.iter().map(|v| 1.0 / v).collect();
    let gross_in: f64 = raw_weights.iter().map(|x| x.abs()).sum();
    let target_sum: f64 = abs_target.iter().sum();
    let scale = gross_in / target_sum;
    for x in abs_target.iter_mut() {
        *x *= scale;
    }
    Ok(signs.iter().zip(abs_target.iter()).map(|(s, a)| s * a).collect())
}


#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }
    fn approx_vec(a: &[f64], b: &[f64], tol: f64) -> bool {
        a.len() == b.len() && a.iter().zip(b.iter()).all(|(x, y)| approx(*x, *y, tol))
    }

    #[test]
    fn dollar_balances_long_short() {
        let raw = vec![0.3, -0.7, 0.5, -0.2];
        let out = neutralize_dollar(&raw).unwrap();
        let longs: f64 = out.iter().filter(|x| **x > 0.0).sum();
        let shorts: f64 = -out.iter().filter(|x| **x < 0.0).sum::<f64>();
        assert!(approx(longs, 0.5, 1e-12));
        assert!(approx(shorts, 0.5, 1e-12));
    }

    #[test]
    fn beta_zeros_market_dot_product() {
        let raw = vec![0.4, -0.6, 0.5];
        let betas = vec![0.9, 1.0, 1.1];
        let w = neutralize_beta(&raw, &betas, Some(1)).unwrap();
        let dot: f64 = w.iter().zip(betas.iter()).map(|(a, b)| a * b).sum();
        assert!(dot.abs() < 1e-12);
    }

    #[test]
    fn sigma_equalises_vol_contribution() {
        let raw = vec![0.5, -0.3, 0.4];
        let vols = vec![0.02, 0.01, 0.03];
        let w = neutralize_sigma(&raw, &vols).unwrap();
        let vcs: Vec<f64> = w.iter().zip(vols.iter()).map(|(a, b)| a.abs() * b).collect();
        let first = vcs[0];
        for v in &vcs {
            assert!((v - first).abs() < 1e-12);
        }
    }

    #[test]
    fn sigma_preserves_gross() {
        let raw = vec![0.5, -0.3, 0.4];
        let vols = vec![0.02, 0.01, 0.03];
        let w = neutralize_sigma(&raw, &vols).unwrap();
        let gross_in: f64 = raw.iter().map(|x| x.abs()).sum();
        let gross_out: f64 = w.iter().map(|x| x.abs()).sum();
        assert!(approx(gross_in, gross_out, 1e-12));
    }

    #[test]
    fn beta_rejects_shape_mismatch() {
        let r = neutralize_beta(&[1.0, 0.0], &[1.0, 0.5, 0.3], None);
        assert!(r.is_err());
    }

    #[test]
    fn estimate_betas_against_self_is_one() {
        // Asset is the market => beta = 1.
        let returns = vec![vec![0.01], vec![-0.005], vec![0.02], vec![-0.01], vec![0.003]];
        let b = estimate_betas(&returns, 0).unwrap();
        assert!(approx(b[0], 1.0, 1e-12));
    }

    #[test]
    fn estimate_vols_matches_sample_std() {
        let r = vec![vec![0.01], vec![0.02], vec![0.03]];
        let v = estimate_vols(&r).unwrap();
        // sample std of [0.01, 0.02, 0.03] = 0.01
        assert!(approx(v[0], 0.01, 1e-15));
    }
}
