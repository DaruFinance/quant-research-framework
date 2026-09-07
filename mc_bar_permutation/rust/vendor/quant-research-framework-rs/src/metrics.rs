//! Extra metric primitives (Rust mirror).
//!
//! Mirrors ``backtester.metrics`` in Python. Sortino and turnover
//! consume the same per-bar / per-trade return / position arrays the
//! kernel already produces. Pure-function; no time-dependent state.

/// Sortino ratio. Mean return divided by downside deviation. NaN if
/// fewer than 2 returns or zero downside dev. Optional annualisation
/// multiplies the raw ratio by sqrt(annualization).
pub fn sortino(returns: &[f64], annualization: Option<f64>) -> f64 {
    if returns.len() < 2 {
        return f64::NAN;
    }
    let mut neg_count = 0usize;
    let mut sum = 0.0f64;
    let mut sum_neg_sq = 0.0f64;
    for &r in returns {
        sum += r;
        if r < 0.0 {
            neg_count += 1;
            sum_neg_sq += r * r;
        }
    }
    if neg_count < 2 {
        return f64::NAN;
    }
    // Match Python: downside dev computed over ALL returns (not just
    // negatives) using min(r, 0)^2. The sum_neg_sq above is exactly
    // sum_i (min(r_i, 0))^2 over all i.
    let semi_dev = (sum_neg_sq / returns.len() as f64).sqrt();
    if semi_dev == 0.0 {
        return f64::NAN;
    }
    let mean = sum / returns.len() as f64;
    let mut out = mean / semi_dev;
    if let Some(ann) = annualization {
        out *= ann.sqrt();
    }
    out
}

/// Sum of |w_{t+1} - w_t| across a position trajectory.
pub fn turnover(positions: &[f64]) -> f64 {
    if positions.len() < 2 {
        return 0.0;
    }
    let mut acc = 0.0f64;
    for w in positions.windows(2) {
        acc += (w[1] - w[0]).abs();
    }
    acc
}


#[cfg(test)]
mod tests {
    use super::*;
    use rand::{rngs::StdRng, Rng, SeedableRng};

    #[test]
    fn sortino_positive_when_mean_positive() {
        let mut rng = StdRng::seed_from_u64(1);
        let r: Vec<f64> = (0..1000).map(|_| 0.001 + rng.random::<f64>() * 0.01 - 0.005).collect();
        let s = sortino(&r, None);
        assert!(s.is_finite());
        assert!(s > 0.0);
    }

    #[test]
    fn sortino_nan_on_no_losses() {
        let r = vec![0.01, 0.02, 0.03];
        assert!(sortino(&r, None).is_nan());
    }

    #[test]
    fn sortino_nan_on_too_few_returns() {
        assert!(sortino(&[0.01], None).is_nan());
    }

    #[test]
    fn sortino_annualization_scales_by_sqrt() {
        let mut rng = StdRng::seed_from_u64(3);
        let r: Vec<f64> = (0..1000).map(|_| 0.001 + rng.random::<f64>() * 0.01 - 0.005).collect();
        let raw = sortino(&r, None);
        let ann = sortino(&r, Some(252.0));
        assert!((ann - raw * (252.0_f64).sqrt()).abs() < 1e-12);
    }

    #[test]
    fn turnover_zero_for_constant_position() {
        assert_eq!(turnover(&[0.5, 0.5, 0.5, 0.5]), 0.0);
    }

    #[test]
    fn turnover_counts_absolute_changes() {
        // |(-0.5 - 0.5)| + |(0.5 - -0.5)| + |(0.0 - 0.5)| = 1 + 1 + 0.5 = 2.5
        assert!((turnover(&[0.5, -0.5, 0.5, 0.0]) - 2.5).abs() < 1e-12);
    }

    #[test]
    fn turnover_short_input_is_zero() {
        assert_eq!(turnover(&[0.5]), 0.0);
    }
}
