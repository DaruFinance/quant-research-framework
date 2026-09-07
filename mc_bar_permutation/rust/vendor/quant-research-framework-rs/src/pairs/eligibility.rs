//! Pre-screening eligibility filters (Rust mirror).

#![cfg(feature = "pairs")]

#[derive(Debug, Clone)]
pub struct EligibilityCriteria {
    pub p_max: Option<f64>,
    pub half_life_range: Option<(f64, f64)>,
    pub min_window: usize,
}

impl Default for EligibilityCriteria {
    fn default() -> Self {
        Self {
            p_max: Some(0.05),
            half_life_range: Some((1.0, 1000.0)),
            min_window: 60,
        }
    }
}

/// OU half-life via OLS on `Δs_t = a + b * s_{t-1}`.  Returns
/// `+inf` if the slope is non-negative (no mean reversion).
pub fn half_life_ou(spread: &[f64]) -> f64 {
    let s: Vec<f64> = spread.iter().filter(|v| !v.is_nan()).copied().collect();
    if s.len() < 3 {
        return f64::INFINITY;
    }
    let n = s.len();
    let s_lag: Vec<f64> = s[..n - 1].to_vec();
    let ds: Vec<f64> = (1..n).map(|i| s[i] - s[i - 1]).collect();
    // OLS: ds = a + b * s_lag.  Compute b directly.
    let nn = s_lag.len() as f64;
    let sx: f64 = s_lag.iter().sum();
    let sy: f64 = ds.iter().sum();
    let sxx: f64 = s_lag.iter().map(|v| v * v).sum();
    let sxy: f64 = s_lag.iter().zip(ds.iter()).map(|(a, b)| a * b).sum();
    let denom = nn * sxx - sx * sx;
    if denom == 0.0 {
        return f64::INFINITY;
    }
    let slope = (nn * sxy - sx * sy) / denom;
    if slope >= 0.0 {
        return f64::INFINITY;
    }
    (2.0_f64).ln() / -slope
}

/// Apply the criteria stack.  Returns `(ok, reason)`.  `p_value` is
/// None if the caller has not yet run a cointegration test.
pub fn is_eligible_pair(
    spread: &[f64],
    p_value: Option<f64>,
    criteria: &EligibilityCriteria,
) -> (bool, String) {
    let clean: Vec<f64> = spread.iter().filter(|v| !v.is_nan()).copied().collect();
    if clean.len() < criteria.min_window {
        return (
            false,
            format!(
                "insufficient data ({} < {})",
                clean.len(),
                criteria.min_window
            ),
        );
    }
    if let (Some(p_max), Some(p)) = (criteria.p_max, p_value) {
        if p > p_max {
            return (false, format!("p_value={:.4} > p_max={}", p, p_max));
        }
    }
    if let Some((h_lo, h_hi)) = criteria.half_life_range {
        let hl = half_life_ou(&clean);
        if hl < h_lo || hl > h_hi {
            return (
                false,
                format!("half_life={:.2} outside [{}, {}]", hl, h_lo, h_hi),
            );
        }
    }
    (true, "ok".to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn half_life_inf_or_large_on_explosive_series() {
        // s_t = 1.001 * s_{t-1}, explosive AR(1), slope on the
        // change-vs-lag regression is positive => half_life = +inf.
        let mut s = vec![1.0f64];
        for _ in 1..200 {
            let last = *s.last().unwrap();
            s.push(1.001 * last);
        }
        let hl = half_life_ou(&s);
        assert!(hl.is_infinite(), "hl={}", hl);
    }

    #[test]
    fn half_life_finite_on_mean_reverting_series() {
        // AR(1): s_t = 0.7 * s_{t-1} + e.
        let mut s = vec![0.0f64];
        let mut x = 0.0f64;
        for _ in 1..500 {
            // Deterministic "shock": alternating sign.
            let e = if s.len() % 2 == 0 { 0.5 } else { -0.5 };
            x = 0.7 * x + e;
            s.push(x);
        }
        let hl = half_life_ou(&s);
        assert!(hl.is_finite() && hl > 0.0 && hl < 100.0, "hl={}", hl);
    }

    #[test]
    fn is_eligible_short_window_rejected() {
        let s: Vec<f64> = (0..30).map(|i| i as f64).collect();
        let crit = EligibilityCriteria::default();
        let (ok, _) = is_eligible_pair(&s, None, &crit);
        assert!(!ok);
    }
}
