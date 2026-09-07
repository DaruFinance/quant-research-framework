//! Multi-term IS objective (Rust mirror).
//!
//! Mirrors ``backtester.objectives.MultiTermObjective`` in Python.
//!
//! ```text
//! score = sortino_weight * sortino(rets)
//!       - corr_penalty * |corr(rets, benchmark)|
//!       - turnover_penalty * turnover
//! ```

use crate::metrics::sortino;

#[derive(Debug, Clone, Copy)]
pub struct MultiTermObjective {
    pub sortino_weight: f64,
    pub corr_penalty: f64,
    pub turnover_penalty: f64,
    pub annualization: Option<f64>,
}

impl Default for MultiTermObjective {
    fn default() -> Self {
        Self {
            sortino_weight: 1.0,
            corr_penalty: 0.5,
            turnover_penalty: 0.1,
            annualization: None,
        }
    }
}

impl MultiTermObjective {
    pub fn score(
        &self,
        rets: &[f64],
        benchmark_rets: Option<&[f64]>,
        turnover: f64,
    ) -> Result<f64, String> {
        if rets.len() < 2 {
            return Ok(f64::NEG_INFINITY);
        }
        let mut score = self.sortino_weight * sortino(rets, self.annualization);
        if let Some(b) = benchmark_rets {
            if !b.is_empty() {
                if b.len() != rets.len() {
                    return Err(format!(
                        "multi_term: benchmark length {} != strategy length {}",
                        b.len(),
                        rets.len()
                    ));
                }
                let n = rets.len() as f64;
                let mr: f64 = rets.iter().sum::<f64>() / n;
                let mb: f64 = b.iter().sum::<f64>() / n;
                let var_r: f64 =
                    rets.iter().map(|x| (x - mr).powi(2)).sum::<f64>() / n;
                let var_b: f64 =
                    b.iter().map(|x| (x - mb).powi(2)).sum::<f64>() / n;
                if var_r > 0.0 && var_b > 0.0 {
                    let cov: f64 = rets
                        .iter()
                        .zip(b.iter())
                        .map(|(x, y)| (x - mr) * (y - mb))
                        .sum::<f64>()
                        / n;
                    let mut rho = cov / (var_r.sqrt() * var_b.sqrt());
                    if rho.is_nan() {
                        rho = 0.0;
                    }
                    score -= self.corr_penalty * rho.abs();
                }
            }
        }
        score -= self.turnover_penalty * turnover;
        Ok(score)
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use rand::{rngs::StdRng, Rng, SeedableRng};

    fn rng_returns(seed: u64, n: usize, mean: f64, sd: f64) -> Vec<f64> {
        // Simple normal-ish sample via Box-Muller from rand uniforms.
        let mut rng = StdRng::seed_from_u64(seed);
        let mut out = Vec::with_capacity(n);
        for _ in 0..n {
            let u1: f64 = rng.random::<f64>().max(1e-12);
            let u2: f64 = rng.random::<f64>();
            let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
            out.push(mean + sd * z);
        }
        out
    }

    #[test]
    fn finite_score_on_basic_input() {
        let r = rng_returns(4, 500, 0.001, 0.01);
        let obj = MultiTermObjective::default();
        let s = obj.score(&r, None, 0.0).unwrap();
        assert!(s.is_finite());
    }

    #[test]
    fn high_correlation_lowers_score() {
        let bench = rng_returns(5, 500, 0.001, 0.01);
        let aligned: Vec<f64> = bench.iter().map(|x| *x + 1e-9).collect();
        let independent = rng_returns(99, 500, 0.001, 0.01);
        let obj = MultiTermObjective { corr_penalty: 1.0, ..Default::default() };
        let s_aligned = obj.score(&aligned, Some(&bench), 0.0).unwrap();
        let s_independent = obj.score(&independent, Some(&bench), 0.0).unwrap();
        assert!(s_aligned < s_independent);
    }

    #[test]
    fn high_turnover_lowers_score() {
        let r = rng_returns(6, 500, 0.001, 0.01);
        let obj = MultiTermObjective { turnover_penalty: 0.1, ..Default::default() };
        let low = obj.score(&r, None, 1.0).unwrap();
        let high = obj.score(&r, None, 10.0).unwrap();
        assert!(high < low);
        assert!((((low - high) - 0.9) as f64).abs() < 1e-12);
    }

    #[test]
    fn rejects_mismatched_benchmark_length() {
        let r = vec![0.0; 500];
        let b = vec![0.0; 400];
        let obj = MultiTermObjective::default();
        assert!(obj.score(&r, Some(&b), 0.0).is_err());
    }
}
