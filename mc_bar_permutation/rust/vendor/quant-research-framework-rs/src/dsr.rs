//! Deflated Sharpe Ratio (Bailey & López de Prado, 2014), Rust mirror.
//!
//! 1:1 port of `backtester/dsr.py`. Selection-bias correction for an
//! in-sample maximised Sharpe ratio: when a strategy parameter is chosen
//! by maximising SR over a grid of N trials, the maximised SR is
//! upward-biased. The deflated SR is the probability that the true SR
//! exceeds zero conditional on the observed SR, the trial count, and the
//! higher moments of the per-trade returns.
//!
//! Formulas (Bailey & López de Prado 2014, JPM 40(5):94--107):
//!
//! ```text
//!     SR_0 = sqrt(V[SR_n]) * ((1 - euler_gamma) * Phi^{-1}(1 - 1/N)
//!                           + euler_gamma * Phi^{-1}(1 - 1/(N * e)))
//!     DSR  = Phi( (SR_hat - SR_0) * sqrt(T - 1)
//!                / sqrt(1 - g_3 * SR_hat + (g_4 - 1) * SR_hat^2 / 4) )
//! ```
//!
//! where `g_4` is the *raw* fourth standardised moment (= 3 for a Normal,
//! NOT excess kurtosis); see the comment block in `deflated_sharpe_ratio`.
//! All Sharpe arguments (chosen, trial, benchmark) are mean/sample_std,
//! without sqrt(T) or annualisation. Each trial uses its own return series,
//! with the same observation type as the chosen returns (e.g. trades).
//!
//! Like the Python module this is a *post-processing* utility, it does
//! not run inside the engine and does not affect the engine's stdout
//! metric block. The cross-language guarantee for this module is the
//! dedicated `tools/parity_dsr.py` scalar harness, NOT the stdout-metric
//! parity surface.
//!
//! `Phi` and `Phi^{-1}` are taken from `statrs` (`Normal::cdf` /
//! `Normal::inverse_cdf`, matching scipy `norm.cdf` / `norm.ppf` to
//! ~1e-12). A coarse Abramowitz & Stegun 7.1.26 `erf` approximation is
//! deliberately NOT reused here: it has no inverse and its 1.5e-7 forward
//! error would be amplified into `SR_0` at the `1 - 1/N` tail.

use statrs::distribution::{ContinuousCDF, Normal};

/// Euler–Mascheroni constant. Matches `backtester.dsr.EULER_GAMMA`.
pub const EULER_GAMMA: f64 = 0.5772156649015329;

/// Guard for the division by (SR_hat - SR*) in MinTRL.
const SR_TARGET_FLOOR: f64 = 1e-12;

/// Standard-normal helper. `Normal::new(0, 1)` is infallible for these
/// arguments; we unwrap rather than thread a `Result` to keep the public
/// signatures identical to the Python module.
fn standard_normal() -> Normal {
    Normal::new(0.0, 1.0).expect("standard normal is well-defined")
}

/// Sample variance with `ddof = 1` (matches `numpy.var(arr, ddof=1)`).
/// Caller guarantees `xs.len() >= 2`.
fn sample_var_ddof1(xs: &[f64]) -> f64 {
    let n = xs.len() as f64;
    let mean = xs.iter().sum::<f64>() / n;
    let ss: f64 = xs.iter().map(|&x| (x - mean) * (x - mean)).sum();
    ss / (n - 1.0)
}

/// Mean/sample_std (ddof=1), without sqrt(T) or annualisation. Non-finite
/// observations are dropped; fewer than two or zero dispersion yield NaN.
/// Use each trial's own returns, never the chosen strategy's sample count.
pub fn sharpe_per_observation(returns: &[f64]) -> f64 {
    let rets: Vec<f64> = returns.iter().copied().filter(|r| r.is_finite()).collect();
    if rets.len() < 2 {
        return f64::NAN;
    }
    let sd = sample_var_ddof1(&rets).sqrt();
    if sd > 0.0 {
        (rets.iter().sum::<f64>() / rets.len() as f64) / sd
    } else {
        f64::NAN
    }
}

/// `E[max SR_n]` under the null that the true SR is zero, the `SR_0`
/// quantity of Bailey & López de Prado 2014 §3.
///
/// Mirrors `expected_max_sharpe_under_null`: non-finite trials are
/// dropped; fewer than two finite trials, or non-positive trial-Sharpe
/// variance, return `0.0`.
/// Inputs and output use per-observation mean/sample_std units.
pub fn expected_max_sharpe_under_null(trial_sharpes: &[f64]) -> f64 {
    let finite: Vec<f64> = trial_sharpes
        .iter()
        .copied()
        .filter(|s| s.is_finite())
        .collect();
    let n = finite.len();
    if n < 2 {
        return 0.0;
    }
    let var_sr = sample_var_ddof1(&finite);
    if var_sr <= 0.0 {
        return 0.0;
    }
    let nd = standard_normal();
    let nf = n as f64;
    let e = std::f64::consts::E;
    var_sr.sqrt()
        * ((1.0 - EULER_GAMMA) * nd.inverse_cdf(1.0 - 1.0 / nf)
            + EULER_GAMMA * nd.inverse_cdf(1.0 - 1.0 / (nf * e)))
}

/// Probability the true Sharpe exceeds zero, conditional on the observed
/// in-sample maximised Sharpe, the per-trial Sharpe variance, and the
/// per-trade return moments. Returns a value in `[0, 1]`, or `NaN` on any
/// of the degenerate guards (mirrors `deflated_sharpe_ratio`).
/// Chosen and trial Sharpes must be per-observation mean/sample_std.
pub fn deflated_sharpe_ratio(
    sharpe_chosen: f64,
    trial_sharpes: &[f64],
    returns: &[f64],
) -> f64 {
    let rets: Vec<f64> = returns.iter().copied().filter(|r| r.is_finite()).collect();
    let t = rets.len();
    if t < 3 || !sharpe_chosen.is_finite() {
        return f64::NAN;
    }

    let sr_0 = expected_max_sharpe_under_null(trial_sharpes);

    // Per-trade-return higher moments. Bailey & López de Prado (2014),
    // JPM 40(5):94--107, eq. (9): the variance-correction term is
    //     sqrt( 1 - g_3 * SR + (g_4 - 1) * SR^2 / 4 )
    // where g_4 is the *raw* fourth standardised moment
    //     g_4 = E[(x - mu)^4] / sigma^4   (= 3 for a Normal),
    // NOT excess kurtosis. The (g_4 - 1) coefficient is therefore correct
    // as written below: it reduces to (3 - 1)/4 = 0.5 in the Normal case.
    let denom_sq = match sr_std_correction(&rets, sharpe_chosen) {
        Some(d) => d,
        None => return f64::NAN,
    };
    if denom_sq <= 0.0 {
        return f64::NAN;
    }

    let z_hat = (sharpe_chosen - sr_0) * ((t as f64) - 1.0).sqrt() / denom_sq.sqrt();
    standard_normal().cdf(z_hat)
}

/// Bailey-LdP 2014 eq (9) variance-correction term
///     1 - g_3*SR + (g_4 - 1)*SR^2/4
/// with g_4 the RAW fourth standardised moment. `rets` finite-filtered.
/// None if sd <= 0. Shared by DSR/PSR/MinTRL; arithmetic order matches
/// the Python `_sr_std_correction`.
fn sr_std_correction(rets: &[f64], sharpe: f64) -> Option<f64> {
    let tf = rets.len() as f64;
    let mu = rets.iter().sum::<f64>() / tf;
    let sd = sample_var_ddof1(rets).sqrt();
    if sd <= 0.0 {
        return None;
    }
    let mut sum_z3 = 0.0f64;
    let mut sum_z4 = 0.0f64;
    for &r in rets {
        let z = (r - mu) / sd;
        let z2 = z * z;
        sum_z3 += z2 * z;
        sum_z4 += z2 * z2;
    }
    let g_3 = sum_z3 / tf;
    let g_4 = sum_z4 / tf;
    Some(1.0 - g_3 * sharpe + (g_4 - 1.0) * sharpe * sharpe / 4.0)
}

/// Probabilistic Sharpe Ratio (Bailey-LdP 2014). DSR = PSR with
/// SR* = SR_0. P(SR>SR*) in [0,1], NaN on the DSR guards.
/// `sharpe` and `sr_benchmark` are per-observation mean/sample_std.
pub fn probabilistic_sharpe_ratio(sharpe: f64, returns: &[f64], sr_benchmark: f64) -> f64 {
    let rets: Vec<f64> = returns.iter().copied().filter(|r| r.is_finite()).collect();
    let t = rets.len();
    if t < 3 || !sharpe.is_finite() {
        return f64::NAN;
    }
    let denom_sq = match sr_std_correction(&rets, sharpe) {
        Some(d) if d > 0.0 => d,
        _ => return f64::NAN,
    };
    let z_hat = (sharpe - sr_benchmark) * ((t as f64) - 1.0).sqrt() / denom_sq.sqrt();
    standard_normal().cdf(z_hat)
}

/// Minimum Track Record Length (Bailey-LdP 2014 eq 19). Observation
/// count, inf if SR <= SR*, NaN on the DSR guards.
/// Both Sharpe arguments are per-observation mean/sample_std.
pub fn min_track_record_length(
    sharpe: f64, returns: &[f64], sr_benchmark: f64, prob: f64,
) -> f64 {
    let rets: Vec<f64> = returns.iter().copied().filter(|r| r.is_finite()).collect();
    if rets.len() < 3 || !sharpe.is_finite() {
        return f64::NAN;
    }
    let denom_sq = match sr_std_correction(&rets, sharpe) {
        Some(d) if d > 0.0 => d,
        _ => return f64::NAN,
    };
    let excess = sharpe - sr_benchmark;
    if excess <= SR_TARGET_FLOOR {
        return f64::INFINITY;
    }
    let z_p = standard_normal().inverse_cdf(prob);
    1.0 + denom_sq * (z_p / excess).powi(2)
}

/// Minimum Backtest Length (Bailey-Borwein-LdP-Zhu 2014). `sr_target`
/// must be per-period. inf if sr_target<=0, NaN if n_trials<2.
pub fn min_backtest_length(n_trials: usize, sr_target: f64) -> f64 {
    if n_trials < 2 {
        return f64::NAN;
    }
    if sr_target <= SR_TARGET_FLOOR {
        return f64::INFINITY;
    }
    let nf = n_trials as f64;
    let e = std::f64::consts::E;
    let nd = standard_normal();
    let z_combo = (1.0 - EULER_GAMMA) * nd.inverse_cdf(1.0 - 1.0 / nf)
        + EULER_GAMMA * nd.inverse_cdf(1.0 - 1.0 / (nf * e));
    (z_combo / sr_target).powi(2)
}

/// One-line summary mirroring `backtester.dsr.report`. Kept for API
/// parity; nothing in the engine calls it.
pub fn report(sharpe_chosen: f64, trial_sharpes: &[f64], returns: &[f64]) -> String {
    let sr_0 = expected_max_sharpe_under_null(trial_sharpes);
    let dsr = deflated_sharpe_ratio(sharpe_chosen, trial_sharpes, returns);
    let n = trial_sharpes.iter().filter(|s| s.is_finite()).count();
    format!(
        "  DSR  | SR_chosen:{:6.2}  E[max SR|null,N={}]:{:6.2}  haircut:{:6.2}  P(SR_true>0):{:5.3}",
        sharpe_chosen,
        n,
        sr_0,
        sharpe_chosen - sr_0,
        dsr
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn observation_units_match_symmetric_closed_form() {
        let t = 1000;
        let sr = 0.1;
        let shift = sr * (t as f64 / (t - 1) as f64).sqrt();
        let rets: Vec<f64> = (0..t)
            .map(|i| if i % 2 == 0 { shift - 1.0 } else { shift + 1.0 })
            .collect();
        let g4 = ((t - 1) as f64 / t as f64).powi(2);
        let correction = 1.0 + (g4 - 1.0) * sr * sr / 4.0;
        approx(sharpe_per_observation(&rets), sr, 1e-12);
        // Set the benchmark so the independently derived normal argument
        // is exactly one; pin Phi(1), not merely the probability's range.
        let benchmark = sr - correction.sqrt() / ((t - 1) as f64).sqrt();
        approx(probabilistic_sharpe_ratio(sr, &rets, benchmark), 0.8413447460685429, 1e-10);
        let expected = 1.0 + correction * (1.6448536269514722 / (sr - 0.03)).powi(2);
        approx(min_track_record_length(sr, &rets, 0.03, 0.95), expected, 1e-9);
        assert!(expected > 500.0);
    }

    #[test]
    fn observation_sharpe_filters_nonfinite_and_degenerate() {
        approx(sharpe_per_observation(&[-1.0, 1.0, f64::NAN, f64::INFINITY]), 0.0, 1e-12);
        assert!(sharpe_per_observation(&[]).is_nan());
        assert!(sharpe_per_observation(&[1.0]).is_nan());
        assert!(sharpe_per_observation(&[1.0, 1.0]).is_nan());
    }

    // Golden values precomputed once from backtester/dsr.py on the exact
    // input arrays below; asserted to <1e-9 to pin the port independently
    // of the cross-process parity harness.
    fn approx(a: f64, b: f64, tol: f64) {
        assert!(
            (a - b).abs() <= tol,
            "expected {b} got {a} (|Δ|={})",
            (a - b).abs()
        );
    }

    #[test]
    fn sr0_fewer_than_two_trials_is_zero() {
        assert_eq!(expected_max_sharpe_under_null(&[]), 0.0);
        assert_eq!(expected_max_sharpe_under_null(&[1.3]), 0.0);
        // non-finite trials dropped -> only one finite -> 0.0
        assert_eq!(
            expected_max_sharpe_under_null(&[1.3, f64::NAN, f64::INFINITY]),
            0.0
        );
    }

    #[test]
    fn sr0_zero_variance_is_zero() {
        assert_eq!(expected_max_sharpe_under_null(&[0.5, 0.5, 0.5, 0.5]), 0.0);
    }

    #[test]
    fn dsr_too_few_returns_is_nan() {
        assert!(deflated_sharpe_ratio(1.0, &[0.1, 0.2, 0.3], &[0.01, 0.02]).is_nan());
    }

    #[test]
    fn dsr_nonfinite_sharpe_is_nan() {
        let rets = [0.01, -0.02, 0.03, 0.0, 0.015];
        assert!(deflated_sharpe_ratio(f64::NAN, &[0.1, 0.2], &rets).is_nan());
        assert!(deflated_sharpe_ratio(f64::INFINITY, &[0.1, 0.2], &rets).is_nan());
    }

    #[test]
    fn dsr_zero_dispersion_returns_is_nan() {
        // constant returns -> sd == 0 -> NaN
        assert!(deflated_sharpe_ratio(1.0, &[0.1, 0.2, 0.3], &[0.5; 8]).is_nan());
    }

    #[test]
    fn golden_clean_case() {
        // trial_sharpes = [0.1, 0.4, 0.9, 0.6, 0.3, 0.8]
        // returns       = [0.012, -0.004, 0.021, -0.011, 0.008, 0.015,
        //                  -0.006, 0.019, 0.003, -0.009]
        // sharpe_chosen = 0.9
        // Computed from backtester/dsr.py (scipy norm):
        //   expected_max_sharpe_under_null -> 0.3979082244143515
        //   deflated_sharpe_ratio          -> 0.9301178821563774
        let trials = [0.1, 0.4, 0.9, 0.6, 0.3, 0.8];
        let rets = [
            0.012, -0.004, 0.021, -0.011, 0.008, 0.015, -0.006, 0.019, 0.003, -0.009,
        ];
        let sr0 = expected_max_sharpe_under_null(&trials);
        approx(sr0, 0.3979082244143515, 1e-9);
        let dsr = deflated_sharpe_ratio(0.9, &trials, &rets);
        approx(dsr, 0.9301178821563774, 1e-9);
        assert!((0.0..=1.0).contains(&dsr), "dsr out of range: {dsr}");
    }
}
