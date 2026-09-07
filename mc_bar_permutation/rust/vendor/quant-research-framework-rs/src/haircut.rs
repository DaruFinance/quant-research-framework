//! Haircut Sharpe Ratio (Harvey & Liu 2015), Rust mirror of
//! `backtester/haircut.py`. See the Python module docstring for the
//! derivation. BHY uses c(N)=sum_{i=1..N} 1/i (NOT plain BH). Only the two
//! distinct single-statistic closed forms are offered: 0 = bonferroni,
//! 1 = bhy (Lens B D4, no degenerate 'holm' that equals Bonferroni).
//! Phi/Phi^{-1} from `statrs::Normal`. Cross-language guarantee:
//! `tools/parity_multitest.py`.

use statrs::distribution::{ContinuousCDF, Normal};

fn standard_normal() -> Normal {
    Normal::new(0.0, 1.0).expect("standard normal is well-defined")
}

fn c_harmonic(n: usize) -> f64 {
    (1..=n).map(|i| 1.0 / (i as f64)).sum()
}

/// Result of the Harvey-Liu haircut. Field names mirror the Python dict.
pub struct Haircut {
    pub haircut_sr: f64,
    pub haircut_pct: f64,
    pub p_obs: f64,
    pub p_adj: f64,
    pub t_obs: f64,
    pub t_adj: f64,
}

/// Harvey-Liu (2015) haircut. `method`: 0 = bonferroni, 1 = bhy. Panics on
/// T<2 / n_tests<1 / method>1 (matching the Python ValueErrors).
pub fn haircut_sharpe_ratio(
    sharpe_annual: f64,
    t: usize,
    n_tests: usize,
    method: u8,
    freq: f64,
) -> Haircut {
    assert!(t >= 2, "need T >= 2 observations, got {t}");
    assert!(n_tests >= 1, "n_tests must be >= 1, got {n_tests}");
    assert!(method <= 1, "method must be 0(bonferroni)|1(bhy), got {method}");

    let nd = standard_normal();
    let sr_period = sharpe_annual / freq.sqrt();
    let t_obs = sr_period * (t as f64).sqrt();

    let p_obs = (2.0 * (1.0 - nd.cdf(t_obs.abs()))).clamp(0.0, 1.0);

    let nt = n_tests as f64;
    let p_adj = match method {
        0 => (p_obs * nt).min(1.0),                       // bonferroni
        _ => (p_obs * nt * c_harmonic(n_tests)).min(1.0), // bhy
    };

    let t_adj = if p_adj >= 1.0 {
        0.0
    } else if n_tests == 1 {
        // Single test: no multiple-testing adjustment, so p_adj == p_obs and
        // t_adj == |t_obs| exactly. Compute it directly to avoid cdf/inverse-cdf
        // round-trip noise (statrs vs scipy) on the mathematically-zero haircut.
        t_obs.abs()
    } else {
        nd.inverse_cdf(1.0 - p_adj / 2.0).max(0.0)
    };

    let (haircut_sr, haircut_pct) = if t_obs == 0.0 {
        (sharpe_annual, 0.0)
    } else {
        let ratio = t_adj / t_obs.abs();
        (sharpe_annual * ratio, 1.0 - ratio)
    };

    Haircut { haircut_sr, haircut_pct, p_obs, p_adj, t_obs, t_adj }
}

/// One-line summary mirroring `haircut.py::report`.
pub fn report(sharpe_annual: f64, t: usize, n_tests: usize, method: u8, freq: f64) -> String {
    let out = haircut_sharpe_ratio(sharpe_annual, t, n_tests, method, freq);
    let mname = if method == 0 { "bonferroni" } else { "bhy" };
    format!(
        "  HCUT | method={}  N={}  T={}  SR_obs:{:6.2}  SR_hc:{:6.2}  cut:{:5.1}%  p_adj:{:.3}",
        mname, n_tests, t, sharpe_annual, out.haircut_sr, out.haircut_pct * 100.0, out.p_adj
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn single_test_no_haircut() {
        let h = haircut_sharpe_ratio(1.5, 252, 1, 1, 252.0);
        assert!(h.haircut_pct.abs() < 1e-9, "pct={}", h.haircut_pct);
        assert!((h.haircut_sr - 1.5).abs() < 1e-9);
    }

    #[test]
    fn more_tests_cut_harder() {
        let h10 = haircut_sharpe_ratio(1.5, 252, 10, 1, 252.0);
        let h100 = haircut_sharpe_ratio(1.5, 252, 100, 1, 252.0);
        assert!(h100.haircut_sr <= h10.haircut_sr);
        assert!(h10.haircut_pct >= 0.0 && h10.haircut_pct <= 1.0);
    }
}
