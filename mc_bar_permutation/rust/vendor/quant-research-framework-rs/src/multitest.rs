//! Multiple-testing corrections, Rust mirror of `backtester/multitest.py`.
//!
//! Closed-form (parity-clean, deterministic):
//!   * `bonferroni`    : p <= alpha/N                         (Bonferroni)
//!   * `holm`          : step-down thr = alpha/(N-i)          (Holm 1979)
//!   * `bh_fdr`        : largest i with p_(i) <= alpha*i/N     (B-H 1995)
//!   * `sharpe_pvalues`: 1 - Phi(SR_hat) one-sided            (iid-Normal)
//!
//! Bootstrap (`white_reality_check_indexed`, `romano_wolf_indexed`):
//! NumPy PCG64 != `rand`, so RNG draws cannot match cross-language. These
//! take a PRE-DRAWN index matrix `&[Vec<usize>]` (one row of length T per
//! resample); only the deterministic statistic is parity-tested.
//!
//! `np.percentile(..., 'linear')` (default) replicated exactly in
//! `percentile_linear` for the Romano-Wolf critical value.
//!
//! Post-processing only; cross-language guarantee is
//! `tools/parity_multitest.py`. Needs `statrs::Normal` only for
//! `sharpe_pvalues`.

use statrs::distribution::{ContinuousCDF, Normal};

fn standard_normal() -> Normal {
    Normal::new(0.0, 1.0).expect("standard normal is well-defined")
}

/// Bonferroni mask: p <= alpha/N.
pub fn bonferroni(pvalues: &[f64], alpha: f64) -> Vec<bool> {
    let n = pvalues.len();
    if n == 0 {
        return Vec::new();
    }
    let thr = alpha / (n as f64);
    pvalues.iter().map(|&p| p <= thr).collect()
}

fn argsort_stable(xs: &[f64]) -> Vec<usize> {
    let mut order: Vec<usize> = (0..xs.len()).collect();
    order.sort_by(|&a, &b| match xs[a].partial_cmp(&xs[b]) {
        Some(std::cmp::Ordering::Equal) | None => a.cmp(&b),
        Some(ord) => ord,
    });
    order
}

/// Holm step-down mask (thr = alpha/(N-i), break on first failure).
pub fn holm(pvalues: &[f64], alpha: f64) -> Vec<bool> {
    let n = pvalues.len();
    if n == 0 {
        return Vec::new();
    }
    let order = argsort_stable(pvalues);
    let mut rejected = vec![false; n];
    for (i, &orig) in order.iter().enumerate() {
        let thr = alpha / ((n - i) as f64);
        if pvalues[orig] <= thr {
            rejected[orig] = true;
        } else {
            break;
        }
    }
    rejected
}

/// Benjamini-Hochberg FDR mask (largest i with p_(i) <= alpha*i/N).
pub fn bh_fdr(pvalues: &[f64], alpha: f64) -> Vec<bool> {
    let n = pvalues.len();
    if n == 0 {
        return Vec::new();
    }
    let order = argsort_stable(pvalues);
    let mut k_opt: Option<usize> = None;
    for (k, &orig) in order.iter().enumerate() {
        let thr = alpha * ((k + 1) as f64) / (n as f64);
        if pvalues[orig] <= thr {
            k_opt = Some(k);
        }
    }
    let mut rejected = vec![false; n];
    if let Some(k) = k_opt {
        for &orig in order.iter().take(k + 1) {
            rejected[orig] = true;
        }
    }
    rejected
}

/// One-sided p-values for H0: SR_true=0 under iid-Normal: 1 - Phi(SR_hat).
/// Panics if T < 2 (matching the Python ValueError).
pub fn sharpe_pvalues(trial_sharpes: &[f64], t: usize) -> Vec<f64> {
    assert!(t >= 2, "need T >= 2 returns, got {t}");
    let nd = standard_normal();
    trial_sharpes.iter().map(|&z| 1.0 - nd.cdf(z)).collect()
}

/// `np.percentile(xs, q)` with the default 'linear' method. q in [0,100].
pub fn percentile_linear(xs: &[f64], q: f64) -> f64 {
    let mut v: Vec<f64> = xs.to_vec();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = v.len();
    if n == 0 {
        return f64::NAN;
    }
    if n == 1 {
        return v[0];
    }
    let pos = (q / 100.0) * ((n - 1) as f64);
    let lo = pos.floor() as usize;
    let hi = pos.ceil() as usize;
    if lo == hi {
        return v[lo];
    }
    let frac = pos - (lo as f64);
    v[lo] + (v[hi] - v[lo]) * frac
}

/// Scaled Sharpe (sqrt(T)*mean/std(ddof=1)); 0.0 if sd<=0 (WRC convention).
fn scaled_sharpe_col(col: &[f64]) -> f64 {
    let t = col.len();
    if t < 2 {
        return 0.0;
    }
    let tf = t as f64;
    let mean = col.iter().sum::<f64>() / tf;
    let ss: f64 = col.iter().map(|&x| (x - mean) * (x - mean)).sum();
    let sd = (ss / (tf - 1.0)).sqrt();
    if sd <= 0.0 {
        return 0.0;
    }
    tf.sqrt() * mean / sd
}

/// Result of White's Reality Check.
pub struct WrcResult {
    pub pvalue: f64,
    pub v_obs: f64,
    pub v_dist: Vec<f64>,
}

/// White (2000) Reality Check given a PRE-DRAWN resample-index matrix.
/// `r` is (T,N) as r[t][n]; columns are centred before resampling. V_obs
/// from RAW returns; V_dist from CENTRED resampled returns;
/// pvalue = mean(V_dist >= V_obs). Deterministic in the indices.
pub fn white_reality_check_indexed(r: &[Vec<f64>], index_matrix: &[Vec<usize>]) -> WrcResult {
    let t = r.len();
    let n = if t > 0 { r[0].len() } else { 0 };
    assert!(t >= 2 && n >= 1, "need T>=2, N>=1; got T={t}, N={n}");

    let mut v_obs = f64::NEG_INFINITY;
    for ni in 0..n {
        let col: Vec<f64> = (0..t).map(|ti| r[ti][ni]).collect();
        let s = scaled_sharpe_col(&col);
        if s > v_obs {
            v_obs = s;
        }
    }

    let mut col_mean = vec![0.0f64; n];
    for ni in 0..n {
        let mut acc = 0.0;
        for ti in 0..t {
            acc += r[ti][ni];
        }
        col_mean[ni] = acc / (t as f64);
    }

    let mut v_dist: Vec<f64> = Vec::with_capacity(index_matrix.len());
    for idx in index_matrix {
        assert!(idx.len() == t, "index row length {} != T {}", idx.len(), t);
        let mut v_max = f64::NEG_INFINITY;
        for ni in 0..n {
            let col: Vec<f64> = idx.iter().map(|&row| r[row][ni] - col_mean[ni]).collect();
            let s = scaled_sharpe_col(&col);
            if s > v_max {
                v_max = s;
            }
        }
        v_dist.push(v_max);
    }

    let ge = v_dist.iter().filter(|&&x| x >= v_obs).count();
    let pvalue = if v_dist.is_empty() {
        f64::NAN
    } else {
        (ge as f64) / (v_dist.len() as f64)
    };
    WrcResult { pvalue, v_obs, v_dist }
}

/// Romano-Wolf result: per-strategy mask plus the critical value and
/// observed t-stats (exposed so the parity harness can compare floats, not
/// just the brittle boolean mask, Lens C D8).
pub struct RwResult {
    pub rejected: Vec<bool>,
    pub crit: f64,
    pub t_obs: Vec<f64>,
}

/// Romano-Wolf single-step studentised max-T given a PRE-DRAWN index
/// matrix. `crit = percentile_linear(max_t_dist, 100*(1-alpha))`; reject
/// t_obs > crit. sd==0 -> sd:=1 (matches Python `sd[sd==0]=1.0`).
/// Single-step variant (NOT the full step-down), matching multitest.py.
pub fn romano_wolf_indexed(
    r: &[Vec<f64>],
    index_matrix: &[Vec<usize>],
    alpha: f64,
) -> RwResult {
    let t = r.len();
    let n = if t > 0 { r[0].len() } else { 0 };
    assert!(t >= 2 && n >= 1, "need T>=2, N>=1; got T={t}, N={n}");

    let mut t_obs = vec![0.0f64; n];
    let mut col_mean = vec![0.0f64; n];
    for ni in 0..n {
        let col: Vec<f64> = (0..t).map(|ti| r[ti][ni]).collect();
        let tf = t as f64;
        let mean = col.iter().sum::<f64>() / tf;
        col_mean[ni] = mean;
        let ss: f64 = col.iter().map(|&x| (x - mean) * (x - mean)).sum();
        let mut sd = (ss / (tf - 1.0)).sqrt();
        if sd == 0.0 {
            sd = 1.0;
        }
        t_obs[ni] = tf.sqrt() * mean / sd;
    }

    let mut max_t_dist: Vec<f64> = Vec::with_capacity(index_matrix.len());
    for idx in index_matrix {
        assert!(idx.len() == t, "index row length {} != T {}", idx.len(), t);
        let mut max_t = f64::NEG_INFINITY;
        for ni in 0..n {
            let tf = t as f64;
            let col: Vec<f64> = idx.iter().map(|&row| r[row][ni] - col_mean[ni]).collect();
            let mean = col.iter().sum::<f64>() / tf;
            let ss: f64 = col.iter().map(|&x| (x - mean) * (x - mean)).sum();
            let mut sd = (ss / (tf - 1.0)).sqrt();
            if sd == 0.0 {
                sd = 1.0;
            }
            let tb = tf.sqrt() * mean / sd;
            if tb > max_t {
                max_t = tb;
            }
        }
        max_t_dist.push(max_t);
    }

    let crit = percentile_linear(&max_t_dist, 100.0 * (1.0 - alpha));
    let rejected = t_obs.iter().map(|&to| to > crit).collect();
    RwResult { rejected, crit, t_obs }
}

/// One-line summary mirroring `multitest.py::report`.
pub fn report(trial_sharpes: &[f64], t: usize, alpha: f64) -> String {
    let p = sharpe_pvalues(trial_sharpes, t);
    let nb = bonferroni(&p, alpha).iter().filter(|&&x| x).count();
    let nh = holm(&p, alpha).iter().filter(|&&x| x).count();
    let nf = bh_fdr(&p, alpha).iter().filter(|&&x| x).count();
    format!(
        "  MTC  | N={}  T={}  alpha={:.2}  Bonferroni={}  Holm={}  BH-FDR={}",
        p.len(), t, alpha, nb, nh, nf
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bonferroni_basic() {
        let p = [0.001, 0.02, 0.2, 0.5];
        assert_eq!(bonferroni(&p, 0.05), vec![true, false, false, false]);
    }

    #[test]
    fn holm_step_down_breaks() {
        let p = [0.001, 0.013, 0.2, 0.5];
        assert_eq!(holm(&p, 0.05), vec![true, true, false, false]);
    }

    #[test]
    fn bh_fdr_largest_index() {
        let p = [0.001, 0.008, 0.039, 0.5];
        assert_eq!(bh_fdr(&p, 0.05), vec![true, true, false, false]);
    }

    #[test]
    fn percentile_matches_numpy_linear() {
        let xs = [1.0, 2.0, 3.0, 4.0];
        assert!((percentile_linear(&xs, 50.0) - 2.5).abs() < 1e-12);
        assert!((percentile_linear(&xs, 25.0) - 1.75).abs() < 1e-12);
        assert!((percentile_linear(&xs, 90.0) - 3.7).abs() < 1e-12);
    }
}
