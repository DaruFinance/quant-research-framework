//! Combinatorial-Symmetric Cross-Validation (CSCV) and Probability of
//! Backtest Overfitting (PBO), Rust mirror of `backtester/pbo.py`.
//!
//! Bailey, Borwein, López de Prado & Zhu (2014), "Pseudo-Mathematics and
//! Financial Charlatanism", Notices of the AMS 61(5):458--471.
//!
//! Given a per-bar equity matrix `M` of shape `(T, N)` given row-major as
//! `m[t][n]` (T rows of bars, N strategy columns), CSCV partitions the T
//! bars into S consecutive folds, takes every S/2-sized subset of folds as
//! in-sample and the complement as out-of-sample, picks the IS-best
//! strategy n*, and records the logit of its OOS rank. PBO = fraction of
//! splits whose logit is <= 0.
//!
//! Numeric parity contract with `pbo.py`:
//!   * per-bar returns = first difference with prepend=M[0] so row 0 = 0
//!     (pbo.py:122);
//!   * `sharpe` = (mean/std(ddof=1))*sqrt(n), 0.0 if <2 finite or sd<=0,
//!     computed in that grouping order (pbo.py:55);
//!   * fold edges = np.linspace(0,T,S+1,dtype=int): step = T/S (ONE f64
//!     division), interior edge_k = (k*step) as usize, and the LAST edge
//!     is forced to exactly T (numpy endpoint=True override). See
//!     `fold_edges`; this is the corrected version (Lens C D6 / Lens B D1):
//!     `(k*T/S) as usize` alone does NOT reproduce numpy for S not a power
//!     of two.
//!   * OOS rank ascending (1=lowest), STABLE ties by original index
//!     (np.argsort kind="stable", pbo.py:144);
//!   * logit clip to copysign(20, r_out-(N+1)/2) at the degenerate guard
//!     (pbo.py:151-152);
//!   * pbo = mean(lambda <= 0) (pbo.py:158).
//!
//! No statrs (rank/logit arithmetic). `itertools` is NOT a dependency:
//! the S/2-of-S combinations are produced by a hand-rolled lexicographic
//! iterator. Post-processing only; cross-language guarantee is
//! `tools/parity_pbo.py`.

/// Per-bar Sharpe: (mean/std(ddof=1))*sqrt(n). Mirrors `pbo.py::_sharpe`
/// grouping exactly (division materialised before the sqrt multiply).
fn sharpe(returns: &[f64]) -> f64 {
    let r: Vec<f64> = returns.iter().copied().filter(|x| x.is_finite()).collect();
    let n = r.len();
    if n < 2 {
        return 0.0;
    }
    let nf = n as f64;
    let mean = r.iter().sum::<f64>() / nf;
    let ss: f64 = r.iter().map(|&x| (x - mean) * (x - mean)).sum();
    let sd = (ss / (nf - 1.0)).sqrt();
    if sd <= 0.0 {
        return 0.0;
    }
    (mean / sd) * nf.sqrt()
}

/// Fold edges reproducing `np.linspace(0, T, S+1, dtype=int)`. Interior
/// edges are `(k*step) as usize` with a single `step = T/S` division; the
/// final edge is forced to exactly `T` (numpy's endpoint override). This
/// is the corrected version, `(k*T/S) as usize` alone diverges from numpy
/// for non-power-of-two S (Lens C D6).
fn fold_edges(t: usize, s: usize) -> Vec<usize> {
    let step = (t as f64) / (s as f64);
    let mut edges = vec![0usize; s + 1];
    for k in 0..s {
        edges[k] = ((k as f64) * step) as usize;
    }
    edges[s] = t; // numpy endpoint=True forces the last sample to T.
    edges
}

/// Lexicographic iterator over all size-`k` subsets of `0..n`, in the same
/// order as Python's `itertools.combinations(range(n), k)`. Each yielded
/// vec is strictly increasing. Empty when `k > n`.
pub fn combinations_half(n: usize, k: usize) -> Vec<Vec<usize>> {
    let mut out: Vec<Vec<usize>> = Vec::new();
    if k > n {
        return out;
    }
    if k == 0 {
        out.push(Vec::new());
        return out;
    }
    let mut idx: Vec<usize> = (0..k).collect();
    loop {
        out.push(idx.clone());
        let mut i = k;
        loop {
            if i == 0 {
                return out;
            }
            i -= 1;
            if idx[i] != i + n - k {
                break;
            }
        }
        idx[i] += 1;
        for j in (i + 1)..k {
            idx[j] = idx[j - 1] + 1;
        }
    }
}

/// Stable ascending argsort, matching `np.argsort(kind="stable")`.
fn argsort_stable(xs: &[f64]) -> Vec<usize> {
    let mut order: Vec<usize> = (0..xs.len()).collect();
    order.sort_by(|&a, &b| match xs[a].partial_cmp(&xs[b]) {
        Some(std::cmp::Ordering::Equal) | None => a.cmp(&b),
        Some(ord) => ord,
    });
    order
}

/// Result of a CSCV run. Mirrors the dict returned by `pbo.py::cscv`.
pub struct Cscv {
    pub pbo: f64,
    pub lambdas: Vec<f64>,
    pub n_splits: usize,
    pub s: usize,
    pub n: usize,
    pub t: usize,
}

/// Run CSCV on a `(T, N)` equity matrix given row-major as `m[t][n]`.
/// `s` must be even. Panics on the same invalid inputs `pbo.py::cscv`
/// raises ValueError for (S odd, N<2, T<S).
pub fn cscv(m: &[Vec<f64>], s: usize) -> Cscv {
    let t = m.len();
    let n = if t > 0 { m[0].len() } else { 0 };
    assert!(s % 2 == 0, "S must be even, got {s}");
    assert!(n >= 2, "need at least 2 strategies, got N={n}");
    assert!(t >= s, "need at least S={s} bars, got T={t}");

    // Per-bar returns: prepend=M[0] -> row 0 is 0.
    let mut rets = vec![vec![0.0f64; n]; t];
    for ti in 1..t {
        for ni in 0..n {
            rets[ti][ni] = m[ti][ni] - m[ti - 1][ni];
        }
    }

    let edges = fold_edges(t, s);
    let folds: Vec<(usize, usize)> = (0..s).map(|k| (edges[k], edges[k + 1])).collect();

    let half = s / 2;
    let mut lambdas: Vec<f64> = Vec::new();

    for c in combinations_half(s, half) {
        let mut mask_in = vec![false; t];
        for &k in &c {
            let (a, b) = folds[k];
            for bar in a..b {
                mask_in[bar] = true;
            }
        }

        let mut sr_in = vec![0.0f64; n];
        let mut sr_out = vec![0.0f64; n];
        for ni in 0..n {
            let mut col_in: Vec<f64> = Vec::new();
            let mut col_out: Vec<f64> = Vec::new();
            for bar in 0..t {
                if mask_in[bar] {
                    col_in.push(rets[bar][ni]);
                } else {
                    col_out.push(rets[bar][ni]);
                }
            }
            sr_in[ni] = sharpe(&col_in);
            sr_out[ni] = sharpe(&col_out);
        }

        // n* = argmax sr_in. np.argmax returns the FIRST max on ties.
        let mut n_star = 0usize;
        let mut best = sr_in[0];
        for ni in 1..n {
            if sr_in[ni] > best {
                best = sr_in[ni];
                n_star = ni;
            }
        }

        // OOS rank of n*: ascending, 1..N, stable ties.
        let order = argsort_stable(&sr_out);
        let mut rank = vec![0usize; n];
        for (pos, &orig) in order.iter().enumerate() {
            rank[orig] = pos + 1;
        }
        let r_out = rank[n_star] as f64;

        let denom = (n as f64) + 1.0 - r_out;
        let lam = if denom <= 0.0 || r_out <= 0.0 {
            (20.0f64).copysign(r_out - ((n as f64) + 1.0) / 2.0)
        } else {
            (r_out / denom).ln()
        };
        lambdas.push(lam);
    }

    let n_splits = lambdas.len();
    let n_le0 = lambdas.iter().filter(|&&x| x <= 0.0).count();
    let pbo = if n_splits == 0 {
        0.0
    } else {
        (n_le0 as f64) / (n_splits as f64)
    };

    Cscv { pbo, lambdas, n_splits, s, n, t }
}

/// Convenience: scalar PBO. Mirrors `pbo.py::pbo`.
pub fn pbo(m: &[Vec<f64>], s: usize) -> f64 {
    cscv(m, s).pbo
}

/// One-line summary mirroring `pbo.py::report`.
pub fn report(m: &[Vec<f64>], s: usize) -> String {
    let out = cscv(m, s);
    format!(
        "  PBO  | S={}  splits={}  N={}  T={}  PBO={:.3}",
        out.s, out.n_splits, out.n, out.t, out.pbo
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn combinations_match_itertools_order() {
        let got = combinations_half(4, 2);
        let want: Vec<Vec<usize>> = vec![
            vec![0, 1], vec![0, 2], vec![0, 3],
            vec![1, 2], vec![1, 3], vec![2, 3],
        ];
        assert_eq!(got, want);
        assert_eq!(combinations_half(8, 4).len(), 70);
        assert_eq!(combinations_half(16, 8).len(), 12870);
    }

    #[test]
    fn argsort_stable_breaks_ties_by_index() {
        let xs = [0.5, 0.1, 0.5, 0.1];
        assert_eq!(argsort_stable(&xs), vec![1, 3, 0, 2]);
    }

    #[test]
    fn fold_edges_force_endpoint_nonpow2() {
        // numpy: np.linspace(0,122,15,dtype=int)[7] == 60 (NOT 61).
        let e = fold_edges(122, 14);
        assert_eq!(e[0], 0);
        assert_eq!(e[14], 122); // endpoint forced
        assert_eq!(e[7], 60);
        // np.linspace(0,230,15,dtype=int)[7] == 114.
        assert_eq!(fold_edges(230, 14)[7], 114);
    }

    #[test]
    fn golden_pbo_small_panel() {
        let m: Vec<Vec<f64>> = vec![
            vec![1.00, 1.00, 1.00, 1.00],
            vec![1.02, 1.01, 0.99, 1.03],
            vec![1.05, 1.00, 1.01, 1.02],
            vec![1.03, 1.02, 1.00, 1.05],
            vec![1.07, 1.01, 1.02, 1.04],
            vec![1.06, 1.03, 1.01, 1.07],
            vec![1.09, 1.02, 1.03, 1.06],
            vec![1.08, 1.04, 1.02, 1.09],
        ];
        let out = cscv(&m, 8);
        assert_eq!(out.n_splits, 70);
        assert!((0.0..=1.0).contains(&out.pbo));
    }
}
