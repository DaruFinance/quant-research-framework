"""Multiple-testing corrections for backtest p-value families.

Lightweight Bonferroni / Holm / Benjamini--Hochberg (BH-FDR) corrections
on top of an array of p-values, plus a convenience wrapper that turns
the optimiser's trial Sharpe array into per-trial p-values under an
iid-Normal approximation.

Bonferroni
----------
``alpha_corr_i = alpha / N`` -- familywise error rate (FWER) <= alpha.
Conservative; under-powers when N is large or the trials are
correlated. Reference: Bonferroni (1936).

Holm
----
Step-down: order p-values ascending, at rank i set
``alpha_corr_i = alpha / (N - i + 1)`` and accept rejecting from the
smallest p as long as p_i <= alpha_corr_i. Reference: Holm (1979).

Benjamini--Hochberg (BH-FDR)
----------------------------
Order p-values ascending, find the largest i with
``p_i <= alpha * i / N``, reject all H_j for j <= i. Controls the
false-discovery rate at alpha. Reference: Benjamini & Hochberg (1995).
We bundle the BH variant rather than BHY because the trial-Sharpe
correlation in this framework's optimiser sweep is positive, the
positive-dependence assumption (PRDS) plausibly holds, and BH is
applicable as written.

Public API
----------
    bonferroni(pvalues, alpha=0.05) -> ndarray of bools (rejected mask)
    holm(pvalues, alpha=0.05)       -> ndarray of bools
    bh_fdr(pvalues, alpha=0.05)     -> ndarray of bools
    sharpe_pvalues(trial_sharpes, T) -> ndarray of (one-sided) p-values

This module is post-processing only; it does not run inside the engine
and does not affect cross-language parity. The Rust mirror is on the
v0.5.x roadmap.
"""
from __future__ import annotations

import math
import numpy as np

# scipy is optional; fall back to a numerical Phi if unavailable.
try:
    from scipy.stats import norm
    _PHI = lambda x: float(norm.cdf(x))
except Exception:                                              # pragma: no cover
    def _PHI(x: float) -> float:
        return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def bonferroni(pvalues: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    """Bonferroni rejection mask. ``alpha_corr = alpha / N``."""
    p = np.asarray(pvalues, dtype=float)
    N = len(p)
    if N == 0:
        return np.zeros(0, dtype=bool)
    return p <= (alpha / N)


def holm(pvalues: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    """Holm step-down rejection mask."""
    p = np.asarray(pvalues, dtype=float)
    N = len(p)
    if N == 0:
        return np.zeros(0, dtype=bool)
    order = np.argsort(p, kind="stable")
    sorted_p = p[order]
    rejected = np.zeros(N, dtype=bool)
    for i, pi in enumerate(sorted_p):
        thr = alpha / (N - i)
        if pi <= thr:
            rejected[order[i]] = True
        else:
            break
    return rejected


def bh_fdr(pvalues: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    """Benjamini-Hochberg rejection mask. Controls FDR <= alpha under PRDS."""
    p = np.asarray(pvalues, dtype=float)
    N = len(p)
    if N == 0:
        return np.zeros(0, dtype=bool)
    order = np.argsort(p, kind="stable")
    sorted_p = p[order]
    thr = alpha * (np.arange(1, N + 1)) / N
    below = sorted_p <= thr
    if not below.any():
        return np.zeros(N, dtype=bool)
    k = int(np.where(below)[0].max())            # largest index satisfying p_i <= thr_i
    rejected = np.zeros(N, dtype=bool)
    rejected[order[: k + 1]] = True
    return rejected


def sharpe_pvalues(trial_sharpes: np.ndarray, T: int) -> np.ndarray:
    """One-sided p-values for ``H0: SR_true = 0`` under iid Normal returns.

    The framework reports the *scaled* per-trade Sharpe
    ``SR_hat = sqrt(T) * mean / sd``. Under iid Normal returns and
    ``H0: SR_true = 0``, ``SR_hat`` is asymptotically Normal(0, 1), so
    the one-sided p-value is ``1 - Phi(SR_hat)``.

    The Lo (2002) variance correction ``(1 + SR^2 / 2) / T`` applies to
    *confidence intervals* around the *estimated* SR (where the SR
    parameter enters the variance), not to the null-hypothesis test
    statistic. Users wanting a CI under non-iid returns should pair the
    framework's SR with the stationary-bootstrap variance estimator in
    ``backtester.bootstrap``.

    Parameters
    ----------
    trial_sharpes : 1-D array of length N.
        Per-trade Sharpes scaled by sqrt(T) (the framework's
        convention; see §4.5 of the v0.4.0 paper).
    T : int
        Number of trades the Sharpe was computed from. Used for the
        small-sample warning only; the asymptotic p-value does not
        depend on T directly.

    Returns
    -------
    1-D ndarray of p-values, same length as ``trial_sharpes``.
    """
    sr = np.asarray(trial_sharpes, dtype=float)
    if T < 2:
        raise ValueError(f"need T >= 2 returns, got {T}")
    return np.array([1.0 - _PHI(zi) for zi in sr])


def report(trial_sharpes: np.ndarray, T: int, alpha: float = 0.05) -> str:
    """One-line summary of the three corrections at the given trial count."""
    p = sharpe_pvalues(trial_sharpes, T)
    n_b = int(bonferroni(p, alpha).sum())
    n_h = int(holm(p, alpha).sum())
    n_f = int(bh_fdr(p, alpha).sum())
    return (f"  MTC  | N={len(p)}  T={T}  alpha={alpha:.2f}  "
            f"Bonferroni={n_b}  Holm={n_h}  BH-FDR={n_f}")
