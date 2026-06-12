"""Haircut Sharpe Ratio (Harvey & Liu 2015).

Harvey, C. R. and Liu, Y. (2015), "Backtesting", Journal of Portfolio
Management 42(1):13--28. (SSRN 2345489.)

Given an *observed* annualised Sharpe ratio whose t-statistic was the
winner of a multiple-testing search over ``n_tests`` candidate
strategies, the haircut adjusts the Sharpe downward for multiple-testing
inflation:

  1.  t_obs = SR_annual * sqrt(T / freq)   (= SR_per_period * sqrt(T)).
  2.  Two-sided observed p-value  p_obs = 2 * (1 - Phi(|t_obs|)).
  3.  Adjust p_obs for ``n_tests`` simultaneous tests with Bonferroni or
      BHY (Benjamini-Hochberg-Yekutieli, the paper's default).
  4.  t_adj = Phi^{-1}(1 - p_adj / 2);  SR_haircut = SR_annual*(t_adj/t_obs);
      haircut_pct = 1 - t_adj/t_obs.

The single-test case (``n_tests = 1``) leaves the SR unchanged.

Methods (Lens B D4)
-------------------
This module offers the two distinct *single-reported-statistic* closed
forms: 'bonferroni' (p_adj = min(1, p_obs*N)) and 'bhy' (p_adj =
min(1, p_obs*N*c(N)), c(N)=sum_{i=1..N} 1/i). A 'holm' option is NOT
offered: for the single most-significant reported statistic Holm's
multiplier equals N, so it is numerically IDENTICAL to Bonferroni and
would mislead a reader into thinking a distinct correction was applied.
The genuine step-down Holm/BHY power recovery only appears across a
*family* of p-values, which this single-statistic closed form does not
have. If the full Harvey-Liu simulation-based three-method haircut is
wanted later, that is a separate, larger procedure (see RISKS).

BHY note
--------
BHY uses the dependency constant c(N) = sum_{i=1..N} 1/i, which is NOT
the plain Benjamini-Hochberg in ``backtester.multitest.bh_fdr`` (that one
omits c(N)). BHY is therefore implemented inline here; Bonferroni reuses
``multitest.bonferroni`` so there is one implementation of it.

Post-processing only; the Rust mirror is ``src/haircut.rs`` behind the
``overfit`` Cargo feature; cross-language guarantee is
``tools/parity_multitest.py``.
"""
from __future__ import annotations

import math
from typing import Dict

import numpy as np

# Reuse the framework's normal CDF (scipy with numeric fallback).
from backtester.multitest import _PHI, bonferroni  # noqa: F401  (bonferroni: one-impl invariant)

try:
    from scipy.stats import norm
    _PHI_INV = lambda q: float(norm.ppf(q))
except Exception:                                              # pragma: no cover
    # Acklam's inverse-normal approximation (|err| < 1.15e-9); only used
    # if scipy is unavailable. statrs (Rust) uses its exact inverse_cdf.
    def _PHI_INV(q: float) -> float:
        if q <= 0.0:
            return -math.inf
        if q >= 1.0:
            return math.inf
        a = [-3.969683028665376e+01, 2.209460984245205e+02,
             -2.759285104469687e+02, 1.383577518672690e+02,
             -3.066479806614716e+01, 2.506628277459239e+00]
        b = [-5.447609879822406e+01, 1.615858368580409e+02,
             -1.556989798598866e+02, 6.680131188771972e+01,
             -1.328068155288572e+01]
        c = [-7.784894002430293e-03, -3.223964580411365e-01,
             -2.400758277161838e+00, -2.549732539343734e+00,
             4.374664141464968e+00, 2.938163982698783e+00]
        d = [7.784695709041462e-03, 3.224671290700398e-01,
             2.445134137142996e+00, 3.754408661907416e+00]
        plow, phigh = 0.02425, 1 - 0.02425
        if q < plow:
            ql = math.sqrt(-2 * math.log(q))
            return (((((c[0]*ql+c[1])*ql+c[2])*ql+c[3])*ql+c[4])*ql+c[5]) / \
                   ((((d[0]*ql+d[1])*ql+d[2])*ql+d[3])*ql+1)
        if q > phigh:
            ql = math.sqrt(-2 * math.log(1 - q))
            return -(((((c[0]*ql+c[1])*ql+c[2])*ql+c[3])*ql+c[4])*ql+c[5]) / \
                    ((((d[0]*ql+d[1])*ql+d[2])*ql+d[3])*ql+1)
        ql = q - 0.5
        r = ql*ql
        return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*ql / \
               (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)


def _c_harmonic(n: int) -> float:
    """BHY dependency constant c(N) = sum_{i=1..N} 1/i."""
    return float(sum(1.0 / i for i in range(1, n + 1)))


def haircut_sharpe_ratio(
    sharpe_annual: float,
    T: int,
    n_tests: int,
    method: str = "bhy",
    freq: float = 252.0,
) -> Dict[str, float]:
    """Harvey-Liu (2015) haircut of an observed annualised Sharpe ratio.

    Parameters
    ----------
    sharpe_annual : float   observed annualised SR (the search winner).
    T : int                 number of return observations behind the SR.
    n_tests : int           effective number of strategies tried (NOT
                            windows*combos). n_tests<=1 -> zero haircut.
    method : {'bonferroni','bhy'}, default 'bhy'.
    freq : float            periods/year for SR<->t conversion (252 daily).

    Returns dict: haircut_sr, haircut_pct, p_obs, p_adj, t_obs, t_adj.
    """
    if T < 2:
        raise ValueError(f"need T >= 2 observations, got {T}")
    if n_tests < 1:
        raise ValueError(f"n_tests must be >= 1, got {n_tests}")
    m = method.lower()
    if m not in ("bonferroni", "bhy"):
        raise ValueError(f"method must be bonferroni|bhy, got {method!r}")

    sr_period = sharpe_annual / math.sqrt(freq)
    t_obs = sr_period * math.sqrt(T)

    p_obs = 2.0 * (1.0 - _PHI(abs(t_obs)))
    p_obs = min(max(p_obs, 0.0), 1.0)

    if m == "bonferroni":
        p_adj = min(1.0, p_obs * n_tests)
    else:  # bhy
        p_adj = min(1.0, p_obs * n_tests * _c_harmonic(n_tests))

    if p_adj >= 1.0:
        t_adj = 0.0
    else:
        t_adj = _PHI_INV(1.0 - p_adj / 2.0)
    t_adj = max(t_adj, 0.0)

    if t_obs == 0.0:
        haircut_pct = 0.0
        haircut_sr = sharpe_annual
    else:
        ratio = t_adj / abs(t_obs)
        haircut_sr = sharpe_annual * ratio
        haircut_pct = 1.0 - ratio

    return {
        "haircut_sr": float(haircut_sr),
        "haircut_pct": float(haircut_pct),
        "p_obs": float(p_obs),
        "p_adj": float(p_adj),
        "t_obs": float(t_obs),
        "t_adj": float(t_adj),
    }


def report(
    sharpe_annual: float, T: int, n_tests: int,
    method: str = "bhy", freq: float = 252.0,
) -> str:
    """One-line Harvey-Liu haircut summary. HCUT| prefix avoids the
    field-name collision with dsr.report's 'haircut:' selection field."""
    out = haircut_sharpe_ratio(sharpe_annual, T, n_tests, method, freq)
    return (f"  HCUT | method={method}  N={n_tests}  T={T}  "
            f"SR_obs:{sharpe_annual:6.2f}  SR_hc:{out['haircut_sr']:6.2f}  "
            f"cut:{out['haircut_pct']*100:5.1f}%  p_adj:{out['p_adj']:.3g}")
