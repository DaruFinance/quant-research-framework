"""Deflated Sharpe Ratio (Bailey & López de Prado, 2014).

Selection bias correction for the in-sample Sharpe ratio: when a
strategy's lookback (or any other parameter) is selected by maximising
SR over a grid of N trials, the maximised SR is upward-biased. The
deflated SR is the probability that the true SR exceeds zero
conditional on the observed SR, the trial count, and the higher
moments of the per-trade returns.

Formulas (Bailey & López de Prado 2014, JPM 40(5):94--107):

    SR_0 = sqrt(V[SR_n]) * ((1 - euler_gamma) * Phi^{-1}(1 - 1/N)
                          + euler_gamma * Phi^{-1}(1 - 1/(N * e)))
    DSR  = Phi(  (SR_hat - SR_0) * sqrt(T - 1)
               / sqrt(1 - g_3 * SR_hat + (g_4 - 1) * SR_hat^2 / 4) )

where:
    SR_hat   = the chosen (in-sample maximised) Sharpe
    V[SR_n]  = sample variance of the N trial Sharpes
    N        = number of effectively independent trials
    T        = number of returns the chosen Sharpe was computed from
    g_3, g_4 = sample skewness and excess kurtosis of the chosen
               strategy's per-trade returns
    Phi      = standard-normal CDF, Phi^{-1} = its inverse
    e        = Euler's number, euler_gamma ≈ 0.5772156649

Public API:

    deflated_sharpe_ratio(
        sharpe_chosen, trial_sharpes, returns
    ) -> float in [0, 1]

Returns the DSR (a probability). Reject the null SR_true <= 0 at
1 - alpha if DSR > 1 - alpha.

This module is a *post-processing* utility — it does not run inside
the engine and does not affect cross-language parity. The Rust
mirror is on the v0.4.0 roadmap.
"""
from __future__ import annotations

import math
from typing import Sequence

import numpy as np
from scipy.stats import norm

EULER_GAMMA = 0.5772156649015329


def expected_max_sharpe_under_null(trial_sharpes: Sequence[float]) -> float:
    """E[max SR_n] under the null that the true SR is zero, using the
    closed form of Bailey & López de Prado 2014 §3 (the SR_0 quantity)."""
    arr = np.asarray(trial_sharpes, dtype=float)
    arr = arr[np.isfinite(arr)]
    n = len(arr)
    if n < 2:
        return 0.0
    var_sr = float(np.var(arr, ddof=1))
    if var_sr <= 0.0:
        return 0.0
    sr_0 = math.sqrt(var_sr) * (
        (1.0 - EULER_GAMMA) * norm.ppf(1.0 - 1.0 / n)
        + EULER_GAMMA       * norm.ppf(1.0 - 1.0 / (n * math.e))
    )
    return float(sr_0)


def deflated_sharpe_ratio(
    sharpe_chosen: float,
    trial_sharpes: Sequence[float],
    returns: Sequence[float],
) -> float:
    """Probability the true Sharpe exceeds zero, conditional on the
    observed in-sample maximised Sharpe, the per-trial Sharpe variance,
    and the per-trade return moments.

    Returns a value in [0, 1]; values near 1 indicate the chosen SR is
    unlikely to be a chance maximum over the grid.
    """
    rets = np.asarray(returns, dtype=float)
    rets = rets[np.isfinite(rets)]
    t = len(rets)
    if t < 3 or not math.isfinite(sharpe_chosen):
        return float("nan")

    sr_0 = expected_max_sharpe_under_null(trial_sharpes)

    # Per-trade-return higher moments. Skewness uses the standard
    # bias-corrected estimator; excess kurtosis is g_4 - 3 in
    # statistical software, but Bailey--LdP use g_4 = E[(x-mu)^4]/sigma^4
    # (raw kurtosis) so we report the raw form.
    mu  = float(rets.mean())
    sd  = float(rets.std(ddof=1))
    if sd <= 0.0:
        return float("nan")
    z   = (rets - mu) / sd
    g_3 = float(np.mean(z ** 3))             # skewness
    g_4 = float(np.mean(z ** 4))             # raw kurtosis (3 for normal)

    denom_sq = 1.0 - g_3 * sharpe_chosen + (g_4 - 1.0) * sharpe_chosen ** 2 / 4.0
    if denom_sq <= 0.0:
        return float("nan")

    z_hat = (sharpe_chosen - sr_0) * math.sqrt(t - 1) / math.sqrt(denom_sq)
    return float(norm.cdf(z_hat))


def report(
    sharpe_chosen: float,
    trial_sharpes: Sequence[float],
    returns: Sequence[float],
) -> str:
    """Format DSR + components as a one-line summary suitable for
    appending to the engine's stdout metric block."""
    sr_0 = expected_max_sharpe_under_null(trial_sharpes)
    dsr  = deflated_sharpe_ratio(sharpe_chosen, trial_sharpes, returns)
    n    = sum(1 for s in trial_sharpes if math.isfinite(s))
    return (f"  DSR  | SR_chosen:{sharpe_chosen:6.2f}  "
            f"E[max SR|null,N={n}]:{sr_0:6.2f}  "
            f"haircut:{sharpe_chosen-sr_0:6.2f}  "
            f"P(SR_true>0):{dsr:5.3f}")
