"""Extra metric primitives (item #44, Phase 2).

The single-asset kernel emits a fixed 8-metric tuple (Trades, ROI,
PF, WinRate, Exp, Sharpe, MaxDD, Consistency). This module adds
metrics that consume the same `returns` series the kernel already
produces, without touching the Numba hot loop:

- ``sortino(returns, annualization)``: mean return / downside
  deviation, annualised by ``sqrt(annualization)``. The missing
  cousin to ``Sharpe`` in the v0.4.0 baseline — item #44 finally
  lands it.

- ``turnover(positions)``: sum of absolute position changes per bar.
  Used as the third term of the multi_term objective.

These functions are **pure data-only**: they take a returns / positions
array and emit a scalar. Lookahead concerns sit at the caller (slice
the returns to the IS window before passing in).
"""
from __future__ import annotations

import numpy as np


def sortino(returns: np.ndarray, annualization: float | int | None = None) -> float:
    """Sortino ratio. Like Sharpe but only penalises downside vol.

    Parameters
    ----------
    returns : 1-D array of per-bar / per-trade returns.
    annualization : optional float. If given, multiplies the raw ratio
        by ``sqrt(annualization)`` so the output is annualised in the
        same units the caller assumed. If ``None``, returns the
        unannualised ratio.

    Returns
    -------
    float. ``NaN`` if downside deviation is zero (or fewer than 2
    losses).
    """
    r = np.asarray(returns, dtype=np.float64)
    if r.size < 2:
        return float("nan")
    downside = r[r < 0]
    if downside.size < 2:
        return float("nan")
    # Sortino uses downside *deviation* (ddof=0) over all returns by
    # one convention, or downside std (ddof=1) over negative-only by
    # another. We match the latter (LPM_2 with target 0 over all
    # observations) because it's the one consistently produced by
    # pyportfolioopt / numpy-financial / etc.
    semi_dev = float(np.sqrt(np.mean(np.minimum(r, 0.0) ** 2)))
    if semi_dev == 0.0:
        return float("nan")
    mean_r = float(r.mean())
    out = mean_r / semi_dev
    if annualization is not None:
        out *= float(annualization) ** 0.5
    return out


def turnover(positions: np.ndarray) -> float:
    """Sum of absolute position changes across the trajectory. For a
    position series ``[w_1, w_2, ..., w_n]`` returns
    ``sum_t |w_{t+1} - w_t|``. Used as the third term of the
    multi_term objective."""
    p = np.asarray(positions, dtype=np.float64)
    if p.size < 2:
        return 0.0
    return float(np.sum(np.abs(np.diff(p, axis=0))))


__all__ = ["sortino", "turnover"]
