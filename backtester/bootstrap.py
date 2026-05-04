"""Stationary bootstrap (Politis & Romano 1994) for backtest-friendly
serial-correlated resampling.

Politis, D. N. and Romano, J. P. (1994), "The Stationary Bootstrap",
JASA 89(428):1303--1313. DOI 10.1080/01621459.1994.10476870.

Most backtests produce a per-bar return series with material serial
correlation (autocorrelation in the inputs, momentum and mean-reversion
in the strategy itself). Standard iid bootstrap discards the
correlation structure; the stationary bootstrap preserves it by
resampling whole *blocks* of bars whose lengths are drawn from a
geometric distribution with mean ``mean_block``. The resampled series
inherits the autocorrelation function of the original up to scale,
which makes it appropriate for variance estimation of statistics that
depend on the time order (Sharpe, max drawdown, OOS Sharpe of an
in-sample-fit strategy, etc.).

This module provides:

  - ``stationary_bootstrap(returns, n_resamples=1000, mean_block=10, seed=42)``
    -> ndarray of shape (n_resamples, T): one resampled return series
    per row.

  - ``bootstrap_ci(returns, statistic, level=0.95, ...)`` -> (low, high):
    a percentile confidence interval for an arbitrary statistic (any
    callable mapping a 1-D return array to a scalar).

The procedure matches the principled alternative to the framework's
fixed-multiplier robustness scenarios (FEE_SHOCK 2x, SLIPPAGE_SHOCK 3x,
INDICATOR_VARIANCE +/-1) flagged as folk engineering in the paper.
This is post-processing only; it does not run inside the engine and
does not affect cross-language parity. The Rust mirror is on the
v0.5.x roadmap.
"""
from __future__ import annotations

from typing import Callable
import numpy as np


def _draw_block_lengths(
    n_blocks_max: int, mean_block: float, rng: np.random.Generator
) -> np.ndarray:
    """Draw block lengths from a Geometric(p) distribution with mean
    mean_block. NumPy's geometric is shifted (lengths >= 1)."""
    p = 1.0 / mean_block
    return rng.geometric(p, size=n_blocks_max)


def stationary_bootstrap(
    returns: np.ndarray,
    n_resamples: int = 1000,
    mean_block: float = 10.0,
    seed: int | None = 42,
) -> np.ndarray:
    """Draw ``n_resamples`` stationary-bootstrap replicas of ``returns``.

    Each replica is constructed by:
      1. Picking a uniformly random start index in ``[0, T)``.
      2. Copying ``L`` consecutive bars (with wrap-around) where
         ``L`` is Geometric(1/mean_block), expected length ``mean_block``.
      3. Repeating until the replica has length ``T``.

    Parameters
    ----------
    returns : 1-D array of length ``T``.
        The realised per-bar return series.
    n_resamples : int, default 1000.
        Number of bootstrap replicas to draw.
    mean_block : float, default 10.0.
        Expected block length. Politis & Romano recommend
        ``mean_block ~ T^{1/3}`` as a rule of thumb; for daily data with
        ``T = 250``, that gives ``~ 6``; for ``T = 2500`` (10 years
        daily), ``~ 14``.
    seed : int or None.
        RNG seed for reproducibility.

    Returns
    -------
    ndarray of shape ``(n_resamples, T)``.
    """
    r = np.asarray(returns, dtype=float)
    if r.ndim != 1:
        raise ValueError("returns must be 1-D")
    T = len(r)
    if T < 2:
        raise ValueError(f"need at least 2 returns, got {T}")
    if mean_block <= 0:
        raise ValueError(f"mean_block must be positive, got {mean_block}")
    if mean_block >= T:
        # collapses to iid bootstrap of the whole series; fine.
        pass

    rng = np.random.default_rng(seed)
    out = np.empty((n_resamples, T), dtype=float)

    # Pre-allocate generous block-length budget per replica:
    # E[# blocks] = T / mean_block; allocate 4x that to be safe.
    n_blocks_budget = max(1, int(4.0 * T / mean_block))

    for k in range(n_resamples):
        starts = rng.integers(0, T, size=n_blocks_budget)
        lens   = _draw_block_lengths(n_blocks_budget, mean_block, rng)
        idx_list = []
        filled = 0
        for s, L in zip(starts, lens):
            for j in range(L):
                idx_list.append((s + j) % T)
                filled += 1
                if filled >= T:
                    break
            if filled >= T:
                break
        # In case the budget was exhausted (rare with 4x oversize), fill
        # with iid samples for the remainder.
        while filled < T:
            idx_list.append(int(rng.integers(0, T)))
            filled += 1
        out[k] = r[idx_list[:T]]
    return out


def bootstrap_ci(
    returns: np.ndarray,
    statistic: Callable[[np.ndarray], float],
    level: float = 0.95,
    n_resamples: int = 1000,
    mean_block: float = 10.0,
    seed: int | None = 42,
) -> tuple[float, float, np.ndarray]:
    """Percentile confidence interval for ``statistic`` via stationary bootstrap.

    Parameters
    ----------
    returns : 1-D array.
    statistic : callable returning a scalar.
        E.g. ``lambda r: np.sqrt(len(r)) * r.mean() / r.std(ddof=1)``
        for the per-trade Sharpe.
    level : float in (0, 1).
        Two-sided coverage; default 0.95.
    n_resamples, mean_block, seed : as in ``stationary_bootstrap``.

    Returns
    -------
    (low, high, replicas) : tuple
        ``low``, ``high`` are the empirical percentile bounds; ``replicas``
        is the 1-D ndarray of bootstrap statistic values.
    """
    if not (0.0 < level < 1.0):
        raise ValueError(f"level must be in (0, 1), got {level}")
    boot = stationary_bootstrap(returns, n_resamples=n_resamples,
                                mean_block=mean_block, seed=seed)
    replicas = np.array([statistic(boot[k]) for k in range(n_resamples)])
    alpha = 1.0 - level
    low  = float(np.percentile(replicas, 100 * (alpha / 2)))
    high = float(np.percentile(replicas, 100 * (1 - alpha / 2)))
    return low, high, replicas


def per_trade_sharpe(returns: np.ndarray) -> float:
    """Convenience: per-trade Sharpe (sqrt(N) * mean / std), a typical
    statistic of interest."""
    r = np.asarray(returns, dtype=float)
    if len(r) < 2:
        return 0.0
    sd = r.std(ddof=1)
    if sd <= 0.0:
        return 0.0
    return float(np.sqrt(len(r)) * r.mean() / sd)
