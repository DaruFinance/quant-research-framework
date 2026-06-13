"""Position-construction neutralizations (item #7, Phase 2).

Three neutralization modes layer on top of a raw weight vector
(typically the output of an alpha/momentum signal or item #6's
ERC sizer):

- ``dollar``  : enforce gross long notional == gross short notional.
                Pure rescaling; no auxiliary inputs required.
- ``beta``    : enforce ``sum(w_i * beta_i) == 0``. Requires per-asset
                betas estimated from a window of returns at times
                ``< t``.
- ``sigma``   : enforce ``|w_i| * sigma_i`` equal across legs (each
                leg contributes the same volatility). Requires per-
                asset volatilities estimated from returns at ``< t``.

All three functions are **pure data-only**: weights in, weights out;
no series indexing, no time-dependent constants. The lookahead
concern lives at the *caller*, which must build betas / vols from a
returns window strictly preceding the rebalance bar.

The output of ``neutralize`` is always a real-valued NumPy array of
the same length as the input. Sign convention is preserved (positive
weights stay positive, negative stay negative) unless the
neutralization explicitly requires a sign flip — which is documented
on the relevant mode.
"""
from __future__ import annotations

from typing import Literal, Optional

import numpy as np


Mode = Literal["dollar", "beta", "sigma"]


def estimate_betas(returns_window: np.ndarray, market_idx: int) -> np.ndarray:
    """Per-asset beta against a market column inside the same
    returns matrix. Pure OLS slope: ``cov(r_i, r_m) / var(r_m)``.

    Caller MUST pass a returns window that ends strictly before the
    rebalance bar so the betas are leak-free.
    """
    if returns_window.ndim != 2:
        raise ValueError(
            f"returns_window must be 2-D (bars x assets); got "
            f"shape {returns_window.shape}"
        )
    n_assets = returns_window.shape[1]
    if not (0 <= market_idx < n_assets):
        raise ValueError(
            f"market_idx={market_idx} out of range [0, {n_assets})"
        )
    mask = ~np.any(np.isnan(returns_window), axis=1)
    clean = returns_window[mask]
    if clean.shape[0] < 2:
        raise ValueError("returns_window has < 2 clean rows for beta")
    market = clean[:, market_idx]
    market_var = float(np.var(market, ddof=1))
    if market_var <= 0:
        raise ValueError(
            f"market column {market_idx} has zero variance; "
            "cannot estimate betas"
        )
    betas = np.empty(n_assets, dtype=np.float64)
    for i in range(n_assets):
        cov_im = float(np.cov(clean[:, i], market, ddof=1)[0, 1])
        betas[i] = cov_im / market_var
    return betas


def estimate_vols(returns_window: np.ndarray) -> np.ndarray:
    """Per-asset stdev of returns. Caller responsible for using a
    window at times ``< t``."""
    if returns_window.ndim != 2:
        raise ValueError("returns_window must be 2-D")
    mask = ~np.any(np.isnan(returns_window), axis=1)
    clean = returns_window[mask]
    if clean.shape[0] < 2:
        raise ValueError("returns_window has < 2 clean rows for vol")
    return np.std(clean, axis=0, ddof=1)


def _dollar_neutralize(raw_weights: np.ndarray) -> np.ndarray:
    """Gross long notional == gross short notional. Implementation:
    rescale longs and shorts independently so each leg sums to 0.5 in
    absolute value (=> total gross 1.0; net 0)."""
    w = np.asarray(raw_weights, dtype=np.float64)
    longs = w > 0
    shorts = w < 0
    if not longs.any() or not shorts.any():
        raise ValueError(
            "dollar-neutral requires both long and short raw weights; "
            f"got longs={longs.sum()} shorts={shorts.sum()}"
        )
    out = w.copy()
    long_sum = float(out[longs].sum())
    short_sum = float(-out[shorts].sum())
    out[longs]  = out[longs]  * (0.5 / long_sum)
    out[shorts] = out[shorts] * (0.5 / short_sum)
    return out


def _beta_neutralize(
    raw_weights: np.ndarray,
    betas: np.ndarray,
    market_idx: Optional[int] = None,
) -> np.ndarray:
    """Adjust weights so ``sum(w_i * beta_i) == 0``. Uses a single
    scalar shift on the market leg (defaults to the highest-|beta|
    asset if ``market_idx`` not given).

    Sign of the original weights is preserved on non-market legs;
    the market leg's weight may flip sign if needed.
    """
    w = np.asarray(raw_weights, dtype=np.float64).copy()
    b = np.asarray(betas, dtype=np.float64)
    if w.shape != b.shape:
        raise ValueError(
            f"raw_weights and betas shape mismatch: {w.shape} vs {b.shape}"
        )
    if market_idx is None:
        market_idx = int(np.argmax(np.abs(b)))
    if not (0 <= market_idx < len(w)):
        raise ValueError(f"market_idx {market_idx} out of range")
    if b[market_idx] == 0:
        raise ValueError(
            f"market_idx={market_idx} has beta=0; cannot neutralize"
        )
    # Net beta excluding the market leg:
    net_no_market = float(
        np.dot(w[:market_idx], b[:market_idx])
        + np.dot(w[market_idx + 1:], b[market_idx + 1:])
    )
    # Solve w[market_idx] * b[market_idx] + net_no_market = 0
    w[market_idx] = -net_no_market / b[market_idx]
    return w


def _sigma_neutralize(
    raw_weights: np.ndarray,
    vols: np.ndarray,
) -> np.ndarray:
    """Adjust |w_i| so ``|w_i| * sigma_i`` is constant across legs.
    Sign of the raw weight is preserved. The constant is fixed by
    requiring ``sum(|w_i|) == sum(|raw_weights|)`` (gross-preserving
    rescaling)."""
    w = np.asarray(raw_weights, dtype=np.float64)
    v = np.asarray(vols, dtype=np.float64)
    if w.shape != v.shape:
        raise ValueError(
            f"raw_weights and vols shape mismatch: {w.shape} vs {v.shape}"
        )
    if np.any(v <= 0):
        raise ValueError(
            f"sigma-neutral requires positive vols; got {v.tolist()}"
        )
    signs = np.sign(w)
    if not np.all(np.abs(signs) == 1):
        raise ValueError(
            "sigma-neutral requires every raw weight non-zero (signs needed)"
        )
    # equal-vol-contribution: |w_i| = c / sigma_i; choose c so the
    # gross sum is preserved.
    abs_target = 1.0 / v
    gross_in = float(np.abs(w).sum())
    abs_target = abs_target * (gross_in / abs_target.sum())
    return signs * abs_target


def neutralize(
    raw_weights: np.ndarray,
    mode: Mode,
    *,
    betas: Optional[np.ndarray] = None,
    vols: Optional[np.ndarray] = None,
    market_idx: Optional[int] = None,
) -> np.ndarray:
    """Dispatch to one of the three neutralizers.

    ``mode``:
        ``"dollar"`` — no auxiliary input required.
        ``"beta"``   — pass ``betas`` (array of length n_assets) and
                       optionally ``market_idx``.
        ``"sigma"``  — pass ``vols`` (array of length n_assets).
    """
    if mode == "dollar":
        return _dollar_neutralize(raw_weights)
    if mode == "beta":
        if betas is None:
            raise ValueError("mode='beta' requires betas")
        return _beta_neutralize(raw_weights, betas, market_idx)
    if mode == "sigma":
        if vols is None:
            raise ValueError("mode='sigma' requires vols")
        return _sigma_neutralize(raw_weights, vols)
    raise ValueError(f"unknown neutralization mode {mode!r}")


__all__ = [
    "Mode",
    "neutralize",
    "estimate_betas",
    "estimate_vols",
]
