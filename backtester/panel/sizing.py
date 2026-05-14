"""Portfolio sizing primitives (item #6, Phase 2).

Currently exposes:

- ``equal_weights(n) -> np.ndarray``: trivial 1/N baseline.
- ``erc_weights(returns_window) -> np.ndarray``: equal-risk-contribution
  weights. At convergence, each asset's contribution to portfolio
  variance ``w_i * (Σw)_i`` is equal to ``port_var / n``.

The ERC solver uses ``scipy.optimize.minimize(SLSQP)`` with:
- objective: dispersion of risk contributions
- equality constraint: weights sum to 1
- bounds: each weight in (0, 1]

The function is pure data-only: it takes a 2-D ``returns_window`` of
shape ``(n_bars, n_assets)`` (or a precomputed covariance matrix), and
emits a weight vector. **It reads zero data outside its input** — so
polluting the future of any of its callers' inputs cannot change its
output, which is the no-lookahead guarantee that's exercised by the
property test in ``tests/test_panel_sizing.py``.

Future items #7 (β-/$-/σ-neutral) and #8 (long-short basket) add more
weight-construction primitives in this same module.
"""
from __future__ import annotations

import numpy as np


def equal_weights(n: int) -> np.ndarray:
    """1/N baseline. Returns ``np.ones(n) / n``."""
    if n <= 0:
        raise ValueError(f"equal_weights: n must be > 0, got {n}")
    return np.full(n, 1.0 / n, dtype=np.float64)


def _cov_from_returns(returns_window: np.ndarray) -> np.ndarray:
    """Sample covariance of a (n_bars, n_assets) return matrix.

    Drops bars containing any NaN before computing covariance so that
    callers can pass an incomplete window without separate cleanup.
    """
    if returns_window.ndim != 2:
        raise ValueError(
            f"returns_window must be 2-D (bars x assets); got shape "
            f"{returns_window.shape}"
        )
    # Mask any bar with a NaN across assets.
    mask = ~np.any(np.isnan(returns_window), axis=1)
    clean = returns_window[mask]
    if clean.shape[0] < 2:
        raise ValueError(
            f"returns_window has only {clean.shape[0]} clean rows; "
            f"need at least 2 for a sample covariance"
        )
    return np.cov(clean, rowvar=False, ddof=1)


def erc_weights(
    returns_window: np.ndarray | None = None,
    *,
    cov: np.ndarray | None = None,
    max_iter: int = 1_000,
    tol: float = 1e-10,
) -> np.ndarray:
    """Equal-risk-contribution weights.

    Pass either ``returns_window`` (2-D bars × assets) or a precomputed
    ``cov`` matrix; the function builds the covariance from the
    returns if only the former is given.

    The constrained convex problem:

        min_w   sum_i (w_i * (Σw)_i - V/n)^2
        s.t.    sum_i w_i = 1
                0 < w_i <= 1

    where V is portfolio variance ``w' Σ w``. Solved via SLSQP. For a
    well-conditioned covariance the optimum has all
    ``w_i * (Σw)_i == V / n`` and weights are unique.
    """
    if cov is None:
        if returns_window is None:
            raise ValueError("erc_weights: must pass returns_window or cov")
        cov = _cov_from_returns(returns_window)
    cov = np.asarray(cov, dtype=np.float64)
    if cov.ndim != 2 or cov.shape[0] != cov.shape[1]:
        raise ValueError(
            f"cov must be a square matrix; got shape {cov.shape}"
        )
    n = cov.shape[0]
    if n == 1:
        return np.array([1.0])

    # Lazy import; scipy is in the panel extras.
    from scipy.optimize import minimize

    # Rescale to unit trace so the optimisation problem is well-
    # conditioned regardless of the absolute return magnitude. SLSQP
    # convergence on raw daily/hourly return covariances (entries
    # ~1e-6) is noisy; scaling pulls those into O(1) where ftol = tol
    # actually means something.
    trace = float(np.trace(cov))
    if trace <= 0:
        raise ValueError(
            f"erc_weights: covariance has non-positive trace {trace}; "
            "cannot solve ERC"
        )
    cov_s = cov / trace

    def objective(w: np.ndarray) -> float:
        port_var = float(w @ cov_s @ w)
        rc = w * (cov_s @ w)
        target = port_var / n
        return float(np.sum((rc - target) ** 2))

    w0 = equal_weights(n)
    constraints = [{"type": "eq", "fun": lambda w: float(w.sum() - 1.0)}]
    bounds = [(1e-9, 1.0)] * n
    result = minimize(
        objective, w0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": max_iter, "ftol": tol},
    )
    if not result.success:
        # Fallback: iterative fixed-point. Numerically inferior but
        # always returns a valid weight vector.
        w = w0.copy()
        for _ in range(max_iter):
            sigma_w = cov @ w
            port_var = float(w @ cov @ w)
            target = port_var / n
            new_w_unnorm = target / np.where(np.abs(sigma_w) < 1e-12, 1e-12, sigma_w)
            new_w_unnorm = np.clip(new_w_unnorm, 1e-9, None)
            new_w = new_w_unnorm / new_w_unnorm.sum()
            if np.max(np.abs(new_w - w)) < tol:
                break
            w = new_w
        return w
    w = np.clip(result.x, 1e-9, None)
    return w / w.sum()


def risk_contributions(weights: np.ndarray, cov: np.ndarray) -> np.ndarray:
    """For a given weight vector and covariance matrix, return the
    per-asset risk contribution ``w_i * (Σw)_i``.

    Useful for verifying the ERC invariant in tests / verification
    logs without re-implementing the math inline.
    """
    return np.asarray(weights) * (np.asarray(cov) @ np.asarray(weights))


__all__ = [
    "equal_weights",
    "erc_weights",
    "risk_contributions",
    "_cov_from_returns",
]
