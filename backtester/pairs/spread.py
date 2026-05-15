"""Spread-definition primitives (item #10, Phase 3).

Given a panel (or a pair of return series) and a target endpoint
``t_idx``, each primitive computes the spread series up to ``t_idx``
using only data at indices ``<= t_idx``. Five primitives ship:

- ``log_ratio`` — ``log(asset_a / asset_b)``. Constant β=1 implicit.
- ``ols_resid`` — Rolling OLS-residual spread. β is the slope of
  ``log(a) ~ log(b)`` over the window ending at ``t_idx``.
- ``kalman_beta_spread`` — Kalman-filter dynamic β; smoother and
  more responsive than a fixed window.
- ``pca_resid`` — First-PC residual on an N-asset panel.
- ``ml_resid`` — Generic residual where the user supplies a fitted
  predictor; e.g. RandomForestRegressor with sklearn.

All primitives are leak-free by construction. The HIGH-RISK
counterpart is item #11's *cadence engine*, which re-fits β at chosen
times — its leak test is the 50-T pollute battery.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Sequence

import numpy as np
import pandas as pd

from ..panel import PanelData


@dataclass(frozen=True)
class SpreadResult:
    """Output of every spread-definition primitive.

    ``spread`` is a 1-D array of length ``t_idx + 1``; entry ``i`` is
    the spread value at bar ``i`` computed from data at ``<= i``.

    ``beta`` is the regression coefficient as of ``t_idx`` (a scalar
    for static-β methods like OLS / log-ratio, the full Kalman series
    for dynamic-β methods). NaN where the window has insufficient
    data for fitting.
    """
    spread: np.ndarray
    beta: float | np.ndarray
    method: str
    asset_a: str
    asset_b: str


def _close_series(panel: PanelData, asset: str) -> np.ndarray:
    ai = panel.assets.index(asset)
    return panel.ds["close"].values[:, ai]


def log_ratio(
    panel: PanelData,
    asset_a: str,
    asset_b: str,
    t_idx: int,
) -> SpreadResult:
    """``spread[i] = log(close_a[i] / close_b[i])`` for ``i in [0, t_idx]``.

    Trivially leak-free: each bar's spread reads only that bar's
    closes from the two assets.
    """
    a = _close_series(panel, asset_a)[: t_idx + 1]
    b = _close_series(panel, asset_b)[: t_idx + 1]
    spread = np.log(a) - np.log(b)
    return SpreadResult(spread=spread, beta=1.0, method="log_ratio",
                         asset_a=asset_a, asset_b=asset_b)


def ols_resid(
    panel: PanelData,
    asset_a: str,
    asset_b: str,
    t_idx: int,
    lookback: int = 60,
) -> SpreadResult:
    """Rolling OLS residual: regress ``log(close_a) ~ alpha + beta *
    log(close_b)`` on the ``lookback``-bar window ending at ``t_idx``.

    For bars before warmup completes (``i < lookback``) the spread is
    NaN. The β reported is the slope estimated on the ``[t_idx - lookback,
    t_idx]`` window (so callers can audit it). Leak-free: each bar's
    spread is built from a regression that ends at-or-before that bar.
    """
    a = _close_series(panel, asset_a)[: t_idx + 1]
    b = _close_series(panel, asset_b)[: t_idx + 1]
    n = len(a)
    spread = np.full(n, np.nan)
    if n < lookback:
        return SpreadResult(spread=spread, beta=float("nan"), method="ols_resid",
                             asset_a=asset_a, asset_b=asset_b)

    log_a = np.log(a)
    log_b = np.log(b)
    # For each bar i in [lookback-1, n), fit OLS on window [i-lookback+1, i+1).
    for i in range(lookback - 1, n):
        x = log_b[i - lookback + 1 : i + 1]
        y = log_a[i - lookback + 1 : i + 1]
        beta, alpha = np.polyfit(x, y, 1)
        spread[i] = log_a[i] - alpha - beta * log_b[i]

    # Report the β fitted on the window ending at t_idx.
    x_final = log_b[n - lookback : n]
    y_final = log_a[n - lookback : n]
    beta_final, _ = np.polyfit(x_final, y_final, 1)
    return SpreadResult(spread=spread, beta=float(beta_final),
                         method="ols_resid",
                         asset_a=asset_a, asset_b=asset_b)


def kalman_beta_spread(
    panel: PanelData,
    asset_a: str,
    asset_b: str,
    t_idx: int,
    delta: float = 1e-4,
    observation_var: float = 1e-3,
) -> SpreadResult:
    """Dynamic β via a Kalman filter. The state is ``(alpha, beta)``
    of ``log_a = alpha + beta * log_b + noise``. Returns a spread
    series and the full β trajectory.

    Standard 2-D random-walk-state Kalman update. ``delta`` controls
    the state covariance scale (small = slow β drift, large =
    responsive); ``observation_var`` is the measurement noise. Each
    bar's update uses only that bar's observation, so the algorithm
    is leak-free by construction.
    """
    a = _close_series(panel, asset_a)[: t_idx + 1]
    b = _close_series(panel, asset_b)[: t_idx + 1]
    n = len(a)
    log_a = np.log(a)
    log_b = np.log(b)

    # State: [alpha, beta]; covariance P; observation H_t = [1, log_b[t]].
    state = np.zeros(2)
    P = np.eye(2)
    Q = delta * np.eye(2)
    spread = np.full(n, np.nan)
    beta_traj = np.full(n, np.nan)
    for i in range(n):
        H = np.array([1.0, log_b[i]])
        # Predict
        P = P + Q
        # Innovation
        y_hat = H @ state
        v = log_a[i] - y_hat
        S = float(H @ P @ H + observation_var)
        K = P @ H / S
        state = state + K * v
        P = P - np.outer(K, H) @ P
        spread[i] = log_a[i] - state[0] - state[1] * log_b[i]
        beta_traj[i] = state[1]
    return SpreadResult(spread=spread, beta=beta_traj,
                         method="kalman_beta",
                         asset_a=asset_a, asset_b=asset_b)


def pca_resid(
    panel: PanelData,
    asset_a: str,
    t_idx: int,
    other_assets: Optional[Sequence[str]] = None,
    lookback: int = 60,
) -> SpreadResult:
    """First-principal-component residual on an N-asset panel.

    Builds the covariance of log-returns over the window ending at
    ``t_idx``, extracts the top eigenvector, and reports the residual
    of ``asset_a``'s log-return after projection onto that PC. Other
    assets default to ``panel.assets`` minus ``asset_a``.
    """
    if other_assets is None:
        others = [a for a in panel.assets if a != asset_a]
    else:
        others = list(other_assets)
    all_assets = [asset_a] + others
    close = np.stack([_close_series(panel, a)[: t_idx + 1] for a in all_assets],
                     axis=1)
    log_close = np.log(close)
    n = log_close.shape[0]
    spread = np.full(n, np.nan)
    if n < lookback + 1:
        return SpreadResult(spread=spread, beta=float("nan"), method="pca_resid",
                             asset_a=asset_a, asset_b=",".join(others))
    # Compute log-returns once.
    log_rets = np.diff(log_close, axis=0)
    for i in range(lookback, n):
        window = log_rets[i - lookback : i]  # shape (lookback, n_assets)
        # Demean.
        window = window - window.mean(axis=0)
        # First PC via covariance eigendecomposition.
        cov = window.T @ window / (window.shape[0] - 1)
        eigvals, eigvecs = np.linalg.eigh(cov)
        pc1 = eigvecs[:, -1]  # largest eigenvalue (eigh sorts ascending)
        # Project today's log-return onto PC1 to get the projection
        # length, then the residual is log_ret_a - projection_component
        # corresponding to asset_a.
        today_ret = log_rets[i - 1]  # log-return into bar i, computed from <= i closes
        projection = (pc1 @ today_ret) * pc1
        residual = today_ret - projection
        spread[i] = residual[0]  # asset_a is index 0 by construction
    return SpreadResult(spread=spread, beta=float("nan"),
                         method="pca_resid",
                         asset_a=asset_a, asset_b=",".join(others))


def ml_resid(
    panel: PanelData,
    asset_a: str,
    asset_b: str,
    t_idx: int,
    lookback: int = 60,
    predictor_factory: Optional[Callable] = None,
) -> SpreadResult:
    """Generic ML-residual primitive. ``predictor_factory`` returns a
    fresh predictor object exposing ``fit(X, y)`` and ``predict(X)``.
    Defaults to scikit-learn ``LinearRegression`` so the function has
    a sane out-of-the-box behaviour.

    Each bar refits the predictor on the previous ``lookback`` bars
    (``log_a ~ predictor(log_b)``) and computes today's residual.
    Caller's responsibility to ensure the predictor itself doesn't
    leak (e.g. stateful kernels accumulating across bars).
    """
    a = _close_series(panel, asset_a)[: t_idx + 1]
    b = _close_series(panel, asset_b)[: t_idx + 1]
    n = len(a)
    log_a = np.log(a)
    log_b = np.log(b)
    spread = np.full(n, np.nan)
    if n < lookback + 1:
        return SpreadResult(spread=spread, beta=float("nan"), method="ml_resid",
                             asset_a=asset_a, asset_b=asset_b)
    if predictor_factory is None:
        try:
            from sklearn.linear_model import LinearRegression
        except ImportError as e:
            raise ImportError(
                "ml_resid default predictor needs scikit-learn. Pass a "
                "custom predictor_factory or install with "
                "`pip install quant-research-framework[examples]`."
            ) from e
        predictor_factory = lambda: LinearRegression()  # noqa: E731

    for i in range(lookback, n):
        x = log_b[i - lookback : i].reshape(-1, 1)
        y = log_a[i - lookback : i]
        model = predictor_factory()
        model.fit(x, y)
        pred_today = float(model.predict([[log_b[i]]])[0])
        spread[i] = float(log_a[i] - pred_today)
    return SpreadResult(spread=spread, beta=float("nan"), method="ml_resid",
                         asset_a=asset_a, asset_b=asset_b)


__all__ = [
    "SpreadResult",
    "log_ratio", "ols_resid", "kalman_beta_spread",
    "pca_resid", "ml_resid",
]
