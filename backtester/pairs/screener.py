"""Pair / spread screener (item #9, Phase 3 — HIGH-RISK).

Given a panel and a screening window ending at ``t_idx``, evaluate
every pair `(a, b)` of assets with one of the supported methods and
return a ranked list of candidates with metadata.

The HIGH-RISK aspect: the screener at logical time ``t`` must use
**only** data at indices ``<= t``. Polluting indices ``> t`` cannot
change the returned pair list. The 10-window pollute test in
``tests/test_pairs_screener.py`` enforces this.

Supported methods (Phase 3 launch):

- ``engle_granger`` — co-integration ADF p-value on the OLS residual.
- ``distance_ssd`` — sum of squared deviations between standardised
  log-prices over the window. Lower = closer.
- (Future) ``johansen``, ``copula``, ``pca``, ``ml_cluster``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Literal, Optional

import numpy as np
import pandas as pd

from ..panel import PanelData


@dataclass
class ScreenedPair:
    asset_a: str
    asset_b: str
    method: str
    statistic: float            # method-specific (p-value, distance, etc.)
    extras: Dict[str, float] = field(default_factory=dict)


Method = Literal["engle_granger", "distance_ssd"]


def engle_granger(close_a: np.ndarray, close_b: np.ndarray) -> tuple[float, float]:
    """Run Engle-Granger cointegration test on ``log(a) ~ log(b)``.

    Returns ``(adf_pvalue, ols_beta)``. Caller responsible for
    slicing the input series to the window ending at ``<= t``.
    """
    from statsmodels.tsa.stattools import adfuller
    log_a = np.log(close_a)
    log_b = np.log(close_b)
    beta, alpha = np.polyfit(log_b, log_a, 1)
    resid = log_a - alpha - beta * log_b
    # Newer statsmodels uses 'n' for the no-constant ADF variant; the
    # residuals are already mean-zero from the regression intercept.
    adf = adfuller(resid, regression="n")
    return float(adf[1]), float(beta)


def distance_ssd(close_a: np.ndarray, close_b: np.ndarray) -> float:
    """Sum of squared deviations between z-scored log-prices. Lower
    is closer (better cointegration candidate)."""
    log_a = np.log(close_a)
    log_b = np.log(close_b)
    za = (log_a - log_a.mean()) / log_a.std()
    zb = (log_b - log_b.mean()) / log_b.std()
    return float(np.sum((za - zb) ** 2))


def screen_pairs(
    panel: PanelData,
    t_idx: int,
    *,
    method: Method = "engle_granger",
    lookback: int = 500,
    top_n: Optional[int] = None,
) -> List[ScreenedPair]:
    """Score every ordered pair ``(a, b)`` with ``a < b`` (by panel
    ordering) over the ``lookback`` bars ending at ``t_idx``. Returns
    a list sorted by score (ascending for engle_granger p-value,
    ascending for distance — lower is better in both cases).

    Reads only ``panel`` cells at row indices ``<= t_idx``. Polluting
    rows past ``t_idx`` cannot change the returned pair list.
    """
    n_assets = len(panel.assets)
    if t_idx < lookback - 1:
        return []  # not enough data yet
    start = t_idx - lookback + 1
    end = t_idx + 1
    close_window = {}
    for asset in panel.assets:
        ai = panel.assets.index(asset)
        close_window[asset] = panel.ds["close"].values[start:end, ai]

    # Validate method up-front so a typo raises loudly instead of
    # silently being stored as an error per-pair.
    if method not in ("engle_granger", "distance_ssd"):
        raise ValueError(f"unknown method {method!r}")

    results: List[ScreenedPair] = []
    for i in range(n_assets):
        for j in range(i + 1, n_assets):
            a = panel.assets[i]
            b = panel.assets[j]
            try:
                if method == "engle_granger":
                    p, beta = engle_granger(close_window[a], close_window[b])
                    results.append(ScreenedPair(
                        asset_a=a, asset_b=b, method=method, statistic=p,
                        extras={"beta": beta},
                    ))
                else:  # distance_ssd (only other validated method)
                    d = distance_ssd(close_window[a], close_window[b])
                    results.append(ScreenedPair(
                        asset_a=a, asset_b=b, method=method, statistic=d,
                    ))
            except Exception as e:  # noqa: BLE001
                results.append(ScreenedPair(
                    asset_a=a, asset_b=b, method=method,
                    statistic=float("inf"),
                    extras={"error": str(e)[:100]},  # type: ignore[dict-item]
                ))
    # Sort ascending (lower = better for both p-value and distance).
    results.sort(key=lambda r: r.statistic)
    if top_n is not None:
        return results[:top_n]
    return results


__all__ = ["ScreenedPair", "screen_pairs", "engle_granger", "distance_ssd"]
