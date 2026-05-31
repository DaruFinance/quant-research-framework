"""Pre-screening eligibility filters (item #13, Phase 3).

Quick acceptance tests applied to a candidate spread before any
trading decision:

- ``half_life_ou``: OU half-life estimate. A spread with half-life
  outside ``[h_min, h_max]`` is rejected — too fast and slippage
  dominates; too slow and inventory cost dominates.
- ``is_eligible_pair``: dispatcher applying the full filter stack
  (ADF p-value, half-life, optional volume floor).

The HIGH-RISK item #11 (re-estimation cadence) uses these as
acceptance gates during its β refits.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class EligibilityCriteria:
    """Bundle of filter parameters. ``None`` skips that particular
    gate (so callers can mix and match)."""
    p_max: Optional[float] = 0.05
    half_life_range: Optional[tuple[float, float]] = (1.0, 1000.0)
    min_window: int = 60


def half_life_ou(spread: np.ndarray) -> float:
    """OU half-life estimator. Fits ``ds_t = -lambda * s_{t-1} + e`` and
    returns ``ln(2) / lambda``. Caller must pass a NaN-free spread.

    Returns ``+inf`` if the regression slope is non-negative (i.e. no
    mean reversion detected) — half-life is undefined.
    """
    s = np.asarray(spread, dtype=np.float64)
    s = s[~np.isnan(s)]
    if len(s) < 3:
        return float("inf")
    ds = np.diff(s)
    s_lag = s[:-1]
    # OLS: ds = a + b * s_lag + e. b should be negative for MR.
    slope, _ = np.polyfit(s_lag, ds, 1)
    if slope >= 0:
        return float("inf")
    return float(np.log(2.0) / -slope)


def is_eligible_pair(
    spread: np.ndarray,
    *,
    p_value: Optional[float] = None,
    criteria: Optional[EligibilityCriteria] = None,
) -> tuple[bool, str]:
    """Apply the criteria stack to ``spread``. Returns ``(ok, reason)``
    so the caller can log why a pair was rejected.

    ``p_value`` is the ADF (or whichever cointegration test) p-value;
    the function does not run the test itself — that lives in the
    screener (#9).
    """
    crit = criteria or EligibilityCriteria()
    s = np.asarray(spread, dtype=np.float64)
    clean = s[~np.isnan(s)]
    if len(clean) < crit.min_window:
        return False, f"insufficient data ({len(clean)} < {crit.min_window})"

    if crit.p_max is not None and p_value is not None:
        if p_value > crit.p_max:
            return False, f"p_value={p_value:.4f} > p_max={crit.p_max}"

    if crit.half_life_range is not None:
        hl = half_life_ou(clean)
        h_lo, h_hi = crit.half_life_range
        if hl < h_lo or hl > h_hi:
            return False, f"half_life={hl:.2f} outside [{h_lo}, {h_hi}]"

    return True, "ok"


__all__ = [
    "EligibilityCriteria", "half_life_ou", "is_eligible_pair",
]
