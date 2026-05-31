"""Portfolio-level constraints (item #45, Phase 2 — closes Phase 2).

Apply hard caps to a weight vector after sizing and neutralization
(items #6, #7) have produced their proposed weights. Two caps land
today:

- ``single_asset_weight_max``: no asset's weight exceeds this in
  absolute value. Excess mass redistributes proportionally to the
  remaining (unbound) assets, preserving the original gross
  notional.
- ``gross_lev_max``: total gross notional (`sum |w_i|`) does not
  exceed this. Excess scales every weight by a common factor,
  preserving the relative composition.

The two caps compose: ``apply_constraints`` enforces the single-
asset cap first (which is non-linear because of the redistribution
loop) then rescales for the gross cap. Both caps are **idempotent**:
applying twice gives the same result as once.

Pure pointwise function. No time-dependent inputs, no series. The
property tests verify idempotence and convexity.
"""
from __future__ import annotations

from typing import Optional

import numpy as np


def apply_constraints(
    weights: np.ndarray,
    *,
    single_asset_max: Optional[float] = None,
    gross_lev_max: Optional[float] = None,
    max_iter: int = 100,
    tol: float = 1e-12,
) -> np.ndarray:
    """Apply optional caps to a proposed weight vector.

    Parameters
    ----------
    weights : 1-D real array, real-valued (positive = long, negative
        = short).
    single_asset_max : optional float in (0, 1]. Cap on `|w_i|` per
        asset.
    gross_lev_max : optional positive float. Cap on `sum(|w_i|)`.
    max_iter : redistribution-loop iteration cap (defensive).
    tol : convergence tolerance for the redistribution.

    Returns
    -------
    Constrained weight vector. Same sign per asset as input.
    """
    w = np.asarray(weights, dtype=np.float64).copy()

    if single_asset_max is not None:
        if not (0 < single_asset_max <= 1.0):
            raise ValueError(
                f"single_asset_max must lie in (0, 1]; got "
                f"{single_asset_max}"
            )
        w = _cap_single_asset(w, single_asset_max, max_iter, tol)

    if gross_lev_max is not None:
        if gross_lev_max <= 0:
            raise ValueError(
                f"gross_lev_max must be > 0; got {gross_lev_max}"
            )
        gross = float(np.abs(w).sum())
        if gross > gross_lev_max:
            w = w * (gross_lev_max / gross)

    return w


def _cap_single_asset(
    w: np.ndarray, cap: float, max_iter: int, tol: float,
) -> np.ndarray:
    """Iteratively cap |w_i| at ``cap`` and redistribute the trimmed
    mass to legs **strictly below** ``cap``. Using `~over` (i.e.
    including legs already at cap) creates an oscillation loop, so we
    only feed the residual to assets with room to grow. Legs that hit
    cap are frozen; the redistribution converges in a finite number of
    iterations (≤ n_assets).

    If everyone hits cap before the residual is absorbed, the
    remaining excess is dropped — the gross notional may shrink
    accordingly (mathematically unavoidable: with cap c and n legs,
    the max representable gross is n*c).
    """
    if w.size == 0:
        return w
    signs = np.sign(w)
    abs_w = np.abs(w).copy()
    for _ in range(max_iter):
        over = abs_w > cap + tol
        if not over.any():
            break
        excess = float(np.sum(abs_w[over] - cap))
        abs_w[over] = cap
        # Strict under: legs with headroom for more weight. Legs at
        # cap or over (just capped) are frozen.
        under = abs_w < cap - tol
        if not under.any():
            # No headroom; drop the residual.
            break
        under_sum = float(abs_w[under].sum())
        if under_sum == 0:
            # All under-legs are zero-weighted; distribute uniformly.
            abs_w[under] = excess / int(under.sum())
        else:
            abs_w[under] = abs_w[under] + excess * (abs_w[under] / under_sum)
    # Defensive final clip: any tiny over-shoot from float arithmetic.
    abs_w = np.minimum(abs_w, cap)
    return signs * abs_w


__all__ = ["apply_constraints"]
