"""Long-short basket primitive (item #8, Phase 2).

A panel-level strategy that:
1. Evaluates an alpha function at the rebalance bar (function gets
   ``(panel, t_idx)`` and returns a length-N alpha vector using
   only data at indices `<= t_idx`).
2. Ranks assets by alpha.
3. Selects the top ``n_long`` as longs and the bottom ``n_short`` as
   shorts; remaining assets get weight 0.
4. Builds raw weights (+1 for each long, -1 for each short).
5. Applies one of the item #7 neutralizations (``dollar`` / ``beta``
   / ``sigma``).
6. Returns per-asset position weights as a dict ``{asset: weight}``.

Lookahead-freeness: the basket reads exclusively values at indices
``<= t_idx``. The bundled momentum alpha proves the pattern.

Future panel strategies (regime-conditional long-short, sector-
neutral baskets, factor-based composites) will add classes alongside
``LongShortBasket`` in this subpackage.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional, Sequence, Tuple

import numpy as np

from .. import PanelData
from ..neutralize import (
    Mode as NeutralizeMode,
    estimate_betas,
    estimate_vols,
    neutralize,
)


AlphaFn = Callable[[PanelData, int], np.ndarray]
"""Alpha function signature. Returns a length-N alpha vector. Must
read only ``panel`` cells at row indices ``<= t_idx``."""


def momentum_alpha(lookback: int = 20) -> AlphaFn:
    """Standard N-bar momentum: ``close[t] / close[t - lookback] - 1``.

    Returns NaN per asset when ``t_idx < lookback`` (the basket
    treats NaN alphas by excluding those assets from ranking).
    """
    def _alpha(panel: PanelData, t_idx: int) -> np.ndarray:
        close = panel.ds["close"].values  # (T, A)
        if t_idx < lookback:
            return np.full(close.shape[1], np.nan)
        return (close[t_idx] / close[t_idx - lookback]) - 1.0
    _alpha.__name__ = f"momentum_alpha_{lookback}"
    return _alpha


@dataclass
class LongShortBasket:
    """Long-short basket builder.

    Parameters
    ----------
    alpha_fn
        Returns a length-N alpha vector when called as
        ``alpha_fn(panel, t_idx)``. Must read only ``<= t_idx``.
    neutralize_mode
        One of ``"dollar"``, ``"beta"``, ``"sigma"``. Drives the
        post-ranking weight construction (item #7).
    n_long, n_short
        Number of top / bottom alpha-ranked assets to enter long /
        short.
    market_asset
        Required for ``neutralize_mode="beta"``; the asset whose β
        the basket targets to zero.
    returns_lookback
        Window length used to estimate per-asset betas / vols when
        the neutralization needs them. Window ends at ``t_idx`` so
        it's strictly ``<= t_idx``.
    """
    alpha_fn: AlphaFn
    neutralize_mode: NeutralizeMode = "dollar"
    n_long: int = 1
    n_short: int = 1
    market_asset: Optional[str] = None
    returns_lookback: int = 60

    def __post_init__(self) -> None:
        if self.n_long < 0 or self.n_short < 0:
            raise ValueError(
                f"n_long and n_short must be non-negative; got "
                f"n_long={self.n_long} n_short={self.n_short}"
            )
        if self.neutralize_mode == "beta" and self.market_asset is None:
            raise ValueError(
                "neutralize_mode='beta' requires market_asset"
            )

    def positions(self, panel: PanelData, t_idx: int) -> Dict[str, float]:
        """Return per-asset weights as ``{asset: weight}``.

        Reads only ``panel`` cells at row indices ``<= t_idx`` —
        consult the alpha at ``t_idx`` and the returns window
        ``[t_idx - returns_lookback, t_idx)``.
        """
        n_assets = len(panel.assets)
        if self.n_long + self.n_short > n_assets:
            raise ValueError(
                f"n_long+n_short={self.n_long + self.n_short} exceeds "
                f"n_assets={n_assets}"
            )

        alpha = np.asarray(self.alpha_fn(panel, t_idx), dtype=np.float64)
        if alpha.shape != (n_assets,):
            raise ValueError(
                f"alpha_fn returned shape {alpha.shape}; expected "
                f"({n_assets},)"
            )
        if np.all(np.isnan(alpha)):
            # Pre-warmup: no positions.
            return {a: 0.0 for a in panel.assets}

        # Argsort ignoring NaNs by mapping them to -inf for shorts /
        # +inf for longs — they fall to the bottom of the ranking
        # automatically.
        order_asc = np.argsort(np.where(np.isnan(alpha), np.inf, alpha))
        order_desc = order_asc[::-1]
        long_ids = list(order_desc[: self.n_long])
        short_ids = list(order_asc[: self.n_short])
        # Remove overlap (shouldn't happen unless n_long+n_short > n_assets,
        # but defensive).
        short_ids = [i for i in short_ids if i not in long_ids]

        raw = np.zeros(n_assets, dtype=np.float64)
        for i in long_ids:
            raw[i] = 1.0
        for i in short_ids:
            raw[i] = -1.0

        # Neutralization (only on the selected legs).
        # For modes that need returns: build returns window ending at t_idx.
        if self.neutralize_mode == "dollar":
            return self._weights_to_dict(panel, neutralize(raw, "dollar"))
        elif self.neutralize_mode == "beta":
            mi = panel.assets.index(self.market_asset)  # type: ignore[arg-type]
            window = self._returns_window(panel, t_idx)
            betas = estimate_betas(window, market_idx=mi)
            w = neutralize(raw, "beta", betas=betas, market_idx=mi)
            return self._weights_to_dict(panel, w)
        elif self.neutralize_mode == "sigma":
            # Sigma requires non-zero weights for every entry; zero
            # out unselected legs after rescaling the selected ones.
            window = self._returns_window(panel, t_idx)
            vols = estimate_vols(window)
            selected_mask = raw != 0
            if not selected_mask.any():
                return self._weights_to_dict(panel, raw)
            raw_sel = raw[selected_mask]
            vols_sel = vols[selected_mask]
            w_sel = neutralize(raw_sel, "sigma", vols=vols_sel)
            w_full = np.zeros(n_assets, dtype=np.float64)
            w_full[selected_mask] = w_sel
            return self._weights_to_dict(panel, w_full)
        raise ValueError(
            f"unknown neutralize_mode {self.neutralize_mode!r}"
        )

    def _returns_window(self, panel: PanelData, t_idx: int) -> np.ndarray:
        """Log-returns window strictly ending at ``t_idx`` (rows
        ``[t_idx - returns_lookback, t_idx)``)."""
        if t_idx < self.returns_lookback + 1:
            raise ValueError(
                f"t_idx={t_idx} insufficient for "
                f"returns_lookback={self.returns_lookback}"
            )
        close = panel.ds["close"].values  # (T, A)
        slc = close[t_idx - self.returns_lookback - 1 : t_idx]
        rets = np.diff(np.log(slc), axis=0)
        return rets

    @staticmethod
    def _weights_to_dict(panel: PanelData, weights: np.ndarray) -> Dict[str, float]:
        return {a: float(weights[i]) for i, a in enumerate(panel.assets)}
