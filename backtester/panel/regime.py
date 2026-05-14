"""Cross-asset regime detection (item #4, Phase 2; HIGH-RISK).

The single-asset ``detect_regimes(df)`` lives in
``backtester/__init__.py`` and uses EMA-200 + 8-bar consistency to
emit one of {Uptrend, Downtrend, Ranging} per bar. This module
generalises that contract to the panel:

    detect_regimes_panel(panel: PanelData) -> Dict[asset, pd.Series]

Every asset gets its own label series of length ``len(panel)``. The
contract: an asset's label at time ``t`` may depend on **any asset's
data at times <= t**, never on data at times > t. Two implementations
ship by default; either may be plugged into the orchestrator via
``bt.panel.detect_regimes_panel = my_fn``.

- ``detect_regimes_panel_per_asset`` (the registered default):
  each asset's regime is computed from its own close series only.
  Trivially leak-free across assets — no cross-asset coupling.
- ``detect_regimes_panel_market(market_asset='BTC')``: every asset
  inherits ``market_asset``'s regime label at each bar. The
  classic "BTC dominance" regime. Leak-free because the market
  asset's own regime at ``t`` reads only its own past.

Both functions are registered with the lookahead-leak harness from
item #14 so the cross-asset pollute test catches any future variant
that accidentally peeks at another asset's future close.
"""
from __future__ import annotations

from typing import Callable, Dict, Optional

import numpy as np
import pandas as pd

from . import PanelData
from ..invariants import registers_invariant


def _per_asset_ema_regime(close: np.ndarray, length: int = 8) -> pd.Series:
    """Replicate ``backtester.detect_regimes`` on a 1-D close series.

    Uses EMA-200 and an N-bar consistency window. Returns string
    labels (Uptrend / Downtrend / Ranging) so panel routes can keep
    using the same downstream regime-handling code paths.
    """
    s = pd.Series(close)
    ema200 = s.ewm(span=200, adjust=False).mean()
    above = (s > ema200).rolling(length).sum()
    below = (s < ema200).rolling(length).sum()
    labels = np.full(len(s), "Ranging", dtype=object)
    labels[above >= length] = "Uptrend"
    labels[below >= length] = "Downtrend"
    return pd.Series(labels, name="regime")


@registers_invariant(name="panel_regime_per_asset", data_kind="panel")
def detect_regimes_panel_per_asset(panel: PanelData) -> Dict[str, pd.Series]:
    """Default panel regime detector: each asset gets its own
    independent EMA-200 regime. Bit-equivalent to running the
    single-asset detector on each asset separately."""
    out: Dict[str, pd.Series] = {}
    for ai, asset in enumerate(panel.assets):
        close = panel.ds["close"].values[:, ai]
        out[asset] = _per_asset_ema_regime(close)
    return out


def detect_regimes_panel_market(
    market_asset: str = "BTC",
) -> Callable[[PanelData], Dict[str, pd.Series]]:
    """Build a cross-asset detector that broadcasts ``market_asset``'s
    EMA-200 regime to every asset in the panel. Every asset's label at
    bar ``t`` is exactly ``market_asset``'s label at ``t``.

    Leak-free across assets because the market asset's regime at ``t``
    only consults its own close at ``<= t``; polluting any other
    asset's data has zero effect.
    """
    def detector(panel: PanelData) -> Dict[str, pd.Series]:
        if market_asset not in panel.assets:
            raise ValueError(
                f"detect_regimes_panel_market: {market_asset!r} not in panel "
                f"assets {panel.assets}"
            )
        mi = panel.assets.index(market_asset)
        market_close = panel.ds["close"].values[:, mi]
        market_labels = _per_asset_ema_regime(market_close)
        return {asset: market_labels.copy() for asset in panel.assets}

    detector.__name__ = f"detect_regimes_panel_market_{market_asset}"
    return detector


# The single-callable plugin point. Users override via
# ``backtester.panel.regime.detect_regimes_panel = my_fn`` or by
# overriding the per-asset / market variants and rewiring.
detect_regimes_panel = detect_regimes_panel_per_asset


__all__ = [
    "detect_regimes_panel",
    "detect_regimes_panel_per_asset",
    "detect_regimes_panel_market",
]
