"""Panel walk-forward orchestrator (item #5 iter, Phase 2).

Adds the ``multi_asset=True`` routes to the central dispatch table
introduced by Phase 1 #5. The panel WFO iterates over the assets in a
``PanelData`` and runs the existing single-asset
``_walk_forward_*_path`` on each asset's slice in turn, returning a
dict of per-asset results.

The contract is: when the strategy is per-asset independent (the
typical case for Tree-1 trend / momentum panels), each asset's
trade ledger is bit-identical to running the single-asset WFO on that
asset alone. Item #6 (ERC sizing), #7 (β-/$-/σ-neutral), #8
(long-short basket), and #44 (multi-term objective) layer
*portfolio-level* state on top of the per-asset ledgers; the per-asset
core stays the same.

The panel WFO uses the same global window grid (cur_start advances by
``WFO_TRIGGER_VAL`` candles per iteration) as the single-asset path —
multi-asset iteration does NOT shift window boundaries per asset.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, Optional

import numpy as np
import pandas as pd

from . import PanelData
from .. import orchestrator as _orch


def _asset_to_dataframe(panel: PanelData, asset: str) -> pd.DataFrame:
    """Convert one asset's slice of a PanelData into the DataFrame
    shape that the single-asset WFO consumes.

    The single-asset path expects a tz-aware datetime ``time`` column
    (set by ``bt.load_ohlc``: UNIX seconds → UTC → America/New_York).
    The panel loader stores raw UNIX seconds; we redo the conversion
    here so each per-asset DataFrame is indistinguishable from a
    ``load_ohlc`` output.
    """
    import backtester as bt  # late import; avoids panel <-> __init__ cycle

    ai = panel.assets.index(asset)
    times_dt = (
        pd.to_datetime(panel.times, unit="s", utc=True).tz_convert(bt.NY_TZ)
    )
    cols: Dict[str, Any] = {"time": times_dt}
    for field in panel.fields:
        cols[field] = panel.ds[field].values[:, ai].copy()
    return pd.DataFrame(cols)


def walk_forward_panel(
    panel: PanelData,
    *,
    single_asset_runner: Optional[Callable[[pd.DataFrame, str], Any]] = None,
) -> Dict[str, Any]:
    """Run the panel WFO: per-asset single-asset WFO, results aggregated.

    Parameters
    ----------
    panel : PanelData
        Loaded multi-asset panel.
    single_asset_runner : optional callable (df, asset_symbol) -> result
        Plug point for unit tests. Defaults to a thin wrapper that
        calls ``bt.walk_forward(df, met_is_baseline=None,
        eq_is_baseline=None)`` so the per-asset path mirrors the
        single-asset entry point exactly.

    Returns
    -------
    dict mapping asset symbol to the runner's return value (typically
    a 4-tuple of (oos_rets, eq_wfo, rb_eq_curves, split_wfo_is)).
    """
    if single_asset_runner is None:
        # Lazy import: avoids panel <-> __init__ cycle at module load.
        import backtester as bt

        def single_asset_runner(df: pd.DataFrame, asset: str) -> Any:  # noqa: ARG001
            return bt.walk_forward(df, met_is_baseline=None,
                                    eq_is_baseline=np.array([1.0]))

    out: Dict[str, Any] = {}
    for asset in panel.assets:
        df = _asset_to_dataframe(panel, asset)
        out[asset] = single_asset_runner(df, asset)
    return out


def _walk_forward_panel_path(
    df_or_panel: Any, met_is_baseline: Any, eq_is_baseline: Any, rb_scenarios: Any,
) -> Any:
    """Dispatch-table entry for ``RouteKey(multi_asset=True)``.

    The Phase 1 single-asset routes take a DataFrame; this panel route
    expects a ``PanelData`` instead. We call out to ``walk_forward_panel``
    and return its per-asset dict; downstream basket primitives (item
    #8) consume the dict.

    ``rb_scenarios`` is the same robustness-scenario list the
    single-asset paths receive; ``walk_forward_panel`` doesn't use it
    today (each per-asset call rebuilds it from globals), but the
    parameter is in the signature for future basket-aware variants.
    """
    if not isinstance(df_or_panel, PanelData):
        raise TypeError(
            "panel route expects a PanelData; got "
            f"{type(df_or_panel).__name__}. Use the single-asset routes "
            "for DataFrame inputs."
        )
    _ = met_is_baseline, eq_is_baseline, rb_scenarios  # forward-looking
    return walk_forward_panel(df_or_panel)


# Register the Phase 2 panel route. Phase 1's RouteKey(multi_asset=False)
# entries remain untouched; this adds the multi_asset=True entries.
# Phase 2 only registers the (multi_asset=True, regime=False) variant;
# panel + regime composition lands when #4's panel regime detector is
# wired into the WFO at item #44.
_orch.register(
    _orch.RouteKey(regime=False, multi_asset=True),
    _walk_forward_panel_path,
)


__all__ = ["walk_forward_panel", "_walk_forward_panel_path"]
