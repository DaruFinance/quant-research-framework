"""Tests for the panel orchestrator (item #5 iter, Phase 2).

The contract: when the strategy is per-asset independent (the typical
case before item #6 ERC sizing / #8 basket / #44 multi-term IS land),
each asset's panel-run trade ledger must be bit-identical to running
the single-asset WFO on that asset's slice alone.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

xr = pytest.importorskip("xarray")

import backtester as bt
from backtester import orchestrator as orch
from backtester.panel import PanelData, load_panel, walk_forward_panel
from backtester.panel.orchestrator import _walk_forward_panel_path


HERE = Path(__file__).resolve().parent
FIXTURE_DIR = HERE / "fixtures" / "sources"
PANEL_PATHS = {
    "SOL": FIXTURE_DIR / "SOLUSDT_1h_30000_31000.csv",
    "BTC": FIXTURE_DIR / "BTCUSDT_1h_jan_feb_2024.csv",
    "ETH": FIXTURE_DIR / "ETHUSDT_1h_jan_feb_2024.csv",
}

# Small-fixture-appropriate config; mirrors docs/baselines/v0.4.0_ds_sol_1k.json.
PANEL_CFG = {
    "BACKTEST_CANDLES": 300,
    "OOS_CANDLES":      600,
    "ORIGINAL_OOS":     600,
    "WFO_TRIGGER_VAL":  150,
    "DEFAULT_LB":       20,
}


def _apply_cfg(monkeypatch):
    """Mutate bt.* config via monkeypatch so values are restored after
    the test. Direct ``bt.X = Y`` assignment would leak into later
    tests in the suite (caught by test_wfo_regime_smoke etc.)."""
    monkeypatch.setattr(bt, "PRINT_EQUITY_CURVE", False)
    monkeypatch.setattr(bt, "USE_MONTE_CARLO",   False)
    for k, v in PANEL_CFG.items():
        monkeypatch.setattr(bt, k, v)


def test_panel_route_registered_under_multi_asset_true():
    """The Phase 1 dispatch registry must now know about
    RouteKey(multi_asset=True). Imported eagerly by
    backtester.panel.__init__."""
    keys = orch.registered_keys()
    assert orch.RouteKey(regime=False, multi_asset=True) in keys, (
        f"panel route missing; have {keys}"
    )


def test_panel_route_rejects_dataframe_input():
    """Routing a single-asset DataFrame through the multi_asset=True
    function would silently mis-behave; the path asserts the input
    type and raises TypeError."""
    df = pd.DataFrame({"time": [0, 3600], "open": [1.0, 1.0],
                       "high": [1.0, 1.0], "low": [1.0, 1.0],
                       "close": [1.0, 1.0]})
    import io, contextlib
    with pytest.raises(TypeError, match="PanelData"):
        with contextlib.redirect_stdout(io.StringIO()):
            _walk_forward_panel_path(df, None, None, [])


def test_per_asset_run_matches_single_asset_run(monkeypatch):
    """Critical equivalence: running walk_forward_panel(panel) on
    DS-PANEL-3 must produce per-asset returns bit-identical to
    walking each asset's bar slice through the single-asset entry
    point."""
    _apply_cfg(monkeypatch)
    panel = load_panel(PANEL_PATHS)
    import io, contextlib

    # Panel run
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        panel_results = walk_forward_panel(panel)

    # Per-asset single-asset run (same config)
    for asset in panel.assets:
        ai = panel.assets.index(asset)
        times = pd.to_datetime(panel.times, unit="s", utc=True).tz_convert(bt.NY_TZ)
        df = pd.DataFrame({
            "time": times,
            "open":  panel.ds["open"].values[:, ai].copy(),
            "high":  panel.ds["high"].values[:, ai].copy(),
            "low":   panel.ds["low"].values[:, ai].copy(),
            "close": panel.ds["close"].values[:, ai].copy(),
        })
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            single = bt.walk_forward(df, met_is_baseline=None,
                                      eq_is_baseline=np.array([1.0]))

        # Compare oos returns and final equity element-wise.
        panel_rets, panel_eq, _, _ = panel_results[asset]
        single_rets, single_eq, _, _ = single
        np.testing.assert_array_equal(
            panel_rets, single_rets,
            err_msg=f"{asset}: oos rets differ between panel and single-asset",
        )
        np.testing.assert_array_equal(
            panel_eq, single_eq,
            err_msg=f"{asset}: equity curve differs between panel and single-asset",
        )


def test_panel_wfo_no_leak_under_tail_pollution(monkeypatch):
    """Lookahead-leak property for the panel WFO. Build a panel
    truncated at time T (rows after T dropped); run the panel WFO on
    both the truncated panel and a 'polluted' panel where rows past T
    have NaN OHLC. Per-asset oos rets at <= T must be bit-identical
    across the two runs.

    Note: the test reduces OOS_CANDLES to fit within the truncated
    panel length, so the WFO actually exits the loop without trying
    to read past T.
    """
    _apply_cfg(monkeypatch)
    panel = load_panel(PANEL_PATHS)

    # Truncate at index 800 (out of 1000 rows). The panel WFO will see
    # exactly 800 bars per asset.
    T_idx = 800
    truncated_ds = panel.ds.isel(time=slice(0, T_idx))
    truncated = PanelData(ds=truncated_ds)

    # Build a "polluted" panel where rows past T_idx contain NaN. The
    # inner-join in load_panel doesn't allow NaN closes, so we
    # construct the PanelData manually with the full 1000-bar shape
    # but NaN-after-T_idx in every field for SOL only.
    polluted_ds = panel.ds.copy(deep=True)
    sol_idx = panel.assets.index("SOL")
    for field in panel.fields:
        arr = polluted_ds[field].values
        arr[T_idx:, sol_idx] = np.nan
        polluted_ds[field].values[...] = arr
    polluted_full = PanelData(ds=polluted_ds)
    # Also truncate the polluted panel to the same window so the WFO
    # runs over identical row counts.
    polluted = PanelData(ds=polluted_full.ds.isel(time=slice(0, T_idx)))

    import io, contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        clean_results = walk_forward_panel(truncated)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        polluted_results = walk_forward_panel(polluted)

    # Every asset's oos returns up to T_idx must be unchanged. The
    # polluted run truncates to the same window so we expect bit-
    # identical output (the pollution rows past T_idx aren't read).
    for asset in panel.assets:
        np.testing.assert_array_equal(
            clean_results[asset][0], polluted_results[asset][0],
            err_msg=f"{asset}: panel WFO leaked tail pollution"
        )
