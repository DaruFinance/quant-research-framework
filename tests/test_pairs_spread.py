"""Tests for spread-definition primitives (item #10, Phase 3)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

xr = pytest.importorskip("xarray")
sm = pytest.importorskip("statsmodels")

from backtester.panel import PanelData, load_panel
from backtester.pairs import (
    SpreadResult,
    log_ratio,
    ols_resid,
    kalman_beta_spread,
    pca_resid,
)


HERE = Path(__file__).resolve().parent
FIXTURE_DIR = HERE / "fixtures" / "sources"
PAIR_PATHS = {
    "BTC": FIXTURE_DIR / "BTCUSDT_1h_pair_q3_2023.csv",
    "ETH": FIXTURE_DIR / "ETHUSDT_1h_pair_q3_2023.csv",
}


@pytest.fixture(scope="module")
def panel():
    return load_panel(PAIR_PATHS)


def test_log_ratio_basic(panel):
    res = log_ratio(panel, "BTC", "ETH", t_idx=499)
    assert isinstance(res, SpreadResult)
    assert res.method == "log_ratio"
    assert res.beta == 1.0
    assert res.spread.shape == (500,)
    # log(BTC/ETH) at i=0 equals log(BTC[0]) - log(ETH[0]).
    btc = panel.ds["close"].values[:, panel.assets.index("BTC")][0]
    eth = panel.ds["close"].values[:, panel.assets.index("ETH")][0]
    assert abs(res.spread[0] - (np.log(btc) - np.log(eth))) < 1e-12


def test_log_ratio_no_lookahead(panel):
    # Polluting bars > t_idx must not change res.spread[<= t_idx].
    res_clean = log_ratio(panel, "BTC", "ETH", t_idx=500)
    polluted_ds = panel.ds.copy(deep=True)
    rng = np.random.default_rng(0xCA1F)
    for field in panel.fields:
        arr = polluted_ds[field].values
        arr[501:] = rng.normal(100, 10, size=arr[501:].shape)
        polluted_ds[field].values[...] = arr
    polluted_panel = PanelData(ds=polluted_ds)
    res_polluted = log_ratio(polluted_panel, "BTC", "ETH", t_idx=500)
    np.testing.assert_array_equal(res_clean.spread, res_polluted.spread)


def test_ols_resid_warmup_returns_nan(panel):
    res = ols_resid(panel, "BTC", "ETH", t_idx=20, lookback=60)
    assert np.all(np.isnan(res.spread))
    assert np.isnan(res.beta)


def test_ols_resid_emits_spread_after_warmup(panel):
    res = ols_resid(panel, "BTC", "ETH", t_idx=499, lookback=60)
    # First 59 entries NaN (need lookback), rest finite.
    assert np.all(np.isnan(res.spread[:59]))
    assert np.all(np.isfinite(res.spread[59:]))
    assert np.isfinite(res.beta)


def test_ols_resid_no_lookahead(panel):
    res_clean = ols_resid(panel, "BTC", "ETH", t_idx=500, lookback=60)
    polluted_ds = panel.ds.copy(deep=True)
    rng = np.random.default_rng(0xCA1F)
    for field in panel.fields:
        arr = polluted_ds[field].values
        arr[501:] = rng.normal(100, 10, size=arr[501:].shape)
        polluted_ds[field].values[...] = arr
    polluted_panel = PanelData(ds=polluted_ds)
    res_polluted = ols_resid(polluted_panel, "BTC", "ETH", t_idx=500, lookback=60)
    np.testing.assert_array_equal(res_clean.spread, res_polluted.spread)
    assert res_clean.beta == res_polluted.beta


def test_kalman_beta_spread_emits_trajectory(panel):
    res = kalman_beta_spread(panel, "BTC", "ETH", t_idx=499)
    assert res.spread.shape == (500,)
    assert isinstance(res.beta, np.ndarray)
    assert res.beta.shape == (500,)
    assert np.all(np.isfinite(res.spread))


def test_kalman_beta_no_lookahead(panel):
    res_clean = kalman_beta_spread(panel, "BTC", "ETH", t_idx=500)
    polluted_ds = panel.ds.copy(deep=True)
    rng = np.random.default_rng(0xCA1F)
    for field in panel.fields:
        arr = polluted_ds[field].values
        arr[501:] = rng.normal(100, 10, size=arr[501:].shape)
        polluted_ds[field].values[...] = arr
    polluted_panel = PanelData(ds=polluted_ds)
    res_polluted = kalman_beta_spread(polluted_panel, "BTC", "ETH", t_idx=500)
    np.testing.assert_array_equal(res_clean.spread, res_polluted.spread)


def test_pca_resid_emits_spread(panel):
    res = pca_resid(panel, "BTC", t_idx=499, other_assets=["ETH"], lookback=60)
    assert res.spread.shape == (500,)
    # First lookback bars are NaN.
    assert np.all(np.isnan(res.spread[:60]))
    assert np.all(np.isfinite(res.spread[60:]))


def test_pca_resid_no_lookahead(panel):
    res_clean = pca_resid(panel, "BTC", t_idx=500, other_assets=["ETH"], lookback=60)
    polluted_ds = panel.ds.copy(deep=True)
    rng = np.random.default_rng(0xCA1F)
    for field in panel.fields:
        arr = polluted_ds[field].values
        arr[501:] = rng.normal(100, 10, size=arr[501:].shape)
        polluted_ds[field].values[...] = arr
    polluted_panel = PanelData(ds=polluted_ds)
    res_polluted = pca_resid(polluted_panel, "BTC", t_idx=500, other_assets=["ETH"], lookback=60)
    np.testing.assert_array_equal(res_clean.spread, res_polluted.spread)
