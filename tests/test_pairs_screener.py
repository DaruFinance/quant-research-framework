"""Tests for the pair / spread screener (item #9, Phase 3 — HIGH-RISK)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

xr = pytest.importorskip("xarray")
sm = pytest.importorskip("statsmodels")

from backtester.panel import PanelData, load_panel
from backtester.pairs import (
    ScreenedPair,
    distance_ssd,
    engle_granger,
    screen_pairs,
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


def test_engle_granger_returns_pvalue_and_beta(panel):
    btc = panel.ds["close"].values[:1000, panel.assets.index("BTC")]
    eth = panel.ds["close"].values[:1000, panel.assets.index("ETH")]
    p, beta = engle_granger(btc, eth)
    assert 0.0 <= p <= 1.0
    assert 0.0 < beta < 5.0  # reasonable range for log-log slope


def test_engle_granger_cointegrates_btc_eth_on_fixture(panel):
    """DS-PAIR-BTCETH was specifically chosen as a cointegrating
    window — ADF p-value on the full window must be < 0.05."""
    btc = panel.ds["close"].values[:, panel.assets.index("BTC")]
    eth = panel.ds["close"].values[:, panel.assets.index("ETH")]
    p, _ = engle_granger(btc, eth)
    assert p < 0.05, f"BTC/ETH should cointegrate on fixture; got p={p}"


def test_distance_ssd_lower_for_more_similar_log_prices(panel):
    # Standardised log-prices are forced to mean 0 std 1, so distance
    # is bounded; lower is closer.
    btc = panel.ds["close"].values[:500, panel.assets.index("BTC")]
    eth = panel.ds["close"].values[:500, panel.assets.index("ETH")]
    d_real = distance_ssd(btc, eth)
    # Constructed near-identical series: d should be much smaller.
    d_identical = distance_ssd(btc, btc)
    assert d_identical < d_real


def test_screen_pairs_returns_ranked_list(panel):
    pairs = screen_pairs(panel, t_idx=499, method="engle_granger",
                          lookback=500)
    assert len(pairs) == 1  # only one ordered (a, b) pair for 2 assets
    pair = pairs[0]
    assert pair.asset_a == "BTC" and pair.asset_b == "ETH"
    assert pair.statistic < 0.05  # cointegration confirmed
    assert "beta" in pair.extras


def test_screen_pairs_returns_empty_when_insufficient_data(panel):
    pairs = screen_pairs(panel, t_idx=10, lookback=500)
    assert pairs == []


# ---------------------------------------------------------------------------
# HIGH-RISK: 10-window pollute test
# ---------------------------------------------------------------------------
def test_screener_no_lookahead_10_windows(panel):
    """For 10 different ``t_idx`` endpoints, pollute panel rows past
    that endpoint and assert the returned pair list is bit-identical
    to the unpolluted run."""
    rng = np.random.default_rng(0xCA1F)
    n = len(panel)
    cuts = sorted(rng.choice(range(600, n - 1), size=10, replace=False).tolist())
    for cut_idx in cuts:
        clean = screen_pairs(panel, t_idx=cut_idx, method="engle_granger",
                              lookback=500)
        polluted_ds = panel.ds.copy(deep=True)
        for field in panel.fields:
            arr = polluted_ds[field].values
            arr[cut_idx + 1:] = rng.normal(100, 10, size=arr[cut_idx + 1:].shape)
            polluted_ds[field].values[...] = arr
        polluted_panel = PanelData(ds=polluted_ds)
        polluted = screen_pairs(polluted_panel, t_idx=cut_idx,
                                  method="engle_granger", lookback=500)
        assert len(clean) == len(polluted), f"cut={cut_idx}"
        for c, p in zip(clean, polluted):
            assert c.asset_a == p.asset_a
            assert c.asset_b == p.asset_b
            assert c.statistic == p.statistic, (
                f"cut={cut_idx}: statistic differs {c.statistic} vs "
                f"{p.statistic}"
            )
            assert c.extras == p.extras


def test_screen_pairs_rejects_unknown_method(panel):
    with pytest.raises(ValueError, match="unknown method"):
        screen_pairs(panel, t_idx=499, method="nonsense", lookback=500)  # type: ignore[arg-type]
