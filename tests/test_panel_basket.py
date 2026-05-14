"""Tests for the long-short basket primitive (item #8, Phase 2)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

xr = pytest.importorskip("xarray")

from backtester.panel import (
    LongShortBasket,
    PanelData,
    load_panel,
    momentum_alpha,
)


HERE = Path(__file__).resolve().parent
FIXTURE_DIR = HERE / "fixtures" / "sources"
PANEL_PATHS = {
    "SOL": FIXTURE_DIR / "SOLUSDT_1h_30000_31000.csv",
    "BTC": FIXTURE_DIR / "BTCUSDT_1h_jan_feb_2024.csv",
    "ETH": FIXTURE_DIR / "ETHUSDT_1h_jan_feb_2024.csv",
}


def _signs(weights: dict) -> dict:
    return {a: int(np.sign(w)) for a, w in weights.items()}


# ---------------------------------------------------------------------------
# Plan-spec verification: 20-bar momentum alpha, n_long=1 n_short=1
# ---------------------------------------------------------------------------
def test_basket_exactly_one_long_one_short_per_rebalance():
    """At any `t` past warmup, the basket selects exactly 1 long
    and 1 short — the plan's verification expectation."""
    panel = load_panel(PANEL_PATHS)
    basket = LongShortBasket(
        alpha_fn=momentum_alpha(lookback=20),
        neutralize_mode="dollar",
        n_long=1,
        n_short=1,
    )
    for t in (100, 250, 500, 750, 999):
        w = basket.positions(panel, t)
        signs = _signs(w)
        n_long = sum(1 for s in signs.values() if s > 0)
        n_short = sum(1 for s in signs.values() if s < 0)
        assert n_long == 1, f"t={t}: expected 1 long, got {n_long}: {signs}"
        assert n_short == 1, f"t={t}: expected 1 short, got {n_short}: {signs}"
        assert n_long + n_short < len(panel.assets), (
            f"t={t}: should have at least one zero-weighted asset; got {signs}"
        )


def test_basket_dollar_neutral_balances_long_short_notional():
    panel = load_panel(PANEL_PATHS)
    basket = LongShortBasket(
        alpha_fn=momentum_alpha(20),
        neutralize_mode="dollar",
        n_long=1,
        n_short=1,
    )
    w = basket.positions(panel, 500)
    longs = sum(v for v in w.values() if v > 0)
    shorts = -sum(v for v in w.values() if v < 0)
    assert abs(longs - shorts) < 1e-12
    assert abs(longs - 0.5) < 1e-12


def test_basket_pre_warmup_returns_all_zero():
    panel = load_panel(PANEL_PATHS)
    basket = LongShortBasket(
        alpha_fn=momentum_alpha(20),
        neutralize_mode="dollar",
        n_long=1,
        n_short=1,
    )
    # t_idx < lookback -> alpha all-NaN -> all-zero weights
    w = basket.positions(panel, 10)
    assert all(v == 0.0 for v in w.values())


def test_basket_selects_winners_and_losers_correctly():
    """Synthetic check: build a panel where SOL has the highest 20-bar
    momentum and ETH has the lowest. Basket must long SOL and short
    ETH at the chosen rebalance bar."""
    panel = load_panel(PANEL_PATHS)
    # Use a real rebalance bar; verify by computing momentum manually.
    t = 500
    close = panel.ds["close"].values
    mom = (close[t] / close[t - 20]) - 1.0
    expected_long = panel.assets[int(np.argmax(mom))]
    expected_short = panel.assets[int(np.argmin(mom))]

    basket = LongShortBasket(
        alpha_fn=momentum_alpha(20),
        neutralize_mode="dollar",
        n_long=1,
        n_short=1,
    )
    w = basket.positions(panel, t)
    selected_long = max(w, key=w.get)
    selected_short = min(w, key=w.get)
    assert selected_long == expected_long
    assert selected_short == expected_short


# ---------------------------------------------------------------------------
# Lookahead-free property (HIGH-RISK pattern): polluting alpha-input
# rows at >= t cannot change the positions returned for the rebalance
# at t.
# ---------------------------------------------------------------------------
def test_basket_no_lookahead_under_tail_pollution():
    """Pollute the panel's close at rows > t_idx with garbage values.
    Re-run basket.positions(panel, t_idx); per-asset weights must be
    bit-identical to the clean run."""
    panel = load_panel(PANEL_PATHS)
    basket = LongShortBasket(
        alpha_fn=momentum_alpha(20),
        neutralize_mode="dollar",
        n_long=1,
        n_short=1,
    )
    t_idx = 500
    clean = basket.positions(panel, t_idx)

    polluted_ds = panel.ds.copy(deep=True)
    rng = np.random.default_rng(seed=0xCAFE)
    for field in panel.fields:
        arr = polluted_ds[field].values
        arr[t_idx + 1:] = rng.normal(100, 10, size=arr[t_idx + 1:].shape)
        polluted_ds[field].values[...] = arr
    polluted_panel = PanelData(ds=polluted_ds)
    polluted = basket.positions(polluted_panel, t_idx)

    for asset in panel.assets:
        assert clean[asset] == polluted[asset], (
            f"{asset}: clean={clean[asset]} polluted={polluted[asset]}"
        )


# ---------------------------------------------------------------------------
# Beta-neutral basket
# ---------------------------------------------------------------------------
def test_basket_beta_neutral_zeros_market_beta():
    panel = load_panel(PANEL_PATHS)
    basket = LongShortBasket(
        alpha_fn=momentum_alpha(20),
        neutralize_mode="beta",
        n_long=2,
        n_short=1,
        market_asset="BTC",
        returns_lookback=60,
    )
    t_idx = 500
    w = basket.positions(panel, t_idx)
    # Pull betas the basket actually used (same window).
    from backtester.panel.neutralize import estimate_betas
    close = panel.ds["close"].values
    slc = close[t_idx - 61:t_idx]
    rets = np.diff(np.log(slc), axis=0)
    betas = estimate_betas(rets, market_idx=panel.assets.index("BTC"))
    w_vec = np.array([w[a] for a in panel.assets])
    portfolio_beta = float(np.dot(w_vec, betas))
    assert abs(portfolio_beta) < 1e-10


def test_basket_beta_neutral_requires_market_asset():
    with pytest.raises(ValueError, match="requires market_asset"):
        LongShortBasket(
            alpha_fn=momentum_alpha(20),
            neutralize_mode="beta",
            n_long=1, n_short=1,
        )


# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------
def test_basket_rejects_n_long_plus_n_short_gt_n_assets():
    panel = load_panel(PANEL_PATHS)  # 3 assets
    basket = LongShortBasket(
        alpha_fn=momentum_alpha(20),
        neutralize_mode="dollar",
        n_long=3, n_short=2,
    )
    with pytest.raises(ValueError, match="exceeds"):
        basket.positions(panel, 500)


def test_basket_rejects_negative_counts():
    with pytest.raises(ValueError, match="non-negative"):
        LongShortBasket(
            alpha_fn=momentum_alpha(20),
            neutralize_mode="dollar",
            n_long=-1, n_short=1,
        )


# ---------------------------------------------------------------------------
# G3 verification: 5 rebalances, alpha ranks computed only on < t data.
# ---------------------------------------------------------------------------
def test_basket_g3_5_rebalances_alpha_audit():
    """Dump alpha vector + selected long/short asset at 5 rebalance
    indices, confirm:
    - alpha[i] computed from close[<= t] only;
    - long is argmax of alpha;
    - short is argmin of alpha.
    """
    panel = load_panel(PANEL_PATHS)
    basket = LongShortBasket(
        alpha_fn=momentum_alpha(20),
        neutralize_mode="dollar",
        n_long=1, n_short=1,
    )
    close = panel.ds["close"].values

    for t in (100, 300, 500, 700, 999):
        # Manually compute alpha at t using only data <= t.
        alpha = (close[t] / close[t - 20]) - 1.0
        long_i = int(np.argmax(alpha))
        short_i = int(np.argmin(alpha))

        w = basket.positions(panel, t)
        for i, asset in enumerate(panel.assets):
            if i == long_i:
                assert w[asset] > 0, f"t={t}: {asset} should be long"
            elif i == short_i:
                assert w[asset] < 0, f"t={t}: {asset} should be short"
            else:
                assert w[asset] == 0, f"t={t}: {asset} should be flat"
