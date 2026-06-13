"""Tests for the neutralization primitives (item #7, Phase 2)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

xr = pytest.importorskip("xarray")

from backtester.panel import load_panel
from backtester.panel.neutralize import (
    Mode,
    estimate_betas,
    estimate_vols,
    neutralize,
)


HERE = Path(__file__).resolve().parent
FIXTURE_DIR = HERE / "fixtures" / "sources"
PANEL_PATHS = {
    "SOL": FIXTURE_DIR / "SOLUSDT_1h_30000_31000.csv",
    "BTC": FIXTURE_DIR / "BTCUSDT_1h_jan_feb_2024.csv",
    "ETH": FIXTURE_DIR / "ETHUSDT_1h_jan_feb_2024.csv",
}


def _panel_returns(panel) -> np.ndarray:
    close = panel.ds["close"].values
    return np.diff(np.log(close), axis=0)


# ---------------------------------------------------------------------------
# dollar-neutral
# ---------------------------------------------------------------------------
def test_dollar_neutral_balances_gross_long_short():
    raw = np.array([0.3, -0.7, 0.5, -0.2])
    w = neutralize(raw, "dollar")
    long_sum = w[w > 0].sum()
    short_sum = -w[w < 0].sum()
    assert abs(long_sum - short_sum) < 1e-12
    assert abs(long_sum - 0.5) < 1e-12
    assert abs(short_sum - 0.5) < 1e-12


def test_dollar_neutral_preserves_sign_of_each_position():
    raw = np.array([0.3, -0.7, 0.5, -0.2])
    w = neutralize(raw, "dollar")
    assert np.all(np.sign(w) == np.sign(raw))


def test_dollar_neutral_rejects_all_long_basket():
    with pytest.raises(ValueError, match="both long and short"):
        neutralize(np.array([1.0, 0.5]), "dollar")


# ---------------------------------------------------------------------------
# beta-neutral
# ---------------------------------------------------------------------------
def test_beta_neutral_zeros_portfolio_beta_to_chosen_market():
    raw = np.array([0.4, -0.6, 0.5])
    betas = np.array([0.9, 1.0, 1.1])
    w = neutralize(raw, "beta", betas=betas, market_idx=1)
    assert abs(float(np.dot(w, betas))) < 1e-12


def test_beta_neutral_preserves_non_market_weights():
    raw = np.array([0.4, -0.6, 0.5])
    betas = np.array([0.9, 1.0, 1.1])
    w = neutralize(raw, "beta", betas=betas, market_idx=1)
    # Non-market legs stay as-is.
    assert w[0] == raw[0]
    assert w[2] == raw[2]
    # Market leg adjusted to absorb residual beta.
    expected_market = -(raw[0] * betas[0] + raw[2] * betas[2]) / betas[1]
    assert abs(w[1] - expected_market) < 1e-12


def test_beta_neutral_auto_picks_highest_abs_beta_if_unspecified():
    raw = np.array([0.4, -0.6, 0.5])
    betas = np.array([0.5, 1.5, 0.9])
    w = neutralize(raw, "beta", betas=betas)
    # Highest |beta| is at index 1 (1.5). After neutralization, dot
    # product is zero.
    assert abs(float(np.dot(w, betas))) < 1e-12


def test_beta_neutral_rejects_shape_mismatch():
    with pytest.raises(ValueError, match="shape mismatch"):
        neutralize(np.array([1.0, 0.0]), "beta", betas=np.array([1.0, 0.5, 0.3]))


def test_beta_neutral_rejects_zero_market_beta():
    raw = np.array([0.3, -0.7])
    betas = np.array([0.5, 0.0])
    with pytest.raises(ValueError, match="beta=0"):
        neutralize(raw, "beta", betas=betas, market_idx=1)


def test_beta_neutral_requires_betas():
    with pytest.raises(ValueError, match="requires betas"):
        neutralize(np.array([0.3, -0.7]), "beta")


# ---------------------------------------------------------------------------
# sigma-neutral
# ---------------------------------------------------------------------------
def test_sigma_neutral_equalises_vol_contribution_per_leg():
    raw = np.array([0.5, -0.3, 0.4])
    vols = np.array([0.02, 0.01, 0.03])
    w = neutralize(raw, "sigma", vols=vols)
    vc = np.abs(w) * vols
    # All vol contributions equal across legs.
    np.testing.assert_allclose(vc, vc[0], atol=1e-12)


def test_sigma_neutral_preserves_gross_notional():
    raw = np.array([0.5, -0.3, 0.4])
    vols = np.array([0.02, 0.01, 0.03])
    w = neutralize(raw, "sigma", vols=vols)
    assert abs(np.abs(w).sum() - np.abs(raw).sum()) < 1e-12


def test_sigma_neutral_preserves_signs():
    raw = np.array([0.5, -0.3, 0.4])
    vols = np.array([0.02, 0.01, 0.03])
    w = neutralize(raw, "sigma", vols=vols)
    assert np.all(np.sign(w) == np.sign(raw))


def test_sigma_neutral_rejects_zero_vol():
    with pytest.raises(ValueError, match="positive vols"):
        neutralize(np.array([0.5, -0.3]), "sigma",
                    vols=np.array([0.02, 0.0]))


def test_sigma_neutral_rejects_zero_weight():
    with pytest.raises(ValueError, match="non-zero"):
        neutralize(np.array([0.5, 0.0, 0.3]), "sigma",
                    vols=np.array([0.02, 0.01, 0.03]))


# ---------------------------------------------------------------------------
# Lookahead-free property: same as for sizing / loader, neutralize is a
# pure function of its inputs. We exercise this by polluting the
# returns matrix the caller would use to estimate betas/vols at >= cut,
# computing both versions of betas, and checking the neutralize output
# matches when fed the < cut-derived inputs.
# ---------------------------------------------------------------------------
def test_neutralize_no_lookahead_under_tail_pollution():
    panel = load_panel(PANEL_PATHS)
    rets = _panel_returns(panel)
    cut = 600
    raw = np.array([1.0, -1.0, 1.0])

    # Clean run: betas from rets[200:cut].
    betas_clean = estimate_betas(rets[200:cut], market_idx=1)
    vols_clean = estimate_vols(rets[200:cut])
    w_beta_clean = neutralize(raw, "beta", betas=betas_clean, market_idx=1)
    w_sigma_clean = neutralize(raw, "sigma", vols=vols_clean)
    w_dollar_clean = neutralize(raw, "dollar")

    # Pollute rows past cut.
    rng = np.random.default_rng(seed=0xCAFE)
    polluted = rets.copy()
    polluted[cut:] = rng.normal(0, 1.0, size=polluted[cut:].shape)
    betas_polluted = estimate_betas(polluted[200:cut], market_idx=1)
    vols_polluted = estimate_vols(polluted[200:cut])
    w_beta_polluted = neutralize(raw, "beta", betas=betas_polluted, market_idx=1)
    w_sigma_polluted = neutralize(raw, "sigma", vols=vols_polluted)
    w_dollar_polluted = neutralize(raw, "dollar")

    np.testing.assert_array_equal(w_beta_clean, w_beta_polluted)
    np.testing.assert_array_equal(w_sigma_clean, w_sigma_polluted)
    np.testing.assert_array_equal(w_dollar_clean, w_dollar_polluted)


# ---------------------------------------------------------------------------
# Real-data G3 verification: 5 rebalances on DS-PANEL-3 with the plan's
# basket (+SOL, -BTC, +ETH) under beta-neutral. Portfolio beta to BTC
# must be < 0.05 in absolute value.
# ---------------------------------------------------------------------------
def test_beta_neutral_5_rebalances_residual_beta_under_5pct():
    panel = load_panel(PANEL_PATHS)
    rets = _panel_returns(panel)
    raw = np.array([1.0, -1.0, 1.0])  # +SOL, -BTC, +ETH
    market_idx = panel.assets.index("BTC")

    rebalances = []
    lookback = 60  # plan says 60-bar OLS
    for end in (200, 400, 600, 800, 999):
        window = rets[end - lookback:end]
        betas = estimate_betas(window, market_idx=market_idx)
        w = neutralize(raw, "beta", betas=betas, market_idx=market_idx)
        residual_beta = float(np.dot(w, betas))
        rebalances.append((end, betas, w, residual_beta))
        assert abs(residual_beta) < 0.05, (
            f"end={end}: |residual beta| = {abs(residual_beta):.4f} "
            f">= 0.05; betas={betas} w={w}"
        )

    # Sanity: at least some non-trivial neutralization happened —
    # i.e. the market-leg weight differs from the raw -1.0.
    for end, betas, w, _ in rebalances:
        assert w[market_idx] != raw[market_idx], (
            f"end={end}: neutralization didn't adjust the market leg"
        )


def test_unknown_mode_raises():
    with pytest.raises(ValueError, match="unknown neutralization mode"):
        neutralize(np.array([1.0, -1.0]), "unknown")  # type: ignore[arg-type]
