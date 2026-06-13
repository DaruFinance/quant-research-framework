"""Tests for funding-signal models (Phase 3 item #43)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from backtester.carry.funding import load_funding
from backtester.carry.models import (
    FundingMomentumModel,
    FundingOICointegrationModel,
    PersistentFundingSignModel,
)
from backtester.carry.oi import load_oi


HERE = Path(__file__).resolve().parent
FUNDING = HERE / "fixtures" / "funding_btcusdt_200evt.parquet"
OI = HERE / "fixtures" / "oi_btc_perp_1h_7d.parquet"


def test_persistent_sign_emits_flat_below_streak():
    rates = pd.DataFrame({
        "time": np.arange(10) * 28800,
        "rate": [1e-4, -1e-4, 1e-4, -1e-4, 1e-4, -1e-4, 1e-4, -1e-4, 1e-4, -1e-4],
    })
    model = PersistentFundingSignModel(min_streak=3)
    sig = model.signal_at(rates, int(rates["time"].iloc[-1]))
    assert sig.direction == 0


def test_persistent_sign_emits_carry_direction_after_streak():
    rates = pd.DataFrame({
        "time": np.arange(5) * 28800,
        "rate": [1e-4, 1e-4, 1e-4, 1e-4, 1e-4],
    })
    model = PersistentFundingSignModel(min_streak=3)
    sig = model.signal_at(rates, int(rates["time"].iloc[-1]))
    # Five positive funding events => short carry (we are short).
    assert sig.direction == -1
    assert sig.strength >= 3


def test_persistent_sign_no_lookahead():
    df = load_funding(FUNDING)
    model = PersistentFundingSignModel(min_streak=3)
    rng = np.random.default_rng(0xCA1F)
    for _ in range(10):
        t = int(rng.choice(df["time"].values[10:-5]))
        clean = model.signal_at(df, t)
        polluted = df.copy()
        polluted.loc[polluted["time"] > t, "rate"] *= -100
        p = model.signal_at(polluted, t)
        assert clean.direction == p.direction
        assert clean.strength == p.strength


def test_funding_momentum_flat_below_threshold():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "time": np.arange(50) * 28800,
        "rate": rng.normal(0, 1e-5, size=50),  # small noise
    })
    sig = FundingMomentumModel(window=20, z_thresh=3.0).signal_at(
        df, int(df["time"].iloc[-1]))
    assert sig.direction == 0


def test_funding_momentum_fires_on_spike():
    rng = np.random.default_rng(0)
    rates = rng.normal(0, 1e-5, size=50)
    rates[-1] = 1.0  # massive z
    df = pd.DataFrame({"time": np.arange(50) * 28800, "rate": rates})
    sig = FundingMomentumModel(window=20, z_thresh=3.0).signal_at(
        df, int(df["time"].iloc[-1]))
    assert sig.direction != 0
    assert sig.strength > 3.0


def test_funding_momentum_no_lookahead():
    df = load_funding(FUNDING)
    model = FundingMomentumModel(window=20, z_thresh=1.5)
    rng = np.random.default_rng(0xCA1F)
    for _ in range(10):
        t = int(rng.choice(df["time"].values[30:-1]))
        clean = model.signal_at(df, t)
        polluted = df.copy()
        polluted.loc[polluted["time"] > t, "rate"] *= 1e6
        p = model.signal_at(polluted, t)
        assert clean.direction == p.direction


def test_funding_oi_cointegration_emits():
    funding = load_funding(FUNDING)
    oi = load_oi(OI)
    # Common timestamp present in both fixtures.
    t = int(min(funding["time"].max(), oi["time"].max()))
    if t > int(funding["time"].iloc[20]):
        sig = FundingOICointegrationModel(window=10).signal_at(funding, oi, t)
        # Signal valid; direction may be +1, 0 or -1 depending on data.
        assert sig.direction in {-1, 0, 1}
