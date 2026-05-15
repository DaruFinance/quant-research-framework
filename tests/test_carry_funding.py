"""Tests for funding-rate loader (Phase 3 item #38)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from backtester.carry.funding import (
    FUNDING_INTERVAL_S,
    load_funding,
    next_funding_time,
    rate_at,
)


HERE = Path(__file__).resolve().parent
FIX = HERE / "fixtures" / "funding_btcusdt_200evt.parquet"


def test_load_funding_basic():
    df = load_funding(FIX)
    assert len(df) == 200
    assert list(df.columns) == ["time", "rate"]
    assert df["time"].is_monotonic_increasing
    assert not df["rate"].isna().any()


def test_load_funding_boundary_aligned():
    df = load_funding(FIX, strict_boundary=True)
    rems = df["time"].values % FUNDING_INTERVAL_S
    assert (rems == 0).all()


def test_load_funding_rejects_misaligned(tmp_path):
    df = pd.read_parquet(FIX).copy()
    df.loc[0, "time"] = int(df.loc[0, "time"]) + 7  # off by 7s
    bad = tmp_path / "bad.parquet"
    df.to_parquet(bad)
    with pytest.raises(ValueError, match="not aligned"):
        load_funding(bad, strict_boundary=True)


def test_load_funding_rejects_nan(tmp_path):
    df = pd.read_parquet(FIX).copy()
    df.loc[5, "rate"] = float("nan")
    bad = tmp_path / "nan.parquet"
    df.to_parquet(bad)
    with pytest.raises(ValueError, match="NaN"):
        load_funding(bad)


def test_next_funding_time_aligned():
    # 1773072000 is aligned (00:00 UTC). +1 second → +28799s to next.
    assert next_funding_time(1773072000) == 1773072000
    assert (next_funding_time(1773072001)
              == 1773072000 + FUNDING_INTERVAL_S)


def test_rate_at_no_lookahead():
    df = load_funding(FIX)
    rng = np.random.default_rng(0xCA1F)
    # 20 random query timestamps in the loaded range.
    cuts = rng.choice(df["time"].values[10:-1], size=20, replace=False)
    for t in cuts:
        clean = rate_at(df, int(t))
        polluted = df.copy()
        # Pollute rows past t.
        mask = polluted["time"].values > t
        polluted.loc[mask, "rate"] = 9.99
        polluted_rate = rate_at(polluted, int(t))
        assert clean == polluted_rate, f"t={t}: rate differs"


def test_rate_at_strict_raises_before_first_event():
    df = load_funding(FIX)
    with pytest.raises(LookupError):
        rate_at(df, 0, fill="strict")


def test_rate_at_ffill_returns_none_before_first():
    df = load_funding(FIX)
    assert rate_at(df, 0, fill="ffill") is None
