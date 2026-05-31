"""Tests for OI loader (Phase 3 item #40)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from backtester.carry.oi import load_oi, oi_at


HERE = Path(__file__).resolve().parent
FIX = HERE / "fixtures" / "oi_btc_perp_1h_7d.parquet"


def test_load_oi_basic():
    df = load_oi(FIX)
    assert len(df) == 168  # 7 days × 24 hours
    assert "open_interest" in df.columns


def test_load_oi_cadence_check_passes():
    df = load_oi(FIX, expected_cadence_s=3600, cadence_tol_s=60)
    deltas = np.diff(df["time"].values)
    assert (np.abs(deltas - 3600) <= 60).all()


def test_load_oi_rejects_misread_cadence(tmp_path):
    df = pd.read_parquet(FIX).copy()
    # Halving cadence == 1800s rows; loader expecting 3600 ± 60 must fail.
    df["time"] = df.iloc[0]["time"] + np.arange(len(df)) * 1800
    bad = tmp_path / "misread.parquet"
    df.to_parquet(bad)
    with pytest.raises(ValueError, match="cadence"):
        load_oi(bad, expected_cadence_s=3600, cadence_tol_s=60)


def test_oi_at_no_lookahead():
    df = load_oi(FIX)
    rng = np.random.default_rng(0xCA1F)
    cuts = rng.choice(df["time"].values[10:-1], size=10, replace=False)
    for t in cuts:
        clean = oi_at(df, int(t))
        polluted = df.copy()
        polluted.loc[polluted["time"].values > t, "open_interest"] = 0.0
        p = oi_at(polluted, int(t))
        assert clean == p
