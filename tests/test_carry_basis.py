"""Tests for basis loader (Phase 3 item #39)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from backtester.carry.basis import basis_at, load_basis


HERE = Path(__file__).resolve().parent
FIX = HERE / "fixtures" / "basis_btc_perp_spot_1d.parquet"


def test_load_basis_basic():
    df = load_basis(FIX)
    assert len(df) == 24
    assert set(["time", "close_spot", "close_perp", "basis_bp"]).issubset(df.columns)


def test_load_basis_recompute_matches_input():
    """The fixture's basis_bp must agree with a fresh recompute to
    well within the 0.01bp tolerance — this is what makes the loader
    a reliable lookahead-screen for forward-looking sources."""
    df = load_basis(FIX)
    fresh = (df["close_perp"] - df["close_spot"]) / df["close_spot"] * 1e4
    assert np.allclose(df["basis_bp"], fresh, atol=1e-6)


def test_load_basis_rejects_smoothed_input(tmp_path):
    """Inject a 5-bp drift into the basis_bp column and assert the
    loader catches it."""
    raw = pd.read_parquet(FIX).copy()
    raw["basis_bp"] = raw["basis_bp"] + 5.0
    bad = tmp_path / "drifted.parquet"
    raw.to_parquet(bad)
    with pytest.raises(ValueError, match="drifted from fresh recompute"):
        load_basis(bad)


def test_basis_at_no_lookahead():
    df = load_basis(FIX)
    rng = np.random.default_rng(0xCA1F)
    cuts = rng.choice(df["time"].values[5:-1], size=10, replace=False)
    for t in cuts:
        clean = basis_at(df, int(t))
        polluted = df.copy()
        mask = polluted["time"].values > t
        polluted.loc[mask, "close_perp"] = 9999.0
        polluted.loc[mask, "basis_bp"] = 999.0
        p = basis_at(polluted, int(t))
        assert clean == p, f"t={t}: basis_at differs after future pollution"


def test_basis_at_returns_none_before_first():
    df = load_basis(FIX)
    assert basis_at(df, 0) is None
