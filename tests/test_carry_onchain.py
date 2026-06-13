"""Tests for on-chain loader & snapshot pinning (Phase 3 item #41)."""
from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd
import pytest

from backtester.carry.onchain import load_onchain, value_at


HERE = Path(__file__).resolve().parent
FIX = HERE / "fixtures" / "onchain_nvt_50d.csv"


def test_load_onchain_basic():
    df = load_onchain(FIX, metric="nvt")
    assert len(df) == 50
    assert list(df.columns) == ["time", "value"]
    assert df["time"].is_monotonic_increasing
    assert "snapshot_sha256" in df.attrs


def test_load_onchain_unknown_metric_raises(tmp_path):
    p = tmp_path / "bad.csv"
    pd.DataFrame({"time": [0, 1], "nvt": [1.0, 2.0]}).to_csv(p, index=False)
    with pytest.raises(ValueError, match="needs columns"):
        load_onchain(p, metric="missing_col")


def test_load_onchain_snapshot_pin_rejects_revised_value(tmp_path):
    """The CRITICAL on-chain leak: a provider revises a historical
    value AFTER our snapshot was loaded.  We require the snapshot's
    SHA-256 to identify a future re-load that should NOT match our
    pinned dataframe.

    We simulate it by loading the fixture, then writing a *modified*
    copy to disk, then reloading; the two DataFrames must have
    different ``snapshot_sha256`` attrs so the consumer can detect
    the revision."""
    pinned = load_onchain(FIX, metric="nvt")
    revised_path = tmp_path / "revised.csv"
    shutil.copy(FIX, revised_path)
    revised_df = pd.read_csv(revised_path)
    revised_df.loc[5, "nvt"] = 999999.0  # backfill revision
    revised_df.to_csv(revised_path, index=False)
    revised = load_onchain(revised_path, metric="nvt")
    assert pinned.attrs["snapshot_sha256"] != revised.attrs["snapshot_sha256"]
    # And the pinned DataFrame's value at the revised row is unchanged.
    assert pinned.iloc[5]["value"] != revised.iloc[5]["value"]


def test_value_at_no_lookahead():
    df = load_onchain(FIX, metric="nvt")
    # Pick a midpoint timestamp; ensure pollution of future rows does
    # not alter the value at this point in time.
    cut_ts = int(df["time"].iloc[25])
    clean = value_at(df, cut_ts)
    polluted = df.copy()
    polluted.loc[polluted["time"] > cut_ts, "value"] = -1.0
    p = value_at(polluted, cut_ts)
    assert clean == p


def test_value_at_returns_none_before_first():
    df = load_onchain(FIX, metric="nvt")
    assert value_at(df, 0) is None
