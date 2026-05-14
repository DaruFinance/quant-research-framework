"""Tests for the panel data loader (item #1, Phase 2).

Verification gates:

G1 (parity): pure new code path — none of the four parity scripts are
   affected. Verified by running them after this item lands; here we
   only assert the loader's contract.

G2 (property tests):
   - idempotence (same paths -> identical Dataset).
   - gap-detection (one asset missing a bar -> PanelGapError naming
     the offending timestamp).
   - schema-detection (missing required column -> PanelSchemaError).
   - inner-join (only timestamps common to ALL assets appear).
   - lookahead-free (polluting tail of one input doesn't change earlier
     rows).

G3 (real-data verification): 5 random `(t, asset)` cells from the
   DS-PANEL-3 sources reconcile bit-identically against the raw CSVs.
"""
from __future__ import annotations

import csv
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Note: backtester.panel is pulled via `pip install -e .[panel]` in CI;
# skip the whole file if xarray isn't available so non-panel users can
# run pytest without errors.
xr = pytest.importorskip("xarray")

from backtester.panel import (  # noqa: E402
    PanelData,
    PanelGapError,
    PanelSchemaError,
    load_panel,
)


HERE = Path(__file__).resolve().parent
FIXTURE_DIR = HERE / "fixtures" / "sources"
PANEL_PATHS = {
    "SOL": FIXTURE_DIR / "SOLUSDT_1h_30000_31000.csv",
    "BTC": FIXTURE_DIR / "BTCUSDT_1h_jan_feb_2024.csv",
    "ETH": FIXTURE_DIR / "ETHUSDT_1h_jan_feb_2024.csv",
}


# ---------------------------------------------------------------------------
# Real-data verification: DS-PANEL-3 reconciliation (G3-style)
# ---------------------------------------------------------------------------
def test_load_panel_ds_panel_3_shape():
    panel = load_panel(PANEL_PATHS)
    assert isinstance(panel, PanelData)
    assert panel.assets == ["SOL", "BTC", "ETH"]
    assert panel.fields == ["open", "high", "low", "close"]
    assert len(panel) == 1000
    assert panel.ds.attrs["interval_seconds"] == 3600


def test_load_panel_5_random_cells_match_sources():
    panel = load_panel(PANEL_PATHS)
    sources = {a: pd.read_csv(p) for a, p in PANEL_PATHS.items()}
    rng = np.random.default_rng(seed=42)
    for _ in range(5):
        ti = int(rng.integers(0, len(panel)))
        ai = int(rng.integers(0, 3))
        asset = panel.assets[ai]
        t = int(panel.times[ti])
        for field in panel.fields:
            cell = float(panel.ds[field].values[ti, ai])
            src = float(sources[asset][sources[asset]["time"] == t][field].iloc[0])
            assert abs(cell - src) < 1e-12, (
                f"t={t} asset={asset} field={field}: panel={cell} src={src}"
            )


# ---------------------------------------------------------------------------
# Property tests
# ---------------------------------------------------------------------------
def test_idempotent_loads_produce_identical_data():
    a = load_panel(PANEL_PATHS)
    b = load_panel(PANEL_PATHS)
    # Compare every data var element-wise.
    for field in a.fields:
        np.testing.assert_array_equal(a.ds[field].values, b.ds[field].values)
    np.testing.assert_array_equal(a.times, b.times)
    assert a.assets == b.assets
    assert a.ds.attrs == b.ds.attrs


def test_inner_join_drops_unmatched_timestamps(tmp_path):
    """If asset X has an extra timestamp not in the others, it must be
    dropped silently (inner-join semantics)."""
    base = pd.read_csv(PANEL_PATHS["SOL"])
    # Truncate BTC to 500 bars so its time set is a strict subset.
    btc = pd.read_csv(PANEL_PATHS["BTC"]).head(500)
    btc_path = tmp_path / "btc_short.csv"
    btc.to_csv(btc_path, index=False)
    paths = {
        "SOL": PANEL_PATHS["SOL"],
        "BTC": btc_path,
        "ETH": PANEL_PATHS["ETH"],
    }
    panel = load_panel(paths)
    # Intersection ≤ min(N_each); BTC has 500 bars so intersection ≤ 500.
    assert len(panel) <= 500
    # Every retained time must appear in all three input files.
    btc_times = set(btc["time"].astype(int))
    sol_times = set(base["time"].astype(int))
    eth_times = set(pd.read_csv(PANEL_PATHS["ETH"])["time"].astype(int))
    for t in panel.times:
        t = int(t)
        assert t in btc_times and t in sol_times and t in eth_times


def test_gap_in_one_asset_raises_panel_gap_error(tmp_path):
    """If the inner-join is no longer uniformly spaced because one
    asset's CSV has a hole, ``load_panel`` must raise PanelGapError
    naming the first offending timestamp."""
    sol = pd.read_csv(PANEL_PATHS["SOL"])
    btc = pd.read_csv(PANEL_PATHS["BTC"])
    eth = pd.read_csv(PANEL_PATHS["ETH"])
    # Drop one BTC row from the middle to create a hole.
    drop_idx = 500
    dropped_t = int(btc.iloc[drop_idx]["time"])
    btc_holed = pd.concat([btc.iloc[:drop_idx], btc.iloc[drop_idx + 1:]],
                           ignore_index=True)
    btc_path = tmp_path / "btc_holed.csv"
    btc_holed.to_csv(btc_path, index=False)
    paths = {
        "SOL": PANEL_PATHS["SOL"],
        "BTC": btc_path,
        "ETH": PANEL_PATHS["ETH"],
    }
    with pytest.raises(PanelGapError) as exc:
        load_panel(paths)
    # Error message must surface a specific offending timestamp.
    assert "gap" in str(exc.value).lower()
    # The offending ts is the bar AFTER the dropped one (post-join
    # diff > 1 hour at that boundary).
    assert exc.value.ts == dropped_t + 3600


def test_missing_required_column_raises_schema_error(tmp_path):
    bad = pd.read_csv(PANEL_PATHS["SOL"]).drop(columns=["high"])
    bad_path = tmp_path / "sol_no_high.csv"
    bad.to_csv(bad_path, index=False)
    paths = {
        "SOL": bad_path,
        "BTC": PANEL_PATHS["BTC"],
        "ETH": PANEL_PATHS["ETH"],
    }
    with pytest.raises(PanelSchemaError) as exc:
        load_panel(paths)
    assert "high" in str(exc.value)


def test_duplicate_timestamps_raise_schema_error(tmp_path):
    sol = pd.read_csv(PANEL_PATHS["SOL"])
    # Duplicate the first row's timestamp.
    dup = pd.concat([sol, sol.iloc[[0]]], ignore_index=True)
    dup_path = tmp_path / "sol_dup.csv"
    dup.to_csv(dup_path, index=False)
    paths = {
        "SOL": dup_path,
        "BTC": PANEL_PATHS["BTC"],
        "ETH": PANEL_PATHS["ETH"],
    }
    with pytest.raises(PanelSchemaError) as exc:
        load_panel(paths)
    assert "duplicate" in str(exc.value).lower()


def test_empty_paths_dict_raises():
    with pytest.raises(PanelSchemaError) as exc:
        load_panel({})
    assert "empty" in str(exc.value).lower()


def test_loader_no_lookahead_under_tail_pollution(tmp_path):
    """The loader is pure inner-join + alignment. Polluting the tail of
    one CSV with garbage rows (still in monotonically-increasing time)
    must not change the data at earlier rows in the panel — the inner-
    join semantics guarantee this, and we surface the guarantee as a
    test for the future plugin code that depends on it."""
    sol = pd.read_csv(PANEL_PATHS["SOL"])
    btc = pd.read_csv(PANEL_PATHS["BTC"])
    eth = pd.read_csv(PANEL_PATHS["ETH"])

    # Append 50 garbage rows to BTC at the same 1h cadence after its
    # natural end. They won't survive the inner-join (no SOL/ETH match).
    last_t = int(btc["time"].iloc[-1])
    garbage = pd.DataFrame({
        "time":  [last_t + 3600 * (i + 1) for i in range(50)],
        "open":  np.full(50, 999_999.0),
        "high":  np.full(50, 999_999.0),
        "low":   np.full(50, 999_999.0),
        "close": np.full(50, 999_999.0),
    })
    btc_polluted = pd.concat([btc, garbage], ignore_index=True)
    btc_path = tmp_path / "btc_tail_polluted.csv"
    btc_polluted.to_csv(btc_path, index=False)

    clean = load_panel(PANEL_PATHS)
    polluted = load_panel({
        "SOL": PANEL_PATHS["SOL"],
        "BTC": btc_path,
        "ETH": PANEL_PATHS["ETH"],
    })

    np.testing.assert_array_equal(clean.times, polluted.times)
    for field in clean.fields:
        np.testing.assert_array_equal(
            clean.ds[field].values, polluted.ds[field].values
        )
