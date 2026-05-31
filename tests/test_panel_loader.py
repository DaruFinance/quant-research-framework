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


# ---------------------------------------------------------------------------
# Item #4: cross-asset regime detector lookahead leak harness (HIGH-RISK).
# For every (victim, witness) pair and every T in a 50-point grid, polluting
# `victim`'s data at rows > T must leave `witness`'s regime labels at
# rows <= T bit-identical. Both default detectors are leak-free by
# construction; this test is the safety net for future variants.
# ---------------------------------------------------------------------------
def _pollute_asset_after(panel, asset: str, cut_idx: int):
    """Return a deep copy of `panel` with `asset`'s OHLC values
    replaced by NaN at rows >= cut_idx."""
    ds = panel.ds.copy(deep=True)
    ai = panel.assets.index(asset)
    for field in panel.fields:
        arr = ds[field].values
        arr[cut_idx:, ai] = np.nan
        ds[field].values[...] = arr
    return type(panel)(ds=ds)


def _assert_no_cross_asset_leak(panel, detector_fn, *, n_cut_points: int = 50):
    """50-point pollute battery across all (victim, witness) asset pairs."""
    clean = detector_fn(panel)
    n = len(panel)
    rng = np.random.default_rng(seed=0xCA1F)
    # Sample cut indices across the panel; skip the warmup window where
    # EMA-200 hasn't stabilised and labels are all "Ranging" anyway.
    cuts = sorted(set(int(rng.integers(220, n - 1)) for _ in range(n_cut_points)))
    for cut_idx in cuts:
        for victim in panel.assets:
            polluted = _pollute_asset_after(panel, victim, cut_idx)
            poll_labels = detector_fn(polluted)
            for witness in panel.assets:
                if witness == victim:
                    continue
                clean_head = clean[witness].iloc[:cut_idx].astype(str).reset_index(drop=True)
                poll_head = poll_labels[witness].iloc[:cut_idx].astype(str).reset_index(drop=True)
                assert (clean_head == poll_head).all(), (
                    f"cross-asset leak: polluting {victim!r} at idx >= {cut_idx} "
                    f"changed {witness!r}'s labels in [0, {cut_idx})"
                )


def test_panel_regime_per_asset_no_cross_asset_leak():
    """Default per-asset detector must not couple any pair of assets.
    The HIGH-RISK 50-T pollute battery from the plan."""
    from backtester.panel import (
        detect_regimes_panel_per_asset, load_panel,
    )
    panel = load_panel(PANEL_PATHS)
    _assert_no_cross_asset_leak(panel, detect_regimes_panel_per_asset)


def test_panel_regime_market_no_cross_asset_leak():
    """Market-regime detector (BTC labels broadcast to all) is leak-free
    too: polluting SOL or ETH cannot change BTC's own labels at <= T
    (BTC reads only its own past); polluting BTC at > T cannot change
    BTC's labels at <= T."""
    from backtester.panel import (
        detect_regimes_panel_market, load_panel,
    )
    panel = load_panel(PANEL_PATHS)
    detector = detect_regimes_panel_market("BTC")
    _assert_no_cross_asset_leak(panel, detector)


def test_panel_regime_market_broadcasts_market_asset_labels():
    """All assets inherit the market asset's regime, bit-identical."""
    from backtester.panel import (
        detect_regimes_panel_market, load_panel,
    )
    panel = load_panel(PANEL_PATHS)
    labels = detect_regimes_panel_market("BTC")(panel)
    btc = labels["BTC"]
    for asset in panel.assets:
        if asset == "BTC":
            continue
        assert (labels[asset].values == btc.values).all(), (
            f"market detector did not broadcast BTC labels to {asset}"
        )


def test_cross_asset_leak_harness_catches_known_leak():
    """The 50-T pollute battery must catch a deliberately-leaking
    panel detector. We register a detector that copies the NEXT bar's
    SOL close into BTC's regime — a clear cross-asset future-read —
    and assert ``_assert_no_cross_asset_leak`` raises with the
    offending (victim, witness) pair in the message."""
    from backtester.panel import load_panel
    panel = load_panel(PANEL_PATHS)

    def leaky_detector(panel):
        # BTC's "label" at t reads SOL.close at t+1 — a future leak
        # across assets. Polluting SOL at >T must change BTC at <=T.
        sol_idx = panel.assets.index("SOL")
        sol = panel.ds["close"].values[:, sol_idx]
        sol_next = np.roll(sol, -1)
        sol_next[-1] = sol[-1]
        out = {}
        for a in panel.assets:
            labels = np.where(sol_next > sol, "UP", "DOWN")
            out[a] = pd.Series(labels)
        return out

    raised = False
    try:
        _assert_no_cross_asset_leak(panel, leaky_detector, n_cut_points=5)
    except AssertionError as e:
        raised = True
        msg = str(e)
        assert "cross-asset leak" in msg
        assert "SOL" in msg, "victim asset must appear in the error message"
    assert raised, "deliberate cross-asset leak was NOT caught by the harness"


def test_panel_regime_market_rejects_missing_market_asset():
    """If the named market asset isn't in the panel, fail loud."""
    from backtester.panel import (
        detect_regimes_panel_market, load_panel,
    )
    panel = load_panel(PANEL_PATHS)
    with pytest.raises(ValueError, match="DOGE"):
        detect_regimes_panel_market("DOGE")(panel)


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
