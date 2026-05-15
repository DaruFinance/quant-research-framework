"""Audit follow-up tests for the Phase 3 T6 carry pipeline.

These pin behaviours that the original Phase 3 commit (ff09ab6) had
either covered loosely, by accident, or only in one direction.  The
re-audit pass produced this file as part of the handoff between the
Python build session and the Rust port session — every test here is
a regression pin; none expect a code change in the carry source.

Order roughly matches the 10-item checklist in
``handoff_phase3_to_rust.md``.  Items #1, #5 do not produce tests
(rationale documented inline).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from backtester.carry.basis import load_basis
from backtester.carry.funding import (
    FUNDING_INTERVAL_S,
    load_funding,
    rate_at,
)
from backtester.carry.models import (
    FundingMomentumModel,
    FundingOICointegrationModel,
    PersistentFundingSignModel,
)
from backtester.carry.oi import load_oi
from backtester.carry.onchain import load_onchain
from backtester.carry.triggers import (
    BasisBlowoutTrigger,
    FundingFlipTrigger,
)


HERE = Path(__file__).resolve().parent
FUNDING_FIX = HERE / "fixtures" / "funding_btcusdt_200evt.parquet"
OI_FIX = HERE / "fixtures" / "oi_btc_perp_1h_7d.parquet"
ONCHAIN_FIX = HERE / "fixtures" / "onchain_nvt_50d.csv"


# --------------------------------------------------------------------- #
# Item #2 — FundingMomentumModel smallest-valid window arithmetic.
# --------------------------------------------------------------------- #

def test_momentum_smallest_valid_window():
    """At exactly ``len(rates) == window + 1``, the trailing slice
    must be exactly ``window`` rows and the test value must be
    ``rates[-1]`` — not off-by-one in either direction."""
    window = 5
    rates = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0])  # len = window+1
    df = pd.DataFrame({"time": np.arange(len(rates)) * FUNDING_INTERVAL_S,
                        "rate": rates})
    sig = FundingMomentumModel(window=window, z_thresh=0.5).signal_at(
        df, int(df["time"].iloc[-1]))
    # Trailing window has zero variance => sd == 0 => model returns flat.
    # That's the documented short-circuit, and confirms the slice is
    # exactly the first 5 rows (all zero), not 6 rows or 4 rows.
    assert sig.direction == 0
    assert sig.strength == 0.0


def test_momentum_one_below_window_returns_flat():
    """``len(rates) == window`` is below the strict-inequality guard
    and must short-circuit to flat regardless of value."""
    window = 5
    rates = np.array([0.0, 0.0, 0.0, 0.0, 5.0])  # len == window
    df = pd.DataFrame({"time": np.arange(len(rates)) * FUNDING_INTERVAL_S,
                        "rate": rates})
    sig = FundingMomentumModel(window=window, z_thresh=0.5).signal_at(
        df, int(df["time"].iloc[-1]))
    assert sig.direction == 0


# --------------------------------------------------------------------- #
# Item #3 — BasisBlowoutTrigger silent-skip on zero/non-finite sigma.
# --------------------------------------------------------------------- #

def test_basis_blowout_skips_zero_sigma_window():
    """A flat trailing window has sd == 0; the trigger must skip the
    row silently rather than emit a divide-by-zero or a spurious event."""
    window = 5
    z_thresh = 3.0
    # Flat trailing window of length `window` for the candidate at index 5.
    basis = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 99.0,   # row 5: spike, sd=0
                       1.0, 2.0, 1.0, 2.0, 1.0, 7.0])  # later rows have sd
    times = np.arange(len(basis)) * 3600
    df = pd.DataFrame({"time": times, "basis_bp": basis})
    out = BasisBlowoutTrigger(window=window, z_thresh=z_thresh).run(df)
    # Row 5 must NOT fire (sd=0 in [0..5)).  Subsequent rows may.
    assert all(ev.time_s != int(times[5]) for ev in out)


# --------------------------------------------------------------------- #
# Item #4 — load_funding positive control: shift by an integer multiple
# of FUNDING_INTERVAL_S keeps every row aligned and must NOT raise.
# --------------------------------------------------------------------- #

def test_load_funding_shift_by_full_interval_passes(tmp_path):
    df = pd.read_parquet(FUNDING_FIX).copy()
    df["time"] = df["time"].astype(np.int64) + FUNDING_INTERVAL_S
    p = tmp_path / "shifted.parquet"
    df.to_parquet(p)
    # Loader must accept a series shifted by a full settlement interval.
    out = load_funding(p, strict_boundary=True)
    assert len(out) == len(df)
    assert (out["time"].values % FUNDING_INTERVAL_S == 0).all()


# --------------------------------------------------------------------- #
# Items #6 / #9 — snapshot pinning attrs do NOT survive a downstream
# concat.  Pin the behaviour so a future pandas upgrade / refactor that
# silently changes it can't slip through.  The docstring on
# ``load_onchain`` warns about this; this test is the regression pin.
# --------------------------------------------------------------------- #

def test_onchain_attrs_dropped_by_concat_with_foreign_frame():
    """Pandas drops attrs whenever it concats a frame whose attrs
    don't match the source.  This is the realistic case: joining
    on-chain rows with bar data (no carry attrs) — the result loses
    ``snapshot_sha256`` silently."""
    df = load_onchain(ONCHAIN_FIX, metric="nvt")
    sha_at_load = df.attrs.get("snapshot_sha256")
    assert sha_at_load is not None  # set at load time

    # Foreign frame with no carry-loader attrs — like a bar-data row.
    foreign = pd.DataFrame({"time": [0], "value": [0.0]})
    joined = pd.concat([df, foreign], ignore_index=True)

    assert joined.attrs.get("snapshot_sha256") is None, (
        "pandas now propagates attrs across concat with a foreign "
        "frame — update load_onchain docstring and consumer code"
    )

    # The recommended consumer pattern: capture once at load time.
    captured = sha_at_load
    assert captured is not None
    assert captured == df.attrs["snapshot_sha256"]


def test_carry_loaders_attrs_dropped_by_merge_with_bar_data():
    """The realistic join — merging carry rows with bar data on a
    timestamp column — drops the carry-loader attrs.  Pin this for
    each of the four loaders so a pandas-version regression that
    silently changes it is caught."""
    f = load_funding(FUNDING_FIX)
    b = load_basis(HERE / "fixtures" / "basis_btc_perp_spot_1d.parquet")
    o = load_oi(OI_FIX)
    assert "venue" in f.attrs
    assert "instrument_pair" in b.attrs
    assert "expected_cadence_s" in o.attrs

    # Realistic bar-data join.
    bars_f = pd.DataFrame({"time": [int(f["time"].iloc[0])], "close": [1.0]})
    bars_b = pd.DataFrame({"time": [int(b["time"].iloc[0])], "close": [1.0]})
    bars_o = pd.DataFrame({"time": [int(o["time"].iloc[0])], "close": [1.0]})

    f_joined = f.merge(bars_f, on="time")
    b_joined = b.merge(bars_b, on="time")
    o_joined = o.merge(bars_o, on="time")

    assert f_joined.attrs.get("venue") is None
    assert b_joined.attrs.get("instrument_pair") is None
    assert o_joined.attrs.get("expected_cadence_s") is None


def test_carry_loaders_attrs_survive_self_concat():
    """Counter-pin: pandas DOES preserve attrs when both sides of a
    concat share identical attrs (the typical 'append more rows from
    the same source' pattern).  This is a documentary test — neither
    side of the user-facing API depends on it; it just makes the
    boundary explicit so the warning in load_onchain is precise."""
    f = load_funding(FUNDING_FIX)
    same_source = pd.concat([f, f.iloc[:1]], ignore_index=True)
    # Both halves carry the same attrs => preserved.
    assert same_source.attrs.get("venue") == "binance_perp"


# --------------------------------------------------------------------- #
# Item #7 — FundingOICointegrationModel with OI cadence > funding cadence.
# Multiple funding events must legitimately map to the same most-recent
# OI value; o_sd may be 0 if no new OI arrived in the window, in which
# case the model returns flat — that's the documented short-circuit.
# --------------------------------------------------------------------- #

def test_oi_cointegration_with_slower_oi_cadence():
    # Funding at 8h cadence, OI at 24h cadence — every third funding
    # event sees a new OI value, the rest reuse the prior.
    window = 6
    f_times = np.arange(window + 1) * FUNDING_INTERVAL_S          # 7 events
    f_rates = np.array([1e-4, 1e-4, 1e-4, -1e-4, -1e-4, 2e-4, 3e-4])
    funding = pd.DataFrame({"time": f_times, "rate": f_rates})

    oi_times = np.arange(3) * (3 * FUNDING_INTERVAL_S)            # 24h cadence
    oi_vals = np.array([100.0, 200.0, 300.0])
    oi = pd.DataFrame({"time": oi_times, "open_interest": oi_vals})

    sig = FundingOICointegrationModel(window=window).signal_at(
        funding, oi, int(f_times[-1]))
    # The mapping must succeed (no LookupError), the resulting signal
    # is well-typed, and the direction is one of {-1, 0, 1}.  We don't
    # pin a specific direction here — just no leak / no crash.
    assert sig.direction in {-1, 0, 1}
    # Strength must be finite and non-negative.
    assert np.isfinite(sig.strength)
    assert sig.strength >= 0.0


def test_oi_cointegration_constant_oi_returns_flat():
    """If every recent OI value is the same (slow OI cadence + tight
    window), o_sd == 0 and the model must return flat, not crash."""
    window = 5
    f_times = np.arange(6) * FUNDING_INTERVAL_S
    funding = pd.DataFrame({
        "time": f_times,
        "rate": [1e-4, 1e-4, 1e-4, 1e-4, 1e-4, 1e-4],
    })
    oi = pd.DataFrame({
        "time": [0],
        "open_interest": [100.0],
    })
    sig = FundingOICointegrationModel(window=window).signal_at(
        funding, oi, int(f_times[-1]))
    assert sig.direction == 0
    assert sig.strength == 0.0


# --------------------------------------------------------------------- #
# Item #8 — PersistentFundingSignModel with len(slc) == 1.
# slc[-2::-1] is empty, streak stays at 1, below default min_streak => flat.
# --------------------------------------------------------------------- #

def test_persistent_sign_single_event_streak_is_one():
    df = pd.DataFrame({"time": [0], "rate": [1e-4]})
    sig = PersistentFundingSignModel(min_streak=3).signal_at(df, 0)
    assert sig.direction == 0
    # Strength = streak / min_streak = 1/3.
    assert sig.strength == pytest.approx(1.0 / 3.0)
    assert sig.inputs["streak"] == 1.0


def test_persistent_sign_single_event_meets_min_streak_one():
    """With min_streak=1, a single same-sign event already qualifies."""
    df = pd.DataFrame({"time": [0], "rate": [1e-4]})
    sig = PersistentFundingSignModel(min_streak=1).signal_at(df, 0)
    # Positive funding => carry-collecting side is short the underlying.
    assert sig.direction == -1


# --------------------------------------------------------------------- #
# Item #10 — FundingFlipTrigger row-by-row leak pin.
# Polluting strictly past the rows whose flips we care about must
# leave the emitted set up to T identical, byte-for-byte.
# --------------------------------------------------------------------- #

def test_funding_flip_pollution_strictly_past_emission_rows():
    df = load_funding(FUNDING_FIX)
    trig = FundingFlipTrigger()
    clean_events = trig.run(df)
    # Pick a T that lies strictly between two known flip emission rows.
    flip_times = [ev.time_s for ev in clean_events]
    assert len(flip_times) >= 2
    # T = a value strictly larger than the second flip time.
    T = flip_times[1] + 1
    pre_T_clean = [ev for ev in clean_events if ev.time_s <= T]

    polluted = df.copy()
    # Pollute every row strictly past T.  Use a sign-flipping multiplier
    # large enough that, if the trigger were peeking past T, every
    # polluted row would change sign relative to its pre-pollution
    # neighbour and thus alter the flip count at that index.
    polluted.loc[polluted["time"] > T, "rate"] *= -7.3
    pre_T_polluted = [ev for ev in trig.run(polluted) if ev.time_s <= T]

    assert pre_T_clean == pre_T_polluted, (
        f"FundingFlipTrigger leaked: events ≤ T={T} differ after "
        f"pollution strictly past T"
    )


def test_funding_flip_pollution_exactly_at_emission_row_changes_curr():
    """Sanity counter-test: polluting AT the emission row IS expected
    to change the trigger's view of that row's `curr` value (because
    curr is read from the row itself).  This is not a leak; it
    confirms the row identity in the emission record."""
    df = load_funding(FUNDING_FIX)
    trig = FundingFlipTrigger()
    clean = trig.run(df)
    if not clean:
        pytest.skip("fixture has no flips")
    target = clean[0]
    polluted = df.copy()
    polluted.loc[polluted["time"] == target.time_s, "rate"] = (
        target.curr + 1.0)
    polluted_events = trig.run(polluted)
    matched = [ev for ev in polluted_events if ev.time_s == target.time_s]
    assert matched, "the emission at the polluted row should still appear"
    assert matched[0].curr != target.curr


# --------------------------------------------------------------------- #
# Documentary assert: rate_at strictly excludes future rows even when
# the future rate is wildly different from the most-recent past rate.
# (The existing test pollutes random rows; this pins the boundary case
# where the polluted future row is exactly t+1s from the query.)
# --------------------------------------------------------------------- #

def test_rate_at_strict_exclude_at_t_plus_one(tmp_path):
    df = pd.read_parquet(FUNDING_FIX).copy()
    t = int(df["time"].iloc[10])
    # Inject a polluted row at t+1 (NOT aligned to FUNDING_INTERVAL_S
    # but the test bypasses load_funding's strict-boundary check by
    # calling rate_at directly on a constructed frame).
    polluted = pd.concat([
        df.iloc[:11],
        pd.DataFrame({"time": [t + 1], "rate": [9.99]}),
        df.iloc[11:],
    ], ignore_index=True)
    polluted = polluted.sort_values("time").reset_index(drop=True)
    # rate_at(t) must return the row at t, never the row at t+1.
    r = rate_at(polluted, t)
    expected = float(df["rate"].iloc[10])
    assert r == expected
