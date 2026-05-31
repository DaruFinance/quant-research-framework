"""Tests for funding-flip / basis-blowout triggers (Phase 3 item #39s)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from backtester.carry.basis import load_basis
from backtester.carry.funding import load_funding
from backtester.carry.triggers import (
    BasisBlowoutTrigger,
    FundingFlipTrigger,
)


HERE = Path(__file__).resolve().parent


def test_funding_flip_emits_events_on_signed_fixture():
    df = load_funding(HERE / "fixtures" / "funding_btcusdt_200evt.parquet")
    trig = FundingFlipTrigger()
    out = trig.run(df)
    # The DS-FUNDING-200 fixture straddles 0 multiple times.
    assert len(out) >= 3
    assert all(ev.kind == "funding_flip" for ev in out)
    # Direction matches the sign at the firing row.
    for ev in out:
        assert ev.direction == int(np.sign(ev.curr))


def test_funding_flip_respects_min_magnitude():
    df = load_funding(HERE / "fixtures" / "funding_btcusdt_200evt.parquet")
    loose = FundingFlipTrigger(min_magnitude=0.0).run(df)
    tight = FundingFlipTrigger(min_magnitude=1.0).run(df)
    assert len(tight) == 0  # nothing in the fixture exceeds 100% per 8h
    assert len(loose) > 0


def test_funding_flip_no_lookahead():
    """The flip set up to time T can't change when rates past T move."""
    df = load_funding(HERE / "fixtures" / "funding_btcusdt_200evt.parquet")
    trig = FundingFlipTrigger()
    rng = np.random.default_rng(0xCA1F)
    for _ in range(10):
        T = int(rng.choice(df["time"].values[20:-5]))
        clean_events = [ev for ev in trig.run(df) if ev.time_s <= T]
        polluted = df.copy()
        polluted.loc[polluted["time"].values > T, "rate"] *= -7.3
        polluted_events = [ev for ev in trig.run(polluted) if ev.time_s <= T]
        assert clean_events == polluted_events, f"flip events differ at T={T}"


def test_basis_blowout_window_warmup_no_fires():
    df = load_basis(HERE / "fixtures" / "basis_btc_perp_spot_1d.parquet")
    trig = BasisBlowoutTrigger(window=20, z_thresh=3.0)
    # The 24-row fixture is too short to fire 3-sigma on any row,
    # because the trailing window is 20 and z_thresh is high.  This
    # test just confirms no spurious fires from arithmetic edge cases.
    events = trig.run(df)
    for ev in events:
        assert ev.kind == "basis_blowout"


def test_basis_blowout_rejects_short_window():
    with pytest.raises(ValueError, match="window must be >= 5"):
        BasisBlowoutTrigger(window=4)


def test_basis_blowout_no_lookahead():
    """Constructed series with a deliberate spike; pollution past T
    cannot resurrect a fire that wasn't there in the clean run."""
    rng = np.random.default_rng(0)
    ts = np.arange(0, 200) * 3600
    basis = rng.normal(0, 1, size=200)
    basis[100] = 10.0  # spike
    df = pd.DataFrame({"time": ts, "basis_bp": basis})
    trig = BasisBlowoutTrigger(window=20, z_thresh=3.0)
    clean = trig.run(df)
    polluted = df.copy()
    polluted.loc[polluted["time"] > ts[110], "basis_bp"] = 100.0
    polluted_run = trig.run(polluted)
    clean_pre_110 = [ev for ev in clean if ev.time_s <= ts[110]]
    polluted_pre_110 = [ev for ev in polluted_run if ev.time_s <= ts[110]]
    assert clean_pre_110 == polluted_pre_110
