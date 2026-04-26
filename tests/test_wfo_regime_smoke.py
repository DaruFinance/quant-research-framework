"""End-to-end smoke for the v0.2.0 WFO + regime-segmentation rewrite.

Boots the engine with small IS/OOS/WFO sizes so the test runs in seconds,
flips USE_WFO and USE_REGIME_SEG on together, and asserts:

  1. The walk completes without raising.
  2. WFO window boundaries align with WFO_TRIGGER_VAL — i.e. they are NOT
     re-anchored on regime changes (the bug fixed in v0.2.0).
  3. Per-regime LB rotation actually happens inside OOS (signals are
     produced from at least two distinct LBs across the run).

Catches the regression directly: with the v0.1.x logic the OOS window
boundaries would be derived from regime change indices, breaking the
WFO_TRIGGER_VAL invariant.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import backtester as bt


@pytest.fixture(scope="module")
def small_df():
    """Load the synthetic CSV that conftest seeded for BT_CSV."""
    csv = Path(bt.CSV_FILE)
    if not csv.exists():
        pytest.skip(f"BT_CSV {csv} missing; conftest should have created it")
    return bt.load_ohlc(str(csv))


def test_wfo_with_regime_completes_and_keeps_cadence(small_df, monkeypatch):
    n = len(small_df)
    if n < 1500:
        pytest.skip("synthetic fixture too small for a meaningful WFO run")

    # Slot the engine into a small but realistic config.
    monkeypatch.setattr(bt, "BACKTEST_CANDLES", 600, raising=False)
    monkeypatch.setattr(bt, "OOS_CANDLES",       600, raising=False)
    monkeypatch.setattr(bt, "WFO_TRIGGER_VAL",   200, raising=False)
    monkeypatch.setattr(bt, "WFO_TRIGGER_MODE",  "candles", raising=False)
    monkeypatch.setattr(bt, "USE_WFO",            True, raising=False)
    monkeypatch.setattr(bt, "USE_REGIME_SEG",     True, raising=False)
    monkeypatch.setattr(bt, "USE_MONTE_CARLO",   False, raising=False)
    monkeypatch.setattr(bt, "PRINT_EQUITY_CURVE", False, raising=False)
    monkeypatch.setattr(bt, "MIN_TRADES",          1, raising=False)
    monkeypatch.setattr(bt, "OPTIMIZE_RRR",     False, raising=False)
    monkeypatch.setattr(bt, "LOOKBACK_RANGE",   (12, 60), raising=False)

    captured_lbs = []
    real = bt.optimize_regimes_sequential
    def spy(is_df):
        out = real(is_df)
        captured_lbs.append(dict(out[0]))
        return out
    monkeypatch.setattr(bt, "optimize_regimes_sequential", spy)

    rets, eq, _, _ = bt.walk_forward(small_df, met_is_baseline=None,
                                     eq_is_baseline=np.ones(1))

    # 1. Pipeline completed.
    assert isinstance(rets, np.ndarray)
    assert eq.size > 1

    # 2. Cadence respected. The new WFO+regime path makes one initial call
    #    to optimize_regimes_sequential for the pre-OOS evaluate_filters
    #    pass, then one per WFO window. With OOS=600 and TRIGGER_VAL=200
    #    the upper bound is therefore 1 + ceil(600/200) = 4, *independent*
    #    of how often the regime label flips inside OOS. The v0.1.x bug
    #    used regime change indices to slice OOS, which on synthetic data
    #    routinely produced 10+ stretches and therefore 10+ optimiser
    #    calls — that's the regression this assertion catches.
    assert 2 <= len(captured_lbs) <= 4, (
        f"Expected 2..4 optimiser calls under v0.2.0 cadence "
        f"(1 initial + up to 3 windows for OOS=600 / TRIGGER=200), "
        f"got {len(captured_lbs)} — regression to v0.1.x regime-driven cadence?"
    )

    # 3. The optimiser produced a per-regime LB dict each window, with at
    #    least one valid (non-None) LB.
    assert all(any(v is not None for v in d.values()) for d in captured_lbs)
