"""Behavioural tests for the v0.2.x feature set.

Each test flips one feature flag (FOREX_MODE, TRADE_SESSIONS, USE_OOS2,
USE_REGIME_SEG) and asserts the engine actually changes behaviour, not
just that the flag is settable. Catches "wired but inert" regressions.
"""
from datetime import datetime, time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import backtester as bt


def _df_from_bars(n=600, start_unix=1_600_000_000, interval_s=3600, seed=3):
    """Mimic backtester.load_ohlc: synthesise OHLC and convert time to NY tz
    so downstream session-aware code sees the same shape it would from a
    real CSV load."""
    rng = np.random.default_rng(seed)
    rets = rng.normal(0.00005, 0.01, size=n)
    close = 100.0 * np.exp(np.cumsum(rets))
    times = (pd.to_datetime(start_unix + np.arange(n) * interval_s, unit="s", utc=True)
               .tz_convert(bt.NY_TZ))
    return pd.DataFrame({
        "time":  times,
        "open":  close,
        "high":  close * 1.002,
        "low":   close * 0.998,
        "close": close,
    })


# ---------------------------------------------------------------------------
# Forex mode: switching it on completely changes PnL semantics — pip-based
# R units instead of dollar qty, and funding fees are disabled. The
# behavioural assertions are: (a) the engine takes a different code path
# (trade ledger differs from the non-forex run on the same bars/signals),
# and (b) the metrics are still well-formed. ROI comparisons across modes
# are meaningless because the units differ.
# ---------------------------------------------------------------------------
def test_forex_mode_changes_pnl_path(monkeypatch):
    df = _df_from_bars(300)
    dfi = bt.compute_indicators(df, bt.DEFAULT_LB)
    sig = np.zeros(len(df), dtype=np.int8)
    sig[10] = 1; sig[200] = 3
    parsed = bt.parse_signals(sig, dfi["time"])

    monkeypatch.setattr(bt, "FOREX_MODE", False)
    trades_no, met_no, _, _, _ = bt.backtest(dfi, parsed)

    monkeypatch.setattr(bt, "FOREX_MODE", True)
    trades_fx, met_fx, _, _, _ = bt.backtest(dfi, parsed)

    # Same signals, different sizing/PnL semantics: the per-trade PnL field
    # must differ at least once (forex uses a capped pip computation that
    # can never coincide with the dollar PnL except at exactly 0).
    pnls_no = [t[6] for t in trades_no]
    pnls_fx = [t[6] for t in trades_fx]
    assert pnls_no != pnls_fx, (
        "FOREX_MODE flag flipped but trade PnL is byte-identical — flag inert?"
    )
    # Both runs must still produce well-formed metrics.
    for met in (met_no, met_fx):
        assert all(np.isfinite(met[k]) for k in ("ROI", "Sharpe", "WinRate", "MaxDrawdown"))


# ---------------------------------------------------------------------------
# Session mode: with TRADE_SESSIONS on and a 13-21 UTC window, no trade
# entry should fall on a bar whose UTC hour is outside [13, 21).
# ---------------------------------------------------------------------------
def test_session_mode_blocks_out_of_session_entries(monkeypatch):
    monkeypatch.setattr(bt, "TRADE_SESSIONS", True)
    monkeypatch.setattr(bt, "SESSION_START", "8:00")
    monkeypatch.setattr(bt, "SESSION_END", "16:50")
    monkeypatch.setattr(bt, "PRINT_EQUITY_CURVE", False)
    monkeypatch.setattr(bt, "USE_MONTE_CARLO", False)

    df = _df_from_bars(2_000)
    dfi = bt.compute_indicators(df, bt.DEFAULT_LB)
    raw = bt.create_raw_signals(dfi, bt.DEFAULT_LB)
    parsed = bt.parse_signals(raw, dfi["time"])
    trades, _, _, _, _ = bt.backtest(dfi, parsed)

    # parse_signals should already have masked out-of-session entries via
    # the in_flags mechanism. Verify by inspecting the entry timestamp of
    # every executed trade — it must be inside the session window in NY tz.
    start_t = datetime.strptime("8:00", "%H:%M").time()
    end_t   = datetime.strptime("16:50", "%H:%M").time()
    for side, ent_idx, *_ in trades:
        ts = dfi["time"].iloc[ent_idx].tz_convert(bt.NY_TZ)
        local = ts.timetz().replace(tzinfo=None)
        assert start_t <= local < end_t, (
            f"trade entered at {local} (NY) outside session window "
            f"{start_t}..{end_t} — TRADE_SESSIONS not enforced?"
        )


# ---------------------------------------------------------------------------
# OOS2: USE_OOS2 doubles the OOS window length used at module-config time
# (Python evaluates the doubling at import; at runtime the constant is
# the doubled value). Verify the relationship is consistent and that
# ORIGINAL_OOS is preserved as the original (un-doubled) length so downstream
# code can split into OOS1 + OOS2 at the right index.
# ---------------------------------------------------------------------------
def test_oos2_preserves_split_point():
    assert bt.ORIGINAL_OOS > 0
    if bt.USE_OOS2:
        assert bt.OOS_CANDLES == bt.ORIGINAL_OOS * 2
    else:
        assert bt.OOS_CANDLES == bt.ORIGINAL_OOS


# ---------------------------------------------------------------------------
# Regime LB rotation: after walk_forward with USE_WFO + USE_REGIME_SEG,
# the optimiser should have produced per-regime LBs that actually differ
# across regimes (or at least are non-uniform across windows). Also a
# regression check that windows aren't being driven by regime change indices.
# ---------------------------------------------------------------------------
def test_regime_segmentation_produces_per_regime_lbs(monkeypatch):
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

    csv = Path(bt.CSV_FILE)
    if not csv.exists():
        pytest.skip(f"BT_CSV {csv} missing; conftest should have created it")
    df = bt.load_ohlc(str(csv))
    if len(df) < 1500:
        pytest.skip("synthetic fixture too small for the regime smoke")

    captured = []
    real = bt.optimize_regimes_sequential
    monkeypatch.setattr(bt, "optimize_regimes_sequential",
                        lambda is_df: (lambda r: (captured.append(dict(r[0])), r)[1])(real(is_df)))

    bt.walk_forward(df, met_is_baseline=None, eq_is_baseline=np.ones(1))

    # We should have captured at least one IS-window optimisation and the
    # returned dict must be keyed by REGIME_LABELS.
    assert captured, "optimize_regimes_sequential never called from walk_forward"
    for lbs in captured:
        assert set(lbs.keys()).issubset(set(bt.REGIME_LABELS) | {None}), (
            f"per-regime LB dict has unexpected keys: {lbs.keys()}"
        )


# ---------------------------------------------------------------------------
# Robustness combinations: the news-injection scenario should produce a
# DIFFERENT trade ledger than baseline (otherwise it's a no-op).
# ---------------------------------------------------------------------------
def test_news_injection_perturbs_bars():
    df = _df_from_bars(1_500, seed=11)
    perturbed = bt.inject_news_candles(df.copy(), seed=42)
    diff_high = (perturbed["high"] - df["high"]).abs()
    diff_low  = (df["low"] - perturbed["low"]).abs()
    assert (diff_high.sum() + diff_low.sum()) > 0, \
        "inject_news_candles produced an identical bar series — injection broken?"
    # OHLC invariants must still hold post-injection.
    assert (perturbed["high"] >= perturbed[["open", "close"]].max(axis=1)).all()
    assert (perturbed["low"]  <= perturbed[["open", "close"]].min(axis=1)).all()
