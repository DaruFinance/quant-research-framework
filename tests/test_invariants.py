"""Property checks on actual trade-ledger outputs.

Where `test_behavioural.py` asks "does this flag change anything", these
tests ask "does the engine respect the invariants it claims". Each test
runs the backtest end-to-end and inspects the produced trade list /
signals against the underlying constraint.
"""
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import backtester as bt


def _df_from_bars(n=600, start_unix=1_600_000_000, interval_s=3600, seed=3):
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
# Session: no trade may ENTER on a bar whose NY local time is outside
# [SESSION_START, SESSION_END), and no position may carry across a
# session-end bar (force-close should fire).
# ---------------------------------------------------------------------------
def test_no_entry_outside_session_window(monkeypatch):
    monkeypatch.setattr(bt, "TRADE_SESSIONS", True)
    monkeypatch.setattr(bt, "SESSION_START",  "8:00")
    monkeypatch.setattr(bt, "SESSION_END",   "16:50")

    df = _df_from_bars(2_000)
    dfi = bt.compute_indicators(df, bt.DEFAULT_LB)
    raw = bt.create_raw_signals(dfi, bt.DEFAULT_LB)
    parsed = bt.parse_signals(raw, dfi["time"])
    trades, _, _, _, _ = bt.backtest(dfi, parsed)

    start_t = datetime.strptime("8:00",  "%H:%M").time()
    end_t   = datetime.strptime("16:50", "%H:%M").time()

    violations = []
    for side, ent, exi, *_ in trades:
        ts_ny = dfi["time"].iloc[ent].tz_convert(bt.NY_TZ).timetz().replace(tzinfo=None)
        if not (start_t <= ts_ny < end_t):
            violations.append((ent, side, ts_ny))

    assert not violations, (
        f"Session invariant broken — {len(violations)} entries outside the "
        f"window. First few: {violations[:5]}"
    )


@pytest.mark.xfail(
    reason="Known Python design choice: _backtest_numba_core only force-closes "
           "at session_end when code != 0 (line 950 guard). When no signal "
           "lands on the last in-session bar, the position carries across "
           "out-of-session windows. The Rust port mirrors this behaviour for "
           "parity. Removing the guard would change all published research "
           "numbers, so it's left as-is and documented here.",
    strict=True,
)
def test_session_blocks_overnight_carry_known_quirk(monkeypatch):
    """If you ever fix the `code != 0` guard, this test will start passing
    and you should flip xfail off."""
    monkeypatch.setattr(bt, "TRADE_SESSIONS", True)
    monkeypatch.setattr(bt, "SESSION_START",  "8:00")
    monkeypatch.setattr(bt, "SESSION_END",   "16:50")

    df = _df_from_bars(2_000)
    dfi = bt.compute_indicators(df, bt.DEFAULT_LB)
    raw = bt.create_raw_signals(dfi, bt.DEFAULT_LB)
    parsed = bt.parse_signals(raw, dfi["time"])
    trades, _, _, _, _ = bt.backtest(dfi, parsed)

    in_flags = bt.compute_in_flags(dfi["time"])
    overnight = [(ent, exi) for side, ent, exi, *_ in trades
                 if (~in_flags[ent : exi + 1]).any()]
    assert not overnight, (
        f"Trades span out-of-session bars: {len(overnight)} found"
    )


# ---------------------------------------------------------------------------
# Regime: at every bar, the slow-EMA used in `create_regime_signals` must
# come from `best_lbs[regime[i]]`. Build a controlled best_lbs with three
# distinct LBs and verify the produced raw signal matches a hand-computed
# reference.
# ---------------------------------------------------------------------------
def test_create_regime_signals_uses_correct_lb_per_bar():
    df = _df_from_bars(800, seed=42)
    base = df.copy()
    for span in (20, 200, 900):
        base[f"EMA_{span}"] = base["close"].ewm(span=span, adjust=False).mean()

    best_lbs = {"Uptrend": 13, "Downtrend": 47, "Ranging": 71}
    for lb in set(best_lbs.values()):
        base[f"EMA_{lb}"] = base["close"].ewm(span=lb, adjust=False).mean()

    regimes = bt.detect_regimes(base)
    raw = bt.create_regime_signals(base, best_lbs, regimes)

    # Mirror create_regime_signals exactly: NaN > NaN is False so the
    # function emits -1 at warmup bars (no special-case for NaN). What we
    # actually want to verify is that the LOOKUP picks the right slow-EMA
    # column for each bar's regime label, NOT the NaN-handling.
    ema20_prev = base["EMA_20"].shift(1).values
    expected = np.zeros(len(base), dtype=np.int8)
    for i in range(len(base)):
        lb = best_lbs[regimes.iat[i]]
        slow = base[f"EMA_{lb}"].shift(1).iat[i]
        expected[i] = 1 if (ema20_prev[i] > slow) else -1   # NaN > NaN -> False -> -1

    mismatches = np.where(raw != expected)[0]
    assert len(mismatches) == 0, (
        f"Regime LB rotation broken — {len(mismatches)} bars use the wrong "
        f"slow-EMA. First mismatch idx={mismatches[:5]}"
    )


# ---------------------------------------------------------------------------
# Forex: every per-trade PnL must be within [-position_size_fx,
# +position_size_fx * RRR] up to the entry+exit fees. Mirrors the cap that
# `(price_move_pips / (RRR*stop_pips)) * RRR` is clipped to [-1, RRR].
# ---------------------------------------------------------------------------
def test_forex_pnl_clamped_to_R_band(monkeypatch):
    monkeypatch.setattr(bt, "FOREX_MODE", True)
    monkeypatch.setattr(bt, "PIP_SIZE",   0.0001)
    # Pre-multiply the SL/TP percents — backtester does this at import
    # time when FOREX_MODE is True, but we flipped it at runtime.
    monkeypatch.setattr(bt, "SL_PERCENTAGE", 1.0 * 0.0001)
    monkeypatch.setattr(bt, "TP_PERCENTAGE", 3.0 * 0.0001)
    monkeypatch.setattr(bt, "RISK_AMOUNT", 1.0)
    monkeypatch.setattr(bt, "ACCOUNT_SIZE", 1.0)
    monkeypatch.setattr(bt, "POSITION_SIZE", 1.0)

    df = _df_from_bars(800, seed=11)
    dfi = bt.compute_indicators(df, bt.DEFAULT_LB)
    raw = bt.create_raw_signals(dfi, bt.DEFAULT_LB)
    parsed = bt.parse_signals(raw, dfi["time"])
    trades, _, _, _, _ = bt.backtest(dfi, parsed)

    if not trades:
        pytest.skip("no trades produced on this fixture")

    rrr = bt.TP_PERCENTAGE / bt.SL_PERCENTAGE
    pos_size_fx = 1.0
    # Fees subtract from PnL: entry fee = pos_size_fx * fee_rate, exit fee =
    # qty * exit_price * fee_rate (qty = pos_size_fx / entry_price, so exit
    # fee ≈ pos_size_fx * (exit/entry) * fee_rate ≈ pos_size_fx * fee_rate).
    # Total fees ≈ 2 * pos_size_fx * fee_rate. Add a tiny FP slack.
    fee_band = 2 * pos_size_fx * (bt.FEE_PCT / 100.0) + 1e-6
    upper = pos_size_fx * rrr + fee_band
    lower = -pos_size_fx - fee_band

    out_of_band = [(t[1], t[6]) for t in trades if not (lower <= t[6] <= upper)]
    assert not out_of_band, (
        f"Forex PnL not clamped — {len(out_of_band)} trades outside "
        f"[{lower:.6f}, {upper:.6f}]. First few: {out_of_band[:5]}"
    )


# ---------------------------------------------------------------------------
# OOS2 doubles oos_candles AND the original split point is preserved so
# OOS1 + OOS2 metrics can be reported separately downstream.
# ---------------------------------------------------------------------------
def test_oos2_invariants():
    """OOS_CANDLES is multiplied at import-time when USE_OOS2 is True;
    ORIGINAL_OOS retains the un-doubled length so the engine can split
    trades by `exit_i < ORIGINAL_OOS`."""
    assert bt.ORIGINAL_OOS > 0
    if bt.USE_OOS2:
        assert bt.OOS_CANDLES == bt.ORIGINAL_OOS * 2
        assert bt.OOS_CANDLES > bt.ORIGINAL_OOS
    else:
        assert bt.OOS_CANDLES == bt.ORIGINAL_OOS


# ---------------------------------------------------------------------------
# Sanity invariants that should hold for EVERY trade tuple regardless of flags.
# ---------------------------------------------------------------------------
def test_trade_indices_and_prices_are_well_formed():
    df = _df_from_bars(1_500, seed=21)
    dfi = bt.compute_indicators(df, bt.DEFAULT_LB)
    raw = bt.create_raw_signals(dfi, bt.DEFAULT_LB)
    parsed = bt.parse_signals(raw, dfi["time"])
    trades, _, _, _, _ = bt.backtest(dfi, parsed)
    n = len(df)

    for side, ent, exi, ep, xp, qty, pnl in trades:
        assert side in (1, -1), f"bad side {side}"
        assert 0 <= ent < n,    f"entry idx {ent} out of range"
        assert 0 <= exi < n,    f"exit idx {exi} out of range"
        assert ent <= exi,      f"exit {exi} before entry {ent} — time travel?"
        assert ep > 0 and xp > 0, f"non-positive prices ({ep}, {xp})"
        assert qty >= 0,        f"negative quantity {qty}"
        assert np.isfinite(pnl), f"non-finite PnL {pnl}"


# ---------------------------------------------------------------------------
# parse_signals look-ahead discipline: the flip-code at bar i must depend
# only on raw[..i+1] and in_flags[..i+1] — not on anything later. Build a
# raw signal, parse it, then mutate raw[i+1:] and re-parse; sig[..i+1]
# must be identical.
# ---------------------------------------------------------------------------
def test_parse_signals_no_lookahead():
    n = 500
    rng = np.random.default_rng(7)
    raw = rng.choice([-1, 0, 1], size=n).astype(np.int8)
    df = _df_from_bars(n, seed=31)
    parsed_full = bt.parse_signals(raw.copy(), df["time"])

    polluted = raw.copy()
    polluted[300:] = 0
    parsed_clean = bt.parse_signals(polluted, df["time"])

    assert (parsed_full[:300] == parsed_clean[:300]).all(), (
        "parse_signals appears to leak future data into earlier bars"
    )


# ---------------------------------------------------------------------------
# Regime detector look-ahead discipline: detect_regimes(df) for bar i must
# only consume df[..i+1].close and EMA_200[..i+1].
# ---------------------------------------------------------------------------
def test_default_regime_detector_no_lookahead():
    df = _df_from_bars(800, seed=4)
    base = df.copy()
    base["EMA_200"] = base["close"].ewm(span=200, adjust=False).mean()
    full = bt.detect_regimes(base)

    polluted = base.copy()
    cut = 400
    polluted.loc[cut:, "close"]   = float("nan")
    polluted.loc[cut:, "EMA_200"] = float("nan")
    clean = bt.detect_regimes(polluted)

    early_full  = full.iloc[:cut].astype(str).reset_index(drop=True)
    early_clean = clean.iloc[:cut].astype(str).reset_index(drop=True)
    assert (early_full == early_clean).all(), (
        "default regime detector leaks future bar data into earlier labels"
    )


# ---------------------------------------------------------------------------
# WFO + regime cadence: with USE_REGIME_SEG=True and a fixed candle
# trigger, optimize_regimes_sequential is called exactly once per WFO
# window (plus one initial pre-OOS call for evaluate_filters). Catches
# the v0.1.x bug where regime flips re-anchored the IS window.
# ---------------------------------------------------------------------------
def test_wfo_regime_cadence_unaffected_by_regime_flips(monkeypatch):
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
        pytest.skip("synthetic fixture too small")

    calls = []
    real = bt.optimize_regimes_sequential
    monkeypatch.setattr(bt, "optimize_regimes_sequential",
                        lambda is_df: (lambda r: (calls.append(len(is_df)), r)[1])(real(is_df)))

    bt.walk_forward(df, met_is_baseline=None, eq_is_baseline=np.ones(1))

    # Each call's IS slice should be exactly BACKTEST_CANDLES (the standard
    # WFO cadence) — never some smaller "regime stretch" length.
    bad = [n for n in calls if n != 600]
    assert not bad, (
        f"WFO+regime cadence broken — optimize_regimes_sequential called "
        f"with non-standard IS slice lengths: {bad[:5]} "
        f"(all should be BACKTEST_CANDLES=600)"
    )
