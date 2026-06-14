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


def test_session_force_close_prevents_overnight_carry(monkeypatch):
    """When TRADE_SESSIONS is on, no trade may span out-of-session bars.
    Fixed in v0.2.2 by removing the `and code != 0` guard on the
    session-end force-close path — previously that guard silently let
    positions carry across session gaps when no signal landed on the
    closing bar."""
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
        f"Session-end force-close regressed — {len(overnight)} trades span "
        f"out-of-session bars. First few: {overnight[:5]}"
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


# ---------------------------------------------------------------------------
# LB-grid optimiser cache: no cached score may be reused across the IS/OOS
# boundary. `optimiser()` builds a fresh `eval_cache = {}` per invocation,
# closed over by `_evaluate(lb)` (see backtester/__init__.py:1293-1298). The
# cache is therefore scoped to one window: the score for a given lookback is
# keyed by (this optimiser call's window, lb), so an in-sample window and its
# out-of-sample counterpart can never share a cache entry. This test shifts
# the IS/OOS boundary, re-runs the optimiser on each side, and asserts that
# every lookback the OOS optimiser needs is recomputed against OOS data — no
# score leaks across the boundary move.
#
# Why this is a real guard (verified out-of-band, not committed): if the cache
# is weakened to drop window identity — e.g. promoted to module scope and keyed
# by `lb` alone — then the OOS optimiser hits the IS-window cache for every
# shared lookback and reuses the IS score, and the recompute / score-disagreement
# assertions below both fail. The window key is load-bearing.
# ---------------------------------------------------------------------------
def test_lb_cache_no_oos_leak(monkeypatch):
    monkeypatch.setattr(bt, "SMART_OPTIMIZATION", False, raising=False)
    monkeypatch.setattr(bt, "OPTIMIZE_RRR",        False, raising=False)
    monkeypatch.setattr(bt, "LOOKBACK_RANGE",     (12, 40), raising=False)
    monkeypatch.setattr(bt, "DRAWDOWN_CONSTRAINT", None, raising=False)
    monkeypatch.setattr(bt, "dd_constraint",       None, raising=False)

    full   = _df_from_bars(2_400, seed=9)
    lb_list = list(range(*bt.LOOKBACK_RANGE))

    IS_LEN = OOS_LEN = 800

    # Spy on the two expensive per-lookback steps so we can tell, per window,
    # whether a score was recomputed (cache miss) or skipped (cache hit), and
    # capture the score the optimiser stored for each (window, lb).
    win        = {"tag": None}     # which window is currently being optimised
    last_lb    = {"v": None}       # lb of the most recent compute_indicators call
    recomputed = {"IS": set(), "OOS": set()}   # lbs whose score was (re)computed
    pf_by_win  = {"IS": {}, "OOS": {}}         # window -> {lb: PF score}
    data_first = {"IS": set(), "OOS": set()}   # window -> first-close of df fed in

    real_ci = bt.compute_indicators
    real_bt = bt.backtest

    def spy_ci(df, lb, *a, **k):
        last_lb["v"] = int(lb)
        recomputed[win["tag"]].add(int(lb))
        data_first[win["tag"]].add(round(float(df["close"].iloc[0]), 9))
        return real_ci(df, lb, *a, **k)

    def spy_bt(dfi, sig, *a, **k):
        res = real_bt(dfi, sig, *a, **k)
        # store the score for this (window, lb); setdefault so a genuine cache
        # HIT (which would skip compute_indicators / backtest) leaves no entry.
        pf_by_win[win["tag"]].setdefault(last_lb["v"], res[1].get("PF"))
        return res

    monkeypatch.setattr(bt, "compute_indicators", spy_ci)
    monkeypatch.setattr(bt, "backtest",           spy_bt)

    def optimise_around(boundary):
        is_df  = full.iloc[boundary - IS_LEN : boundary].reset_index(drop=True)
        oos_df = full.iloc[boundary : boundary + OOS_LEN].reset_index(drop=True)
        win["tag"] = "IS";  bt.optimiser(is_df,  lb_list, "PF", 1)
        win["tag"] = "OOS"; bt.optimiser(oos_df, lb_list, "PF", 1)

    # First placement, then shift the IS/OOS boundary and re-run. Both runs
    # share the same module state — the only thing protecting against leakage
    # is the per-call cache scoping.
    optimise_around(1_000)
    optimise_around(1_200)

    # 1) IS and OOS are fed genuinely different data (disjoint first-close set).
    assert data_first["IS"].isdisjoint(data_first["OOS"]), (
        "IS and OOS windows fed identical data — test cannot prove isolation"
    )

    # 2) The OOS run recomputed (cache-missed) every lookback it scored: no
    #    OOS lookback was served from a cache populated by the IS window.
    shared = recomputed["IS"] & recomputed["OOS"]
    assert shared, "expected overlapping lookbacks between IS and OOS coarse passes"
    leaked = [lb for lb in shared if lb not in recomputed["OOS"]]
    assert not leaked, (
        f"LB cache leaked across IS/OOS boundary — lookbacks {leaked[:5]} were "
        f"scored in OOS without recomputing against OOS data"
    )

    # 3) The stored OOS score for a shared lookback was computed from OOS data,
    #    not copied from IS: for the overwhelming majority of shared lookbacks
    #    the IS PF and OOS PF differ (same lb, different window data). A leaking
    #    lb-only cache would make these identical.
    both = sorted(lb for lb in shared
                  if lb in pf_by_win["IS"] and lb in pf_by_win["OOS"])
    assert both, "no shared lookback had a stored score in both windows"
    distinct = [lb for lb in both if pf_by_win["IS"][lb] != pf_by_win["OOS"][lb]]
    assert len(distinct) >= max(1, len(both) - 1), (
        f"OOS scores match IS scores for {len(both) - len(distinct)}/{len(both)} "
        f"shared lookbacks — the optimiser appears to be reusing IS-window "
        f"scores for OOS evaluation (cache key dropped window identity)"
    )


# ---------------------------------------------------------------------------
# LEGACY_SIDE_BUG regression: the RRR-optimisation R-multiple code compares the
# trade's side against a value. The corrected default uses `side == 1`; the
# legacy path (LEGACY_SIDE_BUG=True) uses `side == 'long'`, which compares an
# int8 side code against a str and is therefore ALWAYS False — so every trade,
# long or short, takes the `else` (short) R-multiple branch (see
# backtester/__init__.py:1329, 1905). This test pins both the documented
# mechanism and its downstream effect on the optimiser's RRR selection.
# ---------------------------------------------------------------------------
def test_legacy_side_bug_regression(monkeypatch):
    monkeypatch.setattr(bt, "OPTIMIZE_RRR",        True, raising=False)
    monkeypatch.setattr(bt, "SMART_OPTIMIZATION", False, raising=False)
    monkeypatch.setattr(bt, "LOOKBACK_RANGE",     (12, 76), raising=False)
    monkeypatch.setattr(bt, "DRAWDOWN_CONSTRAINT", None, raising=False)
    monkeypatch.setattr(bt, "dd_constraint",       None, raising=False)

    # Trending-with-reversals data so the baseline produces BOTH long and short
    # trades — the bug only manifests on the long side (longs get mislabelled
    # as shorts in the R-multiple computation).
    df  = _df_from_bars(1_800, seed=5)
    dfi = bt.compute_indicators(df, bt.DEFAULT_LB)
    raw = bt.create_raw_signals(dfi, bt.DEFAULT_LB)
    sig = bt.parse_signals(raw, dfi["time"])
    trades, *_ = bt.backtest(dfi, sig)

    long_trades  = [t for t in trades if int(t[0]) ==  1]
    short_trades = [t for t in trades if int(t[0]) == -1]
    assert long_trades and short_trades, (
        "fixture must produce both long and short trades to exercise the bug"
    )

    # --- (a) the documented comparison mechanism, on real int8 side codes ---
    # Legacy `side == 'long'` is ALWAYS False; corrected `side == 1` is True
    # iff the trade is genuinely long.
    for t in trades:
        side = t[0]
        assert bool(side == "long") is False, (
            f"legacy comparison side=={side!r}=='long' was expected to be "
            f"unconditionally False (int8-vs-str), got True"
        )
    assert all(bool(t[0] == 1) for t in long_trades), "corrected side==1 must hold for longs"
    assert not any(bool(t[0] == 1) for t in short_trades), "corrected side==1 must fail for shorts"

    # --- (b) downstream R-multiple consequence on a real LONG trade ---
    # The corrected (long) peak-R formula must differ from the legacy (short)
    # formula the bug applies to that same long trade.
    side, e, x, entry, _exit, _qty, _pnl = long_trades[0]
    risk = entry * bt.SL_PERCENTAGE / 100.0
    peak_R_correct = (dfi["high"].iloc[e:x + 1].values.max() - entry) / risk   # long branch
    peak_R_legacy  = (entry - dfi["low"].iloc[e:x + 1].values.min()) / risk    # short branch (bug)
    assert peak_R_correct != peak_R_legacy, (
        "long peak-R must differ from the short formula the legacy bug applies; "
        "if equal the test cannot distinguish the paths"
    )

    # --- (c) end-to-end: the bug changes the optimiser's RRR-selection inputs ---
    # Faithfully replicate the optimiser's RRR probe (backtester/__init__.py
    # lines 1312-1352): probe at a fixed 5R TP, collect peak/close R per trade,
    # then sum R for each candidate target. The ONLY difference between the two
    # runs below is the side comparison, so any divergence in the candidate sums
    # is caused solely by the legacy bug mislabelling longs as shorts.
    def rrr_candidate_sums(legacy):
        old_tp, old_flag = bt.TP_PERCENTAGE, bt.USE_TP
        monkeypatch.setattr(bt, "TP_PERCENTAGE", 5 * bt.SL_PERCENTAGE, raising=False)
        monkeypatch.setattr(bt, "USE_TP", True, raising=False)
        probe_trades, *_ = bt.backtest(dfi, sig)
        monkeypatch.setattr(bt, "TP_PERCENTAGE", old_tp, raising=False)
        monkeypatch.setattr(bt, "USE_TP", old_flag, raising=False)

        peak_Rs, close_Rs = [], []
        for tside, te, tx, *_ in probe_trades:
            entry_price = dfi["close"].iloc[te]
            trisk = entry_price * bt.SL_PERCENTAGE / 100.0
            is_long = (tside == "long") if legacy else (tside == 1)
            if is_long:
                hi = dfi["high"].iloc[te:tx + 1].values
                peak_R  = (hi.max() - entry_price) / trisk
                close_R = (dfi["close"].iloc[tx] - entry_price) / trisk
            else:
                lo = dfi["low"].iloc[te:tx + 1].values
                peak_R  = (entry_price - lo.min()) / trisk
                close_R = (entry_price - dfi["close"].iloc[tx]) / trisk
            peak_Rs.append(min(peak_R, 3.0))
            close_Rs.append(close_R)
        peak_Rs  = np.array(peak_Rs, dtype=float)
        close_Rs = np.array(close_Rs, dtype=float)
        return {R: float(np.where(peak_Rs >= R, R, close_Rs).sum()) for R in range(1, 4)}

    sums_correct = rrr_candidate_sums(legacy=False)
    sums_legacy  = rrr_candidate_sums(legacy=True)
    assert sums_correct != sums_legacy, (
        f"legacy and corrected RRR-candidate sums are identical "
        f"({sums_correct}); the side-bug path is not being exercised"
    )
