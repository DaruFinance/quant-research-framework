# put this at the VERY TOP of the file, above any numpy/pandas/numba imports
import os, gc, psutil
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("BLIS_NUM_THREADS", "1")

import sys
# Ensure repo root is on sys.path so `import backtester as gb` works reliably
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)





import gc

def _worker_entry(args):
    spec, base_output, regime_seg = args
    return process_strategy(spec, base_output=base_output, regime_seg=regime_seg)



#!/usr/bin/env python3
import os
import numpy  as np
import pandas as pd
import importlib
import types
import sys, os
from contextlib import redirect_stdout
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from contextlib import redirect_stdout
import multiprocessing as mp
import pytz
import re
import math

class Tee:
    def __init__(self, *writers):
        self.writers = writers
    def write(self, data):
        for w in self.writers:
            w.write(data)
    def flush(self):
        for w in self.writers:
            w.flush()

# 1) Import your back-tester module (make sure PYTHONPATH is set so this resolves)
import backtester as gb
importlib.reload(gb)

def _patched_create_regime_signals(df, best_lbs, regimes):
    """Use the CURRENT gb.create_raw_signals() for each bar."""
    # pre-compute raw arrays for every LB we will use
    raws = {}
    for lb in set(best_lbs.values()):
        raws[lb] = gb.create_raw_signals(df, lb)

    out = np.zeros(len(df), dtype=np.int8)
    for i, reg in enumerate(regimes):
        lb = best_lbs[reg]
        out[i] = raws[lb][i]
    return out

gb.create_regime_signals = types.FunctionType(
    _patched_create_regime_signals.__code__, globals(),
    name='create_regime_signals')

def _get_rrr_candidates():
    """Return validated RRR candidates from gb.RRR_used (fallback: 1..5)."""
    values = getattr(gb, "RRR_used", [1, 2, 3, 4, 5])
    if isinstance(values, (int, float)):
        values = [values]

    cleaned = []
    for v in values:
        try:
            r = float(v)
        except (TypeError, ValueError):
            continue
        if r <= 0:
            continue
        if r not in cleaned:
            cleaned.append(r)

    return cleaned if cleaned else [1.0, 2.0, 3.0, 4.0, 5.0]


def _patched_optimize_regimes_sequential(is_df):
    """
    Three-phase optimisation that calls **your** strategy for every LB,
    but otherwise behaves exactly like the original (prints, RRR, etc.)
    """
    regs = ['Uptrend', 'Downtrend', 'Ranging']
    best_lbs   = {r: gb.DEFAULT_LB for r in regs}
    best_rrrs  = {r: None for r in regs}

    # -------- pre-compute common stuff -------------------------------
    dfi = is_df.copy()
    dfi['EMA_20']  = is_df['close'].ewm(span=20,  adjust=False).mean()
    dfi['EMA_200'] = is_df['close'].ewm(span=200, adjust=False).mean()
    regimes = gb.get_regimes(dfi)

    lb_space = range(*gb.LOOKBACK_RANGE)

    # cache raw arrays for every LB we might test
    raw_pool = {lb: gb.create_raw_signals(dfi, lb) for lb in lb_space}

    for target in regs:
        print(f"🔍 Phase: optimise {target}")
        best_val = -np.inf
        best_met = None
        best_lb  = None
        best_rrr = None

        for lb in lb_space:
            cand_lbs = best_lbs.copy()
            cand_lbs[target] = lb

            # build regime-aware raw with our patched helper
            raw = gb.create_regime_signals(dfi, cand_lbs, regimes)
            gb.last_unfiltered_raw = raw.copy()          # keep parser happy
            sig = gb.parse_signals(raw, dfi['time'])

            # ---- RRR probe (unchanged) ---------------------------------
            if gb.OPTIMIZE_RRR:
                rrr_candidates = _get_rrr_candidates()
                rrr_cap = max(rrr_candidates)
                tp_old, flag_old = gb.TP_PERCENTAGE, gb.USE_TP
                gb.TP_PERCENTAGE, gb.USE_TP = rrr_cap * gb.SL_PERCENTAGE, True
                trades_probe, *_ = gb.backtest(dfi, sig)

                # peak-/close-R calc
                peak_Rs, close_Rs = [], []
                for side, e, x, *_ in trades_probe:
                    entry = dfi['close'].iloc[e]
                    risk  = entry * gb.SL_PERCENTAGE / 100
                    if side == 'long':
                        peak  = dfi['high'].iloc[e:x+1].max()
                        close = dfi['close'].iloc[x]
                        peak_R  = (peak  - entry)/risk
                        close_R = (close - entry)/risk
                    else:
                        trough = dfi['low'].iloc[e:x+1].min()
                        close  = dfi['close'].iloc[x]
                        peak_R  = (entry - trough)/risk
                        close_R = (entry - close )/risk
                    peak_Rs.append(min(peak_R, rrr_cap))
                    close_Rs.append(close_R)

                arr_peak, arr_close = np.array(peak_Rs), np.array(close_Rs)
                sums = {R: np.where(arr_peak>=R, float(R), arr_close).sum()
                        for R in rrr_candidates}
                chosen_rrr = max(sums, key=sums.get)

                # re-run with chosen RRR
                gb.TP_PERCENTAGE, gb.USE_TP = chosen_rrr*gb.SL_PERCENTAGE, True
                gb.last_unfiltered_raw = raw.copy()
                _, met, *_ = gb.backtest(dfi, sig)
                met['RRR'] = chosen_rrr
                gb.TP_PERCENTAGE, gb.USE_TP = tp_old, flag_old
            else:
                _, met, *_ = gb.backtest(dfi, sig)

            if met['Trades'] < gb.MIN_TRADES:
                continue

            score = -met[gb.OPT_METRIC] if gb.OPT_METRIC=='MaxDrawdown' else met[gb.OPT_METRIC]
            if score > best_val:
                best_val, best_lb, best_met = score, lb, met
                best_rrr = met.get('RRR')

        best_lbs[target]  = best_lb
        best_rrrs[target] = best_rrr
        rrr_info = f" | RRR={best_rrr}" if best_rrr is not None else ""
        print(f"{target:>9} best LB = {best_lb} | {gb.OPT_METRIC}: {best_met[gb.OPT_METRIC]:.4f}{rrr_info}")

    return best_lbs, best_rrrs

gb.optimize_regimes_sequential = types.FunctionType(
    _patched_optimize_regimes_sequential.__code__, globals(),
    name='optimize_regimes_sequential')


def _patched_create_regime_signals(df, best_lbs, regimes):
    """Use the CURRENT gb.create_raw_signals() for each bar."""
    # pre-compute raw arrays for every LB we will use
    raws = {}
    for lb in set(best_lbs.values()):
        raws[lb] = gb.create_raw_signals(df, lb)

    out = np.zeros(len(df), dtype=np.int8)
    for i, reg in enumerate(regimes):
        lb = best_lbs[reg]
        out[i] = raws[lb][i]
    return out

gb.create_regime_signals = types.FunctionType(
    _patched_create_regime_signals.__code__, globals(),
    name='create_regime_signals')

def _patched_bcr(df: pd.DataFrame, best_lbs: dict):
    """
    * Identical metrics & RRR handling to the original,
      but the +1/-1 decisions come from **gb.create_regime_signals()**,
      which in turn calls your current gb.create_raw_signals().
    """
    dfi = df.copy()

    # columns needed only for regime labelling, NOT for signals
    dfi['EMA_20']  = dfi['close'].ewm(span=20 , adjust=False).mean()
    dfi['EMA_200'] = dfi['close'].ewm(span=200, adjust=False).mean()

    regimes = gb.get_regimes(dfi)

    # ---- build raw array with your strategy ------------------------
    raw_full = gb.create_regime_signals(dfi, best_lbs, regimes)
    raw_full = gb.filter_raw_signals(raw_full, regimes)

    gb.last_unfiltered_raw = raw_full.copy()          # keep exits aligned
    sig = gb.parse_signals(raw_full, dfi['time'])

    # ---- unchanged RRR-optimisation --------------------------------
    if gb.OPTIMIZE_RRR:
        rrr_candidates = _get_rrr_candidates()
        rrr_cap = max(rrr_candidates)
        tp_old, flag_old = gb.TP_PERCENTAGE, gb.USE_TP
        gb.TP_PERCENTAGE, gb.USE_TP = rrr_cap * gb.SL_PERCENTAGE, True
        trades_probe, *_ = gb.backtest(dfi, sig)

        peak_Rs, close_Rs = [], []
        for side, e, x, *_ in trades_probe:
            entry = dfi['close'].iloc[e]
            risk  = entry * gb.SL_PERCENTAGE / 100
            if side == 'long':
                peak  = dfi['high'].iloc[e:x+1].max()
                close = dfi['close'].iloc[x]
                peak_R  = (peak  - entry)/risk
                close_R = (close - entry)/risk
            else:
                trough = dfi['low'].iloc[e:x+1].min()
                close  = dfi['close'].iloc[x]
                peak_R  = (entry - trough)/risk
                close_R = (entry - close )/risk
            peak_Rs.append(min(peak_R, rrr_cap))
            close_Rs.append(close_R)

        arr_peak, arr_close = np.array(peak_Rs), np.array(close_Rs)
        sums = {R: np.where(arr_peak>=R, float(R), arr_close).sum()
                for R in rrr_candidates}
        best_rrr = max(sums, key=sums.get)

        gb.TP_PERCENTAGE, gb.USE_TP = best_rrr*gb.SL_PERCENTAGE, True
        gb.last_unfiltered_raw = raw_full.copy()
        trades, metrics, eq, rets, _ = gb.backtest(dfi, sig)
        metrics['RRR'] = best_rrr
        gb.TP_PERCENTAGE, gb.USE_TP = tp_old, flag_old
    else:
        trades, metrics, eq, rets, _ = gb.backtest(dfi, sig)

    return trades, metrics, eq, rets

# replace the original implementation
gb.backtest_continuous_regime = types.FunctionType(
    _patched_bcr.__code__, globals(), name='backtest_continuous_regime')

def get_completed_strategies(base_output):
    completed = set()
    if not os.path.isdir(base_output):
        return completed
    # base_output/<group>/<strategy> folders
    for group in os.listdir(base_output):
        group_path = os.path.join(base_output, group)
        if not os.path.isdir(group_path):
            continue
        for strat in os.listdir(group_path):
            strat_path = os.path.join(group_path, strat)
            if not os.path.isdir(strat_path):
                continue
            for fn in os.listdir(strat_path):
                low = fn.lower()
                # consider any of these as a completion marker
                if low.endswith(('.png', '.jpg', '.jpeg', '.txt', '.csv', '.done')):
                    completed.add(strat)
                    break
    return completed

# ---------------------------------------------------------------------------
# === OPTIONAL: Override any of Backtester’s USER CONFIG settings here ===
# ---------------------------------------------------------------------------
BACKTESTER_OVERRIDES = {
    # Numeric parameters
    'SLIPPAGE_PCT':            0.03,         # slippage % per order (entry and exit)
    'FEE_PCT':                 0.05,         # commission % per order (entry and exit)
    'FUNDING_FEE':             0.01,         # funding fee % charged at 00:00,08:00,16:00 UTC (crypto only)

    # ─── SESSION TRADING CONFIG ────────────────────────────────────────────
    'TRADE_SESSIONS': False,                # if True, only trade during NY session
    'SESSION_START': "8:00",            # NY open time (HH:MM)
    'SESSION_END': "16:50",            # NY close time (HH:MM)
    'NY_TZ': pytz.timezone("America/New_York"),
    # ───────────────────────────────────────────────────────────────────────

    'BACKTEST_CANDLES':        10_000,       # length of IS window
    'OOS_CANDLES':             120_000    ,       # length of RAW-OOS
    'USE_OOS2':                False,         # Split OOS into two windows

    'OPT_METRIC':              'Sharpe',         # ROI, PF, Sharpe, WinRate, Exp, MaxDrawdown, Consistency
    'MIN_TRADES':              1,
    'SMART_OPTIMIZATION':      True,         # True to skip “spiky” optimisations
    'DRAWDOWN_CONSTRAINT':     None,           # Skip optimizations with a drawdown higher than the value input. Use "None" for OFF
    'NEWS_AVOIDER':            False,         # Closes trades 2 candles prior to red folder news

    # Plotting & Monte Carlo
    'PRINT_EQUITY_CURVE':      True,         # plot
    'USE_MONTE_CARLO':         False,         # on first IS only
    'MC_RUNS':                 1000,         # simulations

    # Stop-loss & Take-profit
    'USE_SL':                  True,         # whether to enable stop loss
    'SL_PERCENTAGE':           1.0,          # stop loss percent (e.g. 1.0 for 1%)
    'USE_TP':                  True,         # whether to enable take profit
    'TP_PERCENTAGE':           3.0,          # take profit percent (e.g. 2.0 for 2%)
    'OPTIMIZE_RRR':            True,         # True to auto-optimise Risk-to-Reward ratio
    'RRR_used':                [0.5, 1, 2, 3, 4, 5],  # candidate RRRs when OPTIMIZE_RRR=True (supports floats)

    # --- FOREX MODE -----------------------------------------------------------
    'FOREX_MODE':              False,         # Turns all percentages values into pips
    # --------------------------------------------------------------------------


    # Filtering
    'FILTER_REGIMES':          False,        # works only when USE_REGIME_SEG == True
    'FILTER_DIRECTIONS':       False,        # blocks long/short based on IS stats

    'USE_REGIME_SEG':          False,        # enable regime segmentation?

    # --- ROBUSTNESS TESTS ----------------------------------------------------
    'FEE_SHOCK':               False,         # Set True to run Fee Shock (5× fees) stress-test
    'SLIPPAGE_SHOCK':          False,         # True to apply slippage shock (5× slippage)
    'NEWS_CANDLES_INJECTION':  False,         # NEW robustness scenario
    'ENTRY_DRIFT':             True,         # NEW – shift every entry one bar forward
    'INDICATOR_VARIANCE':      True,         # NEW: ±1 look-back

    # --- NEW WFO SWITCHES ---------------------------------------------------
    'USE_WFO':                 True,        # do rolling windows?
    'WFO_TRIGGER_MODE':        'candles',     # "candles" or "trades"
    'WFO_TRIGGER_VAL':         5000,           # n-candles or n-trades per window
}

# --- USE DIFFERENT STOP LOSS LEVELS --------------------------------------------
STOP_LOSS_VALUES = [2.0, 3.0] # 5.0, 7.0
# -------------------------------------------------------------------------------

def _free_mem_gb() -> float:
    """
    Effective free memory in GiB.
    Use 'available' plus a portion of cached (Windows can reclaim cache quickly).
    """
    v = psutil.virtual_memory()
    avail = v.available / (1024**3)
    cached = getattr(v, "cached", 0) / (1024**3)
    # Count 60% of cache as practically reclaimable
    return avail + 0.60 * cached


def _estimate_worker_mem_gb(overrides: dict | None) -> float:
    """
    Heuristic per-worker footprint (GiB) with equity curves & robustness ON,
    but tuned to be less conservative so we don't clamp to tiny pools.
    """
    base = 0.85  # numpy+pandas+numba+DF+plots on Windows spawn (leaner baseline)
    cfg = (overrides or BACKTESTER_OVERRIDES)
    if cfg.get("USE_MONTE_CARLO", False):
        base += 0.35
    if cfg.get("PRINT_EQUITY_CURVE", True):
        base += 0.10
    if cfg.get("ENTRY_DRIFT", False):
        base += 0.05
    if cfg.get("INDICATOR_VARIANCE", False):
        base += 0.05
    if cfg.get("NEWS_CANDLES_INJECTION", False):
        base += 0.10
    return base


def _mem_ok(expected_extra_gb: float, floor_gb: float = 2.0) -> bool:
    """
    True if, after allocating expected_extra_gb, at least floor_gb would remain.
    (Uses effective free memory from _free_mem_gb.)
    """
    free_gb = _free_mem_gb()
    return (free_gb - expected_extra_gb) >= floor_gb


def _worker_entry(args):
    spec, base_output, regime_seg = args
    return process_strategy(spec, base_output=base_output, regime_seg=regime_seg)


def _init_worker(overrides):
    """
    Per-process init:
      • BLAS is single-threaded via env vars (set at top of file)
      • Keep Numba to a small number of threads so we can run *more* processes
      • Apply overrides
    """
    try:
        import numba, os
        n = int(os.environ.get("NUMBA_THREADS_PER_WORKER", "3"))  # small by default
        numba.set_num_threads(max(1, n))
    except Exception:
        pass
    BACKTESTER_OVERRIDES.update(overrides)

def _warmup_numba():
    """Pre-JIT hot kernels in parent so workers load from cache."""
    try:
        import numpy as np
        from backtester import _parse_signals_numba
        raw = np.zeros(1024, dtype=np.int8)
        flags = np.zeros(1024, dtype=np.uint8)
        _ = _parse_signals_numba(raw, flags)
    except Exception:
        pass


def apply_global_overrides():
    # 1) shove in everything from BACKTESTER_OVERRIDES
    for var, val in BACKTESTER_OVERRIDES.items():
        setattr(gb, var, val)

    # 2) now re-enforce the module's own FOREX_MODE sizing & percent→pip scaling
    if gb.FOREX_MODE:
        # re-scale your SL/TP percentages into pip fractions
        gb.SL_PERCENTAGE *= gb.PIP_SIZE
        gb.TP_PERCENTAGE *= gb.PIP_SIZE
        # enforce the 1 R account/risk sizing
        gb.RISK_AMOUNT   = 1.0
        gb.ACCOUNT_SIZE  = 1.0
        gb.POSITION_SIZE = 1.0

    # 3) recompute any dependent globals (lookback range, OOS doubling, etc.)
    if 'DEFAULT_LB' in BACKTESTER_OVERRIDES:
        reset_lookback_range()
    gb.ORIGINAL_OOS = gb.OOS_CANDLES
    if gb.USE_OOS2:
        gb.OOS_CANDLES = gb.ORIGINAL_OOS * 2



# ---------------------------------------------------------------------------
# === Indicator helpers (so _make_raw_func can use them) ===
# ---------------------------------------------------------------------------
def compute_sma(data, length, column='close'):
    length = int(length)          # ← make sure it’s an integer
    return data[column].rolling(window=length).mean()


def compute_ema(data, length):
    return data['close'].ewm(span=length, adjust=False).mean()

def compute_macd(data, fast_length=12, slow_length=26, signal_length=9):
    fast = compute_ema(data, fast_length)
    slow = compute_ema(data, slow_length)
    macd = fast - slow
    sig  = macd.ewm(span=signal_length, adjust=False).mean()
    return macd, sig

def compute_rsi(data, length):
    delta = data['close'].diff()
    gain  = delta.where(delta > 0, 0)
    loss  = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(com=length-1, min_periods=length).mean()
    avg_loss = loss.ewm(com=length-1, min_periods=length).mean()
    rs  = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def compute_atr(data, length):
    hl = data['high'] - data['low']
    hc = (data['high'] - data['close'].shift()).abs()
    lc = (data['low']  - data['close'].shift()).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return tr.ewm(alpha=1/length, min_periods=length).mean()

def compute_stoch(data, length):
    lo = data['low'].rolling(window=length).min()
    hi = data['high'].rolling(window=length).max()
    return 100 * (data['close'] - lo) / (hi - lo)

def compute_ppo_master(data: pd.DataFrame, fastLen: int) -> pd.Series:
    # rolling lowest low and highest high
    lo = data['low'].rolling(window=fastLen).min()
    hi = data['high'].rolling(window=fastLen).max()
    
    # sums for numerator
    sum_up   = (data['close'] - lo).rolling(window=fastLen).sum()
    sum_down = (hi - data['close']).rolling(window=fastLen).sum()
    numMaster = sum_up - sum_down
    
    # sum for denominator
    denMaster = (hi - lo).rolling(window=fastLen).sum()
    
    # match PineScript: if denMaster == 0 → 0, else ratio
    ppoMaster = numMaster.divide(denMaster).where(denMaster != 0, 0)
    
    return ppoMaster

def compute_vr_osc(data: pd.DataFrame, length: int, q: int = 2) -> pd.Series:
    """
    Variance‑Ratio Oscillator over the last length bars of log‑returns,
    but suppress any warnings from log(…) on the warm‑up NaNs.
    
    R_t    = sum_{i=0..q-1} r_{t-i}
    VR     = var(R_t, N) / (q * var(r_t, N))
    VR_osc = (VR - 1) / (VR + 1)
    """
    # raw log‑returns; the first shift produces a NaN
    with np.errstate(divide='ignore', invalid='ignore'):
        r = np.log(data['close'] / data['close'].shift(1))
    
    # q‑period aggregation
    R = r.rolling(window=q).sum()
    
    # numerator and denominator of VR
    num = R.rolling(window=length).var(ddof=0)
    den = q * r.rolling(window=length).var(ddof=0)
    
    vr = num / den
    return (vr - 1) / (vr + 1)

# ---------------------------------------------------------------------------
# === Transformation helpers (so _make_raw_func can use them) ===
# ---------------------------------------------------------------------------

def f_zscore(src: pd.Series, length: int) -> pd.Series:
    """
    Z-score: (src - mean) / std, zero where std is zero.
    """
    mean = src.rolling(window=length, min_periods=length).mean()
    std  = src.rolling(window=length, min_periods=length).std(ddof=0)
    return ((src - mean).where(std != 0, 0)) / std

def f_roc(src: pd.Series, length: int) -> pd.Series:
    """
    Rate of Change: (src - src.shift(length)) / src.shift(length) * 100, NaN if denominator is zero.
    """
    prev = src.shift(length)
    roc  = (src - prev) / prev * 100
    return roc.where(prev != 0)

def f_bias(src: pd.Series, length: int, smooth: int) -> pd.Series:
    """
    EMA of slope: slope = (src - src.shift(length)) / length, then EMA(slope, span=smooth).
    """
    slope = (src - src.shift(length)) / length
    return slope.ewm(span=smooth, adjust=False).mean()

def f_volZ(src: pd.Series, length: int) -> pd.Series:
    """
    Volatility Z-score: compute rolling std of src, 
    then z-score that volatility over the same window.
    """
    vol     = src.rolling(window=length, min_periods=length).std(ddof=0)
    meanVol = vol.rolling(window=length, min_periods=length).mean()
    devVol  = vol.rolling(window=length, min_periods=length).std(ddof=0)
    return ((vol - meanVol).where(devVol != 0, 0)) / devVol

def f_accel(src: pd.Series, length: int) -> pd.Series:
    """
    Second derivative approximation:
      (src - 2*src.shift(length) + src.shift(2*length)) / (length^2).
    """
    shifted1 = src.shift(length)
    shifted2 = src.shift(2 * length)
    return (src - 2*shifted1 + shifted2) / (length * length)

def f_distFromMedian(src: pd.Series, length: int) -> pd.Series:
    if length <= 0:
        return pd.Series(0, index=src.index)
    highest = src.rolling(window=length, min_periods=length).max()
    lowest  = src.rolling(window=length, min_periods=length).min()
    median  = (highest + lowest) / 2
    return src - median

def f_slope(src: pd.Series, length: int) -> pd.Series:
    """
    1:1 translation of PineScript slope:
    _prev = _src[_len]
    _len != 0 ? (_src - _prev) / _len : 0
    """
    if length == 0:
        # avoid division by zero
        return pd.Series(0, index=src.index)
    prev = src.shift(length)
    return (src - prev) / length

def f_normalized_price(src: pd.Series, length: int) -> pd.Series:
    """
    1:1 translation of PineScript normalized_price:
    _high  = highest(_src, _len)
    _low   = lowest(_src, _len)
    _range = _high - _low
    _range != 0 ? (_src - _low) / _range : 0
    """
    high = src.rolling(window=length).max()
    low = src.rolling(window=length).min()
    rng = high - low
    return pd.Series(
        np.where(rng != 0, (src - low) / rng, 0),
        index=src.index
    )

# 0) Helper: safe rolling funcs
def _roll(s, n, func, **kw):
    return getattr(s.rolling(n, min_periods=n), func)(**kw)

# 1) Folded Deviation (price folded around its rolling median)
def f_fold_dev(src: pd.Series, lb: int) -> pd.Series:
    med = _roll(src, lb, 'median')
    return (src - med).abs() * np.sign(src - med.shift(1))

# 2) Rank Osc Residual (demeaned rolling rank; behaves like a quirky oscillator)
def f_rank_resid(src: pd.Series, lb: int) -> pd.Series:
    r = _roll(src, lb, 'rank')
    return (r - _roll(r, lb, 'mean')) / (lb - 1)

# 3) Quantile Stretch (pushes tails harder than center; non‑linear & bounded)
def f_quant_stretch(src: pd.Series, lb: int, q_low=0.1, q_hi=0.9) -> pd.Series:
    lo  = _roll(src, lb, 'quantile', q=q_low)
    hi  = _roll(src, lb, 'quantile', q=q_hi)
    num = src - lo
    den = (hi - lo).replace(0, np.nan)
    return 2 * (num / den) - 1  # map to roughly [-1, 1]

# ---------------------------------------------------------------------------
# === Strategy definitions ===
# Each spec drives _make_raw_func so that gb.create_raw_signals(df, lb)
# will return exactly your +1/-1 array for that rule.
# ---------------------------------------------------------------------------

STRATEGIES = [
    *[
        {
            "name": f"EMA_x_EMA{p}{f'_{trans}' if trans else ''}{('_' + conf if conf else '')}",
            "default_lb": p + 20,
            "primary":  "EMA",
            "partner":  ("EMA", p),
            "transformation": trans,
            "confluence":     conf        
        }
        for p in (20, 50, 100)
        for trans in (None, "zscore", "slope", "normalized_price", "roc", "bias", "volZ", "accel", "disFromMedian", "quant_stretch", "rank_resid", "fold_dev")
        for conf  in (None, "RSIge40", "Pge0.7", "kurtosis10", "skew0.75", "atr_pct0.8", "RSIge50", "Pge0.8", "BW_filter", "pi", "vr", "kurtosis", "skew", "atr_pct", "burstfreq", "TinyBody", "NoNewLowGreen", "RangeSpike", "YesterdayPeak", "DeadFlat10", "InsideBar", "SameDirection", "TopOfRange", "VolContraction", "EMAHug")
    ],
         
    *[
        {
            "name": f"SMA_x_SMA{p}{f'_{trans}' if trans else ''}{('_' + conf if conf else '')}",
            "default_lb": p + 20,
            "primary":  "SMA",
            "partner":  ("SMA", p),
            "transformation": trans,
            "confluence":     conf        
        }
        for p in (20, 50, 100)
        for trans in (None, "zscore", "slope", "normalized_price", "roc", "bias", "volZ", "accel", "disFromMedian", "quant_stretch", "rank_resid", "fold_dev")
        for conf  in (None, "RSIge40", "Pge0.7", "kurtosis10", "skew0.75", "atr_pct0.8", "RSIge50", "Pge0.8", "BW_filter", "pi", "vr", "kurtosis", "skew", "atr_pct", "burstfreq", "TinyBody", "NoNewLowGreen", "RangeSpike", "YesterdayPeak", "DeadFlat10", "InsideBar", "SameDirection", "TopOfRange", "VolContraction", "EMAHug")
    ],

    *[
        {
            "name": f"RSI_x_{typ}{p}{f'_{trans}_{mode}' if trans else ''}{('_' + conf if conf else '')}",
            "default_lb": p + 20,
            "primary":    "RSI",
            "partner":    (typ, p),
            "transformation": trans,
            "mode":       mode,
            "confluence":     conf       
        }
        for typ   in ("EMA", "SMA")
        for p     in (8, 20, 50, 100)
        for trans in (None, "zscore", "slope", "normalized_price",
                      "roc", "bias", "volZ", "accel", "disFromMedian", "quant_stretch", "rank_resid", "fold_dev")
        for mode  in ("calc", "src")
        for conf  in (None, "RSIge40", "Pge0.7", "kurtosis10", "skew0.75", "atr_pct0.8", "RSIge50", "Pge0.8", "BW_filter", "pi", "vr", "kurtosis", "skew", "atr_pct", "burstfreq", "TinyBody", "NoNewLowGreen", "RangeSpike", "YesterdayPeak", "DeadFlat10", "InsideBar", "SameDirection", "TopOfRange", "VolContraction", "EMAHug")

    ],
    *[
        {"name": f"RSI_{typ}_{lvl}",
         "default_lb": 50,
         "primary":  "RSI_LEVEL",
         "partner":   (typ, lvl)}
        for typ in ("SMA","EMA") for lvl in (40,50,60)
    ],

    *[
        {
            "name": f"MACD({f},{s}){f'_{trans}' if trans else ''}{('_' + conf if conf else '')}",
            "default_lb": f,
            "primary":    "MACD",
            "params":     (f, s),
            "partner":    (None, None),
            "transformation": trans,
            "confluence":     conf       
        }
        for f, s in ((24, 52), (16, 42), (26, 68), (42, 110))
        for trans in (None, "zscore", "slope", "normalized_price", "roc", "bias", "volZ", "accel", "disFromMedian", "quant_stretch", "rank_resid", "fold_dev")
        for conf  in (None, "RSIge40", "Pge0.7", "kurtosis10", "skew0.75", "atr_pct0.8", "RSIge50", "Pge0.8", "BW_filter", "pi", "vr", "kurtosis", "skew", "atr_pct", "burstfreq", "TinyBody", "NoNewLowGreen", "RangeSpike", "YesterdayPeak", "DeadFlat10", "InsideBar", "SameDirection", "TopOfRange", "VolContraction", "EMAHug")
    ],

    *[
        {
            "name": f"STOCHK_{kind}_{p}{f'_{trans}_{mode}' if trans else ''}{('_' + conf if conf else '')}",
            "default_lb": p,
            "primary":      "STOCHK",
            "partner":      (kind, None),
            "transformation": trans,
            "mode":         mode,
            "confluence":     conf       
        }
        for kind  in ("SMA", "EMA")
        for p     in (21, 55, 89)
        for trans in (None, "zscore", "slope", "normalized_price",
                      "roc", "bias", "volZ", "accel", "disFromMedian", "quant_stretch", "rank_resid", "fold_dev")
        for mode  in ("calc", "src")
        for conf  in (None, "RSIge40", "Pge0.7", "kurtosis10", "skew0.75", "atr_pct0.8", "RSIge50", "Pge0.8", "BW_filter", "pi", "vr", "kurtosis", "skew", "atr_pct", "burstfreq", "TinyBody", "NoNewLowGreen", "RangeSpike", "YesterdayPeak", "DeadFlat10", "InsideBar", "SameDirection", "TopOfRange", "VolContraction", "EMAHug")
    ],

    *[
        {
            "name": f"ATR_x_{typ}{p}{f'_{trans}_{mode}' if trans else ''}{('_' + conf if conf else '')}",
            "default_lb": 50,
            "primary":      "ATR",
            "partner":      (typ, p),
            "transformation": trans,
            "mode":         mode,
            "confluence":     conf       
        }
        for typ   in ("EMA", "SMA")
        for p     in (50, 100, 200)
        for trans in (None, "zscore", "slope", "normalized_price",
                      "roc", "bias", "volZ", "accel", "disFromMedian", "quant_stretch", "rank_resid", "fold_dev")
        for mode  in ("calc", "src")
        for conf  in (None, "RSIge40", "Pge0.7", "kurtosis10", "skew0.75", "atr_pct0.8", "RSIge50", "Pge0.8", "BW_filter", "pi", "vr", "kurtosis", "skew", "atr_pct", "burstfreq", "TinyBody", "NoNewLowGreen", "RangeSpike", "YesterdayPeak", "DeadFlat10", "InsideBar", "SameDirection", "TopOfRange", "VolContraction", "EMAHug")
    ],

    *[
        {
            "name": f"PPO_x_{typ}{p}{f'_{trans}_{mode}' if trans else ''}{('_' + conf if conf else '')}",
            "default_lb": p + 20,
            "primary":    "PPO",
            "partner":    (typ, p),
            "transformation": trans,
            "mode":       mode,
            "confluence":     conf       
        }
        for typ   in ("EMA", "SMA")
        for p     in (8, 20, 50, 100)
        for trans in (None, "zscore", "slope", "normalized_price",
                      "roc", "bias", "volZ", "accel", "disFromMedian", "quant_stretch", "rank_resid", "fold_dev")
        for mode  in ("calc", "src")
        for conf  in (None, "RSIge40", "Pge0.7", "kurtosis10", "skew0.75", "atr_pct0.8", "RSIge50", "Pge0.8", "BW_filter", "pi", "vr", "kurtosis", "skew", "atr_pct", "burstfreq", "TinyBody", "NoNewLowGreen", "RangeSpike", "YesterdayPeak", "DeadFlat10", "InsideBar", "SameDirection", "TopOfRange", "VolContraction", "EMAHug")

    ],
]

_original = STRATEGIES
STRATEGIES = [
    {
        **spec,
        "stop_loss": sl,
        "name":       f"{spec['name']}_SL{int(sl)}"
    }
    for spec in _original
    for sl   in STOP_LOSS_VALUES
]


# ---------------------------------------------------------------------------
# Recompute LOOKBACK_RANGE after changing DEFAULT_LB
def reset_lookback_range():
    lb = gb.DEFAULT_LB
    gb.LOOKBACK_RANGE = (int(lb*0.25), int(lb*1.5)+1)

# ---------------------------------------------------------------------------
# Build a custom create_raw_signals(df, lb) for each spec
def _make_raw_func(spec):
    prim, (kind, val) = spec['primary'], spec['partner']
    params            = spec.get('params', None)
    transformation = spec.get('transformation')
    mode           = spec.get('mode', 'calc')
    confluence     = spec.get('confluence')      

    def _raw(df_slice, lb):

        def maybe_transform(src, _lb=None):
            use_lb = _lb if _lb is not None else lb      # fall back to outer lb

            if not transformation:
                return src
            if transformation == 'bias':
                return f_bias(src, lb, lb)
            if transformation == 'quant_stretch':
                return f_quant_stretch(src, lb, 0.10, 0.90)
            func_name = 'f_' + ('distFromMedian' if transformation == 'disFromMedian'
                                else transformation)
            return globals()[func_name](src, lb)

        # ─── FAST / SLOW ────────────────────────────────────────────────
        if prim == 'EMA':
            src = maybe_transform(df_slice['close'])
            if kind == 'PRICE':
                fast = compute_ema(pd.DataFrame({'close': src}), lb).shift(1)
                slow = df_slice['close'].shift(1)
            else:
                fast = compute_ema(pd.DataFrame({'close': src}), val).shift(1)
                slow = compute_ema(pd.DataFrame({'close': src}), lb).shift(1)


        elif prim == 'SMA':
            src = maybe_transform(df_slice['close'])
            if kind == 'PRICE':
                fast = compute_sma(pd.DataFrame({'close': src}), lb, 'close').shift(1)
                slow = df_slice['close'].shift(1)
            else:
                fast = compute_sma(pd.DataFrame({'close': src}), val, 'close').shift(1)
                slow = compute_sma(pd.DataFrame({'close': src}), lb, 'close').shift(1)

        elif prim == 'RSI':
            if transformation and mode == 'src':
                # 1) transform the raw price first
                src_price = maybe_transform(df_slice['close'], lb)
                df_for_rsi = pd.DataFrame({'close': src_price})
                rsi_full   = compute_rsi(df_for_rsi, lb)
            else:
                # 2) compute RSI on raw price
                rsi_full = compute_rsi(df_slice, lb)
                # 3) optionally transform the RSI values
                if transformation and mode == 'calc':
                    rsi_full = maybe_transform(rsi_full, lb)

            # 4) build fast/slow MA of that RSI
            tmp = pd.DataFrame({'close': rsi_full})
            if kind == 'EMA':
                fast = compute_ema(tmp, lb).shift(1)
                slow = compute_ema(tmp, int(1.5*lb)).shift(1)
            else:
                fast = compute_sma(tmp, lb, 'close').shift(1)
                slow = compute_sma(tmp, int(1.5*lb), 'close').shift(1)

        elif prim == 'RSI_LEVEL':
            # compute raw RSI using lb
            rsi_full = compute_rsi(df_slice, lb)
            # smooth that RSI via lb-period MA:
            if kind == 'EMA':
                tmp  = pd.DataFrame({'close': rsi_full})
                fast = compute_ema(tmp, 3).shift(1)
            else:  # SMA
                tmp  = pd.DataFrame({'rsi': rsi_full})
                fast = compute_sma(tmp.rename(columns={'rsi':'close'}),
                                   3, 'close').shift(1)
            # constant level threshold
            slow = pd.Series(val, index=df_slice.index)


        elif prim == 'MACD':
            # 1) transform the price if needed
            src = maybe_transform(df_slice['close'])
            data2 = pd.DataFrame({'close': src})

            # 2) compute MACD on the transformed source
            fast_len, slow_len = lb, params[1]
            macd_line, sig_line = compute_macd(
                data2,
                fast_length=fast_len,
                slow_length=slow_len,
                signal_length=9
            )

            # 3) smooth both lines
            macd_sm = macd_line.rolling(window=10).mean()
            sig_sm  = sig_line.rolling(window=10).mean()

            # 4) shift to avoid look-ahead
            fast = macd_sm.shift(1)
            slow = sig_sm.shift(1)


        elif prim == 'STOCHK':
            if transformation and mode == 'src':
                # transform the price, then use existing compute_stoch
                src_price = maybe_transform(df_slice['close'], lb)
                tmp_df    = df_slice.copy()
                tmp_df['close'] = src_price
                k_full    = compute_stoch(tmp_df, lb)
            else:
                # raw %K
                k_full = compute_stoch(df_slice, lb)
                # optional transform of %K itself
                if transformation and mode == 'calc':
                    k_full = maybe_transform(k_full, lb)

            tmp = pd.DataFrame({'close': k_full})
            if kind == 'EMA':
                fast = compute_ema(tmp, lb).shift(1)
                slow = compute_ema(tmp, 2*lb).shift(1)
            else:
                fast = compute_sma(tmp, lb, 'close').shift(1)
                slow = compute_sma(tmp, 2*lb, 'close').shift(1)

        elif prim == 'STOCHK_LEVEL':
            k_full = compute_stoch(df_slice, lb)
            tmp    = pd.DataFrame({'close': k_full})
            if kind == 'EMA':
                fast = compute_ema(tmp, lb).shift(1)
            else:
                fast = compute_sma(tmp, lb, 'close').shift(1)
            slow = pd.Series(val, index=df_slice.index)

        elif prim == 'ATR':
            if transformation and mode == 'src':
                # 1) transform price, then ATR
                price_warped = maybe_transform(df_slice['close'], lb)
                tmp_df = df_slice.copy()
                tmp_df['close'] = price_warped
                atr_full = compute_atr(tmp_df, lb)
            else:
                # 2) raw ATR
                atr_full = compute_atr(df_slice, lb)
                if transformation and mode == 'calc':
                    atr_full = maybe_transform(atr_full, lb)

            tmp_fast = pd.DataFrame({'atr': atr_full})
            fast     = compute_sma(tmp_fast.rename(columns={'atr':'close'}), 3, 'close').shift(1)

            if kind == 'EMA':
                tmp_slow = pd.DataFrame({'atr': atr_full})
                slow     = compute_ema(tmp_slow.rename(columns={'atr':'close'}), val).shift(1)
            else:
                tmp_slow = pd.DataFrame({'atr': atr_full})
                slow     = compute_sma(tmp_slow.rename(columns={'atr':'close'}), val, 'close').shift(1)

        elif prim == 'PPO':
            if transformation and mode == 'src':
                # 1) transform the raw price first
                src_price = maybe_transform(df_slice['close'], lb)
                df_for_ppo = df_slice[['low','high','close']].copy()  # include low & high
                df_for_ppo['close'] = src_price
                ppo_full   = compute_ppo_master(df_for_ppo, lb)
            else:
                # 2) compute PPO on raw price
                ppo_full = compute_ppo_master(df_slice, lb)
                # 3) optionally transform the PPO values
                if transformation and mode == 'calc':
                    ppo_full = maybe_transform(ppo_full, lb)

            # 4) build fast/slow MA of that PPO
            tmp = pd.DataFrame({'close': ppo_full})
            fast = ppo_full.shift(1)
            if kind == 'EMA':
                slow = compute_ema(tmp, int(0.5*lb)).shift(1)
            else:
                slow = compute_sma(tmp, int(0.5*lb), 'close').shift(1)

        elif prim == 'VR':
            if transformation and mode == 'src':
                # 1) transform the raw price first
                src_price = maybe_transform(df_slice['close'], lb)
                df_for_vr = df_slice[['low','high','close']].copy()  # include low & high
                df_for_vr['close'] = src_price
                vr_full   = compute_vr_osc(df_for_vr, lb)
            else:
                # 2) compute PPO on raw price
                vr_full = compute_vr_osc(df_slice, lb)
                # 3) optionally transform the PPO values
                if transformation and mode == 'calc':
                    vr_full = maybe_transform(vr_full, lb)

            # 4) build fast/slow MA of that PPO
            tmp = pd.DataFrame({'close': vr_full})
            if kind == 'EMA':
                fast = compute_ema(tmp, lb)
                slow = compute_ema(tmp, int(1.5*lb)).shift(1)
            else:
                fast = compute_sma(tmp, lb)
                slow = compute_sma(tmp, int(1.5*lb), 'close').shift(1)


        else:
            raise ValueError(f"Unknown primary {prim}")

        # ─── BUILD RAW +1 / -1 ─────────────────────────────────────────
        cross_up   = (fast > slow)  & (fast.shift(1) <= slow.shift(1))
        cross_down = (fast < slow)  & (fast.shift(1) >= slow.shift(1))

        raw = np.zeros(len(df_slice), dtype=np.int8)
        raw[cross_up]   =  1
        raw[cross_down] = -1

        gb._last_df = df_slice
        gb._last_lb = lb

        return raw

    return _raw


def process_strategy(spec,
                     base_output="strategy_outputs_forex",
                     regime_seg=False):
    """
    Run a single strategy and save results under *base_output*.
    Keeps equity curves and robustness; aggressively frees memory after save.
    """
    apply_global_overrides()
    if 'stop_loss' in spec:
        gb.SL_PERCENTAGE = spec['stop_loss']

    gb.USE_REGIME_SEG = regime_seg
    gb.CONFLUENCES    = spec.get('confluence', None)

    df_full = gb.load_ohlc(gb.CSV_FILE)
    orig    = gb.create_raw_signals
    os.makedirs(base_output, exist_ok=True)

    name  = spec['name'] + ('_reg' if regime_seg else '')
    group = spec['primary']
    folder = os.path.join(base_output, group, name)
    os.makedirs(folder, exist_ok=True)

    gb.EXPORT_PATH = os.path.join(folder, "trade_list.csv")

    log_path = os.path.join(folder, f"{name}.txt")
    try:
        with open(log_path, "w", encoding="utf-8") as log_f:
            tee = Tee(sys.stdout, log_f)
            with redirect_stdout(tee):
                print(f"\n=== RUNNING STRATEGY: {name} ===")
                print("Reproducibility Parameters:")
                params = {
                    "SLIPPAGE_PCT":           gb.SLIPPAGE_PCT,
                    "FEE_PCT":                gb.FEE_PCT,
                    "FUNDING_FEE":            gb.FUNDING_FEE,
                    "TRADE_SESSIONS":         gb.TRADE_SESSIONS,
                    "SESSION_START":          gb.SESSION_START,
                    "SESSION_END":            gb.SESSION_END,
                    "BACKTEST_CANDLES":       gb.BACKTEST_CANDLES,
                    "OOS_CANDLES":            gb.OOS_CANDLES,
                    "USE_OOS2":               gb.USE_OOS2,
                    "OPT_METRIC":             gb.OPT_METRIC,
                    "MIN_TRADES":             gb.MIN_TRADES,
                    "PRINT_EQUITY_CURVE":     gb.PRINT_EQUITY_CURVE,
                    "USE_MONTE_CARLO":        gb.USE_MONTE_CARLO,
                    "MC_RUNS":                gb.MC_RUNS,
                    "USE_SL":                 gb.USE_SL,
                    "SL_PERCENTAGE":          gb.SL_PERCENTAGE,
                    "USE_TP":                 gb.USE_TP,
                    "TP_PERCENTAGE":          gb.TP_PERCENTAGE,
                    "OPTIMIZE_RRR":           gb.OPTIMIZE_RRR,
                    "RRR_used":               gb.RRR_used,
                    "FOREX_MODE":             gb.FOREX_MODE,
                    "FILTER_REGIMES":         gb.FILTER_REGIMES,
                    "FILTER_DIRECTIONS":      gb.FILTER_DIRECTIONS,
                    "USE_REGIME_SEG":         gb.USE_REGIME_SEG,
                    "NEWS_AVOIDER":           gb.NEWS_AVOIDER,
                    "FEE_SHOCK":              gb.FEE_SHOCK,
                    "SLIPPAGE_SHOCK":         gb.SLIPPAGE_SHOCK,
                    "NEWS_CANDLES_INJECTION": gb.NEWS_CANDLES_INJECTION,
                    "ENTRY_DRIFT":            gb.ENTRY_DRIFT,
                    "INDICATOR_VARIANCE":     gb.INDICATOR_VARIANCE,
                    "USE_WFO":                gb.USE_WFO,
                    "WFO_TRIGGER_MODE":       gb.WFO_TRIGGER_MODE,
                    "WFO_TRIGGER_VAL":        gb.WFO_TRIGGER_VAL,
                }
                for k, v in params.items():
                    print(f"  {k} = {v}")

                gb.DEFAULT_LB = spec['default_lb']
                reset_lookback_range()
                gb.signals_cache.clear()
                gb.signals_cache['df'] = df_full  # no .copy()
                gb.create_raw_signals = _make_raw_func(spec)

                # run your full pipeline (equity curves & robustness happen inside)
                gb.main()

        # Save and release any figures created by gb.main()
        for num in plt.get_fignums():
            fig = plt.figure(num)
            fig_path = os.path.join(folder, f"{name}_fig{num}.png")
            fig.savefig(fig_path, dpi=100)  # modest dpi to reduce buffer size
        # completion marker
        with open(os.path.join(folder, "_done"), "w") as _f:
            _f.write("ok")
        return name
    finally:
        plt.close('all')
        gb.create_raw_signals = orig
        del df_full
        gc.collect()
# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------
def run_all(base_output: str,
            regime_seg: bool,
            overrides: dict | None = None,
            max_workers: int | None = None,
            chunksize: int = 6,
            maxtasksperchild: int = 24,
            reserve_free_gb: float = 2.0):
    """
    High-throughput runner:
      • Starts at desired workers (up to 32 on 7950X).
      • If pre-launch RAM is *critically* low, step down by 2 until safe.
      • Otherwise, launch at full desired and only back off by 2 on MemoryError.
      • Tracks progress; streams tasks; equity curves/robustness stay ON.
    """
    apply_global_overrides()

    cpu_cnt = mp.cpu_count() or 4
    desired = min(cpu_cnt, 32) if max_workers is None else max(2, min(max_workers, 32))
    per_worker_gb = _estimate_worker_mem_gb(overrides or BACKTESTER_OVERRIDES)
    current = desired

    # Pending list (don’t re-run finished stuff)
    done = get_completed_strategies(base_output)
    if done:
        print(f"Skipping {len(done)} already completed strategies.")
    pending = [spec for spec in STRATEGIES if spec['name'] not in done]

    # Only pre-trim if we’d start with dangerously low headroom
    # (i.e., effective free RAM already below reserve).
    if _free_mem_gb() < reserve_free_gb and current > 2:
        # minimal pre-trim: bring it just above reserve
        while current > 2 and not _mem_ok(current * per_worker_gb, floor_gb=reserve_free_gb):
            current -= 2
        print(f"[pretrim] starting with {current} workers due to very low free RAM")
    else:
        print(f"[launch] trying {current} workers (target), per-worker≈{per_worker_gb:.2f} GiB, "
              f"effective_free≈{_free_mem_gb():.1f} GiB, reserve={reserve_free_gb:.1f} GiB")

    # Outer loop: on MemoryError, reduce workers by 2 and resume remaining
    while pending:
        print(f"[pool] {current} workers | chunksize={chunksize} | maxtasksperchild={maxtasksperchild} "
              f"| projected≈{current*per_worker_gb:.1f} GiB | free≈{_free_mem_gb():.1f} GiB")
        completed_names = []
        try:
            ctx = mp.get_context('spawn')
            pool_kwargs = dict(processes=current, maxtasksperchild=maxtasksperchild)
            init_over = (overrides or BACKTESTER_OVERRIDES)
            pool_kwargs.update({'initializer': _init_worker, 'initargs': (init_over,)})
            with ctx.Pool(**pool_kwargs) as pool:
                for name in pool.imap_unordered(
                        _worker_entry,
                        ((spec, base_output, regime_seg) for spec in pending),
                        chunksize):
                    print(f"✔ {name}")
                    completed_names.append(name)
            # remove completed
            if completed_names:
                done_now = set(completed_names)
                pending = [s for s in pending if s['name'] not in done_now]
        except MemoryError:
            # keep progress, back off by 2, and retry
            if completed_names:
                done_now = set(completed_names)
                pending = [s for s in pending if s['name'] not in done_now]
            if current > 2:
                current -= 2
                print(f"[autoscale] MemoryError → reducing workers to {current} and resuming...")
                continue
            raise

    seg_txt = " (regime-segmented)" if regime_seg else ""
    print(f"\n✅ All pending strategies complete → {base_output}{seg_txt}")





# ---------------------------------------------------------------------------
# Wrapper – run once (vanilla) and once with regime segmentation enabled
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    mp.freeze_support()

    run_all(
        base_output=r"YOUR_OUTPUT_FOLDER_HERE",  # <-- Set this to your desired output folder path (e.g. r"C:\Strategies\MyRun" on Windows)
        regime_seg=False,
        max_workers=32,      # target; scaler will step down by 2 if needed
        chunksize=6,         # fat enough to keep workers busy
        maxtasksperchild=24, # recycle sometimes to cap RAM creep
        reserve_free_gb=2.0  # keep at least 4 GiB headroom
    )


