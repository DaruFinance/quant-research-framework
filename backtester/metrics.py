#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Metric computation extracted from ``backtester/__init__.py``.

Mirrors the Rust port ``src/metrics.rs``. Pure move/re-export: every
function here is byte-for-byte identical in behaviour to its prior
location. Module-level globals (FOREX_MODE, POSITION_SIZE, ACCOUNT_SIZE,
SL_PERCENTAGE, TP_PERCENTAGE, PIP_SIZE, ...) are read LIVE at call time
via ``_bt.<name>`` so the ``Config.with_config()`` runtime-mutation
contract (setattr on the ``backtester`` module) keeps working unchanged.

The two ``@njit`` helpers (``_cummax``, ``_five_segment_sums``) read no
globals and move verbatim — numba cannot resolve ``_bt.<name>`` inside a
jitted body, so nothing is qualified there.
"""

import numpy as np
import pandas as pd
from math import sqrt
from numba import njit

import backtester as _bt

__all__ = [
    "_metrics_from_trades",
    "_cummax",
    "_five_segment_sums",
    "_bar_based_sharpe",
    "prettyprint",
]


def _metrics_from_trades(trades):
    # 1) per-trade returns in R-units vs USD
    if _bt.FOREX_MODE:
        rets = np.array([pnl / _bt.POSITION_SIZE for *_, pnl in trades], dtype=float)
    else:
        rets = np.array([pnl / _bt.ACCOUNT_SIZE for *_, pnl in trades], dtype=float)
    tc   = len(rets)
    wr   = np.mean(rets > 0) if tc else 0.0
    roi  = rets.sum() if tc else 0.0   # sum of R or sum of fracreturns
    wins = rets[rets > 0]
    losses = -rets[rets <= 0]
    pf   = wins.sum() / losses.sum() if losses.size else float('inf')
    expc = (wins.mean() if wins.size else 0) * wr \
         - (losses.mean() if losses.size else 0) * (1 - wr)
    shp  = (rets.mean() / rets.std() * sqrt(tc)) if tc > 1 and rets.std() else 0.0

    # Max drawdown
    if _bt.FOREX_MODE:
        eq = np.concatenate(([0.0], np.cumsum(rets)))
        hw = np.maximum.accumulate(eq)
        dd = np.max(hw - eq) if tc else 0.0
    else:
        eq = 1.0 + np.cumsum(rets)
        hw = np.maximum.accumulate(eq)
        dd = np.max((hw - eq) / hw) if tc else 0.0

    # Consistency
    segs = np.array_split(rets, 5)
    w    = np.array([0.0117,0.0317,0.0861,0.2341,0.6364])
    consistency = 0.6 * np.dot(w, [s.sum() for s in segs]) + 0.4 * roi

    return {
        'Trades':      tc,
        'ROI':         roi,
        'PF':          pf,
        'WinRate':     wr,
        'Exp':         expc,
        'Sharpe':      shp,
        'MaxDrawdown': dd,
        'Consistency': consistency
    }


@njit(cache=True)
def _cummax(arr):
    """Numbafriendly replacement for np.maximum.accumulate(arr)."""
    out = np.empty_like(arr)
    m   = -1e100          # works for any realistic equity fraction
    for i in range(arr.size):
        v = arr[i]
        if v > m:
            m = v
        out[i] = m
    return out

@njit(cache=True)
def _five_segment_sums(vec):
    """
    Split 1D array into five equalsized (1) segments and
    return their sums (length5 np.ndarray).
    """
    n   = vec.size
    seg = np.empty(5, dtype=np.float64)
    start = 0
    for k in range(5):
        end   = start + (n + k) // 5     # distributes the remainder nicely
        seg[k] = vec[start:end].sum()
        start = end
    return seg


def _bar_based_sharpe(df, trades, use_forex, account_size):
    """Standard calendar-time Sharpe on the per-bar mark-to-market equity curve.

    Reconstructs equity at every bar as account + realised PnL of closed trades
    + unrealised mark-to-market of any open position at that bar's close, then
    annualises mean/std of per-bar returns by the bar frequency (periods per
    Julian year from the median bar spacing). Crypto path compounds (pct-change
    returns, matching vectorbt/empyrical); forex path is in R-units and uses
    additive increments. The Rust port reconstructs this identically (parity).
    """
    n = len(df)
    if n < 3 or not trades:
        return 0.0
    close = df["close"].to_numpy(dtype=float)
    realized = np.zeros(n)
    unreal = np.zeros(n)
    if use_forex:
        sl_dist = _bt.SL_PERCENTAGE * _bt.PIP_SIZE
        rrr = (_bt.TP_PERCENTAGE / _bt.SL_PERCENTAGE) if _bt.SL_PERCENTAGE else 1.0
    for side, ent, exi, ep, xp, qty, pnl in trades:
        e, x = int(ent), int(exi)
        if x < e:
            continue
        if x < n:
            realized[x:] += pnl
        d = 1.0 if side == 1 else -1.0
        for i in range(e, min(x, n)):
            if use_forex:
                r = (d * (close[i] - ep) / sl_dist) if sl_dist else 0.0
                unreal[i] = min(max(r, -1.0), rrr) * _bt.POSITION_SIZE
            else:
                unreal[i] = d * qty * (close[i] - ep)
    if use_forex:
        bar_eq = realized + unreal               # R-units around 0
        ret = np.diff(bar_eq)                     # additive R increments
    else:
        bar_eq = account_size + realized + unreal
        ret = np.diff(bar_eq) / bar_eq[:-1]       # compounding (pct-change)
    ret = ret[np.isfinite(ret)]
    if ret.size < 2 or ret.std() <= 0:
        return 0.0
    # periods per Julian year from the median bar spacing (seconds). The bar
    # timestamps may live in the index or a `time` column, as (tz-aware)
    # datetime or unix; `.asi8` gives UTC nanoseconds in all cases.
    ts = None
    if isinstance(df.index, pd.DatetimeIndex):
        ts = df.index
    elif "time" in df.columns:
        col = df["time"]
        ts = pd.DatetimeIndex(col) if pd.api.types.is_datetime64_any_dtype(col) \
             else pd.to_datetime(col.to_numpy(), unit="s", utc=True)
    sec = float(np.median(np.diff(np.asarray(ts.asi8))) / 1e9) \
          if ts is not None and len(ts) > 1 else 0.0
    ppy = (31_557_600.0 / sec) if sec > 0 else 1.0
    return float(ret.mean() / ret.std() * np.sqrt(ppy))


def prettyprint(tag, m, lb=None):
    lb_note  = f"(LB {lb}) " if lb else ""
    rrr_note = f"  RRR:{m['RRR']}" if 'RRR' in m else ""
    if _bt.FOREX_MODE:
        print(f"{tag:>8} {lb_note}| Trades:{m['Trades']:4d}  "
              f"ROI:{m['ROI']:7.2f}R  PF:{m['PF']:6.2f}  Shp:{m['Sharpe']:6.2f}  "
              f"Win:{m['WinRate']*100:6.2f}%  Exp:{m['Exp']:7.2f}R  "
              f"MaxDD:{m['MaxDrawdown']:7.2f}R{rrr_note}")
    else:
        print(f"{tag:>8} {lb_note}| Trades:{m['Trades']:4d}  "
              f"ROI:${m['ROI'] * _bt.ACCOUNT_SIZE:,.2f}  "
              f"PF:{m['PF']:6.2f}  Shp:{m['Sharpe']:6.2f}  "
              f"Win:{m['WinRate']*100:6.2f}%  "
              f"Exp:${m['Exp'] * _bt.ACCOUNT_SIZE:,.2f}  "
              f"MaxDD:${m['MaxDrawdown'] * _bt.ACCOUNT_SIZE:,.2f}{rrr_note}")
