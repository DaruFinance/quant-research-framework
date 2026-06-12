#!/usr/bin/env python3
"""Volume strategy family (item #2, v0.5.0 Python). Python mirrors of the four
Rust volume examples. Each is a `create_raw_signals`-shaped function:
(df, lb) -> int8 array of -1/0/+1. No look-ahead: every signal at index i uses
only data evaluated at bar i-1 (and earlier).

Requires a df with a 'volume' column (load_ohlc auto-includes it when the CSV
has a 6th 'volume' column; see backtester/__init__.py).

    python examples/volume_strategies.py data/volume_fixture.csv
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from backtester.volume_indicators import (  # noqa: E402
    ny_session_resets,
    obv,
    relative_volume,
    volume_sma,
    vwap_session,
)

FAST, SLOW, VOL_LEN, K = 12, 26, 20, 1.5
ANCHOR_HOUR, BAND = 0, 0.01
LOOK = 14
RELVOL_K, CHANNEL = 2.0, 20


def vol_confirmed_ema_cross(df: pd.DataFrame, lb: int) -> np.ndarray:
    n = len(df)
    raw = np.zeros(n, dtype=np.int8)
    if n < 3:
        return raw
    close = df['close']
    fast = close.ewm(span=FAST, adjust=False).mean().to_numpy()
    slow = close.ewm(span=SLOW, adjust=False).mean().to_numpy()
    vol = df['volume'].to_numpy()
    vsma = volume_sma(df, VOL_LEN)
    for i in range(2, n):
        if np.isnan(vsma[i - 1]) or not (vol[i - 1] > K * vsma[i - 1]):
            continue
        cu = fast[i - 1] > slow[i - 1] and fast[i - 2] <= slow[i - 2]
        cd = fast[i - 1] < slow[i - 1] and fast[i - 2] >= slow[i - 2]
        if cu:
            raw[i] = 1
        elif cd:
            raw[i] = -1
    return raw


def vwap_mean_reversion(df: pd.DataFrame, lb: int) -> np.ndarray:
    n = len(df)
    raw = np.zeros(n, dtype=np.int8)
    if n < 2:
        return raw
    reset = ny_session_resets(df, ANCHOR_HOUR)
    vwap = vwap_session(df, reset)
    close = df['close'].to_numpy()
    for i in range(1, n):
        v1 = vwap[i - 1]
        if np.isnan(v1) or v1 == 0.0:
            continue
        dev = (close[i - 1] - v1) / v1
        if dev <= -BAND:
            raw[i] = 1
        elif dev >= BAND:
            raw[i] = -1
    return raw


def obv_divergence(df: pd.DataFrame, lb: int) -> np.ndarray:
    n = len(df)
    raw = np.zeros(n, dtype=np.int8)
    if n < LOOK + 2:
        return raw
    o = obv(df)
    close = df['close'].to_numpy()
    for i in range(LOOK + 1, n):
        p_now, p_then = close[i - 1], close[i - 1 - LOOK]
        o_now, o_then = o[i - 1], o[i - 1 - LOOK]
        if p_now < p_then and o_now > o_then:
            raw[i] = 1
        elif p_now > p_then and o_now < o_then:
            raw[i] = -1
    return raw


def relvol_breakout(df: pd.DataFrame, lb: int) -> np.ndarray:
    n = len(df)
    raw = np.zeros(n, dtype=np.int8)
    if n < CHANNEL + 3:
        return raw
    relvol = relative_volume(df, VOL_LEN)
    high = df['high'].to_numpy()
    low = df['low'].to_numpy()
    close = df['close'].to_numpy()
    for i in range(CHANNEL + 2, n):
        rv = relvol[i - 1]
        if np.isnan(rv) or rv < RELVOL_K:
            continue
        lo = i - 1 - CHANNEL
        hi_end = i - 1  # bars[lo:hi_end], excludes breakout bar i-1
        hh = high[lo:hi_end].max()
        ll = low[lo:hi_end].min()
        if close[i - 1] > hh:
            raw[i] = 1
        elif close[i - 1] < ll:
            raw[i] = -1
    return raw


if __name__ == "__main__":
    from backtester import load_ohlc
    path = sys.argv[1] if len(sys.argv) > 1 else "data/volume_fixture.csv"
    df = load_ohlc(path)
    for name, fn in [
        ("vol_confirmed_ema_cross", vol_confirmed_ema_cross),
        ("vwap_mean_reversion", vwap_mean_reversion),
        ("obv_divergence", obv_divergence),
        ("relvol_breakout", relvol_breakout),
    ]:
        sig = fn(df, 50)
        print(f"{name}: {int((sig != 0).sum())} signals "
              f"(+{int((sig == 1).sum())} / -{int((sig == -1).sum())})")
