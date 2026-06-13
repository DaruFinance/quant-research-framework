"""Volume indicators (item #2, v0.6.0). Python reference for
`src/volume.rs`. No look-ahead: every value at index i uses only bars 0..=i.
Seeds and session-reset boundaries are specified IDENTICALLY to the Rust
module so the two engines match within f64 noise (parity gate 1e-3 paper /
1e-9 default in tools/parity_volume.py).

Conventions (kept in lockstep with the Rust mirror):
  * Volume SMA / z-score: trailing window ENDING at i; NaN before full
    window; z-score uses POPULATION std (ddof=0) via the SAME naive two-pass
    loop as Rust, guarded on var <= VAR_FLOOR (not exact ==0).
  * Volume EMA: ewm(span, adjust=False) seeded at volume[0].
  * OBV: OBV[0] = 0 (TradingView); tie close==close.shift adds 0.
  * A/D: AD[0] = mfm[0]*vol[0]; mfm = 0 when high==low.
  * MFI: NaN until `length` typical-price deltas exist (i < length);
    tie tp==tp.shift contributes to neither; flat window
    (pos_sum==0 and neg_sum==0) returns NaN (TradingView-faithful).
  * VWAP rolling-N: window ENDING at i; NaN before full window or zero
    window-volume.
  * VWAP session-anchored: cumulative, reset when `reset[i]` is True.

CANONICAL IMPORT: `from backtester.volume_indicators import ...` (NOT via the
deprecated indicators_tradingview shim).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Magnitude floor for the z-score zero-variance guard. Matches Rust VAR_FLOOR.
VAR_FLOOR = 1e-300


def typical_price(df: pd.DataFrame) -> pd.Series:
    return (df['high'] + df['low'] + df['close']) / 3.0


def _vol(df: pd.DataFrame) -> pd.Series:
    return df['volume'].astype(float)


def volume_sma(df: pd.DataFrame, length: int) -> np.ndarray:
    length = int(length)
    return _vol(df).rolling(window=length, min_periods=length).mean().to_numpy()


def volume_ema(df: pd.DataFrame, span: int) -> np.ndarray:
    return _vol(df).ewm(span=int(span), adjust=False).mean().to_numpy()


def relative_volume(df: pd.DataFrame, length: int) -> np.ndarray:
    v = _vol(df).to_numpy()
    sma = volume_sma(df, length)
    out = np.full(len(v), np.nan)
    mask = ~np.isnan(sma) & (sma != 0.0)
    out[mask] = v[mask] / sma[mask]
    return out


def volume_zscore(df: pd.DataFrame, length: int) -> np.ndarray:
    """Naive two-pass population z-score, IDENTICAL to src/volume.rs. We do
    NOT use pandas.std() here: pandas may yield std==0 exactly while the Rust
    naive loop yields a tiny nonzero (or vice-versa), desyncing the zero-var
    guard. Computing the same naive var on both sides removes that landmine."""
    length = int(length)
    v = _vol(df).to_numpy()
    n = len(v)
    out = np.full(n, np.nan)
    if length == 0 or n < length:
        return out
    for i in range(length - 1, n):
        window = v[i + 1 - length:i + 1]
        mean = window.sum() / length
        var = ((window - mean) ** 2).sum() / length
        if var <= VAR_FLOOR:
            continue
        out[i] = (v[i] - mean) / np.sqrt(var)
    return out


def obv(df: pd.DataFrame) -> np.ndarray:
    close = df['close'].to_numpy()
    v = _vol(df).to_numpy()
    n = len(close)
    out = np.zeros(n)
    if n == 0:
        return out
    out[0] = 0.0  # TradingView seed
    for i in range(1, n):
        d = close[i] - close[i - 1]
        if d > 0.0:
            out[i] = out[i - 1] + v[i]
        elif d < 0.0:
            out[i] = out[i - 1] - v[i]
        else:
            out[i] = out[i - 1]
    return out


def ad_line(df: pd.DataFrame) -> np.ndarray:
    high = df['high'].to_numpy()
    low = df['low'].to_numpy()
    close = df['close'].to_numpy()
    v = _vol(df).to_numpy()
    rng = high - low
    with np.errstate(divide='ignore', invalid='ignore'):
        mfm = ((close - low) - (high - close)) / rng
    mfm = np.where(rng == 0.0, 0.0, mfm)
    return np.cumsum(mfm * v)


def mfi(df: pd.DataFrame, length: int) -> np.ndarray:
    length = int(length)
    tp = typical_price(df).to_numpy()
    v = _vol(df).to_numpy()
    n = len(tp)
    out = np.full(n, np.nan)
    if n < 2 or length == 0:
        return out
    pos = np.zeros(n)
    neg = np.zeros(n)
    rmf = tp * v
    for i in range(1, n):
        if tp[i] > tp[i - 1]:
            pos[i] = rmf[i]
        elif tp[i] < tp[i - 1]:
            neg[i] = rmf[i]
    for i in range(length, n):
        lo = i + 1 - length
        pos_sum = pos[lo:i + 1].sum()
        neg_sum = neg[lo:i + 1].sum()
        if pos_sum == 0.0 and neg_sum == 0.0:
            out[i] = np.nan  # flat window: TradingView returns na
        elif neg_sum == 0.0:
            out[i] = 100.0
        else:
            mr = pos_sum / neg_sum
            out[i] = 100.0 - 100.0 / (1.0 + mr)
    return out


def vwap_rolling(df: pd.DataFrame, length: int) -> np.ndarray:
    length = int(length)
    tp = typical_price(df)
    v = _vol(df)
    pv = (tp * v).rolling(window=length, min_periods=length).sum()
    vv = v.rolling(window=length, min_periods=length).sum()
    out = (pv / vv)
    out = out.where(vv != 0.0, other=np.nan)
    return out.to_numpy()


def vwap_session(df: pd.DataFrame, reset: np.ndarray) -> np.ndarray:
    """Session-anchored cumulative VWAP. `reset[i]` True restarts the
    accumulator AT i (bar i included). reset[0] is treated as an open.
    NaN only if cumulative session volume is 0 (degenerate)."""
    tp = typical_price(df).to_numpy()
    v = _vol(df).to_numpy()
    n = len(tp)
    out = np.full(n, np.nan)
    pv = 0.0
    vv = 0.0
    for i in range(n):
        if reset[i] or i == 0:
            pv = 0.0
            vv = 0.0
        pv += tp[i] * v[i]
        vv += v[i]
        if vv != 0.0:
            out[i] = pv / vv
    return out


def ny_session_resets(df: pd.DataFrame, anchor_hour: int) -> np.ndarray:
    """Reset flags from the already-NY-localised 'time' column (load_ohlc
    tz_converts to America/New_York). reset[i] True when the NY calendar
    date changes vs i-1, OR bar i crosses the anchor hour. reset[0]=True.
    Mirrors src/volume.rs::ny_session_resets exactly."""
    t = df['time']
    hour = t.dt.hour.to_numpy()
    date = t.dt.normalize().to_numpy()  # NY-midnight per row
    n = len(t)
    out = np.zeros(n, dtype=bool)
    if n == 0:
        return out
    out[0] = True
    for i in range(1, n):
        date_changed = date[i] != date[i - 1]
        crossed = (hour[i - 1] < anchor_hour) and (hour[i] >= anchor_hour)
        out[i] = bool(date_changed or crossed)
    return out
