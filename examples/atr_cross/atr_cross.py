#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Example strategy: ATR-cross with an RSI confluence filter.

Primary signal: 3-bar SMA of ATR(lb) crosses length-50 EMA of ATR(lb).
    - cross up   -> go long
    - cross down -> go short
Confluence filter: RSI(14) on the previous bar must be >= 50, else drop
the signal.

This mirrors the proprietary spec `ATR_x_EMA50_RSIge50` from
run_strategies.py, but written the way an end user would: one file, no
framework metadata, indicators imported from the bundled
indicators_tradingview.py. The only thing we override in backtester is
`create_raw_signals` — everything else (IS/OOS split, smart optimiser,
walk-forward, robustness, Monte Carlo, trade export) is reused as-is.

Run from the repo root:

    python examples/atr_cross/atr_cross.py
    python examples/atr_cross/atr_cross.py path/to/ohlc.csv
"""

import os, sys

# Resolve the project root and expose it on sys.path so `import backtester`
# works regardless of where the user invokes this script from.
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, _ROOT)

# Let users pass the CSV as a positional arg; fall through to BT_CSV, then
# to backtester's built-in default ("data/your_ohlc.csv").
if len(sys.argv) > 1:
    os.environ["BT_CSV"] = sys.argv.pop(1)

import numpy as np                                         # noqa: E402
import pandas as pd                                        # noqa: E402
import backtester as bt                                    # noqa: E402
from indicators_tradingview import compute_atr, compute_rsi  # noqa: E402


# ---------------------------------------------------------------------------
# Strategy
# ---------------------------------------------------------------------------
ATR_PARTNER   = 50     # slow ATR-EMA length (the "EMA50" in the spec name)
RSI_LEN       = 14     # classic RSI window for the confluence
RSI_THRESHOLD = 50.0


def atr_cross_rsi(df: pd.DataFrame, lb: int) -> np.ndarray:
    """
    Raw-signals function. Must return a 1-D numpy int8 array of length
    `len(df)` where each element is:
        +1  -> long intent at this bar
        -1  -> short intent at this bar
         0  -> no signal (indicator not warmed up, or no crossover)

    `raw[i]` may only use data from bar `i-1` or earlier (no look-ahead).
    The backtester will flip-detect these via parse_signals and execute
    trades at `df.open[i]`.
    """
    # 1) Primary: ATR on (high/low/close) with length `lb`, then
    #    fast = 3-bar SMA of ATR, slow = 50-bar EMA of ATR.
    atr = compute_atr(df, lb)
    fast = atr.rolling(window=3, min_periods=3).mean()
    slow = atr.ewm(span=ATR_PARTNER, adjust=False).mean()

    # 2) Cross-events on the previous bar (no look-ahead).
    fast_prev, slow_prev = fast.shift(1), slow.shift(1)
    fast_prev2, slow_prev2 = fast.shift(2), slow.shift(2)
    cross_up   = (fast_prev >  slow_prev) & (fast_prev2 <= slow_prev2)
    cross_down = (fast_prev <  slow_prev) & (fast_prev2 >= slow_prev2)

    # 3) Confluence: RSI(14) on the previous bar >= 50.
    rsi_prev = compute_rsi(df, RSI_LEN).shift(1)
    rsi_ok   = (rsi_prev >= RSI_THRESHOLD)

    raw = np.zeros(len(df), dtype=np.int8)
    raw[(cross_up   & rsi_ok).fillna(False).values] =  1
    raw[(cross_down & rsi_ok).fillna(False).values] = -1
    return raw


# Plug the strategy in. backtester.create_raw_signals is called from
# classic_single_run / optimiser / walk_forward / run_robustness_tests, so
# a single module-level reassignment re-routes every call site.
bt.create_raw_signals = atr_cross_rsi

if __name__ == "__main__":
    bt.main()
