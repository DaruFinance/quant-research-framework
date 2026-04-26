#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ML signal example (pre-computed predictions).

Pattern: train your model offline, attach its predictions to the OHLC frame
as a column (here: ``pred``), and let the strategy function threshold that
column into long/short intents. The backtester never sees your model — it
just consumes a numpy array of signals like every other strategy.

This is the recommended path when you can train ahead of time:
  * fast (no per-bar inference inside the inner loop)
  * trivially compatible with any framework (sklearn, lightgbm, torch, jax,
    R, MATLAB, ONNX, ...) — you just need a CSV/Parquet column of scores
  * easy to audit: one column in, one signal out

Look-ahead discipline: the predictor's score for bar ``i`` must only use
information available at bar ``i-1`` or earlier. ``parse_signals`` reads
``raw[i]`` and trades at ``df.open[i]``, so any leakage in the training
features lands directly in your backtest.

Run:
    python examples/ml_precomputed/ml_precomputed.py
    python examples/ml_precomputed/ml_precomputed.py path/to/ohlc.csv
"""

import os, sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, _ROOT)

if len(sys.argv) > 1:
    os.environ["BT_CSV"] = sys.argv.pop(1)

import numpy as np                                        # noqa: E402
import pandas as pd                                       # noqa: E402
import backtester as bt                                   # noqa: E402


PRED_COL    = "pred"     # name of the column holding model scores
LONG_THRESH =  0.55      # score >= this => long intent
SHORT_THRESH = 0.45      # score <= this => short intent


def _ensure_predictions(df: pd.DataFrame) -> pd.DataFrame:
    """
    If ``df`` already carries a `pred` column we use it as-is. Otherwise we
    fall back to a tiny, deterministic stand-in so the example runs on any
    OHLC CSV without needing a model file. The stand-in is *not* a real
    strategy — it just demonstrates the data shape the backtester expects.
    """
    if PRED_COL in df.columns:
        return df

    # Stand-in predictor: 50-bar momentum z-score squashed into [0, 1].
    ret = df['close'].pct_change()
    mom = ret.rolling(50, min_periods=50).mean().shift(1)
    sd  = ret.rolling(50, min_periods=50).std().shift(1).replace(0, np.nan)
    z   = (mom / sd).fillna(0.0)
    df = df.copy()
    df[PRED_COL] = 1.0 / (1.0 + np.exp(-z * 2.0))   # logistic squashing
    return df


def ml_precomputed_signals(df: pd.DataFrame, lb: int) -> np.ndarray:
    """
    Strategy function. Reads ``df[PRED_COL]`` and turns it into a raw
    signal array. The ``lb`` argument is unused here — pre-computed
    predictions don't sweep a look-back. Set ``LOOKBACK_RANGE = (1, 2)`` in
    backtester to keep the optimiser from wasting work, or ignore it (the
    optimiser will still pick the single non-trivial point).
    """
    df = _ensure_predictions(df)
    pred_prev = df[PRED_COL].shift(1).values     # no look-ahead
    raw = np.zeros(len(df), dtype=np.int8)
    raw[pred_prev >= LONG_THRESH]  =  1
    raw[pred_prev <= SHORT_THRESH] = -1
    return raw


bt.create_raw_signals = ml_precomputed_signals

if __name__ == "__main__":
    bt.main()
