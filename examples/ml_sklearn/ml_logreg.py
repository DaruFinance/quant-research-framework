#!/usr/bin/env python3
"""ML strategy example: scikit-learn LogisticRegression on lagged
return + RSI + ATR features. Uses the *precomputed* contract — train
once on the IS slice, predict probabilities for every bar of the full
series, attach as a `pred` column, then threshold inside the
strategy function.

Why this pattern:
  - sklearn fit + predict is far faster than per-bar callback inference.
  - Train-on-IS / predict-everywhere is the right discipline: the
    strategy function only ever reads `pred` shifted by one bar
    (`shift(1)`), so the model's fit-window seeing future test bars
    would still be a look-ahead bug — but we control that by fitting
    only on the IS slice the engine declares.
  - All features the model trains on are themselves shifted by one bar
    so each bar's feature row contains only data available at that
    bar's prior close.

Run:
    BT_CSV=data_SOLUSDT_1h.csv \\
        python examples/ml_sklearn/ml_logreg.py
"""
import os, sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent.parent
sys.path.insert(0, str(_ROOT))

if len(sys.argv) > 1:
    os.environ["BT_CSV"] = sys.argv.pop(1)

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

import backtester as bt
from backtester.indicators import compute_atr, compute_rsi


PRED_COL     = "pred"
LONG_THRESH  = 0.55
SHORT_THRESH = 0.45
HORIZON      = 5      # predict sign of close[i+HORIZON] - close[i]
LOOKBACK     = 20     # bars of history per feature row


def _build_features(df: pd.DataFrame) -> pd.DataFrame:
    """All features must be derivable from data at or before bar i.
    We then shift(1) before the model sees them so prediction at bar i
    uses only data available at bar i-1's close."""
    f = pd.DataFrame(index=df.index)
    rets = df["close"].pct_change()
    f["ret_1"]  = rets
    f["ret_5"]  = rets.rolling(5).sum()
    f["ret_20"] = rets.rolling(20).sum()
    f["vol_20"] = rets.rolling(20).std()
    f["rsi_14"] = compute_rsi(df, 14)
    f["atr_14"] = compute_atr(df, 14)
    f["hl_range"] = (df["high"] - df["low"]) / df["close"]
    return f.shift(1)            # ★ no look-ahead: features at i use up to i-1


def _label(df: pd.DataFrame, horizon: int) -> pd.Series:
    """Binary label: 1 if close[i+horizon] > close[i] (will go up)."""
    fwd = df["close"].shift(-horizon)
    return (fwd > df["close"]).astype(int)


def _attach_predictions(df: pd.DataFrame) -> pd.DataFrame:
    """Train logreg on the IS portion only (the engine's BACKTEST_CANDLES
    convention: bars [is_start, oos_start) are IS), predict probabilities
    for every bar, attach as `pred`. The strategy function then shifts
    `pred` by one before thresholding for the look-ahead invariant.
    """
    if PRED_COL in df.columns:
        return df

    feats = _build_features(df)
    label = _label(df, HORIZON)

    # IS slice: same convention as bt.classic_single_run
    n            = len(df)
    oos_candles  = bt.OOS_CANDLES
    is_candles   = bt.BACKTEST_CANDLES
    oos_start    = max(0, n - oos_candles)
    is_start     = max(0, oos_start - is_candles)

    X_train = feats.iloc[is_start:oos_start].dropna()
    y_train = label.iloc[X_train.index].dropna()
    common  = X_train.index.intersection(y_train.index)
    X_train, y_train = X_train.loc[common], y_train.loc[common]

    if len(X_train) < 100 or y_train.nunique() < 2:
        # Degenerate fixture — fall back to constant 0.5 so the engine
        # still has a well-defined `pred` column. Pure stand-in.
        df = df.copy()
        df[PRED_COL] = 0.5
        return df

    model = Pipeline([
        ("scale", StandardScaler()),
        ("lr",    LogisticRegression(max_iter=200, C=1.0)),
    ])
    model.fit(X_train.values, y_train.values)

    X_full = feats.fillna(0.0)
    proba  = model.predict_proba(X_full.values)[:, 1]
    df = df.copy()
    df[PRED_COL] = proba
    return df


def ml_logreg_signals(df: pd.DataFrame, lb: int) -> np.ndarray:
    """Strategy contract: long when prev-bar pred >= LONG_THRESH; short
    when <= SHORT_THRESH; flat otherwise. lb is unused (predictions
    don't sweep a lookback)."""
    df = _attach_predictions(df)
    pred_prev = df[PRED_COL].shift(1).values
    raw = np.zeros(len(df), dtype=np.int8)
    raw[pred_prev >= LONG_THRESH]  =  1
    raw[pred_prev <= SHORT_THRESH] = -1
    return raw


bt.create_raw_signals = ml_logreg_signals
# Predictions don't sweep a lookback — pin LB to a single value to skip
# wasted optimiser cycles.
bt.LOOKBACK_RANGE = (50, 52)
# Default OOS_CANDLES (90,000) exceeds the bundled SOLUSDT 1h dataset
# (48,094 bars), which would zero out the IS slice. Smaller window so
# both IS and OOS fit the dataset.
bt.BACKTEST_CANDLES = 30_000
bt.OOS_CANDLES      = 10_000
bt.ORIGINAL_OOS     = 10_000

if __name__ == "__main__":
    bt.main()
