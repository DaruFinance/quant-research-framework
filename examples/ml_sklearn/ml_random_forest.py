#!/usr/bin/env python3
"""ML strategy example: scikit-learn RandomForestClassifier on the same
feature set as the LogisticRegression sibling. Demonstrates that
swapping the model is a one-line change once the feature/labelling
pipeline is in place.

Run:
    BT_CSV=data_SOLUSDT_1h.csv \\
        python examples/ml_sklearn/ml_random_forest.py
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
from sklearn.ensemble import RandomForestClassifier

import backtester as bt
from backtester.indicators import compute_atr, compute_rsi


PRED_COL     = "pred"
LONG_THRESH  = 0.55
SHORT_THRESH = 0.45
HORIZON      = 5
N_ESTIMATORS = 50
MAX_DEPTH    = 4


def _build_features(df):
    f = pd.DataFrame(index=df.index)
    rets = df["close"].pct_change()
    f["ret_1"]  = rets
    f["ret_5"]  = rets.rolling(5).sum()
    f["ret_20"] = rets.rolling(20).sum()
    f["vol_20"] = rets.rolling(20).std()
    f["rsi_14"] = compute_rsi(df, 14)
    f["atr_14"] = compute_atr(df, 14)
    f["hl_range"] = (df["high"] - df["low"]) / df["close"]
    return f.shift(1)


def _attach_predictions(df: pd.DataFrame) -> pd.DataFrame:
    if PRED_COL in df.columns:
        return df

    feats = _build_features(df)
    fwd   = df["close"].shift(-HORIZON)
    label = (fwd > df["close"]).astype(int)

    n           = len(df)
    oos_candles = bt.OOS_CANDLES
    is_candles  = bt.BACKTEST_CANDLES
    oos_start   = max(0, n - oos_candles)
    is_start    = max(0, oos_start - is_candles)

    X_train = feats.iloc[is_start:oos_start].dropna()
    y_train = label.iloc[X_train.index].dropna()
    common  = X_train.index.intersection(y_train.index)
    X_train, y_train = X_train.loc[common], y_train.loc[common]

    if len(X_train) < 100 or y_train.nunique() < 2:
        df = df.copy()
        df[PRED_COL] = 0.5
        return df

    model = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        random_state=42,
        n_jobs=1,         # leave parallelism to the batch_runner
    )
    model.fit(X_train.values, y_train.values)

    X_full = feats.fillna(0.0)
    proba  = model.predict_proba(X_full.values)[:, 1]
    df = df.copy()
    df[PRED_COL] = proba
    return df


def ml_rf_signals(df: pd.DataFrame, lb: int) -> np.ndarray:
    df = _attach_predictions(df)
    pred_prev = df[PRED_COL].shift(1).values
    raw = np.zeros(len(df), dtype=np.int8)
    raw[pred_prev >= LONG_THRESH]  =  1
    raw[pred_prev <= SHORT_THRESH] = -1
    return raw


bt.create_raw_signals = ml_rf_signals
bt.LOOKBACK_RANGE = (50, 52)
bt.BACKTEST_CANDLES = 30_000
bt.OOS_CANDLES      = 10_000
bt.ORIGINAL_OOS     = 10_000

if __name__ == "__main__":
    bt.main()
