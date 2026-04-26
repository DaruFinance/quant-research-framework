#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ML signal example (per-bar callback).

Pattern: keep the model in memory, extract a feature vector at each bar,
call ``predict(features)`` and emit a long/short intent. Use this when
your model is online (state changes through the run) or when you want
training and inference to live in the same process.

Trade-offs vs. the pre-computed path (`examples/ml_precomputed`):
  * Slower — Python-level inference inside the inner loop.
  * Trickier look-ahead discipline — features for bar ``i`` must come from
    ``df[: i]`` only. The helper ``extract_window_features`` enforces this.
  * More flexible — works with any predictor that has a ``.predict(x)``
    method (sklearn, pytorch, your own class).

The demo predictor below is a hand-coded linear model so this example has
zero extra dependencies. Swap it for ``sklearn.linear_model.LogisticRegression``
or any callable returning a scalar/probability and the rest works as-is.

Run:
    python examples/ml_callback/ml_callback.py
    python examples/ml_callback/ml_callback.py path/to/ohlc.csv
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


# --- Predictor protocol ----------------------------------------------------
# Anything that exposes ``predict(features: np.ndarray) -> float`` (one
# probability/score per call) plugs straight in. Replace the demo below with
# your own model class — load weights in __init__, do inference in predict.

class TinyMomentumModel:
    """Deterministic linear model: weighted sum of normalised lagged
    returns squashed through a logistic. Stand-in for a real estimator."""
    def __init__(self, weights=None):
        self.weights = np.asarray(weights if weights is not None
                                  else [0.6, 0.3, 0.1], dtype=float)

    def predict(self, features: np.ndarray) -> float:
        x = np.asarray(features, dtype=float)
        # Pad / truncate so this works for any feature length.
        n = min(len(x), len(self.weights))
        z = float(np.dot(x[:n], self.weights[:n]))
        return 1.0 / (1.0 + np.exp(-z))


MODEL = TinyMomentumModel()
LONG_THRESH  = 0.55
SHORT_THRESH = 0.45


def extract_window_features(df: pd.DataFrame, i: int, lb: int) -> np.ndarray:
    """
    Build a feature vector for bar ``i`` using only data from bars
    ``[i - lb, i - 1]`` (no look-ahead). Returns ``np.empty`` when not
    enough history exists yet — caller should treat that as "no signal".
    """
    if i <= lb:
        return np.empty(0, dtype=float)
    closes = df['close'].values[i - lb : i]
    if len(closes) < 2:
        return np.empty(0, dtype=float)
    rets = np.diff(np.log(closes))
    sd = rets.std()
    if sd <= 0 or not np.isfinite(sd):
        return np.empty(0, dtype=float)
    z = rets[-3:] / sd                                # last 3 z-scored rets
    if z.size < 3:
        z = np.concatenate([np.zeros(3 - z.size), z])
    return z


def ml_callback_signals(df: pd.DataFrame, lb: int) -> np.ndarray:
    """
    Per-bar inference loop. Slower than vectorised strategies but the only
    way to support online / stateful predictors.
    """
    raw = np.zeros(len(df), dtype=np.int8)
    for i in range(len(df)):
        feats = extract_window_features(df, i, lb)
        if feats.size == 0:
            continue
        score = MODEL.predict(feats)
        if score >= LONG_THRESH:
            raw[i] =  1
        elif score <= SHORT_THRESH:
            raw[i] = -1
    return raw


bt.create_raw_signals = ml_callback_signals

if __name__ == "__main__":
    bt.main()
