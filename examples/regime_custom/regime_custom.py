#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Custom regime-detector example.

The default regime detector is EMA-200 / 8-bar consistency producing three
labels (Uptrend, Downtrend, Ranging). The framework lets you swap in any
detector you want, including ML-based ones, by overriding two module-level
symbols on ``backtester``:

    bt.REGIME_LABELS  = [...]    # length 2..5, the canonical label set
    bt.detect_regimes = my_fn    # (df) -> pd.Series[label]

Once those are set, every code path that looks at regimes (single-run with
USE_REGIME_SEG, WFO+regime in walk_forward, evaluate_filters, etc.)
picks them up automatically.

Two demos are included below. Pick one by setting ``DEMO`` near the top.

Run:
    python examples/regime_custom/regime_custom.py
    python examples/regime_custom/regime_custom.py path/to/ohlc.csv
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


DEMO = "vol4"   # one of: "vol2", "vol4", "ml5"


# --- Demo 1: 2-regime volatility detector ---------------------------------
VOL2_LABELS = ['Calm', 'Volatile']

def detect_regimes_vol2(df: pd.DataFrame) -> pd.Series:
    """Two regimes by 50-bar realised vol vs its 250-bar median."""
    ret = df['close'].pct_change()
    sd  = ret.rolling(50, min_periods=50).std().shift(1)
    cutoff = sd.rolling(250, min_periods=250).median()
    out = pd.Series(VOL2_LABELS[0], index=df.index)
    out.loc[sd > cutoff] = VOL2_LABELS[1]
    return out


# --- Demo 2: 4-regime trend × volatility detector -------------------------
VOL4_LABELS = ['CalmUp', 'CalmDown', 'VolUp', 'VolDown']

def detect_regimes_vol4(df: pd.DataFrame) -> pd.Series:
    ret = df['close'].pct_change()
    trend = (df['close'] - df['close'].shift(50)).shift(1)
    sd    = ret.rolling(50, min_periods=50).std().shift(1)
    cutoff = sd.rolling(250, min_periods=250).median()
    is_vol  = sd > cutoff
    is_up   = trend > 0
    out = pd.Series(VOL4_LABELS[0], index=df.index)   # CalmUp default
    out.loc[(~is_vol) & (~is_up)] = VOL4_LABELS[1]    # CalmDown
    out.loc[( is_vol) & ( is_up)] = VOL4_LABELS[2]    # VolUp
    out.loc[( is_vol) & (~is_up)] = VOL4_LABELS[3]    # VolDown
    return out


# --- Demo 3: 5-regime ML-style detector (k-means stand-in) ----------------
ML5_LABELS = ['R0', 'R1', 'R2', 'R3', 'R4']

class TinyKMeansLikeDetector:
    """
    Stand-in for an ML clusterer: bins the previous bar's (return, vol)
    pair into 5 quantile buckets. Replace this with a fitted
    sklearn.cluster.KMeans / HMM / mixture model — the contract is the
    same: a function (df) -> pd.Series[label].
    """
    def __init__(self):
        self._labels = ML5_LABELS

    def __call__(self, df: pd.DataFrame) -> pd.Series:
        ret  = df['close'].pct_change().shift(1)
        sd   = df['close'].pct_change().rolling(50, min_periods=50).std().shift(1)
        # Combine into a single score, bucket into 5 quantiles.
        score = (ret.fillna(0) / sd.replace(0, np.nan)).fillna(0)
        try:
            buckets = pd.qcut(score, q=5, labels=self._labels, duplicates='drop')
        except ValueError:
            buckets = pd.Series(self._labels[0], index=df.index)
        return buckets.astype(object).fillna(self._labels[0])


# --- Pick a demo and wire it in -------------------------------------------
if DEMO == "vol2":
    bt.REGIME_LABELS  = VOL2_LABELS
    bt.detect_regimes = detect_regimes_vol2
elif DEMO == "vol4":
    bt.REGIME_LABELS  = VOL4_LABELS
    bt.detect_regimes = detect_regimes_vol4
elif DEMO == "ml5":
    bt.REGIME_LABELS  = ML5_LABELS
    bt.detect_regimes = TinyKMeansLikeDetector()
else:
    raise ValueError(f"Unknown DEMO: {DEMO!r}")


if __name__ == "__main__":
    # Make sure regime segmentation is on for this example.
    bt.USE_REGIME_SEG = True
    bt.main()
