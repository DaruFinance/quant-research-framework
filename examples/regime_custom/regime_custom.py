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

Three demos are included below. Pick one by setting ``DEMO`` near the top.

Run:
    python examples/regime_custom/regime_custom.py
    python examples/regime_custom/regime_custom.py path/to/ohlc.csv
"""

import os, sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, _ROOT)

import numpy as np                                        # noqa: E402
import pandas as pd                                       # noqa: E402
import backtester as bt                                   # noqa: E402
from backtester.invariants import registers_invariant      # noqa: E402


DEMO = "vol4"   # one of: "vol2", "vol4", "ml5"


# --- Demo 1: 2-regime volatility detector ---------------------------------
VOL2_LABELS = ['Calm', 'Volatile']

@registers_invariant(name="example_vol2", data_kind="ohlc_df")
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

@registers_invariant(name="example_vol4", data_kind="ohlc_df")
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
    pair using its expanding historical percentile. At bar i, both the
    score and its reference distribution use prices through i-1 only.
    A replacement clusterer must also fit only on already available data.
    """
    def __init__(self):
        self._labels = ML5_LABELS

    def __call__(self, df: pd.DataFrame) -> pd.Series:
        ret = df['close'].pct_change(fill_method=None)
        sd = ret.rolling(50, min_periods=50).std()
        score = ret / sd.replace(0, np.nan)
        # Expanding.rank uses an ordered window, avoiding a quadratic
        # expanding.apply scan. Shift the entire fit and score together.
        rank = score.expanding(min_periods=50).rank(pct=True).shift(1)
        buckets = pd.cut(rank, bins=[0, .2, .4, .6, .8, 1],
                         labels=self._labels, include_lowest=True)
        return buckets.astype(object).fillna(self._labels[0])


@registers_invariant(name="ml5_quantile", data_kind="ohlc_df")
def detect_regimes_ml5(df: pd.DataFrame) -> pd.Series:
    return TinyKMeansLikeDetector()(df)


def main():
    if len(sys.argv) > 1:
        # backtester has already imported its defaults; update the runtime
        # path as well as the environment used by callers.
        bt.CSV_FILE = os.environ["BT_CSV"] = sys.argv[1]
    demos = {
        "vol2": (VOL2_LABELS, detect_regimes_vol2),
        "vol4": (VOL4_LABELS, detect_regimes_vol4),
        "ml5": (ML5_LABELS, detect_regimes_ml5),
    }
    if DEMO not in demos:
        raise ValueError(f"Unknown DEMO: {DEMO!r}")
    bt.REGIME_LABELS, bt.detect_regimes = demos[DEMO]
    bt.USE_REGIME_SEG = True
    bt.main()


if __name__ == "__main__":
    main()
