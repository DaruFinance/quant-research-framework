#!/usr/bin/env python3
"""ML regime detector example: scikit-learn KMeans on volatility +
momentum features. Replaces the framework's default 3-regime
EMA-200/8-bar detector with a learned 3-cluster classifier whose
labels (Calm / Trending / Choppy) come from the cluster centres'
ranking of mean realised volatility.

Why this pattern:
  - Demonstrates the regime-detector seam works with arbitrary
    sklearn estimators, not just hand-written rules.
  - The fit happens on the IS slice (the engine's BACKTEST_CANDLES
    bars before the OOS window); cluster assignment then runs over
    the full series. Any feature that uses data from bar i must come
    from data at or before bar i-1, enforced via shift(1).
  - Labels are remapped after fit so the same physical regime
    (e.g. "low-volatility, trending") gets the same string label
    across runs — KMeans's raw cluster IDs are arbitrary.

Run:
    BT_CSV=data_SOLUSDT_1h.csv \\
        python examples/ml_regime_kmeans/ml_regime_kmeans.py
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
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

import backtester as bt


REGIME_LABELS = ["Calm", "Trending", "Choppy"]


def _build_regime_features(df: pd.DataFrame) -> pd.DataFrame:
    """Volatility (rolling std), trend strength (rolling mean of
    abs returns / std), and high-low range. All shifted by one bar so
    bar i's label depends only on data up to and including bar i-1."""
    f = pd.DataFrame(index=df.index)
    rets = df["close"].pct_change(fill_method=None)
    f["vol_50"]  = rets.rolling(50).std()
    f["vol_200"] = rets.rolling(200).std()
    f["mom_50"]  = rets.rolling(50).sum()
    f["abs_mean_50"] = rets.abs().rolling(50).mean()
    f["range_50"] = (df["high"] - df["low"]).rolling(50).mean() / df["close"]
    return f.shift(1)


def _label_from_centres(model: KMeans) -> dict:
    """Remap KMeans's cluster IDs to stable label names by sorting
    the centres on the *first* feature (vol_50): lowest → 'Calm',
    highest → 'Choppy', middle → 'Trending'. Any 3-cluster permutation
    therefore gets the same label assignment across runs."""
    centres = model.cluster_centers_
    order   = np.argsort(centres[:, 0])
    return {int(order[0]): "Calm",
            int(order[1]): "Trending",
            int(order[2]): "Choppy"}


def ml_kmeans_detect_regimes(df: pd.DataFrame) -> pd.Series:
    """Fit KMeans on the IS slice; assign labels for every bar.
    Returns a pd.Series of regime label strings indexed identically
    to df."""
    feats = _build_regime_features(df)

    # IS slice for fit (same convention as classic_single_run)
    n           = len(df)
    oos_candles = bt.OOS_CANDLES
    is_candles  = bt.BACKTEST_CANDLES
    oos_start   = max(0, n - oos_candles)
    is_start    = max(0, oos_start - is_candles)

    X_train = feats.iloc[is_start:oos_start].dropna()

    if len(X_train) < 100:
        # Degenerate fixture — return a constant label so the engine
        # still has well-defined regimes. Demo-only fallback.
        return pd.Series(["Calm"] * n, index=df.index)

    pipe = Pipeline([
        ("scale",  StandardScaler()),
        ("kmeans", KMeans(n_clusters=3, n_init=10, random_state=42)),
    ])
    pipe.fit(X_train.values)
    label_map = _label_from_centres(pipe.named_steps["kmeans"])

    X_full = feats.fillna(0.0)
    cluster_ids = pipe.predict(X_full.values)
    labels = [label_map[int(c)] for c in cluster_ids]
    return pd.Series(labels, index=df.index)


# Wire the new detector + label set into the engine.
bt.REGIME_LABELS = REGIME_LABELS
bt.detect_regimes = ml_kmeans_detect_regimes

# Regime-segmented backtest: optimiser fits a separate lookback per regime.
bt.USE_REGIME_SEG = True

# Smaller IS / OOS windows so the 48k-bar SOLUSDT dataset has both
# (default OOS_CANDLES=90,000 zeros out the IS slice on smaller data).
bt.BACKTEST_CANDLES = 30_000
bt.OOS_CANDLES      = 10_000
bt.ORIGINAL_OOS     = 10_000

if __name__ == "__main__":
    bt.main()
