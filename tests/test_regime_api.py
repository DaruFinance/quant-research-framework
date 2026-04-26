"""Contract tests for the pluggable regime detector API added in v0.2.0."""
import numpy as np
import pandas as pd
import pytest

import backtester as bt


@pytest.fixture()
def sample_df():
    n = 800
    rng = np.random.default_rng(0)
    rets = rng.normal(0.0, 0.01, size=n)
    close = 100.0 * np.exp(np.cumsum(rets))
    return pd.DataFrame({
        "time":  pd.to_datetime(np.arange(n) * 3600, unit="s", utc=True),
        "open":  close,
        "high":  close * 1.001,
        "low":   close * 0.999,
        "close": close,
    })


def test_default_labels_length():
    assert 2 <= len(bt.REGIME_LABELS) <= 5
    assert bt.REGIME_LABELS == ["Uptrend", "Downtrend", "Ranging"]


def test_default_detector_returns_known_labels(sample_df):
    dfi = bt.compute_indicators(sample_df, bt.DEFAULT_LB)
    labels = bt.detect_regimes(dfi)
    assert isinstance(labels, pd.Series)
    assert len(labels) == len(sample_df)
    assert set(labels.unique()).issubset(set(bt.REGIME_LABELS))


def test_custom_two_regime_detector(sample_df):
    custom_labels = ["Calm", "Volatile"]
    def detector(df: pd.DataFrame) -> pd.Series:
        out = pd.Series(custom_labels[0], index=df.index)
        rets = df["close"].pct_change().abs().shift(1)
        out.loc[rets > rets.median()] = custom_labels[1]
        return out

    saved_labels, saved_fn = bt.REGIME_LABELS, bt.detect_regimes
    try:
        bt.REGIME_LABELS  = custom_labels
        bt.detect_regimes = detector
        labels = bt.detect_regimes(sample_df)
        assert set(labels.unique()).issubset(set(custom_labels))
    finally:
        bt.REGIME_LABELS  = saved_labels
        bt.detect_regimes = saved_fn


def test_label_count_validation():
    saved_labels, saved_fn = bt.REGIME_LABELS, bt.detect_regimes
    try:
        bt.REGIME_LABELS = ["Only"]                  # length 1: invalid
        with pytest.raises(ValueError):
            bt.detect_regimes(pd.DataFrame({"close": [1.0, 2.0, 3.0]}))
    finally:
        bt.REGIME_LABELS  = saved_labels
        bt.detect_regimes = saved_fn


def test_five_regime_set_accepted(sample_df):
    saved_labels, saved_fn = bt.REGIME_LABELS, bt.detect_regimes
    five_labels = ["A", "B", "C", "D", "E"]
    try:
        bt.REGIME_LABELS = five_labels
        bt.detect_regimes = lambda df: pd.Series([five_labels[i % 5] for i in range(len(df))],
                                                  index=df.index)
        labels = bt.detect_regimes(sample_df)
        assert set(labels.unique()) == set(five_labels)
    finally:
        bt.REGIME_LABELS  = saved_labels
        bt.detect_regimes = saved_fn
