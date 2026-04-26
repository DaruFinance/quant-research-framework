"""Smoke tests for the ML signal example modules added in v0.2.0."""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import backtester as bt


def _load(name, relpath):
    path = Path(__file__).resolve().parent.parent / relpath
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture()
def sample_df():
    n = 600
    rng = np.random.default_rng(1)
    rets = rng.normal(0.0, 0.01, size=n)
    close = 100.0 * np.exp(np.cumsum(rets))
    return pd.DataFrame({
        "time":  pd.to_datetime(np.arange(n) * 3600, unit="s", utc=True),
        "open":  close,
        "high":  close * 1.001,
        "low":   close * 0.999,
        "close": close,
    })


def test_precomputed_returns_int8_signals(sample_df):
    saved = bt.create_raw_signals
    try:
        mod = _load("ex_ml_pre", "examples/ml_precomputed/ml_precomputed.py")
        bt.create_raw_signals = saved              # the module reassigns it; reset
        raw = mod.ml_precomputed_signals(sample_df.copy(), bt.DEFAULT_LB)
        assert isinstance(raw, np.ndarray)
        assert raw.dtype == np.int8
        assert len(raw) == len(sample_df)
        assert set(np.unique(raw)).issubset({-1, 0, 1})
    finally:
        bt.create_raw_signals = saved


def test_callback_returns_int8_signals(sample_df):
    saved = bt.create_raw_signals
    try:
        mod = _load("ex_ml_cb", "examples/ml_callback/ml_callback.py")
        bt.create_raw_signals = saved
        raw = mod.ml_callback_signals(sample_df.copy(), 50)
        assert isinstance(raw, np.ndarray)
        assert raw.dtype == np.int8
        assert len(raw) == len(sample_df)
        assert set(np.unique(raw)).issubset({-1, 0, 1})
    finally:
        bt.create_raw_signals = saved


def test_callback_helper_no_lookahead(sample_df):
    """`extract_window_features` for bar i must only touch bars [i-lb : i]."""
    mod = _load("ex_ml_cb", "examples/ml_callback/ml_callback.py")
    i, lb = 100, 50
    feats_full = mod.extract_window_features(sample_df, i, lb)

    polluted = sample_df.copy()
    polluted.loc[i:, "close"] = float("nan")            # blank bar i and after
    feats_clean = mod.extract_window_features(polluted, i, lb)

    assert np.allclose(feats_full, feats_clean), "feature builder leaked future data"


def test_version_constant():
    assert hasattr(bt, "__version__")
    parts = bt.__version__.split(".")
    assert int(parts[0]) >= 0 and int(parts[1]) >= 2, bt.__version__
