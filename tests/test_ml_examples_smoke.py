"""Smoke tests for the ML strategy + ML regime detector examples.

These tests import each example module and call its strategy /
regime function on a small synthetic OHLC frame. They do NOT verify
metric correctness — they verify only that:
  - the module imports cleanly (sklearn pipeline builds);
  - the strategy function obeys the (df, lb) -> int8[n] contract;
  - the regime detector returns a Series of allowed labels indexed
    identically to the input frame.

Slow paths (the engine's full IS/OOS/WFO pipeline) are NOT run here;
that's left to the per-example `python examples/X.py` smoke runs.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import backtester as bt


def _df_synthetic(n=2_000, seed=7):
    rng = np.random.default_rng(seed)
    rets = rng.normal(0.00005, 0.01, size=n)
    close = 100.0 * np.exp(np.cumsum(rets))
    times = pd.to_datetime(1_600_000_000 + np.arange(n) * 3600,
                           unit="s", utc=True).tz_convert(bt.NY_TZ)
    return pd.DataFrame({
        "time":  times,
        "open":  close,
        "high":  close * 1.002,
        "low":   close * 0.998,
        "close": close,
    })


def _import_example(rel_path: str):
    """Import an example module by file path, isolated from import-side
    effects on other tests (the example modules touch bt globals)."""
    import importlib.util
    here = Path(__file__).resolve().parent.parent
    src  = here / rel_path
    spec = importlib.util.spec_from_file_location(
        f"_ex_{src.stem}", src)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# Keep the original engine globals so we restore them after each test —
# the example modules patch bt.create_raw_signals / detect_regimes /
# REGIME_LABELS / OOS_CANDLES on import.
@pytest.fixture(autouse=True)
def _isolate_bt_globals():
    saved = {
        "create_raw_signals": bt.create_raw_signals,
        "detect_regimes":     bt.detect_regimes,
        "REGIME_LABELS":      list(bt.REGIME_LABELS),
        "BACKTEST_CANDLES":   bt.BACKTEST_CANDLES,
        "OOS_CANDLES":        bt.OOS_CANDLES,
        "ORIGINAL_OOS":       bt.ORIGINAL_OOS,
        "USE_REGIME_SEG":     bt.USE_REGIME_SEG,
        "LOOKBACK_RANGE":     bt.LOOKBACK_RANGE,
    }
    yield
    for k, v in saved.items():
        setattr(bt, k, v)


# ---------------------------------------------------------------------------
# Strategy examples
# ---------------------------------------------------------------------------
def test_ml_logreg_signals_obey_contract():
    mod = _import_example("examples/ml_sklearn/ml_logreg.py")
    df = _df_synthetic(n=600)
    sig = mod.ml_logreg_signals(df, lb=50)
    assert isinstance(sig, np.ndarray)
    assert sig.dtype == np.int8
    assert sig.shape == (len(df),)
    assert set(np.unique(sig)).issubset({-1, 0, 1})


def test_ml_random_forest_signals_obey_contract():
    mod = _import_example("examples/ml_sklearn/ml_random_forest.py")
    df = _df_synthetic(n=600)
    sig = mod.ml_rf_signals(df, lb=50)
    assert isinstance(sig, np.ndarray)
    assert sig.dtype == np.int8
    assert sig.shape == (len(df),)
    assert set(np.unique(sig)).issubset({-1, 0, 1})


# ---------------------------------------------------------------------------
# Regime detector example
# ---------------------------------------------------------------------------
def test_ml_kmeans_regime_detector_obeys_contract():
    mod = _import_example("examples/ml_regime_kmeans/ml_regime_kmeans.py")
    df = _df_synthetic(n=1_500)
    regimes = mod.ml_kmeans_detect_regimes(df)
    assert isinstance(regimes, pd.Series)
    assert len(regimes) == len(df)
    assert (regimes.index == df.index).all()
    allowed = set(mod.REGIME_LABELS)
    assert set(regimes.unique()).issubset(allowed), (
        f"detector returned labels outside REGIME_LABELS: "
        f"{set(regimes.unique()) - allowed}"
    )


def test_ml_kmeans_regime_detector_no_lookahead():
    """Bar i's label must depend only on data at or before bar i-1.
    Pollute only the post-IS tail (so the KMeans fit is unchanged);
    verify pre-IS and most-of-IS labels are unchanged. Bars within
    the rolling-feature window of the polluted region (200 bars) may
    differ via shift-and-rolling propagation, so we exclude those."""
    mod = _import_example("examples/ml_regime_kmeans/ml_regime_kmeans.py")

    # Configure smaller IS/OOS so we can pollute past the IS end.
    bt.BACKTEST_CANDLES = 800
    bt.OOS_CANDLES      = 400
    bt.ORIGINAL_OOS     = 400

    n = 2_000
    df = _df_synthetic(n=n, seed=11)
    full = mod.ml_kmeans_detect_regimes(df)

    # IS occupies [n - OOS - IS, n - OOS) = [800, 1600). Pollute beyond
    # that window so KMeans's fit is unchanged.
    cut = 1_700
    polluted = df.copy()
    polluted.loc[cut:, ["close", "high", "low", "open"]] = float("nan")
    clean = mod.ml_kmeans_detect_regimes(polluted)

    # Compare labels strictly before the rolling-window's reach into
    # the polluted region. Largest rolling window in the feature set
    # is 200 bars, and features are shift(1), so bars before
    # cut - 200 - 1 are guaranteed unaffected.
    safe_end = cut - 201
    early_full  = full.iloc[:safe_end].astype(str).reset_index(drop=True)
    early_clean = clean.iloc[:safe_end].astype(str).reset_index(drop=True)
    assert (early_full == early_clean).all(), (
        f"kmeans regime detector leaked post-bar-{cut} pollution into "
        f"labels at or before bar {safe_end - 1}"
    )
