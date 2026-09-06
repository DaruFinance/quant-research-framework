"""Causal detector checks and deliberately leaking global-fit controls.

The global-fit functions below are test sentinels, not supported detectors.
"""
from pathlib import Path
import importlib.util

import numpy as np
import pandas as pd
import pytest

import backtester as bt
from backtester.invariants import InvariantSpec, assert_no_lookahead, list_invariants

_spec = importlib.util.spec_from_file_location(
    "qrf_regime_example", Path(__file__).parents[1] / "examples/regime_custom/regime_custom.py")
_example = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_example)
ML5_LABELS = _example.ML5_LABELS
TinyKMeansLikeDetector = _example.TinyKMeansLikeDetector
detect_regimes_ml5 = _example.detect_regimes_ml5
detect_regimes_vol2 = _example.detect_regimes_vol2
detect_regimes_vol4 = _example.detect_regimes_vol4


@pytest.fixture
def bars():
    return pd.read_csv(Path(__file__).parent / "fixtures/sol_1h_30000_31000.csv")


def _shock_tail(df, cut):
    polluted = df.copy()
    for col in ("open", "high", "low", "close"):
        polluted.loc[polluted.index[cut:], col] *= np.linspace(1, 20, len(df) - cut)
    return polluted


def test_default_detector_is_causal(bars):
    def detector(df):
        work = df.copy()
        work["EMA_200"] = work["close"].ewm(span=200, adjust=False).mean()
        return bt.detect_regimes(work)
    assert_no_lookahead(InvariantSpec("default", detector), bars, 500,
                        pollute=_shock_tail)


@pytest.mark.parametrize("anti_pattern_kind", ["kmeans", "vol_quantile", "trend_vol"])
def test_globally_fit_detectors_are_known_anti_patterns(anti_pattern_kind, bars):
    def globally_fit(df):
        if anti_pattern_kind == "kmeans":
            from scipy.cluster.vq import kmeans2
            # Deliberately fit all rows: future data changes fitted centers.
            close = df["close"].to_numpy()
            centers = np.quantile(close, [0, .5, 1])
            return kmeans2(close, centers, minit="matrix")[1]
        ret = df["close"].pct_change(fill_method=None)
        vol = ret.rolling(20, min_periods=2).std().fillna(0)
        if anti_pattern_kind == "vol_quantile":
            return pd.qcut(vol, 3, labels=False)
        trend = df["close"].diff(20).fillna(0)
        return (vol > vol.median()).astype(int) * 2 + (trend > trend.median()).astype(int)

    with pytest.raises(AssertionError, match=f"{anti_pattern_kind}.*leaked future data"):
        assert_no_lookahead(InvariantSpec(anti_pattern_kind, globally_fit), bars,
                            500, pollute=_shock_tail)


@pytest.mark.parametrize("detector", [detect_regimes_vol2, detect_regimes_vol4, detect_regimes_ml5])
@pytest.mark.parametrize("cut", [1, 50, 100, 500, 999])
def test_custom_detectors_are_causal(detector, cut, bars):
    cut = min(cut, len(bars) - 1)
    assert_no_lookahead(InvariantSpec(detector.__name__, detector), bars, cut,
                        pollute=_shock_tail)


def test_ml5_excludes_the_current_bar_and_matches_prefixes(bars):
    detector = TinyKMeansLikeDetector()
    full = detector(bars)
    assert set(full).issubset(ML5_LABELS)
    assert full.nunique() == 5
    for cut in (1, 49, 50, 99, 100, 501, len(bars) - 1):
        polluted = _shock_tail(bars, cut)
        polluted.loc[polluted.index[cut], "close"] *= 100
        pd.testing.assert_series_equal(full.iloc[:cut + 1], detector(polluted).iloc[:cut + 1])
        pd.testing.assert_series_equal(full.iloc[:cut], detector(bars.iloc[:cut]))


@pytest.mark.parametrize("length", [0, 1, 50, 200])
def test_ml5_empty_and_flat_warmup(length):
    frame = pd.DataFrame({"close": np.full(length, 100.0)})
    assert TinyKMeansLikeDetector()(frame).tolist() == [ML5_LABELS[0]] * length


def test_custom_detectors_registered():
    names = {spec.name for spec in list_invariants()}
    assert {"example_vol2", "example_vol4", "ml5_quantile"} <= names
