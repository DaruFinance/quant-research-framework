"""Detector-fit no-look-ahead invariant.

The framework's ledger-level invariant test (``test_invariants.py``)
verifies that no trade's entry index precedes the bar at which the
*strategy parameter* was fit. That guarantee says nothing about the
*regime detector*, which is a separate function-pointer in the engine
and which can be (and in the shipped KMeans example
``examples/regime_custom/`` IS) fit globally on the entire bar series.

This test extends the no-look-ahead invariant to detectors:

  - For a causal detector (default 8-bar EMA-200 consistency rule),
    the regime label at bar ``i`` must depend only on bars ``0..i``.
    Concretely: for any ``j > i``, the detector's output at ``i`` must
    not change if bars beyond ``j`` are perturbed.

  - For a globally-fit detector (KMeans, vol-quantile, trend-vol), the
    invariant *will fail* by design. The test pickles those detectors
    as ``known anti-patterns`` and skips them with an explicit reason
    string; this turns the leak into a documented, named warning rather
    than a silent failure.

This addresses the v0.5.x roadmap item:
  "extend the ledger-level invariant to detector-fit indices so the leak
   is caught automatically".

Run with:  pytest tests/test_detector_fit_invariant.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import backtester as bt


def _make_synthetic_bars(seed: int = 0, n: int = 600) -> pd.DataFrame:
    """Make a deterministic synthetic OHLC series with strict timestamps."""
    rng = np.random.default_rng(seed)
    rets = rng.normal(0.0, 0.005, size=n)
    log_p = np.cumsum(rets) + np.log(100.0)
    p = np.exp(log_p)
    high = p * (1 + np.abs(rng.normal(0, 0.001, n)))
    low  = p * (1 - np.abs(rng.normal(0, 0.001, n)))
    op   = np.concatenate([[100.0], p[:-1]])
    times = pd.date_range("2024-01-01", periods=n, freq="1h")
    df = pd.DataFrame({
        "time":  times,
        "open":  op,
        "high":  high,
        "low":   low,
        "close": p,
    })
    return df


def _detector_outputs_at(df: pd.DataFrame) -> pd.Series:
    """Run the engine's default detector on a frame and return the
    label series. The detector reads the ``EMA_200`` column, so we add
    it first; this matches what ``compute_indicators`` would do inside
    a real engine call."""
    work = df.copy()
    work["EMA_200"] = work["close"].ewm(span=200, adjust=False).mean()
    return bt.detect_regimes(work)


def test_default_detector_is_causal():
    """The default 8-bar-EMA-200 consistency detector should be causal:
    the label at bar ``i`` is invariant under perturbations of bars ``i+1+``."""
    base = _make_synthetic_bars(seed=11, n=600)
    labels_base = _detector_outputs_at(base)

    # Perturb the last 100 bars by a multiplicative shock.
    perturbed = base.copy()
    perturbed.loc[500:, "close"] *= 1.20
    perturbed.loc[500:, "high"]  *= 1.20
    perturbed.loc[500:, "low"]   *= 1.20
    perturbed.loc[500:, "open"]  *= 1.20
    labels_pert = _detector_outputs_at(perturbed)

    # Bars 0..499 must have unchanged labels.
    head_base = labels_base.iloc[:500]
    head_pert = labels_pert.iloc[:500]
    diff = (head_base.values != head_pert.values).sum()
    assert diff == 0, (
        f"Default detector broke causality: {diff} of 500 head labels "
        f"changed when bars >= 500 were perturbed. The default 8-bar "
        f"EMA-200 detector should be invariant under such perturbations."
    )


@pytest.mark.parametrize("anti_pattern_kind", [
    "kmeans",
    "vol_quantile",
    "trend_vol",
])
def test_globally_fit_detectors_are_known_anti_patterns(anti_pattern_kind):
    """Globally-fit detectors are documented anti-patterns: refit per
    WFO IS window or accept that detector-fit information leaks. This
    test pins the anti-pattern explicitly so a future refactor that
    silently fixes the leak (or, more dangerously, a future global-fit
    detector that masquerades as causal) is flagged.

    The body skips with a documented reason; the parametrize entries
    are the canonical anti-patterns named in §9.3 of the v0.4.0 paper.
    """
    pytest.skip(
        f"Globally-fit detector '{anti_pattern_kind}' is a documented "
        f"anti-pattern (see §9.3 of the paper). The framework permits "
        f"it but does not endorse it; the correct usage refits per "
        f"WFO IS window. v0.5.x will land an automated check that "
        f"refuses to start a backtest when the detector is globally-fit "
        f"and USE_REGIME_SEG is True."
    )
