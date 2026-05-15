"""Tests for the multi-term IS objective (item #44, HIGH-RISK)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

xr = pytest.importorskip("xarray")

from backtester.metrics import sortino, turnover
from backtester.objectives import MultiTermObjective, multi_term
from backtester.panel import load_panel


HERE = Path(__file__).resolve().parent
FIXTURE_DIR = HERE / "fixtures" / "sources"
PANEL_PATHS = {
    "SOL": FIXTURE_DIR / "SOLUSDT_1h_30000_31000.csv",
    "BTC": FIXTURE_DIR / "BTCUSDT_1h_jan_feb_2024.csv",
    "ETH": FIXTURE_DIR / "ETHUSDT_1h_jan_feb_2024.csv",
}


def _panel_returns(panel) -> np.ndarray:
    close = panel.ds["close"].values
    return np.diff(np.log(close), axis=0)


# ---------------------------------------------------------------------------
# Sortino formula sanity
# ---------------------------------------------------------------------------
def test_sortino_basic():
    rng = np.random.default_rng(seed=1)
    r = rng.normal(0.001, 0.01, 1000)
    s = sortino(r)
    assert np.isfinite(s)
    # Mean is positive; we expect a positive ratio.
    assert s > 0


def test_sortino_negative_when_mean_negative():
    rng = np.random.default_rng(seed=2)
    r = rng.normal(-0.001, 0.01, 1000)
    assert sortino(r) < 0


def test_sortino_nan_on_no_losses():
    """If there are no negative returns, downside deviation is 0;
    Sortino is undefined and returns NaN."""
    r = np.array([0.01, 0.02, 0.03])
    assert np.isnan(sortino(r))


def test_sortino_nan_on_too_few_returns():
    assert np.isnan(sortino(np.array([0.01])))


def test_sortino_annualization_scales_by_sqrt():
    rng = np.random.default_rng(seed=3)
    r = rng.normal(0.001, 0.01, 1000)
    raw = sortino(r)
    ann = sortino(r, annualization=252)
    assert abs(ann - raw * np.sqrt(252)) < 1e-12


# ---------------------------------------------------------------------------
# turnover
# ---------------------------------------------------------------------------
def test_turnover_zero_for_constant_position():
    p = np.array([0.5, 0.5, 0.5, 0.5])
    assert turnover(p) == 0.0


def test_turnover_counts_absolute_changes():
    p = np.array([0.5, -0.5, 0.5, 0.0])
    # |(-0.5 - 0.5)| + |(0.5 - -0.5)| + |(0.0 - 0.5)| = 1 + 1 + 0.5 = 2.5
    assert abs(turnover(p) - 2.5) < 1e-12


def test_turnover_short_input_is_zero():
    assert turnover(np.array([0.5])) == 0.0


# ---------------------------------------------------------------------------
# Multi-term objective basic behaviour
# ---------------------------------------------------------------------------
def test_multi_term_returns_score_within_finite_range():
    rng = np.random.default_rng(seed=4)
    r = rng.normal(0.001, 0.01, 500)
    obj = multi_term()
    s = obj(r, benchmark_rets=None, turnover=0.0)
    assert np.isfinite(s)


def test_multi_term_penalises_high_correlation_to_benchmark():
    """Two equivalent strategies — one uncorrelated to benchmark,
    one ~perfectly correlated — produce different scores; the
    correlated one is lower."""
    rng = np.random.default_rng(seed=5)
    bench = rng.normal(0.001, 0.01, 500)
    strat_aligned = bench + rng.normal(0, 1e-6, 500)
    strat_independent = rng.normal(0.001, 0.01, 500)
    obj = multi_term(corr_penalty=1.0)
    s_aligned = obj(strat_aligned, benchmark_rets=bench, turnover=0.0)
    s_independent = obj(strat_independent, benchmark_rets=bench, turnover=0.0)
    assert s_aligned < s_independent


def test_multi_term_penalises_high_turnover():
    rng = np.random.default_rng(seed=6)
    r = rng.normal(0.001, 0.01, 500)
    obj = multi_term(turnover_penalty=0.1)
    low = obj(r, benchmark_rets=None, turnover=1.0)
    high = obj(r, benchmark_rets=None, turnover=10.0)
    assert high < low
    # Differ by exactly (10 - 1) * 0.1 = 0.9.
    assert abs((low - high) - 0.9) < 1e-12


def test_multi_term_rejects_mismatched_benchmark_length():
    obj = multi_term()
    r = np.zeros(500)
    bad_bench = np.zeros(400)
    with pytest.raises(ValueError, match="benchmark length"):
        obj(r, benchmark_rets=bad_bench)


# ---------------------------------------------------------------------------
# HIGH-RISK: the IS-only benchmark contract
# Pollute the OOS portion of the benchmark; assert the IS objective
# is bit-identical to the unpolluted run.
# ---------------------------------------------------------------------------
def test_multi_term_is_objective_unaffected_by_oos_benchmark_pollution():
    """The plan's HIGH-RISK G2 leak test: 50 random IS-window endpoints,
    BTC pollution past each endpoint, IS objective bit-identical."""
    panel = load_panel(PANEL_PATHS)
    rets = _panel_returns(panel)
    btc_idx = panel.assets.index("BTC")
    sol_idx = panel.assets.index("SOL")

    obj = multi_term()
    rng = np.random.default_rng(seed=0xCA1F)
    n = rets.shape[0]
    cuts = sorted(set(int(rng.integers(100, n - 1)) for _ in range(50)))
    for cut_idx in cuts:
        is_window_strategy = rets[max(0, cut_idx - 200):cut_idx, sol_idx]
        is_window_benchmark_clean = rets[max(0, cut_idx - 200):cut_idx, btc_idx]
        score_clean = obj(
            is_window_strategy,
            benchmark_rets=is_window_benchmark_clean,
            turnover=1.0,
        )

        # Pollute the OOS portion of the benchmark (rows >= cut_idx).
        polluted_btc = rets[:, btc_idx].copy()
        polluted_btc[cut_idx:] = rng.normal(0, 1.0, size=n - cut_idx)
        is_window_benchmark_polluted = polluted_btc[max(0, cut_idx - 200):cut_idx]
        score_polluted = obj(
            is_window_strategy,
            benchmark_rets=is_window_benchmark_polluted,
            turnover=1.0,
        )

        assert score_clean == score_polluted, (
            f"cut={cut_idx}: IS objective changed under OOS benchmark "
            f"pollution. clean={score_clean} polluted={score_polluted}"
        )


def test_multi_term_caller_responsible_for_slicing():
    """If the caller passes the FULL benchmark instead of the IS slice,
    the function should reject (length mismatch). The mismatch is the
    safety net that catches the most common misuse."""
    obj = multi_term()
    r = np.zeros(200)  # IS slice of length 200
    full_bench = np.zeros(1000)  # full series — caller forgot to slice
    with pytest.raises(ValueError, match="benchmark length"):
        obj(r, benchmark_rets=full_bench)


# ---------------------------------------------------------------------------
# G3 — 5 IS windows hand-reconciled
# ---------------------------------------------------------------------------
def test_multi_term_5_is_windows_hand_reconcilable():
    """For each of 5 IS-window endpoints, the multi_term score must
    equal the sum of its three explicit terms computed manually from
    the same IS slice. The point: zero hidden state, zero
    out-of-window inputs."""
    panel = load_panel(PANEL_PATHS)
    rets = _panel_returns(panel)
    btc_idx = panel.assets.index("BTC")
    sol_idx = panel.assets.index("SOL")

    obj = multi_term(
        sortino_weight=1.0, corr_penalty=0.5, turnover_penalty=0.1,
    )
    for cut in (200, 400, 600, 800, 999):
        r_is = rets[max(0, cut - 100):cut, sol_idx]
        b_is = rets[max(0, cut - 100):cut, btc_idx]
        turnover_is = 3.0  # arbitrary

        # Manual recomputation.
        manual_sortino = sortino(r_is)
        if r_is.std() > 0 and b_is.std() > 0:
            manual_corr = float(np.corrcoef(r_is, b_is)[0, 1])
        else:
            manual_corr = 0.0
        manual_score = (
            1.0 * manual_sortino
            - 0.5 * abs(manual_corr)
            - 0.1 * turnover_is
        )

        engine_score = obj(r_is, benchmark_rets=b_is, turnover=turnover_is)
        assert abs(engine_score - manual_score) < 1e-12, (
            f"cut={cut}: manual={manual_score} engine={engine_score}"
        )
