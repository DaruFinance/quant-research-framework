# tests/test_overfit_gapfill.py  (NEW, Python repo)
import numpy as np
from backtester.dsr import (
    deflated_sharpe_ratio, probabilistic_sharpe_ratio,
    expected_max_sharpe_under_null, min_track_record_length,
    min_backtest_length,
)
from backtester.haircut import haircut_sharpe_ratio


def test_dsr_equals_psr_at_sr0():
    rng = np.random.default_rng(7)
    trials = list(rng.normal(0.3, 0.4, 24))
    rets = list(rng.normal(0.001, 0.01, 300))
    sr_chosen = 0.9
    sr0 = expected_max_sharpe_under_null(trials)
    dsr = deflated_sharpe_ratio(sr_chosen, trials, rets)
    psr = probabilistic_sharpe_ratio(sr_chosen, rets, sr_benchmark=sr0)
    assert abs(dsr - psr) < 1e-12   # DSR is PSR with SR* = SR_0


def test_mintrl_monotone_in_confidence():
    rng = np.random.default_rng(11)
    rets = list(rng.normal(0.001, 0.01, 300))
    a = min_track_record_length(0.9, rets, 0.0, 0.90)
    b = min_track_record_length(0.9, rets, 0.0, 0.99)
    assert b > a > 0


def test_mintrl_infinite_when_sr_below_benchmark():
    rng = np.random.default_rng(13)
    rets = list(rng.normal(0.0, 0.01, 200))
    assert min_track_record_length(0.2, rets, 0.5, 0.95) == float("inf")


def test_minbtl_scales_inverse_square_sr():
    a = min_backtest_length(32, 0.05)
    b = min_backtest_length(32, 0.10)
    assert abs(a / b - 4.0) < 1e-9   # SR halved -> 4x the obs


def test_haircut_zero_for_single_test():
    out = haircut_sharpe_ratio(1.5, 252, 1, "bhy", 252.0)
    assert abs(out["haircut_pct"]) < 1e-12
    assert abs(out["haircut_sr"] - 1.5) < 1e-12


def test_haircut_monotone_in_ntests():
    s10 = haircut_sharpe_ratio(1.5, 252, 10, "bhy")["haircut_sr"]
    s100 = haircut_sharpe_ratio(1.5, 252, 100, "bhy")["haircut_sr"]
    assert s100 <= s10
