import math

import numpy as np
import pytest

import backtester as bt
from backtester.monte_carlo import MonteCarloMode, run_trade_monte_carlo


RETURNS = np.array([0.02, -0.01, 0.03, -0.015, 0.01], dtype=float)


def test_permutation_preserves_order_invariant_metrics():
    result = run_trade_monte_carlo(
        RETURNS, mode="permutation", runs=25, seed=7
    )
    for name in ("ROI", "PF", "WinRate", "Exp", "Sharpe"):
        rank = result.ranks[name]
        assert rank.ties == 25, name
        assert rank.percentile == 50.0, name


def test_resampling_is_seeded_and_changes_order_invariant_metrics():
    first = run_trade_monte_carlo(
        RETURNS, mode="resampling", runs=25, seed=9
    )
    second = run_trade_monte_carlo(
        RETURNS, mode=MonteCarloMode.RESAMPLING, runs=25, seed=9
    )
    np.testing.assert_array_equal(first.equity_paths, second.equity_paths)
    assert first.ranks["ROI"].ties < 25


def test_no_loss_profit_factor_is_infinite_for_actual_and_samples():
    result = run_trade_monte_carlo(
        np.array([0.01, 0.02, 0.03]),
        mode="permutation",
        runs=10,
        seed=1,
    )
    profit_factor = result.ranks["PF"]
    assert math.isinf(profit_factor.actual)
    assert profit_factor.ties == 10
    assert result.summary()["metrics"]["PF"]["actual"] == "Infinity"


def test_forex_uses_absolute_drawdown():
    result = run_trade_monte_carlo(
        np.array([1.0, -2.0, 0.5]),
        mode="permutation",
        runs=1,
        seed=1,
        forex_mode=True,
    )
    assert result.ranks["MaxDrawdown"].actual == 2.0
    assert result.drawdown_convention == "absolute R from equity 0"


def test_public_wrapper_uses_config_and_returns_manifest(capsys):
    cfg = bt.Config.from_module()
    cfg.print_equity_curve = False
    cfg.mc_mode = "permutation"
    cfg.mc_runs = 7
    cfg.mc_seed = 13
    result = bt.monte_carlo(RETURNS, {}, config=cfg)
    assert result.mode is MonteCarloMode.PERMUTATION
    assert result.completed_runs == 7
    assert result.seed == 13
    output = capsys.readouterr().out
    assert "Monte Carlo (permutation, seed=13, runs=7)" in output
    assert "ties=" in output


@pytest.mark.parametrize("mode", ["mixed", "bootstrap-and-shuffle", ""])
def test_unknown_mode_is_rejected(mode):
    with pytest.raises(ValueError, match="unknown Monte Carlo mode"):
        run_trade_monte_carlo(RETURNS, mode=mode, runs=1)
