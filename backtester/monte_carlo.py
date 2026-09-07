"""Trade-return Monte Carlo modes used by the backtester.

This module keeps sampling policy separate from metric calculation.  A
permutation is a shuffle without replacement; resampling is a bootstrap with
replacement.  Both operate on completed trade returns, not market bars.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import sqrt
from typing import Mapping

import numpy as np


DEFAULT_TRADE_RUNS = 1_000
DEFAULT_BAR_RUNS = 500
DEFAULT_SEED = 42
_WEIGHTS = np.array([0.0117, 0.0317, 0.0861, 0.2341, 0.6364])


class MonteCarloMode(str, Enum):
    """Supported completed-trade sampling policies."""

    PERMUTATION = "permutation"
    RESAMPLING = "resampling"
    BAR_PERMUTATION = "bar-permutation"

    @property
    def default_runs(self) -> int:
        return DEFAULT_BAR_RUNS if self is self.BAR_PERMUTATION else DEFAULT_TRADE_RUNS

    @classmethod
    def parse(cls, value: str | "MonteCarloMode") -> "MonteCarloMode":
        if isinstance(value, cls):
            return value
        try:
            return cls(str(value).lower())
        except ValueError as exc:
            choices = ", ".join(mode.value for mode in cls)
            raise ValueError(f"unknown Monte Carlo mode {value!r}; choose {choices}") from exc


@dataclass(frozen=True)
class MetricRank:
    actual: float
    percentile: float
    less: int
    ties: int


@dataclass
class MonteCarloResult:
    mode: MonteCarloMode
    seed: int
    requested_runs: int
    completed_runs: int
    sharpe_convention: str
    drawdown_convention: str
    distributions: dict[str, np.ndarray]
    equity_paths: np.ndarray
    ranks: dict[str, MetricRank]

    def summary(self) -> dict:
        """Return a JSON-safe manifest without the full simulated paths."""

        metrics = {}
        for name, rank in self.ranks.items():
            metrics[name] = {
                "actual": _json_float(rank.actual),
                "percentile_midrank": rank.percentile,
                "less": rank.less,
                "ties": rank.ties,
            }
        return {
            "mode": self.mode.value,
            "seed": self.seed,
            "requested_runs": self.requested_runs,
            "completed_runs": self.completed_runs,
            "sharpe_convention": self.sharpe_convention,
            "drawdown_convention": self.drawdown_convention,
            "metrics": metrics,
        }


def _json_float(value: float) -> float | str:
    if np.isposinf(value):
        return "Infinity"
    if np.isneginf(value):
        return "-Infinity"
    if np.isnan(value):
        return "NaN"
    return float(value)


def _metric_values(sim: np.ndarray, forex_mode: bool) -> tuple[dict[str, float], np.ndarray]:
    n = sim.size
    roi = float(sim.sum())
    wins = sim[sim > 0]
    losses = -sim[sim <= 0]
    wins_sum = float(wins.sum())
    losses_sum = float(losses.sum())
    pf = wins_sum / losses_sum if losses_sum > 0 else float("inf")
    wr = float(wins.size / n)
    mean_win = wins_sum / wins.size if wins.size else 0.0
    mean_loss = losses_sum / losses.size if losses.size else 0.0
    expectancy = mean_win * wr - mean_loss * (1.0 - wr)
    std = float(sim.std())
    sharpe = float(sim.mean() / std * sqrt(n)) if n > 1 and std > 0 else 0.0

    if forex_mode:
        equity = np.cumsum(sim)
        high_water = np.maximum.accumulate(np.concatenate(([0.0], equity)))[1:]
        max_drawdown = float(np.max(high_water - equity))
        drawdown_convention = "absolute R from equity 0"
    else:
        equity = 1.0 + np.cumsum(sim)
        high_water = np.maximum.accumulate(equity)
        with np.errstate(divide="ignore", invalid="ignore"):
            drawdowns = np.divide(
                high_water - equity,
                high_water,
                out=np.zeros_like(equity),
                where=high_water > 0,
            )
        max_drawdown = float(np.max(drawdowns))
        drawdown_convention = "fractional from equity 1"

    segments = np.array_split(sim, 5)
    weighted = float(np.dot(_WEIGHTS, [segment.sum() for segment in segments]))
    consistency = 0.6 * weighted + 0.4 * roi
    metrics = {
        "ROI": roi,
        "PF": pf,
        "WinRate": wr,
        "Exp": expectancy,
        "Sharpe": sharpe,
        "MaxDrawdown": max_drawdown,
        "Consistency": consistency,
    }
    return metrics, equity, drawdown_convention


def _midrank(distribution: np.ndarray, actual: float) -> MetricRank:
    if np.isposinf(actual):
        ties_mask = np.isposinf(distribution)
    elif np.isneginf(actual):
        ties_mask = np.isneginf(distribution)
    else:
        ties_mask = np.isclose(distribution, actual, rtol=1e-12, atol=1e-15)
    less = int(np.count_nonzero((distribution < actual) & ~ties_mask))
    ties = int(np.count_nonzero(ties_mask))
    percentile = (less + 0.5 * ties) / distribution.size * 100.0
    return MetricRank(actual=actual, percentile=percentile, less=less, ties=ties)


def run_trade_monte_carlo(
    returns,
    *,
    mode: str | MonteCarloMode = MonteCarloMode.RESAMPLING,
    runs: int = DEFAULT_TRADE_RUNS,
    seed: int = DEFAULT_SEED,
    forex_mode: bool = False,
) -> MonteCarloResult:
    """Run one completed-trade Monte Carlo null.

    Strategy logic and market bars are not rerun here.  Use the separate
    ``mc_bar_permutation`` runner when the null must rebuild OHLCV and execute
    a frozen strategy on every replay.
    """

    arr = np.asarray(returns, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError("returns must be a one-dimensional array")
    if arr.size == 0:
        raise ValueError("returns must not be empty")
    if not np.all(np.isfinite(arr)):
        raise ValueError("returns must contain only finite values")
    if runs <= 0:
        raise ValueError("runs must be positive")
    if seed < 0:
        raise ValueError("seed must be non-negative")

    selected_mode = MonteCarloMode.parse(mode)
    if selected_mode is MonteCarloMode.BAR_PERMUTATION:
        raise ValueError(
            "bar-permutation reruns a frozen strategy on OHLCV; use "
            "mc_bar_permutation/run.py --mode bar-permutation"
        )
    rng = np.random.default_rng(seed)
    names = ("ROI", "PF", "WinRate", "Exp", "Sharpe", "MaxDrawdown", "Consistency")
    distributions = {name: np.empty(runs, dtype=np.float64) for name in names}
    equity_paths = np.empty((runs, arr.size), dtype=np.float64)

    for index in range(runs):
        if selected_mode is MonteCarloMode.RESAMPLING:
            sample = rng.choice(arr, size=arr.size, replace=True)
        else:
            sample = rng.permutation(arr)
        metrics, equity, drawdown_convention = _metric_values(sample, forex_mode)
        equity_paths[index] = equity
        for name in names:
            distributions[name][index] = metrics[name]

    observed, _, drawdown_convention = _metric_values(arr, forex_mode)
    ranks = {name: _midrank(distributions[name], observed[name]) for name in names}
    return MonteCarloResult(
        mode=selected_mode,
        seed=seed,
        requested_runs=runs,
        completed_runs=runs,
        sharpe_convention="completed-trade mean/std * sqrt(trade count)",
        drawdown_convention=drawdown_convention,
        distributions=distributions,
        equity_paths=equity_paths,
        ranks=ranks,
    )
