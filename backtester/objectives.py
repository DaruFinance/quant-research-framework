"""Multi-term IS objective functions (item #44, Phase 2).

Default scoring in the single-asset optimiser is one of
``ROI / PF / Sharpe / WinRate / Exp / MaxDrawdown / Consistency``
selected via the ``OPT_METRIC`` module constant. Item #44 adds a
composite that combines:

    score = sortino - lambda * |corr(strategy_rets, benchmark_rets)|
          - mu * turnover_per_bar

Three motivations:

- ``sortino`` is the missing downside-aware cousin of Sharpe. The
  v0.4.0 baseline never computed it; the panel-WFO path needs it
  alongside per-asset risk metrics so basket allocation can prefer
  strategies with asymmetric loss profiles.
- ``-|corr(s, BTC)|`` penalises strategies whose returns merely
  mirror the underlying asset's directional moves; a market-neutral
  strategy on a crypto panel should explicitly target low
  correlation with BTC.
- ``-turnover`` discourages overfit, high-churn parameter choices
  that look good on backtest Sharpe but vanish under realistic
  execution friction.

The HIGH-RISK lookahead surface (per the plan): the benchmark
returns used to compute the correlation **must come from the IS
window only**. Polluting the OOS / future portion of the benchmark
series cannot change the IS objective value. The harness in
``tests/test_objective_multi_term.py`` exercises this with 50
random IS-window endpoints.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .metrics import sortino


@dataclass(frozen=True)
class MultiTermObjective:
    """Composite IS objective.

    Higher score is better (the optimiser maximises). The terms are
    additive:

        score = sortino_weight * sortino(rets, annualization)
              - corr_penalty   * |corr(rets, benchmark_rets)|
              - turnover_penalty * turnover

    Defaults are conservative (heavier Sortino weight, modest corr
    and turnover penalties). Callers tune via the dataclass fields.
    """
    sortino_weight:   float = 1.0
    corr_penalty:     float = 0.5
    turnover_penalty: float = 0.1
    annualization:    Optional[float] = None
    """If set, Sortino is annualised by sqrt(annualization). For 1h
    bars annualization = 24*365 = 8760. For daily bars 252.
    Leaving None passes through the unannualised ratio."""

    def __call__(
        self,
        rets: np.ndarray,
        benchmark_rets: Optional[np.ndarray] = None,
        turnover: float = 0.0,
    ) -> float:
        """Compute the objective score on a single returns slice.

        Parameters
        ----------
        rets : 1-D returns array sliced to the IS window.
        benchmark_rets : 1-D benchmark returns array sliced to the
            same IS window. Length must match ``rets``. If ``None``
            or empty, the corr penalty term contributes 0.
        turnover : scalar turnover for the IS window (typically
            ``backtester.metrics.turnover(positions_in_window)``).

        Returns
        -------
        float score. Higher is better.

        Raises
        ------
        ValueError if ``benchmark_rets`` is given but length
        mismatches ``rets``.
        """
        r = np.asarray(rets, dtype=np.float64)
        if r.size < 2:
            return float("-inf")

        score = self.sortino_weight * sortino(r, annualization=self.annualization)

        if benchmark_rets is not None and len(benchmark_rets) > 0:
            b = np.asarray(benchmark_rets, dtype=np.float64)
            if len(b) != len(r):
                raise ValueError(
                    f"multi_term: benchmark length {len(b)} != "
                    f"strategy length {len(r)}; the benchmark must be "
                    f"sliced to the same IS window as the strategy"
                )
            # Pearson correlation; handle zero-variance edge case.
            if r.std() > 0 and b.std() > 0:
                rho = float(np.corrcoef(r, b)[0, 1])
                if np.isnan(rho):
                    rho = 0.0
                score -= self.corr_penalty * abs(rho)
            # Else: penalty contributes 0 (no movement means no
            # directional coupling to penalise).

        score -= self.turnover_penalty * float(turnover)
        return score


def multi_term(
    sortino_weight: float = 1.0,
    corr_penalty: float = 0.5,
    turnover_penalty: float = 0.1,
    annualization: Optional[float] = None,
) -> MultiTermObjective:
    """Factory mirroring the plan's API. Equivalent to constructing
    ``MultiTermObjective`` directly; the function form is exposed for
    config-string callers that may want a callable producer."""
    return MultiTermObjective(
        sortino_weight=sortino_weight,
        corr_penalty=corr_penalty,
        turnover_penalty=turnover_penalty,
        annualization=annualization,
    )


__all__ = ["MultiTermObjective", "multi_term"]
