"""Spread re-estimation cadence engine (item #11, Phase 3 — HIGH-RISK).

Schedules β refits at fixed bar cadence, fixed time cadence, or
trigger-driven (e.g. when z-score breaches a threshold). Every refit
uses **only data at indices ≤ the refit bar**, which is what makes
this item HIGH-RISK: a sloppy implementation that fits β at refit
bar t using data > t would silently leak future information into
every trading decision derived from that β.

The 50-T pollute battery in ``tests/test_pairs_cadence.py`` exercises
this directly: pollute panel rows past T, re-run the cadence engine,
assert every β fit at refit bars ≤ T is bit-identical to the clean
run.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Literal, Optional

import numpy as np

from ..panel import PanelData


CadenceMode = Literal["bars", "trigger", "on_breakdown"]


@dataclass
class Cadence:
    """Schedule specification for a re-estimation engine.

    Exactly one of the fields below should be set (others left
    default). ``bars`` is the simple every-N-bars cadence;
    ``trigger`` invokes a user-supplied predicate at each bar to
    decide whether to refit.
    """
    mode: CadenceMode = "bars"
    every: int = 100
    trigger_fn: Optional[Callable[[np.ndarray, int], bool]] = None


@dataclass
class CadenceEngine:
    """Drives spread-β refits over a panel.

    ``spread_fn(panel, asset_a, asset_b, t_idx)`` returns a
    ``SpreadResult``. The engine re-invokes it at the configured
    cadence and records the β at each refit; the spread series
    pre-refit uses the most recent β, and the resulting time series
    is the concatenation. Reads only ``panel`` cells at row indices
    ``<= refit_bar``.
    """
    spread_fn: Callable
    cadence: Cadence = field(default_factory=Cadence)

    def run(
        self,
        panel: PanelData,
        asset_a: str,
        asset_b: str,
        t_start: int,
        t_end: int,
    ) -> List[tuple[int, object]]:
        """Iterate refit bars from ``t_start`` to ``t_end``. Returns a
        list of ``(refit_idx, SpreadResult)`` pairs. The first refit
        is always at ``t_start``."""
        results = [(t_start, self.spread_fn(panel, asset_a, asset_b, t_start))]
        if self.cadence.mode == "bars":
            t = t_start + self.cadence.every
            while t <= t_end:
                results.append((t, self.spread_fn(panel, asset_a, asset_b, t)))
                t += self.cadence.every
        elif self.cadence.mode == "trigger":
            if self.cadence.trigger_fn is None:
                raise ValueError("trigger mode requires trigger_fn")
            # Walk every bar; refit when trigger fires.
            last_refit = t_start
            for t in range(t_start + 1, t_end + 1):
                # Build the spread up to t using the most recent β.
                latest = results[-1][1]
                if self.cadence.trigger_fn(latest.spread[: t + 1], t):
                    results.append((t, self.spread_fn(panel, asset_a, asset_b, t)))
                    last_refit = t
            _ = last_refit
        else:  # on_breakdown
            # Simple breakdown rule: refit when |z| > 3 on a 60-bar
            # rolling stat of the current spread, capped at every-50
            # bars to avoid pathological refit storms.
            min_gap = 50
            last_refit = t_start
            for t in range(t_start + 1, t_end + 1):
                if t - last_refit < min_gap:
                    continue
                latest = results[-1][1]
                seg = latest.spread[max(0, t - 60): t + 1]
                seg = seg[~np.isnan(seg)]
                if len(seg) < 10:
                    continue
                z = (seg[-1] - seg[:-1].mean()) / (seg[:-1].std() + 1e-12)
                if abs(z) > 3.0:
                    results.append((t, self.spread_fn(panel, asset_a, asset_b, t)))
                    last_refit = t
        return results


__all__ = ["Cadence", "CadenceEngine", "CadenceMode"]
