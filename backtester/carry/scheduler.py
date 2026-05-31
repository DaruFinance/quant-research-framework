"""Event-driven rebalance scheduler (item #42, Phase 3).

The scheduler fans in three event streams — fixed-cadence bar clock,
funding events (#38), and arbitrary trigger events (#39s) — and
emits a merged, time-sorted rebalance schedule.  Every decision at
time ``t`` consumes only state available at ``<= t``.  Pollution of
the input streams past ``t`` cannot change the next scheduled time
or its tag.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Literal, Optional, Sequence

import numpy as np
import pandas as pd

from .funding import FUNDING_INTERVAL_S, next_funding_time
from .triggers import TriggerEvent

RebalanceKind = Literal["bar", "funding", "trigger"]


@dataclass(frozen=True)
class ScheduledRebalance:
    time_s: int
    kind: RebalanceKind
    tag: Optional[str] = None
    payload: Optional[object] = None


class EventDrivenScheduler:
    """Build a merged rebalance schedule.

    Parameters
    ----------
    bar_cadence_s
        Fixed-cadence rebalance interval (e.g. 3600 for 1h bars).
        Set to ``None`` to disable bar-clock rebalances.
    funding_df
        DataFrame from :func:`load_funding`; each row schedules a
        funding-time rebalance.  Set ``None`` to disable.
    triggers
        Iterable of :class:`TriggerEvent` records produced by #39s.
    t_start_s, t_end_s
        Inclusive window for the merged schedule.
    """

    def __init__(
        self,
        *,
        bar_cadence_s: Optional[int] = None,
        funding_df: Optional[pd.DataFrame] = None,
        triggers: Sequence[TriggerEvent] = (),
        t_start_s: int = 0,
        t_end_s: Optional[int] = None,
    ) -> None:
        self.bar_cadence_s = bar_cadence_s
        self.funding_df = funding_df
        self.triggers = list(triggers)
        self.t_start_s = int(t_start_s)
        self.t_end_s = (int(t_end_s) if t_end_s is not None
                         else (10**12))   # ~year 33658 — effectively no cap

    def run(self) -> List[ScheduledRebalance]:
        out: List[ScheduledRebalance] = []
        if self.bar_cadence_s is not None and self.bar_cadence_s > 0:
            t = self.t_start_s
            while t <= self.t_end_s:
                out.append(ScheduledRebalance(time_s=int(t), kind="bar"))
                t += int(self.bar_cadence_s)

        if self.funding_df is not None and len(self.funding_df) > 0:
            times = self.funding_df["time"].values
            for t in times:
                t = int(t)
                if self.t_start_s <= t <= self.t_end_s:
                    out.append(ScheduledRebalance(
                        time_s=t, kind="funding", tag="funding_settle",
                    ))

        for ev in self.triggers:
            if self.t_start_s <= ev.time_s <= self.t_end_s:
                out.append(ScheduledRebalance(
                    time_s=ev.time_s, kind="trigger",
                    tag=ev.kind, payload=ev,
                ))

        # Stable sort by (time, kind) so two events at the same second
        # have a deterministic order: bar < funding < trigger.
        kind_rank = {"bar": 0, "funding": 1, "trigger": 2}
        out.sort(key=lambda r: (r.time_s, kind_rank[r.kind]))
        return out

    def next_rebalance(self, after_s: int) -> Optional[ScheduledRebalance]:
        """Find the next scheduled rebalance strictly after ``after_s``.
        Used by the orchestrator when running event-driven mode
        instead of a precomputed full schedule.

        Reads only ``after_s`` and the loaded streams; never consults
        any future state.
        """
        candidates: List[ScheduledRebalance] = []
        if self.bar_cadence_s is not None and self.bar_cadence_s > 0:
            rem = (after_s - self.t_start_s) % self.bar_cadence_s
            nxt = after_s + (self.bar_cadence_s - rem) if rem else (
                after_s + self.bar_cadence_s)
            if nxt <= self.t_end_s:
                candidates.append(ScheduledRebalance(time_s=int(nxt),
                                                       kind="bar"))
        if self.funding_df is not None and len(self.funding_df) > 0:
            arr = self.funding_df["time"].values
            mask = arr > after_s
            if mask.any():
                t = int(arr[np.flatnonzero(mask)[0]])
                if t <= self.t_end_s:
                    candidates.append(ScheduledRebalance(
                        time_s=t, kind="funding", tag="funding_settle"))
        for ev in self.triggers:
            if ev.time_s > after_s and ev.time_s <= self.t_end_s:
                candidates.append(ScheduledRebalance(
                    time_s=ev.time_s, kind="trigger",
                    tag=ev.kind, payload=ev))
        if not candidates:
            return None
        kind_rank = {"bar": 0, "funding": 1, "trigger": 2}
        candidates.sort(key=lambda r: (r.time_s, kind_rank[r.kind]))
        return candidates[0]
