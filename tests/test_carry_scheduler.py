"""Tests for the event-driven rebalance scheduler (Phase 3 item #42 — HIGH-RISK)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from backtester.carry.funding import load_funding
from backtester.carry.scheduler import EventDrivenScheduler
from backtester.carry.triggers import FundingFlipTrigger, TriggerEvent


HERE = Path(__file__).resolve().parent
FUNDING = HERE / "fixtures" / "funding_btcusdt_200evt.parquet"


def test_bar_only_schedule():
    sch = EventDrivenScheduler(
        bar_cadence_s=3600,
        t_start_s=0,
        t_end_s=3600 * 10,
    )
    out = sch.run()
    assert len(out) == 11
    assert all(r.kind == "bar" for r in out)
    assert [r.time_s for r in out] == [3600 * i for i in range(11)]


def test_merged_schedule_orders_bar_funding_trigger():
    funding_df = load_funding(FUNDING)
    triggers = [TriggerEvent(time_s=int(funding_df["time"].iloc[0]),
                              kind="funding_flip", direction=1,
                              prev=-1e-4, curr=1e-4)]
    sch = EventDrivenScheduler(
        bar_cadence_s=3600,
        funding_df=funding_df,
        triggers=triggers,
        t_start_s=int(funding_df["time"].iloc[0]),
        t_end_s=int(funding_df["time"].iloc[0]) + 3600,
    )
    out = sch.run()
    same_ts = [r for r in out if r.time_s == int(funding_df["time"].iloc[0])]
    assert [r.kind for r in same_ts] == ["bar", "funding", "trigger"]


def test_next_rebalance_no_lookahead():
    """HIGH-RISK property: the next-rebalance computation at ``after_s``
    must never consult any stream entry past ``after_s`` plus the
    chosen cadence.  We pollute the streams past ``after_s`` and
    require ``next_rebalance`` to return the same record."""
    funding_df = load_funding(FUNDING)
    triggers = [
        TriggerEvent(time_s=int(funding_df["time"].iloc[i]),
                       kind="funding_flip", direction=1, prev=0.0, curr=1e-4)
        for i in range(0, 20, 3)
    ]
    sch = EventDrivenScheduler(
        bar_cadence_s=3600,
        funding_df=funding_df,
        triggers=triggers,
        t_start_s=int(funding_df["time"].iloc[0]),
        t_end_s=int(funding_df["time"].iloc[-1]),
    )
    rng = np.random.default_rng(0xCA1F)
    for _ in range(20):
        T = int(rng.choice(funding_df["time"].values[5:-5]))
        clean = sch.next_rebalance(T)
        # Pollute future funding rows.
        polluted_funding = funding_df.copy()
        polluted_funding.loc[polluted_funding["time"] > T, "time"] = 10**11
        # Pollute trigger times past T.
        polluted_triggers = [t if t.time_s <= T
                              else TriggerEvent(time_s=10**11, kind=t.kind,
                                                  direction=t.direction,
                                                  prev=t.prev, curr=t.curr)
                              for t in triggers]
        polluted = EventDrivenScheduler(
            bar_cadence_s=3600,
            funding_df=polluted_funding,
            triggers=polluted_triggers,
            t_start_s=sch.t_start_s,
            t_end_s=sch.t_end_s,
        )
        p = polluted.next_rebalance(T)
        assert clean == p, f"T={T}: next_rebalance differs after pollution"


def test_full_run_pipeline():
    funding_df = load_funding(FUNDING)
    flip_events = FundingFlipTrigger().run(funding_df)
    sch = EventDrivenScheduler(
        bar_cadence_s=3600 * 24,
        funding_df=funding_df,
        triggers=flip_events,
        t_start_s=int(funding_df["time"].iloc[0]),
        t_end_s=int(funding_df["time"].iloc[-1]),
    )
    out = sch.run()
    bar_count = sum(1 for r in out if r.kind == "bar")
    funding_count = sum(1 for r in out if r.kind == "funding")
    trigger_count = sum(1 for r in out if r.kind == "trigger")
    assert bar_count > 0
    assert funding_count == 200
    assert trigger_count == len(flip_events)
