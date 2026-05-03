"""Hypothesis-based property tests for the strategy contract.

These complement the example-based tests in `test_invariants.py`
(which use fixed seeds — kept as regression cases for specific bugs)
by letting Hypothesis search for counter-examples across a generated
input space, with shrinking on failure.

Invariants under test (each with a generated input space, not a single
fixed fixture):

  parse_signals_no_lookahead    sig[..k] invariant under raw[k:] pollution
  detect_regimes_no_lookahead   regime[..k] invariant under df[k:] pollution
  trade_indices_well_formed     for all (seed, n, lb): every trade tuple
                                obeys 0 <= ent <= exi < n, side ∈ {±1},
                                positive prices, finite PnL, qty >= 0
  session_no_entry_outside      for all (seed, n, session window): no
                                trade enters at a NY-local time outside
                                [SESSION_START, SESSION_END)

Hypothesis settings: deterministic seed via a global database off, max
examples bounded so CI stays under one minute. Increase locally with
`HYPOTHESIS_MAX_EXAMPLES=200 pytest tests/test_invariants_property.py`.
"""
import os
from datetime import datetime

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings, HealthCheck, strategies as st
from hypothesis.strategies import composite

import backtester as bt


_MAX_EXAMPLES = int(os.environ.get("HYPOTHESIS_MAX_EXAMPLES", "30"))


def _df_from(seed: int, n: int, start_unix: int = 1_600_000_000,
             interval_s: int = 3600) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rets = rng.normal(0.00005, 0.01, size=n)
    close = 100.0 * np.exp(np.cumsum(rets))
    times = (pd.to_datetime(start_unix + np.arange(n) * interval_s,
                            unit="s", utc=True)
               .tz_convert(bt.NY_TZ))
    return pd.DataFrame({
        "time":  times,
        "open":  close,
        "high":  close * 1.002,
        "low":   close * 0.998,
        "close": close,
    })


# ---------------------------------------------------------------------------
# Property 1: parse_signals never leaks future raw[] data into earlier sig[].
# Holds for any choice of raw signal sequence and any cut point.
# ---------------------------------------------------------------------------
@given(
    seed=st.integers(0, 2**31 - 1),
    n=st.integers(60, 1500),
    cut_frac=st.floats(0.2, 0.9),
)
@settings(max_examples=_MAX_EXAMPLES * 4,  # pure-Python, fast
          deadline=None,
          suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_parse_signals_no_lookahead_property(seed: int, n: int, cut_frac: float):
    rng = np.random.default_rng(seed)
    raw = rng.choice([-1, 0, 1], size=n).astype(np.int8)
    df = _df_from(seed, n)
    cut = max(10, int(n * cut_frac))

    parsed_full = bt.parse_signals(raw.copy(), df["time"])

    polluted = raw.copy()
    polluted[cut:] = rng.choice([-1, 0, 1], size=n - cut).astype(np.int8)
    parsed_polluted = bt.parse_signals(polluted, df["time"])

    assert (parsed_full[:cut] == parsed_polluted[:cut]).all(), (
        f"parse_signals leaked raw[{cut}:] into sig[..{cut}] "
        f"(seed={seed}, n={n})"
    )


# ---------------------------------------------------------------------------
# Property 2: detect_regimes never leaks future close/EMA200 into earlier
# labels. Holds for any seed / length / cut.
# ---------------------------------------------------------------------------
@given(
    seed=st.integers(0, 2**31 - 1),
    n=st.integers(400, 1500),  # need >= EMA-200 warmup
    cut_frac=st.floats(0.4, 0.9),
)
@settings(max_examples=_MAX_EXAMPLES * 2, deadline=None)
def test_default_regime_detector_no_lookahead_property(
    seed: int, n: int, cut_frac: float
):
    df = _df_from(seed, n)
    df["EMA_200"] = df["close"].ewm(span=200, adjust=False).mean()
    cut = max(250, int(n * cut_frac))

    full = bt.detect_regimes(df)

    polluted = df.copy()
    polluted.loc[cut:, "close"]   = float("nan")
    polluted.loc[cut:, "EMA_200"] = float("nan")
    clean = bt.detect_regimes(polluted)

    early_full  = full.iloc[:cut].astype(str).reset_index(drop=True)
    early_clean = clean.iloc[:cut].astype(str).reset_index(drop=True)
    assert (early_full == early_clean).all(), (
        f"detect_regimes leaked df[{cut}:] into regime[..{cut}] "
        f"(seed={seed}, n={n})"
    )


# ---------------------------------------------------------------------------
# Property 3: every trade tuple satisfies the well-formedness invariants
# regardless of input. Engine is heavier here so max_examples is lower.
# ---------------------------------------------------------------------------
@given(
    seed=st.integers(0, 2**31 - 1),
    n=st.integers(400, 1500),
    lb=st.integers(10, 30),
)
@settings(max_examples=_MAX_EXAMPLES, deadline=None,
          suppress_health_check=[HealthCheck.function_scoped_fixture,
                                 HealthCheck.too_slow])
def test_trade_indices_well_formed_property(seed: int, n: int, lb: int):
    df = _df_from(seed, n)
    dfi = bt.compute_indicators(df, lb)
    raw = bt.create_raw_signals(dfi, lb)
    parsed = bt.parse_signals(raw, dfi["time"])
    trades, _, _, _, _ = bt.backtest(dfi, parsed)

    for side, ent, exi, ep, xp, qty, pnl in trades:
        assert side in (1, -1), f"bad side {side} (seed={seed},n={n},lb={lb})"
        assert 0 <= ent < n,    f"entry idx {ent} oob"
        assert 0 <= exi < n,    f"exit idx {exi} oob"
        assert ent <= exi,      f"exit {exi} before entry {ent}"
        assert ep > 0 and xp > 0, f"non-positive prices ({ep}, {xp})"
        assert qty >= 0,        f"negative quantity {qty}"
        assert np.isfinite(pnl), f"non-finite PnL {pnl}"


# ---------------------------------------------------------------------------
# Property 4: with TRADE_SESSIONS on, no trade entry timestamp falls outside
# the configured session window. Generates session windows from a small
# discrete set so monkeypatch is well-defined.
# ---------------------------------------------------------------------------
_SESSIONS = [
    ("8:00",  "16:50"),
    ("9:30",  "16:00"),
    ("0:00",  "12:00"),
    ("13:00", "23:00"),
]


@given(
    seed=st.integers(0, 2**31 - 1),
    n=st.integers(800, 2000),
    sess_idx=st.integers(0, len(_SESSIONS) - 1),
)
@settings(max_examples=_MAX_EXAMPLES, deadline=None,
          suppress_health_check=[HealthCheck.function_scoped_fixture,
                                 HealthCheck.too_slow])
def test_session_no_entry_outside_window_property(
    seed: int, n: int, sess_idx: int, monkeypatch
):
    start_str, end_str = _SESSIONS[sess_idx]
    monkeypatch.setattr(bt, "TRADE_SESSIONS", True)
    monkeypatch.setattr(bt, "SESSION_START", start_str)
    monkeypatch.setattr(bt, "SESSION_END",   end_str)

    df = _df_from(seed, n)
    dfi = bt.compute_indicators(df, bt.DEFAULT_LB)
    raw = bt.create_raw_signals(dfi, bt.DEFAULT_LB)
    parsed = bt.parse_signals(raw, dfi["time"])
    trades, _, _, _, _ = bt.backtest(dfi, parsed)

    start_t = datetime.strptime(start_str, "%H:%M").time()
    end_t   = datetime.strptime(end_str,   "%H:%M").time()

    violations = []
    for side, ent, exi, *_ in trades:
        ts_ny = (dfi["time"].iloc[ent].tz_convert(bt.NY_TZ)
                 .timetz().replace(tzinfo=None))
        if not (start_t <= ts_ny < end_t):
            violations.append((ent, side, ts_ny))

    assert not violations, (
        f"Session invariant: {len(violations)} entries outside "
        f"[{start_str}, {end_str}) (seed={seed},n={n})"
    )
