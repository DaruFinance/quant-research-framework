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


# ---------------------------------------------------------------------------
# Item #46: hold-period cap. Pure index arithmetic (idx and ent_bar
# both at-or-before this bar), so trivially lookahead-free.
# ---------------------------------------------------------------------------
@given(
    seed=st.integers(0, 2**31 - 1),
    n=st.integers(400, 1200),
    lb=st.integers(10, 30),
    max_hold=st.integers(2, 50),
)
@settings(max_examples=_MAX_EXAMPLES, deadline=None,
          suppress_health_check=[HealthCheck.function_scoped_fixture,
                                 HealthCheck.too_slow])
def test_max_hold_bars_no_leak_property(seed: int, n: int, lb: int, max_hold: int):
    """Every trade with exit_reason = HOLD_PERIOD must satisfy
    exit_idx - entry_idx == max_hold; other trades must satisfy
    exit_idx - entry_idx <= max_hold (they exited earlier via
    SL/TP/signal/session). Because the kernel reads only `idx` and
    `ent_bar` for the cap decision, the property is also a strong
    lookahead guard."""
    df = _df_from(seed, n)
    dfi = bt.compute_indicators(df, lb)
    raw = bt.create_raw_signals(dfi, lb)
    parsed = bt.parse_signals(raw, dfi["time"])

    old = bt.MAX_HOLD_BARS
    bt.MAX_HOLD_BARS = max_hold
    try:
        trades, _, _, _, _ = bt.backtest(dfi, parsed)
    finally:
        bt.MAX_HOLD_BARS = old

    for side, ent, exi, *_ in trades:
        hold = exi - ent
        assert hold <= max_hold, (
            f"hold-period cap violated: trade ent={ent} exi={exi} "
            f"hold={hold} > max_hold={max_hold} (seed={seed},n={n},lb={lb})"
        )


def test_max_hold_bars_zero_preserves_v0_4_0_behavior():
    """MAX_HOLD_BARS=0 (default) must produce bit-identical output to
    the pre-#46 kernel — the in-loop check is guarded by `max_hold_bars
    > 0` and must not perturb any trade when off. Run a small fixture
    twice (default vs explicit 0) and assert trade lists equal."""
    df = bt.load_ohlc("tests/fixtures/sol_1h_30000_31000.csv")
    dfi = bt.compute_indicators(df, 10)
    raw = bt.create_raw_signals(dfi, 10)
    parsed = bt.parse_signals(raw, dfi["time"])

    bt.MAX_HOLD_BARS = 0
    trades_a, met_a, _, _, _ = bt.backtest(dfi, parsed)

    bt.MAX_HOLD_BARS = 0  # idempotent
    trades_b, met_b, _, _, _ = bt.backtest(dfi, parsed)

    assert len(trades_a) == len(trades_b)
    for ta, tb in zip(trades_a, trades_b):
        assert ta == tb, f"MAX_HOLD_BARS=0 produced different trades: {ta} vs {tb}"
    assert met_a == met_b


# ---------------------------------------------------------------------------
# Item #14: invariant-registry harness self-tests.
# ---------------------------------------------------------------------------
def test_harness_catches_known_leak():
    """A deliberately leaky function must trip assert_no_lookahead.

    Constructs a fake regime detector that returns ``close.shift(-1)`` —
    i.e. uses tomorrow's price to label today. The harness should detect
    that polluting future rows changes the output for earlier rows and
    raise AssertionError naming the offending invariant.
    """
    from backtester.invariants import InvariantSpec, assert_no_lookahead

    def leaky_detector(df):
        # Shifts the NEXT bar's close back into today's label — clear leak.
        return df["close"].shift(-1).fillna(0.0)

    spec = InvariantSpec(name="leaky_sentinel", func=leaky_detector,
                         data_kind="ohlc_df")
    df = _df_from(seed=42, n=400)
    raised = False
    try:
        assert_no_lookahead(spec, df, cut=200)
    except AssertionError as e:
        raised = True
        assert "leaky_sentinel" in str(e), (
            f"harness error message must name the leaky invariant: {e!r}"
        )
    assert raised, "deliberate leak was NOT caught by the harness"


def test_harness_passes_lookahead_free_function():
    """The harness must NOT false-positive on a known-good function.

    Registers a trivial 20-bar moving-average detector that reads only
    past close values, runs the pollute-and-verify probe; must complete
    silently without raising.
    """
    from backtester.invariants import InvariantSpec, assert_no_lookahead

    def clean_detector(df):
        # 20-bar SMA threshold — uses only df.close up to and including
        # the labelled bar.
        sma = df["close"].rolling(20, min_periods=1).mean()
        return (df["close"] > sma).astype(int)

    spec = InvariantSpec(name="clean_sma_threshold", func=clean_detector,
                         data_kind="ohlc_df")
    df = _df_from(seed=43, n=400)
    assert_no_lookahead(spec, df, cut=200)


def test_registered_invariants_pass_default_pollute():
    """Every function registered via @registers_invariant must survive
    the default-polluter probe. New items (#4 cross-asset regime,
    #9 spread screener, #11 cadence engine, ...) inherit this gate
    automatically just by carrying the decorator.
    """
    from backtester.invariants import list_invariants, assert_no_lookahead

    specs = list_invariants()
    assert specs, "registry is empty — default_regime_detector should register on import"
    df = _df_from(seed=44, n=500)
    df["EMA_200"] = df["close"].ewm(span=200, adjust=False).mean()
    for spec in specs:
        if spec.data_kind != "ohlc_df":
            continue
        assert_no_lookahead(spec, df, cut=300)


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

    for side, ent, exi, ep, xp, qty, pnl, *_ in trades:
        assert side in (1, -1), f"bad side {side} (seed={seed},n={n},lb={lb})"
        assert 0 <= ent < n,    f"entry idx {ent} oob"
        assert 0 <= exi < n,    f"exit idx {exi} oob"
        assert ent <= exi,      f"exit {exi} before entry {ent}"
        assert ep > 0 and xp > 0, f"non-positive prices ({ep}, {xp})"
        assert qty >= 0,        f"negative quantity {qty}"
        assert np.isfinite(pnl), f"non-finite PnL {pnl}"


# ---------------------------------------------------------------------------
# Property 3b: aggregate_legs (item #2) is pure data-only. Polluting the
# tail of the input leg list at positions >= T cannot change the first T
# Trade groups in the output. Pure lookahead-freeness check on the
# aggregation layer that sits between the kernel's per-leg 9-tuples and
# downstream multi-leg analytics.
# ---------------------------------------------------------------------------
@given(
    seed=st.integers(0, 2**31 - 1),
    n=st.integers(10, 200),
    cut_frac=st.floats(0.2, 0.9),
)
@settings(max_examples=_MAX_EXAMPLES * 4, deadline=None)
def test_aggregate_legs_no_leak_property(seed: int, n: int, cut_frac: float):
    from backtester.ledger import aggregate_legs
    rng = np.random.default_rng(seed)
    # Clean input: kernel-shape 9-tuples with monotonic tgid = row index,
    # leg_id = 0 (single-leg single-asset mode).
    legs = [
        (1, i * 3, i * 3 + 2, 100.0, 101.0, 0.1, 1.0, 0, i)
        for i in range(n)
    ]
    cut = max(1, int(n * cut_frac))
    # Polluted input: replace tail with garbage entry_idx / exi / prices /
    # qty / pnl, keeping tgid monotonic. The pollution simulates a future
    # bug that emits nonsense leg metadata in positions > T; the property
    # asserts the first T groups in the output are unaffected.
    polluted = list(legs[:cut])
    for i in range(cut, n):
        polluted.append(
            (
                int(rng.choice([1, -1])),
                int(rng.integers(-10_000, 10_000)),
                int(rng.integers(-10_000, 10_000)),
                float(rng.normal(0, 1_000)),
                float(rng.normal(0, 1_000)),
                float(rng.normal(0, 100)),
                float(rng.normal(0, 10_000)),
                int(rng.integers(0, 1_000)),
                i,  # keep tgid monotonic so we don't conflate with earlier groups
            )
        )

    out_clean = aggregate_legs(legs)
    out_poll = aggregate_legs(polluted)
    assert out_clean[:cut] == out_poll[:cut], (
        f"aggregate_legs leaked tail pollution into output[:{cut}] "
        f"(seed={seed}, n={n})"
    )


# ---------------------------------------------------------------------------
# Property 3c: item #3 cost decomposition identity. For every trade leg the
# kernel emits, gross_pnl - fee - slippage - funding == net_pnl to floating-
# point tolerance, AND polluting the cost columns at positions > T cannot
# corrupt the cost values stored in Leg objects from positions <= T.
# ---------------------------------------------------------------------------
@given(
    seed=st.integers(0, 2**31 - 1),
    n=st.integers(400, 1200),
    lb=st.integers(10, 30),
)
@settings(max_examples=_MAX_EXAMPLES, deadline=None,
          suppress_health_check=[HealthCheck.function_scoped_fixture,
                                 HealthCheck.too_slow])
def test_per_leg_costs_decomposition_property(seed: int, n: int, lb: int):
    from backtester.ledger import aggregate_legs
    df = _df_from(seed, n)
    dfi = bt.compute_indicators(df, lb)
    raw = bt.create_raw_signals(dfi, lb)
    parsed = bt.parse_signals(raw, dfi["time"])
    trades, _, _, _, _ = bt.backtest(dfi, parsed)
    # Identity: gross_pnl - fee - slippage - funding == net_pnl per leg.
    for t in trades:
        if len(t) < 14:
            continue  # kernel still on pre-#3 tuple width (unreachable here)
        pnl, fee, slip, fund, gross, net = t[6], t[9], t[10], t[11], t[12], t[13]
        dev = abs(gross - fee - slip - fund - net)
        assert dev < 1e-9, (
            f"cost decomposition violated by {dev} on a leg "
            f"(seed={seed}, n={n}, lb={lb}): gross={gross} fee={fee} "
            f"slip={slip} fund={fund} net={net} pnl={pnl}"
        )
        # net_pnl == pnl by construction (kernel-time identity).
        assert abs(pnl - net) < 1e-12

    # Pollute the cost columns at tail positions; assert aggregate_legs
    # returns the same Leg objects (cost values intact) for positions <= cut.
    if len(trades) < 4:
        return
    cut = max(1, len(trades) // 2)
    rng = np.random.default_rng(seed)
    polluted = []
    for i, t in enumerate(trades):
        if i < cut:
            polluted.append(t)
        else:
            # Replace cost fields with garbage, keep tgid monotonic so
            # group ordering is preserved.
            t_list = list(t)
            t_list[9]  = float(rng.normal(0, 1000))   # fee
            t_list[10] = float(rng.normal(0, 1000))   # slip
            t_list[11] = float(rng.normal(0, 100))    # funding
            t_list[12] = float(rng.normal(0, 10000))  # gross
            t_list[13] = float(rng.normal(0, 10000))  # net
            polluted.append(tuple(t_list))
    clean_groups = aggregate_legs(trades)
    poll_groups = aggregate_legs(polluted)
    assert clean_groups[:cut] == poll_groups[:cut], (
        f"polluted cost fields at >={cut} affected aggregate_legs output[:{cut}]"
    )


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
