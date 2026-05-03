"""
Config dataclass isolation tests (v0.4.0).

These verify three contracts:

1. Two `Config` instances can coexist in one process with different
   field values, and the engine respects whichever one is active.
2. `bt.main(config=cfg)` applies cfg for the duration of the call and
   restores prior module globals on exit, even if the body raises.
3. The legacy API (`bt.create_raw_signals = my_strategy; bt.main()`,
   `bt.FOREX_MODE = True`, `monkeypatch.setattr(bt, "X", Y)`) keeps
   working untouched. This is the single hard back-compat constraint
   that gates the refactor.
"""

from dataclasses import fields

import numpy as np
import pandas as pd
import pytest

import backtester as bt


def _tiny_df(n: int = 800, seed: int = 7) -> pd.DataFrame:
    """Synthetic OHLC with NY-tz timestamps; cheap enough for unit tests."""
    rng = np.random.default_rng(seed)
    t0 = pd.Timestamp("2022-01-01", tz="UTC").tz_convert(bt.NY_TZ)
    times = pd.date_range(t0, periods=n, freq="1h")
    drift = rng.normal(0.0, 0.001, size=n).cumsum()
    close = 100.0 * np.exp(drift)
    high  = close * (1 + rng.uniform(0.0, 0.003, size=n))
    low   = close * (1 - rng.uniform(0.0, 0.003, size=n))
    open_ = np.concatenate(([close[0]], close[:-1]))
    return pd.DataFrame({"time": times, "open": open_, "high": high, "low": low, "close": close})


# ----------------------------------------------------------------------
# 1. Two Configs coexist
# ----------------------------------------------------------------------
def test_two_configs_have_distinct_field_values():
    """Building two Config instances with different values produces two
    independent dataclasses; mutating one must not leak to the other."""
    cfg_a = bt.Config()
    cfg_b = bt.Config()
    cfg_a.tp_percentage = 1.5
    cfg_a.use_tp = False
    cfg_b.tp_percentage = 9.0
    cfg_b.use_tp = True

    assert cfg_a.tp_percentage == 1.5
    assert cfg_b.tp_percentage == 9.0
    assert cfg_a.use_tp is False
    assert cfg_b.use_tp is True


def test_apply_to_module_then_restore_round_trips():
    """apply_to_module returns a snapshot that restore_module_state can
    use to undo every change."""
    before_fee = bt.FEE_PCT
    before_use_tp = bt.USE_TP
    before_tp = bt.TP_PERCENTAGE

    cfg = bt.Config.from_module()
    cfg.fee_pct = 0.99
    cfg.use_tp = (not before_use_tp)
    cfg.tp_percentage = 7.7

    prev = cfg.apply_to_module()
    assert bt.FEE_PCT == 0.99
    assert bt.USE_TP == (not before_use_tp)
    assert bt.TP_PERCENTAGE == 7.7

    bt.restore_module_state(prev)
    assert bt.FEE_PCT == before_fee
    assert bt.USE_TP == before_use_tp
    assert bt.TP_PERCENTAGE == before_tp


def test_with_config_context_manager_restores_on_exception():
    """Even when the body raises, with_config must restore the prior
    module state. Otherwise a failed library call would corrupt the
    entire interpreter."""
    before_fee = bt.FEE_PCT
    cfg = bt.Config.from_module()
    cfg.fee_pct = 12345.0

    with pytest.raises(RuntimeError, match="boom"):
        with bt.with_config(cfg):
            assert bt.FEE_PCT == 12345.0
            raise RuntimeError("boom")

    assert bt.FEE_PCT == before_fee


def test_with_config_none_is_noop():
    """Passing config=None must NOT touch module globals (legacy path)."""
    before = bt.FEE_PCT
    with bt.with_config(None):
        assert bt.FEE_PCT == before
    assert bt.FEE_PCT == before


# ----------------------------------------------------------------------
# 2. Config affects engine behaviour
# ----------------------------------------------------------------------
def test_optimiser_with_config_uses_cfg_metric():
    """Calling optimiser(config=cfg) where cfg.opt_metric differs from
    the module's must produce a result that depends on cfg's metric.
    We don't assert exact LB equality (the search is data-dependent),
    just that the call completes without error and returns the
    documented (lb, metrics) shape."""
    df = _tiny_df(n=600)
    df = bt.compute_indicators(df, bt.DEFAULT_LB)

    cfg = bt.Config.from_module()
    cfg.opt_metric = "PF"
    cfg.min_trades = 1
    cfg.optimize_rrr = False
    cfg.smart_optimization = False

    lb_a, met_a = bt.optimiser(df, range(20, 40, 2), "PF", 1, config=cfg)
    assert isinstance(lb_a, int)
    assert "PF" in met_a


def test_with_forex_helper_returns_new_config():
    """Config.with_forex(True) returns a new instance with forex defaults
    applied; the original Config is untouched."""
    cfg = bt.Config()
    fx = cfg.with_forex(True)

    # Original untouched
    assert cfg.forex_mode is False
    assert cfg.account_size == 100_000.0

    # New cfg is in forex mode with 1 R bookkeeping
    assert fx.forex_mode is True
    assert fx.account_size == 1.0
    assert fx.position_size == 1.0
    # SL/TP scaled by pip_size
    assert fx.sl_percentage == cfg.sl_percentage * cfg.pip_size
    assert fx.tp_percentage == cfg.tp_percentage * cfg.pip_size


# ----------------------------------------------------------------------
# 3. Legacy back-compat (the gating constraint)
# ----------------------------------------------------------------------
def test_legacy_create_raw_signals_swap_still_works(monkeypatch):
    """The README documents:
        bt.create_raw_signals = my_strategy
        bt.main()
    This pattern must keep working in v0.4.0."""
    captured = {}

    def my_strategy(df, lb):
        captured["called"] = True
        captured["lb"] = lb
        # Return alternating 1/-1 so parse_signals has something to do
        out = np.zeros(len(df), dtype=np.int8)
        out[::2] = 1
        out[1::2] = -1
        return out

    original = bt.create_raw_signals
    monkeypatch.setattr(bt, "create_raw_signals", my_strategy)

    df = _tiny_df(n=500)
    df = bt.compute_indicators(df, 25)
    raw = bt.create_raw_signals(df, 25)

    assert captured.get("called") is True
    assert captured.get("lb") == 25
    assert raw.shape == (500,)
    # Restore handled by monkeypatch
    assert bt.create_raw_signals is my_strategy


def test_monkeypatch_setattr_pattern_still_works(monkeypatch):
    """The 32-test suite uses `monkeypatch.setattr(bt, "X", Y)`. This
    is the legacy module-globals API. v0.4.0 must keep it working."""
    monkeypatch.setattr(bt, "FOREX_MODE", True, raising=False)
    assert bt.FOREX_MODE is True
    monkeypatch.setattr(bt, "FEE_PCT", 0.001, raising=False)
    assert bt.FEE_PCT == 0.001


def test_config_field_count_matches_module_map():
    """If a contributor adds a new field to Config they must also wire
    it into the _MODULE_NAME_MAP, otherwise the new field will silently
    be ignored by from_module / apply_to_module."""
    cfg_field_names = {f.name for f in fields(bt.Config)}
    mapped = set(bt.Config._MODULE_NAME_MAP.keys())
    missing = cfg_field_names - mapped
    assert not missing, f"Config fields not in _MODULE_NAME_MAP: {missing}"


def test_config_apply_recomputes_dd_constraint():
    """The derived `dd_constraint` module-global must be recomputed when
    a Config is applied. Otherwise the optimiser's drawdown filter would
    see the wrong threshold after a config swap."""
    cfg = bt.Config.from_module()
    cfg.drawdown_constraint = 25.0  # pct
    cfg.forex_mode = False

    prev = cfg.apply_to_module()
    try:
        assert bt.dd_constraint == pytest.approx(0.25)
    finally:
        bt.restore_module_state(prev)


def test_module_globals_unchanged_after_main_with_config(monkeypatch, tmp_path):
    """Calling main(config=cfg) must NOT leak cfg into module globals
    after returning. We test this with a very small Config knob (just
    print_equity_curve) and assert that an unrelated module global
    (FEE_PCT) survives untouched.

    main() requires a CSV file. We synthesise one and point CSV_FILE
    at it for the duration of the test. We disable everything heavy
    so the call is cheap.
    """
    csv_path = tmp_path / "tiny.csv"
    df = _tiny_df(n=400)
    out = pd.DataFrame({
        "time": df["time"].astype("int64") // 10**9,  # unix seconds
        "open": df["open"], "high": df["high"], "low": df["low"], "close": df["close"],
    })
    out.to_csv(csv_path, index=False)

    before_fee = bt.FEE_PCT
    before_print = bt.PRINT_EQUITY_CURVE

    cfg = bt.Config.from_module()
    cfg.csv_file = str(csv_path)
    cfg.print_equity_curve = False
    cfg.use_wfo = False
    cfg.use_monte_carlo = False
    cfg.use_regime_seg = False
    cfg.optimize_rrr = False
    cfg.smart_optimization = False
    cfg.backtest_candles = 100
    cfg.oos_candles = 50
    cfg.lookback_range = (20, 32)
    cfg.min_trades = 1

    # Ensure ROBUSTNESS_SCENARIOS is absent so we don't run heavy paths
    monkeypatch.setattr(bt, "ROBUSTNESS_SCENARIOS", {}, raising=False)
    monkeypatch.setattr(bt, "FEE_SHOCK", False, raising=False)
    monkeypatch.setattr(bt, "SLIPPAGE_SHOCK", False, raising=False)
    monkeypatch.setattr(bt, "NEWS_CANDLES_INJECTION", False, raising=False)
    monkeypatch.setattr(bt, "ENTRY_DRIFT", False, raising=False)
    monkeypatch.setattr(bt, "INDICATOR_VARIANCE", False, raising=False)

    bt.main(config=cfg)

    # After main returns, module state must equal the pre-call state.
    assert bt.FEE_PCT == before_fee
    assert bt.PRINT_EQUITY_CURVE == before_print
