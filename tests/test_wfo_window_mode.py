from pathlib import Path

import numpy as np
import pytest

import backtester as bt


REAL_SOL = Path(__file__).parents[1] / "data" / "SOLUSDT_1h.csv"


def test_config_window_mode_round_trips_without_leaking():
    before = bt.WFO_WINDOW_MODE
    cfg = bt.Config.from_module()
    cfg.wfo_window_mode = "expanding"
    with bt.with_config(cfg):
        assert bt.WFO_WINDOW_MODE == "expanding"
        assert bt.Config.from_module().wfo_window_mode == "expanding"
    assert bt.WFO_WINDOW_MODE == before


@pytest.mark.parametrize("value", ["", "expand", "ROLLING"])
def test_config_rejects_unknown_window_mode(value):
    with pytest.raises(ValueError, match="wfo_window_mode"):
        bt.Config(wfo_window_mode=value)


def _run_and_record_optimizer_windows(monkeypatch, tmp_path, mode):
    df = bt.load_ohlc(str(REAL_SOL)).iloc[:15_500].reset_index(drop=True)
    cfg = bt.Config(
        backtest_candles=10_000,
        oos_candles=2_500,
        wfo_trigger_val=1_000,
        wfo_window_mode=mode,
        lookback_range=(48, 50),
        min_trades=1,
        smart_optimization=False,
        optimize_rrr=False,
        use_monte_carlo=False,
        print_equity_curve=False,
        export_path=str(tmp_path / f"{mode}.csv"),
    )
    seen = []
    original = bt.optimiser

    def recording_optimizer(is_df, *args, **kwargs):
        seen.append((len(is_df), is_df["time"].iloc[0], is_df["time"].iloc[-1]))
        return original(is_df, *args, **kwargs)

    monkeypatch.setattr(bt, "optimiser", recording_optimizer)
    monkeypatch.setattr(bt, "ROBUSTNESS_SCENARIOS", {})
    bt.walk_forward(df, None, np.array([1.0]), config=cfg)
    return seen


def test_real_bar_optimizer_uses_rolling_or_expanding_geometry(monkeypatch, tmp_path):
    rolling = _run_and_record_optimizer_windows(monkeypatch, tmp_path, "rolling")
    monkeypatch.undo()
    expanding = _run_and_record_optimizer_windows(monkeypatch, tmp_path, "expanding")

    assert [item[0] for item in rolling] == [10_000, 10_000, 10_000]
    assert [item[0] for item in expanding] == [10_000, 11_000, 12_000]
    assert rolling[0] == expanding[0]
    assert expanding[0][1] == expanding[1][1]
    assert expanding[0][1] == expanding[2][1]
    assert expanding[0][1] != bt.load_ohlc(str(REAL_SOL))["time"].iloc[0]
    assert rolling[0][2] == expanding[0][2]


def test_trade_trigger_uses_same_expanding_is_for_probe_and_final_optimizer(
    monkeypatch, tmp_path
):
    df = bt.load_ohlc(str(REAL_SOL)).iloc[:15_500].reset_index(drop=True)
    cfg = bt.Config(
        backtest_candles=10_000,
        oos_candles=2_500,
        wfo_trigger_mode="trades",
        wfo_trigger_val=20,
        wfo_window_mode="expanding",
        lookback_range=(48, 50),
        min_trades=1,
        smart_optimization=False,
        optimize_rrr=False,
        use_monte_carlo=False,
        print_equity_curve=False,
        export_path=str(tmp_path / "trades-trigger.csv"),
    )
    seen = []
    original = bt.optimiser

    def recording_optimizer(is_df, *args, **kwargs):
        seen.append((len(is_df), is_df["time"].iloc[0], is_df["time"].iloc[-1]))
        return original(is_df, *args, **kwargs)

    monkeypatch.setattr(bt, "optimiser", recording_optimizer)
    monkeypatch.setattr(bt, "ROBUSTNESS_SCENARIOS", {})
    bt.walk_forward(df, None, np.array([1.0]), config=cfg)
    assert len(seen) >= 4 and len(seen) % 2 == 0
    for probe, final in zip(seen[0::2], seen[1::2]):
        assert probe == final
    assert all(item[1] == seen[0][1] for item in seen)


def test_expanding_rejects_insufficient_initial_history():
    df = bt.load_ohlc(str(REAL_SOL)).iloc[:11_999].reset_index(drop=True)
    cfg = bt.Config(
        backtest_candles=10_000,
        oos_candles=2_000,
        wfo_window_mode="expanding",
        use_monte_carlo=False,
    )
    with pytest.raises(ValueError, match=r"len\(data\)"):
        bt.walk_forward(df, None, np.array([1.0]), config=cfg)


@pytest.mark.parametrize("use_regime", [False, True])
def test_future_bar_pollution_cannot_change_completed_window(
    monkeypatch, tmp_path, use_regime
):
    clean = bt.load_ohlc(str(REAL_SOL)).iloc[:15_500].reset_index(drop=True)
    polluted = clean.copy()
    for column in ("open", "high", "low", "close"):
        polluted.loc[14_000:, column] *= 7.0

    def run(frame, suffix):
        cfg = bt.Config(
            backtest_candles=10_000,
            oos_candles=2_500,
            wfo_trigger_val=1_000,
            wfo_window_mode="expanding",
            lookback_range=(48, 51),
            min_trades=1,
            smart_optimization=False,
            optimize_rrr=False,
            use_regime_seg=use_regime,
            use_monte_carlo=False,
            print_equity_curve=False,
            export_path=str(tmp_path / f"{suffix}.csv"),
        )
        completed = []
        original = bt._run_wfo_window

        def recording_window(*args, **kwargs):
            result = original(*args, **kwargs)
            selected = kwargs.get("best_lbs", args[2] if len(args) > 2 else None)
            completed.append((selected, np.asarray(result[0]).copy()))
            return result

        monkeypatch.setattr(bt, "_run_wfo_window", recording_window)
        monkeypatch.setattr(bt, "ROBUSTNESS_SCENARIOS", {})
        bt.walk_forward(frame, None, np.array([1.0]), config=cfg)
        monkeypatch.undo()
        ledger = bt.pd.read_csv(cfg.export_path)
        return completed, ledger

    original, original_ledger = run(clean, f"clean-{use_regime}")
    rerun, rerun_ledger = run(polluted, f"polluted-{use_regime}")
    assert len(original) == len(rerun) == 3
    assert original[0][0]
    assert len(original[0][1]) > 0
    assert original[0][0] == rerun[0][0]
    np.testing.assert_array_equal(original[0][1], rerun[0][1])
    key = lambda ledger: ledger[
        (ledger["window"] == "W01") & (ledger["sample"] == "OOS")
    ].reset_index(drop=True)
    first_original = key(original_ledger)
    first_rerun = key(rerun_ledger)
    assert not first_original.empty
    bt.pd.testing.assert_frame_equal(first_original, first_rerun, check_exact=True)

    current_oos = clean.copy()
    for column in ("open", "high", "low", "close"):
        current_oos.loc[13_000:, column] *= 11.0
    current, _ = run(current_oos, f"current-{use_regime}")
    assert original[0][0] == current[0][0]
