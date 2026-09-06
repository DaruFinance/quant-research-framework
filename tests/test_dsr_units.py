"""Mathematical unit contracts, independent of cross-language agreement."""
import math
from statistics import NormalDist

import numpy as np
import pandas as pd
import pytest

import backtester as bt
from backtester import dsr, overfit_report


def symmetric_returns(t=1000, sr=0.1):
    # Exact sample SR=sr, skew=0 and raw kurtosis=((T-1)/T)^2.
    return np.tile([-1.0, 1.0], t // 2) + sr * math.sqrt(t / (t - 1))


def test_psr_and_mintrl_against_symmetric_closed_form():
    rets = symmetric_returns()
    t, sr, benchmark = len(rets), 0.1, 0.03
    correction = 1 + (((t - 1) / t) ** 2 - 1) * sr**2 / 4
    expected_psr = NormalDist().cdf((sr - benchmark) * math.sqrt(t - 1) /
                                    math.sqrt(correction))
    expected_mintrl = 1 + correction * (NormalDist().inv_cdf(.95) /
                                       (sr - benchmark)) ** 2
    assert dsr.sharpe_per_observation(rets) == pytest.approx(sr)
    assert dsr.probabilistic_sharpe_ratio(sr, rets, benchmark) == pytest.approx(expected_psr)
    assert dsr.min_track_record_length(sr, rets, benchmark) == pytest.approx(expected_mintrl)
    assert expected_mintrl > 500  # Counting sample size twice would give ~1.


def test_report_uses_observation_units_and_nonzero_benchmark(capsys):
    rets = symmetric_returns()
    trials = [-0.03, 0.01, 0.04, 0.08]
    overfit_report.emit(trials, rets, sr_benchmark=.03)
    output = capsys.readouterr().out
    assert "SR_chosen(per-observation)=0.1000" in output
    assert "OOS observations T=1000" in output
    assert dsr.report(.1, trials, rets) in output
    expected = dsr.min_track_record_length(.1, rets, .03)
    assert f"min_obs={expected:.1f}  (have 1000)" in output
    expected_psr = dsr.probabilistic_sharpe_ratio(.1, rets, .03)
    assert f"P(SR>SR*):{expected_psr:5.3f}" in output


@pytest.mark.parametrize("mode", ["trade", "bar"])
@pytest.mark.parametrize("optimize_rrr", [False, True])
def test_optimizer_captures_each_trials_own_observations(monkeypatch, mode, optimize_rrr):
    # Two different counts with deliberately unrelated ranking statistics.
    # Diagnostics must consume actual returns in both ranking modes.
    returns = {2: np.array([-.01, .02, -.02, .03]),
               3: np.array([-.02, .01, .02, -.03, .04, .01, -.01, .03])}
    frame = pd.DataFrame({"time": pd.date_range("2024-01-01", periods=8, freq="h")})
    monkeypatch.setattr(bt, "OVERFIT_REPORT", True)
    monkeypatch.setattr(bt, "SHARPE_MODE", mode)
    monkeypatch.setattr(bt, "OPTIMIZE_RRR", optimize_rrr)
    monkeypatch.setattr(bt, "SMART_OPTIMIZATION", False)
    monkeypatch.setattr(bt, "DRAWDOWN_CONSTRAINT", None)
    monkeypatch.setattr(bt, "dd_constraint", None)
    monkeypatch.setattr(bt, "compute_indicators", lambda df, lb: df.assign(lb=lb))
    monkeypatch.setattr(bt, "create_raw_signals", lambda df, lb: np.full(len(df), lb))
    monkeypatch.setattr(bt, "parse_signals", lambda raw, times: raw)

    def backtest(df, sig):
        lb = int(df.lb.iloc[0])
        met = {"Trades": len(returns[lb]), "Sharpe": float(lb * 10),
               "PF": 1.0, "MaxDrawdown": 0.0}
        return [], met, [], returns[lb], None

    monkeypatch.setattr(bt, "backtest", backtest)
    monkeypatch.setattr(bt, "_runtime_state", {})
    bt.optimiser(frame, [2, 3], "Sharpe", 1)
    expected = [dsr.sharpe_per_observation(returns[lb]) for lb in [2, 3]]
    assert bt._runtime_state["_last_trial_sharpes"] == pytest.approx(expected)


def test_observation_sharpe_filters_nonfinite_and_rejects_degenerate():
    rets = [-.02, .01, .03, -.01]
    assert dsr.sharpe_per_observation(rets + [math.nan, math.inf]) == pytest.approx(
        dsr.sharpe_per_observation(rets))
    for bad in [[], [1.0], [1.0, 1.0]]:
        assert math.isnan(dsr.sharpe_per_observation(bad))


@pytest.mark.parametrize("t", [100, 1000, 4000])
def test_independent_gaussian_null_does_not_saturate_with_sample_size(t):
    # Distributional regression for the issue's explicitly requested null
    # corpus, not a universal false-positive guarantee for trading returns.
    rng = np.random.default_rng(20260906)
    probabilities = []
    for _ in range(64):
        corpus = rng.standard_normal((t, 64))
        trials = corpus.mean(axis=0) / corpus.std(axis=0, ddof=1)
        chosen = int(np.argmax(trials))
        probabilities.append(dsr.deflated_sharpe_ratio(
            trials[chosen], trials, corpus[:, chosen]))
    probabilities = np.array(probabilities)
    assert .3 < np.median(probabilities) < .7
    assert np.mean(probabilities > .95) < .1
    assert np.mean((probabilities < .01) | (probabilities > .99)) < .1
