"""Negative-example look-ahead demonstration. Two strategies, one pollution test.

The future-pollution test mirrors the discipline that
tests/test_invariants.py::test_parse_signals_no_lookahead applies to engine
internals, lifted to the user-supplied strategy. Replace close[k:] with
noise; re-run; signals at bars [..k) must match. A leaky strategy diverges.
"""
import os, sys
# Resolve the engine and data repos portably: honour QRF_PYTHON_REPO /
# QRF_RUST_REPO when set (as in the paper's Appendix A command), else fall back
# to the sibling layout relative to this file (…/quant-research-framework/listings).
_PY_REPO = os.environ.get(
    "QRF_PYTHON_REPO", os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
_RS_REPO = os.environ.get(
    "QRF_RUST_REPO", os.path.join(os.path.dirname(_PY_REPO), "quant-research-framework-rs"))
sys.path.insert(0, _PY_REPO)
os.environ.setdefault("BT_CSV", os.path.join(_RS_REPO, "data", "SOLUSDT_1h.csv"))
import numpy as np, pandas as pd
import backtester as bt

def _df(n=600, seed=11):
    rng = np.random.default_rng(seed)
    rets = rng.normal(0.00005, 0.01, size=n)
    close = 100.0 * np.exp(np.cumsum(rets))
    times = (pd.to_datetime(1_600_000_000 + np.arange(n)*3600, unit="s",
             utc=True).tz_convert(bt.NY_TZ))
    return pd.DataFrame({"time":times,"open":close,"high":close*1.002,
                         "low":close*0.998,"close":close})

def good_strategy(df, lb):
    fast = df["close"].rolling(lb).mean()
    slow = df["close"].rolling(lb*4).mean()
    sig  = np.where(fast > slow, 1, np.where(fast < slow, -1, 0))
    return np.asarray(sig, dtype=np.int8).take(np.arange(len(df))-1, mode="clip")

def buggy_strategy(df, lb):
    nxt = df["close"].shift(-5).ffill().values
    cur = df["close"].values
    sig = np.where(nxt > cur, 1, np.where(nxt < cur, -1, 0))
    return np.asarray(sig, dtype=np.int8)

def future_pollution_test(strategy_fn, name, k=400):
    rng = np.random.default_rng(99)
    df_clean    = _df()
    df_polluted = df_clean.copy()
    df_polluted.loc[k:, "close"] = rng.normal(100, 5, size=len(df_polluted)-k)
    for col in ("high","low","open"):
        if col == "high": df_polluted.loc[k:, col] = df_polluted.loc[k:, "close"] * 1.002
        if col == "low":  df_polluted.loc[k:, col] = df_polluted.loc[k:, "close"] * 0.998
        if col == "open": df_polluted.loc[k:, col] = df_polluted.loc[k:, "close"]

    sc = strategy_fn(df_clean,    bt.DEFAULT_LB)
    sp = strategy_fn(df_polluted, bt.DEFAULT_LB)
    diff = np.where(sc[:k] != sp[:k])[0]
    if len(diff)==0:
        print(f"[PASS] {name}: 0 of {k} bars affected by post-bar-{k} pollution.")
    else:
        first, last = int(diff[0]), int(diff[-1])
        print(f"[FAIL] {name}: {len(diff)} of {k} bars affected by post-bar-{k} "
              f"pollution; first leak at bar {first}, last at bar {last}.")

future_pollution_test(good_strategy,  "good (paper Listing 1, .take(idx-1) shift)")
future_pollution_test(buggy_strategy, "buggy (deliberate close.shift(-5) peek)")
