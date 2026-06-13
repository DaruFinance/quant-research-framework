"""Negative-example look-ahead demonstration. Two strategies, one pollution test.

The future-pollution test mirrors the discipline that
tests/test_invariants.py::test_parse_signals_no_lookahead applies to engine
internals, lifted to the user-supplied strategy. Replace close[k:] with
noise; re-run; signals at bars [..k) must match. A leaky strategy diverges.
"""
import os, sys

# Portable repo resolution (matches the paper's reproducibility appendix):
# QRF_PYTHON_REPO / QRF_RUST_REPO env vars, else sensible fallbacks — the
# Python repo is this file's grandparent (listings/ sits in the repo root),
# the Rust repo its sibling. No hardcoded absolute paths.
_HERE = os.path.dirname(os.path.abspath(__file__))
_PY = (os.environ.get("QRF_PYTHON_REPO") or os.environ.get("QRF_PY_DIR")
       or os.path.dirname(_HERE))
_RS = (os.environ.get("QRF_RUST_REPO")
       or os.path.join(os.path.dirname(_PY), "quant-research-framework-rs"))
sys.path.insert(0, os.path.abspath(_PY))
os.environ.setdefault("BT_CSV", os.path.join(_RS, "data", "SOLUSDT_1h.csv"))
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
    return len(diff)

_good = future_pollution_test(good_strategy,  "good (paper Listing 1, .take(idx-1) shift)")
_bad  = future_pollution_test(buggy_strategy, "buggy (deliberate close.shift(-5) peek)")

# The demo is itself a guard: the causal strategy must leak 0 bars and the
# deliberately-leaky one must be caught (>0). Exit non-zero otherwise so it
# can gate `make repro` / CI.
sys.exit(0 if (_good == 0 and _bad > 0) else 1)
