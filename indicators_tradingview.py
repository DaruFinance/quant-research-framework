"""Backwards-compatibility shim. The indicators are now part of the
``backtester`` package (``backtester.indicators``); this module is kept
so that pre-v0.3.0 user scripts that do ``from indicators_tradingview
import ...`` continue to work.

New code should import from ``backtester.indicators`` directly.
"""
from backtester.indicators import (  # noqa: F401
    compute_sma,
    compute_ema,
    compute_macd,
    compute_rsi,
    compute_atr,
    compute_stoch,
)
