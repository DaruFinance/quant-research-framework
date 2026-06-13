#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Walk-forward backtester for forex and crypto datasets.

Execution flow:
1. Runs baseline in-sample (IS) and out-of-sample (OOS) tests.
2. Optionally runs rolling walk-forward optimization (WFO).
3. Reports performance and robustness metrics.
4. Optionally plots equity curves.

The implementation is designed to avoid look-ahead bias.
"""

__version__ = "0.6.0"

import os, math, random
import time as pytime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import sqrt
from matplotlib.gridspec import GridSpec
import pytz
from datetime import datetime, time
from .indicators import compute_atr, compute_rsi
from numba import njit, types
from numba.typed import List

# Configuration
# CSV_FILE can be overridden without editing this file by setting the BT_CSV
# environment variable — handy when running a strategy example from examples/.
CSV_FILE            = os.environ.get("BT_CSV", "data/your_ohlc.csv") # <-- put your CSV here (not included in repo)

ACCOUNT_SIZE        = 100_000.0      # total account equity in USD
RISK_AMOUNT         = 2_500.0       # dollars you are willing to lose per trade
SLIPPAGE_PCT        = 0.03           # slippage % per order (entry and exit)
FEE_PCT             = 0.02         # commission % per order (entry and exit)
FUNDING_FEE         = 0.01         # funding fee % charged at 00:00,08:00,16:00 UTC (crypto only)

# Session trading window (New York time)
TRADE_SESSIONS = False                # if True, only trade during NY session
SESSION_START   = "8:00"            # NY open time (HH:MM)
SESSION_END     = "16:50"            # NY close time (HH:MM)
NY_TZ           = pytz.timezone("America/New_York")
DEFAULT_LB          = 50                         # for RAW baseline
LOOKBACK_RANGE      = (int(DEFAULT_LB*0.25), int(DEFAULT_LB*1.5)+1)

BACKTEST_CANDLES    = 10_000                     # length of IS window
OOS_CANDLES         = 90_000                    # length of RAW-OOS
USE_OOS2            = False                     # if True, split OOS into two windows

OPT_METRIC          = "Sharpe"              # ROI, PF, Sharpe, WinRate, Exp, MaxDrawdown, Consistency
# How the reported (and, since OPT_METRIC may be "Sharpe", optimised) Sharpe is
# computed. "trade" (default, unchanged) = the per-trade statistic
# mean(trade returns)/std * sqrt(trade count) -- a t-statistic of the mean
# trade return, robust for sparse-trade event strategies. "bar" = the standard
# calendar-time Sharpe on the per-bar mark-to-market equity curve, annualised by
# the bar frequency (matches the convention used by vectorbt / empyrical). The
# two are different statistics by design; see the paper's Sharpe-convention note.
SHARPE_MODE         = "trade"               # "trade" | "bar"
MIN_TRADES          = 10
SMART_OPTIMIZATION  = True    # True to skip spiky optimisations
DRAWDOWN_CONSTRAINT = None    # Skip optimizations with a drawdown higher than the value input. Use "None" for OFF

PRINT_EQUITY_CURVE  = True                       # plot
USE_MONTE_CARLO     = True                       # on first IS only
MC_RUNS             = 1000                       # simulations

USE_SL         = True    # whether to enable stop loss
SL_PERCENTAGE  = 1.0     # stop loss percent (e.g. 1.0 for 1%)
USE_TP         = True    # whether to enable take profit
TP_PERCENTAGE  = 3.0     # take profit percent (e.g. 2.0 for 2%)
OPTIMIZE_RRR  = True     # set True to auto-optimise the Risk-to-Reward ratio

CONFLUENCES: str | None = None        # "RSIge50", "Pge0.8", ...
MASK_EXITS = False        # confluence filter: when False (default), the
                          # confluence rule applies only to entries (codes 1, 3);
                          # exit codes (2, 4) pass through unconditionally.
                          # When True, the rule applies to exits too — useful for
                          # strategies where exit signals also need confirmation.

LEGACY_SIDE_BUG = False   # RRR-optimisation side comparison: the original
                          # `side == 'long'` test compared int8 against str and
                          # always took the else (short) branch. Default is now
                          # the corrected `side == 1` test. Set True to
                          # reproduce numerical results of research published
                          # against versions <= v0.2.4 that depended on the
                          # buggy code path. See CHANGELOG v0.2.5.

# Forex mode: when True, use pip-based risk units.
FOREX_MODE = False        # Convert percentage distances to pip distances.

FILTER_REGIMES     = False      # works only when USE_REGIME_SEG == True
FILTER_DIRECTIONS  = False      # blocks long / short based on IS stats

USE_REGIME_SEG       = False   # enable regime segmentation?

# Regime detector contract (pluggable; reassign at module level to override).
# REGIME_LABELS may have length 2..5. Default = EMA-200 / 8-bar consistency.
REGIME_LABELS: list[str] = ['Uptrend', 'Downtrend', 'Ranging']

# Robustness tests
FEE_SHOCK              = False          # Multiply fees for stress testing.
SLIPPAGE_SHOCK         = False          # True to apply slippage shock (5x slippage)
NEWS_CANDLES_INJECTION = False          # Inject synthetic high-volatility candles.
ENTRY_DRIFT            = False          # Shift entries one candle forward.
INDICATOR_VARIANCE     = False          # Perturb selected lookback by +/- 1.

# Optional queued robustness scenarios (run separately, up to 5).
ROBUSTNESS_SCENARIOS = {
    "Test 1": ("ENTRY_DRIFT",),
    "Test 2": ("FEE_SHOCK",),
    "Test 3": ("SLIPPAGE_SHOCK",),
    "Test 4": ("ENTRY_DRIFT", "INDICATOR_VARIANCE",),
}
MAX_ROBUSTNESS_SCENARIOS = 5

# Walk-forward settings
USE_WFO             = True                       # do rolling windows?
WFO_TRIGGER_MODE    = "candles"                   # "candles" or "trades"
WFO_TRIGGER_VAL     = 5000                         # n-candles or n-trades per window

# Overfitting-statistics report (item #3). OFF by default: when on, an
# ADDITIVE block (DSR/PSR/PBO/MinTRL/MinBTL/haircut) is printed after the
# walk-forward run. The block's lines never carry the metric body
# parity_common.LINE_RE matches, so existing parity harnesses stay
# byte-identical. Env override: QRF_OVERFIT=1 turns it on without code edits.
OVERFIT_REPORT      = (os.environ.get("QRF_OVERFIT") == "1")
EMIT_OPT_SURFACE    = os.environ.get("EMIT_OPT_SURFACE", "0") in ("1", "true", "True")
EMIT_OPT_SURFACE_SL = os.environ.get("EMIT_OPT_SURFACE_SL", "0") in ("1", "true", "True")


EXPORT_PATH         = "trade_list.csv"
METRICS             = ["ROI","PF","Sharpe","WinRate","Exp","MaxDrawdown"]


blocked_regimes     = set()          # whole regimes blocked (Uptrend / Downtrend / Ranging)
blocked_directions  = set()          # global longs / shorts blocked
blocked_pairs       = {}             # per-regime direction blocks  e.g. {'Uptrend':{'short'}}
last_unfiltered_raw = None
ORIGINAL_OOS        = OOS_CANDLES               # preserve the single-window length
if USE_OOS2:
    # NOTE: OOS_CANDLES doubling is applied at module-import time. Flipping
    # USE_OOS2 dynamically AFTER `import backtester` has no effect on this
    # constant. To switch modes mid-process, set OOS_CANDLES = ORIGINAL_OOS * 2
    # (or back) yourself, or re-import. This is the documented constraint.
    OOS_CANDLES     = ORIGINAL_OOS * 2         # everywhere uses the doubled window

# Pip size: explicit env override wins. The legacy substring fallback
# (path containing "JPY") still triggers for convenience but emits a
# warning so users know it can be wrong on filenames like "FUJPYR.csv".
def _resolve_pip_size(csv_path: str) -> float:
    override = os.environ.get("BT_PIP_SIZE")
    if override is not None:
        try:
            return float(override)
        except ValueError:
            raise ValueError(f"BT_PIP_SIZE={override!r} is not a valid float")
    if "JPY" in csv_path:
        import warnings
        warnings.warn(
            f"PIP_SIZE auto-set to 0.01 because CSV path {csv_path!r} contains "
            "'JPY'. This substring heuristic is fragile; set BT_PIP_SIZE "
            "explicitly to silence this warning.",
            stacklevel=2,
        )
        return 0.01
    return 0.0001

PIP_SIZE   = _resolve_pip_size(CSV_FILE)
if FOREX_MODE:
    # turn your percent vars into fraction-of-price pip distances
    SL_PERCENTAGE *= PIP_SIZE
    TP_PERCENTAGE *= PIP_SIZE
if FOREX_MODE:
    # risk is fixed at 1 R
    RISK_AMOUNT   = 1.0
    ACCOUNT_SIZE  = 1.0
    POSITION_SIZE = 1.0
else:
    POSITION_SIZE = RISK_AMOUNT
# Derive internal fractional constraint
if DRAWDOWN_CONSTRAINT is None:
    dd_constraint = None
else:
    # FOREX_MODE  number is already in R; otherwise convert %  fraction
    dd_constraint = DRAWDOWN_CONSTRAINT if FOREX_MODE else DRAWDOWN_CONSTRAINT / 100.0
FAST_EMA_SPAN = 20

# NOTE: the FileNotFoundError check used to live at module top-level here.
# It has been moved into load_ohlc() and main() so `import backtester` works
# in pip-install / library workflows that never read CSVs.

# ============================================================================
# Config dataclass — v0.4.0 library-grade configuration surface.
#
# Background: prior to v0.4.0 every tunable lived as a module-level UPPERCASE
# constant (FEE_PCT, USE_TP, FOREX_MODE, ...). The engine read those constants
# directly via globals(); a handful of functions even rebound them with
# `global FEE_PCT, ...` to apply RRR optimisation or robustness shocks. This
# made the engine non-fork-safe and prevented two backtests with different
# configurations from coexisting in one process.
#
# `Config` carries every configurable field in one immutable-by-convention
# dataclass. The engine still reads from module globals as its default source
# (back-compat: `bt.FOREX_MODE = True; bt.main()` keeps working, as do all
# `monkeypatch.setattr(bt, "X", Y)` test patterns), but library callers can
# now pass `bt.main(config=cfg)` and the engine will apply that Config to the
# module for the duration of the call, then restore the prior values.
#
# This pattern eliminates every `global X` statement inside the engine
# (those that rebound configuration use `globals()['X'] = ...` instead;
# those that mutated runtime caches now use the `_runtime_state` dict),
# giving us an audit-clean surface (`grep -c "^    global " == 0`) without
# breaking the long-standing public API.
# ============================================================================
from dataclasses import dataclass, field, fields, asdict
from typing import Any, Optional


@dataclass
class Config:
    """
    Snapshot of every configurable knob in the backtester.

    Defaults mirror the module-level constants so `Config()` is always
    equivalent to "use the current module defaults". To copy the live
    module state (including any `bt.X = Y` overrides set after import),
    use `Config.from_module()`. To execute a backtest with a Config
    without leaking values into module globals, pass `config=cfg` to
    `main()` / `walk_forward()` / `optimiser()` etc., or use the
    `with_config(cfg)` context manager directly.
    """
    # Account / risk
    account_size: float = 100_000.0
    risk_amount: float = 2_500.0
    position_size: float = 2_500.0  # derived: RISK_AMOUNT (or 1.0 in forex)
    slippage_pct: float = 0.03
    fee_pct: float = 0.02
    funding_fee: float = 0.01

    # Sessions
    trade_sessions: bool = False
    session_start: str = "8:00"
    session_end: str = "16:50"

    # Lookback / windowing
    default_lb: int = 50
    lookback_range: tuple = (12, 76)
    backtest_candles: int = 10_000
    oos_candles: int = 90_000
    use_oos2: bool = False

    # Optimiser
    opt_metric: str = "Sharpe"
    min_trades: int = 10
    smart_optimization: bool = True
    drawdown_constraint: Optional[float] = None
    fast_ema_span: int = 20

    # Plotting / Monte Carlo
    print_equity_curve: bool = True
    use_monte_carlo: bool = True
    mc_runs: int = 1000

    # SL / TP / RRR
    use_sl: bool = True
    sl_percentage: float = 1.0
    use_tp: bool = True
    tp_percentage: float = 3.0
    optimize_rrr: bool = True

    # Confluence + legacy
    confluences: Optional[str] = None
    mask_exits: bool = False
    legacy_side_bug: bool = False

    # Forex
    forex_mode: bool = False
    pip_size: float = 0.0001

    # Regime / filters
    filter_regimes: bool = False
    filter_directions: bool = False
    use_regime_seg: bool = False
    regime_labels: list = field(default_factory=lambda: ['Uptrend', 'Downtrend', 'Ranging'])

    # Robustness
    fee_shock: bool = False
    slippage_shock: bool = False
    news_candles_injection: bool = False
    entry_drift: bool = False
    indicator_variance: bool = False

    # Walk-forward
    use_wfo: bool = True
    wfo_trigger_mode: str = "candles"
    wfo_trigger_val: int = 5000

    # Overfitting-statistics report (item #3); default OFF.
    overfit_report: bool = False

    # ---- item #1 (IS isosurface emit) ----
    emit_opt_surface: bool = False
    emit_opt_surface_sl: bool = False

    # I/O
    csv_file: str = "data/your_ohlc.csv"
    export_path: str = "trade_list.csv"

    # ------------------------------------------------------------------
    # Mapping between Config field names and module-level UPPERCASE names.
    # The two surfaces use different casing conventions: Config follows
    # PEP 8 (lower_snake), the legacy module API uses ALL_CAPS. We keep
    # the mapping explicit so a typo on either side is loud rather than
    # silent.
    # ------------------------------------------------------------------
    _MODULE_NAME_MAP = {
        'account_size': 'ACCOUNT_SIZE',
        'risk_amount': 'RISK_AMOUNT',
        'position_size': 'POSITION_SIZE',
        'slippage_pct': 'SLIPPAGE_PCT',
        'fee_pct': 'FEE_PCT',
        'funding_fee': 'FUNDING_FEE',
        'trade_sessions': 'TRADE_SESSIONS',
        'session_start': 'SESSION_START',
        'session_end': 'SESSION_END',
        'default_lb': 'DEFAULT_LB',
        'lookback_range': 'LOOKBACK_RANGE',
        'backtest_candles': 'BACKTEST_CANDLES',
        'oos_candles': 'OOS_CANDLES',
        'use_oos2': 'USE_OOS2',
        'opt_metric': 'OPT_METRIC',
        'min_trades': 'MIN_TRADES',
        'smart_optimization': 'SMART_OPTIMIZATION',
        'drawdown_constraint': 'DRAWDOWN_CONSTRAINT',
        'fast_ema_span': 'FAST_EMA_SPAN',
        'print_equity_curve': 'PRINT_EQUITY_CURVE',
        'use_monte_carlo': 'USE_MONTE_CARLO',
        'mc_runs': 'MC_RUNS',
        'use_sl': 'USE_SL',
        'sl_percentage': 'SL_PERCENTAGE',
        'use_tp': 'USE_TP',
        'tp_percentage': 'TP_PERCENTAGE',
        'optimize_rrr': 'OPTIMIZE_RRR',
        'confluences': 'CONFLUENCES',
        'mask_exits': 'MASK_EXITS',
        'legacy_side_bug': 'LEGACY_SIDE_BUG',
        'forex_mode': 'FOREX_MODE',
        'pip_size': 'PIP_SIZE',
        'filter_regimes': 'FILTER_REGIMES',
        'filter_directions': 'FILTER_DIRECTIONS',
        'use_regime_seg': 'USE_REGIME_SEG',
        'regime_labels': 'REGIME_LABELS',
        'fee_shock': 'FEE_SHOCK',
        'slippage_shock': 'SLIPPAGE_SHOCK',
        'news_candles_injection': 'NEWS_CANDLES_INJECTION',
        'entry_drift': 'ENTRY_DRIFT',
        'indicator_variance': 'INDICATOR_VARIANCE',
        'use_wfo': 'USE_WFO',
        'wfo_trigger_mode': 'WFO_TRIGGER_MODE',
        'wfo_trigger_val': 'WFO_TRIGGER_VAL',
        'overfit_report': 'OVERFIT_REPORT',
        'emit_opt_surface': 'EMIT_OPT_SURFACE',
        'emit_opt_surface_sl': 'EMIT_OPT_SURFACE_SL',
        'csv_file': 'CSV_FILE',
        'export_path': 'EXPORT_PATH',
    }

    @classmethod
    def from_module(cls) -> 'Config':
        """Snapshot the current module-level globals into a Config instance.

        Useful when you want to start from "whatever is currently set" and
        then tweak a few fields:

            cfg = bt.Config.from_module()
            cfg.use_tp = False
            bt.main(config=cfg)
        """
        kwargs = {}
        g = globals()
        for fname, mname in cls._MODULE_NAME_MAP.items():
            if mname in g:
                kwargs[fname] = g[mname]
        return cls(**kwargs)

    def apply_to_module(self) -> dict:
        """Write every field of this Config back to the module globals.

        Returns the previous values as a dict, so callers can restore them
        via `restore_module_state(prev)`. Use `with_config(cfg)` instead of
        calling this directly when possible — the context manager guarantees
        restore on exception.

        The derived `dd_constraint` and any forex-mode-driven overrides
        are also recomputed so the engine sees a consistent module state.
        """
        g = globals()
        prev = {}
        for fname, mname in self._MODULE_NAME_MAP.items():
            prev[mname] = g.get(mname)
            g[mname] = getattr(self, fname)
        # Recompute derived dd_constraint exactly as the import-time block does
        prev['dd_constraint'] = g.get('dd_constraint')
        if self.drawdown_constraint is None:
            g['dd_constraint'] = None
        else:
            g['dd_constraint'] = (
                self.drawdown_constraint
                if self.forex_mode
                else self.drawdown_constraint / 100.0
            )
        return prev

    def with_forex(self, on: bool = True) -> 'Config':
        """Return a copy with forex-mode defaults applied.

        Mirrors the import-time `if FOREX_MODE:` block in the legacy module:
        SL/TP get scaled by pip_size, account/risk/position go to 1.0 R-units.
        """
        new = Config(**{f.name: getattr(self, f.name) for f in fields(self)})
        new.forex_mode = on
        if on:
            new.sl_percentage = self.sl_percentage * new.pip_size
            new.tp_percentage = self.tp_percentage * new.pip_size
            new.risk_amount = 1.0
            new.account_size = 1.0
            new.position_size = 1.0
        return new

    def with_sessions(self, on: bool, start: str = "8:00", end: str = "16:50") -> 'Config':
        new = Config(**{f.name: getattr(self, f.name) for f in fields(self)})
        new.trade_sessions = on
        new.session_start = start
        new.session_end = end
        return new

    def with_oos2(self, on: bool) -> 'Config':
        new = Config(**{f.name: getattr(self, f.name) for f in fields(self)})
        new.use_oos2 = on
        # If user starts from import-time defaults the OOS doubling
        # already happened; we re-derive against the original single-window.
        base = new.oos_candles if not self.use_oos2 else (new.oos_candles // 2)
        new.oos_candles = base * 2 if on else base
        return new


def restore_module_state(prev: dict) -> None:
    """Restore the module globals from the dict returned by apply_to_module()."""
    g = globals()
    for k, v in prev.items():
        g[k] = v


from contextlib import contextmanager


@contextmanager
def with_config(cfg: Optional[Config]):
    """Context manager: temporarily apply `cfg` to the module globals.

    On entry, snapshots the current module state and applies `cfg`. On exit
    (even if the body raises), restores the previous state. If `cfg` is None,
    runs as a no-op (yields immediately) so call sites can wrap the body
    unconditionally:

        with with_config(config):
            ... engine body ...
    """
    if cfg is None:
        yield
        return
    prev = cfg.apply_to_module()
    try:
        yield
    finally:
        restore_module_state(prev)


# ----------------------------------------------------------------------
# Runtime-state holder: absorbs the per-run scratch values that used to
# live as bare module globals (`last_unfiltered_raw`, `_last_df`, ...).
# By storing them in a dict we can mutate without `global` declarations
# and we keep a single audit point should we ever want to make these
# per-call (e.g. truly fork-safe). Today they remain process-global —
# matching the prior contract — but the surface is cleaner.
# ----------------------------------------------------------------------
_runtime_state: dict[str, Any] = {
    'last_unfiltered_raw': None,
    '_last_df': None,
    '_last_lb': None,
}

def in_session(ts: datetime) -> bool:
    """ts is a timezone-aware NY datetime."""
    start = datetime.strptime(SESSION_START, "%H:%M").time()
    end   = datetime.strptime(SESSION_END,   "%H:%M").time()
    local = ts.timetz()  # keeps tzinfo
    return start <= local.replace(tzinfo=None) < end


def _trade_csv_lock_path(path: str) -> str:
    return f"{path}.lock"


def _acquire_trade_csv_lock(path: str, timeout_s: float = 120.0, stale_s: float = 6 * 3600):
    """
    Lightweight cross-process lock using an adjacent .lock file.
    Prevents concurrent workers from deleting/appending the same CSV on Windows.
    """
    lock_path = _trade_csv_lock_path(path)
    deadline = pytime.time() + timeout_s

    while True:
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            try:
                os.write(fd, f"pid={os.getpid()} ts={pytime.time():.3f}\n".encode("utf-8"))
            except OSError:
                pass
            return fd, lock_path
        except FileExistsError:
            # Clean up stale lock if a worker crashed and left the sentinel behind.
            try:
                age = pytime.time() - os.path.getmtime(lock_path)
                if age > stale_s:
                    try:
                        os.remove(lock_path)
                        continue
                    except OSError:
                        pass
            except OSError:
                pass

            if pytime.time() >= deadline:
                raise TimeoutError(f"Timed out waiting for CSV lock: {lock_path}")
            pytime.sleep(0.10)


def _release_trade_csv_lock(lock_fd, lock_path: str):
    try:
        os.close(lock_fd)
    except OSError:
        pass
    try:
        os.remove(lock_path)
    except OSError:
        pass


def _safe_remove_trade_csv(path: str, retries: int = 20):
    if not os.path.exists(path):
        return

    last_err = None
    for attempt in range(retries):
        lock_fd, lock_path = _acquire_trade_csv_lock(path)
        try:
            if not os.path.exists(path):
                return
            os.remove(path)
            return
        except PermissionError as e:
            last_err = e
        finally:
            _release_trade_csv_lock(lock_fd, lock_path)
        pytime.sleep(min(0.05 * (attempt + 1), 0.5))

    if last_err is not None:
        raise last_err


def _safe_append_or_write_trade_csv(df_export: pd.DataFrame, path: str, write_header: bool, retries: int = 20):
    last_err = None
    mode = 'w' if write_header else 'a'
    for attempt in range(retries):
        lock_fd, lock_path = _acquire_trade_csv_lock(path)
        try:
            df_export.to_csv(
                path,
                index=False,
                header=write_header,
                mode=mode
            )
            return
        except PermissionError as e:
            last_err = e
        finally:
            _release_trade_csv_lock(lock_fd, lock_path)
        pytime.sleep(min(0.05 * (attempt + 1), 0.5))

    if last_err is not None:
        raise last_err

# 1. LOAD DATA
def load_ohlc(path: str) -> pd.DataFrame:
    """
    Load OHLC CSV where 'time' is UNIX seconds (UTC),
    and convert to America/New_York timezone (with DST).
    Returns a DataFrame with a timezone-aware 'time' column.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"CSV file not found: {path}\n\n"
            "Put your OHLC CSV at that path, or change CSV_FILE / set BT_CSV.\n"
            "You can generate one with binance_ohlc_downloader.py (see README)."
        )
    _avail = pd.read_csv(path, nrows=0).columns
    _cols = ['time', 'open', 'high', 'low', 'close'] + (['volume'] if 'volume' in _avail else [])
    df = pd.read_csv(path, usecols=_cols)
    if 'volume' in _cols:
        df['volume'] = df['volume'].fillna(0.0)
    # 1) Parse UNIX seconds as UTC timestamps
    # 2) Convert them into America/New_York (handles DST automatically)
    df['time'] = (
        pd.to_datetime(df['time'], unit='s', utc=True)
          .dt.tz_convert(NY_TZ)
    )
    return df.sort_values('time').reset_index(drop=True)


# 2. INDICATORS (EMA crossover)
def compute_indicators(df, lb):
    return df.assign(
        **{
            'EMA_20': df['close'].ewm(span=20, adjust=False).mean(),
            'EMA_200': df['close'].ewm(span=200, adjust=False).mean(),
            f'EMA_{lb}': df['close'].ewm(span=lb, adjust=False).mean()
        }
    )


_last_df = None
_last_lb = None


def make_codes(df, raw, lb):

    raw = np.asarray(raw)
    if raw.ndim != 1:
        raw = raw.ravel()


    # 1) RSI(previous bar) a 50
    rsi_prev = compute_rsi(df, 14).shift(1).values
    ok_rsi   = (rsi_prev >= 50)

    if CONFLUENCES == "RSIge50":
        keep = ok_rsi

    elif CONFLUENCES == "RSIge40":
        keep = (compute_rsi(df, 14).shift(1).values >= 40)

    elif CONFLUENCES == "Pge0.8":
        close = df['close']
        low   = close.rolling(window=lb, min_periods=lb).min().shift(1)
        high  = close.rolling(window=lb, min_periods=lb).max().shift(1)
        P     = ((close - low) / (high - low)).shift(1).values
        keep = ((raw ==  1) & (P >= 0.8)) | ((raw == -1) & (P <= 0.2))

    elif CONFLUENCES == "Pge0.7":
        close = df['close']
        low   = close.rolling(window=lb, min_periods=lb).min().shift(1)
        high  = close.rolling(window=lb, min_periods=lb).max().shift(1)
        P     = ((close - low) / (high - low)).shift(1).values
        keep = ((raw ==  1) & (P >= 0.7)) | ((raw == -1) & (P <= 0.3))

    elif CONFLUENCES == "BW_filter":
        denom = (df['high'] - df['low']).shift(1).replace(0, np.nan)
        bw    = ((df['close'] - df['open']).abs().shift(1) / denom).fillna(0).values
        keep = np.ones_like(raw, dtype=bool)
        keep[(raw == 1) & (bw <= 0.7)] = False

    elif CONFLUENCES == "pi":
        atr     = compute_atr(df, lb).shift(1)
        body    = (df['close'] - df['open']).abs().shift(1)
        ratio   = body / atr
        pi_mid  = np.pi / 4
        tol     = 0.05
        valid   = ratio.between(pi_mid - tol, pi_mid + tol) & ratio.notna()
        keep    = valid.values

    elif CONFLUENCES == "vr":
        ret      = df['close'].diff()
        sigma1   = ret.rolling(window=lb, min_periods=lb).var(ddof=0)
        r2       = df['close'] - df['close'].shift(2)
        sigma2   = r2.rolling(window=lb, min_periods=lb).var(ddof=0)
        vr       = sigma2 / (2 * sigma1)
        vr_shift = vr.shift(1)
        valid    = (vr_shift >= 1) & vr_shift.notna()
        keep     = valid.values

    elif CONFLUENCES == "kurtosis":
        r       = df['close'].pct_change()
        mu      = r.rolling(window=lb, min_periods=lb).mean()
        dev     = r - mu
        m2      = dev.pow(2).rolling(window=lb, min_periods=lb).mean()
        m4      = dev.pow(4).rolling(window=lb, min_periods=lb).mean()
        kappa   = m4.div(m2.pow(2))
        valid   = kappa.shift(1) >= 5.0
        keep    = valid.fillna(False).values

    elif CONFLUENCES == "kurtosis10":
        r       = df['close'].pct_change()
        mu      = r.rolling(window=lb, min_periods=lb).mean()
        dev     = r - mu
        m2      = dev.pow(2).rolling(window=lb, min_periods=lb).mean()
        m4      = dev.pow(4).rolling(window=lb, min_periods=lb).mean()
        kappa   = m4.div(m2.pow(2))
        valid   = kappa.shift(1) >= 10.0
        keep    = valid.fillna(False).values

    elif CONFLUENCES == "skew":
        r     = df['close'].pct_change()
        mu    = r.rolling(window=lb, min_periods=lb).mean()
        sd    = r.rolling(window=lb, min_periods=lb).std().replace(0, np.nan)
        z     = (r - mu) / sd
        m3    = z.pow(3).rolling(window=lb, min_periods=lb).mean()
        skew  = m3.shift(1)
        keep  = (skew.abs() >= 0.5) & skew.notna()
        keep  = keep.values.astype(bool)

    elif CONFLUENCES == "skew0.75":
        r     = df['close'].pct_change()
        mu    = r.rolling(window=lb, min_periods=lb).mean()
        sd    = r.rolling(window=lb, min_periods=lb).std().replace(0, np.nan)
        z     = (r - mu) / sd
        m3    = z.pow(3).rolling(window=lb, min_periods=lb).mean()
        skew  = m3.shift(1)
        keep  = (skew.abs() >= 0.75) & skew.notna()
        keep  = keep.values.astype(bool)

    elif CONFLUENCES == "atr_pct":
        atr = compute_atr(df, lb)
        lo  = atr.rolling(window=lb, min_periods=lb).min().shift(1)
        hi  = atr.rolling(window=lb, min_periods=lb).max().shift(1)
        pct = (atr.shift(1) - lo) / (hi - lo)
        keep = (pct >= 0.7) & pct.notna()
        keep = keep.values.astype(bool)

    elif CONFLUENCES == "atr_pct0.8":
        atr = compute_atr(df, lb)
        lo  = atr.rolling(window=lb, min_periods=lb).min().shift(1)
        hi  = atr.rolling(window=lb, min_periods=lb).max().shift(1)
        pct = (atr.shift(1) - lo) / (hi - lo)
        keep = (pct >= 0.8) & pct.notna()
        keep = keep.values.astype(bool)

    elif CONFLUENCES == "burstfreq":
        ret   = df['close'].diff()
        sd    = ret.rolling(window=lb, min_periods=lb).std().replace(0, np.nan)
        bursts= (ret.abs() > sd).astype(float)
        freq  = bursts.rolling(window=lb, min_periods=lb).mean().shift(1)
        keep  = (freq >= 0.2) & freq.notna()
        keep  = keep.values.astype(bool)

    elif CONFLUENCES == "TinyBody":
        body  = (df['close'] - df['open']).abs()
        range_ = (df['high'] - df['low'])
        cond  = body < (range_ * 0.1)
        keep  = cond.shift(1).fillna(False).values

    elif CONFLUENCES == "NoNewLowGreen":
        cond = (df['close'] > df['open']) & (df['low'] > df['close'].shift(5))
        keep = cond.shift(1).fillna(False).values

    elif CONFLUENCES == "RangeSpike":
        range_   = df['high'] - df['low']
        avg_rng  = range_.rolling(window=20, min_periods=20).mean()
        cond     = range_ > (1.3 * avg_rng)
        keep     = cond.shift(1).fillna(False).values

    elif CONFLUENCES == "YesterdayPeak":
        cond = df['close'].rolling(window=3, min_periods=3).max() == df['close'].shift(1)
        keep = cond.shift(1).fillna(False).values

    elif CONFLUENCES == "DeadFlat10":
        ret  = (df['close'] / df['close'].shift(10) - 1).abs()
        cond = ret < 0.005
        keep = cond.shift(1).fillna(False).values

    elif CONFLUENCES == "InsideBar":
        cond = (
            (df['high'].shift(1) <= df['high']) &
            (df['low'].shift(1) >= df['low'])
        )
        keep = cond.shift(1).fillna(False).values

    elif CONFLUENCES == "SameDirection":
        curr_up  = df['close'] > df['open']
        prev_up  = df['close'].shift(1) > df['open'].shift(1)
        cond     = curr_up == prev_up
        keep     = cond.shift(1).fillna(False).values

    elif CONFLUENCES == "TopOfRange":
        highest = df['high'].rolling(window=50, min_periods=50).max()
        cond    = df['close'] > (highest * 0.99)
        keep    = cond.shift(1).fillna(False).values

    elif CONFLUENCES == "VolContraction":
        rng     = df['high'] - df['low']
        short   = rng.rolling(window=10, min_periods=10).std()
        long    = rng.rolling(window=100, min_periods=100).std()
        cond    = short < long
        keep    = cond.shift(1).fillna(False).values

    elif CONFLUENCES == "EMAHug":
        ema21 = df['close'].rolling(window=21, min_periods=21).mean()
        dev   = (df['close'] - ema21).abs()
        range_ = (df['high'] - df['low'])
        cond  = dev < (range_ * 0.92)
        keep  = cond.shift(1).fillna(False).values

    else:
        keep = np.ones_like(raw, dtype=bool)

    # ---- HARDEN keep + ALIGN LENGTHS (this is the actual fix) ----
    keep = np.asarray(keep)
    if keep.ndim != 1:
        keep = keep.ravel()

    # ---- HARDEN keep + ALIGN LENGTHS (robust to object dtype) ----
    keep = np.asarray(keep)
    if keep.ndim != 1:
        keep = keep.ravel()

    # Force keep into strict boolean mask safely:
    # - bool stays bool
    # - numeric: True if nonzero and finite-ish (NaN -> False handled via pandas, but here just treat NaN as False)
    # - object: try numeric conversion; if fails, fall back to (x is True)
    if keep.dtype != np.bool_:
        try:
            # Attempt numeric coercion (handles object arrays containing numbers/None/NaN)
            keep_num = pd.to_numeric(keep, errors="coerce").to_numpy(dtype=float, na_value=np.nan)
            keep = (keep_num != 0) & (~np.isnan(keep_num))
        except Exception:
            # Last resort: truthiness, but only literal True counts as True
            keep = np.array([x is True for x in keep], dtype=bool)

    raw = np.asarray(raw)
    if raw.ndim != 1:
        raw = raw.ravel()

    n = min(len(raw), len(keep))
    raw = raw[:n]
    keep = keep[:n]

    # Build the final codes (0/1/3)
    codes = np.zeros_like(raw, dtype=np.int8)
    codes[(raw ==  1) & keep] = 1
    codes[(raw == -1) & keep] = 3
    return codes

# 3. RAW SIGNALS (based on EMA20 vs EMA_lb)
# 3. RAW SIGNALS (based on EMA20 vs EMA_lb)
def create_raw_signals(df, lb):
    # Was: `global _last_df, _last_lb` (config-state mutation). The two
    # values are debug breadcrumbs for ML strategies that want to inspect
    # the most recent (df, lb) pair after a backtest. Stored in
    # `_runtime_state` so we don't need a `global` keyword. Still readable
    # via the legacy module attribute through the property aliases below.
    _runtime_state['_last_df'] = df
    _runtime_state['_last_lb'] = lb

    ema20  = df['EMA_20'].shift(1).values
    ema_lb = df[f'EMA_{lb}'].shift(1).values
    raw    = np.where(ema20 > ema_lb,  1,
             np.where(ema20 < ema_lb, -1, 0)).astype(np.int8)
    return raw


# 4. PARSE SIGNALS
# revert parse_signals to your exact original version

#  1) Helper: build your in_flags outside of Numba 
def compute_in_flags(times: pd.Series) -> np.ndarray:
    """
    Given a pandas Series of Timestamps, returns a bool array saying
    whether each bar is insession (uses your existing in_session()).
    """
    # listcomprehension is fastest here, since in_session() uses pandas.Timestamp
    return np.array([in_session(ts) for ts in times], dtype=np.bool_)


#  2) Numbacompiled core loop 
@njit
def _parse_signals_numba(raw: np.ndarray, in_flags: np.ndarray) -> np.ndarray:
    n = raw.shape[0]
    sig = np.zeros(n, np.int8)
    pos = 0
    in_prev = in_flags[0]

    for i in range(n):
        r = raw[i]
        if not in_flags[i]:
            in_prev = False
            continue

        if not in_prev:
            pos = r
            in_prev = True
            continue

        # flip detection
        if r == 1 and pos != 1:
            sig[i] = 1
            pos = 1
        elif r == -1 and pos != -1:
            sig[i] = 3
            pos = -1

    return sig


#  3) Public wrapper: precompute flags, then call the fast loop 
def parse_signals(raw: np.ndarray, times: pd.Series) -> np.ndarray:
    """
    Fast, Numbapowered version, but with confluencemasking of flips.
    """
    # 1) build in_flags as before
    if TRADE_SESSIONS:
        in_flags = compute_in_flags(times)
    else:
        in_flags = np.ones(raw.shape[0], dtype=np.bool_)

    # 2) get the raw flip codes
    raw_i8 = raw.astype(np.int8, copy=False)
    sig    = _parse_signals_numba(raw_i8, in_flags)

    # 3) now strip out any flips that fail your confluence
    last_df = _runtime_state['_last_df']
    last_lb = _runtime_state['_last_lb']
    if CONFLUENCES is not None and last_df is not None:
        # rebuild make_codes mask on that same df & lb
        codes = make_codes(last_df, raw, last_lb)
        # codes==1 means allow a long entry (sig==1), codes==3 means allow short
        # everywhere else we kill the flip code
        mask_allow = np.zeros_like(raw, dtype=bool)
        mask_allow[codes==1] = True   # long
        mask_allow[codes==3] = True   # short

        # *entry* flips are sig==1 or sig==3
        # zero them out where mask_allow is False
        sig[(sig == 1) & (~mask_allow)] = 0
        sig[(sig == 3) & (~mask_allow)] = 0

        # Optional symmetric exit-side filter. When MASK_EXITS is True the
        # confluence rule applies to exit codes (2 = long-close, 4 = short-close)
        # too — the bar must satisfy the confluence (codes != 0) for the exit
        # to fire. This is desirable for strategies whose exits also need
        # confirmation (e.g. only close on a confirming candle); the default
        # (False) preserves the v0.2.x behaviour where exits are unconditional
        # on signal flip.
        if MASK_EXITS:
            mask_allow_exit = (codes != 0)
            sig[(sig == 2) & (~mask_allow_exit)] = 0
            sig[(sig == 4) & (~mask_allow_exit)] = 0

    return sig


# Regime segmentation helpers.
def get_regimes(df, length=8):
    """
    Label each bar as 'Uptrend', 'Downtrend' or 'Ranging' based on EMA_200.
    Uptrend  = close > EMA_200 for the past length bars
    Downtrend= close < EMA_200 for the past length bars

    This is the default implementation used by `detect_regimes`. To use a
    different detector (volatility regimes, ML-based regimes, 2- or 5-regime
    schemes, etc.), reassign the module-level `detect_regimes` symbol and
    update `REGIME_LABELS` to match.
    """
    close = df['close'].shift(1)
    ema200 = df['EMA_200'].shift(1)

    up   = close > ema200
    dn   = close < ema200

    uptrend   = up.rolling(length).apply(lambda x: x.all(), raw=True).eq(1)
    downtrend = dn.rolling(length).apply(lambda x: x.all(), raw=True).eq(1)

    regimes = pd.Series('Ranging', index=df.index)
    regimes.loc[uptrend]   = 'Uptrend'
    regimes.loc[downtrend] = 'Downtrend'

    pct = regimes.value_counts(normalize=True) * 100
    parts = [f"{lbl} {pct.get(lbl, 0):.1f}%" for lbl in REGIME_LABELS]
    print("Regime distribution: " + ", ".join(parts))
    return regimes


def detect_regimes(df: pd.DataFrame) -> pd.Series:
    """
    Pluggable regime detector. Returns a pd.Series of labels (strings drawn
    from REGIME_LABELS) of the same length as `df`. The default implementation
    delegates to `get_regimes` (EMA-200 / 8-bar consistency, 3 regimes).

    Custom detectors must:
      * Return labels that are a subset of `REGIME_LABELS` (length 2..5)
      * Be free of look-ahead — only use information available at bar i-1 or
        earlier when labelling bar i.
    Override pattern (in your strategy file):
        import backtester as bt
        bt.REGIME_LABELS  = ['Calm', 'Volatile']
        bt.detect_regimes = my_detector
    """
    if not (2 <= len(REGIME_LABELS) <= 5):
        raise ValueError(f"REGIME_LABELS must have length 2..5, got {len(REGIME_LABELS)}")
    return get_regimes(df)

def create_regime_signals(df, best_lbs, regimes):
    """
    Create raw regime-aware EMA signals.
    The slow EMA is selected per bar from best_lbs[regime].
    The output is +1/-1 on each bar; parse_signals() applies flip logic.
    """
    ema20 = df['EMA_20'].shift(1)
    raw   = np.zeros(len(df), dtype=np.int8)

    for i, reg in enumerate(regimes):
        lb   = best_lbs[reg]
        slow = df[f'EMA_{lb}'].shift(1).iat[i]
        raw[i] = 1 if ema20.iat[i] > slow else -1

    return raw

def evaluate_filters(trades, rets, regimes=None):
    """
    Inspect the IS trades and decide which regimes and/or directions to disable.
    Prints PF, ROI and Trade count for every tested bucket, then fills the
    module-level *blocked_* structures according to:
         TradeCount > 50  and  PF < 1    block
    """
    # Was: `global blocked_regimes, ...` — but those names are mutated
    # in-place via .clear()/.add()/.setdefault(), no rebind, so the
    # `global` keyword was redundant. Removed in v0.4.0.
    blocked_regimes.clear()
    blocked_directions.clear()
    blocked_pairs.clear()

    filt_reg = globals().get("FILTER_REGIMES", False)
    filt_dir = globals().get("FILTER_DIRECTIONS", False)
    if not (filt_reg or filt_dir):
        return  # nothing to do

    # ---------- helpers ----------------------------------------------------
    def metrics(arr):
        tc   = len(arr)
        wins = sum(x for x in arr if x > 0)
        loss = -sum(x for x in arr if x <= 0)
        pf   = wins / loss if loss else float('inf')
        roi  = sum(arr)
        return tc, pf, roi

    # ---------- 1. filter-by-regime only -----------------------------------
    if filt_reg and not filt_dir:
        buckets = {}
        # Now each trade has 5 elements: (side, ent, exit, entry_price, exit_price)
        for i, (side, ent, _, *_ ) in enumerate(trades):
            reg = regimes[ent] if regimes is not None else None
            if reg is None:
                # should not happen if segmentation ON
                continue
            buckets.setdefault(reg, []).append(rets[i])

        for reg, arr in buckets.items():
            tc, pf_, roi_ = metrics(arr)
            print(f"{reg:>9}  PF:{pf_:6.2f}  ROI:{roi_*100:8.2f}%  Trades:{tc:4}")
            if tc > 50 and pf_ < 1:
                blocked_regimes.add(reg)

    # ---------- 2. filter-by-direction only --------------------------------
    elif filt_dir and not filt_reg:
        buckets = {'long': [], 'short': []}
        for i, (side, *_ ) in enumerate(trades):
            # here we only care about 'side'; the rest (entry, exit, prices) go into *_
            buckets[side].append(rets[i])

        for d in ('long', 'short'):
            tc, pf_, roi_ = metrics(buckets[d])
            print(f"{d.capitalize():>6}  PF:{pf_:6.2f}  ROI:{roi_*100:8.2f}%  Trades:{tc:4}")
            if tc > 50 and pf_ < 1:
                blocked_directions.add(d)

    # ---------- 3. BOTH filters at once  (per-regime *and* per-direction) ---
    else:
        buckets = {}
        for i, (side, ent, _, *_ ) in enumerate(trades):
            reg = regimes[ent]
            buckets.setdefault((reg, side), []).append(rets[i])

        for (reg, d), arr in buckets.items():
            tc, pf_, roi_ = metrics(arr)
            tag = f"{reg[:3]}-{d[0].upper()}"
            print(f"{tag:>8}  PF:{pf_:6.2f}  ROI:{roi_*100:8.2f}%  Trades:{tc:4}")
            if tc > 50 and pf_ < 1:
                blocked_pairs.setdefault(reg, set()).add(d)

def filter_raw_signals(raw, regimes=None):
    """
    Zeroout raw entries that fall into a blocked regime / direction,
    but preserve the original raw in _runtime_state['last_unfiltered_raw']
    so exits still fire.
    """
    # Was: `global last_unfiltered_raw, blocked_regimes, ...`. blocked_*
    # are read-only here (no rebind), and last_unfiltered_raw now lives
    # in the _runtime_state dict so we can mutate without `global`.
    _runtime_state['last_unfiltered_raw'] = raw.copy()

    if not (FILTER_REGIMES or FILTER_DIRECTIONS):
        return raw

    out = raw.copy()
    for i, v in enumerate(raw):
        if v == 0:
            continue
        side = 'long' if v == 1 else 'short'
        reg  = regimes[i] if regimes is not None else None

        # direction-wide blocks
        if side in blocked_directions:
            out[i] = 0
            continue
        # regime-wide blocks
        if reg and reg in blocked_regimes:
            out[i] = 0
            continue
        # perregime & direction blocks
        if reg and reg in blocked_pairs and side in blocked_pairs[reg]:
            out[i] = 0

    return out


# 5. BACKTEST CORE (modified to include Consistency metric)
# 5. BACKTEST CORE (modified to include Consistency metric + carry_in/out)
# 5. BACKTEST CORE (fixed tuple unpacking)
# 5. BACKTEST CORE (fixed drawdown/ROI clipping)

def _prepare_backtest_inputs(df, sig):
    """
    Converts the DataFrame (pandas) into plain NumPy objects that Numba can use.
    Returns a dict with everything the JIT kernel needs.
    """
    # price arrays
    o = df["open"].to_numpy(dtype=np.float64)
    h = df["high"].to_numpy(dtype=np.float64)
    l = df["low"].to_numpy(dtype=np.float64)
    c = df["close"].to_numpy(dtype=np.float64)

    # --- timestamps: precompute everything we will need inside the kernel ---
    times = df["time"]
    n     = len(times)

    # 1 funding (crypto only) -------------------------------------------------
    if not FOREX_MODE:
        ts_utc        = times.dt.tz_convert("UTC")
        funding_mask  = ((ts_utc.dt.minute == 0) &
                         (ts_utc.dt.hour.isin([0, 8, 16]))).to_numpy(np.bool_)
    else:
        funding_mask = np.zeros(n, dtype=np.bool_)

    # 2 session windows -------------------------------------------------------
    if TRADE_SESSIONS:
        t_start = datetime.strptime(SESSION_START, "%H:%M").time()
        t_end   = datetime.strptime(SESSION_END,   "%H:%M").time()

        session_mask     = ((times.dt.time >= t_start) &
                            (times.dt.time <= t_end)).to_numpy(np.bool_)
        session_idxs     = np.flatnonzero(session_mask).astype(np.int64)
        # v0.2.3 fix: previously this required the bar's local time to
        # *exactly* equal SESSION_END (e.g. 16:50). Bars at any other
        # minute (e.g. crypto bars at HH:00) silently never triggered the
        # force-close, so positions could carry across out-of-session
        # windows. Mark the LAST in-session bar of each contiguous
        # in-session run instead.
        next_in = np.concatenate([session_mask[1:], [False]])
        session_end_mask = session_mask & ~next_in
    else:
        session_idxs     = np.arange(n, dtype=np.int64)
        session_end_mask = np.zeros(n, dtype=np.bool_)

    # 3 signals ---------------------------------------------------------------
    sig_arr = sig.astype(np.int8)   # ensure small dtype

    return dict(
        o=o, h=h, l=l, c=c,
        sig=sig_arr,
        session_idxs=session_idxs,
        session_end=session_end_mask,
        funding_mask=funding_mask,
        n=n
    )

@njit(cache=True)
def _backtest_numba_core(o, h, l, c, sig,
                         session_idxs, session_end, funding_mask,
                         # config flags -------------------------------------------------
                         use_forex, use_sessions, use_regime,
                         use_sl, use_tp,
                         # numeric constants --------------------------------------------
                         sl_perc, tp_perc, pip_size,
                         fee_rate, slip, funding_rate,
                         position_size, account_size,
                         # carryin -----------------------------------------------------
                         carry_side, carry_ent_idx, carry_entry_price):
    """
    Fullycompiled backtest.  Returns:
       trades_list, metrics_dict, eq_frac, rets
    """
    # ---------- helper typed lists for variablelength results --------------
    trades_side  = List.empty_list(types.int8)        #  1 = long, -1 = short
    trades_ent   = List.empty_list(types.int32)       #  entry index
    trades_exit  = List.empty_list(types.int32)       #  exit index
    trades_ep    = List.empty_list(types.float64)     #  entry price
    trades_xp    = List.empty_list(types.float64)     #  exit price
    trades_qty   = List.empty_list(types.float64)     #  quantity
    trades_pnl   = List.empty_list(types.float64)     #  pnl

    equity_list  = List()
    funding_acc  = 0.0
    open_pos     = 0        # 0 = flat,  1 = long,  -1 = short
    ent_bar      = -1
    entry_price  = 0.0
    qty          = 0.0
    fee_entry    = 0.0

    if use_forex:
        position_size_fx = position_size   # = 1.0 (see wrapper)
    else:
        equity_list.append(account_size)

    # -------- restore carryin ----------------------------------------------
    if carry_side != 0:
        open_pos    = 1 if carry_side == 1 else -1
        ent_bar     = carry_ent_idx
        entry_price = carry_entry_price
        qty         = position_size / entry_price
        fee_entry   = position_size * fee_rate
        if not use_forex:
            # no equity update yet (will happen on exit)
            pass

    # ----------------------------- main loop --------------------------------
    for idx in session_idxs:
        # 1 funding every 8 h (crypto)
        if open_pos != 0 and funding_mask[idx]:
            fee_f = qty * o[idx] * funding_rate
            funding_acc += fee_f
            if not use_forex:
                equity_list[-1] = equity_list[-1] - fee_f

        code   = sig[idx]

        if use_regime and idx < 200:
            continue

        # forced exit for session end. v0.2.3 fix: drop the prior
        # `and code != 0` guard — the force-close should fire whenever an
        # open position exists at a session-end bar, regardless of whether
        # the strategy happens to emit a signal on that same bar.
        # Prior behaviour silently carried positions across out-of-session
        # windows; no published research used TRADE_SESSIONS so this is a
        # zero-cost correctness fix.
        end_bar_flag = session_end[idx] if use_sessions else False
        if open_pos != 0:
            if use_sessions and end_bar_flag:
                code = 2 if open_pos == 1 else 4
        if use_sessions and (code in (1, 3)) and end_bar_flag:
            code = 0                                    # block new entry

        price_open = o[idx]

        # ---------- intrabar SL / TP check ----------------------------------
        if open_pos != 0 and code not in (1, 3) and not end_bar_flag:
            if use_forex:
                sl_pr = entry_price - sl_perc if open_pos==1 else entry_price + sl_perc
                tp_pr = entry_price + tp_perc if open_pos==1 else entry_price - tp_perc
            else:
                sl_pr = entry_price * (1 - sl_perc/100) if open_pos==1 else entry_price * (1 + sl_perc/100)
                tp_pr = entry_price * (1 + tp_perc/100) if open_pos==1 else entry_price * (1 - tp_perc/100)

            hit_sl = (l[idx] <= sl_pr) if open_pos==1 else (h[idx] >= sl_pr)
            hit_tp = (h[idx] >= tp_pr) if open_pos==1 else (l[idx] <= tp_pr)
            if hit_sl and hit_tp:          # mutually exclusive
                hit_tp = False

            is_sl_hit = None
            if use_sl and hit_sl:
                is_sl_hit = True
            elif use_tp and hit_tp:
                is_sl_hit = False

            if is_sl_hit is not None:
                raw_exit   = sl_pr if is_sl_hit else tp_pr
                exit_price = raw_exit * (1 - slip) if open_pos==1 else raw_exit * (1 + slip)
                fee_exit   = qty * exit_price * fee_rate

                # ----- PNL --------------------------------------------------
                if use_forex:
                    if is_sl_hit:
                        pnl = -position_size_fx - (fee_entry + fee_exit)
                    else:
                        RRR = tp_perc / sl_perc
                        pnl = position_size_fx * RRR - (fee_entry + fee_exit)
                else:
                    pnl = qty * ((exit_price - entry_price) if open_pos==1 else (entry_price - exit_price)) \
                          - (fee_entry + fee_exit + funding_acc)
                    funding_acc = 0.0

                # save trade
                trades_side.append(open_pos)
                trades_ent.append(ent_bar)
                trades_exit.append(idx)
                trades_ep.append(entry_price)
                trades_xp.append(exit_price)
                trades_qty.append(qty)
                trades_pnl.append(pnl)

                if use_forex:
                    # equity as fraction of risk
                    pass
                else:
                    equity_list.append(equity_list[-1] + pnl)

                # flat
                open_pos = 0
                continue

        # ------------- signaldriven entry & exit at bar OPEN ---------------
        if code == 1:                       # go long
            # 1a) close short
            if open_pos == -1:
                exit_price = price_open * (1 + slip)
                fee_exit   = qty * exit_price * fee_rate
                if use_forex:
                    price_move      = entry_price - exit_price
                    price_move_pips = price_move / pip_size
                    stop_pips       = sl_perc / pip_size
                    RRR             = tp_perc / sl_perc
                    trade_res       = (price_move_pips / (RRR*stop_pips)) * RRR
                    trade_res       = max(min(trade_res, RRR), -1)
                    pnl = trade_res * position_size_fx - (fee_entry + fee_exit)
                else:
                    pnl = qty * (entry_price - exit_price) - (fee_entry + fee_exit + funding_acc)
                    funding_acc = 0.0

                # store
                trades_side.append(-1)
                trades_ent.append(ent_bar)
                trades_exit.append(idx)
                trades_ep.append(entry_price)
                trades_xp.append(exit_price)
                trades_qty.append(qty)
                trades_pnl.append(pnl)
                if not use_forex:
                    equity_list.append(equity_list[-1] + pnl)
                open_pos = 0            # flat

            # 1b) open long
            if open_pos == 0:
                fee_entry = position_size * fee_rate
                entry_price = price_open * (1 + slip)
                qty         = position_size / entry_price
                open_pos    = 1
                ent_bar     = idx

        elif code == 3:                   # go short (mirror)
            if open_pos == 1:
                exit_price = price_open * (1 - slip)
                fee_exit   = qty * exit_price * fee_rate
                if use_forex:
                    price_move      = exit_price - entry_price
                    price_move_pips = price_move / pip_size
                    stop_pips       = sl_perc / pip_size
                    RRR             = tp_perc / sl_perc
                    trade_res       = (price_move_pips / (RRR*stop_pips)) * RRR
                    trade_res       = max(min(trade_res, RRR), -1)
                    pnl = trade_res * position_size_fx - (fee_entry + fee_exit)
                else:
                    pnl = qty * (exit_price - entry_price) - (fee_entry + fee_exit + funding_acc)
                    funding_acc = 0.0

                trades_side.append(1)
                trades_ent.append(ent_bar)
                trades_exit.append(idx)
                trades_ep.append(entry_price)
                trades_xp.append(exit_price)
                trades_qty.append(qty)
                trades_pnl.append(pnl)
                if not use_forex:
                    equity_list.append(equity_list[-1] + pnl)
                open_pos = 0

            if open_pos == 0:
                fee_entry = position_size * fee_rate
                entry_price = price_open * (1 - slip)
                qty         = position_size / entry_price
                open_pos    = -1
                ent_bar     = idx

        elif code == 2 and open_pos == 1:  # long  close
            exit_price = price_open * (1 - slip)
            fee_exit   = qty * exit_price * fee_rate
            if use_forex:
                price_move      = exit_price - entry_price
                price_move_pips = price_move / pip_size
                stop_pips       = sl_perc / pip_size
                RRR             = tp_perc / sl_perc
                trade_res       = (price_move_pips / (RRR*stop_pips)) * RRR
                trade_res       = max(min(trade_res, RRR), -1)
                pnl = trade_res * position_size_fx - (fee_entry + fee_exit)
            else:
                pnl = qty * (exit_price - entry_price) - (fee_entry + fee_exit + funding_acc)
                funding_acc = 0.0

            trades_side.append(1)
            trades_ent.append(ent_bar)
            trades_exit.append(idx)
            trades_ep.append(entry_price)
            trades_xp.append(exit_price)
            trades_qty.append(qty)
            trades_pnl.append(pnl)
            if not use_forex:
                equity_list.append(equity_list[-1] + pnl)
            open_pos = 0

        elif code == 4 and open_pos == -1: # short  close
            exit_price = price_open * (1 + slip)
            fee_exit   = qty * exit_price * fee_rate
            if use_forex:
                price_move      = entry_price - exit_price
                price_move_pips = price_move / pip_size
                stop_pips       = sl_perc / pip_size
                RRR             = tp_perc / sl_perc
                trade_res       = (price_move_pips / (RRR*stop_pips)) * RRR
                trade_res       = max(min(trade_res, RRR), -1)
                pnl = trade_res * position_size_fx - (fee_entry + fee_exit)
            else:
                pnl = qty * (entry_price - exit_price) - (fee_entry + fee_exit + funding_acc)
                funding_acc = 0.0

            trades_side.append(-1)
            trades_ent.append(ent_bar)
            trades_exit.append(idx)
            trades_ep.append(entry_price)
            trades_xp.append(exit_price)
            trades_qty.append(qty)
            trades_pnl.append(pnl)
            if not use_forex:
                equity_list.append(equity_list[-1] + pnl)
            open_pos = 0

    # ------------ forceclose any open trade on the last bar ----------------
    if open_pos != 0:
        price_last = o[-1]
        exit_price = price_last * (1 - slip) if open_pos==1 else price_last * (1 + slip)
        fee_exit   = qty * exit_price * fee_rate
        if use_forex:
            price_move      = (exit_price - entry_price) if open_pos==1 else (entry_price - exit_price)
            price_move_pips = price_move / pip_size
            stop_pips       = sl_perc / pip_size
            RRR             = tp_perc / sl_perc
            trade_res       = (price_move_pips / (RRR*stop_pips)) * RRR
            trade_res       = max(min(trade_res, RRR), -1)
            pnl = trade_res * position_size_fx - (fee_entry + fee_exit)
        else:
            pnl = qty * ((exit_price - entry_price) if open_pos==1 else (entry_price - exit_price)) \
                  - (fee_entry + fee_exit + funding_acc)
        trades_side.append(open_pos)
        trades_ent.append(ent_bar)
        trades_exit.append(len(o)-1)
        trades_ep.append(entry_price)
        trades_xp.append(exit_price)
        trades_qty.append(qty)
        trades_pnl.append(pnl)
        if not use_forex:
            equity_list.append(equity_list[-1] + pnl)

    # ------------- convert typed lists  numpy ------------------------------
    side   = np.asarray(trades_side,  dtype=np.int8)
    ent    = np.asarray(trades_ent,   dtype=np.int32)
    exi    = np.asarray(trades_exit,  dtype=np.int32)
    ep     = np.asarray(trades_ep,    dtype=np.float64)
    xp     = np.asarray(trades_xp,    dtype=np.float64)
    qtys   = np.asarray(trades_qty,   dtype=np.float64)
    pnl    = np.asarray(trades_pnl,   dtype=np.float64)

    # equity curve & returns --------------------------------------------------
    if use_forex:
        rets    = pnl / position_size_fx
        eq_frac = np.concatenate((np.array([0.0]), np.cumsum(rets)))
    else:
        eq_usd  = np.asarray(equity_list, dtype=np.float64)
        eq_frac = eq_usd / account_size
        rets    = pnl / account_size

    tc = rets.size
    wr = np.mean(rets > 0) if tc else 0.0
    roi = eq_frac[-1] if use_forex else (eq_frac[-1] - 1.0)

    wins   = rets[rets > 0]
    losses = -rets[rets <= 0]
    pf   = wins.sum()/losses.sum() if losses.size else np.inf
    expc = (wins.mean() if wins.size else 0)*wr - (losses.mean() if losses.size else 0)*(1-wr)
    shp  = (rets.mean()/rets.std()*np.sqrt(tc)) if tc>1 and rets.std() else 0.0
    hw   = _cummax(eq_frac) 
    dd   = (np.max(hw - eq_frac) if use_forex else np.max((hw - eq_frac)/hw)) if tc else 0.0
    segment_sums = _five_segment_sums(rets)
    w    = np.array([0.0117,0.0317,0.0861,0.2341,0.6364])
    consistency  = 0.6 * np.dot(w, segment_sums) + 0.4 * roi

    metrics_tuple = (tc, roi, pf, wr, expc, shp, dd, consistency)


    # pack trades as list of tuples for identical API
    trades = list(zip(side, ent, exi, ep, xp, qtys, pnl))

    return trades, metrics_tuple, eq_frac, rets, None


# User-facing wrapper (keeps backward-compatible signature).
def backtest(df, raw_sig, carry_in=None):
    # Build normalized signal inputs and route them to the Numba core.
    prep = _prepare_backtest_inputs(df, raw_sig)

    if carry_in:
        carry_side, carry_ent, carry_price = carry_in
        carry_side_num = 1 if carry_side == "long" else -1
    else:
        carry_side_num = 0
        carry_ent      = -1
        carry_price    = 0.0

    trades, metrics_tup, eq_frac, rets, _ = _backtest_numba_core(
        prep["o"], prep["h"], prep["l"], prep["c"], prep["sig"],
        prep["session_idxs"], prep["session_end"],
        prep["funding_mask"],
        FOREX_MODE, TRADE_SESSIONS, USE_REGIME_SEG,
        USE_SL, USE_TP,
        SL_PERCENTAGE, TP_PERCENTAGE, PIP_SIZE,
        FEE_PCT/100, SLIPPAGE_PCT*(PIP_SIZE if FOREX_MODE else 0.01),
        0.0 if FOREX_MODE else FUNDING_FEE/100,
        1.0 if FOREX_MODE else RISK_AMOUNT,
        ACCOUNT_SIZE,
        carry_side_num, carry_ent, carry_price
    )

    tc, roi, pf, wr, expc, shp, dd, cons = metrics_tup
    metrics = dict(
        Trades=int(tc), ROI=float(roi), PF=float(pf), WinRate=float(wr),
        Exp=float(expc), Sharpe=float(shp), MaxDrawdown=float(dd),
        Consistency=float(cons)
    )
    # Standard (bar-based) Sharpe override. Default SHARPE_MODE="trade" leaves
    # the per-trade statistic untouched (bit-identical to prior releases).
    if SHARPE_MODE == "bar":
        metrics["Sharpe"] = _bar_based_sharpe(df, trades, FOREX_MODE, ACCOUNT_SIZE)

    return trades, metrics, eq_frac, rets, None


# Module-level run-flow state. These stay in __init__ as module globals (the
# orchestrator submodule reads/mutates them live via `_bt.signals_cache` /
# `_bt.AGE_DATASET`). signals_cache is a per-process scratch cache populated by
# classic_single_run / main; AGE_DATASET trims the most-recent N candles.
signals_cache = {}
AGE_DATASET  = 0


# ============================================================================
# Submodule re-exports (v0.6.0 split).
#
# The engine was split into metrics.py / objectives.py / orchestrator.py to
# mirror the Rust port (src/metrics.rs, objectives.rs, orchestrator.rs). These
# imports live at the END of the module, AFTER every global/constant and every
# function that STAYS here is defined, so the submodules can `import backtester
# as _bt` and read the live module surface at call time without circular-import
# or stale-value hazards. The re-imports below rebind every moved name back into
# this namespace so `import backtester as bt; bt.<name>` resolves exactly as it
# did before the split. (The numba core `_backtest_numba_core` references the
# moved njit helpers `_cummax` / `_five_segment_sums` as module globals; those
# names are bound here before any backtest() call triggers JIT compilation.)
# ============================================================================
from .metrics import (
    _metrics_from_trades,
    _cummax,
    _five_segment_sums,
    _bar_based_sharpe,
    prettyprint,
)
from .objectives import (
    optimiser,
    _optimiser_impl,
    monte_carlo,
    _monte_carlo_impl,
)
from .orchestrator import (
    export_trades,
    optimise_regime_full,
    backtest_continuous_regime,
    optimize_regimes_sequential,
    _optimize_regimes_sequential_impl,
    classic_single_run,
    _classic_single_run_impl,
    _run_wfo_window,
    walk_forward,
    _walk_forward_impl,
    inject_news_candles,
    apply_news_injection,
    _apply_news_injection_impl,
    drift_entries,
    apply_combined_robustness,
    _normalize_rb_flag,
    _opts_from_flags,
    _label_from_flags,
    run_robustness_tests,
    age_dataset,
    main,
    _main_impl,
)


if __name__ == '__main__':
    main()

