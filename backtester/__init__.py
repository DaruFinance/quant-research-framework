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

def _metrics_from_trades(trades):
    # 1) per-trade returns in R-units vs USD
    if FOREX_MODE:
        rets = np.array([pnl / POSITION_SIZE for *_, pnl in trades], dtype=float)
    else:
        rets = np.array([pnl / ACCOUNT_SIZE for *_, pnl in trades], dtype=float)
    tc   = len(rets)
    wr   = np.mean(rets > 0) if tc else 0.0
    roi  = rets.sum() if tc else 0.0   # sum of R or sum of fracreturns
    wins = rets[rets > 0]
    losses = -rets[rets <= 0]
    pf   = wins.sum() / losses.sum() if losses.size else float('inf')
    expc = (wins.mean() if wins.size else 0) * wr \
         - (losses.mean() if losses.size else 0) * (1 - wr)
    shp  = (rets.mean() / rets.std() * sqrt(tc)) if tc > 1 and rets.std() else 0.0

    # Max drawdown
    if FOREX_MODE:
        eq = np.concatenate(([0.0], np.cumsum(rets)))
        hw = np.maximum.accumulate(eq)
        dd = np.max(hw - eq) if tc else 0.0
    else:
        eq = 1.0 + np.cumsum(rets)
        hw = np.maximum.accumulate(eq)
        dd = np.max((hw - eq) / hw) if tc else 0.0

    # Consistency
    segs = np.array_split(rets, 5)
    w    = np.array([0.0117,0.0317,0.0861,0.2341,0.6364])
    consistency = 0.6 * np.dot(w, [s.sum() for s in segs]) + 0.4 * roi

    return {
        'Trades':      tc,
        'ROI':         roi,
        'PF':          pf,
        'WinRate':     wr,
        'Exp':         expc,
        'Sharpe':      shp,
        'MaxDrawdown': dd,
        'Consistency': consistency
    }


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
def _cummax(arr):
    """Numbafriendly replacement for np.maximum.accumulate(arr)."""
    out = np.empty_like(arr)
    m   = -1e100          # works for any realistic equity fraction
    for i in range(arr.size):
        v = arr[i]
        if v > m:
            m = v
        out[i] = m
    return out

@njit(cache=True)
def _five_segment_sums(vec):
    """
    Split 1D array into five equalsized (1) segments and
    return their sums (length5 np.ndarray).
    """
    n   = vec.size
    seg = np.empty(5, dtype=np.float64)
    start = 0
    for k in range(5):
        end   = start + (n + k) // 5     # distributes the remainder nicely
        seg[k] = vec[start:end].sum()
        start = end
    return seg


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


def _bar_based_sharpe(df, trades, use_forex, account_size):
    """Standard calendar-time Sharpe on the per-bar mark-to-market equity curve.

    Reconstructs equity at every bar as account + realised PnL of closed trades
    + unrealised mark-to-market of any open position at that bar's close, then
    annualises mean/std of per-bar returns by the bar frequency (periods per
    Julian year from the median bar spacing). Crypto path compounds (pct-change
    returns, matching vectorbt/empyrical); forex path is in R-units and uses
    additive increments. The Rust port reconstructs this identically (parity).
    """
    n = len(df)
    if n < 3 or not trades:
        return 0.0
    close = df["close"].to_numpy(dtype=float)
    realized = np.zeros(n)
    unreal = np.zeros(n)
    if use_forex:
        sl_dist = SL_PERCENTAGE * PIP_SIZE
        rrr = (TP_PERCENTAGE / SL_PERCENTAGE) if SL_PERCENTAGE else 1.0
    for side, ent, exi, ep, xp, qty, pnl in trades:
        e, x = int(ent), int(exi)
        if x < e:
            continue
        if x < n:
            realized[x:] += pnl
        d = 1.0 if side == 1 else -1.0
        for i in range(e, min(x, n)):
            if use_forex:
                r = (d * (close[i] - ep) / sl_dist) if sl_dist else 0.0
                unreal[i] = min(max(r, -1.0), rrr) * POSITION_SIZE
            else:
                unreal[i] = d * qty * (close[i] - ep)
    if use_forex:
        bar_eq = realized + unreal               # R-units around 0
        ret = np.diff(bar_eq)                     # additive R increments
    else:
        bar_eq = account_size + realized + unreal
        ret = np.diff(bar_eq) / bar_eq[:-1]       # compounding (pct-change)
    ret = ret[np.isfinite(ret)]
    if ret.size < 2 or ret.std() <= 0:
        return 0.0
    # periods per Julian year from the median bar spacing (seconds). The bar
    # timestamps may live in the index or a `time` column, as (tz-aware)
    # datetime or unix; `.asi8` gives UTC nanoseconds in all cases.
    ts = None
    if isinstance(df.index, pd.DatetimeIndex):
        ts = df.index
    elif "time" in df.columns:
        col = df["time"]
        ts = pd.DatetimeIndex(col) if pd.api.types.is_datetime64_any_dtype(col) \
             else pd.to_datetime(col.to_numpy(), unit="s", utc=True)
    sec = float(np.median(np.diff(np.asarray(ts.asi8))) / 1e9) \
          if ts is not None and len(ts) > 1 else 0.0
    ppy = (31_557_600.0 / sec) if sec > 0 else 1.0
    return float(ret.mean() / ret.std() * np.sqrt(ppy))


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


# 6. OPTIMISER  now with Smart Optimization support
# 6. OPTIMISER  now with Smart Optimization support (patched for WFO)
def optimiser(df, lb_range, metric, min_trades, config: Optional[Config] = None):
    """
    Updated optimiser:
      1) Coarse search in steps of 2 over lb_range.
      2) Take the best coarse lookback.
      3) Fine-tune by testing best_lb - 1 and best_lb + 1.
      4) Return the overall best of those three.

    `config` is optional. When provided, the engine uses cfg's values for
    the duration of this call (and restores prior values on exit). When
    omitted, the call reads from module globals as it always has —
    the documented `bt.X = Y` / `monkeypatch.setattr(bt, "X", Y)` API
    keeps working unchanged.
    """
    with with_config(config):
        return _optimiser_impl(df, lb_range, metric, min_trades)


def _optimiser_impl(df, lb_range, metric, min_trades):
    # Was: `global last_unfiltered_raw`. Moved to _runtime_state dict
    # so we can mutate without `global`. See _runtime_state docstring.
    eval_cache = {}

    # Helper to evaluate a single lookback
    def _evaluate(lb):
        if lb in eval_cache:
            return eval_cache[lb]

        # 1) Compute indicators & raw signals for this lookback
        dfi = compute_indicators(df, lb)
        raw = create_raw_signals(dfi, lb)
        _runtime_state['last_unfiltered_raw'] = raw.copy()
        sig = parse_signals(raw, dfi['time'])

        # 2) Backtest (with or without RRR optimisation)
        if not OPTIMIZE_RRR:
            _, met, _, _, _ = backtest(dfi, sig)
        else:
            # --- quick probe at fixed 5 R ---
            old_tp, old_tp_flag = TP_PERCENTAGE, USE_TP
            globals()['TP_PERCENTAGE'] = 5 * SL_PERCENTAGE
            globals()['USE_TP']        = True

            _runtime_state['last_unfiltered_raw'] = raw.copy()
            trades_probe, _, _, _, _ = backtest(dfi, sig)

            # compute peak and close R multiples
            peak_Rs, close_Rs = [], []
            for side, e, x, *_ in trades_probe:
                entry_price = dfi['close'].iloc[e]
                risk = entry_price * SL_PERCENTAGE / 100.0

                # NOTE: pre-v0.2.5 used `side == 'long'`, which compared int8
                # against a str and always took the else branch. Default now is
                # the corrected `side == 1` test; LEGACY_SIDE_BUG=True reverts.
                is_long = (side == 'long') if LEGACY_SIDE_BUG else (side == 1)
                if is_long:
                    high_slice = dfi['high'].iloc[e:x+1].values
                    peak_R = (high_slice.max() - entry_price) / risk
                    close_R = (dfi['close'].iloc[x] - entry_price) / risk
                else:  # short
                    low_slice = dfi['low'].iloc[e:x+1].values
                    peak_R = (entry_price - low_slice.min()) / risk
                    close_R = (entry_price - dfi['close'].iloc[x]) / risk

                peak_Rs.append(min(peak_R, 3.0))
                close_Rs.append(close_R)

            peak_Rs = np.array(peak_Rs, dtype=float)
            close_Rs = np.array(close_Rs, dtype=float)

            # sum R for each candidate R_target
            sum_R_for_candidates = {
                R_target: np.where(peak_Rs >= R_target, R_target, close_Rs).sum()
                for R_target in range(1, 4)
            }

            # pick best RRR
            best_rrr = max(sum_R_for_candidates, key=sum_R_for_candidates.get)

            # re-run with optimal TP
            globals()['TP_PERCENTAGE'] = best_rrr * SL_PERCENTAGE
            _runtime_state['last_unfiltered_raw'] = raw.copy()
            _, met, _, _, _ = backtest(dfi, sig)
            met['RRR'] = best_rrr

            # restore TP settings
            globals()['TP_PERCENTAGE'] = old_tp
            globals()['USE_TP']        = old_tp_flag

        # 3) filter by minimum trades
        if met['Trades'] < min_trades:
            eval_cache[lb] = None
            return None

        # 4) Drawdown constraint
        if dd_constraint is not None and met['MaxDrawdown'] > dd_constraint:
            eval_cache[lb] = None
            return None

        # 5) Compute value to optimize (flip for MaxDrawdown)
        val = -met[metric] if metric == 'MaxDrawdown' else met[metric]
        eval_cache[lb] = (val, lb, met)
        return eval_cache[lb]

    # Build list of all lookbacks and coarse subset
    all_lbs = list(lb_range)
    coarse_lbs = all_lbs[::2]

    # 1) Coarse pass
    coarse_results = [res for lb in coarse_lbs if (res := _evaluate(lb)) is not None]


    if not coarse_results:
        print(f"No lookback meets drawdown  {DRAWDOWN_CONSTRAINT}, using raw LB {DEFAULT_LB}")
        dfi = compute_indicators(df, DEFAULT_LB)
        raw = create_raw_signals(dfi, DEFAULT_LB)
        sig = parse_signals(raw, dfi['time'])
        _, met_raw, *_ = backtest(dfi, sig)
        return DEFAULT_LB, met_raw


    # Pick best coarse, then fine-tune
    coarse_results.sort(key=lambda x: x[0], reverse=True)
    best_val, best_lb, best_met = coarse_results[0]

    idx = all_lbs.index(best_lb)
    candidates = [(best_val, best_lb, best_met)]
    for ni in (idx - 1, idx + 1):
        if 0 <= ni < len(all_lbs):
            if (res := _evaluate(all_lbs[ni])) is not None:
                candidates.append(res)

    candidates.sort(key=lambda x: x[0], reverse=True)

    # --- Smart optimization guard against spiky PF -------------------------
    selected = candidates[0]
    if SMART_OPTIMIZATION:
        best_candidate = candidates[0]
        best_lb = best_candidate[1]
        all_lb_set = set(all_lbs)

        for cand in candidates:
            _, lb_cand, met_cand = cand
            pf_cand = met_cand['PF']

            ok = True
            for neigh in (lb_cand - 1, lb_cand + 1):
                if neigh in all_lb_set:
                    neigh_res = _evaluate(neigh)
                    if neigh_res is None:
                        continue
                    pf_neigh = neigh_res[2]['PF']
                    if pf_cand > 1.10 * pf_neigh:
                        ok = False
                        break
            if ok:
                selected = cand
                if lb_cand != best_lb:
                    print(f"Smart Optimization: switched from LB {best_lb} to LB {lb_cand} because PF spike exceeded 10% vs neighbors.")
                break

    # --- item #3 opt-in side-channel: capture the distinct trial Sharpes ---
    # eval_cache maps each EVALUATED lookback -> (val, lb, met). The set of
    # distinct lookbacks IS the "strategies tried" set for this single IS
    # optimisation — window-count-free (effective-trials discipline). Pure
    # write to the scratch dict; no return value, printed line, or
    # control-flow change. Gated so default runs are inert.
    if OVERFIT_REPORT:
        _runtime_state['_last_trial_sharpes'] = [
            float(v[2]['Sharpe']) for v in eval_cache.values()
            if v is not None and 'Sharpe' in v[2]
        ]

    return selected[1], selected[2]

# 7. MONTE CARLO
def monte_carlo(arr, actual, runs, config: Optional[Config] = None):
    """Monte Carlo bootstrap & shuffle of the realised return series.

    `config` is optional; see `optimiser` docstring for the contract.
    The MC routine itself is config-free (no engine knobs participate),
    but the parameter is accepted for API symmetry with the rest of
    the public surface.
    """
    with with_config(config):
        return _monte_carlo_impl(arr, actual, runs)


def _monte_carlo_impl(arr, actual, runs):
    N = arr.size
    if N == 0:
        print(" Monte Carlo skipped: no return series provided.")
        return

    # bootstrap and shuffle
    sims_boot  = np.random.choice(arr, size=(runs, N), replace=True)
    sims_shuff = np.array([np.random.permutation(arr) for _ in range(runs)])
    sims_all   = np.concatenate((sims_boot, sims_shuff), axis=0)

    # --- metric calculations ---
    roi    = sims_all.sum(axis=1)
    wins   = np.where(sims_all > 0, sims_all, 0)
    losses = np.where(sims_all <= 0, -sims_all, 0)

    wins_sum   = wins.sum(axis=1)
    losses_sum = losses.sum(axis=1)

    # Profit factor: avoid inf by using NaNlarge finite
    pf = np.divide(
        wins_sum, 
        losses_sum,
        out=np.full_like(wins_sum, np.nan),
        where=losses_sum > 0
    )
    pf = np.where(np.isnan(pf), 1e9, pf)

    wr  = np.mean(sims_all > 0, axis=1)
    mw  = np.where(
        wins_sum > 0,
        wins_sum / np.maximum(np.count_nonzero(wins, axis=1), 1),
        0
    )
    ml  = np.where(
        losses_sum > 0,
        losses_sum / np.maximum(np.count_nonzero(losses, axis=1), 1),
        0
    )
    exp = mw * wr - ml * (1 - wr)
    std = sims_all.std(axis=1)
    shp = np.where(std > 0, sims_all.mean(axis=1) / std * sqrt(N), 0)

    # consistency
    weights = np.array([0.0117, 0.0317, 0.0861, 0.2341, 0.6364])
    cons = np.empty(sims_all.shape[0])
    for i, sim in enumerate(sims_all):
        segs = np.array_split(sim, 5)
        rois = [s.sum() for s in segs]
        cons[i] = 0.6 * np.dot(weights, rois) + 0.4 * sim.sum()

    # equity curves & drawdown
    eqs = 1.0 + np.cumsum(sims_all, axis=1)
    hw  = np.maximum.accumulate(eqs, axis=1)
    dd  = ((hw - eqs) / hw).max(axis=1)

    dist = {
        'ROI':         roi,
        'PF':          pf,
        'WinRate':     wr,
        'Exp':         exp,
        'Sharpe':      shp,
        'MaxDrawdown': dd,
        'Consistency': cons
    }

    # prints
    print("\n Monte-Carlo Percentile Ranks vs ACTUAL ")
    for m in METRICS + ['Consistency']:
        pct = np.mean(dist[m] <= actual[m]) * 100
        print(f"  {m:>12}: {pct:6.1f}th percentile")

    percs  = [5, 25, 50, 75, 95]
    eq_pct = np.percentile(eqs, percs, axis=0)

    print("\n Equity Curve Final Value Percentiles ")
    for p, val in zip(percs, eq_pct[:, -1]):
        print(f"  {p:>2}th pct: {val:9.4f}")
    loss_pct = np.mean(roi < 0) * 100
    dd80_pct = np.mean(dd > 0.80) * 100
    print(f"\nSimulations ending with LOSS:           {loss_pct:5.1f}%")
    print(f"Simulations max-DD > 80 %:              {dd80_pct:5.1f}%\n")

    if PRINT_EQUITY_CURVE:
        x   = np.arange(1, N + 1)
        fig = plt.figure(figsize=(10, 5))
        ax  = fig.add_subplot(1, 1, 1)

        # actual equity
        ax.plot(ACCOUNT_SIZE * (1.0 + np.cumsum(arr)),
                color='black', label='Actual equity ($)', linewidth=1.6)

        # simulated bands
        ax.fill_between(x,
                        ACCOUNT_SIZE * eq_pct[0],
                        ACCOUNT_SIZE * eq_pct[-1],
                        alpha=0.18, label='5-95% band')
        ax.fill_between(x,
                        ACCOUNT_SIZE * eq_pct[1],
                        ACCOUNT_SIZE * eq_pct[-2],
                        alpha=0.28, label='25-75% band')
        ax.plot(x,
                ACCOUNT_SIZE * eq_pct[2],
                color='royalblue',
                label='Median (50%)',
                linewidth=1.2)

        ax.set_title('Monte-Carlo Equity Curve Bands (USD)')
        ax.legend(); ax.grid(True)
        fig.tight_layout()

        # histograms
        hist_keys = [
          ('ROI',         'ROI'),
          ('MaxDrawdown','Max DD'),
          ('PF',          'Profit Factor'),
          ('Sharpe',      'Sharpe'),
          ('Consistency','Consistency')
        ]
        fig2 = plt.figure(figsize=(14, 8))
        gs   = GridSpec(2, 3, fig2)
        for idx, (k, title) in enumerate(hist_keys):
            r, c = divmod(idx, 3)
            axh  = fig2.add_subplot(gs[r, c])

            data = dist[k]
            # drop any non-finite just in case
            data = data[np.isfinite(data)]
            if data.size == 0:
                axh.text(0.5, 0.5, 'No data', ha='center', va='center')
            else:
                # clip PF for readability
                if k == 'PF':
                    data = np.clip(data, 0, 50)
                axh.hist(data, bins=100)
                axh.axvline(min(actual[k], 50 if k=='PF' else actual[k]),
                            color='red', linestyle='--', linewidth=1.5)

            axh.set_title(title)
            axh.grid(True)

        fig2.add_subplot(gs[1, 2]).axis('off')
        fig2.suptitle('Simulation Metric Distributions')
        fig2.tight_layout(rect=[0, 0, 1, 0.96])
# 8. PRINTER
def prettyprint(tag, m, lb=None):
    lb_note  = f"(LB {lb}) " if lb else ""
    rrr_note = f"  RRR:{m['RRR']}" if 'RRR' in m else ""
    if FOREX_MODE:
        print(f"{tag:>8} {lb_note}| Trades:{m['Trades']:4d}  "
              f"ROI:{m['ROI']:7.2f}R  PF:{m['PF']:6.2f}  Shp:{m['Sharpe']:6.2f}  "
              f"Win:{m['WinRate']*100:6.2f}%  Exp:{m['Exp']:7.2f}R  "
              f"MaxDD:{m['MaxDrawdown']:7.2f}R{rrr_note}")
    else:
        print(f"{tag:>8} {lb_note}| Trades:{m['Trades']:4d}  "
              f"ROI:${m['ROI'] * ACCOUNT_SIZE:,.2f}  "
              f"PF:{m['PF']:6.2f}  Shp:{m['Sharpe']:6.2f}  "
              f"Win:{m['WinRate']*100:6.2f}%  "
              f"Exp:${m['Exp'] * ACCOUNT_SIZE:,.2f}  "
              f"MaxDD:${m['MaxDrawdown'] * ACCOUNT_SIZE:,.2f}{rrr_note}")


# 9. EXPORT
def export_trades(trades, df, strat, window, sample, path, write_header):
    cols = [
        'strategy','window','sample','side',                  #  added 'side'
        'entry_time','open_entry','high_entry','low_entry','close_entry',
        'exit_time','open_exit','high_exit','low_exit','close_exit',
        'pnl'                                                 #  added 'pnl'
    ]
    t, o, h, l, c = (df[x].values for x in ('time','open','high','low','close'))
    rows = []
    for trade in trades:
        side, ei, xi, _entry_price, _exit_price, _qty, pnl = trade
        rows.append([
            strat,
            window,
            sample,
            side,                                            #  include side
            t[ei], o[ei], h[ei], l[ei], c[ei],
            t[xi], o[xi], h[xi], l[xi], c[xi],
            pnl                                              #  include P&L
        ])
    df_export = pd.DataFrame(rows, columns=cols)
    _safe_append_or_write_trade_csv(df_export, path, write_header)


def optimise_regime_full(df_full, regimes, target_regime, current_lbs,
                         lb_range, metric, min_trades):
    """
    Two-stage regime optimiser:
      1) Coarse search in steps of 2 over lb_range for target_regime,
         holding other regimes at current_lbs.
      2) Pick best coarse lookback, then fine-tune by testing
         best_lb - 1 and best_lb + 1.
      3) Return the best (lb, rrr, metrics) tuple.
    """
    import math
    # Was: `global last_unfiltered_raw`. Moved to _runtime_state.

    # helper: run one candidate lookback and return (val, lb, rrr, met) or None
    def _evaluate(lb):
        # 1) set candidate look-backs
        cand_lbs = current_lbs.copy()
        cand_lbs[target_regime] = lb

        # 2) generate raw signals and parse
        base = compute_indicators(df_full.copy(), DEFAULT_LB) if 'EMA_20' not in df_full else df_full.copy()
        raw = create_regime_signals(base, cand_lbs, regimes)
        _runtime_state['last_unfiltered_raw'] = raw.copy()
        sig = parse_signals(raw, df_full['time'])

        # 3) backtest (with or without RRR probe)
        if not OPTIMIZE_RRR:
            _, met, _, _, _ = backtest(base, sig)
            rrr_used = None
        else:
            # probe at 5R
            tp_old, flag_old = TP_PERCENTAGE, USE_TP
            globals()['TP_PERCENTAGE'], globals()['USE_TP'] = 5 * SL_PERCENTAGE, True
            trades_p, _, _, _, _ = backtest(base, sig)

            # collect peak/close only for target_regime entries
            peak_Rs, close_Rs = [], []
            for side, ent, exi, *_ in trades_p:
                if regimes.iloc[ent] != target_regime:
                    continue
                entry = base['close'].iloc[ent]
                risk = entry * SL_PERCENTAGE / 100.0
                is_long = (side == 'long') if LEGACY_SIDE_BUG else (side == 1)
                if is_long:
                    peak = base['high'].iloc[ent:exi+1].max()
                    close = base['close'].iloc[exi]
                    peak_Rs.append(min((peak - entry) / risk, 5.0))
                    close_Rs.append((close - entry) / risk)
                else:
                    trough = base['low'].iloc[ent:exi+1].min()
                    close = base['close'].iloc[exi]
                    peak_Rs.append(min((entry - trough) / risk, 5.0))
                    close_Rs.append((entry - close) / risk)

            # choose best RRR if we have any trades
            best_rrr_cand = None
            if peak_Rs:
                sums = {
                    r: np.where(np.array(peak_Rs) >= r, r, close_Rs).sum()
                    for r in range(1, 6)
                }
                best_rrr_cand = max(sums, key=sums.get)

            # rerun full backtest with candidate RRR
            if best_rrr_cand is not None:
                globals()['TP_PERCENTAGE'], globals()['USE_TP'] = best_rrr_cand * SL_PERCENTAGE, True
            _, met, _, _, _ = backtest(base, sig)
            rrr_used = best_rrr_cand
            if best_rrr_cand is not None:
                met['RRR'] = best_rrr_cand

            # restore TP settings
            globals()['TP_PERCENTAGE'], globals()['USE_TP'] = tp_old, flag_old

        # 4) filter by minimum trades
        if met['Trades'] < min_trades:
            return None

        # 5) Drawdown constraint
        if dd_constraint is not None and met['MaxDrawdown'] > dd_constraint:
            return None

        # 6) Compute value
        val = -met[metric] if metric == 'MaxDrawdown' else met[metric]
        return val, lb, met

    # build full and coarse lookback lists
    all_lbs = list(lb_range)
    coarse_lbs = all_lbs[::2]

    # Coarse pass
    coarse_results = [res for lb in coarse_lbs if (res := _evaluate(lb)) is not None]


    # Fallback if none
    if not coarse_results:
        raw_lb = current_lbs[target_regime]
        print(f"No lookback for {target_regime} meets drawdown  {DRAWDOWN_CONSTRAINT}, using raw LB {raw_lb}")
        # Recompute metrics at raw_lb
        base = (compute_indicators(df_full.copy(), DEFAULT_LB)
                if 'EMA_20' not in df_full else df_full.copy())
        raw = create_regime_signals(base, current_lbs, regimes)
        sig = parse_signals(raw, df_full['time'])
        _, met_raw, *_ = backtest(base, sig)
        return raw_lb, None, met_raw


    # sort by val descending
    coarse_results.sort(key=lambda x: x[0], reverse=True)
    best_val, best_lb, best_rrr, best_met = coarse_results[0]

    # 2) Fine pass: test neighbors of best_lb
    idx = all_lbs.index(best_lb)
    candidates = [(best_val, best_lb, best_rrr, best_met)]
    for neighbor_idx in (idx - 1, idx + 1):
        if 0 <= neighbor_idx < len(all_lbs):
            res = _evaluate(all_lbs[neighbor_idx])
            if res is not None:
                candidates.append(res)

    # pick the best overall
    candidates.sort(key=lambda x: x[0], reverse=True)
    _, final_lb, final_rrr, final_met = candidates[0]
    return final_lb, final_rrr, final_met
def backtest_continuous_regime(df, best_lbs):
    """
    Continuous regime-switching backtest with RRR applied exactly
    as in optimiser().
    """
    # 1) compute EMAs once
    dfi = df.copy()
    for span in (20, 200, 900):
        dfi[f'EMA_{span}'] = dfi['close'].ewm(span=span, adjust=False).mean()
    for lb in set(best_lbs.values()):
        if lb is not None:
            dfi[f'EMA_{lb}'] = dfi['close'].ewm(span=lb, adjust=False).mean()

    # 2) label regimes
    regimes = detect_regimes(dfi)

    # 3) build full raw & filter
    ema20    = dfi['EMA_20'].shift(1)
    ema_lbs  = {r: dfi[f'EMA_{lb}'].shift(1) for r, lb in best_lbs.items()}
    raw_full = np.zeros(len(dfi), dtype=np.int8)
    for reg, lb in best_lbs.items():
        m = regimes == reg
        raw_full[m] = np.where(ema20[m] > ema_lbs[reg][m], 1, -1)
    raw_full = filter_raw_signals(raw_full, regimes)

    # 4) RRR optimisation on the full signal (exactly like optimiser)
    if OPTIMIZE_RRR:
        old_tp, old_flag = TP_PERCENTAGE, USE_TP
        globals()['TP_PERCENTAGE'] = 5 * SL_PERCENTAGE
        globals()['USE_TP']        = True

        _runtime_state['last_unfiltered_raw'] = raw_full.copy()
        trades_probe, _, _, _, _ = backtest(dfi, parse_signals(raw_full, dfi['time']))

        # compute peak_Rs & close_Rs
        peak_Rs, close_Rs = [], []
        for side, e, x, _, _, *_ in trades_probe:
            entry = dfi['close'].iloc[e]
            risk  = entry * SL_PERCENTAGE/100
            is_long = (side == 'long') if LEGACY_SIDE_BUG else (side == 1)
            if is_long:
                seg       = dfi['high'].iloc[e:x+1].values
                peak_R    = (seg.max() - entry) / risk
                close_R   = (dfi['close'].iloc[x] - entry) / risk
            else:
                seg       = dfi['low'].iloc[e:x+1].values
                peak_R    = (entry - seg.min()) / risk
                close_R   = (entry - dfi['close'].iloc[x]) / risk

            peak_Rs.append(min(peak_R, 5.0))
            close_Rs.append(close_R)

        arr_peak  = np.array(peak_Rs)
        arr_close = np.array(close_Rs)
        sums = {
            R: np.where(arr_peak>=R, float(R), arr_close).sum()
            for R in range(1,6)
        }
        best_rrr = max(sums, key=sums.get)

        # apply it
        globals()['TP_PERCENTAGE'] = best_rrr * SL_PERCENTAGE
        globals()['USE_TP']        = True

        # restore after final backtest below
    else:
        best_rrr = None
        old_tp, old_flag = TP_PERCENTAGE, USE_TP

    # 5) final full-series backtest with RRR applied
    _runtime_state['last_unfiltered_raw'] = raw_full.copy()
    sig = parse_signals(raw_full, dfi['time'])
    trades, metrics, eq_frac, rets, _ = backtest(dfi, sig)
    if best_rrr is not None:
        metrics['RRR'] = best_rrr

    # restore globals
    globals()['TP_PERCENTAGE'] = old_tp
    globals()['USE_TP']        = old_flag


    return trades, metrics, eq_frac, rets

def optimize_regimes_sequential(is_df, config: Optional[Config] = None):
    """
    Sequential per-regime LB optimisation with coarse/fine search (and optional
    RRR optimisation). Iterates over `REGIME_LABELS` in order; on phase k the
    LB for label k is searched while LBs for labels 0..k-1 are held at their
    previously chosen values and labels k+1.. are held at DEFAULT_LB.
    Returns:
      best_lbs:  dict {label: lb}        (one entry per REGIME_LABELS)
      best_rrrs: dict {label: rrr | None}
    Works with REGIME_LABELS of any length in {2, 3, 4, 5}.

    `config` is optional; see `optimiser` docstring for the contract.
    """
    with with_config(config):
        return _optimize_regimes_sequential_impl(is_df)


def _optimize_regimes_sequential_impl(is_df):
    import math
    # Was: `global last_unfiltered_raw`. Moved to _runtime_state.

    dfi = is_df.copy()
    dfi['EMA_20']  = is_df['close'].ewm(span=20,  adjust=False).mean()
    dfi['EMA_200'] = is_df['close'].ewm(span=200, adjust=False).mean()
    dfi['EMA_900'] = is_df['close'].ewm(span=900, adjust=False).mean()
    regimes = detect_regimes(dfi)

    ema20 = dfi['EMA_20'].shift(1).values

    lbs_candidates = [
        lb for lb in range(*LOOKBACK_RANGE)
        if lb != FAST_EMA_SPAN
    ]
    slow_emas = {
        lb: is_df['close'].ewm(span=lb, adjust=False).mean().shift(1).values
        for lb in lbs_candidates
    }

    regs = list(REGIME_LABELS)
    best_lbs  = {r: DEFAULT_LB for r in regs}
    best_rrrs = {r: None       for r in regs}

    # Phase-by-phase search
    for reg in regs:
        print(f" Phase: optimize {reg}")

        # helper to evaluate one lookback for this regime
        def _evaluate(lb):
            # build lookback map with candidate for this regime
            temp_lbs = best_lbs.copy()
            temp_lbs[reg] = lb

            # construct raw signal array
            raw = np.empty(len(dfi), dtype=np.int8)
            for i, r in enumerate(regimes):
                chosen_lb = temp_lbs[r]
                raw[i]    = 1 if ema20[i] > slow_emas[chosen_lb][i] else -1

            _runtime_state['last_unfiltered_raw'] = None
            sig = parse_signals(raw, dfi['time'])

            # backtest (with or without RRR probe)
            if OPTIMIZE_RRR:
                old_tp, old_tp_flag = TP_PERCENTAGE, USE_TP
                globals()['TP_PERCENTAGE'] = 5 * SL_PERCENTAGE
                globals()['USE_TP']        = True

                _runtime_state['last_unfiltered_raw'] = None
                trades_p, _, _, _, _ = backtest(dfi, sig)

                # collect peak/close R for this target regime
                peak_Rs, close_Rs = [], []
                for side, ent, exi, entry, exit_p, *_ in trades_p:
                    if regimes.iloc[ent] != reg:
                        continue
                    risk = entry * SL_PERCENTAGE / 100.0
                    is_long = (side == 'long') if LEGACY_SIDE_BUG else (side == 1)
                    if is_long:
                        seg = dfi['high'].iloc[ent:exi+1].values
                        peak_Rs.append(min((seg.max() - entry) / risk, 5.0))
                        close_Rs.append((exit_p - entry) / risk)
                    else:
                        seg = dfi['low'].iloc[ent:exi+1].values
                        peak_Rs.append(min((entry - seg.min()) / risk, 5.0))
                        close_Rs.append((entry - exit_p) / risk)

                # choose best RRR if any trades
                best_rrr_cand = None
                if peak_Rs:
                    arr_peak = np.array(peak_Rs)
                    arr_close = np.array(close_Rs)
                    sums = {
                        r: np.where(arr_peak >= r, r, arr_close).sum()
                        for r in range(1, 6)
                    }
                    best_rrr_cand = max(sums, key=sums.get)

                # rerun backtest with chosen RRR
                if best_rrr_cand is not None:
                    globals()['TP_PERCENTAGE'] = best_rrr_cand * SL_PERCENTAGE
                    globals()['USE_TP']        = True

                _runtime_state['last_unfiltered_raw'] = None
                trades, met, eq, rets, _ = backtest(dfi, sig)
                if best_rrr_cand is not None:
                    met['RRR'] = best_rrr_cand

                # restore TP settings
                globals()['TP_PERCENTAGE'] = old_tp
                globals()['USE_TP']        = old_tp_flag

                rrr_used = best_rrr_cand
            else:
                _runtime_state['last_unfiltered_raw'] = None
                trades, met, eq, rets, _ = backtest(dfi, sig)
                rrr_used = None

            # skip too-few trades
            if met['Trades'] < MIN_TRADES:
                return None

            # compute score (maximize)
            score = -met[OPT_METRIC] if OPT_METRIC == 'MaxDrawdown' else met[OPT_METRIC]
            return score, lb, rrr_used, met

        # build coarse subset and run coarse pass
        coarse_lbs = lbs_candidates[::2]
        coarse_results = []
        for lb in coarse_lbs:
            res = _evaluate(lb)
            if res is not None:
                coarse_results.append(res)

        if not coarse_results:
            print(f"{reg:>9} no valid lookbacks!")
            best_lbs[reg] = None
            best_rrrs[reg] = None
            continue

        coarse_results.sort(key=lambda x: x[0], reverse=True)
        best_score, best_lb_coarse, best_rrr_coarse, best_met_coarse = coarse_results[0]

        # fine-tune: test neighbors of best coarse
        idx = lbs_candidates.index(best_lb_coarse)
        candidates = [(best_score, best_lb_coarse, best_rrr_coarse, best_met_coarse)]
        for neighbor_idx in (idx - 1, idx + 1):
            if 0 <= neighbor_idx < len(lbs_candidates):
                res = _evaluate(lbs_candidates[neighbor_idx])
                if res is not None:
                    candidates.append(res)

        candidates.sort(key=lambda x: x[0], reverse=True)
        _, final_lb, final_rrr, final_met = candidates[0]

        # save and report
        best_lbs[reg]  = final_lb
        best_rrrs[reg] = final_rrr
        rrr_info = f" | RRR={final_rrr}" if final_rrr is not None else ""
        print(f"{reg:>9} best LB = {final_lb} | {OPT_METRIC}: {final_met[OPT_METRIC]:.4f}{rrr_info}")

    return best_lbs, best_rrrs
signals_cache = {}

## 10. CLASSIC SINGLE-RUN  updated so that RRR is optimized and applied per regime
# 10. CLASSIC SINGLE-RUN  corrected to collect OOS-regime returns by segment
def classic_single_run(df, config: Optional[Config] = None):
    """Single-window IS/OOS run with optional RRR optimisation.

    `config` is optional; see `optimiser` docstring for the contract.
    """
    with with_config(config):
        return _classic_single_run_impl(df)


def _classic_single_run_impl(df):
    # Was: `global TP_PERCENTAGE, USE_TP, signals_cache` — redundant.
    # TP/USE_TP rebinds inside this function are done via `globals()['X']=...`
    # (the only path that the engine respects when reading them back through
    # `backtest()` -> `_backtest_numba_core`); signals_cache is mutated
    # in-place. Removed in v0.4.0.
    m1 = None
    m2 = None
    m1r = None
    m2r = None

    signals_cache['mode'] = 'classic'   # <-- add this

    if os.path.exists(EXPORT_PATH):
        _safe_remove_trade_csv(EXPORT_PATH)
    first_export = True


    N = len(df)
    is_df  = df.iloc[N - OOS_CANDLES - BACKTEST_CANDLES : N - OOS_CANDLES].reset_index(drop=True)
    oos_df = df.iloc[N - OOS_CANDLES : N].reset_index(drop=True)

    # ---------- RAW baseline (console-only) --------------------------------
    for tag, subset in [('IS-raw', is_df), ('OOS-raw', oos_df)]:
        # 1) compute indicators
        dfi = compute_indicators(subset, DEFAULT_LB)
        # 2) build raw signals
        raw = create_raw_signals(dfi, DEFAULT_LB)
        # 3) reset true-raw buffer so exits align with this segment
        # Was: `global last_unfiltered_raw; last_unfiltered_raw = None`
        _runtime_state['last_unfiltered_raw'] = None
        # 4) parse signals (entries + exits)
        sig = parse_signals(raw, dfi['time'])
        # 5) backtest
        tr, m, eq, rets, _ = backtest(dfi, sig)

        # --- print & export ---
        if tag == 'IS-raw':
            prettyprint('IS-raw', m)
        else:  # OOS-raw
            if USE_OOS2:
                # split trades into OOS1 vs OOS2
                o1 = [t for t in tr if t[2] < ORIGINAL_OOS]
                o2 = [t for t in tr if t[2] >= ORIGINAL_OOS]
                m1 = _metrics_from_trades(o1)
                m2 = _metrics_from_trades(o2)
                prettyprint('OOS1-raw', m1)
                prettyprint('OOS2-raw', m2)
            else:
                prettyprint('OOS-raw', m)

        # store IS / OOS metrics & equity
        if tag == 'IS-raw':
            tr_is_raw,   met_is_raw,   eq_is_raw,   rets_is_raw   = tr, m, eq, rets
        else:
            tr_oos_raw,  met_oos_raw,  eq_oos_raw,  rets_oos_raw  = tr, m, eq, rets
    if PRINT_EQUITY_CURVE: import numpy as np, matplotlib.pyplot as plt; plt.figure(figsize=(10,5)); plt.plot((np.concatenate((eq_is_raw, eq_oos_raw + (eq_is_raw[-1]-1))) * ACCOUNT_SIZE), label='RAW Equity'); plt.title('RAW Equity Curve'); plt.legend(); plt.grid(True); plt.tight_layout(); plt.show()

    print("\n Replication BEFORE optimisation ")
    for mm in METRICS:
        r = met_oos_raw[mm] / met_is_raw[mm] if met_is_raw[mm] else math.nan
        print(f"  {mm:>12}: {r:6.3f}")

    # ---------- CASE A  regime segmentation ------------------------------
    if USE_REGIME_SEG and not USE_WFO:
        # 1) find best LBs per regime on IS
        dfi_full   = compute_indicators(is_df, DEFAULT_LB)
        regimes_is = detect_regimes(dfi_full)
        best_lbs, best_rrrs = optimize_regimes_sequential(is_df)

        # report RRRs
        print("\n Best RRR per regime ")
        for reg in REGIME_LABELS:
            rrr = best_rrrs.get(reg)
            print(f"  {reg:>9}: RRR = {rrr if rrr is not None else 'None'}")

        # 2) IS backtest by regime
        tr_is_reg, met_is_reg, eq_is_reg, rets_is_reg = \
            backtest_continuous_regime(is_df, best_lbs)
        print("\nIS-regime backtest")
        prettyprint('IS-reg', met_is_reg)

        if not USE_WFO:
            export_trades(
                tr_is_reg, is_df,
                'EMA-crossover-regime',      # strategy name
                f"LB-regime",                # window label, you can customize
                'IS-reg',                    # sample label
                EXPORT_PATH,
                first_export                 # True on very first call
            )
            first_export = False

        # 3) apply filters if enabled
        evaluate_filters(tr_is_reg, rets_is_reg,
                         detect_regimes(compute_indicators(is_df, DEFAULT_LB)))
        print("\n Filter Conclusion for OOS:")
        if FILTER_REGIMES:
            print(f"  Regimes removed: {', '.join(blocked_regimes) or 'None'}")
        if FILTER_DIRECTIONS:
            print(f"  Directions removed: {', '.join(blocked_directions) or 'None'}")
        if FILTER_REGIMES and FILTER_DIRECTIONS and blocked_pairs:
            print("  Regime-Direction pairs removed:")
            for r, dirs in blocked_pairs.items():
                if dirs:
                    print(f"    {r}: {', '.join(dirs)}")

        # 4) OOS backtest by regime
        tr_oos_reg, met_oos_reg, eq_oos_reg, rets_oos_reg = backtest_continuous_regime(oos_df, best_lbs)
        print("\nOOS-regime backtest")
        if USE_OOS2:
            # split into first vs. second window
            o1r = [t for t in tr_oos_reg if t[2] < ORIGINAL_OOS]
            o2r = [t for t in tr_oos_reg if t[2] >= ORIGINAL_OOS]
            m1r = _metrics_from_trades(o1r)
            m2r = _metrics_from_trades(o2r)
            prettyprint('OOS1-reg', m1r)
            prettyprint('OOS2-reg', m2r)
        else:
            prettyprint('OOS-reg', met_oos_reg)

        # 5) count how many of those OOS trades exited in the first ORIGINAL_OOS bars
        if USE_OOS2:
            n_oos1_trades = sum(
                1
                for side, ent, exit_i, *rest in tr_oos_reg
                if exit_i < ORIGINAL_OOS
            )
            o1r = [t for t in tr_oos_reg if t[2] < ORIGINAL_OOS]
            o2r = [t for t in tr_oos_reg if t[2] >= ORIGINAL_OOS]
            if not USE_WFO:
                export_trades(o1r, oos_df, 'EMA-crossover-regime', 'LB-regime', 'OOS1-reg', EXPORT_PATH, first_export)
                export_trades(o2r, oos_df, 'EMA-crossover-regime', 'LB-regime', 'OOS2-reg', EXPORT_PATH, first_export)
        else:
            n_oos1_trades = None
            if not USE_WFO:
                export_trades(tr_oos_reg, oos_df, 'EMA-crossover-regime', 'LB-regime', 'OOS-reg', EXPORT_PATH, first_export)

        # 6) build equity for plotting
        if PRINT_EQUITY_CURVE:
            # 1) offset OOS so it starts where IS left off
            offset = eq_is_reg[-1] - 1
            eq_baseline_oos = eq_oos_reg + offset
            # 2) stitch IS + OOS into one curve
            eq_baseline = np.concatenate((eq_is_reg, eq_baseline_oos))
        else:
            eq_baseline = None


        # cache for robustness
        signals_cache['mode']     = 'regime'
        signals_cache['is_df']    = is_df
        signals_cache['oos_df']   = oos_df
        signals_cache['best_lbs'] = best_lbs

        # 7) compute trade-based splits
        split1 = len(eq_is_reg) - 1
        split2 = (split1 + n_oos1_trades) if n_oos1_trades is not None else None

        # --- 4) Monte Carlo on optimised IS ---
        if USE_MONTE_CARLO:
            monte_carlo(rets_is_reg, met_is_reg, MC_RUNS)
            
        # 8) return everything
        return {
            'met_is':          met_is_reg,
            'met_oos_reg':     met_oos_reg,
            'eq_is':           eq_is_reg,
            'met_is_opt':      None,
            'eq_is_reg':       eq_is_reg,
            'eq_oos_reg':      eq_oos_reg,
            'eq_baseline_is':  eq_is_reg,
            'eq_baseline_oos': eq_baseline_oos,
            'n_oos1_trades':   n_oos1_trades,
            'split1':          split1,
            'split2':          split2,
            'met_oos1_reg':     m1r,
            'met_oos2_reg':     m2r,
            'eq_baseline':     eq_baseline,

        }


    # ---------- CASE B  classic optimisation (no regime segment) ----------
    best_lb, met_is_opt = optimiser(is_df, range(*LOOKBACK_RANGE), OPT_METRIC, MIN_TRADES)

    # Item #1: emit the baseline IS objective surface (opt-in, default off).
    if EMIT_OPT_SURFACE:
        import backtester as _bt
        from backtester import opt_surface as _osf
        _hdr = not os.path.exists(_osf._surface_path(EXPORT_PATH, _osf._resolve_format()))
        _osf.emit_surface_classic(_bt, is_df, "baseline", write_header=_hdr)

    if best_lb:
        best_rrr = met_is_opt.get('RRR') if OPTIMIZE_RRR else None
        rrr_note = f"  |  Best RRR = {best_rrr}" if best_rrr is not None else ""
        print(f"\nBest {OPT_METRIC} look-back = {best_lb}{rrr_note}\n")
        prettyprint('IS-opt', met_is_opt, best_lb)

        # --- 1) IS-opt run ---
        tp_old, tp_flag_old = TP_PERCENTAGE, USE_TP
        if best_rrr is not None:
            globals()['TP_PERCENTAGE'] = best_rrr * SL_PERCENTAGE
            globals()['USE_TP']        = True

        dfi_is_opt = compute_indicators(is_df, best_lb)
        raw_is_opt = create_raw_signals(dfi_is_opt, best_lb)
        raw_is_opt = filter_raw_signals(raw_is_opt, None)
        sig_is_opt = parse_signals(raw_is_opt, dfi_is_opt['time'])
        tr_is_opt, met_is_opt, eq_is_opt, rets_is_opt, _ = backtest(dfi_is_opt, sig_is_opt)

        globals()['TP_PERCENTAGE'] = tp_old
        globals()['USE_TP']        = tp_flag_old

        # --- 2) OOS-opt run ---
        tp_old, tp_flag_old = TP_PERCENTAGE, USE_TP
        if best_rrr is not None:
            globals()['TP_PERCENTAGE'] = best_rrr * SL_PERCENTAGE
            globals()['USE_TP']        = True

        dfo_opt     = compute_indicators(oos_df, best_lb)
        raw_tmp     = create_raw_signals(dfo_opt, best_lb)
        raw_tmp     = filter_raw_signals(raw_tmp, None)
        sig_oos_opt = parse_signals(raw_tmp, dfo_opt['time'])
        tr_oos_opt, met_oos_opt, eq_oos_opt, rets_oos_opt, _ = backtest(dfo_opt, sig_oos_opt)
        if best_rrr is not None:
            met_oos_opt['RRR'] = best_rrr
 
        if not USE_WFO:
            export_trades(tr_is_opt, dfi_is_opt,
                          'EMA-crossover', f'LB{best_lb}', 'IS-opt',
                          EXPORT_PATH, first_export)
            first_export = False

        # split into OOS1 vs OOS2
        if USE_OOS2:
            o1 = [t for t in tr_oos_opt if t[2] < ORIGINAL_OOS]
            o2 = [t for t in tr_oos_opt if t[2] >= ORIGINAL_OOS]
            n_oos1_trades = len(o1)

            m1 = _metrics_from_trades(o1)
            m2 = _metrics_from_trades(o2)
            print(f"\nOOS1 back-test (first {ORIGINAL_OOS} bars, LB{best_lb})")
            prettyprint('OOS1-opt', m1, best_lb)
            print(f"\nOOS2 back-test (last {ORIGINAL_OOS} bars, LB{best_lb})")
            prettyprint('OOS2-opt', m2, best_lb)
            if not USE_WFO:
                export_trades(o1, dfo_opt,
                              'EMA-crossover', f'LB{best_lb}', 'OOS1-opt',
                              EXPORT_PATH, first_export)
                export_trades(o2, dfo_opt,
                              'EMA-crossover', f'LB{best_lb}', 'OOS2-opt',
                              EXPORT_PATH, first_export)            
        else:
            prettyprint('OOS-opt', met_oos_opt, best_lb)
            n_oos1_trades = None
            if not USE_WFO:
                export_trades(tr_oos_opt, dfo_opt,
                              'EMA-crossover', f'LB{best_lb}', 'OOS-opt',
                              EXPORT_PATH, first_export)

        globals()['TP_PERCENTAGE'] = tp_old
        globals()['USE_TP']        = tp_flag_old

        print("\n Replication OOS-opt / IS-opt ")
        for mm in METRICS:
            r = met_oos_opt[mm] / met_is_opt[mm] if met_is_opt[mm] else math.nan
            print(f"  {mm:>12}: {r:6.3f}")

        # --- 3) Build equity curve for plotting (additive, not multiplicative) ---
        if PRINT_EQUITY_CURVE:
            # 3.a) combine IS and OOS trades in chronological order
            trades_all = tr_is_opt + tr_oos_opt

            # 3.b) extract pnl from each trade tuple
            pnl_list = [trade[-1] for trade in trades_all]  # last element is pnl

            # 3.c) cumulative-sum into an equity curve
            if FOREX_MODE:
                # pnl_list is in R-units; equity starts at 0R
                eq_baseline = np.concatenate(([0.0], np.cumsum(pnl_list)))
            else:
                # pnl_list is in USD; equity starts at ACCOUNT_SIZE
                eq_usd      = np.concatenate(([ACCOUNT_SIZE], ACCOUNT_SIZE + np.cumsum(pnl_list)))
                eq_baseline = eq_usd / ACCOUNT_SIZE
        else:
            eq_baseline = None

        # --- 4) Monte Carlo on optimised IS ---
        if USE_MONTE_CARLO:
            monte_carlo(rets_is_opt, met_is_opt, MC_RUNS)

        signals_cache['best_lb'] = best_lb
        signals_cache['best_rrr'] = best_rrr
        signals_cache['sig_is'] = sig_is_opt
        signals_cache['dfi_is'] = dfi_is_opt
        signals_cache['sig_oos'] = sig_oos_opt
        signals_cache['dfo_oos'] = dfo_opt

        split1 = len(eq_is_opt) - 1
        split2 = split1 + n_oos1_trades if n_oos1_trades is not None else None


        return {
            'met_is':         met_is_raw,
            'eq_is':          eq_is_raw,
            'met_is_opt':     met_is_opt,
            'eq_is_opt':      eq_is_opt,
            'met_oos_opt':    met_oos_opt,
            'eq_oos_opt':     eq_oos_opt,
            'eq_baseline':    eq_baseline,
            'best_lb':        best_lb,
            'best_rrr':       best_rrr,
            'split1':       split1,
            'split2':       split2,
            'n_oos1_trades':n_oos1_trades,
            'met_oos1_opt':   m1,
            'met_oos2_opt':   m2
        }

    # --------- fallback (no valid optimisation) ----------------------------
    if PRINT_EQUITY_CURVE:
        offset = eq_is_raw[-1] - 1
        eq_baseline = np.concatenate((eq_is_raw, eq_oos_raw + offset))
    else:
        eq_baseline = None

    if USE_MONTE_CARLO:
        monte_carlo(rets_is_raw, met_is_raw, MC_RUNS)

    # count raw OOS-1 trades
    if USE_OOS2:
        n_oos1_trades = sum(
            1
            for side, ent, exit_i, *rest in tr_oos_raw
            if exit_i < ORIGINAL_OOS
        )
    else:
        n_oos1_trades = None

    split1 = len(eq_is_raw) - 1
    split2 = split1 + n_oos1_trades if n_oos1_trades is not None else None


    return {
        'met_is'     : met_is_raw,
        'eq_is'      : eq_is_raw,
        'met_is_opt' : None,
        'eq_baseline': eq_baseline,
        'split1':       split1,
        'split2':       split2
    }

# 12. MAIN  add equity-curve plot when USE_WFO is False


# 11. WALK-FORWARD (only windows)
def _run_wfo_window(is_df, oos_df, lb, window_tag, regimes_is, regimes_oos, rb_scenarios, export_is=False, best_lbs=None):
    """
    Backtest a single WFO window (IS + OOS), optionally with robustness tweaks.

    If `best_lbs` is given (regime-segmentation mode), the active LB rotates per
    bar according to `regimes_is` / `regimes_oos`. Otherwise a single `lb` is
    used for the whole window. The WFO walk cadence (window boundaries) is
    decided by the caller and is *not* affected by regime changes — only the
    per-bar choice of LB inside the window changes.
    """
    use_regime = best_lbs is not None

    def _run_segment(df_seg, lb_use, regimes_seg, drift=False, best_lbs_seg=None):
        if best_lbs_seg is not None:
            dfi = df_seg.copy()
            for span in (20, 200, 900):
                dfi[f'EMA_{span}'] = dfi['close'].ewm(span=span, adjust=False).mean()
            for lb_v in {v for v in best_lbs_seg.values() if v is not None}:
                dfi[f'EMA_{lb_v}'] = dfi['close'].ewm(span=lb_v, adjust=False).mean()
            raw = create_regime_signals(dfi, best_lbs_seg, regimes_seg)
        else:
            dfi = compute_indicators(df_seg.copy(), lb_use)
            raw = create_raw_signals(dfi, lb_use)
        raw = filter_raw_signals(raw, regimes_seg)
        sig = parse_signals(raw, dfi['time'])
        if drift:
            sig = drift_entries(sig)
        return backtest(dfi, sig)

    bs_lbs = best_lbs if use_regime else None
    lb_tag = (",".join(f"{r}:{v}" for r, v in best_lbs.items() if v is not None)
              if use_regime else lb)

    # baseline
    tr_is,  met_is,  eq_is,  _, _ = _run_segment(is_df,  lb, regimes_is, drift=False, best_lbs_seg=bs_lbs)
    tr_oos, met_oos, _, rets_oos, _ = _run_segment(oos_df, lb, regimes_oos, drift=False, best_lbs_seg=bs_lbs)
    prettyprint(f"{window_tag} IS",  met_is,  lb_tag)
    prettyprint(f"{window_tag} OOS", met_oos, lb_tag)

    # export WFO trades for this window (IS + OOS) to the shared trade list
    header_needed = not os.path.exists(EXPORT_PATH)
    if export_is:
        export_trades(tr_is,  is_df,  'EMA-crossover-WFO', window_tag, 'IS',  EXPORT_PATH, header_needed)
        header_needed = False  # header already written if needed
    export_trades(tr_oos, oos_df, 'EMA-crossover-WFO', window_tag, 'OOS', EXPORT_PATH, header_needed)

    # robustness overlays (one line per scenario)
    rb_rets    = {}
    rb_eq_is   = {}
    for label, opts in rb_scenarios:
        fee_mult  = opts["fee_mult"]
        slip_mult = opts["slip_mult"]
        news_on   = opts["news_on"]
        drift_on  = opts["drift_on"]
        var_on    = opts["var_on"]
        # IND_VARIANCE_SEED: previously unseeded (paper-time review noted
        # the +/- 1 LB perturbation propagated to a different optimised
        # parameter across host architectures). Seeded to 42 here so the
        # perturbation is deterministic. Mirrors IND_VARIANCE_SEED in
        # quant-research-framework-rs/src/lib.rs.
        rng       = opts.get("rng", random.Random(42))

        if fee_mult == 1 and slip_mult == 1 and not news_on and not drift_on and not var_on:
            continue

        fee_old, slip_old = FEE_PCT, SLIPPAGE_PCT
        globals()['FEE_PCT']      = fee_old  * fee_mult
        globals()['SLIPPAGE_PCT'] = slip_old * slip_mult

        is_work  = inject_news_candles(is_df.copy())  if news_on else is_df
        oos_work = inject_news_candles(oos_df.copy()) if news_on else oos_df

        if use_regime:
            lb_rb_dict = {r: (max(1, v + rng.choice([-1, 1])) if (var_on and v is not None) else v)
                          for r, v in best_lbs.items()}
            lb_rb_tag = ",".join(f"{r}:{v}" for r, v in lb_rb_dict.items() if v is not None)
            try:
                _, met_is_rb, eq_is_rb, _, _ = _run_segment(is_work,  lb, regimes_is, drift=drift_on, best_lbs_seg=lb_rb_dict)
                _, met_oos_rb, _, rets_rb, _ = _run_segment(oos_work, lb, regimes_oos, drift=drift_on, best_lbs_seg=lb_rb_dict)
            finally:
                globals()['FEE_PCT'], globals()['SLIPPAGE_PCT'] = fee_old, slip_old
            prettyprint(f"{window_tag} IS+{label}",  met_is_rb,  lb_rb_tag)
            prettyprint(f"{window_tag} OOS+{label}", met_oos_rb, lb_rb_tag)
        else:
            lb_rb    = max(1, lb + rng.choice([-1, 1])) if var_on else lb
            try:
                _, met_is_rb, eq_is_rb, _, _ = _run_segment(is_work,  lb_rb, regimes_is, drift=drift_on)
                _, met_oos_rb, _, rets_rb, _ = _run_segment(oos_work, lb_rb, regimes_oos, drift=drift_on)
            finally:
                globals()['FEE_PCT'], globals()['SLIPPAGE_PCT'] = fee_old, slip_old
            prettyprint(f"{window_tag} IS+{label}",  met_is_rb,  lb_rb)
            prettyprint(f"{window_tag} OOS+{label}", met_oos_rb, lb_rb)
        rb_rets[label] = rets_rb
        rb_eq_is[label] = eq_is_rb

    return rets_oos, rb_rets, eq_is, rb_eq_is


def walk_forward(df, met_is_baseline, eq_is_baseline, config: Optional[Config] = None):
    """Rolling walk-forward driver. `config` is optional; see `optimiser`."""
    with with_config(config):
        return _walk_forward_impl(df, met_is_baseline, eq_is_baseline)


def _walk_forward_impl(df, met_is_baseline, eq_is_baseline):
    # Build robustness scenarios using the queued flags
    if ROBUSTNESS_SCENARIOS:
        items = list(ROBUSTNESS_SCENARIOS.items())[:MAX_ROBUSTNESS_SCENARIOS]
    else:
        default_flags = []
        if FEE_SHOCK:              default_flags.append("fee_shock")
        if SLIPPAGE_SHOCK:         default_flags.append("slippage_shock")
        if NEWS_CANDLES_INJECTION: default_flags.append("news_candles_injection")
        if ENTRY_DRIFT:            default_flags.append("entry_drift")
        if INDICATOR_VARIANCE:     default_flags.append("indicator_variance")
        items = [((" + ".join(default_flags)), tuple(default_flags))]

    rb_scenarios = []
    for name, flags in items:
        opts = _opts_from_flags(flags)
        if not any([opts["fee_mult"] != 1, opts["slip_mult"] != 1, opts["news_on"], opts["drift_on"], opts["var_on"]]):
            continue
        label = _label_from_flags(flags)
        rb_scenarios.append((label, opts))

    # ===== 1.  WFO **with** regime segmentation ============================
    # Rewritten in v0.2.0: WFO walks the standard cadence (candles or trades).
    # Regime segmentation only changes which per-regime LB is active for each
    # OOS bar; the WFO test/train boundaries are NOT shifted by regime changes.
    if USE_WFO and USE_REGIME_SEG:
        n           = len(df)
        start_total = n - OOS_CANDLES
        cur_start   = start_total
        window_no   = 1
        all_oos_rets = []
        rb_totals = {label: [] for label, _ in rb_scenarios}
        eq_is_first = None
        rb_eq_seed: dict[str, np.ndarray] = {}

        # full-data regimes (computed once; pluggable detector)
        dfi_full     = compute_indicators(df, DEFAULT_LB)
        regimes_full = detect_regimes(dfi_full)

        # one-shot evaluate_filters on the very first IS window (identical to
        # the legacy behaviour, but anchored to the WFO IS window not a regime
        # stretch). This matters when FILTER_REGIMES / FILTER_DIRECTIONS is on.
        is_start_init = max(0, start_total - BACKTEST_CANDLES)
        is_df_init    = df.iloc[is_start_init:start_total].reset_index(drop=True)
        regimes_init  = regimes_full.iloc[is_start_init:start_total].reset_index(drop=True)
        initial_lbs, _ = optimize_regimes_sequential(is_df_init)
        if initial_lbs and any(v is not None for v in initial_lbs.values()):
            dfi_init = is_df_init.copy()
            for span in (20, 200, 900):
                dfi_init[f'EMA_{span}'] = dfi_init['close'].ewm(span=span, adjust=False).mean()
            for lb_v in {v for v in initial_lbs.values() if v is not None}:
                dfi_init[f'EMA_{lb_v}'] = dfi_init['close'].ewm(span=lb_v, adjust=False).mean()
            raw_init = create_regime_signals(dfi_init, initial_lbs, regimes_init)
            sig_init = parse_signals(raw_init, dfi_init['time'])
            tr_init, _, _, rets_init, _ = backtest(dfi_init, sig_init)
            evaluate_filters(tr_init, rets_init, regimes_init)

        while cur_start < n:
            # --- decide window end (same cadence as the no-regime path) ---
            if WFO_TRIGGER_MODE == 'candles':
                cur_end = min(cur_start + WFO_TRIGGER_VAL, n)
            else:   # by trade-count
                is_win_start_p = cur_start - BACKTEST_CANDLES
                is_df_p        = df.iloc[is_win_start_p:cur_start].reset_index(drop=True)
                best_lbs_p, _  = optimize_regimes_sequential(is_df_p)
                if not best_lbs_p or all(v is None for v in best_lbs_p.values()):
                    break
                dfo_p     = df.iloc[cur_start:n].reset_index(drop=True)
                regimes_p = regimes_full.iloc[cur_start:n].reset_index(drop=True)
                dfi_p = dfo_p.copy()
                for span in (20, 200, 900):
                    dfi_p[f'EMA_{span}'] = dfi_p['close'].ewm(span=span, adjust=False).mean()
                for lb_v in {v for v in best_lbs_p.values() if v is not None}:
                    dfi_p[f'EMA_{lb_v}'] = dfi_p['close'].ewm(span=lb_v, adjust=False).mean()
                raw_p = create_regime_signals(dfi_p, best_lbs_p, regimes_p)
                raw_p = filter_raw_signals(raw_p, regimes_p)
                sig_p = parse_signals(raw_p, dfi_p['time'])
                tr_p, _, _, _, _ = backtest(dfi_p, sig_p)
                if not tr_p:
                    cur_end = n
                else:
                    idx     = min(WFO_TRIGGER_VAL, len(tr_p)) - 1
                    cur_end = min(cur_start + tr_p[idx][2] + 1, n)

            # --- IS slice + per-regime optimisation -----------------------
            is_win_start = cur_start - BACKTEST_CANDLES
            is_df_roll   = df.iloc[is_win_start:cur_start].reset_index(drop=True)
            regimes_is   = regimes_full.iloc[is_win_start:cur_start].reset_index(drop=True)

            best_lbs, _  = optimize_regimes_sequential(is_df_roll)
            if not best_lbs or all(v is None for v in best_lbs.values()):
                break
            if EMIT_OPT_SURFACE:
                import backtester as _bt
                from backtester import opt_surface as _osf
                _hdr = not os.path.exists(_osf._surface_path(EXPORT_PATH, _osf._resolve_format()))
                _osf.emit_surface_regime(_bt, is_df_roll, best_lbs,
                                         f"{window_no:02d}", write_header=_hdr)

            # --- OOS slice with regime-rotated signals --------------------
            dfo          = df.iloc[cur_start:cur_end].reset_index(drop=True)
            regimes_oos  = regimes_full.iloc[cur_start:cur_end].reset_index(drop=True)

            rets_oos, rb_rets_window, eq_is_window, rb_eq_is_window = _run_wfo_window(
                is_df_roll, dfo, lb=None,
                window_tag=f"W{window_no:02d}",
                regimes_is=regimes_is,
                regimes_oos=regimes_oos,
                rb_scenarios=rb_scenarios,
                export_is=(window_no == 1),
                best_lbs=best_lbs,
            )
            if eq_is_first is None:
                eq_is_first = eq_is_window
                for label, eq_is_rb in rb_eq_is_window.items():
                    rb_eq_seed[label] = eq_is_rb
            all_oos_rets.extend(rets_oos)
            for label in rb_totals:
                rb_totals[label].extend(rb_rets_window.get(label, []))

            cur_start = cur_end
            window_no += 1

        eq_seed = eq_is_first if eq_is_first is not None else eq_is_baseline
        all_oos_rets = np.array(all_oos_rets, dtype=float)
        eq_wfo = np.concatenate([eq_seed,
                                 eq_seed[-1] + np.cumsum(all_oos_rets)])

        rb_eq_curves: dict[str, np.ndarray] = {}
        if rb_totals:
            for label, vals in rb_totals.items():
                if not vals:
                    continue
                arr = np.array(vals, dtype=float)
                seed_rb = rb_eq_seed.get(label, eq_seed)
                rb_eq_curves[label] = np.concatenate([seed_rb,
                                                      seed_rb[-1] + np.cumsum(arr)])

        split_wfo_is = len(eq_seed) - 1
        return all_oos_rets, eq_wfo, rb_eq_curves, split_wfo_is

    # ===== 2.  WFO **without** regime segmentation =========================
    n           = len(df)
    start_total = n - OOS_CANDLES
    cur_start   = start_total
    window_no   = 1
    all_oos_rets = []
    rb_totals = {label: [] for label, _ in rb_scenarios}
    eq_is_first = None
    rb_eq_seed: dict[str, np.ndarray] = {}

    while cur_start < n:
        # --- decide window end ---------------------------------------------
        if WFO_TRIGGER_MODE == 'candles':
            cur_end = min(cur_start + WFO_TRIGGER_VAL, n)
        else:   # by trade-count
            is_win_start = cur_start - BACKTEST_CANDLES
            is_df_roll   = df.iloc[is_win_start:cur_start].reset_index(drop=True)
            lb_roll, _   = optimiser(is_df_roll, range(*LOOKBACK_RANGE), OPT_METRIC, MIN_TRADES)
            if not lb_roll:
                break
            dfo_tmp      = df.iloc[cur_start:n].reset_index(drop=True)
            raw_tmp      = create_raw_signals(compute_indicators(dfo_tmp, lb_roll), lb_roll)
            raw_tmp      = filter_raw_signals(raw_tmp, None)          # just dirs
            sig_tmp      = parse_signals(raw_tmp, dfo_tmp['time'])
            tr_tmp, _, _, _, _ = backtest(dfo_tmp, sig_tmp)
            if not tr_tmp:
                cur_end = n
            else:
                idx     = min(WFO_TRIGGER_VAL, len(tr_tmp)) - 1
                cur_end = min(cur_start + tr_tmp[idx][2] + 1, n)

        # --- real rolling IS  current OOS ---------------------------------
        is_win_start = cur_start - BACKTEST_CANDLES
        is_df_roll   = df.iloc[is_win_start:cur_start].reset_index(drop=True)
        lb_roll, _   = optimiser(is_df_roll, range(*LOOKBACK_RANGE), OPT_METRIC, MIN_TRADES)
        if not lb_roll:
            break
        if EMIT_OPT_SURFACE:
            import backtester as _bt
            from backtester import opt_surface as _osf
            _hdr = not os.path.exists(_osf._surface_path(EXPORT_PATH, _osf._resolve_format()))
            _osf.emit_surface_classic(_bt, is_df_roll, f"{window_no:02d}", write_header=_hdr)

        dfo          = df.iloc[cur_start:cur_end].reset_index(drop=True)
        rets_oos, rb_rets_window, eq_is_window, rb_eq_is_window = _run_wfo_window(
            is_df_roll, dfo, lb_roll,
            window_tag=f"W{window_no:02d}",
            regimes_is=None,
            regimes_oos=None,
            rb_scenarios=rb_scenarios,
            export_is=(window_no == 1)  # only export IS for first WFO window
        )
        if eq_is_first is None:
            eq_is_first = eq_is_window
            for label, eq_is_rb in rb_eq_is_window.items():
                rb_eq_seed[label] = eq_is_rb
        all_oos_rets.extend(rets_oos)
        for label in rb_totals:
            rb_totals[label].extend(rb_rets_window.get(label, []))

        cur_start = cur_end
        window_no += 1

    eq_seed = eq_is_first if eq_is_first is not None else eq_is_baseline
    all_oos_rets = np.array(all_oos_rets, dtype=float)
    eq_wfo = np.concatenate([eq_seed,
                             eq_seed[-1] + np.cumsum(all_oos_rets)])

    rb_eq_curves = {}
    if rb_totals:
        for label, vals in rb_totals.items():
            if not vals:
                continue
            arr = np.array(vals, dtype=float)
            seed_rb = rb_eq_seed.get(label, eq_seed)
            rb_eq_curves[label] = np.concatenate([seed_rb,
                                                  seed_rb[-1] + np.cumsum(arr)])

    split_wfo_is = len(eq_seed) - 1
    return all_oos_rets, eq_wfo, rb_eq_curves, split_wfo_is

def inject_news_candles(df: pd.DataFrame, seed: int | None = None) -> pd.DataFrame:
    """Return **new** DataFrame where every 5001000 bars a burst of 12 candles
    gets oversized wicks (25 average true range of the previous 100 bars).
    Open/Close stay unchanged; only High/Low are stretched.
    """
    rng = random.Random(seed)
    i = 0
    n = len(df)
    while i < n:
        i += rng.randint(500, 1000)           # skip 5001000 bars
        if i >= n:
            break
        burst = rng.randint(1, 2)             # 12 candles in this burst
        for j in range(burst):
            idx = i + j
            if idx >= n:
                break
            w_start = max(0, idx - 100)
            avg_range = (df['high'].iloc[w_start:idx] - df['low'].iloc[w_start:idx]).abs().mean()
            if math.isnan(avg_range) or avg_range == 0:
                avg_range = (df['high'] - df['low']).abs().mean()
            extent = avg_range * rng.uniform(2, 5)
            direction = rng.choice(['up', 'down', 'both'])
            op   = df.at[idx, 'open']
            cp   = df.at[idx, 'close']
            hi   = df.at[idx, 'high']
            lo   = df.at[idx, 'low']
            top  = max(op, cp, hi)
            bot  = min(op, cp, lo)
            if direction in ('up', 'both'):
                hi_new = top + extent
            else:
                hi_new = top
            if direction in ('down', 'both'):
                lo_new = bot - extent
            else:
                lo_new = bot
            # Safety: ensure logical ordering
            if hi_new < max(op, cp):
                hi_new = max(op, cp)
            if lo_new > min(op, cp):
                lo_new = min(op, cp)
            df.at[idx, 'high'] = hi_new
            df.at[idx, 'low']  = lo_new
        i += burst
    return df

def apply_news_injection(config: Optional[Config] = None):
    """Injects synthetic newswick candles and rebacktests with *fixed* params.

    `config` is optional; see `optimiser` docstring for the contract.
    """
    with with_config(config):
        return _apply_news_injection_impl()


def _apply_news_injection_impl():
    # Was: `global TP_PERCENTAGE, USE_TP`. Replaced with the
    # `globals()['X']=...` pattern used elsewhere in this file so we do
    # not need a `global` keyword. The save/restore semantics are
    # identical: snapshot before the override, restore after.
    if 'df' not in signals_cache:
        raise RuntimeError("Baseline DF not cached  run main() first!")

    df_mod = inject_news_candles(signals_cache['df'].copy())
    N      = len(df_mod)
    is_df  = df_mod.iloc[N - OOS_CANDLES - BACKTEST_CANDLES : N - OOS_CANDLES].reset_index(drop=True)
    oos_df = df_mod.iloc[N - OOS_CANDLES : N].reset_index(drop=True)

    # === Classic optimisation path =======================================
    if signals_cache.get('mode') == 'classic':
        best_lb  = signals_cache['best_lb']
        best_rrr = signals_cache['best_rrr']

        # set fixed RRR (if any)
        old_tp, old_flag = TP_PERCENTAGE, USE_TP
        if best_rrr is not None:
            globals()['TP_PERCENTAGE'] = best_rrr * SL_PERCENTAGE
            globals()['USE_TP']        = True

        # IS
        dfi = compute_indicators(is_df.copy(), best_lb)
        raw = create_raw_signals(dfi, best_lb)
        raw = filter_raw_signals(raw, None)
        sig = parse_signals(raw, dfi['time'])
        _, met_is_rb, eq_is_rb, _, _ = backtest(dfi, sig)

        # OOS
        dfo = compute_indicators(oos_df.copy(), best_lb)
        raw = create_raw_signals(dfo, best_lb)
        raw = filter_raw_signals(raw, None)
        sig = parse_signals(raw, dfo['time'])
        _, met_oos_rb, eq_oos_rb, _, _ = backtest(dfo, sig)

        eq_baseline_rb = np.concatenate((eq_is_rb, eq_oos_rb + (eq_is_rb[-1] - 1)))

        # restore TP settings
        globals()['TP_PERCENTAGE'] = old_tp
        globals()['USE_TP'] = old_flag

    # === Regimesegmentation path ========================================
    else:
        best_lbs = signals_cache['best_lbs']
        # IS
        _, met_is_rb, eq_is_rb, _   = backtest_continuous_regime(is_df.copy(),  best_lbs)
        # OOS
        _, met_oos_rb, eq_oos_rb, _ = backtest_continuous_regime(oos_df.copy(), best_lbs)
        eq_baseline_rb = np.concatenate((eq_is_rb, eq_oos_rb + (eq_is_rb[-1] - 1)))

    return {
        'met_is_rb':      met_is_rb,
        'met_oos_rb':     met_oos_rb,
        'eq_baseline_rb': eq_baseline_rb
    }

def drift_entries(sig: np.ndarray) -> np.ndarray:
    """Return a copy where codes 1 & 3 are shifted to the *next* bar.
    Pure exits (2, 4) stay in place.  If an entry falls on the last bar
    it is simply dropped."""
    out = np.zeros_like(sig)
    for i, code in enumerate(sig):
        if code in (1, 3):                 # entries that also auto-exit the
            if i + 1 < len(sig):           # opposite side get drifted too
                out[i + 1] = code
        elif code in (2, 4):
            out[i] = code
    return out


def apply_combined_robustness(
        fee_mult=1,
        slip_mult=1,
        inject_news=False,
        drift=False,
        variance=False):
    """Layer *all* enabled robustness tweaks in one pass.

    Parameters
    ----------
    fee_mult, slip_mult : int or float
        Multipliers for commission and slippage.
    inject_news : bool
        If True add synthetic newswick candles.
    drift : bool
        If True delay every entry signal by one bar.
    variance : bool
        If True perturb the chosen slowEMA lookbacks by 1.
    """
    # Was: `global FEE_PCT, SLIPPAGE_PCT, TP_PERCENTAGE, USE_TP`. Replaced
    # with `globals()['X']=...` so we do not need a `global` keyword.
    # Save/restore semantics unchanged.

    # 1) build working price series ---------------------------------------
    df_work = signals_cache['df'].copy()
    if inject_news:
        df_work = inject_news_candles(df_work)

    # split IS / OOS --------------------------------------------------------
    N      = len(df_work)
    is_df  = df_work.iloc[N - BACKTEST_CANDLES - OOS_CANDLES : N - OOS_CANDLES].reset_index(drop=True)
    oos_df = df_work.iloc[N - OOS_CANDLES : N].reset_index(drop=True)

    # 2) fee / slippage shock ---------------------------------------------
    fee_old, slip_old = FEE_PCT, SLIPPAGE_PCT
    globals()['FEE_PCT']      = fee_old * fee_mult
    globals()['SLIPPAGE_PCT'] = slip_old * slip_mult

    # 3) choose variant LBs -----------------------------------------------
    # IND_VARIANCE_SEED: seeded to 42 (was unseeded; cross-arch review
    # flagged that the +/- 1 LB perturbation could pick a different
    # optimum LB on different hosts and silently propagate into the
    # printed metric ledger).
    rng = random.Random(42)

    if signals_cache['mode'] == 'classic':
        base_lb   = signals_cache['best_lb']
        best_rrr  = signals_cache['best_rrr']
        lb_use    = base_lb
        if variance:
            offset   = rng.choice([-1, 1])
            lb_use   = max(1, base_lb + offset)

        # apply RRR if any
        tp_old, flag_old = TP_PERCENTAGE, USE_TP
        if best_rrr is not None:
            globals()['TP_PERCENTAGE'] = best_rrr * SL_PERCENTAGE
            globals()['USE_TP']        = True

        # ---------- IS -----------------------------------------------
        dfi = compute_indicators(is_df.copy(), lb_use)
        raw = filter_raw_signals(create_raw_signals(dfi, lb_use), None)
        sig = parse_signals(raw, dfi['time'])
        if drift:
            sig = drift_entries(sig)
        _, met_is_rb, eq_is_rb, _, _ = backtest(dfi, sig)

        # ---------- OOS ----------------------------------------------
        dfo = compute_indicators(oos_df.copy(), lb_use)
        raw = filter_raw_signals(create_raw_signals(dfo, lb_use), None)
        sig = parse_signals(raw, dfo['time'])
        if drift:
            sig = drift_entries(sig)
        _, met_oos_rb, eq_oos_rb, _, _ = backtest(dfo, sig)

        globals()['TP_PERCENTAGE'] = tp_old
        globals()['USE_TP']        = flag_old

    else:  # regime segmentation path
        base_lbs = signals_cache['best_lbs'].copy()
        lbs_use  = base_lbs
        if variance:
            lbs_use = {r: max(1, lb + rng.choice([-1, 1])) for r, lb in base_lbs.items()}

        # IS
        _, met_is_rb, eq_is_rb, _   = backtest_continuous_regime(is_df.copy(),  lbs_use)
        # OOS
        _, met_oos_rb, eq_oos_rb, _ = backtest_continuous_regime(oos_df.copy(), lbs_use)

    # 4) restore globals ----------------------------------------------------
    globals()['FEE_PCT']      = fee_old
    globals()['SLIPPAGE_PCT'] = slip_old

    # 5) stitch equity curve ----------------------------------------------
    eq_baseline_rb = np.concatenate((eq_is_rb, eq_oos_rb + (eq_is_rb[-1] - 1)))

    return {
        'met_is_rb':      met_is_rb,
        'met_oos_rb':     met_oos_rb,
        'eq_baseline_rb': eq_baseline_rb
    }


# === Run robustness tests ===
# === Run robustness tests (all-in-one print & plot) ===
def _normalize_rb_flag(flag: str) -> str:
    return flag.strip().lower().replace(" ", "_")


def _opts_from_flags(flags: tuple[str, ...] | list[str] | None) -> dict[str, object]:
    flags = flags or ()
    tokens = {_normalize_rb_flag(f) for f in flags if isinstance(f, str)}
    fee_on  = any(t in tokens for t in ("fee_shock", "fee", "fees"))
    slip_on = any(t in tokens for t in ("slippage_shock", "slippage", "slip"))
    news_on = any(t in tokens for t in ("news_candles_injection", "news"))
    drift_on= any(t in tokens for t in ("entry_drift", "drift"))
    var_on  = any(t in tokens for t in ("indicator_variance", "variance", "lb_variance"))
    return {
        "fee_mult": 2 if fee_on else 1,
        "slip_mult":3 if slip_on else 1,
        "news_on":  news_on,
        "drift_on": drift_on,
        "var_on":   var_on,
    }


def _label_from_flags(flags: tuple[str, ...] | list[str]) -> str:
    """Build a short label using 3-char abbreviations joined by '+'."""
    norm = [_normalize_rb_flag(f) for f in flags if isinstance(f, str)]
    pretty_map = {
        "fee_shock": "FEE",
        "slippage_shock": "SLI",
        "news_candles_injection": "NWS",
        "entry_drift": "ENT",
        "indicator_variance": "IND",
    }
    parts = []
    for t in norm:
        parts.append(pretty_map.get(t, t[:3].upper()))
    return "+".join(parts) if parts else "NONE"


def run_robustness_tests():
    """Runs one or more robustness scenarios separately (no overlapping toggles)."""

    def _run_single(name: str, opts: dict[str, object]):
        fee_mult  = opts["fee_mult"]
        slip_mult = opts["slip_mult"]
        news_on   = opts["news_on"]
        drift_on  = opts["drift_on"]
        var_on    = opts["var_on"]
        flags_current = ROBUSTNESS_SCENARIOS.get(name, ()) if ROBUSTNESS_SCENARIOS else ()

        if fee_mult == 1 and slip_mult == 1 and not news_on and not drift_on and not var_on:
            return None

        # if we've doubled OOS_CANDLES, shrink it back so robustness only hits OOS1
        if USE_OOS2:
            old_oos = OOS_CANDLES
            globals()['OOS_CANDLES'] = ORIGINAL_OOS
        else:
            old_oos = None

        parts = []
        if fee_mult > 1: parts.append('Fee')
        if slip_mult>1:  parts.append('Slippage')
        if news_on:      parts.append('News')
        if drift_on:     parts.append('Drift')
        if var_on:       parts.append('LB')
        label = _label_from_flags(flags_current if flags_current else parts)
        verbose_name = name if name else (' + '.join(parts) + ' Robustness')

        print(f"\n Robustness Test: {label} ({verbose_name}) ")
        res = apply_combined_robustness(
            fee_mult, slip_mult,
            inject_news=news_on,
            drift=drift_on,
            variance=var_on
        )

        # restore full OOS window for plotting & WFO
        if old_oos is not None:
            globals()['OOS_CANDLES'] = old_oos

        prettyprint(f"{label} IS",  res['met_is_rb'])
        prettyprint(f"{label} OOS1", res['met_oos_rb'])
        return label, res

    # decide which scenarios to run: custom queue first, else fall back to legacy combined toggle
    if ROBUSTNESS_SCENARIOS:
        items = list(ROBUSTNESS_SCENARIOS.items())[:MAX_ROBUSTNESS_SCENARIOS]
    else:
        default_flags = []
        if FEE_SHOCK:              default_flags.append("fee_shock")
        if SLIPPAGE_SHOCK:         default_flags.append("slippage_shock")
        if NEWS_CANDLES_INJECTION: default_flags.append("news_candles_injection")
        if ENTRY_DRIFT:            default_flags.append("entry_drift")
        if INDICATOR_VARIANCE:     default_flags.append("indicator_variance")
        items = [((' + '.join(f.title().replace('_', ' ') for f in default_flags) + ' Robustness') if default_flags else "", tuple(default_flags))]

    results = {}
    for name, flags in items:
        label_res = _run_single(str(name), _opts_from_flags(flags))
        if label_res is None:
            continue
        label, res = label_res
        results[label] = res

    return results


AGE_DATASET  = 0

def age_dataset(df, age):
    """
    Remove the last `age` rows (most recent candles) from df.
    """
    if age <= 0:
        return df
    if age >= len(df):
        raise ValueError(f"AGE_DATASET ({age}) >= dataset length ({len(df)})")
    # drop last `age` rows and reset index
    return df.iloc[:-age].reset_index(drop=True)


# 12. MAIN
def main(config: Optional[Config] = None):
    """
    Top-level entry point. Runs the full backtest pipeline:
    classic single run -> robustness -> walk-forward -> plot.

    `config` is optional. When provided, applies that Config to module
    globals for the duration of this call (and restores prior values on
    exit). Pass `bt.Config()` to use library defaults regardless of the
    current module state, or `bt.Config.from_module()` to snapshot the
    current state and tweak fields. When omitted, reads from module
    globals — the legacy `bt.X = Y` API works exactly as before.
    """
    with with_config(config):
        return _main_impl()


def _main_impl():
    # Explicit early check so users running `python -m backtester` get a
    # clear error before any heavy imports / numba JIT warm-up happens.
    if not os.path.exists(CSV_FILE):
        raise FileNotFoundError(
            f"CSV file not found: {CSV_FILE}\n\n"
            "Put your OHLC CSV at that path, or change CSV_FILE / set BT_CSV.\n"
            "You can generate one with binance_ohlc_downloader.py (see README)."
        )
    df = load_ohlc(CSV_FILE)

    df = age_dataset(df, AGE_DATASET)

    df['is_traded'] = df['time'].apply(lambda ts: in_session(ts) if TRADE_SESSIONS else True)

    # item #3: clear any stale side-channel from a prior main() call in the
    # same process. On the regime-no-WFO path optimiser() is never called,
    # so without this the report could surface a phantom trial vector from
    # an unrelated earlier run (Lens A3). An empty capture then honestly
    # reports N=0 instead of stale-N.
    if OVERFIT_REPORT:
        _runtime_state.pop('_last_trial_sharpes', None)

    # 1) baseline run
    base = classic_single_run(df)
    # item #3: snapshot the baseline IS optimisation's distinct trial
    # Sharpes NOW, before the WFO loop's per-window optimiser() calls
    # overwrite the side-channel. Canonical "strategies tried" set.
    _overfit_trials = list(_runtime_state.get('_last_trial_sharpes', [])) \
        if OVERFIT_REPORT else []
    signals_cache['df'] = df.copy()

    # 1.a) print baseline metrics
    print(" Baseline Optimized Metrics ")
    # --- Classic optimization path ---
    if base.get('met_is_opt') is not None:
        prettyprint('Baseline IS', base['met_is_opt'])
        # two-window?
        if base.get('met_oos1_opt') is not None:
            prettyprint('Baseline OOS1', base['met_oos1_opt'])
            prettyprint('Baseline OOS2', base['met_oos2_opt'])
        else:
            prettyprint('Baseline OOS', base['met_oos_opt'])

    # --- Regime segmentation path ---
    elif base.get('met_is') is not None:
        # met_is holds the IS-regime metrics
        prettyprint('Baseline IS-Regime', base['met_is'])
        # two-window?
        if base.get('met_oos1_reg') is not None:
            prettyprint('Baseline OOS1-Regime', base['met_oos1_reg'])
            prettyprint('Baseline OOS2-Regime', base['met_oos2_reg'])
        else:
            prettyprint('Baseline OOS-Regime', base['met_oos_reg'])


    # 2) robustness tests
    rb = run_robustness_tests()

    # 3) plotting
    if USE_WFO:
        print(" Running Walk-Forward Windows ")
        oos_rets, eq_wfo, rb_eq_wfo, split_wfo_is = walk_forward(df, base['met_is'], base['eq_is'])
        # item #3: opt-in overfitting diagnostics. Additive lines only; none
        # carry the LINE_RE metric body, so parity harnesses are unaffected.
        # trial count = distinct baseline strategies (NOT windows*combos). The
        # chosen Sharpe is recomputed FROM oos_rets inside emit() in the
        # Bailey-LdP sqrt(T)*mean/std convention (NOT met_is).
        if OVERFIT_REPORT:
            from backtester import overfit_report
            overfit_report.emit(
                trial_sharpes=_overfit_trials,
                oos_returns=oos_rets,
                sharpe_mode=SHARPE_MODE,
                equity_matrix=_runtime_state.get('_overfit_equity_matrix'),
            )
        # Plot baseline and WFO + robustness
        if PRINT_EQUITY_CURVE:
            plt.figure(figsize=(10,5))
            # WFO equity only
            plt.plot(eq_wfo * ACCOUNT_SIZE, label='WFO Rolling')
            # robustness scenarios (on the WFO rolling curve)
            for name, eq_rb in rb_eq_wfo.items():
                plt.plot(eq_rb * ACCOUNT_SIZE, label=name)
            # split at the end of the first WFO IS segment (length of eq_is passed to walk_forward)
            plt.axvline(split_wfo_is, color='red', linestyle='--', label='End IS (W1)')
            plt.title('Equity Curve: WFO & Robustness')
            plt.legend(); plt.grid(True); plt.tight_layout(); plt.show()
    else:
    # single-run plotting
        if PRINT_EQUITY_CURVE:
            plt.figure(figsize=(10,5))

            #  pick the right equity curve 
            if USE_REGIME_SEG and base.get('eq_baseline') is not None:
                # regimesegmented: always use the unified array we built
                eq_all = base['eq_baseline']
            else:
                # fallback for classic or any path without unified baseline
                eq_all = base.get('eq_baseline')
                if eq_all is None:
                    # legacy stitch
                    eq_all = np.concatenate((base['eq_baseline_is'], base['eq_baseline_oos']))

           
           
            plt.plot(eq_all * ACCOUNT_SIZE, label='Baseline Equity')

            # 2) robustness overlays
            for name, res in rb.items():
                eq_rb = res['eq_baseline_rb']
                if base.get('split2') is not None:
                    # splice OOS2 from the baseline returns onto the end of the robustness-tested OOS1
                    split2 = base['split2']
                    eq_base_split2 = eq_all[split2]
                    baseline_tail = eq_all[split2+1:]
                    rel_changes = baseline_tail - eq_base_split2
                    tail = rel_changes + eq_rb[-1]
                    eq_rb_plot = np.concatenate((eq_rb, tail))
                else:
                    eq_rb_plot = eq_rb

                plt.plot(eq_rb_plot * ACCOUNT_SIZE, label=name)

            # 3) vertical lines at trade-based splits
            split1 = base['split1']
            plt.axvline(split1, color='red', linestyle='--', label='End IS')

            split2 = base.get('split2')
            if split2 is not None:
                plt.axvline(split2, color='red', linestyle='--', label='End OOS1')

            # 4) finalize
            plt.title('Equity Curve with Robustness Tests')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.show()


if __name__ == '__main__':
    main()

