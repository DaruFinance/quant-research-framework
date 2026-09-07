"""Shared contract for the external execution-workload adapters.

This is intentionally not the QRF walk-forward workload. Each process loads
the same real OHLC file and frozen causal event arrays outside the engine
timer, then times ten full-history execution passes.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

for _name in (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_name] = "1"

import numpy as np
import pandas as pd


QRF_PY_REPO = Path(os.environ.get("QRF_PY_DIR", Path(__file__).resolve().parents[2]))
DEFAULT_CSV = Path(
    os.environ.get("QRF_BENCH_CSV", QRF_PY_REPO / "data" / "BTCUSDT_30m_150k.csv")
)
COMBINED_CHARGE_RATE = 0.0007  # 5 bp taker fee + 2 bp slippage proxy, per fill
INIT_CASH = 1_000_000_000.0


@dataclass(frozen=True)
class Spec:
    name: str
    function_name: str
    lookback: int


SPECS = (
    Spec("ema_cross_lb14_tp2.0", "signal_ema_cross", 14),
    Spec("ema_cross_lb40_tp4.5", "signal_ema_cross", 40),
    Spec("atr_cross_lb20_tp1.6", "signal_atr_cross", 20),
    Spec("atr_cross_lb50_tp2.0", "signal_atr_cross", 50),
    Spec("macd_zero_lb12_tp3.0", "signal_macd_zero", 12),
    Spec("macd_zero_lb26_tp3.0", "signal_macd_zero", 26),
    Spec("rsi_revert_lb14_tp0.5", "signal_rsi_revert", 14),
    Spec("rsi_revert_lb28_tp2.0", "signal_rsi_revert", 28),
    Spec("stoch_kd_lb14_tp0.8", "signal_stoch_kd", 14),
    Spec("stoch_kd_lb21_tp1.2", "signal_stoch_kd", 21),
)

LEDGER_COLUMNS = (
    "strategy",
    "trade_id",
    "side",
    "entry_idx",
    "exit_idx",
    "entry_price",
    "exit_price",
    "quantity",
    "gross_pnl",
    "execution_charge",
    "net_pnl",
)


def load_bars(csv_path: Path, bars: int | None = None) -> pd.DataFrame:
    df = pd.read_csv(csv_path, nrows=bars)
    required = {"time", "open", "high", "low", "close"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"missing OHLC columns: {sorted(missing)}")
    df = df.loc[:, ["time", "open", "high", "low", "close"]].copy()
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    if len(df) < 3:
        raise ValueError("at least three real bars are required")
    if bars is None and len(df) != 150_000:
        raise ValueError(f"formal workload requires exactly 150,000 bars, got {len(df)}")
    return df


def _load_signal_library():
    sys.path.insert(0, str(QRF_PY_REPO))
    path = QRF_PY_REPO / "examples" / "batch_runner" / "run_batch.py"
    spec = importlib.util.spec_from_file_location("qrf_competitor_signal_library", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load signal library: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def qrf_event_codes(shifted_raw: np.ndarray) -> np.ndarray:
    """Encode the approved long-only interpretation; raw zero means hold.

    Signal functions already shift one bar, so code at i is executable at
    open[i].  The first bar is forbidden because no prior bar exists.  A
    possible flip on the last bar is suppressed and replaced with an explicit
    close there, giving every adapter the same final-open liquidation.
    """
    raw = np.asarray(shifted_raw, dtype=np.int8)
    if not np.isin(raw, (-1, 0, 1)).all():
        raise ValueError("raw signal contains a value outside {-1, 0, 1}")
    codes = np.zeros(raw.size, dtype=np.int8)
    is_long = False
    for i in range(1, raw.size - 1):
        value = int(raw[i])
        if value == 1 and not is_long:
            codes[i] = 1
            is_long = True
        elif value == -1 and is_long:
            codes[i] = 2
            is_long = False
    if is_long:
        codes[-1] = 2
    return codes


def build_event_sets(df: pd.DataFrame) -> list[tuple[Spec, np.ndarray]]:
    library = _load_signal_library()
    out: list[tuple[Spec, np.ndarray]] = []
    for spec in SPECS:
        signal_fn = getattr(library, spec.function_name)
        shifted_raw = np.asarray(signal_fn(df, spec.lookback), dtype=np.int8)
        if shifted_raw.shape != (len(df),):
            raise ValueError(f"{spec.name}: wrong signal shape {shifted_raw.shape}")
        codes = qrf_event_codes(shifted_raw)
        # Backtesting.py begins Strategy.next() at bar 1, so the earliest
        # representable next-open event is bar 2.  These ten causal indicators
        # satisfy that boundary; fail loudly if a future signal does not.
        if codes[0] != 0 or codes[1] != 0 or codes[-1] not in (0, 2):
            raise AssertionError(f"{spec.name}: event boundary invariant failed")
        out.append((spec, codes))
    return out


def array_sha256(values: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(values, dtype=np.int8).tobytes()).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_event_sets(path: Path, df: pd.DataFrame) -> list[tuple[Spec, np.ndarray]]:
    events = pd.read_csv(path)
    expected_columns = ["time", *(spec.name for spec in SPECS)]
    if list(events.columns) != expected_columns:
        raise ValueError(f"event columns differ: {list(events.columns)}")
    if len(events) != len(df):
        raise ValueError(f"event rows differ from OHLC rows: {len(events)} != {len(df)}")
    expected_time = (pd.DatetimeIndex(df["time"]).asi8 // 1_000_000_000).astype(np.int64)
    actual_time = events["time"].to_numpy(dtype=np.int64)
    if not np.array_equal(actual_time, expected_time):
        raise ValueError("event timestamps do not match OHLC timestamps")
    out = []
    for spec in SPECS:
        codes = events[spec.name].to_numpy(dtype=np.int8)
        if not np.isin(codes, (0, 1, 2)).all():
            raise ValueError(f"{spec.name}: event code outside {{0, 1, 2}}")
        if codes[0] != 0 or codes[1] != 0 or codes[-1] not in (0, 2):
            raise ValueError(f"{spec.name}: event boundary invariant failed")
        is_long = False
        for code in codes:
            if code == 1:
                if is_long:
                    raise ValueError(f"{spec.name}: repeated entry")
                is_long = True
            elif code == 2:
                if not is_long:
                    raise ValueError(f"{spec.name}: close while flat")
                is_long = False
        if is_long:
            raise ValueError(f"{spec.name}: unclosed final position")
        out.append((spec, codes))
    return out


def validate_ledger(frame: pd.DataFrame, strategy: str) -> pd.DataFrame:
    frame = frame.loc[:, LEDGER_COLUMNS].copy()
    if len(frame):
        if not (frame["strategy"] == strategy).all():
            raise AssertionError(f"{strategy}: ledger strategy mismatch")
        if not (frame["entry_idx"] <= frame["exit_idx"]).all():
            raise AssertionError(f"{strategy}: exit precedes entry")
        expected = frame["gross_pnl"] - frame["execution_charge"]
        if not np.allclose(frame["net_pnl"], expected, rtol=1e-10, atol=1e-8):
            raise AssertionError(f"{strategy}: gross - charge != net")
    return frame


def write_outputs(
    *,
    out_dir: Path,
    engine: str,
    csv_path: Path,
    event_path: Path,
    df: pd.DataFrame,
    event_sets: list[tuple[Spec, np.ndarray]],
    ledgers: list[pd.DataFrame],
    engine_seconds: list[float],
    versions: dict[str, str],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=False)
    ledger_dir = out_dir / "ledgers"
    ledger_dir.mkdir()
    results = []
    for (spec, codes), ledger, elapsed in zip(event_sets, ledgers, engine_seconds):
        ledger = validate_ledger(ledger, spec.name)
        path = ledger_dir / f"{spec.name}.csv"
        ledger.to_csv(path, index=False, lineterminator="\n")
        results.append(
            {
                "name": spec.name,
                "lookback": spec.lookback,
                "event_sha256": array_sha256(codes),
                "events": int(np.count_nonzero(codes)),
                "trades": int(len(ledger)),
                "engine_elapsed_s": elapsed,
                "ledger": str(path),
                "ledger_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "gross_pnl": float(ledger["gross_pnl"].sum()),
                "execution_charge": float(ledger["execution_charge"].sum()),
                "net_pnl": float(ledger["net_pnl"].sum()),
            }
        )
    metadata = {
        "engine": engine,
        "versions": versions,
        "bars": len(df),
        "is_smoke": len(df) != 150_000,
        "csv": str(csv_path),
        "events": str(event_path),
        "events_file_sha256": file_sha256(event_path),
        "strategies": len(event_sets),
        "engine_elapsed_total_s": float(sum(engine_seconds)),
        "signal_semantics": "long-only: positive enters, negative closes, zero holds; causal next-bar-open events",
        "boundary": "fresh process; one OHLC load and one frozen-event CSV load; ten serial full-history engine calls; ledger normalization outside engine timer",
        "finalization": "suppress final-bar flip and explicitly close existing position at final-bar open",
        "combined_execution_charge_per_fill": COMBINED_CHARGE_RATE,
        "combined_execution_charge_note": "5 bp taker fee + 2 bp slippage proxy; zero price offset; not a price-slippage model",
        "features_disabled": ["walk_forward", "optimization", "stop_loss", "take_profit", "funding"],
        "strategy_name_note": "names are retained from the audited 10-config batch for traceability; embedded tp labels are not active in this workload",
        "results": results,
    }
    (out_dir / "results.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
