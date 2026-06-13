"""Funding-rate stream loader (item #38, Phase 3).

Loads a perpetual-futures funding-rate series into an in-memory
representation aligned to the bar-clock at downstream consumption
time. Critical property: a funding record's *payable* timestamp
matches the venue's settlement boundary (Binance perp: 00:00, 08:00,
16:00 UTC; deltas of 8h), and the loader must not accidentally pull
forward an "announcement" or "predicted-next" value into the row of
the current event.

The bundled DS-FUNDING-200 fixture stores 200 8h Binance events
(~67 days) with the schema ``(time:int64 unix-sec, rate:float64)``.
Future feeds with extra columns are tolerated as long as those two
exist.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Union

import numpy as np
import pandas as pd


# Funding settlement boundary for the perp products we support, in
# seconds.  All Binance / Bybit / OKX perps clear at 8h.  If a venue
# with a different cadence ever lands, parametrise this rather than
# hard-coding a second value.
FUNDING_INTERVAL_S = 8 * 3600


@dataclass(frozen=True)
class FundingEvent:
    time_s: int          # unix epoch second, payable timestamp
    rate: float          # signed funding rate (fraction, not bp)


def load_funding(
    path: Union[str, Path],
    *,
    venue: str = "binance_perp",
    strict_boundary: bool = True,
) -> pd.DataFrame:
    """Load a funding stream, returning a tidy DataFrame indexed by
    payable timestamp.

    - ``strict_boundary``: when True, every event timestamp must align
      to the venue's settlement boundary (modulo zero against
      ``FUNDING_INTERVAL_S``).  Raises with the first offender index
      if it doesn't, so a downstream backtest never silently uses a
      midnight-aligned series misread from an 8h venue (or vice versa).
    - No NaN ``rate`` values pass through; the loader raises on the
      first NaN.
    - The returned DataFrame is sorted by time ascending and de-duped
      on the timestamp column (last value wins for a duplicate, with
      a warning surfaced via the returned DataFrame attribute
      ``attrs['dup_count']``).
    """
    path = Path(path)
    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
    elif path.suffix == ".csv":
        df = pd.read_csv(path)
    else:
        raise ValueError(f"unsupported funding feed extension: {path.suffix}")

    missing = {"time", "rate"} - set(df.columns)
    if missing:
        raise ValueError(f"funding feed missing columns: {missing}")

    if df["rate"].isna().any():
        bad = df.index[df["rate"].isna()].tolist()[:5]
        raise ValueError(f"funding feed has NaN rate(s) at rows {bad}")

    dup_count = int(df["time"].duplicated().sum())
    if dup_count:
        df = df.drop_duplicates(subset=["time"], keep="last")
    df = df.sort_values("time").reset_index(drop=True)

    if strict_boundary:
        off = df.loc[df["time"] % FUNDING_INTERVAL_S != 0]
        if len(off):
            row = off.iloc[0]
            raise ValueError(
                f"funding row {off.index[0]} time={int(row['time'])} not "
                f"aligned to {FUNDING_INTERVAL_S}s {venue} boundary"
            )

    df.attrs["venue"] = venue
    df.attrs["dup_count"] = dup_count
    return df


def next_funding_time(
    t_s: int,
    *,
    interval_s: int = FUNDING_INTERVAL_S,
) -> int:
    """Smallest funding settlement timestamp ``>= t_s``.  Used by the
    scheduler (#42) to align next-rebalance with the next payable
    event without consulting any future state."""
    rem = t_s % interval_s
    return t_s if rem == 0 else t_s + (interval_s - rem)


def iter_events(df: pd.DataFrame) -> Iterable[FundingEvent]:
    """Yield FundingEvent records.  Convenience for code paths that
    prefer a streaming form over a DataFrame."""
    for _, row in df.iterrows():
        yield FundingEvent(time_s=int(row["time"]), rate=float(row["rate"]))


def rate_at(
    df: pd.DataFrame,
    t_s: int,
    *,
    fill: str = "ffill",
) -> Optional[float]:
    """Return the most recent funding rate at or before ``t_s``.

    Critical leak-free property: never returns a rate from an event
    with ``time_s > t_s``.  ``fill='strict'`` raises if no record at
    or before ``t_s`` exists.  ``fill='ffill'`` returns None when no
    record exists yet (e.g., at backtest start).
    """
    mask = df["time"].values <= t_s
    if not mask.any():
        if fill == "strict":
            raise LookupError(f"no funding rate at or before {t_s}")
        return None
    idx = int(np.flatnonzero(mask)[-1])
    return float(df["rate"].iloc[idx])
