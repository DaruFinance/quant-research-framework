"""Open-interest stream loader (item #40, Phase 3).

OI feeds typically come at fixed cadence (Binance: 5m or 1h).  The
loader normalises columns, asserts the cadence is monotonic and
uniform within a configurable tolerance, and exposes a point-in-time
look-up that never returns a value from after ``t_s``.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class OIRecord:
    time_s: int
    open_interest: float
    open_interest_usd: Optional[float]


def load_oi(
    path: Union[str, Path],
    *,
    expected_cadence_s: int = 3600,
    cadence_tol_s: int = 60,
) -> pd.DataFrame:
    """Load an OI feed.  Verifies the inter-row spacing is uniform
    within ``cadence_tol_s`` of ``expected_cadence_s`` to catch a
    misread cadence early; the BTC perp 1h fixture allows up to 60s
    of jitter from the venue's emit-time variance.
    """
    path = Path(path)
    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
    elif path.suffix == ".csv":
        df = pd.read_csv(path)
    else:
        raise ValueError(f"unsupported OI feed extension: {path.suffix}")

    if "time" not in df.columns or "open_interest" not in df.columns:
        raise ValueError("OI feed needs columns time, open_interest")
    df = df.sort_values("time").reset_index(drop=True)

    if len(df) >= 2:
        deltas = np.diff(df["time"].values)
        worst = int(np.max(np.abs(deltas - expected_cadence_s)))
        if worst > cadence_tol_s:
            bad = int(np.argmax(np.abs(deltas - expected_cadence_s)))
            raise ValueError(
                f"OI cadence at row {bad} = {int(deltas[bad])}s, expected "
                f"{expected_cadence_s}s ± {cadence_tol_s}s"
            )

    if df["open_interest"].isna().any():
        raise ValueError("OI feed has NaN open_interest")

    df.attrs["expected_cadence_s"] = expected_cadence_s
    return df


def oi_at(
    df: pd.DataFrame,
    t_s: int,
) -> Optional[OIRecord]:
    """Most-recent OI snapshot at or before ``t_s``."""
    mask = df["time"].values <= t_s
    if not mask.any():
        return None
    idx = int(np.flatnonzero(mask)[-1])
    row = df.iloc[idx]
    return OIRecord(
        time_s=int(row["time"]),
        open_interest=float(row["open_interest"]),
        open_interest_usd=(float(row["open_interest_usd"])
                             if "open_interest_usd" in df.columns
                             else None),
    )
