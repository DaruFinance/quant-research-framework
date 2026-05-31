"""Panel data loader (item #1, Phase 2).

``load_panel`` reads N per-asset OHLC[+V] CSVs, inner-joins them on the
``time`` column, and returns an ``xarray.Dataset`` keyed by
``(time, asset)`` with one data variable per OHLC field. Strict
contract:

- Every asset's CSV must have a ``time`` column whose values are UNIX
  seconds (UTC).
- The columns ``open``, ``high``, ``low``, ``close`` are required;
  ``volume`` is included if present in every asset's CSV.
- Inner-join across all assets; the resulting time grid must be
  **uniformly spaced** (no gaps). The expected interval is inferred
  from the most-common delta in the intersection; any deviation
  raises ``PanelGapError`` naming the first offending timestamp.
- Asset symbols are taken from the input dict's keys; ordering is
  preserved.

The loader is pure data-only: it reads CSVs, computes the intersection,
and emits a Dataset. No time-dependent decisions, no forward-looking
inputs — trivially lookahead-free.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping

import numpy as np
import pandas as pd
import xarray as xr


REQUIRED_FIELDS = ("open", "high", "low", "close")
OPTIONAL_FIELDS = ("volume",)


class PanelSchemaError(ValueError):
    """A source CSV is missing required columns or has the wrong dtype."""


class PanelGapError(ValueError):
    """The inner-join produced a non-uniform time grid (gap inside the
    requested window). The offending timestamp is in ``args[1]``."""

    def __init__(self, message: str, ts: int) -> None:
        super().__init__(message, ts)
        self.ts = ts


@dataclass(frozen=True)
class PanelData:
    """Lightweight handle around the loaded ``xarray.Dataset``.

    Holding it in a wrapper rather than returning a bare Dataset gives
    downstream items (#4 cross-asset regime, #5(iter) basket
    orchestrator) a stable type to dispatch on without binding the
    whole stack to a single xarray version.
    """
    ds: xr.Dataset

    @property
    def assets(self) -> List[str]:
        return [str(a) for a in self.ds["asset"].values]

    @property
    def times(self) -> np.ndarray:
        return self.ds["time"].values

    @property
    def fields(self) -> List[str]:
        return [str(v) for v in self.ds.data_vars]

    def __len__(self) -> int:
        return int(self.ds.sizes["time"])


def _read_one(path: Path, asset: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [c for c in ("time",) + REQUIRED_FIELDS if c not in df.columns]
    if missing:
        raise PanelSchemaError(
            f"asset {asset!r} ({path}) missing columns: {missing}; "
            f"got {list(df.columns)}"
        )
    df = df[["time"] + list(REQUIRED_FIELDS)
            + [c for c in OPTIONAL_FIELDS if c in df.columns]]
    if not np.issubdtype(df["time"].dtype, np.integer):
        # Accept floats that round to int (some downloaders emit float).
        if not (df["time"] % 1 == 0).all():
            raise PanelSchemaError(
                f"asset {asset!r} ({path}): time column must be integer "
                f"UNIX seconds; got dtype {df['time'].dtype}"
            )
        df["time"] = df["time"].astype(np.int64)
    df = df.sort_values("time").reset_index(drop=True)
    if df["time"].duplicated().any():
        first_dup = int(df.loc[df["time"].duplicated(), "time"].iloc[0])
        raise PanelSchemaError(
            f"asset {asset!r} ({path}): duplicate timestamp at "
            f"ts={first_dup}"
        )
    return df


def _infer_interval(times: np.ndarray) -> int:
    """Return the most common gap between consecutive timestamps. If the
    panel is degenerate (≤1 row) we surface a SchemaError; otherwise the
    modal delta defines the expected bar spacing."""
    if len(times) < 2:
        raise PanelSchemaError(
            f"panel must have at least 2 timestamps after inner-join; "
            f"got {len(times)}"
        )
    deltas = np.diff(times)
    counter = Counter(int(d) for d in deltas)
    return counter.most_common(1)[0][0]


def load_panel(paths: Mapping[str, Path]) -> PanelData:
    """Load N CSVs into a single time × asset × field xarray.Dataset.

    Parameters
    ----------
    paths : Mapping[str, Path]
        ``{asset_symbol: csv_path}``. Order preserved in the Dataset's
        ``asset`` coordinate.

    Raises
    ------
    PanelSchemaError
        If any input CSV is malformed (missing columns, non-integer
        time, duplicate timestamps, fewer than 2 rows post-join).
    PanelGapError
        If the inner-join is not uniformly spaced.
    """
    if not paths:
        raise PanelSchemaError("paths is empty")

    frames = {asset: _read_one(Path(p), asset) for asset, p in paths.items()}

    # Inner-join on time.
    common = None
    for asset, df in frames.items():
        s = set(df["time"].tolist())
        common = s if common is None else common & s
    if not common:
        raise PanelSchemaError(
            "inner-join across assets produced an empty set of timestamps"
        )
    times = np.array(sorted(common), dtype=np.int64)

    # Detect gaps in the joined grid.
    interval = _infer_interval(times)
    diffs = np.diff(times)
    bad_idx = np.where(diffs != interval)[0]
    if len(bad_idx) > 0:
        i = int(bad_idx[0])
        raise PanelGapError(
            f"panel grid has a gap at index {i}: ts {int(times[i])} -> "
            f"{int(times[i + 1])} = {int(diffs[i])}s "
            f"(expected {interval}s, inferred from modal delta)",
            ts=int(times[i + 1]),
        )

    # Decide which OHLC[+V] fields exist in EVERY input. Volume only
    # makes it in if every asset has it; otherwise drop it (no
    # half-populated columns).
    have_volume = all("volume" in df.columns for df in frames.values())
    fields = list(REQUIRED_FIELDS) + (["volume"] if have_volume else [])

    asset_list = list(paths.keys())
    n_t, n_a, n_f = len(times), len(asset_list), len(fields)
    arr = np.empty((n_t, n_a, n_f), dtype=np.float64)

    for ai, asset in enumerate(asset_list):
        df = frames[asset]
        sub = df[df["time"].isin(times)].sort_values("time").reset_index(drop=True)
        # `times` is sorted; sub should align row-by-row. Defensive check:
        if len(sub) != n_t or not (sub["time"].values == times).all():
            raise PanelSchemaError(
                f"asset {asset!r}: post-filter alignment failed; "
                f"got {len(sub)} rows, expected {n_t}"
            )
        for fi, f in enumerate(fields):
            arr[:, ai, fi] = sub[f].astype(np.float64).values

    data_vars = {
        f: (("time", "asset"), arr[:, :, fi]) for fi, f in enumerate(fields)
    }
    ds = xr.Dataset(
        data_vars=data_vars,
        coords={"time": times, "asset": asset_list},
        attrs={"interval_seconds": interval, "n_assets": n_a, "n_bars": n_t},
    )
    return PanelData(ds=ds)
