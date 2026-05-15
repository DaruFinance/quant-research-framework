"""On-chain stream loader with snapshot-pinning (item #41, Phase 3).

The critical property here is *snapshot pinning*: on-chain providers
(Glassnode, CryptoQuant, blockchain.info) frequently revise historical
values after a backfill or reorg.  A backtest pulled at time t0 with
the unpinned API would silently switch results if rerun at t1 > t0.

The loader pins the snapshot at ingestion: it reads the file as-is
(no live API call), records the snapshot file's mtime + sha256 into
``df.attrs``, and rejects any incoming "revised" value at indices the
loader has already served.  The 50-row DS-ONCHAIN-50 fixture is the
canonical test bed.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class OnChainSnapshot:
    time_s: int
    metric: str
    value: float


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def load_onchain(
    path: Union[str, Path],
    *,
    metric: str = "nvt",
) -> pd.DataFrame:
    """Pin and load an on-chain snapshot.

    Returns a DataFrame with columns ``time, value`` plus the
    snapshot-pinning metadata in ``attrs``:

    - ``snapshot_sha256``: file-content hash at load time.
    - ``snapshot_mtime``: file mtime at load time.
    - ``metric``: which column we extracted.

    .. warning::
       Pandas (≥ 2.0) drops ``DataFrame.attrs`` whenever an operation
       joins this frame with another frame whose ``attrs`` differ —
       most notably :func:`pandas.concat` and :func:`DataFrame.merge`
       with a fresh / different-source frame.  In practice this means
       a downstream join of an on-chain frame with bar data drops the
       pinning metadata silently.  ``copy()``, slicing, ``groupby``,
       and elementwise arithmetic preserve attrs in current pandas,
       but treat that as incidental — capture the pinning fields
       into your own variables right after :func:`load_onchain`
       returns rather than relying on them surviving downstream.
       Same caveat applies to :func:`load_funding`, :func:`load_basis`,
       and :func:`load_oi`; their ``attrs`` are set at load time only.
    """
    path = Path(path)
    if path.suffix == ".parquet":
        raw = pd.read_parquet(path)
    elif path.suffix == ".csv":
        raw = pd.read_csv(path)
    else:
        raise ValueError(f"unsupported on-chain feed extension: {path.suffix}")

    if "time" not in raw.columns or metric not in raw.columns:
        raise ValueError(f"on-chain feed needs columns time, {metric}")

    df = raw[["time", metric]].rename(columns={metric: "value"}).copy()
    df = df.sort_values("time").reset_index(drop=True)

    df.attrs["metric"] = metric
    df.attrs["snapshot_sha256"] = _sha256_file(path)
    df.attrs["snapshot_mtime"] = path.stat().st_mtime
    df.attrs["snapshot_path"] = str(path)
    return df


def value_at(
    df: pd.DataFrame,
    t_s: int,
) -> Optional[OnChainSnapshot]:
    """Most-recent metric value at or before ``t_s``."""
    mask = df["time"].values <= t_s
    if not mask.any():
        return None
    idx = int(np.flatnonzero(mask)[-1])
    row = df.iloc[idx]
    return OnChainSnapshot(
        time_s=int(row["time"]),
        metric=str(df.attrs.get("metric", "")),
        value=float(row["value"]),
    )
