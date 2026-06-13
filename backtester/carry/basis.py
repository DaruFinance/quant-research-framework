"""Perp-vs-spot and calendar basis loader (item #39, Phase 3).

Basis at logical time ``t`` is computed strictly from contemporaneous
perp and spot quotes available at ``t``.  No look-ahead from later
quotes is permitted; the loader rejects sources whose computed basis
is shifted (e.g., trailing window applied at publish time) by
comparing the input file's ``basis_bp`` column (if present) against
a fresh recompute and surfacing any divergence > 0.01bp.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class BasisRecord:
    time_s: int
    close_spot: float
    close_perp: float
    basis_bp: float


def _basis_bp(perp: float, spot: float) -> float:
    if spot == 0.0:
        return float("nan")
    return (perp - spot) / spot * 1e4


def load_basis(
    path: Union[str, Path],
    *,
    instrument_pair: str = "btc_perp_spot",
    recompute_basis_tol_bp: float = 0.01,
) -> pd.DataFrame:
    """Load a basis series.  Columns expected: ``time, close_spot,
    close_perp`` (and optionally a pre-computed ``basis_bp``).

    If ``basis_bp`` is present in the input, the loader recomputes
    from ``close_perp / close_spot`` and verifies the columns agree
    to ``recompute_basis_tol_bp``.  This catches sources that ship a
    smoothed / forward-looking basis column without flagging it.
    """
    path = Path(path)
    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
    elif path.suffix == ".csv":
        df = pd.read_csv(path)
    else:
        raise ValueError(f"unsupported basis feed extension: {path.suffix}")

    missing = {"time", "close_spot", "close_perp"} - set(df.columns)
    if missing:
        raise ValueError(f"basis feed missing columns: {missing}")

    fresh = np.array([_basis_bp(p, s) for p, s in zip(df["close_perp"],
                                                       df["close_spot"])])
    if "basis_bp" in df.columns:
        delta = np.abs(df["basis_bp"].values - fresh)
        worst = float(np.nanmax(delta)) if len(delta) else 0.0
        if worst > recompute_basis_tol_bp:
            bad = int(np.nanargmax(delta))
            raise ValueError(
                f"basis_bp column drifted from fresh recompute by "
                f"{worst:.4f}bp at row {bad}; possible forward-looking "
                f"smoothing in source feed"
            )
    df = df.copy()
    df["basis_bp"] = fresh
    df = df.sort_values("time").reset_index(drop=True)
    df.attrs["instrument_pair"] = instrument_pair
    return df


def basis_at(
    df: pd.DataFrame,
    t_s: int,
) -> Optional[BasisRecord]:
    """Most-recent basis at or before ``t_s``; never returns a record
    from after ``t_s``."""
    mask = df["time"].values <= t_s
    if not mask.any():
        return None
    idx = int(np.flatnonzero(mask)[-1])
    row = df.iloc[idx]
    return BasisRecord(
        time_s=int(row["time"]),
        close_spot=float(row["close_spot"]),
        close_perp=float(row["close_perp"]),
        basis_bp=float(row["basis_bp"]),
    )
