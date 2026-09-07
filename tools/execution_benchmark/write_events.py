#!/usr/bin/env python3
"""Freeze the shared causal events once, outside every engine process."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from shared import (
    DEFAULT_CSV,
    QRF_PY_REPO,
    SPECS,
    array_sha256,
    build_event_sets,
    file_sha256,
    load_bars,
)

import numpy as np
import pandas as pd


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--bars", type=int, default=None, help="real-data API smoke only")
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)

    df = load_bars(args.csv, args.bars)
    event_sets = build_event_sets(df)
    unix_time = (pd.DatetimeIndex(df["time"]).asi8 // 1_000_000_000).astype(np.int64)
    table = {"time": unix_time}
    table.update({spec.name: codes for spec, codes in event_sets})
    pd.DataFrame(table).to_csv(args.out, index=False, lineterminator="\n")

    signal_source = QRF_PY_REPO / "examples" / "batch_runner" / "run_batch.py"
    metadata = {
        "bars": len(df),
        "is_smoke": len(df) != 150_000,
        "csv": str(args.csv),
        "csv_sha256": file_sha256(args.csv),
        "events": str(args.out),
        "events_sha256": file_sha256(args.out),
        "generator": str(Path(__file__).resolve()),
        "generator_command": [sys.executable, *sys.argv],
        "generator_sha256": file_sha256(Path(__file__).resolve()),
        "signal_source": str(signal_source),
        "signal_source_sha256": file_sha256(signal_source),
        "strategies": [
            {
                "name": spec.name,
                "lookback": spec.lookback,
                "events": int(np.count_nonzero(codes)),
                "event_sha256": array_sha256(codes),
            }
            for spec, codes in event_sets
        ],
    }
    meta_path = args.out.with_suffix(args.out.suffix + ".meta.json")
    meta_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
