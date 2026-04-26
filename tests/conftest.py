"""Pytest configuration: ensure a synthetic CSV exists before backtester
imports (the module raises FileNotFoundError at import time if BT_CSV /
CSV_FILE does not point at a real file)."""
import os
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
sys.path.insert(0, str(_ROOT))

_FIXTURE = _ROOT / "data" / "SYNTHETIC_TEST.csv"
if not _FIXTURE.exists():
    _FIXTURE.parent.mkdir(parents=True, exist_ok=True)
    from gen_synthetic import generate
    data = generate(2_000, 3600, 1_600_000_000, seed=7)
    with _FIXTURE.open("w", encoding="utf-8") as fh:
        fh.write("time,open,high,low,close\n")
        for t, o, h, l, c in data:
            fh.write(f"{int(t)},{o:.8f},{h:.8f},{l:.8f},{c:.8f}\n")

os.environ["BT_CSV"] = str(_FIXTURE)
