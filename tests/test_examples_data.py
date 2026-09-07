"""Shipped examples resolve real bundled data from any working directory."""
import os
from pathlib import Path
import subprocess
import sys

import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]


def environment():
    env = os.environ.copy()
    env.pop("BT_CSV", None)
    env["PYTHONPATH"] = str(ROOT)
    for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        env[key] = "1"
    return env


def test_bundled_history_has_real_positive_volume():
    bars = pd.read_csv(ROOT / "data/SOLUSDT_1h.csv")
    assert bars.columns.tolist() == ["time", "open", "high", "low", "close", "volume"]
    assert len(bars) == 53182
    assert bars["time"].is_monotonic_increasing and bars["time"].is_unique
    assert bars["volume"].gt(0).any()
    assert bars.iloc[0]["time"] == 1597125600
    assert bars.iloc[-1]["time"] == 1788649200


def test_volume_default_runs_outside_checkout(tmp_path):
    result = subprocess.run([sys.executable, str(ROOT / "examples/volume_strategies.py")],
                            cwd=tmp_path, env=environment(), text=True,
                            capture_output=True, timeout=60)
    assert result.returncode == 0, result.stderr
    assert len([line for line in result.stdout.splitlines() if "signals" in line]) == 4


@pytest.mark.parametrize("override", ["argument", "environment"])
def test_volume_missing_column_names_downloader_flag(tmp_path, override):
    path = str(ROOT / "tests/fixtures/sol_1h_30000_31000.csv")
    args = [sys.executable, str(ROOT / "examples/volume_strategies.py")]
    env = environment()
    if override == "argument":
        args.append(path)
    else:
        env["BT_CSV"] = path
    result = subprocess.run(args, cwd=tmp_path, env=env, text=True,
                            capture_output=True, timeout=60)
    assert result.returncode != 0
    assert "--volume" in result.stderr
    assert "Traceback" not in result.stderr


def test_end_to_end_default_resolves_bundled_history(tmp_path):
    # Verify entry-point setup without repeating the entire backtest suite.
    script = (
        "import runpy; "
        f"ns = runpy.run_path({str(ROOT / 'examples/end_to_end/end_to_end.py')!r}); "
        "print('resolved=' + str(ns['bt'].CSV_FILE))"
    )
    result = subprocess.run([sys.executable, "-c", script], cwd=tmp_path,
                            env=environment(), text=True, capture_output=True, timeout=60)
    assert result.returncode == 0, result.stderr
    assert f"resolved={ROOT / 'data/SOLUSDT_1h.csv'}" in result.stdout
