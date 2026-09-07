import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).parents[1]


def load_module(name, path):
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_resume_requires_current_outputs_and_worker_identity(tmp_path):
    runner = load_module("mc_bar_runner", ROOT / "mc_bar_permutation" / "run.py")
    metrics = tmp_path / "metrics.json"
    ledger = tmp_path / "ledger.bin"
    metrics.write_text("{}", encoding="utf-8")
    ledger.write_bytes(b"ledger")
    status = {
        "status": "complete", "seed": 42,
        "source_sha256": "source", "spec_sha256": "spec",
        "worker_sha256": "worker", "ledger_file": "ledger.bin",
        "metrics_sha256": runner.sha256(metrics),
        "ledger_sha256": runner.sha256(ledger),
    }
    (tmp_path / "status.json").write_text(json.dumps(status), encoding="utf-8")
    assert runner.complete_status(tmp_path, "source", "spec", 42, "worker")
    assert not runner.complete_status(tmp_path, "source", "spec", 42, "different")
    ledger.unlink()
    assert not runner.complete_status(tmp_path, "source", "spec", 42, "worker")


@pytest.mark.parametrize("bad", [0.5, 257.0, np.nan])
def test_callback_signals_are_validated_before_int8_cast(monkeypatch, bad):
    worker = load_module("mc_bar_worker", ROOT / "mc_bar_permutation" / "worker.py")
    frame = pd.DataFrame({
        "time": pd.to_datetime([0, 1], unit="s", utc=True),
        "open": [1.0, 1.0], "high": [1.0, 1.0], "low": [1.0, 1.0],
        "close": [1.0, 1.0], "volume": [1.0, 1.0],
    })
    monkeypatch.setattr(worker, "load_callable", lambda _target: lambda *_args: [0, bad])
    with pytest.raises(ValueError, match="signals must"):
        worker.strategy_signals(frame, {
            "kind": "python-callable", "callable": "module:function",
            "lookback": 1, "parameters": {},
        })


def test_callback_schema_is_reachable_and_validated(tmp_path):
    runner = load_module("mc_bar_runner_schema", ROOT / "mc_bar_permutation" / "run.py")
    base = json.loads((ROOT / "mc_bar_permutation" / "spec.example.json").read_text())
    base["strategy"] = {
        "kind": "python-callable", "callable": "package.module:function",
        "lookback": 20, "parameters": {"threshold": 1.5},
    }
    path = tmp_path / "callback.json"
    path.write_text(json.dumps(base), encoding="utf-8")
    strategy, _ = runner.load_spec(path)
    assert strategy["parameters"] == {"threshold": 1.5}
