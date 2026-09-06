"""Ledger ownership tests use file markers, not simulated strategy results."""
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import subprocess
import sys

import pandas as pd
import pytest

import backtester as bt
from backtester import ledger_lock


def competitor(path):
    script = """
import runpy, sys
lock = runpy.run_path(sys.argv[1])
with lock['ledger_run'](sys.argv[2]):
    pass
"""
    return subprocess.run(
        [sys.executable, "-c", script, ledger_lock.__file__, str(path)],
        capture_output=True, text=True, timeout=20,
    )


def test_nested_lock_blocks_process_and_preserves_existing_file(tmp_path):
    path = tmp_path / "trades.csv"
    path.write_text("existing ledger")
    with ledger_lock.ledger_run(path):
        with ledger_lock.ledger_run(path):
            pass
        result = competitor(path)
        assert result.returncode != 0
        assert "Trade ledger is already locked" in result.stderr
        assert path.read_text() == "existing ledger"
        assert competitor(tmp_path / "independent.csv").returncode == 0
    assert not Path(str(path) + ".lock").exists()
    assert competitor(path).returncode == 0


def test_exception_cleanup_and_no_timer_lock_stealing(tmp_path):
    path = tmp_path / "trades.csv"
    with pytest.raises(ValueError):
        with ledger_lock.ledger_run(path):
            raise ValueError("failed run")
    lock = Path(str(path) + ".lock")
    assert not lock.exists()
    lock.write_text("pid=unknown\n")
    os.utime(lock, (0, 0))
    with pytest.raises(RuntimeError, match="already locked"):
        with ledger_lock.ledger_run(path):
            pass
    assert lock.read_text() == "pid=unknown\n"


def test_env_default_and_explicit_config(monkeypatch, tmp_path):
    path = str(tmp_path / "env.csv")
    monkeypatch.setenv("BT_EXPORT_PATH", path)
    assert bt.Config().export_path == path
    assert bt.Config(export_path="explicit.csv").export_path == "explicit.csv"


def test_opt_surface_follows_ledger_directory_and_name(tmp_path):
    from backtester.opt_surface import _surface_path
    assert Path(_surface_path(str(tmp_path / "trade_list.csv"), "csv")) == tmp_path / "opt_surface.csv"
    assert Path(_surface_path(str(tmp_path / "custom.csv"), "csv")) == tmp_path / "custom.csv.opt_surface.csv"


def test_main_owns_ledger_between_nested_stages(monkeypatch, tmp_path):
    path = tmp_path / "nested" / "trades.csv"
    original = bt.EXPORT_PATH

    def classic(_df):
        bt._safe_append_or_write_trade_csv(pd.DataFrame({"stage": ["classic"]}), bt.EXPORT_PATH, True)

    def wfo(*_args):
        bt._safe_append_or_write_trade_csv(pd.DataFrame({"stage": ["wfo"]}), bt.EXPORT_PATH, False)

    def main():
        bt.classic_single_run(None)
        assert competitor(path).returncode != 0
        bt.walk_forward(None, None, None)

    monkeypatch.setattr(bt, "_classic_single_run_impl", classic)
    monkeypatch.setattr(bt, "_walk_forward_impl", wfo)
    monkeypatch.setattr(bt, "_main_impl", main)
    bt.main(bt.Config(export_path=str(path)))
    assert path.read_text().splitlines() == ["stage", "classic", "wfo"]
    assert bt.EXPORT_PATH == original
    assert not Path(str(path) + ".lock").exists()


@pytest.mark.parametrize("entry", ["main", "classic_single_run", "walk_forward"])
def test_public_entry_refuses_foreign_lock_before_running(entry, monkeypatch, tmp_path):
    path = tmp_path / "trades.csv"
    Path(str(path) + ".lock").write_text("pid=another\n")
    cfg = bt.Config(export_path=str(path))
    args = {"main": (), "classic_single_run": (None,), "walk_forward": (None, None, None)}[entry]
    with pytest.raises(RuntimeError, match="already locked"):
        getattr(bt, entry)(*args, config=cfg)


def test_threaded_entry_cannot_change_active_module_config(tmp_path):
    first = str(tmp_path / "first.csv")
    with bt._ledger_run(bt.Config(export_path=first)):
        with ThreadPoolExecutor(max_workers=1) as executor:
            result = executor.submit(bt.main, bt.Config(export_path=str(tmp_path / "second.csv")))
            with pytest.raises(RuntimeError, match="separate processes"):
                result.result(timeout=5)
        assert bt.EXPORT_PATH == first
