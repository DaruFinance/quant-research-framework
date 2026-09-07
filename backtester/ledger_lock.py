"""Run-scoped, cross-process ledger ownership shared with the Rust engine.

A killed process can leave a sentinel. We never steal it on a timer: remove
the named lock only after verifying its owner has stopped.
"""
from contextlib import contextmanager
from pathlib import Path
import os
import threading

_held = threading.local()
engine_run_lock = threading.RLock()


def acquire(path):
    target = Path(path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    lock = str(target) + ".lock"
    key = (os.getpid(), lock)
    held = getattr(_held, "paths", None)
    if held is None:
        held = _held.paths = set()
    if key in held:
        return None, lock
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        raise RuntimeError(
            f"Trade ledger is already locked: {lock}. Choose a distinct "
            "BT_EXPORT_PATH (or Config.export_path). If its owner crashed, "
            "verify it has stopped before removing the lock."
        ) from exc
    try:
        os.write(fd, f"pid={os.getpid()}\n".encode("ascii"))
    except BaseException:
        os.close(fd)
        os.unlink(lock)
        raise
    held.add(key)
    return fd, lock


def release(fd, lock):
    if fd is None:  # nested call; the outer run owns the sentinel
        return
    try:
        os.close(fd)  # Windows cannot unlink the open sentinel.
    finally:
        try:
            os.unlink(lock)
        finally:
            _held.paths.discard((os.getpid(), lock))


@contextmanager
def ledger_run(path):
    fd, lock = acquire(path)
    try:
        yield
    finally:
        release(fd, lock)
