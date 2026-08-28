"""RM-314: cross-process locking for the per-user state files.

`settings.json`, `queue_state.json`, and `presets.json` live in one
per-user directory and every process that starts writes them. The writes
themselves are atomic (temp file plus `os.replace`), so a reader never sees
a torn file, but two processes doing read-modify-write can still interleave
and drop the newer state. A process-local `threading.Lock` cannot see the
other process, so this module takes a real OS lock on a sidecar file.

The lock is advisory and only meaningful between processes that use it, so
every path that persists shared user state must go through
`state_file_lock()`.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# A write is a few hundred microseconds, so anything approaching this is a
# stuck or suspended peer rather than ordinary contention.
DEFAULT_TIMEOUT_SECONDS = 10.0

_local = threading.local()
# One handle per path per process. Windows byte-range locks are held by the
# file handle, so reusing a single handle keeps re-entry from deadlocking.
_handles: dict = {}
_handles_guard = threading.Lock()


def _open_lock_handle(path: Path):
    with _handles_guard:
        handle = _handles.get(str(path))
        if handle is not None and not handle.closed:
            return handle
        path.parent.mkdir(parents=True, exist_ok=True)
        handle = open(path, "a+b")
        _handles[str(path)] = handle
        return handle


if os.name == "nt":
    import msvcrt

    def _try_lock(handle) -> bool:
        try:
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            return True
        except OSError:
            return False

    def _unlock(handle) -> None:
        try:
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        except OSError as exc:
            logger.debug("State lock release failed: %s", exc)
else:
    import fcntl

    def _try_lock(handle) -> bool:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            return True
        except OSError:
            return False

    def _unlock(handle) -> None:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        except OSError as exc:
            logger.debug("State lock release failed: %s", exc)


@contextmanager
def state_file_lock(path: Path,
                    timeout: float = DEFAULT_TIMEOUT_SECONDS):
    """Hold an exclusive cross-process lock on `path` for the block.

    Re-entrant within a thread. On timeout the block still runs, because
    refusing to save the user's settings is worse than the lost-update risk
    the lock exists to remove, but the wait is logged as a warning so a
    stuck peer is visible rather than silent.
    """
    depth = getattr(_local, "depth", None)
    if depth is None:
        depth = {}
        _local.depth = depth
    key = str(path)
    if depth.get(key, 0) > 0:
        depth[key] += 1
        try:
            yield True
        finally:
            depth[key] -= 1
        return

    handle: Optional[object] = None
    acquired = False
    try:
        handle = _open_lock_handle(Path(path))
    except OSError as exc:
        logger.warning("Could not open the state lock at %s: %s", path, exc)

    if handle is not None:
        deadline = time.monotonic() + max(0.0, float(timeout))
        delay = 0.002
        while True:
            if _try_lock(handle):
                acquired = True
                break
            if time.monotonic() >= deadline:
                logger.warning(
                    "Timed out after %.1fs waiting for the state lock at %s; "
                    "another Video Subtitle Remover process may be stuck. "
                    "Writing anyway.",
                    timeout, path,
                )
                break
            time.sleep(delay)
            delay = min(delay * 2, 0.05)

    depth[key] = depth.get(key, 0) + 1
    try:
        yield acquired
    finally:
        depth[key] -= 1
        if acquired and handle is not None:
            _unlock(handle)


def close_lock_handles() -> None:
    """Release cached lock handles. For tests and interpreter teardown."""
    with _handles_guard:
        for handle in list(_handles.values()):
            try:
                handle.close()
            except OSError:
                pass
        _handles.clear()
