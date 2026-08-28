"""RM-314: one interactive Video Subtitle Remover per user session.

Two GUI processes share `settings.json` and `queue_state.json`, so a second
launch is not a second workspace, it is a race over the first one's state.
The Windows build already publishes `Local\\VideoSubtitleRemoverPro.Running`
so the installer can tell whether the app is open; `CreateMutexW` was
already reporting whether that name existed and the answer was thrown away.

Detection is deliberately passive. Raising or focusing the first window
would steal the foreground from whatever the user is doing, so the second
process reports the condition on stderr and in the log, then exits without
writing any state.
"""

from __future__ import annotations

import ctypes
import logging
import os
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

MUTEX_NAME = "Local\\VideoSubtitleRemoverPro.Running"
ERROR_ALREADY_EXISTS = 183

ALREADY_RUNNING_MESSAGE = (
    "Video Subtitle Remover Pro is already running for this user. "
    "Switch to the open window instead; a second copy would share the same "
    "settings and queue files and overwrite the first one's state."
)


class SingleInstanceGuard:
    """The result of asking for the interactive-instance slot."""

    def __init__(self, *, already_running: bool, handle=None,
                 lock_handle=None, name: str = MUTEX_NAME):
        self.already_running = bool(already_running)
        self.name = name
        self._handle = handle
        self._lock_handle = lock_handle

    def release(self) -> None:
        """Give up the slot. Safe to call more than once."""
        handle, self._handle = self._handle, None
        if handle:
            try:
                ctypes.windll.kernel32.CloseHandle(handle)
            except Exception as exc:  # pragma: no cover - teardown only
                logger.debug("Instance mutex close failed: %s", exc)
        lock_handle, self._lock_handle = self._lock_handle, None
        if lock_handle is not None:
            try:
                lock_handle.close()
            except OSError as exc:  # pragma: no cover - teardown only
                logger.debug("Instance lock close failed: %s", exc)


def _acquire_windows(name: str) -> SingleInstanceGuard:
    kernel32 = ctypes.windll.kernel32
    handle = kernel32.CreateMutexW(None, False, name)
    if not handle:
        # Without a handle there is no answer to give, so let the launch
        # continue rather than refusing on a failed syscall.
        logger.warning("Could not create the instance mutex %s", name)
        return SingleInstanceGuard(already_running=False, name=name)
    already = kernel32.GetLastError() == ERROR_ALREADY_EXISTS
    return SingleInstanceGuard(
        already_running=already, handle=handle, name=name)


def _acquire_posix(lock_path: Path) -> SingleInstanceGuard:
    import fcntl

    try:
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        handle = open(lock_path, "a+b")
    except OSError as exc:
        logger.warning("Could not open the instance lock at %s: %s",
                       lock_path, exc)
        return SingleInstanceGuard(already_running=False, name=str(lock_path))
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        handle.close()
        return SingleInstanceGuard(already_running=True, name=str(lock_path))
    return SingleInstanceGuard(
        already_running=False, lock_handle=handle, name=str(lock_path))


def acquire(name: str = MUTEX_NAME,
            lock_path: Optional[Path] = None) -> SingleInstanceGuard:
    """Claim the interactive-instance slot for this user session."""
    if os.name == "nt":
        return _acquire_windows(name)
    if lock_path is None:
        from gui.config import LOG_DIR

        lock_path = Path(LOG_DIR) / "instance.lock"
    return _acquire_posix(Path(lock_path))
