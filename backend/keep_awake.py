"""RM-316: keep the system awake for the length of a job, and no longer.

A long removal run can be interrupted by idle sleep even though Windows
offers a scoped power request for exactly this. The request here is
deliberately narrow: the system stays up, the display is free to switch off,
and a user who closes the lid or picks Sleep still gets a sleeping machine.

The hold is reference counted, so overlapping jobs cannot release each
other's, and it is a no-op everywhere but Windows.
"""

from __future__ import annotations

import ctypes
import logging
import platform
import threading

logger = logging.getLogger(__name__)

KEEP_AWAKE_STATUS_SCHEMA = "vsr.keep_awake.v1"

# https://learn.microsoft.com/en-us/windows/win32/api/winbase/nf-winbase-setthreadexecutionstate
ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001
# Deliberately never requested. ES_DISPLAY_REQUIRED keeps the screen lit for
# a job the user is not watching, and ES_AWAYMODE_REQUIRED overrides a
# user-initiated sleep, which is not ours to override.
ES_DISPLAY_REQUIRED = 0x00000002
ES_AWAYMODE_REQUIRED = 0x00000040

_LOCK = threading.RLock()
_DEPTH = 0
_HELD = False
_LAST_ERROR = ""


def _is_windows() -> bool:
    return platform.system() == "Windows"


def _set_execution_state(flags: int) -> bool:
    """Call SetThreadExecutionState, returning whether Windows accepted it."""
    try:
        kernel32 = ctypes.windll.kernel32
        kernel32.SetThreadExecutionState.restype = ctypes.c_uint
        kernel32.SetThreadExecutionState.argtypes = [ctypes.c_uint]
        # A zero return means the request was refused.
        return bool(kernel32.SetThreadExecutionState(ctypes.c_uint(flags)))
    except Exception as exc:  # pragma: no cover - platform specific
        global _LAST_ERROR
        _LAST_ERROR = str(exc)[:300]
        logger.debug("SetThreadExecutionState failed: %s", exc)
        return False


def _acquire_locked() -> None:
    global _HELD, _LAST_ERROR
    if _HELD or not _is_windows():
        return
    if _set_execution_state(ES_CONTINUOUS | ES_SYSTEM_REQUIRED):
        _HELD = True
        _LAST_ERROR = ""
        logger.debug("System sleep held for processing")
    else:
        _LAST_ERROR = _LAST_ERROR or "SetThreadExecutionState returned 0"
        logger.warning(
            "Could not hold the system awake for processing; a long job may "
            "be interrupted by idle sleep"
        )


def _release_locked() -> None:
    global _HELD
    if not _HELD:
        return
    # ES_CONTINUOUS on its own clears the request and restores the machine's
    # normal idle timers.
    _set_execution_state(ES_CONTINUOUS)
    _HELD = False
    logger.debug("System sleep hold released")


def acquire() -> None:
    """Take one reference on the processing sleep hold."""
    global _DEPTH
    with _LOCK:
        _DEPTH += 1
        if _DEPTH == 1:
            _acquire_locked()


def release() -> None:
    """Drop one reference, releasing the hold when the last job ends."""
    global _DEPTH
    with _LOCK:
        if _DEPTH <= 0:
            _DEPTH = 0
            return
        _DEPTH -= 1
        if _DEPTH == 0:
            _release_locked()


def release_all() -> None:
    """Drop every reference. For shutdown and crash cleanup."""
    global _DEPTH
    with _LOCK:
        _DEPTH = 0
        _release_locked()


def status() -> dict:
    with _LOCK:
        return {
            "schema": KEEP_AWAKE_STATUS_SCHEMA,
            "supported": _is_windows(),
            "held": _HELD,
            "depth": _DEPTH,
            "error": _LAST_ERROR,
        }


class keep_system_awake:
    """Hold the system awake for the duration of a block.

    Safe to nest and safe to leave by an exception, a cancellation, or a
    pause: the reference is dropped either way.
    """

    def __enter__(self) -> "keep_system_awake":
        acquire()
        return self

    def __exit__(self, *exc) -> None:
        release()
