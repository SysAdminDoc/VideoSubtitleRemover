"""RM-209: contain an isolated worker and everything it spawns.

Terminating a process on Windows reaches that process only. The job worker
spawns ffmpeg for muxing and post-restore, so killing the worker -- on the
cancel escalation or the wall-clock timeout -- can leave an ffmpeg running to
completion, holding the output and temp files open. The item is reported
cancelled while its output file is still being written, and an immediate retry
fails with a sharing violation.

A Job Object fixes both halves of that. Processes the worker starts after it
joins the job are in the job too, so terminating the job reaps the whole tree.
And with KILL_ON_JOB_CLOSE the tree dies when the last handle closes, which
covers the case no cleanup code can: the GUI process itself dying. Without it
a crashed GUI leaves the worker running to completion, because the worker's
control reader deliberately treats a missing control file as "no request".

Containment is best effort. Every failure here leaves the job running exactly
as it did before, because a job that cannot be contained is still a job worth
finishing.
"""

from __future__ import annotations

import logging
import sys
from typing import Optional

logger = logging.getLogger(__name__)

# JOBOBJECTINFOCLASS.JobObjectExtendedLimitInformation
_JOB_OBJECT_EXTENDED_LIMIT_INFORMATION = 9
_JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000
# AssignProcessToJobObject needs both of these on the process handle.
_PROCESS_TERMINATE = 0x0001
_PROCESS_SET_QUOTA = 0x0100


def _limit_structures():
    """Build the JOBOBJECT_EXTENDED_LIMIT_INFORMATION layout."""
    import ctypes
    from ctypes import wintypes

    class IoCounters(ctypes.Structure):
        _fields_ = [
            ("ReadOperationCount", ctypes.c_ulonglong),
            ("WriteOperationCount", ctypes.c_ulonglong),
            ("OtherOperationCount", ctypes.c_ulonglong),
            ("ReadTransferCount", ctypes.c_ulonglong),
            ("WriteTransferCount", ctypes.c_ulonglong),
            ("OtherTransferCount", ctypes.c_ulonglong),
        ]

    class BasicLimits(ctypes.Structure):
        _fields_ = [
            ("PerProcessUserTimeLimit", wintypes.LARGE_INTEGER),
            ("PerJobUserTimeLimit", wintypes.LARGE_INTEGER),
            ("LimitFlags", wintypes.DWORD),
            ("MinimumWorkingSetSize", ctypes.c_size_t),
            ("MaximumWorkingSetSize", ctypes.c_size_t),
            ("ActiveProcessLimit", wintypes.DWORD),
            ("Affinity", ctypes.POINTER(ctypes.c_ulong)),
            ("PriorityClass", wintypes.DWORD),
            ("SchedulingClass", wintypes.DWORD),
        ]

    class ExtendedLimits(ctypes.Structure):
        _fields_ = [
            ("BasicLimitInformation", BasicLimits),
            ("IoInfo", IoCounters),
            ("ProcessMemoryLimit", ctypes.c_size_t),
            ("JobMemoryLimit", ctypes.c_size_t),
            ("PeakProcessMemoryUsed", ctypes.c_size_t),
            ("PeakJobMemoryUsed", ctypes.c_size_t),
        ]

    return ExtendedLimits


class ProcessJob:
    """A Windows Job Object holding one worker and its descendants."""

    def __init__(self, handle: int) -> None:
        self._handle: Optional[int] = handle

    @property
    def active(self) -> bool:
        return self._handle is not None

    @classmethod
    def create(cls) -> Optional["ProcessJob"]:
        """Return a kill-on-close job, or None where that is unavailable."""
        if sys.platform != "win32":
            return None
        try:
            import ctypes
            from ctypes import wintypes

            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            kernel32.CreateJobObjectW.restype = wintypes.HANDLE
            kernel32.CreateJobObjectW.argtypes = [
                ctypes.c_void_p, wintypes.LPCWSTR]
            handle = kernel32.CreateJobObjectW(None, None)
            if not handle:
                raise ctypes.WinError(ctypes.get_last_error())

            extended = _limit_structures()()
            extended.BasicLimitInformation.LimitFlags = (
                _JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE)
            kernel32.SetInformationJobObject.argtypes = [
                wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
            ok = kernel32.SetInformationJobObject(
                handle,
                _JOB_OBJECT_EXTENDED_LIMIT_INFORMATION,
                ctypes.byref(extended),
                ctypes.sizeof(extended),
            )
            if not ok:
                error = ctypes.get_last_error()
                kernel32.CloseHandle(handle)
                raise ctypes.WinError(error)
            return cls(int(handle))
        except Exception:
            logger.debug("Job object unavailable; worker runs uncontained",
                         exc_info=True)
            return None

    def assign_pid(self, pid: int) -> bool:
        """Put an already-running process into the job."""
        if self._handle is None:
            return False
        try:
            import ctypes
            from ctypes import wintypes

            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            kernel32.OpenProcess.restype = wintypes.HANDLE
            kernel32.OpenProcess.argtypes = [
                wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
            process = kernel32.OpenProcess(
                _PROCESS_SET_QUOTA | _PROCESS_TERMINATE, False, int(pid))
            if not process:
                raise ctypes.WinError(ctypes.get_last_error())
            try:
                kernel32.AssignProcessToJobObject.argtypes = [
                    wintypes.HANDLE, wintypes.HANDLE]
                if not kernel32.AssignProcessToJobObject(
                        self._handle, process):
                    raise ctypes.WinError(ctypes.get_last_error())
            finally:
                kernel32.CloseHandle(process)
            return True
        except Exception:
            logger.debug("Could not assign the worker to its job object",
                         exc_info=True)
            return False

    def terminate(self, exit_code: int = 1) -> bool:
        """Kill every process still in the job."""
        if self._handle is None:
            return False
        try:
            import ctypes
            from ctypes import wintypes

            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            kernel32.TerminateJobObject.argtypes = [
                wintypes.HANDLE, wintypes.UINT]
            if not kernel32.TerminateJobObject(self._handle, int(exit_code)):
                raise ctypes.WinError(ctypes.get_last_error())
            return True
        except Exception:
            logger.debug("Could not terminate the job object", exc_info=True)
            return False

    def close(self) -> None:
        """Release the job. Anything still inside it dies with the handle."""
        handle, self._handle = self._handle, None
        if handle is None:
            return
        try:
            import ctypes

            ctypes.WinDLL("kernel32", use_last_error=True).CloseHandle(handle)
        except Exception:
            logger.debug("Could not close the job object handle",
                         exc_info=True)
