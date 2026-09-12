"""Run a command on a throwaway Windows desktop so no window reaches the user.

GUI tests create real Tk windows. On Windows there is no headless display, so
the only way to keep them off the interactive desktop is to give the child
process a desktop of its own: CreateDesktopW makes one, and CreateProcessW
launches the child with STARTUPINFOW.lpDesktop pointing at it.

subprocess cannot do this. CPython's subprocess.STARTUPINFO has no lpDesktop
field, and assigning the attribute anyway is silently ignored, so the child
still lands on the interactive desktop. Hence the ctypes CreateProcessW call.

The child inherits this process's standard handles, so pytest output streams
back normally and the exit code is passed through.

Usage:
    python scripts/run_offscreen.py python -m pytest tests -q

On a non-Windows platform the command runs unchanged; X11 and Wayland already
have their own offscreen options.
"""

from __future__ import annotations

import ctypes
import os
import subprocess
import sys
import uuid

if sys.platform == "win32":
    from ctypes import wintypes

    user32 = ctypes.WinDLL("user32", use_last_error=True)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

    GENERIC_ALL = 0x10000000
    STARTF_USESTDHANDLES = 0x00000100
    INFINITE = 0xFFFFFFFF
    STD_INPUT_HANDLE = -10
    STD_OUTPUT_HANDLE = -11
    STD_ERROR_HANDLE = -12

    class STARTUPINFOW(ctypes.Structure):
        _fields_ = [
            ("cb", wintypes.DWORD),
            ("lpReserved", wintypes.LPWSTR),
            ("lpDesktop", wintypes.LPWSTR),
            ("lpTitle", wintypes.LPWSTR),
            ("dwX", wintypes.DWORD),
            ("dwY", wintypes.DWORD),
            ("dwXSize", wintypes.DWORD),
            ("dwYSize", wintypes.DWORD),
            ("dwXCountChars", wintypes.DWORD),
            ("dwYCountChars", wintypes.DWORD),
            ("dwFillAttribute", wintypes.DWORD),
            ("dwFlags", wintypes.DWORD),
            ("wShowWindow", wintypes.WORD),
            ("cbReserved2", wintypes.WORD),
            ("lpReserved2", ctypes.POINTER(ctypes.c_byte)),
            ("hStdInput", wintypes.HANDLE),
            ("hStdOutput", wintypes.HANDLE),
            ("hStdError", wintypes.HANDLE),
        ]

    class PROCESS_INFORMATION(ctypes.Structure):
        _fields_ = [
            ("hProcess", wintypes.HANDLE),
            ("hThread", wintypes.HANDLE),
            ("dwProcessId", wintypes.DWORD),
            ("dwThreadId", wintypes.DWORD),
        ]

    user32.CreateDesktopW.restype = wintypes.HANDLE
    user32.CreateDesktopW.argtypes = [
        wintypes.LPCWSTR, wintypes.LPCWSTR, ctypes.c_void_p,
        wintypes.DWORD, wintypes.DWORD, ctypes.c_void_p,
    ]
    user32.CloseDesktop.argtypes = [wintypes.HANDLE]
    kernel32.CreateProcessW.argtypes = [
        wintypes.LPCWSTR, wintypes.LPWSTR, ctypes.c_void_p, ctypes.c_void_p,
        wintypes.BOOL, wintypes.DWORD, ctypes.c_void_p, wintypes.LPCWSTR,
        ctypes.POINTER(STARTUPINFOW), ctypes.POINTER(PROCESS_INFORMATION),
    ]
    kernel32.GetStdHandle.restype = wintypes.HANDLE


def _run_on_private_desktop(argv: list[str], cwd: str) -> int:
    name = "VsrOffscreen-" + uuid.uuid4().hex[:12]
    desktop = user32.CreateDesktopW(name, None, None, 0, GENERIC_ALL, None)
    if not desktop:
        raise ctypes.WinError(ctypes.get_last_error())

    info = STARTUPINFOW()
    info.cb = ctypes.sizeof(STARTUPINFOW)
    info.lpDesktop = name
    info.dwFlags = STARTF_USESTDHANDLES
    info.hStdInput = kernel32.GetStdHandle(STD_INPUT_HANDLE)
    info.hStdOutput = kernel32.GetStdHandle(STD_OUTPUT_HANDLE)
    info.hStdError = kernel32.GetStdHandle(STD_ERROR_HANDLE)

    process = PROCESS_INFORMATION()
    command = subprocess.list2cmdline(argv)
    try:
        started = kernel32.CreateProcessW(
            None, ctypes.create_unicode_buffer(command), None, None,
            True, 0, None, cwd,
            ctypes.byref(info), ctypes.byref(process),
        )
        if not started:
            raise ctypes.WinError(ctypes.get_last_error())
        kernel32.WaitForSingleObject(process.hProcess, INFINITE)
        code = wintypes.DWORD()
        kernel32.GetExitCodeProcess(process.hProcess, ctypes.byref(code))
        kernel32.CloseHandle(process.hThread)
        kernel32.CloseHandle(process.hProcess)
        return int(code.value)
    finally:
        user32.CloseDesktop(desktop)


def main(argv: list[str]) -> int:
    if not argv:
        print("usage: run_offscreen.py <command> [args...]", file=sys.stderr)
        return 2

    cwd = os.environ.get("VSR_RUN_CWD") or os.getcwd()
    if sys.platform != "win32":
        # subprocess-policy-exempt: this wrapper must pass through arbitrary
        # test commands without imposing the production policy's time limit
        return subprocess.call(argv, cwd=cwd)
    return _run_on_private_desktop(argv, cwd)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
