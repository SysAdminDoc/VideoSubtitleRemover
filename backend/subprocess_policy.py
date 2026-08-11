"""Shared policy for every production child process.

The GUI must never surface a console window or leave a child process waiting
for terminal input. Short-lived commands use :func:`run_process`, which drains
captured streams into bounded tail buffers and owns timeout/cancel escalation.
Streaming FFmpeg adapters use :func:`popen_process` and remain responsible for
closing their explicitly requested pipes before calling :func:`terminate_process`.
"""

from __future__ import annotations

import os
import subprocess
import threading
import time
from collections.abc import Callable, Sequence
from typing import Any, Optional, Union


DEFAULT_MAX_OUTPUT_BYTES = 8 * 1024 * 1024
WINDOWS_CREATE_NO_WINDOW = int(
    getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)
)

Command = Union[str, Sequence[Union[str, os.PathLike[str]]]]


def _hidden_creationflags(existing: int = 0) -> int:
    flags = int(existing or 0)
    if os.name == "nt":
        flags |= WINDOWS_CREATE_NO_WINDOW
    return flags


def popen_process(command: Command, **kwargs: Any) -> subprocess.Popen:
    """Start a non-shell child with hidden Windows UI and closed stdin.

    Callers that intentionally stream data may override ``stdin`` with
    ``subprocess.PIPE``. The returned handle must be waited or terminated by
    the caller; one-shot work should use :func:`run_process` instead.
    """
    if kwargs.pop("shell", False):
        raise ValueError("shell=True is forbidden by the subprocess policy")
    kwargs["shell"] = False
    kwargs.setdefault("stdin", subprocess.DEVNULL)
    kwargs.setdefault("close_fds", True)
    creationflags = _hidden_creationflags(kwargs.pop("creationflags", 0))
    if creationflags:
        kwargs["creationflags"] = creationflags
    return subprocess.Popen(command, **kwargs)


def terminate_process(proc: subprocess.Popen, timeout: float = 2.0) -> None:
    """Terminate a child and escalate to kill when graceful exit stalls."""
    try:
        if proc.poll() is not None:
            return
    except Exception:
        pass
    try:
        proc.terminate()
    except Exception:
        pass
    try:
        proc.wait(timeout=max(0.0, float(timeout)))
        return
    except subprocess.TimeoutExpired:
        pass
    except Exception:
        return
    try:
        proc.kill()
    except Exception:
        pass
    try:
        proc.wait(timeout=max(0.0, float(timeout)))
    except Exception:
        pass


class _BoundedCollector:
    def __init__(self, stream: Any, limit: int, *, text: bool) -> None:
        self._stream = stream
        self._limit = max(1, int(limit))
        self._empty: Union[str, bytes] = "" if text else b""
        self._chunks: list[Union[str, bytes]] = []
        self._size = 0
        self._thread = threading.Thread(
            target=self._drain,
            name="vsr-subprocess-drain",
            daemon=True,
        )

    def start(self) -> None:
        self._thread.start()

    def join(self) -> None:
        self._thread.join(timeout=5.0)
        if self._thread.is_alive():
            try:
                self._stream.close()
            except Exception:
                pass
            self._thread.join(timeout=5.0)

    def _drain(self) -> None:
        try:
            while True:
                chunk = self._stream.read(65536)
                if not chunk:
                    break
                self._chunks.append(chunk)
                self._size += len(chunk)
                while self._size > self._limit and self._chunks:
                    overflow = self._size - self._limit
                    first = self._chunks[0]
                    if len(first) <= overflow:
                        self._chunks.pop(0)
                        self._size -= len(first)
                    else:
                        self._chunks[0] = first[overflow:]
                        self._size -= overflow
        finally:
            try:
                self._stream.close()
            except Exception:
                pass

    def value(self) -> Union[str, bytes]:
        if not self._chunks:
            return self._empty
        return self._empty.join(self._chunks)


class _StdinWriter:
    """Feed a child's stdin from a worker thread.

    RM-142: writing a large payload inline blocks the caller before the
    deadline/cancel loop starts, so a child that never reads stdin could hang
    ``run_process`` forever despite its timeout. The write happens off-thread
    instead; terminating the child breaks the pipe and releases it.
    """

    def __init__(self, stream: Any, payload: Union[str, bytes]) -> None:
        self._stream = stream
        self._payload = payload
        self.error: Optional[BaseException] = None
        self._thread = threading.Thread(
            target=self._write,
            name="vsr-subprocess-stdin",
            daemon=True,
        )

    def start(self) -> None:
        self._thread.start()

    def join(self, timeout: float = 5.0) -> bool:
        self._thread.join(timeout=max(0.0, float(timeout)))
        return not self._thread.is_alive()

    def _write(self) -> None:
        try:
            self._stream.write(self._payload)
            self._stream.flush()
        except BaseException as exc:  # child exited early or was terminated
            self.error = exc
        finally:
            try:
                self._stream.close()
            except Exception:
                pass


def run_process(
    command: Command,
    *,
    timeout: float,
    capture_output: bool = False,
    check: bool = False,
    text: bool = False,
    input: Optional[Union[str, bytes]] = None,
    max_output_bytes: int = DEFAULT_MAX_OUTPUT_BYTES,
    cancel_check: Optional[Callable[[], bool]] = None,
    on_process: Optional[Callable[[Optional[subprocess.Popen]], None]] = None,
    **kwargs: Any,
) -> subprocess.CompletedProcess:
    """Run a child under the shared hidden, bounded, cancellable policy."""
    if timeout is None or float(timeout) <= 0.0:
        raise ValueError("a positive subprocess timeout is required")
    if input is not None and "stdin" in kwargs:
        raise ValueError("stdin and input arguments may not both be used")
    if capture_output and ("stdout" in kwargs or "stderr" in kwargs):
        raise ValueError("stdout/stderr may not be used with capture_output")

    if capture_output:
        kwargs["stdout"] = subprocess.PIPE
        kwargs["stderr"] = subprocess.PIPE
    if input is not None:
        kwargs["stdin"] = subprocess.PIPE
    kwargs["text"] = bool(text)

    proc = popen_process(command, **kwargs)
    stdout_collector = (
        _BoundedCollector(proc.stdout, max_output_bytes, text=text)
        if getattr(proc, "stdout", None) is not None
        and kwargs.get("stdout") == subprocess.PIPE
        else None
    )
    stderr_collector = (
        _BoundedCollector(proc.stderr, max_output_bytes, text=text)
        if getattr(proc, "stderr", None) is not None
        and kwargs.get("stderr") == subprocess.PIPE
        else None
    )
    for collector in (stdout_collector, stderr_collector):
        if collector is not None:
            collector.start()

    if on_process is not None:
        on_process(proc)
    started = time.monotonic()
    stdin_writer: Optional[_StdinWriter] = None
    if input is not None and proc.stdin is not None:
        stdin_writer = _StdinWriter(proc.stdin, input)
        stdin_writer.start()
    try:
        while True:
            if cancel_check is not None and cancel_check():
                terminate_process(proc, timeout=2.0)
                raise InterruptedError("subprocess cancelled")
            remaining = float(timeout) - (time.monotonic() - started)
            if remaining <= 0.0:
                terminate_process(proc, timeout=2.0)
                raise subprocess.TimeoutExpired(command, timeout)
            try:
                returncode = proc.wait(timeout=min(0.1, remaining))
                if cancel_check is not None and cancel_check():
                    # A cancel that arrived while we were blocked in wait()
                    # -- typically because the caller terminated the child
                    # itself -- must stay a cancel. Reporting the resulting
                    # nonzero exit as a plain failure let callers mistake a
                    # user Stop for an encoder error and "recover" from it.
                    raise InterruptedError("subprocess cancelled")
                break
            except subprocess.TimeoutExpired:
                continue
    finally:
        if on_process is not None:
            on_process(None)
        if stdin_writer is not None and not stdin_writer.join(timeout=5.0):
            # The child is gone or ignoring stdin; break the pipe so the
            # writer thread cannot outlive this call.
            terminate_process(proc, timeout=2.0)
            try:
                proc.stdin.close()  # type: ignore[union-attr]
            except Exception:
                pass
            stdin_writer.join(timeout=5.0)
        for collector in (stdout_collector, stderr_collector):
            if collector is not None:
                collector.join()

    stdout = stdout_collector.value() if stdout_collector is not None else None
    stderr = stderr_collector.value() if stderr_collector is not None else None
    result = subprocess.CompletedProcess(command, returncode, stdout, stderr)
    if check:
        result.check_returncode()
    return result
