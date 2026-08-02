"""RM-155: supervise one isolated queue job from the parent process.

The parent spawns `python -m backend.job_worker`, streams the job in as
JSON, and reads newline-delimited events back. The point of the exercise
is the failure case: when the child dies of a native fault there is no
exception to catch and no traceback to read, only a process that stopped
existing. This supervisor treats "the child exited without sending a
result" as a first-class outcome, reports it against the one item that
faulted with the decoded exit status and the child's stderr tail, and
leaves the caller free to continue the queue.

Everything crossing the boundary is text. Progress and warnings arrive as
JSON; preview frames arrive as a path to a PNG the child wrote.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import tempfile
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

from backend.job_worker import (
    EVENT_PREVIEW,
    EVENT_PROGRESS,
    EVENT_READY,
    EVENT_RESULT,
    EVENT_WARNING,
    JOB_PROTOCOL_SCHEMA,
    JOB_PROTOCOL_VERSION,
    describe_exit_code,
    write_control_file,
)

logger = logging.getLogger(__name__)

# How long to wait for a terminated child to actually go away before
# escalating from terminate() to kill().
TERMINATE_GRACE_SECONDS = 10.0

# Keep the tail of the child's stderr for the failure report. A native
# crash often prints something useful (a CUDA error, a DLL name) right
# before dying, and that is the only diagnostic that survives.
STDERR_TAIL_LINES = 40


@dataclass
class JobOutcome:
    """What a supervised job did. `status` is the single source of truth."""

    status: str = "error"        # complete | error | paused | cancelled | crashed
    success: bool = False
    error: str = ""
    reason: str = ""
    evidence: dict = field(default_factory=dict)
    exit_code: Optional[int] = None
    stderr_tail: str = ""
    protocol_version: int = JOB_PROTOCOL_VERSION

    @property
    def crashed(self) -> bool:
        return self.status == "crashed"


def worker_command(request_path: str, python_executable: str = "") -> list:
    """Return the argv that launches the worker for `request_path`.

    A frozen build has no importable `-m` target, so it re-executes its
    own executable with a marker argument the entry point understands.
    """
    if getattr(sys, "frozen", False):
        return [sys.executable, "--job-worker", "--request", str(request_path)]
    return [
        python_executable or sys.executable,
        "-m", "backend.job_worker",
        "--request", str(request_path),
    ]


def build_request(
    *,
    input_path: str,
    output_path: str,
    config_payload: dict,
    is_image: bool,
    preview_dir: str = "",
    control_path: str = "",
    resume_checkpoint: bool = True,
    selective_rerun_from: str = "",
    selective_rerun_ranges: Any = None,
    auto_band: bool = False,
) -> dict:
    return {
        "schema": JOB_PROTOCOL_SCHEMA,
        "version": JOB_PROTOCOL_VERSION,
        "input_path": str(input_path),
        "output_path": str(output_path),
        "is_image": bool(is_image),
        "preview_dir": str(preview_dir or ""),
        "control_path": str(control_path or ""),
        "resume_checkpoint": bool(resume_checkpoint),
        "selective_rerun_from": str(selective_rerun_from or ""),
        "selective_rerun_ranges": selective_rerun_ranges or None,
        "auto_band": bool(auto_band),
        "config": dict(config_payload),
    }


class JobSupervisor:
    """Run one job in a child process and translate its event stream."""

    def __init__(
        self,
        request: dict,
        *,
        on_progress: Optional[Callable[[float, str], None]] = None,
        on_preview: Optional[Callable[[str, int, int], None]] = None,
        on_warning: Optional[Callable[[str], None]] = None,
        cwd: Optional[str] = None,
        env: Optional[dict] = None,
        command: Optional[list] = None,
        scratch_dir: Optional[str] = None,
    ):
        self.request = dict(request)
        self.on_progress = on_progress
        self.on_preview = on_preview
        self.on_warning = on_warning
        self._cwd = cwd or str(Path(__file__).resolve().parents[1])
        self._env = env
        self._explicit_command = list(command) if command else None
        self._process: Optional[subprocess.Popen] = None
        self._result: Optional[dict] = None
        self._stderr_lines: list = []
        self._ready = threading.Event()
        self._cancel_sent = False
        self._paused = False
        self._owned_scratch: Optional[tempfile.TemporaryDirectory] = None
        if scratch_dir:
            self._scratch = Path(scratch_dir)
        else:
            self._owned_scratch = tempfile.TemporaryDirectory(
                prefix="vsr_job_")
            self._scratch = Path(self._owned_scratch.name)
        self._scratch.mkdir(parents=True, exist_ok=True)
        self._request_path = self._scratch / "request.json"
        self._control_path = self._scratch / "control.json"
        self.request.setdefault("control_path", str(self._control_path))
        if not self.request.get("control_path"):
            self.request["control_path"] = str(self._control_path)
        # A caller that wires on_preview wants frames: give the child a
        # place to write them without every call site having to invent
        # its own scratch directory.
        if self.on_preview is not None and not self.request.get("preview_dir"):
            self.request["preview_dir"] = str(self._scratch / "preview")

    # -- lifecycle ---------------------------------------------------

    def _popen(self) -> subprocess.Popen:
        env = dict(os.environ if self._env is None else self._env)
        # The child must not inherit a UI-thread assumption, and its
        # stdout is a protocol stream: unbuffered text only.
        env.setdefault("PYTHONUNBUFFERED", "1")
        env.setdefault("PYTHONIOENCODING", "utf-8")
        creationflags = 0
        if sys.platform == "win32":
            # No console window, and a fatal fault must not raise the
            # Windows Error Reporting dialog and block the child forever.
            creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        command = self._explicit_command or worker_command(
            str(self._request_path))
        return subprocess.Popen(
            command,
            # Closed on purpose: the child reads its job from a file and
            # its control state from another, so nothing needs stdin --
            # and a live stdin pipe would be inherited by every grandchild
            # (import probes, ffmpeg) the job spawns.
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=self._cwd,
            env=env,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            creationflags=creationflags,
        )

    def run(self, timeout: Optional[float] = None) -> JobOutcome:
        """Start the child, pump its events, and return the outcome."""
        try:
            self._write_request()
            self._process = self._popen()
        except OSError as exc:
            self._cleanup_scratch()
            return JobOutcome(
                status="error",
                reason="worker_spawn_failed",
                error=f"could not start the job worker: {exc}",
            )

        stderr_thread = threading.Thread(
            target=self._drain_stderr, name="vsr-job-stderr", daemon=True)
        stderr_thread.start()

        try:
            self._pump_events()
        finally:
            code = self._await_exit(timeout)
            stderr_thread.join(timeout=2.0)
            self._close_streams()
        outcome = self._build_outcome(code)
        self._cleanup_scratch()
        return outcome

    def _write_request(self) -> None:
        self._request_path.write_text(
            json.dumps(self.request, ensure_ascii=False, default=str),
            encoding="utf-8",
        )
        write_control_file(self._control_path, cancel=False, pause=False)

    def _cleanup_scratch(self) -> None:
        if self._owned_scratch is not None:
            try:
                self._owned_scratch.cleanup()
            except OSError:
                pass
            self._owned_scratch = None

    def _pump_events(self) -> None:
        stream = self._process.stdout
        if stream is None:
            return
        for raw in stream:
            line = raw.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except (TypeError, ValueError):
                # Anything a library printed to stdout is noise, not a
                # protocol violation; the result event is what matters.
                logger.debug("Ignoring non-protocol worker output: %s", line[:200])
                continue
            if not isinstance(event, dict):
                continue
            self._dispatch(event)

    def _dispatch(self, event: dict) -> None:
        name = str(event.get("event") or "")
        if name == EVENT_READY:
            self._ready.set()
            return
        if name == EVENT_PROGRESS:
            if self.on_progress is not None:
                try:
                    self.on_progress(
                        float(event.get("progress") or 0.0),
                        str(event.get("message") or ""),
                    )
                except Exception:
                    logger.debug("progress callback failed", exc_info=True)
            return
        if name == EVENT_PREVIEW:
            if self.on_preview is not None:
                try:
                    self.on_preview(
                        str(event.get("path") or ""),
                        int(event.get("frameIndex") or 0),
                        int(event.get("totalFrames") or 0),
                    )
                except Exception:
                    logger.debug("preview callback failed", exc_info=True)
            return
        if name == EVENT_WARNING:
            if self.on_warning is not None:
                try:
                    self.on_warning(str(event.get("message") or ""))
                except Exception:
                    logger.debug("warning callback failed", exc_info=True)
            return
        if name == EVENT_RESULT:
            self._result = dict(event)

    def _drain_stderr(self) -> None:
        stream = self._process.stderr if self._process else None
        if stream is None:
            return
        try:
            for raw in stream:
                text = raw.rstrip()
                if not text:
                    continue
                self._stderr_lines.append(text)
                if len(self._stderr_lines) > STDERR_TAIL_LINES:
                    del self._stderr_lines[0]
        except (OSError, ValueError):
            pass

    def _await_exit(self, timeout: Optional[float]) -> Optional[int]:
        if self._process is None:
            return None
        try:
            return self._process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            logger.warning(
                "Job worker exceeded its time budget; terminating")
            self.terminate()
            try:
                return self._process.wait(timeout=TERMINATE_GRACE_SECONDS)
            except subprocess.TimeoutExpired:
                return None

    def _close_streams(self) -> None:
        if self._process is None:
            return
        for stream in (
            self._process.stdin, self._process.stdout, self._process.stderr
        ):
            try:
                if stream is not None:
                    stream.close()
            except (OSError, ValueError):
                pass

    def _build_outcome(self, code: Optional[int]) -> JobOutcome:
        tail = "\n".join(self._stderr_lines[-STDERR_TAIL_LINES:])
        if self._result is None:
            # The defining case: no result event. Either the child was
            # killed, or it faulted natively. Report it as its own status
            # so a caller can tell a crash from a normal failure.
            detail = (
                describe_exit_code(code) if code is not None
                else "the worker did not exit"
            )
            if self._cancel_sent:
                status, error = "cancelled", "Cancelled"
            else:
                status = "crashed"
                error = (
                    f"The job worker stopped before finishing ({detail}). "
                    f"Only this item failed; the rest of the queue is "
                    f"unaffected."
                )
            return JobOutcome(
                status=status,
                success=False,
                error=error,
                reason="worker_crashed" if status == "crashed" else "cancelled",
                exit_code=code,
                stderr_tail=tail,
            )
        evidence = self._result.get("evidence")
        return JobOutcome(
            status=str(self._result.get("status") or "error"),
            success=bool(self._result.get("success")),
            error=str(self._result.get("error") or ""),
            reason=str(self._result.get("reason") or ""),
            evidence=dict(evidence) if isinstance(evidence, dict) else {},
            exit_code=code,
            stderr_tail=tail,
        )

    # -- control -----------------------------------------------------

    def _publish_control(self) -> bool:
        try:
            write_control_file(
                self._control_path,
                cancel=self._cancel_sent,
                pause=self._paused,
            )
            return True
        except OSError:
            logger.warning(
                "Could not publish job control state", exc_info=True)
            return False

    def cancel(self) -> bool:
        self._cancel_sent = True
        return self._publish_control()

    def pause(self) -> bool:
        self._paused = True
        return self._publish_control()

    def resume(self) -> bool:
        self._paused = False
        return self._publish_control()

    def terminate(self) -> None:
        """Stop the child, escalating to kill if it will not go."""
        if self._process is None or self._process.poll() is not None:
            return
        try:
            self._process.terminate()
        except OSError:
            pass
        try:
            self._process.wait(timeout=TERMINATE_GRACE_SECONDS)
            return
        except subprocess.TimeoutExpired:
            pass
        try:
            self._process.kill()
        except OSError:
            pass

    @property
    def pid(self) -> Optional[int]:
        return self._process.pid if self._process is not None else None


__all__ = [
    "JOB_PROTOCOL_VERSION",
    "JobOutcome",
    "JobSupervisor",
    "build_request",
    "describe_exit_code",
    "worker_command",
]
