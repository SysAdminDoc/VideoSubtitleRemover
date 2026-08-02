"""RM-155: run one queue job in an isolated child process.

A fatal native fault -- an access violation inside OpenCV, ONNX Runtime,
or a model's own kernels -- cannot be caught by Python. In a single
in-process worker it takes down the interpreter, which means it takes down
the GUI and every remaining queued job along with the one that faulted.
Checkpoints and retries do not help: the process that would have resumed
them is gone.

This module is the child half of a versioned local job protocol. The
parent spawns `python -m backend.job_worker`, hands it one job as JSON,
and reads newline-delimited JSON events back. Because the fault is
confined to the child, an abrupt death is just a missing `result` event
that the parent reports against that one item.

The protocol is deliberately small and text-only:

  parent -> child   a JSON request file, and a JSON control file the
                    parent rewrites to ask for cancel/pause/resume
  child  -> parent  one JSON event per line on stdout:
                    ready / progress / preview / warning / result

Control deliberately does *not* travel over stdin. A reader thread parked
in a blocking stdin read deadlocks against C-extension module
initialisation during the child's own imports (numpy and cv2 both load
native modules), and on Windows that read cannot be interrupted, so the
worker would hang before it ever started work. A polled control file has
none of those failure modes, costs nothing next to video processing, and
lets the child run with stdin closed -- which also stops any grandchild
(an import probe, an ffmpeg call) from inheriting a live pipe.

Preview frames are the one payload JSON cannot carry cheaply, so the
child writes a throttled PNG into a scratch directory and sends its path.

The version is checked on both sides. A parent and child that disagree
refuse to run rather than silently misinterpreting each other's fields.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any, Optional


JOB_PROTOCOL_SCHEMA = "vsr.job_protocol.v1"
JOB_PROTOCOL_VERSION = 1

EVENT_READY = "ready"
EVENT_PROGRESS = "progress"
EVENT_PREVIEW = "preview"
EVENT_WARNING = "warning"
EVENT_RESULT = "result"

COMMAND_CANCEL = "cancel"
COMMAND_PAUSE = "pause"
COMMAND_RESUME = "resume"

# The evidence a completed job carries back. Every one of these is a
# plain dict/str/float on the remover, so the whole result crosses the
# process boundary as JSON with no pickling and no shared memory.
RESULT_ATTRIBUTES = (
    "last_mask_export",
    "last_mask_import",
    "last_frozen_matte",
    "last_timing_report",
    "last_output_contract",
    "last_selective_rerun",
    "last_stage_timings",
    "last_detection_stats",
    "last_quality_report",
    "last_translation",
    "last_output_path",
    "last_error_message",
    "last_error_reason",
    "last_resume_warning",
    "last_pause_checkpoint",
    "last_pause_checkpoint_path",
    "last_work_directory_warning",
)

PREVIEW_MIN_INTERVAL = 1.0 / 12.0

# Windows surfaces a hard native fault as a large unsigned exit status.
# Naming the common ones turns "exit code 3221225477" into a sentence a
# person can act on.
NATIVE_FAULT_CODES = {
    0xC0000005: "access violation (native memory fault)",
    0xC000001D: "illegal instruction",
    0xC00000FD: "stack overflow",
    0xC0000094: "integer division by zero",
    0xC0000374: "heap corruption",
    0x80000003: "breakpoint / abort",
}


def describe_exit_code(code: int) -> str:
    """Render a child exit status as something a user can read."""
    value = int(code)
    unsigned = value & 0xFFFFFFFF
    if unsigned in NATIVE_FAULT_CODES:
        return f"{NATIVE_FAULT_CODES[unsigned]} (0x{unsigned:08X})"
    if value < 0:
        # POSIX: negative means terminated by signal N.
        return f"terminated by signal {-value}"
    return f"exit code {value}"


class _EventWriter:
    """Serialize events to stdout, one JSON object per line."""

    def __init__(self, stream=None):
        self._stream = stream if stream is not None else sys.stdout
        self._lock = threading.Lock()

    def emit(self, event: str, **fields: Any) -> None:
        payload = {"schema": JOB_PROTOCOL_SCHEMA, "event": event}
        payload.update(fields)
        line = json.dumps(payload, ensure_ascii=False, default=str)
        with self._lock:
            try:
                self._stream.write(line + "\n")
                self._stream.flush()
            except (OSError, ValueError):
                # The parent went away. Nothing useful remains to say.
                pass


CONTROL_POLL_SECONDS = 0.2


def write_control_file(path: str | Path, *, cancel: bool = False,
                       pause: bool = False) -> None:
    """Publish the current control state for a running job, atomically."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_name(target.name + ".tmp")
    temp.write_text(
        json.dumps({"cancel": bool(cancel), "pause": bool(pause)}),
        encoding="utf-8",
    )
    os.replace(temp, target)


class _ControlFile:
    """Poll a small JSON file for cancel/pause requests.

    Reads are cached for `CONTROL_POLL_SECONDS` so a per-frame progress
    callback cannot turn into a per-frame stat() storm. A missing or
    unparsable file means "no request pending", never "cancel": losing
    the file must not abort a long job that was running fine.
    """

    def __init__(self, path: str | Path = ""):
        self._path = Path(path) if path else None
        self._checked = 0.0
        self._cancel = False
        self._pause = False

    def _refresh(self) -> None:
        if self._path is None:
            return
        now = time.monotonic()
        if now - self._checked < CONTROL_POLL_SECONDS:
            return
        self._checked = now
        try:
            payload = json.loads(self._path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, TypeError, ValueError):
            return
        if isinstance(payload, dict):
            self._cancel = bool(payload.get("cancel"))
            self._pause = bool(payload.get("pause"))

    class _Flag:
        def __init__(self, owner, attribute):
            self._owner = owner
            self._attribute = attribute

        def is_set(self) -> bool:
            self._owner._refresh()
            return bool(getattr(self._owner, self._attribute))

        def set(self) -> None:
            setattr(self._owner, self._attribute, True)

        def clear(self) -> None:
            setattr(self._owner, self._attribute, False)

    @property
    def cancelled(self):
        return self._Flag(self, "_cancel")

    @property
    def paused(self):
        return self._Flag(self, "_pause")

    def start(self) -> "_ControlFile":
        """Present the same interface the parent's spawn path expects."""
        return self


def _write_preview(frame, scratch: Path, index: int) -> Optional[str]:
    """Persist one preview frame as PNG and return its path."""
    try:
        import cv2

        scratch.mkdir(parents=True, exist_ok=True)
        path = scratch / f"preview_{index:06d}.png"
        if not cv2.imwrite(str(path), frame):
            return None
        # Keep only the newest few frames; the parent reads immediately
        # and a stalled parent must not fill the disk.
        for stale in sorted(scratch.glob("preview_*.png"))[:-3]:
            try:
                stale.unlink()
            except OSError:
                pass
        return str(path)
    except Exception:
        return None


def run_job(request: dict, *, writer=None, control=None) -> int:
    """Execute one job described by `request`. Returns a process exit code."""
    events = writer if writer is not None else _EventWriter()
    if control is None:
        control = _ControlFile(str(request.get("control_path") or ""))
    commands = control

    version = int(request.get("version") or 0)
    if version != JOB_PROTOCOL_VERSION:
        events.emit(
            EVENT_RESULT,
            status="error",
            error=(
                f"job protocol version mismatch: parent sent {version}, "
                f"this worker speaks {JOB_PROTOCOL_VERSION}"
            ),
            reason="protocol_version_mismatch",
        )
        return 2

    from backend.config import ProcessingConfig
    from backend.config_schema import apply_backend_payload
    from backend.processor import SubtitleRemover
    from backend.resume_checkpoint import (
        ProcessingPaused,
        _checkpoint_key,
        _default_checkpoint_dir,
    )

    input_path = str(request.get("input_path") or "")
    output_path = str(request.get("output_path") or "")
    if not input_path or not output_path:
        events.emit(
            EVENT_RESULT, status="error", reason="invalid_request",
            error="job request needs input_path and output_path")
        return 2

    scratch = Path(str(request.get("preview_dir") or "")) if request.get(
        "preview_dir") else None
    want_preview = scratch is not None

    try:
        config = apply_backend_payload(
            ProcessingConfig(), dict(request.get("config") or {}))
    except (TypeError, ValueError) as exc:
        events.emit(
            EVENT_RESULT, status="error", reason="invalid_config",
            error=f"job config was rejected: {exc}")
        return 2

    remover = SubtitleRemover(config)
    preview_state = {"last": 0.0, "index": 0}

    def on_progress(progress: float, message: str) -> None:
        if commands.cancelled.is_set():
            raise InterruptedError("Processing cancelled")
        events.emit(
            EVENT_PROGRESS,
            progress=float(progress),
            message=str(message),
        )

    def on_preview_frame(frame, cur_idx, total) -> None:
        if commands.cancelled.is_set() or not want_preview:
            return
        now = time.monotonic()
        if now - preview_state["last"] < PREVIEW_MIN_INTERVAL:
            return
        preview_state["last"] = now
        preview_state["index"] += 1
        path = _write_preview(frame, scratch, preview_state["index"])
        if path:
            events.emit(
                EVENT_PREVIEW,
                path=path,
                frameIndex=int(cur_idx or 0),
                totalFrames=int(total or 0),
            )

    remover.on_progress = on_progress
    if want_preview:
        remover.on_preview_frame = on_preview_frame

    warning = getattr(remover, "last_work_directory_warning", None)
    if warning:
        events.emit(EVENT_WARNING, message=str(warning))

    events.emit(EVENT_READY, version=JOB_PROTOCOL_VERSION, pid=os.getpid())

    # Auto subtitle-band detection runs here, in the child, so the probe's
    # OCR model never loads into the GUI process the isolation exists to
    # protect. `_apply_auto_band_override` only probes when no explicit
    # region is configured, matching the in-process path.
    if bool(request.get("auto_band")) and not bool(request.get("is_image")):
        from backend.config import _apply_auto_band_override

        try:
            band = _apply_auto_band_override(
                remover,
                input_path,
                auto_band=True,
                base_subtitle_area=config.subtitle_area,
                base_subtitle_areas=config.subtitle_areas,
                base_subtitle_region_spans=getattr(
                    config, "subtitle_region_spans", None),
                base_subtitle_region_keyframes=getattr(
                    config, "subtitle_region_keyframes", None),
            )
            if band:
                events.emit(
                    EVENT_PROGRESS, progress=0.0,
                    message=f"Auto-detected subtitle band: {band}")
        except Exception as exc:
            events.emit(
                EVENT_WARNING,
                message=f"Subtitle band detection failed: {exc}")

    status = "error"
    error = ""
    reason = ""
    success = False
    try:
        if bool(request.get("is_image")):
            success = bool(remover.process_image(input_path, output_path))
        else:
            checkpoint_dir = _default_checkpoint_dir(config.work_directory)
            success = bool(remover.process_video(
                input_path,
                output_path,
                checkpoint_dir=checkpoint_dir,
                checkpoint_key=_checkpoint_key(input_path, output_path),
                resume_checkpoint=bool(request.get("resume_checkpoint", True)),
                pause_check=commands.paused.is_set,
                selective_rerun_from=(
                    request.get("selective_rerun_from") or None),
                selective_rerun_ranges=(
                    request.get("selective_rerun_ranges") or None),
            ))
        status = "complete" if success else "error"
        if not success:
            error = str(
                getattr(remover, "last_error_message", "")
                or "Processing failed")
            reason = str(getattr(remover, "last_error_reason", "") or "")
    except ProcessingPaused as exc:
        status = "paused"
        error = str(exc)
    except InterruptedError as exc:
        status = "cancelled"
        error = str(exc)
    except BaseException as exc:  # noqa: BLE001 - reported, never swallowed
        status = "error"
        error = f"{type(exc).__name__}: {exc}"
        reason = str(getattr(remover, "last_error_reason", "") or "")

    evidence = {}
    for name in RESULT_ATTRIBUTES:
        value = getattr(remover, name, None)
        if value is not None:
            evidence[name] = value
    provenance = getattr(remover, "execution_provenance", None)
    if provenance is not None:
        try:
            evidence["execution_provenance"] = provenance.to_dict()
        except Exception:
            pass

    events.emit(
        EVENT_RESULT,
        status=status,
        success=bool(success),
        error=error,
        reason=reason,
        evidence=evidence,
    )
    return 0 if status in {"complete", "paused", "cancelled"} else 1


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run one VSR queue job in an isolated child process.")
    parser.add_argument(
        "--request", default="", metavar="PATH",
        help="JSON job request file written by the parent.")
    parser.add_argument(
        "--protocol-version", action="store_true",
        help="Print the supported protocol version and exit.")
    args = parser.parse_args(argv)

    # A version probe must not need a job: the supervisor and packaging
    # checks use it to confirm parent and child agree before any work.
    if args.protocol_version:
        print(JOB_PROTOCOL_VERSION)
        return 0
    if not args.request:
        parser.error("--request is required to run a job")

    writer = _EventWriter()
    try:
        request = json.loads(Path(args.request).read_text(encoding="utf-8"))
        if not isinstance(request, dict):
            raise ValueError("job request must be a JSON object")
    except (OSError, UnicodeError, TypeError, ValueError) as exc:
        writer.emit(
            EVENT_RESULT, status="error", reason="unreadable_request",
            error=f"job request could not be read: {exc}")
        return 2
    code = run_job(request, writer=writer)
    # Hard-exit rather than unwinding. `run_job` has returned, so every
    # `finally` in the pipeline has run and the result line is flushed;
    # what remains is third-party atexit handlers (CUDA, ORT, Tk) that on
    # a worker process can only add latency or a spurious teardown error
    # to a job that already reported its outcome.
    _flush_streams()
    os._exit(code)


def _flush_streams() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.flush()
        except (OSError, ValueError):
            pass


if __name__ == "__main__":
    raise SystemExit(main())
