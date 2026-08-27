"""Bounded failure classification for queue rows and batch reports.

The processor already records a free-form ``last_error_reason`` (for example
``truncated_decode`` or ``intermediate_writer_timeout``) and a curated
``last_error_message``.  Both are useful to a human reading one row and
useless for aggregation: the vocabulary is open, so a batch report cannot
count "how many runs died in the writer" without pattern matching.

This module projects every failure path onto a small closed set.  The
curated message is untouched; the reason is an extra field.
"""

from __future__ import annotations

from typing import Optional

from backend.execution_provenance import RequestedStageError


REASON_NONE = ""
REASON_NO_SPACE = "no_space"
REASON_WRITER_FAILED = "writer_failed"
REASON_OUTPUT_EMPTY = "output_empty"
REASON_FFMPEG_FAILED = "ffmpeg_failed"
REASON_MODEL_MISSING = "model_missing"
REASON_REQUESTED_STAGE_FAILED = "requested_stage_failed"
REASON_DECODE_FAILED = "decode_failed"
REASON_INPUT_MISSING = "input_missing"
REASON_PERMISSION_DENIED = "permission_denied"
REASON_TIMED_OUT = "timed_out"
REASON_FROZEN_MATTE = "frozen_matte"
REASON_WORKER_CRASHED = "worker_crashed"
REASON_CANCELLED = "cancelled"
REASON_PAUSED = "paused"
REASON_UNKNOWN = "unknown"

FAILURE_REASONS = (
    REASON_NO_SPACE,
    REASON_WRITER_FAILED,
    REASON_OUTPUT_EMPTY,
    REASON_FFMPEG_FAILED,
    REASON_MODEL_MISSING,
    REASON_REQUESTED_STAGE_FAILED,
    REASON_DECODE_FAILED,
    REASON_INPUT_MISSING,
    REASON_PERMISSION_DENIED,
    REASON_TIMED_OUT,
    REASON_FROZEN_MATTE,
    REASON_WORKER_CRASHED,
    REASON_CANCELLED,
    REASON_PAUSED,
    REASON_UNKNOWN,
)

# Human-readable labels for the batch report. These are report text, not GUI
# strings, so they stay English like every other batch-report column.
FAILURE_REASON_LABELS = {
    REASON_NO_SPACE: "Out of disk space",
    REASON_WRITER_FAILED: "Output writer failed",
    REASON_OUTPUT_EMPTY: "Output was empty",
    REASON_FFMPEG_FAILED: "FFmpeg failed",
    REASON_MODEL_MISSING: "Model or engine unavailable",
    REASON_REQUESTED_STAGE_FAILED: "Requested processing stage failed",
    REASON_DECODE_FAILED: "Source could not be decoded",
    REASON_INPUT_MISSING: "Source file missing",
    REASON_PERMISSION_DENIED: "Permission denied",
    REASON_TIMED_OUT: "Timed out",
    REASON_FROZEN_MATTE: "Frozen matte rejected",
    REASON_WORKER_CRASHED: "Worker stopped unexpectedly",
    REASON_CANCELLED: "Cancelled",
    REASON_PAUSED: "Paused",
    REASON_UNKNOWN: "Unclassified failure",
}

# backend.processor / backend.io reason codes, projected onto the closed set.
_REASON_CODES = {
    "corrupt_or_truncated": REASON_DECODE_FAILED,
    "invalid_container": REASON_DECODE_FAILED,
    "unreadable": REASON_DECODE_FAILED,
    "empty_file": REASON_DECODE_FAILED,
    "no_video_stream": REASON_DECODE_FAILED,
    "unsupported_codec": REASON_DECODE_FAILED,
    "invalid_dimensions": REASON_DECODE_FAILED,
    "no_decodable_frames": REASON_DECODE_FAILED,
    "truncated_decode": REASON_DECODE_FAILED,
    "decoder_seek_failed": REASON_DECODE_FAILED,
    "worker_timeout": REASON_TIMED_OUT,
    "worker_spawn_failed": REASON_WORKER_CRASHED,
    "output_integrity_failed": REASON_OUTPUT_EMPTY,
    "intermediate_writer_died": REASON_WRITER_FAILED,
    "intermediate_writer_timeout": REASON_WRITER_FAILED,
    "intermediate_writer_failed": REASON_WRITER_FAILED,
    "frame_write_failed": REASON_WRITER_FAILED,
}

_EXCEPTION_NAMES = {
    "FileNotFoundError": REASON_INPUT_MISSING,
    "PermissionError": REASON_PERMISSION_DENIED,
    "TimeoutError": REASON_TIMED_OUT,
    "TimeoutExpired": REASON_TIMED_OUT,
    "MediaWriteError": REASON_WRITER_FAILED,
    "OutputIntegrityError": REASON_OUTPUT_EMPTY,
    "MediaInputError": REASON_DECODE_FAILED,
    "FrozenMatteError": REASON_FROZEN_MATTE,
    "InpainterUnavailableError": REASON_MODEL_MISSING,
    "InterruptedError": REASON_CANCELLED,
    "ProcessingPaused": REASON_PAUSED,
}

# Ordered: the first marker found in the text wins, so specific phrases have
# to precede the generic ones they contain.
_TEXT_MARKERS = (
    ("insufficient disk space", REASON_NO_SPACE),
    ("no space left", REASON_NO_SPACE),
    ("disk full", REASON_NO_SPACE),
    ("out of disk", REASON_NO_SPACE),
    ("frozen matte", REASON_FROZEN_MATTE),
    ("timed out", REASON_TIMED_OUT),
    ("timeout", REASON_TIMED_OUT),
    ("wall-clock", REASON_TIMED_OUT),
    ("cancelled", REASON_CANCELLED),
    ("canceled", REASON_CANCELLED),
    ("paused", REASON_PAUSED),
    ("job worker", REASON_WORKER_CRASHED),
    ("stopped unexpectedly", REASON_WORKER_CRASHED),
    ("crash", REASON_WORKER_CRASHED),
    # The phrases backend.io actually emits from validate_video_output.
    ("output has no decodable video stream", REASON_OUTPUT_EMPTY),
    ("output duration", REASON_OUTPUT_EMPTY),
    ("output frame count", REASON_OUTPUT_EMPTY),
    ("output file is empty", REASON_OUTPUT_EMPTY),
    ("output is empty", REASON_OUTPUT_EMPTY),
    ("produced no output", REASON_OUTPUT_EMPTY),
    ("ffprobe", REASON_FFMPEG_FAILED),
    ("ffmpeg", REASON_FFMPEG_FAILED),
    ("could not write the output", REASON_WRITER_FAILED),
    ("writer", REASON_WRITER_FAILED),
    ("inpaint engine", REASON_MODEL_MISSING),
    ("no registered backend", REASON_MODEL_MISSING),
    ("model", REASON_MODEL_MISSING),
    ("permission denied", REASON_PERMISSION_DENIED),
    ("could not be found", REASON_INPUT_MISSING),
    ("no such file", REASON_INPUT_MISSING),
    ("decode", REASON_DECODE_FAILED),
    ("codec", REASON_DECODE_FAILED),
    ("corrupt", REASON_DECODE_FAILED),
    ("not a readable video", REASON_DECODE_FAILED),
)


def normalize_failure_reason(value: Optional[str]) -> str:
    """Return ``value`` when it is already a member of the closed set."""
    text = str(value or "").strip()
    return text if text in FAILURE_REASONS else REASON_NONE


def classify_failure_reason(
    *,
    exc: Optional[BaseException] = None,
    reason: Optional[str] = None,
    message: Optional[str] = None,
) -> str:
    """Project one failure onto the closed reason set.

    ``reason`` is the processor's free-form code, ``exc`` the raised
    exception, ``message`` the curated user-facing text. The most specific
    evidence wins; an unrecognised failure is ``unknown``, never blank.
    """
    already = normalize_failure_reason(reason)
    if already:
        return already
    code = str(reason or "").strip()
    if code in _REASON_CODES:
        return _REASON_CODES[code]
    if code.startswith("frozen_matte"):
        return REASON_FROZEN_MATTE
    if exc is not None:
        if isinstance(exc, RequestedStageError):
            cause = getattr(exc, "cause", None)
            if isinstance(cause, BaseException):
                cause_reason = classify_failure_reason(exc=cause)
                if cause_reason not in {REASON_UNKNOWN, REASON_NONE}:
                    return cause_reason
            failure_class = str(getattr(exc, "failure_class", "") or "")
            if failure_class in {
                "dependency_missing",
                "policy_blocked",
                "initialization_failed",
            }:
                return REASON_MODEL_MISSING
            return REASON_REQUESTED_STAGE_FAILED
        # errno 28 is ENOSPC on every platform Python supports.
        if getattr(exc, "errno", None) == 28:
            return REASON_NO_SPACE
        for klass in type(exc).__mro__:
            mapped = _EXCEPTION_NAMES.get(klass.__name__)
            if mapped:
                return mapped
    haystack = " ".join(
        part for part in (
            message,
            f"{type(exc).__name__}: {exc}" if exc is not None else "",
        ) if part
    ).lower()
    for marker, mapped in _TEXT_MARKERS:
        if marker in haystack:
            return mapped
    return REASON_UNKNOWN
