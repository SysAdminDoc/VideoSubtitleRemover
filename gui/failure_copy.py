"""Stable English queue-row failure copy.

QueueItem.message is persisted in queue_state.json and must stay in English
so a locale change does not orphan restored rows. Translation happens in
gui.utils.queue_message_text via CANONICAL_QUEUE_MESSAGES.
"""

from __future__ import annotations

from backend.i18n import N_


MSG_FAILED = N_("Processing failed")
MSG_MISSING = N_("The source file could not be found")
MSG_PERMISSION = N_("Permission denied while reading or writing a file")
MSG_TIMEOUT = N_("Processing timed out")
MSG_WRITE = N_("Could not write the output file")
MSG_INPAINTER = N_("The selected inpaint engine is not available")
MSG_PAUSED = N_("Paused at checkpoint")
MSG_CANCELLED = N_("Cancelled")
MSG_CRASHED = N_("The processing worker stopped unexpectedly")
MSG_WORKER_STOPPED = N_("The job worker stopped before finishing.")
MSG_COMPLETE = N_("Complete")
MSG_INITIALIZING = N_("Initializing...")
MSG_PREPARING_MODELS = N_("Preparing model downloads if needed...")
MSG_READY_RETRY = N_("Ready to retry with suggested settings")
MSG_STRIP_SOFT = N_("Removing embedded subtitle tracks...")
MSG_COPY_SOFT = N_("Copying embedded subtitle tracks...")

CANONICAL_FAILURE_MESSAGES = frozenset({
    MSG_FAILED,
    MSG_MISSING,
    MSG_PERMISSION,
    MSG_TIMEOUT,
    MSG_WRITE,
    MSG_INPAINTER,
    MSG_PAUSED,
    MSG_CANCELLED,
    MSG_CRASHED,
    MSG_WORKER_STOPPED,
    MSG_COMPLETE,
    MSG_INITIALIZING,
    MSG_PREPARING_MODELS,
    MSG_READY_RETRY,
    MSG_STRIP_SOFT,
    MSG_COPY_SOFT,
})


def user_facing_processing_error(exc: BaseException) -> str:
    """Return stable English queue-row text for an in-process exception."""
    if isinstance(exc, FileNotFoundError):
        return MSG_MISSING
    if isinstance(exc, PermissionError):
        return MSG_PERMISSION
    if isinstance(exc, TimeoutError):
        return MSG_TIMEOUT
    name = type(exc).__name__
    if name == "MediaWriteError":
        return MSG_WRITE
    if name == "InpainterUnavailableError":
        return MSG_INPAINTER
    if name == "ProcessingPaused":
        return MSG_PAUSED
    return MSG_FAILED


def user_facing_isolated_error(text: str) -> str:
    """Return stable English queue-row text for an isolated-worker outcome."""
    lowered = (text or "").lower()
    if "timed out" in lowered or "timeout" in lowered or "wall-clock" in lowered:
        return MSG_TIMEOUT
    if "cancelled" in lowered or "canceled" in lowered:
        return MSG_CANCELLED
    if "paused" in lowered:
        return MSG_PAUSED
    if "job worker" in lowered:
        return MSG_WORKER_STOPPED
    if "crash" in lowered or "exit" in lowered:
        return MSG_CRASHED
    if "filenotfound" in lowered:
        return MSG_MISSING
    if "permission" in lowered:
        return MSG_PERMISSION
    if "mediawrite" in lowered:
        return MSG_WRITE
    if "inpainterunavailable" in lowered:
        return MSG_INPAINTER
    return MSG_FAILED
