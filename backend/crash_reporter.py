"""Opt-in crash reporting (RM-52).

GlitchTip is a self-hosted Sentry-API-compatible service that runs in
~512 MB RAM. When the user has set both:

- `VSR_GLITCHTIP_DSN` (the Sentry-format DSN of the user's own
  GlitchTip instance), AND
- `VSR_CRASH_REPORTS=1` (a second explicit consent gate so an
  accidental DSN leak from a CI environment does not silently start
  shipping stack traces),

we install a global excepthook that ships:
- Exception type + message
- Stack trace WITH local paths replaced by "<path>" so absolute
  Windows paths never leak
- Python version + platform
- VSR APP_VERSION

We deliberately DO NOT ship:
- Frame contents
- File names / paths the user processed
- OCR text (privacy)
- The full Python environment / installed packages

Strict opt-in is the project philosophy. Default off; the scaffold lands so a user who wants
crash visibility can wire it without monkey-patching.
"""

from __future__ import annotations

import logging
import os
import re
import sys
from typing import Optional

logger = logging.getLogger(__name__)

_INSTALLED = False
_ORIG_EXCEPTHOOK = sys.excepthook


MAX_VALUE_CHARS = 400
MAX_FRAMES = 50

_MEDIA_SUFFIXES = (
    "mp4", "mkv", "mov", "avi", "webm", "m4v", "mpg", "mpeg", "wmv", "flv",
    "ts", "m2ts", "png", "jpg", "jpeg", "webp", "bmp", "tif", "tiff", "gif",
    "srt", "vtt", "ass", "ssa", "sub", "idx", "wav", "mp3", "aac", "flac",
    "onnx", "pth", "pt", "safetensors", "json", "csv", "log", "txt",
)

# Ordered: file URLs and UNC roots first, then drive-letter and POSIX paths.
# Each pattern consumes the leaf name too -- a basename such as
# "family holiday.mp4" is exactly the kind of thing this must not upload.
_PATH_PATTERNS = (
    re.compile(r"file:/{2,3}[^\s'\"<>|]*", re.IGNORECASE),
    re.compile(r"\\\\[^\s\\/:*?\"<>|]+(?:[\\/][^\s\\/:*?\"<>|]+)*"),
    re.compile(r"[A-Za-z]:[\\/][^\s:*?\"<>|]*"),
    re.compile(
        r"/(?:home|Users|users|var|tmp|mnt|media|opt|srv|root|Volumes)"
        r"/[^\s'\"<>|]*"
    ),
)

_BASENAME_PATTERN = re.compile(
    r"\b[\w~ .()\[\]#&+,'-]+\.(?:" + "|".join(_MEDIA_SUFFIXES) + r")\b",
    re.IGNORECASE,
)

# Directory-only variants: strip the tree but keep the leaf name. Used by the
# local support bundle, which the user reads and chooses to share, and where
# the filename is what makes a log line useful.
_DIR_PATTERNS = (
    (re.compile(r"file:/{2,3}(?:[^\s'\"<>|/]+/)*", re.IGNORECASE), "<path>/"),
    (re.compile(r"\\\\[^\s\\/:*?\"<>|]+\\(?:[^\s\\/:*?\"<>|]+\\)*"), "<path>\\\\"),
    (re.compile(r"[A-Za-z]:\\(?:[^\s\\/:*?\"<>|]+\\)*"), "<path>\\\\"),
    (re.compile(r"[A-Za-z]:/(?:[^\s\\/:*?\"<>|]+/)*"), "<path>/"),
    (
        re.compile(
            r"/(?:home|Users|users|var|tmp|mnt|media|opt|srv|root|Volumes)"
            r"/(?:[^\s'\"<>|/]+/)*"
        ),
        "<path>/",
    ),
)


def _path_scrub(text: str, *, keep_basename: bool = False) -> str:
    """Remove filesystem identifiers from a string.

    RM-145: for crash telemetry (the default) the previous version only
    replaced the *directory* part of a path, so `C:\\videos\\family holiday.mp4`
    still uploaded the processed filename despite the documented privacy
    contract. Windows drive paths, UNC shares, ``file://`` URLs, POSIX
    home/temp paths, and bare media basenames are now replaced outright.

    ``keep_basename=True`` selects the older directory-only behaviour for the
    local support bundle, where the leaf filename is what makes a log line
    useful and the user reviews the archive before sharing it.
    """
    if keep_basename:
        for pattern, replacement in _DIR_PATTERNS:
            text = pattern.sub(replacement, text)
        return text
    for pattern in _PATH_PATTERNS:
        text = pattern.sub("<path>", text)
    return _BASENAME_PATTERN.sub("<file>", text)


def is_enabled() -> bool:
    return (
        os.environ.get("VSR_CRASH_REPORTS", "").strip().lower() in {"1", "true", "yes", "on"}
        and bool(os.environ.get("VSR_GLITCHTIP_DSN", "").strip())
    )


def install() -> bool:
    """Install the excepthook. Returns True when the hook is active,
    False when the user has not opted in or the optional `sentry-sdk`
    package is missing."""
    global _INSTALLED
    if _INSTALLED:
        return True
    if not is_enabled():
        return False
    dsn = os.environ.get("VSR_GLITCHTIP_DSN", "").strip()
    try:
        import sentry_sdk  # type: ignore
    except ImportError:
        logger.info(
            "VSR_GLITCHTIP_DSN is set but sentry-sdk is not installed. "
            "`pip install sentry-sdk` to enable crash reporting."
        )
        return False
    try:
        sentry_sdk.init(
            dsn=dsn,
            traces_sample_rate=0.0,  # we only ship exceptions, never traces
            release=os.environ.get("VSR_APP_VERSION", ""),
            send_default_pii=False,
            attach_stacktrace=True,
            # RM-145: nothing that could carry user data is collected in the
            # first place; before_send then rebuilds from an allowlist.
            max_breadcrumbs=0,
            include_local_variables=False,
            server_name=None,
            default_integrations=False,
            before_send=_before_send,
            before_send_transaction=lambda *args: None,
        )
        _INSTALLED = True
        logger.info("Opt-in crash reporting active (GlitchTip)")
        return True
    except Exception as exc:
        logger.warning(f"GlitchTip init failed: {exc}")
        return False


def _scrub_tree(value):
    """Recursively replace filesystem identifiers in every nested string."""
    if isinstance(value, str):
        return _path_scrub(value)
    if isinstance(value, dict):
        return {k: _scrub_tree(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_scrub_tree(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_scrub_tree(v) for v in value)
    return value


def _short_text(value, limit: int = MAX_VALUE_CHARS) -> str:
    if not isinstance(value, str):
        value = "" if value is None else str(value)
    return _path_scrub(value)[:limit]


def _safe_identifier(value) -> str:
    """Keep only dotted module/function identifiers; drop anything else."""
    if not isinstance(value, str):
        return ""
    candidate = value.strip()
    if not candidate or len(candidate) > 200:
        return ""
    return candidate if re.fullmatch(r"[\w.<>]+", candidate) else ""


def _minimal_frame(frame) -> dict:
    if not isinstance(frame, dict):
        return {}
    payload = {
        "module": _safe_identifier(frame.get("module")),
        "function": _safe_identifier(frame.get("function")),
        "in_app": bool(frame.get("in_app")),
    }
    lineno = frame.get("lineno")
    if isinstance(lineno, int):
        payload["lineno"] = lineno
    return payload


def _minimal_exception(value) -> dict:
    if not isinstance(value, dict):
        return {}
    stacktrace = value.get("stacktrace")
    frames = []
    if isinstance(stacktrace, dict):
        raw = stacktrace.get("frames")
        if isinstance(raw, list):
            frames = [_minimal_frame(item) for item in raw[-MAX_FRAMES:]]
    mechanism = value.get("mechanism")
    payload = {
        "type": _safe_identifier(value.get("type")) or "Exception",
        "value": _short_text(value.get("value")),
        "module": _safe_identifier(value.get("module")),
        "stacktrace": {"frames": frames},
    }
    if isinstance(mechanism, dict):
        payload["mechanism"] = {
            "type": _safe_identifier(mechanism.get("type")) or "excepthook",
            "handled": bool(mechanism.get("handled")),
        }
    return payload


def _minimal_contexts(contexts) -> dict:
    """Python version and OS name only -- never hostname, user, or device."""
    payload = {}
    if not isinstance(contexts, dict):
        return payload
    runtime = contexts.get("runtime")
    if isinstance(runtime, dict):
        payload["runtime"] = {
            "name": _short_text(runtime.get("name"), 40),
            "version": _short_text(runtime.get("version"), 40),
        }
    operating_system = contexts.get("os")
    if isinstance(operating_system, dict):
        payload["os"] = {
            "name": _short_text(operating_system.get("name"), 40),
            "version": _short_text(operating_system.get("version"), 40),
        }
    return payload


def build_minimal_event(event) -> Optional[dict]:
    """Rebuild a report from an allowlist, or return None if that fails.

    RM-145: the previous hook scrubbed the incoming event in place and returned
    the *original* on any exception, so a scrubber failure shipped unredacted
    data. Nothing is copied across unless it is explicitly listed here, and
    breadcrumbs, extra data, request payloads, frame locals, filenames, tags,
    loaded modules, server name, and user data are simply never carried.
    """
    if not isinstance(event, dict):
        return None
    values = []
    exception = event.get("exception")
    if isinstance(exception, dict):
        raw = exception.get("values")
        if isinstance(raw, list):
            values = [_minimal_exception(item) for item in raw]
    payload = {
        "event_id": _safe_identifier(event.get("event_id")) or "",
        "timestamp": event.get("timestamp"),
        "platform": _short_text(event.get("platform"), 20) or "python",
        "level": _short_text(event.get("level"), 20) or "error",
        "logger": _safe_identifier(event.get("logger")),
        "release": _short_text(event.get("release"), 40),
        "environment": _short_text(event.get("environment"), 40),
        "contexts": _minimal_contexts(event.get("contexts")),
    }
    if values:
        payload["exception"] = {"values": values}
    return payload


def _before_send(event: dict, hint: dict) -> Optional[dict]:
    """Build an allowlisted minimal event; drop the report on any failure."""
    try:
        return build_minimal_event(event)
    except Exception:
        # Fail closed: an unsendable report is always better than one that
        # might carry a user's filenames, locals, or OCR text.
        logger.warning("Crash report dropped: scrubbing failed", exc_info=True)
        return None
