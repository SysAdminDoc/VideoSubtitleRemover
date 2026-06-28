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
import traceback
from typing import Optional

logger = logging.getLogger(__name__)

_INSTALLED = False
_ORIG_EXCEPTHOOK = sys.excepthook


def _path_scrub(text: str) -> str:
    """Strip Windows / POSIX absolute paths out of traceback text so
    the upload doesn't carry filesystem layout information. Replaces
    `C:\\Users\\xxx\\repos\\VSR\\...` with `<path>\\...` and
    `/home/xxx/...` with `<path>/...`."""
    text = re.sub(r"[A-Za-z]:\\(?:[^\\:]+\\)*", "<path>\\\\", text)
    text = re.sub(r"[A-Za-z]:/(?:[^/:]+/)*", "<path>/", text)
    text = re.sub(r"\\\\\\\\[^\\:]+\\\\(?:[^\\:]+\\\\)*", "<path>\\\\", text)
    text = re.sub(r"/(?:home|Users|var|tmp)/(?:[^/]+/)+", "<path>/", text)
    return text


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
            before_send=_before_send,
        )
        _INSTALLED = True
        logger.info("Opt-in crash reporting active (GlitchTip)")
        return True
    except Exception as exc:
        logger.warning(f"GlitchTip init failed: {exc}")
        return False


def _before_send(event: dict, hint: dict) -> Optional[dict]:
    """Scrub paths out of every string field in the event."""
    try:
        for key in list(event.keys()):
            value = event[key]
            if isinstance(value, str):
                event[key] = _path_scrub(value)
        # Walk known stack-frame structures.
        for exc in event.get("exception", {}).get("values", []):
            for frame in exc.get("stacktrace", {}).get("frames", []):
                if "abs_path" in frame:
                    frame["abs_path"] = _path_scrub(str(frame["abs_path"]))
                if "filename" in frame:
                    frame["filename"] = _path_scrub(str(frame["filename"]))
                # Drop locals -- they can carry frame buffers / mask
                # arrays we don't want uploaded.
                frame.pop("vars", None)
    except Exception:
        pass
    return event
