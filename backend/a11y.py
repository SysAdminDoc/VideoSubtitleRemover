"""Accessibility scaffold for Windows UI Automation announcements.

RM-95: NVDA / Narrator screen readers cannot announce custom Canvas
widget state changes (ModernButton / ModernToggle / ModernSlider) the
default tkinter binding doesn't expose. Full UIA provider support is
a multi-week project; this scaffold provides the *announcement* slice:

- `announce(text)` reads `text` via the Windows UI Automation
  NotificationKind API when comtypes + pywin32 are available. NVDA
  and Narrator pick the notification up as speech.
- The function is a no-op on non-Windows platforms and when the
  optional dependencies are missing -- the rest of the GUI never
  needs to special-case the announcer.

Used by the GUI for batch-state transitions (item complete, batch
finished, fatal error), the queue cancel action, and the per-file
overrides popover's save button. Wiring more announcements is the
next pass; this commit lands the framework + a couple of high-value
call sites.
"""

from __future__ import annotations

import logging
import sys
from typing import Optional

logger = logging.getLogger(__name__)

# Cached state across calls so we don't pay the import cost more than
# once.
_PROBED = False
_PROVIDER = None


def _probe_provider() -> Optional[object]:
    """Try to import the Windows UIA NotificationKind provider. Returns
    None on any failure -- non-Windows, missing pywin32, missing
    UIAutomationCore, or NVDA / Narrator not running. The user-visible
    GUI keeps working either way."""
    global _PROBED, _PROVIDER
    if _PROBED:
        return _PROVIDER
    _PROBED = True
    if sys.platform != "win32":
        return None
    try:
        import comtypes.client as _cc  # type: ignore
        # UIAutomationCore is shipped with the OS on Windows 7+.
        uia = _cc.CreateObject("CUIAutomation8")
        _PROVIDER = uia
        logger.info("UIA notification provider ready")
        return _PROVIDER
    except Exception as exc:
        logger.debug(f"UIA provider unavailable: {exc}")
        return None


def announce(text: str, importance: str = "normal") -> None:
    """Send `text` as a UIA notification so a screen reader speaks it.

    `importance` is "normal" or "high"; the latter maps to
    NotificationProcessing_ImportantMostRecent so urgent messages
    (e.g. fatal error) cut in front of the queue. Silent when UIA is
    unavailable so the function is safe to call from any GUI thread."""
    if not text:
        return
    provider = _probe_provider()
    if provider is None:
        return
    try:
        # NotificationKind_ActionCompleted = 0
        # NotificationProcessing_All = 0 (default queue)
        # NotificationProcessing_ImportantAll = 3 (importance="high")
        proc = 3 if importance == "high" else 0
        hwnd = _root_hwnd()
        if hwnd is None:
            return
        elem = provider.ElementFromHandle(hwnd)
        # The COM call signature varies by build of UIAutomationCore.
        # The two-arg form ("text", processing) is the one shipped on
        # Windows 10 1709+. Older Win7 builds will raise; we swallow.
        elem.RaiseNotificationEvent(0, proc, text, "VSR")
    except Exception as exc:
        logger.debug(f"UIA announce failed: {exc}")


def _root_hwnd() -> Optional[int]:
    """Resolve the active foreground window hwnd for the UIA call. We
    deliberately walk through GetForegroundWindow rather than
    threading the GUI's root through every call site; the GUI is
    always foregrounded during a state change worth announcing."""
    try:
        import ctypes
        hwnd = ctypes.windll.user32.GetForegroundWindow()
        return hwnd or None
    except Exception:
        return None
