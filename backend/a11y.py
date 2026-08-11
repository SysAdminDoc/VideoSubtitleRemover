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
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Cached state across calls so we don't pay the import cost more than
# once.
_PROBED = False
_PROVIDER = None


def set_accessible_metadata(
    widget: Any,
    *,
    role: str,
    label: str,
    state: str = "",
    value: str = "",
    description: str = "",
) -> dict:
    """Attach a testable accessibility description to a custom Tk widget."""
    metadata = {
        "role": str(role or "").strip(),
        "label": str(label or "").strip(),
        "state": str(state or "").strip(),
        "value": str(value or "").strip(),
        "description": str(description or "").strip(),
    }
    try:
        setattr(widget, "_vsr_a11y", metadata)
    except Exception:
        pass
    return metadata


def accessible_metadata(widget: Any) -> dict:
    """Return the metadata set by set_accessible_metadata()."""
    metadata = getattr(widget, "_vsr_a11y", None)
    return dict(metadata) if isinstance(metadata, dict) else {}


def accessible_text(metadata: dict) -> str:
    """Format metadata into a concise screen-reader announcement."""
    parts = []
    for key in ("label", "role", "state", "value", "description"):
        value = str(metadata.get(key) or "").strip()
        if value:
            parts.append(value)
    return ". ".join(parts)


def announce_widget(widget: Any, importance: str = "normal") -> None:
    """Announce a custom-widget state snapshot when UIA is available."""
    text = accessible_text(accessible_metadata(widget))
    if text:
        announce(text, importance=importance)


# NotificationKind_Other -- the announcement is informational, not the
# completion of a specific UIA action the reader is tracking.
_NOTIFICATION_KIND_OTHER = 4
# NotificationProcessing_ImportantAll = 0 keeps urgent messages; _All = 2
# lets the reader coalesce routine chatter.
_PROCESSING_IMPORTANT_ALL = 0
_PROCESSING_ALL = 2


def _probe_provider() -> Optional[object]:
    """Bind the UIAutomationCore announcement entry points.

    Announcements are raised through the *provider* side of UI Automation
    (`UiaRaiseNotificationEvent`), not the client side: the client
    `IUIAutomationElement` has no RaiseNotificationEvent method at all, so
    the previous `CreateObject("CUIAutomation8")` client object could not
    have announced anything even had that (unregistered) class string
    resolved. Every window already has a default provider via
    `UiaHostProviderFromHwnd`, so this needs no COM registration, no
    typelib, and no third-party package -- only ctypes.

    Returns None on any failure (non-Windows, missing export); the GUI
    keeps working either way.
    """
    global _PROBED, _PROVIDER
    if _PROBED:
        return _PROVIDER
    _PROBED = True
    if sys.platform != "win32":
        return None
    try:
        import ctypes
        from ctypes import wintypes

        core = ctypes.windll.UIAutomationCore
        core.UiaHostProviderFromHwnd.argtypes = [
            wintypes.HWND, ctypes.POINTER(ctypes.c_void_p)
        ]
        core.UiaHostProviderFromHwnd.restype = ctypes.HRESULT
        core.UiaRaiseNotificationEvent.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_wchar_p,
            ctypes.c_wchar_p,
        ]
        core.UiaRaiseNotificationEvent.restype = ctypes.HRESULT
        core.UiaClientsAreListening.restype = ctypes.c_bool
        _PROVIDER = core
        logger.info("UIA notification provider ready")
        return _PROVIDER
    except Exception as exc:
        # INFO, not DEBUG: a silently missing announcer is exactly the
        # failure that went unnoticed for several releases.
        logger.info(f"UIA provider unavailable, announcements disabled: {exc}")
        return None


def _release_provider(provider) -> None:
    """Release the IRawElementProviderSimple returned by the host call."""
    import ctypes

    if not provider:
        return
    try:
        vtable = ctypes.cast(
            provider, ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p))
        ).contents
        release = ctypes.CFUNCTYPE(ctypes.c_ulong, ctypes.c_void_p)(vtable[2])
        release(provider)
    except Exception as exc:
        logger.debug(f"UIA provider release failed: {exc}")


def announce(text: str, importance: str = "normal") -> None:
    """Send `text` as a UIA notification so a screen reader speaks it.

    `importance` is "normal" or "high"; the latter maps to
    NotificationProcessing_ImportantAll so urgent messages (e.g. a fatal
    error) are not coalesced away. Silent when UIA is unavailable, so the
    function is safe to call from any GUI thread."""
    if not text:
        return
    core = _probe_provider()
    if core is None:
        return
    import ctypes

    provider = ctypes.c_void_p()
    try:
        if not core.UiaClientsAreListening():
            # No screen reader attached; skip the provider allocation.
            return
        hwnd = _root_hwnd()
        if hwnd is None:
            return
        if core.UiaHostProviderFromHwnd(hwnd, ctypes.byref(provider)) != 0:
            return
        processing = (
            _PROCESSING_IMPORTANT_ALL if importance == "high"
            else _PROCESSING_ALL
        )
        core.UiaRaiseNotificationEvent(
            provider, _NOTIFICATION_KIND_OTHER, processing, str(text), "VSR"
        )
    except Exception as exc:
        logger.debug(f"UIA announce failed: {exc}")
    finally:
        _release_provider(provider)


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
