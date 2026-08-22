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
import time
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
    help_text: str = "",
    live_region: bool = False,
) -> dict:
    """Attach a testable accessibility description to a custom Tk widget.

    RM-282: the same metadata is pushed onto the widget's own HWND with
    MSAA Dynamic Annotation, so a screen reader reads a role and a name
    instead of an anonymous pane. Annotation failure is silent -- the
    in-process metadata is the contract, the HWND push is a bonus.
    """
    metadata = {
        "role": str(role or "").strip(),
        "label": str(label or "").strip(),
        "state": str(state or "").strip(),
        "value": str(value or "").strip(),
        "description": str(description or "").strip(),
        "help": str(help_text or "").strip(),
    }
    try:
        setattr(widget, "_vsr_a11y", metadata)
    except Exception:
        pass
    annotate_widget(widget, live_region=live_region)
    return metadata


def accessible_metadata(widget: Any) -> dict:
    """Return the metadata set by set_accessible_metadata()."""
    metadata = getattr(widget, "_vsr_a11y", None)
    return dict(metadata) if isinstance(metadata, dict) else {}


def set_accessible_subtree_visible(widget: Any, visible: bool) -> None:
    """Keep a disclosed Tk subtree in both the control view and tab order.

    Tk normally removes an unpacked parent from painting, but focusable child
    HWNDs can remain discoverable to Windows accessibility clients. Preserve
    each widget's original ``takefocus`` value while the subtree is collapsed
    so keyboard traversal and UI Automation expose the same disclosure state.
    """
    pending = [widget]
    while pending:
        current = pending.pop()
        try:
            pending.extend(current.winfo_children())
        except Exception:
            pass
        try:
            setattr(current, "_vsr_a11y_control_view", bool(visible))
        except Exception:
            pass
        if visible:
            original = getattr(current, "_vsr_a11y_saved_takefocus", None)
            if original is None:
                continue
            if getattr(current, "enabled", True) is False:
                original = 0
            try:
                current.configure(takefocus=original)
            except Exception:
                pass
            try:
                delattr(current, "_vsr_a11y_saved_takefocus")
            except Exception:
                pass
            continue
        if hasattr(current, "_vsr_a11y_saved_takefocus"):
            continue
        try:
            original = current.cget("takefocus")
            setattr(current, "_vsr_a11y_saved_takefocus", original)
            current.configure(takefocus=0)
        except Exception:
            pass


def accessible_text(metadata: dict) -> str:
    """Format metadata into a concise screen-reader announcement."""
    parts = []
    for key in ("label", "role", "state", "value", "description", "help"):
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
# NotificationProcessing_ImportantAll = 0 keeps urgent messages.
# NotificationProcessing_MostRecent = 3 lets the reader drop a queued routine
# message when a newer one arrives, which is what a status line wants.
# _All = 2 is the deliver-everything value and was the wrong default here.
_PROCESSING_IMPORTANT_ALL = 0
_PROCESSING_MOST_RECENT = 3


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
            else _PROCESSING_MOST_RECENT
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


# -- RM-282: MSAA Dynamic Annotation on the widget's own HWND ---------------
#
# Tk 8.6.15 has no `tk accessible` (TIP 733 lands in Tk 9.1), and a full UI
# Automation provider means answering WM_GETOBJECT with a custom
# IRawElementProviderSimple -- a separate, much larger project. Dynamic
# Annotation sits in between: every Tk widget on Windows is a real HWND, so
# IAccPropServices::SetHwndProp can give that HWND a role, a name, a value
# and help text without registering anything. The MSAA-to-UIA bridge then
# surfaces those to NVDA and Narrator.
#
# LiveSetting is deliberately absent here: it is a UIA-only property and the
# MSAA bridge cannot carry it. A widget marked `live_region=True` gets its
# value spoken through the existing UIA notification path instead, which is
# what a reader actually consumes for a changing status line.

_ACC_PROBED = False
_ACC_SERVICES = None

# oleacc.h role constants, mapped from this app's role vocabulary.
_MSAA_ROLES = {
    "button": 0x2B,          # ROLE_SYSTEM_PUSHBUTTON
    "checkbox": 0x2C,        # ROLE_SYSTEM_CHECKBUTTON
    "radio button": 0x2D,    # ROLE_SYSTEM_RADIOBUTTON
    "slider": 0x33,          # ROLE_SYSTEM_SLIDER
    "progressbar": 0x30,     # ROLE_SYSTEM_PROGRESSBAR
    "progress": 0x30,
    "status": 0x17,          # ROLE_SYSTEM_STATUSBAR
    "heading": 0x29,         # ROLE_SYSTEM_STATICTEXT
    "text": 0x2A,            # ROLE_SYSTEM_TEXT
    "text box": 0x2A,
    "search box": 0x2A,
    "numeric input": 0x34,   # ROLE_SYSTEM_SPINBUTTON
    "coordinate input": 0x34,
    "combo box": 0x2E,       # ROLE_SYSTEM_COMBOBOX
    "queue item": 0x22,      # ROLE_SYSTEM_LISTITEM
    "notification": 0x08,    # ROLE_SYSTEM_ALERT
    "region editor canvas": 0x28,   # ROLE_SYSTEM_GRAPHIC
    "mask painting canvas": 0x28,
    "link": 0x1E,            # ROLE_SYSTEM_LINK
    "dialog": 0x12,          # ROLE_SYSTEM_DIALOG
    "drop target": 0x14,     # ROLE_SYSTEM_GROUPING
    "group": 0x14,
}
_MSAA_ROLE_DEFAULT = 0x14    # ROLE_SYSTEM_GROUPING

_OBJID_CLIENT = 0xFFFFFFFC
_CHILDID_SELF = 0
_VT_I4 = 3

# oleacc.h MSAAPROPID GUIDs, as (Data1, Data2, Data3, Data4-bytes).
_PROPID_SPECS = {
    "name": (0x608D3DF8, 0x8128, 0x4AA7,
             (0xA4, 0x28, 0xF5, 0x5E, 0x49, 0x26, 0x72, 0x91)),
    "value": (0x123FE443, 0x211A, 0x4615,
              (0x95, 0x27, 0xC4, 0x5A, 0x7E, 0x93, 0x71, 0x7A)),
    "description": (0x4D48DFE4, 0xBD3F, 0x491F,
                    (0xA6, 0x48, 0x49, 0x2D, 0x6F, 0x20, 0xC5, 0x88)),
    "role": (0xCB905FF2, 0x7BD1, 0x4C05,
             (0xB3, 0xC8, 0xE6, 0xC2, 0x41, 0x36, 0x4D, 0x70)),
    "help": (0xC831E11F, 0x44DB, 0x4A99,
             (0x97, 0x68, 0xCB, 0x8F, 0x97, 0x8B, 0x72, 0x31)),
    "rolemap": (0xF79ACDA2, 0x140D, 0x4FE6,
                (0x89, 0x14, 0x20, 0x84, 0x76, 0x32, 0x82, 0x69)),
}

_CLSID_ACC_PROP_SERVICES = (0xB5F8350B, 0x0548, 0x48B1,
                            (0xA6, 0xEE, 0x88, 0xBD, 0x00, 0xB4, 0xA5, 0xE7))
_IID_IACC_PROP_SERVICES = (0x6E26E776, 0x04F0, 0x495D,
                           (0x80, 0xE4, 0x33, 0x30, 0x35, 0x2E, 0x31, 0x69))


_ACC_TYPES = None


def _acc_types():
    """Return the ctypes structures the annotation calls need.

    Built once and cached: ctypes compares argument types by class
    identity, so handing a freshly-built GUID class to a prototype
    declared with a different one fails with "expected GUID instead of
    GUID".
    """
    global _ACC_TYPES
    if _ACC_TYPES is not None:
        return _ACC_TYPES
    import ctypes

    class GUID(ctypes.Structure):
        _fields_ = [
            ("Data1", ctypes.c_ulong),
            ("Data2", ctypes.c_ushort),
            ("Data3", ctypes.c_ushort),
            ("Data4", ctypes.c_ubyte * 8),
        ]

    class VARIANT(ctypes.Structure):
        # VARTYPE + three reserved WORDs, then the union. The union holds a
        # BRECORD (two pointers), so sizing it from the pointer width gives
        # the right total on both 32- and 64-bit builds.
        _fields_ = [
            ("vt", ctypes.c_ushort),
            ("wReserved1", ctypes.c_ushort),
            ("wReserved2", ctypes.c_ushort),
            ("wReserved3", ctypes.c_ushort),
            ("data", ctypes.c_byte * (2 * ctypes.sizeof(ctypes.c_void_p))),
        ]

    _ACC_TYPES = (GUID, VARIANT)
    return _ACC_TYPES


def _make_guid(spec) -> Any:
    import ctypes

    GUID, _ = _acc_types()
    data1, data2, data3, data4 = spec
    return GUID(data1, data2, data3, (ctypes.c_ubyte * 8)(*data4))


def _acc_prop_services():
    """Return the cached IAccPropServices pointer, or None."""
    global _ACC_PROBED, _ACC_SERVICES
    if _ACC_PROBED:
        return _ACC_SERVICES
    _ACC_PROBED = True
    if sys.platform != "win32":
        return None
    try:
        import ctypes

        ole32 = ctypes.windll.ole32
        # S_FALSE (already initialised) and RPC_E_CHANGED_MODE (Tk got
        # there first with a different model) are both fine: some other
        # component already set the thread's apartment up for us.
        ole32.CoInitializeEx(None, 0x2)
        clsid = _make_guid(_CLSID_ACC_PROP_SERVICES)
        iid = _make_guid(_IID_IACC_PROP_SERVICES)
        services = ctypes.c_void_p()
        hr = ole32.CoCreateInstance(
            ctypes.byref(clsid), None, 1,  # CLSCTX_INPROC_SERVER
            ctypes.byref(iid), ctypes.byref(services),
        )
        if hr != 0 or not services:
            logger.info(
                f"IAccPropServices unavailable (hr=0x{hr & 0xFFFFFFFF:08x}); "
                "custom controls stay unannotated"
            )
            return None
        _ACC_SERVICES = services
        logger.info("MSAA dynamic annotation ready")
        return _ACC_SERVICES
    except Exception as exc:
        logger.info(f"MSAA dynamic annotation unavailable: {exc}")
        return None


def _acc_method(services, index, restype, argtypes):
    import ctypes

    vtable = ctypes.cast(
        services, ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p))
    ).contents
    prototype = ctypes.WINFUNCTYPE(restype, *argtypes)
    return prototype(vtable[index])


def annotate_hwnd(
    hwnd: int,
    *,
    role: str = "",
    name: str = "",
    value: str = "",
    description: str = "",
    help_text: str = "",
) -> bool:
    """Push MSAA properties onto ``hwnd``. True when anything was set."""
    services = _acc_prop_services()
    if services is None or not hwnd:
        return False
    try:
        import ctypes
        from ctypes import wintypes

        GUID, VARIANT = _acc_types()
        set_str = _acc_method(
            services, 7, ctypes.HRESULT,
            [ctypes.c_void_p, wintypes.HWND, wintypes.DWORD, wintypes.DWORD,
             GUID, ctypes.c_wchar_p],
        )
        applied = False
        strings = (
            ("name", name),
            ("value", value),
            ("description", description),
            ("help", help_text),
        )
        for key, text in strings:
            # Empty strings are pushed too: a control whose description was
            # cleared must stop reading the stale one.
            if _acc_call(set_str, services, hwnd,
                         _make_guid(_PROPID_SPECS[key]), str(text)):
                applied = True
        if role:
            set_prop = _acc_method(
                services, 6, ctypes.HRESULT,
                [ctypes.c_void_p, wintypes.HWND, wintypes.DWORD,
                 wintypes.DWORD, GUID, VARIANT],
            )
            var = VARIANT()
            var.vt = _VT_I4
            ctypes.memmove(
                ctypes.byref(var, VARIANT.data.offset),
                ctypes.byref(ctypes.c_int32(
                    _MSAA_ROLES.get(role, _MSAA_ROLE_DEFAULT))),
                4,
            )
            if _acc_call(set_prop, services, hwnd,
                         _make_guid(_PROPID_SPECS["role"]), var):
                applied = True
            elif _acc_call(set_str, services, hwnd,
                           _make_guid(_PROPID_SPECS["rolemap"]), str(role)):
                # Older shells reject the VARIANT role; the localized role
                # string is read by NVDA and Narrator just the same.
                applied = True
        return applied
    except Exception as exc:
        logger.debug(f"HWND annotation failed: {exc}")
        return False


def _acc_call(method, services, hwnd, prop, payload) -> bool:
    """Invoke one Set*Prop* method, treating any HRESULT failure as False."""
    try:
        method(services, hwnd, _OBJID_CLIENT, _CHILDID_SELF, prop, payload)
    except OSError as exc:
        logger.debug(f"HWND annotation call rejected: {exc}")
        return False
    return True


def annotate_widget(widget: Any, *, live_region: bool = False) -> bool:
    """Annotate ``widget``'s own HWND from its stored metadata.

    Re-annotating on every hover and focus change would mean a COM call per
    mouse move, so the last applied snapshot is cached on the widget and an
    unchanged one is skipped.
    """
    metadata = accessible_metadata(widget)
    # A tooltip attaches help before the widget has any other metadata, so a
    # bare tooltip is still worth annotating on its own.
    tooltip_help = str(getattr(widget, "_vsr_a11y_help", "") or "")
    if not metadata and not tooltip_help:
        return False
    snapshot = (
        metadata.get("role", ""),
        metadata.get("label", ""),
        metadata.get("state", ""),
        metadata.get("value", ""),
        metadata.get("description", ""),
        metadata.get("help", "") or tooltip_help,
    )
    if getattr(widget, "_vsr_a11y_hwnd_applied", None) == snapshot:
        return False
    try:
        hwnd = int(widget.winfo_id())
    except Exception:
        return False
    role, label, state, value, description, help_text = snapshot
    # MSAA has no separate state string, so it rides along with the value --
    # that is the field a reader re-reads when the control changes.
    spoken_value = "; ".join(part for part in (value, state) if part)
    applied = annotate_hwnd(
        hwnd,
        role=role,
        name=label,
        value=spoken_value,
        description=description,
        help_text=help_text,
    )
    try:
        setattr(widget, "_vsr_a11y_hwnd_applied", snapshot)
    except Exception:
        pass
    if live_region:
        # The value alone, never the state: a footer whose tone token is
        # "info" must not read out "...; info" after every message.
        announce_live(value)
    return applied


# A live region that speaks every progress tick is worse than one that says
# nothing -- the backend emits a new "Processing frame N/M" string per batch,
# so an unthrottled announcement is hundreds of interruptions per file.
LIVE_REGION_MIN_INTERVAL_SECONDS = 2.0
_live_last_text = ""
_live_last_time = 0.0


def announce_live(text: str) -> None:
    """Announce a changing status value, throttled and de-duplicated."""
    global _live_last_text, _live_last_time

    message = str(text or "").strip()
    if not message or message == _live_last_text:
        return
    now = time.monotonic()
    if now - _live_last_time < LIVE_REGION_MIN_INTERVAL_SECONDS:
        return
    _live_last_text = message
    _live_last_time = now
    announce(message)


def reset_live_region_throttle() -> None:
    """Forget the last live announcement (used by tests)."""
    global _live_last_text, _live_last_time

    _live_last_text = ""
    _live_last_time = 0.0


def set_tooltip_help(widget: Any, text: str) -> None:
    """Record a widget's tooltip text as its accessible help.

    A sighted user gets the tooltip on hover; without this a screen-reader
    user gets nothing at all, because the tooltip is a separate Toplevel
    that never takes focus.
    """
    help_text = str(text or "").strip()
    if not help_text:
        return
    try:
        setattr(widget, "_vsr_a11y_help", help_text)
        setattr(widget, "_vsr_a11y_hwnd_applied", None)
    except Exception:
        return
    annotate_widget(widget)
