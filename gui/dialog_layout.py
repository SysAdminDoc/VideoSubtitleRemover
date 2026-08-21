"""Shared work-area fitting and internal scrolling for modal dialogs.

RM-148: the main workbench is responsive, but the onboarding modal, the region
editor, and the mask-correction editor were fixed-size and non-resizable. At
125-200% text scale their content grew past the screen work area, pushing the
primary actions off the bottom with no way to reach them.

Every major dialog now builds into a scrollable body and is clamped to the
usable work area, so a control can always be reached by scrolling or by
keyboard even when the dialog would otherwise be taller than the screen.
"""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk

from gui.theme import Theme
from gui.utils import desktop_bounds


# Reserve room for the taskbar / dock and the window chrome. Deliberately
# conservative: a dialog that is slightly smaller than it could be is fine, one
# that runs under the taskbar is not.
WORK_AREA_WIDTH_FRACTION = 0.96
WORK_AREA_HEIGHT_FRACTION = 0.90
MIN_DIALOG_WIDTH = 320
MIN_DIALOG_HEIGHT = 220


def work_area(root) -> tuple[int, int]:
    """Usable width/height for a dialog on the screen holding ``root``."""
    try:
        screen_w = int(root.winfo_screenwidth())
        screen_h = int(root.winfo_screenheight())
    except Exception:
        return (1024, 720)
    override = getattr(root, "_vsr_work_area_override", None)
    if isinstance(override, (tuple, list)) and len(override) == 2:
        try:
            return (max(MIN_DIALOG_WIDTH, int(override[0])),
                    max(MIN_DIALOG_HEIGHT, int(override[1])))
        except (TypeError, ValueError):
            pass
    return (
        max(MIN_DIALOG_WIDTH, int(screen_w * WORK_AREA_WIDTH_FRACTION)),
        max(MIN_DIALOG_HEIGHT, int(screen_h * WORK_AREA_HEIGHT_FRACTION)),
    )


def register_wheel_surface(canvas) -> None:
    """Mark a canvas as a scroll target for the application wheel router."""
    canvas._vsr_wheel_surface = True
    ensure_wheel_router(canvas)


def ensure_wheel_router(widget) -> None:
    """Install the application-wide wheel router once per interpreter.

    On Windows the wheel event goes to the widget under the pointer, so a
    binding on the canvas leaves every card, toggle, and label inside it a
    dead zone, and per-child rebinding never keeps up. One ``bind_all``
    handler walks up from ``event.widget`` to the nearest registered
    scrollable surface instead. The old dialog variant bound and unbound on
    the canvas's Enter/Leave, so crossing from the canvas onto its own body
    frame turned scrolling off.
    """
    try:
        root = widget.winfo_toplevel().nametowidget(".")
    except (tk.TclError, KeyError):
        root = widget.winfo_toplevel()
    if getattr(root, "_vsr_wheel_router_installed", False):
        return

    def _scroll(canvas, event):
        try:
            if canvas.yview() == (0.0, 1.0):
                # Everything is visible; let the event fall through so a
                # parent surface (if any) can take it.
                return None
        except tk.TclError:
            return None
        delta = getattr(event, "delta", 0)
        if delta:
            step = int(-1 * (delta / 120)) or (-1 if delta > 0 else 1)
        elif getattr(event, "num", 0) == 4:
            step = -1
        elif getattr(event, "num", 0) == 5:
            step = 1
        else:
            return None
        canvas.yview_scroll(step, "units")
        return "break"

    def _route(event):
        widget = event.widget
        if isinstance(widget, str):
            return None
        while widget is not None:
            if getattr(widget, "_vsr_wheel_surface", False):
                outcome = _scroll(widget, event)
                if outcome is not None:
                    return outcome
            widget = getattr(widget, "master", None)
        return None

    for sequence in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
        root.bind_all(sequence, _route, add="+")
    root._vsr_wheel_router_installed = True


def _bind_wheel(canvas: tk.Canvas) -> None:
    register_wheel_surface(canvas)


def scrollable_dialog_body(dialog, *, bg: str = "") -> tk.Frame:
    """Add a scrollable body to ``dialog`` and return the frame to build into.

    The returned frame is the only parent dialog content should use; the
    canvas, scrollbars, and keyboard bindings are owned here.
    """
    background = bg or Theme.BG_OVERLAY
    container = tk.Frame(dialog, bg=background)
    container.pack(fill="both", expand=True)
    container.rowconfigure(0, weight=1)
    container.columnconfigure(0, weight=1)

    canvas = tk.Canvas(
        container, bg=background, highlightthickness=0, takefocus=False)
    canvas.grid(row=0, column=0, sticky="nsew")
    vbar = ttk.Scrollbar(container, orient="vertical", command=canvas.yview)
    hbar = ttk.Scrollbar(container, orient="horizontal", command=canvas.xview)
    canvas.configure(yscrollcommand=vbar.set, xscrollcommand=hbar.set)

    body = tk.Frame(canvas, bg=background)
    window = canvas.create_window((0, 0), window=body, anchor="nw")

    def _sync(_event=None):
        bbox = canvas.bbox("all") or (0, 0, 0, 0)
        canvas.configure(scrollregion=bbox)
        needs_v = bbox[3] - bbox[1] > canvas.winfo_height() + 1
        needs_h = bbox[2] - bbox[0] > canvas.winfo_width() + 1
        if needs_v and not vbar.winfo_ismapped():
            vbar.grid(row=0, column=1, sticky="ns")
        elif not needs_v and vbar.winfo_ismapped():
            vbar.grid_remove()
        if needs_h and not hbar.winfo_ismapped():
            hbar.grid(row=1, column=0, sticky="ew")
        elif not needs_h and hbar.winfo_ismapped():
            hbar.grid_remove()
        # Stretch the body to the canvas when there is room to spare so the
        # content is not left-hugging in a wide dialog.
        if canvas.winfo_width() > body.winfo_reqwidth():
            canvas.itemconfigure(window, width=canvas.winfo_width())
        else:
            canvas.itemconfigure(window, width=body.winfo_reqwidth())

    body.bind("<Configure>", _sync)
    canvas.bind("<Configure>", _sync)

    for sequence, delta in (
        ("<Up>", -1), ("<Down>", 1),
    ):
        canvas.bind(
            sequence, lambda _e, step=delta: (canvas.yview_scroll(step, "units"), "break")[1])
    canvas.bind(
        "<Prior>", lambda _e: (canvas.yview_scroll(-1, "pages"), "break")[1])
    canvas.bind(
        "<Next>", lambda _e: (canvas.yview_scroll(1, "pages"), "break")[1])
    canvas.bind("<Home>", lambda _e: (canvas.yview_moveto(0.0), "break")[1])
    canvas.bind("<End>", lambda _e: (canvas.yview_moveto(1.0), "break")[1])
    _bind_wheel(canvas)

    dialog._vsr_scroll_canvas = canvas
    dialog._vsr_scroll_body = body
    dialog._vsr_scroll_sync = _sync
    return body


def fit_dialog_to_work_area(
    dialog,
    root,
    *,
    min_width: int = MIN_DIALOG_WIDTH,
    min_height: int = MIN_DIALOG_HEIGHT,
    center: bool = True,
) -> tuple[int, int]:
    """Size ``dialog`` to its content, clamped to the usable work area."""
    try:
        dialog.update_idletasks()
    except tk.TclError:
        return (0, 0)
    area_w, area_h = work_area(root)
    # Canvas windows do not contribute their child request size to the
    # Toplevel. Measure the scroll body directly or wide dialogs collapse to
    # the minimum width and immediately grow a horizontal scrollbar.
    body = getattr(dialog, "_vsr_scroll_body", None)
    body_w = int(body.winfo_reqwidth()) if body is not None else 0
    body_h = int(body.winfo_reqheight()) if body is not None else 0
    want_w = max(int(dialog.winfo_reqwidth()), body_w, min_width)
    want_h = max(int(dialog.winfo_reqheight()), body_h, min_height)
    width = max(min(want_w, area_w), min(min_width, area_w))
    height = max(min(want_h, area_h), min(min_height, area_h))
    dialog.resizable(True, True)
    dialog.minsize(min(min_width, width), min(min_height, height))
    dialog.maxsize(area_w, area_h)
    if center:
        # Center on the parent window. Centering on the primary screen put
        # every dialog on the wrong monitor whenever the app was moved.
        try:
            parent_x = int(root.winfo_rootx())
            parent_y = int(root.winfo_rooty())
            parent_w = int(root.winfo_width())
            parent_h = int(root.winfo_height())
        except Exception:
            parent_x = parent_y = 0
            parent_w, parent_h = area_w, area_h
        if parent_w <= 1 or parent_h <= 1:
            parent_w, parent_h = area_w, area_h
        try:
            screen_w = int(root.winfo_screenwidth())
            screen_h = int(root.winfo_screenheight())
        except Exception:
            screen_w, screen_h = area_w, area_h
        bx, by, bw, bh = desktop_bounds(screen_w, screen_h)
        x = parent_x + (parent_w - width) // 2
        y = parent_y + max(0, (parent_h - height) // 3)
        x = min(max(x, bx), bx + bw - width)
        y = min(max(y, by), by + bh - height)
        dialog.geometry(f"{width}x{height}+{x}+{y}")
    else:
        dialog.geometry(f"{width}x{height}")
    sync = getattr(dialog, "_vsr_scroll_sync", None)
    if callable(sync):
        dialog.update_idletasks()
        sync()
    return (width, height)
