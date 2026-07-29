"""RM-152: logical-to-physical layout direction mapping.

Every layout call site in this GUI is written in *logical* terms -- the
left-to-right reading order English uses. Under an RTL locale those
logical values have to become their physical mirror: a row that packs
left-to-right must pack right-to-left, a label anchored to the start of
its cell must anchor to the right edge, and a "sort A -> Z" affordance
must point the other way.

Hand-mirroring roughly four hundred `side=`, `anchor=`, `sticky=`, and
`justify=` call sites would be both enormous and permanently prone to
drift: every new widget would silently be LTR-only. Instead this module
installs one interception point.

`tkinter.Misc._options` is the single funnel every option dictionary
passes through -- widget construction, `configure()`, and the `pack`,
`grid`, and `place` geometry managers all build their Tcl argument list
from it. Mirroring there covers the whole widget tree at once and
automatically covers code written later.

Canvas *items* are excluded on purpose. Their anchors are coupled to
explicit x/y coordinates the caller already computed, so mirroring the
anchor without mirroring the coordinate would tear the two apart.
Canvas-drawing widgets mirror their own geometry (see `gui/widgets.py`).

The mirror is inert until `install_direction_mirror()` runs and
`Theme.RTL_LAYOUT` is true, so an LTR session pays nothing but one
dictionary lookup per option dict.
"""

from __future__ import annotations

import contextlib
import threading
import tkinter as tk

from gui.theme import Theme

SIDE_MIRROR = {"left": "right", "right": "left"}

ANCHOR_MIRROR = {
    "w": "e",
    "e": "w",
    "nw": "ne",
    "ne": "nw",
    "sw": "se",
    "se": "sw",
}

JUSTIFY_MIRROR = {"left": "right", "right": "left"}

# Only spaced ASCII arrows are mirrored. `<->` is symmetric and must not
# move, and a bare `->` inside a path, log line, or type annotation is
# not a direction affordance.
TEXT_MIRROR = ((" -> ", " <- "), (" >> ", " << "))


def is_rtl() -> bool:
    """True when the active theme is laid out right-to-left."""
    return bool(getattr(Theme, "RTL_LAYOUT", False))


def mirroring_active() -> bool:
    """True when a mirror call should actually transform its input.

    Both gates matter: the theme has to be RTL, and the caller must not
    have suspended mirroring for this block. Every public `mirror_*`
    helper honours this so `no_mirror()` means the same thing whether a
    value goes through Tk or through a direct call.
    """
    return is_rtl() and not _suspended()


def mirror_side(value):
    """Mirror a `pack(side=...)` value."""
    if not mirroring_active() or not isinstance(value, str):
        return value
    return SIDE_MIRROR.get(value.strip().lower(), value)


def mirror_anchor(value):
    """Mirror a widget or geometry-manager anchor."""
    if not mirroring_active() or not isinstance(value, str):
        return value
    return ANCHOR_MIRROR.get(value.strip().lower(), value)


def mirror_justify(value):
    """Mirror a multi-line text justification."""
    if not mirroring_active() or not isinstance(value, str):
        return value
    return JUSTIFY_MIRROR.get(value.strip().lower(), value)


def mirror_sticky(value):
    """Mirror the west/east components of a `grid(sticky=...)` mask.

    `n` and `s` are direction-neutral and stay put; `ew` and `we` are
    already symmetric and survive the swap unchanged.
    """
    if not mirroring_active() or not isinstance(value, str):
        return value
    swapped = []
    for char in value:
        lowered = char.lower()
        if lowered == "w":
            swapped.append("e" if char.islower() else "E")
        elif lowered == "e":
            swapped.append("w" if char.islower() else "W")
        else:
            swapped.append(char)
    return "".join(swapped)


def mirror_text(value):
    """Flip directional arrow affordances inside user-visible text."""
    if not mirroring_active() or not isinstance(value, str):
        return value
    if "<->" in value or "<-->" in value:
        return value
    result = value
    for forward, backward in TEXT_MIRROR:
        if forward in result:
            result = result.replace(forward, backward)
        elif backward in result:
            result = result.replace(backward, forward)
    return result


MIRRORED_OPTIONS = {
    "side": mirror_side,
    "anchor": mirror_anchor,
    "justify": mirror_justify,
    "sticky": mirror_sticky,
    "text": mirror_text,
    # Menu entries carry their caption as `label`, not `text`.
    "label": mirror_text,
}


_state = threading.local()


def _suspended() -> bool:
    return bool(getattr(_state, "suspended", False))


@contextlib.contextmanager
def no_mirror():
    """Run a block with mirroring switched off.

    Use this for the rare call site that has already mirrored its own
    geometry by hand and would otherwise be flipped back.
    """
    previous = _suspended()
    _state.suspended = True
    try:
        yield
    finally:
        _state.suspended = previous


def mirror_options(mapping):
    """Return `mapping` with every direction-sensitive option mirrored.

    The input is never mutated: a caller-owned dict passed to `pack()`
    and reused for a second widget must not pick up the mirror twice.
    """
    if not isinstance(mapping, dict) or not mapping:
        return mapping
    changed = None
    for key, value in mapping.items():
        transform = MIRRORED_OPTIONS.get(key)
        if transform is None:
            continue
        mirrored = transform(value)
        if mirrored != value:
            if changed is None:
                changed = dict(mapping)
            changed[key] = mirrored
    return mapping if changed is None else changed


_original_options = None
_original_canvas_create = None
_original_canvas_itemconfigure = None


def direction_mirror_installed() -> bool:
    return _original_options is not None


def install_direction_mirror() -> bool:
    """Patch Tk's option funnel. Idempotent; returns True on first install."""
    global _original_options, _original_canvas_create
    global _original_canvas_itemconfigure
    if _original_options is not None:
        return False

    _original_options = tk.Misc._options
    _original_canvas_create = tk.Canvas._create
    _original_canvas_itemconfigure = tk.Canvas.itemconfigure

    def _options(self, cnf, kw=None):
        if mirroring_active():
            cnf = mirror_options(cnf)
            if kw is not None:
                kw = mirror_options(kw)
        return _original_options(self, cnf, kw)

    def _create(self, itemType, args, kw):
        with no_mirror():
            return _original_canvas_create(self, itemType, args, kw)

    def _itemconfigure(self, tagOrId, cnf=None, **kw):
        with no_mirror():
            return _original_canvas_itemconfigure(self, tagOrId, cnf, **kw)

    tk.Misc._options = _options
    tk.Canvas._create = _create
    tk.Canvas.itemconfigure = _itemconfigure
    tk.Canvas.itemconfig = _itemconfigure
    return True


def uninstall_direction_mirror() -> bool:
    """Restore Tk's originals. Idempotent; returns True when it undid a patch."""
    global _original_options, _original_canvas_create
    global _original_canvas_itemconfigure
    if _original_options is None:
        return False
    tk.Misc._options = _original_options
    tk.Canvas._create = _original_canvas_create
    tk.Canvas.itemconfigure = _original_canvas_itemconfigure
    tk.Canvas.itemconfig = _original_canvas_itemconfigure
    _original_options = None
    _original_canvas_create = None
    _original_canvas_itemconfigure = None
    return True
