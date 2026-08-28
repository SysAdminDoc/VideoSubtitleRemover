"""Design-token system and font helpers."""

from __future__ import annotations

import os
import sys


_TRUE_VALUES = {"1", "true", "yes", "on"}
_FALSE_VALUES = {"0", "false", "no", "off"}
_SYSTEM_REDUCED_MOTION = None
_TEXT_SCALE_PERCENT = 100


class Theme:
    """Design system. Dark-first, refined tonal layering, calm accents."""

    # Surfaces
    BG_DARK = "#080d15"
    BG_SECONDARY = "#0c1420"
    # Cards intentionally share the workbench tone. Hierarchy comes from
    # spacing and separators instead of stacked outlined rectangles.
    BG_CARD = BG_SECONDARY
    BG_CARD_HOVER = "#192231"
    BG_CARD_SELECTED = "#1c2940"
    BG_TERTIARY = "#121d2b"
    # Disabled controls: a recessed, desaturated grey that reads as "inert"
    # against the blue-tinted card (#0c1420) and enabled tertiary (#121d2b)
    # fills without vanishing into either.
    BG_DISABLED = "#14171d"
    BG_RAISED = "#1a283a"
    BG_LOG = "#070b12"
    BG_OVERLAY = "#080c14"

    # Accents
    GREEN_PRIMARY = "#38d9a9"
    GREEN_HOVER = "#20c997"
    GREEN_PRESS = "#12a77d"
    GREEN_MUTED = "#10382f"

    # RM-333: used as accent text and as the focus ring, so it has to
    # clear 4.5:1 on every surface. It measured 3.65 on the selected
    # card.
    BLUE_PRIMARY = "#4e8cff"
    # Raising BLUE_PRIMARY for contrast left BLUE_HOVER darker than the
    # resting fill, so hovering an accent button made it fractionally
    # darker instead of lighter and the affordance disappeared.
    BLUE_HOVER = "#689dff"
    BLUE_PRESS = "#2a71ec"
    # The selected state draws accent text on this fill, which measured
    # 4.07:1 against BLUE_HOVER and 4.16:1 against BLUE_PRIMARY.
    BLUE_MUTED = "#17274e"
    CYAN = "#36c5f0"

    # Text
    TEXT_PRIMARY = "#f5f7fb"
    TEXT_SECONDARY = "#cbd4e4"
    TEXT_MUTED = "#9ca9bf"
    TEXT_DISABLED = "#73829a"

    # Ink: dark foreground for bright-filled controls
    INK_ON_GREEN = "#04120b"
    INK_ON_BLUE = "#000000"
    INK_ON_DANGER = "#2a0505"

    # Danger / destructive action
    DANGER = "#f87171"
    DANGER_HOVER = "#ef4444"
    # RM-333: the pressed fill was 3.87:1 against INK_ON_DANGER, so the
    # label on a destructive button dropped below 4.5:1 exactly while
    # it was being pressed. Still darker than the hover fill.
    DANGER_PRESS = "#e14343"

    # Status
    SUCCESS = "#34d399"
    SUCCESS_BG = "#0e2e22"
    WARNING = "#f4b860"
    WARNING_BG = "#382817"
    ERROR = "#f87171"
    ERROR_BG = "#351821"
    INFO = "#36c5f0"
    INFO_BG = "#102f42"

    # Borders
    # RM-333: BORDER and BORDER_STRONG draw the boundary of controls that
    # have no other visual boundary, so WCAG 2.2 1.4.11 applies and they
    # have to clear 3:1 against every surface they are drawn on, including
    # hover, selected, raised and disabled. They used to measure 1.18 and
    # 1.60 at worst. BORDER_SUBTLE stays where it was: it is the divider
    # tone, decorative by design, and the test exempts it by name.
    BORDER = "#5b739c"
    BORDER_STRONG = "#7d93bb"
    BORDER_SUBTLE = "#1c2635"
    BORDER_FOCUS = BLUE_PRIMARY

    # Progress
    PROGRESS_BG = "#182236"
    PROGRESS_FILL = BLUE_PRIMARY

    # Typography
    FONT_FAMILY = "Segoe UI"
    FONT_MONO = "Consolas"
    RTL_LAYOUT = False

    # Size tokens
    F_DISPLAY = 20
    F_HEADING = 17
    F_TITLE = 15
    F_BODY = 15
    F_BODY_SM = 14
    F_LABEL = 14
    F_META = 13
    F_EYEBROW = 12
    F_MICRO = 12

    # Spacing rhythm (4pt baseline)
    S_XS = 4
    S_SM = 8
    S_MD = 12
    S_LG = 16
    S_XL = 20
    S_2XL = 24
    S_3XL = 32

    # Radii
    R_SM = 4
    R_MD = 6
    R_LG = 8
    R_XL = 10


def apply_high_contrast_theme():
    """RM-96: Swap the design tokens for a higher-contrast palette."""
    if not hasattr(Theme, "_defaults"):
        Theme._defaults = {
            k: v for k, v in Theme.__dict__.items()
            if not k.startswith("_") and isinstance(v, str)
        }
    Theme.BG_DARK = "#000000"
    Theme.BG_SECONDARY = "#000000"
    Theme.BG_CARD = "#0c0c0c"
    Theme.BG_CARD_HOVER = "#1a1a1a"
    Theme.BG_CARD_SELECTED = "#1f1f1f"
    Theme.BG_TERTIARY = "#1a1a1a"
    Theme.BG_DISABLED = "#141414"
    Theme.BG_RAISED = "#262626"
    Theme.BG_LOG = "#000000"
    Theme.BG_OVERLAY = "#000000"
    Theme.GREEN_PRIMARY = "#00ff7f"
    Theme.GREEN_HOVER = "#00cc66"
    Theme.GREEN_PRESS = "#00994d"
    Theme.GREEN_MUTED = "#003319"
    Theme.BLUE_PRIMARY = "#00d4ff"
    Theme.BLUE_HOVER = "#00b3d9"
    Theme.BLUE_PRESS = "#0099b3"
    Theme.BLUE_MUTED = "#002633"
    Theme.CYAN = "#00d4ff"
    Theme.TEXT_PRIMARY = "#ffffff"
    Theme.TEXT_SECONDARY = "#ffffff"
    Theme.TEXT_MUTED = "#dcdcdc"
    Theme.TEXT_DISABLED = "#888888"
    Theme.SUCCESS = "#00ff7f"
    Theme.SUCCESS_BG = "#003319"
    Theme.WARNING = "#ffff00"
    Theme.WARNING_BG = "#332f00"
    Theme.ERROR = "#ff5555"
    Theme.ERROR_BG = "#330000"
    Theme.INFO = "#00d4ff"
    Theme.INFO_BG = "#002633"
    Theme.BORDER = "#ffffff"
    Theme.BORDER_STRONG = "#ffffff"
    Theme.BORDER_SUBTLE = "#aaaaaa"
    Theme.BORDER_FOCUS = "#ffff00"
    Theme.INK_ON_GREEN = "#000000"
    Theme.INK_ON_BLUE = "#000000"
    Theme.INK_ON_DANGER = "#2a0505"
    Theme.DANGER = "#ff5555"
    Theme.DANGER_HOVER = "#ff3333"
    Theme.DANGER_PRESS = "#fa0000"  # RM-333: 3.18:1 before
    Theme.PROGRESS_BG = "#1a1a1a"
    Theme.PROGRESS_FILL = "#00d4ff"


def apply_default_theme():
    """Restore the original Theme palette."""
    defaults = getattr(Theme, "_defaults", None)
    if not defaults:
        return
    for k, v in defaults.items():
        setattr(Theme, k, v)


def prefers_reduced_motion() -> bool:
    """Return the explicit or Windows system reduced-animation preference."""
    override = os.environ.get("VSR_REDUCED_MOTION", "").strip().lower()
    if override in _TRUE_VALUES:
        return True
    if override in _FALSE_VALUES:
        return False

    global _SYSTEM_REDUCED_MOTION
    if _SYSTEM_REDUCED_MOTION is not None:
        return bool(_SYSTEM_REDUCED_MOTION)
    if sys.platform != "win32":
        _SYSTEM_REDUCED_MOTION = False
        return False
    try:
        import ctypes

        animations_enabled = ctypes.c_int(1)
        ok = ctypes.windll.user32.SystemParametersInfoW(
            0x1042,  # SPI_GETCLIENTAREAANIMATION
            0,
            ctypes.byref(animations_enabled),
            0,
        )
        _SYSTEM_REDUCED_MOTION = bool(ok and not animations_enabled.value)
    except Exception:
        _SYSTEM_REDUCED_MOTION = False
    return bool(_SYSTEM_REDUCED_MOTION)


def normalize_text_scale_percent(value: object) -> int:
    """Clamp text scaling to the supported 100-200 percent range."""
    try:
        percent = int(value)
    except (TypeError, ValueError, OverflowError):
        percent = 100
    percent = max(100, min(200, percent))
    choices = (100, 125, 150, 175, 200)
    return min(choices, key=lambda choice: (abs(choice - percent), choice))


def set_text_scale_percent(value: object) -> int:
    """Set the process-wide text scale before constructing Tk widgets."""
    global _TEXT_SCALE_PERCENT
    _TEXT_SCALE_PERCENT = normalize_text_scale_percent(value)
    return _TEXT_SCALE_PERCENT


def text_scale_percent() -> int:
    return int(_TEXT_SCALE_PERCENT)


def text_scale_factor() -> float:
    return text_scale_percent() / 100.0


def scaled_font_size(size: int) -> int:
    return max(1, int(round(int(size) * text_scale_factor())))


def scaled_control_size(size: int) -> int:
    """Scale geometry that must grow with text, such as Canvas heights."""
    return max(1, int(round(int(size) * text_scale_factor())))


def f(size: int, weight: str = "normal") -> tuple:
    """Build a Segoe UI font tuple."""
    size = scaled_font_size(size)
    if weight == "bold":
        return (Theme.FONT_FAMILY, size, "bold")
    return (Theme.FONT_FAMILY, size)


def mono(size: int) -> tuple:
    return (Theme.FONT_MONO, scaled_font_size(size))
