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
    BG_DARK = "#0b1020"
    BG_SECONDARY = "#10182a"
    BG_CARD = "#121a2b"
    BG_CARD_HOVER = "#182238"
    BG_CARD_SELECTED = "#1d2b48"
    BG_TERTIARY = "#182238"
    BG_RAISED = "#202c45"
    BG_LOG = "#080d19"
    BG_OVERLAY = "#090e1b"

    # Accents
    GREEN_PRIMARY = "#38d9a9"
    GREEN_HOVER = "#20c997"
    GREEN_PRESS = "#12a77d"
    GREEN_MUTED = "#10382f"

    BLUE_PRIMARY = "#4f7cff"
    BLUE_HOVER = "#668cff"
    BLUE_PRESS = "#3d66df"
    BLUE_MUTED = "#1a2d5a"
    CYAN = "#36c5f0"

    # Text
    TEXT_PRIMARY = "#f5f7fb"
    TEXT_SECONDARY = "#cbd4e4"
    TEXT_MUTED = "#9ca9bf"
    TEXT_DISABLED = "#59677f"

    # Ink: dark foreground for bright-filled controls
    INK_ON_GREEN = "#04120b"
    INK_ON_BLUE = "#071226"
    INK_ON_DANGER = "#ffffff"

    # Danger / destructive action
    DANGER = "#f87171"
    DANGER_HOVER = "#ef4444"
    DANGER_PRESS = "#dc2626"

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
    BORDER = "#28344d"
    BORDER_STRONG = "#3c4a68"
    BORDER_SUBTLE = "#202b40"
    BORDER_FOCUS = "#4f7cff"

    # Progress
    PROGRESS_BG = "#182236"
    PROGRESS_FILL = BLUE_PRIMARY

    # Typography
    FONT_FAMILY = "Segoe UI"
    FONT_MONO = "Consolas"
    RTL_LAYOUT = False

    # Size tokens
    F_DISPLAY = 22
    F_HEADING = 16
    F_TITLE = 12
    F_BODY = 10
    F_BODY_SM = 9
    F_LABEL = 9
    F_META = 8
    F_EYEBROW = 8
    F_MICRO = 7

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
    R_XL = 12


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
    Theme.INK_ON_DANGER = "#ffffff"
    Theme.DANGER = "#ff5555"
    Theme.DANGER_HOVER = "#ff3333"
    Theme.DANGER_PRESS = "#cc0000"
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
