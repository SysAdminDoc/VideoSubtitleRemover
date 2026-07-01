"""Design-token system and font helpers."""

from __future__ import annotations


class Theme:
    """Design system. Dark-first, refined tonal layering, calm accents."""

    # Surfaces
    BG_DARK = "#06080f"
    BG_SECONDARY = "#0c111c"
    BG_CARD = "#121927"
    BG_CARD_HOVER = "#182132"
    BG_CARD_SELECTED = "#1a2944"
    BG_TERTIARY = "#1b2438"
    BG_RAISED = "#222d44"
    BG_LOG = "#070b13"
    BG_OVERLAY = "#0a0e17"

    # Accents
    GREEN_PRIMARY = "#34d399"
    GREEN_HOVER = "#10b981"
    GREEN_PRESS = "#059669"
    GREEN_MUTED = "#0f3324"

    BLUE_PRIMARY = "#60a5fa"
    BLUE_HOVER = "#3b82f6"
    BLUE_PRESS = "#2563eb"
    BLUE_MUTED = "#13294a"

    # Text
    TEXT_PRIMARY = "#f4f7fd"
    TEXT_SECONDARY = "#c5cfe2"
    TEXT_MUTED = "#8391ad"
    TEXT_DISABLED = "#4c5877"

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
    WARNING = "#fbbf24"
    WARNING_BG = "#352412"
    ERROR = "#f87171"
    ERROR_BG = "#351821"
    INFO = "#60a5fa"
    INFO_BG = "#0f2744"

    # Borders
    BORDER = "#27324a"
    BORDER_STRONG = "#364364"
    BORDER_SUBTLE = "#1a2234"
    BORDER_FOCUS = "#60a5fa"

    # Progress
    PROGRESS_BG = "#182236"
    PROGRESS_FILL = BLUE_PRIMARY

    # Typography
    FONT_FAMILY = "Segoe UI"
    FONT_MONO = "Consolas"

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


def f(size: int, weight: str = "normal") -> tuple:
    """Build a Segoe UI font tuple."""
    if weight == "bold":
        return (Theme.FONT_FAMILY, size, "bold")
    return (Theme.FONT_FAMILY, size)


def mono(size: int) -> tuple:
    return (Theme.FONT_MONO, size)
