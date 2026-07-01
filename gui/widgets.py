"""Custom widgets extracted from the GUI monolith (RM-114)."""

from __future__ import annotations

import ctypes
import logging
import os
import queue
import sys
from datetime import datetime
import tkinter as tk
from tkinter import filedialog
import tkinter.font as tkfont
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from backend.i18n import tr
from gui.theme import Theme, f, mono
from gui.config import ProcessingStatus, QueueItem, STATUS_UI, status_ui
from backend.a11y import (
    accessible_metadata,
    announce,
    announce_widget,
    set_accessible_metadata,
)

try:
    from PIL import Image, ImageTk
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

logger = logging.getLogger(__name__)
_FORCE_QUOTED_CLI_VALUE_FLAGS = {"-i", "-o", "--watermark"}


def _state_text(*values: str) -> str:
    return ", ".join(value for value in values if value)


def _build_cli_command(item: QueueItem) -> str:
    """Build a CLI command string that reproduces an item's settings."""
    args = ["python", "-m", "backend.processor"]
    args.extend(["-i", item.file_path])
    args.extend(["-o", item.output_path])
    cfg = item.config
    mode_map = {"STTN": "sttn", "LAMA": "lama", "ProPainter": "propainter",
                "AUTO": "auto"}
    mode_str = getattr(cfg, "mode", None)
    if mode_str:
        if hasattr(mode_str, "value"):
            mode_str = mode_str.value
        cli_mode = mode_map.get(str(mode_str), str(mode_str).lower())
        args.extend(["-m", cli_mode])
    lang = getattr(cfg, "detection_lang", "en") or "en"
    if lang != "en":
        args.extend(["-l", lang])
    crf = getattr(cfg, "output_quality", 23)
    if crf != 23:
        args.extend(["--crf", str(crf)])
    codec = getattr(cfg, "output_codec", "h264")
    if codec and codec != "h264":
        args.extend(["--codec", str(codec)])
    threshold = getattr(cfg, "detection_threshold", 0.5)
    if threshold != 0.5:
        args.extend(["--threshold", str(threshold)])
    dilate = getattr(cfg, "mask_dilate_px", 8)
    if dilate != 8:
        args.extend(["--mask-dilate", str(dilate)])
    feather = getattr(cfg, "mask_feather_px", 4)
    if feather != 4:
        args.extend(["--mask-feather", str(feather)])
    skip = getattr(cfg, "detection_frame_skip", 0)
    if skip:
        args.extend(["--frame-skip", str(skip)])
    edge_ring = getattr(cfg, "edge_ring_px", 2)
    if edge_ring != 2:
        args.extend(["--edge-ring", str(edge_ring)])
    temporal = getattr(cfg, "temporal_smooth_radius", 0)
    if temporal:
        args.extend(["--temporal-smooth", str(temporal)])
    if getattr(cfg, "lama_super_fast", False):
        args.append("--fast")
    if not getattr(cfg, "preserve_audio", True):
        args.append("--no-audio")
    if not getattr(cfg, "use_hw_encode", True):
        args.append("--no-hw-encode")
    if getattr(cfg, "detection_vertical", False):
        args.append("--vertical")
    start = getattr(cfg, "time_start", 0.0) or 0.0
    if start > 0:
        args.extend(["--start", str(start)])
    end = getattr(cfg, "time_end", 0.0) or 0.0
    if end > 0:
        args.extend(["--end", str(end)])
    loudnorm = getattr(cfg, "loudnorm_target", 0.0) or 0.0
    if loudnorm != 0.0:
        args.extend(["--loudnorm", str(loudnorm)])
    if not getattr(cfg, "multi_audio_passthrough", True):
        args.append("--single-audio")
    if getattr(cfg, "tbe_flow_warp", False):
        args.append("--flow-warp")
    if getattr(cfg, "colour_tune_enable", False):
        args.append("--colour-tune")
        tolerance = getattr(cfg, "colour_tune_tolerance", 25)
        if tolerance != 25:
            args.extend(["--colour-tolerance", str(tolerance)])
    if not getattr(cfg, "kalman_tracking", True):
        args.append("--no-kalman")
    if not getattr(cfg, "phash_skip_enable", True):
        args.append("--no-phash")
    if getattr(cfg, "confidence_weighted_dilation", False):
        args.append("--confidence-dilate")
    if getattr(cfg, "whisper_fallback", False):
        args.append("--whisper-fallback")
        model = getattr(cfg, "whisper_model_size", "tiny")
        if model != "tiny":
            args.extend(["--whisper-model", str(model)])
        backend = getattr(cfg, "whisper_backend", "faster-whisper")
        if backend != "faster-whisper":
            args.extend(["--whisper-backend", str(backend)])
    if not getattr(cfg, "remove_subtitles", True):
        args.append("--keep-subtitles")
    if not getattr(cfg, "remove_chyrons", True):
        args.append("--keep-chyrons")
    if getattr(cfg, "karaoke_grouping", False):
        args.append("--karaoke-grouping")
    if getattr(cfg, "export_srt", False):
        args.append("--export-srt")
    if getattr(cfg, "export_mask_video", False):
        args.append("--export-mask")
    if getattr(cfg, "output_frames", False):
        args.append("--output-frames")
    if getattr(cfg, "quality_report", False):
        args.append("--quality-report")
    if getattr(cfg, "quality_report_sheet", False):
        args.append("--quality-sheet")
    nle = getattr(cfg, "nle_sidecar", "off")
    if nle and nle != "off":
        args.extend(["--nle-sidecar", str(nle)])
    wm = getattr(cfg, "watermark_image", "")
    if wm:
        args.extend(["--watermark", wm])
        wm_pos = getattr(cfg, "watermark_position", "bottom-right")
        if wm_pos != "bottom-right":
            args.extend(["--watermark-position", str(wm_pos)])
        wm_op = getattr(cfg, "watermark_opacity", 1.0)
        if wm_op != 1.0:
            args.extend(["--watermark-opacity", str(wm_op)])
        wm_margin = getattr(cfg, "watermark_margin", 16)
        if wm_margin != 16:
            args.extend(["--watermark-margin", str(wm_margin)])
    return " ".join(
        _quote_cli_arg(
            arg,
            force=i > 0 and args[i - 1] in _FORCE_QUOTED_CLI_VALUE_FLAGS,
        )
        for i, arg in enumerate(args)
    )


def _quote_cli_arg(value: Any, *, force: bool = False) -> str:
    """Quote for a copied Windows shell command, not just CreateProcess."""
    text = str(value)
    if not force and text and all(ch.isalnum() or ch in "._-/:\\"
                    for ch in text):
        return text
    out = '"'
    backslashes = 0
    for ch in text:
        if ch == "\\":
            backslashes += 1
            continue
        if ch == '"':
            out += "\\" * (backslashes * 2 + 1)
            out += '"'
            backslashes = 0
            continue
        if backslashes:
            out += "\\" * backslashes
            backslashes = 0
        out += ch
    if backslashes:
        out += "\\" * (backslashes * 2)
    out += '"'
    return out


from gui.utils import (
    IMAGE_EXTENSIONS,
    SUPPORTED_EXTENSIONS,
    VIDEO_EXTENSIONS,
    filepicker_pattern,
    format_time,
    truncate_middle,
    _queue_item_info_text,
)

def _get_dpi_scale(root) -> float:
    """Get the DPI scaling factor relative to 96 DPI baseline."""
    try:
        return root.winfo_fpixels('1i') / 96.0
    except Exception:
        return 1.0


def _scaled(root, px: int) -> int:
    """Scale a pixel value by the current DPI factor."""
    return int(px * _get_dpi_scale(root))


class Tooltip:
    """Refined hover tooltip. Appears after a short delay, styled as a raised
    surface with subtle border and proper text wrapping."""

    DELAY_MS = 380

    def __init__(self, widget, text):
        self.widget = widget
        self.text = text
        self._tip = None
        self._after_id = None
        widget.bind("<Enter>", self._schedule, add="+")
        widget.bind("<Leave>", self._hide, add="+")
        widget.bind("<ButtonPress>", self._hide, add="+")
        widget.bind("<Destroy>", self._hide, add="+")

    def _schedule(self, event):
        self._cancel()
        try:
            self._after_id = self.widget.after(self.DELAY_MS, self._show)
        except tk.TclError:
            self._after_id = None

    def _cancel(self):
        if self._after_id:
            try:
                self.widget.after_cancel(self._after_id)
            except tk.TclError:
                pass
            self._after_id = None

    def _show(self):
        if self._tip:
            self._tip.destroy()
            self._tip = None
        try:
            self._tip = tk.Toplevel(self.widget)
            self._tip.wm_overrideredirect(True)
            self._tip.configure(bg=Theme.BORDER_STRONG)
            display_text = self.text if len(self.text) <= 160 else self.text[:157] + "..."
            inner = tk.Frame(self._tip, bg=Theme.BG_RAISED)
            inner.pack(padx=1, pady=1)
            tk.Label(
                inner,
                text=display_text,
                font=f(Theme.F_LABEL),
                bg=Theme.BG_RAISED,
                fg=Theme.TEXT_PRIMARY,
                padx=10, pady=6,
                wraplength=360,
                justify="left",
            ).pack()
            self._tip.update_idletasks()
            x = self.widget.winfo_rootx() + 14
            y = self.widget.winfo_rooty() + self.widget.winfo_height() + 6
            sw = self._tip.winfo_screenwidth()
            sh = self._tip.winfo_screenheight()
            tw = self._tip.winfo_reqwidth()
            th = self._tip.winfo_reqheight()
            if x + tw > sw:
                x = sw - tw - 6
            if y + th > sh:
                y = self.widget.winfo_rooty() - th - 6
            self._tip.wm_geometry(f"+{x}+{y}")
        except tk.TclError:
            self._tip = None

    def _hide(self, event=None):
        self._cancel()
        if self._tip:
            try:
                self._tip.destroy()
            except tk.TclError:
                pass
            self._tip = None


class ModernButton(tk.Canvas):
    """A refined button with hover/press/focus states, icon support,
    and consistent size tokens. Canvas-based so corner radius is rendered
    crisply regardless of ttk theme.

    Style variants: primary, accent, secondary, ghost, danger, success
    Size variants: sm (28), md (32), lg (36)
    """

    SIZES = {"sm": (28, Theme.F_META), "md": (32, Theme.F_LABEL), "lg": (36, Theme.F_BODY_SM)}

    def __init__(self, parent, text="Button", command=None, width=120, height=None,
                 bg=None, hover_bg=None, fg=Theme.TEXT_PRIMARY,
                 corner_radius=None, font_size=None, style="primary",
                 size="md", icon=None, **kwargs):
        if height is None:
            height = self.SIZES.get(size, self.SIZES["md"])[0]
        if font_size is None:
            font_size = self.SIZES.get(size, self.SIZES["md"])[1]
        if corner_radius is None:
            corner_radius = Theme.R_MD if height <= 30 else Theme.R_LG

        parent_bg = parent.cget('bg') if hasattr(parent, 'cget') else Theme.BG_DARK
        super().__init__(parent, width=width, height=height, highlightthickness=0,
                        bg=parent_bg, takefocus=1)

        self.text = text
        self.icon = icon  # optional single-char glyph (ASCII)
        self.command = command
        self.width = width
        self.height = height
        self.corner_radius = corner_radius
        self.font_size = font_size
        self.enabled = True
        self.focused = False
        self.pressed = False
        self.hovered = False
        self.style = style
        self.disabled_reason = ""

        self._apply_style(style)
        self.current_bg = self.bg_color
        self._sync_a11y()
        self._draw()

        self.bind("<Enter>", self._on_enter)
        self.bind("<Leave>", self._on_leave)
        self.bind("<Button-1>", self._on_click)
        self.bind("<ButtonRelease-1>", self._on_release)
        self.bind("<FocusIn>", self._on_focus_in)
        self.bind("<FocusOut>", self._on_focus_out)
        self.bind("<Return>", self._on_keyboard_activate)
        self.bind("<space>", self._on_keyboard_activate)

    def _apply_style(self, style):
        if style == "primary":
            self.bg_color = Theme.BLUE_PRIMARY
            self.hover_color = Theme.BLUE_HOVER
            self.press_color = Theme.BLUE_PRESS
            self.fg_color = Theme.INK_ON_BLUE
            self.border_color = Theme.BLUE_HOVER
        elif style == "accent":
            self.bg_color = Theme.BLUE_MUTED
            self.hover_color = Theme.BG_RAISED
            self.press_color = Theme.BG_CARD_SELECTED
            self.fg_color = Theme.TEXT_PRIMARY
            self.border_color = Theme.BORDER
        elif style == "secondary":
            self.bg_color = Theme.BG_TERTIARY
            self.hover_color = Theme.BG_RAISED
            self.press_color = Theme.BG_CARD_HOVER
            self.fg_color = Theme.TEXT_PRIMARY
            self.border_color = Theme.BORDER
        elif style == "ghost":
            self.bg_color = Theme.BG_CARD
            self.hover_color = Theme.BG_CARD_HOVER
            self.press_color = Theme.BG_CARD_SELECTED
            self.fg_color = Theme.TEXT_SECONDARY
            self.border_color = Theme.BORDER_SUBTLE
        elif style == "danger":
            self.bg_color = Theme.DANGER
            self.hover_color = Theme.DANGER_HOVER
            self.press_color = Theme.DANGER_PRESS
            self.fg_color = Theme.INK_ON_DANGER
            self.border_color = Theme.DANGER_HOVER
        elif style == "success":
            self.bg_color = Theme.GREEN_MUTED
            self.hover_color = Theme.SUCCESS_BG
            self.press_color = Theme.GREEN_MUTED
            self.fg_color = Theme.GREEN_PRIMARY
            self.border_color = Theme.GREEN_HOVER
        else:
            self.bg_color = Theme.BG_TERTIARY
            self.hover_color = Theme.BG_CARD_HOVER
            self.press_color = Theme.BG_CARD_HOVER
            self.fg_color = Theme.TEXT_PRIMARY
            self.border_color = Theme.BORDER

    def _draw(self):
        self.delete("all")

        # Focus ring -- crisp outer glow
        if self.focused and self.enabled:
            self._create_rounded_rect(
                0, 0, self.width, self.height,
                self.corner_radius + 2,
                fill=Theme.BG_DARK, outline=Theme.BORDER_FOCUS, width=2,
            )
            pad = 2
        else:
            pad = 0

        if not self.enabled:
            fill = Theme.BG_TERTIARY
            border = Theme.BORDER_SUBTLE
            text_color = Theme.TEXT_DISABLED
        else:
            fill = self.current_bg
            border = self.border_color if (self.hovered or self.focused) else self._subtle_border()
            text_color = self.fg_color

        self._create_rounded_rect(
            pad, pad, self.width - pad, self.height - pad,
            self.corner_radius,
            fill=fill, outline=border, width=1,
        )

        # Press offset
        text_y = self.height // 2 + (1 if self.pressed else 0)

        if self.icon:
            gap = 6
            text_size = self._fit_text_size(self.text, self.icon, gap)
            icon_font = (Theme.FONT_FAMILY, text_size + 1, "bold")
            text_font = (Theme.FONT_FAMILY, text_size, "bold")
            icon_w = self._text_width(self.icon, icon_font)
            text_w = self._text_width(self.text, text_font)
            total = icon_w + gap + text_w
            start_x = (self.width - total) // 2
            self.create_text(start_x + icon_w // 2, text_y,
                             text=self.icon, fill=text_color, font=icon_font)
            self.create_text(start_x + icon_w + gap + text_w // 2, text_y,
                             text=self.text, fill=text_color, font=text_font)
        else:
            text_size = self._fit_text_size(self.text)
            self.create_text(self.width // 2, text_y, text=self.text,
                             fill=text_color,
                             font=(Theme.FONT_FAMILY, text_size, "bold"))

    def _sync_a11y(self):
        state = _state_text(
            "enabled" if self.enabled else "disabled",
            "focused" if self.focused else "",
            "pressed" if self.pressed else "",
        )
        set_accessible_metadata(
            self,
            role="button",
            label=self.text,
            state=state,
            description=self.disabled_reason if not self.enabled else "",
        )

    def accessibility_snapshot(self) -> dict:
        return accessible_metadata(self)

    def _subtle_border(self):
        # For filled CTAs, border should match the fill for a flat look
        if self.style in ("primary", "danger"):
            return self.bg_color
        return Theme.BORDER_SUBTLE

    def _text_width(self, text, font):
        try:
            return tkfont.Font(font=font).measure(text)
        except Exception:
            return len(text) * 7

    def _fit_text_size(self, text: str, icon: str = "", gap: int = 0) -> int:
        """Keep long labels inside fixed-width canvas buttons."""
        size = self.font_size
        available = max(24, self.width - 18)
        while size > Theme.F_MICRO:
            text_font = (Theme.FONT_FAMILY, size, "bold")
            measured = self._text_width(text, text_font)
            if icon:
                icon_font = (Theme.FONT_FAMILY, size + 1, "bold")
                measured += self._text_width(icon, icon_font) + gap
            if measured <= available:
                return size
            size -= 1
        return Theme.F_MICRO

    def _create_rounded_rect(self, x1, y1, x2, y2, r, **kwargs):
        points = [
            x1 + r, y1, x2 - r, y1, x2, y1, x2, y1 + r,
            x2, y2 - r, x2, y2, x2 - r, y2, x1 + r, y2,
            x1, y2, x1, y2 - r, x1, y1 + r, x1, y1
        ]
        return self.create_polygon(points, smooth=True, **kwargs)

    def _on_enter(self, event):
        if self.enabled:
            self.hovered = True
            self.current_bg = self.hover_color
            self._draw()
            self.config(cursor="hand2")

    def _on_leave(self, event):
        if self.enabled:
            self.hovered = False
            self.pressed = False
            self.current_bg = self.bg_color
            self._sync_a11y()
            self._draw()
            self.config(cursor="")

    def _on_click(self, event):
        if self.enabled:
            self.focus_set()
            self.pressed = True
            self.current_bg = self.press_color
            self._sync_a11y()
            self._draw()

    def _on_release(self, event):
        if self.enabled:
            inside = 0 <= event.x <= self.width and 0 <= event.y <= self.height
            self.pressed = False
            self.current_bg = self.hover_color if inside else self.bg_color
            self._sync_a11y()
            self._draw()
            if inside and self.command:
                self.command()

    def _on_focus_in(self, event):
        self.focused = True
        self._sync_a11y()
        self._draw()
        announce_widget(self)

    def _on_focus_out(self, event):
        self.focused = False
        self.pressed = False
        self.current_bg = self.bg_color
        self._sync_a11y()
        self._draw()

    def _on_keyboard_activate(self, event):
        if self.enabled and self.command:
            self.command()

    def set_enabled(self, enabled: bool, reason: str = ""):
        self.enabled = enabled
        self.disabled_reason = "" if enabled else reason
        self.hovered = False
        self.pressed = False
        if not enabled:
            self.focused = False
        self.current_bg = self.bg_color if enabled else Theme.BG_TERTIARY
        self.config(cursor="hand2" if enabled else "", takefocus=1 if enabled else 0)
        self._sync_a11y()
        self._draw()

    def set_text(self, text: str):
        self.text = text
        self._sync_a11y()
        self._draw()

    def set_style(self, style: str):
        """Re-skin the button (e.g., primary -> danger during processing)."""
        self._apply_style(style)
        self.style = style
        self.current_bg = self.bg_color
        self._sync_a11y()
        self._draw()


class ModernProgressBar(tk.Canvas):
    """A refined progress bar. Rounded track + fill. Smoothly tweens to
    target progress values so updates feel continuous rather than stepped."""

    TWEEN_STEP = 0.04
    TWEEN_DELAY_MS = 16  # ~60fps cap

    def __init__(self, parent, width=400, height=6, bg=Theme.PROGRESS_BG,
                 fill=Theme.PROGRESS_FILL, corner_radius=None, **kwargs):
        if corner_radius is None:
            corner_radius = max(2, height // 2)
        super().__init__(parent, width=width, height=height, highlightthickness=0,
                        bg=parent.cget('bg') if hasattr(parent, 'cget') else Theme.BG_DARK)

        self.bar_width = width
        self.bar_height = height
        self.corner_radius = corner_radius
        self.bg_color = bg
        self.fill_color = fill
        self.progress = 0.0
        self._target = 0.0
        self._tween_id = None

        self._draw()

    def _draw(self):
        self.delete("all")
        r = self.corner_radius

        self._create_rounded_rect(0, 0, self.bar_width, self.bar_height, r, fill=self.bg_color)

        if self.progress > 0:
            fill_width = max(r * 2, int(self.bar_width * self.progress))
            self._create_rounded_rect(0, 0, fill_width, self.bar_height, r, fill=self.fill_color)

    def _create_rounded_rect(self, x1, y1, x2, y2, r, **kwargs):
        points = [
            x1 + r, y1, x2 - r, y1, x2, y1, x2, y1 + r,
            x2, y2 - r, x2, y2, x2 - r, y2, x1 + r, y2,
            x1, y2, x1, y2 - r, x1, y1 + r, x1, y1
        ]
        return self.create_polygon(points, smooth=True, **kwargs)

    def set_progress(self, value: float, animate: bool = True):
        """Set the displayed progress. With `animate=True`, eases from the
        current value to the target over several frames."""
        target = max(0.0, min(1.0, value))
        self._target = target
        if self._tween_id:
            try:
                self.after_cancel(self._tween_id)
            except tk.TclError:
                pass
            self._tween_id = None
        # For big backward jumps (e.g. reset to 0), snap directly
        if not animate or target == 0.0 or abs(target - self.progress) < 0.005:
            self.progress = target
            self._draw()
            return
        self._tween_step()

    def _tween_step(self):
        delta = self._target - self.progress
        if abs(delta) < 0.003:
            self.progress = self._target
            self._draw()
            self._tween_id = None
            return
        # Ease-out: move 18% of remaining distance per frame, min 0.4%
        step = delta * 0.18
        if abs(step) < 0.004:
            step = 0.004 if delta > 0 else -0.004
        self.progress = max(0.0, min(1.0, self.progress + step))
        self._draw()
        try:
            self._tween_id = self.after(self.TWEEN_DELAY_MS, self._tween_step)
        except tk.TclError:
            self._tween_id = None

    def set_color(self, color: str):
        self.fill_color = color
        self._draw()

    def resize(self, width: int, height: int = None):
        """Resize the progress bar (for DPI/layout changes)."""
        self.bar_width = width
        if height:
            self.bar_height = height
            self.corner_radius = max(2, height // 2)
        self.config(width=self.bar_width, height=self.bar_height)
        self._draw()


class ModernToggle(tk.Canvas):
    """Custom checkbox/toggle replacement for tk.Checkbutton.

    Renders as a rounded square indicator with a checkmark, followed by
    a text label. Full support for hover/focus/disabled states, keyboard
    activation, and tk.BooleanVar binding.
    """

    BOX = 18
    GAP = 10

    def __init__(self, parent, text="", variable=None, command=None,
                 bg=None, fg=None, **kwargs):
        self.variable = variable if variable is not None else tk.BooleanVar(value=False)
        self.text = text
        self.command = command
        self.enabled = True
        self.focused = False
        self.hovered = False
        self.parent_bg = bg or (parent.cget('bg') if hasattr(parent, 'cget') else Theme.BG_CARD)
        self.fg_color = fg or Theme.TEXT_PRIMARY

        # Measure text width for canvas sizing
        self._font = f(Theme.F_BODY_SM)
        text_w = tkfont.Font(font=self._font).measure(text)
        total_w = self.BOX + self.GAP + text_w + 4
        super().__init__(parent, width=total_w, height=max(self.BOX + 4, 24),
                         highlightthickness=0, bg=self.parent_bg, takefocus=1)

        self._sync_a11y()
        self._draw()
        self.bind("<Button-1>", self._toggle)
        self.bind("<space>", self._toggle)
        self.bind("<Return>", self._toggle)
        self.bind("<Enter>", self._on_enter)
        self.bind("<Leave>", self._on_leave)
        self.bind("<FocusIn>", self._on_focus_in)
        self.bind("<FocusOut>", self._on_focus_out)
        if self.variable is not None:
            self.variable.trace_add("write", lambda *_: self._on_variable_changed())

    def _on_variable_changed(self):
        self._sync_a11y()
        self._draw()

    def _sync_a11y(self):
        checked = bool(self.variable.get())
        state = _state_text(
            "enabled" if self.enabled else "disabled",
            "checked" if checked else "not checked",
            "focused" if self.focused else "",
        )
        set_accessible_metadata(
            self,
            role="checkbox",
            label=self.text,
            state=state,
            value="on" if checked else "off",
        )

    def accessibility_snapshot(self) -> dict:
        return accessible_metadata(self)

    def _draw(self):
        self.delete("all")
        y0 = (int(self["height"]) - self.BOX) // 2
        x0 = 2

        checked = bool(self.variable.get())

        # Focus ring
        if self.focused and self.enabled:
            self._rounded(x0 - 2, y0 - 2, x0 + self.BOX + 2, y0 + self.BOX + 2,
                          Theme.R_SM + 2, fill=Theme.BG_DARK, outline=Theme.BORDER_FOCUS, width=1)

        # Box
        if not self.enabled:
            box_fill = Theme.BG_TERTIARY
            box_border = Theme.BORDER_SUBTLE
        elif checked:
            box_fill = Theme.BLUE_PRIMARY
            box_border = Theme.BLUE_HOVER
        else:
            box_fill = Theme.BG_TERTIARY
            box_border = Theme.BORDER_STRONG if self.hovered else Theme.BORDER

        self._rounded(x0, y0, x0 + self.BOX, y0 + self.BOX, Theme.R_SM,
                      fill=box_fill, outline=box_border, width=1)

        # Checkmark
        if checked:
            stroke = Theme.INK_ON_GREEN if self.enabled else Theme.TEXT_DISABLED
            self.create_line(x0 + 4, y0 + 9, x0 + 8, y0 + 13,
                             fill=stroke, width=2, capstyle="round")
            self.create_line(x0 + 8, y0 + 13, x0 + 14, y0 + 5,
                             fill=stroke, width=2, capstyle="round")

        # Label
        text_color = self.fg_color if self.enabled else Theme.TEXT_DISABLED
        self.create_text(x0 + self.BOX + self.GAP, int(self["height"]) // 2,
                         text=self.text, anchor="w",
                         font=self._font, fill=text_color)

    def _rounded(self, x1, y1, x2, y2, r, **kw):
        points = [
            x1 + r, y1, x2 - r, y1, x2, y1, x2, y1 + r,
            x2, y2 - r, x2, y2, x2 - r, y2, x1 + r, y2,
            x1, y2, x1, y2 - r, x1, y1 + r, x1, y1
        ]
        return self.create_polygon(points, smooth=True, **kw)

    def _toggle(self, event=None):
        if not self.enabled:
            return
        self.focus_set()
        self.variable.set(not self.variable.get())
        self._sync_a11y()
        self._draw()
        announce_widget(self)
        if self.command:
            self.command()

    def _on_enter(self, event):
        if self.enabled:
            self.hovered = True
            self.config(cursor="hand2")
            self._draw()

    def _on_leave(self, event):
        self.hovered = False
        self.config(cursor="")
        self._draw()

    def _on_focus_in(self, event):
        self.focused = True
        self._sync_a11y()
        self._draw()
        announce_widget(self)

    def _on_focus_out(self, event):
        self.focused = False
        self._sync_a11y()
        self._draw()

    def set_enabled(self, enabled: bool):
        self.enabled = enabled
        if not enabled:
            self.focused = False
        self.config(cursor="hand2" if enabled else "", takefocus=1 if enabled else 0)
        self._sync_a11y()
        self._draw()


class ModernSlider(tk.Frame):
    """Premium slider: rounded track, filled portion in accent color,
    prominent thumb, value pill on the right. Canvas-based so styling is
    fully controlled."""

    TRACK_H = 4
    THUMB_R = 8
    HEIGHT = 28

    def __init__(self, parent, from_=0, to=100, value=0,
                 command=None, bg=None, width=220,
                 accessible_label: str = "Slider", **kwargs):
        self.parent_bg = bg or (parent.cget('bg') if hasattr(parent, 'cget') else Theme.BG_CARD)
        super().__init__(parent, bg=self.parent_bg)

        self.from_ = from_
        self.to = to
        self.value = max(from_, min(to, value))
        self.command = command
        self._width = width
        self._dragging = False
        self._focused = False
        self.enabled = True
        self.accessible_label = accessible_label or "Slider"

        self.canvas = tk.Canvas(self, width=width, height=self.HEIGHT,
                                highlightthickness=0, bg=self.parent_bg, takefocus=1)
        self.canvas.pack(side="left", fill="x", expand=True, padx=(0, 0))

        self.canvas.bind("<Configure>", self._on_resize)
        self.canvas.bind("<Button-1>", self._on_press)
        self.canvas.bind("<B1-Motion>", self._on_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_release)
        self.canvas.bind("<Left>", lambda e: self._step(-1))
        self.canvas.bind("<Right>", lambda e: self._step(1))
        self.canvas.bind("<MouseWheel>", self._on_wheel)
        self.canvas.bind("<FocusIn>", lambda e: self._set_focused(True))
        self.canvas.bind("<FocusOut>", lambda e: self._set_focused(False))
        self._sync_a11y()
        self._draw()

    def _on_resize(self, event):
        self._width = max(60, event.width)
        self._draw()

    def _value_to_x(self, v):
        if self.to == self.from_:
            return self.THUMB_R
        pct = (v - self.from_) / (self.to - self.from_)
        return int(self.THUMB_R + pct * (self._width - self.THUMB_R * 2))

    def _x_to_value(self, x):
        if self._width <= self.THUMB_R * 2:
            return self.from_
        pct = (x - self.THUMB_R) / (self._width - self.THUMB_R * 2)
        pct = max(0.0, min(1.0, pct))
        return self.from_ + pct * (self.to - self.from_)

    def _draw(self):
        self.canvas.delete("all")
        mid = self.HEIGHT // 2
        left = self.THUMB_R
        right = self._width - self.THUMB_R

        # Track background
        self.canvas.create_rectangle(
            left, mid - self.TRACK_H // 2, right, mid + self.TRACK_H // 2,
            fill=Theme.BG_TERTIARY, outline="",
        )

        thumb_x = self._value_to_x(self.value)
        # Filled portion
        if thumb_x > left:
            self.canvas.create_rectangle(
                left, mid - self.TRACK_H // 2, thumb_x, mid + self.TRACK_H // 2,
                fill=Theme.BLUE_PRIMARY if self.enabled else Theme.BORDER_SUBTLE,
                outline="",
            )

        # Thumb
        self.canvas.create_oval(
            thumb_x - self.THUMB_R - 1, mid - self.THUMB_R - 1,
            thumb_x + self.THUMB_R + 1, mid + self.THUMB_R + 1,
            fill=Theme.BG_DARK, outline="",
        )
        self.canvas.create_oval(
            thumb_x - self.THUMB_R, mid - self.THUMB_R,
            thumb_x + self.THUMB_R, mid + self.THUMB_R,
            fill=Theme.BLUE_PRIMARY if self.enabled else Theme.BG_TERTIARY,
            outline=Theme.BLUE_HOVER if self.enabled else Theme.BORDER_SUBTLE,
            width=1,
        )
        if self._focused and self.enabled:
            self.canvas.create_rectangle(
                1, 1, self._width - 1, self.HEIGHT - 1,
                outline=Theme.BORDER_FOCUS, width=1,
            )

    def _set_focused(self, focused: bool):
        self._focused = focused
        self._sync_a11y()
        self._draw()
        if focused:
            announce_widget(self)

    def _sync_a11y(self):
        value = getattr(self, "value", getattr(self, "from_", 0))
        from_value = getattr(self, "from_", 0)
        to_value = getattr(self, "to", 100)
        enabled = getattr(self, "enabled", True)
        focused = getattr(self, "_focused", False)
        set_accessible_metadata(
            self,
            role="slider",
            label=getattr(self, "accessible_label", "Slider"),
            state=_state_text(
                "enabled" if enabled else "disabled",
                "focused" if focused else "",
            ),
            value=f"{int(value)} (range {from_value} to {to_value})",
        )
        try:
            set_accessible_metadata(
                self.canvas,
                role="slider",
                label=getattr(self, "accessible_label", "Slider"),
                state=accessible_metadata(self).get("state", ""),
                value=accessible_metadata(self).get("value", ""),
            )
        except Exception:
            pass

    def accessibility_snapshot(self) -> dict:
        return accessible_metadata(self)

    def _on_press(self, event):
        if not self.enabled:
            return
        self.canvas.focus_set()
        self._dragging = True
        self._set_from_x(event.x)

    def _on_drag(self, event):
        if self.enabled and self._dragging:
            self._set_from_x(event.x)

    def _on_release(self, event):
        self._dragging = False

    def _on_wheel(self, event):
        if not self.enabled:
            return
        self._step(1 if event.delta > 0 else -1)

    def _step(self, direction):
        if not self.enabled:
            return
        step = max(1, int((self.to - self.from_) / 50))
        new_val = max(self.from_, min(self.to, int(self.value) + direction * step))
        self._set_value(new_val)

    def _set_from_x(self, x):
        new_val = int(round(self._x_to_value(x)))
        self._set_value(new_val)

    def _set_value(self, v):
        v = max(self.from_, min(self.to, v))
        if v == self.value:
            return
        self.value = v
        self._sync_a11y()
        self._draw()
        if self._focused:
            announce_widget(self)
        if self.command:
            self.command(v)

    def set(self, v):
        self._set_value(int(v))

    def get(self):
        return int(self.value)

    def set_enabled(self, enabled: bool):
        self.enabled = enabled
        if not enabled:
            self._dragging = False
            self._focused = False
        self.canvas.config(
            cursor="hand2" if enabled else "",
            takefocus=1 if enabled else 0,
        )
        self._sync_a11y()
        self._draw()


def _confirm_default_focus(tone: str, requested: Optional[str] = None) -> str:
    """Return the safest initial focus target for a confirmation dialog."""
    if requested in {"confirm", "cancel"}:
        return requested
    return "cancel" if tone == "danger" else "confirm"


def show_confirm(parent, title: str, message: str, detail: str = "",
                 confirm_label: str = "Confirm",
                 cancel_label: str = "Cancel",
                 tone: str = "primary",
                 default_focus: Optional[str] = None) -> bool:
    """Themed modal confirmation dialog that matches the app aesthetic.

    Returns True if confirmed, False if cancelled (or closed).
    `tone` selects the confirm button style: primary / danger / accent.
    Destructive dialogs focus the safe action unless explicitly overridden.
    """
    result = {"value": False}

    dialog = tk.Toplevel(parent)
    dialog.withdraw()
    title_text = tr(title)
    message_text = tr(message)
    detail_text = tr(detail) if detail else ""
    confirm_text = tr(confirm_label)
    cancel_text = tr(cancel_label)

    dialog.title(title_text)
    dialog.configure(bg=Theme.BG_OVERLAY)
    dialog.resizable(False, False)
    dialog.transient(parent)
    set_accessible_metadata(
        dialog,
        role="dialog",
        label=title_text,
        state="modal",
        description=" ".join(part for part in (message_text, detail_text) if part),
    )

    outer = tk.Frame(dialog, bg=Theme.BORDER, padx=1, pady=1)
    outer.pack()
    body = tk.Frame(outer, bg=Theme.BG_SECONDARY)
    body.pack()

    # Content
    content = tk.Frame(body, bg=Theme.BG_SECONDARY)
    content.pack(padx=28, pady=(24, 14))

    tk.Label(content, text=title_text, font=f(Theme.F_HEADING, "bold"),
             bg=Theme.BG_SECONDARY, fg=Theme.TEXT_PRIMARY,
             anchor="w", justify="left").pack(anchor="w")
    tk.Label(content, text=message_text, font=f(Theme.F_BODY),
             bg=Theme.BG_SECONDARY, fg=Theme.TEXT_SECONDARY,
             anchor="w", justify="left", wraplength=420).pack(
                 anchor="w", pady=(6, 0))
    if detail_text:
        tk.Label(content, text=detail_text, font=f(Theme.F_BODY_SM),
                 bg=Theme.BG_SECONDARY, fg=Theme.TEXT_MUTED,
                 anchor="w", justify="left", wraplength=420).pack(
                     anchor="w", pady=(8, 0))

    # Action row
    actions = tk.Frame(body, bg=Theme.BG_CARD)
    actions.pack(fill="x")
    inner_actions = tk.Frame(actions, bg=Theme.BG_CARD)
    inner_actions.pack(side="right", padx=16, pady=14)

    def _cancel():
        dialog.grab_release()
        dialog.destroy()

    def _confirm():
        result["value"] = True
        dialog.grab_release()
        dialog.destroy()

    cancel_btn = ModernButton(inner_actions, text=cancel_text, width=96,
                              command=_cancel, style="ghost", size="md")
    cancel_btn.pack(side="left")

    confirm_btn = ModernButton(inner_actions, text=confirm_text, width=118,
                               command=_confirm, style=tone, size="md")
    confirm_btn.pack(side="left", padx=(Theme.S_SM, 0))

    dialog.bind("<Escape>", lambda e: _cancel())
    dialog.protocol("WM_DELETE_WINDOW", _cancel)

    # Center on parent
    dialog.update_idletasks()
    try:
        px = parent.winfo_rootx()
        py = parent.winfo_rooty()
        pw = parent.winfo_width()
        ph = parent.winfo_height()
        dw = dialog.winfo_reqwidth()
        dh = dialog.winfo_reqheight()
        x = px + (pw - dw) // 2
        y = py + (ph - dh) // 3
        dialog.geometry(f"+{x}+{y}")
    except Exception:
        pass

    dialog.deiconify()
    dialog.grab_set()
    if _confirm_default_focus(tone, default_focus) == "cancel":
        cancel_btn.focus_set()
    else:
        confirm_btn.focus_set()
    announce(
        f"{title_text}. {message_text}",
        importance="high" if tone == "danger" else "normal",
    )
    dialog.wait_window()
    return result["value"]


class TaskbarProgress:
    """Thin wrapper over ITaskbarList3 for Windows 7+ taskbar progress.

    Falls back to no-op on non-Windows or when COM is unavailable.
    State values per MSDN:
        0 = NOPROGRESS, 1 = INDETERMINATE, 2 = NORMAL, 4 = ERROR, 8 = PAUSED
    """

    STATE_NONE = 0
    STATE_INDETERMINATE = 1
    STATE_NORMAL = 2
    STATE_ERROR = 4
    STATE_PAUSED = 8

    def __init__(self, hwnd):
        self._taskbar = None
        self._hwnd = hwnd
        if sys.platform != "win32":
            return
        try:
            import comtypes.client  # type: ignore
            # CLSID_TaskbarList
            self._taskbar = comtypes.client.CreateObject(
                "{56FDF344-FD6D-11D0-958A-006097C9A090}",
                interface=comtypes.GUID("{EA1AFB91-9E28-4B86-90E9-9E9F8A5EEFAF}"),
            )
            self._taskbar.HrInit()
        except Exception:
            self._taskbar = None

    def set_value(self, current: int, total: int):
        if not self._taskbar or not self._hwnd:
            return
        try:
            self._taskbar.SetProgressValue(self._hwnd, current, max(total, 1))
        except Exception:
            pass

    def set_state(self, state: int):
        if not self._taskbar or not self._hwnd:
            return
        try:
            self._taskbar.SetProgressState(self._hwnd, state)
        except Exception:
            pass

    def clear(self):
        self.set_state(self.STATE_NONE)


def make_themed_menu(parent) -> tk.Menu:
    """Create a `tk.Menu` styled for the dark theme."""
    menu = tk.Menu(
        parent,
        tearoff=0,
        bg=Theme.BG_RAISED,
        fg=Theme.TEXT_PRIMARY,
        activebackground=Theme.BLUE_MUTED,
        activeforeground=Theme.TEXT_PRIMARY,
        disabledforeground=Theme.TEXT_DISABLED,
        relief="flat",
        bd=0,
        font=f(Theme.F_BODY_SM),
        activeborderwidth=0,
    )
    return menu


class Toast:
    """Lightweight transient notification, anchored to the bottom-right of
    the root window. Fades after TIMEOUT_MS."""

    TIMEOUT_MS = 2600
    _active: List['Toast'] = []

    def __init__(self, root, message: str, tone: str = "success"):
        self.root = root
        self.message = message
        self.tone = tone
        self._win = None
        self._fade_id = None
        self._build()
        Toast._active.append(self)
        self._schedule_close()

    @classmethod
    def show(cls, root, message: str, tone: str = "success"):
        return cls(root, message, tone)

    def _tone_color(self):
        return {
            "success": Theme.SUCCESS,
            "warning": Theme.WARNING,
            "error": Theme.ERROR,
            "info": Theme.INFO,
        }.get(self.tone, Theme.TEXT_SECONDARY)

    def _build(self):
        try:
            self._win = tk.Toplevel(self.root)
            self._win.wm_overrideredirect(True)
            self._win.configure(bg=Theme.BORDER_STRONG)
            set_accessible_metadata(
                self._win,
                role="notification",
                label="Notification",
                state=self.tone,
                value=self.message,
            )
            self._win.attributes("-topmost", True)
            try:
                self._win.attributes("-alpha", 0.97)
            except tk.TclError:
                pass

            card = tk.Frame(self._win, bg=Theme.BG_RAISED)
            card.pack(padx=1, pady=1)

            # Left color stripe
            stripe = tk.Frame(card, bg=self._tone_color(), width=3)
            stripe.pack(side="left", fill="y")

            content = tk.Frame(card, bg=Theme.BG_RAISED)
            content.pack(side="left", padx=(12, 18), pady=10)

            tk.Label(
                content,
                text=self.message,
                font=f(Theme.F_BODY_SM, "bold"),
                bg=Theme.BG_RAISED,
                fg=Theme.TEXT_PRIMARY,
                justify="left",
                wraplength=420,
            ).pack(anchor="w")

            self._win.update_idletasks()
            self._position()
        except tk.TclError:
            self._win = None

    def _position(self):
        try:
            w = self._win.winfo_reqwidth()
            h = self._win.winfo_reqheight()
            rx = self.root.winfo_rootx()
            ry = self.root.winfo_rooty()
            rw = self.root.winfo_width()
            rh = self.root.winfo_height()
            # Stack toasts upward from the bottom-right. During _build
            # self is not yet in _active, so every active toast is a
            # predecessor; on restack only the toasts before self are.
            if self in Toast._active:
                predecessors = Toast._active[:Toast._active.index(self)]
            else:
                predecessors = list(Toast._active)
            offset = sum((t._win.winfo_reqheight() + 8)
                         for t in predecessors if t._win)
            x = rx + rw - w - 20
            y = ry + rh - h - 52 - offset
            self._win.wm_geometry(f"+{x}+{y}")
        except Exception:
            pass

    def _schedule_close(self):
        try:
            self._fade_id = self.root.after(self.TIMEOUT_MS, self._begin_fade)
        except tk.TclError:
            pass

    def _begin_fade(self):
        """Fade the toast out over ~300ms using the -alpha attribute, then
        destroy and restack any later toasts."""
        if not self._win:
            return
        steps = [0.85, 0.65, 0.45, 0.25, 0.08]

        def apply(i):
            if not self._win:
                return
            try:
                self._win.attributes("-alpha", steps[i])
            except tk.TclError:
                pass
            if i + 1 < len(steps):
                try:
                    self.root.after(45, lambda: apply(i + 1))
                except tk.TclError:
                    pass
            else:
                self._close()

        apply(0)

    def _close(self):
        try:
            if self._win:
                self._win.destroy()
        except tk.TclError:
            pass
        self._win = None
        if self in Toast._active:
            Toast._active.remove(self)
        for t in list(Toast._active):
            try:
                t._position()
            except Exception:
                pass


class SegmentedPicker(tk.Frame):
    """A segmented radio-style selector. Renders a horizontal group of
    Canvas-based buttons. Used for the algorithm picker."""

    def __init__(self, parent, options: List[Tuple[str, str]],
                 value: str = None, command: Callable = None,
                 bg: str = None, group_label: str = "Selection", **kwargs):
        """options: list of (value, label) tuples."""
        self.parent_bg = bg or (parent.cget('bg') if hasattr(parent, 'cget')
                                else Theme.BG_CARD)
        super().__init__(parent, bg=self.parent_bg)
        self.options = options
        self.value = value or (options[0][0] if options else None)
        self.command = command
        self.group_label = group_label
        self._segments: dict = {}

        wrap = tk.Frame(self, bg=Theme.BG_TERTIARY, highlightthickness=1,
                        highlightbackground=Theme.BORDER)
        wrap.pack(fill="x")

        for val, label in options:
            seg = _Segment(wrap, label=label, value=val,
                            on_select=self._select,
                            selected=(val == self.value),
                            group_label=self.group_label)
            seg.pack(side="left", fill="x", expand=True, padx=1, pady=1)
            self._segments[val] = seg

    def _select(self, val):
        if val == self.value:
            return
        self.value = val
        for v, seg in self._segments.items():
            seg.set_selected(v == val)
        if self.command:
            self.command(val)

    def set(self, val: str):
        if val in self._segments:
            self._select(val)

    def get(self) -> str:
        return self.value


class _Segment(tk.Canvas):
    """Single button inside a SegmentedPicker."""

    H = 30

    def __init__(self, parent, label: str, value: str, on_select: Callable,
                 selected: bool = False, group_label: str = "Selection"):
        super().__init__(parent, height=self.H, highlightthickness=0,
                         bg=Theme.BG_TERTIARY, takefocus=1)
        self.label = label
        self.value = value
        self.on_select = on_select
        self.selected = selected
        self.group_label = group_label
        self.enabled = True
        self.hovered = False
        self.focused = False

        self.bind("<Button-1>", self._click)
        self.bind("<Return>", self._click)
        self.bind("<space>", self._click)
        self.bind("<Enter>", self._on_enter)
        self.bind("<Leave>", self._on_leave)
        self.bind("<FocusIn>", lambda e: self._set_focused(True))
        self.bind("<FocusOut>", lambda e: self._set_focused(False))
        self.bind("<Configure>", lambda e: self._draw())
        self._sync_a11y()
        self._draw()

    def _on_enter(self, event):
        if not self.enabled:
            return
        self.hovered = True
        self.config(cursor="hand2")
        self._draw()

    def _on_leave(self, event):
        self.hovered = False
        self.config(cursor="")
        self._draw()

    def _set_focused(self, focused):
        if focused and not self.enabled:
            return
        self.focused = focused
        self._sync_a11y()
        self._draw()
        if focused:
            announce_widget(self)

    def _click(self, event=None):
        if not self.enabled:
            return "break"
        self.focus_set()
        self.on_select(self.value)
        announce_widget(self)
        return "break"

    def set_selected(self, selected: bool):
        self.selected = selected
        self._sync_a11y()
        self._draw()

    def _sync_a11y(self):
        set_accessible_metadata(
            self,
            role="radio button",
            label=self.label,
            state=_state_text(
                "enabled" if self.enabled else "disabled",
                "selected" if self.selected else "not selected",
                "focused" if self.focused else "",
            ),
            value=str(self.value),
            description=self.group_label,
        )

    def accessibility_snapshot(self) -> dict:
        return accessible_metadata(self)

    def set_enabled(self, enabled: bool):
        self.enabled = enabled
        if not enabled:
            self.focused = False
            self.hovered = False
        self.config(cursor="hand2" if enabled else "", takefocus=1 if enabled else 0)
        self._sync_a11y()
        self._draw()

    def _draw(self):
        self.delete("all")
        w = max(1, int(self["width"]) if int(self["width"]) > 1 else self.winfo_width())
        if w <= 1:
            w = self.winfo_width()
        h = self.H
        if not self.enabled:
            bg = Theme.BG_TERTIARY
            fg = Theme.TEXT_DISABLED
            border = Theme.BORDER_SUBTLE
        elif self.selected:
            bg = Theme.GREEN_MUTED
            fg = Theme.GREEN_PRIMARY
            border = Theme.GREEN_HOVER
        elif self.hovered:
            bg = Theme.BG_CARD_HOVER
            fg = Theme.TEXT_PRIMARY
            border = Theme.BORDER_STRONG
        else:
            bg = Theme.BG_TERTIARY
            fg = Theme.TEXT_SECONDARY
            border = Theme.BG_TERTIARY
        self.create_rectangle(0, 0, w, h, fill=bg, outline=border, width=1)
        if self.focused:
            self.create_rectangle(1, 1, w - 1, h - 1, outline=Theme.BORDER_FOCUS,
                                  width=1)
        font_w = "bold" if self.selected else "normal"
        self.create_text(w // 2, h // 2, text=self.label,
                         fill=fg, font=f(Theme.F_BODY_SM, font_w))


class DragDropFrame(tk.Frame):
    """A calm drop target surface with a single clear import action."""

    def __init__(self, parent, on_drop: Callable[[List[str]], None],
                 width=400, height=200, **kwargs):
        super().__init__(parent, bg=Theme.BG_CARD, highlightthickness=1,
                        highlightbackground=Theme.BORDER, highlightcolor=Theme.BLUE_PRIMARY,
                        takefocus=1)

        self.on_drop = on_drop
        self.normal_bg = Theme.BG_CARD
        self.hover_bg = Theme.BG_CARD_HOVER
        self.hovered = False
        self.focused = False
        self.import_enabled = True
        self.configure(height=height)
        self.pack_propagate(False)
        self.grid_propagate(False)
        self.config(cursor="hand2")

        # Inner content
        inner = tk.Frame(self, bg=self.normal_bg)
        inner.place(relx=0.5, rely=0.5, anchor="center")
        self._surface_widgets = [self, inner]

        # Import glyph
        glyph = tk.Label(inner, text="+", font=f(22, "bold"),
                         bg=self.normal_bg, fg=Theme.BLUE_PRIMARY)
        glyph.pack()

        # Main text
        main_text = tk.Label(inner, text=tr("Add files to the queue"),
                            font=f(Theme.F_TITLE, "bold"), bg=self.normal_bg,
                            fg=Theme.TEXT_PRIMARY)
        main_text.pack(pady=(2, 0))

        # Sub text -- updated after DnD setup to reflect actual capabilities
        self._sub_text = tk.Label(inner,
                           text=tr("Drag files here, choose files, or choose a folder. Originals stay untouched."),
                           font=f(Theme.F_BODY_SM), bg=self.normal_bg,
                           fg=Theme.TEXT_SECONDARY, justify="center", wraplength=480)
        self._sub_text.pack(pady=(6, 12))
        sub_text = self._sub_text

        actions = tk.Frame(inner, bg=self.normal_bg)
        actions.pack()

        self.add_files_btn = ModernButton(actions, text=tr("Choose files"), width=124,
                                          command=self._open_file_dialog,
                                          style="accent", size="md")
        self.add_files_btn.pack(side="left")

        self.add_folder_btn = ModernButton(actions, text=tr("Choose folder"), width=118,
                                           command=self._open_folder_dialog,
                                           style="secondary", size="md")
        self.add_folder_btn.pack(side="left", padx=(8, 0))

        support_text = tk.Label(inner,
                                text=tr("Videos and images supported"),
                                font=f(Theme.F_META, "bold"), bg=self.normal_bg,
                                fg=Theme.TEXT_DISABLED)
        support_text.pack(pady=(12, 0))
        self._surface_widgets.extend([glyph, main_text, sub_text, actions, support_text])

        # Bind click (left = files, right = folder)
        self.bind("<Button-1>", self._on_click)
        self.bind("<Button-3>", self._on_right_click)
        self.bind("<Enter>", self._on_enter, add="+")
        self.bind("<Leave>", self._on_leave, add="+")
        self.bind("<FocusIn>", self._on_focus_in, add="+")
        self.bind("<FocusOut>", self._on_focus_out, add="+")
        self.bind("<Return>", lambda e: self._open_file_dialog())
        self.bind("<space>", lambda e: self._open_file_dialog())
        for child in (inner, glyph, main_text, sub_text, support_text):
            child.bind("<Button-1>", self._on_click)
            child.bind("<Button-3>", self._on_right_click)
            child.bind("<Enter>", self._on_enter, add="+")
            child.bind("<Leave>", self._on_leave, add="+")

        # Try to enable native drag-drop (Windows)
        self._dnd_available = False
        try:
            self._setup_dnd()
        except Exception:
            pass
        if not self._dnd_available:
            self._sub_text.config(
                text=tr("Choose files or a folder below. Originals stay untouched.")
            )
        self._sync_a11y()

    def _set_bg(self, bg: str, border: str):
        self.config(bg=bg, highlightbackground=border)
        for widget in self._surface_widgets:
            if isinstance(widget, tk.Widget):
                try:
                    widget.config(bg=bg)
                except tk.TclError:
                    pass
        for button in (self.add_files_btn, self.add_folder_btn):
            button.config(bg=bg)

    def set_import_enabled(self, enabled: bool):
        """Enable or disable import buttons during processing."""
        self.import_enabled = enabled
        self.add_files_btn.set_enabled(enabled)
        self.add_folder_btn.set_enabled(enabled)
        self.config(cursor="hand2" if enabled else "", takefocus=1 if enabled else 0)
        if not enabled:
            self.focused = False
        self._sync_a11y()

    def _sync_a11y(self):
        set_accessible_metadata(
            self,
            role="drop target",
            label=tr("Add files to the queue"),
            state=_state_text(
                tr("enabled") if self.import_enabled else tr("disabled"),
                tr("focused") if self.focused else "",
                tr("drag and drop available") if self._dnd_available else "",
            ),
            description=self._sub_text.cget("text"),
        )

    def accessibility_snapshot(self) -> dict:
        return accessible_metadata(self)

    def _setup_dnd(self):
        """Setup native drag and drop if available."""
        try:
            import tkinterdnd2
            self.drop_target_register(tkinterdnd2.DND_FILES)
            self.dnd_bind('<<Drop>>', self._handle_drop)
            self._dnd_available = True
        except ImportError:
            pass

    def _handle_drop(self, event):
        if not getattr(self, "import_enabled", True):
            return
        files = self.tk.splitlist(event.data)
        if files:
            self.on_drop(list(files))

    def _open_file_dialog(self):
        if not getattr(self, "import_enabled", True):
            return
        files = filedialog.askopenfilenames(
            title=tr("Choose files to clean"),
            filetypes=[
                (tr("All Supported"), filepicker_pattern(SUPPORTED_EXTENSIONS)),
                (tr("Video Files"), filepicker_pattern(VIDEO_EXTENSIONS)),
                (tr("Image Files"), filepicker_pattern(IMAGE_EXTENSIONS)),
                (tr("All Files"), "*.*"),
            ]
        )
        if files:
            self.on_drop(list(files))

    def _open_folder_dialog(self):
        if not getattr(self, "import_enabled", True):
            return
        folder = filedialog.askdirectory(title=tr("Choose a folder to clean"))
        if folder:
            self.on_drop([folder])

    def _on_click(self, event):
        self.focus_set()
        self._open_file_dialog()

    def _on_right_click(self, event):
        self.focus_set()
        self._open_folder_dialog()

    def _on_enter(self, event):
        self.hovered = True
        self._set_bg(self.hover_bg, Theme.BORDER_FOCUS if self.focused else Theme.BLUE_PRIMARY)

    def _on_leave(self, event):
        self.hovered = False
        self._set_bg(self.normal_bg, Theme.BORDER_FOCUS if self.focused else Theme.BORDER)

    def _on_focus_in(self, event):
        self.focused = True
        self._set_bg(self.hover_bg if self.hovered else self.normal_bg, Theme.BORDER_FOCUS)
        self._sync_a11y()
        announce_widget(self)

    def _on_focus_out(self, event):
        self.focused = False
        self._set_bg(self.hover_bg if self.hovered else self.normal_bg,
                     Theme.BLUE_PRIMARY if self.hovered else Theme.BORDER)
        self._sync_a11y()


class QueueItemWidget(tk.Frame):
    """A single queue item card. Clear hierarchy: filename + status pill,
    compact meta row, progress bar, and row of actions. Selected state
    shows a left-edge accent stripe."""

    def __init__(self, parent, item: QueueItem, on_remove: Callable,
                 on_select: Callable = None, on_rename: Callable = None,
                 on_repeat: Callable = None, on_cancel_item: Callable = None,
                 on_override: Callable = None,
                 on_soft_action: Callable = None,
                 on_retry_suggested: Callable = None,
                 **kwargs):
        super().__init__(parent, bg=Theme.BG_CARD, highlightthickness=1,
                        highlightbackground=Theme.BORDER, takefocus=1)

        self.item = item
        self.on_remove = on_remove
        self.on_select = on_select
        self.on_rename = on_rename
        self.on_repeat = on_repeat
        self.on_cancel_item = on_cancel_item
        self.on_override = on_override
        self.on_soft_action = on_soft_action
        self.on_retry_suggested = on_retry_suggested
        self.is_selected = False
        self.focused = False
        self._surface_bg = Theme.BG_CARD
        self._pulse_id = None
        self._pulse_phase = 0
        self._last_a11y_status = None

        # Left accent stripe (visible only when selected)
        self.accent_stripe = tk.Frame(self, bg=Theme.BG_CARD, width=3)
        self.accent_stripe.pack(side="left", fill="y")

        # Main container with padding
        self.container = tk.Frame(self, bg=self._surface_bg)
        self.container.pack(fill="x", padx=Theme.S_MD, pady=Theme.S_MD)

        # Top row: filename and status
        self.top_row = tk.Frame(self.container, bg=self._surface_bg)
        self.top_row.pack(fill="x")

        self.name_label = tk.Label(self.top_row,
                                   text=truncate_middle(Path(item.file_path).name, 46),
                                   font=f(Theme.F_BODY, "bold"),
                                   bg=self._surface_bg, fg=Theme.TEXT_PRIMARY,
                                   cursor="hand2")
        self.name_label.pack(side="left")
        Tooltip(self.name_label, item.file_path)

        # Status pill (rounded by adding generous padx)
        badge = status_ui(item.status)
        self.status_badge = tk.Label(self.top_row, text=badge["label"],
                                     font=f(Theme.F_META, "bold"),
                                     bg=badge["bg"], fg=badge["color"],
                                     padx=Theme.S_SM, pady=Theme.S_XS)
        self.status_badge.pack(side="right")

        # File info row (meta caption)
        self.info_label = tk.Label(self.container,
                                   text=_queue_item_info_text(item),
                                   font=f(Theme.F_META),
                                   bg=self._surface_bg, fg=Theme.TEXT_MUTED, anchor="w")
        self.info_label.pack(fill="x", pady=(Theme.S_XS, 0))

        # Progress bar (resizes with container)
        self.progress_bar = ModernProgressBar(self.container, width=300, height=5,
                                              fill=self._get_status_color())
        self.progress_bar.pack(fill="x", pady=(Theme.S_MD, Theme.S_XS))
        self.progress_bar.set_progress(item.progress)
        def _resize_bar(event):
            bar_w = event.width - 4
            if bar_w > 20:
                self.progress_bar.resize(bar_w)
        self.container.bind("<Configure>", _resize_bar)

        # Bottom row: message + elapsed time
        self.bottom_row = tk.Frame(self.container, bg=self._surface_bg)
        self.bottom_row.pack(fill="x")

        self.message_label = tk.Label(self.bottom_row, text=item.message or tr("Ready to process"),
                                      font=f(Theme.F_BODY_SM), bg=self._surface_bg,
                                      fg=Theme.TEXT_SECONDARY, anchor="w")
        self.message_label.pack(side="left", fill="x", expand=True)

        self.time_label = tk.Label(self.bottom_row, text="",
                                   font=f(Theme.F_META, "bold"),
                                   bg=self._surface_bg, fg=Theme.TEXT_MUTED, anchor="e")
        self.time_label.pack(side="right")

        self.actions_row = tk.Frame(self.container, bg=self._surface_bg)
        self.actions_row.pack(fill="x", pady=(Theme.S_MD, 0))

        self.remove_btn = ModernButton(self.actions_row, text=tr("Remove"), width=78,
                                       command=lambda: self.on_remove(self.item.id),
                                       style="ghost", size="sm")
        self.remove_btn.pack(side="left")

        self.open_btn = ModernButton(self.actions_row, text=tr("Open result"), width=104,
                                     command=self._open_output, style="accent",
                                     size="sm")
        self.open_btn.pack(side="right")

        self._interactive_widgets = [
            self, self.accent_stripe, self.container, self.top_row,
            self.name_label, self.info_label,
            self.bottom_row, self.message_label, self.time_label, self.actions_row,
            self.status_badge, self.progress_bar,
        ]
        for widget in self._interactive_widgets:
            widget.bind("<Enter>", self._on_enter, add="+")
            widget.bind("<Leave>", self._on_leave, add="+")
            widget.bind("<Button-1>", self._on_card_click, add="+")
            widget.bind("<Button-3>", self._on_context_menu, add="+")
        self.bind("<FocusIn>", self._on_focus_in, add="+")
        self.bind("<FocusOut>", self._on_focus_out, add="+")
        self.bind("<Return>", self._on_card_activate, add="+")
        self.bind("<space>", self._on_card_activate, add="+")

        self._sync_a11y()
        self.update_item(item)

    def _on_context_menu(self, event):
        """Show a themed right-click menu for this queue item."""
        menu = make_themed_menu(self)
        is_active = self.item.status in (
            ProcessingStatus.LOADING, ProcessingStatus.DETECTING,
            ProcessingStatus.PROCESSING, ProcessingStatus.MERGING,
        )
        is_complete = (self.item.status == ProcessingStatus.COMPLETE
                       and Path(self.item.output_path).exists())

        menu.add_command(label=tr("Preview source frame"),
                         command=self._request_preview)
        menu.add_command(label=tr("Review subtitle mask"),
                         command=self._request_mask_preview)
        menu.add_separator()
        menu.add_command(label=tr("Open result"),
                         command=self._open_output,
                         state="normal" if is_complete else "disabled")
        menu.add_command(label=tr("Open quality sheet"),
                         command=self._open_quality_sheet,
                         state="normal" if self._quality_sheet_path() else "disabled")
        menu.add_command(label=tr("Retry with suggested settings"),
                         command=lambda: self.on_retry_suggested(self.item.id)
                         if self.on_retry_suggested else None,
                         state=(
                             "normal"
                             if self.on_retry_suggested and self._needs_quality_review()
                             else "disabled"
                         ))
        menu.add_command(label=tr("Reveal output folder"),
                         command=self._reveal_output,
                         state="normal" if is_complete else "disabled")
        menu.add_separator()
        # Only allow renaming output before processing has started.
        rename_allowed = self.item.status == ProcessingStatus.IDLE and self.on_rename is not None
        menu.add_command(label=tr("Rename output..."),
                         command=lambda: self.on_rename(self.item.id) if self.on_rename else None,
                         state="normal" if rename_allowed else "disabled")
        menu.add_command(label=tr("Copy source path"),
                         command=self._copy_source_path)
        menu.add_command(label=tr("Copy CLI command"),
                         command=self._copy_cli_command)
        # RM-28: re-queue the same source with the snapshot of settings
        # that was active on this item. Useful when re-running with a
        # tweaked global config but you still want exactly the same
        # per-file overrides as a previous run.
        if self.on_repeat is not None:
            menu.add_command(label=tr("Repeat with these settings"),
                             command=lambda: self.on_repeat(self.item.id))
        # F-7: per-item cancel. Only meaningful while the item is
        # actively running -- on IDLE entries we surface "Remove"
        # below as the equivalent action.
        if self.on_cancel_item is not None and is_active:
            menu.add_command(label=tr("Cancel this item"),
                             command=lambda: self.on_cancel_item(self.item.id))
        # RM-29: open the per-file override dialog so users can change
        # mode / language / sensitivity for a single queued item
        # without touching the global settings.
        if self.on_override is not None and self.item.status == ProcessingStatus.IDLE:
            menu.add_command(label=tr("Override settings for this file..."),
                             command=lambda: self.on_override(self.item.id))
        if (self.on_soft_action is not None
                and self.item.status == ProcessingStatus.IDLE
                and getattr(self.item, "soft_subtitle_streams", None)):
            menu.add_separator()
            menu.add_command(
                label=tr("Fast strip embedded subtitles"),
                command=lambda: self.on_soft_action(self.item.id, "strip"),
            )
            menu.add_command(
                label=tr("Fast remux and keep embedded subtitles"),
                command=lambda: self.on_soft_action(self.item.id, "keep_all"),
            )
            menu.add_command(
                label=tr("Run burned-in cleanup instead"),
                command=lambda: self.on_soft_action(self.item.id, "burned_in"),
            )
        menu.add_separator()
        menu.add_command(label=tr("Remove from queue"),
                         command=lambda: self.on_remove(self.item.id),
                         state="disabled" if is_active else "normal")

        try:
            menu.tk_popup(event.x_root, event.y_root)
        finally:
            menu.grab_release()

    def _reveal_output(self):
        """Open the folder containing the output in Explorer."""
        if self.item.status == ProcessingStatus.COMPLETE and Path(self.item.output_path).exists():
            try:
                os.startfile(str(Path(self.item.output_path).parent))
            except Exception as exc:
                logger.warning(f"Could not open output folder: {exc}")

    def _copy_source_path(self):
        """Copy the source file path to the clipboard."""
        try:
            self.clipboard_clear()
            self.clipboard_append(self.item.file_path)
        except tk.TclError:
            pass

    def _copy_cli_command(self):
        """Copy a CLI command that reproduces this item's settings."""
        try:
            cmd = _build_cli_command(self.item)
            self.clipboard_clear()
            self.clipboard_append(cmd)
        except tk.TclError:
            pass

    def _request_preview(self):
        if self.on_select:
            self.on_select(self.item)

    def _request_mask_preview(self):
        if self.on_select:
            self.on_select(self.item, show_mask=True)

    def _on_card_click(self, event):
        self.focus_set()
        if self.on_select:
            self.on_select(self.item)

    def _on_card_activate(self, event=None):
        if self.on_select:
            self.on_select(self.item)
        return "break"

    def _on_enter(self, event):
        if not self.is_selected:
            border = Theme.BORDER_FOCUS if self.focused else Theme.BORDER
            self._apply_surface_state(Theme.BG_CARD_HOVER, border)

    def _on_leave(self, event):
        if not self.is_selected:
            border = Theme.BORDER_FOCUS if self.focused else Theme.BORDER
            accent = Theme.BORDER_FOCUS if self.focused else None
            self._apply_surface_state(Theme.BG_CARD, border, accent=accent)

    def _on_focus_in(self, event):
        self.focused = True
        if not self.is_selected:
            self._apply_surface_state(
                Theme.BG_CARD_HOVER,
                Theme.BORDER_FOCUS,
                accent=Theme.BORDER_FOCUS,
            )
        self._sync_a11y()
        announce_widget(self)

    def _on_focus_out(self, event):
        self.focused = False
        if not self.is_selected:
            self._apply_surface_state(Theme.BG_CARD, Theme.BORDER)
        self._sync_a11y()

    def _apply_surface_state(self, bg: str, border: str, accent: str = None):
        self._surface_bg = bg
        self.config(bg=bg, highlightbackground=border)
        for widget in (self.container, self.name_label, self.info_label, self.message_label,
                       self.time_label):
            widget.config(bg=bg)
        for widget in (self.top_row, self.bottom_row, self.actions_row):
            widget.config(bg=bg)
        self.progress_bar.config(bg=bg)
        for button in (self.remove_btn, self.open_btn):
            button.config(bg=bg)
        # Accent stripe: painted when a value is passed, otherwise matches bg
        self.accent_stripe.config(bg=accent or bg)

    def set_selected(self, selected: bool):
        self.is_selected = selected
        if selected:
            self._apply_surface_state(
                Theme.BG_CARD_SELECTED, Theme.BLUE_PRIMARY, accent=Theme.BLUE_PRIMARY)
        elif self.focused:
            self._apply_surface_state(
                Theme.BG_CARD_HOVER,
                Theme.BORDER_FOCUS,
                accent=Theme.BORDER_FOCUS,
            )
        else:
            self._apply_surface_state(Theme.BG_CARD, Theme.BORDER)
        self._sync_a11y()

    def _open_output(self):
        """Open the output file if processing is complete."""
        if self.item.status == ProcessingStatus.COMPLETE and Path(self.item.output_path).exists():
            try:
                os.startfile(self.item.output_path)
            except Exception:
                pass

    def _quality_sheet_path(self) -> Optional[Path]:
        report = getattr(self.item, "quality_report", None)
        if not isinstance(report, dict):
            return None
        sheet = report.get("sheet")
        if not sheet:
            return None
        path = Path(sheet)
        return path if path.exists() else None

    def _needs_quality_review(self) -> bool:
        report = getattr(self.item, "quality_report", None)
        if not isinstance(report, dict):
            return False
        gate = report.get("quality_gate")
        if isinstance(gate, dict) and gate.get("status") == "review":
            return True
        return str(report.get("tag") or "").strip().lower() == "review"

    def _open_quality_sheet(self):
        path = self._quality_sheet_path()
        if path is None:
            return
        try:
            os.startfile(str(path))
        except Exception:
            pass

    def _get_status_color(self) -> str:
        return status_ui(self.item.status)["color"]

    def _sync_a11y(self):
        badge = status_ui(self.item.status)
        pct = int(max(0.0, min(1.0, float(self.item.progress or 0.0))) * 100)
        set_accessible_metadata(
            self,
            role="queue item",
            label=Path(self.item.file_path).name,
            state=_state_text(
                badge["label"],
                tr("selected") if self.is_selected else tr("not selected"),
                tr("focused") if self.focused else "",
            ),
            value=f"{pct}% complete",
            description=self.item.message or tr("Ready to process"),
        )

    def accessibility_snapshot(self) -> dict:
        return accessible_metadata(self)

    def update_item(self, item: QueueItem):
        self.item = item
        badge = status_ui(item.status)
        self.status_badge.config(text=badge["label"], fg=badge["color"], bg=badge["bg"])
        self.info_label.config(text=_queue_item_info_text(item))
        self.progress_bar.set_progress(item.progress)
        self.progress_bar.set_color(self._get_status_color())
        status_message = truncate_middle(item.message or tr("Ready to process"), 74)
        message_color = {
            ProcessingStatus.COMPLETE: Theme.SUCCESS,
            ProcessingStatus.ERROR: Theme.ERROR,
            ProcessingStatus.PAUSED: Theme.WARNING,
            ProcessingStatus.CANCELLED: Theme.WARNING,
            ProcessingStatus.LOADING: Theme.INFO,
            ProcessingStatus.DETECTING: Theme.INFO,
            ProcessingStatus.PROCESSING: Theme.INFO,
            ProcessingStatus.MERGING: Theme.WARNING,
        }.get(item.status, Theme.TEXT_SECONDARY)
        self.message_label.config(text=status_message, fg=message_color)
        can_open = item.status == ProcessingStatus.COMPLETE and Path(item.output_path).exists()
        self.open_btn.set_enabled(can_open)
        if can_open:
            if not self.open_btn.winfo_ismapped():
                self.open_btn.pack(side="right")
        elif self.open_btn.winfo_manager():
            self.open_btn.pack_forget()
        self.remove_btn.set_enabled(item.status not in (
            ProcessingStatus.LOADING,
            ProcessingStatus.DETECTING,
            ProcessingStatus.PROCESSING,
            ProcessingStatus.MERGING,
        ))

        # Elapsed time
        elapsed_text = ""
        if item.started_at:
            end = item.completed_at or datetime.now()
            elapsed = (end - item.started_at).total_seconds()
            elapsed_text = format_time(elapsed)
        pct_text = f"{int(item.progress * 100)}%" if item.progress > 0 else ""
        if pct_text and elapsed_text:
            meta_text = f"{pct_text} / {elapsed_text}"
        else:
            meta_text = pct_text or elapsed_text
        self.time_label.config(text=meta_text)

        # Active-state pulsing indicator (start/stop based on status)
        active = item.status in (ProcessingStatus.LOADING,
                                 ProcessingStatus.DETECTING,
                                 ProcessingStatus.PROCESSING,
                                 ProcessingStatus.MERGING)
        if active:
            self._start_pulse()
        else:
            self._stop_pulse()
        self._sync_a11y()

    # Pulse-state helpers -----------------------------------------------
    _pulse_id = None
    _pulse_phase = 0

    def _start_pulse(self):
        if getattr(self, "_pulse_id", None) is not None:
            return
        self._pulse_phase = 0
        self._pulse_tick()

    def _stop_pulse(self):
        tid = getattr(self, "_pulse_id", None)
        if tid:
            try:
                self.after_cancel(tid)
            except tk.TclError:
                pass
        self._pulse_id = None
        # Restore the normal border for the current selection state
        border = Theme.BLUE_PRIMARY if self.is_selected else Theme.BORDER
        self.config(highlightbackground=border)
        if self.is_selected:
            self.accent_stripe.config(bg=Theme.BLUE_PRIMARY)
        else:
            self.accent_stripe.config(bg=self._surface_bg)

    def _pulse_tick(self):
        # Alternate between a bright and a calm border / accent stripe
        try:
            bright = (self._pulse_phase % 2 == 0)
            border = Theme.GREEN_PRIMARY if bright else Theme.GREEN_HOVER
            stripe = Theme.GREEN_PRIMARY if bright else Theme.GREEN_HOVER
            self.config(highlightbackground=border)
            self.accent_stripe.config(bg=stripe)
            self._pulse_phase += 1
            self._pulse_id = self.after(720, self._pulse_tick)
        except tk.TclError:
            self._pulse_id = None


# =============================================================================
# LOG PANEL HANDLER -- routes log messages into a tk.Text widget
# =============================================================================

class TextWidgetHandler(logging.Handler):
    """Logging handler that writes to a tk.Text widget and tracks
    WARN/ERROR counts so the UI can show live badges."""

    def __init__(self, text_widget: tk.Text, on_count_change: Callable = None):
        super().__init__()
        self.text_widget = text_widget
        self.on_count_change = on_count_change
        self.warn_count = 0
        self.error_count = 0
        self._pending: "queue.Queue[Tuple[str, int]]" = queue.Queue()
        try:
            self.text_widget.after(100, self._drain_pending)
        except tk.TclError:
            pass

    def emit(self, record):
        msg = self.format(record) + '\n'
        self._pending.put((msg, record.levelno))

    def _drain_pending(self):
        try:
            if not int(self.text_widget.winfo_exists()):
                return
        except tk.TclError:
            return
        for _ in range(200):
            try:
                msg, levelno = self._pending.get_nowait()
            except queue.Empty:
                break
            self._append(msg, levelno)
        try:
            self.text_widget.after(100, self._drain_pending)
        except tk.TclError:
            pass

    def _append(self, msg, levelno):
        try:
            if not int(self.text_widget.winfo_exists()):
                return
        except tk.TclError:
            return
        self.text_widget.config(state="normal")
        tag = "info"
        if levelno >= logging.ERROR:
            tag = "error"
            self.error_count += 1
        elif levelno >= logging.WARNING:
            tag = "warning"
            self.warn_count += 1
        self.text_widget.insert("end", msg, tag)
        # Trim to 2000 lines to prevent unbounded memory growth
        line_count = int(self.text_widget.index("end-1c").split(".")[0])
        if line_count > 2000:
            self.text_widget.delete("1.0", f"{line_count - 2000}.0")
        self.text_widget.see("end")
        self.text_widget.config(state="disabled")
        if self.on_count_change:
            try:
                self.on_count_change(self.warn_count, self.error_count)
            except Exception:
                pass

    def reset_counts(self):
        self.warn_count = 0
        self.error_count = 0
        if self.on_count_change:
            try:
                self.on_count_change(0, 0)
            except Exception:
                pass

