"""Header/card/label layout helpers extracted from app.py."""

from __future__ import annotations

import logging

try:
    import tkinter as tk
except ImportError:
    pass

from gui.theme import (
    Theme, f,
)
from gui.utils import (
    truncate_middle,
)
from gui.widgets import (
    Tooltip,
)
from backend.i18n import tr

logger = logging.getLogger(__name__)


class LayoutHelpersMixin:
    """Header/card/label layout helpers extracted from app.py."""

    def _render_header_chips(self):
        """Render one readiness state and one quiet capability summary."""
        if not hasattr(self, "_header_chips"):
            return
        for child in self._header_chips.winfo_children():
            child.destroy()
        if self._hardware_probe_pending:
            state_text = tr("Checking")
            state_fg = Theme.INFO
            summary = tr("Detecting hardware and media support")
        else:
            gpu_short = (
                truncate_middle(self.gpus[0]["name"], 24)
                if self.gpus else tr("CPU mode")
            )
            detection = self.ai_engines.get("detection", [])
            det_short = detection[0] if detection else tr("OpenCV fallback")
            audio_short = tr("FFmpeg") if self.ffmpeg_ready else tr("No FFmpeg")
            ready = bool(detection) and self.ffmpeg_ready
            state_text = tr("Ready") if ready else tr("Limited")
            state_fg = Theme.SUCCESS if ready else Theme.WARNING
            summary = f"{gpu_short}  /  {det_short}  /  {audio_short}"

        ready_group = tk.Frame(self._header_chips, bg=Theme.BG_DARK)
        ready_dot = tk.Canvas(
            ready_group, width=10, height=10,
            bg=Theme.BG_DARK, highlightthickness=0,
        )
        ready_dot.create_oval(1, 1, 9, 9, fill=state_fg, outline="")
        ready_dot.pack(side="left", padx=(0, Theme.S_XS))
        self._header_ready_label = tk.Label(
            ready_group,
            text=state_text,
            font=f(Theme.F_META, "bold"),
            bg=Theme.BG_DARK,
            fg=state_fg,
        )
        self._header_ready_label.pack(side="left")
        ready_group.pack(side="left")
        # The capability summary (GPU / detector / audio) is surfaced as a
        # tooltip on the ready indicator rather than an inline label.
        Tooltip(self._header_ready_label, summary)

    def _section_title(self, parent, eyebrow: str, title: str, hint: str,
                       pad_x: int = 16, pad_top: int = 12):
        """Compact section header with optional supporting copy."""
        bg = parent.cget("bg")
        if eyebrow:
            tk.Label(parent, text=tr(eyebrow).upper(), font=f(Theme.F_EYEBROW, "bold"),
                     bg=bg, fg=Theme.TEXT_MUTED).pack(
                         anchor="w", padx=pad_x, pady=(pad_top, 0))
        tk.Label(parent, text=tr(title), font=f(Theme.F_HEADING, "bold"),
                 bg=bg, fg=Theme.TEXT_PRIMARY).pack(
                     anchor="w", padx=pad_x,
                     pady=(2 if eyebrow else pad_top, 0))
        if hint:
            tk.Label(parent, text=tr(hint), font=f(Theme.F_BODY_SM),
                     bg=bg, fg=Theme.TEXT_MUTED, wraplength=560,
                     justify="left").pack(anchor="w", padx=pad_x, pady=(4, Theme.S_MD))

    def _create_card(self, parent, bg=Theme.BG_CARD) -> tk.Frame:
        """Create a borderless tonal group."""
        return tk.Frame(parent, bg=bg, highlightthickness=0)

    def _card_header(self, parent, eyebrow: str, title: str, bg=Theme.BG_CARD,
                     pad_x: int = 12, pad_top: int = 10):
        """Card-internal section header with a single clear title."""
        tk.Label(parent, text=tr(title), font=f(Theme.F_TITLE, "bold"),
                 bg=bg, fg=Theme.TEXT_PRIMARY).pack(
                     anchor="w", padx=pad_x, pady=(pad_top, Theme.S_SM))

    def _divider(self, parent, pad: int = 0):
        tk.Frame(parent, bg=Theme.BORDER_SUBTLE, height=1).pack(
            fill="x", padx=pad, pady=0)

    def _update_output_label(self):
        """Refresh the output directory summary."""
        if self._output_dir:
            display = truncate_middle(str(self._output_dir), 54)
            self.output_dir_label.config(text=display, fg=Theme.TEXT_PRIMARY)
            self.output_dir_meta.config(text=tr("Custom location"))
            command_text = tr("Custom folder")
        else:
            self.output_dir_label.config(text=tr("Auto-create an output folder beside each source"),
                                         fg=Theme.TEXT_PRIMARY)
            self.output_dir_meta.config(text=tr("Default workflow"))
            command_text = tr("Same as source")
        if hasattr(self, "_command_output_btn"):
            self._command_output_btn.set_text(command_text)

    def _update_region_label_display(self):
        """Refresh the region summary line."""
        spans = getattr(self.config, "subtitle_region_spans", None) or []
        keyframe_tracks = (
            getattr(self.config, "subtitle_region_keyframes", None) or [])
        areas = getattr(self.config, "subtitle_areas", None) or []
        if keyframe_tracks:
            self.region_label.config(
                text=tr("Moving manual regions: {count} track{suffix}").format(
                    count=len(keyframe_tracks),
                    suffix="s" if len(keyframe_tracks) != 1 else ""),
                fg=Theme.TEXT_PRIMARY,
            )
            self.region_meta.config(text=tr("Interpolated keyframe masks"),
                                    fg=Theme.SUCCESS)
        elif spans:
            self.region_label.config(
                text=tr("Timed manual regions: {count} rectangle{suffix}").format(
                    count=len(spans), suffix="s" if len(spans) != 1 else ""),
                fg=Theme.TEXT_PRIMARY,
            )
            self.region_meta.config(text=tr("Time-ranged mask regions"),
                                    fg=Theme.SUCCESS)
        elif len(areas) > 1:
            self.region_label.config(
                text=tr("Manual regions: {count} fixed rectangles").format(count=len(areas)),
                fg=Theme.TEXT_PRIMARY,
            )
            self.region_meta.config(text=tr("Fixed mask regions"), fg=Theme.SUCCESS)
        elif self.config.subtitle_area:
            x1, y1, x2, y2 = self.config.subtitle_area
            self.region_label.config(
                text=tr("Manual region: ({x1}, {y1}) to ({x2}, {y2})").format(
                    x1=x1, y1=y1, x2=x2, y2=y2),
                fg=Theme.TEXT_PRIMARY,
            )
            self.region_meta.config(text=tr("Fixed mask region"), fg=Theme.SUCCESS)
        else:
            self.region_label.config(text=tr("Automatic subtitle detection"), fg=Theme.TEXT_PRIMARY)
            self.region_meta.config(text=tr("Recommended default"), fg=Theme.TEXT_MUTED)
        if hasattr(self, "region_reset_btn"):
            has_manual = (
                bool(spans) or bool(keyframe_tracks) or bool(areas)
                or self.config.subtitle_area is not None
            )
            self.region_reset_btn.set_enabled(has_manual and not self.is_processing)
