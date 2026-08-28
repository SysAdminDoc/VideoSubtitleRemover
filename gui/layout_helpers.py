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
from backend.i18n import ntr, tr

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
            from gui.utils import ffmpeg_status_summary

            ffmpeg_summary = ffmpeg_status_summary(
                getattr(self, "ffmpeg_state", {}))
            audio_short = ffmpeg_summary["short"]
            # RM-324: a build below the enforced floor stops the run at the
            # security check, so the chip must not read as ready.
            ready = bool(detection) and ffmpeg_summary["safe"]
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

    def _card_header(self, parent, title: str, bg=Theme.BG_CARD,
                     pad_x: int = 12, pad_top: int = 10):
        """Card-internal section header with a single clear title."""
        tk.Label(parent, text=tr(title), font=f(Theme.F_TITLE, "bold"),
                 bg=bg, fg=Theme.TEXT_PRIMARY).pack(
                     anchor="w", padx=pad_x, pady=(pad_top, Theme.S_SM))

    def _divider(self, parent, pad: int = 0):
        line = tk.Frame(parent, bg=Theme.BORDER_SUBTLE, height=1)
        line.pack(fill="x", padx=pad, pady=0)
        return line

    def _update_output_label(self):
        """Refresh the output directory summary."""
        if self._output_dir:
            display = truncate_middle(str(self._output_dir), 54)
            self.output_dir_label.config(text=display, fg=Theme.TEXT_PRIMARY)
            self.output_dir_meta.config(text=tr("Custom location"))
            full_command_text = tr("Custom folder")
            command_text = (
                tr("Output")
                if self._text_scale_percent >= 200
                else full_command_text
            )
        else:
            self.output_dir_label.config(text=tr("Auto-create an output folder beside each source"),
                                         fg=Theme.TEXT_PRIMARY)
            self.output_dir_meta.config(text=tr("Default workflow"))
            full_command_text = tr("Same as source")
            command_text = (
                tr("Output")
                if self._text_scale_percent >= 200
                else full_command_text
            )
        if hasattr(self, "_command_output_btn"):
            self._command_output_btn.accessible_label = full_command_text
            self._command_output_btn.set_text(command_text)

    def _update_region_label_display(self):
        """Refresh the region summary line."""
        spans = getattr(self.config, "subtitle_region_spans", None) or []
        keyframe_tracks = (
            getattr(self.config, "subtitle_region_keyframes", None) or [])
        areas = getattr(self.config, "subtitle_areas", None) or []
        if keyframe_tracks:
            self.region_label.config(
                text=ntr("Moving manual regions: {count} track",
                         "Moving manual regions: {count} tracks",
                         len(keyframe_tracks)).format(
                    count=len(keyframe_tracks)),
                fg=Theme.TEXT_PRIMARY,
            )
            self.region_meta.config(text=tr("Interpolated keyframe masks"),
                                    fg=Theme.SUCCESS)
        elif spans:
            self.region_label.config(
                text=ntr("Timed manual regions: {count} rectangle",
                         "Timed manual regions: {count} rectangles",
                         len(spans)).format(count=len(spans)),
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
            if has_manual and self.config.sttn_skip_detection:
                from backend.config import static_region_degrades_to_cv2

                # RM-321: a static mask leaves the temporal engines nothing
                # to recover from, so the whole band goes through
                # cv2.inpaint. Green here told the user the opposite.
                if static_region_degrades_to_cv2(self.config):
                    self.region_meta.config(
                        text=tr(
                            "Manual-only mask; {mode} cannot use it and will "
                            "fall back to cv2"
                        ).format(mode=self.config.mode.value),
                        fg=Theme.WARNING,
                    )
                else:
                    self.region_meta.config(
                        text=tr("Manual-only mask; automatic detection is off"),
                        fg=Theme.SUCCESS,
                    )
            elif has_manual:
                self.region_meta.config(
                    text=tr("Saved region; automatic detection is on"),
                    fg=Theme.TEXT_SECONDARY,
                )
            self.region_reset_btn.set_enabled(has_manual and not self.is_processing)
        self._refresh_static_region_notice()

    def _refresh_static_region_notice(self):
        """Explain the cv2 fallback and offer the one-click way out.

        RM-321: the GUI used to report this state in success green with no
        hint that the selected engine would not run.
        """
        label = getattr(self, "static_region_notice", None)
        button = getattr(self, "static_region_switch_btn", None)
        if label is None:
            return
        from backend.config import static_region_degrades_to_cv2

        degrades = static_region_degrades_to_cv2(self.config)
        if not degrades:
            label.config(text="")
            label.pack_forget()
            if button is not None:
                button.pack_forget()
            return
        label.config(
            text=tr(
                "{mode} recovers pixels from other frames, and a fixed "
                "manual region looks the same in every frame, so the whole "
                "region will be filled by cv2 instead."
            ).format(mode=self.config.mode.value),
            fg=Theme.WARNING,
        )
        if not label.winfo_ismapped():
            label.pack(anchor="w", pady=(Theme.S_XS, 0))
        if button is not None and not button.winfo_ismapped():
            button.pack(anchor="w", pady=(Theme.S_XS, 0))

    def _switch_job_to_lama(self):
        """Move the job to the engine that does not need temporal exposure.

        Setting the variable is not enough. The combobox binding does not
        fire for a programmatic set, so the dependent controls and the
        algorithm description would keep describing the old engine; route
        through the same handler the picker uses.
        """
        from gui.config import InpaintMode as _GuiMode

        if hasattr(self, "mode_var"):
            self.mode_var.set(_GuiMode.LAMA.value)
        picker = getattr(self, "mode_picker", None)
        if picker is not None:
            picker.set(_GuiMode.LAMA.value)
        combo = getattr(self, "_command_mode_combo", None)
        if combo is not None:
            try:
                combo.set(_GuiMode.LAMA.value)
            except Exception:
                pass
        handler = getattr(self, "_on_mode_changed", None)
        if callable(handler):
            handler()
        else:  # pragma: no cover - the mixin is always present in the app
            self.config.mode = _GuiMode.LAMA
            self._update_region_label_display()
        self._update_status(
            tr("Switched to LaMa so the manual region is repaired properly."),
            "info",
        )
