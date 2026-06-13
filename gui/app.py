"""Main application class extracted from the monolith (RM-114)."""

from __future__ import annotations

import json
import logging
import math
import os
import subprocess
import sys
import tempfile
import threading
import time
import traceback
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

try:
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox
    from tkinter import font as tkfont
except ImportError:
    pass

try:
    from PIL import Image, ImageTk, ImageDraw, ImageFilter
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

from gui.theme import (
    Theme, apply_high_contrast_theme, apply_default_theme, f, mono,
)
from gui.config import (
    APP_NAME, APP_VERSION, APP_AUTHOR,
    LOG_DIR, LOG_FILE, SETTINGS_FILE, VSR_SETTINGS_FORMAT,
    InpaintMode, ProcessingStatus, STATUS_UI,
    ProcessingConfig, QueueItem,
    _coerce_bool, _coerce_int, _coerce_float, _coerce_text,
    _coerce_rect, _coerce_rect_list, _coerce_gui_mode,
    _read_json_object, _write_json_atomic,
    _migrate_settings, load_settings, save_settings,
    PRESETS_FILE, list_presets, apply_preset,
    save_user_preset, delete_user_preset, export_preset, import_preset,
    status_ui,
)
from gui.utils import (
    SUPPORTED_EXTENSIONS, filepicker_pattern,
    get_app_dir, detect_gpu, format_time, format_size,
    is_video_file, is_image_file,
    _CURATED_LANG_NAMES, _engine_supported_languages, _build_language_list,
    detect_ai_engines, detect_ffmpeg, get_file_info,
    _soft_subtitle_stream_record, _format_soft_subtitle_summary,
    _queue_item_info_text, truncate_middle,
    format_quality_report, summarize_quality_reports,
)
from gui.widgets import (
    _get_dpi_scale, _scaled,
    Tooltip, ModernButton, ModernProgressBar, ModernToggle,
    ModernSlider, show_confirm, TaskbarProgress, make_themed_menu,
    Toast, SegmentedPicker, DragDropFrame, QueueItemWidget,
    TextWidgetHandler,
)

logger = logging.getLogger(__name__)


class VideoSubtitleRemoverApp:
    """Main application class."""

    def __init__(self):
        self.root = tk.Tk()
        self.root.title(f"{APP_NAME} v{APP_VERSION}")
        self.root.geometry("1240x860")
        self.root.minsize(980, 720)
        self.root.configure(bg=Theme.BG_DARK)

        # Set window icon
        try:
            icon_candidates = [
                get_app_dir() / "assets" / "icon.ico",
                get_app_dir() / "icon.ico",
                get_app_dir() / "favicon.ico",
            ]
            for icon_path in icon_candidates:
                if icon_path.exists():
                    self.root.iconbitmap(icon_path)
                    break
        except Exception:
            pass
        if PIL_AVAILABLE:
            try:
                for icon_path in (get_app_dir() / "icon.png", get_app_dir() / "banner.png"):
                    if icon_path.exists():
                        icon_img = Image.open(icon_path)
                        if icon_img.width > 128:
                            icon_img.thumbnail((128, 128), Image.LANCZOS)
                        self._app_icon_photo = ImageTk.PhotoImage(icon_img)
                        self.root.iconphoto(True, self._app_icon_photo)
                        break
            except Exception:
                pass

        # State
        self.config = load_settings()
        # RM-96: high-contrast theme applies BEFORE widget construction so
        # every Canvas / ttk style reads the swapped palette on first draw.
        if getattr(self.config, "high_contrast", False):
            apply_high_contrast_theme()
        # RM-98: RTL layout mirror -- set the Tk option DB before widgets
        # build so labels right-align and `pack(side="right")` becomes
        # the dominant orientation. Full pack-side flipping for every
        # widget is a much larger pass; this lands the framework.
        self._rtl_layout = bool(getattr(self.config, "rtl_layout", False))
        # RM-97: bind a gettext catalog if one matches the user's locale.
        # No-op when no `.mo` file ships -- every UI string falls back to
        # the literal english form.
        try:
            import locale as _locale
            _user_locale = (_locale.getlocale()[0] or "").split("_")[0]
            from backend.i18n import bind_locale as _bind_locale
            if _user_locale and _user_locale != "en":
                _bind_locale(_user_locale)
        except Exception:
            pass
        self.queue: List[QueueItem] = []
        self.queue_widgets: dict = {}
        self.is_processing = False
        self._stop_requested = False
        self._processing_thread: Optional[threading.Thread] = None
        self.cancel_event = threading.Event()
        self.queue_lock = threading.Lock()
        self.gpus = detect_gpu()
        self.ai_engines = detect_ai_engines()
        self.ffmpeg_ready = detect_ffmpeg()
        self._elapsed_timer_id = None
        self._output_dir: Optional[Path] = None  # None = use input_dir/output/
        self._preview_detector = None  # cached SubtitleDetector for mask preview
        self._preview_detector_lang = None  # lang the cached detector was created with
        self._cached_remover = None  # cached BackendRemover for batch reuse
        self._cached_remover_key = None  # (mode, device, lang) key for cache invalidation
        self._selected_queue_item_id: Optional[str] = None
        self._brand_photo = None
        self._status_tone = "neutral"
        self._shutdown_started = False
        self._taskbar = None  # created after the root is fully realized
        self._batch_times: List[float] = []  # seconds per item for ETA
        self._batch_started_at: Optional[datetime] = None
        self._batch_report_records: dict = {}
        self._preview_request_id = 0
        self._throbber_id = None
        self._throbber_phase = 0
        self._layout_mode = "wide"
        self._workflow_pills = []

        # Variables
        self.mode_var = tk.StringVar(value=self.config.mode.value)
        self.gpu_var = tk.StringVar()
        self.skip_detection_var = tk.BooleanVar(value=self.config.sttn_skip_detection)
        self.lama_fast_var = tk.BooleanVar(value=self.config.lama_super_fast)
        self.preserve_audio_var = tk.BooleanVar(value=self.config.preserve_audio)
        self.lang_var = tk.StringVar(value=self.config.detection_lang)

        # Build UI
        self._setup_styles()
        self._build_ui()
        self._bind_shortcuts()
        self.root.bind("<Configure>", self._on_root_configure, add="+")

        # GPU setup -- restore saved selection or default to first
        if self.gpus:
            matched = False
            for g in self.gpus:
                if g['index'] == self.config.gpu_id:
                    self.gpu_var.set(f"{g['name']} ({g['memory']})")
                    matched = True
                    break
            if not matched:
                self.gpu_var.set(f"{self.gpus[0]['name']} ({self.gpus[0]['memory']})")
        else:
            self.gpu_var.set("CPU Mode")
            self.config.use_gpu = False

        # Attach log panel handler (tracks warn/error counts for badges)
        self._log_handler = TextWidgetHandler(self.log_text,
                                              on_count_change=self._update_log_badges)
        self._log_handler.setFormatter(logging.Formatter(
            '%(asctime)s [%(levelname)s] %(message)s', datefmt='%H:%M:%S'))
        logging.getLogger().addHandler(self._log_handler)

        self._update_output_label()
        self._update_region_label_display()
        self._refresh_action_states()
        self.root.after(0, lambda: self._apply_responsive_layout(self.root.winfo_width()))

        # Restore persisted panel visibility (defaults: advanced closed, log open)
        try:
            if self.config.adv_panel_open and not self.adv_visible:
                self._toggle_advanced()
            if not self.config.log_panel_open and self._log_visible:
                self._toggle_log_panel()
        except Exception:
            pass

        # Save settings on close
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        # First-run welcome overlay (only shown once, then persisted)
        self._maybe_show_onboarding()

    def _on_close(self):
        """Stop processing, save settings, and close."""
        if self._shutdown_started:
            return
        # Guard against re-entry while the confirmation dialog is open.
        # Without this, a second WM_DELETE_WINDOW could fire while the modal
        # is shown (e.g. from an external window manager).
        if getattr(self, '_close_dialog_open', False):
            return
        self._close_dialog_open = True
        try:
            active_thread = self._has_active_processing_thread()
            if self.is_processing or active_thread:
                n = sum(1 for it in self.queue
                        if it.status in (ProcessingStatus.LOADING,
                                         ProcessingStatus.DETECTING,
                                         ProcessingStatus.PROCESSING,
                                         ProcessingStatus.MERGING))
                label = f"{n} active item{'s' if n != 1 else ''} will be cancelled."
                if not show_confirm(
                    self.root,
                    title="Close while processing?",
                    message="A batch is still running.",
                    detail=label + " Completed outputs on disk are kept.",
                    confirm_label="Close anyway",
                    cancel_label="Keep working",
                    tone="danger",
                ):
                    return
                self.cancel_event.set()
                self._stop_elapsed_timer()
                self._stop_requested = True
                self._update_status(
                    "Closing after the current step stops safely...",
                    "warning",
                )
                if self._taskbar:
                    self._taskbar.set_state(TaskbarProgress.STATE_PAUSED)
        finally:
            self._close_dialog_open = False
        # Set the flag AFTER confirmation so that _on_processing_complete
        # callbacks scheduled before the dialog opened don't race-destroy root.
        self._shutdown_started = True
        self._sync_config_from_ui()
        # Persist window layout and panel states for next launch
        try:
            self.config.window_geometry = self.root.geometry()
            self.config.adv_panel_open = self.adv_visible
            self.config.log_panel_open = self._log_visible
        except Exception:
            pass
        save_settings(self.config)
        self._finish_close_when_safe(time.monotonic() + 2.0)

    def _finish_close_when_safe(self, deadline: float):
        """Wait briefly for active work to notice cancellation before exit."""
        if not self._has_active_processing_thread() or time.monotonic() >= deadline:
            try:
                self.root.destroy()
            except Exception:
                pass
            return
        try:
            self.root.after(100, lambda: self._finish_close_when_safe(deadline))
        except Exception:
            try:
                self.root.destroy()
            except Exception:
                pass

    def _sync_config_from_ui(self):
        """Sync config object from current UI state."""
        try:
            self.config.mode = InpaintMode(self.mode_var.get())
        except ValueError:
            pass
        self.config.sttn_skip_detection = self.skip_detection_var.get()
        self.config.lama_super_fast = self.lama_fast_var.get()
        self.config.preserve_audio = self.preserve_audio_var.get()
        self.config.detection_lang = self.lang_var.get()
        # Threshold slider stores as int percent, convert to float
        pct = getattr(self.config, '_detection_threshold_pct', 50)
        self.config.detection_threshold = pct / 100.0
        # Time range
        self.config.time_start = self._safe_float(self.time_start_entry.get())
        self.config.time_end = self._safe_float(self.time_end_entry.get())
        # HW encode
        self.config.use_hw_encode = self.hw_encode_var.get()
        # v3.9 quality + workflow toggles
        if hasattr(self, 'auto_band_var'):
            self.config.auto_band = self.auto_band_var.get()
        if hasattr(self, 'flow_warp_var'):
            self.config.tbe_flow_warp = self.flow_warp_var.get()
        if hasattr(self, 'scene_split_var'):
            self.config.tbe_scene_cut_split = self.scene_split_var.get()
        if hasattr(self, 'adaptive_batch_var'):
            self.config.adaptive_batch = self.adaptive_batch_var.get()
        if hasattr(self, 'export_srt_var'):
            self.config.export_srt = self.export_srt_var.get()
        if hasattr(self, 'export_mask_var'):
            self.config.export_mask_video = self.export_mask_var.get()
        if hasattr(self, 'kalman_var'):
            self.config.kalman_tracking = self.kalman_var.get()
        if hasattr(self, 'phash_var'):
            self.config.phash_skip_enable = self.phash_var.get()
        if hasattr(self, 'colour_tune_var'):
            self.config.colour_tune_enable = self.colour_tune_var.get()
        if hasattr(self, 'deinterlace_var'):
            self.config.deinterlace_auto = self.deinterlace_var.get()
        if hasattr(self, 'keyframe_var'):
            self.config.keyframe_detection = self.keyframe_var.get()
        if hasattr(self, 'quality_report_var'):
            self.config.quality_report = self.quality_report_var.get()
        # v3.13 GUI-exposed toggles
        if hasattr(self, 'quality_sheet_var'):
            self.config.quality_report_sheet = self.quality_sheet_var.get()
        if hasattr(self, 'multi_audio_var'):
            self.config.multi_audio_passthrough = self.multi_audio_var.get()
        if hasattr(self, 'loudnorm_var'):
            self.config.loudnorm_target = self._safe_float(self.loudnorm_var.get(), 0.0)
        if hasattr(self, 'decode_accel_var'):
            self.config.decode_hw_accel = self.decode_accel_var.get()
        if hasattr(self, 'prefetch_var'):
            self.config.prefetch_decode = self.prefetch_var.get()
        if hasattr(self, 'remove_subs_var'):
            self.config.remove_subtitles = self.remove_subs_var.get()
        if hasattr(self, 'remove_chyrons_var'):
            self.config.remove_chyrons = self.remove_chyrons_var.get()
        if hasattr(self, 'karaoke_grouping_var'):
            self.config.karaoke_grouping = self.karaoke_grouping_var.get()
        if hasattr(self, 'output_codec_var'):
            self.config.output_codec = self.output_codec_var.get()
        if hasattr(self, 'vertical_text_var'):
            self.config.detection_vertical = self.vertical_text_var.get()
        if hasattr(self, 'high_contrast_var'):
            self.config.high_contrast = self.high_contrast_var.get()
        if hasattr(self, 'update_check_var'):
            self.config.update_check = self.update_check_var.get()
        if hasattr(self, 'json_log_var'):
            self.config.json_log_enabled = self.json_log_var.get()
        # GPU sync
        selection = self.gpu_var.get()
        for gpu in self.gpus:
            if f"{gpu['name']} ({gpu['memory']})" == selection:
                self.config.gpu_id = gpu['index']
                break

    def _make_processing_snapshot(self) -> ProcessingConfig:
        """Build a fresh processing snapshot from the current UI state."""
        self._sync_config_from_ui()
        return ProcessingConfig.from_dict(self.config.to_dict())

    def _apply_current_settings_to_idle_items(self) -> int:
        """Refresh all not-yet-running queue items from the current UI state."""
        snapshot = self._make_processing_snapshot()
        updated = 0
        with self.queue_lock:
            for item in self.queue:
                if item.status == ProcessingStatus.IDLE:
                    item.config = ProcessingConfig.from_dict(snapshot.to_dict())
                    updated += 1
        output_updates = self._refresh_idle_output_paths()
        if output_updates:
            self._update_queue_display()
        return updated

    def _setup_styles(self):
        """Configure ttk styles for a cohesive dark theme."""
        style = ttk.Style()
        style.theme_use('clam')

        # ---- Combobox ---------------------------------------------------
        style.configure("Dark.TCombobox",
                       fieldbackground=Theme.BG_TERTIARY,
                       background=Theme.BG_TERTIARY,
                       foreground=Theme.TEXT_PRIMARY,
                       arrowcolor=Theme.TEXT_SECONDARY,
                       bordercolor=Theme.BORDER,
                       darkcolor=Theme.BG_TERTIARY,
                       lightcolor=Theme.BG_TERTIARY,
                       insertcolor=Theme.TEXT_PRIMARY,
                       padding=(10, 6))

        style.map("Dark.TCombobox",
                 fieldbackground=[('readonly', Theme.BG_TERTIARY),
                                  ('disabled', Theme.BG_CARD)],
                 background=[('active', Theme.BG_RAISED)],
                 foreground=[('disabled', Theme.TEXT_DISABLED)],
                 arrowcolor=[('active', Theme.TEXT_PRIMARY),
                             ('disabled', Theme.TEXT_DISABLED)],
                 bordercolor=[('focus', Theme.BORDER_FOCUS),
                              ('hover', Theme.BORDER_STRONG)],
                 selectbackground=[('readonly', Theme.BLUE_MUTED)],
                 selectforeground=[('readonly', Theme.TEXT_PRIMARY)])

        # Theme the combobox dropdown popup listbox
        self.root.option_add('*TCombobox*Listbox.background', Theme.BG_RAISED)
        self.root.option_add('*TCombobox*Listbox.foreground', Theme.TEXT_PRIMARY)
        self.root.option_add('*TCombobox*Listbox.selectBackground', Theme.BLUE_MUTED)
        self.root.option_add('*TCombobox*Listbox.selectForeground', Theme.TEXT_PRIMARY)
        self.root.option_add('*TCombobox*Listbox.borderWidth', 0)
        self.root.option_add('*TCombobox*Listbox.font', f(Theme.F_BODY_SM))

        # ---- Scrollbar (slimmer, quieter) -------------------------------
        style.configure("Dark.Vertical.TScrollbar",
                        background=Theme.BORDER,
                        troughcolor=Theme.BG_SECONDARY,
                        bordercolor=Theme.BG_SECONDARY,
                        arrowcolor=Theme.TEXT_MUTED,
                        gripcount=0,
                        width=10)
        style.map("Dark.Vertical.TScrollbar",
                 background=[('active', Theme.BORDER_STRONG),
                             ('pressed', Theme.BORDER_STRONG)],
                 arrowcolor=[('active', Theme.TEXT_SECONDARY)])

    def _create_surface(self, parent, bg: str = Theme.BG_SECONDARY) -> tk.Frame:
        """Create a bordered surface panel."""
        return tk.Frame(parent, bg=bg, highlightthickness=1,
                        highlightbackground=Theme.BORDER_SUBTLE)

    def _create_chip(self, parent, label: str, value: str, fg: str, bg: str) -> tk.Frame:
        """Minimal status chip with a single clear line of text."""
        chip = tk.Frame(parent, bg=bg, highlightthickness=1,
                        highlightbackground=Theme.BORDER_SUBTLE)
        tk.Label(
            chip,
            text=f"{label}: {value}",
            font=f(Theme.F_META, "bold"),
            bg=bg,
            fg=fg,
            padx=12,
            pady=7,
        ).pack(anchor="w")
        return chip

    def _section_title(self, parent, eyebrow: str, title: str, hint: str,
                       pad_x: int = 20, pad_top: int = 16):
        """Consistent section header: eyebrow label + title + hint line."""
        bg = parent.cget("bg")
        if eyebrow:
            tk.Label(parent, text=eyebrow.upper(), font=f(Theme.F_EYEBROW, "bold"),
                     bg=bg, fg=Theme.TEXT_MUTED).pack(
                         anchor="w", padx=pad_x, pady=(pad_top, 0))
        tk.Label(parent, text=title, font=f(Theme.F_HEADING, "bold"),
                 bg=bg, fg=Theme.TEXT_PRIMARY).pack(
                     anchor="w", padx=pad_x,
                     pady=(2 if eyebrow else pad_top, 0))
        if hint:
            tk.Label(parent, text=hint, font=f(Theme.F_BODY_SM),
                     bg=bg, fg=Theme.TEXT_MUTED, wraplength=560,
                     justify="left").pack(anchor="w", padx=pad_x, pady=(4, Theme.S_MD))

    def _create_card(self, parent, bg=Theme.BG_CARD) -> tk.Frame:
        """Bordered card container with consistent style."""
        return tk.Frame(parent, bg=bg, highlightthickness=1,
                        highlightbackground=Theme.BORDER_SUBTLE)

    def _card_header(self, parent, eyebrow: str, title: str, bg=Theme.BG_CARD,
                     pad_x: int = 16, pad_top: int = 14):
        """Card-internal section header with a single clear title."""
        tk.Label(parent, text=title, font=f(Theme.F_TITLE, "bold"),
                 bg=bg, fg=Theme.TEXT_PRIMARY).pack(anchor="w", padx=pad_x, pady=(pad_top, 10))

    def _divider(self, parent, pad: int = 0):
        tk.Frame(parent, bg=Theme.BORDER_SUBTLE, height=1).pack(
            fill="x", padx=pad, pady=0)

    def _update_output_label(self):
        """Refresh the output directory summary."""
        if self._output_dir:
            display = truncate_middle(str(self._output_dir), 54)
            self.output_dir_label.config(text=display, fg=Theme.TEXT_PRIMARY)
            self.output_dir_meta.config(text="Custom location")
        else:
            self.output_dir_label.config(text="Auto-create an output folder beside each source",
                                         fg=Theme.TEXT_PRIMARY)
            self.output_dir_meta.config(text="Default workflow")

    def _update_region_label_display(self):
        """Refresh the region summary line."""
        areas = getattr(self.config, "subtitle_areas", None) or []
        if len(areas) > 1:
            self.region_label.config(
                text=f"Manual regions: {len(areas)} fixed rectangles",
                fg=Theme.TEXT_PRIMARY,
            )
            self.region_meta.config(text="Fixed mask regions", fg=Theme.SUCCESS)
        elif self.config.subtitle_area:
            x1, y1, x2, y2 = self.config.subtitle_area
            self.region_label.config(
                text=f"Manual region: ({x1}, {y1}) to ({x2}, {y2})",
                fg=Theme.TEXT_PRIMARY,
            )
            self.region_meta.config(text="Fixed mask region", fg=Theme.SUCCESS)
        else:
            self.region_label.config(text="Automatic subtitle detection", fg=Theme.TEXT_PRIMARY)
            self.region_meta.config(text="Recommended default", fg=Theme.TEXT_MUTED)
        if hasattr(self, "region_reset_btn"):
            has_manual = bool(areas) or self.config.subtitle_area is not None
            self.region_reset_btn.set_enabled(has_manual and not self.is_processing)

    def _start_throbber(self):
        """Animate the preview area with a shimmer placeholder and moving dots
        to signal a background task in progress."""
        self._stop_throbber()
        self._throbber_phase = 0
        self._throbber_tick()

    def _stop_throbber(self):
        tid = getattr(self, "_throbber_id", None)
        if tid:
            try:
                self.root.after_cancel(tid)
            except Exception:
                pass
            self._throbber_id = None

    def _throbber_tick(self):
        if not PIL_AVAILABLE:
            self._preview_label.config(
                text="Detecting" + "." * (self._throbber_phase % 4))
            try:
                self._throbber_id = self.root.after(240, self._throbber_tick)
                self._throbber_phase += 1
            except tk.TclError:
                pass
            return
        try:
            w = max(220, self._preview_frame.winfo_width() - 36)
            h = 158
            base = Image.new("RGB", (w, h), self._hex_to_rgb(Theme.BG_TERTIARY))
            d = ImageDraw.Draw(base)
            d.rectangle([(0, 0), (w - 1, h - 1)],
                        outline=self._hex_to_rgb(Theme.BORDER), width=1)
            # Three animated dots pulsing left-to-right
            cx, cy = w // 2, h // 2
            phase = self._throbber_phase % 3
            for i in range(3):
                active = (i == phase)
                color = (Theme.BLUE_PRIMARY if active else Theme.BORDER)
                r = 6 if active else 4
                x = cx - 18 + i * 18
                d.ellipse([(x - r, cy - r), (x + r, cy + r)],
                          fill=self._hex_to_rgb(color))
            d.text((cx - 42, cy + 22), "DETECTING",
                   fill=self._hex_to_rgb(Theme.TEXT_MUTED))
            self._preview_photo = ImageTk.PhotoImage(base)
            self._preview_label.config(image=self._preview_photo, text="")
            self._throbber_phase += 1
            try:
                self._throbber_id = self.root.after(240, self._throbber_tick)
            except tk.TclError:
                pass
        except Exception:
            # Render failures shouldn't block detection
            pass

    def _push_live_preview(self, pil_img, cur_idx: int, total: int, file_name: str):
        """Render an inpainted frame into the preview pane during processing.
        Called on the Tk main thread via `root.after` from the worker thread."""
        try:
            self._stop_throbber()
            # Throttle: coalesce to at most ~15 FPS of UI updates
            now = time.monotonic()
            last = getattr(self, "_live_preview_last_ts", 0.0)
            if (now - last) < (1.0 / 15.0):
                return
            self._live_preview_last_ts = now
            if PIL_AVAILABLE:
                self._preview_photo = ImageTk.PhotoImage(pil_img)
                self._preview_label.config(image=self._preview_photo, text="")
            if total:
                pct = int(cur_idx / max(1, total) * 100)
                self.preview_title_label.config(text=f"Live preview: {file_name}")
                self.preview_meta_label.config(
                    text=f"Frame {cur_idx}/{total} ({pct}%)")
        except Exception:
            pass

    def _set_preview_placeholder(self, title: str, body: str):
        """Show the empty-state preview guidance with a subtle illustration."""
        self._stop_throbber()
        self.preview_title_label.config(text=title)
        self.preview_meta_label.config(text=body)
        # Render a minimalist placeholder card via PIL (if available) so the
        # preview never collapses to empty space.
        if PIL_AVAILABLE:
            try:
                w, h = 420, 128
                base = Image.new("RGB", (w, h), self._hex_to_rgb(Theme.BG_TERTIARY))
                draw = ImageDraw.Draw(base)
                # Outer border
                draw.rectangle([(0, 0), (w - 1, h - 1)],
                               outline=self._hex_to_rgb(Theme.BORDER_SUBTLE), width=1)
                # Faux film-strip glyph (three tall rects in the center)
                cx, cy = w // 2, h // 2
                for dx in (-44, 0, 44):
                    draw.rectangle(
                        [(cx + dx - 10, cy - 22), (cx + dx + 10, cy + 22)],
                        outline=self._hex_to_rgb(Theme.BORDER),
                        fill=self._hex_to_rgb(Theme.BG_CARD_HOVER),
                    )
                # Underline
                draw.line([(cx - 70, cy + 32), (cx + 70, cy + 32)],
                          fill=self._hex_to_rgb(Theme.BORDER_SUBTLE), width=1)
                self._preview_photo = ImageTk.PhotoImage(base)
                self._preview_label.config(image=self._preview_photo, text="")
            except Exception:
                self._preview_label.config(text="", image="")
                self._preview_photo = None
        else:
            self._preview_label.config(text="", image="")
            self._preview_photo = None

    @staticmethod
    def _hex_to_rgb(hex_str: str):
        hex_str = hex_str.lstrip('#')
        return tuple(int(hex_str[i:i + 2], 16) for i in (0, 2, 4))

    def _set_selected_queue_item(self, item_id: Optional[str]):
        """Update queue item selection state."""
        self._selected_queue_item_id = item_id
        for wid, widget in self.queue_widgets.items():
            widget.set_selected(wid == item_id)
        self._update_preview_actions()

    def _refresh_action_states(self):
        """Enable or disable primary queue actions based on current state."""
        has_queue = bool(self.queue)
        has_complete = any(item.status == ProcessingStatus.COMPLETE for item in self.queue)
        has_retry = any(item.status in (ProcessingStatus.ERROR, ProcessingStatus.CANCELLED)
                        for item in self.queue)
        active_thread = self._has_active_processing_thread()
        batch_busy = self.is_processing or active_thread

        if hasattr(self, "start_btn"):
            can_stop = active_thread and not self._stop_requested
            can_start = (not batch_busy) and has_queue
            self.start_btn.set_enabled(can_stop or can_start)
        if hasattr(self, "open_output_btn"):
            self.open_output_btn.set_enabled(has_complete)
        if hasattr(self, "retry_btn"):
            self.retry_btn.set_enabled((not batch_busy) and has_retry)
        if hasattr(self, "clear_btn"):
            self.clear_btn.set_enabled((not batch_busy) and has_queue)
        if hasattr(self, "repeat_btn"):
            self.repeat_btn.set_enabled(
                (not batch_busy) and self._last_completed_config() is not None)
        if hasattr(self, "batch_label") and not batch_busy:
            pending = sum(1 for item in self.queue if item.status == ProcessingStatus.IDLE)
            if pending:
                self.batch_label.config(
                    text=f"{pending} queued and ready to process",
                    fg=Theme.TEXT_SECONDARY,
                )
            elif has_complete:
                self.batch_label.config(
                    text="Outputs are ready for review",
                    fg=Theme.SUCCESS,
                )
            elif has_retry:
                self.batch_label.config(
                    text="Some items need attention",
                    fg=Theme.WARNING,
                )
            else:
                self.batch_label.config(text="Ready", fg=Theme.TEXT_MUTED)
        self._update_preview_actions()
        self._update_guidance_surface()

    def _bind_shortcuts(self):
        """No app-wide keyboard accelerators -- all actions are click-only."""
        pass

    def _open_file_picker(self):
        if hasattr(self, "drop_area"):
            self.drop_area._open_file_dialog()

    def _focus_queue_filter(self, event=None):
        if len(self.queue) < 6 or not hasattr(self, "_queue_filter_entry"):
            return "break"
        try:
            self._queue_filter_frame.pack(
                fill="x", padx=Theme.S_XL, pady=(0, Theme.S_SM),
                before=self._queue_container)
            self._queue_filter_entry.focus_set()
            self._queue_filter_entry.selection_range(0, "end")
        except tk.TclError:
            pass
        return "break"

    def _on_root_configure(self, event):
        """Keep layout responsive as the window width changes."""
        if event.widget is not self.root:
            return
        self._apply_responsive_layout(event.width)

    def _apply_responsive_layout(self, width: int):
        """Stack columns and footer/help clusters on narrower windows."""
        if not hasattr(self, "_content"):
            return

        mode = "stacked" if width < 1180 else "wide"
        if mode == self._layout_mode:
            if hasattr(self, "preview_meta_label"):
                self.preview_meta_label.config(wraplength=520 if mode == "stacked" else 360)
            if hasattr(self, "header_guidance_body"):
                self.header_guidance_body.config(wraplength=520 if mode == "stacked" else 300)
            if hasattr(self, "status_hint"):
                self.status_hint.config(wraplength=520 if mode == "stacked" else 360)
            return

        self._layout_mode = mode
        stacked = (mode == "stacked")

        self._left_col.grid_forget()
        self._right_col.grid_forget()

        if stacked:
            self._content.columnconfigure(0, weight=1, minsize=0)
            self._content.columnconfigure(1, weight=0, minsize=0)
            self._content.rowconfigure(0, weight=0)
            self._content.rowconfigure(1, weight=1)
            self._left_col.grid(row=0, column=0, sticky="nsew", padx=0, pady=(0, Theme.S_MD))
            self._right_col.grid(row=1, column=0, sticky="nsew", padx=0, pady=0)

            self._header_right.pack_forget()
            self._header_right.pack(fill="x", pady=(Theme.S_LG, 0))
            self._header_chips.pack_forget()
            self._header_chips.pack(anchor="w")
            self._header_help_btn.pack_forget()
            self._header_help_btn.pack(anchor="w", pady=(Theme.S_SM, 0))
            self._header_guidance_panel.pack_forget()
            self._header_guidance_panel.pack(fill="x", pady=(Theme.S_SM, 0))

            self._footer_left.pack_forget()
            self._footer_left.pack(anchor="w")
            self.status_hint.pack_forget()
            self.status_hint.pack(fill="x", pady=(Theme.S_XS, 0))
        else:
            self._content.columnconfigure(0, weight=57, minsize=440)
            self._content.columnconfigure(1, weight=43, minsize=360)
            self._content.rowconfigure(0, weight=1)
            self._content.rowconfigure(1, weight=0)
            self._left_col.grid(row=0, column=0, sticky="nsew", padx=(0, Theme.S_MD))
            self._right_col.grid(row=0, column=1, sticky="nsew", padx=(Theme.S_MD, 0))

            self._header_right.pack_forget()
            self._header_right.pack(side="right", anchor="n")
            self._header_chips.pack_forget()
            self._header_chips.pack(anchor="e")
            self._header_help_btn.pack_forget()
            self._header_help_btn.pack(anchor="e", pady=(Theme.S_SM, 0))
            self._header_guidance_panel.pack_forget()
            self._header_guidance_panel.pack(anchor="e", fill="x", pady=(Theme.S_SM, 0))

            self._footer_left.pack_forget()
            self._footer_left.pack(side="left")
            self.status_hint.pack_forget()
            self.status_hint.pack(side="right")

        self.preview_meta_label.config(wraplength=520 if stacked else 360)
        self.header_guidance_body.config(wraplength=520 if stacked else 300)
        self.status_hint.config(wraplength=520 if stacked else 360)

    def _get_selected_queue_item(self) -> Optional[QueueItem]:
        """Return the currently selected queue item, if any."""
        if not self._selected_queue_item_id:
            return None
        return next((item for item in self.queue if item.id == self._selected_queue_item_id), None)

    def _set_workflow_stage(self, stage: int):
        """Update the compact workflow pills in the header."""
        for idx, pill in enumerate(self._workflow_pills, start=1):
            if idx < stage:
                frame_bg = Theme.SUCCESS_BG
                frame_border = Theme.GREEN_HOVER
                badge_bg = Theme.GREEN_PRIMARY
                badge_fg = "#04120b"
                text_fg = Theme.SUCCESS
            elif idx == stage:
                frame_bg = Theme.BLUE_MUTED
                frame_border = Theme.BLUE_PRIMARY
                badge_bg = Theme.BLUE_PRIMARY
                badge_fg = "#071226"
                text_fg = Theme.TEXT_PRIMARY
            else:
                frame_bg = Theme.BG_CARD
                frame_border = Theme.BORDER
                badge_bg = Theme.BG_TERTIARY
                badge_fg = Theme.TEXT_MUTED
                text_fg = Theme.TEXT_SECONDARY
            pill["frame"].config(bg=frame_bg, highlightbackground=frame_border)
            pill["badge"].config(bg=badge_bg, fg=badge_fg)
            pill["text"].config(bg=frame_bg, fg=text_fg)

    def _update_guidance_surface(self):
        """Keep the header guidance card and footer hint aligned with state."""
        if not hasattr(self, "header_guidance_title"):
            return

        selected = self._get_selected_queue_item()
        has_queue = bool(self.queue)
        has_complete = any(item.status == ProcessingStatus.COMPLETE for item in self.queue)
        has_retry = any(item.status in (ProcessingStatus.ERROR, ProcessingStatus.CANCELLED)
                        for item in self.queue)

        if self._stop_requested:
            stage = 3
            title = "Stopping batch"
            body = ("The current item is wrapping up so the app can stop cleanly without risking overlapping work. "
                    "Finished outputs stay on disk and remaining items will be marked as stopped.")
            hint = "Stopping safely. Please wait for the current item to finish its active step."
        elif self.is_processing:
            stage = 3
            title = "Batch running"
            body = ("Live preview, ETA, and the activity log stay up to date while the batch works. "
                    "Stop is safe: completed outputs stay on disk.")
            hint = "Use Stop batch if you need to pause. Finished outputs are preserved."
        elif not has_queue:
            stage = 1
            title = "Build your batch"
            body = "Import files or choose a folder to start."
            hint = "Import files or choose a folder to start."
        elif has_retry:
            stage = 3 if has_complete else 2
            title = "Review the outliers"
            body = "Retry failed items or open the log for details."
            hint = "Retry failed items or open the log for details."
        elif not selected:
            stage = 2
            title = "Inspect a sample frame"
            body = "Select one item and review the mask before starting."
            hint = "Select one item and review the mask before starting."
        elif has_complete:
            stage = 3
            title = "Outputs are ready"
            body = "Preview a finished item or open the output folder."
            hint = "Preview a finished item or open the output folder."
        else:
            stage = 3
            title = "Ready to run"
            body = "Start the batch when the preview framing looks right."
            hint = "Start the batch when the preview framing looks right."

        self._set_workflow_stage(stage)
        self.header_guidance_title.config(text=title)
        self.header_guidance_body.config(text=body)
        if hasattr(self, "status_hint"):
            self.status_hint.config(text=hint)

    def _update_preview_actions(self):
        """Enable preview tools only when they make sense for the selection."""
        if not hasattr(self, "preview_mask_btn"):
            return
        selected = self._get_selected_queue_item()
        can_preview = bool(selected and PIL_AVAILABLE)
        self.preview_mask_btn.set_enabled(bool(selected) and not self.is_processing)
        self.preview_zoom_btn.set_enabled(can_preview)
        if hasattr(self, "preview_inpaint_btn"):
            self.preview_inpaint_btn.set_enabled(
                bool(selected) and not self.is_processing
            )
        if hasattr(self, "preview_ab_btn"):
            ab_ready = bool(
                selected
                and selected.status == ProcessingStatus.COMPLETE
                and Path(selected.output_path).exists()
            )
            self.preview_ab_btn.set_enabled(ab_ready)
        self._preview_label.config(cursor="hand2" if can_preview else "")

        if selected:
            badge = status_ui(selected.status)
            self.preview_status_chip.config(
                text=badge["label"],
                fg=badge["color"],
                bg=badge["bg"],
            )
        else:
            self.preview_status_chip.config(
                text="Waiting",
                fg=Theme.TEXT_MUTED,
                bg=Theme.BG_TERTIARY,
            )

    def _open_selected_mask_preview(self):
        item = self._get_selected_queue_item()
        if item:
            self._show_preview(item, show_mask=True)

    def _open_ab_scrubber(self):
        """RM-30: open a Toplevel that lets the user scrub frames AND
        wipe a vertical seam left/right to compare the original vs the
        cleaned output side-by-side at any frame.

        The window opens both video captures, holds them open for the
        duration of the modal, and composes a single image per scrub.
        """
        item = self._get_selected_queue_item()
        if item is None or item.status != ProcessingStatus.COMPLETE:
            self._update_status("Select a completed item first", "warning")
            return
        in_path = item.file_path
        out_path = item.output_path
        if not Path(out_path).exists():
            self._update_status("Output file is missing", "warning")
            return
        if not PIL_AVAILABLE:
            self._update_status("Pillow required for A/B compare", "warning")
            return

        import cv2 as _cv2
        cap_a = _cv2.VideoCapture(in_path)
        cap_b = _cv2.VideoCapture(out_path)
        if not cap_a.isOpened() or not cap_b.isOpened():
            cap_a.release(); cap_b.release()
            self._update_status("Could not open input/output for compare", "warning")
            return

        n_a = int(cap_a.get(_cv2.CAP_PROP_FRAME_COUNT)) or 1
        n_b = int(cap_b.get(_cv2.CAP_PROP_FRAME_COUNT)) or 1
        n_total = max(1, min(n_a, n_b))
        fps = cap_a.get(_cv2.CAP_PROP_FPS) or 30.0
        if fps <= 0:
            fps = 30.0
        max_w = min(1024, int(self.root.winfo_screenwidth() * 0.7))
        max_h = min(576, int(self.root.winfo_screenheight() * 0.6))

        win = tk.Toplevel(self.root)
        win.title(f"A/B compare: {Path(in_path).name}")
        win.configure(bg=Theme.BG_OVERLAY)
        win.resizable(False, False)

        canvas = tk.Canvas(win, width=max_w, height=max_h,
                            highlightthickness=0, bg=Theme.BG_DARK)
        canvas.pack()
        image_id = canvas.create_image(0, 0, anchor="nw")
        canvas._photo = None

        state = {"frame_idx": 0, "seam": max_w // 2}

        def _render():
            cap_a.set(_cv2.CAP_PROP_POS_FRAMES, state["frame_idx"])
            ok_a, fa = cap_a.read()
            cap_b.set(_cv2.CAP_PROP_POS_FRAMES, min(n_b - 1, state["frame_idx"]))
            ok_b, fb = cap_b.read()
            if not (ok_a and ok_b):
                return
            if fa.shape != fb.shape:
                fb = _cv2.resize(fb, (fa.shape[1], fa.shape[0]),
                                  interpolation=_cv2.INTER_AREA)
            h, w = fa.shape[:2]
            scale = min(max_w / w, max_h / h, 1.0)
            dw, dh = int(w * scale), int(h * scale)
            seam = max(0, min(dw, state["seam"]))
            fa_r = _cv2.resize(fa, (dw, dh), interpolation=_cv2.INTER_AREA)
            fb_r = _cv2.resize(fb, (dw, dh), interpolation=_cv2.INTER_AREA)
            composite = fa_r.copy()
            composite[:, seam:] = fb_r[:, seam:]
            # Draw a 2-pixel green seam line for the wipe boundary.
            if 0 < seam < dw:
                _cv2.line(composite, (seam, 0), (seam, dh - 1), (0, 255, 0), 2)
            rgb = _cv2.cvtColor(composite, _cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(rgb)
            canvas._photo = ImageTk.PhotoImage(pil)
            canvas.itemconfig(image_id, image=canvas._photo)

        # Frame slider (vertical -- below the image).
        slider_row = tk.Frame(win, bg=Theme.BG_OVERLAY)
        slider_row.pack(fill="x", padx=Theme.S_MD, pady=(Theme.S_SM, 0))
        tk.Label(slider_row, text="Frame", font=f(Theme.F_BODY_SM),
                 bg=Theme.BG_OVERLAY, fg=Theme.TEXT_SECONDARY).pack(side="left")
        ts_label = tk.Label(slider_row, text="00:00:00",
                            font=f(Theme.F_META),
                            bg=Theme.BG_OVERLAY, fg=Theme.TEXT_MUTED)
        ts_label.pack(side="right")

        def _on_frame(value):
            try:
                state["frame_idx"] = max(0, min(n_total - 1, int(float(value))))
            except (TypeError, ValueError):
                return
            secs = state["frame_idx"] / fps
            hh = int(secs // 3600); mm = int((secs % 3600) // 60); ss = int(secs % 60)
            ts_label.config(text=f"{hh:02d}:{mm:02d}:{ss:02d}")
            _render()

        frame_slider = tk.Scale(
            win, from_=0, to=n_total - 1, orient="horizontal",
            command=_on_frame, showvalue=False, length=max_w - 24,
            bg=Theme.BG_OVERLAY, fg=Theme.TEXT_PRIMARY,
            troughcolor=Theme.BG_TERTIARY,
            activebackground=Theme.BLUE_PRIMARY, highlightthickness=0,
        )
        frame_slider.pack(fill="x", padx=Theme.S_MD, pady=(0, Theme.S_SM))

        # Seam slider (wipe boundary).
        seam_row = tk.Frame(win, bg=Theme.BG_OVERLAY)
        seam_row.pack(fill="x", padx=Theme.S_MD, pady=(0, 0))
        tk.Label(seam_row, text="Wipe", font=f(Theme.F_BODY_SM),
                 bg=Theme.BG_OVERLAY, fg=Theme.TEXT_SECONDARY).pack(side="left")
        tk.Label(seam_row, text="source <-> cleaned",
                 font=f(Theme.F_META),
                 bg=Theme.BG_OVERLAY, fg=Theme.TEXT_MUTED).pack(side="right")

        def _on_seam(value):
            try:
                state["seam"] = max(0, min(max_w, int(float(value))))
            except (TypeError, ValueError):
                return
            _render()

        seam_slider = tk.Scale(
            win, from_=0, to=max_w, orient="horizontal",
            command=_on_seam, showvalue=False, length=max_w - 24,
            bg=Theme.BG_OVERLAY, fg=Theme.TEXT_PRIMARY,
            troughcolor=Theme.BG_TERTIARY,
            activebackground=Theme.GREEN_PRIMARY, highlightthickness=0,
        )
        seam_slider.set(state["seam"])
        seam_slider.pack(fill="x", padx=Theme.S_MD, pady=(0, Theme.S_MD))

        def _close():
            try:
                cap_a.release(); cap_b.release()
            except Exception:
                pass
            win.destroy()

        win.bind("<Escape>", lambda e: _close())
        win.protocol("WM_DELETE_WINDOW", _close)
        win.transient(self.root)
        win.grab_set()
        _render()

    def _open_selected_inpaint_preview(self):
        """F-3: run a single-frame detect + inpaint pass on the selected
        item and render the result in the preview pane. Lets users A/B
        settings without committing a full batch run."""
        item = self._get_selected_queue_item()
        if item is None:
            self._update_status("Select a queue item first", "warning")
            return
        if self.is_processing:
            self._update_status("Pause the batch before previewing", "warning")
            return
        if not PIL_AVAILABLE:
            self._update_status("Pillow required for inpaint preview", "warning")
            return

        self.preview_title_label.config(text=f"Inpainting {Path(item.file_path).name}")
        self.preview_meta_label.config(text="Running detect + inpaint on the first frame...")
        self._preview_label.config(image="", text="")
        self._preview_photo = None
        self._start_throbber()
        self._preview_label.update_idletasks()
        self._preview_request_id += 1
        request_id = self._preview_request_id
        snapshot_cfg = ProcessingConfig.from_dict(item.config.to_dict())
        source_path = item.file_path

        def _worker():
            import cv2 as _cv2
            try:
                from backend.processor import (
                    SubtitleRemover as _Remover,
                    ProcessingConfig as _BackendCfg,
                )
                if is_image_file(source_path):
                    frame = _cv2.imread(source_path)
                elif is_video_file(source_path):
                    cap = _cv2.VideoCapture(source_path)
                    try:
                        ret, frame = cap.read()
                    finally:
                        cap.release()
                    if not ret:
                        frame = None
                else:
                    frame = None
                if frame is None:
                    self.root.after(0, lambda: (
                        self._stop_throbber(),
                        self.preview_title_label.config(text="Preview unavailable"),
                        self.preview_meta_label.config(text="The selected file could not be read."),
                    ))
                    return

                # Build a backend config snapshot from the item.
                backend_cfg = _BackendCfg(
                    mode=self._gui_to_backend_mode(snapshot_cfg.mode.value),
                    device=self._gui_to_backend_device(
                        snapshot_cfg.use_gpu, snapshot_cfg.gpu_id),
                    detection_lang=snapshot_cfg.detection_lang,
                    detection_threshold=snapshot_cfg.detection_threshold,
                    subtitle_area=snapshot_cfg.subtitle_area,
                    subtitle_areas=snapshot_cfg.subtitle_areas,
                    mask_dilate_px=snapshot_cfg.mask_dilate_px,
                    mask_feather_px=snapshot_cfg.mask_feather_px,
                    tbe_enable=snapshot_cfg.tbe_enable,
                )
                remover = _Remover(backend_cfg)
                # Single-frame inpaint -- detect, build mask, inpaint.
                fixed = (snapshot_cfg.subtitle_areas
                          or ([snapshot_cfg.subtitle_area] if snapshot_cfg.subtitle_area else None))
                if fixed:
                    boxes = list(fixed)
                else:
                    boxes = remover.detector.detect(
                        frame, snapshot_cfg.detection_threshold)
                if not boxes:
                    # No detection -- show the source with a hint.
                    pil = Image.fromarray(_cv2.cvtColor(frame, _cv2.COLOR_BGR2RGB))
                    self.root.after(0, lambda: self._apply_inpaint_preview(
                        pil, "No text detected on the first frame", request_id, item.id))
                    return
                mask = remover._create_mask(frame.shape, boxes)
                [filled] = remover.inpainter.inpaint([frame], [mask])
                pil = Image.fromarray(_cv2.cvtColor(filled, _cv2.COLOR_BGR2RGB))
                meta = (f"Cleanup preview using {snapshot_cfg.mode.value}; "
                        f"{len(boxes)} region{'s' if len(boxes) != 1 else ''} masked.")
                self.root.after(0, lambda: self._apply_inpaint_preview(
                    pil, meta, request_id, item.id))
            except Exception as exc:
                logger.warning(f"Inpaint preview failed: {exc}")
                self.root.after(0, lambda: (
                    self._stop_throbber(),
                    self.preview_title_label.config(text="Inpaint preview failed"),
                    self.preview_meta_label.config(text=str(exc)),
                ))

        threading.Thread(target=_worker, daemon=True).start()

    def _apply_inpaint_preview(self, pil_img, meta_text, request_id, item_id):
        if (request_id != self._preview_request_id
                or self._selected_queue_item_id != item_id):
            return
        try:
            self._stop_throbber()
            max_w = max(220, self._preview_frame.winfo_width() - 36)
            max_h = 158
            pil_img.thumbnail((max_w, max_h), Image.LANCZOS)
            self._preview_photo = ImageTk.PhotoImage(pil_img)
            self._preview_label.config(image=self._preview_photo, text="")
            self.preview_title_label.config(text="Inpaint preview")
            self.preview_meta_label.config(text=meta_text)
        except Exception:
            pass

    def _build_ui(self):
        """Build the main user interface with balanced spacing rhythm."""
        main_container = tk.Frame(self.root, bg=Theme.BG_DARK)
        main_container.pack(fill="both", expand=True,
                            padx=Theme.S_XL, pady=(Theme.S_LG, Theme.S_MD))

        # Header
        self._build_header(main_container)

        # Content area (two columns via grid)
        content = tk.Frame(main_container, bg=Theme.BG_DARK)
        content.pack(fill="both", expand=True, pady=(Theme.S_MD, 0))
        content.columnconfigure(0, weight=57, minsize=440)
        content.columnconfigure(1, weight=43, minsize=360)
        content.rowconfigure(0, weight=1)

        # Left column - Input & Settings
        left_col = tk.Frame(content, bg=Theme.BG_DARK)
        left_col.grid(row=0, column=0, sticky="nsew", padx=(0, Theme.S_MD))
        self._content = content
        self._left_col = left_col

        self._build_input_section(left_col)
        self._build_settings_section(left_col)

        # Right column - Queue & Preview
        right_col = tk.Frame(content, bg=Theme.BG_DARK)
        right_col.grid(row=0, column=1, sticky="nsew", padx=(Theme.S_MD, 0))
        self._right_col = right_col

        self._build_queue_section(right_col)

        # Log panel
        self._build_log_panel(main_container)

        # Footer
        self._build_footer(main_container)

    def _build_header(self, parent):
        """Minimal app header with short guidance and a few live status signals."""
        header = self._create_surface(parent)
        header.pack(fill="x")

        inner = tk.Frame(header, bg=Theme.BG_SECONDARY)
        inner.pack(fill="x", padx=Theme.S_XL, pady=Theme.S_LG)

        left = tk.Frame(inner, bg=Theme.BG_SECONDARY)
        left.pack(side="left", fill="both", expand=True)
        self._header_left = left

        tk.Label(left, text="Video Subtitle Remover",
                 font=f(Theme.F_DISPLAY, "bold"), bg=Theme.BG_SECONDARY,
                 fg=Theme.TEXT_PRIMARY).pack(anchor="w")
        tk.Label(left, text=f"Version {APP_VERSION}",
                 font=f(Theme.F_META, "bold"), bg=Theme.BG_SECONDARY,
                 fg=Theme.TEXT_MUTED).pack(anchor="w", pady=(2, 0))
        tk.Label(
            left,
            text="Add files, review one sample, then run the batch.",
            font=f(Theme.F_BODY),
            bg=Theme.BG_SECONDARY,
            fg=Theme.TEXT_SECONDARY,
        ).pack(anchor="w", pady=(8, 0))

        right = tk.Frame(inner, bg=Theme.BG_SECONDARY)
        right.pack(side="right", anchor="n")
        self._header_right = right

        gpu_short = truncate_middle(self.gpus[0]["name"], 26) if self.gpus else "CPU mode"
        gpu_fg = Theme.SUCCESS if self.gpus else Theme.WARNING
        det_short = self.ai_engines["detection"][0] if self.ai_engines["detection"] else "OpenCV fallback"
        audio_short = "FFmpeg ready" if self.ffmpeg_ready else "No FFmpeg"
        audio_fg = Theme.SUCCESS if self.ffmpeg_ready else Theme.WARNING

        chips = tk.Frame(right, bg=Theme.BG_SECONDARY)
        chips.pack(anchor="e")
        self._header_chips = chips

        self._create_chip(chips, "Device", gpu_short, gpu_fg, Theme.BG_CARD).pack(side="left")
        self._create_chip(chips, "Detection", det_short, Theme.INFO, Theme.BG_CARD).pack(
            side="left", padx=(Theme.S_SM, 0))
        self._create_chip(chips, "Audio", audio_short, audio_fg, Theme.BG_CARD).pack(
            side="left", padx=(Theme.S_SM, 0))

        # About / help
        help_btn = ModernButton(right, text="Help", width=80,
                                command=self._show_about, style="ghost",
                                size="sm", icon="?")
        help_btn.pack(anchor="e", pady=(Theme.S_SM, 0))
        self._header_help_btn = help_btn

        self._header_guidance_panel = tk.Frame(right, bg=Theme.BG_SECONDARY)
        self._header_guidance_panel.pack(anchor="e", fill="x", pady=(Theme.S_SM, 0))

        # Workflow step pills (Import -> Inspect -> Run)
        pills_row = tk.Frame(self._header_guidance_panel, bg=Theme.BG_SECONDARY)
        pills_row.pack(anchor="w", pady=(0, Theme.S_SM))
        for idx, step_label in enumerate(("Import", "Inspect", "Run"), start=1):
            pill_frame = tk.Frame(pills_row, bg=Theme.BG_CARD,
                                  highlightthickness=1, highlightbackground=Theme.BORDER)
            badge_lbl = tk.Label(pill_frame, text=str(idx),
                                 font=f(Theme.F_META, "bold"),
                                 bg=Theme.BG_TERTIARY, fg=Theme.TEXT_MUTED,
                                 padx=5, pady=2)
            badge_lbl.pack(side="left", padx=(5, 0), pady=4)
            text_lbl = tk.Label(pill_frame, text=step_label,
                                font=f(Theme.F_BODY_SM),
                                bg=Theme.BG_CARD, fg=Theme.TEXT_SECONDARY)
            text_lbl.pack(side="left", padx=(Theme.S_XS, 8), pady=4)
            pill_frame.pack(side="left",
                            padx=(0 if idx == 1 else Theme.S_XS, 0))
            self._workflow_pills.append({
                "frame": pill_frame, "badge": badge_lbl, "text": text_lbl,
            })

        self.header_guidance_title = tk.Label(
            self._header_guidance_panel,
            text="Build your batch",
            font=f(Theme.F_TITLE, "bold"),
            bg=Theme.BG_SECONDARY,
            fg=Theme.TEXT_PRIMARY,
        )
        self.header_guidance_title.pack(anchor="w")
        self.header_guidance_body = tk.Label(
            self._header_guidance_panel,
            text="Import files or choose a folder to start.",
            font=f(Theme.F_BODY_SM),
            wraplength=300,
            justify="left",
            bg=Theme.BG_SECONDARY,
            fg=Theme.TEXT_MUTED,
        )
        self.header_guidance_body.pack(anchor="w", fill="x", pady=(4, 0))

    def _build_input_section(self, parent):
        """Workspace section: drop zone + output location."""
        section = self._create_surface(parent)
        section.pack(fill="x")

        self._section_title(
            section,
            eyebrow="Workspace",
            title="Import media",
            hint="Add videos or images. Originals are never modified.",
        )

        self.drop_area = DragDropFrame(section, self._on_files_dropped, height=142)
        self.drop_area.pack(fill="x", padx=Theme.S_XL, pady=(0, Theme.S_MD))

        out_surface = self._create_card(section)
        out_surface.pack(fill="x", padx=Theme.S_XL, pady=(0, Theme.S_LG))

        out_row = tk.Frame(out_surface, bg=Theme.BG_CARD)
        out_row.pack(fill="x", padx=Theme.S_LG, pady=Theme.S_MD)

        label_col = tk.Frame(out_row, bg=Theme.BG_CARD)
        label_col.pack(side="left", fill="x", expand=True)

        tk.Label(label_col, text="OUTPUT LOCATION", font=f(Theme.F_EYEBROW, "bold"),
                 bg=Theme.BG_CARD, fg=Theme.TEXT_MUTED).pack(anchor="w")

        self.output_dir_label = tk.Label(label_col, text="", font=f(Theme.F_BODY, "bold"),
                                         bg=Theme.BG_CARD, fg=Theme.TEXT_PRIMARY, anchor="w")
        self.output_dir_label.pack(anchor="w", pady=(4, 0))

        self.output_dir_meta = tk.Label(label_col, text="", font=f(Theme.F_META),
                                        bg=Theme.BG_CARD, fg=Theme.TEXT_MUTED, anchor="w")
        self.output_dir_meta.pack(anchor="w", pady=(2, 0))

        actions = tk.Frame(out_row, bg=Theme.BG_CARD)
        actions.pack(side="right", padx=(Theme.S_MD, 0))

        choose_btn = ModernButton(actions, text="Choose folder", width=120,
                                  command=self._choose_output_dir, style="accent",
                                  size="sm")
        choose_btn.pack(side="left")

        reset_btn = ModernButton(actions, text="Reset", width=76,
                                 command=self._reset_output_dir, style="ghost",
                                 size="sm")
        reset_btn.pack(side="left", padx=(Theme.S_SM, 0))

        self._update_output_label()

    def _build_settings_section(self, parent):
        """Settings section: profile + workflow + collapsible advanced controls."""
        section = self._create_surface(parent)
        section.pack(fill="both", expand=True, pady=(Theme.S_MD, 0))

        self._section_title(
            section,
            eyebrow="Processing",
            title="Settings",
            hint="Pick a profile, confirm the region, then start the batch.",
        )

        settings = tk.Frame(section, bg=Theme.BG_SECONDARY)
        settings.pack(fill="both", expand=True, padx=Theme.S_XL, pady=(0, Theme.S_LG))

        # ---- Profile card -----------------------------------------------
        profile_panel = self._create_card(settings)
        profile_panel.pack(fill="x")

        self._card_header(profile_panel, "Profile", "Processing profile")

        # Preset picker -- one-click recipe application. Built-ins + user-saved.
        preset_row = tk.Frame(profile_panel, bg=Theme.BG_CARD)
        preset_row.pack(fill="x", padx=Theme.S_LG, pady=(Theme.S_XS, Theme.S_SM))

        tk.Label(preset_row, text="Preset", font=f(Theme.F_BODY_SM),
                 bg=Theme.BG_CARD, fg=Theme.TEXT_SECONDARY).pack(side="left")

        self.preset_var = tk.StringVar(value="(custom)")
        preset_names = ["(custom)"] + [n for n, _ in list_presets()]
        self.preset_combo = ttk.Combobox(
            preset_row, textvariable=self.preset_var, values=preset_names,
            state="readonly", style="Dark.TCombobox", width=32,
            font=f(Theme.F_BODY_SM),
        )
        self.preset_combo.pack(side="left", padx=(Theme.S_SM, Theme.S_SM))
        self.preset_combo.bind("<<ComboboxSelected>>", self._on_preset_applied)

        save_preset_btn = ModernButton(
            preset_row, text="Save as...", command=self._save_preset_dialog,
            size="sm", style="ghost",
        )
        save_preset_btn.pack(side="left")

        export_preset_btn = ModernButton(
            preset_row, text="Export", command=self._export_preset_dialog,
            size="sm", style="ghost",
        )
        export_preset_btn.pack(side="left", padx=(Theme.S_XS, 0))
        Tooltip(export_preset_btn, "Write the current preset to a shareable JSON file.")

        import_preset_btn = ModernButton(
            preset_row, text="Import", command=self._import_preset_dialog,
            size="sm", style="ghost",
        )
        import_preset_btn.pack(side="left", padx=(Theme.S_XS, 0))
        Tooltip(import_preset_btn, "Load a preset JSON file into the user library.")

        # Algorithm -- segmented picker replaces the Combobox for speed + clarity
        tk.Label(profile_panel, text="Algorithm", font=f(Theme.F_BODY_SM),
                 bg=Theme.BG_CARD, fg=Theme.TEXT_SECONDARY).pack(
                     anchor="w", padx=Theme.S_LG)

        self.mode_picker = SegmentedPicker(
            profile_panel,
            options=[(m.value, m.value) for m in InpaintMode],
            value=self.mode_var.get(),
            command=self._on_mode_picker_changed,
            bg=Theme.BG_CARD,
        )
        self.mode_picker.pack(fill="x", padx=Theme.S_LG, pady=(Theme.S_XS, 0))

        self.algo_desc = tk.Label(profile_panel, text=self._get_algo_description(),
                                  font=f(Theme.F_BODY_SM), bg=Theme.BG_CARD,
                                  fg=Theme.TEXT_SECONDARY, justify="left", anchor="w",
                                  wraplength=520)
        self.algo_desc.pack(fill="x", padx=Theme.S_LG, pady=(2, Theme.S_MD))

        if self.gpus:
            row2 = tk.Frame(profile_panel, bg=Theme.BG_CARD)
            row2.pack(fill="x", padx=Theme.S_LG, pady=(0, Theme.S_SM))

            tk.Label(row2, text="Compute device", font=f(Theme.F_BODY_SM),
                     bg=Theme.BG_CARD, fg=Theme.TEXT_SECONDARY).pack(side="left")

            gpu_options = [f"{g['name']} ({g['memory']})" for g in self.gpus]
            self.gpu_combo = ttk.Combobox(row2, textvariable=self.gpu_var, width=36,
                                          values=gpu_options, style="Dark.TCombobox",
                                          state="readonly", font=f(Theme.F_BODY_SM))
            self.gpu_combo.pack(side="right")
            self.gpu_combo.bind("<<ComboboxSelected>>", self._on_gpu_changed)

        lang_row = tk.Frame(profile_panel, bg=Theme.BG_CARD)
        lang_row.pack(fill="x", padx=Theme.S_LG, pady=(0, Theme.S_LG))

        tk.Label(lang_row, text="Subtitle language", font=f(Theme.F_BODY_SM),
                 bg=Theme.BG_CARD, fg=Theme.TEXT_SECONDARY).pack(side="left")

        # F-5: language list = the union of curated friendly names and
        # any extra codes the active OCR engine declares it supports.
        # PaddleOCR / RapidOCR ship 100+ languages; we expose the union
        # so users can pick e.g. Thai or Polish without modifying code.
        self._lang_display = _build_language_list()
        self._lang_labels = [f"{name} ({code})" for code, name in self._lang_display]
        self._lang_by_label = {label: code for label, (code, _) in
                               zip(self._lang_labels, self._lang_display)}
        self._lang_display_var = tk.StringVar()
        self._set_lang_display(self.lang_var.get())

        self.lang_combo = ttk.Combobox(lang_row, textvariable=self._lang_display_var,
                                       width=20, values=self._lang_labels,
                                       style="Dark.TCombobox",
                                       state="readonly", font=f(Theme.F_BODY_SM))
        self.lang_combo.pack(side="right")
        self.lang_combo.bind("<<ComboboxSelected>>", self._on_lang_changed)

        # ---- Workflow card ----------------------------------------------
        workflow_panel = self._create_card(settings)
        workflow_panel.pack(fill="x", pady=(Theme.S_MD, 0))

        self._card_header(workflow_panel, "Workflow", "Detection and output")

        checks_frame = tk.Frame(workflow_panel, bg=Theme.BG_CARD)
        checks_frame.pack(fill="x", padx=Theme.S_LG, pady=(0, Theme.S_MD))

        self.skip_check = ModernToggle(
            checks_frame,
            text="Reuse a fixed subtitle region (skip per-frame scanning)",
            variable=self.skip_detection_var,
        )
        self.skip_check.pack(anchor="w")
        Tooltip(self.skip_check, "Skip repeated detection when you have already set a precise subtitle region.")

        self.lama_check = ModernToggle(
            checks_frame,
            text="LaMa fast mode - favor speed over fill detail",
            variable=self.lama_fast_var,
        )
        self.lama_check.pack(anchor="w", pady=(Theme.S_SM, 0))
        Tooltip(self.lama_check, "LaMa fast mode is useful for quick passes and lower-resolution drafts.")

        self.preserve_audio_check = ModernToggle(
            checks_frame,
            text="Preserve the source audio track",
            variable=self.preserve_audio_var,
        )
        self.preserve_audio_check.pack(anchor="w", pady=(Theme.S_SM, 0))
        if not self.ffmpeg_ready:
            tk.Label(
                checks_frame,
                text="FFmpeg is not available, so outputs will be saved without original audio until it is installed.",
                font=f(Theme.F_META),
                bg=Theme.BG_CARD,
                fg=Theme.WARNING,
                wraplength=520,
                justify="left",
            ).pack(anchor="w", pady=(Theme.S_XS, 0))

        # Region surface -- raised card-within-card
        region_surface = tk.Frame(workflow_panel, bg=Theme.BG_TERTIARY,
                                  highlightthickness=1,
                                  highlightbackground=Theme.BORDER_SUBTLE)
        region_surface.pack(fill="x", padx=Theme.S_LG, pady=(Theme.S_XS, Theme.S_LG))

        region_text = tk.Frame(region_surface, bg=Theme.BG_TERTIARY)
        region_text.pack(side="left", fill="x", expand=True, padx=Theme.S_MD, pady=Theme.S_MD)

        tk.Label(region_text, text="SUBTITLE REGION", font=f(Theme.F_EYEBROW, "bold"),
                 bg=Theme.BG_TERTIARY, fg=Theme.TEXT_MUTED).pack(anchor="w")

        self.region_label = tk.Label(region_text, text="", font=f(Theme.F_BODY, "bold"),
                                     bg=Theme.BG_TERTIARY, fg=Theme.TEXT_PRIMARY,
                                     anchor="w")
        self.region_label.pack(anchor="w", pady=(4, 0))

        self.region_meta = tk.Label(region_text, text="", font=f(Theme.F_META),
                                    bg=Theme.BG_TERTIARY, fg=Theme.TEXT_MUTED,
                                    anchor="w")
        self.region_meta.pack(anchor="w", pady=(2, 0))

        region_actions = tk.Frame(region_surface, bg=Theme.BG_TERTIARY)
        region_actions.pack(side="right", padx=Theme.S_MD, pady=Theme.S_MD)

        self.region_btn = ModernButton(region_actions, text="Set region", width=100,
                                       command=self._open_region_selector, style="accent",
                                       size="sm")
        self.region_btn.pack(side="left")

        self.region_reset_btn = ModernButton(region_actions, text="Reset", width=76,
                                             command=self._reset_region, style="ghost",
                                             size="sm")
        self.region_reset_btn.pack(side="left", padx=(Theme.S_SM, 0))

        # ---- Advanced toggle --------------------------------------------
        adv_frame = tk.Frame(settings, bg=Theme.BG_SECONDARY)
        adv_frame.pack(fill="x", pady=(Theme.S_MD, 0))

        self.adv_visible = False
        self.adv_toggle = ModernButton(adv_frame, text="Show detailed controls", width=188,
                                       command=self._toggle_advanced,
                                       style="ghost", size="sm", icon="+")
        self.adv_toggle.pack(anchor="w")

        self.adv_panel = tk.Frame(settings, bg=Theme.BG_SECONDARY)

        # STTN Motion card
        sttn_frame = self._create_card(self.adv_panel)
        sttn_frame.pack(fill="x", pady=(Theme.S_MD, Theme.S_SM))
        self._card_header(sttn_frame, "STTN motion", "Temporal coherence")

        self._create_slider(sttn_frame, "Neighbor stride", 5, 30,
                            self.config.sttn_neighbor_stride, "sttn_neighbor_stride")
        self._create_slider(sttn_frame, "Reference length", 5, 30,
                            self.config.sttn_reference_length, "sttn_reference_length")
        self._create_slider(sttn_frame, "Max load frames", 10, 100,
                            self.config.sttn_max_load_num, "sttn_max_load_num")
        tk.Frame(sttn_frame, bg=Theme.BG_CARD, height=Theme.S_SM).pack(fill="x")

        # Detection Precision card
        det_frame = self._create_card(self.adv_panel)
        det_frame.pack(fill="x", pady=(0, Theme.S_SM))
        self._card_header(det_frame, "Detection", "Precision tuning")

        self._create_slider(det_frame, "Sensitivity", 10, 90,
                            int(self.config.detection_threshold * 100),
                            "_detection_threshold_pct",
                            hint="Higher catches more text (lower confidence floor). Lower is stricter.")
        self._create_slider(det_frame, "Frame skip", 0, 10,
                            self.config.detection_frame_skip, "detection_frame_skip",
                            hint="Reuse the last mask for N frames to speed up long videos.")
        self._create_slider(det_frame, "Mask dilate", 0, 20,
                            self.config.mask_dilate_px, "mask_dilate_px",
                            hint="Expand detected regions for cleaner fill edges.")
        self._create_slider(det_frame, "Mask feather", 0, 15,
                            self.config.mask_feather_px, "mask_feather_px",
                            hint="Soft-blend the removal edge for seamless boundaries.")
        self._create_slider(det_frame, "Colour match ring", 0, 8,
                            self.config.edge_ring_px, "edge_ring_px",
                            hint="Post-inpaint edge-ring colour correction to kill faint seams.")

        self.auto_band_var = tk.BooleanVar(value=self.config.auto_band)
        auto_band_toggle = ModernToggle(
            det_frame,
            text="Auto-detect subtitle band on load",
            variable=self.auto_band_var,
        )
        auto_band_toggle.pack(anchor="w", padx=Theme.S_LG, pady=(Theme.S_SM, 0))
        Tooltip(auto_band_toggle, "Scan the first 30 frames and pin the dominant subtitle band before processing.")

        self.flow_warp_var = tk.BooleanVar(value=self.config.tbe_flow_warp)
        flow_toggle = ModernToggle(
            det_frame,
            text="Flow-warped temporal exposure (motion-heavy)",
            variable=self.flow_warp_var,
        )
        flow_toggle.pack(anchor="w", padx=Theme.S_LG, pady=(Theme.S_SM, 0))
        Tooltip(flow_toggle, "Farneback optical flow aligns frames before TBE aggregation. Slower but cleaner on pans and zooms.")

        self.scene_split_var = tk.BooleanVar(value=self.config.tbe_scene_cut_split)
        scene_toggle = ModernToggle(
            det_frame,
            text="Split TBE batches at scene cuts",
            variable=self.scene_split_var,
        )
        scene_toggle.pack(anchor="w", padx=Theme.S_LG, pady=(Theme.S_SM, 0))
        Tooltip(scene_toggle, "Prevents background aggregation across hard cuts. Turn off if your footage is uncut.")

        self.kalman_var = tk.BooleanVar(value=self.config.kalman_tracking)
        kalman_toggle = ModernToggle(
            det_frame,
            text="Kalman box tracking (flicker reduction)",
            variable=self.kalman_var,
        )
        kalman_toggle.pack(anchor="w", padx=Theme.S_LG, pady=(Theme.S_SM, 0))
        Tooltip(kalman_toggle, "Smooths per-frame OCR jitter and fills single-frame misses. Recommended.")

        self.phash_var = tk.BooleanVar(value=self.config.phash_skip_enable)
        phash_toggle = ModernToggle(
            det_frame,
            text="Adaptive mask reuse (perceptual hash)",
            variable=self.phash_var,
        )
        phash_toggle.pack(anchor="w", padx=Theme.S_LG, pady=(Theme.S_SM, 0))
        Tooltip(phash_toggle, "Skip OCR on frames nearly identical to the last detected one. Speeds up long static shots.")

        self.colour_tune_var = tk.BooleanVar(value=self.config.colour_tune_enable)
        colour_toggle = ModernToggle(
            det_frame,
            text="Colour-tuned mask expansion",
            variable=self.colour_tune_var,
        )
        colour_toggle.pack(anchor="w", padx=Theme.S_LG, pady=(Theme.S_SM, 0))
        Tooltip(colour_toggle, "Grow the mask to cover serifs / drop shadows that match the subtitle colour. Catches decorative lettering.")

        # RM-24: vertical-text toggle for Japanese tategaki / classical CN.
        self.vertical_text_var = tk.BooleanVar(value=getattr(self.config, "detection_vertical", False))
        vertical_toggle = ModernToggle(
            det_frame,
            text="Vertical text mode (Japanese tategaki / classical Chinese)",
            variable=self.vertical_text_var,
        )
        vertical_toggle.pack(anchor="w", padx=Theme.S_LG, pady=(Theme.S_SM, Theme.S_MD))
        Tooltip(vertical_toggle, "Rotates each frame 90 CCW before OCR so columnar CJK reads as a line. Boxes rotate back to the source frame.")

        tk.Frame(det_frame, bg=Theme.BG_CARD, height=Theme.S_SM).pack(fill="x")

        # Output Quality card
        quality_frame = self._create_card(self.adv_panel)
        quality_frame.pack(fill="x", pady=(0, Theme.S_SM))
        self._card_header(quality_frame, "Output", "Encoding quality")

        self._create_slider(quality_frame, "CRF target", 15, 35,
                            self.config.output_quality, "output_quality",
                            hint="Lower = higher quality. 23 is a balanced default.")

        self.hw_encode_var = tk.BooleanVar(value=self.config.use_hw_encode)
        self.hw_encode_check = ModernToggle(
            quality_frame,
            text="Hardware encoding (NVENC / QSV / AMF) with software fallback",
            variable=self.hw_encode_var,
        )
        self.hw_encode_check.pack(anchor="w", padx=Theme.S_LG, pady=(Theme.S_SM, 0))
        Tooltip(self.hw_encode_check, "If hardware encoding fails the app retries automatically with libx264.")

        # F-8: output codec selector lives next to the HW-encode toggle.
        codec_row = tk.Frame(quality_frame, bg=Theme.BG_CARD)
        codec_row.pack(fill="x", padx=Theme.S_LG, pady=(Theme.S_SM, 0))
        tk.Label(codec_row, text="Output codec", font=f(Theme.F_BODY_SM),
                 bg=Theme.BG_CARD, fg=Theme.TEXT_SECONDARY).pack(side="left")
        self.output_codec_var = tk.StringVar(value=getattr(self.config, "output_codec", "h264"))
        codec_combo = ttk.Combobox(
            codec_row, textvariable=self.output_codec_var, width=10,
            values=["h264", "h265", "av1"],
            state="readonly", style="Dark.TCombobox", font=f(Theme.F_BODY_SM),
        )
        codec_combo.pack(side="right")
        Tooltip(codec_combo,
                "h264 is universal; h265 and av1 cut bitrate ~50% on 4K. Uses NVENC/QSV/AMF when available.")

        self.adaptive_batch_var = tk.BooleanVar(value=self.config.adaptive_batch)
        adaptive_toggle = ModernToggle(
            quality_frame,
            text="Adaptive batch sizing (probe free VRAM on init)",
            variable=self.adaptive_batch_var,
        )
        adaptive_toggle.pack(anchor="w", padx=Theme.S_LG, pady=(Theme.S_SM, 0))
        Tooltip(adaptive_toggle, "Scale the TBE window to fit free VRAM. Prevents OOM on 4K, unlocks headroom on 24 GB cards.")

        self.export_srt_var = tk.BooleanVar(value=self.config.export_srt)
        srt_toggle = ModernToggle(
            quality_frame,
            text="Export detected text as .srt sidecar",
            variable=self.export_srt_var,
        )
        srt_toggle.pack(anchor="w", padx=Theme.S_LG, pady=(Theme.S_SM, 0))
        Tooltip(srt_toggle, "Writes an .srt file next to the output using OCR text and timings.")

        self.export_mask_var = tk.BooleanVar(value=self.config.export_mask_video)
        mask_toggle = ModernToggle(
            quality_frame,
            text="Export debug mask video (.mask.mp4)",
            variable=self.export_mask_var,
        )
        mask_toggle.pack(anchor="w", padx=Theme.S_LG, pady=(Theme.S_SM, 0))
        Tooltip(mask_toggle, "Writes a black-and-white mp4 of the per-frame detection mask alongside the output.")

        self.deinterlace_var = tk.BooleanVar(value=self.config.deinterlace_auto)
        deinterlace_toggle = ModernToggle(
            quality_frame,
            text="Auto-deinterlace interlaced sources (yadif)",
            variable=self.deinterlace_var,
        )
        deinterlace_toggle.pack(anchor="w", padx=Theme.S_LG, pady=(Theme.S_SM, 0))
        Tooltip(deinterlace_toggle, "ffprobe-checks the input for combing; runs ffmpeg yadif if detected.")

        self.keyframe_var = tk.BooleanVar(value=self.config.keyframe_detection)
        keyframe_toggle = ModernToggle(
            quality_frame,
            text="Keyframe-driven detection (OCR only at I-frames)",
            variable=self.keyframe_var,
        )
        keyframe_toggle.pack(anchor="w", padx=Theme.S_LG, pady=(Theme.S_SM, 0))
        Tooltip(keyframe_toggle, "Large speedup for long videos. Falls back to pHash skip if ffprobe is missing.")

        self.quality_report_var = tk.BooleanVar(value=self.config.quality_report)
        quality_toggle = ModernToggle(
            quality_frame,
            text="Compute PSNR / SSIM quality report after run",
            variable=self.quality_report_var,
        )
        quality_toggle.pack(anchor="w", padx=Theme.S_LG, pady=(Theme.S_SM, Theme.S_MD))
        Tooltip(quality_toggle, "Samples 10 random frames, compares input vs output; logged and shown in the batch summary.")

        # Video Range card
        time_frame = self._create_card(self.adv_panel)
        time_frame.pack(fill="x")
        self._card_header(time_frame, "Video range", "Trim (videos only)")

        time_inner = tk.Frame(time_frame, bg=Theme.BG_CARD)
        time_inner.pack(fill="x", padx=Theme.S_LG, pady=(0, Theme.S_MD))

        tk.Label(time_inner, text="Start (s)", font=f(Theme.F_BODY_SM),
                 bg=Theme.BG_CARD, fg=Theme.TEXT_SECONDARY).pack(side="left")
        self.time_start_entry = tk.Entry(
            time_inner, width=7, bg=Theme.BG_TERTIARY,
            fg=Theme.TEXT_PRIMARY, font=f(Theme.F_BODY_SM),
            insertbackground=Theme.TEXT_PRIMARY,
            highlightthickness=1,
            highlightbackground=Theme.BORDER,
            highlightcolor=Theme.BORDER_FOCUS,
            relief="flat", bd=6)
        self.time_start_entry.insert(0, str(self.config.time_start or 0))
        self.time_start_entry.pack(side="left", padx=(Theme.S_SM, Theme.S_MD))

        tk.Label(time_inner, text="End (s)", font=f(Theme.F_BODY_SM),
                 bg=Theme.BG_CARD, fg=Theme.TEXT_SECONDARY).pack(side="left")
        self.time_end_entry = tk.Entry(
            time_inner, width=7, bg=Theme.BG_TERTIARY,
            fg=Theme.TEXT_PRIMARY, font=f(Theme.F_BODY_SM),
            insertbackground=Theme.TEXT_PRIMARY,
            highlightthickness=1,
            highlightbackground=Theme.BORDER,
            highlightcolor=Theme.BORDER_FOCUS,
            relief="flat", bd=6)
        self.time_end_entry.insert(0, str(self.config.time_end or 0))
        self.time_end_entry.pack(side="left", padx=(Theme.S_SM, 0))

        tk.Label(time_inner, text="0 uses the full clip", font=f(Theme.F_META),
                 bg=Theme.BG_CARD, fg=Theme.TEXT_MUTED).pack(side="left", padx=(Theme.S_MD, 0))

        # ---- v3.13 GUI-exposed knobs ------------------------------------
        # Editorial: chyron vs subtitle filter + karaoke grouping
        editorial_frame = self._create_card(self.adv_panel)
        editorial_frame.pack(fill="x", pady=(Theme.S_MD, Theme.S_SM))
        self._card_header(editorial_frame, "Editorial", "Filter what gets removed")

        self.remove_subs_var = tk.BooleanVar(value=self.config.remove_subtitles)
        remove_subs_toggle = ModernToggle(
            editorial_frame,
            text="Remove dialogue subtitles (short-lived OCR tracks)",
            variable=self.remove_subs_var,
        )
        remove_subs_toggle.pack(anchor="w", padx=Theme.S_LG, pady=(Theme.S_XS, 0))
        Tooltip(remove_subs_toggle, "Tracks the chyron classifier marks as dialogue subtitles.")

        self.remove_chyrons_var = tk.BooleanVar(value=self.config.remove_chyrons)
        remove_chyrons_toggle = ModernToggle(
            editorial_frame,
            text="Remove persistent text (logos, tickers, lower-thirds)",
            variable=self.remove_chyrons_var,
        )
        remove_chyrons_toggle.pack(anchor="w", padx=Theme.S_LG, pady=(Theme.S_SM, 0))
        Tooltip(remove_chyrons_toggle, "Kalman tracks lasting longer than ~3s are treated as chyrons.")

        self.karaoke_grouping_var = tk.BooleanVar(value=self.config.karaoke_grouping)
        karaoke_toggle = ModernToggle(
            editorial_frame,
            text="Karaoke grouping: fuse per-syllable boxes on the same line",
            variable=self.karaoke_grouping_var,
        )
        karaoke_toggle.pack(anchor="w", padx=Theme.S_LG, pady=(Theme.S_SM, Theme.S_MD))
        Tooltip(karaoke_toggle, "Stops karaoke captions leaking original text through the gaps between syllables.")

        # Audio card: loudnorm target + multi-track passthrough
        audio_frame = self._create_card(self.adv_panel)
        audio_frame.pack(fill="x", pady=(0, Theme.S_SM))
        self._card_header(audio_frame, "Audio", "Loudness + tracks")

        self.multi_audio_var = tk.BooleanVar(value=self.config.multi_audio_passthrough)
        multi_audio_toggle = ModernToggle(
            audio_frame,
            text="Pass through every audio stream (Bluray/DVD multi-track)",
            variable=self.multi_audio_var,
        )
        multi_audio_toggle.pack(anchor="w", padx=Theme.S_LG, pady=(Theme.S_XS, 0))
        Tooltip(multi_audio_toggle, "Mux every audio stream from the source. Off keeps only the first track.")

        loudnorm_row = tk.Frame(audio_frame, bg=Theme.BG_CARD)
        loudnorm_row.pack(fill="x", padx=Theme.S_LG, pady=(Theme.S_SM, Theme.S_MD))
        tk.Label(loudnorm_row, text="EBU R128 loudness target", font=f(Theme.F_BODY_SM),
                 bg=Theme.BG_CARD, fg=Theme.TEXT_SECONDARY).pack(side="left")
        self.loudnorm_var = tk.StringVar(value=str(self.config.loudnorm_target or 0.0))
        loudnorm_entry = tk.Entry(
            loudnorm_row, width=7, bg=Theme.BG_TERTIARY,
            fg=Theme.TEXT_PRIMARY, font=f(Theme.F_BODY_SM),
            insertbackground=Theme.TEXT_PRIMARY,
            highlightthickness=1,
            highlightbackground=Theme.BORDER,
            highlightcolor=Theme.BORDER_FOCUS,
            relief="flat", bd=6, textvariable=self.loudnorm_var)
        loudnorm_entry.pack(side="left", padx=(Theme.S_SM, Theme.S_MD))
        tk.Label(loudnorm_row,
                 text="LUFS. 0 = off. YouTube -14, Apple -16, broadcast -23.",
                 font=f(Theme.F_META),
                 bg=Theme.BG_CARD, fg=Theme.TEXT_MUTED).pack(side="left")

        # Performance card: decode HW accel + prefetch
        perf_frame = self._create_card(self.adv_panel)
        perf_frame.pack(fill="x", pady=(0, Theme.S_SM))
        self._card_header(perf_frame, "Performance", "Decode pipeline")

        accel_row = tk.Frame(perf_frame, bg=Theme.BG_CARD)
        accel_row.pack(fill="x", padx=Theme.S_LG, pady=(Theme.S_XS, 0))
        tk.Label(accel_row, text="Hardware-decode hint", font=f(Theme.F_BODY_SM),
                 bg=Theme.BG_CARD, fg=Theme.TEXT_SECONDARY).pack(side="left")
        self.decode_accel_var = tk.StringVar(value=self.config.decode_hw_accel or "off")
        accel_combo = ttk.Combobox(
            accel_row, textvariable=self.decode_accel_var, width=10,
            values=["off", "auto", "any", "d3d11", "vaapi", "mfx"],
            state="readonly", style="Dark.TCombobox", font=f(Theme.F_BODY_SM),
        )
        accel_combo.pack(side="right")
        Tooltip(accel_combo, "Hint for cv2.VideoCapture. Falls back to software if the HW path returns no frames.")

        self.prefetch_var = tk.BooleanVar(value=self.config.prefetch_decode)
        prefetch_toggle = ModernToggle(
            perf_frame,
            text="Worker-thread frame prefetch (overlap decode and inpaint)",
            variable=self.prefetch_var,
        )
        prefetch_toggle.pack(anchor="w", padx=Theme.S_LG, pady=(Theme.S_SM, Theme.S_MD))
        Tooltip(prefetch_toggle, "Decouples cv2.VideoCapture.read() from the detect+inpaint critical path. On by default.")

        # Quality sheet toggle (lives under Output but kept separate so we
        # don't disturb the existing Output card layout)
        self.quality_sheet_var = tk.BooleanVar(value=self.config.quality_report_sheet)
        quality_sheet_toggle = ModernToggle(
            quality_frame,
            text="Quality report sheet (side-by-side PNG comparison)",
            variable=self.quality_sheet_var,
        )
        quality_sheet_toggle.pack(anchor="w", padx=Theme.S_LG, pady=(0, Theme.S_SM))
        Tooltip(quality_sheet_toggle, "Renders <output>.qualitysheet.png with per-sample PSNR/SSIM. Implies the numeric report.")

        self.json_log_var = tk.BooleanVar(value=getattr(self.config, "json_log_enabled", False))
        json_log_toggle = ModernToggle(
            quality_frame,
            text="Structured JSON log",
            variable=self.json_log_var,
        )
        json_log_toggle.pack(anchor="w", padx=Theme.S_LG, pady=(0, Theme.S_SM))
        Tooltip(json_log_toggle, "Write a structured JSON-lines log alongside the text log. Useful for long batch runs and scripted post-processing.")

        # RM-96: high-contrast theme toggle. Takes effect on next launch
        # because re-skinning every live widget mid-session would force
        # a tree-wide redraw the design tokens were not built for.
        self.high_contrast_var = tk.BooleanVar(value=getattr(self.config, "high_contrast", False))
        hc_toggle = ModernToggle(
            quality_frame,
            text="High-contrast theme (restart required)",
            variable=self.high_contrast_var,
        )
        hc_toggle.pack(anchor="w", padx=Theme.S_LG, pady=(0, Theme.S_MD))
        Tooltip(hc_toggle, "Alternative palette tuned for low-vision users. Persists across sessions.")

        self.update_check_var = tk.BooleanVar(value=getattr(self.config, "update_check", False))
        uc_toggle = ModernToggle(
            quality_frame,
            text="Check for updates on startup",
            variable=self.update_check_var,
        )
        uc_toggle.pack(anchor="w", padx=Theme.S_LG, pady=(0, Theme.S_MD))
        Tooltip(uc_toggle, "When enabled, checks GitHub Releases for a newer version on launch. Off by default; no telemetry.")

        self._update_region_label_display()
        self._update_mode_options()

    def _create_slider(self, parent, label, min_val, max_val, default, attr_name,
                       hint: str = ""):
        """Create a labeled row with a ModernSlider and a value pill. Optional
        helper hint below."""
        parent_bg = parent.cget("bg") if hasattr(parent, "cget") else Theme.BG_CARD
        row = tk.Frame(parent, bg=parent_bg)
        row.pack(fill="x", padx=Theme.S_LG, pady=(Theme.S_XS, 2))

        tk.Label(row, text=label, font=f(Theme.F_BODY_SM),
                 bg=parent_bg, fg=Theme.TEXT_SECONDARY,
                 width=16, anchor="w").pack(side="left")

        # Value pill on the right
        pill = tk.Frame(row, bg=Theme.BG_TERTIARY, highlightthickness=1,
                        highlightbackground=Theme.BORDER_SUBTLE)
        pill.pack(side="right", padx=(Theme.S_MD, 0))
        value_label = tk.Label(pill, text=str(default), font=f(Theme.F_BODY_SM, "bold"),
                               bg=Theme.BG_TERTIARY, fg=Theme.GREEN_PRIMARY,
                               padx=8, pady=1, width=4)
        value_label.pack()

        slider = ModernSlider(row, from_=min_val, to=max_val, value=default,
                              bg=parent_bg)
        slider.pack(side="left", fill="x", expand=True, padx=(Theme.S_SM, 0))

        def on_change(val):
            value_label.config(text=str(int(val)))
            setattr(self.config, attr_name, int(val))

        slider.command = on_change

        if hint:
            tk.Label(parent, text=hint, font=f(Theme.F_META),
                     bg=parent_bg, fg=Theme.TEXT_MUTED,
                     anchor="w", justify="left").pack(
                         fill="x", padx=(Theme.S_LG, Theme.S_LG),
                         pady=(0, Theme.S_XS))

    def _toggle_advanced(self, event=None):
        """Toggle advanced settings visibility."""
        self.adv_visible = not self.adv_visible
        if self.adv_visible:
            self.adv_toggle.icon = "-"
            self.adv_toggle.set_text("Hide detailed controls")
            self.adv_panel.pack(fill="x")
        else:
            self.adv_toggle.icon = "+"
            self.adv_toggle.set_text("Show detailed controls")
            self.adv_panel.pack_forget()

    def _build_queue_section(self, parent):
        """Queue + preview + batch controls column."""
        section = self._create_surface(parent)
        section.pack(fill="both", expand=True)

        header = tk.Frame(section, bg=Theme.BG_SECONDARY)
        header.pack(fill="x", padx=Theme.S_XL, pady=(Theme.S_LG, Theme.S_XS))

        heading = tk.Frame(header, bg=Theme.BG_SECONDARY)
        heading.pack(side="left", fill="x", expand=True)

        tk.Label(heading, text="Queue",
                 font=f(Theme.F_HEADING, "bold"),
                 bg=Theme.BG_SECONDARY, fg=Theme.TEXT_PRIMARY).pack(anchor="w")
        tk.Label(heading, text="Review the list, then start the batch when ready.",
                 font=f(Theme.F_BODY_SM),
                 bg=Theme.BG_SECONDARY, fg=Theme.TEXT_MUTED).pack(anchor="w", pady=(2, 0))

        # Count + status chip cluster (right-aligned)
        count_cluster = tk.Frame(header, bg=Theme.BG_SECONDARY)
        count_cluster.pack(side="right", anchor="n")

        def _mk_stat_pill(fg=Theme.TEXT_SECONDARY, bg=Theme.BG_TERTIARY):
            pill = tk.Frame(count_cluster, bg=Theme.BG_SECONDARY)
            lbl = tk.Label(pill, text="", font=f(Theme.F_META, "bold"),
                           bg=bg, fg=fg, padx=8, pady=2)
            lbl.pack()
            return pill, lbl

        self.queue_total_pill, self.queue_count = _mk_stat_pill(
            fg=Theme.TEXT_PRIMARY, bg=Theme.BG_TERTIARY)
        self.queue_done_pill, self.queue_done_lbl = _mk_stat_pill(
            fg=Theme.SUCCESS, bg=Theme.SUCCESS_BG)
        self.queue_err_pill, self.queue_err_lbl = _mk_stat_pill(
            fg=Theme.ERROR, bg=Theme.ERROR_BG)

        self.queue_total_pill.pack(side="left")
        # done/err pills get shown conditionally in _update_queue_display
        self.queue_count.config(text="0 items")

        # Sort button -- hidden until queue has >= 3 items
        self._sort_btn = ModernButton(
            count_cluster, text="Sort", width=72,
            command=self._open_sort_menu, style="ghost", size="sm")
        # packed conditionally in _update_queue_display

        # Batch progress -- labels row above the bar
        batch_frame = tk.Frame(section, bg=Theme.BG_SECONDARY)
        batch_frame.pack(fill="x", padx=Theme.S_XL, pady=(Theme.S_MD, 0))

        meta_row = tk.Frame(batch_frame, bg=Theme.BG_SECONDARY)
        meta_row.pack(fill="x")

        self.batch_label = tk.Label(meta_row, text="Ready",
                                    font=f(Theme.F_META, "bold"),
                                    bg=Theme.BG_SECONDARY, fg=Theme.TEXT_MUTED)
        self.batch_label.pack(side="left")

        self.batch_percent_label = tk.Label(meta_row, text="",
                                            font=f(Theme.F_META, "bold"),
                                            bg=Theme.BG_SECONDARY, fg=Theme.TEXT_SECONDARY)
        self.batch_percent_label.pack(side="right")

        batch_bar_frame = tk.Frame(section, bg=Theme.BG_SECONDARY)
        batch_bar_frame.pack(fill="x", padx=Theme.S_XL, pady=(4, Theme.S_SM))

        self.batch_progress = ModernProgressBar(batch_bar_frame, width=300, height=5,
                                                 fill=Theme.BLUE_PRIMARY)
        self.batch_progress.pack(fill="x")
        def _resize_batch(event):
            if event.width > 40:
                self.batch_progress.resize(event.width - 4)
        batch_bar_frame.bind("<Configure>", _resize_batch)

        # Queue filter input -- appears when there are >5 items
        self._queue_filter_var = tk.StringVar()
        self._queue_filter_frame = tk.Frame(
            section, bg=Theme.BG_TERTIARY,
            highlightthickness=1, highlightbackground=Theme.BORDER)
        # Packed/unpacked dynamically in _update_queue_display
        filter_inner = tk.Frame(self._queue_filter_frame, bg=Theme.BG_TERTIARY)
        filter_inner.pack(fill="x", padx=Theme.S_SM, pady=2)

        tk.Label(filter_inner, text="Filter", font=f(Theme.F_META, "bold"),
                 bg=Theme.BG_TERTIARY, fg=Theme.TEXT_MUTED).pack(
                     side="left", padx=(Theme.S_SM, Theme.S_SM))
        self._queue_filter_entry = tk.Entry(
            filter_inner, textvariable=self._queue_filter_var,
            bg=Theme.BG_TERTIARY, fg=Theme.TEXT_PRIMARY,
            insertbackground=Theme.TEXT_PRIMARY,
            font=f(Theme.F_BODY_SM), relief="flat", bd=6,
            highlightthickness=0)
        self._queue_filter_entry.pack(side="left", fill="x", expand=True)
        self._queue_filter_entry.bind(
            "<FocusIn>",
            lambda e: self._queue_filter_frame.config(highlightbackground=Theme.BORDER_FOCUS),
        )
        self._queue_filter_entry.bind(
            "<FocusOut>",
            lambda e: self._queue_filter_frame.config(highlightbackground=Theme.BORDER),
        )
        self._queue_filter_clear = ModernButton(
            filter_inner, text="Clear", width=68,
            command=lambda: self._queue_filter_var.set(""),
            style="ghost", size="sm")
        self._queue_filter_clear.pack(side="right", padx=(Theme.S_SM, 0))
        self._queue_filter_var.trace_add(
            "write", lambda *_: self._apply_queue_filter())

        self._queue_container = tk.Frame(section, bg=Theme.BG_SECONDARY)
        self._queue_container.pack(fill="both", expand=True,
                                   padx=Theme.S_XL, pady=(0, Theme.S_SM))
        queue_container = self._queue_container

        self.queue_canvas = tk.Canvas(queue_container, bg=Theme.BG_SECONDARY,
                                     highlightthickness=0)
        scrollbar = ttk.Scrollbar(queue_container, orient="vertical",
                                 command=self.queue_canvas.yview,
                                 style="Dark.Vertical.TScrollbar")

        self.queue_frame = tk.Frame(self.queue_canvas, bg=Theme.BG_SECONDARY)

        self.queue_canvas.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y")
        self.queue_canvas.pack(side="left", fill="both", expand=True)

        self.queue_window = self.queue_canvas.create_window((0, 0), window=self.queue_frame,
                                                            anchor="nw")

        self.queue_frame.bind("<Configure>", self._on_queue_configure)
        self.queue_canvas.bind("<Configure>", self._on_canvas_configure)

        # Mousewheel scrolling
        self.queue_canvas.bind("<Enter>", self._bind_mousewheel)
        self.queue_canvas.bind("<Leave>", self._unbind_mousewheel)

        self._build_queue_empty_state()

        # Preview card
        self._preview_frame = self._create_card(section)
        self._preview_frame.pack(fill="x", padx=Theme.S_XL, pady=(0, Theme.S_MD))

        preview_header = tk.Frame(self._preview_frame, bg=Theme.BG_CARD)
        preview_header.pack(fill="x", padx=Theme.S_LG, pady=(Theme.S_MD, 0))

        preview_text = tk.Frame(preview_header, bg=Theme.BG_CARD)
        preview_text.pack(side="left", fill="x", expand=True)

        self.preview_title_label = tk.Label(preview_text, text="Preview a sample frame",
                                            font=f(Theme.F_TITLE, "bold"),
                                            bg=Theme.BG_CARD, fg=Theme.TEXT_PRIMARY)
        self.preview_title_label.pack(anchor="w")
        self.preview_meta_label = tk.Label(
            preview_text,
            text="Select a queued item and review the mask before processing.",
            font=f(Theme.F_META), wraplength=360,
            justify="left", bg=Theme.BG_CARD,
            fg=Theme.TEXT_MUTED)
        self.preview_meta_label.pack(anchor="w", pady=(4, 0))

        preview_actions = tk.Frame(preview_header, bg=Theme.BG_CARD)
        preview_actions.pack(side="right", anchor="ne")
        self.preview_status_chip = tk.Label(
            preview_actions,
            text="Waiting",
            font=f(Theme.F_META, "bold"),
            bg=Theme.BG_TERTIARY,
            fg=Theme.TEXT_MUTED,
            padx=10,
            pady=4,
        )
        self.preview_status_chip.pack(side="left", padx=(0, Theme.S_SM))
        self.preview_mask_btn = ModernButton(
            preview_actions,
            text="Review mask",
            width=108,
            command=self._open_selected_mask_preview,
            style="ghost",
            size="sm",
        )
        self.preview_mask_btn.pack(side="left")
        Tooltip(self.preview_mask_btn,
                "Run detection on the selected item and show the first-frame mask.")
        self.preview_zoom_btn = ModernButton(
            preview_actions,
            text="Full size",
            width=92,
            command=self._open_preview_zoom,
            style="ghost",
            size="sm",
        )
        self.preview_zoom_btn.pack(side="left", padx=(Theme.S_SM, 0))
        Tooltip(self.preview_zoom_btn,
                "Open the selected source frame in a larger viewer.")
        # F-3: cheap inpaint preview for the first frame of the selected
        # item. Runs detect + inpaint once and renders the result inline
        # so users can A/B settings before committing the batch.
        self.preview_inpaint_btn = ModernButton(
            preview_actions,
            text="Preview cleanup",
            width=128,
            command=self._open_selected_inpaint_preview,
            style="ghost",
            size="sm",
        )
        self.preview_inpaint_btn.pack(side="left", padx=(Theme.S_SM, 0))
        Tooltip(self.preview_inpaint_btn,
                "Run detect + inpaint on the first frame of the selected item.")
        # RM-30: A/B flicker scrubber for completed items.
        self.preview_ab_btn = ModernButton(
            preview_actions,
            text="A/B compare",
            width=108,
            command=self._open_ab_scrubber,
            style="ghost",
            size="sm",
        )
        self.preview_ab_btn.pack(side="left", padx=(Theme.S_SM, 0))
        Tooltip(self.preview_ab_btn,
                "Open a frame slider that wipes between source and cleaned output.")

        self._preview_label = tk.Label(self._preview_frame, bg=Theme.BG_CARD,
                                       text="", font=f(Theme.F_META),
                                       fg=Theme.TEXT_MUTED, compound="bottom",
                                       justify="center", cursor="hand2")
        self._preview_label.pack(fill="x", padx=Theme.S_LG, pady=(Theme.S_MD, Theme.S_LG))
        self._preview_photo = None
        self._preview_label.bind("<Double-Button-1>", self._open_preview_zoom)
        Tooltip(self._preview_label,
                "Double-click to view at full size. Right-click a queue item for more actions.")

        # Action bar -- Start is primary, secondary actions right-aligned
        btn_frame = tk.Frame(section, bg=Theme.BG_SECONDARY)
        btn_frame.pack(fill="x", padx=Theme.S_XL, pady=(0, Theme.S_LG))

        self.start_btn = ModernButton(btn_frame, text="Start batch", width=156,
                                     command=self._start_processing,
                                     style="primary", size="lg", icon=">")
        self.start_btn.pack(side="left")

        self.open_output_btn = ModernButton(btn_frame, text="Open output", width=132,
                                            command=self._open_output_folder,
                                            style="ghost", size="lg", icon="^")
        self.open_output_btn.pack(side="left", padx=(Theme.S_SM, 0))

        self.retry_btn = ModernButton(btn_frame, text="Retry failed", width=124,
                                      command=self._retry_failed,
                                      style="ghost", size="lg")
        self.retry_btn.pack(side="right")

        self.repeat_btn = ModernButton(btn_frame, text="Repeat last", width=120,
                                      command=self._repeat_last_job,
                                      style="ghost", size="lg")
        self.repeat_btn.pack(side="right")

        self.clear_btn = ModernButton(btn_frame, text="Clear queue", width=120,
                                     command=self._clear_queue,
                                     style="ghost", size="lg")
        self.clear_btn.pack(side="right", padx=(0, Theme.S_SM))

        self._set_preview_placeholder(
            "Preview a sample frame",
            "Select a queued item to inspect it before processing.",
        )
        self._refresh_action_states()

    def _build_queue_empty_state(self):
        """Queue empty state with short, clear guidance."""
        self.empty_container = tk.Frame(self.queue_frame, bg=Theme.BG_SECONDARY)
        self.empty_container.pack(pady=(Theme.S_3XL, Theme.S_LG), fill="x")

        icon = tk.Canvas(self.empty_container, width=60, height=60,
                         bg=Theme.BG_SECONDARY, highlightthickness=0)
        icon.pack()
        # Simple minimalist film-strip icon
        icon.create_rectangle(6, 12, 54, 48, outline=Theme.BORDER_STRONG, width=2)
        for x in (14, 30, 46):
            icon.create_rectangle(x - 5, 20, x + 5, 40,
                                  fill=Theme.BG_TERTIARY, outline="")

        tk.Label(self.empty_container, text="Your queue is empty",
                 font=f(Theme.F_TITLE, "bold"),
                 bg=Theme.BG_SECONDARY, fg=Theme.TEXT_SECONDARY).pack(pady=(Theme.S_MD, 4))
        tk.Label(self.empty_container,
                 text="Add files on the left to start a batch.",
                 font=f(Theme.F_BODY_SM),
                 bg=Theme.BG_SECONDARY, fg=Theme.TEXT_MUTED,
                 wraplength=340, justify="center").pack()

    def _ensure_filter_empty_state(self):
        """Create the queue filter empty state on demand."""
        if hasattr(self, "_filter_empty_container") and self._filter_empty_container.winfo_exists():
            return
        self._filter_empty_container = tk.Frame(self.queue_frame, bg=Theme.BG_SECONDARY)

        icon = tk.Canvas(self._filter_empty_container, width=52, height=52,
                         bg=Theme.BG_SECONDARY, highlightthickness=0)
        icon.pack()
        icon.create_oval(10, 10, 34, 34, outline=Theme.BORDER_STRONG, width=2)
        icon.create_line(30, 30, 42, 42, fill=Theme.BORDER_STRONG, width=2)

        self._filter_empty_title = tk.Label(
            self._filter_empty_container,
            text="No queued items match this search",
            font=f(Theme.F_TITLE, "bold"),
            bg=Theme.BG_SECONDARY,
            fg=Theme.TEXT_SECONDARY,
        )
        self._filter_empty_title.pack(pady=(Theme.S_MD, 4))
        self._filter_empty_body = tk.Label(
            self._filter_empty_container,
            text="Clear the filter or search for part of a filename.",
            font=f(Theme.F_BODY_SM),
            bg=Theme.BG_SECONDARY,
            fg=Theme.TEXT_MUTED,
            wraplength=340,
            justify="center",
        )
        self._filter_empty_body.pack()
        ModernButton(
            self._filter_empty_container,
            text="Clear filter",
            width=110,
            command=lambda: self._queue_filter_var.set(""),
            style="ghost",
            size="sm",
        ).pack(pady=(Theme.S_MD, 0))

    def _hide_filter_empty_state(self):
        if hasattr(self, "_filter_empty_container") and self._filter_empty_container.winfo_exists():
            self._filter_empty_container.pack_forget()

    def _bind_mousewheel(self, event):
        self._mousewheel_bound = True
        self.queue_canvas.bind("<MouseWheel>", self._on_mousewheel)
        # Also bind on children so scroll works when hovering queue items
        for child in self.queue_frame.winfo_children():
            child.bind("<MouseWheel>", self._on_mousewheel)

    def _unbind_mousewheel(self, event):
        self._mousewheel_bound = False
        self.queue_canvas.unbind("<MouseWheel>")

    def _on_mousewheel(self, event):
        self.queue_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    def _scroll_queue_to_item(self, item_id: str):
        """Scroll the queue canvas so the given item is fully visible."""
        widget = self.queue_widgets.get(item_id)
        if not widget:
            return
        try:
            self.queue_canvas.update_idletasks()
            bbox = self.queue_canvas.bbox("all")
            if not bbox:
                return
            total_h = max(1, bbox[3] - bbox[1])
            wy = widget.winfo_y()
            wh = widget.winfo_height()
            view_h = self.queue_canvas.winfo_height()
            top_frac, bot_frac = self.queue_canvas.yview()
            top_px = int(top_frac * total_h)
            bot_px = int(bot_frac * total_h)
            # Only scroll if not already in view
            if wy < top_px:
                self.queue_canvas.yview_moveto(max(0.0, wy / total_h))
            elif wy + wh > bot_px:
                target_top = wy + wh - view_h
                self.queue_canvas.yview_moveto(max(0.0, target_top / total_h))
        except Exception:
            pass

    def _on_queue_configure(self, event):
        self.queue_canvas.configure(scrollregion=self.queue_canvas.bbox("all"))

    def _on_canvas_configure(self, event):
        self.queue_canvas.itemconfig(self.queue_window, width=event.width)

    def _build_log_panel(self, parent):
        """Embedded, collapsible activity log."""
        log_section = self._create_surface(parent)
        log_section.pack(fill="x", pady=(Theme.S_MD, 0))

        log_header = tk.Frame(log_section, bg=Theme.BG_SECONDARY)
        log_header.pack(fill="x", padx=Theme.S_XL, pady=(Theme.S_MD, 0))

        # Title cluster (left)
        title_cluster = tk.Frame(log_header, bg=Theme.BG_SECONDARY)
        title_cluster.pack(side="left")
        tk.Label(title_cluster, text="ACTIVITY", font=f(Theme.F_EYEBROW, "bold"),
                 bg=Theme.BG_SECONDARY, fg=Theme.TEXT_MUTED).pack(anchor="w")
        tk.Label(title_cluster, text="Runtime log",
                 font=f(Theme.F_BODY, "bold"),
                 bg=Theme.BG_SECONDARY, fg=Theme.TEXT_SECONDARY).pack(anchor="w", pady=(2, 0))

        # Level badges: warn / error counts, packed in a row between title and toggle
        self._badge_row = tk.Frame(log_header, bg=Theme.BG_SECONDARY)
        self._badge_row.pack(side="left", padx=(Theme.S_MD, 0))
        self._log_warn_badge = tk.Label(
            self._badge_row, text="", font=f(Theme.F_META, "bold"),
            bg=Theme.WARNING_BG, fg=Theme.WARNING, padx=8, pady=3)
        self._log_error_badge = tk.Label(
            self._badge_row, text="", font=f(Theme.F_META, "bold"),
            bg=Theme.ERROR_BG, fg=Theme.ERROR, padx=8, pady=3)

        self._log_visible = True
        self._log_toggle_btn = ModernButton(log_header, text="Hide activity", width=120,
                                            command=self._toggle_log_panel,
                                            style="ghost", size="sm")
        self._log_toggle_btn.pack(side="left", padx=(Theme.S_MD, 0))

        open_log_btn = ModernButton(
            log_header, text="Open log file", width=118,
            command=self._open_log_file,
            style="ghost", size="sm")
        open_log_btn.pack(side="right")

        clear_log_btn = ModernButton(log_header, text="Clear", width=72,
                                     command=self._clear_log,
                                     style="ghost", size="sm")
        clear_log_btn.pack(side="right", padx=(0, Theme.S_SM))

        self._log_body = tk.Frame(log_section, bg=Theme.BG_LOG,
                                  highlightthickness=1,
                                  highlightbackground=Theme.BORDER_SUBTLE)
        self._log_body.pack(fill="x", padx=Theme.S_XL, pady=(Theme.S_SM, Theme.S_LG))

        self.log_text = tk.Text(self._log_body, height=6, bg=Theme.BG_LOG,
                                fg=Theme.TEXT_SECONDARY, font=mono(Theme.F_BODY_SM),
                                relief="flat", bd=8, state="disabled",
                                wrap="word", insertbackground=Theme.TEXT_PRIMARY,
                                selectbackground=Theme.BLUE_MUTED)
        log_scroll = ttk.Scrollbar(self._log_body, orient="vertical",
                                   command=self.log_text.yview,
                                   style="Dark.Vertical.TScrollbar")
        self.log_text.configure(yscrollcommand=log_scroll.set)
        log_scroll.pack(side="right", fill="y")
        self.log_text.pack(side="left", fill="both", expand=True)

        # Tag colors
        self.log_text.tag_configure("info", foreground=Theme.TEXT_MUTED)
        self.log_text.tag_configure("warning", foreground=Theme.WARNING)
        self.log_text.tag_configure("error", foreground=Theme.ERROR)

        # Initialize closed-state toggle (no flip on first run)
        # We start visible, so text stays "Hide activity"

    def _toggle_log_panel(self):
        """Toggle log panel visibility."""
        self._log_visible = not self._log_visible
        if self._log_visible:
            self._log_body.pack(fill="x", padx=Theme.S_XL, pady=(Theme.S_SM, Theme.S_LG))
            self._log_toggle_btn.set_text("Hide activity")
        else:
            self._log_body.pack_forget()
            self._log_toggle_btn.set_text("Show activity")

    def _update_log_badges(self, warn_count: int, error_count: int):
        """Show/hide warn/error count pills in the log header (always before toggle)."""
        try:
            if warn_count > 0:
                self._log_warn_badge.config(
                    text=f"{warn_count} warning{'s' if warn_count != 1 else ''}")
                self._log_warn_badge.pack(side="left", padx=(0, Theme.S_XS))
            else:
                self._log_warn_badge.pack_forget()
            if error_count > 0:
                self._log_error_badge.config(
                    text=f"{error_count} error{'s' if error_count != 1 else ''}")
                self._log_error_badge.pack(side="left", padx=(0, Theme.S_XS))
            else:
                self._log_error_badge.pack_forget()
        except Exception:
            pass

    def _clear_log(self):
        """Clear the log panel."""
        self.log_text.config(state="normal")
        self.log_text.delete("1.0", "end")
        self.log_text.config(state="disabled")
        if hasattr(self, "_log_handler"):
            self._log_handler.reset_counts()
        self._update_status("Activity log cleared")

    def _open_log_file(self):
        """Reveal the current log file in the system shell."""
        if not LOG_FILE.exists():
            self._update_status("The log file is not available yet", "warning")
            return
        try:
            os.startfile(str(LOG_FILE))
            self._update_status("Opened the log file", "info")
        except Exception:
            self._update_status("The log file could not be opened", "warning")

    def _open_settings_folder(self):
        try:
            os.startfile(str(LOG_DIR))
            self._update_status("Opened the settings folder", "info")
        except Exception:
            self._update_status("The settings folder could not be opened", "warning")

    def _build_footer(self, parent):
        """Footer status bar with a colored dot + message and a right-side hint."""
        footer = tk.Frame(parent, bg=Theme.BG_DARK)
        footer.pack(fill="x", pady=(Theme.S_SM, 0))
        self._footer = footer

        left = tk.Frame(footer, bg=Theme.BG_DARK)
        left.pack(side="left")
        self._footer_left = left

        # Status dot
        self.status_dot = tk.Canvas(left, width=10, height=10, bg=Theme.BG_DARK,
                                    highlightthickness=0)
        self._status_dot_item = self.status_dot.create_oval(
            1, 1, 9, 9, fill=Theme.TEXT_SECONDARY, outline="")
        self.status_dot.pack(side="left", padx=(0, Theme.S_SM), pady=2)

        self.status_label = tk.Label(left, text="Ready to process",
                                     font=f(Theme.F_BODY_SM, "bold"),
                                     bg=Theme.BG_DARK, fg=Theme.TEXT_SECONDARY, anchor="w")
        self.status_label.pack(side="left")

        self.status_hint = tk.Label(
            footer,
            text="Add files, review a sample frame, then start.",
            font=f(Theme.F_META),
            bg=Theme.BG_DARK,
            fg=Theme.TEXT_MUTED,
        )
        self.status_hint.pack(side="right")

    def _get_algo_description(self) -> str:
        """Get description for current algorithm."""
        descriptions = {
            "Auto": "Routes each batch to TBE or LaMa based on temporal exposure. Fastest on easy footage, automatically falls back to neural fill on hard frames.",
            "STTN": "Temporal background exposure. Reconstructs the real background from neighbouring frames where the subtitle is absent. Fastest, usually the best choice for live action.",
            "LAMA": "Neural single-frame fill. Highest-quality spatial inpaint for stills, animation, and clean backgrounds. Slower per frame.",
            "ProPainter": "Hybrid temporal + LaMa refinement. Best for motion-heavy footage or thick text. Higher VRAM and slower than STTN.",
        }
        return descriptions.get(self.mode_var.get(), "")

    def _on_mode_changed(self, event=None):
        """Handle algorithm mode change."""
        self.config.mode = InpaintMode(self.mode_var.get())
        self.algo_desc.config(text=self._get_algo_description())
        self._update_mode_options()
        self._update_status(f"Switched to the {self.mode_var.get()} profile")

    def _on_mode_picker_changed(self, value: str):
        """Segmented picker callback -- keep `mode_var` and the combobox path
        compatible."""
        self.mode_var.set(value)
        self._on_mode_changed()

    def _on_preset_applied(self, event=None):
        """Apply the chosen preset to the live config and refresh the UI."""
        name = self.preset_var.get()
        if name == "(custom)":
            return
        if not apply_preset(self.config, name):
            self._update_status(f"Preset '{name}' not found", "warning")
            return
        # Reflect preset changes in the mode picker + toggle vars that back
        # the detection / quality / output cards. The dataclass carries the
        # authoritative state; just push it out to every widget we track.
        self.mode_var.set(self.config.mode.value)
        try:
            self.mode_picker.set(self.config.mode.value)
        except Exception:
            pass
        for attr, field in (
            ("auto_band_var", "auto_band"),
            ("flow_warp_var", "tbe_flow_warp"),
            ("scene_split_var", "tbe_scene_cut_split"),
            ("kalman_var", "kalman_tracking"),
            ("phash_var", "phash_skip_enable"),
            ("colour_tune_var", "colour_tune_enable"),
            ("adaptive_batch_var", "adaptive_batch"),
            ("export_srt_var", "export_srt"),
            ("export_mask_var", "export_mask_video"),
        ):
            if hasattr(self, attr):
                getattr(self, attr).set(getattr(self.config, field))
        self._on_mode_changed()
        save_settings(self.config)
        self._update_status(f"Applied preset '{name}'", "success")

    def _export_preset_dialog(self):
        """Export the currently-selected preset to a shareable JSON file."""
        try:
            from tkinter import filedialog
            name = self.preset_var.get()
            if name == "(custom)":
                self._update_status("Pick a preset first, then export", "warning")
                return
            path = filedialog.asksaveasfilename(
                parent=self.root,
                title=f"Export preset '{name}'",
                defaultextension=".json",
                filetypes=[("VSR preset", "*.json"), ("All files", "*.*")],
                initialfile=f"{name.replace('/', '-')}.vsr-preset.json",
            )
            if not path:
                return
            if export_preset(name, path):
                self._update_status(f"Exported '{name}' to {Path(path).name}", "success")
            else:
                self._update_status("Export failed", "error")
        except Exception as exc:
            self._update_status(f"Export failed: {exc}", "error")

    def _import_preset_dialog(self):
        """Import a preset JSON into the user library and select it."""
        try:
            from tkinter import filedialog
            path = filedialog.askopenfilename(
                parent=self.root,
                title="Import preset",
                filetypes=[("VSR preset", "*.json"), ("All files", "*.*")],
            )
            if not path:
                return
            new_name = import_preset(path)
            if new_name is None:
                self._update_status("Not a valid VSR preset file", "error")
                return
            self.preset_combo['values'] = ["(custom)"] + [n for n, _ in list_presets()]
            self.preset_var.set(new_name)
            self._on_preset_applied()
            self._update_status(f"Imported preset '{new_name}'", "success")
        except Exception as exc:
            self._update_status(f"Import failed: {exc}", "error")

    def _prompt_preset_details(self) -> Optional[Tuple[str, str]]:
        """Open a themed modal for naming and describing a user preset."""
        result = {"value": None}

        dialog = tk.Toplevel(self.root)
        dialog.withdraw()
        dialog.title("Save preset")
        dialog.configure(bg=Theme.BG_OVERLAY)
        dialog.resizable(False, False)
        dialog.transient(self.root)

        outer = tk.Frame(dialog, bg=Theme.BORDER, padx=1, pady=1)
        outer.pack()
        body = tk.Frame(outer, bg=Theme.BG_SECONDARY)
        body.pack()

        content = tk.Frame(body, bg=Theme.BG_SECONDARY)
        content.pack(padx=28, pady=(24, 14))

        tk.Label(content, text="Save the current setup as a preset",
                 font=f(Theme.F_HEADING, "bold"),
                 bg=Theme.BG_SECONDARY, fg=Theme.TEXT_PRIMARY).pack(anchor="w")
        tk.Label(content,
                 text="Use a short name you will recognize later. Saving to an existing user preset name will update it.",
                 font=f(Theme.F_BODY_SM),
                 bg=Theme.BG_SECONDARY, fg=Theme.TEXT_MUTED,
                 justify="left", wraplength=420).pack(anchor="w", pady=(6, Theme.S_LG))

        form = tk.Frame(content, bg=Theme.BG_SECONDARY)
        form.pack(fill="x")

        def entry_row(label_text: str, initial: str = ""):
            row = tk.Frame(form, bg=Theme.BG_SECONDARY)
            row.pack(fill="x", pady=(0, Theme.S_MD))
            tk.Label(row, text=label_text, font=f(Theme.F_BODY_SM),
                     bg=Theme.BG_SECONDARY, fg=Theme.TEXT_SECONDARY).pack(anchor="w")
            entry = tk.Entry(
                row, bg=Theme.BG_TERTIARY, fg=Theme.TEXT_PRIMARY,
                insertbackground=Theme.TEXT_PRIMARY,
                font=f(Theme.F_BODY_SM), relief="flat", bd=6,
                highlightthickness=1, highlightbackground=Theme.BORDER,
                highlightcolor=Theme.BORDER_FOCUS,
            )
            entry.pack(fill="x", pady=(Theme.S_XS, 0))
            entry.insert(0, initial)
            return entry

        name_entry = entry_row("Preset name")
        desc_entry = entry_row("Description", "User preset")

        helper = tk.Label(content, text="Built-in preset names are reserved.",
                          font=f(Theme.F_META), bg=Theme.BG_SECONDARY,
                          fg=Theme.TEXT_MUTED)
        helper.pack(anchor="w")

        error_label = tk.Label(content, text="", font=f(Theme.F_META, "bold"),
                               bg=Theme.BG_SECONDARY, fg=Theme.ERROR)
        error_label.pack(anchor="w", pady=(Theme.S_SM, 0))

        actions = tk.Frame(body, bg=Theme.BG_CARD)
        actions.pack(fill="x")
        actions_inner = tk.Frame(actions, bg=Theme.BG_CARD)
        actions_inner.pack(side="right", padx=16, pady=14)

        def _cancel():
            dialog.grab_release()
            dialog.destroy()

        def _submit():
            name = name_entry.get().strip()
            description = desc_entry.get().strip() or "User preset"
            if not name:
                error_label.config(text="Give this preset a short name.")
                name_entry.focus_set()
                return
            if name in BUILTIN_PRESETS:
                error_label.config(text="Built-in preset names are reserved.")
                name_entry.focus_set()
                return
            result["value"] = (name, description)
            dialog.grab_release()
            dialog.destroy()

        ModernButton(actions_inner, text="Cancel", width=96,
                     command=_cancel, style="ghost", size="md").pack(side="left")
        ModernButton(actions_inner, text="Save preset", width=120,
                     command=_submit, style="primary", size="md").pack(
                         side="left", padx=(Theme.S_SM, 0))

        dialog.bind("<Escape>", lambda e: _cancel())
        dialog.bind("<Return>", lambda e: _submit())
        dialog.protocol("WM_DELETE_WINDOW", _cancel)

        dialog.update_idletasks()
        try:
            px = self.root.winfo_rootx()
            py = self.root.winfo_rooty()
            pw = self.root.winfo_width()
            ph = self.root.winfo_height()
            dw = dialog.winfo_reqwidth()
            dh = dialog.winfo_reqheight()
            dialog.geometry(f"+{px + (pw - dw) // 2}+{py + (ph - dh) // 3}")
        except Exception:
            pass

        dialog.deiconify()
        dialog.grab_set()
        name_entry.focus_set()
        dialog.wait_window()
        return result["value"]

    def _save_preset_dialog(self):
        """Prompt for a name + description and save a user preset."""
        try:
            details = self._prompt_preset_details()
            if not details:
                return
            name, description = details
            existing_user = name in _load_user_presets()
            self._sync_config_from_ui()
            if save_user_preset(name, description, self.config):
                # Refresh combo
                self.preset_combo['values'] = ["(custom)"] + [n for n, _ in list_presets()]
                self.preset_var.set(name)
                verb = "Updated" if existing_user else "Saved"
                self._update_status(f"{verb} preset '{name}'", "success")
            else:
                self._update_status(f"Could not save preset '{name}'", "error")
        except Exception as exc:
            self._update_status(f"Save preset failed: {exc}", "error")

    def _update_mode_options(self):
        """Enable/disable mode-specific toggles based on selected algorithm."""
        mode = self.mode_var.get()

        # Skip detection only for STTN
        if mode == "STTN":
            self.skip_check.set_enabled(True)
        else:
            self.skip_detection_var.set(False)
            self.skip_check.set_enabled(False)

        # LAMA fast only for LAMA
        if mode == "LAMA":
            self.lama_check.set_enabled(True)
        else:
            self.lama_fast_var.set(False)
            self.lama_check.set_enabled(False)

    def _maybe_show_onboarding(self):
        """Show a short 3-card welcome overlay on first launch."""
        if self.config.onboarding_seen:
            return
        # Guard against showing twice in the same session
        self.config.onboarding_seen = True
        # Let the main window settle first
        try:
            self.root.after(420, self._show_onboarding)
        except tk.TclError:
            pass

    def _show_onboarding(self):
        dialog = tk.Toplevel(self.root)
        dialog.withdraw()
        dialog.title(f"Welcome to {APP_NAME}")
        dialog.configure(bg=Theme.BG_OVERLAY)
        dialog.resizable(False, False)
        dialog.transient(self.root)

        outer = tk.Frame(dialog, bg=Theme.BORDER, padx=1, pady=1)
        outer.pack()
        body = tk.Frame(outer, bg=Theme.BG_SECONDARY)
        body.pack()

        content = tk.Frame(body, bg=Theme.BG_SECONDARY)
        content.pack(padx=36, pady=(28, 16))

        # Headline
        hero = tk.Frame(content, bg=Theme.BG_SECONDARY)
        hero.pack(anchor="w")
        tk.Label(hero, text="Welcome", font=f(Theme.F_DISPLAY, "bold"),
                 bg=Theme.BG_SECONDARY, fg=Theme.TEXT_PRIMARY).pack(
                     side="left")
        tk.Label(hero, text=f"v{APP_VERSION}", font=f(Theme.F_BODY_SM),
                 bg=Theme.BG_SECONDARY, fg=Theme.TEXT_MUTED).pack(
                     side="left", padx=(Theme.S_SM, 0), pady=(14, 0))

        tk.Label(content,
                 text="Three things that make batch cleanup painless.",
                 font=f(Theme.F_BODY),
                 bg=Theme.BG_SECONDARY, fg=Theme.TEXT_SECONDARY).pack(
                     anchor="w", pady=(4, Theme.S_LG))

        # Cue cards
        cards = tk.Frame(content, bg=Theme.BG_SECONDARY)
        cards.pack(anchor="w")

        def card(num: str, heading: str, body_text: str, tone: str):
            c = tk.Frame(cards, bg=Theme.BG_CARD, highlightthickness=1,
                         highlightbackground=Theme.BORDER)
            inner = tk.Frame(c, bg=Theme.BG_CARD)
            inner.pack(fill="both", expand=True, padx=16, pady=14)
            top = tk.Frame(inner, bg=Theme.BG_CARD)
            top.pack(anchor="w")
            # Numbered step badge
            badge_bg = {"info": Theme.INFO_BG, "success": Theme.SUCCESS_BG,
                        "warning": Theme.WARNING_BG}.get(tone, Theme.BG_TERTIARY)
            badge_fg = {"info": Theme.INFO, "success": Theme.SUCCESS,
                        "warning": Theme.WARNING}.get(tone, Theme.TEXT_SECONDARY)
            tk.Label(top, text=num, font=f(Theme.F_BODY_SM, "bold"),
                     bg=badge_bg, fg=badge_fg, padx=8, pady=2).pack(side="left")
            tk.Label(top, text=heading, font=f(Theme.F_BODY, "bold"),
                     bg=Theme.BG_CARD, fg=Theme.TEXT_PRIMARY).pack(
                         side="left", padx=(Theme.S_SM, 0))
            tk.Label(inner, text=body_text, font=f(Theme.F_BODY_SM),
                     bg=Theme.BG_CARD, fg=Theme.TEXT_SECONDARY,
                     wraplength=220, justify="left", anchor="w").pack(
                         anchor="w", pady=(Theme.S_SM, 0))
            return c

        card("1", "Import media",
             "Drop videos or images on the left, or pick an entire folder. "
             "Originals are never touched.",
             "info").pack(side="left", fill="both", expand=True,
                          padx=(0, Theme.S_SM))
        card("2", "Inspect the region",
             "Select a queued item and review the mask to confirm the subtitle "
             "mask before running the batch.",
             "warning").pack(side="left", fill="both", expand=True,
                             padx=(0, Theme.S_SM))
        card("3", "Run the batch",
             "Hit Start batch when the framing looks right. Progress, ETA, "
             "and completion summary are all live.",
             "success").pack(side="left", fill="both", expand=True)

        # Action row
        actions = tk.Frame(body, bg=Theme.BG_CARD)
        actions.pack(fill="x")
        actions_inner = tk.Frame(actions, bg=Theme.BG_CARD)
        actions_inner.pack(side="right", padx=16, pady=14)

        def _close():
            dialog.grab_release()
            dialog.destroy()

        ModernButton(actions_inner, text="Got it", width=118,
                     command=_close, style="primary", size="md").pack(
                         side="left")

        dialog.bind("<Escape>", lambda e: _close())
        dialog.bind("<Return>", lambda e: _close())
        dialog.protocol("WM_DELETE_WINDOW", _close)

        dialog.update_idletasks()
        try:
            px, py = self.root.winfo_rootx(), self.root.winfo_rooty()
            pw, ph = self.root.winfo_width(), self.root.winfo_height()
            dw, dh = dialog.winfo_reqwidth(), dialog.winfo_reqheight()
            dialog.geometry(f"+{px + (pw - dw) // 2}+{py + (ph - dh) // 3}")
        except Exception:
            pass
        dialog.deiconify()
        dialog.grab_set()

    def _open_preview_zoom(self, event=None):
        """Open the currently selected queue item's frame at a larger size."""
        if not PIL_AVAILABLE:
            return
        item_id = self._selected_queue_item_id
        if not item_id:
            return
        item = next((i for i in self.queue if i.id == item_id), None)
        if not item:
            return

        try:
            import cv2 as _cv2

            if is_video_file(item.file_path):
                cap = _cv2.VideoCapture(item.file_path)
                try:
                    ret, frame = cap.read()
                    if not ret:
                        return
                finally:
                    cap.release()
            else:
                frame = _cv2.imread(item.file_path)
                if frame is None:
                    return

            frame_rgb = _cv2.cvtColor(frame, _cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
        except Exception:
            return

        screen_w = self.root.winfo_screenwidth()
        screen_h = self.root.winfo_screenheight()
        max_w = int(screen_w * 0.82)
        max_h = int(screen_h * 0.82)
        if img.width > max_w or img.height > max_h:
            img.thumbnail((max_w, max_h), Image.LANCZOS)

        win = tk.Toplevel(self.root)
        win.withdraw()
        win.title(f"Preview - {Path(item.file_path).name}")
        win.configure(bg=Theme.BG_DARK)
        win.transient(self.root)

        header = tk.Frame(win, bg=Theme.BG_SECONDARY,
                          highlightthickness=1,
                          highlightbackground=Theme.BORDER_SUBTLE)
        header.pack(fill="x")
        tk.Label(header, text=Path(item.file_path).name,
                 font=f(Theme.F_BODY, "bold"),
                 bg=Theme.BG_SECONDARY, fg=Theme.TEXT_PRIMARY).pack(
                     side="left", padx=Theme.S_LG, pady=Theme.S_MD)
        tk.Label(header, text=f"{img.width} x {img.height}",
                 font=f(Theme.F_BODY_SM),
                 bg=Theme.BG_SECONDARY, fg=Theme.TEXT_MUTED).pack(
                     side="left", padx=(0, Theme.S_LG), pady=Theme.S_MD)
        ModernButton(header, text="Close", width=86,
                     command=win.destroy, style="ghost", size="sm").pack(
                         side="right", padx=Theme.S_LG, pady=Theme.S_SM)

        canvas = tk.Frame(win, bg=Theme.BG_DARK)
        canvas.pack(fill="both", expand=True, padx=Theme.S_LG,
                    pady=(Theme.S_LG, Theme.S_LG))
        photo = ImageTk.PhotoImage(img)
        label = tk.Label(canvas, image=photo, bg=Theme.BG_DARK)
        label.image = photo  # prevent GC
        label.pack(anchor="center")

        win.bind("<Escape>", lambda e: win.destroy())
        win.update_idletasks()
        try:
            w = win.winfo_reqwidth()
            h = win.winfo_reqheight()
            x = (screen_w - w) // 2
            y = max(20, (screen_h - h) // 2)
            win.geometry(f"+{x}+{y}")
        except Exception:
            pass
        win.deiconify()

    def _show_batch_summary(self, complete: int, errors: int,
                            cancelled: int, elapsed: str,
                            quality_summary: Optional[dict] = None):
        """Themed summary modal shown when a batch finishes."""
        total = complete + errors + cancelled
        is_clean = errors == 0 and cancelled == 0

        dialog = tk.Toplevel(self.root)
        dialog.withdraw()
        dialog.title("Batch finished")
        dialog.configure(bg=Theme.BG_OVERLAY)
        dialog.resizable(False, False)
        dialog.transient(self.root)

        outer = tk.Frame(dialog, bg=Theme.BORDER, padx=1, pady=1)
        outer.pack()
        body = tk.Frame(outer, bg=Theme.BG_SECONDARY)
        body.pack()

        content = tk.Frame(body, bg=Theme.BG_SECONDARY)
        content.pack(padx=32, pady=(26, 16))

        title_text = "Batch finished" if is_clean else "Batch finished with issues"
        title_color = Theme.SUCCESS if is_clean else Theme.WARNING
        tk.Label(content, text=title_text, font=f(Theme.F_HEADING, "bold"),
                 bg=Theme.BG_SECONDARY, fg=title_color).pack(anchor="w")
        if elapsed:
            tk.Label(content, text=f"Total time {elapsed}  -  {total} item"
                                   f"{'s' if total != 1 else ''} processed",
                     font=f(Theme.F_BODY_SM),
                     bg=Theme.BG_SECONDARY, fg=Theme.TEXT_MUTED).pack(
                         anchor="w", pady=(2, 0))
        summary_note = ("Outputs are ready to review."
                        if is_clean else
                        "Completed outputs are ready. Review the outliers or open the log for details.")
        tk.Label(content, text=summary_note, font=f(Theme.F_BODY_SM),
                 bg=Theme.BG_SECONDARY, fg=Theme.TEXT_SECONDARY,
                 wraplength=420, justify="left").pack(anchor="w", pady=(Theme.S_SM, 0))

        # Stat row (compact pills)
        stats = tk.Frame(content, bg=Theme.BG_SECONDARY)
        stats.pack(anchor="w", pady=(Theme.S_LG, 0))

        def stat(parent, label, count, fg, bg):
            p = tk.Frame(parent, bg=bg, highlightthickness=1,
                         highlightbackground=Theme.BORDER_SUBTLE)
            tk.Label(
                p,
                text=str(count),
                font=f(Theme.F_HEADING, "bold"),
                bg=bg,
                fg=fg,
                padx=18,
                pady=0,
            ).pack(pady=(10, 0))
            tk.Label(
                p,
                text=label,
                font=f(Theme.F_META, "bold"),
                bg=bg,
                fg=Theme.TEXT_MUTED,
                padx=18,
                pady=0,
            ).pack(pady=(0, 10))
            return p

        stat(stats, "COMPLETED", complete, Theme.SUCCESS, Theme.SUCCESS_BG).pack(
            side="left")
        stat(stats, "FAILED", errors, Theme.ERROR, Theme.ERROR_BG).pack(
            side="left", padx=(Theme.S_SM, 0))
        stat(stats, "STOPPED", cancelled, Theme.WARNING, Theme.WARNING_BG).pack(
            side="left", padx=(Theme.S_SM, 0))

        if quality_summary:
            quality_card = tk.Frame(content, bg=Theme.BG_CARD, highlightthickness=1,
                                    highlightbackground=Theme.BORDER_SUBTLE)
            quality_card.pack(fill="x", pady=(Theme.S_LG, 0))

            tk.Label(
                quality_card,
                text="Sampled quality check",
                font=f(Theme.F_BODY_SM, "bold"),
                bg=Theme.BG_CARD,
                fg=Theme.TEXT_PRIMARY,
            ).pack(anchor="w", padx=Theme.S_LG, pady=(Theme.S_MD, 0))

            items_measured = int(quality_summary.get("items", 0) or 0)
            samples = int(quality_summary.get("samples", 0) or 0)
            tk.Label(
                quality_card,
                text=(
                    f"Measured {items_measured} completed item"
                    f"{'s' if items_measured != 1 else ''} across {samples} sampled frame"
                    f"{'s' if samples != 1 else ''}. Higher is generally better."
                ),
                font=f(Theme.F_META),
                bg=Theme.BG_CARD,
                fg=Theme.TEXT_MUTED,
                wraplength=420,
                justify="left",
            ).pack(anchor="w", padx=Theme.S_LG, pady=(4, Theme.S_MD))

            metrics = tk.Frame(quality_card, bg=Theme.BG_CARD)
            metrics.pack(anchor="w", padx=Theme.S_LG, pady=(0, Theme.S_MD))

            stat(metrics, "AVG PSNR", f"{quality_summary['psnr']:.2f} dB",
                 Theme.INFO, Theme.INFO_BG).pack(side="left")
            stat(metrics, "AVG SSIM", f"{quality_summary['ssim']:.4f}",
                 Theme.SUCCESS, Theme.SUCCESS_BG).pack(side="left", padx=(Theme.S_SM, 0))

        # Actions row
        actions = tk.Frame(body, bg=Theme.BG_CARD)
        actions.pack(fill="x")
        actions_inner = tk.Frame(actions, bg=Theme.BG_CARD)
        actions_inner.pack(side="right", padx=16, pady=14)

        def _close():
            dialog.grab_release()
            dialog.destroy()

        def _open_output_and_close():
            self._open_output_folder()
            _close()

        def _retry_failed_and_close():
            self._retry_failed()
            _close()

        if complete > 0:
            ModernButton(actions_inner, text="Open output", width=132,
                         command=_open_output_and_close,
                         style="accent", size="md", icon="^").pack(side="left")
        report_paths = getattr(self, "_last_batch_report_paths", [])
        if report_paths:
            def _open_report_and_close():
                import subprocess as _sp
                for rp in report_paths:
                    if str(rp).endswith(".md") and Path(rp).exists():
                        _sp.Popen(["cmd", "/c", "start", "", str(rp)],
                                  creationflags=0x08000000)
                        break
                else:
                    for rp in report_paths:
                        if Path(rp).exists():
                            _sp.Popen(["cmd", "/c", "start", "", str(rp)],
                                      creationflags=0x08000000)
                            break
                _close()
            ModernButton(actions_inner, text="Open report", width=116,
                         command=_open_report_and_close,
                         style="ghost", size="md").pack(side="left", padx=(Theme.S_SM, 0))
        if errors > 0:
            ModernButton(actions_inner, text="Open log", width=104,
                         command=self._open_log_file,
                         style="ghost", size="md").pack(side="left", padx=(Theme.S_SM, 0))
        if errors > 0 or cancelled > 0:
            ModernButton(actions_inner, text="Retry failed", width=110,
                         command=_retry_failed_and_close,
                         style="ghost", size="md").pack(side="left", padx=(Theme.S_SM, 0))
        ModernButton(actions_inner, text="Close", width=92,
                     command=_close, style="primary", size="md").pack(
                         side="left", padx=(Theme.S_SM, 0))

        dialog.bind("<Escape>", lambda e: _close())
        dialog.bind("<Return>", lambda e: _close())
        dialog.protocol("WM_DELETE_WINDOW", _close)

        dialog.update_idletasks()
        try:
            px, py = self.root.winfo_rootx(), self.root.winfo_rooty()
            pw, ph = self.root.winfo_width(), self.root.winfo_height()
            dw, dh = dialog.winfo_reqwidth(), dialog.winfo_reqheight()
            dialog.geometry(f"+{px + (pw - dw) // 2}+{py + (ph - dh) // 3}")
        except Exception:
            pass
        dialog.deiconify()
        dialog.grab_set()

    def _show_about(self):
        """Open a themed About dialog with version, credits, and quick links."""
        dialog = tk.Toplevel(self.root)
        dialog.withdraw()
        dialog.title(f"About {APP_NAME}")
        dialog.configure(bg=Theme.BG_OVERLAY)
        dialog.resizable(False, False)
        dialog.transient(self.root)

        outer = tk.Frame(dialog, bg=Theme.BORDER, padx=1, pady=1)
        outer.pack()
        body = tk.Frame(outer, bg=Theme.BG_SECONDARY)
        body.pack()

        content = tk.Frame(body, bg=Theme.BG_SECONDARY)
        content.pack(padx=32, pady=(28, 14))

        # Brand row
        brand_row = tk.Frame(content, bg=Theme.BG_SECONDARY)
        brand_row.pack(anchor="w")
        if self._brand_photo:
            tk.Label(brand_row, image=self._brand_photo,
                     bg=Theme.BG_SECONDARY).pack(side="left", padx=(0, Theme.S_MD))
        title_stack = tk.Frame(brand_row, bg=Theme.BG_SECONDARY)
        title_stack.pack(side="left")
        tk.Label(title_stack, text=APP_NAME, font=f(Theme.F_HEADING, "bold"),
                 bg=Theme.BG_SECONDARY, fg=Theme.TEXT_PRIMARY).pack(anchor="w")
        tk.Label(title_stack, text=f"Version {APP_VERSION}",
                 font=f(Theme.F_BODY_SM),
                 bg=Theme.BG_SECONDARY, fg=Theme.TEXT_MUTED).pack(anchor="w", pady=(2, 0))

        # Fact rows
        fact_card = tk.Frame(content, bg=Theme.BG_CARD, highlightthickness=1,
                             highlightbackground=Theme.BORDER_SUBTLE)
        fact_card.pack(fill="x", pady=(Theme.S_LG, 0))

        def fact(label, value, tone=Theme.TEXT_PRIMARY):
            row = tk.Frame(fact_card, bg=Theme.BG_CARD)
            row.pack(fill="x", padx=14, pady=6)
            tk.Label(row, text=label, font=f(Theme.F_BODY_SM),
                     bg=Theme.BG_CARD, fg=Theme.TEXT_MUTED).pack(side="left")
            tk.Label(row, text=value, font=f(Theme.F_BODY_SM, "bold"),
                     bg=Theme.BG_CARD, fg=tone).pack(side="right")

        det_label = ", ".join(self.ai_engines["detection"]) or "None"
        inp_label = ", ".join(self.ai_engines["inpainting"]) or "None"
        gpu_count = len(self.gpus)
        gpu_label = f"{gpu_count} GPU{'s' if gpu_count != 1 else ''}" if self.gpus else "CPU only"

        fact("Detection engines", det_label, Theme.INFO)
        fact("Inpainting engines", inp_label, Theme.SUCCESS)
        fact("Compute", gpu_label,
             Theme.SUCCESS if self.gpus else Theme.WARNING)
        fact("FFmpeg", "Ready" if self.ffmpeg_ready else "Missing",
             Theme.SUCCESS if self.ffmpeg_ready else Theme.WARNING)
        fact("Input", "Click Import or drag files onto the queue")
        fact("Settings", str(SETTINGS_FILE))
        fact("Log file", str(LOG_FILE))

        # Action row
        actions = tk.Frame(body, bg=Theme.BG_CARD)
        actions.pack(fill="x")
        actions_inner = tk.Frame(actions, bg=Theme.BG_CARD)
        actions_inner.pack(side="right", padx=16, pady=14)

        ModernButton(actions_inner, text="Open log", width=110,
                     command=self._open_log_file, style="ghost", size="md").pack(side="left")
        ModernButton(actions_inner, text="Settings folder", width=140,
                     command=self._open_settings_folder, style="ghost",
                     size="md").pack(side="left", padx=(Theme.S_SM, 0))
        ModernButton(actions_inner, text="Close", width=90,
                     command=dialog.destroy,
                     style="primary", size="md").pack(side="left", padx=(Theme.S_SM, 0))

        dialog.bind("<Escape>", lambda e: dialog.destroy())

        dialog.update_idletasks()
        try:
            px, py = self.root.winfo_rootx(), self.root.winfo_rooty()
            pw, ph = self.root.winfo_width(), self.root.winfo_height()
            dw, dh = dialog.winfo_reqwidth(), dialog.winfo_reqheight()
            dialog.geometry(f"+{px + (pw - dw) // 2}+{py + (ph - dh) // 3}")
        except Exception:
            pass
        dialog.deiconify()
        dialog.grab_set()

    def _set_lang_display(self, code: str):
        """Sync the friendly-name label to the underlying lang code."""
        for label, (c, _) in zip(self._lang_labels, self._lang_display):
            if c == code:
                self._lang_display_var.set(label)
                return
        # Unknown -- default to English
        self._lang_display_var.set(self._lang_labels[0])
        self.lang_var.set(self._lang_display[0][0])

    def _on_lang_changed(self, event=None):
        """Map selected friendly label back to the lang code."""
        label = self._lang_display_var.get()
        code = self._lang_by_label.get(label)
        if code:
            self.lang_var.set(code)
            self.config.detection_lang = code

    def _on_gpu_changed(self, event=None):
        """Handle GPU device selection change."""
        selection = self.gpu_var.get()
        for i, gpu in enumerate(self.gpus):
            label = f"{gpu['name']} ({gpu['memory']})"
            if label == selection:
                self.config.gpu_id = gpu['index']
                self.config.use_gpu = True
                self._update_status(f"Compute device set to {gpu['name']}", "info")
                logger.info(f"GPU set to: {gpu['name']} (index {gpu['index']})")
                break

    def _choose_output_dir(self):
        """Let user pick a custom output directory."""
        d = filedialog.askdirectory(title="Select Output Directory")
        if d:
            self._output_dir = Path(d)
            self._update_output_label()
            refreshed = self._refresh_idle_output_paths()
            if refreshed:
                self._update_queue_display()
            message = "Custom output location selected"
            if refreshed:
                message += f". Updated {refreshed} idle output path{'s' if refreshed != 1 else ''}"
            self._update_status(message, "success")
            logger.info(f"Output directory: {self._output_dir}")

    def _reset_output_dir(self):
        """Reset output directory to default (input_dir/output/)."""
        self._output_dir = None
        self._update_output_label()
        refreshed = self._refresh_idle_output_paths()
        if refreshed:
            self._update_queue_display()
        message = "Output location reset to the default per-folder workflow"
        if refreshed:
            message += f". Updated {refreshed} idle output path{'s' if refreshed != 1 else ''}"
        self._update_status(message)

    def _open_region_selector(self):
        """Open a region-selector window with frame scrubbing (F-1) and
        multi-rectangle drawing (F-2).

        Drag = primary rect (or new rect when "Add another" was clicked).
        The frame slider re-loads the chosen frame so users can target a
        non-zero timecode for clips that open on a black intro card. The
        backend already accepts a `subtitle_areas` list; once the user
        clicks "Save" we write a single rect to `subtitle_area` (back-
        compat) AND the full rect list to `subtitle_areas`.
        """
        source_path = None
        selected = self._get_selected_queue_item()
        if selected:
            source_path = selected.file_path
        else:
            for item in self.queue:
                source_path = item.file_path
                break

        if not source_path:
            source_path = filedialog.askopenfilename(
                title="Select a video/image to define subtitle region",
                filetypes=[("All Supported", filepicker_pattern(SUPPORTED_EXTENSIONS))]
            )
        if not source_path:
            return
        if not PIL_AVAILABLE:
            self._update_status("Pillow required for region selector")
            return

        import cv2 as _cv2

        is_video = is_video_file(source_path)
        cap = _cv2.VideoCapture(source_path) if is_video else None
        if is_video and not cap.isOpened():
            logger.error("Could not open video for region selection")
            cap.release()
            return
        try:
            if is_video:
                frame_count = int(cap.get(_cv2.CAP_PROP_FRAME_COUNT)) or 1
                fps = cap.get(_cv2.CAP_PROP_FPS) or 30.0
                if fps <= 0:
                    fps = 30.0
                ret, frame = cap.read()
                if not ret:
                    # Early return bypasses both the except-release below
                    # and the <Destroy> binding (not installed yet) --
                    # release here or the capture leaks.
                    logger.error("Could not read video frame for region selection")
                    cap.release()
                    return
            else:
                frame = _cv2.imread(source_path)
                if frame is None:
                    logger.error("Could not read image for region selection")
                    return
                frame_count, fps = 1, 30.0

            orig_h, orig_w = frame.shape[:2]
            screen_w = self.root.winfo_screenwidth()
            screen_h = self.root.winfo_screenheight()
            max_w = min(900, int(screen_w * 0.8))
            max_h = min(540, int(screen_h * 0.7))
            scale = min(max_w / orig_w, max_h / orig_h, 1.0)
            disp_w, disp_h = int(orig_w * scale), int(orig_h * scale)

            # Selector state: every saved rect lives in `rects` (image-
            # space coordinates). Drag creates the current rect; on
            # release it joins the list. Adding another rect re-arms the
            # canvas for the next drag.
            rects: List[Tuple[int, int, int, int]] = []
            preload = self.config.subtitle_areas or (
                [self.config.subtitle_area] if self.config.subtitle_area else []
            )
            rects.extend([tuple(r) for r in preload if r])

            win = tk.Toplevel(self.root)
            win.title("Choose subtitle region")
            win.configure(bg=Theme.BG_OVERLAY)
            win.resizable(False, False)

            canvas = tk.Canvas(win, width=disp_w, height=disp_h, highlightthickness=0,
                               bg=Theme.BG_DARK, cursor="cross")
            canvas.pack()
            canvas_image_id = canvas.create_image(0, 0, anchor="nw")
            canvas._photo = None

            rect_ids: List[int] = []
            drag_id = [None]
            drag_start = [0, 0]

            def _frame_to_pil(bgr):
                rgb = _cv2.cvtColor(bgr, _cv2.COLOR_BGR2RGB)
                return Image.fromarray(rgb).resize((disp_w, disp_h), Image.LANCZOS)

            def _draw_image(bgr):
                pil = _frame_to_pil(bgr)
                canvas._photo = ImageTk.PhotoImage(pil)
                canvas.itemconfig(canvas_image_id, image=canvas._photo)

            def _draw_saved_rects():
                for rid in rect_ids:
                    canvas.delete(rid)
                rect_ids.clear()
                for (x1, y1, x2, y2) in rects:
                    rect_ids.append(canvas.create_rectangle(
                        x1 * scale, y1 * scale, x2 * scale, y2 * scale,
                        outline=Theme.GREEN_PRIMARY, width=2,
                        stipple="gray25", fill=Theme.GREEN_PRIMARY,
                    ))

            def _seek_video(frame_idx: int):
                if not is_video or cap is None:
                    return
                frame_idx = max(0, min(frame_count - 1, int(frame_idx)))
                cap.set(_cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ok, f = cap.read()
                if ok:
                    _draw_image(f)
                    _draw_saved_rects()

            _draw_image(frame)
            _draw_saved_rects()

            # Drawing handlers.
            def on_press(event):
                drag_start[0], drag_start[1] = event.x, event.y
                if drag_id[0]:
                    canvas.delete(drag_id[0])
                drag_id[0] = canvas.create_rectangle(
                    event.x, event.y, event.x, event.y,
                    outline=Theme.BLUE_PRIMARY, width=2,
                    stipple="gray12", fill=Theme.BLUE_PRIMARY,
                )

            def on_drag(event):
                if drag_id[0]:
                    canvas.coords(drag_id[0], drag_start[0], drag_start[1], event.x, event.y)

            def on_release(event):
                x1 = int(min(drag_start[0], event.x) / scale)
                y1 = int(min(drag_start[1], event.y) / scale)
                x2 = int(max(drag_start[0], event.x) / scale)
                y2 = int(max(drag_start[1], event.y) / scale)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(orig_w, x2), min(orig_h, y2)
                if (x2 - x1) > 10 and (y2 - y1) > 5:
                    rects.append((x1, y1, x2, y2))
                if drag_id[0] is not None:
                    canvas.delete(drag_id[0])
                    drag_id[0] = None
                _draw_saved_rects()

            canvas.bind("<ButtonPress-1>", on_press)
            canvas.bind("<B1-Motion>", on_drag)
            canvas.bind("<ButtonRelease-1>", on_release)

            # Frame slider for videos.
            if is_video and frame_count > 1:
                slider_row = tk.Frame(win, bg=Theme.BG_OVERLAY)
                slider_row.pack(fill="x", padx=Theme.S_MD, pady=(Theme.S_SM, 0))
                tk.Label(slider_row, text="Frame",
                         font=f(Theme.F_BODY_SM),
                         bg=Theme.BG_OVERLAY, fg=Theme.TEXT_SECONDARY).pack(side="left")
                ts_label = tk.Label(slider_row, text="00:00:00",
                                    font=f(Theme.F_META),
                                    bg=Theme.BG_OVERLAY, fg=Theme.TEXT_MUTED)
                ts_label.pack(side="right")

                def _on_slider(value):
                    try:
                        idx = int(float(value))
                    except (TypeError, ValueError):
                        return
                    _seek_video(idx)
                    secs = idx / fps
                    hh = int(secs // 3600); mm = int((secs % 3600) // 60); ss = int(secs % 60)
                    ts_label.config(text=f"{hh:02d}:{mm:02d}:{ss:02d}")

                slider = tk.Scale(
                    win, from_=0, to=frame_count - 1, orient="horizontal",
                    command=_on_slider, length=disp_w - 24,
                    bg=Theme.BG_OVERLAY, fg=Theme.TEXT_PRIMARY,
                    troughcolor=Theme.BG_TERTIARY,
                    activebackground=Theme.BLUE_PRIMARY,
                    highlightthickness=0, showvalue=False,
                )
                slider.pack(fill="x", padx=Theme.S_MD, pady=(0, Theme.S_SM))

            # Action row: Add another, Clear all, Save.
            actions = tk.Frame(win, bg=Theme.BG_OVERLAY)
            actions.pack(fill="x", padx=Theme.S_MD, pady=(Theme.S_SM, Theme.S_MD))

            def _clear_all():
                rects.clear()
                _draw_saved_rects()

            def _save_and_close():
                # Single-rect back-compat field stores the union or the
                # first rect; multi-rect field carries the full list.
                if rects:
                    self.config.subtitle_areas = [tuple(r) for r in rects]
                    self.config.subtitle_area = rects[0]
                    self._update_status(
                        f"Saved {len(rects)} subtitle region{'s' if len(rects) != 1 else ''}",
                        "success",
                    )
                else:
                    self.config.subtitle_areas = None
                    self.config.subtitle_area = None
                    self._update_status("Cleared manual subtitle regions", "info")
                self._update_region_label_display()
                win.destroy()

            ModernButton(actions, text="Clear all", command=_clear_all,
                         style="ghost", size="sm", width=92).pack(side="left")
            ModernButton(actions, text="Save", command=_save_and_close,
                         style="primary", size="sm", width=92).pack(
                             side="right")
            ModernButton(actions, text="Cancel", command=win.destroy,
                         style="ghost", size="sm", width=92).pack(
                             side="right", padx=(0, Theme.S_SM))

            hint_frame = tk.Frame(win, bg=Theme.BG_OVERLAY)
            hint_frame.pack(fill="x", pady=(0, Theme.S_MD))
            tk.Label(hint_frame,
                     text="Drag to add a region. Drag again for multi-region.",
                     font=f(Theme.F_BODY_SM, "bold"),
                     bg=Theme.BG_OVERLAY, fg=Theme.TEXT_PRIMARY).pack()
            tk.Label(hint_frame,
                     text="Scrub the slider to pick a frame where the subtitle is visible.",
                     font=f(Theme.F_META),
                     bg=Theme.BG_OVERLAY, fg=Theme.TEXT_MUTED).pack(pady=(2, 0))

            # The cap must outlive this function so slider scrubs work
            # while the (non-blocking) modal stays open. Release it when
            # the window is closed instead of at function return.
            def _release_cap():
                if cap is not None:
                    try:
                        cap.release()
                    except Exception:
                        pass

            win.bind("<Destroy>", lambda e: _release_cap() if e.widget is win else None)
            win.bind("<Escape>", lambda e: win.destroy())
            win.transient(self.root)
            win.grab_set()
        except Exception as e:
            logger.error(f"Region selector error: {e}")
            if cap is not None:
                try:
                    cap.release()
                except Exception:
                    pass

    def _reset_region(self):
        """Reset subtitle region to auto-detect. Clears BOTH region
        fields -- the backend honours subtitle_areas as fixed masks, so
        leaving it set would keep using the old rectangles while the UI
        claims automatic detection."""
        self.config.subtitle_area = None
        self.config.subtitle_areas = None
        self._update_region_label_display()
        self._update_status("Subtitle detection returned to automatic mode")

    @staticmethod
    def _safe_float(value: str, default: float = 0.0) -> float:
        """Parse a float from a string, returning default on failure."""
        try:
            return float(value or default)
        except (ValueError, TypeError):
            return default

    @staticmethod
    def _gui_to_backend_mode(gui_mode_value: str):
        """Map a GUI InpaintMode value onto the backend enum. The two
        enums are deliberately separate (see CLAUDE.md) -- this mapping
        is the single place they meet."""
        from backend.processor import InpaintMode as _BM
        return {
            "Auto": _BM.AUTO,
            "STTN": _BM.STTN,
            "LAMA": _BM.LAMA,
            "ProPainter": _BM.PROPAINTER,
        }.get(gui_mode_value, _BM.STTN)

    def _gui_to_backend_device(self, use_gpu: bool, gpu_id: int) -> str:
        """DirectML GPUs map to device='directml' (ONNX Runtime DirectML
        EP, not torch-directml); CUDA GPUs map to 'cuda:N'."""
        if not use_gpu:
            return "cpu"
        for g in self.gpus:
            if g['index'] == gpu_id and g.get('type') == "DirectML":
                return "directml"
        return f"cuda:{gpu_id}"

    @staticmethod
    def _normalized_path_key(path: str | Path) -> str:
        """Return a case-folded absolute path for reliable Windows comparisons."""
        try:
            return str(Path(path).resolve(strict=False)).casefold()
        except TypeError:
            return str(Path(path).resolve()).casefold()
        except OSError:
            return str(Path(path).absolute()).casefold()

    @staticmethod
    def _new_import_stats() -> dict:
        return {
            "added": 0,
            "duplicate": 0,
            "missing": 0,
            "unsupported": 0,
            "queue_full": 0,
            "folders": 0,
            "supported_in_folders": 0,
        }

    def _merge_import_stats(self, base: dict, extra: dict):
        for key, value in extra.items():
            base[key] = base.get(key, 0) + value

    def _occupied_output_paths(self, exclude_item_id: Optional[str] = None) -> set[str]:
        with self.queue_lock:
            return {
                self._normalized_path_key(item.output_path)
                for item in self.queue
                if item.id != exclude_item_id
            }

    def _make_unique_output_path(self, desired: Path,
                                 exclude_item_id: Optional[str] = None) -> Path:
        """Avoid overwriting existing files or reserved queue outputs."""
        occupied = self._occupied_output_paths(exclude_item_id=exclude_item_id)
        candidate = desired
        counter = 2
        while candidate.exists() or self._normalized_path_key(candidate) in occupied:
            candidate = desired.with_name(f"{desired.stem}({counter}){desired.suffix}")
            counter += 1
        return candidate

    def _suggest_output_path(self, file_path: str, *,
                             output_dir: Optional[Path] = None,
                             exclude_item_id: Optional[str] = None) -> Path:
        input_path = Path(file_path)
        target_dir = output_dir if output_dir is not None else (
            self._output_dir or (input_path.parent / "output")
        )
        desired = target_dir / f"{input_path.stem}_no_sub{input_path.suffix}"
        return self._make_unique_output_path(desired, exclude_item_id=exclude_item_id)

    def _probe_soft_subtitle_records(self, path: str) -> List[dict]:
        from backend.io import _probe_subtitle_streams
        return [
            _soft_subtitle_stream_record(stream)
            for stream in _probe_subtitle_streams(path)
        ]

    def _apply_soft_subtitle_probe_records(self, item_id: str,
                                           records: List[dict]) -> None:
        with self.queue_lock:
            target = next((it for it in self.queue if it.id == item_id), None)
        if target is None:
            return
        target.soft_subtitle_streams = records
        target.soft_subtitle_probe_done = True
        if target.status == ProcessingStatus.IDLE:
            if records:
                target.message = (
                    "Embedded subtitle tracks found. Right-click for "
                    "fast strip/keep, or run burned-in cleanup."
                )
            elif target.message == "Checking embedded subtitle tracks...":
                target.message = "Ready to process"
        if target.id in self.queue_widgets:
            self.queue_widgets[target.id].update_item(target)
        if self._selected_queue_item_id == target.id:
            self.preview_meta_label.config(
                text=(
                    _format_soft_subtitle_summary(records)
                    if records else
                    "Review mask to confirm the subtitle band, then start the batch when the framing looks right."
                )
            )

    def _start_soft_subtitle_probe(self, item: QueueItem) -> None:
        """Probe embedded subtitle streams off the Tk thread."""
        if not is_video_file(item.file_path):
            item.soft_subtitle_probe_done = True
            return
        item.soft_subtitle_probe_done = False
        if item.status == ProcessingStatus.IDLE:
            item.message = "Checking embedded subtitle tracks..."

        def _worker(item_id: str, path: str):
            records: List[dict] = []
            try:
                records = self._probe_soft_subtitle_records(path)
            except Exception as exc:
                logger.debug(f"Soft-subtitle probe failed for {path}: {exc}")

            try:
                self.root.after(0, self._apply_soft_subtitle_probe_records,
                                item_id, records)
            except RuntimeError:
                pass

        threading.Thread(
            target=_worker,
            args=(item.id, item.file_path),
            name="vsr-soft-subtitle-probe",
            daemon=True,
        ).start()

    def _set_soft_subtitle_action(self, item_id: str, action: str) -> None:
        labels = {
            "strip": "Fast strip embedded subtitles",
            "keep_all": "Fast remux and keep embedded subtitles",
            "burned_in": "Burned-in cleanup",
        }
        if action not in labels:
            return
        with self.queue_lock:
            item = next((it for it in self.queue if it.id == item_id), None)
        if item is None or item.status != ProcessingStatus.IDLE:
            self._update_status("Only idle queue items can change subtitle action", "warning")
            return
        item.soft_subtitle_action = action
        if action == "burned_in":
            item.message = "Ready for burned-in cleanup"
        else:
            item.message = labels[action]
        if not item.output_path_locked:
            item.output_path = str(self._suggest_output_path(
                item.file_path,
                exclude_item_id=item.id,
            ))
        if item.id in self.queue_widgets:
            self.queue_widgets[item.id].update_item(item)
        self._update_status(f"{Path(item.file_path).name}: {labels[action]}", "info")

    def _refresh_idle_output_paths(self) -> int:
        """Recompute output paths for idle items that still follow the live output rule."""
        refreshed = 0
        with self.queue_lock:
            idle_items = [
                item for item in self.queue
                if item.status == ProcessingStatus.IDLE and not item.output_path_locked
            ]
        for item in idle_items:
            new_path = self._suggest_output_path(item.file_path, exclude_item_id=item.id)
            if self._normalized_path_key(item.output_path) != self._normalized_path_key(new_path):
                item.output_path = str(new_path)
                refreshed += 1
        return refreshed

    def _announce_import_summary(self, stats: dict):
        """Surface one calm import summary instead of a burst of per-file notices."""
        added = stats.get("added", 0)
        duplicate = stats.get("duplicate", 0)
        missing = stats.get("missing", 0)
        unsupported = stats.get("unsupported", 0)
        queue_full = stats.get("queue_full", 0)
        folders = stats.get("folders", 0)
        supported_in_folders = stats.get("supported_in_folders", 0)

        if added > 0:
            parts = [f"Added {added} item{'s' if added != 1 else ''} to the queue"]
            if duplicate:
                parts.append(f"skipped {duplicate} duplicate{'s' if duplicate != 1 else ''}")
            if queue_full:
                parts.append("queue reached the 500-item limit")
            detail = ". ".join(parts)
            self._update_status(detail, "success")
            logger.info(detail)
            return

        if queue_full:
            self._update_status("The queue is already full (500 items max)", "warning")
            logger.warning("Queue full while importing items")
            return

        if folders and supported_in_folders == 0:
            self._update_status("No supported videos or images were found in the selected folder", "warning")
            logger.warning("No supported files found while importing folder selection")
            return

        if duplicate and not (missing or unsupported):
            self._update_status("Everything selected is already in the queue", "info")
            logger.info("Import skipped because every selected item was already queued")
            return

        if unsupported and not (duplicate or missing):
            self._update_status("Only supported video and image formats can be queued", "warning")
            logger.warning("Import skipped because the selection only contained unsupported files")
            return

        if missing and not (duplicate or unsupported):
            self._update_status("Some selected files could not be found", "warning")
            logger.warning("Import skipped because selected files were missing")
            return

        self._update_status("Nothing new was added to the queue", "warning")
        logger.warning("Import completed without adding new queue items")

    def _on_files_dropped(self, files: List[str]):
        """Handle dropped files."""
        stats = self._new_import_stats()
        for file_path in files:
            if Path(file_path).is_dir():
                self._merge_import_stats(stats, self._add_folder_to_queue(file_path))
            else:
                result = self._add_to_queue(file_path)
                stats[result] = stats.get(result, 0) + 1
        self._announce_import_summary(stats)

    def _add_folder_to_queue(self, folder_path: str):
        """Recursively add all supported files from a folder."""
        folder = Path(folder_path)
        stats = self._new_import_stats()
        stats["folders"] = 1
        for f in sorted(folder.rglob("*")):
            if f.is_file() and (is_video_file(str(f)) or is_image_file(str(f))):
                stats["supported_in_folders"] += 1
                result = self._add_to_queue(str(f))
                stats[result] = stats.get(result, 0) + 1
                if result == "queue_full":
                    break
        return stats

    def _add_to_queue(self, file_path: str):
        """Add a file to the processing queue."""
        # Check file exists and is valid
        if not Path(file_path).is_file():
            logger.warning(f"File not found: {file_path}")
            return "missing"
        if not (is_video_file(file_path) or is_image_file(file_path)):
            logger.warning(f"Unsupported file type: {file_path}")
            return "unsupported"

        # Queue size limit
        if len(self.queue) >= 500:
            logger.warning("Queue full (500 items max)")
            return "queue_full"

        # Prevent duplicate files in queue
        normalized = self._normalized_path_key(file_path)
        with self.queue_lock:
            for existing in self.queue:
                if self._normalized_path_key(existing.file_path) == normalized:
                    logger.info(f"Already in queue: {Path(file_path).name}")
                    return "duplicate"

        # Generate a collision-proof unique ID for this queue slot
        item_id = uuid.uuid4().hex

        # Generate an output path that stays unique against both disk and the
        # rest of the queued items.
        output_path = self._suggest_output_path(file_path)

        # Create config copy from the latest UI state.
        config = self._make_processing_snapshot()

        # Create queue item
        initial_message = (
            "Checking embedded subtitle tracks..."
            if is_video_file(file_path) else
            "Ready to process"
        )
        item = QueueItem(
            id=item_id,
            file_path=file_path,
            output_path=str(output_path),
            output_path_locked=False,
            config=config,
            message=initial_message
        )

        with self.queue_lock:
            self.queue.append(item)
        self._update_queue_display()
        if is_video_file(file_path):
            self._start_soft_subtitle_probe(item)
        if len(self.queue) == 1 and not self.is_processing:
            self._show_preview(item)
        logger.info(f"Queued: {Path(file_path).name} ({get_file_info(file_path)})")
        return "added"

    def _open_sort_menu(self):
        """Pop up a themed sort menu anchored to the sort button."""
        if self.is_processing:
            self._update_status(
                "Sorting is disabled while a batch is running", "warning")
            return
        menu = make_themed_menu(self.root)
        menu.add_command(label="Filename (A -> Z)",
                         command=lambda: self._sort_queue("name_asc"))
        menu.add_command(label="Filename (Z -> A)",
                         command=lambda: self._sort_queue("name_desc"))
        menu.add_separator()
        menu.add_command(label="File size (largest first)",
                         command=lambda: self._sort_queue("size_desc"))
        menu.add_command(label="File size (smallest first)",
                         command=lambda: self._sort_queue("size_asc"))
        menu.add_separator()
        menu.add_command(label="Status (pending first)",
                         command=lambda: self._sort_queue("status"))
        menu.add_command(label="Reverse current order",
                         command=lambda: self._sort_queue("reverse"))
        try:
            bx = self._sort_btn.winfo_rootx()
            by = self._sort_btn.winfo_rooty() + self._sort_btn.winfo_height() + 2
            menu.tk_popup(bx, by)
        finally:
            menu.grab_release()

    def _sort_queue(self, strategy: str):
        """Reorder queue items by the chosen strategy and re-render."""
        if self.is_processing:
            return
        key_map = {
            "name_asc": lambda it: Path(it.file_path).name.lower(),
            "name_desc": lambda it: Path(it.file_path).name.lower(),
            "size_asc": lambda it: self._safe_size(it.file_path),
            "size_desc": lambda it: self._safe_size(it.file_path),
            "status": lambda it: {
                ProcessingStatus.IDLE: 0,
                ProcessingStatus.LOADING: 1,
                ProcessingStatus.DETECTING: 2,
                ProcessingStatus.PROCESSING: 3,
                ProcessingStatus.MERGING: 4,
                ProcessingStatus.COMPLETE: 5,
                ProcessingStatus.CANCELLED: 6,
                ProcessingStatus.ERROR: 7,
            }.get(it.status, 99),
        }
        with self.queue_lock:
            if strategy == "reverse":
                self.queue.reverse()
            elif strategy in key_map:
                reverse = strategy.endswith("_desc")
                self.queue.sort(key=key_map[strategy], reverse=reverse)
        # Destroy all widgets so they get rebuilt in new order
        for wid, w in list(self.queue_widgets.items()):
            try:
                w.destroy()
            except Exception:
                pass
        self.queue_widgets.clear()
        self._update_queue_display()
        self._update_status("Queue sorted")

    @staticmethod
    def _safe_size(path: str) -> int:
        try:
            return Path(path).stat().st_size
        except OSError:
            return 0

    def _open_per_file_overrides(self, item_id: str):
        """RM-29: themed popover that edits a single queue item's
        ProcessingConfig without touching the global UI state.

        Only the most-asked fields are surfaced (mode, language,
        sensitivity, output codec). The rest of the config carries over
        from the snapshot taken when the item was queued.
        """
        item = next((it for it in self.queue if it.id == item_id), None)
        if item is None or item.status != ProcessingStatus.IDLE:
            return

        dialog = tk.Toplevel(self.root)
        dialog.title(f"Override settings: {Path(item.file_path).name}")
        dialog.configure(bg=Theme.BG_OVERLAY)
        dialog.resizable(False, False)
        dialog.transient(self.root)

        outer = tk.Frame(dialog, bg=Theme.BORDER, padx=1, pady=1)
        outer.pack()
        body = tk.Frame(outer, bg=Theme.BG_SECONDARY)
        body.pack()

        content = tk.Frame(body, bg=Theme.BG_SECONDARY)
        content.pack(padx=24, pady=(20, 12))

        tk.Label(content, text="Per-file overrides",
                 font=f(Theme.F_HEADING, "bold"),
                 bg=Theme.BG_SECONDARY, fg=Theme.TEXT_PRIMARY).pack(anchor="w")
        tk.Label(content,
                 text="These apply to this queued item only and survive a global settings change.",
                 font=f(Theme.F_BODY_SM),
                 bg=Theme.BG_SECONDARY, fg=Theme.TEXT_SECONDARY,
                 wraplength=380, justify="left").pack(anchor="w", pady=(2, Theme.S_LG))

        # Mode picker.
        mode_var = tk.StringVar(value=item.config.mode.value)
        tk.Label(content, text="Mode", font=f(Theme.F_BODY_SM),
                 bg=Theme.BG_SECONDARY, fg=Theme.TEXT_SECONDARY).pack(anchor="w")
        mode_picker = SegmentedPicker(
            content,
            options=[(m.value, m.value) for m in InpaintMode],
            value=mode_var.get(),
            command=lambda v: mode_var.set(v),
            bg=Theme.BG_SECONDARY,
        )
        mode_picker.pack(fill="x", pady=(2, Theme.S_MD))

        # Detection language.
        lang_row = tk.Frame(content, bg=Theme.BG_SECONDARY)
        lang_row.pack(fill="x", pady=(0, Theme.S_SM))
        tk.Label(lang_row, text="Subtitle language", font=f(Theme.F_BODY_SM),
                 bg=Theme.BG_SECONDARY, fg=Theme.TEXT_SECONDARY).pack(side="left")
        lang_codes = [code for code, _ in self._lang_display]
        lang_var = tk.StringVar(value=item.config.detection_lang)
        lang_combo = ttk.Combobox(
            lang_row, textvariable=lang_var, width=18,
            values=self._lang_labels,
            state="readonly", style="Dark.TCombobox", font=f(Theme.F_BODY_SM),
        )
        # Match the friendly label for the current code.
        for label, (code, _) in zip(self._lang_labels, self._lang_display):
            if code == lang_var.get():
                lang_combo.set(label)
                break
        lang_combo.pack(side="right")

        # Sensitivity slider (1-9 maps to 0.1-0.9).
        sens_row = tk.Frame(content, bg=Theme.BG_SECONDARY)
        sens_row.pack(fill="x", pady=(Theme.S_SM, Theme.S_SM))
        sens_var = tk.IntVar(value=int(round(item.config.detection_threshold * 100)))
        tk.Label(sens_row, text="Sensitivity", font=f(Theme.F_BODY_SM),
                 bg=Theme.BG_SECONDARY, fg=Theme.TEXT_SECONDARY).pack(side="left")
        sens_label = tk.Label(sens_row, text=f"{sens_var.get()}%",
                              font=f(Theme.F_BODY_SM, "bold"),
                              bg=Theme.BG_SECONDARY, fg=Theme.BLUE_PRIMARY)
        sens_label.pack(side="right")

        def _on_sens(value):
            try:
                sens_var.set(int(value))
                sens_label.config(text=f"{int(value)}%")
            except (TypeError, ValueError):
                pass

        sens_slider = tk.Scale(
            content, from_=10, to=90, orient="horizontal",
            command=_on_sens, showvalue=False, length=380,
            bg=Theme.BG_SECONDARY, fg=Theme.TEXT_PRIMARY,
            troughcolor=Theme.BG_TERTIARY,
            activebackground=Theme.BLUE_PRIMARY,
            highlightthickness=0,
        )
        sens_slider.set(sens_var.get())
        sens_slider.pack(fill="x", pady=(0, Theme.S_MD))

        # Output codec.
        codec_row = tk.Frame(content, bg=Theme.BG_SECONDARY)
        codec_row.pack(fill="x", pady=(0, Theme.S_SM))
        tk.Label(codec_row, text="Output codec", font=f(Theme.F_BODY_SM),
                 bg=Theme.BG_SECONDARY, fg=Theme.TEXT_SECONDARY).pack(side="left")
        codec_var = tk.StringVar(value=getattr(item.config, "output_codec", "h264"))
        ttk.Combobox(
            codec_row, textvariable=codec_var, width=8,
            values=["h264", "h265", "av1"],
            state="readonly", style="Dark.TCombobox", font=f(Theme.F_BODY_SM),
        ).pack(side="right")

        # Action buttons.
        actions = tk.Frame(body, bg=Theme.BG_CARD)
        actions.pack(fill="x")
        actions_inner = tk.Frame(actions, bg=Theme.BG_CARD)
        actions_inner.pack(side="right", padx=16, pady=14)

        def _save():
            try:
                item.config.mode = InpaintMode(mode_var.get())
            except ValueError:
                pass
            label = lang_combo.get()
            new_code = self._lang_by_label.get(label, item.config.detection_lang)
            item.config.detection_lang = new_code
            item.config.detection_threshold = sens_var.get() / 100.0
            item.config.output_codec = codec_var.get()
            item.config.normalized()
            if item.id in self.queue_widgets:
                self.queue_widgets[item.id].update_item(item)
            self._update_status(
                f"Overrides saved for {Path(item.file_path).name}",
                "success",
            )
            dialog.destroy()

        ModernButton(actions_inner, text="Cancel", command=dialog.destroy,
                     style="ghost", size="md", width=96).pack(side="left")
        ModernButton(actions_inner, text="Save", command=_save,
                     style="primary", size="md", width=96).pack(
                         side="left", padx=(Theme.S_SM, 0))

        dialog.bind("<Escape>", lambda e: dialog.destroy())
        dialog.transient(self.root)
        dialog.grab_set()
        dialog.update_idletasks()
        try:
            px, py = self.root.winfo_rootx(), self.root.winfo_rooty()
            pw, ph = self.root.winfo_width(), self.root.winfo_height()
            dw, dh = dialog.winfo_reqwidth(), dialog.winfo_reqheight()
            dialog.geometry(f"+{px + (pw - dw) // 2}+{py + (ph - dh) // 3}")
        except Exception:
            pass

    def _cancel_queue_item(self, item_id: str):
        """F-7: per-item cancellation. Sets the QueueItem's cancel flag;
        _process_item's progress callback raises InterruptedError next
        time it fires so the worker drops this file and moves on to the
        next one. The global cancel_event stays untouched so the rest
        of the batch survives."""
        with self.queue_lock:
            item = next((it for it in self.queue if it.id == item_id), None)
        if item is None:
            return
        if item.status not in (ProcessingStatus.LOADING,
                                ProcessingStatus.DETECTING,
                                ProcessingStatus.PROCESSING,
                                ProcessingStatus.MERGING):
            self._update_status("That item is not running", "warning")
            return
        item.cancel_requested = True
        self._update_status(
            f"Cancelling {Path(item.file_path).name}", "warning", toast=True,
        )

    def _repeat_item_with_settings(self, item_id: str):
        """RM-28: snapshot the named item's config and re-queue the same
        source file with a fresh IDLE entry. Lets users re-run a clip
        with exactly the per-file settings that already worked once,
        even after the global UI state has changed."""
        with self.queue_lock:
            template = next((it for it in self.queue if it.id == item_id), None)
        if template is None:
            self._update_status("Item not found in queue", "warning")
            return
        # Snapshot via to_dict / from_dict so any future field churn does
        # not need an explicit copy update here.
        snapshot = ProcessingConfig.from_dict(template.config.to_dict())
        # Build a fresh output path (unique vs the original).
        desired = self._suggest_output_path(template.file_path)
        new_item = QueueItem(
            id=str(uuid.uuid4()),
            file_path=template.file_path,
            output_path=str(desired),
            config=snapshot,
            output_path_locked=False,
            status=ProcessingStatus.IDLE,
            progress=0.0,
            message="Ready to process",
            soft_subtitle_streams=list(template.soft_subtitle_streams),
            soft_subtitle_probe_done=template.soft_subtitle_probe_done,
            soft_subtitle_action=template.soft_subtitle_action,
        )
        with self.queue_lock:
            self.queue.append(new_item)
        self._update_queue_display()
        self._refresh_action_states()
        self._update_status(
            f"Re-queued {Path(template.file_path).name} with the same settings",
            "info", toast=True,
        )

    def _rename_output_for(self, item_id: str):
        """Open a file picker to customize the output path of a queued item.

        Disabled for items that have already started processing.
        """
        item = next((i for i in self.queue if i.id == item_id), None)
        if not item:
            return
        if item.status != ProcessingStatus.IDLE:
            self._update_status(
                "Only idle items can have their output renamed", "warning")
            return

        current = Path(item.output_path)
        suffix = current.suffix or Path(item.file_path).suffix
        ext_star = f"*{suffix}" if suffix else "*.*"
        new_path = filedialog.asksaveasfilename(
            parent=self.root,
            title="Choose an output path",
            initialdir=str(current.parent),
            initialfile=current.name,
            defaultextension=suffix,
            filetypes=[("Keep extension", ext_star), ("All files", "*.*")],
        )
        if not new_path:
            return

        resolved_path = self._make_unique_output_path(
            Path(new_path),
            exclude_item_id=item.id,
        )
        item.output_path = str(resolved_path)
        item.output_path_locked = True
        if item.id in self.queue_widgets:
            self.queue_widgets[item.id].update_item(item)
        if self._normalized_path_key(new_path) != self._normalized_path_key(resolved_path):
            self._update_status(
                f"Output renamed to {resolved_path.name} to avoid an overwrite",
                "success",
            )
        else:
            self._update_status(
                f"Output renamed to {resolved_path.name}", "success")

    def _remove_from_queue(self, item_id: str):
        """Remove an item from the queue."""
        with self.queue_lock:
            # Don't remove items that are currently being processed
            item = next((i for i in self.queue if i.id == item_id), None)
            if item and item.status in (ProcessingStatus.LOADING, ProcessingStatus.DETECTING,
                                         ProcessingStatus.PROCESSING, ProcessingStatus.MERGING):
                self._update_status("Wait for the active item to finish before removing it", "warning")
                return
            self.queue = [i for i in self.queue if i.id != item_id]
        if self._selected_queue_item_id == item_id:
            self._selected_queue_item_id = None
        self._update_queue_display()
        if item:
            self._update_status(f"Removed {Path(item.file_path).name} from the queue")

    def _clear_queue(self):
        """Clear all items from the queue."""
        if self.is_processing:
            self._update_status("Stop the batch before clearing the queue", "warning")
            return
        if self.queue:
            n = len(self.queue)
            if not show_confirm(
                self.root,
                title="Clear the queue?",
                message=f"Remove {n} item{'s' if n != 1 else ''} from the batch.",
                detail="Completed outputs on disk are not deleted.",
                confirm_label="Clear queue",
                cancel_label="Keep",
                tone="danger",
            ):
                return

        with self.queue_lock:
            self.queue.clear()
        self._selected_queue_item_id = None
        self._update_queue_display()
        self._update_status("Queue cleared")

    def _update_queue_display(self):
        """Update the queue display. Only rebuilds widgets that changed."""
        with self.queue_lock:
            current_ids = {item.id for item in self.queue}

        # Remove widgets for items no longer in queue
        stale_ids = [wid for wid in self.queue_widgets if wid not in current_ids]
        for wid in stale_ids:
            self.queue_widgets[wid].destroy()
            del self.queue_widgets[wid]

        # Update count + stat chips
        total = len(self.queue)
        self.queue_count.config(text=f"{total} item{'s' if total != 1 else ''}")
        done = sum(1 for i in self.queue if i.status == ProcessingStatus.COMPLETE)
        err = sum(1 for i in self.queue
                  if i.status in (ProcessingStatus.ERROR, ProcessingStatus.CANCELLED))
        if done > 0:
            self.queue_done_lbl.config(text=f"{done} done")
            self.queue_done_pill.pack(side="left", padx=(Theme.S_XS, 0))
        else:
            self.queue_done_pill.pack_forget()
        if err > 0:
            self.queue_err_lbl.config(text=f"{err} failed")
            self.queue_err_pill.pack(side="left", padx=(Theme.S_XS, 0))
        else:
            self.queue_err_pill.pack_forget()
        # Sort button visibility
        try:
            if total >= 3:
                self._sort_btn.pack(side="left", padx=(Theme.S_SM, 0))
            else:
                self._sort_btn.pack_forget()
        except Exception:
            pass

        if not self.queue:
            # Clear any remaining children and show empty state
            for widget in self.queue_frame.winfo_children():
                widget.destroy()
            self.queue_widgets.clear()
            self._hide_filter_empty_state()
            self._build_queue_empty_state()
            self._set_preview_placeholder(
                "Preview a sample frame",
                "Select a queued item to inspect it before processing. Review mask is the fastest way to confirm the subtitle region.",
            )
        else:
            # Remove empty label if present
            for child in self.queue_frame.winfo_children():
                if child not in self.queue_widgets.values():
                    child.destroy()

            # Add widgets for new items only
            for item in self.queue:
                if item.id not in self.queue_widgets:
                    widget = QueueItemWidget(self.queue_frame, item, self._remove_from_queue,
                                             on_select=self._show_preview,
                                             on_rename=self._rename_output_for,
                                             on_repeat=self._repeat_item_with_settings,
                                             on_cancel_item=self._cancel_queue_item,
                                             on_override=self._open_per_file_overrides,
                                             on_soft_action=self._set_soft_subtitle_action)
                    widget.pack(fill="x", pady=(0, 8))
                    self.queue_widgets[item.id] = widget
                    # Forward mousewheel to queue canvas
                    widget.bind("<MouseWheel>", self._on_mousewheel)
                    for child in widget.winfo_children():
                        child.bind("<MouseWheel>", self._on_mousewheel)
                        for subchild in child.winfo_children():
                            subchild.bind("<MouseWheel>", self._on_mousewheel)
                else:
                    self.queue_widgets[item.id].update_item(item)

        if self._selected_queue_item_id and self._selected_queue_item_id in self.queue_widgets:
            self._set_selected_queue_item(self._selected_queue_item_id)
        else:
            self._set_selected_queue_item(None)
        self._refresh_action_states()
        # Show filter only when the queue is long enough to justify it
        try:
            if len(self.queue) >= 6:
                self._queue_filter_frame.pack(
                    fill="x", padx=Theme.S_XL, pady=(0, Theme.S_SM),
                    before=self._queue_container)
            else:
                self._queue_filter_frame.pack_forget()
                if self._queue_filter_var.get():
                    self._queue_filter_var.set("")
        except Exception:
            pass
        # Re-apply any active filter so newly added items get filtered too
        if self._queue_filter_var.get():
            self._apply_queue_filter()

    def _apply_queue_filter(self):
        """Hide/show queue widgets whose filename doesn't match the filter."""
        query = (self._queue_filter_var.get() or "").strip().lower()
        visible = 0
        total = len(self.queue)
        for item in self.queue:
            widget = self.queue_widgets.get(item.id)
            if not widget:
                continue
            fname = Path(item.file_path).name.lower()
            match = (query in fname) or (query in item.file_path.lower())
            if not query or match:
                if not widget.winfo_ismapped():
                    widget.pack(fill="x", pady=(0, Theme.S_SM))
                visible += 1
            else:
                widget.pack_forget()
        if query:
            self.queue_count.config(text=f"{visible} of {total} shown")
        else:
            self.queue_count.config(text=f"{total} item{'s' if total != 1 else ''}")

        if query and total and visible == 0:
            self._ensure_filter_empty_state()
            self._filter_empty_title.config(
                text=f'No items match "{truncate_middle(query, 28)}"')
            self._filter_empty_body.config(
                text="Try a shorter filename search, or clear the filter to see the full batch again.")
            if not self._filter_empty_container.winfo_ismapped():
                self._filter_empty_container.pack(
                    pady=(Theme.S_3XL, Theme.S_LG), fill="x")
        else:
            self._hide_filter_empty_state()

    def _update_status(self, message: str, tone: str = "neutral", toast: bool = False):
        """Update the footer status dot + message.

        If `toast=True`, also surface as a transient toast in the bottom-right.
        """
        colors = {
            "neutral": Theme.TEXT_SECONDARY,
            "success": Theme.SUCCESS,
            "warning": Theme.WARNING,
            "error": Theme.ERROR,
            "info": Theme.INFO,
        }
        color = colors.get(tone, Theme.TEXT_SECONDARY)
        self.status_label.config(text=message, fg=color)
        try:
            self.status_dot.itemconfig(self._status_dot_item, fill=color)
        except Exception:
            pass
        self._status_tone = tone
        if toast:
            try:
                Toast.show(self.root, message, tone=tone)
            except Exception:
                pass

    def _open_output_folder(self):
        """Open the output folder for the most recently completed item."""
        selected = self._get_selected_queue_item()
        if selected and selected.status == ProcessingStatus.COMPLETE and Path(selected.output_path).exists():
            target = selected
        else:
            completed = [i for i in self.queue if i.status == ProcessingStatus.COMPLETE]
            target = completed[-1] if completed else None
        if target:
            output_dir = str(Path(target.output_path).parent)
            try:
                os.startfile(output_dir)
                self._update_status("Opened the output folder", "info")
            except Exception:
                logger.warning(f"Could not open folder: {output_dir}")
        else:
            self._update_status("No completed results are available yet", "warning")

    def _show_preview(self, item: QueueItem, show_mask: bool = False):
        """Show thumbnail preview. Side-by-side before/after for completed items.
        If show_mask=True, run detection and overlay red boxes on the frame."""
        self._preview_request_id += 1
        preview_request_id = self._preview_request_id
        # Any switch cancels a running throbber so it can't overwrite later UI
        if not show_mask:
            self._stop_throbber()
        self._set_selected_queue_item(item.id)
        if not PIL_AVAILABLE:
            self.preview_title_label.config(text="Preview unavailable")
            self.preview_meta_label.config(text="Install Pillow to enable image previews.")
            self._preview_label.config(text="Install Pillow for previews", image="")
            return

        try:
            import cv2 as _cv2

            def load_first_frame_raw(path):
                """Load first frame as BGR numpy array."""
                if is_image_file(path):
                    return _cv2.imread(path)
                elif is_video_file(path):
                    cap = _cv2.VideoCapture(path)
                    try:
                        ret, frame = cap.read()
                        return frame if ret else None
                    finally:
                        cap.release()
                return None

            def to_pil(bgr_frame):
                return Image.fromarray(_cv2.cvtColor(bgr_frame, _cv2.COLOR_BGR2RGB))

            raw_frame = load_first_frame_raw(item.file_path)
            if raw_frame is None:
                self.preview_title_label.config(text="Preview unavailable")
                self.preview_meta_label.config(text="The selected file could not be read for preview.")
                self._preview_label.config(text="Could not read file", image="")
                return

            badge = status_ui(item.status)
            self.preview_status_chip.config(text=badge["label"], fg=badge["color"], bg=badge["bg"])

            try:
                max_w = max(220, self._preview_frame.winfo_width() - 36)
            except Exception:
                max_w = 390
            max_h = 158

            # Mask preview mode -- run detection in background thread
            if show_mask:
                self.preview_title_label.config(text=f"Detecting {Path(item.file_path).name}")
                self.preview_meta_label.config(
                    text="Running detection on the first frame..."
                )
                # Clear any existing preview image, then start animated throbber
                self._preview_label.config(image="", text="")
                self._preview_photo = None
                self._start_throbber()
                self._preview_label.update_idletasks()
                frame_copy = raw_frame.copy()
                lang = self.lang_var.get()
                threshold = getattr(self.config, '_detection_threshold_pct', 50) / 100.0
                # Match what processing will actually use: all manual
                # regions when set, else the single back-compat rect.
                sub_areas = list(getattr(self.config, "subtitle_areas", None) or [])
                if not sub_areas and self.config.subtitle_area:
                    sub_areas = [self.config.subtitle_area]

                def _detect_bg():
                    try:
                        from backend.processor import SubtitleDetector
                        # Reuse cached detector if lang hasn't changed
                        if self._preview_detector is None or self._preview_detector_lang != lang:
                            self._preview_detector = SubtitleDetector(lang=lang)
                            self._preview_detector_lang = lang
                        det = self._preview_detector
                        if sub_areas:
                            boxes = sub_areas
                        else:
                            boxes = det.detect(frame_copy, threshold)
                        vis = frame_copy.copy()
                        for (bx1, by1, bx2, by2) in boxes:
                            _cv2.rectangle(vis, (bx1, by1), (bx2, by2), (0, 0, 255), 2)
                        img = to_pil(vis)
                        img.thumbnail((max_w, max_h), Image.LANCZOS)
                        engine = det._engine_name
                        n = len(boxes)
                        def _update_ui():
                            if (preview_request_id != self._preview_request_id
                                    or self._selected_queue_item_id != item.id):
                                return
                            self._stop_throbber()
                            self._preview_photo = ImageTk.PhotoImage(img)
                            self.preview_title_label.config(text=f"Detection mask for {Path(item.file_path).name}")
                            if sub_areas:
                                meta = "Manual region applied. Detection used your saved subtitle band."
                            elif n:
                                meta = f"{engine} found {n} region{'s' if n != 1 else ''} on the first frame."
                            else:
                                meta = ("No regions were found on the first frame. Try Set region, or lower the "
                                        "Threshold in detailed controls.")
                            self.preview_meta_label.config(text=meta)
                            self._preview_label.config(
                                image=self._preview_photo,
                                text=f"{engine}: {n} detected" if n else "No text detected")
                        self.root.after(0, _update_ui)
                    except Exception as exc:
                        def _show_error():
                            if (preview_request_id != self._preview_request_id
                                    or self._selected_queue_item_id != item.id):
                                return
                            self._stop_throbber()
                            self.preview_title_label.config(text="Detection preview failed")
                            self.preview_meta_label.config(text="The detection preview could not be generated.")
                            self._preview_label.config(text=f"Detection error: {exc}", image="")
                        self.root.after(0, _show_error)

                threading.Thread(target=_detect_bg, daemon=True).start()
                return

            input_img = to_pil(raw_frame)

            # Check if completed and output exists -- show before/after
            output_img = None
            if item.status == ProcessingStatus.COMPLETE and Path(item.output_path).exists():
                out_frame = load_first_frame_raw(item.output_path)
                if out_frame is not None:
                    output_img = to_pil(out_frame)

            if output_img:
                half_w = max_w // 2 - 2
                input_img.thumbnail((half_w, max_h), Image.LANCZOS)
                output_img.thumbnail((half_w, max_h), Image.LANCZOS)
                total_w = input_img.width + output_img.width + 4
                total_h = max(input_img.height, output_img.height)
                composite = Image.new("RGB", (total_w, total_h), (15, 23, 42))
                composite.paste(input_img, (0, 0))
                composite.paste(output_img, (input_img.width + 4, 0))
                draw = ImageDraw.Draw(composite)
                draw.line([(input_img.width + 1, 0), (input_img.width + 1, total_h)],
                          fill="#22c55e", width=2)
                draw.rectangle((10, 10, 82, 28), fill=self._hex_to_rgb(Theme.BG_TERTIARY))
                draw.text((18, 14), "Source", fill=self._hex_to_rgb(Theme.TEXT_SECONDARY))
                draw.rectangle((input_img.width + 16, 10, input_img.width + 96, 28),
                               fill=self._hex_to_rgb(Theme.SUCCESS_BG))
                draw.text((input_img.width + 24, 14), "Cleaned",
                          fill=self._hex_to_rgb(Theme.SUCCESS))
                self._preview_photo = ImageTk.PhotoImage(composite)
                self.preview_title_label.config(text=f"Before / after for {Path(item.file_path).name}")
                meta = ("Completed items show the source frame beside the cleaned result so you can "
                        "spot-check the cleanup immediately.")
                quality_note = format_quality_report(item.quality_report)
                if quality_note:
                    meta += f" Quality check: {quality_note}."
                self.preview_meta_label.config(text=meta)
                self._preview_label.config(image=self._preview_photo, text="")
            else:
                input_img.thumbnail((max_w, max_h), Image.LANCZOS)
                self._preview_photo = ImageTk.PhotoImage(input_img)
                self.preview_title_label.config(text=f"Source frame for {Path(item.file_path).name}")
                soft_summary = _format_soft_subtitle_summary(
                    getattr(item, "soft_subtitle_streams", [])
                )
                if soft_summary:
                    preview_meta = (
                        f"{soft_summary}. Right-click the queue item for fast "
                        "strip/keep, or review the mask for burned-in cleanup."
                    )
                else:
                    preview_meta = (
                        "Review mask to confirm the subtitle band, then start "
                        "the batch when the framing looks right."
                    )
                self.preview_meta_label.config(
                    text=preview_meta
                )
                self._preview_label.config(image=self._preview_photo, text="")
        except Exception as e:
            self.preview_title_label.config(text="Preview unavailable")
            self.preview_meta_label.config(text="An unexpected preview error occurred.")
            self._preview_label.config(text=f"Preview error: {e}", image="")

    def _retry_failed(self):
        """Reset failed/cancelled items so they can be reprocessed."""
        if self.is_processing:
            self._update_status("Stop the active batch before retrying failed items", "warning")
            return
        count = 0
        with self.queue_lock:
            for item in self.queue:
                if item.status in (ProcessingStatus.ERROR, ProcessingStatus.CANCELLED):
                    item.status = ProcessingStatus.IDLE
                    item.progress = 0.0
                    item.message = "Ready to retry"
                    item.error = None
                    item.quality_report = None
                    item.started_at = None
                    item.completed_at = None
                    count += 1
        if count:
            self._update_queue_display()
            # Force-refresh all widgets to show reset state
            for item in self.queue:
                if item.message == "Ready to retry" and item.id in self.queue_widgets:
                    self.queue_widgets[item.id].update_item(item)
            self._update_status(f"Reset {count} item{'s' if count != 1 else ''} for retry", "success")
        else:
            self._update_status("There are no failed items to retry", "warning")

    def _last_completed_config(self) -> "ProcessingConfig | None":
        """Return the config snapshot from the most recently completed
        queue item, or None if no item has completed this session."""
        best = None
        for item in self.queue:
            if item.status == ProcessingStatus.COMPLETE and item.completed_at:
                if best is None or item.completed_at > best.completed_at:
                    best = item
        return best.config if best else None

    def _repeat_last_job(self):
        """Open a file picker and enqueue the selected files with the
        config from the most recently completed job."""
        if self.is_processing:
            self._update_status("Stop the active batch first", "warning")
            return
        source_config = self._last_completed_config()
        if source_config is None:
            self._update_status("No completed job to repeat", "warning")
            return
        paths = filedialog.askopenfilenames(
            title="Select files to process with the last job's settings",
            filetypes=[
                ("All Supported",
                 "*.mp4;*.avi;*.mkv;*.mov;*.wmv;*.flv;*.webm;*.m4v;"
                 "*.mpeg;*.mpg;*.jpg;*.jpeg;*.png;*.bmp;*.tiff;*.webp"),
            ],
        )
        if not paths:
            return
        added = 0
        for p in paths:
            result = self._add_to_queue(p)
            if result == "added":
                with self.queue_lock:
                    new_item = self.queue[-1]
                    new_item.config = ProcessingConfig.from_dict(
                        source_config.to_dict())
                added += 1
        if added:
            self._update_queue_display()
            self._update_status(
                f"Queued {added} file{'s' if added != 1 else ''} "
                f"with the last job's settings",
                "success", toast=True,
            )
        self._refresh_action_states()

    def _set_settings_locked(self, locked: bool):
        """Lock or unlock settings controls during processing."""
        entry_state = "disabled" if locked else "normal"
        combo_state = "disabled" if locked else "readonly"
        try:
            # Custom toggles
            self.skip_check.set_enabled(not locked)
            self.lama_check.set_enabled(not locked)
            self.preserve_audio_check.set_enabled(not locked)
            self.hw_encode_check.set_enabled(not locked)

            self.lang_combo.config(state=combo_state)
            if hasattr(self, 'gpu_combo'):
                self.gpu_combo.config(state=combo_state)
            self.time_start_entry.config(state=entry_state)
            self.time_end_entry.config(state=entry_state)

            self.region_btn.set_enabled(not locked)
            self.region_reset_btn.set_enabled(
                (not locked) and self.config.subtitle_area is not None)
            self.adv_toggle.set_enabled(not locked)
            # Segmented algo picker: dim/undim each segment
            try:
                for seg in self.mode_picker._segments.values():
                    seg.config(state="disabled" if locked else "normal")
            except Exception:
                pass
        except Exception:
            pass

        # Re-apply mode-specific toggle availability
        if not locked:
            try:
                self._update_mode_options()
            except Exception:
                pass

    def _start_processing(self):
        """Start processing the queue."""
        if not self.queue:
            self._update_status("Add media to the queue before starting a batch", "warning")
            return

        active_thread = self._has_active_processing_thread()
        batch_busy = self.is_processing or active_thread
        if batch_busy:
            if self._stop_requested or self.cancel_event.is_set():
                self._update_status(
                    "Batch is already stopping. Please wait for the current item to wrap up.",
                    "warning",
                )
                return
            if active_thread:
                self._stop_processing()
            else:
                self._update_status("Finalizing the previous batch...", "info")
            return

        self._apply_current_settings_to_idle_items()
        if self.preserve_audio_var.get() and not self.ffmpeg_ready:
            has_video = any(is_video_file(item.file_path) for item in self.queue)
            if has_video:
                self._update_status(
                    "FFmpeg is missing, so video outputs will be saved without original audio.",
                    "warning",
                    toast=True,
                )

        self.is_processing = True
        self._stop_requested = False
        self.cancel_event.clear()
        self._set_settings_locked(True)
        self.start_btn.set_style("danger")
        self.start_btn.icon = "x"
        self.start_btn.set_text("Stop batch")
        self._batch_times = []
        # F-9: the ETA probe loads an OCR model and detects 30 frames --
        # far too slow for the Tk main thread. _process_queue runs it on
        # the worker thread before the first item; until then the ETA
        # line is simply empty.
        self._probe_eta_seconds = 0.0
        self._batch_started_at = datetime.now()
        self._prepare_batch_report_records()
        self._write_batch_preflight_plan()
        self._last_batch_report_paths = []
        self._refresh_action_states()
        self._update_status("Batch processing started", "info")
        # Kick off Windows taskbar progress in indeterminate until first tick
        self._ensure_taskbar()
        if self._taskbar:
            self._taskbar.set_state(TaskbarProgress.STATE_INDETERMINATE)

        # Start elapsed timer
        self._start_elapsed_timer()

        # Start processing thread
        self._processing_thread = threading.Thread(target=self._process_queue, daemon=True)
        self._processing_thread.start()

    def _stop_processing(self):
        """Stop the current processing."""
        if self._stop_requested:
            self._update_status("Batch is already stopping...", "warning")
            return
        self._stop_requested = True
        self.cancel_event.set()
        # Invalidate the cached remover so the next batch re-initialises with
        # fresh state. A cancelled run may have left detector / inpainter /
        # SRT buffers in an intermediate state.
        self._cached_remover = None
        self._cached_remover_key = None

        self.start_btn.set_style("primary")
        self.start_btn.icon = "x"
        self.start_btn.set_text("Stopping...")
        self._refresh_action_states()
        self._update_status(
            "Stopping after the current step. Finished outputs stay on disk.",
            "warning",
        )
        if self._taskbar:
            self._taskbar.set_state(TaskbarProgress.STATE_PAUSED)

    def _has_active_processing_thread(self) -> bool:
        return self._processing_thread is not None and self._processing_thread.is_alive()

    def _start_elapsed_timer(self):
        """Start a timer that updates elapsed times on in-progress queue items."""
        # Cancel any existing timer before starting a new one to avoid
        # stacking multiple concurrent tick loops.
        self._stop_elapsed_timer()
        def tick():
            if not self.is_processing:
                return
            try:
                for widget in list(self.queue_widgets.values()):
                    if widget.item.started_at and not widget.item.completed_at:
                        elapsed = (datetime.now() - widget.item.started_at).total_seconds()
                        widget.time_label.config(text=format_time(elapsed))
            except Exception:
                pass
            self._elapsed_timer_id = self.root.after(1000, tick)
        self._elapsed_timer_id = self.root.after(1000, tick)

    def _stop_elapsed_timer(self):
        if self._elapsed_timer_id:
            self.root.after_cancel(self._elapsed_timer_id)
            self._elapsed_timer_id = None

    def _batch_report_device(self, item: QueueItem) -> str:
        if not getattr(item.config, "use_gpu", False):
            return "cpu"
        for gpu in self.gpus:
            if gpu.get("index") == item.config.gpu_id:
                if gpu.get("type") == "DirectML":
                    return "directml"
                break
        return f"cuda:{item.config.gpu_id}"

    def _prepare_batch_report_records(self):
        """Build preflight report records for the queue without processing frames."""
        from backend.batch_report import make_batch_item_record

        with self.queue_lock:
            items = [
                item for item in self.queue
                if item.status not in (
                    ProcessingStatus.COMPLETE,
                    ProcessingStatus.ERROR,
                    ProcessingStatus.CANCELLED,
                )
            ]
        records = {}
        for item in items:
            soft_action = (
                item.soft_subtitle_action
                if item.soft_subtitle_action in {"strip", "keep_all"}
                else None
            )
            try:
                records[item.id] = make_batch_item_record(
                    item.file_path,
                    item.output_path,
                    config={
                        "mode": item.config.mode.value,
                        "device": self._batch_report_device(item),
                        "output_codec": getattr(item.config, "output_codec", "h264"),
                    },
                    soft_action=soft_action,
                )
            except Exception as exc:
                logger.warning(
                    f"Batch preflight report failed for {Path(item.file_path).name}: {exc}"
                )
        self._batch_report_records = records

    def _finalize_batch_report_records(self) -> List[dict]:
        from backend.batch_report import (
            STATUS_CANCELLED,
            STATUS_FAILED,
            STATUS_HARDCODED_PROCESSED,
            STATUS_SOFT_REMUXED,
            finish_batch_item,
        )

        records = getattr(self, "_batch_report_records", {}) or {}
        if not records:
            return []
        by_id = {item.id: item for item in self.queue}
        finished: List[dict] = []
        for item_id, record in records.items():
            item = by_id.get(item_id)
            if item is None:
                continue
            elapsed = None
            if item.started_at and item.completed_at:
                elapsed = (item.completed_at - item.started_at).total_seconds()
            if item.status == ProcessingStatus.COMPLETE:
                status = (
                    STATUS_SOFT_REMUXED
                    if item.soft_subtitle_action in {"strip", "keep_all"}
                    else STATUS_HARDCODED_PROCESSED
                )
                message = item.message or "Complete"
            elif item.status == ProcessingStatus.ERROR:
                status = STATUS_FAILED
                message = item.error or item.message or "Processing failed"
            elif item.status == ProcessingStatus.CANCELLED:
                status = STATUS_CANCELLED
                message = item.message or "Cancelled"
            else:
                status = STATUS_CANCELLED
                message = item.message or "Not processed"
            finish_batch_item(
                record,
                status,
                message=message,
                elapsed_seconds=elapsed,
                quality_report=(
                    item.quality_report
                    if status == STATUS_HARDCODED_PROCESSED
                    else None
                ),
            )
            finished.append(record)
        return finished

    def _write_batch_preflight_plan(self) -> List[Path]:
        """Write a preflight plan JSON before processing starts, so
        overnight runs are fully accounted for even on crash."""
        records = getattr(self, "_batch_report_records", {}) or {}
        if not records:
            return []
        from backend.io import _write_text_atomic
        import json as _json
        grouped: dict[Path, List[dict]] = {}
        for item_id, record in records.items():
            out_dir = Path(record.get("output") or ".").parent
            grouped.setdefault(out_dir, []).append(record)
        written: List[Path] = []
        for out_dir, group in grouped.items():
            plan_path = out_dir / "vsr_batch_plan.json"
            payload = {
                "schema": "vsr.batch_plan.v1",
                "created_at": datetime.now().astimezone().isoformat(
                    timespec="seconds"),
                "count": len(group),
                "files": [
                    {
                        "input_name": r.get("input_name", ""),
                        "output_name": r.get("output_name", ""),
                        "planned_result": r.get("planned_result", ""),
                        "mode": r.get("mode", ""),
                        "device": r.get("device", ""),
                        "duration_seconds": r.get("duration_seconds", 0),
                        "estimated_seconds": r.get("estimated_seconds", 0),
                    }
                    for r in group
                ],
            }
            _write_text_atomic(
                plan_path,
                _json.dumps(payload, indent=2, ensure_ascii=True) + "\n",
            )
            written.append(plan_path)
            logger.info(f"Batch preflight plan written: {plan_path}")
        return written

    def _write_batch_report_files(self) -> List[Path]:
        from backend.batch_report import write_batch_reports

        records = self._finalize_batch_report_records()
        if not records:
            return []
        started_at = self._batch_started_at or datetime.now()
        grouped: dict[Path, List[dict]] = {}
        for record in records:
            out_dir = Path(record.get("output") or ".").parent
            grouped.setdefault(out_dir, []).append(record)
        written: List[Path] = []
        for out_dir, group in grouped.items():
            json_path, md_path = write_batch_reports(
                out_dir,
                group,
                kind="gui-batch",
                started_at=started_at,
                completed_at=datetime.now(),
            )
            written.extend([json_path, md_path])
            logger.info(f"Batch report written: {json_path}")
        self._last_batch_report_paths = written
        return written

    def _process_queue(self):
        """Process all items in the queue."""
        # F-9: pre-batch ETA probe runs here, on the worker thread, so
        # model load + 30-frame detection never block the Tk main loop.
        try:
            self._probe_eta_seconds = self._probe_batch_eta()
        except Exception:
            self._probe_eta_seconds = 0.0
        with self.queue_lock:
            items_to_process = [i for i in self.queue
                                if i.status not in (ProcessingStatus.COMPLETE,
                                                     ProcessingStatus.ERROR,
                                                     ProcessingStatus.CANCELLED)]

        total = len(items_to_process)
        for idx, item in enumerate(items_to_process):
            if self.cancel_event.is_set():
                # Mark ALL remaining items as cancelled
                now = datetime.now()
                for remaining in items_to_process[idx:]:
                    remaining.status = ProcessingStatus.CANCELLED
                    remaining.message = "Cancelled"
                    remaining.completed_at = now
                    self._update_item_display(remaining)
                break

            # Update batch progress + window title
            try:
                self.root.after(0, self._update_batch_progress, idx, total)
            except RuntimeError:
                return  # root destroyed during shutdown
            self._process_item(item)

        # Final batch state
        try:
            self.root.after(0, self._update_batch_progress, total, total)
            self.root.after(0, self._on_processing_complete)
        except RuntimeError:
            pass  # root destroyed during shutdown

    def _process_soft_subtitle_item(self, item: QueueItem) -> bool:
        action_value = getattr(item, "soft_subtitle_action", "burned_in")
        if action_value not in {"strip", "keep_all"}:
            return False

        from backend.remux import SoftSubtitleAction, remux_soft_subtitles

        action_map = {
            "strip": SoftSubtitleAction.STRIP,
            "keep_all": SoftSubtitleAction.KEEP_ALL,
        }
        action = action_map[action_value]

        item.status = ProcessingStatus.MERGING
        item.progress = 0.2
        item.message = (
            "Stripping embedded subtitle tracks..."
            if action == SoftSubtitleAction.STRIP else
            "Remuxing embedded subtitle tracks..."
        )
        self._update_item_display(item)

        Path(item.output_path).parent.mkdir(parents=True, exist_ok=True)
        remux_soft_subtitles(item.file_path, item.output_path, action=action)

        item.status = ProcessingStatus.COMPLETE
        item.progress = 1.0
        item.error = None
        item.quality_report = None
        item.completed_at = datetime.now()
        item.message = (
            "Embedded subtitles stripped"
            if action == SoftSubtitleAction.STRIP else
            "Embedded subtitles remuxed"
        )
        elapsed = (item.completed_at - item.started_at).total_seconds()
        self._batch_times.append(elapsed)
        logger.info(
            f"Soft-subtitle {action.value}: {Path(item.file_path).name} "
            f"in {format_time(elapsed)}"
        )
        self._update_item_display(item)
        return True

    def _process_item(self, item: QueueItem):
        """Process a single queue item using the backend processor."""
        try:
            item.status = ProcessingStatus.LOADING
            item.started_at = datetime.now()
            item.completed_at = None
            item.progress = 0.0
            item.message = "Initializing..."
            item.error = None
            item.quality_report = None
            item.cancel_requested = False  # F-7 reset on fresh attempt
            self._update_item_display(item)

            if self._process_soft_subtitle_item(item):
                return

            from backend.processor import (
                SubtitleRemover as BackendRemover,
                ProcessingConfig as BackendConfig,
            )

            backend_mode = self._gui_to_backend_mode(item.config.mode.value)
            device = self._gui_to_backend_device(
                item.config.use_gpu, item.config.gpu_id)
            lang = getattr(item.config, 'detection_lang', 'en')
            vertical = bool(getattr(item.config, 'detection_vertical', False))
            cache_key = (backend_mode, device, lang, vertical)

            backend_config = BackendConfig(
                mode=backend_mode,
                device=device,
                sttn_skip_detection=item.config.sttn_skip_detection,
                sttn_neighbor_stride=item.config.sttn_neighbor_stride,
                sttn_reference_length=item.config.sttn_reference_length,
                sttn_max_load_num=item.config.sttn_max_load_num,
                lama_super_fast=item.config.lama_super_fast,
                preserve_audio=item.config.preserve_audio,
                output_quality=item.config.output_quality,
                detection_lang=lang,
                detection_threshold=getattr(item.config, 'detection_threshold', 0.5),
                detection_vertical=getattr(item.config, 'detection_vertical', False),
                whisper_fallback=getattr(item.config, 'whisper_fallback', False),
                whisper_backend=getattr(item.config, 'whisper_backend', 'faster-whisper'),
                whisper_model_size=getattr(item.config, 'whisper_model_size', 'tiny'),
                whisper_model_path=getattr(item.config, 'whisper_model_path', ''),
                whisper_queue_seconds=getattr(item.config, 'whisper_queue_seconds', 3.0),
                upscale_factor=getattr(item.config, 'upscale_factor', 0),
                film_grain_strength=getattr(item.config, 'film_grain_strength', 0.0),
                swinir_restore=getattr(item.config, 'swinir_restore', False),
                seedvr2_restore=getattr(item.config, 'seedvr2_restore', False),
                preserve_color_metadata=getattr(item.config, 'preserve_color_metadata', True),
                nle_sidecar=getattr(item.config, 'nle_sidecar', 'off'),
                subtitle_area=item.config.subtitle_area,
                time_start=getattr(item.config, 'time_start', 0.0),
                time_end=getattr(item.config, 'time_end', 0.0),
                detection_frame_skip=getattr(item.config, 'detection_frame_skip', 0),
                mask_dilate_px=getattr(item.config, 'mask_dilate_px', 8),
                mask_feather_px=getattr(item.config, 'mask_feather_px', 4),
                tbe_enable=getattr(item.config, 'tbe_enable', True),
                tbe_min_coverage=getattr(item.config, 'tbe_min_coverage', 3),
                tbe_use_median=getattr(item.config, 'tbe_use_median', True),
                tbe_flow_warp=getattr(item.config, 'tbe_flow_warp', False),
                tbe_scene_cut_split=getattr(item.config, 'tbe_scene_cut_split', True),
                tbe_scene_cut_threshold=getattr(item.config, 'tbe_scene_cut_threshold', 0.35),
                tbe_scene_cut_use_pyscenedetect=getattr(item.config, 'tbe_scene_cut_use_pyscenedetect', False),
                tbe_scene_cut_use_transnetv2=getattr(item.config, 'tbe_scene_cut_use_transnetv2', False),
                detection_denoise=getattr(item.config, 'detection_denoise', False),
                sam2_refine=getattr(item.config, 'sam2_refine', False),
                edge_ring_px=getattr(item.config, 'edge_ring_px', 2),
                subtitle_areas=getattr(item.config, 'subtitle_areas', None),
                export_srt=getattr(item.config, 'export_srt', False),
                export_mask_video=getattr(item.config, 'export_mask_video', False),
                adaptive_batch=getattr(item.config, 'adaptive_batch', True),
                auto_exposure_threshold=getattr(item.config, 'auto_exposure_threshold', 0.55),
                deinterlace=getattr(item.config, 'deinterlace', False),
                deinterlace_auto=getattr(item.config, 'deinterlace_auto', True),
                keyframe_detection=getattr(item.config, 'keyframe_detection', False),
                quality_report=getattr(item.config, 'quality_report', False),
                kalman_tracking=getattr(item.config, 'kalman_tracking', True),
                kalman_iou_threshold=getattr(item.config, 'kalman_iou_threshold', 0.3),
                kalman_max_age=getattr(item.config, 'kalman_max_age', 2),
                phash_skip_enable=getattr(item.config, 'phash_skip_enable', True),
                phash_skip_distance=getattr(item.config, 'phash_skip_distance', 4),
                colour_tune_enable=getattr(item.config, 'colour_tune_enable', False),
                colour_tune_tolerance=getattr(item.config, 'colour_tune_tolerance', 25),
                use_hw_encode=getattr(item.config, 'use_hw_encode', True),
                output_codec=getattr(item.config, 'output_codec', 'h264'),
                # v3.13 GUI-exposed fields: previously CLI-only, now plumbed
                # through so a GUI user can drive every backend feature.
                loudnorm_target=getattr(item.config, 'loudnorm_target', 0.0),
                multi_audio_passthrough=getattr(item.config, 'multi_audio_passthrough', True),
                decode_hw_accel=getattr(item.config, 'decode_hw_accel', 'off'),
                prefetch_decode=getattr(item.config, 'prefetch_decode', True),
                prefetch_queue_size=getattr(item.config, 'prefetch_queue_size', 0),
                input_fps=getattr(item.config, 'input_fps', 24.0),
                quality_report_sheet=getattr(item.config, 'quality_report_sheet', False),
                remove_subtitles=getattr(item.config, 'remove_subtitles', True),
                remove_chyrons=getattr(item.config, 'remove_chyrons', True),
                chyron_min_hits=getattr(item.config, 'chyron_min_hits', 90),
                karaoke_grouping=getattr(item.config, 'karaoke_grouping', False),
                karaoke_x_gap_px=getattr(item.config, 'karaoke_x_gap_px', 20),
                karaoke_y_overlap=getattr(item.config, 'karaoke_y_overlap', 0.5),
            )

            # Auto subtitle-band detection -- run before the main pass so we
            # can pin the dominant band once per file. Cheap (30-frame probe).
            if getattr(item.config, 'auto_band', False) and not item.config.subtitle_area:
                try:
                    # Use a minimal config just for the band probe
                    probe_cfg = BackendConfig(
                        mode=backend_mode,
                        device=device,
                        detection_lang=lang,
                        detection_threshold=getattr(item.config, 'detection_threshold', 0.5),
                    )
                    probe = BackendRemover(probe_cfg)
                    band = probe.detect_subtitle_band(item.file_path, probe_frames=30)
                    if band:
                        backend_config.subtitle_area = band
                        logger.info(f"Auto-band: {band} for {Path(item.file_path).name}")
                except Exception as exc:
                    logger.warning(f"Auto-band detection failed: {exc}")

            # Reuse cached remover if mode/device/lang match (avoids reloading
            # OCR models and re-probing HW encoders for every queue item).
            # The constructor normalises the config; on hot-swap we re-run
            # normalisation explicitly so a NaN/inf/out-of-range value from a
            # bad per-item override cannot reach the pipeline.
            if self._cached_remover is not None and self._cached_remover_key == cache_key:
                remover = self._cached_remover
                from backend.processor import normalize_processing_config as _normalize_backend_config
                remover.config = _normalize_backend_config(backend_config)
            else:
                remover = BackendRemover(backend_config)
                self._cached_remover = remover
                self._cached_remover_key = cache_key
            if hasattr(remover, "last_quality_report"):
                remover.last_quality_report = None

            def on_progress(progress: float, message: str):
                if self.cancel_event.is_set():
                    raise InterruptedError("Processing cancelled")
                # F-7: per-item cancel raises the same exception so
                # process_video bails on THIS file; the outer
                # _process_queue loop then advances to the next item
                # because cancel_event was never set.
                if getattr(item, "cancel_requested", False):
                    raise InterruptedError("Item cancelled by user")
                # Map backend progress to GUI status
                if progress < 0.3:
                    item.status = ProcessingStatus.DETECTING
                elif progress < 0.9:
                    item.status = ProcessingStatus.PROCESSING
                elif progress < 1.0:
                    item.status = ProcessingStatus.MERGING
                else:
                    item.status = ProcessingStatus.COMPLETE
                item.progress = progress
                item.message = message
                self._update_item_display(item)

            remover.on_progress = on_progress

            # Live preview: pipe the latest inpainted frame into the preview
            # pane. The backend emits frames on its worker thread, so we
            # marshal to the Tk main loop via root.after.
            #
            # EI-4: also throttle on wall-clock so the worker does not
            # queue PIL conversions faster than the Tk thread can absorb
            # ImageTk.PhotoImage calls (~50 ms on 4K). The receiver still
            # throttles to ~15 FPS, but throttling in the worker too
            # avoids burning CPU on conversions that get dropped.
            preview_throttle_state = {"last_ts": 0.0}
            def on_preview_frame(frame, cur_idx, total):
                if self.cancel_event.is_set():
                    return
                now = time.monotonic()
                if (now - preview_throttle_state["last_ts"]) < (1.0 / 15.0):
                    return
                preview_throttle_state["last_ts"] = now
                try:
                    max_w, max_h = 520, 320
                    h, w = frame.shape[:2]
                    scale = min(max_w / max(1, w), max_h / max(1, h), 1.0)
                    if scale < 1.0:
                        new_w = max(1, int(w * scale))
                        new_h = max(1, int(h * scale))
                        import cv2 as _cv2_live
                        small = _cv2_live.resize(frame, (new_w, new_h),
                                                  interpolation=_cv2_live.INTER_AREA)
                    else:
                        small = frame
                    rgb = small[..., ::-1]  # BGR -> RGB
                    from PIL import Image as _Image
                    pil = _Image.fromarray(rgb)
                    self.root.after(0, self._push_live_preview, pil, cur_idx, total,
                                     Path(item.file_path).name)
                except Exception:
                    pass

            remover.on_preview_frame = on_preview_frame

            # Ensure output directory exists
            Path(item.output_path).parent.mkdir(parents=True, exist_ok=True)

            # Run the actual processing
            file_name = Path(item.file_path).name
            logger.info(f"Processing: {file_name} with {item.config.mode.value}")

            if is_video_file(item.file_path):
                success = remover.process_video(item.file_path, item.output_path)
            elif is_image_file(item.file_path):
                success = remover.process_image(item.file_path, item.output_path)
            else:
                raise ValueError(f"Unsupported file type: {Path(item.file_path).suffix}")

            if success:
                item.status = ProcessingStatus.COMPLETE
                item.progress = 1.0
                item.error = None
                item.quality_report = getattr(remover, "last_quality_report", None)
                item.message = "Complete!"
                quality_note = format_quality_report(item.quality_report, compact=True)
                if quality_note:
                    item.message = f"Complete - {quality_note}"
                item.completed_at = datetime.now()
                elapsed = (item.completed_at - item.started_at).total_seconds()
                # Track for ETA rolling average
                self._batch_times.append(elapsed)
                logger.info(f"Completed: {file_name} in {format_time(elapsed)}")
            else:
                item.status = ProcessingStatus.ERROR
                item.message = "Processing failed"
                item.quality_report = None
                item.completed_at = datetime.now()
                logger.error(f"Failed: {file_name}")
            self._update_item_display(item)

        except InterruptedError:
            item.status = ProcessingStatus.CANCELLED
            item.message = "Cancelled"
            item.error = None
            item.quality_report = None
            item.completed_at = datetime.now()
            self._update_item_display(item)
            logger.info(f"Cancelled: {Path(item.file_path).name}")
        except Exception as e:
            item.status = ProcessingStatus.ERROR
            item.error = str(e)
            item.message = f"Error: {str(e)}"
            item.quality_report = None
            item.completed_at = datetime.now()
            self._update_item_display(item)
            logger.error(f"Processing error for {item.file_path}: {e}")

    def _ensure_taskbar(self):
        """Lazily create the Windows taskbar progress client once the window
        is fully realized."""
        if self._taskbar is not None:
            return
        try:
            hwnd = self.root.winfo_id()
            # Walk up to the top-level window (important on some tk builds)
            import ctypes
            hwnd = ctypes.windll.user32.GetParent(hwnd) or hwnd
            self._taskbar = TaskbarProgress(hwnd)
        except Exception:
            self._taskbar = None

    def _compute_eta(self, current: int, total: int) -> str:
        """Estimate time-remaining based on rolling average per-item time.

        F-9: when no items have completed yet we fall back to the
        pre-batch probe estimate (`_probe_eta_seconds`) so users get a
        sensible "about X left" line from the very first frame instead
        of an empty string until the first item finishes.
        """
        remaining = total - current
        if remaining <= 0:
            return ""
        if self._batch_times:
            recent = self._batch_times[-5:]
            avg = sum(recent) / len(recent)
            eta_seconds = avg * remaining
            return format_time(eta_seconds)
        probe = getattr(self, "_probe_eta_seconds", 0.0) or 0.0
        if probe > 0:
            return format_time(probe * remaining) + " (estimated)"
        return ""

    def _probe_batch_eta(self) -> float:
        """F-9: cheap pre-batch ETA probe. Reads a 30-frame slice from
        the first queued video, runs detect + inpaint on that slice,
        scales the wall-time by the video's frame count divided by the
        probe size. Returns the estimated per-item seconds (or 0 if the
        probe can't run -- e.g. only images in the queue).

        Called from _process_queue on the worker thread so the GUI
        stays responsive; the detect loop is capped at ~10 s so the
        first item still starts promptly on slow CPUs.
        """
        first_video = None
        for item in self.queue:
            if is_video_file(item.file_path) and item.status == ProcessingStatus.IDLE:
                first_video = item
                break
        if first_video is None:
            return 0.0
        try:
            import cv2 as _cv2
            cap = _cv2.VideoCapture(first_video.file_path)
            try:
                if not cap.isOpened():
                    return 0.0
                total_frames = int(cap.get(_cv2.CAP_PROP_FRAME_COUNT)) or 1
                fps = cap.get(_cv2.CAP_PROP_FPS) or 30.0
                if fps <= 0:
                    fps = 30.0
                duration = total_frames / fps
                probe_frames = min(30, total_frames)
                if probe_frames <= 0:
                    return 0.0
                from backend.processor import SubtitleDetector
                detector = self._preview_detector
                lang = first_video.config.detection_lang or "en"
                if detector is None or self._preview_detector_lang != lang:
                    detector = SubtitleDetector(lang=lang)
                    self._preview_detector = detector
                    self._preview_detector_lang = lang
                threshold = getattr(first_video.config, "detection_threshold", 0.5)
                t0 = time.monotonic()
                frames_done = 0
                for _ in range(probe_frames):
                    ok, frame = cap.read()
                    if not ok:
                        break
                    detector.detect(frame, threshold)
                    frames_done += 1
                    if time.monotonic() - t0 > 10.0:
                        break
                elapsed = time.monotonic() - t0
            finally:
                cap.release()
        except Exception as exc:
            logger.debug(f"Pre-batch ETA probe failed: {exc}")
            return 0.0
        if elapsed <= 0 or frames_done <= 0:
            return 0.0
        # Scale to the full video duration. Add a fudge factor for the
        # inpaint pass and ffmpeg mux which the detect-only probe does
        # not see. 1.8x leaves room for slower inpainters without
        # over-estimating to the point of being useless.
        per_frame_detect = elapsed / frames_done
        est_per_video = per_frame_detect * total_frames * 1.8 + max(2.0, duration * 0.05)
        return est_per_video

    def _update_batch_progress(self, current: int, total: int):
        """Update the overall batch progress bar, percent label, and title."""
        if total > 0:
            progress = current / total
            pct = int(progress * 100)
            self.batch_progress.set_progress(progress)
            eta = self._compute_eta(current, total)
            label = f"{current} of {total} complete"
            if eta:
                label += f"   -   about {eta} left"
            self.batch_label.config(text=label, fg=Theme.TEXT_SECONDARY)
            self.batch_percent_label.config(text=f"{pct}%", fg=Theme.BLUE_PRIMARY)
            self.root.title(f"[{current}/{total}] {APP_NAME} v{APP_VERSION}")
            # Windows taskbar
            self._ensure_taskbar()
            if self._taskbar:
                self._taskbar.set_state(TaskbarProgress.STATE_NORMAL)
                self._taskbar.set_value(current, total)
        else:
            self.batch_progress.set_progress(0)
            self.batch_label.config(text="Ready", fg=Theme.TEXT_MUTED)
            self.batch_percent_label.config(text="")
            if self._taskbar:
                self._taskbar.clear()

    def _update_item_display(self, item: QueueItem):
        """Update the display for a queue item."""
        def update():
            if item.id in self.queue_widgets:
                self.queue_widgets[item.id].update_item(item)
                # Auto-scroll the queue to keep the active item visible
                if item.status in (ProcessingStatus.LOADING,
                                   ProcessingStatus.DETECTING,
                                   ProcessingStatus.PROCESSING,
                                   ProcessingStatus.MERGING):
                    self._scroll_queue_to_item(item.id)
            fname = Path(item.file_path).name
            if item.status == ProcessingStatus.COMPLETE:
                self._update_status(f"Completed {fname}", "success")
            elif item.status == ProcessingStatus.ERROR:
                self._update_status(f"{fname} needs attention: {item.message}", "error")
            elif item.status == ProcessingStatus.CANCELLED:
                self._update_status(f"Stopped {fname}", "warning")
            else:
                self._update_status(f"{fname}: {item.message}", "info")
            self._refresh_action_states()

        try:
            self.root.after(0, update)
        except RuntimeError:
            pass  # root already destroyed during shutdown

    def _on_processing_complete(self):
        """Handle processing completion."""
        self.is_processing = False
        self._stop_requested = False
        self._processing_thread = None
        self.cancel_event.clear()
        self._stop_elapsed_timer()
        self._set_settings_locked(False)
        # Clear cached remover so next batch picks up any setting changes
        self._cached_remover = None
        self._cached_remover_key = None
        report_paths = self._write_batch_report_files()
        if self._shutdown_started:
            if self._taskbar:
                self._taskbar.clear()
            try:
                self.root.destroy()
            except Exception:
                pass
            return
        self.start_btn.set_style("primary")
        self.start_btn.icon = ">"
        self.start_btn.set_text("Start batch")
        self.root.title(f"{APP_NAME} v{APP_VERSION}")
        self.batch_progress.set_progress(0)
        self.batch_label.config(text="Ready", fg=Theme.TEXT_MUTED)
        if hasattr(self, "batch_percent_label"):
            self.batch_percent_label.config(text="")
        if self._taskbar:
            self._taskbar.clear()
        self._refresh_action_states()

        complete = sum(1 for item in self.queue if item.status == ProcessingStatus.COMPLETE)
        errors = sum(1 for item in self.queue if item.status == ProcessingStatus.ERROR)
        cancelled = sum(1 for item in self.queue if item.status == ProcessingStatus.CANCELLED)

        summary = f"Batch finished: {complete} completed, {errors} failed"
        if cancelled:
            summary += f", {cancelled} stopped"
        is_clean = errors == 0 and cancelled == 0
        quality_summary = summarize_quality_reports(
            [item.quality_report for item in self.queue if item.status == ProcessingStatus.COMPLETE]
        )
        if quality_summary:
            summary += (
                f" | avg PSNR {quality_summary['psnr']:.2f} dB"
                f", avg SSIM {quality_summary['ssim']:.4f}"
            )
        self._update_status(summary, "success" if is_clean else "warning")
        logger.info(summary)
        if report_paths:
            logger.info(
                "Batch reports: "
                + ", ".join(str(path) for path in report_paths)
            )
        self._notify_completion(complete, errors)
        # Surface a themed summary modal for meaningful batches
        total = complete + errors + cancelled
        if total >= 1:
            elapsed = ""
            if self._batch_started_at:
                secs = (datetime.now() - self._batch_started_at).total_seconds()
                elapsed = format_time(secs)
            self._show_batch_summary(
                complete,
                errors,
                cancelled,
                elapsed,
                quality_summary=quality_summary,
            )

    def _notify_completion(self, complete: int, errors: int):
        """Flash taskbar + play sound when batch processing finishes."""
        # RM-95: screen-reader announcement so NVDA / Narrator users
        # learn the batch finished without polling the activity log.
        try:
            from backend.a11y import announce
            if errors == 0:
                announce(f"Batch complete. {complete} items processed.")
            else:
                announce(
                    f"Batch finished with {errors} errors. "
                    f"{complete} items processed.",
                    importance="high",
                )
        except Exception:
            pass
        # Flash the taskbar icon to draw attention
        try:
            import ctypes
            import ctypes.wintypes
            hwnd = ctypes.windll.user32.GetParent(self.root.winfo_id())

            class FLASHWINFO(ctypes.Structure):
                _fields_ = [
                    ('cbSize', ctypes.wintypes.UINT),
                    ('hwnd', ctypes.wintypes.HWND),
                    ('dwFlags', ctypes.wintypes.DWORD),
                    ('uCount', ctypes.wintypes.UINT),
                    ('dwTimeout', ctypes.wintypes.DWORD),
                ]

            FLASHW_ALL = 0x03
            FLASHW_TIMERNOFG = 0x0C
            fwi = FLASHWINFO(
                ctypes.sizeof(FLASHWINFO), hwnd,
                FLASHW_ALL | FLASHW_TIMERNOFG, 5, 0)
            ctypes.windll.user32.FlashWindowEx(ctypes.byref(fwi))
        except Exception:
            pass
        # Completion sound
        try:
            import winsound
            if errors == 0:
                winsound.MessageBeep(winsound.MB_OK)
            else:
                winsound.MessageBeep(winsound.MB_ICONEXCLAMATION)
        except Exception:
            pass

    def _queue_argv_files(self):
        """RM-58: queue files passed via sys.argv (e.g. 'Send to VSR')."""
        for arg in sys.argv[1:]:
            try:
                p = Path(arg)
                if p.is_file() and (is_video_file(str(p)) or is_image_file(str(p))):
                    self._add_to_queue(str(p.resolve()))
            except Exception:
                pass

    def _check_for_update(self):
        """RM-116: opt-in startup update check against GitHub Releases."""
        try:
            from backend.update_check import check_for_update
        except ImportError:
            return

        def _on_update(tag, url):
            try:
                self.root.after(0, lambda: self._show_update_toast(tag, url))
            except Exception:
                pass

        check_for_update(APP_VERSION, _on_update)

    def _show_update_toast(self, tag, url):
        try:
            Toast.show(self.root, f"Update available: {tag}", "info")
            if url:
                # The toast is transient; surface the release link where
                # the user can actually reach it.
                logger.info(f"Update {tag} available: {url}")
                self._update_status(
                    f"Update {tag} available -- link in the log panel",
                    "info",
                )
        except Exception:
            pass

    def run(self):
        """Run the application."""
        self.root.update_idletasks()
        screen_w = self.root.winfo_screenwidth()
        screen_h = self.root.winfo_screenheight()

        # Try to restore saved geometry if it still fits on-screen; otherwise
        # fall back to a sensibly centered default.
        restored = False
        saved = (self.config.window_geometry or "").strip()
        if saved:
            try:
                size_part, _, pos_part = saved.partition('+')
                w_s, _, h_s = size_part.partition('x')
                w = int(w_s); h = int(h_s)
                if pos_part:
                    x_s, _, y_s = pos_part.partition('+')
                    x = int(x_s); y = int(y_s)
                    # Reject off-screen saved positions
                    if (x < -80 or y < -40
                            or x + 120 > screen_w or y + 80 > screen_h):
                        raise ValueError("off-screen")
                    self.root.geometry(f"{w}x{h}+{x}+{y}")
                else:
                    self.root.geometry(f"{w}x{h}")
                restored = True
            except Exception:
                restored = False

        if not restored:
            width = min(self.root.winfo_width(), max(960, screen_w - 120))
            height = min(self.root.winfo_height(), max(720, screen_h - 120))
            x = max(24, (screen_w // 2) - (width // 2))
            y = max(24, (screen_h // 2) - (height // 2))
            self.root.geometry(f"{width}x{height}+{x}+{y}")

        logger.info(f"{APP_NAME} v{APP_VERSION} started")
        logger.info(f"Log file: {LOG_FILE}")
        self._queue_argv_files()
        if self.config.json_log_enabled:
            try:
                from backend.processor import attach_json_log
                json_path = str(LOG_DIR / "vsr_pro.jsonl")
                attach_json_log(json_path)
                logger.info(f"JSON log: {json_path}")
            except Exception as exc:
                logger.debug(f"JSON log setup failed: {exc}")
        if self.config.update_check:
            self._check_for_update()
        self.root.mainloop()


# =============================================================================
# ENTRY POINT
# =============================================================================

