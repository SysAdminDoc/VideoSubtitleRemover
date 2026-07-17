"""Main application class extracted from the monolith (RM-114)."""

from __future__ import annotations

import ctypes
import logging
import math
import os
import subprocess
import sys
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Optional

try:
    import tkinter as tk
    from tkinter import ttk, filedialog
except ImportError:
    pass

try:
    from PIL import Image, ImageTk
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

from gui.theme import (
    Theme, apply_high_contrast_theme, apply_default_theme, f, scaled_control_size, set_text_scale_percent, text_scale_factor,
    text_scale_percent,
)
from gui.config import (
    APP_NAME, APP_VERSION, LOG_DIR, LOG_FILE, InpaintMode, ProcessingStatus, ProcessingConfig, QueueItem,
    _coerce_int, _coerce_region_span_list,
    consume_settings_load_notice, load_settings, save_settings,
    list_presets, save_queue_state, load_queue_state, clear_queue_state,
)
from gui.utils import (
    get_app_dir, detect_gpu, is_video_file, is_image_file,
    _build_language_list,
    detect_ai_engines, detect_ffmpeg, get_file_info,
    _soft_subtitle_stream_record, _format_soft_subtitle_summary,
    truncate_middle,
)
from gui.widgets import (
    Tooltip, ModernButton, ModernProgressBar, ModernToggle,
    ModernSlider, show_confirm, TaskbarProgress, make_themed_menu,
    Toast, SegmentedPicker, DragDropFrame, QueueItemWidget,
    TextWidgetHandler,
)
from backend.ffmpeg_profiles import (
    FFMPEG_PROFILE_SCHEMA,
    collect_ffmpeg_capability_profiles,
    missing_profile_requirements_for_config,
    summarize_missing_profile_requirements,
)
from backend.model_downloads import installed_backend_status
from backend.a11y import set_accessible_metadata
from backend.i18n import available_catalogs, bind_locale, tr
from backend.region_keyframes import (
    normalize_region_keyframe_tracks,
    region_shapes_at,
    shape_bounds,
)
from gui.preview_controller import PreviewControllerMixin
from gui.mask_correction_controller import MaskCorrectionControllerMixin
from gui.processing_controller import ProcessingControllerMixin
from gui.quality_controller import QualityReviewControllerMixin
from gui.region_controller import RegionEditorControllerMixin
from gui.settings_controller import AdvancedSettingsControllerMixin
from gui.support_controller import SupportControllerMixin

logger = logging.getLogger(__name__)


class VideoSubtitleRemoverApp(
    RegionEditorControllerMixin,
    AdvancedSettingsControllerMixin,
    MaskCorrectionControllerMixin,
    PreviewControllerMixin,
    SupportControllerMixin,
    QualityReviewControllerMixin,
    ProcessingControllerMixin,
):
    """Main application class."""

    def __init__(self):
        # Resolve the palette before creating the root so the native root
        # surface and every child start in the same theme. Explicitly restore
        # defaults for embedders/tests that create more than one app instance
        # in a process after a high-contrast instance.
        self.config = load_settings()
        self._settings_load_notice = consume_settings_load_notice()
        bind_locale(getattr(self.config, "ui_locale", "system"))
        self._text_scale_percent = set_text_scale_percent(
            getattr(self.config, "text_scale_percent", 100))
        self._text_scale_factor = text_scale_factor()
        self._rtl_layout = bool(getattr(self.config, "rtl_layout", False))
        Theme.RTL_LAYOUT = self._rtl_layout
        if getattr(self.config, "high_contrast", False):
            apply_high_contrast_theme()
        else:
            apply_default_theme()

        self._background_ui = bool(
            os.environ.get("VSR_UI_BACKGROUND", "").strip().lower()
            in {"1", "true", "yes", "on"}
            or "--smoke-test" in sys.argv
            or "--uia-background" in sys.argv
        )
        self.root = tk.Tk()
        # Route unhandled Tk callback / .after exceptions through the logger
        # with a full traceback instead of Tk's default stderr dump, so a real
        # bug in a scheduled callback is never lost. Set before any widget or
        # .after work so early setup is covered too.
        self.root.report_callback_exception = self._log_callback_exception
        if self._background_ui:
            self.root.withdraw()
        self.root.title(f"{APP_NAME} v{APP_VERSION}")
        self.root.geometry("980x720" if self._background_ui else "1380x880")
        self.root.minsize(980, 720)
        self.root.configure(bg=Theme.BG_DARK)
        self._ui_resources_released = False
        self._running_mutex_handle = None
        if sys.platform == "win32":
            try:
                self._running_mutex_handle = ctypes.windll.kernel32.CreateMutexW(
                    None,
                    False,
                    "Local\\VideoSubtitleRemoverPro.Running",
                )
            except Exception:
                self._running_mutex_handle = None
        self.root.bind(
            "<Destroy>",
            self._on_root_destroyed,
            add="+",
        )

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
        # RM-98: RTL layout mirror -- set the Tk option DB before widgets
        # build so labels right-align and `pack(side="right")` becomes
        # the dominant orientation. Full pack-side flipping for every
        # widget is a much larger pass; this lands the framework.
        self.queue: List[QueueItem] = []
        self.queue_widgets: dict = {}
        self.is_processing = False
        self._stop_requested = False
        self._pause_requested = False
        self._processing_thread: Optional[threading.Thread] = None
        self.cancel_event = threading.Event()
        self.pause_event = threading.Event()
        self.queue_lock = threading.Lock()
        self.gpus = []
        self.ai_engines = {"detection": [], "inpainting": []}
        self.backend_status = {
            "schema": "vsr.backend_status.v1",
            "summary": {
                "detection": "Checking...",
                "inpainting": "Checking...",
                "providers": "Checking...",
                "language_support": "Checking...",
                "model_files": "Checking...",
                "hash_status": "Checking...",
                "next_action": tr("Backend status is still being probed."),
                "tone": "neutral",
            },
        }
        self.ffmpeg_ready = False
        self.ffmpeg_profiles = {
            "schema": FFMPEG_PROFILE_SCHEMA,
            "profiles": [],
        }
        self._hardware_probe_pending = True
        self._hardware_probe_thread: Optional[threading.Thread] = None
        self._elapsed_timer_id = None
        self._output_dir: Optional[Path] = None  # None = use input_dir/output/
        self._preview_detector = None  # cached SubtitleDetector for mask preview
        self._preview_detector_lang = None  # lang the cached detector was created with
        self._preview_detector_engine = None
        self._detector_lock = threading.Lock()  # serializes _preview_detector access
        self._preview_mask_cache = None
        self._preview_mask_render_after_id = None
        self._preview_mask_save_after_id = None
        self._cached_remover = None  # cached BackendRemover for batch reuse
        self._cached_remover_key = None  # (mode, device, lang) key for cache invalidation
        self._active_remover = None
        self._active_subprocess: Optional[subprocess.Popen] = None
        self._selected_queue_item_id: Optional[str] = None
        self._brand_photo = None
        self._status_tone = "neutral"
        self._shutdown_started = False
        self._taskbar = None  # created after the root is fully realized
        self._batch_times: List[float] = []  # seconds per item for ETA
        self._batch_started_at: Optional[datetime] = None
        self._batch_report_records: dict = {}
        self._last_batch_report_records: List[dict] = []
        self._model_download_guidance_seen: set = set()
        self._preview_request_id = 0
        self._preview_region_editor_state = None
        self._preview_region_drag_start = None
        self._preview_region_pending_rect = None
        self._throbber_id = None
        self._throbber_phase = 0
        self._layout_mode = "wide"
        self._workflow_pills = []
        self._settings_sliders: List[ModernSlider] = []
        self._settings_slider_by_attr: dict[str, tuple[ModernSlider, object]] = {}

        # Variables
        self.mode_var = tk.StringVar(value=self.config.mode.value)
        self.gpu_var = tk.StringVar(value="Detecting hardware...")
        self.skip_detection_var = tk.BooleanVar(value=self.config.sttn_skip_detection)
        self.lama_fast_var = tk.BooleanVar(value=self.config.lama_super_fast)
        self.preserve_audio_var = tk.BooleanVar(value=self.config.preserve_audio)
        self.lang_var = tk.StringVar(value=self.config.detection_lang)
        self._ocr_engine_by_label = {
            "Automatic (recommended)": "auto",
            "RapidOCR": "rapidocr",
            "OpenCV 5 DNN": "opencv-dnn",
            "PaddleOCR": "paddleocr",
            "EasyOCR": "easyocr",
            "OpenCV fallback": "opencv",
        }
        current_engine = getattr(self.config, "detection_engine", "auto")
        current_label = next(
            (label for label, value in self._ocr_engine_by_label.items()
             if value == current_engine),
            "Automatic (recommended)",
        )
        self.ocr_engine_var = tk.StringVar(value=current_label)

        # Build UI
        self._setup_styles()
        self._build_ui()
        self._apply_translation_safe_reflow()
        self._bind_shortcuts()
        self.root.bind("<Configure>", self._on_root_configure, add="+")

        # Attach log panel handler (tracks warn/error counts for badges)
        self._log_handler = TextWidgetHandler(self.log_text,
                                              on_count_change=self._update_log_badges)
        self._log_handler.setFormatter(logging.Formatter(
            '%(asctime)s [%(levelname)s] %(message)s', datefmt='%H:%M:%S'))
        logging.getLogger().addHandler(self._log_handler)

        self._update_output_label()
        self._update_region_label_display()
        self._update_status("Detecting hardware...", "info")
        self._surface_settings_load_notice()
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
        if not self._background_ui:
            self._start_startup_hardware_probe()
            self._maybe_show_onboarding()
            self.root.after(500, self._maybe_restore_queue)

    def _log_callback_exception(self, exc, val, tb):
        """Log unhandled Tk callback / ``.after`` exceptions with a traceback.

        Tk's default handler dumps these to stderr where they are easy to
        miss; routing them through the logger means a genuine bug in a
        scheduled callback surfaces in the log file. Teardown races -- the app
        being destroyed while a callback is still queued -- raise TclError or
        RuntimeError and are expected, so they are logged at debug rather than
        error.
        """
        if isinstance(exc, type) and issubclass(exc, (tk.TclError, RuntimeError)):
            logger.debug("Tk callback race during teardown",
                         exc_info=(exc, val, tb))
            return
        logger.error("Unhandled Tk callback exception",
                     exc_info=(exc, val, tb))

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
                self._terminate_active_backend_work()
                self._join_processing_thread(0.1)
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
        save_queue_state(self.queue)
        self._finish_close_when_safe(time.monotonic() + 2.0)

    def _finish_close_when_safe(self, deadline: float):
        """Wait briefly for active work to notice cancellation before exit."""
        if self._has_active_processing_thread():
            self._terminate_active_backend_work()
            self._join_processing_thread(0.05)
        if not self._has_active_processing_thread() or time.monotonic() >= deadline:
            self._join_processing_thread(0.2)
            save_queue_state(self.queue)
            self._shutdown_ui_resources()
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

    def _on_root_destroyed(self, event):
        if event.widget is self.root:
            self._shutdown_ui_resources()

    def _shutdown_ui_resources(self):
        """Release callbacks and handlers before the Tcl interpreter closes."""
        if getattr(self, "_ui_resources_released", False):
            return
        self._ui_resources_released = True
        self._shutdown_started = True
        self._preview_request_id = getattr(self, "_preview_request_id", 0) + 1

        handler = getattr(self, "_log_handler", None)
        if handler is not None:
            logging.getLogger().removeHandler(handler)
            try:
                handler.close()
            except Exception:
                pass
            self._log_handler = None

        root = getattr(self, "root", None)
        if root is not None:
            try:
                pending = root.tk.splitlist(root.tk.call("after", "info"))
                for callback_id in pending:
                    try:
                        root.tk.call("after", "cancel", callback_id)
                    except (tk.TclError, RuntimeError):
                        pass
            except (tk.TclError, RuntimeError):
                pass
        self._elapsed_timer_id = None
        self._throbber_id = None
        self._release_running_mutex()

    def _release_running_mutex(self):
        handle = getattr(self, "_running_mutex_handle", None)
        if not handle:
            return
        try:
            ctypes.windll.kernel32.CloseHandle(handle)
        except Exception:
            pass
        self._running_mutex_handle = None

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
        self.config.detection_engine = self._ocr_engine_by_label.get(
            self.ocr_engine_var.get(), "auto")
        if hasattr(self, "language_filter_var"):
            self.config.language_mask_filter = self.language_filter_var.get()
        # Threshold slider stores as int percent, convert to float
        pct = getattr(self.config, '_detection_threshold_pct', 50)
        self.config.detection_threshold = pct / 100.0
        # Time range
        self.config.time_start = self._safe_float(self.time_start_entry.get())
        self.config.time_end = self._safe_float(self.time_end_entry.get())
        # HW encode
        self.config.use_hw_encode = self.hw_encode_var.get()
        if hasattr(self, 'd3d12_accel_var'):
            self.config.d3d12_accel = self.d3d12_accel_var.get()
        # v3.9 quality + workflow toggles
        if hasattr(self, 'auto_band_var'):
            self.config.auto_band = self.auto_band_var.get()
        if hasattr(self, 'flow_warp_var'):
            self.config.tbe_flow_warp = self.flow_warp_var.get()
        if hasattr(self, 'scene_split_var'):
            self.config.tbe_scene_cut_split = self.scene_split_var.get()
        if hasattr(self, 'adaptive_batch_var'):
            self.config.adaptive_batch = self.adaptive_batch_var.get()
        if hasattr(self, 'temporal_mask_union_var'):
            self.config.temporal_mask_union = self.temporal_mask_union_var.get()
        if hasattr(self, 'export_srt_var'):
            self.config.export_srt = self.export_srt_var.get()
        if hasattr(self, 'translation_enabled_var'):
            self.config.translation_enabled = self.translation_enabled_var.get()
            self.config.translation_srt = self.translation_srt_var.get()
            self.config.translation_provider = self.translation_provider_var.get()
            self.config.translation_source_lang = (
                self.translation_source_lang_var.get())
            self.config.translation_target_lang = (
                self.translation_target_lang_var.get())
            self.config.translation_command = self.translation_command_var.get()
            self.config.translation_style = self.translation_style_var.get()
        if hasattr(self, 'export_mask_var'):
            self.config.export_mask_video = self.export_mask_var.get()
        if hasattr(self, 'mask_export_format_var'):
            self.config.mask_export_format = self.mask_export_format_var.get()
        if hasattr(self, 'mask_import_mode_var'):
            self.config.mask_import_mode = self.mask_import_mode_var.get()
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
        if hasattr(self, 'rife_stride_var'):
            self.config.rife_fast_stride = self._safe_int(
                self.rife_stride_var.get(), 0)
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
        if hasattr(self, 'text_scale_var'):
            self.config.text_scale_percent = _coerce_int(
                str(self.text_scale_var.get()).replace("%", ""),
                100,
                100,
                200,
            )
        if hasattr(self, 'locale_var'):
            self.config.ui_locale = self._locale_display_to_tag.get(
                self.locale_var.get(), "system")
        if hasattr(self, 'update_check_var'):
            self.config.update_check = self.update_check_var.get()
        if hasattr(self, 'json_log_var'):
            self.config.json_log_enabled = self.json_log_var.get()
        if hasattr(self, 'conf_dilate_var'):
            self.config.confidence_weighted_dilation = self.conf_dilate_var.get()
        if hasattr(self, 'work_dir_var'):
            self.config.work_directory = self.work_dir_var.get().strip()
        # GPU sync
        if self._hardware_probe_pending or not self.gpus:
            self.config.use_gpu = False
            return
        selection = self.gpu_var.get()
        for gpu in self.gpus:
            if f"{gpu['name']} ({gpu['memory']})" == selection:
                self.config.gpu_id = gpu['index']
                self.config.use_gpu = True
                break

    def _surface_settings_load_notice(self):
        notice = getattr(self, "_settings_load_notice", None)
        if not notice:
            return

        def _show_notice():
            self._update_status(notice, "warning", toast=True)

        try:
            self.root.after(350, _show_notice)
        except Exception:
            logger.warning(notice)

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
                if (
                    item.status == ProcessingStatus.IDLE
                    and not isinstance(getattr(item, "retry_config", None), dict)
                ):
                    item.config = ProcessingConfig.from_dict(snapshot.to_dict())
                    updated += 1
        output_updates = self._refresh_idle_output_paths()
        if output_updates:
            self._update_queue_display()
        if updated or output_updates:
            save_queue_state(self.queue)
        return updated

    def _apply_region_settings_to_idle_items(self) -> int:
        """Copy the current manual region fields into idle queue snapshots."""
        area = tuple(self.config.subtitle_area) if self.config.subtitle_area else None
        areas = (
            [tuple(region) for region in self.config.subtitle_areas]
            if self.config.subtitle_areas else None
        )
        spans = _coerce_region_span_list(
            getattr(self.config, "subtitle_region_spans", None))
        keyframe_tracks = normalize_region_keyframe_tracks(
            getattr(self.config, "subtitle_region_keyframes", None))
        updated = 0
        with self.queue_lock:
            for item in self.queue:
                if item.status == ProcessingStatus.IDLE:
                    item.config.subtitle_area = area
                    item.config.subtitle_areas = list(areas) if areas else None
                    item.config.subtitle_region_spans = (
                        list(spans) if spans else None
                    )
                    item.config.subtitle_region_keyframes = (
                        list(keyframe_tracks) if keyframe_tracks else None
                    )
                    item.config.normalized()
                    updated += 1
        if updated:
            save_queue_state(self.queue)
        return updated

    def _apply_translation_safe_reflow(self):
        """Wrap verbose labels and stack crowded Canvas-button rows."""
        wrap_limit = 360 if self._text_scale_percent >= 150 else 520
        anchor = "e" if self._rtl_layout else "w"
        justify = "right" if self._rtl_layout else "left"

        def walk(widget):
            yield widget
            for child in widget.winfo_children():
                yield from walk(child)

        widgets = list(walk(self.root))
        for widget in widgets:
            if isinstance(widget, tk.Label):
                try:
                    text = str(widget.cget("text") or "")
                    if self._rtl_layout:
                        widget.configure(anchor=anchor, justify=justify)
                    if len(text) >= 24:
                        current = int(float(str(widget.cget("wraplength") or 0)))
                        if current <= 0 or current > wrap_limit:
                            widget.configure(wraplength=wrap_limit)
                except (tk.TclError, TypeError, ValueError):
                    pass

        if self._text_scale_percent < 150:
            return
        for widget in widgets:
            buttons = [
                child for child in widget.winfo_children()
                if isinstance(child, ModernButton)
                and child.winfo_manager() == "pack"
            ]
            if len(buttons) < 2:
                continue
            required_width = sum(button.winfo_reqwidth() for button in buttons)
            required_width += Theme.S_SM * (len(buttons) - 1)
            if required_width <= 760:
                continue
            for index, button in enumerate(buttons):
                button.pack_forget()
                button.pack(
                    anchor=anchor,
                    pady=(0 if index == 0 else Theme.S_XS, 0),
                )

        if hasattr(self, "_header_chips"):
            self._header_chips.pack_forget()
        if self._text_scale_percent >= 200 and hasattr(
            self, "_header_guidance_panel"
        ):
            self._header_guidance_panel.pack_forget()
        if getattr(self, "_log_visible", False):
            self._toggle_log_panel()

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
                       padding=(scaled_control_size(10), scaled_control_size(6)))

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
                        width=scaled_control_size(10))
        style.map("Dark.Vertical.TScrollbar",
                 background=[('active', Theme.BORDER_STRONG),
                             ('pressed', Theme.BORDER_STRONG)],
                 arrowcolor=[('active', Theme.TEXT_SECONDARY)])

    def _create_surface(self, parent, bg: str = Theme.BG_SECONDARY) -> tk.Frame:
        """Create a quiet tonal surface without an ornamental outline."""
        return tk.Frame(parent, bg=bg, highlightthickness=0)

    def _create_status_tile(self, parent, label: str, value: str, fg: str,
                            bg: str) -> tk.Frame:
        """Create an inline status summary for compatibility surfaces."""
        tile = tk.Frame(parent, bg=bg, highlightthickness=0)
        inner = tk.Frame(tile, bg=bg)
        inner.pack(fill="x")
        tk.Label(
            inner,
            text=label,
            font=f(Theme.F_META, "bold"),
            bg=bg,
            fg=fg,
        ).pack(side="left")
        tk.Label(
            inner,
            text=f"  {value}",
            font=f(Theme.F_META),
            bg=bg,
            fg=Theme.TEXT_MUTED,
        ).pack(side="left")
        return tile

    @staticmethod
    def _fallback_ai_engines() -> dict:
        return {
            "detection": ["OpenCV fallback"],
            "inpainting": ["Temporal BG (TBE)", "OpenCV"],
        }

    def _start_startup_hardware_probe(self):
        """Run slow startup hardware probes off the Tk main thread."""
        if self._hardware_probe_thread and self._hardware_probe_thread.is_alive():
            return
        self._hardware_probe_thread = threading.Thread(
            target=self._probe_startup_hardware,
            daemon=True,
            name="startup-hardware-probe",
        )
        self._hardware_probe_thread.start()

    def _probe_startup_hardware(self):
        """Collect startup hardware facts, then marshal results to Tk."""
        try:
            gpus = detect_gpu()
        except Exception:
            logger.warning("Startup GPU probe failed", exc_info=True)
            gpus = []
        try:
            ai_engines = detect_ai_engines()
        except Exception:
            logger.warning("Startup AI engine probe failed", exc_info=True)
            ai_engines = self._fallback_ai_engines()
        try:
            ffmpeg_ready = detect_ffmpeg()
        except Exception:
            logger.warning("Startup FFmpeg probe failed", exc_info=True)
            ffmpeg_ready = False
        try:
            ffmpeg_profiles = collect_ffmpeg_capability_profiles(timeout=8.0)
        except Exception:
            logger.warning("Startup FFmpeg profile probe failed", exc_info=True)
            ffmpeg_profiles = {
                "schema": FFMPEG_PROFILE_SCHEMA,
                "profiles": [],
            }
        try:
            backend_status = installed_backend_status(getattr(self, "config", None))
        except Exception:
            logger.warning("Startup backend status probe failed", exc_info=True)
            backend_status = {
                "schema": "vsr.backend_status.v1",
                "summary": {
                    "detection": "Unavailable",
                    "inpainting": "Unavailable",
                    "providers": "Unavailable",
                    "language_support": "Unavailable",
                    "model_files": "Unavailable",
                    "hash_status": "Unavailable",
                    "next_action": "Open the support bundle for raw diagnostics.",
                    "tone": "warning",
                },
            }

        try:
            self.root.after(
                0,
                lambda: self._apply_startup_hardware_probe(
                    gpus, ai_engines, ffmpeg_ready, backend_status,
                    ffmpeg_profiles
                ),
            )
        except (tk.TclError, RuntimeError):
            pass

    def _apply_startup_hardware_probe(
        self,
        gpus,
        ai_engines,
        ffmpeg_ready,
        backend_status=None,
        ffmpeg_profiles=None,
    ):
        """Apply background probe results on the Tk main thread."""
        if self._shutdown_started:
            return
        self.gpus = list(gpus or [])
        self.ai_engines = ai_engines or self._fallback_ai_engines()
        if backend_status:
            self.backend_status = backend_status
        if ffmpeg_profiles:
            self.ffmpeg_profiles = ffmpeg_profiles
        self.ffmpeg_ready = bool(ffmpeg_ready)
        self._hardware_probe_pending = False
        self._apply_gpu_selection_from_config()
        self._refresh_gpu_selector()
        self._render_header_chips()
        self._refresh_ffmpeg_warning()
        self._refresh_action_states()

        gpu_label = self.gpus[0]["name"] if self.gpus else "CPU mode"
        audio_label = "FFmpeg ready" if self.ffmpeg_ready else "FFmpeg missing"
        self._update_status(f"Hardware detected: {gpu_label}; {audio_label}", "info")
        logger.info(
            "Startup hardware probe complete: gpus=%s detection=%s inpainting=%s ffmpeg=%s",
            len(self.gpus),
            self.ai_engines.get("detection", []),
            self.ai_engines.get("inpainting", []),
            self.ffmpeg_ready,
        )

    def _apply_gpu_selection_from_config(self):
        """Restore the saved GPU choice after async hardware detection."""
        if not self.gpus:
            self.gpu_var.set("CPU Mode")
            self.config.use_gpu = False
            return
        selected = None
        for gpu in self.gpus:
            if gpu["index"] == self.config.gpu_id:
                selected = gpu
                break
        if selected is None:
            selected = self.gpus[0]
            self.config.gpu_id = selected["index"]
        self.gpu_var.set(f"{selected['name']} ({selected['memory']})")
        self.config.use_gpu = True

    def _refresh_gpu_selector(self):
        """Refresh the compute-device combobox after async probing."""
        if not hasattr(self, "gpu_combo"):
            return
        if self._hardware_probe_pending:
            self.gpu_combo.configure(values=["Detecting hardware..."], state="disabled")
            self.gpu_var.set("Detecting hardware...")
            return
        if self.gpus:
            options = [f"{g['name']} ({g['memory']})" for g in self.gpus]
            self.gpu_combo.configure(values=options, state="readonly")
        else:
            self.gpu_combo.configure(values=["CPU Mode"], state="disabled")
            self.gpu_var.set("CPU Mode")

    def _refresh_ffmpeg_warning(self):
        """Show FFmpeg pending/missing state without rebuilding the settings card."""
        if not hasattr(self, "ffmpeg_warning_label"):
            return
        if self._hardware_probe_pending:
            self.ffmpeg_warning_label.config(
                text=tr("Checking FFmpeg availability for audio preservation..."),
                fg=Theme.INFO,
            )
            if not self.ffmpeg_warning_label.winfo_ismapped():
                self.ffmpeg_warning_label.pack(anchor="w", pady=(Theme.S_XS, 0))
        elif not self.ffmpeg_ready:
            self.ffmpeg_warning_label.config(
                text=tr("FFmpeg is not available, so outputs will be saved without original audio until it is installed."),
                fg=Theme.WARNING,
            )
            if not self.ffmpeg_warning_label.winfo_ismapped():
                self.ffmpeg_warning_label.pack(anchor="w", pady=(Theme.S_XS, 0))
        else:
            self.ffmpeg_warning_label.pack_forget()

    def _render_header_chips(self):
        """Render one readiness state and one quiet capability summary."""
        if not hasattr(self, "_header_chips"):
            return
        for child in self._header_chips.winfo_children():
            child.destroy()
        if self._hardware_probe_pending:
            state_text = tr("Checking")
            state_fg = Theme.INFO
            state_bg = Theme.INFO_BG
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
            state_bg = Theme.SUCCESS_BG if ready else Theme.WARNING_BG
            summary = f"{gpu_short}  /  {det_short}  /  {audio_short}"

        self._header_ready_label = tk.Label(
            self._header_chips,
            text=state_text,
            font=f(Theme.F_META, "bold"),
            bg=state_bg,
            fg=state_fg,
            padx=Theme.S_SM,
            pady=2,
        )
        self._header_ready_label.pack(side="left")
        self._header_capability_label = tk.Label(
            self._header_chips,
            text=summary,
            font=f(Theme.F_META),
            bg=Theme.BG_SECONDARY,
            fg=Theme.TEXT_MUTED,
            wraplength=420,
            justify="left",
        )
        if getattr(self, "_layout_mode", "wide") == "wide":
            self._header_capability_label.pack(
                side="left", padx=(Theme.S_SM, 0))

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
        else:
            self.output_dir_label.config(text=tr("Auto-create an output folder beside each source"),
                                         fg=Theme.TEXT_PRIMARY)
            self.output_dir_meta.config(text=tr("Default workflow"))

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

    @staticmethod
    def _active_timed_region_rects(
        spans,
        seconds: float = 0.0,
        keyframe_tracks=None,
    ):
        """Return timed region rects active at ``seconds`` in GUI config shape."""
        normalized = _coerce_region_span_list(spans) or []
        try:
            current = float(seconds)
        except (TypeError, ValueError):
            current = 0.0
        if not math.isfinite(current) or current < 0.0:
            current = 0.0
        rects = []
        for span in normalized:
            start = float(span.get("start", 0.0) or 0.0)
            end = float(span.get("end", 0.0) or 0.0)
            if start <= current and (end <= 0.0 or current < end):
                rects.append(tuple(span["rect"]))
        for shape in region_shapes_at(keyframe_tracks, current):
            bounds = shape_bounds(shape)
            if bounds is not None:
                rects.append(bounds)
        return rects

    @staticmethod
    def _hex_to_rgb(hex_str: str):
        hex_str = hex_str.lstrip('#')
        return tuple(int(hex_str[i:i + 2], 16) for i in (0, 2, 4))

    def _set_selected_queue_item(self, item_id: Optional[str]):
        """Update queue item selection state."""
        region_state = getattr(self, "_preview_region_editor_state", None)
        if region_state and region_state.get("item_id") != item_id:
            self._clear_preview_region_editor()
        self._selected_queue_item_id = item_id
        for wid, widget in self.queue_widgets.items():
            widget.set_selected(wid == item_id)
        self._update_preview_actions()
        self._update_guidance_surface()

    def _refresh_action_states(self):
        """Enable or disable primary queue actions based on current state."""
        has_queue = bool(self.queue)
        has_complete = any(item.status == ProcessingStatus.COMPLETE for item in self.queue)
        has_retry = any(item.status in (ProcessingStatus.ERROR, ProcessingStatus.CANCELLED)
                        for item in self.queue)
        active_thread = self._has_active_processing_thread()
        batch_busy = self.is_processing or active_thread

        if hasattr(self, "start_btn"):
            can_pause = (
                active_thread
                and not self._stop_requested
                and not self._pause_requested
            )
            can_start = (not batch_busy) and has_queue
            if can_pause or can_start:
                start_reason = ""
            elif not has_queue:
                start_reason = tr("Add media to the queue before starting.")
            elif self._pause_requested:
                start_reason = tr("Pause is already in progress.")
            elif self._stop_requested:
                start_reason = tr("Stop is already in progress.")
            else:
                start_reason = tr("The batch is already running.")
            for button_name in ("start_btn", "inspector_start_btn"):
                button = getattr(self, button_name, None)
                if button is not None:
                    button.set_enabled(
                        can_pause or can_start, reason=start_reason)
        if hasattr(self, "open_output_btn"):
            self.open_output_btn.set_enabled(
                has_complete,
                reason=tr("Finish processing an item to open its output folder."),
            )
        if hasattr(self, "retry_btn"):
            self.retry_btn.set_enabled(
                (not batch_busy) and has_retry,
                reason=(
                    tr("Wait for the current batch to finish.")
                    if batch_busy else tr("Nothing to retry - every item is done.")
                ),
            )
        if hasattr(self, "clear_btn"):
            self.clear_btn.set_enabled(
                (not batch_busy) and has_queue,
                reason=(
                    tr("Wait for the current batch to finish.")
                    if batch_busy else tr("The queue is empty - add media to get started.")
                ),
            )
        if hasattr(self, "repeat_btn"):
            self.repeat_btn.set_enabled(
                (not batch_busy) and self._last_completed_config() is not None,
                reason=(
                    tr("Wait for the current batch to finish.")
                    if batch_busy else tr("Run a batch first to repeat its settings.")
                ),
            )
        if hasattr(self, "batch_label") and not batch_busy:
            pending = sum(1 for item in self.queue if item.status == ProcessingStatus.IDLE)
            if pending:
                self.batch_label.config(
                    text=tr("{count} queued and ready to process").format(count=pending),
                    fg=Theme.TEXT_SECONDARY,
                )
            elif has_complete:
                self.batch_label.config(
                    text=tr("Outputs are ready for review"),
                    fg=Theme.SUCCESS,
                )
            elif has_retry:
                self.batch_label.config(
                    text=tr("Some items need attention"),
                    fg=Theme.WARNING,
                )
            else:
                self.batch_label.config(text=tr("Ready"), fg=Theme.TEXT_MUTED)
        self._update_preview_actions()
        self._update_guidance_surface()

    def _bind_shortcuts(self):
        """No app-wide keyboard accelerators -- all actions are click-only."""
        pass

    def _open_file_picker(self):
        if hasattr(self, "drop_area"):
            self.drop_area._open_file_dialog()

    def _open_folder_picker(self):
        if hasattr(self, "drop_area"):
            self.drop_area._open_folder_dialog()

    def _focus_queue_filter(self, event=None):
        if len(self.queue) < 6 or not hasattr(self, "_queue_filter_entry"):
            return "break"
        try:
            self._queue_filter_frame.pack(
                fill="x", padx=Theme.S_MD, pady=(0, Theme.S_SM),
                before=self._queue_container)
            self._queue_filter_entry.focus_set()
            self._queue_filter_entry.selection_range(0, "end")
        except tk.TclError:
            pass
        return "break"

    def _on_content_configure(self, event):
        """Keep the middle workbench scrollable when settings exceed the viewport."""
        if hasattr(self, "_content_canvas"):
            self._content_canvas.configure(
                scrollregion=self._content_canvas.bbox("all"))

    def _on_content_canvas_configure(self, event):
        """Lock the scrollable content frame to the canvas width."""
        if hasattr(self, "_content_window"):
            self._content_canvas.itemconfig(self._content_window, width=event.width)
            self._content_canvas.configure(
                scrollregion=self._content_canvas.bbox("all"))

    def _on_content_mousewheel(self, event):
        """Scroll the workbench unless the content already fits."""
        if not hasattr(self, "_content_canvas"):
            return
        bbox = self._content_canvas.bbox("all")
        if not bbox:
            return
        if bbox[3] <= self._content_canvas.winfo_height():
            return
        self._content_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    def _on_root_configure(self, event):
        """Keep layout responsive as the window width changes."""
        if event.widget is not self.root:
            return
        self._apply_responsive_layout(event.width)

    def _apply_responsive_layout(self, width: int):
        """Reflow the workbench without changing its workflow order."""
        if not hasattr(self, "_content"):
            return
        if width < 1180 and getattr(self, "_log_visible", False):
            self._toggle_log_panel()
        compact = width < 1180 or self._text_scale_percent >= 150
        if hasattr(self, "_header_title_label"):
            title_wrap = (
                1200 if self._text_scale_percent >= 150
                else (520 if compact else 760)
            )
            self._header_title_label.configure(wraplength=title_wrap)
            self._header_version_label.pack_forget()
            self._header_intro_label.pack_forget()
        if hasattr(self, "_log_title_cluster"):
            self._log_title_cluster.pack_forget()
            self._badge_row.pack_forget()
            self._log_open_btn.pack_forget()
            self._log_clear_btn.pack_forget()
            self._log_toggle_btn.pack_forget()
            if compact:
                self._log_toggle_btn.pack(side="left")
            else:
                self._log_title_cluster.pack(side="left")
                self._badge_row.pack(side="left", padx=(Theme.S_MD, 0))
                self._log_toggle_btn.pack(side="left", padx=(Theme.S_MD, 0))
                self._log_open_btn.pack(side="right")
                self._log_clear_btn.pack(side="right", padx=(0, Theme.S_SM))

        mode = "stacked" if compact else "wide"
        if mode == self._layout_mode:
            self._layout_queue_actions(
                compact=compact,
                dense=self._text_scale_percent >= 150,
            )
            if hasattr(self, "preview_meta_label"):
                self.preview_meta_label.config(
                    wraplength=720 if mode == "stacked" else 520)
            if hasattr(self, "preview_action_hint"):
                self.preview_action_hint.config(
                    wraplength=720 if mode == "stacked" else 520)
            if hasattr(self, "status_hint"):
                self.status_hint.config(
                    wraplength=520 if mode == "stacked" else 360)
            return

        self._layout_mode = mode
        stacked = (mode == "stacked")
        self._layout_queue_actions(
            compact=stacked,
            dense=self._text_scale_percent >= 150,
        )

        self._workflow_col.grid_forget()
        self._preview_col.grid_forget()
        self._settings_col.grid_forget()

        self._header_left.pack_forget()
        self._header_right.pack_forget()
        self._header_chips.pack_forget()

        if stacked:
            self._content.columnconfigure(0, weight=1, minsize=0, uniform="")
            self._content.columnconfigure(1, weight=0, minsize=0, uniform="")
            self._content.columnconfigure(2, weight=0, minsize=0, uniform="")
            for row in range(2):
                self._content.rowconfigure(row, weight=0)
            self._content.rowconfigure(0, weight=1)
            self._preview_col.grid(row=0, column=0, sticky="nsew",
                                   pady=(0, Theme.S_MD))
            self._settings_col.grid(row=1, column=0, sticky="nsew",
                                    pady=(0, Theme.S_MD))

            self._header_left.pack(side="left", fill="y")
            self._header_right.pack(side="right", anchor="n")
            if self._text_scale_percent < 150:
                self._header_chips.pack(
                    side="right", padx=(Theme.S_SM, Theme.S_LG))

            self._footer_left.pack_forget()
            self._footer_left.pack(anchor="w")
            self.status_hint.pack_forget()
            if width >= 1180:
                self.status_hint.pack(fill="x", pady=(Theme.S_XS, 0))
        else:
            self._content.columnconfigure(0, weight=7, minsize=500, uniform="")
            self._content.columnconfigure(1, weight=4, minsize=360, uniform="")
            self._content.columnconfigure(2, weight=0, minsize=0, uniform="")
            self._content.rowconfigure(0, weight=1)
            self._content.rowconfigure(1, weight=0)
            self._content.rowconfigure(2, weight=0)
            self._preview_col.grid(row=0, column=0, sticky="nsew",
                                   padx=(0, Theme.S_MD))
            self._settings_col.grid(row=0, column=1, sticky="nsew")

            self._header_left.pack(side="left", fill="y")
            self._header_right.pack(side="right", anchor="n")
            if width >= 1180 and self._text_scale_percent < 150:
                self._header_chips.pack(side="right",
                                        padx=(Theme.S_XL, Theme.S_LG))

            self._footer_left.pack_forget()
            self._footer_left.pack(side="left")
            self.status_hint.pack_forget()
            self.status_hint.pack(side="right")

        self.preview_meta_label.config(wraplength=720 if stacked else 520)
        if hasattr(self, "preview_action_hint"):
            self.preview_action_hint.config(wraplength=720 if stacked else 520)
        self.status_hint.config(wraplength=520 if stacked else 360)
        self._render_header_chips()

    def _layout_workflow_rail(self, *, compact: bool):
        """Switch the workflow rail between horizontal and vertical forms."""
        if not hasattr(self, "_workflow_step_blocks"):
            return
        for block in self._workflow_step_blocks:
            block.pack_forget()
        for connector in self._workflow_connectors:
            connector.pack_forget()

        if compact:
            for index, block in enumerate(self._workflow_step_blocks):
                block.pack(
                    side="left", fill="x", expand=True,
                    padx=(0 if index == 0 else Theme.S_SM, 0),
                )
                description = self._workflow_pills[index].get("description")
                if description is not None:
                    description.pack_forget()
            self._header_guidance_panel.pack_forget()
        else:
            for index, block in enumerate(self._workflow_step_blocks):
                block.pack(fill="x")
                description = self._workflow_pills[index].get("description")
                if description is not None and not description.winfo_manager():
                    description.pack(anchor="w", pady=(2, 0))
                if index < len(self._workflow_connectors):
                    self._workflow_connectors[index].pack(
                        anchor="w", padx=14, pady=3)
            if not self._header_guidance_panel.winfo_manager():
                self._header_guidance_panel.pack(
                    side="bottom", fill="x", padx=Theme.S_MD,
                    pady=(Theme.S_XL, Theme.S_MD))

    def _layout_queue_actions(self, *, compact: bool, dense: bool):
        """Keep primary queue controls visible at narrow or scaled layouts."""
        if not hasattr(self, "_queue_action_frame"):
            return
        for button in (
            self.start_btn, self.open_output_btn, self.retry_btn,
            self.repeat_btn, self.clear_btn, self._queue_more_btn,
        ):
            button.pack_forget()

        if compact or not hasattr(self, "inspector_start_btn"):
            self.start_btn.pack(side="right")
        if hasattr(self, "inspector_start_btn"):
            self.inspector_start_btn.pack_forget()
            if not compact:
                self.inspector_start_btn.pack(fill="x")
        self.open_output_btn.pack(side="right", padx=(0, Theme.S_SM))
        if compact:
            self._queue_more_btn.pack(side="left")
        else:
            self.retry_btn.pack(side="left")
            self.repeat_btn.pack(side="left", padx=(Theme.S_SM, 0))
            self.clear_btn.pack(side="left", padx=(Theme.S_SM, 0))

        self.queue_canvas.configure(height=64 if dense else 88)
        self._queue_dense_mode = dense
        self._queue_subtitle_label.pack_forget()
        if dense:
            self._queue_batch_frame.pack_forget()
            self._queue_batch_bar_frame.pack_forget()
        show_batch_progress = bool(self.queue or self.is_processing)
        if not dense and show_batch_progress:
            if not self._queue_batch_frame.winfo_manager():
                self._queue_batch_frame.pack(
                    fill="x", padx=Theme.S_MD, pady=(Theme.S_SM, 0),
                    before=self._queue_container,
                )
            if not self._queue_batch_bar_frame.winfo_manager():
                self._queue_batch_bar_frame.pack(
                    fill="x", padx=Theme.S_MD,
                    pady=(4, Theme.S_SM), before=self._queue_container,
                )
        elif not show_batch_progress:
            self._queue_batch_frame.pack_forget()
            self._queue_batch_bar_frame.pack_forget()
        dense_expanded = getattr(self, "_queue_dense_expanded", False)
        if dense and not dense_expanded:
            self._queue_container.pack_forget()
        elif not self._queue_container.winfo_manager():
            self._queue_container.pack(
                fill="both", expand=True, padx=Theme.S_MD,
                pady=(0, Theme.S_SM), before=self._queue_action_frame,
            )

    def _open_queue_actions_menu(self):
        """Expose less frequent queue commands in compact layouts."""
        menu = tk.Menu(
            self.root, tearoff=False,
            bg=Theme.BG_RAISED, fg=Theme.TEXT_PRIMARY,
            activebackground=Theme.BLUE_MUTED,
            activeforeground=Theme.TEXT_PRIMARY,
            bd=1, relief="solid",
        )
        for label, command, enabled in (
            (tr("Retry failed"), self._retry_failed, self.retry_btn.enabled),
            (tr("Repeat last"), self._repeat_last_job, self.repeat_btn.enabled),
            (tr("Clear queue"), self._clear_queue, self.clear_btn.enabled),
        ):
            menu.add_command(
                label=label, command=command,
                state="normal" if enabled else "disabled",
            )
        if getattr(self, "_queue_dense_mode", False):
            menu.add_separator()
            menu.add_command(
                label=(
                    tr("Hide queued files")
                    if getattr(self, "_queue_dense_expanded", False)
                    else tr("Show queued files")
                ),
                command=self._toggle_dense_queue_list,
            )
        try:
            menu.tk_popup(
                self._queue_more_btn.winfo_rootx(),
                self._queue_more_btn.winfo_rooty()
                + self._queue_more_btn.winfo_height(),
            )
        finally:
            menu.grab_release()

    def _toggle_dense_queue_list(self):
        """Expand or collapse the queue list at very large text scales."""
        self._queue_dense_expanded = not getattr(
            self, "_queue_dense_expanded", False)
        self._layout_queue_actions(compact=True, dense=True)

    def _get_selected_queue_item(self, fallback_to_first: bool = False) -> Optional[QueueItem]:
        """Return the selected queue item, optionally falling back to the first item."""
        if self._selected_queue_item_id:
            selected = next(
                (item for item in self.queue if item.id == self._selected_queue_item_id),
                None,
            )
            if selected is not None:
                return selected
        if fallback_to_first:
            return next(iter(self.queue), None)
        return None

    def _queue_item_by_id(self, item_id: str) -> Optional[QueueItem]:
        return next((item for item in self.queue if item.id == item_id), None)

    def _set_workflow_stage(self, stage: int):
        """Update the compact workflow pills in the header."""
        for idx, pill in enumerate(self._workflow_pills, start=1):
            if idx < stage:
                frame_border = Theme.GREEN_HOVER
                badge_bg = Theme.GREEN_PRIMARY
                badge_fg = Theme.INK_ON_GREEN
                text_fg = Theme.SUCCESS
            elif idx == stage:
                frame_border = Theme.BLUE_PRIMARY
                badge_bg = Theme.BLUE_PRIMARY
                badge_fg = Theme.INK_ON_BLUE
                text_fg = Theme.TEXT_PRIMARY
            else:
                frame_border = Theme.BORDER
                badge_bg = Theme.BG_TERTIARY
                badge_fg = Theme.TEXT_MUTED
                text_fg = Theme.TEXT_SECONDARY
            pill["frame"].config(bg=Theme.BG_SECONDARY)
            pill["badge"].config(
                bg=badge_bg, fg=badge_fg, highlightbackground=frame_border)
            pill["text"].config(bg=Theme.BG_SECONDARY, fg=text_fg)
            description = pill.get("description")
            if description is not None:
                description.config(
                    fg=text_fg if idx == stage else Theme.TEXT_MUTED)

    def _update_guidance_surface(self):
        """Keep the header guidance card and footer hint aligned with state."""
        if not hasattr(self, "header_guidance_title"):
            return

        selected = self._get_selected_queue_item()
        has_queue = bool(self.queue)
        has_complete = any(item.status == ProcessingStatus.COMPLETE for item in self.queue)
        has_retry = any(item.status in (ProcessingStatus.ERROR, ProcessingStatus.CANCELLED)
                        for item in self.queue)
        has_paused = any(item.status == ProcessingStatus.PAUSED for item in self.queue)

        if self._pause_requested:
            stage = 3
            title = "Pausing batch"
            body = ("The current item is saving a checkpoint at the next safe frame boundary. "
                    "Resume later without restarting that video.")
            hint = "Pausing safely. Keep the app open until the current checkpoint is written."
        elif self._stop_requested:
            stage = 3
            title = "Stopping batch"
            body = ("The current item is wrapping up so the app can stop cleanly without risking overlapping work. "
                    "Finished outputs stay on disk and remaining items will be marked as stopped.")
            hint = "Stopping safely. Please wait for the current item to finish its active step."
        elif self.is_processing:
            stage = 3
            title = "Batch running"
            body = ("Live preview, ETA, and the activity log stay up to date while the batch works. "
                    "Pause is safe: completed outputs stay on disk and the current video gets a checkpoint.")
            hint = "Use Pause batch if you need to step away. Resume continues from the checkpoint."
        elif not has_queue:
            stage = 1
            title = "Build your batch"
            body = "Import files or choose a folder to start."
            hint = "Import files or choose a folder to start."
        elif has_paused:
            stage = 2
            title = "Resume paused work"
            body = "Paused videos keep their checkpoint frames and continue from the first missing frame."
            hint = "Start batch resumes paused items and continues the queue."
        elif has_retry:
            stage = 3 if has_complete else 2
            title = "Review the outliers"
            body = "Retry failed items or open the log for details."
            hint = "Retry failed items or open the log for details."
        elif not selected:
            stage = 2
            title = "Inspect a sample frame"
            body = "Click a queue item, then use Set region or Review mask."
            hint = "Click a queue item, then use Set region or Review mask."
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
        self.header_guidance_title.config(text=tr(title))
        self.header_guidance_body.config(text=tr(body))
        if hasattr(self, "status_hint"):
            self.status_hint.config(text=tr(hint))

    def _build_ui(self):
        """Build a preview-first workbench with a dedicated inspector and queue."""
        main_container = tk.Frame(self.root, bg=Theme.BG_DARK)
        main_container.pack(fill="both", expand=True,
                            padx=Theme.S_LG, pady=(Theme.S_MD, Theme.S_SM))

        # Header
        self._build_header(main_container)

        # Reserve persistent bottom surfaces before the expanding workbench so
        # they cannot be clipped at the minimum 980x720 viewport.
        self._build_footer(main_container)
        self._build_log_panel(main_container)

        # Keep the operational queue and primary action visible while the
        # preview/inspector workbench scrolls independently above it.
        queue_row = tk.Frame(main_container, bg=Theme.BG_DARK)
        queue_row.pack(side="bottom", fill="x", pady=(Theme.S_MD, 0))
        self._queue_row = queue_row
        self._build_queue_section(queue_row)

        # A single scroll surface keeps the three-part workbench usable at the
        # 980x720 minimum without compromising the desktop hierarchy.
        content_shell = tk.Frame(main_container, bg=Theme.BG_DARK)
        content_shell.pack(fill="both", expand=True, pady=(Theme.S_MD, 0))
        self._content_canvas = tk.Canvas(
            content_shell, bg=Theme.BG_DARK, highlightthickness=0)
        content_scroll = ttk.Scrollbar(
            content_shell, orient="vertical",
            command=self._content_canvas.yview,
            style="Dark.Vertical.TScrollbar")
        self._content_canvas.configure(yscrollcommand=content_scroll.set)
        content_scroll.pack(side="right", fill="y")
        self._content_canvas.pack(side="left", fill="both", expand=True)

        content = tk.Frame(self._content_canvas, bg=Theme.BG_DARK)
        self._content_window = self._content_canvas.create_window(
            (0, 0), window=content, anchor="nw")
        content.bind("<Configure>", self._on_content_configure)
        self._content_canvas.bind("<Configure>", self._on_content_canvas_configure)
        self._content_canvas.bind("<MouseWheel>", self._on_content_mousewheel)
        content.bind("<MouseWheel>", self._on_content_mousewheel)
        content.columnconfigure(0, weight=7, minsize=500)
        content.columnconfigure(1, weight=4, minsize=360)
        content.columnconfigure(2, weight=0, minsize=0)
        content.rowconfigure(0, weight=1)
        self._content = content

        # Keep the legacy workflow state widgets alive for status/a11y updates,
        # but do not spend viewport space on an onboarding rail.
        workflow_col = tk.Frame(content, bg=Theme.BG_DARK)
        self._workflow_col = workflow_col
        self._build_workflow_rail(workflow_col)

        # Preview is the primary work surface. Import remains immediately
        # discoverable above it, but no longer competes with configuration.
        preview_col = tk.Frame(content, bg=Theme.BG_DARK)
        preview_col.grid(row=0, column=0, sticky="nsew",
                         padx=(0, Theme.S_MD))
        self._preview_col = preview_col
        self._build_input_section(preview_col)
        self._build_preview_section(preview_col)

        # Focused inspector: cleanup profile, region, output, then details.
        settings_col = tk.Frame(content, bg=Theme.BG_DARK)
        settings_col.grid(row=0, column=1, sticky="nsew")
        self._settings_col = settings_col
        self._build_settings_section(settings_col)


    def _build_header(self, parent):
        """Compact command bar with product identity and live readiness signals."""
        header = self._create_surface(parent)
        header.pack(fill="x")

        inner = tk.Frame(header, bg=Theme.BG_SECONDARY)
        inner.pack(fill="x", padx=Theme.S_LG, pady=Theme.S_SM)

        header_top = tk.Frame(inner, bg=Theme.BG_SECONDARY)
        header_top.pack(fill="x")

        left = tk.Frame(header_top, bg=Theme.BG_SECONDARY)
        left.pack(side="left", fill="y")
        self._header_left = left

        self._header_title_label = tk.Label(
            left,
            text=tr("Video Subtitle Remover"),
            font=f(Theme.F_DISPLAY, "bold"),
            bg=Theme.BG_SECONDARY,
            fg=Theme.TEXT_PRIMARY,
        )
        self._header_title_label.pack(side="left", anchor="w")
        Tooltip(self._header_title_label, f"Video Subtitle Remover v{APP_VERSION}")
        self._header_version_label = tk.Label(
            left,
            text=f"v{APP_VERSION}",
            font=f(Theme.F_META, "bold"),
            bg=Theme.BG_TERTIARY,
            fg=Theme.TEXT_SECONDARY,
            padx=8,
            pady=3,
        )
        self._header_intro_label = tk.Label(
            left,
            text=tr("Private, local cleanup"),
            font=f(Theme.F_META, "bold"),
            bg=Theme.BG_SECONDARY,
            fg=Theme.GREEN_PRIMARY,
        )

        right = tk.Frame(header_top, bg=Theme.BG_SECONDARY)
        right.pack(side="right", anchor="n")
        self._header_right = right

        settings_btn = ModernButton(
            right, text=tr("Settings"), width=92,
            command=self._focus_settings_panel, style="ghost",
            size="sm",
        )
        settings_btn.pack(side="left")
        self._header_settings_btn = settings_btn

        help_btn = ModernButton(right, text=tr("Help"), width=80,
                                command=self._show_about, style="ghost",
                                size="sm")
        help_btn.pack(side="left", padx=(Theme.S_SM, 0))
        self._header_help_btn = help_btn

        chips = tk.Frame(header_top, bg=Theme.BG_SECONDARY)
        chips.pack(side="right", padx=(Theme.S_XL, Theme.S_LG))
        self._header_chips = chips
        self._render_header_chips()

    def _focus_settings_panel(self):
        """Bring the inspector into view without activating another window."""
        if not hasattr(self, "_settings_col"):
            return
        try:
            if self._layout_mode == "stacked":
                bbox = self._content_canvas.bbox("all")
                if bbox and bbox[3] > 0:
                    self._content_canvas.yview_moveto(
                        max(0.0, self._settings_col.winfo_y() / bbox[3]))
            self._settings_col.focus_set()
        except tk.TclError:
            pass

    def _build_workflow_rail(self, parent):
        """Build the three-step workflow rail from the redesign reference."""
        rail = self._create_surface(parent)
        rail.pack(fill="both", expand=True)
        self._workflow_rail = rail

        tk.Label(
            rail, text=tr("Workflow").upper(), font=f(Theme.F_EYEBROW, "bold"),
            bg=Theme.BG_SECONDARY, fg=Theme.TEXT_MUTED,
        ).pack(anchor="w", padx=Theme.S_LG, pady=(Theme.S_LG, Theme.S_MD))

        steps = tk.Frame(rail, bg=Theme.BG_SECONDARY)
        steps.pack(fill="x", padx=Theme.S_MD)
        self._workflow_steps_row = steps
        self._workflow_step_blocks = []
        self._workflow_connectors = []
        step_data = (
            (tr("Import"), tr("Add video or image files")),
            (tr("Configure"), tr("Choose cleanup and region")),
            (tr("Process"), tr("Run and monitor the batch")),
        )
        for idx, (step_label, description) in enumerate(step_data, start=1):
            block = tk.Frame(steps, bg=Theme.BG_SECONDARY)
            block.pack(fill="x")
            badge_lbl = tk.Label(
                block, text=str(idx), width=3, height=1,
                font=f(Theme.F_BODY_SM, "bold"),
                bg=Theme.BG_TERTIARY, fg=Theme.TEXT_MUTED,
                highlightthickness=1, highlightbackground=Theme.BORDER,
            )
            badge_lbl.pack(side="left", anchor="n", pady=2)
            copy = tk.Frame(block, bg=Theme.BG_SECONDARY)
            copy.pack(side="left", fill="x", expand=True,
                      padx=(Theme.S_SM, 0))
            text_lbl = tk.Label(
                copy, text=step_label, font=f(Theme.F_BODY_SM, "bold"),
                bg=Theme.BG_SECONDARY, fg=Theme.TEXT_SECONDARY,
            )
            text_lbl.pack(anchor="w")
            description_lbl = tk.Label(
                copy, text=description, font=f(Theme.F_META),
                bg=Theme.BG_SECONDARY, fg=Theme.TEXT_MUTED,
                wraplength=112, justify="left",
            )
            description_lbl.pack(anchor="w", pady=(2, 0))
            self._workflow_step_blocks.append(block)
            self._workflow_pills.append({
                "frame": block, "badge": badge_lbl, "text": text_lbl,
                "description": description_lbl,
            })
            if idx < len(step_data):
                connector = tk.Frame(
                    steps, bg=Theme.BORDER_STRONG, width=2, height=24)
                connector.pack(anchor="w", padx=14, pady=3)
                connector.pack_propagate(False)
                self._workflow_connectors.append(connector)

        self._header_guidance_panel = self._create_card(rail)
        self._header_guidance_panel.pack(
            side="bottom", fill="x", padx=Theme.S_MD,
            pady=(Theme.S_XL, Theme.S_MD))

        guidance_copy = tk.Frame(self._header_guidance_panel, bg=Theme.BG_CARD)
        guidance_copy.pack(fill="x", padx=Theme.S_MD, pady=Theme.S_MD)
        self._header_guidance_copy = guidance_copy

        self.header_guidance_title = tk.Label(
            guidance_copy,
            text=tr("Build your batch"),
            font=f(Theme.F_TITLE, "bold"),
            bg=Theme.BG_CARD,
            fg=Theme.TEXT_PRIMARY,
        )
        self.header_guidance_title.pack(anchor="w")
        self.header_guidance_body = tk.Label(
            guidance_copy,
            text=tr("Import files or choose a folder to start."),
            font=f(Theme.F_BODY_SM),
            wraplength=124,
            justify="left",
            bg=Theme.BG_CARD,
            fg=Theme.TEXT_MUTED,
        )
        self.header_guidance_body.pack(anchor="w", fill="x", pady=(4, 0))

    def _build_input_section(self, parent):
        """Compact import surface above the primary preview."""
        section = self._create_surface(parent)
        section.pack(fill="x")
        self._import_section = section

        self.drop_area = DragDropFrame(
            section, self._on_files_dropped, height=52, compact=True)
        self.drop_area.pack(fill="x", padx=Theme.S_MD, pady=Theme.S_SM)

    def _build_output_card(self, parent):
        """Build the output destination card inside the inspector."""
        out_surface = self._create_card(parent)
        out_surface.pack(fill="x", pady=(Theme.S_SM, 0))
        self._card_header(out_surface, "Output", "Output")

        out_row = tk.Frame(out_surface, bg=Theme.BG_CARD)
        out_row.pack(fill="x", padx=Theme.S_MD, pady=(0, Theme.S_MD))

        label_col = tk.Frame(out_row, bg=Theme.BG_CARD)
        label_col.pack(fill="x")

        self.output_dir_label = tk.Label(label_col, text="", font=f(Theme.F_BODY, "bold"),
                                         bg=Theme.BG_CARD, fg=Theme.TEXT_PRIMARY, anchor="w")
        self.output_dir_label.pack(anchor="w")

        self.output_dir_meta = tk.Label(label_col, text="", font=f(Theme.F_META),
                                        bg=Theme.BG_CARD, fg=Theme.TEXT_MUTED, anchor="w")
        self.output_dir_meta.pack(anchor="w", pady=(2, 0))

        actions = tk.Frame(out_row, bg=Theme.BG_CARD)
        actions.pack(fill="x", pady=(Theme.S_SM, 0))

        choose_btn = ModernButton(actions, text=tr("Choose folder"), width=120,
                                  command=self._choose_output_dir, style="accent",
                                  size="sm")
        choose_btn.pack(side="left")

        reset_btn = ModernButton(actions, text=tr("Reset"), width=76,
                                 command=self._reset_output_dir, style="ghost",
                                 size="sm")
        reset_btn.pack(side="left", padx=(Theme.S_SM, 0))

        self._update_output_label()

    def _build_inspector_primary_action(self, parent):
        """Place the dominant action beside the primary configuration."""
        self._inspector_primary_frame = tk.Frame(
            parent, bg=Theme.BG_SECONDARY)
        self._inspector_primary_frame.pack(fill="x", pady=(Theme.S_MD, 0))
        self.inspector_start_btn = ModernButton(
            self._inspector_primary_frame,
            text=tr("Start cleanup"),
            width=320,
            height=44,
            command=self._start_processing,
            style="primary",
            size="lg",
            icon=">",
        )
        self.inspector_start_btn.pack(fill="x")
        self._refresh_action_states()
        self._layout_queue_actions(compact=False, dense=False)

    def _build_profile_settings_group(self, settings):
        # ---- Profile card -----------------------------------------------
        profile_panel = self._create_card(settings)
        profile_panel.pack(fill="x")

        self._card_header(profile_panel, "Cleanup method", "Cleanup method")

        profile_details = tk.Frame(profile_panel, bg=Theme.BG_CARD)
        self._inspector_profile_details = profile_details

        # Preset picker -- one-click recipe application. Built-ins + user-saved.
        preset_row = tk.Frame(profile_details, bg=Theme.BG_CARD)
        preset_row.pack(fill="x", padx=Theme.S_MD, pady=(Theme.S_XS, Theme.S_SM))

        tk.Label(preset_row, text=tr("Preset"), font=f(Theme.F_BODY_SM),
                 bg=Theme.BG_CARD, fg=Theme.TEXT_SECONDARY).pack(anchor="w")

        self.preset_var = tk.StringVar(value="(custom)")
        preset_names = ["(custom)"] + [n for n, _ in list_presets()]
        self.preset_combo = ttk.Combobox(
            preset_row, textvariable=self.preset_var, values=preset_names,
            state="readonly", style="Dark.TCombobox", width=20,
            font=f(Theme.F_BODY_SM),
        )
        self.preset_combo.pack(fill="x", pady=(Theme.S_XS, Theme.S_SM))
        self.preset_combo.bind("<<ComboboxSelected>>", self._on_preset_applied)

        preset_actions = tk.Frame(preset_row, bg=Theme.BG_CARD)
        preset_actions.pack(fill="x")

        save_preset_btn = ModernButton(
            preset_actions, text=tr("Save as..."), command=self._save_preset_dialog,
            size="sm", style="ghost",
        )
        save_preset_btn.pack(side="left")

        export_preset_btn = ModernButton(
            preset_actions, text=tr("Export"), command=self._export_preset_dialog,
            size="sm", style="ghost",
        )
        export_preset_btn.pack(side="left", padx=(Theme.S_XS, 0))
        Tooltip(export_preset_btn, tr("Write the current preset to a shareable JSON file."))

        import_preset_btn = ModernButton(
            preset_actions, text=tr("Import"), command=self._import_preset_dialog,
            size="sm", style="ghost",
        )
        import_preset_btn.pack(side="left", padx=(Theme.S_XS, 0))
        Tooltip(import_preset_btn, tr("Load a preset JSON file into the user library."))

        # Algorithm -- segmented picker replaces the Combobox for speed + clarity
        tk.Label(profile_panel, text=tr("Algorithm"), font=f(Theme.F_BODY_SM),
                  bg=Theme.BG_CARD, fg=Theme.TEXT_SECONDARY).pack(
                     anchor="w", padx=Theme.S_MD)

        self.mode_picker = SegmentedPicker(
            profile_panel,
            options=[(m.value, m.value) for m in InpaintMode],
            value=self.mode_var.get(),
            command=self._on_mode_picker_changed,
            bg=Theme.BG_CARD,
            group_label=tr("Cleanup algorithm"),
            columns=2,
        )
        self.mode_picker.pack(fill="x", padx=Theme.S_MD, pady=(Theme.S_XS, Theme.S_MD))

        self.algo_desc = tk.Label(profile_panel, text=self._get_algo_description(),
                                  font=f(Theme.F_BODY_SM), bg=Theme.BG_CARD,
                                  fg=Theme.TEXT_SECONDARY, justify="left", anchor="w",
                                  wraplength=320)
        Tooltip(self.mode_picker, self._get_algo_description())

        row2 = tk.Frame(profile_details, bg=Theme.BG_CARD)
        row2.pack(fill="x", padx=Theme.S_MD, pady=(0, Theme.S_SM))

        tk.Label(row2, text=tr("Compute device"), font=f(Theme.F_BODY_SM),
                 bg=Theme.BG_CARD, fg=Theme.TEXT_SECONDARY).pack(anchor="w")

        self.gpu_combo = ttk.Combobox(row2, textvariable=self.gpu_var, width=20,
                                      values=[tr("Detecting hardware...")],
                                      style="Dark.TCombobox", state="disabled",
                                      font=f(Theme.F_BODY_SM))
        self.gpu_combo.pack(fill="x", pady=(Theme.S_XS, 0))
        self.gpu_combo.bind("<<ComboboxSelected>>", self._on_gpu_changed)
        self._refresh_gpu_selector()

        lang_row = tk.Frame(profile_details, bg=Theme.BG_CARD)
        lang_row.pack(fill="x", padx=Theme.S_MD, pady=(0, Theme.S_MD))

        tk.Label(lang_row, text=tr("Subtitle language"), font=f(Theme.F_BODY_SM),
                 bg=Theme.BG_CARD, fg=Theme.TEXT_SECONDARY).pack(anchor="w")

        # F-5: language list = curated friendly names plus compatible
        # OCR engine codes so users can pick e.g. Thai or Polish without
        # modifying code. Backend status reports engine capacity separately.
        self._lang_display = _build_language_list()
        self._lang_labels = [f"{name} ({code})" for code, name in self._lang_display]
        self._lang_by_label = {label: code for label, (code, _) in
                               zip(self._lang_labels, self._lang_display)}
        self._lang_display_var = tk.StringVar()
        self._set_lang_display(self.lang_var.get())

        language_picker = tk.Frame(lang_row, bg=Theme.BG_CARD)
        language_picker.pack(fill="x", pady=(Theme.S_XS, 0))
        self.lang_combo = ttk.Combobox(language_picker, textvariable=self._lang_display_var,
                                       width=20, values=self._lang_labels,
                                       style="Dark.TCombobox",
                                       state="readonly", font=f(Theme.F_BODY_SM))
        self.lang_combo.pack(side="left", fill="x", expand=True)
        self.lang_combo.bind("<<ComboboxSelected>>", self._on_lang_changed)
        self._lang_detect_btn = ModernButton(
            language_picker, text=tr("Detect"), width=68,
            command=self._probe_language_from_preview,
            style="ghost", size="sm")
        self._lang_detect_btn.pack(side="right", padx=(Theme.S_SM, 0))
        Tooltip(self._lang_detect_btn,
                tr("Auto-detect the subtitle language from a sample frame."))

        return profile_details

    def _build_workflow_settings_group(self, settings):
        # ---- Workflow card ----------------------------------------------
        workflow_panel = self._create_card(settings)
        workflow_panel.pack(fill="x", pady=(Theme.S_SM, 0))

        self._card_header(workflow_panel, "Workflow", "Region and audio")

        workflow_details = tk.Frame(workflow_panel, bg=Theme.BG_CARD)
        self._inspector_workflow_details = workflow_details
        checks_frame = tk.Frame(workflow_details, bg=Theme.BG_CARD)
        checks_frame.pack(fill="x", padx=Theme.S_MD, pady=(0, Theme.S_MD))

        self.skip_check = ModernToggle(
            checks_frame,
            text=tr("Use a fixed subtitle region"),
            variable=self.skip_detection_var,
            wraplength=300,
        )
        self.skip_check.pack(anchor="w")
        Tooltip(self.skip_check, tr("Skip repeated detection when you have already set a precise subtitle region."))

        self.lama_check = ModernToggle(
            checks_frame,
            text=tr("Fast LaMa cleanup"),
            variable=self.lama_fast_var,
            wraplength=300,
        )
        self.lama_check.pack(anchor="w", pady=(Theme.S_SM, 0))
        Tooltip(self.lama_check, tr("LaMa fast mode is useful for quick passes and lower-resolution drafts."))

        self.preserve_audio_check = ModernToggle(
            checks_frame,
            text=tr("Preserve source audio"),
            variable=self.preserve_audio_var,
            wraplength=300,
        )
        self.preserve_audio_check.pack(anchor="w", pady=(Theme.S_SM, 0))
        self.ffmpeg_warning_label = tk.Label(
            checks_frame,
            text=tr("Checking FFmpeg availability for audio preservation..."),
            font=f(Theme.F_META),
            bg=Theme.BG_CARD,
            fg=Theme.INFO,
            wraplength=320,
            justify="left",
        )
        self._refresh_ffmpeg_warning()

        # The region reads as a control group, not a card nested in a card.
        region_surface = tk.Frame(workflow_panel, bg=Theme.BG_CARD,
                                  highlightthickness=0)
        self._inspector_region_surface = region_surface
        region_surface.pack(fill="x", padx=Theme.S_MD, pady=(0, Theme.S_MD))

        region_text = tk.Frame(region_surface, bg=Theme.BG_CARD)
        region_text.pack(side="left", fill="x", expand=True,
                         pady=Theme.S_XS)

        tk.Label(region_text, text=tr("Subtitle region"), font=f(Theme.F_BODY_SM),
                 bg=Theme.BG_CARD, fg=Theme.TEXT_SECONDARY).pack(anchor="w")

        self.region_label = tk.Label(region_text, text="", font=f(Theme.F_BODY, "bold"),
                                     bg=Theme.BG_CARD, fg=Theme.TEXT_PRIMARY,
                                     anchor="w")
        self.region_label.pack(anchor="w", pady=(2, 0))

        self.region_meta = tk.Label(region_text, text="", font=f(Theme.F_META),
                                    bg=Theme.BG_CARD, fg=Theme.TEXT_MUTED,
                                    anchor="w")
        self.region_meta.pack(anchor="w", pady=(2, 0))

        region_actions = tk.Frame(region_surface, bg=Theme.BG_CARD)
        region_actions.pack(side="right", pady=Theme.S_XS)

        self.region_btn = ModernButton(region_actions, text=tr("Set region"), width=88,
                                       command=self._open_region_selector_modal, style="accent",
                                       size="sm")
        self.region_btn.pack(side="left")

        self.region_reset_btn = ModernButton(region_actions, text=tr("Reset"), width=64,
                                             command=self._reset_region, style="ghost",
                                             size="sm")
        self.region_reset_btn.pack(side="left", padx=(Theme.S_SM, 0))
        return workflow_details, region_surface

    def _build_sttn_settings_group(self):
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


    def _build_detection_settings_group(self):
        # Detection Precision card
        det_frame = self._create_card(self.adv_panel)
        det_frame.pack(fill="x", pady=(0, Theme.S_SM))
        self._card_header(det_frame, "Detection", "Precision tuning")

        self._create_slider(det_frame, "Sensitivity", 10, 90,
                            int(self.config.detection_threshold * 100),
                            "_detection_threshold_pct",
                            hint="Higher catches more text (lower confidence floor). Lower is stricter.")
        engine_row = tk.Frame(det_frame, bg=Theme.BG_CARD)
        engine_row.pack(fill="x", padx=Theme.S_LG, pady=(Theme.S_SM, 0))
        tk.Label(
            engine_row,
            text=tr("OCR engine"),
            font=f(Theme.F_BODY_SM),
            bg=Theme.BG_CARD,
            fg=Theme.TEXT_SECONDARY,
        ).pack(side="left")
        self.ocr_engine_combo = ttk.Combobox(
            engine_row,
            textvariable=self.ocr_engine_var,
            values=tuple(self._ocr_engine_by_label),
            width=24,
            state="readonly",
            style="Dark.TCombobox",
            font=f(Theme.F_BODY_SM),
        )
        self.ocr_engine_combo.pack(side="right")
        self.ocr_engine_combo.bind(
            "<<ComboboxSelected>>", self._on_ocr_engine_changed)
        Tooltip(
            self.ocr_engine_combo,
            tr("Automatic uses the best installed detector. Select an engine "
               "to compare results or reproduce a run."),
        )
        self.language_filter_var = tk.BooleanVar(
            value=self.config.language_mask_filter)
        self.language_filter_toggle = ModernToggle(
            det_frame,
            text=tr("Only remove the selected language"),
            variable=self.language_filter_var,
        )
        self.language_filter_toggle.pack(
            anchor="w", padx=Theme.S_LG, pady=(Theme.S_SM, 0))
        Tooltip(
            self.language_filter_toggle,
            tr("Keep OCR boxes whose recognized script does not match the "
               "subtitle language. Latin-script languages share one family."),
        )
        self._create_slider(det_frame, "Frame skip", 0, 10,
                            self.config.detection_frame_skip, "detection_frame_skip",
                            hint="Reuse the last mask for N frames. At 5, scheduled OCR drops by up to about 83% on stable footage.")
        self._create_slider(det_frame, "Mask dilate", 0, 20,
                            self.config.mask_dilate_px, "mask_dilate_px",
                            hint="Expand detected regions for cleaner fill edges.")
        self.conf_dilate_var = tk.BooleanVar(
            value=self.config.confidence_weighted_dilation)
        conf_dilate_toggle = ModernToggle(
            det_frame,
            text=tr("Adaptive edge padding"),
            variable=self.conf_dilate_var,
        )
        conf_dilate_toggle.pack(anchor="w", padx=Theme.S_LG, pady=(Theme.S_SM, 0))
        Tooltip(conf_dilate_toggle,
                tr("Add more padding around uncertain text detections while "
                   "keeping confident detections tight. Helps catch faint edges."))
        self._create_slider(det_frame, "Mask feather", 0, 15,
                            self.config.mask_feather_px, "mask_feather_px",
                            hint="Soft-blend the removal edge for seamless boundaries.")
        self._create_slider(det_frame, "Colour match ring", 0, 8,
                            self.config.edge_ring_px, "edge_ring_px",
                            hint="Post-inpaint edge-ring colour correction to kill faint seams.")

        self.auto_band_var = tk.BooleanVar(value=self.config.auto_band)
        auto_band_toggle = ModernToggle(
            det_frame,
            text=tr("Auto-detect subtitle band on load"),
            variable=self.auto_band_var,
        )
        auto_band_toggle.pack(anchor="w", padx=Theme.S_LG, pady=(Theme.S_SM, 0))
        Tooltip(auto_band_toggle, tr("Scan the first 30 frames and pin the dominant subtitle band before processing."))

        self.flow_warp_var = tk.BooleanVar(value=self.config.tbe_flow_warp)
        flow_toggle = ModernToggle(
            det_frame,
            text=tr("Motion-aligned background recovery"),
            variable=self.flow_warp_var,
        )
        flow_toggle.pack(anchor="w", padx=Theme.S_LG, pady=(Theme.S_SM, 0))
        Tooltip(flow_toggle, tr("Align nearby frames before rebuilding hidden "
                                "background pixels. Slower, but cleaner on pans and zooms."))

        self.scene_split_var = tk.BooleanVar(value=self.config.tbe_scene_cut_split)
        scene_toggle = ModernToggle(
            det_frame,
            text=tr("Keep scene changes separate"),
            variable=self.scene_split_var,
        )
        scene_toggle.pack(anchor="w", padx=Theme.S_LG, pady=(Theme.S_SM, 0))
        Tooltip(scene_toggle, tr("Keep background recovery from mixing frames "
                                 "across scene changes. Recommended for edited video."))

        self.kalman_var = tk.BooleanVar(value=self.config.kalman_tracking)
        kalman_toggle = ModernToggle(
            det_frame,
            text=tr("Smooth detection between frames"),
            variable=self.kalman_var,
        )
        kalman_toggle.pack(anchor="w", padx=Theme.S_LG, pady=(Theme.S_SM, 0))
        Tooltip(kalman_toggle, tr("Smooths per-frame OCR jitter and fills single-frame misses. Recommended."))

        self.phash_var = tk.BooleanVar(value=self.config.phash_skip_enable)
        phash_toggle = ModernToggle(
            det_frame,
            text=tr("Reuse masks on unchanged frames"),
            variable=self.phash_var,
        )
        phash_toggle.pack(anchor="w", padx=Theme.S_LG, pady=(Theme.S_SM, 0))
        Tooltip(phash_toggle, tr("Skip OCR on frames nearly identical to the last detected one. Static shots can avoid most OCR calls."))
        self._create_slider(
            det_frame, "pHash distance", 0, 16,
            self.config.phash_skip_distance, "phash_skip_distance",
            hint="Higher values reuse masks across more near-identical frames; 4 is conservative.",
        )

        self.colour_tune_var = tk.BooleanVar(value=self.config.colour_tune_enable)
        colour_toggle = ModernToggle(
            det_frame,
            text=tr("Colour-tuned mask expansion"),
            variable=self.colour_tune_var,
        )
        colour_toggle.pack(anchor="w", padx=Theme.S_LG, pady=(Theme.S_SM, 0))
        Tooltip(colour_toggle, tr("Grow the mask to cover serifs / drop shadows that match the subtitle colour. Catches decorative lettering."))

        # RM-24: vertical-text toggle for Japanese tategaki / classical CN.
        self.vertical_text_var = tk.BooleanVar(value=getattr(self.config, "detection_vertical", False))
        vertical_toggle = ModernToggle(
            det_frame,
            text=tr("Vertical text mode (Japanese tategaki / classical Chinese)"),
            variable=self.vertical_text_var,
        )
        vertical_toggle.pack(anchor="w", padx=Theme.S_LG, pady=(Theme.S_SM, Theme.S_MD))
        Tooltip(vertical_toggle, tr("Improve detection for top-to-bottom Japanese "
                                    "or Chinese text. Saved masks stay aligned to the source."))

        tk.Frame(det_frame, bg=Theme.BG_CARD, height=Theme.S_SM).pack(fill="x")


    def _build_output_settings_group(self):
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
            text=tr("Hardware encoding (NVENC / QSV / AMF) with software fallback"),
            variable=self.hw_encode_var,
        )
        self.hw_encode_check.pack(anchor="w", padx=Theme.S_LG, pady=(Theme.S_SM, 0))
        Tooltip(self.hw_encode_check, tr("Use the GPU's media encoder when "
                                         "available. The app retries with CPU encoding if it fails."))

        self.d3d12_accel_var = tk.BooleanVar(
            value=getattr(self.config, "d3d12_accel", False))
        d3d12_toggle = ModernToggle(
            quality_frame,
            text=tr("Try FFmpeg D3D12 acceleration (experimental)"),
            variable=self.d3d12_accel_var,
        )
        d3d12_toggle.pack(anchor="w", padx=Theme.S_LG, pady=(Theme.S_SM, 0))
        Tooltip(d3d12_toggle, tr(
            "Windows only. Requires FFmpeg 8.1 or newer and uses D3D12 only "
            "after a byte-valid runtime test; NVENC, QSV, AMF, or CPU encoding "
            "remains the automatic fallback."))

        # F-8: output codec selector lives next to the HW-encode toggle.
        codec_row = tk.Frame(quality_frame, bg=Theme.BG_CARD)
        codec_row.pack(fill="x", padx=Theme.S_LG, pady=(Theme.S_SM, 0))
        tk.Label(codec_row, text=tr("Output codec"), font=f(Theme.F_BODY_SM),
                 bg=Theme.BG_CARD, fg=Theme.TEXT_SECONDARY).pack(side="left")
        self.output_codec_var = tk.StringVar(value=getattr(self.config, "output_codec", "h264"))
        codec_combo = ttk.Combobox(
            codec_row, textvariable=self.output_codec_var, width=10,
            values=["h264", "h265", "av1", "vvc"],
            state="readonly", style="Dark.TCombobox", font=f(Theme.F_BODY_SM),
        )
        codec_combo.pack(side="right")
        Tooltip(codec_combo,
                tr("H.264 plays almost everywhere. H.265 and AV1 make smaller "
                   "files but take longer; VVC needs a compatible FFmpeg build."))

        self.adaptive_batch_var = tk.BooleanVar(value=self.config.adaptive_batch)
        adaptive_toggle = ModernToggle(
            quality_frame,
            text=tr("Fit frame batches to GPU memory"),
            variable=self.adaptive_batch_var,
        )
        adaptive_toggle.pack(anchor="w", padx=Theme.S_LG, pady=(Theme.S_SM, 0))
        Tooltip(adaptive_toggle, tr("Adjust how many frames are processed together "
                                    "to fit available GPU memory. Recommended for 4K video."))

        self.temporal_mask_union_var = tk.BooleanVar(
            value=self.config.temporal_mask_union)
        temporal_mask_toggle = ModernToggle(
            quality_frame,
            text=tr("Carry masks across missed detections"),
            variable=self.temporal_mask_union_var,
        )
        temporal_mask_toggle.pack(anchor="w", padx=Theme.S_LG, pady=(Theme.S_SM, 0))
        Tooltip(temporal_mask_toggle, tr(
            "Automatic detection only. Carry recent text masks forward for a "
            "few frames so a single missed detection does not leave a flash."))
        self._create_slider(
            quality_frame, "Mask carry window", 1, 15,
            self.config.temporal_mask_window, "temporal_mask_window",
            hint="Number of recent masks available to repair a missed detection.",
        )

        self.export_srt_var = tk.BooleanVar(value=self.config.export_srt)
        srt_toggle = ModernToggle(
            quality_frame,
            text=tr("Export detected text as .srt sidecar"),
            variable=self.export_srt_var,
        )
        srt_toggle.pack(anchor="w", padx=Theme.S_LG, pady=(Theme.S_SM, 0))
        Tooltip(srt_toggle, tr("Writes an .srt file next to the output using OCR text and timings."))

        self.export_mask_var = tk.BooleanVar(value=self.config.export_mask_video)
        mask_toggle = ModernToggle(
            quality_frame,
            text=tr("Export lossless mask / alpha matte"),
            variable=self.export_mask_var,
        )
        mask_toggle.pack(anchor="w", padx=Theme.S_LG, pady=(Theme.S_SM, 0))
        Tooltip(mask_toggle, tr(
            "Writes an exact grayscale matte and versioned timestamp manifest "
            "alongside the output."))

        matte_format_row = tk.Frame(quality_frame, bg=Theme.BG_CARD)
        matte_format_row.pack(fill="x", padx=Theme.S_LG, pady=(Theme.S_XS, 0))
        tk.Label(
            matte_format_row, text=tr("Matte format"), font=f(Theme.F_META),
            bg=Theme.BG_CARD, fg=Theme.TEXT_MUTED,
        ).pack(side="left")
        self.mask_export_format_var = tk.StringVar(
            value=self.config.mask_export_format)
        matte_format = ttk.Combobox(
            matte_format_row, textvariable=self.mask_export_format_var,
            values=("ffv1", "png"), state="readonly", width=9,
            style="Dark.TCombobox", font=f(Theme.F_BODY_SM),
        )
        matte_format.pack(side="right")
        set_accessible_metadata(
            matte_format, role="combo box", label=tr("Lossless matte format"),
            description=tr("Choose FFV1 video or an ordered PNG sequence."))

        matte_import_row = tk.Frame(quality_frame, bg=Theme.BG_CARD)
        matte_import_row.pack(fill="x", padx=Theme.S_LG, pady=(Theme.S_SM, 0))
        self.mask_import_label_var = tk.StringVar(value=(
            Path(self.config.mask_import_path).name
            if self.config.mask_import_path else tr("No imported matte")
        ))
        tk.Label(
            matte_import_row, textvariable=self.mask_import_label_var,
            font=f(Theme.F_META), bg=Theme.BG_CARD, fg=Theme.TEXT_MUTED,
            anchor="w",
        ).pack(side="left", fill="x", expand=True)
        ModernButton(
            matte_import_row, text=tr("Import matte"), width=112,
            command=self._choose_mask_import_manifest,
            style="ghost", size="sm",
        ).pack(side="right")

        matte_mode_row = tk.Frame(quality_frame, bg=Theme.BG_CARD)
        matte_mode_row.pack(fill="x", padx=Theme.S_LG, pady=(Theme.S_XS, 0))
        ModernButton(
            matte_mode_row, text=tr("Clear import"), width=100,
            command=self._clear_mask_import_manifest,
            style="ghost", size="sm",
        ).pack(side="left")
        self.mask_import_mode_var = tk.StringVar(
            value=self.config.mask_import_mode)
        matte_mode = ttk.Combobox(
            matte_mode_row, textvariable=self.mask_import_mode_var,
            values=("replace", "add", "subtract"), state="readonly",
            width=11, style="Dark.TCombobox", font=f(Theme.F_BODY_SM),
        )
        matte_mode.pack(side="right")
        set_accessible_metadata(
            matte_mode, role="combo box", label=tr("Imported matte mode"),
            description=tr(
                "Replace, add to, or subtract from the native composed mask."))

        self.deinterlace_var = tk.BooleanVar(value=self.config.deinterlace_auto)
        deinterlace_toggle = ModernToggle(
            quality_frame,
            text=tr("Clean up interlaced video automatically"),
            variable=self.deinterlace_var,
        )
        deinterlace_toggle.pack(anchor="w", padx=Theme.S_LG, pady=(Theme.S_SM, 0))
        Tooltip(deinterlace_toggle, tr("Detect and remove comb-like lines from "
                                       "older interlaced video before cleanup."))

        self.keyframe_var = tk.BooleanVar(value=self.config.keyframe_detection)
        keyframe_toggle = ModernToggle(
            quality_frame,
            text=tr("Faster keyframe-only detection"),
            variable=self.keyframe_var,
        )
        keyframe_toggle.pack(anchor="w", padx=Theme.S_LG, pady=(Theme.S_SM, 0))
        Tooltip(keyframe_toggle, tr("Check text on major scene frames and reuse "
                                    "masks between them. Faster on long videos, less precise on motion."))

        self.quality_report_var = tk.BooleanVar(value=self.config.quality_report)
        quality_toggle = ModernToggle(
            quality_frame,
            text=tr("Check output quality after processing"),
            variable=self.quality_report_var,
        )
        quality_toggle.pack(anchor="w", padx=Theme.S_LG, pady=(Theme.S_SM, Theme.S_MD))
        Tooltip(quality_toggle, tr("Sample the finished video for visual damage, "
                                   "remaining text, and flicker, then flag results that need review."))

        translation_frame = self._create_card(self.adv_panel)
        translation_frame.pack(fill="x", pady=(0, Theme.S_SM))
        self._card_header(
            translation_frame, "Localization", "Erase, translate, and re-embed")

        self.translation_enabled_var = tk.BooleanVar(
            value=self.config.translation_enabled)
        translation_toggle = ModernToggle(
            translation_frame,
            text=tr("Re-embed translated subtitles after cleanup"),
            variable=self.translation_enabled_var,
        )
        translation_toggle.pack(
            anchor="w", padx=Theme.S_LG, pady=(Theme.S_XS, 0))
        Tooltip(translation_toggle, tr(
            "Use a provided translated SRT, or send OCR/Whisper cues to a "
            "local translation command. VSR itself does not contact a "
            "translation service; the command you choose controls text handling."))

        self.translation_srt_var = tk.StringVar(
            value=self.config.translation_srt)
        self.translation_srt_label_var = tk.StringVar(value=(
            Path(self.config.translation_srt).name
            if self.config.translation_srt else tr("No translated SRT selected")
        ))
        translated_row = tk.Frame(translation_frame, bg=Theme.BG_CARD)
        translated_row.pack(
            fill="x", padx=Theme.S_LG, pady=(Theme.S_SM, 0))
        tk.Label(
            translated_row,
            textvariable=self.translation_srt_label_var,
            font=f(Theme.F_META),
            bg=Theme.BG_CARD,
            fg=Theme.TEXT_MUTED,
            anchor="w",
        ).pack(side="left", fill="x", expand=True)
        ModernButton(
            translated_row,
            text=tr("Choose translated SRT"),
            width=148,
            command=self._choose_translated_srt,
            style="ghost",
            size="sm",
        ).pack(side="right")
        ModernButton(
            translated_row,
            text=tr("Clear"),
            width=68,
            command=self._clear_translated_srt,
            style="ghost",
            size="sm",
        ).pack(side="right", padx=(0, Theme.S_XS))

        language_row = tk.Frame(translation_frame, bg=Theme.BG_CARD)
        language_row.pack(
            fill="x", padx=Theme.S_LG, pady=(Theme.S_SM, 0))
        tk.Label(
            language_row, text=tr("Language tags"),
            font=f(Theme.F_BODY_SM), bg=Theme.BG_CARD,
            fg=Theme.TEXT_SECONDARY,
        ).pack(side="left")
        language_controls = tk.Frame(language_row, bg=Theme.BG_CARD)
        language_controls.pack(side="right")
        self.translation_source_lang_var = tk.StringVar(
            value=self.config.translation_source_lang or "auto")
        tk.Label(
            language_controls, text=tr("Source"),
            font=f(Theme.F_META), bg=Theme.BG_CARD,
            fg=Theme.TEXT_MUTED,
        ).pack(side="left", padx=(0, Theme.S_XS))
        source_lang_entry = tk.Entry(
            language_controls, width=8, bg=Theme.BG_TERTIARY,
            fg=Theme.TEXT_PRIMARY, font=f(Theme.F_BODY_SM),
            insertbackground=Theme.TEXT_PRIMARY, highlightthickness=1,
            highlightbackground=Theme.BORDER,
            highlightcolor=Theme.BORDER_FOCUS, relief="flat", bd=6,
            textvariable=self.translation_source_lang_var,
        )
        source_lang_entry.pack(side="left")
        self.translation_target_lang_var = tk.StringVar(
            value=self.config.translation_target_lang)
        tk.Label(
            language_controls, text=tr("Target"),
            font=f(Theme.F_META), bg=Theme.BG_CARD,
            fg=Theme.TEXT_MUTED,
        ).pack(side="left", padx=(Theme.S_SM, Theme.S_XS))
        target_lang_entry = tk.Entry(
            language_controls, width=8, bg=Theme.BG_TERTIARY,
            fg=Theme.TEXT_PRIMARY, font=f(Theme.F_BODY_SM),
            insertbackground=Theme.TEXT_PRIMARY, highlightthickness=1,
            highlightbackground=Theme.BORDER,
            highlightcolor=Theme.BORDER_FOCUS, relief="flat", bd=6,
            textvariable=self.translation_target_lang_var,
        )
        target_lang_entry.pack(side="left")
        set_accessible_metadata(
            source_lang_entry, role="text box", label=tr("Source language tag"))
        set_accessible_metadata(
            target_lang_entry, role="text box", label=tr("Target language tag"))

        self.translation_provider_var = tk.StringVar(
            value=self.config.translation_provider or "command")
        provider_row = tk.Frame(translation_frame, bg=Theme.BG_CARD)
        provider_row.pack(
            fill="x", padx=Theme.S_LG, pady=(Theme.S_SM, 0))
        tk.Label(
            provider_row, text=tr("Local provider"),
            font=f(Theme.F_BODY_SM), bg=Theme.BG_CARD,
            fg=Theme.TEXT_SECONDARY,
        ).pack(side="left")
        provider_entry = tk.Entry(
            provider_row, width=16, bg=Theme.BG_TERTIARY,
            fg=Theme.TEXT_PRIMARY, font=f(Theme.F_BODY_SM),
            insertbackground=Theme.TEXT_PRIMARY, highlightthickness=1,
            highlightbackground=Theme.BORDER,
            highlightcolor=Theme.BORDER_FOCUS, relief="flat", bd=6,
            textvariable=self.translation_provider_var,
        )
        provider_entry.pack(side="right")

        self.translation_command_var = tk.StringVar(
            value=self.config.translation_command)
        command_row = tk.Frame(translation_frame, bg=Theme.BG_CARD)
        command_row.pack(
            fill="x", padx=Theme.S_LG, pady=(Theme.S_SM, 0))
        tk.Label(
            command_row, text=tr("Translator command"),
            font=f(Theme.F_BODY_SM), bg=Theme.BG_CARD,
            fg=Theme.TEXT_SECONDARY,
        ).pack(side="left")
        command_entry = tk.Entry(
            command_row, bg=Theme.BG_TERTIARY, fg=Theme.TEXT_PRIMARY,
            font=f(Theme.F_BODY_SM), insertbackground=Theme.TEXT_PRIMARY,
            highlightthickness=1, highlightbackground=Theme.BORDER,
            highlightcolor=Theme.BORDER_FOCUS, relief="flat", bd=6,
            textvariable=self.translation_command_var,
        )
        command_entry.pack(
            side="left", fill="x", expand=True, padx=(Theme.S_SM, Theme.S_SM))
        ModernButton(
            command_row, text=tr("Browse"), width=78,
            command=self._choose_translation_command,
            style="ghost", size="sm",
        ).pack(side="right")

        self.translation_style_var = tk.StringVar(
            value=self.config.translation_style)
        style_row = tk.Frame(translation_frame, bg=Theme.BG_CARD)
        style_row.pack(
            fill="x", padx=Theme.S_LG, pady=(Theme.S_SM, Theme.S_MD))
        tk.Label(
            style_row, text=tr("ASS style override"),
            font=f(Theme.F_BODY_SM), bg=Theme.BG_CARD,
            fg=Theme.TEXT_SECONDARY,
        ).pack(side="left")
        style_entry = tk.Entry(
            style_row, bg=Theme.BG_TERTIARY, fg=Theme.TEXT_PRIMARY,
            font=f(Theme.F_BODY_SM), insertbackground=Theme.TEXT_PRIMARY,
            highlightthickness=1, highlightbackground=Theme.BORDER,
            highlightcolor=Theme.BORDER_FOCUS, relief="flat", bd=6,
            textvariable=self.translation_style_var,
        )
        style_entry.pack(
            side="right", fill="x", expand=True, padx=(Theme.S_SM, 0))
        Tooltip(style_entry, tr(
            "Optional FFmpeg force_style text, for example FontSize=24."))

        return quality_frame

    def _build_range_settings_group(self):
        # Video Range card
        time_frame = self._create_card(self.adv_panel)
        time_frame.pack(fill="x")
        self._card_header(time_frame, "Video range", "Trim (videos only)")

        time_inner = tk.Frame(time_frame, bg=Theme.BG_CARD)
        time_inner.pack(fill="x", padx=Theme.S_LG, pady=(0, Theme.S_MD))

        tk.Label(time_inner, text=tr("Start (s)"), font=f(Theme.F_BODY_SM),
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

        tk.Label(time_inner, text=tr("End (s)"), font=f(Theme.F_BODY_SM),
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

        tk.Label(time_inner, text=tr("0 uses the full clip"), font=f(Theme.F_META),
                 bg=Theme.BG_CARD, fg=Theme.TEXT_MUTED).pack(side="left", padx=(Theme.S_MD, 0))


    def _build_performance_settings_groups(self):
        # ---- v3.13 GUI-exposed knobs ------------------------------------
        # Editorial: chyron vs subtitle filter + karaoke grouping
        editorial_frame = self._create_card(self.adv_panel)
        editorial_frame.pack(fill="x", pady=(Theme.S_MD, Theme.S_SM))
        self._card_header(editorial_frame, "Editorial", "Filter what gets removed")

        self.remove_subs_var = tk.BooleanVar(value=self.config.remove_subtitles)
        remove_subs_toggle = ModernToggle(
            editorial_frame,
            text=tr("Remove dialogue subtitles (short-lived OCR tracks)"),
            variable=self.remove_subs_var,
        )
        remove_subs_toggle.pack(anchor="w", padx=Theme.S_LG, pady=(Theme.S_XS, 0))
        Tooltip(remove_subs_toggle, tr("Remove short-lived dialogue text. Persistent "
                                       "logos and lower-thirds are controlled separately below."))

        self.remove_chyrons_var = tk.BooleanVar(value=self.config.remove_chyrons)
        remove_chyrons_toggle = ModernToggle(
            editorial_frame,
            text=tr("Remove persistent text (logos, tickers, lower-thirds)"),
            variable=self.remove_chyrons_var,
        )
        remove_chyrons_toggle.pack(anchor="w", padx=Theme.S_LG, pady=(Theme.S_SM, 0))
        Tooltip(remove_chyrons_toggle, tr("Remove text that stays in one place for "
                                          "about three seconds or longer, such as lower-thirds."))

        self.karaoke_grouping_var = tk.BooleanVar(value=self.config.karaoke_grouping)
        karaoke_toggle = ModernToggle(
            editorial_frame,
            text=tr("Karaoke grouping: fuse per-syllable boxes on the same line"),
            variable=self.karaoke_grouping_var,
        )
        karaoke_toggle.pack(anchor="w", padx=Theme.S_LG, pady=(Theme.S_SM, Theme.S_MD))
        Tooltip(karaoke_toggle, tr("Stops karaoke captions leaking original text through the gaps between syllables."))

        # Audio card: loudnorm target + multi-track passthrough
        audio_frame = self._create_card(self.adv_panel)
        audio_frame.pack(fill="x", pady=(0, Theme.S_SM))
        self._card_header(audio_frame, "Audio", "Loudness + tracks")

        self.multi_audio_var = tk.BooleanVar(value=self.config.multi_audio_passthrough)
        multi_audio_toggle = ModernToggle(
            audio_frame,
            text=tr("Pass through every audio stream (Bluray/DVD multi-track)"),
            variable=self.multi_audio_var,
        )
        multi_audio_toggle.pack(anchor="w", padx=Theme.S_LG, pady=(Theme.S_XS, 0))
        Tooltip(multi_audio_toggle, tr("Mux every audio stream from the source. Off keeps only the first track."))

        loudnorm_row = tk.Frame(audio_frame, bg=Theme.BG_CARD)
        loudnorm_row.pack(fill="x", padx=Theme.S_LG, pady=(Theme.S_SM, Theme.S_MD))
        tk.Label(loudnorm_row, text=tr("EBU R128 loudness target"), font=f(Theme.F_BODY_SM),
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
                 text=tr("LUFS. 0 = off. YouTube -14, Apple -16, broadcast -23."),
                 font=f(Theme.F_META),
                 bg=Theme.BG_CARD, fg=Theme.TEXT_MUTED).pack(side="left")

        # Performance card: decode HW accel + prefetch
        perf_frame = self._create_card(self.adv_panel)
        perf_frame.pack(fill="x", pady=(0, Theme.S_SM))
        self._card_header(perf_frame, "Performance", "Decode pipeline")

        accel_row = tk.Frame(perf_frame, bg=Theme.BG_CARD)
        accel_row.pack(fill="x", padx=Theme.S_LG, pady=(Theme.S_XS, 0))
        tk.Label(accel_row, text=tr("Hardware-decode hint"), font=f(Theme.F_BODY_SM),
                 bg=Theme.BG_CARD, fg=Theme.TEXT_SECONDARY).pack(side="left")
        self.decode_accel_var = tk.StringVar(value=self.config.decode_hw_accel or "off")
        accel_combo = ttk.Combobox(
            accel_row, textvariable=self.decode_accel_var, width=10,
            values=["off", "auto", "any", "d3d11", "vaapi", "mfx", "pynv", "nvdec"],
            state="readonly", style="Dark.TCombobox", font=f(Theme.F_BODY_SM),
        )
        accel_combo.pack(side="right")
        Tooltip(accel_combo, tr("Try hardware video decoding for faster input. "
                                "The app falls back to software decoding if needed."))

        rife_row = tk.Frame(perf_frame, bg=Theme.BG_CARD)
        rife_row.pack(fill="x", padx=Theme.S_LG, pady=(Theme.S_SM, 0))
        tk.Label(rife_row, text=tr("Frame interpolation stride"), font=f(Theme.F_BODY_SM),
                 bg=Theme.BG_CARD, fg=Theme.TEXT_SECONDARY).pack(side="left")
        self.rife_stride_var = tk.StringVar(
            value=str(getattr(self.config, "rife_fast_stride", 0) or 0)
        )
        rife_entry = tk.Entry(
            rife_row, width=5, bg=Theme.BG_TERTIARY,
            fg=Theme.TEXT_PRIMARY, font=f(Theme.F_BODY_SM),
            insertbackground=Theme.TEXT_PRIMARY,
            highlightthickness=1,
            highlightbackground=Theme.BORDER,
            highlightcolor=Theme.BORDER_FOCUS,
            relief="flat", bd=6, textvariable=self.rife_stride_var)
        rife_entry.pack(side="right")
        Tooltip(rife_entry, tr("0 cleans every frame. Values above 1 clean fewer "
                               "frames and recreate the gaps when Practical-RIFE is installed. "
                               "Stride 3 can reduce inpaint calls by up to about 67%."))

        self.prefetch_var = tk.BooleanVar(value=self.config.prefetch_decode)
        prefetch_toggle = ModernToggle(
            perf_frame,
            text=tr("Read frames ahead in the background"),
            variable=self.prefetch_var,
        )
        prefetch_toggle.pack(anchor="w", padx=Theme.S_LG, pady=(Theme.S_SM, Theme.S_MD))
        Tooltip(prefetch_toggle, tr("Read upcoming frames in the background so "
                                    "cleanup spends less time waiting on the source file."))


    def _build_accessibility_storage_settings(
        self, quality_frame,
    ):
        # Quality sheet toggle (lives under Output but kept separate so we
        # don't disturb the existing Output card layout)
        self.quality_sheet_var = tk.BooleanVar(value=self.config.quality_report_sheet)
        quality_sheet_toggle = ModernToggle(
            quality_frame,
            text=tr("Quality report sheet (side-by-side PNG comparison)"),
            variable=self.quality_sheet_var,
        )
        quality_sheet_toggle.pack(anchor="w", padx=Theme.S_LG, pady=(0, Theme.S_SM))
        Tooltip(quality_sheet_toggle, tr("Save a side-by-side source/output image "
                                         "for sampled frames. Also enables the numeric quality report."))

        self.json_log_var = tk.BooleanVar(value=getattr(self.config, "json_log_enabled", False))
        json_log_toggle = ModernToggle(
            quality_frame,
            text=tr("Structured JSON log"),
            variable=self.json_log_var,
        )
        json_log_toggle.pack(anchor="w", padx=Theme.S_LG, pady=(0, Theme.S_SM))
        Tooltip(json_log_toggle, tr("Write a structured JSON-lines log alongside the text log. Useful for long batch runs and scripted post-processing."))

        text_scale_row = tk.Frame(quality_frame, bg=Theme.BG_CARD)
        text_scale_row.pack(
            fill="x", padx=Theme.S_LG, pady=(0, Theme.S_SM))
        tk.Label(
            text_scale_row,
            text=tr("Interface text size (restart required)"),
            font=f(Theme.F_BODY_SM),
            bg=Theme.BG_CARD,
            fg=Theme.TEXT_SECONDARY,
        ).pack(side="left")
        self.text_scale_var = tk.StringVar(
            value=f"{text_scale_percent()}%")
        text_scale_combo = ttk.Combobox(
            text_scale_row,
            textvariable=self.text_scale_var,
            values=("100%", "125%", "150%", "175%", "200%"),
            width=7,
            state="readonly",
            style="Dark.TCombobox",
            font=f(Theme.F_BODY_SM),
        )
        text_scale_combo.pack(side="right")
        Tooltip(
            text_scale_combo,
            tr("Scales interface text and dependent controls on the next launch."),
        )

        locale_row = tk.Frame(quality_frame, bg=Theme.BG_CARD)
        locale_row.pack(fill="x", padx=Theme.S_LG, pady=(0, Theme.S_SM))
        tk.Label(
            locale_row,
            text=tr("Interface language (restart required)"),
            font=f(Theme.F_BODY_SM),
            bg=Theme.BG_CARD,
            fg=Theme.TEXT_SECONDARY,
        ).pack(side="left")
        self._locale_display_to_tag = {
            tr("System"): "system",
            tr("English"): "en",
        }
        for locale_tag in available_catalogs():
            self._locale_display_to_tag.setdefault(locale_tag, locale_tag)
        saved_locale = getattr(self.config, "ui_locale", "system")
        locale_display = next(
            (
                display
                for display, locale_tag in self._locale_display_to_tag.items()
                if locale_tag.lower() == str(saved_locale).lower()
            ),
            str(saved_locale),
        )
        if locale_display not in self._locale_display_to_tag:
            self._locale_display_to_tag[locale_display] = str(saved_locale)
        self.locale_var = tk.StringVar(value=locale_display)
        self.locale_combo = ttk.Combobox(
            locale_row,
            textvariable=self.locale_var,
            values=tuple(self._locale_display_to_tag),
            width=14,
            state="readonly",
            style="Dark.TCombobox",
            font=f(Theme.F_BODY_SM),
        )
        self.locale_combo.pack(side="right")
        Tooltip(
            self.locale_combo,
            tr("Uses the system language or a compiled catalog on the next launch."),
        )

        # RM-96: high-contrast theme toggle. Takes effect on next launch
        # because re-skinning every live widget mid-session would force
        # a tree-wide redraw the design tokens were not built for.
        self.high_contrast_var = tk.BooleanVar(value=getattr(self.config, "high_contrast", False))
        hc_toggle = ModernToggle(
            quality_frame,
            text=tr("High-contrast theme (restart required)"),
            variable=self.high_contrast_var,
        )
        hc_toggle.pack(anchor="w", padx=Theme.S_LG, pady=(0, Theme.S_MD))
        Tooltip(hc_toggle, tr("Alternative palette tuned for low-vision users. Persists across sessions."))

        self.update_check_var = tk.BooleanVar(value=getattr(self.config, "update_check", False))
        uc_toggle = ModernToggle(
            quality_frame,
            text=tr("Check for updates on startup"),
            variable=self.update_check_var,
        )
        uc_toggle.pack(anchor="w", padx=Theme.S_LG, pady=(0, Theme.S_MD))
        Tooltip(uc_toggle, tr("When enabled, checks GitHub Releases for a newer version on launch. Off by default; no telemetry."))

        storage_frame = self._create_card(self.adv_panel)
        storage_frame.pack(fill="x", pady=(Theme.S_MD, Theme.S_SM))
        self._card_header(
            storage_frame,
            "Storage",
            "Temporary, mask, checkpoint, and resume files",
        )
        storage_row = tk.Frame(storage_frame, bg=Theme.BG_CARD)
        storage_row.pack(fill="x", padx=Theme.S_LG, pady=(0, Theme.S_MD))
        self.work_dir_var = tk.StringVar(
            value=getattr(self.config, "work_directory", ""))
        self.work_dir_entry = tk.Entry(
            storage_row,
            textvariable=self.work_dir_var,
            bg=Theme.BG_TERTIARY,
            fg=Theme.TEXT_PRIMARY,
            insertbackground=Theme.TEXT_PRIMARY,
            font=f(Theme.F_BODY_SM),
            relief="flat",
            bd=6,
            highlightthickness=1,
            highlightbackground=Theme.BORDER,
            highlightcolor=Theme.BORDER_FOCUS,
        )
        self.work_dir_entry.pack(side="left", fill="x", expand=True)
        Tooltip(
            self.work_dir_entry,
            tr("Leave empty for the system temporary directory. The selected "
               "folder is write-tested before a batch starts."),
        )
        work_browse_btn = ModernButton(
            storage_row,
            text=tr("Choose folder"),
            width=118,
            command=self._choose_work_directory,
            style="accent",
            size="sm",
        )
        work_browse_btn.pack(side="left", padx=(Theme.S_SM, 0))
        work_reset_btn = ModernButton(
            storage_row,
            text=tr("Use system"),
            width=96,
            command=self._reset_work_directory,
            style="ghost",
            size="sm",
        )
        work_reset_btn.pack(side="left", padx=(Theme.S_SM, 0))
        self._work_directory_buttons = (work_browse_btn, work_reset_btn)


    def _build_settings_section(self, parent):
        """Settings section: profile + workflow + collapsible advanced controls."""
        section = self._create_surface(parent)
        section.pack(fill="both", expand=True)

        self._section_title(
            section,
            eyebrow="",
            title="Configure",
            hint="",
        )

        settings = tk.Frame(section, bg=Theme.BG_SECONDARY)
        settings.pack(fill="both", expand=True, padx=Theme.S_MD, pady=(0, Theme.S_MD))

        profile_details = self._build_profile_settings_group(settings)
        workflow_details, region_surface = (
            self._build_workflow_settings_group(settings)
        )
        self._build_output_card(settings)
        self._build_inspector_primary_action(settings)

        self._inspector_detail_panels = (
            (profile_details, {
                "fill": "x", "padx": Theme.S_LG,
                "pady": (Theme.S_SM, Theme.S_MD),
            }),
            (workflow_details, {
                "fill": "x", "before": region_surface,
            }),
        )

        # ---- Advanced toggle --------------------------------------------
        adv_frame = tk.Frame(settings, bg=Theme.BG_SECONDARY)
        adv_frame.pack(fill="x", pady=(Theme.S_SM, 0))

        self.adv_visible = False
        self.adv_toggle = ModernButton(adv_frame, text=tr("Show detailed controls"), width=188,
                                       command=self._toggle_advanced,
                                       style="ghost", size="sm", icon="+")
        self.adv_toggle.pack(anchor="w")

        self.adv_panel = tk.Frame(settings, bg=Theme.BG_SECONDARY)

        self._build_sttn_settings_group()
        self._build_detection_settings_group()
        quality_frame = self._build_output_settings_group()
        self._build_range_settings_group()
        self._build_performance_settings_groups()
        self._build_accessibility_storage_settings(quality_frame)

        self._update_region_label_display()
        self._update_mode_options()

    def _build_preview_section(self, parent):
        """Build the central 16:9 preview and its contextual tools."""
        section = self._create_surface(parent)
        section.pack(fill="both", expand=True, pady=(Theme.S_SM, 0))
        self._preview_frame = section

        preview_header = tk.Frame(section, bg=Theme.BG_SECONDARY)
        preview_header.pack(fill="x", padx=Theme.S_MD,
                            pady=(Theme.S_MD, Theme.S_SM))

        preview_text = tk.Frame(preview_header, bg=Theme.BG_SECONDARY)
        preview_text.pack(side="left", fill="x", expand=True)
        self.preview_title_label = tk.Label(
            preview_text, text=tr("Preview"),
            font=f(Theme.F_HEADING, "bold"),
            bg=Theme.BG_SECONDARY, fg=Theme.TEXT_PRIMARY,
        )
        self.preview_title_label.pack(anchor="w")
        self.preview_meta_label = tk.Label(
            preview_text,
            text=tr("Select a queue item to inspect its subtitle region."),
            font=f(Theme.F_META), wraplength=520, justify="left",
            bg=Theme.BG_SECONDARY, fg=Theme.TEXT_MUTED,
        )
        self.preview_meta_label.pack(anchor="w", pady=(3, 0))

        self.preview_status_chip = tk.Label(
            preview_header, text=tr("Waiting"),
            font=f(Theme.F_META, "bold"),
            bg=Theme.BG_SECONDARY, fg=Theme.TEXT_MUTED,
        )
        self.preview_status_chip.pack(side="right", anchor="ne")
        self.preview_ab_btn = ModernButton(
            preview_header, text=tr("Before / after"), width=118,
            command=self._open_ab_scrubber, style="ghost", size="sm",
        )
        self.preview_ab_btn.pack(side="right", padx=(0, Theme.S_SM))
        Tooltip(
            self.preview_ab_btn,
            tr("Wipe between the source frame and cleaned output."),
        )

        media_surface = tk.Frame(
            section, bg=Theme.BG_CARD, highlightthickness=1,
            highlightbackground=Theme.BORDER_SUBTLE,
        )
        media_surface.pack(fill="x", padx=Theme.S_MD)
        self._preview_media_surface = media_surface

        self._preview_label = tk.Label(
            media_surface, bg=Theme.BG_CARD, text="",
            font=f(Theme.F_META), fg=Theme.TEXT_MUTED,
            compound="bottom", justify="center", cursor="hand2",
        )
        self._preview_label.pack(fill="both", expand=True, padx=Theme.S_SM,
                                 pady=Theme.S_SM)
        self._preview_photo = None
        self._preview_label.bind("<Double-Button-1>", self._open_preview_zoom)
        self._preview_label.bind(
            "<ButtonPress-1>", self._on_preview_region_press, add="+")
        self._preview_label.bind(
            "<B1-Motion>", self._on_preview_region_drag, add="+")
        self._preview_label.bind(
            "<ButtonRelease-1>", self._on_preview_region_release, add="+")
        Tooltip(
            self._preview_label,
            tr("Double-click for full size, or draw the subtitle region directly."),
        )

        self.preview_action_hint = tk.Label(
            section,
            text=tr("Add media, then select a queue item to enable preview tools."),
            font=f(Theme.F_META), bg=Theme.BG_SECONDARY,
            fg=Theme.TEXT_MUTED, wraplength=520, justify="left", anchor="w",
        )
        # State is already expressed in the preview title, status, and disabled
        # controls. Keep this label for accessibility/status updates, not layout.

        preview_actions = tk.Frame(section, bg=Theme.BG_SECONDARY)
        preview_actions.pack(fill="x", padx=Theme.S_MD,
                             pady=(Theme.S_SM, Theme.S_MD))
        self._preview_primary_actions = preview_actions

        self.preview_region_btn = ModernButton(
            preview_actions, text=tr("Set region"), width=104,
            command=self._open_region_selector, style="accent", size="sm",
        )
        self.preview_region_btn.pack(side="left")
        Tooltip(self.preview_region_btn,
                tr("Draw the subtitle region directly on the preview frame."))
        self.preview_mask_btn = ModernButton(
            preview_actions, text=tr("Review mask"), width=108,
            command=self._open_selected_mask_preview,
            style="ghost", size="sm",
        )
        self.preview_mask_btn.pack(side="left", padx=(Theme.S_SM, 0))
        Tooltip(self.preview_mask_btn,
                tr("Detect subtitles and inspect the first-frame mask."))
        self.preview_inpaint_btn = ModernButton(
            preview_actions, text=tr("Test cleanup"), width=112,
            command=self._open_selected_inpaint_preview,
            style="ghost", size="sm",
        )
        self.preview_inpaint_btn.pack(side="left", padx=(Theme.S_SM, 0))
        Tooltip(self.preview_inpaint_btn,
                tr("Run detection and cleanup on one sample frame."))

        self.preview_zoom_btn = ModernButton(
            preview_actions, text=tr("Full size"), width=86,
            command=self._open_preview_zoom, style="ghost", size="sm",
        )
        self.preview_zoom_btn.pack(side="left", padx=(Theme.S_SM, 0))
        self.preview_correction_btn = ModernButton(
            preview_actions, text=tr("Correct mask"), width=112,
            command=self._open_selected_mask_correction,
            style="ghost", size="sm",
        )
        self.preview_correction_btn.pack(side="left", padx=(Theme.S_SM, 0))

        Tooltip(self.preview_zoom_btn, tr("Open the source frame at full size."))
        Tooltip(
            self.preview_correction_btn,
            tr("Paint frame-local mask corrections for a selective rerun."),
        )

        self.preview_mask_tuning = tk.Frame(section, bg=Theme.BG_SECONDARY)
        tuning_label = tk.Frame(
            self.preview_mask_tuning, bg=Theme.BG_SECONDARY)
        tuning_label.pack(side="left")
        tk.Label(
            tuning_label,
            text=tr("Mask dilation"),
            font=f(Theme.F_BODY_SM, "bold"),
            bg=Theme.BG_SECONDARY,
            fg=Theme.TEXT_PRIMARY,
        ).pack(anchor="w")
        tk.Label(
            tuning_label,
            text=tr("Adjust the cached detection without running OCR again."),
            font=f(Theme.F_META),
            bg=Theme.BG_SECONDARY,
            fg=Theme.TEXT_MUTED,
        )
        self.preview_mask_dilate_slider = ModernSlider(
            self.preview_mask_tuning,
            from_=0,
            to=20,
            value=self.config.mask_dilate_px,
            command=self._on_preview_mask_dilate_changed,
            bg=Theme.BG_SECONDARY,
            width=190,
            accessible_label=tr("Preview mask dilation"),
        )
        self.preview_mask_dilate_slider.pack(
            side="left", fill="x", expand=True, padx=(Theme.S_LG, Theme.S_SM))
        self.preview_mask_dilate_value_var = tk.StringVar(
            value=f"{self.config.mask_dilate_px} px")
        tk.Label(
            self.preview_mask_tuning,
            textvariable=self.preview_mask_dilate_value_var,
            font=f(Theme.F_META, "bold"),
            bg=Theme.BG_SECONDARY,
            fg=Theme.TEXT_PRIMARY,
        ).pack(side="right")
        self._set_preview_placeholder(
            "Preview",
            "Select a queue item to inspect its subtitle region before cleanup.",
        )

    def _build_queue_section(self, parent):
        """Build the full-width processing queue and batch controls."""
        section = self._create_surface(parent)
        section.pack(fill="both", expand=True)

        header = tk.Frame(section, bg=Theme.BG_SECONDARY)
        header.pack(fill="x", padx=Theme.S_MD, pady=(Theme.S_MD, Theme.S_XS))

        heading = tk.Frame(header, bg=Theme.BG_SECONDARY)
        heading.pack(side="left", fill="x", expand=True)

        tk.Label(heading, text=tr("Processing queue"),
                 font=f(Theme.F_HEADING, "bold"),
                 bg=Theme.BG_SECONDARY, fg=Theme.TEXT_PRIMARY).pack(anchor="w")
        self._queue_subtitle_label = tk.Label(
            heading,
            text=tr("Review files, progress, and output status in one place."),
            font=f(Theme.F_BODY_SM), bg=Theme.BG_SECONDARY,
            fg=Theme.TEXT_MUTED, wraplength=360, justify="left",
        )

        # One inline summary replaces a cluster of decorative status pills.
        count_cluster = tk.Frame(header, bg=Theme.BG_SECONDARY)
        count_cluster.pack(side="right", anchor="n")

        def _mk_stat_pill(fg=Theme.TEXT_SECONDARY, bg=Theme.BG_SECONDARY):
            pill = tk.Frame(count_cluster, bg=Theme.BG_SECONDARY)
            lbl = tk.Label(pill, text="", font=f(Theme.F_META),
                           bg=Theme.BG_SECONDARY, fg=fg)
            lbl.pack()
            return pill, lbl

        self.queue_total_pill, self.queue_count = _mk_stat_pill(
            fg=Theme.TEXT_SECONDARY)
        self.queue_done_pill, self.queue_done_lbl = _mk_stat_pill(
            fg=Theme.SUCCESS)
        self.queue_err_pill, self.queue_err_lbl = _mk_stat_pill(
            fg=Theme.WARNING)

        self.queue_total_pill.pack(side="left")
        # done/err pills get shown conditionally in _update_queue_display
        self.queue_count.config(text=tr("{count} items").format(count=0))

        # Sort button -- hidden until queue has >= 3 items
        self._sort_btn = ModernButton(
            count_cluster, text=tr("Sort"), width=72,
            command=self._open_sort_menu, style="ghost", size="sm")
        # packed conditionally in _update_queue_display

        # Batch progress -- labels row above the bar
        batch_frame = tk.Frame(section, bg=Theme.BG_SECONDARY)
        batch_frame.pack(fill="x", padx=Theme.S_MD, pady=(Theme.S_SM, 0))
        self._queue_batch_frame = batch_frame

        meta_row = tk.Frame(batch_frame, bg=Theme.BG_SECONDARY)
        meta_row.pack(fill="x")

        self.batch_label = tk.Label(meta_row, text=tr("Ready"),
                                    font=f(Theme.F_META, "bold"),
                                    bg=Theme.BG_SECONDARY, fg=Theme.TEXT_MUTED)
        self.batch_label.pack(side="left")

        self.batch_percent_label = tk.Label(meta_row, text="",
                                            font=f(Theme.F_META, "bold"),
                                            bg=Theme.BG_SECONDARY, fg=Theme.TEXT_SECONDARY)
        self.batch_percent_label.pack(side="right")

        batch_bar_frame = tk.Frame(section, bg=Theme.BG_SECONDARY)
        batch_bar_frame.pack(fill="x", padx=Theme.S_MD, pady=(4, Theme.S_SM))
        self._queue_batch_bar_frame = batch_bar_frame

        self.batch_progress = ModernProgressBar(batch_bar_frame, width=300, height=5,
                                                 fill=Theme.BLUE_PRIMARY)
        self.batch_progress.pack(fill="x")

        # Queue filter input -- appears when there are >5 items
        self._queue_filter_var = tk.StringVar()
        self._queue_filter_frame = tk.Frame(
            section, bg=Theme.BG_TERTIARY,
            highlightthickness=1, highlightbackground=Theme.BORDER)
        # Packed/unpacked dynamically in _update_queue_display
        filter_inner = tk.Frame(self._queue_filter_frame, bg=Theme.BG_TERTIARY)
        filter_inner.pack(fill="x", padx=Theme.S_SM, pady=2)

        tk.Label(filter_inner, text=tr("Filter"), font=f(Theme.F_META, "bold"),
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
            filter_inner, text=tr("Clear"), width=68,
            command=lambda: self._queue_filter_var.set(""),
            style="ghost", size="sm")
        self._queue_filter_clear.pack(side="right", padx=(Theme.S_SM, 0))
        self._queue_filter_var.trace_add(
            "write", lambda *_: self._apply_queue_filter())

        self._queue_container = tk.Frame(section, bg=Theme.BG_SECONDARY)
        self._queue_container.pack(fill="both", expand=True,
                                   padx=Theme.S_MD, pady=(0, Theme.S_SM))
        queue_container = self._queue_container

        self.queue_canvas = tk.Canvas(queue_container, bg=Theme.BG_SECONDARY,
                                     highlightthickness=0, height=88)
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

        # One quiet command row with the dominant action anchored at right.
        btn_frame = tk.Frame(section, bg=Theme.BG_SECONDARY)
        btn_frame.pack(fill="x", padx=Theme.S_MD, pady=(0, Theme.S_MD))
        self._queue_action_frame = btn_frame

        self.start_btn = ModernButton(btn_frame, text=tr("Start batch"), width=180,
                                     height=44,
                                     command=self._start_processing,
                                     style="primary", size="lg", icon=">")
        self.start_btn.pack(side="right")

        self.open_output_btn = ModernButton(btn_frame, text=tr("Open output"), width=132,
                                            command=self._open_output_folder,
                                            style="ghost", size="lg", icon="^")
        self.open_output_btn.pack(side="right", padx=(0, Theme.S_SM))

        self.retry_btn = ModernButton(btn_frame, text=tr("Retry failed"), width=124,
                                      command=self._retry_failed,
                                      style="ghost", size="lg")
        self.retry_btn.pack(side="left")

        self.repeat_btn = ModernButton(btn_frame, text=tr("Repeat last"), width=120,
                                      command=self._repeat_last_job,
                                      style="ghost", size="lg")
        self.repeat_btn.pack(side="left", padx=(Theme.S_SM, 0))

        self.clear_btn = ModernButton(btn_frame, text=tr("Clear queue"), width=120,
                                     command=self._clear_queue,
                                     style="ghost", size="lg")
        self.clear_btn.pack(side="left", padx=(Theme.S_SM, 0))

        self._queue_more_btn = ModernButton(
            btn_frame, text=tr("Queue actions"), width=124,
            command=self._open_queue_actions_menu,
            style="ghost", size="lg", icon="...",
        )

        self._refresh_action_states()
        if not self.queue:
            self._queue_batch_frame.pack_forget()
            self._queue_batch_bar_frame.pack_forget()

    def _build_queue_empty_state(self):
        """Compact empty state that fits the persistent queue surface."""
        self.empty_container = tk.Frame(self.queue_frame, bg=Theme.BG_SECONDARY)
        self.empty_container.pack(pady=Theme.S_MD, fill="x")

        copy = tk.Frame(self.empty_container, bg=Theme.BG_SECONDARY)
        copy.pack(side="left", fill="x", expand=True)
        tk.Label(
            copy, text=tr("No media queued"),
            font=f(Theme.F_TITLE, "bold"),
            bg=Theme.BG_SECONDARY, fg=Theme.TEXT_SECONDARY,
        ).pack(anchor="w")
        tk.Label(
            copy,
            text=tr("Add videos or images. Originals are never modified."),
            font=f(Theme.F_BODY_SM), bg=Theme.BG_SECONDARY,
            fg=Theme.TEXT_MUTED, wraplength=460, justify="left",
        ).pack(anchor="w", pady=(Theme.S_XS, 0))

        actions = tk.Frame(self.empty_container, bg=Theme.BG_SECONDARY)
        actions.pack(side="right", padx=(Theme.S_LG, 0))
        choose_files = ModernButton(
            actions,
            text=tr("Choose files"),
            width=116,
            command=self._open_file_picker,
            style="accent",
            size="sm",
        )
        choose_files.pack(side="left")
        Tooltip(choose_files, tr("Choose one or more videos or images to add to the queue."))
        choose_folder = ModernButton(
            actions,
            text=tr("Choose folder"),
            width=122,
            command=self._open_folder_picker,
            style="ghost",
            size="sm",
        )
        choose_folder.pack(side="left", padx=(Theme.S_SM, 0))
        Tooltip(choose_folder, tr("Add every supported media file from a folder."))

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
            text=tr("No queued items match this search"),
            font=f(Theme.F_TITLE, "bold"),
            bg=Theme.BG_SECONDARY,
            fg=Theme.TEXT_SECONDARY,
        )
        self._filter_empty_title.pack(pady=(Theme.S_MD, 4))
        self._filter_empty_body = tk.Label(
            self._filter_empty_container,
            text=tr("Clear the filter or search for part of a filename."),
            font=f(Theme.F_BODY_SM),
            bg=Theme.BG_SECONDARY,
            fg=Theme.TEXT_MUTED,
            wraplength=340,
            justify="center",
        )
        self._filter_empty_body.pack()
        ModernButton(
            self._filter_empty_container,
            text=tr("Clear filter"),
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

    def _build_footer(self, parent):
        """Footer status bar with a colored dot + message and a right-side hint."""
        footer = tk.Frame(parent, bg=Theme.BG_DARK)
        footer.pack(side="bottom", fill="x", pady=(Theme.S_SM, 0))
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

        self.status_label = tk.Label(left, text=tr("Ready to process"),
                                     font=f(Theme.F_BODY_SM, "bold"),
                                     bg=Theme.BG_DARK, fg=Theme.TEXT_SECONDARY, anchor="w")
        self.status_label.pack(side="left")
        try:
            from backend.a11y import set_accessible_metadata
            set_accessible_metadata(
                self.status_label,
                role="status",
                label=tr("Application status"),
                state="neutral",
                value=tr("Ready to process"),
            )
        except Exception:
            pass

        self.status_hint = tk.Label(
            footer,
            text=tr("Add files, review a sample frame, then start."),
            font=f(Theme.F_META),
            bg=Theme.BG_DARK,
            fg=Theme.TEXT_MUTED,
        )
        self.status_hint.pack(side="right")

        self._footer_activity_btn = ModernButton(
            footer, text=tr("Activity"), width=92,
            command=self._toggle_log_panel,
            style="ghost", size="sm",
        )
        self._footer_activity_btn.pack(side="right", padx=(0, Theme.S_MD))















    def _maybe_show_onboarding(self):
        """Show a short 3-card welcome overlay on first launch."""
        if self.config.onboarding_seen:
            return
        # In-session guard prevents a double-schedule, but the *persisted*
        # onboarding_seen flag is only set once the dialog actually builds
        # (in _show_onboarding) so a failure here does not permanently hide
        # onboarding from the user.
        if getattr(self, "_onboarding_scheduled", False):
            return
        self._onboarding_scheduled = True
        # Let the main window settle first
        try:
            self.root.after(420, self._show_onboarding)
        except tk.TclError:
            self._onboarding_scheduled = False

    def _apply_onboarding_preset(self, name: str):
        """Apply a first-run preset through the regular settings path."""
        self.preset_var.set(name)
        self._on_preset_applied()

    def _enable_onboarding_auto_band(self):
        """Enable automatic subtitle-band detection and persist it."""
        self.config.auto_band = True
        if hasattr(self, "auto_band_var"):
            self.auto_band_var.set(True)
        save_settings(self.config)
        self._update_status("Automatic subtitle-band detection enabled", "success")

    def _schedule_onboarding_test_cleanup(self):
        """Open the ordinary one-frame cleanup flow after the modal closes."""
        try:
            self.root.after_idle(self._open_selected_inpaint_preview)
        except tk.TclError:
            pass

    def _show_onboarding(self):
        dialog = tk.Toplevel(self.root)
        dialog.withdraw()
        dialog.title(f"Welcome to {APP_NAME}")
        dialog.configure(bg=Theme.BG_OVERLAY)
        dialog.resizable(False, False)
        dialog.transient(self.root)
        try:
            from backend.a11y import set_accessible_metadata
            set_accessible_metadata(
                dialog,
                role="dialog",
                label=tr("Welcome to {app_name}").format(app_name=APP_NAME),
                state="modal",
                description=(
                    tr("Three first-run cues: import media, inspect the "
                       "region, and run the batch.")
                ),
            )
        except Exception:
            pass

        outer = tk.Frame(dialog, bg=Theme.BORDER, padx=1, pady=1)
        outer.pack()
        body = tk.Frame(outer, bg=Theme.BG_SECONDARY)
        body.pack()

        content = tk.Frame(body, bg=Theme.BG_SECONDARY)
        content.pack(padx=36, pady=(28, 16))

        # Headline
        hero = tk.Frame(content, bg=Theme.BG_SECONDARY)
        hero.pack(anchor="w")
        tk.Label(hero, text=tr("Welcome"), font=f(Theme.F_DISPLAY, "bold"),
                 bg=Theme.BG_SECONDARY, fg=Theme.TEXT_PRIMARY).pack(
                     side="left")
        tk.Label(hero, text=f"v{APP_VERSION}", font=f(Theme.F_BODY_SM),
                 bg=Theme.BG_SECONDARY, fg=Theme.TEXT_MUTED).pack(
                     side="left", padx=(Theme.S_SM, 0), pady=(14, 0))

        tk.Label(content,
                 text=tr("Three things that make batch cleanup painless."),
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
            tk.Label(top, text=tr(heading), font=f(Theme.F_BODY, "bold"),
                     bg=Theme.BG_CARD, fg=Theme.TEXT_PRIMARY).pack(
                         side="left", padx=(Theme.S_SM, 0))
            tk.Label(inner, text=tr(body_text), font=f(Theme.F_BODY_SM),
                     bg=Theme.BG_CARD, fg=Theme.TEXT_SECONDARY,
                     wraplength=220, justify="left", anchor="w").pack(
                         anchor="w", pady=(Theme.S_SM, 0))
            return c

        card("1", "Import media",
             "Drop videos or images on the left, or pick an entire folder. "
             "Originals are never modified.",
             "info").pack(side="left", fill="both", expand=True,
                          padx=(0, Theme.S_SM))
        card("2", "Inspect the region",
             "Click a queue item, then use Set region to draw the subtitle band "
             "or Review mask to see what the detector finds.",
             "warning").pack(side="left", fill="both", expand=True,
                             padx=(0, Theme.S_SM))
        card("3", "Run the batch",
             "Hit Start batch when the framing looks right. Progress, ETA, "
             "and completion summary are all live.",
             "success").pack(side="left", fill="both", expand=True)

        # First-run profile chooser. These use the normal preset application
        # path so every dependent toggle and slider refreshes immediately.
        starter = tk.Frame(content, bg=Theme.BG_SECONDARY)
        starter.pack(fill="x", pady=(Theme.S_LG, 0))
        tk.Label(
            starter,
            text=tr("Choose a starting profile"),
            font=f(Theme.F_BODY, "bold"),
            bg=Theme.BG_SECONDARY,
            fg=Theme.TEXT_PRIMARY,
        ).pack(anchor="w")
        tk.Label(
            starter,
            text=tr("You can change every setting later."),
            font=f(Theme.F_META),
            bg=Theme.BG_SECONDARY,
            fg=Theme.TEXT_MUTED,
        ).pack(anchor="w", pady=(2, Theme.S_SM))
        starter_buttons = tk.Frame(starter, bg=Theme.BG_SECONDARY)
        starter_buttons.pack(anchor="w")
        onboarding_choice_var = tk.StringVar(value="")

        def _choose_preset(name: str):
            self._apply_onboarding_preset(name)
            onboarding_choice_var.set(
                tr("Selected: {profile}").format(profile=name)
            )

        for index, (label, preset_name) in enumerate((
            ("YouTube", "YouTube (default)"),
            ("Film", "Film / Live action"),
            ("Fast", "Fast"),
        )):
            ModernButton(
                starter_buttons,
                text=tr(label),
                width=104,
                command=lambda name=preset_name: _choose_preset(name),
                style="ghost",
                size="sm",
            ).pack(side="left", padx=(0 if index == 0 else Theme.S_SM, 0))
        tk.Label(
            starter_buttons,
            textvariable=onboarding_choice_var,
            font=f(Theme.F_META),
            bg=Theme.BG_SECONDARY,
            fg=Theme.SUCCESS,
        ).pack(side="left", padx=(Theme.S_MD, 0))

        # Action row
        actions = tk.Frame(body, bg=Theme.BG_CARD)
        actions.pack(fill="x")
        quick_actions = tk.Frame(actions, bg=Theme.BG_CARD)
        quick_actions.pack(side="left", padx=16, pady=14)
        actions_inner = tk.Frame(actions, bg=Theme.BG_CARD)
        actions_inner.pack(side="right", padx=16, pady=14)

        def _close():
            self.config.onboarding_seen = True
            save_settings(self.config)
            dialog.grab_release()
            dialog.destroy()

        def _try_cleanup():
            _close()
            self._schedule_onboarding_test_cleanup()

        ModernButton(
            quick_actions,
            text=tr("Enable auto-detect"),
            width=156,
            command=self._enable_onboarding_auto_band,
            style="ghost",
            size="sm",
        ).pack(side="left")
        ModernButton(
            quick_actions,
            text=tr("Try test cleanup"),
            width=142,
            command=_try_cleanup,
            style="ghost",
            size="sm",
        ).pack(side="left", padx=(Theme.S_SM, 0))
        ModernButton(actions_inner, text=tr("Got it"), width=118,
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
        # The dialog is now on screen; mark it seen in memory. The close path
        # persists the flag so a background-scheduled dialog cannot write
        # unrelated in-progress settings before the user dismisses it.
        self.config.onboarding_seen = True











    @staticmethod
    def _safe_float(value: str, default: float = 0.0) -> float:
        """Parse a float from a string, returning default on failure."""
        try:
            return float(value or default)
        except (ValueError, TypeError):
            return default

    @staticmethod
    def _safe_int(value: str, default: int = 0) -> int:
        """Parse an int from a string, returning default on failure."""
        try:
            return int(value or default)
        except (ValueError, TypeError):
            return default

    @staticmethod
    def _gui_to_backend_mode(gui_mode_value: str):
        """Map a GUI InpaintMode value onto the backend enum. The two
        enums are deliberately separate -- this mapping
        is the single place they meet."""
        from backend.config import InpaintMode as _BM
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
                    "Use Set region to draw the subtitle band, or Review mask to confirm what the detector finds automatically."
                )
            )
        save_queue_state(self.queue)

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
        save_queue_state(self.queue)
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
        if refreshed:
            save_queue_state(self.queue)
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
            if unsupported:
                parts.append(f"skipped {unsupported} unsupported file{'s' if unsupported != 1 else ''}")
            if missing:
                parts.append(f"skipped {missing} missing file{'s' if missing != 1 else ''}")
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
            detail = (
                f"Skipped {unsupported} unsupported file{'s' if unsupported != 1 else ''}. "
                "Only video and image formats can be queued"
            )
            self._update_status(detail, "warning")
            logger.warning(detail)
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
        for candidate in sorted(folder.rglob("*")):
            if candidate.is_file() and (
                is_video_file(str(candidate)) or is_image_file(str(candidate))
            ):
                stats["supported_in_folders"] += 1
                result = self._add_to_queue(str(candidate))
                stats[result] = stats.get(result, 0) + 1
                if result == "queue_full":
                    break
        return stats

    def _try_enqueue_queue_item(self, item: QueueItem) -> tuple[str, int]:
        """Atomically enforce queue capacity/deduplication and append *item*."""
        normalized = self._normalized_path_key(item.file_path)
        with self.queue_lock:
            if len(self.queue) >= 500:
                return "queue_full", len(self.queue)
            if any(
                self._normalized_path_key(existing.file_path) == normalized
                for existing in self.queue
            ):
                return "duplicate", len(self.queue)
            self.queue.append(item)
            return "added", len(self.queue)

    def _add_to_queue(self, file_path: str):
        """Add a file to the processing queue."""
        # Check file exists and is valid
        if not Path(file_path).is_file():
            logger.warning(f"File not found: {file_path}")
            return "missing"
        if not (is_video_file(file_path) or is_image_file(file_path)):
            logger.warning(f"Unsupported file type: {file_path}")
            return "unsupported"

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

        result, queue_size = self._try_enqueue_queue_item(item)
        if result == "queue_full":
            logger.warning("Queue full (500 items max)")
            return result
        if result == "duplicate":
            logger.info(f"Already in queue: {Path(file_path).name}")
            return result
        self._update_queue_display()
        if is_video_file(file_path):
            self._start_soft_subtitle_probe(item)
        if queue_size == 1 and not self.is_processing:
            self._show_preview(item)
        logger.info(f"Queued: {Path(file_path).name} ({get_file_info(file_path)})")
        save_queue_state(self.queue)
        return "added"

    def _open_sort_menu(self):
        """Pop up a themed sort menu anchored to the sort button."""
        if self.is_processing:
            self._update_status(
                "Sorting is disabled while a batch is running", "warning")
            return
        menu = make_themed_menu(self.root)
        menu.add_command(label=tr("Filename (A -> Z)"),
                         command=lambda: self._sort_queue("name_asc"))
        menu.add_command(label=tr("Filename (Z -> A)"),
                         command=lambda: self._sort_queue("name_desc"))
        menu.add_separator()
        menu.add_command(label=tr("File size (largest first)"),
                         command=lambda: self._sort_queue("size_desc"))
        menu.add_command(label=tr("File size (smallest first)"),
                         command=lambda: self._sort_queue("size_asc"))
        menu.add_separator()
        menu.add_command(label=tr("Status (pending first)"),
                         command=lambda: self._sort_queue("status"))
        menu.add_command(label=tr("Reverse current order"),
                         command=lambda: self._sort_queue("reverse"))
        try:
            bx = self._sort_btn.winfo_rootx()
            by = self._sort_btn.winfo_rooty() + self._sort_btn.winfo_height() + 2
            menu.tk_popup(bx, by)
        finally:
            menu.grab_release()
            menu.destroy()

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
                ProcessingStatus.PAUSED: 6,
                ProcessingStatus.CANCELLED: 7,
                ProcessingStatus.ERROR: 8,
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
        save_queue_state(self.queue)
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
        dialog.title(
            tr("Override settings: {name}").format(
                name=Path(item.file_path).name))
        dialog.configure(bg=Theme.BG_OVERLAY)
        dialog.resizable(False, False)
        dialog.transient(self.root)
        try:
            from backend.a11y import set_accessible_metadata
            set_accessible_metadata(
                dialog,
                role="dialog",
                label=tr("Per-file overrides"),
                state="modal",
                description=tr(
                    "Change cleanup settings for this queue item only. "
                    "Press Control+Enter to save or Escape to cancel."
                ),
            )
        except Exception:
            pass

        outer = tk.Frame(dialog, bg=Theme.BORDER, padx=1, pady=1)
        outer.pack()
        body = tk.Frame(outer, bg=Theme.BG_SECONDARY)
        body.pack()

        content = tk.Frame(body, bg=Theme.BG_SECONDARY)
        content.pack(padx=24, pady=(20, 12))

        tk.Label(content, text=tr("Per-file overrides"),
                 font=f(Theme.F_HEADING, "bold"),
                 bg=Theme.BG_SECONDARY, fg=Theme.TEXT_PRIMARY).pack(anchor="w")
        tk.Label(content,
                 text=tr("These apply to this queued item only and survive a global settings change."),
                 font=f(Theme.F_BODY_SM),
                 bg=Theme.BG_SECONDARY, fg=Theme.TEXT_SECONDARY,
                 wraplength=380, justify="left").pack(anchor="w", pady=(2, Theme.S_LG))

        # Mode picker.
        mode_var = tk.StringVar(value=item.config.mode.value)
        tk.Label(content, text=tr("Mode"), font=f(Theme.F_BODY_SM),
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
        tk.Label(lang_row, text=tr("Subtitle language"), font=f(Theme.F_BODY_SM),
                 bg=Theme.BG_SECONDARY, fg=Theme.TEXT_SECONDARY).pack(side="left")
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
        tk.Label(sens_row, text=tr("Sensitivity"), font=f(Theme.F_BODY_SM),
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
        tk.Label(codec_row, text=tr("Output codec"), font=f(Theme.F_BODY_SM),
                 bg=Theme.BG_SECONDARY, fg=Theme.TEXT_SECONDARY).pack(side="left")
        codec_var = tk.StringVar(value=getattr(item.config, "output_codec", "h264"))
        ttk.Combobox(
            codec_row, textvariable=codec_var, width=8,
            values=["h264", "h265", "av1", "vvc"],
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
            save_queue_state(self.queue)
            self._update_status(
                f"Overrides saved for {Path(item.file_path).name}",
                "success",
            )
            dialog.destroy()

        ModernButton(actions_inner, text=tr("Cancel"), command=dialog.destroy,
                     style="ghost", size="md", width=96).pack(side="left")
        ModernButton(actions_inner, text=tr("Save"), command=_save,
                     style="primary", size="md", width=96).pack(
                         side="left", padx=(Theme.S_SM, 0))

        dialog.bind("<Escape>", lambda e: dialog.destroy())
        dialog.bind("<Control-Return>", lambda e: _save())
        dialog.protocol("WM_DELETE_WINDOW", dialog.destroy)
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
        mode_picker.focus_set()

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
            id=uuid.uuid4().hex,
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
        save_queue_state(self.queue)
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
        save_queue_state(self.queue)
        if self._normalized_path_key(new_path) != self._normalized_path_key(resolved_path):
            self._update_status(
                f"Output renamed to {resolved_path.name} to avoid an overwrite",
                "success",
            )
        else:
            self._update_status(
                f"Output renamed to {resolved_path.name}", "success")

    def _try_dequeue_queue_item(
        self, item_id: str,
    ) -> tuple[Optional[QueueItem], str]:
        """Atomically remove an inactive item, returning its decision state."""
        with self.queue_lock:
            item = next((i for i in self.queue if i.id == item_id), None)
            if item is None:
                return None, "missing"
            if item.status in (
                ProcessingStatus.LOADING,
                ProcessingStatus.DETECTING,
                ProcessingStatus.PROCESSING,
                ProcessingStatus.MERGING,
            ):
                return item, "busy"
            self.queue = [i for i in self.queue if i.id != item_id]
            return item, "removed"

    def _remove_from_queue(self, item_id: str):
        """Remove an item from the queue."""
        item, result = self._try_dequeue_queue_item(item_id)
        if result == "busy":
            self._update_status(
                "Wait for the active item to finish before removing it",
                "warning",
            )
            return
        if self._selected_queue_item_id == item_id:
            self._selected_queue_item_id = None
        self._update_queue_display()
        if item:
            self._update_status(f"Removed {Path(item.file_path).name} from the queue")
        save_queue_state(self.queue)

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
        clear_queue_state()

    @staticmethod
    def _queue_item_needs_quality_review(item: QueueItem) -> bool:
        report = getattr(item, "quality_report", None)
        if not isinstance(report, dict):
            return False
        gate = report.get("quality_gate")
        if isinstance(gate, dict) and gate.get("status") == "review":
            return True
        return str(report.get("tag") or "").strip().lower() == "review"

    @classmethod
    def _queue_attention_count(cls, queue: List[QueueItem]) -> int:
        attention_states = (
            ProcessingStatus.ERROR,
            ProcessingStatus.PAUSED,
            ProcessingStatus.CANCELLED,
        )
        return sum(
            1
            for item in queue
            if item.status in attention_states
            or (
                item.status == ProcessingStatus.COMPLETE
                and cls._queue_item_needs_quality_review(item)
            )
        )

    def _update_queue_display(self):
        """Update the queue display. Only rebuilds widgets that changed."""
        with self.queue_lock:
            current_ids = {item.id for item in self.queue}

        # Remove widgets for items no longer in queue
        stale_ids = [wid for wid in self.queue_widgets if wid not in current_ids]
        for wid in stale_ids:
            self.queue_widgets[wid].destroy()
            del self.queue_widgets[wid]

        # Update the inline queue summary.
        total = len(self.queue)
        self.queue_count.config(text=f"{total} item{'s' if total != 1 else ''}")
        done = sum(1 for i in self.queue if i.status == ProcessingStatus.COMPLETE)
        attention = self._queue_attention_count(self.queue)
        if done > 0:
            self.queue_done_lbl.config(text=f" / {done} done")
            self.queue_done_pill.pack(side="left", padx=(Theme.S_XS, 0))
        else:
            self.queue_done_pill.pack_forget()
        if attention > 0:
            self.queue_err_lbl.config(text=f" / {attention} needs attention")
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
                "Add files to preview them. Use Set region to mark the subtitle band before processing.",
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
                                             on_soft_action=self._set_soft_subtitle_action,
                                             on_retry_suggested=self._retry_review_item_with_suggested_settings)
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
        elif self.queue:
            self._set_selected_queue_item(self.queue[0].id)
        else:
            self._set_selected_queue_item(None)
        self._refresh_action_states()
        self._layout_queue_actions(
            compact=self._layout_mode == "stacked",
            dense=self._text_scale_percent >= 150,
        )
        # Show filter only when the queue is long enough to justify it
        try:
            if len(self.queue) >= 6:
                self._queue_filter_frame.pack(
                    fill="x", padx=Theme.S_MD, pady=(0, Theme.S_SM),
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

    @staticmethod
    def _footer_status_text(message: str) -> str:
        """Keep the footer status line stable for long summaries."""
        return truncate_middle(str(message), 132)

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
        display_message = tr(message)
        self.status_label.config(text=self._footer_status_text(display_message), fg=color)
        try:
            self.status_dot.itemconfig(self._status_dot_item, fill=color)
        except Exception:
            pass
        self._status_tone = tone
        try:
            from backend.a11y import announce, set_accessible_metadata
            set_accessible_metadata(
                self.status_label,
                role="status",
                label=tr("Application status"),
                state=tone,
                value=display_message,
            )
            if toast or tone in {"error", "warning", "success"}:
                announce(display_message, importance="high" if tone == "error" else "normal")
        except Exception:
            pass
        if toast:
            try:
                Toast.show(self.root, display_message, tone=tone)
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
            save_queue_state(self.queue)
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
            save_queue_state(self.queue)
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
            if hasattr(self, "ocr_engine_combo"):
                self.ocr_engine_combo.config(state=combo_state)
            if hasattr(self, "language_filter_toggle"):
                self.language_filter_toggle.set_enabled(not locked)
            if hasattr(self, 'gpu_combo'):
                self.gpu_combo.config(state=combo_state)
            self.time_start_entry.config(state=entry_state)
            self.time_end_entry.config(state=entry_state)
            if hasattr(self, "work_dir_entry"):
                self.work_dir_entry.config(state=entry_state)
            for button in getattr(self, "_work_directory_buttons", ()):
                button.set_enabled(not locked)

            self.region_btn.set_enabled(not locked)
            if hasattr(self, "preview_region_btn"):
                self.preview_region_btn.set_enabled(not locked)
            has_manual_region = (
                self.config.subtitle_area is not None
                or bool(getattr(self.config, "subtitle_areas", None))
                or bool(getattr(self.config, "subtitle_region_spans", None))
                or bool(getattr(
                    self.config, "subtitle_region_keyframes", None))
            )
            self.region_reset_btn.set_enabled(
                (not locked) and has_manual_region)
            self.adv_toggle.set_enabled(not locked)
            if hasattr(self, "drop_area"):
                self.drop_area.set_import_enabled(not locked)
            for slider in getattr(self, "_settings_sliders", []):
                slider.set_enabled(not locked)
            # Segmented algo picker: dim/undim each segment
            try:
                for seg in self.mode_picker._segments.values():
                    if hasattr(seg, "set_enabled"):
                        seg.set_enabled(not locked)
                    else:
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

    def _preflight_free_space_check(self):
        """Validate work storage and estimate every affected batch volume."""
        try:
            from backend.work_directory import (
                StorageRequirement,
                assess_storage_volumes,
                resolve_work_directory,
            )

            resolution = resolve_work_directory(
                getattr(self.config, "work_directory", ""))
            if resolution.warning:
                self._update_status(
                    resolution.warning,
                    "warning",
                    toast=True,
                )
            with self.queue_lock:
                pending = [
                    item for item in self.queue
                    if item.status not in {
                        ProcessingStatus.COMPLETE,
                        ProcessingStatus.CANCELLED,
                    }
                ]
            source_sizes = []
            for item in pending:
                try:
                    source_sizes.append(max(0, Path(item.file_path).stat().st_size))
                except OSError:
                    source_sizes.append(0)
            requirements = [StorageRequirement(
                resolution.path,
                max(512 * 1024 * 1024, sum(source_sizes) * 6),
                "batch temporary/checkpoint files",
            )]
            for item, source_size in zip(pending, source_sizes):
                requirements.append(StorageRequirement(
                    Path(item.output_path).parent,
                    max(64 * 1024 * 1024, source_size),
                    "batch outputs",
                ))
            for status in assess_storage_volumes(requirements):
                if status.free_bytes < status.required_bytes:
                    self._update_status(
                        f"Low disk space at {status.path}: about "
                        f"{status.required_bytes / (1024 ** 3):.1f} GB is "
                        f"estimated for {', '.join(status.purposes)}, but "
                        f"{status.free_bytes / (1024 ** 3):.1f} GB is free. "
                        "The backend will run an exact frame-based check.",
                        "warning",
                        toast=True,
                    )
        except Exception:
            logger.warning("Storage preflight failed", exc_info=True)

    def _current_ffmpeg_profiles(self) -> dict:
        profiles = getattr(self, "ffmpeg_profiles", None)
        if isinstance(profiles, dict) and profiles.get("profiles"):
            return profiles
        try:
            profiles = collect_ffmpeg_capability_profiles(timeout=6.0)
            self.ffmpeg_profiles = profiles
            return profiles
        except Exception as exc:
            logger.warning("FFmpeg profile preflight failed", exc_info=True)
            return {
                "schema": FFMPEG_PROFILE_SCHEMA,
                "profiles": [{
                    "name": "basic",
                    "available": False,
                    "reason": str(exc)[:160],
                    "missing": {},
                }],
            }

    def _confirm_ffmpeg_profile_coverage(self) -> bool:
        """Warn when pending video settings exceed the installed FFmpeg build."""
        profiles = self._current_ffmpeg_profiles()
        with self.queue_lock:
            pending = [
                item for item in self.queue
                if item.status not in (
                    ProcessingStatus.COMPLETE,
                    ProcessingStatus.ERROR,
                    ProcessingStatus.CANCELLED,
                ) and is_video_file(item.file_path)
            ]
        issues = []
        for item in pending:
            missing = missing_profile_requirements_for_config(
                item.config,
                profiles,
            )
            if missing:
                issues.append((item, missing))
        if not issues:
            return True

        lines = []
        for item, missing in issues[:4]:
            name = Path(item.file_path).name
            lines.append(
                f"{name}: {summarize_missing_profile_requirements(missing)}"
            )
        if len(issues) > 4:
            lines.append(f"...and {len(issues) - 4} more queued item(s).")
        detail = (
            "\n".join(lines)
            + "\n\nContinuing may skip optional metrics, drop audio, or fall "
              "back to a different encoder."
        )
        if show_confirm(
            self.root,
            title="FFmpeg capability warning",
            message="Some selected batch options are not supported by this FFmpeg build.",
            detail=detail,
            confirm_label="Start anyway",
            cancel_label="Review settings",
            tone="warning",
        ):
            logger.warning("Starting despite FFmpeg capability gaps: %s", detail)
            return True
        self._update_status(
            "Batch not started. Review FFmpeg-dependent settings or install a fuller FFmpeg build.",
            "warning",
        )
        return False

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
                if self._queue_item_needs_quality_review(item):
                    self._update_status(
                        f"{fname} completed; quality review recommended",
                        "warning",
                    )
                else:
                    self._update_status(f"Completed {fname}", "success")
            elif item.status == ProcessingStatus.ERROR:
                self._update_status(f"{fname} needs attention: {item.message}", "error")
            elif item.status == ProcessingStatus.PAUSED:
                self._update_status(f"Paused {fname}: resume from checkpoint", "warning")
            elif item.status == ProcessingStatus.CANCELLED:
                self._update_status(f"Stopped {fname}", "warning")
            else:
                self._update_status(f"{fname}: {item.message}", "info")
            self._refresh_action_states()

        try:
            self.root.after(0, update)
        except RuntimeError:
            pass  # root already destroyed during shutdown

    def _maybe_restore_queue(self):
        """Restore saved queue items from a previous session."""
        saved = load_queue_state()
        if not saved:
            return
        valid = [r for r in saved if Path(r.get("file_path", "")).is_file()]
        if not valid:
            clear_queue_state()
            return
        n = len(valid)
        label = f"{n} queued item{'s' if n != 1 else ''} from last session"
        if not show_confirm(
            self.root,
            title="Restore queue?",
            message=label,
            detail="Idle and paused items from your previous session were saved. Restore them to the queue?",
            confirm_label="Restore",
            cancel_label="Discard",
        ):
            clear_queue_state()
            return
        restored: List[QueueItem] = []
        seen_ids = {item.id for item in self.queue}
        seen_paths = {
            self._normalized_path_key(item.file_path) for item in self.queue
        }
        resumable_statuses = {
            ProcessingStatus.IDLE,
            ProcessingStatus.PAUSED,
            ProcessingStatus.ERROR,
            ProcessingStatus.CANCELLED,
        }
        for record in valid:
            path_key = self._normalized_path_key(record["file_path"])
            if path_key in seen_paths:
                continue
            cfg = ProcessingConfig.from_dict(record.get("config", {}))
            try:
                status = ProcessingStatus(
                    record.get("status", ProcessingStatus.IDLE.value))
            except ValueError:
                status = ProcessingStatus.IDLE
            if status not in resumable_statuses:
                status = (
                    ProcessingStatus.PAUSED
                    if record.get("pause_checkpoint_path")
                    else ProcessingStatus.IDLE
                )
            item_id = str(record.get("id") or uuid.uuid4().hex)
            if item_id in seen_ids:
                item_id = uuid.uuid4().hex
            item = QueueItem(
                id=item_id,
                file_path=record["file_path"],
                output_path=record["output_path"],
                output_path_locked=bool(
                    record.get("output_path_locked", True)),
                config=cfg,
                status=status,
                progress=max(0.0, min(1.0, float(
                    record.get("progress") or 0.0))),
                message=str(record.get("message") or "Ready to process"),
                error=(
                    str(record.get("error"))
                    if record.get("error") is not None else None
                ),
                stage_timings=(
                    dict(record.get("stage_timings"))
                    if isinstance(record.get("stage_timings"), dict)
                    else {}
                ),
                detection_stats=(
                    dict(record.get("detection_stats"))
                    if isinstance(record.get("detection_stats"), dict)
                    else {}
                ),
                pause_checkpoint_path=str(
                    record.get("pause_checkpoint_path") or ""),
                soft_subtitle_streams=list(
                    record.get("soft_subtitle_streams") or []),
                soft_subtitle_probe_done=bool(
                    record.get("soft_subtitle_probe_done", False)),
                soft_subtitle_action=str(
                    record.get("soft_subtitle_action") or "burned_in"),
                retry_config=(
                    record.get("retry_config")
                    if isinstance(record.get("retry_config"), dict)
                    else None
                ),
                retry_attempts=max(0, int(
                    record.get("retry_attempts") or 0)),
                retry_errors=list(record.get("retry_errors") or []),
                mask_export=(
                    dict(record.get("mask_export"))
                    if isinstance(record.get("mask_export"), dict)
                    else {}
                ),
                mask_import=(
                    dict(record.get("mask_import"))
                    if isinstance(record.get("mask_import"), dict)
                    else {}
                ),
                timing_report=(
                    dict(record.get("timing_report"))
                    if isinstance(record.get("timing_report"), dict)
                    else {}
                ),
                output_contract_report=(
                    dict(record.get("output_contract_report"))
                    if isinstance(
                        record.get("output_contract_report"), dict)
                    else {}
                ),
                correction_retry=(
                    dict(record.get("correction_retry"))
                    if isinstance(record.get("correction_retry"), dict)
                    else None
                ),
                selective_rerun=(
                    dict(record.get("selective_rerun"))
                    if isinstance(record.get("selective_rerun"), dict)
                    else {}
                ),
            )
            if item.soft_subtitle_action not in {
                "strip", "keep_all", "burned_in"
            }:
                item.soft_subtitle_action = "burned_in"
            restored.append(item)
            seen_ids.add(item.id)
            seen_paths.add(path_key)

        with self.queue_lock:
            self.queue.extend(restored)
        self._update_queue_display()
        for item in restored:
            if (
                is_video_file(item.file_path)
                and not item.soft_subtitle_probe_done
            ):
                self._start_soft_subtitle_probe(item)
        # Replace the consumed snapshot with the reconstructed queue. Keeping
        # this atomic snapshot makes a second crash/restore cycle lossless.
        save_queue_state(self.queue)
        self._update_status(f"Restored {n} item{'s' if n != 1 else ''} from last session")

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
                w = int(w_s)
                h = int(h_s)
                if pos_part:
                    x_s, _, y_s = pos_part.partition('+')
                    x = int(x_s)
                    y = int(y_s)
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
