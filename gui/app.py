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
)
from gui.config import (
    APP_NAME, APP_VERSION, LOG_DIR, LOG_FILE, InpaintMode, ProcessingStatus, ProcessingConfig, QueueItem,
    _coerce_int, _coerce_region_span_list,
    consume_settings_load_notice, load_settings, save_settings,
    save_queue_state, load_queue_state, clear_queue_state,
)
from gui.utils import (
    get_app_dir, detect_gpu, is_video_file, is_image_file,
    detect_ai_engines, detect_ffmpeg, get_file_info,
    _soft_subtitle_stream_record, _format_soft_subtitle_summary,
    truncate_middle,
)
from gui.widgets import (
    ModernButton, ModernSlider, show_confirm, TaskbarProgress, Toast, TextWidgetHandler,
)
from backend.ffmpeg_profiles import (
    FFMPEG_PROFILE_SCHEMA,
    collect_ffmpeg_capability_profiles,
    missing_profile_requirements_for_config,
    summarize_missing_profile_requirements,
)
from backend.model_downloads import installed_backend_status
from backend.i18n import bind_locale, tr
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
from gui.layout_helpers import LayoutHelpersMixin
from gui.layout_responsive import ResponsiveLayoutMixin
from gui.layout_build import LayoutBuildMixin
from gui.queue_view import QueueViewMixin
from gui.onboarding import OnboardingMixin

logger = logging.getLogger(__name__)


class VideoSubtitleRemoverApp(
    RegionEditorControllerMixin,
    AdvancedSettingsControllerMixin,
    MaskCorrectionControllerMixin,
    PreviewControllerMixin,
    SupportControllerMixin,
    QualityReviewControllerMixin,
    ProcessingControllerMixin,
    LayoutHelpersMixin,
    ResponsiveLayoutMixin,
    LayoutBuildMixin,
    QueueViewMixin,
    OnboardingMixin,
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
                        header_icon = icon_img.copy()
                        header_icon.thumbnail((24, 24), Image.LANCZOS)
                        self._header_icon_photo = ImageTk.PhotoImage(header_icon)
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
        self._brand_photo = getattr(self, "_app_icon_photo", None)
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
            logger.debug("panel visibility restore failed", exc_info=True)

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

        gpu_label = self.gpus[0]["name"] if self.gpus else tr("CPU mode")
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
            self.gpu_var.set(tr("CPU mode"))
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
            self.gpu_combo.configure(values=[tr("CPU mode")], state="disabled")
            self.gpu_var.set(tr("CPU mode"))

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
        if hasattr(self, "queue_remove_btn"):
            self._refresh_action_states()

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
            for button_name in (
                "start_btn", "inspector_start_btn", "command_start_btn",
            ):
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
        if hasattr(self, "queue_add_btn"):
            self.queue_add_btn.set_enabled(
                not batch_busy,
                reason=tr("Wait for the current batch to finish."),
            )
        selected = self._get_selected_queue_item()
        selected_index = next(
            (index for index, item in enumerate(self.queue)
             if selected is not None and item.id == selected.id),
            -1,
        )
        if hasattr(self, "queue_remove_btn"):
            self.queue_remove_btn.set_enabled(
                (not batch_busy) and selected is not None,
                reason=(
                    tr("Wait for the current batch to finish.")
                    if batch_busy else tr("Select a queue item to remove.")
                ),
            )
        if hasattr(self, "queue_clear_completed_btn"):
            self.queue_clear_completed_btn.set_enabled(
                (not batch_busy) and has_complete,
                reason=(
                    tr("Wait for the current batch to finish.")
                    if batch_busy else tr("No completed items are in the queue.")
                ),
            )
        if hasattr(self, "queue_move_up_btn"):
            self.queue_move_up_btn.set_enabled(
                (not batch_busy) and selected_index > 0,
                reason=tr("Select an item below the first row to move it up."),
            )
        if hasattr(self, "queue_move_down_btn"):
            self.queue_move_down_btn.set_enabled(
                (not batch_busy)
                and selected_index >= 0
                and selected_index < len(self.queue) - 1,
                reason=tr("Select an item above the last row to move it down."),
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
        """Bind a small set of standard accelerators for hidden utilities."""
        self.root.bind(
            "<Control-o>",
            lambda _event: self.drop_area._open_file_dialog(),
            add="+",
        )
        self.root.bind(
            "<Control-l>",
            lambda _event: self._toggle_log_panel(),
            add="+",
        )
        self.root.bind(
            "<F1>",
            lambda _event: self._show_about(),
            add="+",
        )

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



    def _layout_command_strip(self, *, compact: bool):
        """Keep the primary workflow controls high and usable at every width."""
        blocks = getattr(self, "_command_blocks", ())
        if not blocks:
            return
        for block in blocks:
            block.grid_forget()
        for column in range(5):
            self._command_inner.columnconfigure(
                column, weight=0, uniform="", minsize=0)
        if compact:
            for column in range(3):
                self._command_inner.columnconfigure(
                    column, weight=1, uniform="command_compact")
            positions = (
                (0, 0, 1), (0, 1, 1), (1, 0, 1), (1, 1, 2), (0, 2, 1),
            )
        else:
            self._command_inner.columnconfigure(0, minsize=176)
            self._command_inner.columnconfigure(4, minsize=188)
            for column in (1, 2, 3):
                self._command_inner.columnconfigure(
                    column, weight=1, uniform="command_fields")
            positions = ((0, 0, 1), (0, 1, 1), (0, 2, 1), (0, 3, 1), (0, 4, 1))
        for block, (row, column, span) in zip(blocks, positions):
            block.grid(
                row=row, column=column, columnspan=span, sticky="ew",
                padx=(0, Theme.S_MD), pady=(0, Theme.S_SM) if compact else 0,
            )

    def _sync_command_region(self, *_args):
        """Reflect the current automatic/manual region mode in the command bar."""
        if not hasattr(self, "_command_region_var"):
            return
        label = tr("Manual region") if self.skip_detection_var.get() else tr("Automatic")
        self._command_region_var.set(label)

    def _on_command_region_changed(self, _event=None):
        """Route the compact region selector to the existing editor."""
        choice = self._command_region_var.get()
        if choice == tr("Set region..."):
            self._open_region_selector()
        self._sync_command_region()



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


















    def _refresh_inspector_summary(self, *_args):
        """Keep the quiet inspector overview aligned with live controls."""
        if not hasattr(self, "_inspector_profile_summary_var"):
            return
        profile_names = {
            "Auto": tr("Balanced"),
            "STTN": tr("Motion"),
            "LAMA": tr("Detail"),
            "ProPainter": tr("Temporal"),
        }
        mode = self.mode_var.get()
        self._inspector_profile_summary_var.set(
            profile_names.get(mode, mode))

    def _open_inspector_details(self, _section: str = ""):
        """Reveal the detailed editor from a compact inspector row."""
        if _section == "advanced":
            self._toggle_advanced()
        elif not self.adv_visible:
            self._toggle_advanced()

    def _sync_inspector_disclosure_state(self):
        """Keep the flat Advanced row aligned with detailed-control state."""
        chevron = getattr(self, "_inspector_advanced_chevron", None)
        if chevron is not None:
            chevron.configure(text="^" if self.adv_visible else "v")
        button = getattr(self, "_inspector_advanced_button", None)
        if button is not None:
            button.configure(
                fg=Theme.TEXT_PRIMARY if self.adv_visible else Theme.TEXT_SECONDARY)


    def _sync_inspector_encoding(self, *_args):
        if hasattr(self, "_inspector_encoding_var"):
            self.config.output_codec = self.output_codec_var.get()
            self._inspector_encoding_var.set(
                str(self.output_codec_var.get()).upper())




    def _open_preview_tools_menu(self, event=None):
        """Group secondary preview commands into one contextual menu."""
        menu = tk.Menu(
            self.root,
            tearoff=False,
            bg=Theme.BG_RAISED,
            fg=Theme.TEXT_PRIMARY,
            activebackground=Theme.BLUE_MUTED,
            activeforeground=Theme.TEXT_PRIMARY,
            bd=1,
            relief="solid",
        )
        entries = (
            (tr("Before / after"), self.preview_ab_btn, self._open_ab_scrubber),
            (tr("Set region"), self.preview_region_btn, self._open_region_selector),
            (tr("Review mask"), self.preview_mask_btn, self._open_selected_mask_preview),
            (tr("Test cleanup"), self.preview_inpaint_btn, self._open_selected_inpaint_preview),
            (tr("Full size"), self.preview_zoom_btn, self._open_preview_zoom),
            (tr("Correct mask"), self.preview_correction_btn,
             self._open_selected_mask_correction),
        )
        for label, button, command in entries:
            menu.add_command(
                label=label,
                command=command,
                state="normal" if button.enabled else "disabled",
            )
        menu.add_separator()
        menu.add_command(label=tr("Activity"), command=self._toggle_log_panel)
        try:
            x = getattr(event, "x_root", None)
            y = getattr(event, "y_root", None)
            if x is None or y is None:
                x = self._preview_label.winfo_rootx() + Theme.S_MD
                y = self._preview_label.winfo_rooty() + Theme.S_MD
            menu.tk_popup(
                x,
                y,
            )
        finally:
            menu.grab_release()



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
            "strip": "Remove embedded subtitles (fast)",
            "keep_all": "Keep embedded subtitles (fast, no re-encode)",
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



    @staticmethod
    def _safe_size(path: str) -> int:
        try:
            return Path(path).stat().st_size
        except OSError:
            return 0


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
            f"Stopping {Path(item.file_path).name}", "warning", toast=True,
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

    def _remove_selected_queue_item(self):
        """Remove the selected inactive queue item."""
        item = self._get_selected_queue_item()
        if item is None:
            self._update_status("Select a queue item to remove", "warning")
            return
        self._remove_from_queue(item.id)

    def _clear_completed_queue_items(self):
        """Remove completed queue records while preserving output files."""
        if self.is_processing or self._has_active_processing_thread():
            self._update_status(
                "Wait for the active batch to finish before clearing completed items",
                "warning",
            )
            return
        with self.queue_lock:
            completed_ids = {
                item.id for item in self.queue
                if item.status == ProcessingStatus.COMPLETE
            }
            self.queue = [
                item for item in self.queue if item.id not in completed_ids
            ]
        if not completed_ids:
            self._update_status("No completed items to clear")
            return
        if self._selected_queue_item_id in completed_ids:
            self._selected_queue_item_id = None
        self._update_queue_display()
        if self.queue:
            save_queue_state(self.queue)
        else:
            clear_queue_state()
        count = len(completed_ids)
        self._update_status(
            f"Cleared {count} completed item{'s' if count != 1 else ''}")

    def _move_selected_queue_item(self, direction: int):
        """Move the selected inactive queue item one position."""
        if self.is_processing or self._has_active_processing_thread():
            self._update_status(
                "Wait for the active batch to finish before reordering the queue",
                "warning",
            )
            return
        item = self._get_selected_queue_item()
        if item is None:
            self._update_status("Select a queue item to move", "warning")
            return
        step = -1 if direction < 0 else 1
        with self.queue_lock:
            index = next(
                (i for i, queued in enumerate(self.queue) if queued.id == item.id),
                -1,
            )
            target = index + step
            if index < 0 or target < 0 or target >= len(self.queue):
                return
            self.queue[index], self.queue[target] = (
                self.queue[target], self.queue[index])
        self._update_queue_display()
        save_queue_state(self.queue)
        self._update_status(
            f"Moved {Path(item.file_path).name} "
            f"{'up' if step < 0 else 'down'}")

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
            if hasattr(self, "_command_mode_combo"):
                self._command_mode_combo.config(state=combo_state)
            if hasattr(self, "_command_region_combo"):
                self._command_region_combo.config(state=combo_state)
            if hasattr(self, "_command_output_btn"):
                self._command_output_btn.set_enabled(not locked)
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
                logger.debug("segmented picker lock failed", exc_info=True)
        except Exception:
            logger.debug("settings lock/unlock failed", exc_info=True)

        # Re-apply mode-specific toggle availability
        if not locked:
            try:
                self._update_mode_options()
            except Exception:
                logger.debug("mode options update failed", exc_info=True)

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
