from __future__ import annotations

import ctypes
import json
import logging
import os
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import List, Optional

try:
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox
except ImportError:  # pragma: no cover - tkinter is optional in headless imports
    pass

try:
    from PIL import Image, ImageTk, ImageDraw, ImageFilter
    PIL_AVAILABLE = True
except ImportError:  # pragma: no cover - preview features degrade without Pillow
    PIL_AVAILABLE = False

from gui.theme import Theme, f, mono
from gui.config import (
    APP_NAME, APP_VERSION, LOG_DIR, LOG_FILE, SETTINGS_FILE,
    InpaintMode, ProcessingConfig, ProcessingStatus, QueueItem,
    clear_queue_state, save_queue_state, status_ui,
)
from gui.utils import (
    _format_soft_subtitle_summary, format_quality_report, format_size,
    format_time, get_app_dir, get_file_info, is_image_file, is_video_file,
    summarize_quality_reports, truncate_middle,
)
from gui.widgets import (
    ModernButton, ModernProgressBar, TaskbarProgress, Tooltip,
    make_themed_menu, show_confirm,
)
from backend.ffmpeg_profiles import ffmpeg_profile_entries
from backend.i18n import tr
from backend.model_downloads import installed_backend_status
from backend.resume_checkpoint import ProcessingPaused
from backend.safe_image import safe_imread

logger = logging.getLogger(__name__)


class ProcessingControllerMixin:
    """Focused controller methods mixed into VideoSubtitleRemoverApp."""

    def _start_processing(self):
        """Start processing the queue."""
        if not self.queue:
            self._update_status("Add media to the queue before starting a batch", "warning")
            return

        active_thread = self._has_active_processing_thread()
        batch_busy = self.is_processing or active_thread
        if batch_busy:
            if self._pause_requested or self.pause_event.is_set():
                self._update_status(
                    "Batch is already pausing. Please wait for the checkpoint to finish.",
                    "warning",
                )
                return
            if self._stop_requested or self.cancel_event.is_set():
                self._update_status(
                    "Batch is already stopping. Please wait for the current item to wrap up.",
                    "warning",
                )
                return
            if active_thread:
                self._pause_processing()
            else:
                self._update_status("Finalizing the previous batch...", "info")
            return

        self._apply_current_settings_to_idle_items()
        self._preflight_free_space_check()
        if self.preserve_audio_var.get() and not self.ffmpeg_ready:
            has_video = any(is_video_file(item.file_path) for item in self.queue)
            if has_video:
                self._update_status(
                    "FFmpeg is missing, so video outputs will be saved without original audio.",
                    "warning",
                    toast=True,
                )
        if not self._confirm_ffmpeg_profile_coverage():
            return

        self.is_processing = True
        self._stop_requested = False
        self._pause_requested = False
        self.cancel_event.clear()
        self.pause_event.clear()
        self._set_settings_locked(True)
        self.start_btn.set_style("secondary")
        self.start_btn.icon = "pause"
        self.start_btn.set_text(tr("Pause batch"))
        self._batch_times = []
        # F-9: the ETA probe loads an OCR model and detects 30 frames --
        # far too slow for the Tk main thread. _process_queue runs it on
        # the worker thread before the first item; until then the ETA
        # line is simply empty.
        self._probe_eta_seconds = 0.0
        self._batch_started_at = datetime.now()
        self._prepare_batch_report_records()
        self._warn_output_quality_preflight()
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

    def _pause_processing(self):
        """Pause the current processing at the next checkpoint boundary."""
        if self._pause_requested:
            self._update_status("Batch is already pausing...", "warning")
            return
        self._pause_requested = True
        self.pause_event.set()
        self.start_btn.set_style("primary")
        self.start_btn.icon = "pause"
        self.start_btn.set_text(tr("Pausing..."))
        self._refresh_action_states()
        self._update_status(
            "Pausing at the next safe frame checkpoint. Current progress will resume later.",
            "warning",
        )
        if self._taskbar:
            self._taskbar.set_state(TaskbarProgress.STATE_PAUSED)

    def _stop_processing(self):
        """Stop the current processing."""
        if self._stop_requested:
            self._update_status("Batch is already stopping...", "warning")
            return
        self._stop_requested = True
        self.cancel_event.set()
        self._terminate_active_backend_work()
        # Invalidate the cached remover so the next batch re-initialises with
        # fresh state. A cancelled run may have left detector / inpainter /
        # SRT buffers in an intermediate state.
        self._cached_remover = None
        self._cached_remover_key = None

        self.start_btn.set_style("primary")
        self.start_btn.icon = "x"
        self.start_btn.set_text(tr("Stopping..."))
        self._refresh_action_states()
        self._update_status(
            "Stopping after the current step. Finished outputs stay on disk.",
            "warning",
        )
        if self._taskbar:
            self._taskbar.set_state(TaskbarProgress.STATE_PAUSED)

    def _has_active_processing_thread(self) -> bool:
        return self._processing_thread is not None and self._processing_thread.is_alive()

    def _join_processing_thread(self, timeout: float) -> None:
        thread = self._processing_thread
        if thread is None or thread is threading.current_thread():
            return
        if not thread.is_alive():
            return
        try:
            thread.join(timeout=timeout)
        except RuntimeError:
            pass

    def _set_active_subprocess(self, proc: Optional[subprocess.Popen]) -> None:
        self._active_subprocess = proc

    @staticmethod
    def _terminate_subprocess_handle(proc: subprocess.Popen, timeout: float) -> None:
        try:
            poll = getattr(proc, "poll", None)
            if callable(poll) and poll() is not None:
                return
        except Exception:
            pass
        try:
            proc.terminate()
        except Exception:
            pass
        try:
            proc.wait(timeout=timeout)
            return
        except subprocess.TimeoutExpired:
            try:
                proc.kill()
            except Exception:
                pass
            try:
                proc.wait(timeout=timeout)
            except Exception:
                pass
        except Exception:
            pass

    def _terminate_active_backend_work(self) -> None:
        remover = self._active_remover or self._cached_remover
        if remover is not None and hasattr(remover, "terminate_active_work"):
            try:
                remover.terminate_active_work(timeout=2.0)
            except Exception:
                logger.warning("Active backend termination failed", exc_info=True)
        proc = self._active_subprocess
        if proc is not None:
            try:
                self._terminate_subprocess_handle(proc, timeout=2.0)
            except Exception:
                logger.warning("Active remux process termination failed", exc_info=True)
            finally:
                if self._active_subprocess is proc:
                    self._active_subprocess = None

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
                        "output_quality": getattr(item.config, "output_quality", 23),
                    },
                    soft_action=soft_action,
                )
                retry_config = getattr(item, "retry_config", None)
                if isinstance(retry_config, dict):
                    records[item.id]["retry_config"] = retry_config
            except Exception as exc:
                logger.warning(
                    f"Batch preflight report failed for {Path(item.file_path).name}: {exc}"
                )
        self._batch_report_records = records

    def _announce_model_download_guidance(self, item: QueueItem):
        """Surface first-run model download guidance before lazy loaders run."""
        try:
            from backend.model_downloads import (
                pending_model_download_hints,
                summarize_hints,
            )
            hints = pending_model_download_hints(item.config)
        except Exception as exc:
            logger.debug(f"Model download guidance probe failed: {exc}")
            return
        if not hints:
            return
        key = tuple((h.label, h.size_estimate) for h in hints)
        seen = getattr(self, "_model_download_guidance_seen", set())
        if key in seen:
            return
        seen.add(key)
        self._model_download_guidance_seen = seen
        summary = summarize_hints(hints)
        status = f"First use may download model files: {summary}"
        detail = (
            "Preparing model downloads if caches are empty. "
            "Keep this window open; failures will appear in the log."
        )
        logger.info("%s. %s", status, detail)
        item.message = "Preparing model downloads if needed..."
        item.progress = max(float(getattr(item, "progress", 0.0) or 0.0), 0.02)
        self._update_item_display(item)

        def _show():
            self._update_status(status, "info", toast=True)

        try:
            self.root.after(0, _show)
        except RuntimeError:
            pass

    def _process_queue(self):
        """Process all items in the queue."""
        with self.queue_lock:
            items_to_process = [i for i in self.queue
                                if i.status not in (ProcessingStatus.COMPLETE,
                                                     ProcessingStatus.ERROR,
                                                     ProcessingStatus.CANCELLED)]
        if items_to_process:
            self._announce_model_download_guidance(items_to_process[0])
        # F-9: pre-batch ETA probe runs here, on the worker thread, so
        # model load + 30-frame detection never block the Tk main loop.
        try:
            self._probe_eta_seconds = self._probe_batch_eta()
        except Exception:
            self._probe_eta_seconds = 0.0

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
            except (RuntimeError, tk.TclError):
                return  # root destroyed during shutdown
            self._process_item(item)
            if self.pause_event.is_set():
                break

        # Final batch state
        try:
            self.root.after(0, self._update_batch_progress, total, total)
            self.root.after(0, self._on_processing_complete)
        except (RuntimeError, tk.TclError):
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
        remux_soft_subtitles(
            item.file_path,
            item.output_path,
            action=action,
            on_process=self._set_active_subprocess,
            cancel_check=self.cancel_event.is_set,
        )

        item.status = ProcessingStatus.COMPLETE
        item.progress = 1.0
        item.error = None
        item.quality_report = None
        item.completed_at = datetime.now()
        elapsed = (item.completed_at - item.started_at).total_seconds()
        item.stage_timings = {"mux": elapsed}
        item.message = (
            "Embedded subtitles stripped"
            if action == SoftSubtitleAction.STRIP else
            "Embedded subtitles remuxed"
        )
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
            if not hasattr(self, "pause_event"):
                self.pause_event = threading.Event()
            self._update_item_display(item)

            if self._process_soft_subtitle_item(item):
                return

            self._announce_model_download_guidance(item)

            from backend.processor import (
                SubtitleRemover as BackendRemover,
                ProcessingConfig as BackendConfig,
                _checkpoint_key,
                _default_checkpoint_dir,
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
                matanyone_refine=getattr(item.config, 'matanyone_refine', False),
                cotracker_propagate=getattr(item.config, 'cotracker_propagate', False),
                rife_fast_stride=getattr(item.config, 'rife_fast_stride', 0),
                edge_ring_px=getattr(item.config, 'edge_ring_px', 2),
                subtitle_areas=getattr(item.config, 'subtitle_areas', None),
                subtitle_region_spans=getattr(
                    item.config, 'subtitle_region_spans', None),
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
            if (getattr(item.config, 'auto_band', False)
                    and not item.config.subtitle_area
                    and not getattr(item.config, 'subtitle_areas', None)
                    and not getattr(item.config, 'subtitle_region_spans', None)):
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
            cached = self._cached_remover
            if cached is not None and self._cached_remover_key == cache_key:
                remover = cached
                from backend.processor import normalize_processing_config as _normalize_backend_config
                remover.config = _normalize_backend_config(backend_config)
            else:
                remover = BackendRemover(backend_config)
                self._cached_remover = remover
                self._cached_remover_key = cache_key
            self._active_remover = remover
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
                    logger.warning("Live preview callback failed", exc_info=True)

            remover.on_preview_frame = on_preview_frame

            # Ensure output directory exists
            Path(item.output_path).parent.mkdir(parents=True, exist_ok=True)

            # Run the actual processing
            file_name = Path(item.file_path).name
            logger.info(f"Processing: {file_name} with {item.config.mode.value}")

            if is_video_file(item.file_path):
                ckpt_dir = _default_checkpoint_dir()
                ckpt_key = _checkpoint_key(item.file_path, item.output_path)
                success = remover.process_video(
                    item.file_path,
                    item.output_path,
                    checkpoint_dir=ckpt_dir,
                    checkpoint_key=ckpt_key,
                    resume_checkpoint=True,
                    pause_check=self.pause_event.is_set,
                )
            elif is_image_file(item.file_path):
                success = remover.process_image(item.file_path, item.output_path)
            else:
                raise ValueError(f"Unsupported file type: {Path(item.file_path).suffix}")

            resume_warning = getattr(remover, "last_resume_warning", None)
            if resume_warning:
                self._update_status(str(resume_warning), "warning", toast=True)

            if success:
                item.stage_timings = dict(
                    getattr(remover, "last_stage_timings", {}) or {}
                )
                actual_output_path = getattr(remover, "last_output_path", None)
                if (
                    actual_output_path
                    and self._normalized_path_key(actual_output_path)
                    != self._normalized_path_key(item.output_path)
                ):
                    logger.warning(
                        "Output path changed after fallback encode: %s -> %s",
                        item.output_path,
                        actual_output_path,
                    )
                    item.output_path = str(actual_output_path)
                    item.output_path_locked = True
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
                item.stage_timings = dict(
                    getattr(remover, "last_stage_timings", {}) or {}
                )
                failure_message = (
                    getattr(remover, "last_error_message", None)
                    or "Processing failed"
                )
                item.status = ProcessingStatus.ERROR
                item.message = failure_message
                item.error = failure_message
                item.quality_report = None
                item.completed_at = datetime.now()
                logger.error(f"Failed: {file_name}: {failure_message}")
            self._update_item_display(item)

        except ProcessingPaused as exc:
            remover_obj = locals().get("remover")
            item.stage_timings = dict(
                getattr(remover_obj, "last_stage_timings", {}) or {}
            )
            checkpoint_payload = (
                getattr(remover_obj, "last_pause_checkpoint", None)
                if remover_obj is not None else None
            )
            if isinstance(checkpoint_payload, dict):
                next_frame = float(checkpoint_payload.get("next_frame") or 0.0)
                total_frames = float(checkpoint_payload.get("total_frames") or 0.0)
                if total_frames > 0:
                    item.progress = max(0.0, min(0.99, next_frame / total_frames))
            item.pause_checkpoint_path = (
                getattr(remover_obj, "last_pause_checkpoint_path", "") or ""
                if remover_obj is not None else ""
            )
            item.status = ProcessingStatus.PAUSED
            item.message = str(exc) or "Paused at checkpoint"
            item.error = None
            item.quality_report = None
            item.completed_at = datetime.now()
            self._update_item_display(item)
            logger.info(f"Paused: {Path(item.file_path).name}")
        except InterruptedError:
            remover_obj = locals().get("remover")
            item.stage_timings = dict(
                getattr(remover_obj, "last_stage_timings", {}) or {}
            )
            item.status = ProcessingStatus.CANCELLED
            item.message = "Cancelled"
            item.error = None
            item.quality_report = None
            item.completed_at = datetime.now()
            self._update_item_display(item)
            logger.info(f"Cancelled: {Path(item.file_path).name}")
        except Exception as e:
            remover_obj = locals().get("remover")
            item.stage_timings = dict(
                getattr(remover_obj, "last_stage_timings", {}) or {}
            )
            item.status = ProcessingStatus.ERROR
            item.error = str(e)
            item.message = f"Error: {str(e)}"
            item.quality_report = None
            item.completed_at = datetime.now()
            self._update_item_display(item)
            logger.error(f"Processing error for {item.file_path}: {e}", exc_info=True)
        finally:
            if self._active_remover is locals().get("remover"):
                self._active_remover = None

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
                lang = first_video.config.detection_lang or "en"
                with self._detector_lock:
                    detector = self._preview_detector
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

    def _on_processing_complete(self):
        """Handle processing completion."""
        self.is_processing = False
        self._stop_requested = False
        self._pause_requested = False
        self._processing_thread = None
        self.cancel_event.clear()
        self.pause_event.clear()
        self._stop_elapsed_timer()
        self._set_settings_locked(False)
        # Clear cached remover so next batch picks up any setting changes
        self._cached_remover = None
        self._cached_remover_key = None
        report_paths = self._write_batch_report_files()
        save_queue_state(self.queue)
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
        self.start_btn.set_text(tr("Start batch"))
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
        paused = sum(1 for item in self.queue if item.status == ProcessingStatus.PAUSED)
        cancelled = sum(1 for item in self.queue if item.status == ProcessingStatus.CANCELLED)
        review_count = len(self._review_needed_records())

        summary = f"Batch finished: {complete} completed, {errors} failed"
        if review_count:
            summary += f", {review_count} needs review"
        if paused:
            summary += f", {paused} paused"
        if cancelled:
            summary += f", {cancelled} stopped"
        is_clean = errors == 0 and paused == 0 and cancelled == 0 and review_count == 0
        quality_summary = summarize_quality_reports(
            [item.quality_report for item in self.queue if item.status == ProcessingStatus.COMPLETE]
        )
        stage_summary = {}
        try:
            from backend.batch_report import summarize_stage_timings
            stage_summary = summarize_stage_timings(
                getattr(self, "_last_batch_report_records", []) or []
            )
        except Exception:
            logger.warning("Could not summarize batch stage timings", exc_info=True)
        slow_text = self._dominant_stage_text(
            stage_summary.get("slowest_stage")
            if isinstance(stage_summary, dict) else None
        )
        if quality_summary:
            summary += (
                f" | avg PSNR {quality_summary['psnr']:.2f} dB"
                f", avg SSIM {quality_summary['ssim']:.4f}"
            )
        if slow_text:
            summary += f" | slowest {slow_text}"
        self._update_status(summary, "success" if is_clean else "warning")
        logger.info(summary)
        if report_paths:
            logger.info(
                "Batch reports: "
                + ", ".join(str(path) for path in report_paths)
            )
        self._notify_completion(complete, errors, paused=paused)
        # Surface a themed summary modal for meaningful batches
        total = complete + errors + paused + cancelled
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
                paused=paused,
                quality_summary=quality_summary,
                review_count=review_count,
                stage_summary=stage_summary,
            )

    def _notify_completion(self, complete: int, errors: int, *,
                           paused: int = 0):
        """Flash taskbar + play sound when batch processing finishes."""
        # RM-95: screen-reader announcement so NVDA / Narrator users
        # learn the batch finished without polling the activity log.
        try:
            from backend.a11y import announce
            if paused:
                announce(
                    f"Batch paused. {paused} item remains paused. "
                    f"{complete} items processed.",
                    importance="high",
                )
            elif errors == 0:
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
        # System toast notification (visible even when minimised)
        if self.config.notify_on_completion:
            self._send_system_notification(complete, errors, paused=paused)

    def _send_system_notification(self, complete: int, errors: int,
                                  *, paused: int = 0):
        """Send a Windows toast notification summarising the batch result."""
        if paused:
            title = "Batch Paused"
            msg = (
                f"{paused} item{'s' if paused != 1 else ''} paused. "
                "Start again to resume."
            )
        elif errors == 0:
            title = "Batch Complete"
            msg = f"{complete} item{'s' if complete != 1 else ''} processed successfully."
        else:
            title = "Batch Finished with Errors"
            msg = (f"{complete} processed, {errors} failed.")
        try:
            from plyer import notification
            notification.notify(
                title=title,
                message=msg,
                app_name="Video Subtitle Remover Pro",
                timeout=10,
            )
            return
        except Exception:
            pass
        try:
            import ctypes
            ctypes.windll.user32.MessageBeep(0)
        except Exception:
            pass

