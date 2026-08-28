from __future__ import annotations

import logging
import subprocess
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Protocol

from gui.theme import Theme
from gui.config import (
    APP_NAME, APP_VERSION, ProcessingStatus, QueueItem,
    save_queue_state,
)
from gui.utils import (
    dispatch_to_ui,
    format_quality_report, format_time, is_image_file, is_video_file,
    summarize_quality_reports,
)
from gui.widgets import (
    TaskbarProgress,
)
from backend.failure_reason import (
    REASON_CANCELLED,
    REASON_NONE,
    REASON_PAUSED,
    classify_failure_reason,
)
from backend.i18n import N_, tr
from backend.job_worker import describe_exit_code
from backend.resume_checkpoint import ProcessingPaused
from gui.failure_copy import (
    MSG_CANCELLED,
    MSG_COMPLETE,
    MSG_FAILED,
    MSG_INITIALIZING,
    MSG_PAUSED,
    MSG_PREPARING_MODELS,
    MSG_STRIP_SOFT,
    MSG_COPY_SOFT,
    user_facing_isolated_error,
    user_facing_processing_error,
)

logger = logging.getLogger(__name__)


class ProcessingControllerHost(Protocol):
    """Queue and UI surface required by the processing controller."""

    root: Any
    queue: list[QueueItem]
    is_processing: bool

    def _update_status(
        self, message: str, tone: str = "neutral", toast: bool = False
    ) -> None:
        ...


class ProcessingControllerMixin:
    """Focused controller methods mixed into VideoSubtitleRemoverApp."""

    def _start_processing(self):
        """Start processing the queue."""
        if not self.queue:
            self._update_status(N_("Add media to the queue before starting a batch"), "warning")
            return

        active_thread = self._has_active_processing_thread()
        batch_busy = self.is_processing or active_thread
        if batch_busy:
            if self._pause_requested or self.pause_event.is_set():
                self._update_status(
                    N_("Batch is already pausing. Please wait for the checkpoint to finish."),
                    "warning",
                )
                return
            if self._stop_requested or self.cancel_event.is_set():
                self._update_status(
                    N_("Batch is already stopping. Please wait for the current item to wrap up."),
                    "warning",
                )
                return
            if active_thread:
                self._pause_processing()
            else:
                self._update_status(N_("Finalizing the previous batch..."), "info")
            return

        self._apply_current_settings_to_idle_items()
        self._preflight_free_space_check()
        if self.preserve_audio_var.get() and not self.ffmpeg_ready:
            has_video = any(is_video_file(item.file_path) for item in self.queue)
            if has_video:
                self._update_status(
                    N_("FFmpeg is missing, so video outputs will be saved without original audio."),
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
        try:
            self._prepare_batch_report_records()
            self._warn_output_quality_preflight()
            self._write_batch_preflight_plan()
            self._last_batch_report_paths = []
            self._refresh_action_states()
            self._update_status(N_("Batch processing started"), "info")
            # Kick off Windows taskbar progress in indeterminate until first tick
            self._ensure_taskbar()
            if self._taskbar:
                self._taskbar.set_state(TaskbarProgress.STATE_INDETERMINATE)

            # Start elapsed timer
            self._start_elapsed_timer()

            # Start processing thread
            self._processing_thread = threading.Thread(
                target=self._process_queue, daemon=True)
            self._processing_thread.start()
        except Exception as exc:
            # The startup preflight runs after the UI has been locked. Roll
            # every bit of that state back when a removable output volume or
            # another preflight dependency fails before the worker starts.
            logger.exception("Batch startup failed")
            self.is_processing = False
            self._stop_requested = False
            self._pause_requested = False
            self.cancel_event.clear()
            self.pause_event.clear()
            self._processing_thread = None
            try:
                self._stop_elapsed_timer()
            except Exception:
                pass
            self._set_settings_locked(False)
            self.start_btn.set_style("primary")
            self.start_btn.icon = ">"
            self.start_btn.set_text(tr("Start batch"))
            self._refresh_action_states()
            self._update_status(
                tr("Could not start batch: {error}").format(error=exc),
                "error", toast=True)

    def _pause_processing(self):
        """Pause the current processing at the next checkpoint boundary."""
        if self._pause_requested:
            self._update_status(N_("Batch is already pausing..."), "warning")
            return
        self._pause_requested = True
        self.pause_event.set()
        self.start_btn.set_style("primary")
        self.start_btn.icon = "pause"
        self.start_btn.set_text(tr("Pausing..."))
        self._refresh_action_states()
        self._update_status(
            N_("Pausing at the next safe frame checkpoint. Current progress will resume later."),
            "warning",
        )
        if self._taskbar:
            self._taskbar.set_state(TaskbarProgress.STATE_PAUSED)

    def _stop_processing(self):
        """Stop the current processing."""
        if self._stop_requested:
            self._update_status(N_("Batch is already stopping..."), "warning")
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
            N_("Stopping after the current step. Finished outputs stay on disk."),
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
        # RM-155: an isolated job's work lives in a child process, so app
        # shutdown has to stop that too or the child keeps encoding after
        # the window is gone.
        supervisor = getattr(self, "_active_supervisor", None)
        if supervisor is not None:
            try:
                supervisor.cancel()
                supervisor.terminate()
            except Exception:
                logger.warning(
                    "Isolated job termination failed", exc_info=True)
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
                    item = widget.item
                    if item.started_at and not item.completed_at:
                        elapsed = (
                            datetime.now() - item.started_at).total_seconds()
                        # The row composes this label as "42% / 1m 3s".
                        # Writing bare elapsed time here dropped the percent
                        # on every tick, so it flickered off and back between
                        # progress events.
                        meta = format_time(elapsed)
                        if item.progress > 0:
                            meta = f"{int(item.progress * 100)}% / {meta}"
                        widget.time_label.config(text=meta)
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
        item.message = MSG_PREPARING_MODELS
        item.progress = max(float(getattr(item, "progress", 0.0) or 0.0), 0.02)
        self._update_item_display(item)

        def _show():
            self._update_status(status, "info", toast=True)

        dispatch_to_ui(self.root, _show)

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
                    remaining.failure_reason = REASON_CANCELLED
                    remaining.completed_at = now
                    self._update_item_display(remaining)
                break

            # Update batch progress + window title
            if dispatch_to_ui(
                self.root, self._update_batch_progress, idx, total
            ) is None:
                return  # root destroyed during shutdown
            self._process_item(item)
            if self.pause_event.is_set():
                break

        # Final batch state
        save_queue_state(self.queue)
        dispatch_to_ui(self.root, self._update_batch_progress, total, total)
        dispatch_to_ui(self.root, self._on_processing_complete)

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
            MSG_STRIP_SOFT
            if action == SoftSubtitleAction.STRIP else
            MSG_COPY_SOFT
        )
        self._update_item_display(item)

        Path(item.output_path).parent.mkdir(parents=True, exist_ok=True)
        remux_soft_subtitles(
            item.file_path,
            item.output_path,
            action=action,
            on_process=self._set_active_subprocess,
            cancel_check=lambda: (self.cancel_event.is_set() or item.cancel_requested),
        )

        item.status = ProcessingStatus.COMPLETE
        item.progress = 1.0
        item.error = None
        item.failure_reason = REASON_NONE
        item.quality_report = None
        item.completed_at = datetime.now()
        elapsed = (item.completed_at - item.started_at).total_seconds()
        item.stage_timings = {"mux": elapsed}
        item.message = (
            "Embedded subtitles removed"
            if action == SoftSubtitleAction.STRIP else
            "Embedded subtitles copied"
        )
        self._batch_times.append(elapsed)
        logger.info(
            f"Soft-subtitle {action.value}: {Path(item.file_path).name} "
            f"in {format_time(elapsed)}"
        )
        self._update_item_display(item)
        return True

    # RM-155: map the child's terminal status onto the queue model. The
    # child reports its own outcome; only "crashed" is inferred by the
    # parent, from a worker that stopped without publishing a result.
    _ISOLATED_STATUS = {
        "complete": ProcessingStatus.COMPLETE,
        "paused": ProcessingStatus.PAUSED,
        "cancelled": ProcessingStatus.CANCELLED,
        "error": ProcessingStatus.ERROR,
        "crashed": ProcessingStatus.ERROR,
    }

    def _apply_isolated_evidence(self, item: QueueItem, evidence: dict) -> None:
        """Copy a child job's reported evidence onto the queue item."""
        def as_dict(name):
            value = evidence.get(name)
            return dict(value) if isinstance(value, dict) else {}

        item.mask_export = as_dict("last_mask_export")
        item.mask_import = as_dict("last_mask_import")
        item.timing_report = as_dict("last_timing_report")
        item.output_contract_report = as_dict("last_output_contract")
        item.selective_rerun = as_dict("last_selective_rerun")
        item.stage_timings = as_dict("last_stage_timings")
        item.detection_stats = as_dict("last_detection_stats")
        item.execution_provenance = as_dict("execution_provenance")
        quality = evidence.get("last_quality_report")
        item.quality_report = quality if isinstance(quality, dict) else None
        checkpoint_path = evidence.get("last_pause_checkpoint_path")
        if checkpoint_path:
            item.pause_checkpoint_path = str(checkpoint_path)
        actual_output = str(evidence.get("last_output_path") or "")
        if (
            actual_output
            and self._normalized_path_key(actual_output)
            != self._normalized_path_key(item.output_path)
        ):
            logger.warning(
                "Isolated job changed its output path: %s -> %s",
                item.output_path, actual_output,
            )
            item.output_path = actual_output
            item.output_path_locked = True

    def _process_item_isolated(self, item: QueueItem) -> None:
        """RM-155: run one item in a supervised child process.

        A fatal native fault inside OpenCV, ONNX Runtime, or a model's own
        kernels cannot be caught by Python. In-process it takes the GUI and
        every remaining queued job with it. Here it takes only the child,
        and the supervisor reports that as this item's failure.
        """
        from backend.config_schema import (
            gui_to_backend_config,
            serialize_dataclass_config,
        )
        from gui.job_supervisor import JobSupervisor, build_request

        backend_config = gui_to_backend_config(item.config)
        # Selective rerun (mask-correction retry) and auto-band both have
        # to cross the process boundary explicitly, or an isolated job
        # silently downgrades to a full re-detect of the whole file.
        correction_retry = (
            item.correction_retry
            if isinstance(getattr(item, "correction_retry", None), dict)
            else {}
        )
        request = build_request(
            input_path=item.file_path,
            output_path=item.output_path,
            config_payload=serialize_dataclass_config(backend_config),
            is_image=is_image_file(item.file_path),
            preview_dir="",
            resume_checkpoint=True,
            selective_rerun_from=str(
                correction_retry.get("source_output") or ""),
            selective_rerun_ranges=correction_retry.get("ranges") or None,
            auto_band=bool(getattr(item.config, "auto_band", False)),
        )

        def on_progress(progress: float, message: str) -> None:
            if progress < 0.3:
                item.status = ProcessingStatus.DETECTING
            elif progress < 0.9:
                item.status = ProcessingStatus.PROCESSING
            elif progress < 1.0:
                item.status = ProcessingStatus.MERGING
            item.progress = float(progress)
            item.message = str(message)
            self._update_item_display(item)

        def on_warning(message: str) -> None:
            dispatch_to_ui(
                self.root, self._update_status,
                message, "warning", True,
            )

        # Live preview parity with the in-process path: the child writes a
        # throttled PNG, this side loads it, downsizes, and marshals the
        # PIL image onto the Tk main loop exactly like on_preview_frame.
        preview_throttle_state = {"last_ts": 0.0}

        def on_preview(path: str, cur_idx: int, total: int) -> None:
            if self.cancel_event.is_set() or item.cancel_requested:
                return
            now = time.monotonic()
            if (now - preview_throttle_state["last_ts"]) < (1.0 / 15.0):
                return
            preview_throttle_state["last_ts"] = now
            try:
                import cv2 as _cv2_live

                from backend.safe_image import safe_imread

                # RM-317: the worker stages this frame under the scratch
                # directory, which lives in %TEMP%. cv2.imread returns None
                # for any path holding non-ASCII characters, and the guard
                # below would swallow that as "no frame yet", so the live
                # preview would stay blank for the whole run.
                frame = safe_imread(path, _cv2_live.IMREAD_COLOR)
                if frame is None:
                    return
                max_w, max_h = 520, 320
                h, w = frame.shape[:2]
                scale = min(max_w / max(1, w), max_h / max(1, h), 1.0)
                if scale < 1.0:
                    frame = _cv2_live.resize(
                        frame,
                        (max(1, int(w * scale)), max(1, int(h * scale))),
                        interpolation=_cv2_live.INTER_AREA)
                from PIL import Image as _Image

                pil = _Image.fromarray(frame[..., ::-1])
                dispatch_to_ui(
                    self.root, self._push_live_preview,
                    pil, cur_idx, total, Path(item.file_path).name,
                )
            except Exception:
                logger.warning(
                    "Isolated live preview failed", exc_info=True)

        max_retries = max(0, int(getattr(
            item.config, 'batch_max_retries', 0)))
        retry_backoff = max(0.0, float(getattr(
            item.config, 'batch_retry_backoff_seconds', 5.0)))
        attempt = 0
        watchdog_stops: list[threading.Event] = []
        while True:
            supervisor = JobSupervisor(
                request, on_progress=on_progress, on_warning=on_warning,
                on_preview=on_preview)
            self._active_supervisor = supervisor
            # Cancel and pause are polled by the child, so the existing GUI
            # events keep working: publish the state, then let it observe it.
            # One watchdog at a time. On a retry the item goes back to
            # LOADING rather than a terminal status, so the previous
            # watchdog never stood down: it kept polling a dead supervisor
            # at 10 Hz and, on a later cancel, published a control file
            # that recreated the already-cleaned scratch directory.
            watchdog_stop = threading.Event()
            if watchdog_stops:
                for previous in watchdog_stops:
                    previous.set()
                watchdog_stops.clear()
            watchdog_stops.append(watchdog_stop)
            watchdog = threading.Thread(
                target=self._watch_isolated_controls,
                args=(supervisor, item, watchdog_stop),
                name="vsr-job-controls",
                daemon=True,
            )
            watchdog.start()
            try:
                outcome = supervisor.run()
            finally:
                self._active_supervisor = None

            if outcome.status in ("complete", "paused", "cancelled"):
                break
            from backend.batch_report import is_retriable_error
            failure = RuntimeError(outcome.error or "Processing failed")
            item.retry_errors = list(item.retry_errors or []) + [str(failure)]
            if (attempt >= max_retries
                    or not is_retriable_error(failure)
                    or self.cancel_event.is_set()
                    or item.cancel_requested
                    or self.pause_event.is_set()):
                break
            attempt += 1
            item.retry_attempts = attempt
            record = getattr(self, "_batch_report_records", {}).get(item.id)
            if isinstance(record, dict):
                record["retry_attempts"] = attempt
                record["retry_errors"] = list(item.retry_errors)
            wait = round(retry_backoff * attempt, 2)
            item.status = ProcessingStatus.LOADING
            item.message = (
                f"Retrying after transient failure "
                f"({attempt}/{max_retries})...")
            self._update_item_display(item)
            logger.warning(
                "Transient isolated failure on %s (attempt %d/%d): %s; "
                "retrying in %.1fs",
                Path(item.file_path).name, attempt, max_retries,
                failure, wait)
            deadline = time.monotonic() + wait
            while time.monotonic() < deadline:
                if (self.cancel_event.is_set() or item.cancel_requested
                        or self.pause_event.is_set()):
                    break
                time.sleep(min(0.1, max(0.0, deadline - time.monotonic())))
            # A cancel or pause that arrived during the backoff must not
            # spawn another child. Report it as what the user asked for,
            # not as the transient error the retry was about to absorb.
            if self.cancel_event.is_set() or item.cancel_requested:
                from gui.job_supervisor import JobOutcome
                outcome = JobOutcome(
                    status="cancelled", error="Cancelled",
                    reason="cancelled")
                break
            if self.pause_event.is_set():
                from gui.job_supervisor import JobOutcome
                outcome = JobOutcome(
                    status="paused",
                    error="Paused before retry; resume to try again.",
                    reason="paused")
                break

        item.completed_at = datetime.now()
        self._apply_isolated_evidence(item, outcome.evidence)
        item.status = self._ISOLATED_STATUS.get(
            outcome.status, ProcessingStatus.ERROR)

        if outcome.status == "complete":
            item.progress = 1.0
            item.error = None
            item.failure_reason = REASON_NONE
            item.correction_retry = None
            item.message = MSG_COMPLETE
            note = format_quality_report(item.quality_report, compact=True)
            if note:
                item.message = f"{MSG_COMPLETE} - {note}"
            elapsed = (item.completed_at - item.started_at).total_seconds()
            self._batch_times.append(elapsed)
            logger.info(
                "Completed (isolated): %s in %s",
                Path(item.file_path).name, format_time(elapsed),
            )
        elif outcome.status == "paused":
            item.error = None
            item.failure_reason = REASON_PAUSED
            item.message = MSG_PAUSED
        elif outcome.status == "cancelled":
            item.error = None
            item.failure_reason = REASON_CANCELLED
            item.message = MSG_CANCELLED
        else:
            raw = outcome.error or MSG_FAILED
            item.error = user_facing_isolated_error(raw)
            item.message = item.error
            item.failure_reason = classify_failure_reason(
                reason=outcome.reason, message=raw)
            item.quality_report = None
            if outcome.crashed:
                # Retain the child's diagnostics: a native fault usually
                # prints the real cause (a CUDA error, a DLL name) right
                # before dying, and that tail is all that survives it.
                logger.error(
                    "Isolated job crashed for %s (%s). Worker stderr tail:\n%s",
                    item.file_path,
                    describe_exit_code(outcome.exit_code or 0),
                    outcome.stderr_tail or "(no output)",
                )
                if outcome.stderr_tail:
                    item.retry_errors = list(item.retry_errors or []) + [
                        outcome.stderr_tail]
            else:
                logger.error(
                    "Isolated job failed for %s: %s",
                    item.file_path, item.error,
                )
        self._update_item_display(item)

    # How long a cancelled isolated child gets to exit gracefully (write
    # its checkpoint, emit its result) before the watchdog force-kills
    # it. A child wedged in native code never reads the control file, so
    # without this escalation a per-item cancel could wait forever.
    _ISOLATED_CANCEL_GRACE_SECONDS = 30.0

    def _report_control_publish_failure(
            self, item: QueueItem, wanted_pause: bool) -> None:
        """Tell the user a pause/resume never reached the isolated worker."""
        message = (
            N_("Could not pause the running job; it is still processing.")
            if wanted_pause
            else N_("Could not resume the paused job.")
        )
        if dispatch_to_ui(
            self.root, self._update_status, message, "error", True
        ) is None:
            logger.debug("Could not surface a control failure")

    def _watch_isolated_controls(
            self, supervisor, item: QueueItem,
            stop: Optional[threading.Event] = None) -> None:
        """Forward GUI cancel/pause requests to a running child."""
        cancelled = False
        cancel_deadline = 0.0
        paused = False
        terminal = (
            ProcessingStatus.COMPLETE, ProcessingStatus.ERROR,
            ProcessingStatus.CANCELLED, ProcessingStatus.PAUSED,
        )

        def done() -> bool:
            # A retry leaves the item at LOADING, so the terminal status
            # alone is not enough to stand this thread down.
            return (stop is not None and stop.is_set()) or item.status in terminal

        while supervisor.pid is None:
            # A failed spawn never produces a pid; the item going terminal
            # is this thread's signal to stand down instead of spinning.
            if done():
                return
            if self.cancel_event.is_set() or getattr(
                    item, "cancel_requested", False):
                break
            time.sleep(0.05)
        while True:
            if done():
                return
            want_cancel = bool(
                self.cancel_event.is_set()
                or getattr(item, "cancel_requested", False))
            want_pause = bool(self.pause_event.is_set())
            if want_cancel and not cancelled:
                cancelled = True
                cancel_deadline = (
                    time.monotonic() + self._ISOLATED_CANCEL_GRACE_SECONDS)
                supervisor.cancel()
            elif cancelled and time.monotonic() > cancel_deadline:
                # The child had its grace period and is still running:
                # it is not going to honour the control file. Kill it;
                # the supervisor reports the outcome as cancelled.
                logger.warning(
                    "Isolated job ignored cancel for %.0fs; terminating",
                    self._ISOLATED_CANCEL_GRACE_SECONDS)
                supervisor.terminate()
                return
            elif want_pause != paused:
                paused = want_pause
                published = (
                    supervisor.pause() if want_pause else supervisor.resume())
                if not published:
                    # Cancel has a watchdog that escalates to terminate, but
                    # a pause that never reaches the child just leaves the UI
                    # claiming "paused" while the job keeps running. Say so.
                    logger.error(
                        "Could not publish the %s request to the job worker",
                        "pause" if want_pause else "resume")
                    self._report_control_publish_failure(item, want_pause)
                    paused = not want_pause
            time.sleep(0.1)

    def _process_item(self, item: QueueItem):
        """Process a single queue item using the backend processor."""
        try:
            item.status = ProcessingStatus.LOADING
            item.started_at = datetime.now()
            item.completed_at = None
            item.progress = 0.0
            item.message = MSG_INITIALIZING
            item.error = None
            item.failure_reason = REASON_NONE
            item.quality_report = None
            item.retry_attempts = 0
            item.retry_errors = []
            item.mask_export = {}
            item.mask_import = {}
            item.timing_report = {}
            item.output_contract_report = {}
            item.selective_rerun = {}
            item.cancel_requested = False  # F-7 reset on fresh attempt
            if not hasattr(self, "pause_event"):
                self.pause_event = threading.Event()
            self._update_item_display(item)

            if self._process_soft_subtitle_item(item):
                return

            self._announce_model_download_guidance(item)

            # RM-155: opt-in process isolation. Delegated before any backend
            # model is loaded in this process, so an isolated job shares
            # nothing with the GUI that a native fault could corrupt.
            if getattr(item.config, "job_isolation", False):
                self._process_item_isolated(item)
                return

            from backend.processor import SubtitleRemover as BackendRemover
            from backend.config import ProcessingConfig as BackendConfig
            from backend.resume_checkpoint import (
                _checkpoint_key,
                _default_checkpoint_dir,
            )
            from backend.config_schema import gui_to_backend_config

            backend_mode = self._gui_to_backend_mode(item.config.mode.value)
            device = self._gui_to_backend_device(
                item.config.use_gpu, item.config.gpu_id)
            lang = getattr(item.config, 'detection_lang', 'en')
            ocr_engine = getattr(item.config, 'detection_engine', 'auto')
            ocr_variant = getattr(item.config, 'rapidocr_variant', 'v6')
            vertical = bool(getattr(item.config, 'detection_vertical', False))
            cache_key = (
                backend_mode, device, lang, ocr_engine, ocr_variant, vertical)

            backend_config = gui_to_backend_config(item.config)

            # Auto subtitle-band detection -- run before the main pass so we
            # can pin the dominant band once per file. Cheap (30-frame probe).
            if (getattr(item.config, 'auto_band', False)
                    and not item.config.subtitle_area
                    and not getattr(item.config, 'subtitle_areas', None)
                    and not getattr(item.config, 'subtitle_region_spans', None)
                    and not getattr(
                        item.config, 'subtitle_region_keyframes', None)):
                try:
                    # Use a minimal config just for the band probe
                    probe_cfg = BackendConfig(
                        mode=backend_mode,
                        device=device,
                        detection_lang=lang,
                        detection_engine=ocr_engine,
                        rapidocr_variant=ocr_variant,
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
                from backend.config import normalize_processing_config as _normalize_backend_config
                remover.config = _normalize_backend_config(backend_config)
            else:
                remover = BackendRemover(backend_config)
                self._cached_remover = remover
                self._cached_remover_key = cache_key
            self._active_remover = remover
            work_warning = getattr(remover, "last_work_directory_warning", None)
            if work_warning:
                dispatch_to_ui(
                    self.root, self._update_status,
                    work_warning, "warning", True,
                )
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
                    dispatch_to_ui(
                        self.root, self._push_live_preview,
                        pil, cur_idx, total, Path(item.file_path).name,
                    )
                except Exception:
                    logger.warning("Live preview callback failed", exc_info=True)

            remover.on_preview_frame = on_preview_frame

            # Ensure output directory exists
            Path(item.output_path).parent.mkdir(parents=True, exist_ok=True)

            # Run the actual processing
            file_name = Path(item.file_path).name
            logger.info(f"Processing: {file_name} with {item.config.mode.value}")

            max_retries = max(0, int(getattr(
                item.config, 'batch_max_retries', 0)))
            retry_backoff = max(0.0, float(getattr(
                item.config, 'batch_retry_backoff_seconds', 5.0)))
            attempt = 0
            correction_retry = (
                item.correction_retry
                if isinstance(getattr(item, "correction_retry", None), dict)
                else {}
            )
            selective_rerun_from = str(
                correction_retry.get("source_output") or "") or None
            selective_rerun_ranges = correction_retry.get("ranges") or None
            while True:
                raised_error = False
                try:
                    if is_video_file(item.file_path):
                        ckpt_dir = _default_checkpoint_dir(
                            item.config.work_directory)
                        ckpt_key = _checkpoint_key(
                            item.file_path, item.output_path)
                        success = remover.process_video(
                            item.file_path,
                            item.output_path,
                            checkpoint_dir=ckpt_dir,
                            checkpoint_key=ckpt_key,
                            resume_checkpoint=True,
                            pause_check=self.pause_event.is_set,
                            selective_rerun_from=selective_rerun_from,
                            selective_rerun_ranges=selective_rerun_ranges,
                        )
                    elif is_image_file(item.file_path):
                        success = remover.process_image(
                            item.file_path, item.output_path)
                    else:
                        raise ValueError(
                            f"Unsupported file type: "
                            f"{Path(item.file_path).suffix}"
                        )
                except (ProcessingPaused, InterruptedError):
                    raise
                except Exception as exc:
                    failure = exc
                    raised_error = True
                    success = False
                else:
                    mask_export = getattr(remover, "last_mask_export", None)
                    if isinstance(mask_export, dict):
                        item.mask_export = dict(mask_export)
                    mask_import = getattr(remover, "last_mask_import", None)
                    if isinstance(mask_import, dict):
                        item.mask_import = dict(mask_import)
                    timing_report = getattr(
                        remover, "last_timing_report", None)
                    if isinstance(timing_report, dict):
                        item.timing_report = dict(timing_report)
                    output_contract = getattr(
                        remover, "last_output_contract", None)
                    if isinstance(output_contract, dict):
                        item.output_contract_report = dict(output_contract)
                    selective_evidence = getattr(
                        remover, "last_selective_rerun", None)
                    if isinstance(selective_evidence, dict):
                        item.selective_rerun = dict(selective_evidence)
                    if success:
                        break
                    failure_message = (
                        getattr(remover, "last_error_message", None)
                        or "Processing failed"
                    )
                    failure_reason = (
                        getattr(remover, "last_error_reason", None) or ""
                    )
                    detail = ": ".join(
                        part for part in (failure_reason, failure_message)
                        if part
                    )
                    failure = RuntimeError(detail or "Processing failed")

                from backend.batch_report import is_retriable_error
                item.retry_errors.append(str(failure))
                if (
                    attempt >= max_retries
                    or not is_retriable_error(failure)
                ):
                    if raised_error:
                        raise failure
                    break

                if self.cancel_event.is_set() or item.cancel_requested:
                    raise InterruptedError("Processing cancelled")
                if self.pause_event.is_set():
                    raise ProcessingPaused("Processing paused before retry")

                attempt += 1
                item.retry_attempts = attempt
                record = getattr(self, "_batch_report_records", {}).get(
                    item.id)
                if isinstance(record, dict):
                    record["retry_attempts"] = attempt
                    record["retry_errors"] = list(item.retry_errors)
                    if item.mask_export:
                        record["mask_export"] = dict(item.mask_export)
                    if item.mask_import:
                        record["mask_import"] = dict(item.mask_import)
                wait = round(retry_backoff * attempt, 2)
                item.status = ProcessingStatus.LOADING
                item.message = (
                    f"Retrying after transient failure "
                    f"({attempt}/{max_retries})..."
                )
                self._update_item_display(item)
                logger.warning(
                    "Transient failure on %s (attempt %d/%d): %s; "
                    "retrying in %.1fs",
                    file_name, attempt, max_retries, failure, wait,
                )
                deadline = time.monotonic() + wait
                while time.monotonic() < deadline:
                    if self.cancel_event.is_set() or item.cancel_requested:
                        raise InterruptedError("Processing cancelled")
                    if self.pause_event.is_set():
                        raise ProcessingPaused(
                            "Processing paused before retry")
                    time.sleep(min(
                        0.1, max(0.0, deadline - time.monotonic())))

            resume_warning = getattr(remover, "last_resume_warning", None)
            if resume_warning:
                # _process_item runs on the worker thread; Tk widget/toast
                # updates must be marshalled to the main loop.
                dispatch_to_ui(
                    self.root, self._update_status,
                    str(resume_warning), "warning", True,
                )

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
                item.failure_reason = REASON_NONE
                item.quality_report = getattr(remover, "last_quality_report", None)
                item.correction_retry = None
                item.message = MSG_COMPLETE
                quality_note = format_quality_report(item.quality_report, compact=True)
                if quality_note:
                    item.message = f"{MSG_COMPLETE} - {quality_note}"
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
                    or MSG_FAILED
                )
                item.status = ProcessingStatus.ERROR
                item.message = failure_message
                item.error = failure_message
                item.failure_reason = classify_failure_reason(
                    reason=getattr(remover, "last_error_reason", None),
                    message=failure_message,
                )
                item.quality_report = None
                item.completed_at = datetime.now()
                logger.error(f"Failed: {file_name}: {failure_message}")
            self._update_item_display(item)

        except ProcessingPaused:
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
            item.message = MSG_PAUSED
            item.error = None
            item.failure_reason = REASON_PAUSED
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
            item.message = MSG_CANCELLED
            item.error = None
            item.failure_reason = REASON_CANCELLED
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
            item.error = type(e).__name__
            # Queue rows and persisted queue_state must not carry paths or
            # traceback fragments; the log already has the full exception.
            item.message = user_facing_processing_error(e)
            item.failure_reason = classify_failure_reason(
                exc=e, reason=getattr(remover_obj, "last_error_reason", None))
            item.quality_report = None
            item.completed_at = datetime.now()
            self._update_item_display(item)
            logger.error(f"Processing error for {item.file_path}: {e}", exc_info=True)
        finally:
            remover_obj = locals().get("remover")
            if remover_obj is not None:
                item.detection_stats = dict(
                    getattr(remover_obj, "last_detection_stats", {}) or {})
                # RM-147: persist how this job actually executed so the queue,
                # report, and support bundle agree.
                provenance = getattr(remover_obj, "execution_provenance", None)
                if provenance is not None:
                    try:
                        item.execution_provenance = provenance.to_dict()
                    except Exception:
                        logger.warning(
                            "Execution provenance capture failed", exc_info=True)
            if self._active_remover is locals().get("remover"):
                self._active_remover = None
            if hasattr(self, "queue"):
                save_queue_state(self.queue)

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
        # Snapshot the queue under the lock: iterating self.queue directly on
        # the worker thread races the main thread adding/removing items and can
        # raise "list changed size during iteration".
        with self.queue_lock:
            queue_snapshot = list(self.queue)
        for item in queue_snapshot:
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
                from backend.detection import SubtitleDetector
                lang = first_video.config.detection_lang or "en"
                engine = getattr(
                    first_video.config, "detection_engine", "auto") or "auto"
                variant = getattr(
                    first_video.config, "rapidocr_variant", "v6") or "v6"
                with self._detector_lock:
                    detector = self._preview_detector
                    if (
                        detector is None
                        or self._preview_detector_lang != lang
                        or getattr(
                            self, "_preview_detector_engine", None) != engine
                        or (getattr(
                            self, "_preview_detector_variant", None) or "v6") != variant
                    ):
                        detector = SubtitleDetector(
                            lang=lang,
                            engine=engine,
                            rapidocr_variant=variant,
                        )
                        self._preview_detector = detector
                        self._preview_detector_lang = lang
                        self._preview_detector_engine = engine
                        self._preview_detector_variant = variant
                threshold = getattr(first_video.config, "detection_threshold", 0.5)
                t0 = time.monotonic()
                frames_done = 0
                for _ in range(probe_frames):
                    ok, frame = cap.read()
                    if not ok:
                        break
                    # Serialize inference on the shared detector: this probe
                    # runs on the worker thread and can overlap a review-mask
                    # preview, and the OCR predictors are not thread-safe.
                    with self._detector_lock:
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
            label = tr("{current} of {total} complete").format(
                current=current, total=total)
            if eta:
                label += tr("   -   about {eta} left").format(eta=eta)
            self.batch_label.config(text=label, fg=Theme.TEXT_SECONDARY)
            self.batch_percent_label.config(text=f"{pct}%", fg=Theme.BLUE_PRIMARY)
            self.root.title(tr("[{current}/{total}] {app} v{version}").format(
                current=current, total=total,
                app=APP_NAME, version=APP_VERSION))
            # Windows taskbar
            self._ensure_taskbar()
            if self._taskbar:
                self._taskbar.set_state(TaskbarProgress.STATE_NORMAL)
                self._taskbar.set_value(current, total)
        else:
            self.batch_progress.set_progress(0)
            self.batch_label.config(text=tr("Ready"), fg=Theme.TEXT_MUTED)
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
                self.root.update_idletasks()
                self._shutdown_ui_resources()
                self.root.destroy()
            except Exception:
                pass
            return
        self.start_btn.set_style("primary")
        self.start_btn.icon = ">"
        self.start_btn.set_text(tr("Start batch"))
        self.root.title(f"{APP_NAME} v{APP_VERSION}")
        self.batch_progress.set_progress(0)
        self.batch_label.config(text=tr("Ready"), fg=Theme.TEXT_MUTED)
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
            summary += f", {review_count} to review"
        if paused:
            summary += f", {paused} paused"
        if cancelled:
            summary += f", {cancelled} stopped"
        is_clean = errors == 0 and paused == 0 and cancelled == 0 and review_count == 0
        completed_items = [
            item for item in self.queue
            if item.status == ProcessingStatus.COMPLETE
        ]
        quality_summary = summarize_quality_reports(
            [item.quality_report for item in completed_items]
        )
        # RM-281: attribute the batch's worst sampled frame back to its item
        # so the summary can open the A/B compare directly on it.
        if quality_summary and isinstance(quality_summary.get("worst_frame"), dict):
            worst = quality_summary["worst_frame"]
            position = int(worst.get("position", -1))
            if 0 <= position < len(completed_items):
                worst["item_id"] = completed_items[position].id
                worst["item_name"] = Path(
                    completed_items[position].file_path).name
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
                _item = "item" if paused == 1 else "items"
                _remain = "remains" if paused == 1 else "remain"
                announce(
                    f"Batch paused. {paused} {_item} {_remain} paused. "
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
