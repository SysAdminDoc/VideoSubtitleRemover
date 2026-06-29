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


class QualityReviewControllerMixin:
    """Focused controller methods mixed into VideoSubtitleRemoverApp."""

    @staticmethod
    def _stage_label(name: str) -> str:
        labels = {
            "decode": "decode",
            "ocr": "OCR",
            "mask": "mask",
            "inpaint": "inpaint",
            "encode": "encode",
            "mux": "mux",
            "quality": "quality",
        }
        return labels.get(str(name or ""), str(name or ""))

    @classmethod
    def _dominant_stage_text(cls, stage: Optional[dict]) -> str:
        if not isinstance(stage, dict):
            return ""
        name = cls._stage_label(str(stage.get("name") or ""))
        try:
            seconds = float(stage.get("seconds") or 0.0)
        except (TypeError, ValueError):
            seconds = 0.0
        if not name or seconds <= 0.0:
            return ""
        return f"{name} {format_time(seconds)}"

    def _show_batch_summary(self, complete: int, errors: int,
                            cancelled: int, elapsed: str,
                            paused: int = 0,
                            quality_summary: Optional[dict] = None,
                            review_count: int = 0,
                            stage_summary: Optional[dict] = None):
        """Themed summary modal shown when a batch finishes."""
        total = complete + errors + paused + cancelled
        is_clean = errors == 0 and paused == 0 and cancelled == 0 and review_count == 0

        dialog = tk.Toplevel(self.root)
        dialog.withdraw()
        dialog.title(tr("Batch finished"))
        dialog.configure(bg=Theme.BG_OVERLAY)
        dialog.resizable(False, False)
        dialog.transient(self.root)

        outer = tk.Frame(dialog, bg=Theme.BORDER, padx=1, pady=1)
        outer.pack()
        body = tk.Frame(outer, bg=Theme.BG_SECONDARY)
        body.pack()

        content = tk.Frame(body, bg=Theme.BG_SECONDARY)
        content.pack(padx=32, pady=(26, 16))

        title_text = (
            tr("Batch finished")
            if is_clean else tr("Batch finished with issues")
        )
        title_color = Theme.SUCCESS if is_clean else Theme.WARNING
        try:
            from backend.a11y import set_accessible_metadata
            set_accessible_metadata(
                dialog,
                role="dialog",
                label=title_text,
                state="modal",
                value=(
                    f"{complete} completed, {errors} failed, "
                    f"{paused} paused, {cancelled} stopped, {review_count} review"
                ),
            )
        except Exception:
            pass
        tk.Label(content, text=title_text, font=f(Theme.F_HEADING, "bold"),
                 bg=Theme.BG_SECONDARY, fg=title_color).pack(anchor="w")
        if elapsed:
            tk.Label(content, text=tr(
                         "Total time {elapsed} - {count} item{suffix} processed"
                     ).format(
                         elapsed=elapsed,
                         count=total,
                         suffix="s" if total != 1 else "",
                     ),
                     font=f(Theme.F_BODY_SM),
                     bg=Theme.BG_SECONDARY, fg=Theme.TEXT_MUTED).pack(
                         anchor="w", pady=(2, 0))
        if is_clean:
            summary_note = tr("Outputs are ready to review.")
        elif review_count:
            summary_note = tr(
                "Some completed outputs need a closer look. Start with "
                "the quality review item, then retry failed items if needed."
            )
        elif paused:
            summary_note = tr(
                "The current video is paused at a checkpoint. Start the batch "
                "again to resume from that frame."
            )
        else:
            summary_note = tr(
                "Completed outputs are ready. Review the outliers or "
                "open the log for details."
            )
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

        stat(stats, tr("COMPLETED"), complete, Theme.SUCCESS, Theme.SUCCESS_BG).pack(
            side="left")
        stat(stats, tr("FAILED"), errors, Theme.ERROR, Theme.ERROR_BG).pack(
            side="left", padx=(Theme.S_SM, 0))
        if paused:
            stat(stats, tr("PAUSED"), paused, Theme.WARNING, Theme.WARNING_BG).pack(
                side="left", padx=(Theme.S_SM, 0))
        stat(stats, tr("STOPPED"), cancelled, Theme.WARNING, Theme.WARNING_BG).pack(
            side="left", padx=(Theme.S_SM, 0))
        if review_count:
            stat(stats, tr("REVIEW"), review_count, Theme.WARNING, Theme.WARNING_BG).pack(
                side="left", padx=(Theme.S_SM, 0))

        slow_stage = None
        if isinstance(stage_summary, dict):
            slow_stage = stage_summary.get("slowest_stage")
        slow_text = self._dominant_stage_text(slow_stage)
        if slow_text:
            stage_card = tk.Frame(content, bg=Theme.BG_CARD, highlightthickness=1,
                                  highlightbackground=Theme.BORDER_SUBTLE)
            stage_card.pack(fill="x", pady=(Theme.S_LG, 0))
            tk.Label(
                stage_card,
                text=tr("Slowest stage"),
                font=f(Theme.F_BODY_SM, "bold"),
                bg=Theme.BG_CARD,
                fg=Theme.TEXT_PRIMARY,
            ).pack(anchor="w", padx=Theme.S_LG, pady=(Theme.S_MD, 0))
            tk.Label(
                stage_card,
                text=tr(
                    "{stage} dominated this run. Open the report for per-item "
                    "decode, OCR, mask, inpaint, encode, mux, and quality timings."
                ).format(stage=slow_text),
                font=f(Theme.F_META),
                bg=Theme.BG_CARD,
                fg=Theme.TEXT_MUTED,
                wraplength=420,
                justify="left",
            ).pack(anchor="w", padx=Theme.S_LG, pady=(4, Theme.S_MD))

        if quality_summary:
            quality_card = tk.Frame(content, bg=Theme.BG_CARD, highlightthickness=1,
                                    highlightbackground=Theme.BORDER_SUBTLE)
            quality_card.pack(fill="x", pady=(Theme.S_LG, 0))

            tk.Label(
                quality_card,
                text=tr("Sampled quality check"),
                font=f(Theme.F_BODY_SM, "bold"),
                bg=Theme.BG_CARD,
                fg=Theme.TEXT_PRIMARY,
            ).pack(anchor="w", padx=Theme.S_LG, pady=(Theme.S_MD, 0))

            items_measured = int(quality_summary.get("items", 0) or 0)
            samples = int(quality_summary.get("samples", 0) or 0)
            tk.Label(
                quality_card,
                text=tr(
                    "Measured {items} completed item{item_suffix} across "
                    "{samples} sampled frame{sample_suffix}. Higher is generally better."
                ).format(
                    items=items_measured,
                    item_suffix="s" if items_measured != 1 else "",
                    samples=samples,
                    sample_suffix="s" if samples != 1 else "",
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

        def _review_first_and_close():
            self._open_first_review_item()
            _close()

        def _retry_suggested_and_close():
            if self._retry_first_review_with_suggested_settings():
                _close()

        default_button = None

        if review_count > 0:
            default_button = ModernButton(
                actions_inner, text=tr("Review first"), width=122,
                command=_review_first_and_close,
                style="accent", size="md",
            )
            default_button.pack(side="left")
            retry_button = ModernButton(
                actions_inner, text=tr("Retry suggested"), width=140,
                command=_retry_suggested_and_close,
                style="ghost", size="md",
            )
            retry_button.pack(side="left", padx=(Theme.S_SM, 0))
        if complete > 0:
            open_button = ModernButton(
                actions_inner, text=tr("Open output"), width=132,
                command=_open_output_and_close,
                style="accent" if review_count == 0 else "ghost",
                size="md", icon="^",
            )
            if default_button is None:
                default_button = open_button
            open_button.pack(
                side="left",
                padx=(Theme.S_SM, 0) if review_count else 0,
            )
        report_paths = getattr(self, "_last_batch_report_paths", [])
        if report_paths:
            def _open_report_and_close():
                self._open_batch_report_path(report_paths)
                _close()
            report_button = ModernButton(
                actions_inner, text=tr("Open report"), width=116,
                command=_open_report_and_close,
                style="ghost", size="md",
            )
            if default_button is None:
                default_button = report_button
            report_button.pack(side="left", padx=(Theme.S_SM, 0))
        if errors > 0:
            log_button = ModernButton(
                actions_inner, text=tr("Open log"), width=104,
                command=self._open_log_file,
                style="ghost", size="md",
            )
            if default_button is None:
                default_button = log_button
            log_button.pack(side="left", padx=(Theme.S_SM, 0))
        if errors > 0 or cancelled > 0:
            retry_failed_button = ModernButton(
                actions_inner, text=tr("Retry failed"), width=110,
                command=_retry_failed_and_close,
                style="ghost", size="md",
            )
            if default_button is None:
                default_button = retry_failed_button
            retry_failed_button.pack(side="left", padx=(Theme.S_SM, 0))
        close_button = ModernButton(
            actions_inner, text=tr("Close"), width=92,
            command=_close, style="primary", size="md",
        )
        close_button.pack(side="left", padx=(Theme.S_SM, 0))
        if default_button is None:
            default_button = close_button

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
        try:
            default_button.focus_set()
        except Exception:
            pass

    def _warn_output_quality_preflight(self):
        from backend.output_quality_preflight import output_quality_preflight_messages

        records = getattr(self, "_batch_report_records", {}) or {}
        warning_items = []
        for record in records.values():
            preflight = record.get("output_quality_preflight")
            if not isinstance(preflight, dict):
                continue
            messages = output_quality_preflight_messages(preflight)
            if messages:
                warning_items.append((record.get("input_name", "input"), messages))
        if not warning_items:
            return
        count = len(warning_items)
        first_name, first_messages = warning_items[0]
        first = first_messages[0] if first_messages else "review output settings"
        self._update_status(
            f"Output quality preflight warning for {count} item(s): {first_name} - {first}",
            "warning",
            toast=True,
        )
        for name, messages in warning_items:
            logger.warning(
                "Output quality preflight for %s: %s",
                name,
                " ".join(messages),
            )

    def _finalize_batch_report_records(self) -> List[dict]:
        from backend.batch_report import (
            STATUS_CANCELLED,
            STATUS_FAILED,
            STATUS_HARDCODED_PROCESSED,
            STATUS_PAUSED,
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
            elif item.status == ProcessingStatus.PAUSED:
                status = STATUS_PAUSED
                message = item.message or "Paused at checkpoint"
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
                stage_timings=item.stage_timings,
            )
            finished.append(record)
        return finished

    def _review_needed_records(self) -> List[dict]:
        records = getattr(self, "_last_batch_report_records", []) or []
        return [
            record for record in records
            if record.get("status") == "review-needed"
        ]

    def _queue_item_for_report_record(self, record: dict) -> Optional[QueueItem]:
        output = record.get("output")
        output_key = self._normalized_path_key(output) if output else ""
        output_name = record.get("output_name")
        for item in self.queue:
            if output_key and self._normalized_path_key(item.output_path) == output_key:
                return item
            if output_name and Path(item.output_path).name == output_name:
                return item
        return None

    def _open_first_review_item(self):
        records = self._review_needed_records()
        if not records:
            self._update_status("No quality review items are available", "info")
            return
        record = records[0]
        item = self._queue_item_for_report_record(record)
        if item is not None:
            self._set_selected_queue_item(item.id)
            self._scroll_queue_to_item(item.id)
            if item.status == ProcessingStatus.COMPLETE:
                self._show_preview(item)
        quality_report = record.get("quality_report") or {}
        gate = record.get("quality_gate") or {}
        stage_text = self._dominant_stage_text(record.get("dominant_stage"))
        stage_suffix = f"; slowest stage {stage_text}" if stage_text else ""
        candidates = []
        sheet = quality_report.get("sheet")
        if sheet:
            candidates.append(sheet)
        for key in ("previewFramePaths", "preview_frame_paths"):
            paths = gate.get(key) or quality_report.get(key)
            if isinstance(paths, (list, tuple)):
                candidates.extend(paths)
        for path in candidates:
            if path and Path(path).exists():
                try:
                    os.startfile(str(path))
                    self._update_status(
                        f"Opened quality review for {record.get('output_name', 'output')}{stage_suffix}",
                        "warning",
                    )
                    return
                except Exception:
                    logger.warning("Could not open quality review artifact", exc_info=True)
        if self._open_batch_report_path(getattr(self, "_last_batch_report_paths", [])):
            self._update_status(
                f"Opened batch report for review{stage_suffix}",
                "warning",
            )
            return
        self._update_status(
            f"Focused {record.get('output_name', 'the first review item')}{stage_suffix}",
            "warning",
        )

    @staticmethod
    def _retry_changes(before: dict, after: dict, patch: dict) -> dict:
        changes = {}
        for key in sorted(patch):
            if before.get(key) != after.get(key):
                changes[key] = {
                    "before": before.get(key),
                    "after": after.get(key),
                }
        return changes

    def _review_record_for_item(self, item: QueueItem) -> Optional[dict]:
        output_key = self._normalized_path_key(item.output_path)
        for record in self._review_needed_records():
            record_output = record.get("output")
            if record_output and self._normalized_path_key(record_output) == output_key:
                return record
            if record.get("output_name") == Path(item.output_path).name:
                return record
        return None

    def _quality_gate_for_item(self, item: QueueItem,
                               record: Optional[dict] = None) -> dict:
        report = item.quality_report if isinstance(item.quality_report, dict) else {}
        gate = report.get("quality_gate")
        if isinstance(gate, dict):
            return gate
        if isinstance(record, dict):
            gate = record.get("quality_gate")
            if isinstance(gate, dict):
                return gate
        return {}

    def _retry_review_item_with_suggested_settings(self, item_id: str) -> bool:
        if self.is_processing:
            self._update_status("Stop the active batch before preparing a retry", "warning")
            return False
        item = self._queue_item_by_id(item_id)
        if item is None:
            self._update_status("The review item is no longer in the queue", "warning")
            return False
        record = self._review_record_for_item(item)
        gate = self._quality_gate_for_item(item, record)
        try:
            from backend.quality_gate import retry_config_patch_for_gate
            patch = retry_config_patch_for_gate(gate, item.config.to_dict())
        except Exception as exc:
            logger.warning("Could not build quality retry config", exc_info=True)
            self._update_status(f"Could not prepare suggested retry: {exc}", "warning")
            return False
        if not patch:
            self._update_status("No automatic retry settings are available for this review item", "warning")
            return False

        before = item.config.to_dict()
        after_payload = dict(before)
        after_payload.update(patch)
        after_config = ProcessingConfig.from_dict(after_payload)
        after = after_config.to_dict()
        changes = self._retry_changes(before, after, patch)
        if not changes:
            self._update_status("Suggested retry settings already match this item", "info")
            return False

        item.config = after_config
        item.status = ProcessingStatus.IDLE
        item.progress = 0.0
        item.message = "Ready to retry with suggested settings"
        item.error = None
        item.quality_report = None
        item.started_at = None
        item.completed_at = None
        item.retry_config = {
            "schema": "vsr.retry_config.v1",
            "source": "quality_gate",
            "ladderStep": str(gate.get("ladderStep") or ""),
            "qualityGateReason": str(gate.get("reason") or ""),
            "before": {key: value["before"] for key, value in changes.items()},
            "after": {key: value["after"] for key, value in changes.items()},
            "changes": changes,
        }
        self._set_selected_queue_item(item.id)
        self._update_queue_display()
        if item.id in self.queue_widgets:
            self.queue_widgets[item.id].update_item(item)
        save_queue_state(self.queue)
        changed_keys = ", ".join(changes)
        self._update_status(
            f"Prepared retry for {Path(item.file_path).name}: {changed_keys}",
            "success",
            toast=True,
        )
        return True

    def _retry_first_review_with_suggested_settings(self) -> bool:
        records = self._review_needed_records()
        if not records:
            self._update_status("No quality review items are available", "info")
            return False
        item = self._queue_item_for_report_record(records[0])
        if item is None:
            self._update_status("The first review item is no longer in the queue", "warning")
            return False
        return self._retry_review_item_with_suggested_settings(item.id)

    @staticmethod
    def _preferred_batch_report_path(report_paths) -> Optional[Path]:
        existing: List[Path] = []
        for report_path in report_paths or []:
            try:
                path = Path(report_path)
            except (TypeError, ValueError):
                continue
            if path.exists():
                existing.append(path)
        for path in existing:
            if path.suffix.lower() == ".md":
                return path
        return existing[0] if existing else None

    def _open_batch_report_path(self, report_paths) -> bool:
        path = self._preferred_batch_report_path(report_paths)
        if path is None:
            return False
        try:
            os.startfile(str(path))
            return True
        except Exception:
            logger.warning("Could not open batch report", exc_info=True)
            return False

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
        self._last_batch_report_records = records
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

