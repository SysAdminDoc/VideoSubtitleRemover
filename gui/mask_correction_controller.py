"""Quality-directed add/subtract mask correction editor."""

from __future__ import annotations

import logging
import threading
from functools import partial
from pathlib import Path
from typing import Any, Optional, Protocol

import numpy as np

try:
    import tkinter as tk
    from tkinter import ttk
except ImportError:  # pragma: no cover
    pass

try:
    from PIL import Image, ImageTk

    PIL_AVAILABLE = True
except ImportError:  # pragma: no cover
    PIL_AVAILABLE = False

from backend.a11y import set_accessible_metadata
from backend.i18n import tr
from backend.mask_corrections import (
    MASK_CORRECTION_SCHEMA,
    SELECTIVE_RERUN_SCHEMA,
    apply_mask_corrections,
    brush_polygon,
    merge_frame_ranges,
    normalize_mask_correction_list,
)
from backend.region_editing import RegionEditHistory
from backend.safe_image import safe_imread
from gui.config import ProcessingStatus, QueueItem, save_queue_state
from gui.theme import Theme, f
from gui.utils import is_video_file
from gui.widgets import ModernButton

logger = logging.getLogger(__name__)


class MaskCorrectionControllerHost(Protocol):
    """Selection and status surface required by mask correction."""

    root: Any
    queue: list[QueueItem]

    def _update_status(
        self, message: str, tone: str = "neutral", toast: bool = False
    ) -> None:
        ...


class MaskCorrectionWindow:
    """Non-blocking quality-directed mask correction editor."""

    def __init__(self, host, item: QueueItem, initial_span=None):
        self.host = host
        self.item = item
        self.initial_span = initial_span

    def __getattr__(self, name: str):
        return getattr(self.host, name)

    def open(self) -> bool:
        initial_span = self.initial_span
        if self.is_processing:
            self._update_status(
                tr("Stop the active batch before correcting a mask"), "warning")
            return False
        if not PIL_AVAILABLE:
            self._update_status(
                tr("Install Pillow to enable mask correction"), "warning")
            return False
        self.cap = None
        try:
            import cv2
            self.cv2 = cv2

            self.is_video = is_video_file(self.item.file_path)
            if self.is_video:
                self.cap = cv2.VideoCapture(self.item.file_path)
                if not self.cap.isOpened():
                    raise ValueError(tr("The source video could not be opened"))
                self.frame_count = max(1, int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)))
                self.fps = float(self.cap.get(cv2.CAP_PROP_FPS) or 30.0)
                self.fps = self.fps if np.isfinite(self.fps) and self.fps > 0 else 30.0
                self.width = max(1, int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
                self.height = max(1, int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
                self.still = None
            else:
                self.still = safe_imread(self.item.file_path)
                if self.still is None:
                    raise ValueError(tr("The source image could not be opened"))
                self.height, self.width = self.still.shape[:2]
                self.frame_count, self.fps = 1, 1.0

            self.spans = self._mask_review_spans_for_item(self.item) or [{
                "kind": "manual",
                "start_frame": 0,
                "end_frame": 1,
                "reason": tr("Manual mask review"),
                "suggested_mode": "add",
            }]
            selected_span_index = 0
            if isinstance(initial_span, dict):
                target = (
                    str(initial_span.get("kind") or ""),
                    int(initial_span.get("start_frame", 0) or 0),
                )
                selected_span_index = next((
                    index for index, span in enumerate(self.spans)
                    if (str(span.get("kind") or ""), span["start_frame"]) == target
                ), 0)

            self.original = normalize_mask_correction_list(
                self.item.config.manual_mask_corrections) or []
            self.state = {
                "corrections": list(self.original),
                "frame": None,
                "frame_index": 0,
                "base_mask": np.zeros((self.height, self.width), dtype=np.uint8),
                "active": None,
                "last_point": None,
                "request": 0,
            }
            self.history = RegionEditHistory(limit=100)
            screen_w = self.root.winfo_screenwidth()
            screen_h = self.root.winfo_screenheight()
            self.scale = min(
                min(900, int(screen_w * 0.78)) / self.width,
                min(430, int(screen_h * 0.52)) / self.height,
                1.0,
            )
            self.display_w = max(1, int(self.width * self.scale))
            self.display_h = max(1, int(self.height * self.scale))

            self.win = tk.Toplevel(self.root)
            self.win.title(tr("Correct subtitle mask"))
            self.win.configure(bg=Theme.BG_DARK)
            self.win.resizable(False, False)
            self.win.transient(self.root)

            header = tk.Frame(self.win, bg=Theme.BG_SECONDARY)
            header.pack(fill="x", padx=Theme.S_MD, pady=(Theme.S_MD, Theme.S_SM))
            self.span_labels = [
                tr("{kind}: frames {start}-{end}").format(
                    kind=str(span.get("kind", "review")).replace("-", " ").title(),
                    start=span["start_frame"],
                    end=max(span["start_frame"], span["end_frame"] - 1),
                )
                for span in self.spans
            ]
            self.span_var = tk.StringVar(value=self.span_labels[selected_span_index])
            span_picker = ttk.Combobox(
                header, state="readonly", width=31, textvariable=self.span_var,
                values=self.span_labels, style="Dark.TCombobox")
            span_picker.pack(side="left")
            set_accessible_metadata(
                span_picker, role="combo box", label=tr("Quality review span"),
                description=tr("Choose a residual, flicker, or low-confidence span."))

            self.mode_labels = {
                tr("Add to mask"): "add",
                tr("Subtract from mask"): "subtract",
            }
            self.mode_var = tk.StringVar(value=next(iter(self.mode_labels)))
            mode_picker = ttk.Combobox(
                header, state="readonly", width=18, textvariable=self.mode_var,
                values=tuple(self.mode_labels), style="Dark.TCombobox")
            mode_picker.pack(side="left", padx=(Theme.S_SM, 0))
            set_accessible_metadata(
                mode_picker, role="combo box", label=tr("Correction paint mode"),
                description=tr("Add missing pixels or subtract over-masked pixels."))

            controls = tk.Frame(self.win, bg=Theme.BG_DARK)
            controls.pack(fill="x", padx=Theme.S_MD, pady=(0, Theme.S_SM))
            self.propagate_var = tk.BooleanVar(value=False)
            propagate = tk.Checkbutton(
                controls, text=tr("Propagate through review span"),
                variable=self.propagate_var, bg=Theme.BG_DARK, fg=Theme.TEXT_PRIMARY,
                activebackground=Theme.BG_DARK, activeforeground=Theme.TEXT_PRIMARY,
                selectcolor=Theme.BG_TERTIARY, highlightthickness=0, takefocus=True)
            propagate.pack(side="left")
            self.start_var = tk.StringVar()
            self.end_var = tk.StringVar()
            frame_entries = []
            for label, variable in ((tr("Start frame"), self.start_var),
                                    (tr("End frame"), self.end_var)):
                tk.Label(
                    controls, text=label, font=f(Theme.F_META),
                    bg=Theme.BG_DARK, fg=Theme.TEXT_MUTED).pack(
                        side="left", padx=(Theme.S_MD, Theme.S_XS))
                entry = tk.Entry(
                    controls, width=8, textvariable=variable,
                    bg=Theme.BG_TERTIARY, fg=Theme.TEXT_PRIMARY,
                    insertbackground=Theme.TEXT_PRIMARY, relief="flat")
                entry.pack(side="left")
                frame_entries.append(entry)
                set_accessible_metadata(
                    entry, role="numeric input", label=label,
                    description=tr("End frame is exclusive and must be in bounds."))
            self.brush_var = tk.IntVar(value=12)
            tk.Label(
                controls, text=tr("Brush"), font=f(Theme.F_META),
                bg=Theme.BG_DARK, fg=Theme.TEXT_MUTED).pack(
                    side="left", padx=(Theme.S_MD, Theme.S_XS))
            brush = tk.Scale(
                controls, from_=2, to=80, orient="horizontal", length=130,
                variable=self.brush_var, bg=Theme.BG_DARK, fg=Theme.TEXT_PRIMARY,
                troughcolor=Theme.BG_TERTIARY,
                activebackground=Theme.BLUE_PRIMARY, highlightthickness=0)
            brush.pack(side="left")
            set_accessible_metadata(
                brush, role="slider", label=tr("Correction brush radius"),
                description=tr("Brush radius in source pixels."))

            self.status_var = tk.StringVar(value=tr("Loading review frame..."))
            status = tk.Label(
                self.win, textvariable=self.status_var, font=f(Theme.F_META),
                bg=Theme.BG_DARK, fg=Theme.TEXT_MUTED, anchor="w")
            status.pack(fill="x", padx=Theme.S_MD, pady=(0, Theme.S_XS))
            set_accessible_metadata(
                status, role="status", label=tr("Mask correction status"))

            self.canvas = tk.Canvas(
                self.win, width=self.display_w, height=self.display_h, bg=Theme.BG_TERTIARY,
                cursor="crosshair", takefocus=True, highlightthickness=2,
                highlightbackground=Theme.BORDER_SUBTLE,
                highlightcolor=Theme.BLUE_PRIMARY)
            self.canvas.pack(padx=Theme.S_MD)
            self.image_id = self.canvas.create_image(0, 0, anchor="nw")
            self.canvas._photo = None
            set_accessible_metadata(
                self.canvas, role="mask painting canvas",
                label=tr("Frame-local mask correction canvas"),
                description=tr("Drag to paint with the selected mode."))

            actions = tk.Frame(self.win, bg=Theme.BG_DARK)
            actions.pack(fill="x", padx=Theme.S_MD, pady=Theme.S_MD)
            self.buttons = {"undo": None, "redo": None}



















            undo_button = ModernButton(
                actions, text=tr("Undo"), width=76, command=self.undo,
                style="ghost", size="sm")
            undo_button.pack(side="left")
            redo_button = ModernButton(
                actions, text=tr("Redo"), width=76, command=self.redo,
                style="ghost", size="sm")
            redo_button.pack(side="left", padx=(Theme.S_XS, 0))
            self.buttons.update(undo=undo_button, redo=redo_button)
            ModernButton(
                actions, text=tr("Clear corrections"), width=132,
                command=self.clear_corrections, style="ghost", size="sm").pack(
                    side="left", padx=(Theme.S_SM, 0))
            ModernButton(
                actions, text=tr("Prepare selective rerun"), width=184,
                command=self.prepare_rerun, style="primary", size="sm").pack(side="right")
            ModernButton(
                actions, text=tr("Cancel"), width=82,
                command=self.win.destroy, style="ghost", size="sm").pack(
                    side="right", padx=(0, Theme.S_SM))

            propagate.configure(command=self.set_span_range)
            span_picker.bind("<<ComboboxSelected>>", self.span_changed)
            self.canvas.bind("<ButtonPress-1>", self.paint_press)
            self.canvas.bind("<B1-Motion>", self.paint_drag)
            self.canvas.bind("<ButtonRelease-1>", self.paint_release)
            self.win.bind("<Control-z>", self.undo)
            self.win.bind("<Control-y>", self.redo)
            self.win.bind("<Control-Shift-Z>", self.redo)
            self.win.bind("<Escape>", lambda _event: self.win.destroy())


            self.win.bind("<Destroy>", self.release_capture)
            self.win._vsr_span_picker = span_picker
            self.win._vsr_mode_picker = mode_picker
            self.win._vsr_frame_entries = frame_entries
            self.win._vsr_correction_canvas = self.canvas
            self.win._vsr_correction_state = self.state
            self.win._vsr_paint_handlers = (self.paint_press, self.paint_drag, self.paint_release)
            self.win._vsr_undo_correction = self.undo
            self.win._vsr_redo_correction = self.redo
            self.win._vsr_prepare_selective_rerun = self.prepare_rerun
            self.update_history_buttons()
            self.span_changed()
            self.win.grab_set()
            return True
        except Exception as exc:
            if self.cap is not None:
                self.cap.release()
            logger.warning("Mask correction editor failed", exc_info=True)
            self._update_status(
                tr("Mask correction editor could not open: {error}").format(error=exc),
                "warning")
            return False

    def selected_span(self):
        try:
            index = self.span_labels.index(self.span_var.get())
        except ValueError:
            index = 0
        return self.spans[index]

    def set_span_range(self):
        span = self.selected_span()
        anchor = min(self.frame_count - 1, span["start_frame"])
        self.start_var.set(str(anchor))
        self.end_var.set(str(
            min(self.frame_count, span["end_frame"])
            if self.propagate_var.get() else min(self.frame_count, anchor + 1)))

    def parse_range(self):
        try:
            start, end = int(self.start_var.get()), int(self.end_var.get())
        except (TypeError, ValueError) as exc:
            raise ValueError(
                tr("Start and end frames must be whole numbers")) from exc
        if start < 0 or end <= start or end > self.frame_count:
            raise ValueError(
                tr("Correction frames must form a non-empty in-bounds range"))
        return start, end

    def render(self):
        frame = self.state["frame"]
        if frame is None:
            return
        frame_index = self.state["frame_index"]
        final = apply_mask_corrections(
            self.state["base_mask"].copy(), self.state["corrections"],
            frame_index / max(self.fps, 1e-9), frame_index) > 0
        base = self.state["base_mask"] > 0
        rgb = self.cv2.cvtColor(frame, self.cv2.COLOR_BGR2RGB).astype(np.float32)
        for selector, color, alpha in (
            (base & final, (245, 190, 55), 0.26),
            (final & ~base, (239, 68, 68), 0.52),
            (base & ~final, (59, 130, 246), 0.58),
        ):
            if np.any(selector):
                rgb[selector] = (
                    rgb[selector] * (1.0 - alpha)
                    + np.asarray(color) * alpha)
        image = Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8))
        image = image.resize((self.display_w, self.display_h), Image.LANCZOS)
        self.canvas._photo = ImageTk.PhotoImage(image)
        self.canvas.itemconfig(self.image_id, image=self.canvas._photo)

    def update_history_buttons(self):
        self.buttons["undo"].set_enabled(
            self.history.can_undo, tr("No correction to undo"))
        self.buttons["redo"].set_enabled(
            self.history.can_redo, tr("No correction to redo"))

    def record_history(self):
        self.history.record(self.state["corrections"])
        self.update_history_buttons()

    def undo(self, event=None):
        if event is not None and isinstance(event.widget, tk.Entry):
            return None
        restored = self.history.undo(self.state["corrections"])
        if restored is not None:
            self.state["corrections"] = restored
            self.render()
            self.status_var.set(tr("Undid mask correction"))
        self.update_history_buttons()
        return "break"

    def redo(self, event=None):
        if event is not None and isinstance(event.widget, tk.Entry):
            return None
        restored = self.history.redo(self.state["corrections"])
        if restored is not None:
            self.state["corrections"] = restored
            self.render()
            self.status_var.set(tr("Redid mask correction"))
        self.update_history_buttons()
        return "break"

    def source_point(self, event):
        return (
            max(0, min(self.width - 1, int(round(event.x / max(self.scale, 1e-9))))),
            max(0, min(self.height - 1, int(round(event.y / max(self.scale, 1e-9))))),
        )

    def add_dab(self, point):
        operation = self.state["active"]
        if operation is not None:
            operation["polygons"].append(brush_polygon(
                point[0], point[1], self.brush_var.get(), self.width, self.height))
            self.state["last_point"] = point
            self.render()

    def paint_press(self, event):
        try:
            start, end = self.parse_range()
        except ValueError as exc:
            self.status_var.set(str(exc))
            return "break"
        self.record_history()
        span = self.selected_span()
        operation = {
            "schema": MASK_CORRECTION_SCHEMA,
            "mode": self.mode_labels.get(self.mode_var.get(), "add"),
            "polygons": [],
            "start": start / max(self.fps, 1e-9),
            "end": end / max(self.fps, 1e-9),
            "start_frame": start,
            "end_frame": end,
            "propagation": "span" if end - start > 1 else "frame",
            "source": span.get("kind", "manual"),
        }
        self.state["corrections"].append(operation)
        self.state["active"] = operation
        self.add_dab(self.source_point(event))
        self.status_var.set(tr("Painting mask correction"))
        return "break"

    def paint_drag(self, event):
        if self.state["active"] is None:
            return None
        point = self.source_point(event)
        last = self.state["last_point"]
        minimum = max(1.0, self.brush_var.get() * 0.45)
        if last is None or np.hypot(point[0] - last[0], point[1] - last[1]) >= minimum:
            self.add_dab(point)
        return "break"

    def paint_release(self, _event):
        self.state["active"] = None
        self.state["last_point"] = None
        self.state["corrections"] = (
            normalize_mask_correction_list(self.state["corrections"]) or [])
        self.render()
        self.status_var.set(tr("Correction stroke added; undo is available"))
        return "break"

    def clear_corrections(self):
        if self.state["corrections"]:
            self.record_history()
            self.state["corrections"] = []
            self.render()
            self.status_var.set(tr("Cleared mask corrections"))

    def detect_mask(self, frame, frame_index):
        base = np.zeros(frame.shape[:2], dtype=np.uint8)
        mask_path = str((getattr(self.item, "mask_export", None) or {}).get("path") or "")
        if mask_path and Path(mask_path).is_file():
            mask_cap = self.cv2.VideoCapture(mask_path)
            try:
                mask_cap.set(self.cv2.CAP_PROP_POS_FRAMES, frame_index)
                ok, mask_frame = mask_cap.read()
            finally:
                mask_cap.release()
            if ok and mask_frame is not None:
                gray = self.cv2.cvtColor(mask_frame, self.cv2.COLOR_BGR2GRAY)
                return np.where(gray > 127, 255, 0).astype(np.uint8), "exported mask"
        try:
            from backend.detection import SubtitleDetector

            lock = getattr(self, "_detector_lock", threading.Lock())
            with lock:
                detector = getattr(self, "_preview_detector", None)
                lang = getattr(self.item.config, "detection_lang", "en")
                if (
                    detector is None
                    or getattr(
                        self, "_preview_detector_lang", None) != lang
                ):
                    detector = SubtitleDetector(lang=lang)
                    self._preview_detector = detector
                    self._preview_detector_lang = lang
            boxes = detector.detect(
                frame, getattr(self.item.config, "detection_threshold", 0.5))
            for x1, y1, x2, y2 in boxes:
                self.cv2.rectangle(base, (x1, y1), (x2, y2), 255, -1)
            dilation = max(0, int(getattr(self.item.config, "mask_dilate_px", 0)))
            if dilation and np.any(base):
                kernel = self.cv2.getStructuringElement(
                    self.cv2.MORPH_ELLIPSE,
                    (dilation * 2 + 1, dilation * 2 + 1))
                base = self.cv2.dilate(base, kernel)
            return base, getattr(detector, "_engine_name", "detector")
        except Exception:
            logger.warning(
                "Mask-correction detection preview failed", exc_info=True)
            return base, "unavailable detection"

    def load_frame(self, index):
        index = max(0, min(self.frame_count - 1, int(index)))
        if self.is_video:
            self.cap.set(self.cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = self.cap.read()
            if not ok or frame is None:
                self.status_var.set(tr("Could not read the selected review frame"))
                return
        else:
            frame = self.still.copy()
        self.state["frame"] = frame
        self.state["frame_index"] = index
        self.state["base_mask"] = np.zeros(frame.shape[:2], dtype=np.uint8)
        self.state["request"] += 1
        request_id = self.state["request"]
        self.render()
        self.status_var.set(tr("Loading detected mask for frame {frame}...").format(
            frame=index))

        threading.Thread(
            target=partial(
                self._load_frame_worker, frame.copy(), index, request_id,
            ),
            daemon=True,
        ).start()

    def span_changed(self, _event=None):
        span = self.selected_span()
        suggested = str(span.get("suggested_mode") or "add")
        self.mode_var.set(next(
            (label for label, value in self.mode_labels.items()
             if value == suggested),
            next(iter(self.mode_labels))))
        self.set_span_range()
        self.load_frame(span["start_frame"])

    def prepare_rerun(self):
        normalized = normalize_mask_correction_list(self.state["corrections"]) or []
        if normalized == self.original:
            self.status_var.set(
                tr("Paint or clear a correction before preparing a rerun"))
            return False
        ranges = []
        for correction in normalized:
            start = correction.get("start_frame")
            end = correction.get("end_frame")
            if start is None:
                start = round(float(correction.get("start", 0.0)) * self.fps)
            if end is None:
                end_s = float(correction.get("end", 0.0))
                end = round(end_s * self.fps) if end_s > 0 else self.frame_count
            ranges.append((max(0, int(start)), min(self.frame_count, int(end))))
        ranges = merge_frame_ranges(ranges)
        if not ranges:
            self.status_var.set(tr("No valid correction frame ranges are available"))
            return False
        prior_output = Path(self.item.output_path)
        self.item.config.manual_mask_corrections = normalized or None
        self.item.config.quality_report = True
        self.item.correction_retry = ({
            "schema": SELECTIVE_RERUN_SCHEMA,
            "source_output": str(prior_output),
            "ranges": [list(frame_range) for frame_range in ranges],
        } if prior_output.is_file() else None)
        self.item.retry_config = {
            "schema": "vsr.retry_config.v1",
            "source": "mask_correction",
            "changes": {"manual_mask_corrections": {
                "before": self.original,
                "after": normalized,
            }},
            "affectedFrameRanges": [list(frame_range) for frame_range in ranges],
        }
        self.item.status = ProcessingStatus.IDLE
        self.item.progress = 0.0
        self.item.message = (
            tr("Ready to rerun corrected frames")
            if self.item.correction_retry
            else tr("Ready for a full rerun with mask corrections"))
        self.item.error = None
        self.item.quality_report = None
        self.item.started_at = None
        self.item.completed_at = None
        self._set_selected_queue_item(self.item.id)
        self._update_queue_display()
        if self.item.id in self.queue_widgets:
            self.queue_widgets[self.item.id].update_item(self.item)
        save_queue_state(self.queue)
        self._update_status(
            tr("Prepared mask-correction rerun for {count} frame range(s)").format(
                count=len(ranges)),
            "success", toast=True)
        self.win.destroy()
        return True

    def release_capture(self, event):
        if event.widget is self.win and self.cap is not None:
            self.cap.release()

    def _load_frame_worker(self, frame, index, request_id):
        result = self.detect_mask(frame, index)
        try:
            self.root.after(
                0, self._apply_frame_mask, request_id, index, result,
            )
        except (RuntimeError, tk.TclError):
            pass

    def _apply_frame_mask(self, request_id, index, result):
        if request_id != self.state["request"] or not self.win.winfo_exists():
            return
        self.state["base_mask"], source = result
        self.render()
        self.status_var.set(
            tr("Frame {frame}; base mask from {source}").format(
                frame=index, source=source,
            )
        )



class MaskCorrectionControllerMixin:
    """Focused correction behavior mixed into the composed GUI host."""

    @staticmethod
    def _mask_review_spans_for_item(item: QueueItem) -> list[dict]:
        report = item.quality_report if isinstance(item.quality_report, dict) else {}
        spans = []
        for raw in report.get("mask_review_spans") or []:
            if not isinstance(raw, dict):
                continue
            try:
                start = max(0, int(raw.get("start_frame", 0)))
                end = max(start + 1, int(raw.get("end_frame", start + 1)))
            except (TypeError, ValueError):
                continue
            spans.append({**raw, "start_frame": start, "end_frame": end})
        return spans

    def _open_selected_mask_correction(self):
        item = self._get_selected_queue_item(fallback_to_first=True)
        if item is None:
            self._update_status(
                tr("Select a queue item before correcting its mask"), "warning")
            return
        self._open_mask_correction_editor(item)

    def _open_mask_correction_editor(
        self,
        item: QueueItem,
        initial_span: Optional[dict] = None,
    ) -> bool:
        return MaskCorrectionWindow(self, item, initial_span).open()
