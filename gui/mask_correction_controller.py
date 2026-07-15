"""Quality-directed add/subtract mask correction editor."""

from __future__ import annotations

import logging
import threading
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
        if self.is_processing:
            self._update_status(
                tr("Stop the active batch before correcting a mask"), "warning")
            return False
        if not PIL_AVAILABLE:
            self._update_status(
                tr("Install Pillow to enable mask correction"), "warning")
            return False
        cap = None
        try:
            import cv2

            is_video = is_video_file(item.file_path)
            if is_video:
                cap = cv2.VideoCapture(item.file_path)
                if not cap.isOpened():
                    raise ValueError(tr("The source video could not be opened"))
                frame_count = max(1, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
                fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
                fps = fps if np.isfinite(fps) and fps > 0 else 30.0
                width = max(1, int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
                height = max(1, int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
                still = None
            else:
                still = safe_imread(item.file_path)
                if still is None:
                    raise ValueError(tr("The source image could not be opened"))
                height, width = still.shape[:2]
                frame_count, fps = 1, 1.0

            spans = self._mask_review_spans_for_item(item) or [{
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
                    index for index, span in enumerate(spans)
                    if (str(span.get("kind") or ""), span["start_frame"]) == target
                ), 0)

            original = normalize_mask_correction_list(
                item.config.manual_mask_corrections) or []
            state = {
                "corrections": list(original),
                "frame": None,
                "frame_index": 0,
                "base_mask": np.zeros((height, width), dtype=np.uint8),
                "active": None,
                "last_point": None,
                "request": 0,
            }
            history = RegionEditHistory(limit=100)
            screen_w = self.root.winfo_screenwidth()
            screen_h = self.root.winfo_screenheight()
            scale = min(
                min(900, int(screen_w * 0.78)) / width,
                min(430, int(screen_h * 0.52)) / height,
                1.0,
            )
            display_w = max(1, int(width * scale))
            display_h = max(1, int(height * scale))

            win = tk.Toplevel(self.root)
            win.title(tr("Correct subtitle mask"))
            win.configure(bg=Theme.BG_DARK)
            win.resizable(False, False)
            win.transient(self.root)

            header = tk.Frame(win, bg=Theme.BG_SECONDARY)
            header.pack(fill="x", padx=Theme.S_MD, pady=(Theme.S_MD, Theme.S_SM))
            span_labels = [
                tr("{kind}: frames {start}-{end}").format(
                    kind=str(span.get("kind", "review")).replace("-", " ").title(),
                    start=span["start_frame"],
                    end=max(span["start_frame"], span["end_frame"] - 1),
                )
                for span in spans
            ]
            span_var = tk.StringVar(value=span_labels[selected_span_index])
            span_picker = ttk.Combobox(
                header, state="readonly", width=31, textvariable=span_var,
                values=span_labels, style="Dark.TCombobox")
            span_picker.pack(side="left")
            set_accessible_metadata(
                span_picker, role="combo box", label=tr("Quality review span"),
                description=tr("Choose a residual, flicker, or low-confidence span."))

            mode_labels = {
                tr("Add to mask"): "add",
                tr("Subtract from mask"): "subtract",
            }
            mode_var = tk.StringVar(value=next(iter(mode_labels)))
            mode_picker = ttk.Combobox(
                header, state="readonly", width=18, textvariable=mode_var,
                values=tuple(mode_labels), style="Dark.TCombobox")
            mode_picker.pack(side="left", padx=(Theme.S_SM, 0))
            set_accessible_metadata(
                mode_picker, role="combo box", label=tr("Correction paint mode"),
                description=tr("Add missing pixels or subtract over-masked pixels."))

            controls = tk.Frame(win, bg=Theme.BG_DARK)
            controls.pack(fill="x", padx=Theme.S_MD, pady=(0, Theme.S_SM))
            propagate_var = tk.BooleanVar(value=False)
            propagate = tk.Checkbutton(
                controls, text=tr("Propagate through review span"),
                variable=propagate_var, bg=Theme.BG_DARK, fg=Theme.TEXT_PRIMARY,
                activebackground=Theme.BG_DARK, activeforeground=Theme.TEXT_PRIMARY,
                selectcolor=Theme.BG_TERTIARY, highlightthickness=0, takefocus=True)
            propagate.pack(side="left")
            start_var = tk.StringVar()
            end_var = tk.StringVar()
            frame_entries = []
            for label, variable in ((tr("Start frame"), start_var),
                                    (tr("End frame"), end_var)):
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
            brush_var = tk.IntVar(value=12)
            tk.Label(
                controls, text=tr("Brush"), font=f(Theme.F_META),
                bg=Theme.BG_DARK, fg=Theme.TEXT_MUTED).pack(
                    side="left", padx=(Theme.S_MD, Theme.S_XS))
            brush = tk.Scale(
                controls, from_=2, to=80, orient="horizontal", length=130,
                variable=brush_var, bg=Theme.BG_DARK, fg=Theme.TEXT_PRIMARY,
                troughcolor=Theme.BG_TERTIARY,
                activebackground=Theme.BLUE_PRIMARY, highlightthickness=0)
            brush.pack(side="left")
            set_accessible_metadata(
                brush, role="slider", label=tr("Correction brush radius"),
                description=tr("Brush radius in source pixels."))

            status_var = tk.StringVar(value=tr("Loading review frame..."))
            status = tk.Label(
                win, textvariable=status_var, font=f(Theme.F_META),
                bg=Theme.BG_DARK, fg=Theme.TEXT_MUTED, anchor="w")
            status.pack(fill="x", padx=Theme.S_MD, pady=(0, Theme.S_XS))
            set_accessible_metadata(
                status, role="status", label=tr("Mask correction status"))

            canvas = tk.Canvas(
                win, width=display_w, height=display_h, bg=Theme.BG_TERTIARY,
                cursor="crosshair", takefocus=True, highlightthickness=2,
                highlightbackground=Theme.BORDER_SUBTLE,
                highlightcolor=Theme.BLUE_PRIMARY)
            canvas.pack(padx=Theme.S_MD)
            image_id = canvas.create_image(0, 0, anchor="nw")
            canvas._photo = None
            set_accessible_metadata(
                canvas, role="mask painting canvas",
                label=tr("Frame-local mask correction canvas"),
                description=tr("Drag to paint with the selected mode."))

            actions = tk.Frame(win, bg=Theme.BG_DARK)
            actions.pack(fill="x", padx=Theme.S_MD, pady=Theme.S_MD)
            buttons = {"undo": None, "redo": None}

            def selected_span():
                try:
                    index = span_labels.index(span_var.get())
                except ValueError:
                    index = 0
                return spans[index]

            def set_span_range():
                span = selected_span()
                anchor = min(frame_count - 1, span["start_frame"])
                start_var.set(str(anchor))
                end_var.set(str(
                    min(frame_count, span["end_frame"])
                    if propagate_var.get() else min(frame_count, anchor + 1)))

            def parse_range():
                try:
                    start, end = int(start_var.get()), int(end_var.get())
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        tr("Start and end frames must be whole numbers")) from exc
                if start < 0 or end <= start or end > frame_count:
                    raise ValueError(
                        tr("Correction frames must form a non-empty in-bounds range"))
                return start, end

            def render():
                frame = state["frame"]
                if frame is None:
                    return
                frame_index = state["frame_index"]
                final = apply_mask_corrections(
                    state["base_mask"].copy(), state["corrections"],
                    frame_index / max(fps, 1e-9), frame_index) > 0
                base = state["base_mask"] > 0
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32)
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
                image = image.resize((display_w, display_h), Image.LANCZOS)
                canvas._photo = ImageTk.PhotoImage(image)
                canvas.itemconfig(image_id, image=canvas._photo)

            def update_history_buttons():
                buttons["undo"].set_enabled(
                    history.can_undo, tr("No correction to undo"))
                buttons["redo"].set_enabled(
                    history.can_redo, tr("No correction to redo"))

            def record_history():
                history.record(state["corrections"])
                update_history_buttons()

            def undo(event=None):
                if event is not None and isinstance(event.widget, tk.Entry):
                    return None
                restored = history.undo(state["corrections"])
                if restored is not None:
                    state["corrections"] = restored
                    render()
                    status_var.set(tr("Undid mask correction"))
                update_history_buttons()
                return "break"

            def redo(event=None):
                if event is not None and isinstance(event.widget, tk.Entry):
                    return None
                restored = history.redo(state["corrections"])
                if restored is not None:
                    state["corrections"] = restored
                    render()
                    status_var.set(tr("Redid mask correction"))
                update_history_buttons()
                return "break"

            def source_point(event):
                return (
                    max(0, min(width - 1, int(round(event.x / max(scale, 1e-9))))),
                    max(0, min(height - 1, int(round(event.y / max(scale, 1e-9))))),
                )

            def add_dab(point):
                operation = state["active"]
                if operation is not None:
                    operation["polygons"].append(brush_polygon(
                        point[0], point[1], brush_var.get(), width, height))
                    state["last_point"] = point
                    render()

            def paint_press(event):
                try:
                    start, end = parse_range()
                except ValueError as exc:
                    status_var.set(str(exc))
                    return "break"
                record_history()
                span = selected_span()
                operation = {
                    "schema": MASK_CORRECTION_SCHEMA,
                    "mode": mode_labels.get(mode_var.get(), "add"),
                    "polygons": [],
                    "start": start / max(fps, 1e-9),
                    "end": end / max(fps, 1e-9),
                    "start_frame": start,
                    "end_frame": end,
                    "propagation": "span" if end - start > 1 else "frame",
                    "source": span.get("kind", "manual"),
                }
                state["corrections"].append(operation)
                state["active"] = operation
                add_dab(source_point(event))
                status_var.set(tr("Painting mask correction"))
                return "break"

            def paint_drag(event):
                if state["active"] is None:
                    return None
                point = source_point(event)
                last = state["last_point"]
                minimum = max(1.0, brush_var.get() * 0.45)
                if last is None or np.hypot(point[0] - last[0], point[1] - last[1]) >= minimum:
                    add_dab(point)
                return "break"

            def paint_release(_event):
                state["active"] = None
                state["last_point"] = None
                state["corrections"] = (
                    normalize_mask_correction_list(state["corrections"]) or [])
                render()
                status_var.set(tr("Correction stroke added; undo is available"))
                return "break"

            def clear_corrections():
                if state["corrections"]:
                    record_history()
                    state["corrections"] = []
                    render()
                    status_var.set(tr("Cleared mask corrections"))

            def detect_mask(frame, frame_index):
                base = np.zeros(frame.shape[:2], dtype=np.uint8)
                mask_path = str((getattr(item, "mask_export", None) or {}).get("path") or "")
                if mask_path and Path(mask_path).is_file():
                    mask_cap = cv2.VideoCapture(mask_path)
                    try:
                        mask_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                        ok, mask_frame = mask_cap.read()
                    finally:
                        mask_cap.release()
                    if ok and mask_frame is not None:
                        gray = cv2.cvtColor(mask_frame, cv2.COLOR_BGR2GRAY)
                        return np.where(gray > 127, 255, 0).astype(np.uint8), "exported mask"
                try:
                    from backend.processor import SubtitleDetector

                    lock = getattr(self, "_detector_lock", threading.Lock())
                    with lock:
                        detector = getattr(self, "_preview_detector", None)
                        lang = getattr(item.config, "detection_lang", "en")
                        if (
                            detector is None
                            or getattr(
                                self, "_preview_detector_lang", None) != lang
                        ):
                            detector = SubtitleDetector(lang=lang)
                            self._preview_detector = detector
                            self._preview_detector_lang = lang
                    boxes = detector.detect(
                        frame, getattr(item.config, "detection_threshold", 0.5))
                    for x1, y1, x2, y2 in boxes:
                        cv2.rectangle(base, (x1, y1), (x2, y2), 255, -1)
                    dilation = max(0, int(getattr(item.config, "mask_dilate_px", 0)))
                    if dilation and np.any(base):
                        kernel = cv2.getStructuringElement(
                            cv2.MORPH_ELLIPSE,
                            (dilation * 2 + 1, dilation * 2 + 1))
                        base = cv2.dilate(base, kernel)
                    return base, getattr(detector, "_engine_name", "detector")
                except Exception:
                    logger.warning(
                        "Mask-correction detection preview failed", exc_info=True)
                    return base, "unavailable detection"

            def load_frame(index):
                index = max(0, min(frame_count - 1, int(index)))
                if is_video:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, index)
                    ok, frame = cap.read()
                    if not ok or frame is None:
                        status_var.set(tr("Could not read the selected review frame"))
                        return
                else:
                    frame = still.copy()
                state["frame"] = frame
                state["frame_index"] = index
                state["base_mask"] = np.zeros(frame.shape[:2], dtype=np.uint8)
                state["request"] += 1
                request_id = state["request"]
                render()
                status_var.set(tr("Loading detected mask for frame {frame}...").format(
                    frame=index))

                def worker():
                    result = detect_mask(frame.copy(), index)

                    def apply_result():
                        if request_id != state["request"] or not win.winfo_exists():
                            return
                        state["base_mask"], source = result
                        render()
                        status_var.set(tr("Frame {frame}; base mask from {source}").format(
                            frame=index, source=source))

                    try:
                        self.root.after(0, apply_result)
                    except (RuntimeError, tk.TclError):
                        pass

                threading.Thread(target=worker, daemon=True).start()

            def span_changed(_event=None):
                span = selected_span()
                suggested = str(span.get("suggested_mode") or "add")
                mode_var.set(next(
                    (label for label, value in mode_labels.items()
                     if value == suggested),
                    next(iter(mode_labels))))
                set_span_range()
                load_frame(span["start_frame"])

            def prepare_rerun():
                normalized = normalize_mask_correction_list(state["corrections"]) or []
                if normalized == original:
                    status_var.set(
                        tr("Paint or clear a correction before preparing a rerun"))
                    return False
                ranges = []
                for correction in normalized:
                    start = correction.get("start_frame")
                    end = correction.get("end_frame")
                    if start is None:
                        start = round(float(correction.get("start", 0.0)) * fps)
                    if end is None:
                        end_s = float(correction.get("end", 0.0))
                        end = round(end_s * fps) if end_s > 0 else frame_count
                    ranges.append((max(0, int(start)), min(frame_count, int(end))))
                ranges = merge_frame_ranges(ranges)
                if not ranges:
                    status_var.set(tr("No valid correction frame ranges are available"))
                    return False
                prior_output = Path(item.output_path)
                item.config.manual_mask_corrections = normalized or None
                item.config.quality_report = True
                item.correction_retry = ({
                    "schema": SELECTIVE_RERUN_SCHEMA,
                    "source_output": str(prior_output),
                    "ranges": [list(frame_range) for frame_range in ranges],
                } if prior_output.is_file() else None)
                item.retry_config = {
                    "schema": "vsr.retry_config.v1",
                    "source": "mask_correction",
                    "changes": {"manual_mask_corrections": {
                        "before": original,
                        "after": normalized,
                    }},
                    "affectedFrameRanges": [list(frame_range) for frame_range in ranges],
                }
                item.status = ProcessingStatus.IDLE
                item.progress = 0.0
                item.message = (
                    tr("Ready to rerun corrected frames")
                    if item.correction_retry
                    else tr("Ready for a full rerun with mask corrections"))
                item.error = None
                item.quality_report = None
                item.started_at = None
                item.completed_at = None
                self._set_selected_queue_item(item.id)
                self._update_queue_display()
                if item.id in self.queue_widgets:
                    self.queue_widgets[item.id].update_item(item)
                save_queue_state(self.queue)
                self._update_status(
                    tr("Prepared mask-correction rerun for {count} frame range(s)").format(
                        count=len(ranges)),
                    "success", toast=True)
                win.destroy()
                return True

            undo_button = ModernButton(
                actions, text=tr("Undo"), width=76, command=undo,
                style="ghost", size="sm")
            undo_button.pack(side="left")
            redo_button = ModernButton(
                actions, text=tr("Redo"), width=76, command=redo,
                style="ghost", size="sm")
            redo_button.pack(side="left", padx=(Theme.S_XS, 0))
            buttons.update(undo=undo_button, redo=redo_button)
            ModernButton(
                actions, text=tr("Clear corrections"), width=132,
                command=clear_corrections, style="ghost", size="sm").pack(
                    side="left", padx=(Theme.S_SM, 0))
            ModernButton(
                actions, text=tr("Prepare selective rerun"), width=184,
                command=prepare_rerun, style="primary", size="sm").pack(side="right")
            ModernButton(
                actions, text=tr("Cancel"), width=82,
                command=win.destroy, style="ghost", size="sm").pack(
                    side="right", padx=(0, Theme.S_SM))

            propagate.configure(command=set_span_range)
            span_picker.bind("<<ComboboxSelected>>", span_changed)
            canvas.bind("<ButtonPress-1>", paint_press)
            canvas.bind("<B1-Motion>", paint_drag)
            canvas.bind("<ButtonRelease-1>", paint_release)
            win.bind("<Control-z>", undo)
            win.bind("<Control-y>", redo)
            win.bind("<Control-Shift-Z>", redo)
            win.bind("<Escape>", lambda _event: win.destroy())

            def release_capture(event):
                if event.widget is win and cap is not None:
                    cap.release()

            win.bind("<Destroy>", release_capture)
            win._vsr_span_picker = span_picker
            win._vsr_mode_picker = mode_picker
            win._vsr_frame_entries = frame_entries
            win._vsr_correction_canvas = canvas
            win._vsr_correction_state = state
            win._vsr_paint_handlers = (paint_press, paint_drag, paint_release)
            win._vsr_undo_correction = undo
            win._vsr_redo_correction = redo
            win._vsr_prepare_selective_rerun = prepare_rerun
            update_history_buttons()
            span_changed()
            win.grab_set()
            return True
        except Exception as exc:
            if cap is not None:
                cap.release()
            logger.warning("Mask correction editor failed", exc_info=True)
            self._update_status(
                tr("Mask correction editor could not open: {error}").format(error=exc),
                "warning")
            return False
