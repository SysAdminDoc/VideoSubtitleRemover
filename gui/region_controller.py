from __future__ import annotations

import math
import logging
from pathlib import Path
from typing import Any, List, Protocol, Tuple

try:
    import tkinter as tk
    from tkinter import filedialog, ttk
except ImportError:  # pragma: no cover - tkinter is optional for headless imports
    pass

try:
    from PIL import Image, ImageTk
    PIL_AVAILABLE = True
except ImportError:  # pragma: no cover - Pillow is optional for headless imports
    PIL_AVAILABLE = False

from backend.a11y import set_accessible_metadata
from backend.i18n import tr
from backend.region_editing import (
    RegionEditHistory,
    format_polygon_vertices,
    frame_to_seconds,
    parse_polygon_vertices,
    rect_from_xywh,
    seconds_to_frame,
    transform_region_shape,
)
from backend.region_keyframes import (
    normalize_region_keyframe_tracks,
    region_shapes_at,
)
from backend.safe_image import safe_imread
from gui.config import (
    ProcessingConfig,
    _coerce_region_span_list,
)
from gui.theme import Theme, f
from gui.utils import SUPPORTED_EXTENSIONS, filepicker_pattern, is_video_file
from gui.widgets import (
    ModernButton,
    ModernToggle,
)

logger = logging.getLogger(__name__)


class RegionControllerHost(Protocol):
    """Host surface required by RegionEditorControllerMixin."""

    root: Any
    config: ProcessingConfig

    def _update_status(self, message: str, tone: str = "neutral", toast: bool = False): ...


class RegionEditorControllerMixin:
    """Focused behavior mixed into the composed GUI host."""

    def _open_region_selector(self):
        """Start direct preview-pane region editing, with modal fallback."""
        selected = self._get_selected_queue_item(fallback_to_first=True)
        if selected:
            self._set_selected_queue_item(selected.id)
            if self._start_preview_region_editor(selected):
                return
        self._open_region_selector_modal()

    def _open_region_selector_modal(self):
        """Open a region-selector window with frame scrubbing (F-1) and
        multi-rectangle drawing (F-2).

        Drag = primary rect (or new rect when "Add another" was clicked).
        The frame slider re-loads the chosen frame so users can target a
        non-zero timecode for clips that open on a black intro card. The
        backend accepts `subtitle_areas` for global regions and
        `subtitle_region_spans` for optional start/end time windows. Legacy
        saves still write the first rect to `subtitle_area` for compatibility.
        """
        source_path = None
        selected = self._get_selected_queue_item(fallback_to_first=True)
        if selected:
            self._set_selected_queue_item(selected.id)
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
            self._update_status(
                "Install Pillow to enable the visual region selector",
                "warning",
            )
            return

        import cv2 as _cv2

        is_video = is_video_file(source_path)
        cap = _cv2.VideoCapture(source_path) if is_video else None
        if is_video and not cap.isOpened():
            logger.error("Could not open video for region selection")
            cap.release()
            self._update_status(
                "The selected video could not be opened for region selection",
                "warning",
            )
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
                    self._update_status(
                        "The selected video did not provide a readable frame",
                        "warning",
                    )
                    return
            else:
                frame = safe_imread(source_path)
                if frame is None:
                    logger.error("Could not read image for region selection")
                    self._update_status(
                        "The selected image could not be read for region selection",
                        "warning",
                    )
                    return
                frame_count, fps = 1, 30.0

            orig_h, orig_w = frame.shape[:2]
            screen_w = self.root.winfo_screenwidth()
            screen_h = self.root.winfo_screenheight()
            max_w = min(900, int(screen_w * 0.8))
            height_ratio = 0.55 if screen_h >= 600 else 0.7
            max_h = min(540, int(screen_h * height_ratio))
            scale = min(max_w / orig_w, max_h / orig_h, 1.0)
            disp_w, disp_h = int(orig_w * scale), int(orig_h * scale)

            # Selector state: every saved rect lives in `rects` (image-
            # space coordinates). Drag creates the current rect; on
            # release it joins the list. Adding another rect re-arms the
            # canvas for the next drag.
            rects: List[Tuple[int, int, int, int]] = []
            region_spans = _coerce_region_span_list(
                getattr(self.config, "subtitle_region_spans", None)) or []
            keyframe_tracks = normalize_region_keyframe_tracks(
                getattr(self.config, "subtitle_region_keyframes", None)) or []
            pending_keyframes: List[dict] = []
            polygon_shapes: List[List[int]] = []
            polygon_points: List[Tuple[int, int]] = []
            current_frame_index = [0]
            current_frame = [frame]
            preload = []
            if not region_spans and not keyframe_tracks:
                preload = self.config.subtitle_areas or (
                    [self.config.subtitle_area] if self.config.subtitle_area else []
                )
            rects.extend([tuple(r) for r in preload if r])

            win = tk.Toplevel(self.root)
            win.title("Choose subtitle region")
            win.configure(bg=Theme.BG_OVERLAY)
            win.resizable(False, False)

            canvas = tk.Canvas(
                win,
                width=disp_w,
                height=disp_h,
                highlightthickness=2,
                highlightbackground=Theme.BORDER_SUBTLE,
                highlightcolor=Theme.BLUE_PRIMARY,
                bg=Theme.BG_DARK,
                cursor="cross",
                takefocus=True,
            )
            canvas.pack()
            set_accessible_metadata(
                canvas,
                role="region editor canvas",
                label=tr("Subtitle region preview"),
                description=tr(
                    "Draw with the pointer, or use arrow keys to edit the selected region."
                ),
            )
            canvas_image_id = canvas.create_image(0, 0, anchor="nw")
            canvas._photo = None

            rect_ids: List[int] = []
            drag_id = [None]
            drag_start = [0, 0]
            shape_mode_var = tk.StringVar(value=tr("Rectangle"))

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
                for span in region_spans:
                    x1, y1, x2, y2 = span["rect"]
                    rect_ids.append(canvas.create_rectangle(
                        x1 * scale, y1 * scale, x2 * scale, y2 * scale,
                        outline=Theme.WARNING, width=2,
                        stipple="gray50", fill=Theme.WARNING,
                    ))
                for (x1, y1, x2, y2) in rects:
                    rect_ids.append(canvas.create_rectangle(
                        x1 * scale, y1 * scale, x2 * scale, y2 * scale,
                        outline=Theme.GREEN_PRIMARY, width=2,
                        stipple="gray25", fill=Theme.GREEN_PRIMARY,
                    ))
                current_seconds = current_frame_index[0] / max(fps, 1.0)
                for shape in region_shapes_at(keyframe_tracks, current_seconds):
                    if "rect" in shape:
                        x1, y1, x2, y2 = shape["rect"]
                        rect_ids.append(canvas.create_rectangle(
                            x1 * scale, y1 * scale, x2 * scale, y2 * scale,
                            outline=Theme.BLUE_PRIMARY, width=3,
                        ))
                    elif "polygon" in shape:
                        coords = shape["polygon"]
                        points = [coord * scale for coord in coords]
                        rect_ids.append(canvas.create_polygon(
                            *points, outline=Theme.BLUE_PRIMARY,
                            fill="", width=3,
                        ))
                for polygon in polygon_shapes:
                    points = [coord * scale for coord in polygon]
                    rect_ids.append(canvas.create_polygon(
                        *points, outline=Theme.GREEN_PRIMARY,
                        fill="", width=2,
                    ))
                if polygon_points:
                    points = []
                    for px, py in polygon_points:
                        points.extend((px * scale, py * scale))
                    if len(points) >= 4:
                        rect_ids.append(canvas.create_line(
                            *points, fill=Theme.BLUE_PRIMARY, width=2))
                    for px, py in polygon_points:
                        radius = 3
                        rect_ids.append(canvas.create_oval(
                            px * scale - radius, py * scale - radius,
                            px * scale + radius, py * scale + radius,
                            fill=Theme.BLUE_PRIMARY, outline=""))

            def _seek_video(frame_idx: int):
                if not is_video or cap is None:
                    return
                frame_idx = max(0, min(frame_count - 1, int(frame_idx)))
                current_frame_index[0] = frame_idx
                cap.set(_cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ok, f = cap.read()
                if ok:
                    current_frame[0] = f
                    _draw_image(f)
                    _draw_saved_rects()

            _draw_image(frame)
            _draw_saved_rects()

            # Drawing handlers.
            def on_press(event):
                if shape_mode_var.get() == tr("Polygon"):
                    _record_history()
                    px = min(orig_w, max(0, int(event.x / scale)))
                    py = min(orig_h, max(0, int(event.y / scale)))
                    polygon_points.append((px, py))
                    _draw_saved_rects()
                    return
                drag_start[0], drag_start[1] = event.x, event.y
                if drag_id[0]:
                    canvas.delete(drag_id[0])
                drag_id[0] = canvas.create_rectangle(
                    event.x, event.y, event.x, event.y,
                    outline=Theme.BLUE_PRIMARY, width=2,
                    stipple="gray12", fill=Theme.BLUE_PRIMARY,
                )

            def on_drag(event):
                if shape_mode_var.get() == tr("Polygon"):
                    return
                if drag_id[0]:
                    canvas.coords(drag_id[0], drag_start[0], drag_start[1], event.x, event.y)

            def on_release(event):
                if shape_mode_var.get() == tr("Polygon"):
                    return
                x1 = int(min(drag_start[0], event.x) / scale)
                y1 = int(min(drag_start[1], event.y) / scale)
                x2 = int(max(drag_start[0], event.x) / scale)
                y2 = int(max(drag_start[1], event.y) / scale)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(orig_w, x2), min(orig_h, y2)
                if (x2 - x1) > 10 and (y2 - y1) > 5:
                    _record_history()
                    rects.append((x1, y1, x2, y2))
                    _refresh_region_editor(f"rect:{len(rects) - 1}")
                if drag_id[0] is not None:
                    canvas.delete(drag_id[0])
                    drag_id[0] = None
                _draw_saved_rects()

            canvas.bind("<ButtonPress-1>", on_press)
            canvas.bind("<B1-Motion>", on_drag)
            canvas.bind("<ButtonRelease-1>", on_release)
            canvas._vsr_region_drag_handlers = (on_press, on_drag, on_release)

            shape_row = tk.Frame(win, bg=Theme.BG_OVERLAY)
            shape_row.pack(fill="x", padx=Theme.S_MD, pady=(Theme.S_SM, 0))
            tk.Label(shape_row, text=tr("Shape"), font=f(Theme.F_META),
                     bg=Theme.BG_OVERLAY, fg=Theme.TEXT_MUTED).pack(side="left")
            shape_picker = ttk.Combobox(
                shape_row,
                width=12,
                state="readonly",
                textvariable=shape_mode_var,
                values=(tr("Rectangle"), tr("Polygon")),
                style="Dark.TCombobox",
            )
            shape_picker.pack(side="left", padx=(Theme.S_XS, Theme.S_MD))
            shape_hint_var = tk.StringVar(
                value=tr("Drag a rectangle, or choose Polygon and click vertices."))
            tk.Label(shape_row, textvariable=shape_hint_var,
                     font=f(Theme.F_META), bg=Theme.BG_OVERLAY,
                     fg=Theme.TEXT_MUTED).pack(side="left")

            def _finish_polygon() -> bool:
                if len(polygon_points) < 3:
                    self._update_status(
                        "A polygon needs at least three vertices", "warning")
                    return False
                _record_history()
                coords = []
                for px, py in polygon_points:
                    coords.extend((px, py))
                polygon_shapes.append(coords)
                polygon_points.clear()
                _draw_saved_rects()
                _refresh_region_editor(f"polygon:{len(polygon_shapes) - 1}")
                return True

            finish_polygon_btn = ModernButton(
                shape_row, text=tr("Finish polygon"), command=_finish_polygon,
                style="secondary", size="sm", width=112)
            finish_polygon_btn.pack(side="right")

            # Frame slider for videos.
            if is_video and frame_count > 1:
                slider_row = tk.Frame(win, bg=Theme.BG_OVERLAY)
                slider_row.pack(fill="x", padx=Theme.S_MD, pady=(Theme.S_SM, 0))
                tk.Label(slider_row, text=tr("Frame"),
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
                    hh = int(secs // 3600)
                    mm = int((secs % 3600) // 60)
                    ss = int(secs % 60)
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

            time_row = None
            start_var = tk.StringVar()
            end_var = tk.StringVar()
            start_frame_var = tk.StringVar()
            end_frame_var = tk.StringVar()
            span_summary_var = tk.StringVar()
            if is_video:
                existing_starts = {round(float(s.get("start", 0.0)), 3)
                                   for s in region_spans}
                existing_ends = {round(float(s.get("end", 0.0)), 3)
                                 for s in region_spans}
                if len(existing_starts) == 1 and len(existing_ends) == 1:
                    start_value = next(iter(existing_starts))
                    end_value = next(iter(existing_ends))
                    start_var.set(f"{start_value:g}" if start_value else "")
                    end_var.set(f"{end_value:g}" if end_value else "")

                time_row = tk.Frame(win, bg=Theme.BG_OVERLAY)
                time_row.pack(fill="x", padx=Theme.S_MD, pady=(0, Theme.S_SM))
                tk.Label(time_row, text=tr("Start sec"),
                         font=f(Theme.F_META),
                         bg=Theme.BG_OVERLAY, fg=Theme.TEXT_MUTED).pack(side="left")
                start_entry = tk.Entry(
                    time_row, width=9, textvariable=start_var,
                    bg=Theme.BG_TERTIARY, fg=Theme.TEXT_PRIMARY,
                    insertbackground=Theme.TEXT_PRIMARY,
                    relief="flat",
                )
                start_entry.pack(side="left", padx=(Theme.S_XS, Theme.S_MD))
                tk.Label(time_row, text=tr("End sec"),
                         font=f(Theme.F_META),
                         bg=Theme.BG_OVERLAY, fg=Theme.TEXT_MUTED).pack(side="left")
                end_entry = tk.Entry(
                    time_row, width=9, textvariable=end_var,
                    bg=Theme.BG_TERTIARY, fg=Theme.TEXT_PRIMARY,
                    insertbackground=Theme.TEXT_PRIMARY,
                    relief="flat",
                )
                end_entry.pack(side="left", padx=(Theme.S_XS, Theme.S_MD))
                tk.Label(time_row, text=tr("Start frame"),
                         font=f(Theme.F_META),
                         bg=Theme.BG_OVERLAY, fg=Theme.TEXT_MUTED).pack(side="left")
                start_frame_entry = tk.Entry(
                    time_row, width=7, textvariable=start_frame_var,
                    bg=Theme.BG_TERTIARY, fg=Theme.TEXT_PRIMARY,
                    insertbackground=Theme.TEXT_PRIMARY,
                    relief="flat",
                )
                start_frame_entry.pack(side="left", padx=(Theme.S_XS, Theme.S_MD))
                tk.Label(time_row, text=tr("End frame"),
                         font=f(Theme.F_META),
                         bg=Theme.BG_OVERLAY, fg=Theme.TEXT_MUTED).pack(side="left")
                end_frame_entry = tk.Entry(
                    time_row, width=7, textvariable=end_frame_var,
                    bg=Theme.BG_TERTIARY, fg=Theme.TEXT_PRIMARY,
                    insertbackground=Theme.TEXT_PRIMARY,
                    relief="flat",
                )
                end_frame_entry.pack(side="left", padx=(Theme.S_XS, Theme.S_MD))
                span_summary_var.set(
                    (
                        f"{len(keyframe_tracks)} motion track"
                        f"{'s' if len(keyframe_tracks) != 1 else ''}"
                    ) if keyframe_tracks else (
                        f"{len(region_spans)} timed region"
                        f"{'s' if len(region_spans) != 1 else ''}"
                        if region_spans else tr("Optional")
                    )
                )
                span_label = tk.Label(
                    time_row, textvariable=span_summary_var,
                    font=f(Theme.F_META),
                    bg=Theme.BG_OVERLAY, fg=Theme.TEXT_MUTED,
                )
                span_label.pack(side="right")
                win._vsr_start_entry = start_entry
                win._vsr_end_entry = end_entry
                win._vsr_start_frame_entry = start_frame_entry
                win._vsr_end_frame_entry = end_frame_entry

                for entry, label, description in (
                    (start_entry, tr("Region start time in seconds"),
                     tr("Enter a nonnegative decimal time.")),
                    (end_entry, tr("Region end time in seconds"),
                     tr("Leave blank for the end of the video.")),
                    (start_frame_entry, tr("Region start frame"),
                     tr("Enter a nonnegative frame number.")),
                    (end_frame_entry, tr("Region end frame"),
                     tr("Leave blank for the end of the video.")),
                ):
                    set_accessible_metadata(
                        entry, role="numeric input", label=label,
                        description=description)

            precise_frame = tk.Frame(
                win,
                bg=Theme.BG_SECONDARY,
                highlightthickness=1,
                highlightbackground=Theme.BORDER_SUBTLE,
            )
            precise_frame.pack(
                fill="x", padx=Theme.S_MD, pady=(Theme.S_XS, Theme.S_SM))
            precise_top = tk.Frame(precise_frame, bg=Theme.BG_SECONDARY)
            precise_top.pack(fill="x", padx=Theme.S_SM, pady=(Theme.S_XS, 0))
            precise_bottom = tk.Frame(precise_frame, bg=Theme.BG_SECONDARY)
            precise_bottom.pack(fill="x", padx=Theme.S_SM, pady=(2, Theme.S_XS))

            selected_region_var = tk.StringVar()
            region_picker = ttk.Combobox(
                precise_top,
                width=21,
                state="readonly",
                textvariable=selected_region_var,
                style="Dark.TCombobox",
            )
            tk.Label(
                precise_top, text=tr("Selected"), font=f(Theme.F_META),
                bg=Theme.BG_SECONDARY, fg=Theme.TEXT_MUTED,
            ).pack(side="left")
            region_picker.pack(side="left", padx=(Theme.S_XS, Theme.S_MD))
            set_accessible_metadata(
                region_picker,
                role="combo box",
                label=tr("Selected manual region"),
                description=tr("Choose a rectangle, polygon, timed region, or keyframe to edit."),
            )

            geometry_vars = {
                "x": tk.StringVar(),
                "y": tk.StringVar(),
                "width": tk.StringVar(),
                "height": tk.StringVar(),
                "vertices": tk.StringVar(),
            }
            geometry_entries = {}
            for key, label, width in (
                ("x", tr("X"), 6),
                ("y", tr("Y"), 6),
                ("width", tr("Width"), 7),
                ("height", tr("Height"), 7),
            ):
                tk.Label(
                    precise_top, text=label, font=f(Theme.F_META),
                    bg=Theme.BG_SECONDARY, fg=Theme.TEXT_MUTED,
                ).pack(side="left")
                entry = tk.Entry(
                    precise_top, width=width, textvariable=geometry_vars[key],
                    bg=Theme.BG_TERTIARY, fg=Theme.TEXT_PRIMARY,
                    disabledbackground=Theme.BG_DARK,
                    disabledforeground=Theme.TEXT_DISABLED,
                    insertbackground=Theme.TEXT_PRIMARY, relief="flat",
                )
                entry.pack(side="left", padx=(Theme.S_XS, Theme.S_SM))
                geometry_entries[key] = entry
                set_accessible_metadata(
                    entry,
                    role="numeric input",
                    label=tr("Region {field} in pixels").format(field=label),
                    description=tr("Enter a whole number within the source-frame bounds."),
                )

            tk.Label(
                precise_bottom, text=tr("Vertices"), font=f(Theme.F_META),
                bg=Theme.BG_SECONDARY, fg=Theme.TEXT_MUTED,
            ).pack(side="left")
            vertices_entry = tk.Entry(
                precise_bottom, width=46,
                textvariable=geometry_vars["vertices"],
                bg=Theme.BG_TERTIARY, fg=Theme.TEXT_PRIMARY,
                disabledbackground=Theme.BG_DARK,
                disabledforeground=Theme.TEXT_DISABLED,
                insertbackground=Theme.TEXT_PRIMARY, relief="flat",
            )
            vertices_entry.pack(side="left", padx=(Theme.S_XS, Theme.S_MD), fill="x", expand=True)
            geometry_entries["vertices"] = vertices_entry
            set_accessible_metadata(
                vertices_entry,
                role="coordinate input",
                label=tr("Polygon vertex coordinates"),
                description=tr("Use x,y pairs separated by semicolons; at least three vertices are required."),
            )

            reference_frame = tk.Frame(
                win,
                bg=Theme.BG_SECONDARY,
                highlightthickness=1,
                highlightbackground=Theme.BORDER_SUBTLE,
            )
            reference_frame.pack(
                fill="x", padx=Theme.S_MD, pady=(0, Theme.S_SM))
            reference_path_var = tk.StringVar(value="")
            reference_status_var = tk.StringVar(
                value=tr("Select a timed region to attach a clean reference."))
            reference_alignment_var = tk.StringVar(value="Auto")
            reference_color_match_var = tk.BooleanVar(value=True)
            reference_confidence_var = tk.StringVar(value="0.75")
            reference_top = tk.Frame(reference_frame, bg=Theme.BG_SECONDARY)
            reference_top.pack(
                fill="x", padx=Theme.S_SM, pady=(Theme.S_XS, 0))
            tk.Label(
                reference_top, text=tr("Clean reference"),
                font=f(Theme.F_BODY_SM), bg=Theme.BG_SECONDARY,
                fg=Theme.TEXT_PRIMARY,
            ).pack(side="left")
            tk.Label(
                reference_top, textvariable=reference_path_var,
                font=f(Theme.F_META), bg=Theme.BG_SECONDARY,
                fg=Theme.TEXT_MUTED, anchor="w",
            ).pack(side="left", fill="x", expand=True, padx=Theme.S_SM)
            reference_buttons = {}
            for key, label, width, command in (
                ("choose", tr("Choose"), 76,
                 lambda: _choose_clean_reference()),
                ("preview", tr("Preview"), 78,
                 lambda: _preview_clean_reference()),
                ("clear", tr("Clear"), 68,
                 lambda: _clear_clean_reference()),
            ):
                button = ModernButton(
                    reference_top, text=label, width=width,
                    command=command, style="ghost", size="sm")
                button.pack(side="right", padx=(Theme.S_XS, 0))
                reference_buttons[key] = button
            reference_options = tk.Frame(
                reference_frame, bg=Theme.BG_SECONDARY)
            reference_options.pack(
                fill="x", padx=Theme.S_SM, pady=(Theme.S_XS, 0))
            tk.Label(
                reference_options, text=tr("Alignment"), font=f(Theme.F_META),
                bg=Theme.BG_SECONDARY, fg=Theme.TEXT_MUTED,
            ).pack(side="left")
            reference_alignment_picker = ttk.Combobox(
                reference_options, width=13, state="readonly",
                textvariable=reference_alignment_var,
                values=(tr("Auto"), tr("Translation"), tr("Homography")),
                style="Dark.TCombobox",
            )
            reference_alignment_picker.pack(
                side="left", padx=(Theme.S_XS, Theme.S_MD))
            reference_color_toggle = ModernToggle(
                reference_options,
                text=tr("Match each frame's color"),
                variable=reference_color_match_var,
                command=lambda: _save_clean_reference_options(),
                bg=Theme.BG_SECONDARY,
            )
            reference_color_toggle.pack(side="left")
            tk.Label(
                reference_options, text=tr("Confidence floor"),
                font=f(Theme.F_META), bg=Theme.BG_SECONDARY,
                fg=Theme.TEXT_MUTED,
            ).pack(side="left", padx=(Theme.S_MD, Theme.S_XS))
            reference_confidence_entry = tk.Entry(
                reference_options, width=6,
                textvariable=reference_confidence_var,
                bg=Theme.BG_TERTIARY, fg=Theme.TEXT_PRIMARY,
                insertbackground=Theme.TEXT_PRIMARY, relief="flat",
            )
            reference_confidence_entry.pack(side="left")
            tk.Label(
                reference_frame, textvariable=reference_status_var,
                font=f(Theme.F_META), bg=Theme.BG_SECONDARY,
                fg=Theme.TEXT_MUTED, anchor="w",
            ).pack(
                fill="x", padx=Theme.S_SM, pady=(Theme.S_XS, Theme.S_XS))
            set_accessible_metadata(
                reference_alignment_picker, role="combo box",
                label=tr("Clean reference alignment mode"))
            set_accessible_metadata(
                reference_confidence_entry, role="numeric input",
                label=tr("Clean reference minimum confidence"),
                description=tr("Values below this floor use normal inpainting."))

            selected_region_key = [None]
            history = RegionEditHistory()
            history_buttons = {"undo": None, "redo": None}
            time_input_source = {"start": "seconds", "end": "seconds"}

            def _editable_region_records():
                records = []
                for index, rect in enumerate(rects):
                    records.append({
                        "key": f"rect:{index}",
                        "label": tr("Rectangle {index}").format(index=index + 1),
                        "source": "rect", "index": index,
                        "shape": {"rect": list(rect)}, "timing": "none",
                    })
                for index, span in enumerate(region_spans):
                    records.append({
                        "key": f"span:{index}",
                        "label": tr("Timed region {index}").format(index=index + 1),
                        "source": "span", "index": index,
                        "shape": {"rect": list(span["rect"])},
                        "timing": "range",
                        "start": float(span.get("start", 0.0)),
                        "end": float(span.get("end", 0.0)),
                    })
                for index, polygon in enumerate(polygon_shapes):
                    records.append({
                        "key": f"polygon:{index}",
                        "label": tr("Polygon {index}").format(index=index + 1),
                        "source": "polygon", "index": index,
                        "shape": {"polygon": list(polygon)}, "timing": "none",
                    })
                for index, keyframe in enumerate(pending_keyframes):
                    kind = "rect" if "rect" in keyframe else "polygon"
                    records.append({
                        "key": f"pending:{index}",
                        "label": tr("Pending keyframe {index}").format(index=index + 1),
                        "source": "pending", "index": index,
                        "shape": {kind: list(keyframe[kind])}, "timing": "point",
                        "start": float(keyframe.get("time", 0.0)), "end": 0.0,
                    })
                for track_index, track in enumerate(keyframe_tracks):
                    for frame_index, keyframe in enumerate(track.get("keyframes", [])):
                        kind = "rect" if "rect" in keyframe else "polygon"
                        records.append({
                            "key": f"track:{track_index}:{frame_index}",
                            "label": tr("Motion {track}, keyframe {frame}").format(
                                track=track_index + 1, frame=frame_index + 1),
                            "source": "track", "track": track_index,
                            "index": frame_index,
                            "shape": {kind: list(keyframe[kind])}, "timing": "point",
                            "start": float(keyframe.get("time", 0.0)), "end": 0.0,
                        })
                return records

            def _editor_snapshot():
                return {
                    "rects": list(rects),
                    "region_spans": list(region_spans),
                    "keyframe_tracks": list(keyframe_tracks),
                    "pending_keyframes": list(pending_keyframes),
                    "polygon_shapes": list(polygon_shapes),
                    "polygon_points": list(polygon_points),
                    "start": start_var.get(),
                    "end": end_var.get(),
                    "start_frame": start_frame_var.get(),
                    "end_frame": end_frame_var.get(),
                    "selected": selected_region_key[0],
                }

            def _record_history():
                history.record(_editor_snapshot())
                _update_history_buttons()

            def _restore_editor_snapshot(snapshot):
                rects[:] = [tuple(rect) for rect in snapshot["rects"]]
                region_spans[:] = snapshot["region_spans"]
                keyframe_tracks[:] = snapshot["keyframe_tracks"]
                pending_keyframes[:] = snapshot["pending_keyframes"]
                polygon_shapes[:] = [list(shape) for shape in snapshot["polygon_shapes"]]
                polygon_points[:] = [tuple(point) for point in snapshot["polygon_points"]]
                start_var.set(snapshot["start"])
                end_var.set(snapshot["end"])
                start_frame_var.set(snapshot["start_frame"])
                end_frame_var.set(snapshot["end_frame"])
                selected_region_key[0] = snapshot.get("selected")
                _draw_saved_rects()
                _refresh_region_editor(selected_region_key[0])
                _update_history_buttons()

            def _update_history_buttons():
                undo_button = history_buttons.get("undo")
                redo_button = history_buttons.get("redo")
                if undo_button is not None:
                    undo_button.set_enabled(
                        history.can_undo, tr("No region edit to undo"))
                if redo_button is not None:
                    redo_button.set_enabled(
                        history.can_redo, tr("No region edit to redo"))

            def _set_time_fields(start=None, end=None):
                if not is_video:
                    return
                if start is None:
                    start_var.set("")
                    start_frame_var.set("")
                else:
                    start_var.set(f"{float(start):g}")
                    start_frame_var.set(str(seconds_to_frame(start, fps)))
                if not end:
                    end_var.set("")
                    end_frame_var.set("")
                else:
                    end_var.set(f"{float(end):g}")
                    end_frame_var.set(str(seconds_to_frame(end, fps)))
                time_input_source.update(start="seconds", end="seconds")

            def _record_by_key(key):
                return next(
                    (record for record in _editable_region_records()
                     if record["key"] == key),
                    None,
                )

            def _set_clean_reference_controls(enabled):
                reason = tr("Clean references attach only to timed regions")
                for button in reference_buttons.values():
                    button.set_enabled(bool(enabled), reason)
                reference_alignment_picker.configure(
                    state="readonly" if enabled else "disabled")
                reference_confidence_entry.configure(
                    state="normal" if enabled else "disabled")
                reference_color_toggle.set_enabled(bool(enabled))

            def _selected_clean_reference_span():
                record = _record_by_key(selected_region_key[0])
                if record is None or record.get("source") != "span":
                    return None, None
                return record, region_spans[record["index"]]

            def _load_clean_reference_controls(record):
                if record is None or record.get("source") != "span":
                    _set_clean_reference_controls(False)
                    reference_path_var.set("")
                    reference_status_var.set(tr(
                        "Select a timed region to attach a clean reference."))
                    return
                _set_clean_reference_controls(True)
                span = region_spans[record["index"]]
                spec = span.get("clean_reference") or {}
                reference_path_var.set(
                    Path(spec.get("path", "")).name if spec else tr("None"))
                mode_labels = {
                    "auto": tr("Auto"),
                    "translation": tr("Translation"),
                    "homography": tr("Homography"),
                }
                reference_alignment_var.set(
                    mode_labels.get(spec.get("alignment", "auto"), tr("Auto")))
                reference_color_match_var.set(
                    bool(spec.get("color_match", True)))
                reference_confidence_var.set(
                    f"{float(spec.get('min_confidence', 0.75)):g}")
                reference_status_var.set(
                    tr("Ready to preview alignment.") if spec else tr(
                        "Choose a same-size clean image for this timed region."))

            def _save_clean_reference_options():
                record, span = _selected_clean_reference_span()
                if span is None or not span.get("clean_reference"):
                    return False
                mode_values = {
                    tr("Auto"): "auto",
                    tr("Translation"): "translation",
                    tr("Homography"): "homography",
                }
                try:
                    confidence = float(reference_confidence_var.get().strip())
                    if not math.isfinite(confidence) or not 0.05 <= confidence <= 0.99:
                        raise ValueError
                except (TypeError, ValueError):
                    self._update_status(
                        tr("Confidence must be between 0.05 and 0.99"),
                        "warning",
                    )
                    return False
                from backend.reference_fill import normalize_clean_reference
                updated = normalize_clean_reference({
                    **span["clean_reference"],
                    "alignment": mode_values.get(
                        reference_alignment_var.get(), "auto"),
                    "color_match": reference_color_match_var.get(),
                    "min_confidence": confidence,
                })
                if updated != span.get("clean_reference"):
                    _record_history()
                    span["clean_reference"] = updated
                _load_clean_reference_controls(record)
                return True

            def _choose_clean_reference():
                record, span = _selected_clean_reference_span()
                if span is None:
                    self._update_status(
                        tr("Select a timed region before choosing a clean reference"),
                        "warning",
                    )
                    return
                path = filedialog.askopenfilename(
                    parent=win,
                    title=tr("Choose clean reference image"),
                    filetypes=[
                        (tr("Image files"),
                         "*.png *.jpg *.jpeg *.bmp *.webp *.tif *.tiff"),
                        (tr("All files"), "*.*"),
                    ],
                )
                if not path:
                    return
                reference = safe_imread(path)
                if reference is None:
                    self._update_status(
                        tr("The clean reference image could not be read"),
                        "warning",
                    )
                    return
                if reference.shape[:2] != (orig_h, orig_w):
                    self._update_status(
                        tr("Clean reference must be {width}x{height}").format(
                            width=orig_w, height=orig_h),
                        "warning",
                    )
                    return
                from backend.reference_fill import normalize_clean_reference
                _record_history()
                span["clean_reference"] = normalize_clean_reference({
                    "path": path,
                    "alignment": "auto",
                    "color_match": True,
                    "min_confidence": 0.75,
                })
                _load_clean_reference_controls(record)
                self._update_status(
                    tr("Clean reference attached to the timed region"),
                    "success",
                )

            def _clear_clean_reference():
                record, span = _selected_clean_reference_span()
                if span is None or "clean_reference" not in span:
                    return
                _record_history()
                span.pop("clean_reference", None)
                _load_clean_reference_controls(record)
                self._update_status(tr("Clean reference cleared"), "info")

            def _preview_clean_reference():
                record, span = _selected_clean_reference_span()
                if span is None or not span.get("clean_reference"):
                    self._update_status(
                        tr("Choose a clean reference for the selected timed region"),
                        "warning",
                    )
                    return
                if not _save_clean_reference_options():
                    return
                try:
                    result = self._render_clean_reference_preview(
                        current_frame[0], span)
                except Exception as exc:
                    self._update_status(str(exc), "warning")
                    return
                _draw_image(result.composite if result.accepted else result.aligned)
                _draw_saved_rects()
                status = tr(
                    "{method} alignment: {confidence:.1%}").format(
                        method=result.method.title(),
                        confidence=result.confidence,
                    )
                if result.accepted:
                    status += tr("; color delta BGR {delta}").format(
                        delta=", ".join(f"{value:g}" for value in result.color_delta))
                    reference_status_var.set(status)
                    self._update_status(tr("Clean reference preview ready"), "success")
                else:
                    reference_status_var.set(
                        status + tr("; would fall back to normal inpainting"))
                    self._update_status(result.reason, "warning")

            def _load_selected_region():
                records = _editable_region_records()
                selected_label = selected_region_var.get()
                record = next(
                    (item for item in records if item["label"] == selected_label),
                    None,
                )
                if record is None:
                    record = _record_by_key(selected_region_key[0])
                if record is None:
                    selected_region_key[0] = None
                    for variable in geometry_vars.values():
                        variable.set("")
                    for entry in geometry_entries.values():
                        entry.configure(state="disabled")
                    _load_clean_reference_controls(None)
                    return
                selected_region_key[0] = record["key"]
                shape = record["shape"]
                if "rect" in shape:
                    x1, y1, x2, y2 = shape["rect"]
                    geometry_vars["x"].set(str(x1))
                    geometry_vars["y"].set(str(y1))
                    geometry_vars["width"].set(str(x2 - x1))
                    geometry_vars["height"].set(str(y2 - y1))
                    geometry_vars["vertices"].set("")
                    for key in ("x", "y", "width", "height"):
                        geometry_entries[key].configure(state="normal")
                    geometry_entries["vertices"].configure(state="disabled")
                else:
                    for key in ("x", "y", "width", "height"):
                        geometry_vars[key].set("")
                        geometry_entries[key].configure(state="disabled")
                    geometry_vars["vertices"].set(
                        format_polygon_vertices(shape["polygon"]))
                    geometry_entries["vertices"].configure(state="normal")
                if record["timing"] == "none":
                    _set_time_fields()
                else:
                    _set_time_fields(record.get("start"), record.get("end"))
                _load_clean_reference_controls(record)

            def _refresh_region_editor(prefer_key=None):
                records = _editable_region_records()
                values = [record["label"] for record in records]
                region_picker.configure(values=values)
                key = prefer_key or selected_region_key[0]
                selected = next(
                    (record for record in records if record["key"] == key),
                    records[-1] if records else None,
                )
                if selected is None:
                    selected_region_var.set("")
                    selected_region_key[0] = None
                else:
                    selected_region_key[0] = selected["key"]
                    selected_region_var.set(selected["label"])
                _load_selected_region()

            def _set_record_shape(record, shape):
                source = record["source"]
                index = record["index"]
                kind = "rect" if "rect" in shape else "polygon"
                values = list(shape[kind])
                if source == "rect":
                    rects[index] = tuple(values)
                elif source == "span":
                    region_spans[index]["rect"] = tuple(values)
                elif source == "polygon":
                    polygon_shapes[index] = values
                elif source == "pending":
                    pending_keyframes[index].pop("rect", None)
                    pending_keyframes[index].pop("polygon", None)
                    pending_keyframes[index][kind] = values
                elif source == "track":
                    keyframe = keyframe_tracks[record["track"]]["keyframes"][index]
                    keyframe.pop("rect", None)
                    keyframe.pop("polygon", None)
                    keyframe[kind] = values

            def _seconds_from_fields(which):
                seconds_var = start_var if which == "start" else end_var
                frame_var = start_frame_var if which == "start" else end_frame_var
                seconds_raw = seconds_var.get().strip()
                frame_raw = frame_var.get().strip()
                use_frame = (
                    time_input_source[which] == "frame"
                    or (not seconds_raw and bool(frame_raw))
                )
                raw = frame_raw if use_frame else seconds_raw
                if not raw:
                    return None
                if use_frame:
                    value = frame_to_seconds(raw, fps)
                    seconds_var.set(f"{value:g}")
                else:
                    try:
                        value = float(raw)
                    except (TypeError, ValueError) as exc:
                        raise ValueError(tr("Time must be a number")) from exc
                    if not math.isfinite(value) or value < 0:
                        raise ValueError(tr("Time must be zero or greater"))
                    frame_var.set(str(seconds_to_frame(value, fps)))
                return value

            def _sync_time_field(which, source):
                if not is_video:
                    return
                time_input_source[which] = source
                try:
                    _seconds_from_fields(which)
                except (TypeError, ValueError):
                    return

            def _apply_numeric_region_edit():
                record = _record_by_key(selected_region_key[0])
                if record is None:
                    self._update_status(tr("Select or draw a region before editing it"), "warning")
                    return False
                try:
                    if "rect" in record["shape"]:
                        rect = rect_from_xywh(
                            geometry_vars["x"].get(), geometry_vars["y"].get(),
                            geometry_vars["width"].get(), geometry_vars["height"].get(),
                            orig_w, orig_h,
                        )
                        shape = {"rect": list(rect)}
                    else:
                        shape = {"polygon": parse_polygon_vertices(
                            geometry_vars["vertices"].get(), orig_w, orig_h)}
                    start_s = _seconds_from_fields("start") if is_video else None
                    end_s = _seconds_from_fields("end") if is_video else None
                    if end_s is not None and (start_s is None or end_s <= start_s):
                        raise ValueError(tr("Region end must be after its start"))
                    if record["timing"] == "point" and end_s is not None:
                        raise ValueError(tr("A motion keyframe uses one frame, not an end frame"))
                    if record["timing"] == "point" and start_s is not None:
                        if record["source"] == "pending":
                            peers = pending_keyframes
                        else:
                            peers = keyframe_tracks[record["track"]]["keyframes"]
                        if any(
                            peer_index != record["index"]
                            and abs(float(peer.get("time", 0.0)) - start_s) <= 1e-9
                            for peer_index, peer in enumerate(peers)
                        ):
                            raise ValueError(tr("Motion keyframes must use distinct frames"))
                    if "polygon" in shape and record["timing"] == "none" and (
                        start_s is not None or end_s is not None
                    ):
                        raise ValueError(tr("Use motion keyframes to time polygon regions"))
                except (TypeError, ValueError) as exc:
                    self._update_status(str(exc), "warning")
                    return False

                _record_history()
                _set_record_shape(record, shape)
                if record["timing"] == "range":
                    region_spans[record["index"]]["start"] = start_s or 0.0
                    region_spans[record["index"]]["end"] = end_s or 0.0
                elif record["timing"] == "point" and start_s is not None:
                    if record["source"] == "pending":
                        pending_keyframes[record["index"]]["time"] = start_s
                    else:
                        track = keyframe_tracks[record["track"]]
                        track["keyframes"][record["index"]]["time"] = start_s
                        times = [float(item["time"]) for item in track["keyframes"]]
                        track["start"], track["end"] = min(times), max(times)
                _draw_saved_rects()
                _refresh_region_editor(record["key"])
                _update_history_buttons()
                self._update_status(tr("Applied precise region edit"), "success")
                return True

            def _transform_selected_region(event):
                if isinstance(event.widget, (tk.Entry, ttk.Combobox, tk.Scale)):
                    return None
                direction = str(event.keysym)
                if direction not in {"Left", "Right", "Up", "Down"}:
                    return None
                record = _record_by_key(selected_region_key[0])
                if record is None:
                    return "break"
                step = 10 if int(getattr(event, "state", 0)) & 0x0001 else 1
                resize = bool(int(getattr(event, "state", 0)) & 0x0004)
                dx = dy = dw = dh = 0
                if resize:
                    dw = -step if direction == "Left" else step if direction == "Right" else 0
                    dh = -step if direction == "Up" else step if direction == "Down" else 0
                else:
                    dx = -step if direction == "Left" else step if direction == "Right" else 0
                    dy = -step if direction == "Up" else step if direction == "Down" else 0
                shape = transform_region_shape(
                    record["shape"], frame_width=orig_w, frame_height=orig_h,
                    dx=dx, dy=dy, dw=dw, dh=dh,
                )
                if shape == record["shape"]:
                    return "break"
                _record_history()
                _set_record_shape(record, shape)
                _draw_saved_rects()
                _refresh_region_editor(record["key"])
                _update_history_buttons()
                return "break"

            def _undo_region_edit(event=None):
                if event is not None and isinstance(event.widget, tk.Entry):
                    return None
                snapshot = history.undo(_editor_snapshot())
                if snapshot is None:
                    return "break"
                _restore_editor_snapshot(snapshot)
                self._update_status(tr("Undid region edit"), "info")
                return "break"

            def _redo_region_edit(event=None):
                if event is not None and isinstance(event.widget, tk.Entry):
                    return None
                snapshot = history.redo(_editor_snapshot())
                if snapshot is None:
                    return "break"
                _restore_editor_snapshot(snapshot)
                self._update_status(tr("Redid region edit"), "info")
                return "break"

            apply_edit_button = ModernButton(
                precise_bottom, text=tr("Apply"), command=_apply_numeric_region_edit,
                style="secondary", size="sm", width=72)
            apply_edit_button.pack(side="right")
            redo_edit_button = ModernButton(
                precise_bottom, text=tr("Redo"), command=_redo_region_edit,
                style="ghost", size="sm", width=68)
            redo_edit_button.pack(side="right", padx=(0, Theme.S_XS))
            undo_edit_button = ModernButton(
                precise_bottom, text=tr("Undo"), command=_undo_region_edit,
                style="ghost", size="sm", width=68)
            undo_edit_button.pack(side="right", padx=(0, Theme.S_XS))
            history_buttons.update(undo=undo_edit_button, redo=redo_edit_button)

            region_picker.bind("<<ComboboxSelected>>", lambda _event: _load_selected_region())
            reference_alignment_picker.bind(
                "<<ComboboxSelected>>",
                lambda _event: _save_clean_reference_options(),
            )
            reference_confidence_entry.bind(
                "<FocusOut>",
                lambda _event: _save_clean_reference_options(),
                add="+",
            )
            if is_video:
                start_entry.bind("<KeyRelease>", lambda _event: time_input_source.update(start="seconds"))
                end_entry.bind("<KeyRelease>", lambda _event: time_input_source.update(end="seconds"))
                start_frame_entry.bind("<KeyRelease>", lambda _event: time_input_source.update(start="frame"))
                end_frame_entry.bind("<KeyRelease>", lambda _event: time_input_source.update(end="frame"))
                start_entry.bind("<FocusOut>", lambda _event: _sync_time_field("start", "seconds"), add="+")
                end_entry.bind("<FocusOut>", lambda _event: _sync_time_field("end", "seconds"), add="+")
                start_frame_entry.bind("<FocusOut>", lambda _event: _sync_time_field("start", "frame"), add="+")
                end_frame_entry.bind("<FocusOut>", lambda _event: _sync_time_field("end", "frame"), add="+")

            win._vsr_region_picker = region_picker
            win._vsr_geometry_entries = geometry_entries
            win._vsr_apply_region_edit = _apply_numeric_region_edit
            win._vsr_undo_region_edit = _undo_region_edit
            win._vsr_redo_region_edit = _redo_region_edit
            win._vsr_region_key_handler = _transform_selected_region
            win._vsr_clean_reference_buttons = reference_buttons
            win._vsr_choose_clean_reference = _choose_clean_reference
            win._vsr_clear_clean_reference = _clear_clean_reference
            win._vsr_preview_clean_reference = _preview_clean_reference
            win._vsr_clean_reference_status = reference_status_var

            # Action row: Add another, Clear all, Save.
            actions = tk.Frame(win, bg=Theme.BG_OVERLAY)
            actions.pack(fill="x", padx=Theme.S_MD, pady=(Theme.S_SM, Theme.S_MD))

            def _clear_all():
                _record_history()
                rects.clear()
                region_spans.clear()
                keyframe_tracks.clear()
                pending_keyframes.clear()
                polygon_shapes.clear()
                polygon_points.clear()
                span_summary_var.set(tr("Optional") if is_video else "")
                _draw_saved_rects()
                _refresh_region_editor()

            def _add_region_keyframe() -> bool:
                if not is_video:
                    return False
                if polygon_points and not _finish_polygon():
                    return False
                shape_count = len(rects) + len(polygon_shapes)
                if shape_count != 1:
                    self._update_status(
                        "Draw exactly one rectangle or polygon for this keyframe",
                        "warning",
                    )
                    return False
                if polygon_shapes:
                    shape = {"polygon": list(polygon_shapes[0])}
                else:
                    shape = {"rect": list(rects[0])}
                if pending_keyframes:
                    first = pending_keyframes[0]
                    kind = "rect" if "rect" in first else "polygon"
                    if kind not in shape or len(first[kind]) != len(shape[kind]):
                        self._update_status(
                            "Keyframes in one motion track need the same shape and vertex count",
                            "warning",
                        )
                        return False
                seconds = current_frame_index[0] / max(fps, 1.0)
                _record_history()
                keyframe = {"time": seconds, **shape}
                pending_keyframes[:] = [
                    item for item in pending_keyframes
                    if abs(float(item["time"]) - seconds) > 1e-9
                ]
                pending_keyframes.append(keyframe)
                pending_keyframes.sort(key=lambda item: float(item["time"]))
                rects.clear()
                polygon_shapes.clear()
                span_summary_var.set(
                    f"{len(pending_keyframes)} pending keyframe"
                    f"{'s' if len(pending_keyframes) != 1 else ''}"
                )
                _draw_saved_rects()
                _refresh_region_editor(f"pending:{len(pending_keyframes) - 1}")
                self._update_status(
                    f"Added motion keyframe at {seconds:.3f} seconds", "success")
                return True

            def _commit_motion_track() -> bool:
                if len(pending_keyframes) < 2:
                    self._update_status(
                        "Add at least two keyframes before saving a motion track",
                        "warning",
                    )
                    return False
                track = normalize_region_keyframe_tracks([{
                    "start": pending_keyframes[0]["time"],
                    "end": pending_keyframes[-1]["time"],
                    "keyframes": list(pending_keyframes),
                }])
                if not track:
                    self._update_status(
                        "The motion keyframes could not be normalized", "warning")
                    return False
                _record_history()
                keyframe_tracks.append(track[0])
                pending_keyframes.clear()
                span_summary_var.set(
                    f"{len(keyframe_tracks)} motion track"
                    f"{'s' if len(keyframe_tracks) != 1 else ''}"
                )
                _draw_saved_rects()
                _refresh_region_editor(f"track:{len(keyframe_tracks) - 1}:0")
                self._update_status("Saved interpolated motion track", "success")
                return True

            def _parse_time_inputs():
                start_raw = start_var.get().strip() if is_video else ""
                end_raw = end_var.get().strip() if is_video else ""
                has_time = bool(
                    start_raw or end_raw
                    or start_frame_var.get().strip()
                    or end_frame_var.get().strip()
                )
                try:
                    start_s = (_seconds_from_fields("start")
                               if start_raw or start_frame_var.get().strip() else None)
                    end_s = (_seconds_from_fields("end")
                             if end_raw or end_frame_var.get().strip() else None)
                except (TypeError, ValueError) as exc:
                    self._update_status(str(exc), "warning")
                    return None
                start_s = start_s or 0.0
                end_s = end_s or 0.0
                if end_s and end_s <= start_s:
                    self._update_status(
                        "Timed region end must be after start", "warning")
                    return None
                return has_time, start_s, end_s

            def _add_timed_regions(close_on_empty: bool = False) -> bool:
                parsed = _parse_time_inputs()
                if parsed is None:
                    return False
                has_time, start_s, end_s = parsed
                if not rects:
                    if close_on_empty:
                        return True
                    self._update_status("Draw a region before adding a timed range", "warning")
                    return False
                if polygon_shapes or polygon_points:
                    self._update_status(
                        "Polygons are saved through motion keyframes", "warning")
                    return False
                if not is_video:
                    return False
                if not has_time and not region_spans:
                    if not close_on_empty:
                        self._update_status(
                            "Enter a start or end second for a timed range",
                            "warning",
                        )
                    return True if close_on_empty else False
                _record_history()
                for rect in rects:
                    region_spans.append({
                        "rect": tuple(rect),
                        "start": start_s,
                        "end": end_s,
                    })
                rects.clear()
                start_var.set("")
                end_var.set("")
                span_summary_var.set(
                    f"{len(region_spans)} timed region"
                    f"{'s' if len(region_spans) != 1 else ''}"
                )
                _draw_saved_rects()
                _refresh_region_editor(
                    f"span:{len(region_spans) - 1}" if region_spans else None)
                return True

            def _save_and_close():
                if pending_keyframes and not _commit_motion_track():
                    return
                if is_video and not _add_timed_regions(close_on_empty=True):
                    return
                spans = _coerce_region_span_list(region_spans) or []
                tracks = normalize_region_keyframe_tracks(keyframe_tracks) or []
                if tracks:
                    self.config.subtitle_region_keyframes = tracks
                    self.config.subtitle_region_spans = None
                    self.config.subtitle_areas = None
                    self.config.subtitle_area = None
                    self._update_status(
                        f"Saved {len(tracks)} moving subtitle track"
                        f"{'s' if len(tracks) != 1 else ''}",
                        "success",
                    )
                elif rects:
                    self.config.subtitle_areas = [tuple(r) for r in rects]
                    self.config.subtitle_area = rects[0]
                    self.config.subtitle_region_spans = None
                    self.config.subtitle_region_keyframes = None
                    self._update_status(
                        f"Saved {len(rects)} subtitle region{'s' if len(rects) != 1 else ''}",
                        "success",
                    )
                elif spans:
                    self.config.subtitle_region_spans = spans
                    self.config.subtitle_areas = None
                    self.config.subtitle_area = None
                    self.config.subtitle_region_keyframes = None
                    self._update_status(
                        f"Saved {len(spans)} timed subtitle region"
                        f"{'s' if len(spans) != 1 else ''}",
                        "success",
                    )
                else:
                    self.config.subtitle_areas = None
                    self.config.subtitle_area = None
                    self.config.subtitle_region_spans = None
                    self.config.subtitle_region_keyframes = None
                    self._update_status("Cleared manual subtitle regions", "info")
                self._apply_region_settings_to_idle_items()
                self._update_region_label_display()
                win.destroy()

            ModernButton(actions, text=tr("Clear all"), command=_clear_all,
                         style="ghost", size="sm", width=92).pack(side="left")
            if is_video:
                ModernButton(actions, text=tr("Add timed"), command=_add_timed_regions,
                             style="secondary", size="sm", width=96).pack(
                                 side="left", padx=(Theme.S_SM, 0))
                ModernButton(actions, text=tr("Add keyframe"),
                             command=_add_region_keyframe,
                             style="secondary", size="sm", width=108).pack(
                                 side="left", padx=(Theme.S_SM, 0))
                ModernButton(actions, text=tr("Save motion"),
                             command=_commit_motion_track,
                             style="secondary", size="sm", width=108).pack(
                                 side="left", padx=(Theme.S_SM, 0))
            ModernButton(actions, text=tr("Save"), command=_save_and_close,
                         style="primary", size="sm", width=92).pack(
                             side="right")
            ModernButton(actions, text=tr("Cancel"), command=win.destroy,
                         style="ghost", size="sm", width=92).pack(
                             side="right", padx=(0, Theme.S_SM))

            hint_frame = tk.Frame(win, bg=Theme.BG_OVERLAY)
            hint_frame.pack(fill="x", pady=(0, Theme.S_MD))
            tk.Label(hint_frame,
                     text=tr("Drag to add; choose a region for exact coordinates and timing."),
                     font=f(Theme.F_BODY_SM, "bold"),
                     bg=Theme.BG_OVERLAY, fg=Theme.TEXT_PRIMARY).pack()
            tk.Label(hint_frame,
                     text=tr("Arrow keys nudge; Shift moves 10 px; Ctrl+arrows resize; Ctrl+Z/Y undo or redo."),
                     font=f(Theme.F_META),
                     bg=Theme.BG_OVERLAY, fg=Theme.TEXT_MUTED).pack(pady=(2, 0))

            for sequence in ("<Left>", "<Right>", "<Up>", "<Down>"):
                canvas.bind(sequence, _transform_selected_region, add="+")
            win.bind("<Control-z>", _undo_region_edit, add="+")
            win.bind("<Control-y>", _redo_region_edit, add="+")
            win.bind("<Control-Shift-Z>", _redo_region_edit, add="+")
            _refresh_region_editor()
            _update_history_buttons()

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
                    logger.warning("Region selector capture release failed", exc_info=True)

    def _reset_region(self):
        """Reset subtitle region to auto-detect.

        Clears every manual-region field so the backend cannot keep using old
        fixed or time-ranged rectangles while the UI claims automatic detection.
        """
        self.config.subtitle_area = None
        self.config.subtitle_areas = None
        self.config.subtitle_region_spans = None
        self.config.subtitle_region_keyframes = None
        self._apply_region_settings_to_idle_items()
        self._update_region_label_display()
        self._update_status("Subtitle detection returned to automatic mode")
