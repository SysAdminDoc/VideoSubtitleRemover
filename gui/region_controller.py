from __future__ import annotations

import math
import logging
import threading
from functools import partial
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


class RegionSelectorWindow:
    """Non-blocking region editor with explicit callback state."""

    def __init__(self, host: RegionControllerHost):
        self.host = host

    def __getattr__(self, name: str):
        return getattr(self.host, name)

    def open(self):
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
        self._cv2 = _cv2

        self.is_video = is_video_file(source_path)
        self.cap = _cv2.VideoCapture(source_path) if self.is_video else None
        if self.is_video and not self.cap.isOpened():
            logger.error("Could not open video for region selection")
            self.cap.release()
            self._update_status(
                "The selected video could not be opened for region selection",
                "warning",
            )
            return
        try:
            if self.is_video:
                self.frame_count = int(self.cap.get(_cv2.CAP_PROP_FRAME_COUNT)) or 1
                self.fps = self.cap.get(_cv2.CAP_PROP_FPS) or 30.0
                if self.fps <= 0:
                    self.fps = 30.0
                ret, frame = self.cap.read()
                if not ret:
                    # Early return bypasses both the except-release below
                    # and the <Destroy> binding (not installed yet) --
                    # release here or the capture leaks.
                    logger.error("Could not read video frame for region selection")
                    self.cap.release()
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
                self.frame_count, self.fps = 1, 30.0

            self.orig_h, self.orig_w = frame.shape[:2]
            screen_w = self.root.winfo_screenwidth()
            screen_h = self.root.winfo_screenheight()
            max_w = min(900, int(screen_w * 0.8))
            height_ratio = 0.55 if screen_h >= 600 else 0.7
            max_h = min(540, int(screen_h * height_ratio))
            self.scale = min(max_w / self.orig_w, max_h / self.orig_h, 1.0)
            self.disp_w, self.disp_h = int(self.orig_w * self.scale), int(self.orig_h * self.scale)

            # Selector state: every saved rect lives in `rects` (image-
            # space coordinates). Drag creates the current rect; on
            # release it joins the list. Adding another rect re-arms the
            # canvas for the next drag.
            self.rects: List[Tuple[int, int, int, int]] = []
            self.region_spans = _coerce_region_span_list(
                getattr(self.config, "subtitle_region_spans", None)) or []
            self.keyframe_tracks = normalize_region_keyframe_tracks(
                getattr(self.config, "subtitle_region_keyframes", None)) or []
            self.pending_keyframes: List[dict] = []
            self.polygon_shapes: List[List[int]] = []
            self.polygon_points: List[Tuple[int, int]] = []
            self.current_frame_index = [0]
            self.current_frame = [frame]
            preload = []
            if not self.region_spans and not self.keyframe_tracks:
                preload = self.config.subtitle_areas or (
                    [self.config.subtitle_area] if self.config.subtitle_area else []
                )
            self.rects.extend([tuple(r) for r in preload if r])

            self.win = tk.Toplevel(self.root)
            self.win.title("Choose subtitle region")
            self.win.configure(bg=Theme.BG_OVERLAY)
            self.win.resizable(False, False)

            self.canvas = tk.Canvas(
                self.win,
                width=self.disp_w,
                height=self.disp_h,
                highlightthickness=2,
                highlightbackground=Theme.BORDER_SUBTLE,
                highlightcolor=Theme.BLUE_PRIMARY,
                bg=Theme.BG_DARK,
                cursor="cross",
                takefocus=True,
            )
            self.canvas.pack()
            set_accessible_metadata(
                self.canvas,
                role="region editor canvas",
                label=tr("Subtitle region preview"),
                description=tr(
                    "Draw with the pointer, or use arrow keys to edit the selected region."
                ),
            )
            self.canvas_image_id = self.canvas.create_image(0, 0, anchor="nw")
            self.canvas._photo = None

            self.rect_ids: List[int] = []
            self.drag_id = [None]
            self.drag_start = [0, 0]
            self.ocr_overlay_ids: List[int] = []
            self.ocr_probe_after_id = None
            self.ocr_probe_inflight = False
            self.ocr_pending_rect = None
            self.ocr_probe_generation = 0
            self.shape_mode_var = tk.StringVar(value=tr("Rectangle"))





            self._draw_image(frame)
            self._draw_saved_rects()

            # Drawing handlers.



            self.canvas.bind("<ButtonPress-1>", self.on_press)
            self.canvas.bind("<B1-Motion>", self.on_drag)
            self.canvas.bind("<ButtonRelease-1>", self.on_release)
            self.canvas._vsr_region_drag_handlers = (self.on_press, self.on_drag, self.on_release)

            shape_row = tk.Frame(self.win, bg=Theme.BG_OVERLAY)
            shape_row.pack(fill="x", padx=Theme.S_MD, pady=(Theme.S_SM, 0))
            tk.Label(shape_row, text=tr("Shape"), font=f(Theme.F_META),
                     bg=Theme.BG_OVERLAY, fg=Theme.TEXT_MUTED).pack(side="left")
            shape_picker = ttk.Combobox(
                shape_row,
                width=12,
                state="readonly",
                textvariable=self.shape_mode_var,
                values=(tr("Rectangle"), tr("Polygon")),
                style="Dark.TCombobox",
            )
            shape_picker.pack(side="left", padx=(Theme.S_XS, Theme.S_MD))
            shape_hint_var = tk.StringVar(
                value=tr("Drag a rectangle, or choose Polygon and click vertices."))
            tk.Label(shape_row, textvariable=shape_hint_var,
                     font=f(Theme.F_META), bg=Theme.BG_OVERLAY,
                     fg=Theme.TEXT_MUTED).pack(side="left")


            finish_polygon_btn = ModernButton(
                shape_row, text=tr("Finish polygon"), command=self._finish_polygon,
                style="secondary", size="sm", width=112)
            finish_polygon_btn.pack(side="right")

            self.ocr_feedback_var = tk.StringVar(
                value=tr("Drag a rectangle to preview OCR boxes and confidence."))
            tk.Label(
                self.win,
                textvariable=self.ocr_feedback_var,
                font=f(Theme.F_META),
                bg=Theme.BG_OVERLAY,
                fg=Theme.INFO,
            ).pack(fill="x", padx=Theme.S_MD, pady=(Theme.S_XS, 0))

            # Frame slider for videos.
            if self.is_video and self.frame_count > 1:
                slider_row = tk.Frame(self.win, bg=Theme.BG_OVERLAY)
                slider_row.pack(fill="x", padx=Theme.S_MD, pady=(Theme.S_SM, 0))
                tk.Label(slider_row, text=tr("Frame"),
                         font=f(Theme.F_BODY_SM),
                         bg=Theme.BG_OVERLAY, fg=Theme.TEXT_SECONDARY).pack(side="left")
                self.ts_label = tk.Label(slider_row, text="00:00:00",
                                    font=f(Theme.F_META),
                                    bg=Theme.BG_OVERLAY, fg=Theme.TEXT_MUTED)
                self.ts_label.pack(side="right")


                slider = tk.Scale(
                    self.win, from_=0, to=self.frame_count - 1, orient="horizontal",
                    command=self._on_slider, length=self.disp_w - 24,
                    bg=Theme.BG_OVERLAY, fg=Theme.TEXT_PRIMARY,
                    troughcolor=Theme.BG_TERTIARY,
                    activebackground=Theme.BLUE_PRIMARY,
                    highlightthickness=0, showvalue=False,
                )
                slider.pack(fill="x", padx=Theme.S_MD, pady=(0, Theme.S_SM))

            time_row = None
            self.start_var = tk.StringVar()
            self.end_var = tk.StringVar()
            self.start_frame_var = tk.StringVar()
            self.end_frame_var = tk.StringVar()
            self.span_summary_var = tk.StringVar()
            if self.is_video:
                existing_starts = {round(float(s.get("start", 0.0)), 3)
                                   for s in self.region_spans}
                existing_ends = {round(float(s.get("end", 0.0)), 3)
                                 for s in self.region_spans}
                if len(existing_starts) == 1 and len(existing_ends) == 1:
                    start_value = next(iter(existing_starts))
                    end_value = next(iter(existing_ends))
                    self.start_var.set(f"{start_value:g}" if start_value else "")
                    self.end_var.set(f"{end_value:g}" if end_value else "")

                time_row = tk.Frame(self.win, bg=Theme.BG_OVERLAY)
                time_row.pack(fill="x", padx=Theme.S_MD, pady=(0, Theme.S_SM))
                tk.Label(time_row, text=tr("Start sec"),
                         font=f(Theme.F_META),
                         bg=Theme.BG_OVERLAY, fg=Theme.TEXT_MUTED).pack(side="left")
                start_entry = tk.Entry(
                    time_row, width=9, textvariable=self.start_var,
                    bg=Theme.BG_TERTIARY, fg=Theme.TEXT_PRIMARY,
                    insertbackground=Theme.TEXT_PRIMARY,
                    relief="flat",
                )
                start_entry.pack(side="left", padx=(Theme.S_XS, Theme.S_MD))
                tk.Label(time_row, text=tr("End sec"),
                         font=f(Theme.F_META),
                         bg=Theme.BG_OVERLAY, fg=Theme.TEXT_MUTED).pack(side="left")
                end_entry = tk.Entry(
                    time_row, width=9, textvariable=self.end_var,
                    bg=Theme.BG_TERTIARY, fg=Theme.TEXT_PRIMARY,
                    insertbackground=Theme.TEXT_PRIMARY,
                    relief="flat",
                )
                end_entry.pack(side="left", padx=(Theme.S_XS, Theme.S_MD))
                tk.Label(time_row, text=tr("Start frame"),
                         font=f(Theme.F_META),
                         bg=Theme.BG_OVERLAY, fg=Theme.TEXT_MUTED).pack(side="left")
                start_frame_entry = tk.Entry(
                    time_row, width=7, textvariable=self.start_frame_var,
                    bg=Theme.BG_TERTIARY, fg=Theme.TEXT_PRIMARY,
                    insertbackground=Theme.TEXT_PRIMARY,
                    relief="flat",
                )
                start_frame_entry.pack(side="left", padx=(Theme.S_XS, Theme.S_MD))
                tk.Label(time_row, text=tr("End frame"),
                         font=f(Theme.F_META),
                         bg=Theme.BG_OVERLAY, fg=Theme.TEXT_MUTED).pack(side="left")
                end_frame_entry = tk.Entry(
                    time_row, width=7, textvariable=self.end_frame_var,
                    bg=Theme.BG_TERTIARY, fg=Theme.TEXT_PRIMARY,
                    insertbackground=Theme.TEXT_PRIMARY,
                    relief="flat",
                )
                end_frame_entry.pack(side="left", padx=(Theme.S_XS, Theme.S_MD))
                self.span_summary_var.set(
                    (
                        f"{len(self.keyframe_tracks)} motion track"
                        f"{'s' if len(self.keyframe_tracks) != 1 else ''}"
                    ) if self.keyframe_tracks else (
                        f"{len(self.region_spans)} timed region"
                        f"{'s' if len(self.region_spans) != 1 else ''}"
                        if self.region_spans else tr("Optional")
                    )
                )
                span_label = tk.Label(
                    time_row, textvariable=self.span_summary_var,
                    font=f(Theme.F_META),
                    bg=Theme.BG_OVERLAY, fg=Theme.TEXT_MUTED,
                )
                span_label.pack(side="right")
                self.win._vsr_start_entry = start_entry
                self.win._vsr_end_entry = end_entry
                self.win._vsr_start_frame_entry = start_frame_entry
                self.win._vsr_end_frame_entry = end_frame_entry

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
                self.win,
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

            self.selected_region_var = tk.StringVar()
            self.region_picker = ttk.Combobox(
                precise_top,
                width=21,
                state="readonly",
                textvariable=self.selected_region_var,
                style="Dark.TCombobox",
            )
            tk.Label(
                precise_top, text=tr("Selected"), font=f(Theme.F_META),
                bg=Theme.BG_SECONDARY, fg=Theme.TEXT_MUTED,
            ).pack(side="left")
            self.region_picker.pack(side="left", padx=(Theme.S_XS, Theme.S_MD))
            set_accessible_metadata(
                self.region_picker,
                role="combo box",
                label=tr("Selected manual region"),
                description=tr("Choose a rectangle, polygon, timed region, or keyframe to edit."),
            )

            self.geometry_vars = {
                "x": tk.StringVar(),
                "y": tk.StringVar(),
                "width": tk.StringVar(),
                "height": tk.StringVar(),
                "vertices": tk.StringVar(),
            }
            self.geometry_entries = {}
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
                    precise_top, width=width, textvariable=self.geometry_vars[key],
                    bg=Theme.BG_TERTIARY, fg=Theme.TEXT_PRIMARY,
                    disabledbackground=Theme.BG_DARK,
                    disabledforeground=Theme.TEXT_DISABLED,
                    insertbackground=Theme.TEXT_PRIMARY, relief="flat",
                )
                entry.pack(side="left", padx=(Theme.S_XS, Theme.S_SM))
                self.geometry_entries[key] = entry
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
                textvariable=self.geometry_vars["vertices"],
                bg=Theme.BG_TERTIARY, fg=Theme.TEXT_PRIMARY,
                disabledbackground=Theme.BG_DARK,
                disabledforeground=Theme.TEXT_DISABLED,
                insertbackground=Theme.TEXT_PRIMARY, relief="flat",
            )
            vertices_entry.pack(side="left", padx=(Theme.S_XS, Theme.S_MD), fill="x", expand=True)
            self.geometry_entries["vertices"] = vertices_entry
            set_accessible_metadata(
                vertices_entry,
                role="coordinate input",
                label=tr("Polygon vertex coordinates"),
                description=tr("Use x,y pairs separated by semicolons; at least three vertices are required."),
            )

            reference_frame = tk.Frame(
                self.win,
                bg=Theme.BG_SECONDARY,
                highlightthickness=1,
                highlightbackground=Theme.BORDER_SUBTLE,
            )
            reference_frame.pack(
                fill="x", padx=Theme.S_MD, pady=(0, Theme.S_SM))
            self.reference_path_var = tk.StringVar(value="")
            self.reference_status_var = tk.StringVar(
                value=tr("Select a timed region to attach a clean reference."))
            self.reference_alignment_var = tk.StringVar(value="Auto")
            self.reference_color_match_var = tk.BooleanVar(value=True)
            self.reference_confidence_var = tk.StringVar(value="0.75")
            reference_top = tk.Frame(reference_frame, bg=Theme.BG_SECONDARY)
            reference_top.pack(
                fill="x", padx=Theme.S_SM, pady=(Theme.S_XS, 0))
            tk.Label(
                reference_top, text=tr("Clean reference"),
                font=f(Theme.F_BODY_SM), bg=Theme.BG_SECONDARY,
                fg=Theme.TEXT_PRIMARY,
            ).pack(side="left")
            tk.Label(
                reference_top, textvariable=self.reference_path_var,
                font=f(Theme.F_META), bg=Theme.BG_SECONDARY,
                fg=Theme.TEXT_MUTED, anchor="w",
            ).pack(side="left", fill="x", expand=True, padx=Theme.S_SM)
            self.reference_buttons = {}
            for key, label, width, command in (
                ("choose", tr("Choose"), 76,
                 lambda: self._choose_clean_reference()),
                ("preview", tr("Preview"), 78,
                 lambda: self._preview_clean_reference()),
                ("clear", tr("Clear"), 68,
                 lambda: self._clear_clean_reference()),
            ):
                button = ModernButton(
                    reference_top, text=label, width=width,
                    command=command, style="ghost", size="sm")
                button.pack(side="right", padx=(Theme.S_XS, 0))
                self.reference_buttons[key] = button
            reference_options = tk.Frame(
                reference_frame, bg=Theme.BG_SECONDARY)
            reference_options.pack(
                fill="x", padx=Theme.S_SM, pady=(Theme.S_XS, 0))
            tk.Label(
                reference_options, text=tr("Alignment"), font=f(Theme.F_META),
                bg=Theme.BG_SECONDARY, fg=Theme.TEXT_MUTED,
            ).pack(side="left")
            self.reference_alignment_picker = ttk.Combobox(
                reference_options, width=13, state="readonly",
                textvariable=self.reference_alignment_var,
                values=(tr("Auto"), tr("Translation"), tr("Homography")),
                style="Dark.TCombobox",
            )
            self.reference_alignment_picker.pack(
                side="left", padx=(Theme.S_XS, Theme.S_MD))
            self.reference_color_toggle = ModernToggle(
                reference_options,
                text=tr("Match each frame's color"),
                variable=self.reference_color_match_var,
                command=lambda: self._save_clean_reference_options(),
                bg=Theme.BG_SECONDARY,
            )
            self.reference_color_toggle.pack(side="left")
            tk.Label(
                reference_options, text=tr("Confidence floor"),
                font=f(Theme.F_META), bg=Theme.BG_SECONDARY,
                fg=Theme.TEXT_MUTED,
            ).pack(side="left", padx=(Theme.S_MD, Theme.S_XS))
            self.reference_confidence_entry = tk.Entry(
                reference_options, width=6,
                textvariable=self.reference_confidence_var,
                bg=Theme.BG_TERTIARY, fg=Theme.TEXT_PRIMARY,
                insertbackground=Theme.TEXT_PRIMARY, relief="flat",
            )
            self.reference_confidence_entry.pack(side="left")
            tk.Label(
                reference_frame, textvariable=self.reference_status_var,
                font=f(Theme.F_META), bg=Theme.BG_SECONDARY,
                fg=Theme.TEXT_MUTED, anchor="w",
            ).pack(
                fill="x", padx=Theme.S_SM, pady=(Theme.S_XS, Theme.S_XS))
            set_accessible_metadata(
                self.reference_alignment_picker, role="combo box",
                label=tr("Clean reference alignment mode"))
            set_accessible_metadata(
                self.reference_confidence_entry, role="numeric input",
                label=tr("Clean reference minimum confidence"),
                description=tr("Values below this floor use normal inpainting."))

            self.selected_region_key = [None]
            self.history = RegionEditHistory()
            self.history_buttons = {"undo": None, "redo": None}
            self.time_input_source = {"start": "seconds", "end": "seconds"}
























            apply_edit_button = ModernButton(
                precise_bottom, text=tr("Apply"), command=self._apply_numeric_region_edit,
                style="secondary", size="sm", width=72)
            apply_edit_button.pack(side="right")
            redo_edit_button = ModernButton(
                precise_bottom, text=tr("Redo"), command=self._redo_region_edit,
                style="ghost", size="sm", width=68)
            redo_edit_button.pack(side="right", padx=(0, Theme.S_XS))
            undo_edit_button = ModernButton(
                precise_bottom, text=tr("Undo"), command=self._undo_region_edit,
                style="ghost", size="sm", width=68)
            undo_edit_button.pack(side="right", padx=(0, Theme.S_XS))
            self.history_buttons.update(undo=undo_edit_button, redo=redo_edit_button)

            self.region_picker.bind("<<ComboboxSelected>>", lambda _event: self._load_selected_region())
            self.reference_alignment_picker.bind(
                "<<ComboboxSelected>>",
                lambda _event: self._save_clean_reference_options(),
            )
            self.reference_confidence_entry.bind(
                "<FocusOut>",
                lambda _event: self._save_clean_reference_options(),
                add="+",
            )
            if self.is_video:
                start_entry.bind("<KeyRelease>", lambda _event: self.time_input_source.update(start="seconds"))
                end_entry.bind("<KeyRelease>", lambda _event: self.time_input_source.update(end="seconds"))
                start_frame_entry.bind("<KeyRelease>", lambda _event: self.time_input_source.update(start="frame"))
                end_frame_entry.bind("<KeyRelease>", lambda _event: self.time_input_source.update(end="frame"))
                start_entry.bind("<FocusOut>", lambda _event: self._sync_time_field("start", "seconds"), add="+")
                end_entry.bind("<FocusOut>", lambda _event: self._sync_time_field("end", "seconds"), add="+")
                start_frame_entry.bind("<FocusOut>", lambda _event: self._sync_time_field("start", "frame"), add="+")
                end_frame_entry.bind("<FocusOut>", lambda _event: self._sync_time_field("end", "frame"), add="+")

            self.win._vsr_region_picker = self.region_picker
            self.win._vsr_geometry_entries = self.geometry_entries
            self.win._vsr_apply_region_edit = self._apply_numeric_region_edit
            self.win._vsr_undo_region_edit = self._undo_region_edit
            self.win._vsr_redo_region_edit = self._redo_region_edit
            self.win._vsr_region_key_handler = self._transform_selected_region
            self.win._vsr_clean_reference_buttons = self.reference_buttons
            self.win._vsr_choose_clean_reference = self._choose_clean_reference
            self.win._vsr_clear_clean_reference = self._clear_clean_reference
            self.win._vsr_preview_clean_reference = self._preview_clean_reference
            self.win._vsr_clean_reference_status = self.reference_status_var
            self.win._vsr_ocr_feedback = self.ocr_feedback_var
            self.win._vsr_ocr_overlay_ids = self.ocr_overlay_ids
            self.win._vsr_run_live_ocr = self._run_live_ocr_probe

            # Action row: Add another, Clear all, Save.
            actions = tk.Frame(self.win, bg=Theme.BG_OVERLAY)
            actions.pack(fill="x", padx=Theme.S_MD, pady=(Theme.S_SM, Theme.S_MD))







            ModernButton(actions, text=tr("Clear all"), command=self._clear_all,
                         style="ghost", size="sm", width=92).pack(side="left")
            if self.is_video:
                ModernButton(actions, text=tr("Add timed"), command=self._add_timed_regions,
                             style="secondary", size="sm", width=96).pack(
                                 side="left", padx=(Theme.S_SM, 0))
                ModernButton(actions, text=tr("Add keyframe"),
                             command=self._add_region_keyframe,
                             style="secondary", size="sm", width=108).pack(
                                 side="left", padx=(Theme.S_SM, 0))
                ModernButton(actions, text=tr("Save motion"),
                             command=self._commit_motion_track,
                             style="secondary", size="sm", width=108).pack(
                                 side="left", padx=(Theme.S_SM, 0))
            ModernButton(actions, text=tr("Save"), command=self._save_and_close,
                         style="primary", size="sm", width=92).pack(
                             side="right")
            ModernButton(actions, text=tr("Cancel"), command=self.win.destroy,
                         style="ghost", size="sm", width=92).pack(
                             side="right", padx=(0, Theme.S_SM))

            hint_frame = tk.Frame(self.win, bg=Theme.BG_OVERLAY)
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
                self.canvas.bind(sequence, self._transform_selected_region, add="+")
            self.win.bind("<Control-z>", self._undo_region_edit, add="+")
            self.win.bind("<Control-y>", self._redo_region_edit, add="+")
            self.win.bind("<Control-Shift-Z>", self._redo_region_edit, add="+")
            self._refresh_region_editor()
            self._update_history_buttons()

            # The cap must outlive this function so slider scrubs work
            # while the (non-blocking) modal stays open. Release it when
            # the window is closed instead of at function return.

            self.win.bind("<Destroy>", lambda e: self._release_cap() if e.widget is self.win else None)
            self.win.bind("<Escape>", lambda e: self.win.destroy())
            self.win.transient(self.root)
            self.win.grab_set()
        except Exception as e:
            logger.error(f"Region selector error: {e}")
            if self.cap is not None:
                try:
                    self.cap.release()
                except Exception:
                    logger.warning("Region selector capture release failed", exc_info=True)

    def _frame_to_pil(self, bgr):
        rgb = self._cv2.cvtColor(bgr, self._cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb).resize((self.disp_w, self.disp_h), Image.LANCZOS)

    def _draw_image(self, bgr):
        pil = self._frame_to_pil(bgr)
        self.canvas._photo = ImageTk.PhotoImage(pil)
        self.canvas.itemconfig(self.canvas_image_id, image=self.canvas._photo)

    def _draw_saved_rects(self):
        for rid in self.rect_ids:
            self.canvas.delete(rid)
        self.rect_ids.clear()
        for span in self.region_spans:
            x1, y1, x2, y2 = span["rect"]
            self.rect_ids.append(self.canvas.create_rectangle(
                x1 * self.scale, y1 * self.scale, x2 * self.scale, y2 * self.scale,
                outline=Theme.WARNING, width=2,
                stipple="gray50", fill=Theme.WARNING,
            ))
        for (x1, y1, x2, y2) in self.rects:
            self.rect_ids.append(self.canvas.create_rectangle(
                x1 * self.scale, y1 * self.scale, x2 * self.scale, y2 * self.scale,
                outline=Theme.GREEN_PRIMARY, width=2,
                stipple="gray25", fill=Theme.GREEN_PRIMARY,
            ))
        current_seconds = self.current_frame_index[0] / max(self.fps, 1.0)
        for shape in region_shapes_at(self.keyframe_tracks, current_seconds):
            if "rect" in shape:
                x1, y1, x2, y2 = shape["rect"]
                self.rect_ids.append(self.canvas.create_rectangle(
                    x1 * self.scale, y1 * self.scale, x2 * self.scale, y2 * self.scale,
                    outline=Theme.BLUE_PRIMARY, width=3,
                ))
            elif "polygon" in shape:
                coords = shape["polygon"]
                points = [coord * self.scale for coord in coords]
                self.rect_ids.append(self.canvas.create_polygon(
                    *points, outline=Theme.BLUE_PRIMARY,
                    fill="", width=3,
                ))
        for polygon in self.polygon_shapes:
            points = [coord * self.scale for coord in polygon]
            self.rect_ids.append(self.canvas.create_polygon(
                *points, outline=Theme.GREEN_PRIMARY,
                fill="", width=2,
            ))
        if self.polygon_points:
            points = []
            for px, py in self.polygon_points:
                points.extend((px * self.scale, py * self.scale))
            if len(points) >= 4:
                self.rect_ids.append(self.canvas.create_line(
                    *points, fill=Theme.BLUE_PRIMARY, width=2))
            for px, py in self.polygon_points:
                radius = 3
                self.rect_ids.append(self.canvas.create_oval(
                    px * self.scale - radius, py * self.scale - radius,
                    px * self.scale + radius, py * self.scale + radius,
                    fill=Theme.BLUE_PRIMARY, outline=""))

    def _seek_video(self, frame_idx: int):
        if not self.is_video or self.cap is None:
            return
        frame_idx = max(0, min(self.frame_count - 1, int(frame_idx)))
        self.current_frame_index[0] = frame_idx
        self.cap.set(self._cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, f = self.cap.read()
        if ok:
            self.ocr_probe_generation += 1
            self._clear_live_ocr_overlay()
            self.current_frame[0] = f
            self._draw_image(f)
            self._draw_saved_rects()

    def on_press(self, event):
        if self.shape_mode_var.get() == tr("Polygon"):
            self._record_history()
            px = min(self.orig_w, max(0, int(event.x / self.scale)))
            py = min(self.orig_h, max(0, int(event.y / self.scale)))
            self.polygon_points.append((px, py))
            self._draw_saved_rects()
            return
        self._clear_live_ocr_overlay()
        self.drag_start[0], self.drag_start[1] = event.x, event.y
        if self.drag_id[0]:
            self.canvas.delete(self.drag_id[0])
        self.drag_id[0] = self.canvas.create_rectangle(
            event.x, event.y, event.x, event.y,
            outline=Theme.BLUE_PRIMARY, width=2,
            stipple="gray12", fill=Theme.BLUE_PRIMARY,
        )

    def on_drag(self, event):
        if self.shape_mode_var.get() == tr("Polygon"):
            return
        if self.drag_id[0]:
            self.canvas.coords(self.drag_id[0], self.drag_start[0], self.drag_start[1], event.x, event.y)
            rect = self._drag_rect(event.x, event.y)
            if rect is not None:
                self._schedule_live_ocr_probe(rect)

    def on_release(self, event):
        if self.shape_mode_var.get() == tr("Polygon"):
            return
        rect = self._drag_rect(event.x, event.y)
        if rect is not None:
            self._record_history()
            self.rects.append(rect)
            self._refresh_region_editor(f"rect:{len(self.rects) - 1}")
            self._schedule_live_ocr_probe(rect)
        if self.drag_id[0] is not None:
            self.canvas.delete(self.drag_id[0])
            self.drag_id[0] = None
        self._draw_saved_rects()

    def _drag_rect(self, x: int, y: int):
        x1 = int(min(self.drag_start[0], x) / self.scale)
        y1 = int(min(self.drag_start[1], y) / self.scale)
        x2 = int(max(self.drag_start[0], x) / self.scale)
        y2 = int(max(self.drag_start[1], y) / self.scale)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(self.orig_w, x2), min(self.orig_h, y2)
        if (x2 - x1) <= 10 or (y2 - y1) <= 5:
            return None
        return (x1, y1, x2, y2)

    def _clear_live_ocr_overlay(self):
        for item_id in getattr(self, "ocr_overlay_ids", []):
            try:
                self.canvas.delete(item_id)
            except tk.TclError:
                pass
        if hasattr(self, "ocr_overlay_ids"):
            self.ocr_overlay_ids.clear()

    def _schedule_live_ocr_probe(self, rect):
        if getattr(self, "_live_region_ocr_enabled", True) is False:
            return
        self.ocr_pending_rect = tuple(rect)
        if self.ocr_probe_inflight or self.ocr_probe_after_id is not None:
            return
        self.ocr_probe_after_id = self.win.after(
            140, self._start_live_ocr_probe)

    def _run_live_ocr_probe(self, rect):
        """Test seam and explicit final-probe entry point."""
        if self.ocr_probe_after_id is not None:
            try:
                self.win.after_cancel(self.ocr_probe_after_id)
            except tk.TclError:
                pass
            self.ocr_probe_after_id = None
        self.ocr_pending_rect = tuple(rect)
        self._start_live_ocr_probe()

    def _start_live_ocr_probe(self):
        self.ocr_probe_after_id = None
        if self.ocr_probe_inflight or self.ocr_pending_rect is None:
            return
        rect = self.ocr_pending_rect
        self.ocr_pending_rect = None
        x1, y1, x2, y2 = rect
        frame = self.current_frame[0]
        crop = frame[y1:y2, x1:x2].copy()
        if crop.size == 0:
            return
        generation = self.ocr_probe_generation
        lang = (
            self.lang_var.get()
            if hasattr(self, "lang_var")
            else getattr(self.config, "detection_lang", "en")
        )
        threshold = getattr(self.config, "detection_threshold", 0.5)
        self.ocr_probe_inflight = True
        self.ocr_feedback_var.set(tr("Scanning this region for text..."))
        worker = partial(
            self._live_ocr_worker,
            crop,
            threshold,
            lang,
            rect,
            generation,
        )
        threading.Thread(target=worker, daemon=True).start()

    def _live_ocr_worker(self, crop, threshold, lang, rect, generation):
        error = None
        results = []
        try:
            from backend.detection import SubtitleDetector
            with self._detector_lock:
                if (
                    self._preview_detector is None
                    or self._preview_detector_lang != lang
                ):
                    self._preview_detector = SubtitleDetector(lang=lang)
                    self._preview_detector_lang = lang
                results = self._preview_detector.detect_with_confidence(
                    crop, threshold)
        except Exception as exc:
            logger.debug("Live region OCR preview failed", exc_info=True)
            error = str(exc)
        try:
            self.root.after(
                0,
                self._apply_live_ocr_probe,
                rect,
                generation,
                results,
                error,
            )
        except (RuntimeError, tk.TclError):
            pass

    def _apply_live_ocr_probe(self, rect, generation, results, error=None):
        self.ocr_probe_inflight = False
        try:
            alive = bool(self.win.winfo_exists())
        except tk.TclError:
            alive = False
        if not alive:
            return
        if generation == self.ocr_probe_generation:
            self._clear_live_ocr_overlay()
            x1, y1, _x2, _y2 = rect
            confidences = []
            for bx1, by1, bx2, by2, confidence in results:
                confidence = max(0.0, min(1.0, float(confidence)))
                confidences.append(confidence)
                sx1 = (x1 + bx1) * self.scale
                sy1 = (y1 + by1) * self.scale
                sx2 = (x1 + bx2) * self.scale
                sy2 = (y1 + by2) * self.scale
                self.ocr_overlay_ids.append(self.canvas.create_rectangle(
                    sx1, sy1, sx2, sy2,
                    outline=Theme.WARNING,
                    width=2,
                ))
                self.ocr_overlay_ids.append(self.canvas.create_text(
                    sx1 + 3,
                    max(8, sy1 - 7),
                    anchor="w",
                    text=f"{confidence:.0%}",
                    fill=Theme.WARNING,
                    font=f(Theme.F_META, "bold"),
                ))
            if error:
                self.ocr_feedback_var.set(
                    tr("OCR preview unavailable; the region can still be saved."))
            elif confidences:
                average = sum(confidences) / len(confidences)
                self.ocr_feedback_var.set(
                    tr("OCR found {count} region(s); average confidence {confidence:.0%}.").format(
                        count=len(confidences), confidence=average))
            else:
                self.ocr_feedback_var.set(
                    tr("No text found in this rectangle yet."))
        if self.ocr_pending_rect is not None:
            self._schedule_live_ocr_probe(self.ocr_pending_rect)

    def _finish_polygon(self) -> bool:
        if len(self.polygon_points) < 3:
            self._update_status(
                "A polygon needs at least three vertices", "warning")
            return False
        self._record_history()
        coords = []
        for px, py in self.polygon_points:
            coords.extend((px, py))
        self.polygon_shapes.append(coords)
        self.polygon_points.clear()
        self._draw_saved_rects()
        self._refresh_region_editor(f"polygon:{len(self.polygon_shapes) - 1}")
        return True

    def _on_slider(self, value):
        try:
            idx = int(float(value))
        except (TypeError, ValueError):
            return
        self._seek_video(idx)
        secs = idx / self.fps
        hh = int(secs // 3600)
        mm = int((secs % 3600) // 60)
        ss = int(secs % 60)
        self.ts_label.config(text=f"{hh:02d}:{mm:02d}:{ss:02d}")

    def _editable_region_records(self):
        records = []
        for index, rect in enumerate(self.rects):
            records.append({
                "key": f"rect:{index}",
                "label": tr("Rectangle {index}").format(index=index + 1),
                "source": "rect", "index": index,
                "shape": {"rect": list(rect)}, "timing": "none",
            })
        for index, span in enumerate(self.region_spans):
            records.append({
                "key": f"span:{index}",
                "label": tr("Timed region {index}").format(index=index + 1),
                "source": "span", "index": index,
                "shape": {"rect": list(span["rect"])},
                "timing": "range",
                "start": float(span.get("start", 0.0)),
                "end": float(span.get("end", 0.0)),
            })
        for index, polygon in enumerate(self.polygon_shapes):
            records.append({
                "key": f"polygon:{index}",
                "label": tr("Polygon {index}").format(index=index + 1),
                "source": "polygon", "index": index,
                "shape": {"polygon": list(polygon)}, "timing": "none",
            })
        for index, keyframe in enumerate(self.pending_keyframes):
            kind = "rect" if "rect" in keyframe else "polygon"
            records.append({
                "key": f"pending:{index}",
                "label": tr("Pending keyframe {index}").format(index=index + 1),
                "source": "pending", "index": index,
                "shape": {kind: list(keyframe[kind])}, "timing": "point",
                "start": float(keyframe.get("time", 0.0)), "end": 0.0,
            })
        for track_index, track in enumerate(self.keyframe_tracks):
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

    def _editor_snapshot(self):
        return {
            "rects": list(self.rects),
            "region_spans": list(self.region_spans),
            "keyframe_tracks": list(self.keyframe_tracks),
            "pending_keyframes": list(self.pending_keyframes),
            "polygon_shapes": list(self.polygon_shapes),
            "polygon_points": list(self.polygon_points),
            "start": self.start_var.get(),
            "end": self.end_var.get(),
            "start_frame": self.start_frame_var.get(),
            "end_frame": self.end_frame_var.get(),
            "selected": self.selected_region_key[0],
        }

    def _record_history(self):
        self.history.record(self._editor_snapshot())
        self._update_history_buttons()

    def _restore_editor_snapshot(self, snapshot):
        self.rects[:] = [tuple(rect) for rect in snapshot["rects"]]
        self.region_spans[:] = snapshot["region_spans"]
        self.keyframe_tracks[:] = snapshot["keyframe_tracks"]
        self.pending_keyframes[:] = snapshot["pending_keyframes"]
        self.polygon_shapes[:] = [list(shape) for shape in snapshot["polygon_shapes"]]
        self.polygon_points[:] = [tuple(point) for point in snapshot["polygon_points"]]
        self.start_var.set(snapshot["start"])
        self.end_var.set(snapshot["end"])
        self.start_frame_var.set(snapshot["start_frame"])
        self.end_frame_var.set(snapshot["end_frame"])
        self.selected_region_key[0] = snapshot.get("selected")
        self._draw_saved_rects()
        self._refresh_region_editor(self.selected_region_key[0])
        self._update_history_buttons()

    def _update_history_buttons(self):
        undo_button = self.history_buttons.get("undo")
        redo_button = self.history_buttons.get("redo")
        if undo_button is not None:
            undo_button.set_enabled(
                self.history.can_undo, tr("No region edit to undo"))
        if redo_button is not None:
            redo_button.set_enabled(
                self.history.can_redo, tr("No region edit to redo"))

    def _set_time_fields(self, start=None, end=None):
        if not self.is_video:
            return
        if start is None:
            self.start_var.set("")
            self.start_frame_var.set("")
        else:
            self.start_var.set(f"{float(start):g}")
            self.start_frame_var.set(str(seconds_to_frame(start, self.fps)))
        if not end:
            self.end_var.set("")
            self.end_frame_var.set("")
        else:
            self.end_var.set(f"{float(end):g}")
            self.end_frame_var.set(str(seconds_to_frame(end, self.fps)))
        self.time_input_source.update(start="seconds", end="seconds")

    def _record_by_key(self, key):
        return next(
            (record for record in self._editable_region_records()
             if record["key"] == key),
            None,
        )

    def _set_clean_reference_controls(self, enabled):
        reason = tr("Clean references attach only to timed regions")
        for button in self.reference_buttons.values():
            button.set_enabled(bool(enabled), reason)
        self.reference_alignment_picker.configure(
            state="readonly" if enabled else "disabled")
        self.reference_confidence_entry.configure(
            state="normal" if enabled else "disabled")
        self.reference_color_toggle.set_enabled(bool(enabled))

    def _selected_clean_reference_span(self):
        record = self._record_by_key(self.selected_region_key[0])
        if record is None or record.get("source") != "span":
            return None, None
        return record, self.region_spans[record["index"]]

    def _load_clean_reference_controls(self, record):
        if record is None or record.get("source") != "span":
            self._set_clean_reference_controls(False)
            self.reference_path_var.set("")
            self.reference_status_var.set(tr(
                "Select a timed region to attach a clean reference."))
            return
        self._set_clean_reference_controls(True)
        span = self.region_spans[record["index"]]
        spec = span.get("clean_reference") or {}
        self.reference_path_var.set(
            Path(spec.get("path", "")).name if spec else tr("None"))
        mode_labels = {
            "auto": tr("Auto"),
            "translation": tr("Translation"),
            "homography": tr("Homography"),
        }
        self.reference_alignment_var.set(
            mode_labels.get(spec.get("alignment", "auto"), tr("Auto")))
        self.reference_color_match_var.set(
            bool(spec.get("color_match", True)))
        self.reference_confidence_var.set(
            f"{float(spec.get('min_confidence', 0.75)):g}")
        self.reference_status_var.set(
            tr("Ready to preview alignment.") if spec else tr(
                "Choose a same-size clean image for this timed region."))

    def _save_clean_reference_options(self):
        record, span = self._selected_clean_reference_span()
        if span is None or not span.get("clean_reference"):
            return False
        mode_values = {
            tr("Auto"): "auto",
            tr("Translation"): "translation",
            tr("Homography"): "homography",
        }
        try:
            confidence = float(self.reference_confidence_var.get().strip())
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
                self.reference_alignment_var.get(), "auto"),
            "color_match": self.reference_color_match_var.get(),
            "min_confidence": confidence,
        })
        if updated != span.get("clean_reference"):
            self._record_history()
            span["clean_reference"] = updated
        self._load_clean_reference_controls(record)
        return True

    def _choose_clean_reference(self):
        record, span = self._selected_clean_reference_span()
        if span is None:
            self._update_status(
                tr("Select a timed region before choosing a clean reference"),
                "warning",
            )
            return
        path = filedialog.askopenfilename(
            parent=self.win,
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
        if reference.shape[:2] != (self.orig_h, self.orig_w):
            self._update_status(
                tr("Clean reference must be {width}x{height}").format(
                    width=self.orig_w, height=self.orig_h),
                "warning",
            )
            return
        from backend.reference_fill import normalize_clean_reference
        self._record_history()
        span["clean_reference"] = normalize_clean_reference({
            "path": path,
            "alignment": "auto",
            "color_match": True,
            "min_confidence": 0.75,
        })
        self._load_clean_reference_controls(record)
        self._update_status(
            tr("Clean reference attached to the timed region"),
            "success",
        )

    def _clear_clean_reference(self):
        record, span = self._selected_clean_reference_span()
        if span is None or "clean_reference" not in span:
            return
        self._record_history()
        span.pop("clean_reference", None)
        self._load_clean_reference_controls(record)
        self._update_status(tr("Clean reference cleared"), "info")

    def _preview_clean_reference(self):
        record, span = self._selected_clean_reference_span()
        if span is None or not span.get("clean_reference"):
            self._update_status(
                tr("Choose a clean reference for the selected timed region"),
                "warning",
            )
            return
        if not self._save_clean_reference_options():
            return
        try:
            result = self._render_clean_reference_preview(
                self.current_frame[0], span)
        except Exception as exc:
            self._update_status(str(exc), "warning")
            return
        self._draw_image(result.composite if result.accepted else result.aligned)
        self._draw_saved_rects()
        status = tr(
            "{method} alignment: {confidence:.1%}").format(
                method=result.method.title(),
                confidence=result.confidence,
            )
        if result.accepted:
            status += tr("; color delta BGR {delta}").format(
                delta=", ".join(f"{value:g}" for value in result.color_delta))
            self.reference_status_var.set(status)
            self._update_status(tr("Clean reference preview ready"), "success")
        else:
            self.reference_status_var.set(
                status + tr("; would fall back to normal inpainting"))
            self._update_status(result.reason, "warning")

    def _load_selected_region(self):
        records = self._editable_region_records()
        selected_label = self.selected_region_var.get()
        record = next(
            (item for item in records if item["label"] == selected_label),
            None,
        )
        if record is None:
            record = self._record_by_key(self.selected_region_key[0])
        if record is None:
            self.selected_region_key[0] = None
            for variable in self.geometry_vars.values():
                variable.set("")
            for entry in self.geometry_entries.values():
                entry.configure(state="disabled")
            self._load_clean_reference_controls(None)
            return
        self.selected_region_key[0] = record["key"]
        shape = record["shape"]
        if "rect" in shape:
            x1, y1, x2, y2 = shape["rect"]
            self.geometry_vars["x"].set(str(x1))
            self.geometry_vars["y"].set(str(y1))
            self.geometry_vars["width"].set(str(x2 - x1))
            self.geometry_vars["height"].set(str(y2 - y1))
            self.geometry_vars["vertices"].set("")
            for key in ("x", "y", "width", "height"):
                self.geometry_entries[key].configure(state="normal")
            self.geometry_entries["vertices"].configure(state="disabled")
        else:
            for key in ("x", "y", "width", "height"):
                self.geometry_vars[key].set("")
                self.geometry_entries[key].configure(state="disabled")
            self.geometry_vars["vertices"].set(
                format_polygon_vertices(shape["polygon"]))
            self.geometry_entries["vertices"].configure(state="normal")
        if record["timing"] == "none":
            self._set_time_fields()
        else:
            self._set_time_fields(record.get("start"), record.get("end"))
        self._load_clean_reference_controls(record)

    def _refresh_region_editor(self, prefer_key=None):
        records = self._editable_region_records()
        values = [record["label"] for record in records]
        self.region_picker.configure(values=values)
        key = prefer_key or self.selected_region_key[0]
        selected = next(
            (record for record in records if record["key"] == key),
            records[-1] if records else None,
        )
        if selected is None:
            self.selected_region_var.set("")
            self.selected_region_key[0] = None
        else:
            self.selected_region_key[0] = selected["key"]
            self.selected_region_var.set(selected["label"])
        self._load_selected_region()

    def _set_record_shape(self, record, shape):
        source = record["source"]
        index = record["index"]
        kind = "rect" if "rect" in shape else "polygon"
        values = list(shape[kind])
        if source == "rect":
            self.rects[index] = tuple(values)
        elif source == "span":
            self.region_spans[index]["rect"] = tuple(values)
        elif source == "polygon":
            self.polygon_shapes[index] = values
        elif source == "pending":
            self.pending_keyframes[index].pop("rect", None)
            self.pending_keyframes[index].pop("polygon", None)
            self.pending_keyframes[index][kind] = values
        elif source == "track":
            keyframe = self.keyframe_tracks[record["track"]]["keyframes"][index]
            keyframe.pop("rect", None)
            keyframe.pop("polygon", None)
            keyframe[kind] = values

    def _seconds_from_fields(self, which):
        seconds_var = self.start_var if which == "start" else self.end_var
        frame_var = self.start_frame_var if which == "start" else self.end_frame_var
        seconds_raw = seconds_var.get().strip()
        frame_raw = frame_var.get().strip()
        use_frame = (
            self.time_input_source[which] == "frame"
            or (not seconds_raw and bool(frame_raw))
        )
        raw = frame_raw if use_frame else seconds_raw
        if not raw:
            return None
        if use_frame:
            value = frame_to_seconds(raw, self.fps)
            seconds_var.set(f"{value:g}")
        else:
            try:
                value = float(raw)
            except (TypeError, ValueError) as exc:
                raise ValueError(tr("Time must be a number")) from exc
            if not math.isfinite(value) or value < 0:
                raise ValueError(tr("Time must be zero or greater"))
            frame_var.set(str(seconds_to_frame(value, self.fps)))
        return value

    def _sync_time_field(self, which, source):
        if not self.is_video:
            return
        self.time_input_source[which] = source
        try:
            self._seconds_from_fields(which)
        except (TypeError, ValueError):
            return

    def _apply_numeric_region_edit(self):
        record = self._record_by_key(self.selected_region_key[0])
        if record is None:
            self._update_status(tr("Select or draw a region before editing it"), "warning")
            return False
        try:
            if "rect" in record["shape"]:
                rect = rect_from_xywh(
                    self.geometry_vars["x"].get(), self.geometry_vars["y"].get(),
                    self.geometry_vars["width"].get(), self.geometry_vars["height"].get(),
                    self.orig_w, self.orig_h,
                )
                shape = {"rect": list(rect)}
            else:
                shape = {"polygon": parse_polygon_vertices(
                    self.geometry_vars["vertices"].get(), self.orig_w, self.orig_h)}
            start_s = self._seconds_from_fields("start") if self.is_video else None
            end_s = self._seconds_from_fields("end") if self.is_video else None
            if end_s is not None and (start_s is None or end_s <= start_s):
                raise ValueError(tr("Region end must be after its start"))
            if record["timing"] == "point" and end_s is not None:
                raise ValueError(tr("A motion keyframe uses one frame, not an end frame"))
            if record["timing"] == "point" and start_s is not None:
                if record["source"] == "pending":
                    peers = self.pending_keyframes
                else:
                    peers = self.keyframe_tracks[record["track"]]["keyframes"]
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

        self._record_history()
        self._set_record_shape(record, shape)
        if record["timing"] == "range":
            self.region_spans[record["index"]]["start"] = start_s or 0.0
            self.region_spans[record["index"]]["end"] = end_s or 0.0
        elif record["timing"] == "point" and start_s is not None:
            if record["source"] == "pending":
                self.pending_keyframes[record["index"]]["time"] = start_s
            else:
                track = self.keyframe_tracks[record["track"]]
                track["keyframes"][record["index"]]["time"] = start_s
                times = [float(item["time"]) for item in track["keyframes"]]
                track["start"], track["end"] = min(times), max(times)
        self._draw_saved_rects()
        self._refresh_region_editor(record["key"])
        self._update_history_buttons()
        self._update_status(tr("Applied precise region edit"), "success")
        return True

    def _transform_selected_region(self, event):
        if isinstance(event.widget, (tk.Entry, ttk.Combobox, tk.Scale)):
            return None
        direction = str(event.keysym)
        if direction not in {"Left", "Right", "Up", "Down"}:
            return None
        record = self._record_by_key(self.selected_region_key[0])
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
            record["shape"], frame_width=self.orig_w, frame_height=self.orig_h,
            dx=dx, dy=dy, dw=dw, dh=dh,
        )
        if shape == record["shape"]:
            return "break"
        self._record_history()
        self._set_record_shape(record, shape)
        self._draw_saved_rects()
        self._refresh_region_editor(record["key"])
        self._update_history_buttons()
        return "break"

    def _undo_region_edit(self, event=None):
        if event is not None and isinstance(event.widget, tk.Entry):
            return None
        snapshot = self.history.undo(self._editor_snapshot())
        if snapshot is None:
            return "break"
        self._restore_editor_snapshot(snapshot)
        self._update_status(tr("Undid region edit"), "info")
        return "break"

    def _redo_region_edit(self, event=None):
        if event is not None and isinstance(event.widget, tk.Entry):
            return None
        snapshot = self.history.redo(self._editor_snapshot())
        if snapshot is None:
            return "break"
        self._restore_editor_snapshot(snapshot)
        self._update_status(tr("Redid region edit"), "info")
        return "break"

    def _clear_all(self):
        self._record_history()
        self.rects.clear()
        self.region_spans.clear()
        self.keyframe_tracks.clear()
        self.pending_keyframes.clear()
        self.polygon_shapes.clear()
        self.polygon_points.clear()
        self.span_summary_var.set(tr("Optional") if self.is_video else "")
        self._draw_saved_rects()
        self._refresh_region_editor()

    def _add_region_keyframe(self) -> bool:
        if not self.is_video:
            return False
        if self.polygon_points and not self._finish_polygon():
            return False
        shape_count = len(self.rects) + len(self.polygon_shapes)
        if shape_count != 1:
            self._update_status(
                "Draw exactly one rectangle or polygon for this keyframe",
                "warning",
            )
            return False
        if self.polygon_shapes:
            shape = {"polygon": list(self.polygon_shapes[0])}
        else:
            shape = {"rect": list(self.rects[0])}
        if self.pending_keyframes:
            first = self.pending_keyframes[0]
            kind = "rect" if "rect" in first else "polygon"
            if kind not in shape or len(first[kind]) != len(shape[kind]):
                self._update_status(
                    "Keyframes in one motion track need the same shape and vertex count",
                    "warning",
                )
                return False
        seconds = self.current_frame_index[0] / max(self.fps, 1.0)
        self._record_history()
        keyframe = {"time": seconds, **shape}
        self.pending_keyframes[:] = [
            item for item in self.pending_keyframes
            if abs(float(item["time"]) - seconds) > 1e-9
        ]
        self.pending_keyframes.append(keyframe)
        self.pending_keyframes.sort(key=lambda item: float(item["time"]))
        self.rects.clear()
        self.polygon_shapes.clear()
        self.span_summary_var.set(
            f"{len(self.pending_keyframes)} pending keyframe"
            f"{'s' if len(self.pending_keyframes) != 1 else ''}"
        )
        self._draw_saved_rects()
        self._refresh_region_editor(f"pending:{len(self.pending_keyframes) - 1}")
        self._update_status(
            f"Added motion keyframe at {seconds:.3f} seconds", "success")
        return True

    def _commit_motion_track(self) -> bool:
        if len(self.pending_keyframes) < 2:
            self._update_status(
                "Add at least two keyframes before saving a motion track",
                "warning",
            )
            return False
        track = normalize_region_keyframe_tracks([{
            "start": self.pending_keyframes[0]["time"],
            "end": self.pending_keyframes[-1]["time"],
            "keyframes": list(self.pending_keyframes),
        }])
        if not track:
            self._update_status(
                "The motion keyframes could not be normalized", "warning")
            return False
        self._record_history()
        self.keyframe_tracks.append(track[0])
        self.pending_keyframes.clear()
        self.span_summary_var.set(
            f"{len(self.keyframe_tracks)} motion track"
            f"{'s' if len(self.keyframe_tracks) != 1 else ''}"
        )
        self._draw_saved_rects()
        self._refresh_region_editor(f"track:{len(self.keyframe_tracks) - 1}:0")
        self._update_status("Saved interpolated motion track", "success")
        return True

    def _parse_time_inputs(self):
        start_raw = self.start_var.get().strip() if self.is_video else ""
        end_raw = self.end_var.get().strip() if self.is_video else ""
        has_time = bool(
            start_raw or end_raw
            or self.start_frame_var.get().strip()
            or self.end_frame_var.get().strip()
        )
        try:
            start_s = (self._seconds_from_fields("start")
                       if start_raw or self.start_frame_var.get().strip() else None)
            end_s = (self._seconds_from_fields("end")
                     if end_raw or self.end_frame_var.get().strip() else None)
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

    def _add_timed_regions(self, close_on_empty: bool = False) -> bool:
        parsed = self._parse_time_inputs()
        if parsed is None:
            return False
        has_time, start_s, end_s = parsed
        if not self.rects:
            if close_on_empty:
                return True
            self._update_status("Draw a region before adding a timed range", "warning")
            return False
        if self.polygon_shapes or self.polygon_points:
            self._update_status(
                "Polygons are saved through motion keyframes", "warning")
            return False
        if not self.is_video:
            return False
        if not has_time and not self.region_spans:
            if not close_on_empty:
                self._update_status(
                    "Enter a start or end second for a timed range",
                    "warning",
                )
            return True if close_on_empty else False
        self._record_history()
        for rect in self.rects:
            self.region_spans.append({
                "rect": tuple(rect),
                "start": start_s,
                "end": end_s,
            })
        self.rects.clear()
        self.start_var.set("")
        self.end_var.set("")
        self.span_summary_var.set(
            f"{len(self.region_spans)} timed region"
            f"{'s' if len(self.region_spans) != 1 else ''}"
        )
        self._draw_saved_rects()
        self._refresh_region_editor(
            f"span:{len(self.region_spans) - 1}" if self.region_spans else None)
        return True

    def _save_and_close(self):
        if self.pending_keyframes and not self._commit_motion_track():
            return
        if self.is_video and not self._add_timed_regions(close_on_empty=True):
            return
        spans = _coerce_region_span_list(self.region_spans) or []
        tracks = normalize_region_keyframe_tracks(self.keyframe_tracks) or []
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
        elif self.rects:
            self.config.subtitle_areas = [tuple(r) for r in self.rects]
            self.config.subtitle_area = self.rects[0]
            self.config.subtitle_region_spans = None
            self.config.subtitle_region_keyframes = None
            self._update_status(
                f"Saved {len(self.rects)} subtitle region{'s' if len(self.rects) != 1 else ''}",
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
        self.win.destroy()

    def _release_cap(self):
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass



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
        """Open the non-blocking region-selector window."""
        RegionSelectorWindow(self).open()

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
