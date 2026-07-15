from __future__ import annotations

import logging
import threading
import time
import numpy as np
from pathlib import Path

try:
    import tkinter as tk
except ImportError:  # pragma: no cover - tkinter is optional in headless imports
    pass

try:
    from PIL import Image, ImageTk, ImageDraw
    PIL_AVAILABLE = True
except ImportError:  # pragma: no cover - preview features degrade without Pillow
    PIL_AVAILABLE = False

from gui.theme import Theme, f, prefers_reduced_motion
from gui.config import (
    ProcessingConfig, ProcessingStatus, QueueItem,
    status_ui,
)
from gui.utils import (
    _format_soft_subtitle_summary, format_quality_report, is_image_file, is_video_file,
)
from gui.widgets import (
    ModernButton,
)
from backend.i18n import tr
from backend.region_keyframes import region_shapes_at
from backend.safe_image import safe_imread

logger = logging.getLogger(__name__)


class PreviewControllerMixin:
    """Focused controller methods mixed into VideoSubtitleRemoverApp."""

    def _start_throbber(self):
        """Animate the preview area with a shimmer placeholder and moving dots
        to signal a background task in progress."""
        self._stop_throbber()
        self._throbber_phase = 0
        self._throbber_tick(animate=not prefers_reduced_motion())

    def _stop_throbber(self):
        tid = getattr(self, "_throbber_id", None)
        if tid:
            try:
                self.root.after_cancel(tid)
            except Exception:
                pass
            self._throbber_id = None

    def _throbber_tick(self, *, animate: bool = True):
        if not PIL_AVAILABLE:
            self._preview_label.config(
                text=tr("Detecting") + "." * (self._throbber_phase % 4))
            try:
                if not animate:
                    return
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
            d.text((cx - 42, cy + 22), tr("DETECTING"),
                   fill=self._hex_to_rgb(Theme.TEXT_MUTED))
            self._preview_photo = ImageTk.PhotoImage(base)
            self._preview_label.config(image=self._preview_photo, text="")
            self._throbber_phase += 1
            try:
                if not animate:
                    return
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
                self.preview_title_label.config(
                    text=tr("Live preview: {file_name}").format(file_name=file_name))
                self.preview_meta_label.config(
                    text=tr("Frame {current}/{total} ({percent}%)").format(
                        current=cur_idx, total=total, percent=pct))
        except Exception:
            pass

    def _set_preview_placeholder(self, title: str, body: str):
        """Show the empty-state preview guidance with a subtle illustration."""
        self._stop_throbber()
        self.preview_title_label.config(text=tr(title))
        self.preview_meta_label.config(text=tr(body))
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

    def _set_preview_unavailable(
        self,
        title: str,
        body: str,
        *,
        label: str = "Preview unavailable",
        tone: str = "warning",
    ):
        """Show a calm, actionable preview failure state."""
        self._stop_throbber()
        self._preview_photo = None
        self.preview_title_label.config(text=tr(title))
        self.preview_meta_label.config(text=tr(body))
        self._preview_label.config(text=tr(label), image="")

        tone_map = {
            "error": (tr("Needs attention"), Theme.ERROR, Theme.ERROR_BG),
            "warning": (tr("Needs attention"), Theme.WARNING, Theme.WARNING_BG),
            "info": (tr("Info"), Theme.INFO, Theme.INFO_BG),
        }
        chip_text, chip_fg, chip_bg = tone_map.get(
            tone,
            (tr("Waiting"), Theme.TEXT_MUTED, Theme.BG_TERTIARY),
        )
        try:
            self.preview_status_chip.config(text=chip_text, fg=chip_fg, bg=chip_bg)
        except Exception:
            pass

    def _update_preview_actions(self):
        """Enable preview tools only when they make sense for the selection."""
        if not hasattr(self, "preview_mask_btn"):
            return
        selected = self._get_selected_queue_item()
        can_preview = bool(selected and PIL_AVAILABLE)
        if self.is_processing:
            unavailable_reason = tr("Preview tools are locked while a batch is running.")
        elif not selected:
            unavailable_reason = tr("Select a queued item to enable preview tools.")
        elif not PIL_AVAILABLE:
            unavailable_reason = tr("Install Pillow to enable image preview tools.")
        else:
            unavailable_reason = ""

        if hasattr(self, "preview_region_btn"):
            self.preview_region_btn.set_enabled(
                not self.is_processing,
                reason=tr("Wait for the active batch to finish before editing regions."),
            )
        self.preview_mask_btn.set_enabled(
            bool(selected) and not self.is_processing,
            reason=unavailable_reason,
        )
        if hasattr(self, "preview_correction_btn"):
            self.preview_correction_btn.set_enabled(
                bool(selected) and not self.is_processing and PIL_AVAILABLE,
                reason=unavailable_reason,
            )
        self.preview_zoom_btn.set_enabled(
            can_preview,
            reason=unavailable_reason,
        )
        if hasattr(self, "preview_inpaint_btn"):
            self.preview_inpaint_btn.set_enabled(
                bool(selected) and not self.is_processing,
                reason=unavailable_reason,
            )
        if hasattr(self, "preview_ab_btn"):
            ab_ready = bool(
                selected
                and selected.status == ProcessingStatus.COMPLETE
                and Path(selected.output_path).exists()
            )
            if ab_ready:
                ab_reason = ""
            elif self.is_processing:
                ab_reason = tr("Wait for the active batch to finish before comparing output.")
            elif not selected:
                ab_reason = tr("Select a completed queue item to compare source and output.")
            elif selected.status != ProcessingStatus.COMPLETE:
                ab_reason = tr("Finish processing this item before using A/B compare.")
            else:
                ab_reason = tr("The cleaned output file is not available on disk.")
            self.preview_ab_btn.set_enabled(ab_ready, reason=ab_reason)
        editing_region = bool(getattr(self, "_preview_region_editor_state", None))
        self._preview_label.config(
            cursor="crosshair" if editing_region else ("hand2" if can_preview else "")
        )

        if hasattr(self, "preview_action_hint"):
            if self.is_processing:
                hint = tr("Preview tools are locked while the batch is running.")
                hint_fg = Theme.WARNING
            elif not selected:
                hint = tr("Select a queue item to enable mask review, test cleanup, and zoom.")
                hint_fg = Theme.TEXT_MUTED
            elif not PIL_AVAILABLE:
                hint = tr("Install Pillow to enable previews and visual inspection tools.")
                hint_fg = Theme.WARNING
            elif (
                selected.status == ProcessingStatus.COMPLETE
                and Path(selected.output_path).exists()
            ):
                hint = tr("Preview tools are ready. Use A/B compare to inspect the cleaned output.")
                hint_fg = Theme.SUCCESS
            else:
                hint = tr("Preview tools are ready. Review the mask or test cleanup before starting.")
                hint_fg = Theme.TEXT_SECONDARY
            self.preview_action_hint.config(text=hint, fg=hint_fg)

        if selected:
            badge = status_ui(selected.status)
            self.preview_status_chip.config(
                text=badge["label"],
                fg=badge["color"],
                bg=badge["bg"],
            )
        else:
            self.preview_status_chip.config(
                text=tr("Waiting"),
                fg=Theme.TEXT_MUTED,
                bg=Theme.BG_TERTIARY,
            )

    def _open_selected_mask_preview(self):
        item = self._get_selected_queue_item(fallback_to_first=True)
        if item:
            self._show_preview(item, show_mask=True)

    def _probe_language_from_preview(self):
        """Auto-detect subtitle language from the selected queue item."""
        item = self._get_selected_queue_item(fallback_to_first=True)
        source = None
        if item:
            source = item.file_path
        else:
            for q in self.queue:
                source = q.file_path
                break
        if not source:
            self._update_status(
                tr("Add a file to the queue first"), "warning")
            return
        self._update_status(tr("Detecting language..."), "info")

        def _probe():
            try:
                import cv2 as _cv2
                if is_video_file(source):
                    cap = _cv2.VideoCapture(source)
                    try:
                        ok, frame = cap.read()
                    finally:
                        cap.release()
                    if not ok or frame is None:
                        return None
                else:
                    frame = safe_imread(source)
                if frame is None:
                    return None
                active_regions = self._active_timed_region_rects(
                    getattr(self.config, "subtitle_region_spans", None), 0.0,
                    getattr(self.config, "subtitle_region_keyframes", None))
                region = (
                    active_regions[0] if active_regions
                    else getattr(self.config, "subtitle_area", None)
                )
                from backend.detection import probe_language
                return probe_language(frame, region=region)
            except Exception as exc:
                logger.debug(f"Language probe failed: {exc}")
                return None

        def _on_result(result):
            if result is None:
                self._update_status(
                    tr("Could not detect language from this frame"), "warning")
                return
            lang, conf, script = result
            for label, (code, _name) in zip(
                self._lang_labels, self._lang_display
            ):
                if code == lang:
                    self._lang_display_var.set(label)
                    self.lang_var.set(lang)
                    break
            self._update_status(
                f"Detected {script} script, suggested language: {lang} "
                f"(confidence {conf:.0%})",
                "success",
            )

        def _worker():
            result = _probe()
            try:
                self.root.after(0, _on_result, result)
            except (RuntimeError, tk.TclError):
                pass

        threading.Thread(target=_worker, daemon=True).start()

    def _open_ab_scrubber(self):
        """RM-30: open a Toplevel that lets the user scrub frames AND
        wipe a vertical seam left/right to compare the original vs the
        cleaned output side-by-side at any frame.

        The window opens both video captures, holds them open for the
        duration of the modal, and composes a single image per scrub.
        """
        item = self._get_selected_queue_item(fallback_to_first=True)
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
            cap_a.release()
            cap_b.release()
            self._update_status("Could not open input/output for compare", "warning")
            return

        n_a = max(1, int(cap_a.get(_cv2.CAP_PROP_FRAME_COUNT)))
        n_b = max(1, int(cap_b.get(_cv2.CAP_PROP_FRAME_COUNT)))
        n_total = max(1, min(n_a, n_b))
        fps = cap_a.get(_cv2.CAP_PROP_FPS) or 30.0
        if fps <= 0:
            fps = 30.0
        max_w = min(1024, int(self.root.winfo_screenwidth() * 0.7))
        max_h = min(576, int(self.root.winfo_screenheight() * 0.6))

        try:
            self._open_ab_scrubber_window(
                in_path, cap_a, cap_b, n_a, n_b, n_total, fps, max_w, max_h)
        except Exception:
            cap_a.release()
            cap_b.release()
            raise

    def _open_ab_scrubber_window(self, in_path, cap_a, cap_b,
                                  n_a, n_b, n_total, fps, max_w, max_h):
        import cv2 as _cv2

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
            # Draw a 2-pixel accent seam line for the wipe boundary. Derive it
            # from the theme focus token so it brightens in high-contrast mode
            # instead of a fixed green.
            if 0 < seam < dw:
                _r, _g, _b = self._hex_to_rgb(Theme.BORDER_FOCUS)
                _cv2.line(composite, (seam, 0), (seam, dh - 1), (_b, _g, _r), 2)
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
            hh = int(secs // 3600)
            mm = int((secs % 3600) // 60)
            ss = int(secs % 60)
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
                cap_a.release()
                cap_b.release()
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
        item = self._get_selected_queue_item(fallback_to_first=True)
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
                    frame = safe_imread(source_path)
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
                    self.root.after(
                        0,
                        lambda: self._set_preview_unavailable(
                            "Test cleanup unavailable",
                            "The selected file could not be read. Verify the file path and add it again.",
                            label="No frame available",
                        ),
                    )
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
                    subtitle_region_spans=snapshot_cfg.subtitle_region_spans,
                    subtitle_region_keyframes=(
                        snapshot_cfg.subtitle_region_keyframes),
                    mask_dilate_px=snapshot_cfg.mask_dilate_px,
                    mask_feather_px=snapshot_cfg.mask_feather_px,
                    tbe_enable=snapshot_cfg.tbe_enable,
                )
                remover = _Remover(backend_cfg)
                # Single-frame inpaint -- detect, build mask, inpaint.
                timed_fixed = self._active_timed_region_rects(
                    getattr(snapshot_cfg, "subtitle_region_spans", None), 0.0,
                    getattr(snapshot_cfg, "subtitle_region_keyframes", None))
                timed_configured = bool(
                    getattr(snapshot_cfg, "subtitle_region_spans", None)
                    or getattr(
                        snapshot_cfg, "subtitle_region_keyframes", None))
                active_shapes = remover._fixed_region_shapes(0.0) or []
                fixed = timed_fixed or (
                    None if timed_configured else (
                        snapshot_cfg.subtitle_areas
                        or ([snapshot_cfg.subtitle_area]
                            if snapshot_cfg.subtitle_area else None)
                    )
                )
                if fixed:
                    boxes = list(fixed)
                elif (timed_configured
                      and getattr(snapshot_cfg, "sttn_skip_detection", False)):
                    boxes = []
                else:
                    boxes = remover.detector.detect(
                        frame, snapshot_cfg.detection_threshold)
                if not boxes and not active_shapes:
                    # No detection -- show the source with a hint.
                    pil = Image.fromarray(_cv2.cvtColor(frame, _cv2.COLOR_BGR2RGB))
                    self.root.after(0, lambda: self._apply_inpaint_preview(
                        pil, "No text detected on the first frame", request_id, item.id))
                    return
                mask = remover._create_mask(frame.shape, boxes)
                mask = remover._apply_polygon_region_shapes(mask, active_shapes)
                [filled] = remover.inpainter.inpaint([frame], [mask])
                pil = Image.fromarray(_cv2.cvtColor(filled, _cv2.COLOR_BGR2RGB))
                meta = (f"Cleanup preview using {snapshot_cfg.mode.value}; "
                        f"{len(boxes)} region{'s' if len(boxes) != 1 else ''} masked.")
                self.root.after(0, lambda: self._apply_inpaint_preview(
                    pil, meta, request_id, item.id))
            except Exception:
                logger.warning("Inpaint preview failed", exc_info=True)
                self.root.after(
                    0,
                    lambda: self._set_preview_unavailable(
                        "Test cleanup failed",
                        "The cleanup preview could not be rendered. Check the activity log, then try Review mask or Set region.",
                        label="Cleanup preview failed",
                        tone="error",
                    ),
                )

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
                frame = safe_imread(item.file_path)
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
        ModernButton(header, text=tr("Close"), width=86,
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

    def _start_preview_region_editor(self, item: QueueItem) -> bool:
        """Render the selected source frame as an inline draggable editor."""
        if self.is_processing:
            self._update_status(
                "Stop the active batch before changing the subtitle region",
                "warning",
            )
            return True
        if not PIL_AVAILABLE:
            self._update_status(
                "Install Pillow to enable preview region editing",
                "warning",
            )
            return False

        try:
            import cv2 as _cv2

            raw_frame = None
            if is_image_file(item.file_path):
                raw_frame = safe_imread(item.file_path)
            elif is_video_file(item.file_path):
                cap = _cv2.VideoCapture(item.file_path)
                try:
                    if cap.isOpened():
                        ok, frame = cap.read()
                        raw_frame = frame if ok else None
                finally:
                    cap.release()
            if raw_frame is None:
                self._update_status(
                    "Could not read the selected file for region editing",
                    "warning",
                )
                return True

            source_img = Image.fromarray(_cv2.cvtColor(raw_frame, _cv2.COLOR_BGR2RGB))
            orig_w, orig_h = source_img.size
            try:
                max_w = max(220, self._preview_frame.winfo_width() - 36)
            except Exception:
                max_w = 390
            max_h = 260
            display_img = source_img.copy()
            display_img.thumbnail((max_w, max_h), Image.LANCZOS)

            timed_rects = self._active_timed_region_rects(
                getattr(self.config, "subtitle_region_spans", None), 0.0,
                getattr(self.config, "subtitle_region_keyframes", None)
            )
            if timed_rects:
                current_rects = timed_rects
            elif getattr(self.config, "subtitle_areas", None):
                current_rects = list(getattr(self.config, "subtitle_areas") or [])
            elif getattr(self.config, "subtitle_area", None):
                current_rects = [self.config.subtitle_area]
            else:
                current_rects = []

            self._preview_request_id += 1
            self._stop_throbber()
            self._preview_region_editor_state = {
                "item_id": item.id,
                "source_size": (orig_w, orig_h),
                "display_size": display_img.size,
                "base_image": display_img,
                "current_rects": [tuple(rect) for rect in current_rects],
            }
            self._preview_region_drag_start = None
            self._preview_region_pending_rect = None
            self.preview_title_label.config(
                text=tr("Draw subtitle region for {name}").format(
                    name=Path(item.file_path).name)
            )
            self.preview_meta_label.config(
                text=tr("Drag over the subtitle text. Release to save and refresh the mask.")
            )
            self._render_preview_region_editor()
            self._update_preview_actions()
            return True
        except Exception:
            logger.warning("Inline region editor failed", exc_info=True)
            self._update_status(
                "Inline region editor unavailable. Opening the full selector is still available.",
                "warning",
            )
            return True

    def _clear_preview_region_editor(self):
        """Clear direct region-edit state without changing the visible image."""
        self._preview_region_editor_state = None
        self._preview_region_drag_start = None
        self._preview_region_pending_rect = None

    def _preview_region_image_bounds(self):
        """Return the preview image bounds within the label widget."""
        state = getattr(self, "_preview_region_editor_state", None)
        if not state:
            return None
        disp_w, disp_h = state["display_size"]
        try:
            widget_w = max(self._preview_label.winfo_width(), disp_w)
            widget_h = max(self._preview_label.winfo_height(), disp_h)
        except Exception:
            widget_w, widget_h = disp_w, disp_h
        offset_x = max(0, (widget_w - disp_w) // 2)
        offset_y = max(0, (widget_h - disp_h) // 2)
        return offset_x, offset_y, disp_w, disp_h

    def _preview_widget_to_image_point(self, x: int, y: int):
        """Convert preview-label coordinates to source image coordinates."""
        state = getattr(self, "_preview_region_editor_state", None)
        bounds = self._preview_region_image_bounds()
        if not state or not bounds:
            return None
        offset_x, offset_y, disp_w, disp_h = bounds
        orig_w, orig_h = state["source_size"]
        px = min(max(0, int(x) - offset_x), disp_w)
        py = min(max(0, int(y) - offset_y), disp_h)
        img_x = int(round(px * orig_w / max(1, disp_w)))
        img_y = int(round(py * orig_h / max(1, disp_h)))
        return min(orig_w - 1, max(0, img_x)), min(orig_h - 1, max(0, img_y))

    @staticmethod
    def _normalized_region_rect(start, end):
        x1, y1 = start
        x2, y2 = end
        return (
            int(min(x1, x2)),
            int(min(y1, y2)),
            int(max(x1, x2)),
            int(max(y1, y2)),
        )

    def _preview_region_rect_to_display(self, rect):
        state = getattr(self, "_preview_region_editor_state", None)
        if not state:
            return None
        orig_w, orig_h = state["source_size"]
        disp_w, disp_h = state["display_size"]
        x1, y1, x2, y2 = rect
        return (
            int(round(x1 * disp_w / max(1, orig_w))),
            int(round(y1 * disp_h / max(1, orig_h))),
            int(round(x2 * disp_w / max(1, orig_w))),
            int(round(y2 * disp_h / max(1, orig_h))),
        )

    def _render_preview_region_editor(self):
        """Render saved and in-progress rectangles over the preview image."""
        state = getattr(self, "_preview_region_editor_state", None)
        if not state or not PIL_AVAILABLE:
            return
        image = state["base_image"].copy().convert("RGBA")
        draw = ImageDraw.Draw(image, "RGBA")

        def rgba(hex_color: str, alpha: int):
            r, g, b = self._hex_to_rgb(hex_color)
            return r, g, b, alpha

        for rect in state.get("current_rects", []):
            display_rect = self._preview_region_rect_to_display(rect)
            if not display_rect:
                continue
            draw.rectangle(
                display_rect,
                outline=rgba(Theme.GREEN_PRIMARY, 230),
                fill=rgba(Theme.GREEN_PRIMARY, 52),
                width=2,
            )
        pending = getattr(self, "_preview_region_pending_rect", None)
        if pending:
            display_rect = self._preview_region_rect_to_display(pending)
            if display_rect:
                draw.rectangle(
                    display_rect,
                    outline=rgba(Theme.BLUE_PRIMARY, 255),
                    fill=rgba(Theme.BLUE_PRIMARY, 46),
                    width=2,
                )

        self._preview_photo = ImageTk.PhotoImage(image.convert("RGB"))
        self._preview_label.config(image=self._preview_photo, text="")

    def _on_preview_region_press(self, event):
        if not getattr(self, "_preview_region_editor_state", None):
            return None
        point = self._preview_widget_to_image_point(event.x, event.y)
        if point is None:
            return "break"
        self._preview_region_drag_start = point
        self._preview_region_pending_rect = self._normalized_region_rect(point, point)
        self._render_preview_region_editor()
        return "break"

    def _on_preview_region_drag(self, event):
        state = getattr(self, "_preview_region_editor_state", None)
        if not state or not self._preview_region_drag_start:
            return None
        point = self._preview_widget_to_image_point(event.x, event.y)
        if point is None:
            return "break"
        rect = self._normalized_region_rect(self._preview_region_drag_start, point)
        self._preview_region_pending_rect = rect
        x1, y1, x2, y2 = rect
        self.preview_meta_label.config(
            text=tr("Release to save region ({x1}, {y1}) to ({x2}, {y2}).").format(
                x1=x1, y1=y1, x2=x2, y2=y2)
        )
        self._render_preview_region_editor()
        return "break"

    def _on_preview_region_release(self, event):
        state = getattr(self, "_preview_region_editor_state", None)
        if not state or not self._preview_region_drag_start:
            return None
        point = self._preview_widget_to_image_point(event.x, event.y)
        if point is None:
            return "break"
        rect = self._normalized_region_rect(self._preview_region_drag_start, point)
        self._preview_region_drag_start = None
        self._preview_region_pending_rect = None
        x1, y1, x2, y2 = rect
        if (x2 - x1) <= 10 or (y2 - y1) <= 5:
            self.preview_meta_label.config(
                text=tr("Drag a larger subtitle region before releasing.")
            )
            self._render_preview_region_editor()
            return "break"

        # Append to any previously configured static regions instead of
        # replacing them, so a multi-region setup is not silently lost when the
        # user draws another box in the inline editor. Timed spans are left
        # untouched for the same reason.
        existing = list(self.config.subtitle_areas or [])
        if self.config.subtitle_area and self.config.subtitle_area not in existing:
            existing.insert(0, tuple(self.config.subtitle_area))
        new_rect = tuple(rect)
        if new_rect not in existing:
            existing.append(new_rect)
        self.config.subtitle_areas = existing
        self.config.subtitle_area = existing[0] if existing else rect
        self._apply_region_settings_to_idle_items()
        self._update_region_label_display()
        item = self._queue_item_by_id(state["item_id"])
        self._clear_preview_region_editor()
        count = len(existing)
        msg = (
            "Added subtitle region from the preview"
            if count > 1 else "Saved subtitle region from the preview"
        )
        self._update_status(msg, "success")
        if item:
            self._show_preview(item, show_mask=True)
        return "break"

    def _show_preview(self, item: QueueItem, show_mask: bool = False):
        """Show thumbnail preview. Side-by-side before/after for completed items.
        If show_mask=True, run detection and overlay red boxes on the frame."""
        self._preview_request_id += 1
        preview_request_id = self._preview_request_id
        self._clear_preview_region_editor()
        # Any switch cancels a running throbber so it can't overwrite later UI
        if not show_mask:
            self._stop_throbber()
        self._set_selected_queue_item(item.id)
        if not PIL_AVAILABLE:
            self._set_preview_unavailable(
                "Preview unavailable",
                "Install Pillow to enable image previews.",
                label="Preview tools need Pillow",
            )
            return

        try:
            import cv2 as _cv2

            def load_first_frame_raw(path):
                """Load first frame as BGR numpy array."""
                if is_image_file(path):
                    return safe_imread(path)
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

            badge = status_ui(item.status)
            self.preview_status_chip.config(text=badge["label"], fg=badge["color"], bg=badge["bg"])

            try:
                max_w = max(220, self._preview_frame.winfo_width() - 36)
            except Exception:
                max_w = 390
            max_h = 260

            self.preview_title_label.config(
                text=f"Loading {Path(item.file_path).name}...")
            self.preview_meta_label.config(text="Reading first frame...")
            self._preview_label.config(image="", text="")
            self._preview_photo = None
            lang = self.lang_var.get()
            threshold = getattr(self.config, 'detection_threshold', 0.5)
            timed_spans = getattr(self.config, "subtitle_region_spans", None) or []
            keyframe_tracks = (
                getattr(self.config, "subtitle_region_keyframes", None) or [])
            timed_regions_configured = bool(timed_spans or keyframe_tracks)
            manual_shapes = region_shapes_at(keyframe_tracks, 0.0)
            sub_areas = self._active_timed_region_rects(timed_spans, 0.0)
            sub_areas.extend(
                tuple(shape["rect"])
                for shape in manual_shapes if "rect" in shape)
            if (not sub_areas and not timed_regions_configured
                    and getattr(self.config, "subtitle_areas", None)):
                sub_areas = list(getattr(self.config, "subtitle_areas", None) or [])
            if (not sub_areas and not timed_regions_configured
                    and self.config.subtitle_area):
                sub_areas = [self.config.subtitle_area]
            item_file = item.file_path
            item_id = item.id
            item_status = item.status
            item_output = item.output_path
            item_quality = item.quality_report
            item_soft = getattr(item, "soft_subtitle_streams", [])
            item_config = getattr(item, "config", self.config)
            mask_corrections = getattr(
                item_config, "manual_mask_corrections", None)
            mask_import_path = getattr(item_config, "mask_import_path", "")
            mask_import_mode = getattr(
                item_config, "mask_import_mode", "replace")
            mask_dilate_px = getattr(item_config, "mask_dilate_px", 0)

            def _preview_bg():
                try:
                    raw_frame = load_first_frame_raw(item_file)
                    if raw_frame is None:
                        def _no_frame():
                            if preview_request_id != self._preview_request_id:
                                return
                            self._set_preview_unavailable(
                                "Preview unavailable",
                                "The selected file could not be read. Verify the file path and add it again.",
                                label="No frame available",
                            )
                        self.root.after(0, _no_frame)
                        return

                    if show_mask:
                        self._preview_bg_mask(
                            raw_frame, lang, threshold, sub_areas,
                            timed_regions_configured, manual_shapes,
                            item_file, item_id, preview_request_id,
                            max_w, max_h, _cv2, to_pil,
                            mask_corrections=mask_corrections,
                            mask_import_path=mask_import_path,
                            mask_import_mode=mask_import_mode,
                            mask_dilate_px=mask_dilate_px)
                    else:
                        self._preview_bg_normal(
                            raw_frame, item_file, item_id, item_status,
                            item_output, item_quality, item_soft,
                            preview_request_id, max_w, max_h,
                            _cv2, to_pil, load_first_frame_raw)
                except Exception:
                    logger.warning("Preview render failed", exc_info=True)
                    def _err():
                        if preview_request_id != self._preview_request_id:
                            return
                        self._set_preview_unavailable(
                            "Preview unavailable",
                            "The preview failed before a frame could be rendered. Check the activity log, then try Set region.",
                            label="Preview failed",
                            tone="error",
                        )
                    self.root.after(0, _err)

            threading.Thread(target=_preview_bg, daemon=True).start()
        except Exception:
            logger.warning("Preview setup failed", exc_info=True)
            self._set_preview_unavailable(
                "Preview unavailable",
                "The preview could not be prepared. Check the activity log, then try adding the file again.",
                label="Preview failed",
                tone="error",
            )

    def _preview_bg_mask(self, raw_frame, lang, threshold, sub_areas,
                          timed_regions_configured, manual_shapes,
                          item_file, item_id, preview_request_id,
                          max_w, max_h, _cv2, to_pil,
                          mask_corrections=None, mask_import_path="",
                          mask_import_mode="replace", mask_dilate_px=0):
        try:
            from backend.processor import SubtitleDetector
            with self._detector_lock:
                if self._preview_detector is None or self._preview_detector_lang != lang:
                    self._preview_detector = SubtitleDetector(lang=lang)
                    self._preview_detector_lang = lang
                det = self._preview_detector
            frame_copy = raw_frame.copy()
            if sub_areas or manual_shapes:
                boxes = sub_areas
            elif (timed_regions_configured
                  and getattr(self.config, "sttn_skip_detection", False)):
                boxes = []
            else:
                boxes = det.detect(frame_copy, threshold)
            vis = frame_copy.copy()
            # Detection boxes use the theme's danger accent (BGR for cv2) so
            # they stay visible in the high-contrast palette.
            _dr, _dg, _db = self._hex_to_rgb(Theme.DANGER)
            for (bx1, by1, bx2, by2) in boxes:
                _cv2.rectangle(vis, (bx1, by1), (bx2, by2), (_db, _dg, _dr), 2)
            for shape in manual_shapes or []:
                coords = shape.get("polygon")
                if not coords:
                    continue
                points = np.asarray(
                    list(zip(coords[::2], coords[1::2])), dtype=np.int32)
                _cv2.polylines(vis, [points], True, (_db, _dg, _dr), 2)
            import cv2 as _mask_cv2

            composed_mask = np.zeros(frame_copy.shape[:2], dtype=np.uint8)
            for bx1, by1, bx2, by2 in boxes:
                _mask_cv2.rectangle(
                    composed_mask, (bx1, by1), (bx2, by2), 255, -1)
            for shape in manual_shapes or []:
                coords = shape.get("polygon")
                if not coords:
                    continue
                points = np.asarray(
                    list(zip(coords[::2], coords[1::2])), dtype=np.int32)
                _mask_cv2.fillPoly(composed_mask, [points], 255)
            dilation = max(0, int(mask_dilate_px or 0))
            if dilation and np.any(composed_mask):
                kernel = _mask_cv2.getStructuringElement(
                    _mask_cv2.MORPH_ELLIPSE,
                    (dilation * 2 + 1, dilation * 2 + 1),
                )
                composed_mask = _mask_cv2.dilate(composed_mask, kernel)
            from backend.mask_corrections import apply_mask_corrections

            composed_mask = apply_mask_corrections(
                composed_mask, mask_corrections, 0.0, 0)
            imported_note = ""
            if mask_import_path:
                from backend.matte_interchange import (
                    compose_imported_matte,
                    load_matte_preview_frame,
                )

                imported, imported_info = load_matte_preview_frame(
                    mask_import_path,
                    frame_index=0,
                    width=frame_copy.shape[1],
                    height=frame_copy.shape[0],
                )
                composed_mask = compose_imported_matte(
                    composed_mask, imported, mask_import_mode)
                imported_note = (
                    f" Imported {imported_info['format'].upper()} matte "
                    f"composed in {mask_import_mode} mode."
                )
            overlay = composed_mask.astype(np.float32) / 255.0 * 0.42
            color = np.asarray([_db, _dg, _dr], dtype=np.float32)
            vis = np.clip(
                vis.astype(np.float32) * (1.0 - overlay[..., None])
                + color * overlay[..., None],
                0,
                255,
            ).astype(np.uint8)
            img = to_pil(vis)
            img.thumbnail((max_w, max_h), Image.LANCZOS)
            engine = det._engine_name
            n = len(boxes) + sum(
                1 for shape in manual_shapes or [] if "polygon" in shape)
            def _update_mask():
                if (preview_request_id != self._preview_request_id
                        or self._selected_queue_item_id != item_id):
                    return
                self._stop_throbber()
                self._preview_photo = ImageTk.PhotoImage(img)
                self.preview_title_label.config(
                    text=f"Composed mask for {Path(item_file).name}")
                if (sub_areas or manual_shapes) and timed_regions_configured:
                    meta = "Timed manual region is active on the first frame."
                elif sub_areas:
                    meta = "Manual region applied. Detection used your saved subtitle band."
                elif timed_regions_configured:
                    meta = "Timed manual regions are configured but inactive on the first frame."
                elif n:
                    meta = f"{engine} found {n} region{'s' if n != 1 else ''} on the first frame."
                else:
                    meta = ("No regions were found on the first frame. Try Set region, or lower the "
                            "Threshold in detailed controls.")
                meta += imported_note
                self.preview_meta_label.config(text=meta)
                self._preview_label.config(
                    image=self._preview_photo,
                    text=f"{engine}: {n} detected" if n else "No text detected")
            self.root.after(0, _update_mask)
        except Exception:
            logger.warning("Detection preview failed", exc_info=True)
            def _show_mask_error():
                if (preview_request_id != self._preview_request_id
                        or self._selected_queue_item_id != item_id):
                    return
                self._set_preview_unavailable(
                    "Detection preview failed",
                    "The mask preview could not be generated. Check the activity log, then draw a manual region if needed.",
                    label="Mask preview failed",
                    tone="error",
                )
            self.root.after(0, _show_mask_error)

    def _preview_bg_normal(self, raw_frame, item_file, item_id, item_status,
                            item_output, item_quality, item_soft,
                            preview_request_id, max_w, max_h,
                            _cv2, to_pil, load_first_frame_raw):
        input_img = to_pil(raw_frame)

        output_img = None
        if item_status == ProcessingStatus.COMPLETE and Path(item_output).exists():
            out_frame = load_first_frame_raw(item_output)
            if out_frame is not None:
                output_img = to_pil(out_frame)

        if output_img:
            half_w = max_w // 2 - 2
            input_img.thumbnail((half_w, max_h), Image.LANCZOS)
            output_img.thumbnail((half_w, max_h), Image.LANCZOS)
            total_w = input_img.width + output_img.width + 4
            total_h = max(input_img.height, output_img.height)
            composite = Image.new("RGB", (total_w, total_h),
                                   self._hex_to_rgb(Theme.BG_DARK))
            composite.paste(input_img, (0, 0))
            composite.paste(output_img, (input_img.width + 4, 0))
            draw = ImageDraw.Draw(composite)
            draw.line([(input_img.width + 1, 0), (input_img.width + 1, total_h)],
                      fill=self._hex_to_rgb(Theme.GREEN_PRIMARY), width=2)
            draw.rectangle((10, 10, 82, 28), fill=self._hex_to_rgb(Theme.BG_TERTIARY))
            draw.text((18, 14), "Source", fill=self._hex_to_rgb(Theme.TEXT_SECONDARY))
            draw.rectangle((input_img.width + 16, 10, input_img.width + 96, 28),
                           fill=self._hex_to_rgb(Theme.SUCCESS_BG))
            draw.text((input_img.width + 24, 14), "Cleaned",
                      fill=self._hex_to_rgb(Theme.SUCCESS))
            display_image = composite
            title = f"Before / after for {Path(item_file).name}"
            meta = ("Completed items show the source frame beside the cleaned result so you can "
                    "spot-check the cleanup immediately.")
            quality_note = format_quality_report(item_quality)
            if quality_note:
                meta += f" Quality check: {quality_note}."
        else:
            input_img.thumbnail((max_w, max_h), Image.LANCZOS)
            display_image = input_img
            title = f"Source frame for {Path(item_file).name}"
            soft_summary = _format_soft_subtitle_summary(item_soft)
            if soft_summary:
                meta = (
                    f"{soft_summary}. Right-click the queue item for fast "
                    "strip/keep, or review the mask for burned-in cleanup."
                )
            else:
                meta = (
                    "Use Set region to draw the subtitle band, or Review mask "
                    "to confirm what the detector finds automatically."
                )

        def _update_normal():
            if (preview_request_id != self._preview_request_id
                    or self._selected_queue_item_id != item_id):
                return
            self._stop_throbber()
            # Tk image objects must be created by the main thread.  The PIL
            # compositing and resizing above intentionally stays on the
            # preview worker, while PhotoImage construction happens only in
            # this callback dispatched through Tk's event loop.
            self._preview_photo = ImageTk.PhotoImage(display_image)
            self.preview_title_label.config(text=title)
            self.preview_meta_label.config(text=meta)
            self._preview_label.config(image=self._preview_photo, text="")
        self.root.after(0, _update_normal)

