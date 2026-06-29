"""RFP-T-4 GUI smoke test (Tk update_idletasks walk).

The full GUI is heavy to instantiate (250+ widgets, custom Canvas
drawing, etc.) so we exercise it via a `headless()` factory that
withdraws the root window and drives the major flows without
actually mapping the GUI to a display.

Skips automatically on:
- Non-display environments (Linux / WSL without DISPLAY / WAYLAND).
- Builds without Pillow (the preview pane construction requires it).
"""

from __future__ import annotations

import os
import json
import sys
import tempfile
import threading
import tkinter as tk
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from PIL import Image, ImageDraw


def _have_display() -> bool:
    if sys.platform == "win32":
        return True
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


@unittest.skipUnless(_have_display(), "GUI smoke test needs a display")
class GuiSmokeTests(unittest.TestCase):
    """Drive the major GUI flows headlessly."""

    @classmethod
    def setUpClass(cls):
        # Pin the settings file to a temp dir so the test doesn't
        # clobber the developer's real %APPDATA%/VSR config.
        cls._tmpdir = tempfile.TemporaryDirectory()
        import VideoSubtitleRemover as g
        from gui import config as gui_config
        cls._g = g
        cls._gui_config = gui_config
        cls._orig_settings = g.SETTINGS_FILE
        cls._orig_queue_state = gui_config.QUEUE_STATE_FILE
        g.SETTINGS_FILE = Path(cls._tmpdir.name) / "settings.json"
        gui_config.QUEUE_STATE_FILE = Path(cls._tmpdir.name) / "queue_state.json"

    @classmethod
    def tearDownClass(cls):
        cls._g.SETTINGS_FILE = cls._orig_settings
        cls._gui_config.QUEUE_STATE_FILE = cls._orig_queue_state
        cls._tmpdir.cleanup()

    def _make_app(self, *, withdraw: bool = True):
        with mock.patch.object(
            self._g.VideoSubtitleRemoverApp,
            "_start_startup_hardware_probe",
        ):
            with mock.patch.object(
                self._g.VideoSubtitleRemoverApp,
                "_maybe_restore_queue",
            ):
                app = self._g.VideoSubtitleRemoverApp()
        if withdraw:
            app.root.withdraw()
        return app

    @staticmethod
    def _walk_widgets(widget):
        yield widget
        for child in widget.winfo_children():
            yield from GuiSmokeTests._walk_widgets(child)

    @staticmethod
    def _drag_canvas(canvas: tk.Canvas, start: tuple[int, int], end: tuple[int, int]):
        before = len([
            item_id for item_id in canvas.find_all()
            if canvas.type(item_id) == "rectangle"
        ])
        canvas.event_generate("<Enter>", x=start[0], y=start[1], when="now")
        canvas.event_generate("<Motion>", x=start[0], y=start[1], when="now")
        canvas.event_generate("<ButtonPress-1>", x=start[0], y=start[1], when="now")
        canvas.event_generate("<B1-Motion>", x=end[0], y=end[1], when="now")
        canvas.event_generate("<ButtonRelease-1>", x=end[0], y=end[1], when="now")
        after = len([
            item_id for item_id in canvas.find_all()
            if canvas.type(item_id) == "rectangle"
        ])
        if after > before:
            return
        handlers = getattr(canvas, "_vsr_region_drag_handlers", None)
        if not handlers:
            return
        press, drag, release = handlers
        press(SimpleNamespace(x=start[0], y=start[1]))
        drag(SimpleNamespace(x=end[0], y=end[1]))
        release(SimpleNamespace(x=end[0], y=end[1]))

    def _destroy_app(self, app):
        app._shutdown_started = True
        try:
            for child in list(app.root.winfo_children()):
                if isinstance(child, tk.Toplevel):
                    try:
                        child.grab_release()
                    except tk.TclError:
                        pass
            app.root.update_idletasks()
        finally:
            try:
                app.root.destroy()
            except tk.TclError:
                pass

    def test_construct_and_close(self):
        app = self._make_app()
        try:
            self.assertEqual(app.root.title()[:23], "Video Subtitle Remover ")
            app.root.update_idletasks()
        finally:
            self._destroy_app(app)

    def test_custom_widgets_expose_accessibility_snapshots(self):
        app = self._make_app()
        try:
            app.root.update_idletasks()

            self.assertIn(
                "Select a queue item",
                app.preview_action_hint.cget("text"),
            )
            self.assertIn(
                "Select a queued item",
                app.preview_mask_btn.accessibility_snapshot()["description"],
            )

            slider_meta = app._settings_sliders[0].accessibility_snapshot()
            self.assertEqual(slider_meta["role"], "slider")
            self.assertTrue(slider_meta["label"])
            self.assertIn("range", slider_meta["value"])

            segment = next(iter(app.mode_picker._segments.values()))
            segment_meta = segment.accessibility_snapshot()
            self.assertEqual(segment_meta["role"], "radio button")
            self.assertEqual(segment_meta["description"], "Cleanup algorithm")
            self.assertIn("selected", segment_meta["state"])
            segment.set_enabled(False)
            self.assertIn("disabled", segment.accessibility_snapshot()["state"])
            segment.set_enabled(True)

            app.drop_area.set_import_enabled(False)
            drop_meta = app.drop_area.accessibility_snapshot()
            self.assertEqual(drop_meta["role"], "drop target")
            self.assertIn("disabled", drop_meta["state"])

            from gui.widgets import QueueItemWidget

            item = self._g.QueueItem(
                id="a11y-row",
                file_path=str(Path(self._tmpdir.name) / "clip.mp4"),
                output_path=str(Path(self._tmpdir.name) / "clip_no_sub.mp4"),
                config=self._g.ProcessingConfig(),
                status=self._g.ProcessingStatus.PROCESSING,
                progress=0.42,
                message="Cleaning frame 12",
            )
            row = QueueItemWidget(app.root, item, on_remove=lambda _id: None)
            row.update_item(item)
            row_meta = row.accessibility_snapshot()
            self.assertEqual(row_meta["role"], "queue item")
            self.assertIn("Removing", row_meta["state"])
            self.assertIn("42% complete", row_meta["value"])
            self.assertIn("Cleaning frame 12", row_meta["description"])
            row.destroy()

            app._set_preview_unavailable(
                "Preview unavailable",
                "The file could not be read.",
                label="No frame available",
            )
            self.assertEqual(app.preview_title_label.cget("text"), "Preview unavailable")
            self.assertEqual(app.preview_meta_label.cget("text"), "The file could not be read.")
            self.assertEqual(app._preview_label.cget("text"), "No frame available")
            self.assertEqual(app.preview_status_chip.cget("text"), "Needs attention")
        finally:
            self._destroy_app(app)

    def test_about_dialog_shows_backend_status_panel(self):
        app = self._make_app(withdraw=False)
        try:
            app.ai_engines = {
                "detection": ["RapidOCR"],
                "inpainting": ["Temporal BG (TBE)", "LaMa ONNX"],
            }
            app.backend_status = {
                "schema": "vsr.backend_status.v1",
                "summary": {
                    "detection": "RapidOCR (ready)",
                    "inpainting": "LaMa ONNX ready",
                    "providers": "CPUExecutionProvider",
                    "language_support": (
                        "GUI picker: 52 selectable OCR codes; "
                        "installed OCR capacity: RapidOCR 100+."
                    ),
                    "model_files": "RapidOCR 3 model file(s); LaMa ONNX verified",
                    "hash_status": "verified",
                    "next_action": "No backend setup action needed.",
                    "tone": "success",
                },
            }

            app._show_about()
            app.root.update()
            dialog = next(
                child for child in app.root.winfo_children()
                if isinstance(child, tk.Toplevel)
                and child.title() == "About Video Subtitle Remover Pro"
            )
            self.assertLessEqual(dialog.winfo_reqwidth(), 650)

            texts = []
            for widget in self._walk_widgets(app.root):
                drawn_text = getattr(widget, "text", None)
                if drawn_text:
                    texts.append(str(drawn_text))
                try:
                    text = widget.cget("text")
                except tk.TclError:
                    continue
                if text:
                    texts.append(str(text))

            self.assertIn("BACKEND STATUS", texts)
            self.assertIn("LaMa ONNX ready", texts)
            self.assertIn("CPUExecutionProvider", texts)
            self.assertIn(
                "GUI picker: 52 selectable OCR codes; installed OCR capacity: RapidOCR 100+.",
                texts,
            )
            self.assertIn("Model cache", texts)
            self.assertIn("No backend setup action needed.", texts)
            try:
                dialog.grab_release()
            except tk.TclError:
                pass
            dialog.destroy()
            app.root.update()
        finally:
            self._destroy_app(app)

    def test_sync_config_round_trip(self):
        app = self._make_app()
        try:
            # Toggle a known field via the tk variable and confirm it
            # propagates into the persisted config.
            app.preserve_audio_var.set(False)
            app._sync_config_from_ui()
            self.assertFalse(app.config.preserve_audio)
        finally:
            self._destroy_app(app)

    def test_queue_display_selects_first_item_for_inspect_actions(self):
        app = self._make_app()
        try:
            source = Path(self._tmpdir.name) / "inspect.png"
            source.write_bytes(b"not a real image")
            item = self._g.QueueItem(
                id="inspect-target",
                file_path=str(source),
                output_path=str(source.with_name("inspect_no_sub.png")),
                config=self._g.ProcessingConfig(),
            )
            app.queue.append(item)
            app._selected_queue_item_id = None

            app._update_queue_display()

            self.assertEqual(app._selected_queue_item_id, item.id)
            self.assertIs(app._get_selected_queue_item(), item)
            self.assertTrue(app.preview_mask_btn.enabled)
            self.assertTrue(app.preview_inpaint_btn.enabled)
            self.assertIn("Preview tools are ready", app.preview_action_hint.cget("text"))
        finally:
            self._destroy_app(app)

    def test_region_changes_update_idle_queue_snapshots(self):
        app = self._g.VideoSubtitleRemoverApp.__new__(self._g.VideoSubtitleRemoverApp)
        app.config = self._g.ProcessingConfig()
        app.queue = []
        app.queue_lock = threading.Lock()

        source = Path(self._tmpdir.name) / "region.png"
        source.write_bytes(b"not a real image")
        item = self._g.QueueItem(
            id="region-target",
            file_path=str(source),
            output_path=str(source.with_name("region_no_sub.png")),
            config=self._g.ProcessingConfig(),
        )
        app.queue.append(item)

        app.config.subtitle_area = (10, 20, 110, 54)
        app.config.subtitle_areas = [(10, 20, 110, 54), (12, 60, 108, 76)]
        app.config.subtitle_region_spans = [
            {"rect": (14, 70, 114, 92), "start": 2.0, "end": 4.0}
        ]
        self.assertEqual(app._apply_region_settings_to_idle_items(), 1)

        self.assertEqual(item.config.subtitle_area, (10, 20, 110, 54))
        self.assertEqual(
            item.config.subtitle_areas,
            [(10, 20, 110, 54), (12, 60, 108, 76)],
        )
        self.assertEqual(
            item.config.subtitle_region_spans,
            [{"rect": (14, 70, 114, 92), "start": 2.0, "end": 4.0}],
        )

        app.config.subtitle_area = None
        app.config.subtitle_areas = None
        app.config.subtitle_region_spans = None
        self.assertEqual(app._apply_region_settings_to_idle_items(), 1)

        self.assertIsNone(item.config.subtitle_area)
        self.assertIsNone(item.config.subtitle_areas)
        self.assertIsNone(item.config.subtitle_region_spans)

    def test_preview_region_drag_saves_and_refreshes_mask(self):
        app = self._make_app(withdraw=False)
        try:
            source = Path(self._tmpdir.name) / "inline-region.png"
            image = Image.new("RGB", (320, 180), (20, 24, 32))
            draw = ImageDraw.Draw(image)
            draw.rectangle((40, 124, 280, 154), fill=(235, 235, 235))
            draw.text((80, 130), "subtitle text", fill=(0, 0, 0))
            image.save(source)

            with mock.patch.object(app, "_show_preview"):
                self.assertEqual(app._add_to_queue(str(source)), "added")
            item = app.queue[0]

            app._open_region_selector()
            app.root.update()
            self.assertIsNotNone(app._preview_region_editor_state)
            self.assertFalse(any(
                isinstance(child, tk.Toplevel)
                and child.title() == "Choose subtitle region"
                for child in app.root.winfo_children()
            ))

            bounds = app._preview_region_image_bounds()
            self.assertIsNotNone(bounds)
            offset_x, offset_y, disp_w, disp_h = bounds
            self.assertGreater(disp_w, 0)
            self.assertGreater(disp_h, 0)

            def widget_point(x, y):
                src_w, src_h = app._preview_region_editor_state["source_size"]
                return SimpleNamespace(
                    x=offset_x + round(x * disp_w / src_w),
                    y=offset_y + round(y * disp_h / src_h),
                )

            start = widget_point(40, 124)
            end = widget_point(280, 154)
            expected = app._normalized_region_rect(
                app._preview_widget_to_image_point(start.x, start.y),
                app._preview_widget_to_image_point(end.x, end.y),
            )
            self.assertTrue(all(
                abs(actual - target) <= 1
                for actual, target in zip(expected, (40, 124, 280, 154))
            ))

            with mock.patch.object(app, "_show_preview") as preview:
                app._on_preview_region_press(start)
                app._on_preview_region_drag(end)
                self.assertEqual(
                    app._preview_region_pending_rect,
                    expected,
                )
                app._on_preview_region_release(end)

            self.assertEqual(app.config.subtitle_area, expected)
            self.assertEqual(app.config.subtitle_areas, [expected])
            self.assertIsNone(app.config.subtitle_region_spans)
            self.assertEqual(item.config.subtitle_area, expected)
            self.assertEqual(item.config.subtitle_areas, [expected])
            self.assertIsNone(app._preview_region_editor_state)
            preview.assert_called_once_with(item, show_mask=True)
        finally:
            self._destroy_app(app)

    def test_region_selector_save_updates_visible_and_queued_config(self):
        app = self._make_app(withdraw=False)
        try:
            from gui.widgets import ModernButton

            source = Path(self._tmpdir.name) / "selector-source.png"
            image = Image.new("RGB", (320, 180), (20, 24, 32))
            draw = ImageDraw.Draw(image)
            draw.rectangle((40, 124, 280, 154), fill=(235, 235, 235))
            draw.text((80, 130), "subtitle text", fill=(0, 0, 0))
            image.save(source)

            with mock.patch.object(app, "_show_preview"):
                self.assertEqual(app._add_to_queue(str(source)), "added")
            item = app.queue[0]
            self.assertEqual(app._selected_queue_item_id, item.id)

            app._open_region_selector_modal()
            app.root.update()

            selector = next(
                child for child in app.root.winfo_children()
                if isinstance(child, tk.Toplevel)
                and child.title() == "Choose subtitle region"
            )
            canvas = next(
                widget for widget in self._walk_widgets(selector)
                if isinstance(widget, tk.Canvas)
                and str(widget.cget("cursor")) == "cross"
            )

            self._drag_canvas(canvas, (40, 124), (280, 154))
            app.root.update()

            save_button = next(
                widget for widget in self._walk_widgets(selector)
                if isinstance(widget, ModernButton)
                and getattr(widget, "text", "") == "Save"
            )
            save_button.command()
            app.root.update()

            expected = (40, 124, 280, 154)
            self.assertEqual(app.config.subtitle_area, expected)
            self.assertEqual(app.config.subtitle_areas, [expected])
            self.assertEqual(item.config.subtitle_area, expected)
            self.assertEqual(item.config.subtitle_areas, [expected])
            self.assertFalse(selector.winfo_exists())
        finally:
            self._destroy_app(app)

    def test_region_selector_saves_timed_video_regions(self):
        app = self._make_app(withdraw=False)
        try:
            import cv2
            import numpy as np
            from gui.widgets import ModernButton

            source = Path(self._tmpdir.name) / "selector-timed.avi"
            frame_w, frame_h = 320, 180
            writer = cv2.VideoWriter(
                str(source),
                cv2.VideoWriter_fourcc(*"MJPG"),
                10.0,
                (frame_w, frame_h),
            )
            if not writer.isOpened():
                self.skipTest("OpenCV video writer unavailable")
            try:
                for idx in range(4):
                    image = Image.new("RGB", (frame_w, frame_h), (20, 24, 32))
                    draw = ImageDraw.Draw(image)
                    draw.rectangle((40, 120, 280, 152), fill=(235, 235, 235))
                    draw.text((80, 128), f"subtitle {idx}", fill=(0, 0, 0))
                    writer.write(cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR))
            finally:
                writer.release()

            with mock.patch.object(app, "_start_soft_subtitle_probe"):
                with mock.patch.object(app, "_show_preview"):
                    self.assertEqual(app._add_to_queue(str(source)), "added")
            item = app.queue[0]

            app._open_region_selector_modal()
            app.root.update()

            selector = next(
                child for child in app.root.winfo_children()
                if isinstance(child, tk.Toplevel)
                and child.title() == "Choose subtitle region"
            )
            canvas = next(
                widget for widget in self._walk_widgets(selector)
                if isinstance(widget, tk.Canvas)
                and str(widget.cget("cursor")) == "cross"
            )
            self._drag_canvas(canvas, (40, 120), (280, 152))
            selector._vsr_start_entry.insert(0, "0.5")
            selector._vsr_end_entry.insert(0, "1.4")
            app.root.update()

            save_button = next(
                widget for widget in self._walk_widgets(selector)
                if isinstance(widget, ModernButton)
                and getattr(widget, "text", "") == "Save"
            )
            save_button.command()
            app.root.update()

            expected = [{"rect": (40, 120, 280, 152),
                         "start": 0.5, "end": 1.4}]
            self.assertIsNone(app.config.subtitle_area)
            self.assertIsNone(app.config.subtitle_areas)
            self.assertEqual(app.config.subtitle_region_spans, expected)
            self.assertEqual(item.config.subtitle_region_spans, expected)
            self.assertFalse(selector.winfo_exists())
        finally:
            self._destroy_app(app)

    def test_region_selector_scaled_video_updates_preview_coordinates(self):
        app = self._make_app(withdraw=False)
        try:
            import cv2
            import numpy as np
            from gui.widgets import ModernButton

            source = Path(self._tmpdir.name) / "selector-scaled.avi"
            frame_w, frame_h = 960, 540
            writer = cv2.VideoWriter(
                str(source),
                cv2.VideoWriter_fourcc(*"MJPG"),
                10.0,
                (frame_w, frame_h),
            )
            if not writer.isOpened():
                self.skipTest("OpenCV video writer unavailable")
            try:
                for idx in range(2):
                    image = Image.new("RGB", (frame_w, frame_h), (20 + idx, 24, 32))
                    draw = ImageDraw.Draw(image)
                    draw.rectangle((120, 420, 840, 500), fill=(235, 235, 235))
                    writer.write(cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR))
            finally:
                writer.release()

            preload = (96, 104, 320, 184)
            app.config.subtitle_area = preload
            app.config.subtitle_areas = [preload]
            with mock.patch.object(app, "_start_soft_subtitle_probe"):
                with mock.patch.object(app, "_show_preview"):
                    self.assertEqual(app._add_to_queue(str(source)), "added")
            item = app.queue[0]

            with mock.patch.object(app.root, "winfo_screenwidth", return_value=300):
                with mock.patch.object(app.root, "winfo_screenheight", return_value=200):
                    app._open_region_selector_modal()
            app.root.update()

            selector = next(
                child for child in app.root.winfo_children()
                if isinstance(child, tk.Toplevel)
                and child.title() == "Choose subtitle region"
            )
            canvas = next(
                widget for widget in self._walk_widgets(selector)
                if isinstance(widget, tk.Canvas)
                and str(widget.cget("cursor")) == "cross"
            )
            self.assertEqual(int(canvas.cget("width")), 240)
            self.assertEqual(int(canvas.cget("height")), 135)

            scaled_preload = next(
                item_id for item_id in canvas.find_all()
                if canvas.type(item_id) == "rectangle"
            )
            self.assertEqual(
                [round(value, 2) for value in canvas.coords(scaled_preload)],
                [24.0, 26.0, 80.0, 46.0],
            )

            clear_button = next(
                widget for widget in self._walk_widgets(selector)
                if isinstance(widget, ModernButton)
                and getattr(widget, "text", "") == "Clear all"
            )
            clear_button.command()
            app.root.update()
            self.assertEqual(
                [item_id for item_id in canvas.find_all()
                 if canvas.type(item_id) == "rectangle"],
                [],
            )

            self._drag_canvas(canvas, (20, 30), (220, 100))
            self._drag_canvas(canvas, (30, 105), (210, 125))
            app.root.update()

            save_button = next(
                widget for widget in self._walk_widgets(selector)
                if isinstance(widget, ModernButton)
                and getattr(widget, "text", "") == "Save"
            )
            save_button.command()
            app.root.update()

            expected = [(80, 120, 880, 400), (120, 420, 840, 500)]
            self.assertEqual(app.config.subtitle_area, expected[0])
            self.assertEqual(app.config.subtitle_areas, expected)
            self.assertEqual(item.config.subtitle_area, expected[0])
            self.assertEqual(item.config.subtitle_areas, expected)
            self.assertFalse(selector.winfo_exists())

            cap = cv2.VideoCapture(str(source))
            try:
                ok, raw_frame = cap.read()
            finally:
                cap.release()
            self.assertTrue(ok)
            calls = []

            def rectangle(vis, pt1, pt2, color, width):
                calls.append((pt1, pt2, color, width))
                return vis

            def to_pil(bgr):
                return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))

            fake_cv2 = SimpleNamespace(rectangle=rectangle)

            class FakeDetector:
                _engine_name = "manual"

                def detect(self, *_args, **_kwargs):
                    raise AssertionError("manual preview should not detect")

            app._preview_request_id += 1
            request_id = app._preview_request_id
            with mock.patch(
                "backend.processor.SubtitleDetector",
                return_value=FakeDetector(),
            ):
                app._preview_bg_mask(
                    raw_frame,
                    "en",
                    0.5,
                    app.config.subtitle_areas,
                    False,
                    item.file_path,
                    item.id,
                    request_id,
                    390,
                    260,
                    fake_cv2,
                    to_pil,
                )
            app.root.update()

            self.assertEqual(
                calls,
                [
                    ((80, 120), (880, 400), (0, 0, 255), 2),
                    ((120, 420), (840, 500), (0, 0, 255), 2),
                ],
            )
            self.assertEqual(
                app.preview_meta_label.cget("text"),
                "Manual region applied. Detection used your saved subtitle band.",
            )
        finally:
            self._destroy_app(app)

    def test_apply_responsive_layout_no_op_when_unchanged(self):
        app = self._make_app()
        try:
            app._apply_responsive_layout(1280)
            initial = app._layout_mode
            app._apply_responsive_layout(1280)
            self.assertEqual(app._layout_mode, initial)
        finally:
            self._destroy_app(app)

    def test_queue_soft_subtitle_probe_surfaces_actions(self):
        app = self._make_app()
        try:
            source = Path(self._tmpdir.name) / "soft.mkv"
            source.write_bytes(b"not a real video")
            stream = {
                "index": 2,
                "codec_name": "subrip",
                "language": "eng",
                "title": "SDH",
                "default": True,
                "forced": False,
            }
            with mock.patch.object(app, "_start_soft_subtitle_probe"):
                with mock.patch.object(app, "_show_preview"):
                    self.assertEqual(app._add_to_queue(str(source)), "added")

            item = app.queue[0]
            app._apply_soft_subtitle_probe_records(item.id, [stream])
            widget = app.queue_widgets[item.id]
            self.assertTrue(item.soft_subtitle_probe_done)
            self.assertEqual(item.soft_subtitle_streams[0]["language"], "eng")
            self.assertIn("Right-click", item.message)
            self.assertIn("embedded subtitle", widget.info_label.cget("text"))

            app._set_soft_subtitle_action(item.id, "strip")

            self.assertEqual(item.soft_subtitle_action, "strip")
            self.assertIn("Fast strip", item.message)

        finally:
            self._destroy_app(app)

    def test_batch_report_files_written_for_completed_queue(self):
        app = self._g.VideoSubtitleRemoverApp.__new__(self._g.VideoSubtitleRemoverApp)
        app.queue = []
        app.queue_lock = threading.Lock()
        app._batch_report_records = {}
        app._last_batch_report_records = []
        app._last_batch_report_paths = []
        app._batch_started_at = self._g.datetime.now()
        app.gpus = []

        from backend import batch_report as _br

        source = Path(self._tmpdir.name) / "batch-source.mp4"
        output_dir = Path(self._tmpdir.name) / "batch-out"
        output = output_dir / "batch-source_no_sub.mp4"
        source.write_bytes(b"not a real video")
        output_dir.mkdir(exist_ok=True)
        item = self._g.QueueItem(
            id="batch-report",
            file_path=str(source),
            output_path=str(output),
            config=self._g.ProcessingConfig(),
        )
        app.queue.append(item)

        with mock.patch.object(_br, "_probe_codec_for_log", return_value="h264,64,48,24/1"):
            with mock.patch.object(_br, "_probe_duration_seconds", return_value=2.0):
                with mock.patch.object(_br, "_probe_subtitle_streams", return_value=[]):
                    app._prepare_batch_report_records()

        item.status = self._g.ProcessingStatus.COMPLETE
        item.message = "Complete!"
        item.quality_report = {
            "tag": "Good",
            "samples": 2,
            "ssim": 0.99,
            "roi_ssim": 0.98,
        }
        item.stage_timings = {
            "decode": 0.25,
            "ocr": 0.75,
            "mask": 0.1,
            "inpaint": 1.5,
            "encode": 0.4,
            "mux": 0.2,
            "quality": 0.0,
        }
        item.started_at = self._g.datetime.now()
        item.completed_at = self._g.datetime.now()

        paths = app._write_batch_report_files()
        payload = json.loads(
            (output_dir / "vsr-batch-summary.json").read_text(encoding="utf-8")
        )

        self.assertEqual(len(paths), 2)
        self.assertEqual(payload["schema"], "vsr.batch_summary.v1")
        self.assertEqual(payload["kind"], "gui-batch")
        self.assertEqual(payload["counts"], {"hardcoded-processed": 1})
        self.assertEqual(payload["files"][0]["input_name"], source.name)
        self.assertEqual(payload["files"][0]["quality_gate"]["status"], "passed")
        self.assertEqual(payload["files"][0]["dominant_stage"]["name"], "inpaint")
        self.assertEqual(payload["stage_summary"]["slowest_stage"]["name"], "inpaint")

    def test_soft_subtitle_action_remuxes_without_backend_remover(self):
        app = self._g.VideoSubtitleRemoverApp.__new__(self._g.VideoSubtitleRemoverApp)
        app.cancel_event = threading.Event()
        app._batch_times = []
        app._active_subprocess = None
        app._active_remover = None
        app._update_item_display = mock.Mock()

        source = Path(self._tmpdir.name) / "remux.mkv"
        output = Path(self._tmpdir.name) / "remux-out.mkv"
        source.write_bytes(b"not a real video")
        item = self._g.QueueItem(
            id="soft-action",
            file_path=str(source),
            output_path=str(output),
            config=self._g.ProcessingConfig(),
            soft_subtitle_action="strip",
        )
        with mock.patch("backend.remux.remux_soft_subtitles") as remux:
            with mock.patch(
                "backend.processor.SubtitleRemover",
                side_effect=AssertionError("backend remover should not load"),
            ):
                app._process_item(item)

        self.assertEqual(item.status, self._g.ProcessingStatus.COMPLETE)
        self.assertEqual(item.progress, 1.0)
        self.assertIn("stripped", item.message)
        remux.assert_called_once()


if __name__ == "__main__":
    unittest.main()
