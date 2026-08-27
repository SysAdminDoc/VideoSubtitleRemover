"""Active release checks for the smallest critical GUI workflow."""

from __future__ import annotations

import logging
import os
import sys
import tempfile
import threading
import time
import tkinter as tk
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from backend.a11y import accessible_metadata
from gui.preview_controller import PreviewControllerMixin, _fit_preview_image
from gui.quality_controller import QualityReviewControllerMixin
from gui.theme import Theme
from gui.utils import (
    dispatch_to_ui,
    install_ui_dispatcher,
    stop_ui_dispatcher,
)


def _have_display() -> bool:
    if sys.platform == "win32":
        return True
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


def _contrast_ratio(foreground: str, background: str) -> float:
    def _luminance(value: str) -> float:
        channels = [int(value[index:index + 2], 16) / 255 for index in (1, 3, 5)]
        linear = [
            channel / 12.92
            if channel <= 0.04045
            else ((channel + 0.055) / 1.055) ** 2.4
            for channel in channels
        ]
        return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]

    foreground_luminance = _luminance(foreground)
    background_luminance = _luminance(background)
    lighter = max(foreground_luminance, background_luminance)
    darker = min(foreground_luminance, background_luminance)
    return (lighter + 0.05) / (darker + 0.05)


class DesignTokenReleaseTests(unittest.TestCase):
    def test_primary_button_states_meet_normal_text_contrast(self):
        for background in (
            Theme.BLUE_PRIMARY,
            Theme.BLUE_HOVER,
            Theme.BLUE_PRESS,
        ):
            with self.subTest(background=background):
                self.assertGreaterEqual(
                    _contrast_ratio(Theme.INK_ON_BLUE, background),
                    4.5,
                )

    def test_radius_scale_uses_the_supported_non_pill_values(self):
        self.assertEqual(
            (Theme.R_SM, Theme.R_MD, Theme.R_LG, Theme.R_XL),
            (4, 6, 8, 10),
        )

    def test_preview_fit_enlarges_small_frames_without_distortion(self):
        from PIL import Image

        source = Image.new("RGB", (160, 120), "#182132")
        fitted = _fit_preview_image(source, 960, 540)

        self.assertEqual(fitted.size, (720, 540))
        self.assertEqual(source.size, (160, 120))

    def test_batch_stage_labels_use_reader_facing_names(self):
        self.assertEqual(
            QualityReviewControllerMixin._stage_label("inpaint"),
            "Inpainting",
        )
        self.assertEqual(
            QualityReviewControllerMixin._stage_label("mux"),
            "Finalizing output",
        )


class UiThreadDispatchTests(unittest.TestCase):
    def test_worker_dispatch_never_calls_tk_and_callback_runs_on_ui_thread(self):
        main_thread = threading.get_ident()

        class FakeRoot:
            def __init__(self):
                self.after_calls = []
                self.report_callback_exception = mock.Mock()

            def after(self, delay, callback, *args):
                self.after_calls.append((threading.get_ident(), delay, callback, args))
                return f"after-{len(self.after_calls)}"

        root = FakeRoot()
        install_ui_dispatcher(root)
        self.assertEqual(len(root.after_calls), 1)
        seen = []

        worker = threading.Thread(
            target=lambda: dispatch_to_ui(
                root, lambda value: seen.append((threading.get_ident(), value)), 7
            )
        )
        worker.start()
        worker.join()

        self.assertEqual(len(root.after_calls), 1, "worker called Tk.after")
        root.after_calls[0][2]()
        self.assertEqual(seen, [(main_thread, 7)])
        self.assertEqual(root.after_calls[-1][0], main_thread)
        stop_ui_dispatcher(root)

    def test_preview_dispatch_stops_after_shutdown(self):
        controller = PreviewControllerMixin()
        controller.root = SimpleNamespace(after=mock.Mock(return_value="after-1"))
        callback = mock.Mock()

        controller._shutdown_started = False
        self.assertEqual(controller._dispatch_preview_ui(callback, 3), "after-1")
        controller.root.after.assert_called_once_with(0, callback, 3)

        controller._shutdown_started = True
        self.assertIsNone(controller._dispatch_preview_ui(callback, 4))
        self.assertEqual(controller.root.after.call_count, 1)


@unittest.skipUnless(_have_display(), "GUI workflow test needs a display")
class GuiWorkflowReleaseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmpdir = tempfile.TemporaryDirectory()
        import VideoSubtitleRemover as app_exports
        from gui import app as gui_app_module
        from gui import config as gui_config

        cls._app_exports = app_exports
        cls._gui_app_module = gui_app_module
        cls._gui_config = gui_config
        cls._shared_root = tk.Tk()
        cls._shared_root.withdraw()
        cls._original_export_settings = app_exports.SETTINGS_FILE
        cls._original_settings = gui_config.SETTINGS_FILE
        cls._original_queue_state = gui_config.QUEUE_STATE_FILE
        settings_path = Path(cls._tmpdir.name) / "settings.json"
        app_exports.SETTINGS_FILE = settings_path
        gui_config.SETTINGS_FILE = settings_path
        gui_config.QUEUE_STATE_FILE = Path(cls._tmpdir.name) / "queue.json"
        gui_config.save_settings(gui_config.ProcessingConfig(
            onboarding_seen=True,
            log_panel_open=False,
        ))

    @classmethod
    def tearDownClass(cls):
        cls._app_exports.SETTINGS_FILE = cls._original_export_settings
        cls._gui_config.SETTINGS_FILE = cls._original_settings
        cls._gui_config.QUEUE_STATE_FILE = cls._original_queue_state
        cls._shared_root.destroy()
        try:
            tk._default_root = None
        except Exception:
            pass
        cls._tmpdir.cleanup()

    def _make_app(self, *, withdraw: bool = True):
        self._gui_config.save_settings(self._gui_config.ProcessingConfig(
            onboarding_seen=True,
            adv_panel_open=False,
            log_panel_open=False,
        ))
        with mock.patch.object(
            self._app_exports.VideoSubtitleRemoverApp,
            "_start_startup_hardware_probe",
        ), mock.patch.object(
            self._app_exports.VideoSubtitleRemoverApp,
            "_maybe_restore_queue",
        ), mock.patch.object(
            self._gui_app_module.tk,
            "Tk",
            side_effect=lambda: tk.Toplevel(self._shared_root),
        ):
            app = self._app_exports.VideoSubtitleRemoverApp()
        app._live_region_ocr_enabled = False
        if withdraw:
            app.root.withdraw()
        else:
            app.root.deiconify()
            app.root.update()
        return app

    def _destroy_app(self, app):
        app._shutdown_started = True
        try:
            app._shutdown_ui_resources()
        finally:
            try:
                app.root.destroy()
            except tk.TclError:
                pass

    @staticmethod
    def _walk(widget):
        yield widget
        for child in widget.winfo_children():
            yield from GuiWorkflowReleaseTests._walk(child)

    @staticmethod
    def _tab_cycle(root, start, *, limit: int = 512) -> set[str]:
        if not str(root.tk.call("info", "commands", "tk_focusNext")):
            if not root.tk.call("auto_load", "tk_focusNext"):
                raise AssertionError("Tk focus traversal command is unavailable")
        paths = set()
        current = str(start)
        for _index in range(limit):
            next_path = str(root.tk.call("tk_focusNext", current) or "")
            if not next_path or next_path in paths:
                break
            paths.add(next_path)
            current = next_path
        return paths

    def test_queue_selection_and_test_cleanup_dispatch(self):
        app = self._make_app()
        try:
            source = Path(self._tmpdir.name) / "inspect.png"
            source.write_bytes(b"not a real image")
            item = self._app_exports.QueueItem(
                id="inspect-target",
                file_path=str(source),
                output_path=str(source.with_name("inspect_no_sub.png")),
                config=self._app_exports.ProcessingConfig(),
            )
            app.queue.append(item)
            app._selected_queue_item_id = None

            app._update_queue_display()
            app.root.update_idletasks()

            self.assertEqual(app._selected_queue_item_id, item.id)
            self.assertIs(app._get_selected_queue_item(), item)
            self.assertEqual(app._preview_primary_actions.winfo_manager(), "pack")
            self.assertEqual(app._preview_tools_btn.winfo_manager(), "pack")
            self.assertEqual(app.preview_status_chip.winfo_manager(), "pack")
            self.assertGreaterEqual(
                int(app.queue_canvas.cget("height")),
                app.queue_frame.winfo_reqheight(),
            )
            self.assertEqual(app._queue_table_header.winfo_manager(), "")
            self.assertEqual(app._queue_scrollbar.winfo_manager(), "")
            self.assertEqual(app._preview_heading_label.winfo_manager(), "")
            self.assertEqual(app.preview_track_plan_btn.winfo_manager(), "")
            self.assertEqual(app.preview_zoom_btn.winfo_manager(), "")
            self.assertEqual(app.queue_remove_btn.winfo_manager(), "")
            self.assertEqual(app.queue_clear_completed_btn.winfo_manager(), "")
            self.assertEqual(app._queue_more_btn.winfo_manager(), "pack")
            preview_menu = mock.Mock()
            with mock.patch.object(
                self._gui_app_module.tk,
                "Menu",
                return_value=preview_menu,
            ):
                app._open_preview_tools_menu()
            preview_menu_labels = [
                call.kwargs.get("label")
                for call in preview_menu.add_command.call_args_list
            ]
            self.assertIn("Track plan", preview_menu_labels)
            self.assertIn("Full size", preview_menu_labels)
            app._set_inspector_section("detection")
            app.root.update_idletasks()
            self.assertLessEqual(app._settings_col.winfo_reqwidth(), 400)
            self.assertTrue(app.preview_inpaint_btn.enabled)
            self.assertEqual(
                app.preview_inpaint_btn.command,
                app._open_selected_inpaint_preview,
            )
            dispatched = mock.Mock()
            app.preview_inpaint_btn.command = dispatched
            self.assertEqual(
                app.preview_inpaint_btn._on_keyboard_activate(SimpleNamespace()),
                "break",
            )
            dispatched.assert_called_once_with()
        finally:
            self._destroy_app(app)

    def test_track_review_explains_selection_and_offers_bulk_actions(self):
        app = self._make_app()
        dialog = None
        try:
            source = Path(self._tmpdir.name) / "tracks.png"
            source.write_bytes(b"not a real image")
            item = self._app_exports.QueueItem(
                id="track-review-target",
                file_path=str(source),
                output_path=str(source.with_name("tracks-clean.png")),
                config=self._app_exports.ProcessingConfig(),
            )
            app.queue.append(item)
            app._show_track_plan_dialog(item.id, {
                "fps": 24.0,
                "tracks": [
                    {
                        "start_frame": 0,
                        "end_frame": 24,
                        "sample_text": "Dialogue",
                        "keep": False,
                    },
                    {
                        "start_frame": 25,
                        "end_frame": 48,
                        "sample_text": "Logo",
                        "keep": True,
                    },
                ],
            })
            dialog = next(
                child for child in app.root.winfo_children()
                if isinstance(child, tk.Toplevel)
            )
            dialog.update_idletasks()
            visible_text = [
                str(widget.cget("text"))
                for widget in self._walk(dialog)
                if isinstance(widget, tk.Label)
            ]
            button_text = {
                str(getattr(widget, "text", ""))
                for widget in self._walk(dialog)
            }
            self.assertTrue(any(
                "Select the tracks to remove" in text
                for text in visible_text
            ))
            self.assertTrue({
                "Remove all", "Keep all", "Apply selection",
            }.issubset(button_text))
        finally:
            if dialog is not None:
                try:
                    dialog.grab_release()
                except tk.TclError:
                    pass
                dialog.destroy()
            self._destroy_app(app)

    def test_help_uses_neutral_copy_while_runtime_probe_is_active(self):
        app = self._make_app()
        dialog = None
        try:
            app._hardware_probe_pending = True
            app.ffmpeg_ready = False
            app.ai_engines = {"detection": [], "inpainting": []}
            app._show_about()
            dialog = next(
                child for child in app.root.winfo_children()
                if isinstance(child, tk.Toplevel)
            )
            dialog.update_idletasks()
            visible_text = [
                str(widget.cget("text"))
                for widget in self._walk(dialog)
                if isinstance(widget, tk.Label)
            ]
            self.assertIn("Checking system", visible_text)
            self.assertNotIn("Needs attention", visible_text)
            self.assertGreaterEqual(visible_text.count("Checking..."), 3)
        finally:
            if dialog is not None:
                try:
                    dialog.grab_release()
                except tk.TclError:
                    pass
                dialog.destroy()
            self._destroy_app(app)

    def test_region_changes_propagate_to_idle_queue_snapshots(self):
        app = self._app_exports.VideoSubtitleRemoverApp.__new__(
            self._app_exports.VideoSubtitleRemoverApp
        )
        app.config = self._app_exports.ProcessingConfig()
        app.queue = []
        app.queue_lock = threading.Lock()
        source = Path(self._tmpdir.name) / "region.png"
        source.write_bytes(b"not a real image")
        item = self._app_exports.QueueItem(
            id="region-target",
            file_path=str(source),
            output_path=str(source.with_name("region_no_sub.png")),
            config=self._app_exports.ProcessingConfig(),
        )
        app.queue.append(item)
        app.config.subtitle_area = (10, 20, 110, 54)
        app.config.subtitle_areas = [(10, 20, 110, 54), (12, 60, 108, 76)]
        app.config.subtitle_region_spans = [
            {"rect": (14, 70, 114, 92), "start": 2.0, "end": 4.0}
        ]

        self.assertEqual(app._apply_region_settings_to_idle_items(), 1)
        self.assertEqual(item.config.subtitle_area, (10, 20, 110, 54))
        self.assertEqual(item.config.subtitle_areas, app.config.subtitle_areas)
        self.assertEqual(
            item.config.subtitle_region_spans,
            app.config.subtitle_region_spans,
        )

    def test_manual_region_command_is_cross_profile_and_requires_a_region(self):
        app = self._make_app()
        try:
            source = Path(self._tmpdir.name) / "manual-region.png"
            source.write_bytes(b"not a real image")
            item = self._app_exports.QueueItem(
                id="manual-region-target",
                file_path=str(source),
                output_path=str(source.with_name("manual-region-clean.png")),
                config=self._app_exports.ProcessingConfig(),
            )
            app.queue.append(item)
            app._selected_queue_item_id = item.id

            with mock.patch.object(app, "_open_region_selector") as editor:
                app._command_region_var.set("Manual region")
                app._on_command_region_changed()

            editor.assert_called_once_with()
            self.assertFalse(app.skip_detection_var.get())
            self.assertFalse(app.config.sttn_skip_detection)
            self.assertEqual(app._command_region_var.get(), "Automatic")

            saved = (10, 20, 110, 54)
            app.config.subtitle_area = saved
            app.config.subtitle_areas = [saved]
            with mock.patch.object(app, "_show_preview") as preview:
                app._command_region_var.set("Manual region")
                app._on_command_region_changed()

            preview.assert_called_once_with(item, show_mask=True)
            self.assertEqual(app.config.subtitle_area, saved)
            self.assertTrue(app.config.sttn_skip_detection)
            self.assertTrue(item.config.sttn_skip_detection)
            self.assertEqual(item.config.subtitle_area, saved)
            self.assertEqual(app._command_region_var.get(), "Manual region")

            for mode in ("Auto", "STTN", "LAMA", "ProPainter"):
                with self.subTest(mode=mode):
                    app._on_mode_picker_changed(mode)
                    self.assertTrue(app.skip_check.enabled)
                    self.assertTrue(app.skip_detection_var.get())
                    self.assertTrue(app.config.sttn_skip_detection)

            app._command_region_var.set("Automatic")
            app._on_command_region_changed()
            self.assertFalse(app.config.sttn_skip_detection)
            self.assertFalse(item.config.sttn_skip_detection)
            self.assertEqual(app.config.subtitle_area, saved)

            with mock.patch.object(app, "_update_status") as status:
                app._reset_region()
            self.assertIsNone(app.config.subtitle_area)
            self.assertIsNone(app.config.subtitle_areas)
            self.assertFalse(app.config.sttn_skip_detection)
            self.assertIn("Cleared manual subtitle regions", status.call_args.args[0])
            self.assertTrue(status.call_args.kwargs["toast"])
        finally:
            self._destroy_app(app)

    def test_manual_region_preview_bypasses_ocr_and_labels_saved_mask(self):
        from PIL import Image

        app = self._make_app()
        try:
            source = Path(self._tmpdir.name) / "manual-mask-preview.png"
            Image.new("RGB", (320, 180), "#182132").save(source)
            saved = (40, 120, 280, 165)
            item = self._app_exports.QueueItem(
                id="manual-mask-preview",
                file_path=str(source),
                output_path=str(source.with_name("manual-mask-preview-clean.png")),
                config=self._app_exports.ProcessingConfig(
                    subtitle_area=saved,
                    subtitle_areas=[saved],
                    sttn_skip_detection=True,
                ),
            )
            app.queue.append(item)
            app._selected_queue_item_id = item.id
            app.config.subtitle_area = saved
            app.config.subtitle_areas = [saved]
            app.config.sttn_skip_detection = True
            app.skip_detection_var.set(True)
            app._preview_detector = None

            with mock.patch(
                "backend.detection.SubtitleDetector",
                side_effect=AssertionError("manual preview must bypass OCR"),
            ) as detector_type:
                app._show_preview(item, show_mask=True)
                deadline = time.monotonic() + 5.0
                while app._preview_photo is None and time.monotonic() < deadline:
                    app.root.update()
                    time.sleep(0.01)

            detector_type.assert_not_called()
            self.assertIsNotNone(app._preview_photo)
            self.assertIn("Manual mask", app._preview_label.cget("text"))
            self.assertIn("manual mask cached", app.preview_meta_label.cget("text"))
        finally:
            self._destroy_app(app)

    def test_automatic_cleanup_preview_combines_saved_region_and_ocr(self):
        from PIL import Image
        from backend import processor

        app = self._make_app()
        try:
            source = Path(self._tmpdir.name) / "automatic-cleanup-preview.png"
            Image.new("RGB", (320, 180), "#182132").save(source)
            saved = (40, 120, 120, 165)
            detected = (220, 20, 280, 55)
            item = self._app_exports.QueueItem(
                id="automatic-cleanup-preview",
                file_path=str(source),
                output_path=str(
                    source.with_name("automatic-cleanup-preview-clean.png")
                ),
                config=self._app_exports.ProcessingConfig(
                    subtitle_area=saved,
                    subtitle_areas=[saved],
                    sttn_skip_detection=False,
                    mask_dilate_px=0,
                ),
            )
            app.queue.append(item)
            app._selected_queue_item_id = item.id
            captured_masks = []

            class FakeDetector:
                def __init__(self):
                    self.calls = 0

                def detect(self, frame, threshold):
                    self.calls += 1
                    return [detected]

            class CapturingInpainter:
                def inpaint(self, frames, masks):
                    captured_masks.extend(mask.copy() for mask in masks)
                    return [frame.copy() for frame in frames]

            detector = FakeDetector()

            def build_remover(backend_cfg):
                remover = processor.SubtitleRemover.__new__(
                    processor.SubtitleRemover
                )
                remover.config = backend_cfg
                remover.detector = detector
                remover.inpainter = CapturingInpainter()
                return remover

            app._preview_remover_for = build_remover
            app._open_selected_inpaint_preview()
            deadline = time.monotonic() + 5.0
            while not captured_masks and time.monotonic() < deadline:
                app.root.update()
                time.sleep(0.01)

            self.assertEqual(detector.calls, 1)
            self.assertTrue(captured_masks)
            mask = captured_masks[0]
            self.assertEqual(int(mask[140, 80]), 255)
            self.assertEqual(int(mask[35, 250]), 255)
            while (
                "2 region(s) masked" not in app.preview_meta_label.cget("text")
                and time.monotonic() < deadline
            ):
                app.root.update()
                time.sleep(0.01)
            self.assertIn("2 region(s) masked", app.preview_meta_label.cget("text"))
        finally:
            self._destroy_app(app)

    def test_collapsed_advanced_controls_leave_control_view_and_tab_order(self):
        app = self._make_app(withdraw=False)
        try:
            section_names = {
                "detection", "inpainting", "encoding", "advanced",
            }
            labels = {
                key: accessible_metadata(button)
                for key, button in app._inspector_summary_buttons.items()
            }
            self.assertEqual(set(labels), section_names)
            self.assertTrue(all(
                metadata["role"] == "button" and metadata["label"]
                for metadata in labels.values()
            ))
            self.assertTrue(all(
                metadata["state"] == "collapsed"
                for metadata in labels.values()
            ))
            roots = [
                *app._inspector_primary_detail_roots,
                *app._inspector_advanced_cards,
            ]
            descendants = {
                widget
                for root in roots
                for widget in self._walk(root)
            }
            descendants.add(app.adv_panel)
            originally_focusable = {
                widget
                for widget in descendants
                if str(getattr(widget, "_vsr_a11y_saved_takefocus", "0"))
                not in {"", "0", "false"}
            }
            self.assertTrue(originally_focusable)
            self.assertTrue(all(
                getattr(widget, "_vsr_a11y_control_view", True) is False
                for widget in descendants
            ))
            focus_leaks = [
                (
                    type(widget).__name__,
                    str(widget),
                    str(widget.cget("takefocus")),
                    str(widget._vsr_a11y_saved_takefocus),
                )
                for widget in originally_focusable
                if str(widget.cget("takefocus")) not in {"", "0", "false"}
            ]
            self.assertEqual(focus_leaks, [])
            hidden_paths = {str(widget) for widget in originally_focusable}
            collapsed_tabs = self._tab_cycle(
                app.root, app._inspector_advanced_button
            )
            self.assertTrue(collapsed_tabs.isdisjoint(hidden_paths))

            for section in ("detection", "inpainting", "encoding", "advanced"):
                app._open_inspector_details(section)
                app.root.update_idletasks()
                states = {
                    key: accessible_metadata(button)["state"]
                    for key, button in app._inspector_summary_buttons.items()
                }
                self.assertEqual(states[section], "expanded")
                self.assertTrue(all(
                    state == "collapsed"
                    for key, state in states.items()
                    if key != section
                ))
                active = {app.adv_panel}
                for panel, _pack_options in (
                    app._inspector_section_primary_panels[section]
                ):
                    active.update(self._walk(panel))
                for panel in app._inspector_section_advanced_cards[section]:
                    active.update(self._walk(panel))
                self.assertTrue(all(
                    getattr(widget, "_vsr_a11y_control_view", False) is True
                    for widget in active
                ))
                self.assertTrue(all(
                    getattr(widget, "_vsr_a11y_control_view", True) is False
                    for widget in descendants - active
                ))
                active_focus_paths = {
                    str(widget)
                    for widget in active.intersection(originally_focusable)
                }
                expanded_tabs = self._tab_cycle(
                    app.root, app._inspector_summary_buttons[section]
                )
                self.assertTrue(expanded_tabs.intersection(active_focus_paths))
                self.assertTrue(
                    expanded_tabs.isdisjoint(hidden_paths - active_focus_paths)
                )

            app._open_inspector_details("advanced")
            app.root.update_idletasks()
            self.assertIsNone(app._inspector_open_section)
            self.assertTrue(all(
                getattr(widget, "_vsr_a11y_control_view", True) is False
                for widget in descendants
            ))
            collapsed_again = self._tab_cycle(
                app.root, app._inspector_advanced_button
            )
            self.assertTrue(collapsed_again.isdisjoint(hidden_paths))
            self.assertEqual(app.adv_panel.winfo_manager(), "")
        finally:
            self._destroy_app(app)

    def test_conditional_scrollbars_do_not_reflow_their_canvases(self):
        app = self._make_app(withdraw=False)
        try:
            app.root.update_idletasks()
            cases = (
                (
                    app._content_canvas,
                    app._content_window,
                    app._content_scrollbar,
                    app._sync_content_scrollbar,
                ),
                (
                    app.queue_canvas,
                    app.queue_window,
                    app._queue_scrollbar,
                    app._sync_queue_scrollbar,
                ),
            )
            for canvas, window, scrollbar, sync_scrollbar in cases:
                with self.subTest(canvas=str(canvas)):
                    width = canvas.winfo_width()
                    canvas.itemconfigure(
                        window, height=canvas.winfo_height() + 200)
                    canvas.configure(scrollregion=canvas.bbox("all"))
                    sync_scrollbar()
                    app.root.update_idletasks()
                    self.assertEqual(scrollbar.winfo_manager(), "place")
                    self.assertEqual(canvas.winfo_width(), width)

                    canvas.itemconfigure(window, height=1)
                    canvas.configure(scrollregion=canvas.bbox("all"))
                    sync_scrollbar()
                    app.root.update_idletasks()
                    self.assertEqual(scrollbar.winfo_manager(), "")
                    self.assertEqual(canvas.winfo_width(), width)
        finally:
            self._destroy_app(app)

    def test_shutdown_cancels_callbacks_and_detaches_log_handler(self):
        app = self._make_app()
        handler = app._log_handler
        try:
            app.root.after(60_000, lambda: None)
            pending = app.root.tk.splitlist(app.root.tk.call("after", "info"))
            self.assertTrue(pending)
            self.assertIn(handler, logging.getLogger().handlers)

            app._shutdown_ui_resources()

            self.assertEqual(
                app.root.tk.splitlist(app.root.tk.call("after", "info")), ()
            )
            self.assertNotIn(handler, logging.getLogger().handlers)
            self.assertTrue(handler._closed)
        finally:
            self._destroy_app(app)
