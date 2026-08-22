"""Active release checks for the smallest critical GUI workflow."""

from __future__ import annotations

import logging
import os
import sys
import tempfile
import threading
import tkinter as tk
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from backend.a11y import accessible_metadata
from gui.preview_controller import PreviewControllerMixin
from gui.utils import (
    dispatch_to_ui,
    install_ui_dispatcher,
    stop_ui_dispatcher,
)


def _have_display() -> bool:
    if sys.platform == "win32":
        return True
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


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

            self.assertEqual(app._selected_queue_item_id, item.id)
            self.assertIs(app._get_selected_queue_item(), item)
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

    def test_collapsed_advanced_controls_leave_control_view_and_tab_order(self):
        app = self._make_app(withdraw=False)
        try:
            labels = {
                key: accessible_metadata(button)
                for key, button in app._inspector_summary_buttons.items()
            }
            self.assertEqual(set(labels), {
                "detection", "inpainting", "encoding", "advanced",
            })
            self.assertTrue(all(
                metadata["role"] == "button" and metadata["label"]
                for metadata in labels.values()
            ))
            self.assertEqual(labels["advanced"]["state"], "collapsed")
            roots = [
                panel for panel, _pack_options in app._inspector_detail_panels
            ] + [app.adv_panel]
            descendants = {
                widget
                for root in roots
                for widget in self._walk(root)
            }
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

            app._toggle_advanced()
            app.root.update_idletasks()
            self.assertEqual(
                accessible_metadata(app._inspector_advanced_button)["state"],
                "expanded",
            )
            self.assertTrue(all(
                getattr(widget, "_vsr_a11y_control_view", False) is True
                for widget in descendants
            ))
            self.assertTrue(any(
                str(widget.cget("takefocus")) not in {"", "0", "false"}
                for widget in originally_focusable
            ))
            expanded_tabs = self._tab_cycle(
                app.root, app._inspector_advanced_button
            )
            self.assertTrue(expanded_tabs.intersection(hidden_paths))

            app._toggle_advanced()
            app.root.update_idletasks()
            self.assertEqual(
                accessible_metadata(app._inspector_advanced_button)["state"],
                "collapsed",
            )
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
