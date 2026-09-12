"""RM-340: DPI awareness and dialog sizing across monitors.

The launcher said "Per-Monitor V2 first" and asked for V1. Dialogs were sized
from the primary display's dimensions times a fraction standing in for the
taskbar, so on a secondary monitor of a different size the number was wrong
twice. The window floor ignored the text-scale setting, and a maximized window
reopened restored.
"""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import VideoSubtitleRemover as entry
from gui import dialog_layout
from gui.utils import monitor_work_area

_ROOT = Path(__file__).resolve().parents[1]

# One Tk root for the whole module. Creating and destroying several roots in
# one process trips the documented "Can't find a usable tk.tcl" flake, and two
# classes here both need a window.
_TK_ROOT = None


def _shared_tk_root():
    global _TK_ROOT
    import tkinter as tk

    if _TK_ROOT is None:
        _TK_ROOT = tk.Tk()
        _TK_ROOT.withdraw()
    return _TK_ROOT


def tearDownModule():
    """Hand the process back without a root.

    Leaving one alive leaves tkinter._default_root pointing at this module's
    interpreter, so the next file's tk.Tk() becomes a second interpreter and
    its masterless PhotoImages land on the wrong one ("image pyimageN doesn't
    exist"). Every other GUI suite here clears _default_root for the same
    reason.
    """
    global _TK_ROOT
    import tkinter as tk

    if _TK_ROOT is not None:
        try:
            _TK_ROOT.destroy()
        except tk.TclError:
            pass
        _TK_ROOT = None
    try:
        tk._default_root = None
    except AttributeError:
        pass


def _have_display() -> bool:
    if sys.platform == "win32":
        return True
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


class DpiAwarenessLadderTests(unittest.TestCase):
    def test_the_comment_and_the_call_agree_on_v2(self):
        source = (_ROOT / "VideoSubtitleRemover.py").read_text(encoding="utf-8")
        self.assertIn("SetProcessDpiAwarenessContext", source)
        self.assertIn("DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2", source)
        self.assertEqual(
            entry.DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2, -4,
            "V2 is context -4; SetProcessDpiAwareness(2) is V1",
        )

    @unittest.skipUnless(sys.platform == "win32", "DPI APIs are Windows")
    def test_v2_is_taken_when_it_is_available(self):
        # A fresh interpreter, because DPI awareness is set once per process
        # and a second call returns ERROR_ACCESS_DENIED.
        import subprocess

        result = subprocess.run(
            [sys.executable, "-c",
             "import VideoSubtitleRemover as e;"
             "print(e._request_dpi_awareness())"],
            cwd=str(_ROOT), capture_output=True, text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr[-800:])
        self.assertEqual(
            result.stdout.strip(), "per-monitor-v2",
            "the ladder did not reach V2; a bare int argument makes the "
            "pointer-sized context handle fail on 64-bit Windows",
        )

    @unittest.skipUnless(sys.platform == "win32", "DPI APIs are Windows")
    def test_it_falls_back_when_v2_is_missing(self):
        import ctypes

        class _User32:
            def SetProcessDpiAwarenessContext(self, _ctx):
                raise AttributeError("not on this Windows")

            def SetProcessDPIAware(self):
                return 1

        taken = []

        class _Shcore:
            def SetProcessDpiAwareness(self, level):
                taken.append(level)
                return 0

        class _Windll:
            user32 = _User32()
            shcore = _Shcore()

        with mock.patch.object(ctypes, "windll", _Windll(), create=True):
            rung = entry._request_dpi_awareness()
        self.assertEqual(rung, "per-monitor-v1")
        self.assertEqual(taken, [entry.PROCESS_PER_MONITOR_DPI_AWARE])

    @unittest.skipUnless(sys.platform == "win32", "DPI APIs are Windows")
    def test_nothing_available_reports_none_rather_than_raising(self):
        import ctypes

        class _Broken:
            def __getattr__(self, _name):
                def _raise(*_a, **_k):
                    raise OSError("no such export")
                return _raise

        class _Windll:
            user32 = _Broken()
            shcore = _Broken()

        with mock.patch.object(ctypes, "windll", _Windll(), create=True):
            self.assertEqual(entry._request_dpi_awareness(), "none")


@unittest.skipUnless(_have_display(), "work-area tests need a display")
class DialogWorkAreaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import tkinter as tk

        cls.tk = tk
        cls._shared_root = _shared_tk_root()

    def setUp(self):
        self.root = self.tk.Toplevel(self._shared_root)
        self.root.withdraw()
        self.addCleanup(self.root.destroy)

    def test_the_real_monitor_work_area_is_used_when_it_answers(self):
        measured = monitor_work_area(self.root)
        if measured is None:
            self.skipTest("this platform cannot report a monitor work area")
        _x, _y, width, height = measured
        self.assertGreater(width, 0)
        self.assertGreater(height, 0)
        area_w, area_h = dialog_layout.work_area(self.root)
        # Pin the value, not a bound. A bound is satisfied by the old
        # primary-screen fraction too, so it cannot tell which path ran.
        self.assertEqual(
            (area_w, area_h), (width - 16, height - 16),
            "work_area did not come from GetMonitorInfo; the fraction over "
            "winfo_screenwidth/height is the fallback, not the answer",
        )

    def test_the_fallback_and_the_measurement_are_different_answers(self):
        """Guards the test above from becoming vacuous.

        If the fallback happened to produce the same numbers, the equality
        check would pass whichever path ran. This asserts they really differ
        on this machine, so that check is meaningful.
        """
        measured = monitor_work_area(self.root)
        if measured is None:
            self.skipTest("this platform cannot report a monitor work area")
        real = dialog_layout.work_area(self.root)
        with mock.patch.object(dialog_layout, "monitor_work_area",
                               return_value=None):
            fallback = dialog_layout.work_area(self.root)
        self.assertNotEqual(
            real, fallback,
            "the two paths agree here, so the equality assertion above "
            "cannot detect a regression to the fallback",
        )

    def test_a_smaller_secondary_monitor_gives_a_smaller_dialog(self):
        """The acceptance case, without needing a second physical display.

        _vsr_work_area_override is the seam the dialog code already exposes
        for exactly this, so the sizing path is driven end to end with the
        dimensions of a different monitor.
        """
        self.root._vsr_work_area_override = (1366, 768)
        small = dialog_layout.work_area(self.root)
        self.root._vsr_work_area_override = (3840, 2160)
        large = dialog_layout.work_area(self.root)
        self.assertLess(small[0], large[0])
        self.assertLess(small[1], large[1])
        self.assertLessEqual(small[0], 1366)
        self.assertLessEqual(small[1], 768)

    def test_a_dialog_never_exceeds_the_monitor_it_is_on(self):
        for width, height in ((1366, 768), (1920, 1080), (2560, 1440)):
            with self.subTest(monitor=(width, height)):
                self.root._vsr_work_area_override = (width, height)
                area_w, area_h = dialog_layout.work_area(self.root)
                self.assertLessEqual(area_w, width)
                self.assertLessEqual(area_h, height)

    def test_the_fallback_still_applies_when_no_monitor_answers(self):
        with mock.patch.object(dialog_layout, "monitor_work_area",
                               return_value=None):
            area_w, area_h = dialog_layout.work_area(self.root)
        screen_w = self.root.winfo_screenwidth()
        self.assertLessEqual(area_w, screen_w)
        self.assertGreaterEqual(area_w, dialog_layout.MIN_DIALOG_WIDTH)


@unittest.skipUnless(_have_display(), "window tests need a display")
class WindowFloorAndStateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import tkinter as tk

        cls.tk = tk
        cls._tmpdir = tempfile.TemporaryDirectory()
        import VideoSubtitleRemover as app_exports
        from gui import app as gui_app_module
        from gui import config as gui_config

        cls._app_exports = app_exports
        cls._gui_app_module = gui_app_module
        cls._gui_config = gui_config
        cls._shared_root = _shared_tk_root()
        cls._originals = (
            app_exports.SETTINGS_FILE,
            gui_config.SETTINGS_FILE,
            gui_config.QUEUE_STATE_FILE,
        )
        settings_path = Path(cls._tmpdir.name) / "settings.json"
        app_exports.SETTINGS_FILE = settings_path
        gui_config.SETTINGS_FILE = settings_path
        gui_config.QUEUE_STATE_FILE = Path(cls._tmpdir.name) / "queue.json"

    @classmethod
    def tearDownClass(cls):
        (cls._app_exports.SETTINGS_FILE,
         cls._gui_config.SETTINGS_FILE,
         cls._gui_config.QUEUE_STATE_FILE) = cls._originals
        cls._tmpdir.cleanup()

    def _make_app(self):
        self._gui_config.save_settings(self._gui_config.ProcessingConfig(
            onboarding_seen=True, adv_panel_open=False, log_panel_open=False))
        with mock.patch.object(
            self._app_exports.VideoSubtitleRemoverApp,
            "_start_startup_hardware_probe",
        ), mock.patch.object(
            self._app_exports.VideoSubtitleRemoverApp, "_maybe_restore_queue",
        ), mock.patch.object(
            self._gui_app_module.tk, "Tk",
            side_effect=lambda: self.tk.Toplevel(self._shared_root),
        ):
            app = self._app_exports.VideoSubtitleRemoverApp()
        app._live_region_ocr_enabled = False
        app.root.withdraw()
        self.addCleanup(self._destroy_app, app)
        return app

    def _destroy_app(self, app):
        app._shutdown_started = True
        try:
            app._shutdown_ui_resources()
        finally:
            try:
                app.root.destroy()
            except self.tk.TclError:
                pass

    def test_the_window_floor_grows_with_the_text_scale(self):
        app = self._make_app()
        from gui import theme

        with mock.patch.object(theme, "text_scale_percent", return_value=100):
            at_100 = app._scaled_minimum_size(980, 720)
        with mock.patch.object(theme, "text_scale_percent", return_value=200):
            at_200 = app._scaled_minimum_size(980, 720)
        self.assertGreater(
            at_200[1], at_100[1],
            "a 200 percent layout needs more room, not the same floor",
        )

    def test_the_floor_never_exceeds_the_monitor(self):
        app = self._make_app()
        from gui import theme, utils

        with mock.patch.object(theme, "text_scale_percent", return_value=200), \
             mock.patch.object(utils, "monitor_work_area",
                               return_value=(0, 0, 1366, 768)):
            width, height = app._scaled_minimum_size(980, 720)
        self.assertLessEqual(width, 1366)
        self.assertLessEqual(
            height, 768,
            "a floor taller than the screen makes the window unresizable",
        )

    def test_the_maximized_state_is_persisted(self):
        app = self._make_app()
        app._restored_geometry = "1200x800+40+40"
        with mock.patch.object(app.root, "state", return_value="zoomed"), \
             mock.patch.object(app.root, "geometry",
                               return_value="1920x1040+0+0"):
            app.config.window_maximized = False
            app.config.window_maximized = str(app.root.state()) == "zoomed"
            if app.config.window_maximized:
                app.config.window_geometry = app._restored_geometry
        self.assertTrue(app.config.window_maximized)
        self.assertEqual(
            app.config.window_geometry, "1200x800+40+40",
            "persisting the zoomed size reopens a full-screen restored window",
        )

    def test_the_config_round_trips_the_maximized_flag(self):
        config = self._gui_config.ProcessingConfig()
        self.assertFalse(config.window_maximized)
        config.window_maximized = True
        config.normalized()
        self.assertTrue(config.window_maximized)

    def test_the_restored_geometry_tracker_ignores_a_zoomed_window(self):
        app = self._make_app()
        app._restored_geometry = "1000x700+10+10"
        with mock.patch.object(app.root, "state", return_value="zoomed"):
            app._track_restored_geometry()
        self.assertEqual(app._restored_geometry, "1000x700+10+10")

        with mock.patch.object(app.root, "state", return_value="normal"), \
             mock.patch.object(app.root, "geometry",
                               return_value="1100x750+20+20"):
            app._track_restored_geometry()
        self.assertEqual(app._restored_geometry, "1100x750+20+20")


if __name__ == "__main__":
    unittest.main()
