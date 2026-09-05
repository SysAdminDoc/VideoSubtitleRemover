"""RM-355: the product must say which algorithms it can run, before the run.

The backend half is tested directly. The GUI half builds the real application
against a real Tk root, because the defect being fixed was that the picker and
the command-bar profile list were populated from an enum with no reference to
what the machine can actually load.

Every GUI test here needs a display and must run on an isolated desktop, which
is why the suite is invoked the way `Repo Test & Build Matrix` records.
"""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from backend.model_downloads import inpaint_mode_availability


def _have_display() -> bool:
    if sys.platform == "win32":
        return True
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


class InpaintModeAvailabilityTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(prefix="vsr-avail-")
        self.addCleanup(self._tmp.cleanup)
        self.cache = (Path(self._tmp.name) / "AppData"
                      / "VideoSubtitleRemoverPro" / "models")
        self.cache.mkdir(parents=True, exist_ok=True)
        self.env = {"APPDATA": str(Path(self._tmp.name) / "AppData")}

    def test_built_in_algorithms_are_always_available(self):
        states = inpaint_mode_availability(self.env)
        for mode in ("sttn", "auto", "propainter"):
            with self.subTest(mode=mode):
                self.assertTrue(
                    states[mode]["available"],
                    "TBE and its hybrids need no downloaded weight",
                )
                self.assertEqual(states[mode]["next_action"], "")
                self.assertEqual(states[mode]["fetch"], "")

    def test_lama_is_unavailable_with_an_empty_model_cache(self):
        states = inpaint_mode_availability(self.env)
        self.assertFalse(states["lama"]["available"])
        self.assertTrue(states["lama"]["next_action"])
        self.assertEqual(
            states["lama"]["fetch"], "opencv-lama",
            "the state has to name the adapter the fetch can resolve",
        )

    @unittest.skipUnless(
        os.environ.get("VSR_MODEL_FETCH_TESTS", "").strip().lower()
        in {"1", "true", "yes", "on"},
        "opt-in: downloads 92 MB. Set VSR_MODEL_FETCH_TESTS=1.",
    )
    def test_lama_becomes_available_once_a_verified_weight_is_installed(self):
        from backend.model_fetch import fetch_weight

        self.assertFalse(inpaint_mode_availability(self.env)["lama"]["available"])
        result = fetch_weight("opencv-lama", env=self.env)
        self.assertTrue(result.ok, result.detail)

        states = inpaint_mode_availability(self.env)
        self.assertTrue(
            states["lama"]["available"],
            "a verified weight in the cache must make the algorithm runnable",
        )
        self.assertEqual(states["lama"]["next_action"], "")
        self.assertEqual(states["lama"]["fetch"], "")

    def test_a_file_that_fails_its_hash_does_not_make_lama_available(self):
        # The gate is the digest, not the filename. A truncated or substituted
        # model sitting in the cache must not read as a working install.
        (self.cache / "inpainting_lama_2025jan.onnx").write_bytes(b"not a model")
        states = inpaint_mode_availability(self.env)
        self.assertFalse(states["lama"]["available"])
        self.assertEqual(states["lama"]["fetch"], "opencv-lama")

    def test_migan_follows_its_environment_variable(self):
        self.assertFalse(inpaint_mode_availability(self.env)["migan"]["available"])
        model = self.cache / "migan_pipeline_v2.onnx"
        model.write_bytes(b"placeholder")
        with_model = dict(self.env, VSR_MIGAN_ONNX=str(model))
        self.assertTrue(inpaint_mode_availability(with_model)["migan"]["available"])

    def test_the_lama_next_action_leads_with_the_download(self):
        action = inpaint_mode_availability(self.env)["lama"]["next_action"]
        self.assertTrue(
            action.lower().startswith("download"),
            f"the fetch is the first thing to offer, got {action!r}",
        )


@unittest.skipUnless(_have_display(), "GUI availability test needs a display")
class AlgorithmPickerAvailabilityTests(unittest.TestCase):
    """Drive the real application, not a stand-in for it."""

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
        cls._shared_root = tk.Tk()
        cls._shared_root.withdraw()
        cls._original_export_settings = app_exports.SETTINGS_FILE
        cls._original_settings = gui_config.SETTINGS_FILE
        cls._original_queue_state = gui_config.QUEUE_STATE_FILE
        settings_path = Path(cls._tmpdir.name) / "settings.json"
        app_exports.SETTINGS_FILE = settings_path
        gui_config.SETTINGS_FILE = settings_path
        gui_config.QUEUE_STATE_FILE = Path(cls._tmpdir.name) / "queue.json"

    @classmethod
    def tearDownClass(cls):
        cls._app_exports.SETTINGS_FILE = cls._original_export_settings
        cls._gui_config.SETTINGS_FILE = cls._original_settings
        cls._gui_config.QUEUE_STATE_FILE = cls._original_queue_state
        cls._shared_root.destroy()
        try:
            cls.tk._default_root = None
        except AttributeError:
            pass
        cls._tmpdir.cleanup()

    def _make_app(self):
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

    @staticmethod
    def _states(available_lama: bool) -> dict:
        return {
            "auto": {"available": True, "reason": "", "next_action": "",
                     "fetch": ""},
            "sttn": {"available": True, "reason": "", "next_action": "",
                     "fetch": ""},
            "propainter": {"available": True, "reason": "", "next_action": "",
                           "fetch": ""},
            "lama": {
                "available": available_lama,
                "reason": "" if available_lama else "no LaMa model",
                "next_action": "" if available_lama else "Download the model.",
                "fetch": "" if available_lama else "opencv-lama",
            },
            "migan": {"available": False, "reason": "", "next_action": "x",
                      "fetch": ""},
        }

    def _apply(self, app, available_lama: bool):
        """Deliver a probe result the way the background thread would."""
        mapped = {
            gui_mode: self._states(available_lama)[backend_mode]
            for gui_mode, backend_mode
            in app._ALGO_BACKEND_MODES.items()
        }
        app._apply_algorithm_availability(mapped)
        app.root.update_idletasks()

    def test_an_unrunnable_algorithm_is_disabled_in_the_picker(self):
        app = self._make_app()
        self._apply(app, available_lama=False)
        self.assertFalse(app.mode_picker.option_enabled("LAMA"))
        for mode in ("Auto", "STTN", "ProPainter"):
            with self.subTest(mode=mode):
                self.assertTrue(app.mode_picker.option_enabled(mode))

    def test_installing_the_model_re_enables_it(self):
        app = self._make_app()
        self._apply(app, available_lama=False)
        self.assertFalse(app.mode_picker.option_enabled("LAMA"))
        self._apply(app, available_lama=True)
        self.assertTrue(
            app.mode_picker.option_enabled("LAMA"),
            "the state has to flip back once the weight is installed",
        )

    def test_the_command_bar_marks_the_profile_before_it_is_chosen(self):
        app = self._make_app()
        self._apply(app, available_lama=False)
        values = list(app._command_mode_combo.cget("values"))
        suffix = app._command_profile_suffix()
        marked = [value for value in values if value.endswith(suffix)]
        self.assertEqual(
            len(marked), 1,
            f"exactly the Detail profile should be marked, got {values}",
        )
        self.assertTrue(marked[0].startswith(app._command_profile_labels()["LAMA"]))

        self._apply(app, available_lama=True)
        values = list(app._command_mode_combo.cget("values"))
        self.assertEqual(
            [value for value in values if value.endswith(suffix)], [],
            "nothing stays marked once every profile can run",
        )

    def test_a_marked_label_still_selects_its_mode(self):
        app = self._make_app()
        self._apply(app, available_lama=False)
        suffix = app._command_profile_suffix()
        app._command_profile_var.set(
            app._command_profile_labels()["LAMA"] + suffix)
        app._on_command_profile_changed()
        self.assertEqual(
            app.mode_var.get(), "LAMA",
            "the suffix must be stripped, not fall through to Auto",
        )

    def test_choosing_it_explains_what_to_do_and_offers_the_download(self):
        app = self._make_app()
        self._apply(app, available_lama=False)
        app.mode_var.set("LAMA")
        app._refresh_algorithm_availability()
        app.root.update_idletasks()

        self.assertTrue(app.algo_unavailable_row.winfo_ismapped()
                        or app.algo_unavailable_row.winfo_manager())
        self.assertEqual(
            app.algo_unavailable_label.cget("text"), "Download the model.")
        self.assertTrue(
            app.algo_fetch_btn.winfo_manager(),
            "an algorithm with a fetch route must offer it here",
        )

    def test_a_runnable_algorithm_shows_no_hint(self):
        app = self._make_app()
        self._apply(app, available_lama=True)
        app.mode_var.set("LAMA")
        app._refresh_algorithm_availability()
        app.root.update_idletasks()
        self.assertFalse(app.algo_unavailable_row.winfo_manager())

    def test_the_picker_offers_exactly_the_modes_the_gui_enum_declares(self):
        from gui.config import InpaintMode

        app = self._make_app()
        self.assertEqual(
            app.mode_picker.option_values(),
            [mode.value for mode in InpaintMode],
        )
        self.assertNotIn(
            "migan", [value.lower() for value in app.mode_picker.option_values()],
            "MiGAN is a CLI-only registry mode and must not reach this picker",
        )

    def test_a_third_click_cannot_start_a_second_download(self):
        """Cancel is not instant, and the button must not re-arm before it is.

        The worker only checks its cancel flag once per megabyte read, so
        there is a real window after the flag is set and before the worker
        reports back. Deciding from is_set() alone let a click inside that
        window start a second worker writing the same .part file.
        """
        import threading

        app = self._make_app()
        self._apply(app, available_lama=False)
        app.mode_var.set("LAMA")
        app._refresh_algorithm_availability()

        started = []
        real_thread = threading.Thread

        def _capture(*args, **kwargs):
            started.append(kwargs.get("name") or "")
            thread = real_thread(target=lambda: None, daemon=True)
            return thread

        with mock.patch.object(threading, "Thread", _capture):
            app._download_missing_algorithm_model()   # starts one
            app._download_missing_algorithm_model()   # cancels it
            app._download_missing_algorithm_model()   # must do nothing
            app._download_missing_algorithm_model()   # still nothing

        self.assertEqual(
            [name for name in started if name == "vsr-model-fetch"],
            ["vsr-model-fetch"],
            "only one fetch worker may exist at a time",
        )
        self.assertTrue(
            app._model_fetch_cancel.is_set(),
            "the in-flight fetch is still the cancelled one",
        )

        app._apply_model_fetch_result("LAMA", False, "cancelled")
        self.assertIsNone(
            app._model_fetch_cancel,
            "the worker reporting back is what clears the in-flight fetch",
        )

    def test_a_download_reports_bytes_not_just_a_spinner(self):
        """RM-328: a multi-gigabyte fetch used to be one toast and 2 percent.

        The worker is driven with a real fetch_weight call against a local
        server, so this exercises the progress plumbing rather than asserting
        the callback shape.
        """
        import hashlib
        import http.server
        import threading as _threading

        from backend.adapter_manifest import ADAPTER_MANIFEST, AdapterManifestEntry
        from backend import model_fetch

        body = b"weight bytes " * 200000  # ~2.6 MB, several chunks
        name = "vsr_progress_weight.onnx"
        entry = AdapterManifestEntry(
            name="vsr-progress",
            env_vars=(),
            expected_filenames=(name,),
            sha256={name: hashlib.sha256(body).hexdigest()},
            repository="vsr/progress",
            revision="0" * 40,
        )

        class _Handler(http.server.BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def do_GET(self):  # noqa: N802 - stdlib callback name
                self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *_a):
                return

        server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        thread = _threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        host, port = server.server_address
        endpoint = f"http://{host}:{port}"

        ADAPTER_MANIFEST["vsr-progress"] = entry
        self.addCleanup(ADAPTER_MANIFEST.pop, "vsr-progress", None)
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)

        app = self._make_app()
        lines = []
        app._update_status = lambda text, tone="neutral", toast=False: (
            lines.append(text))

        import tempfile
        import time

        from backend.model_downloads import format_download_progress

        with tempfile.TemporaryDirectory(prefix="vsr-prog-") as tmp:
            started = time.monotonic()
            last = [-1]
            reported = []

            def _progress(read, total):
                percent = int(read * 100 / total) if total else -1
                if total and percent == last[0]:
                    return
                last[0] = percent
                reported.append((read, total))
                lines.append(format_download_progress(
                    name, read, total, time.monotonic() - started))

            result = model_fetch.fetch_weight(
                "vsr-progress", name,
                env={"APPDATA": tmp}, endpoint=endpoint,
                progress=_progress,
            )

        self.assertTrue(result.ok, result.detail)
        self.assertGreater(len(reported), 1, "one update is a spinner")
        self.assertEqual(
            [read for read, _ in reported],
            sorted(read for read, _ in reported),
            "progress must be monotonic",
        )
        self.assertEqual(reported[-1][0], len(body))
        byte_lines = [line for line in lines if "MB" in line]
        self.assertTrue(byte_lines, "no line reported a byte count")
        self.assertIn(name, byte_lines[-1])
        self.assertIn("%", byte_lines[-1],
                      "the total is known, so a percentage is expected")

    def test_the_download_guidance_is_shown_not_only_logged(self):
        app = self._make_app()
        self._apply(app, available_lama=False)
        app.mode_var.set("LAMA")
        app._refresh_algorithm_availability()

        said = []
        app._update_status = lambda text, tone="neutral", toast=False: (
            said.append(text))
        import threading as _threading

        with mock.patch.object(_threading, "Thread",
                               lambda *a, **k: mock.Mock()):
            app._download_missing_algorithm_model()

        self.assertTrue(said, "starting a download said nothing")
        self.assertIn(
            "Keep this window open", said[0],
            "the guidance text was logged rather than shown",
        )

    def test_availability_is_not_probed_on_the_ui_thread(self):
        app = self._make_app()
        app._algo_availability = None
        with mock.patch(
            "backend.model_downloads.inpaint_mode_availability"
        ) as probe:
            states = app.algorithm_availability(refresh=True)
        self.assertEqual(
            probe.call_count, 0,
            "the probe imports onnxruntime and must stay off the UI thread",
        )
        self.assertTrue(all(state["available"] for state in states.values()),
                        "before the answer arrives, nothing is claimed missing")


if __name__ == "__main__":
    unittest.main()
