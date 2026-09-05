"""RM-358: the support bundle has to be reachable from the failure.

The bug report template asks every reporter to attach a redacted support
bundle. The only way to make one was a ghost button inside the About dialog,
and the reporter of issue #11 wrote that they could not find the option.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest import mock


def _have_display() -> bool:
    if sys.platform == "win32":
        return True
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


class BundleCarriesTheFailureTests(unittest.TestCase):
    """A bundle asked for because something failed must say what failed."""

    def test_extra_facts_reach_the_written_bundle(self):
        from backend.support_bundle import create_support_bundle

        failed = [{
            "id": "item-1",
            "message": "The selected inpaint engine is not available",
            "failure_reason": "dependency_missing",
            "mode": "LAMA",
            "source_suffix": ".mkv",
            "source_bytes": 1234,
        }]
        with tempfile.TemporaryDirectory(prefix="vsr-bundle-") as tmp:
            out = Path(tmp) / "support.zip"
            create_support_bundle(
                out, app_version="0.0.0",
                extra_facts={"failed_items": failed},
            )
            self.assertTrue(out.is_file())
            with zipfile.ZipFile(out) as archive:
                blob = "\n".join(
                    archive.read(name).decode("utf-8", "replace")
                    for name in archive.namelist()
                    if name.lower().endswith(".json")
                )

        self.assertIn("dependency_missing", blob)
        self.assertIn("The selected inpaint engine is not available", blob)
        self.assertIn("item-1", blob)


@unittest.skipUnless(_have_display(), "GUI reach test needs a display")
class SupportBundleReachTests(unittest.TestCase):
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
        cls._shared_root.destroy()
        try:
            cls.tk._default_root = None
        except AttributeError:
            pass
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

    @staticmethod
    def _labels(widget):
        found = []
        stack = [widget]
        while stack:
            node = stack.pop()
            for attr in ("label", "text"):
                value = getattr(node, attr, None)
                if isinstance(value, str) and value:
                    found.append(value)
            stack.extend(node.winfo_children())
        return found

    def _summary_dialog(self, app, errors: int):
        opened = []
        real_toplevel = self.tk.Toplevel

        def _capture(*args, **kwargs):
            window = real_toplevel(*args, **kwargs)
            opened.append(window)
            return window

        with mock.patch.object(self._gui_app_module.tk, "Toplevel", _capture):
            app._show_batch_summary(
                complete=0, errors=errors, cancelled=0, elapsed="0:01")
        app.root.update_idletasks()
        self.assertTrue(opened, "the summary dialog did not open")
        dialog = opened[-1]
        self.addCleanup(lambda: dialog.winfo_exists() and dialog.destroy())
        return dialog

    def test_a_failed_batch_offers_the_bundle_on_its_summary(self):
        app = self._make_app()
        dialog = self._summary_dialog(app, errors=1)
        self.assertIn(
            "Create support bundle", self._labels(dialog),
            "the bug template asks for this file; the failure is where a "
            "user goes looking for it",
        )

    def test_a_clean_batch_does_not_offer_it(self):
        app = self._make_app()
        dialog = self._summary_dialog(app, errors=0)
        self.assertNotIn("Create support bundle", self._labels(dialog))

    def test_the_summary_action_is_the_same_one_the_about_dialog_uses(self):
        app = self._make_app()
        self._summary_dialog(app, errors=1)
        button = app._batch_summary_bundle_button
        self.assertEqual(
            button.command, app._save_support_bundle,
            "a second bundle writer would drift from the first",
        )

    def test_the_bundle_records_which_queue_items_failed(self):
        from gui.config import ProcessingStatus

        app = self._make_app()
        with tempfile.TemporaryDirectory(prefix="vsr-src-") as tmp:
            source = Path(tmp) / "clip.mkv"
            source.write_bytes(b"0" * 4096)
            item = self._gui_config.QueueItem(
                id="job-7",
                file_path=str(source),
                output_path=str(Path(tmp) / "out.mp4"),
                config=self._gui_config.ProcessingConfig(),
            )
            item.status = ProcessingStatus.ERROR
            item.message = "The selected inpaint engine is not available"
            item.failure_reason = "dependency_missing"
            app.queue = [item]

            facts = app._failed_queue_item_facts()

        self.assertEqual(len(facts), 1)
        self.assertEqual(facts[0]["id"], "job-7")
        self.assertEqual(facts[0]["failure_reason"], "dependency_missing")
        self.assertEqual(facts[0]["source_suffix"], ".mkv")
        self.assertEqual(facts[0]["source_bytes"], 4096)
        blob = json.dumps(facts)
        self.assertNotIn(
            "clip.mkv", blob,
            "the bundle is redacted; a file name is the one field here that "
            "can carry personal information",
        )

    def test_the_bundle_the_button_writes_carries_the_failure(self):
        """Drive _save_support_bundle itself, not just its helper.

        Asserting on _failed_queue_item_facts() alone passes even when nothing
        puts its answer into extra_facts, which is the whole point of the
        change. Only the file-picker is substituted here; the bundle is really
        written and really read back.
        """
        from gui.config import ProcessingStatus
        from gui import support_controller

        app = self._make_app()
        with tempfile.TemporaryDirectory(prefix="vsr-e2e-") as tmp:
            source = Path(tmp) / "clip.mkv"
            source.write_bytes(b"0" * 2048)
            item = self._gui_config.QueueItem(
                id="job-42",
                file_path=str(source),
                output_path=str(Path(tmp) / "out.mp4"),
                config=self._gui_config.ProcessingConfig(),
            )
            item.status = ProcessingStatus.ERROR
            item.message = "Processing failed"
            item.failure_reason = "decode_error"
            app.queue = [item]
            app.ai_engines = {"detection": [], "inpainting": []}
            app.gpus = []
            app.ffmpeg_ready = True

            target = Path(tmp) / "bundle.zip"
            finished = []
            with mock.patch.object(
                support_controller.filedialog, "asksaveasfilename",
                return_value=str(target),
            ), mock.patch.object(
                support_controller.SupportControllerMixin,
                "_run_support_task",
                new=lambda self, busy, err, work, describe: finished.append(
                    work()),
            ):
                app._save_support_bundle()

            self.assertEqual(len(finished), 1, "the bundle was never built")
            self.assertTrue(target.is_file())
            with zipfile.ZipFile(target) as archive:
                blob = "\n".join(
                    archive.read(name).decode("utf-8", "replace")
                    for name in archive.namelist()
                    if name.lower().endswith(".json")
                )

        self.assertIn("job-42", blob)
        self.assertIn("decode_error", blob)
        self.assertNotIn("clip.mkv", blob)

    def test_a_path_inside_the_failure_message_is_scrubbed(self):
        """The message is usually str(exc), and raises interpolate paths.

        The bundle's default rule deliberately keeps leaf filenames in log
        lines. This block promises no filename, so it needs the strict rule,
        and an earlier version of this test could not tell the difference
        because it only ever set message to a canonical string with no path
        in it.
        """
        from gui.config import ProcessingStatus

        app = self._make_app()
        item = self._gui_config.QueueItem(
            id="job-9", file_path="x.mkv", output_path="y.mp4",
            config=self._gui_config.ProcessingConfig(),
        )
        item.status = ProcessingStatus.ERROR
        item.message = (
            "Failed to write output image: "
            r"C:\Users\alice\Videos\family_vacation_private.mp4"
        )
        item.failure_reason = "write_error"
        app.queue = [item]

        message = app._failed_queue_item_facts()[0]["message"]
        self.assertNotIn("family_vacation_private", message)
        self.assertNotIn("alice", message)
        self.assertIn(
            "Failed to write output image", message,
            "scrubbing must not throw the diagnosis away with the path",
        )

    def test_items_that_did_not_fail_are_not_reported(self):
        from gui.config import ProcessingStatus

        app = self._make_app()
        item = self._gui_config.QueueItem(
            id="job-ok", file_path="x.mkv", output_path="y.mp4",
            config=self._gui_config.ProcessingConfig(),
        )
        item.status = ProcessingStatus.COMPLETE
        app.queue = [item]
        self.assertEqual(app._failed_queue_item_facts(), [])


if __name__ == "__main__":
    unittest.main()
