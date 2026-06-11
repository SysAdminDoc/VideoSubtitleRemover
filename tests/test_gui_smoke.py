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
import unittest
from pathlib import Path
from unittest import mock


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
        cls._g = g
        cls._orig_settings = g.SETTINGS_FILE
        g.SETTINGS_FILE = Path(cls._tmpdir.name) / "settings.json"

    @classmethod
    def tearDownClass(cls):
        cls._g.SETTINGS_FILE = cls._orig_settings
        cls._tmpdir.cleanup()

    def _make_app(self):
        app = self._g.VideoSubtitleRemoverApp()
        app.root.withdraw()
        return app

    def test_construct_and_close(self):
        app = self._make_app()
        try:
            self.assertEqual(app.root.title()[:23], "Video Subtitle Remover ")
            app.root.update_idletasks()
        finally:
            app.root.destroy()

    def test_sync_config_round_trip(self):
        app = self._make_app()
        try:
            # Toggle a known field via the tk variable and confirm it
            # propagates into the persisted config.
            app.preserve_audio_var.set(False)
            app._sync_config_from_ui()
            self.assertFalse(app.config.preserve_audio)
        finally:
            app.root.destroy()

    def test_apply_responsive_layout_no_op_when_unchanged(self):
        app = self._make_app()
        try:
            app._apply_responsive_layout(1280)
            initial = app._layout_mode
            app._apply_responsive_layout(1280)
            self.assertEqual(app._layout_mode, initial)
        finally:
            app.root.destroy()

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
            app.root.destroy()

    def test_batch_report_files_written_for_completed_queue(self):
        app = self._make_app()
        try:
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
            app._batch_started_at = self._g.datetime.now()

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

        finally:
            app.root.destroy()

    def test_soft_subtitle_action_remuxes_without_backend_remover(self):
        app = self._make_app()
        try:
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

        finally:
            app.root.destroy()


if __name__ == "__main__":
    unittest.main()
