import json
import logging
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

from gui import config as gui_config
from gui.app import VideoSubtitleRemoverApp
from gui.config import InpaintMode, ProcessingConfig, ProcessingStatus, QueueItem
from gui.widgets import DragDropFrame
from gui.widgets import TextWidgetHandler
from gui.widgets import _build_cli_command


class SettingsLoadFeedbackTests(unittest.TestCase):
    def tearDown(self):
        gui_config.consume_settings_load_notice()

    def test_settings_corruption_notice_is_consumed_once(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            original = gui_config.SETTINGS_FILE
            try:
                gui_config.SETTINGS_FILE = Path(tmpdir) / "settings.json"
                gui_config.SETTINGS_FILE.write_text(
                    json.dumps(["bad"]),
                    encoding="utf-8",
                )

                cfg = gui_config.load_settings()

                self.assertIsInstance(cfg, gui_config.ProcessingConfig)
                self.assertEqual(
                    gui_config.consume_settings_load_notice(),
                    "Settings were corrupted and reset to defaults."
                    " A backup was saved as settings.json.bak.",
                )
                self.assertIsNone(gui_config.consume_settings_load_notice())
            finally:
                gui_config.SETTINGS_FILE = original

    def test_invalid_region_rectangles_are_logged(self):
        payload = {
            "subtitle_area": [20, 20, 10, 25],
            "subtitle_areas": [[1, 2, 5, 8], ["bad"], [4, 4, 4, 10]],
        }

        with self.assertLogs("gui.config", level="WARNING") as caught:
            cfg = gui_config.ProcessingConfig.from_dict(payload)

        self.assertIsNone(cfg.subtitle_area)
        self.assertEqual(cfg.subtitle_areas, [(1, 2, 5, 8)])
        joined = "\n".join(caught.output)
        self.assertIn("subtitle_area", joined)
        self.assertIn("subtitle_areas[1]", joined)
        self.assertIn("subtitle_areas[2]", joined)

    def test_oversized_settings_file_resets_with_notice(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            original = gui_config.SETTINGS_FILE
            try:
                gui_config.SETTINGS_FILE = Path(tmpdir) / "settings.json"
                gui_config.SETTINGS_FILE.write_text(
                    " " * (gui_config.MAX_JSON_OBJECT_BYTES + 1),
                    encoding="utf-8",
                )

                with self.assertLogs("gui.config", level="WARNING") as caught:
                    cfg = gui_config.load_settings()

                self.assertIsInstance(cfg, gui_config.ProcessingConfig)
                self.assertEqual(
                    gui_config.consume_settings_load_notice(),
                    "Settings were corrupted and reset to defaults."
                    " A backup was saved as settings.json.bak.",
                )
                self.assertIn("file is too large", "\n".join(caught.output))
            finally:
                gui_config.SETTINGS_FILE = original


class DropFeedbackTests(unittest.TestCase):
    def test_drop_widget_passes_unsupported_paths_to_app_summarizer(self):
        received = []
        frame = object.__new__(DragDropFrame)
        frame.tk = SimpleNamespace(splitlist=lambda value: tuple(value))
        frame.on_drop = lambda files: received.append(files)

        event = SimpleNamespace(data=("clip.mp4", "notes.txt"))
        DragDropFrame._handle_drop(frame, event)

        self.assertEqual(received, [["clip.mp4", "notes.txt"]])

    def test_drop_widget_ignores_drop_when_import_disabled(self):
        received = []
        frame = object.__new__(DragDropFrame)
        frame.import_enabled = False
        frame.tk = SimpleNamespace(splitlist=lambda value: tuple(value))
        frame.on_drop = lambda files: received.append(files)

        event = SimpleNamespace(data=("clip.mp4",))
        DragDropFrame._handle_drop(frame, event)

        self.assertEqual(received, [])

    def test_import_summary_counts_unsupported_only_selection(self):
        app = object.__new__(VideoSubtitleRemoverApp)
        messages = []
        app._update_status = lambda message, tone="neutral", toast=False: messages.append(
            (message, tone, toast)
        )

        app._announce_import_summary({"unsupported": 3})

        self.assertEqual(messages[-1][1], "warning")
        self.assertIn("Skipped 3 unsupported files", messages[-1][0])

    def test_import_summary_reports_mixed_skips(self):
        app = object.__new__(VideoSubtitleRemoverApp)
        messages = []
        app._update_status = lambda message, tone="neutral", toast=False: messages.append(
            (message, tone, toast)
        )

        app._announce_import_summary({
            "added": 2,
            "unsupported": 1,
            "missing": 1,
        })

        self.assertEqual(messages[-1][1], "success")
        self.assertIn("Added 2 items", messages[-1][0])
        self.assertIn("skipped 1 unsupported file", messages[-1][0])
        self.assertIn("skipped 1 missing file", messages[-1][0])


class TextWidgetHandlerThreadingTests(unittest.TestCase):
    def test_emit_queues_without_touching_tk_after(self):
        class FakeText:
            def __init__(self):
                self.after_calls = 0

            def after(self, *args):
                self.after_calls += 1

        fake = FakeText()
        handler = TextWidgetHandler(fake)  # type: ignore[arg-type]
        initial_after_calls = fake.after_calls
        record = logging.LogRecord(
            "test",
            logging.WARNING,
            __file__,
            1,
            "worker warning",
            (),
            None,
        )

        handler.emit(record)

        self.assertEqual(fake.after_calls, initial_after_calls)
        self.assertEqual(handler._pending.qsize(), 1)


class QueueCliCommandTests(unittest.TestCase):
    def test_cli_command_quotes_user_paths_and_values(self):
        cfg = ProcessingConfig(
            mode=InpaintMode.LAMA,
            detection_lang="en",
            output_quality=19,
        )
        cfg.watermark_image = r"C:\media\logo & proof.png"
        cfg.watermark_position = "top-left"
        item = QueueItem(
            id="quoted",
            file_path=r"C:\media\source & clip.mp4",
            output_path=r"C:\media\out dir\cleaned & clip.mp4",
            status=ProcessingStatus.IDLE,
            config=cfg,
        )

        command = _build_cli_command(item)

        self.assertIn(r'"C:\media\source & clip.mp4"', command)
        self.assertIn(r'"C:\media\out dir\cleaned & clip.mp4"', command)
        self.assertIn(r'--watermark "C:\media\logo & proof.png"', command)
        self.assertIn("--watermark-position top-left", command)


if __name__ == "__main__":
    unittest.main()
