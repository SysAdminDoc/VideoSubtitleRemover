from pathlib import Path
import tempfile
import unittest
from unittest import mock

from gui.app import VideoSubtitleRemoverApp
from gui.config import ProcessingConfig, ProcessingStatus, QueueItem


class GuiReviewWorklistTests(unittest.TestCase):
    def _app_stub(self, item: QueueItem, record: dict):
        app = VideoSubtitleRemoverApp.__new__(VideoSubtitleRemoverApp)
        app.queue = [item]
        app.queue_widgets = {}
        app._selected_queue_item_id = None
        app._last_batch_report_records = [record]
        app._last_batch_report_paths = []
        app._status_messages = []
        app._update_status = lambda message, tone="info", **_kw: app._status_messages.append(
            (message, tone)
        )
        app._update_preview_actions = lambda: None
        return app

    def test_open_first_review_item_focuses_item_and_quality_sheet(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "clip_no_sub.mp4"
            sheet = root / "clip_no_sub.qualitysheet.png"
            sheet.write_bytes(b"png")
            item = QueueItem(
                id="review-item",
                file_path=str(root / "clip.mp4"),
                output_path=str(output),
                config=ProcessingConfig(),
                status=ProcessingStatus.COMPLETE,
            )
            record = {
                "status": "review-needed",
                "output": str(output),
                "output_name": output.name,
                "quality_report": {"sheet": str(sheet)},
                "quality_gate": {
                    "status": "review",
                    "previewFramePaths": [str(sheet)],
                },
            }
            app = self._app_stub(item, record)

            with mock.patch.object(app, "_show_preview") as preview:
                with mock.patch("gui.app.os.startfile", create=True) as startfile:
                    app._open_first_review_item()

            self.assertEqual(app._selected_queue_item_id, item.id)
            preview.assert_called_once_with(item)
            startfile.assert_called_once_with(str(sheet))
            self.assertEqual(app._review_needed_records(), [record])

    def test_review_record_can_match_by_output_name(self):
        item = QueueItem(
            id="review-item",
            file_path="clip.mp4",
            output_path=str(Path("out") / "clip_no_sub.mp4"),
            config=ProcessingConfig(),
            status=ProcessingStatus.COMPLETE,
        )
        record = {
            "status": "review-needed",
            "output_name": "clip_no_sub.mp4",
        }
        app = self._app_stub(item, record)

        self.assertIs(app._queue_item_for_report_record(record), item)

    def test_open_batch_report_prefers_markdown_without_shell(self):
        app = VideoSubtitleRemoverApp.__new__(VideoSubtitleRemoverApp)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            json_report = root / "report&unsafe.json"
            md_report = root / "report&unsafe.md"
            json_report.write_text("{}", encoding="utf-8")
            md_report.write_text("# report\n", encoding="utf-8")

            with mock.patch("gui.app.os.startfile", create=True) as startfile:
                with mock.patch("gui.app.subprocess.Popen") as popen:
                    opened = app._open_batch_report_path([json_report, md_report])

            self.assertTrue(opened)
            startfile.assert_called_once_with(str(md_report))
            popen.assert_not_called()


if __name__ == "__main__":
    unittest.main()
