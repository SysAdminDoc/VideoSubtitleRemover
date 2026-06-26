from pathlib import Path
import tempfile
import threading
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
        app._update_guidance_surface = lambda: None
        app.is_processing = False
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

    def test_retry_with_suggested_settings_mutates_only_review_item(self):
        item = QueueItem(
            id="review-item",
            file_path="clip.mp4",
            output_path="clip_no_sub.mp4",
            config=ProcessingConfig(mask_dilate_px=8),
            status=ProcessingStatus.COMPLETE,
            quality_report={
                "quality_gate": {
                    "status": "review",
                    "ladderStep": "increase-dilation",
                    "reason": "residual text score high",
                    "reasons": [{
                        "metric": "residual_text_score",
                        "ladder": "increase-dilation",
                    }],
                },
            },
        )
        untouched = QueueItem(
            id="other",
            file_path="other.mp4",
            output_path="other_no_sub.mp4",
            config=ProcessingConfig(mask_dilate_px=8),
            status=ProcessingStatus.COMPLETE,
        )
        record = {
            "status": "review-needed",
            "output": "clip_no_sub.mp4",
            "output_name": "clip_no_sub.mp4",
            "quality_gate": item.quality_report["quality_gate"],
        }
        app = self._app_stub(item, record)
        app.queue.append(untouched)
        app.queue_lock = threading.Lock()
        app._update_queue_display = mock.Mock()

        with mock.patch("gui.app.save_queue_state") as save_queue:
            self.assertTrue(app._retry_review_item_with_suggested_settings(item.id))

        self.assertEqual(item.status, ProcessingStatus.IDLE)
        self.assertEqual(item.config.mask_dilate_px, 12)
        self.assertEqual(untouched.config.mask_dilate_px, 8)
        self.assertIsNone(item.quality_report)
        self.assertEqual(
            item.retry_config["changes"]["mask_dilate_px"],
            {"before": 8, "after": 12},
        )
        save_queue.assert_called_once()

    def test_retry_config_is_written_to_batch_record(self):
        item = QueueItem(
            id="retry-item",
            file_path="clip.mp4",
            output_path="clip_no_sub.mp4",
            config=ProcessingConfig(mask_dilate_px=12),
            retry_config={
                "schema": "vsr.retry_config.v1",
                "changes": {"mask_dilate_px": {"before": 8, "after": 12}},
            },
        )
        app = VideoSubtitleRemoverApp.__new__(VideoSubtitleRemoverApp)
        app.queue = [item]
        app.queue_lock = threading.Lock()
        app.gpus = []
        app._batch_report_records = {}

        from backend import batch_report as _br
        with mock.patch.object(_br, "_probe_codec_for_log", return_value="h264,64,48,24/1"):
            with mock.patch.object(_br, "_probe_duration_seconds", return_value=2.0):
                with mock.patch.object(_br, "_probe_subtitle_streams", return_value=[]):
                    app._prepare_batch_report_records()

        record = app._batch_report_records[item.id]
        self.assertEqual(
            record["retry_config"]["changes"]["mask_dilate_px"]["after"],
            12,
        )

    def test_global_settings_sync_skips_suggested_retry_items(self):
        retry_item = QueueItem(
            id="retry-item",
            file_path="clip.mp4",
            output_path="clip_no_sub.mp4",
            config=ProcessingConfig(mask_dilate_px=12),
            retry_config={"schema": "vsr.retry_config.v1"},
        )
        normal_item = QueueItem(
            id="normal",
            file_path="other.mp4",
            output_path="other_no_sub.mp4",
            config=ProcessingConfig(mask_dilate_px=8),
        )
        app = VideoSubtitleRemoverApp.__new__(VideoSubtitleRemoverApp)
        app.queue = [retry_item, normal_item]
        app.queue_lock = threading.Lock()
        app._make_processing_snapshot = mock.Mock(
            return_value=ProcessingConfig(mask_dilate_px=16)
        )
        app._refresh_idle_output_paths = mock.Mock(return_value=0)

        self.assertEqual(app._apply_current_settings_to_idle_items(), 1)
        self.assertEqual(retry_item.config.mask_dilate_px, 12)
        self.assertEqual(normal_item.config.mask_dilate_px, 16)


if __name__ == "__main__":
    unittest.main()
