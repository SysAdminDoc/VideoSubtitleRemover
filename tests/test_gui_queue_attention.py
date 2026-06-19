import unittest

from gui.app import VideoSubtitleRemoverApp
from gui.config import ProcessingConfig, ProcessingStatus, QueueItem


def _item(status, quality_report=None):
    return QueueItem(
        id=f"item-{status.value}-{len(str(quality_report))}",
        file_path="input.mp4",
        output_path="output.mp4",
        config=ProcessingConfig(),
        status=status,
        quality_report=quality_report,
    )


class GuiQueueAttentionTests(unittest.TestCase):
    def _app_stub(self):
        app = VideoSubtitleRemoverApp.__new__(VideoSubtitleRemoverApp)
        app.queue_widgets = {}
        app._status_messages = []
        app._update_status = lambda message, tone="info", **_kw: app._status_messages.append(
            (message, tone)
        )
        app._refresh_action_states = lambda: None

        class _Root:
            def after(self, _delay, callback, *args):
                callback(*args)

        app.root = _Root()
        return app

    def test_attention_count_includes_failed_stopped_and_review_outputs(self):
        queue = [
            _item(ProcessingStatus.ERROR),
            _item(ProcessingStatus.CANCELLED),
            _item(
                ProcessingStatus.COMPLETE,
                {"quality_gate": {"status": "review"}},
            ),
            _item(ProcessingStatus.COMPLETE, {"tag": "Review"}),
        ]

        self.assertEqual(VideoSubtitleRemoverApp._queue_attention_count(queue), 4)

    def test_attention_count_excludes_normal_complete_and_pending_items(self):
        queue = [
            _item(ProcessingStatus.COMPLETE),
            _item(
                ProcessingStatus.COMPLETE,
                {"quality_gate": {"status": "passed"}},
            ),
            _item(ProcessingStatus.IDLE),
            _item(ProcessingStatus.PROCESSING),
        ]

        self.assertEqual(VideoSubtitleRemoverApp._queue_attention_count(queue), 0)

    def test_complete_review_item_announces_warning_not_success(self):
        app = self._app_stub()
        item = _item(
            ProcessingStatus.COMPLETE,
            {"quality_gate": {"status": "review"}},
        )

        app._update_item_display(item)

        self.assertEqual(
            app._status_messages,
            [("input.mp4 completed; quality review recommended", "warning")],
        )

    def test_complete_passed_item_announces_success(self):
        app = self._app_stub()
        item = _item(
            ProcessingStatus.COMPLETE,
            {"quality_gate": {"status": "passed"}},
        )

        app._update_item_display(item)

        self.assertEqual(app._status_messages, [("Completed input.mp4", "success")])


if __name__ == "__main__":
    unittest.main()
