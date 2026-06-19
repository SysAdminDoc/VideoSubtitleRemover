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


if __name__ == "__main__":
    unittest.main()
