import unittest

from gui.app import VideoSubtitleRemoverApp


class GuiStatusTextTests(unittest.TestCase):
    def test_footer_status_text_preserves_short_messages(self):
        self.assertEqual(
            VideoSubtitleRemoverApp._footer_status_text("Ready to process"),
            "Ready to process",
        )

    def test_footer_status_text_truncates_long_summaries(self):
        message = (
            "Added 492 items to the queue. skipped 12 duplicates. "
            "skipped 8 unsupported files. skipped 2 missing files. "
            "queue reached the 500-item limit"
        )

        compact = VideoSubtitleRemoverApp._footer_status_text(message)

        self.assertLessEqual(len(compact), 132)
        self.assertIn("...", compact)
        self.assertTrue(compact.startswith("Added 492"))
        self.assertTrue(compact.endswith("500-item limit"))


if __name__ == "__main__":
    unittest.main()
